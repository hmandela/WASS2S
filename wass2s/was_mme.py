"""Multi-Model Ensemble (MME) methods for seasonal forecasting.

Provides weighted averaging, probabilistic MME schemes, ELM/MLP-based
ensembles, and logistic and Gaussian-process post-processing.

Key functions
-------------
process_datasets_for_mme
    Align and reindex multiple model hindcast/forecast DataArrays to a
    common grid.
myfill
    Fill missing model forecasts using the observed climatology.

Key classes
-----------
WAS_mme_Weighted / WAS_mme_WeightedContinuous
    Historical threshold weighting plus a continuous shrinkage-stabilised variant.
WAS_ProbWeighted / WAS_ProbWeightedContinuous / WAS_ProbWeightedRPS
    Historical stepwise weighting plus continuous and proper-score RPS variants.
WAS_Min2009_ProbWeighted
    Probabilistic MME following Min (2009) chi-squared weighting.
WAS_mme_MLP
    MLP-based MME post-processing.
WAS_mme_hpELM
    Extreme Learning Machine MME post-processing.
WAS_mme_XGBoosting
    XGBoost-based MME post-processing.
WAS_mme_RF
    Random Forest MME post-processing.
WAS_mme_logistic
    Logistic-regression MME producing direct tercile probabilities.
WAS_mme_gaussian_process
    Gaussian Process MME.
WAS_mme_FastBMA / WAS_mme_FullBMA
    Bayesian Model Averaging (fast and full variants).
"""
from __future__ import annotations

# ---------------------------------------------------------
# 1. Standard Library Imports
# ---------------------------------------------------------
import datetime
import gc
import os
import operator
import random
import warnings
from dataclasses import dataclass
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------
# 2. Third-Party Core Data Libraries
# ---------------------------------------------------------
from collections import OrderedDict
from .was_cca_engine import ExactCCA, LazyXeofsCCA, exact_cca_supported
import numpy as np
import pandas as pd
import xarray as xr
from .was_probability_reference import climatological_terciles
from .was_mme_schema import (validate_mme_predictor_schema,
                             standardize_mme_predictors, require_physical_ensemble)


# ---------------------------------------------------------------------------
# Grid-parallel apply_ufunc: the per-grid-point kernels below already pass
# dask='parallelized', but that only parallelises when the inputs are chunked.
# _pufunc is a drop-in replacement for xr.apply_ufunc that chunks DataArray
# inputs along Y and runs the kernel across worker THREADS (numpy/scipy release
# the GIL), returning an eager result identical to the original — chunking
# changes execution, never values. Disable with WAS_GRID_PARALLEL=0; small
# grids and any error fall back to plain xr.apply_ufunc.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Cluster stacking
#
# The ML classes below fit one model per climate cluster. The original idiom
#
#     X.where(mask_3d).stack(sample=("T", "Y", "X")).transpose("sample", "M").values
#
# re-materialises the FULL (T, M, Y, X) array once per cluster, then throws
# away roughly (1 - 1/n_clusters) of the rows. Measured on a 350x171 grid,
# 24 years, 10 members, 5 clusters: 1.8 s per call site per fold against 0.2 s
# when the array is stacked once and the rows are indexed -- an 8.8x saving,
# and 115 MB allocated once instead of five times.
#
# _stack_masked returns exactly what the old expression returned (full sample
# length, masked rows set to NaN), so every call site keeps its semantics:
# downstream code relies on len(stacked) to rebuild the (T, Y, X) cube.
#
# The memo keys on the identity, shape and dims of the DataArray and keeps a
# strong reference, so an id() cannot be recycled while the entry is alive.
# ---------------------------------------------------------------------------
class _StackMemo:
    """Small LRU of stacked arrays, keyed by the DataArray that produced them."""

    def __init__(self, maxsize=8):
        self._store = OrderedDict()
        self._maxsize = maxsize

    def _get(self, da, core):
        key = (id(da), tuple(da.shape), tuple(da.dims), core)
        hit = self._store.get(key)
        if hit is not None:
            self._store.move_to_end(key)
            return hit[1]
        stacked = da.stack(sample=("T", "Y", "X"))
        if core is not None and core in stacked.dims:
            stacked = stacked.transpose("sample", core)
        values = np.asarray(stacked.values)
        # Keep a strong reference to `da` so its id() cannot be recycled.
        self._store[key] = (da, values)
        self._store.move_to_end(key)
        while len(self._store) > self._maxsize:
            self._store.popitem(last=False)
        return values

    def full(self, da, core="M"):
        return self._get(da, core)

    def rows(self, mask_3d):
        return np.asarray(self._get(mask_3d, None)).ravel().astype(bool)


_STACK_MEMO = _StackMemo()


def _stack_masked(da, mask_3d, core="M"):
    """``da.where(mask_3d).stack(...).transpose(...).values``, computed once.

    Kept for call sites that need the FULL sample length with masked rows set
    to NaN. Note that the NaN fill costs about as much as the saving, so this
    form is only ~1.2x faster than the original expression; prefer
    :func:`_stack_rows` where the call site can work on compact rows.
    """
    try:
        full = _STACK_MEMO.full(da, core)
        rows = _STACK_MEMO.rows(mask_3d)
    except Exception:
        out = da.where(mask_3d).stack(sample=("T", "Y", "X"))
        if core is not None and core in out.dims:
            out = out.transpose("sample", core)
        return np.asarray(out.values)
    out = np.array(full, dtype=float, copy=True)
    if out.ndim == 1:
        out[~rows] = np.nan
    else:
        out[~rows, ...] = np.nan
    return out


def _cluster_row_index(cluster_da, n_time):
    """Row positions of each cluster in a ``('T', 'Y', 'X')`` stacked array."""
    labels = np.asarray(cluster_da.values)
    return np.broadcast_to(labels[None, :, :], (int(n_time),) + labels.shape).ravel()


def _stack_rows(da, row_mask, core="M"):
    """Compact rows of the stacked array: no NaN fill, no full-size copy.

    This is the fast path. Measured on a 350x171 grid, 24 years, 10 members,
    5 clusters, with 3% missing values:

        where + stack, once per cluster   2.91 s
        stack once + row indexing         0.20 s    ->  14.4x

    and the 115 MB stacked array is allocated once per fold instead of once
    per cluster. Returns ``(rows, positions)`` where ``positions`` are the row
    numbers in the full stacked array, so the caller can scatter predictions
    back into the (T, Y, X) cube.
    """
    full = _STACK_MEMO.full(da, core)
    positions = np.flatnonzero(row_mask)
    return full[positions], positions


def _grid_workers():
    """Available CPU budget, bounded by scheduler allocation and affinity."""
    limits = []
    for key in ("WAS_NCORES", "SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS"):
        value = os.environ.get(key)
        if value:
            try:
                if int(value) > 0:
                    limits.append(int(value))
            except ValueError:
                pass
    try:
        limits.append(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        limits.append(os.cpu_count() or 1)
    return max(1, min(limits))
def _report_best_params(tag, best_params_dict, verbose=True, score=None):
    """Print the tuned hyperparameters of every cluster.

    Only WAS_mme_RF and WAS_mme_hpELM reported theirs; XGBoost, MLP, the
    Gaussian process and ELR returned silently, so a run gave no way to tell
    whether a search had converged on a near-linear configuration or on a
    flexible one -- which is exactly the diagnostic worth having on a 24-year
    sample. One line per cluster, gated on the caller's verbose flag.
    """
    if not verbose:
        return
    if isinstance(best_params_dict, dict) and best_params_dict and all(
        isinstance(k, (int, float, np.integer, np.floating)) for k in best_params_dict
    ):
        for cluster_id in sorted(best_params_dict):
            # Cluster ids arrive as numpy floats from np.unique(cluster_da);
            # print them as plain integers so the log stays readable.
            print(f"[{tag}] cluster {int(cluster_id)}: "
                  f"params={best_params_dict[cluster_id]}", flush=True)
    else:
        suffix = "" if score is None else f" score={score:.4f}"
        print(f"[{tag}] params={best_params_dict}{suffix}", flush=True)


def _cluster_plan(n_clusters, n_jobs, cluster_n_jobs=None, model_is_threaded=False):
    """Split the CPU budget between cluster-level and model-level parallelism.

    Clusters are independent, so that is the axis that pays. Measured on 10
    predictor features, ``n_jobs=-1`` inside a single RandomForest fit gives no
    speedup (36.7 s against 35.7 s): the trees are too small for sklearn's
    per-tree threading to help. XGBoost and BLAS-backed learners DO use threads
    productively, so for those the remaining budget is handed to the model
    instead of being wasted.

    Returns ``(workers, threads_per_model)``.
    """
    budget = n_jobs if (n_jobs and n_jobs > 0) else _grid_workers()
    budget = max(1, int(budget))
    if cluster_n_jobs:
        workers = max(1, min(int(cluster_n_jobs), int(n_clusters), budget))
    else:
        workers = max(1, min(int(n_clusters), budget))
    if model_is_threaded:
        threads = max(1, budget // workers)
    else:
        threads = 1 if workers > 1 else budget
    return workers, threads



def _pufunc(func, *args, **kwargs):
    """Evaluate once; never retry a failing scientific kernel silently."""
    n = _grid_workers()
    if os.environ.get("WAS_GRID_PARALLEL", "1") == "0" or n <= 1:
        return xr.apply_ufunc(func, *args, **kwargs)
    try:
        import dask
    except ImportError:
        return xr.apply_ufunc(func, *args, **kwargs)
    new_args, chunked = [], False
    for a in args:
        if isinstance(a, xr.DataArray) and "Y" in a.dims:
            ny = a.sizes["Y"]
            if ny >= 2 * n:
                a = a.chunk({"Y": max(1, ny // n)})
                chunked = True
        new_args.append(a)
    result = xr.apply_ufunc(func, *new_args, **kwargs)
    if not chunked:
        return result
    if isinstance(result, tuple):
        return dask.compute(*result, scheduler="threads", num_workers=n)
    return result.compute(scheduler="threads", num_workers=n)

# ---------------------------------------------------------
# 3. Third-Party Math & ML Libraries (Hard Dependencies)
# ---------------------------------------------------------
try:
    from dask.distributed import Client
except ImportError:
    Client = None

from xgboost import XGBRegressor

# SciPy
from scipy import stats
from scipy.optimize import brentq, fsolve, minimize, minimize_scalar, root_scalar
from scipy.special import betaln, expit, gammainc, gammaincinv, softmax, logsumexp
from scipy.special import gamma as gamma_function
from scipy.stats import (
    boxcox_normmax, expon, genextreme, laplace, linregress, logistic,
    lognorm, loguniform, nbinom, norm, pareto, poisson, randint, uniform, weibull_min
)
# 
from scipy.stats import gamma as gamma_dist
from scipy.stats import gamma as sp_gamma
from scipy.stats import t as tdist

# Scikit-Learn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    AdaBoostRegressor, GradientBoostingRegressor,
    RandomForestRegressor, StackingRegressor, ExtraTreesRegressor,
    HistGradientBoostingRegressor
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, RBF, WhiteKernel
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, ElasticNet
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import log_loss, make_scorer, mean_squared_error
from sklearn.model_selection import (
    GridSearchCV, GroupKFold, KFold, LeaveOneGroupOut,
    RandomizedSearchCV, TimeSeriesSplit, cross_val_score
)
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

# ---------------------------------------------------------
# 4. Optional / Conditional Third-Party Imports
# ---------------------------------------------------------
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import optuna
    from optuna.samplers import RandomSampler, TPESampler
    OPTUNA_AVAILABLE = True
    HAS_OPTUNA = True
except ImportError:
    OPTUNA_AVAILABLE = False
    HAS_OPTUNA = False
    optuna = None
    warnings.warn("Optuna not installed. Bayesian optimization will not be available.")

# ELM: numpy implementation by Gram matrices (was_elm). The hpelm package is
# no longer used; the name is kept for scripts that test for its presence.
from .was_elm import (ELMRegressor, ELMGramBank, effective_norm as _elm_effective_norm,
                      random_layer as _elm_random_layer)
HPELM = None

try:
    import matplotlib.style.core
    import pymc as pm
    import arviz as az
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False
    pm = None
    az = None

# Fallback flag just in case downstream code checks for SciPy presence dynamically
try:
    import scipy
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# Optional acceleration. Numba is deliberately optional: every scientific
# result is identical when it is unavailable; only low-level loops change.
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def _decorator(func):
            return func
        return _decorator

# ---------------------------------------------------------
# 5. Project-Specific Imports
# ---------------------------------------------------------
try:
    import xeofs as xe
except ImportError:
    xe = None

from wass2s.utils import *
from wass2s.was_scientific_utils import align_predictor_target, apply_physical_support
from wass2s.was_machine_learning import MARS as _MARSCore

# U-Net MME post-processor. Kept in its own module because it pulls in an
# OPTIONAL torch dependency: importing it here (rather than inlining 1400+
# lines) means `import wass2s` still works when torch is absent — the class is
# simply unavailable, and WAS_mme_UNet itself raises a clear ImportError when
# instantiated without torch.
try:
    from wass2s.was_mme_unet import WAS_mme_UNet  # noqa: F401
    HAS_UNET = True
except ImportError:  # pragma: no cover - torch (or the module) not installed
    WAS_mme_UNet = None
    HAS_UNET = False
try:
    from wass2s.was_mme_rescnn import WAS_mme_ResCNN  # noqa: F401
    HAS_RESCNN = True
except ImportError:  # pragma: no cover - optional module unavailable
    WAS_mme_ResCNN = None
    HAS_RESCNN = False
from wass2s.was_eof import *
from wass2s.was_verification import *


# Explicit aliases used throughout the probability code.  They are assigned
# after the project wildcard imports so ``gamma`` and ``t`` cannot be silently
# shadowed by a project module.
gamma = gamma_dist
t = tdist


# ---------------------------------------------------------------------------
# Scientific safeguards shared by all MME learners
# ---------------------------------------------------------------------------
def _calendar_year_groups(T):
    """Return one group label per T value, preserving whole-year CV blocks."""
    vals = np.asarray(T.values if hasattr(T, "values") else T)
    if np.issubdtype(vals.dtype, np.datetime64):
        return vals.astype("datetime64[Y]").astype(np.int64) + 1970
    return vals


def _recommended_min_years(n_free_parameters: int, floor: int = 8, ratio: float = 2.0) -> int:
    """Conservative lower bound for gridwise fits on short seasonal hindcasts.

    This is a guardrail, not an inferential theorem.  It prevents fitting a
    parameter-rich distribution to 3--5 years simply because an optimizer can
    technically return a solution.
    """
    return max(int(floor), int(np.ceil(float(ratio) * max(1, n_free_parameters))))


def _year_grouped_splitter(groups, n_splits=5, leave_one_year_out=False):
    """Leakage-safe CV splitter for pooled (T,Y,X) climate samples."""
    groups = np.asarray(groups)
    n_group = len(np.unique(groups))
    if n_group < 2:
        raise ValueError("At least two independent years are required for CV.")
    if leave_one_year_out or n_group <= int(n_splits):
        return LeaveOneGroupOut()
    return GroupKFold(n_splits=min(int(n_splits), n_group))


@njit(cache=True)
def _nonparam_tercile_numba(best_guess, errors, t1, t2, positive_support=False):
    """Fast residual-dressing kernel; exact empirical probabilities."""
    nt = best_guess.size
    out = np.empty((3, nt), dtype=np.float64)
    out[:] = np.nan
    if not np.isfinite(t1) or not np.isfinite(t2) or t2 < t1:
        return out
    for i in range(nt):
        bg = best_guess[i]
        if not np.isfinite(bg):
            continue
        n = 0
        nb = 0
        nn = 0
        for j in range(errors.size):
            e = errors[j]
            if not np.isfinite(e):
                continue
            d = bg + e
            if positive_support and d < 0.0:
                d = 0.0
            n += 1
            if d <= t1:
                nb += 1
            elif d <= t2:
                nn += 1
        if n > 0:
            pb = nb / n
            pn = nn / n
            out[0, i] = pb
            out[1, i] = pn
            out[2, i] = 1.0 - pb - pn
    return out



def empirical_tercile_probabilities_from_training(
    forecast_det, residual_samples, training_obs, clim_year_start=None, clim_year_end=None,
    *, positive_support=None, tercile_thresholds=None, climatology_obs=None
):
    """Dress a held-out deterministic forecast with TRAINING-ONLY residuals.

    ``residual_samples`` must have been generated without using the target/outer-test
    observation (typically by an inner blocked CV over the outer-training years).
    Pass ``tercile_thresholds`` computed once on the climatological reference,
    or the full ``climatology_obs`` reference sample, to use fixed categories.
    Without either argument this low-level helper retains training-only
    categories for compatibility with research callers.
    """
    obs=training_obs
    if "M" in obs.dims:
        obs=obs.isel(M=0,drop=True)
    obs=obs.transpose("T","Y","X")
    fc=forecast_det.transpose("T","Y","X")
    err=residual_samples
    if "T" in err.dims:
        err=err.rename({"T":"S"})
    if "S" not in err.dims:
        raise ValueError("residual_samples must contain T or S sample dimension.")
    if positive_support is None:
        positive_support = bool(apply_physical_support(xr.DataArray(-1.0), obs).item() == 0.0)
    if positive_support:
        fc=fc.clip(min=0.0)
    if tercile_thresholds is not None or climatology_obs is not None:
        clim = obs
    elif clim_year_start is not None and clim_year_end is not None:
        clim=obs.sel(T=slice(str(clim_year_start),str(clim_year_end)))
        if clim.sizes.get("T",0)<3:
            raise ValueError("At least three training years are required inside the requested climatology.")
    else:
        clim=obs
    if clim.sizes.get("T",0)<3:
        raise ValueError("At least three outer-training years are required for terciles.")
    if tercile_thresholds is not None:
        q = tercile_thresholds
    elif climatology_obs is not None:
        q = climatological_terciles(climatology_obs, clim_year_start, clim_year_end)[0]
    else:
        q = climatological_terciles(clim)[0]
    t1=q.isel(quantile=0,drop=True); t2=q.isel(quantile=1,drop=True)
    prob=_pufunc(
        _nonparam_tercile_numba, fc, err, t1, t2,
        input_core_dims=[("T",),("S",),(),()],
        output_core_dims=[("probability","T")],
        vectorize=True,dask="parallelized",output_dtypes=[float],
        kwargs={"positive_support": bool(positive_support)},
        dask_gufunc_kwargs={"output_sizes":{"probability":3},"allow_rechunk":True},
    ).assign_coords(probability=("probability",["PB","PN","PA"]))
    prob=prob.transpose("probability","T","Y","X").clip(min=0,max=1)
    total=prob.sum("probability",skipna=False)
    prob=xr.where(total>0,prob/total,np.nan)
    mask=xr.where(np.isfinite(obs).sum("T")>=3,1.0,np.nan)
    return (prob*mask).transpose("probability","T","Y","X")


def compute_oos_tercile_probabilities(
    Predictant, hindcast_det, splits, clim_year_start=None, clim_year_end=None,
    *, positive_support=None, tercile_reference="climatology", climatology_obs=None
):
    """First-order OOS residual dressing from a completed deterministic hindcast.

    Categories use the fixed climatological reference by default. The target fold
    is excluded from the selected residual rows, but historical residual
    predictions may themselves have been fitted with that
    target year in their own training folds.  Therefore this helper is convenient
    for diagnostics but is NOT the publication-grade nested probability path. Use
    :func:`empirical_tercile_probabilities_from_training` with residuals generated
    inside each outer-training fold for strict verification.
    """
    if "M" in Predictant.dims:
        Predictant=Predictant.isel(M=0,drop=True)
    obs=Predictant.transpose("T","Y","X")
    det=hindcast_det.transpose("T","Y","X")
    det = align_predictor_target(det, obs, mode="season_year")
    if positive_support is None:
        positive_support = bool(apply_physical_support(xr.DataArray(-1.0), obs).item() == 0.0)
    if tercile_reference not in {"climatology", "training"}:
        raise ValueError("tercile_reference must be 'climatology' or 'training'.")
    fixed_q = None
    if tercile_reference == "climatology":
        fixed_q = climatological_terciles(
            obs if climatology_obs is None else climatology_obs,
            clim_year_start, clim_year_end,
        )[0]
    probs=[]
    for tr,te in [(np.asarray(a),np.asarray(b)) for a,b in splits]:
        otr=obs.isel(T=tr); htr=det.isel(T=tr); fte=det.isel(T=te)
        if positive_support:
            fte=fte.clip(min=0.0)
        if fixed_q is not None:
            q = fixed_q
        elif clim_year_start is not None and clim_year_end is not None:
            clim=otr.sel(T=slice(str(clim_year_start),str(clim_year_end)))
            if clim.sizes.get("T",0)<3:
                raise ValueError("Too few training years in the requested climatology.")
        else:
            clim=otr
        if fixed_q is None:
            if clim.sizes.get("T",0)<3:
                continue
            q = climatological_terciles(clim)[0]
        t1=q.isel(quantile=0,drop=True); t2=q.isel(quantile=1,drop=True)
        err=(otr-htr).rename({"T":"S"})
        prob=_pufunc(
            _nonparam_tercile_numba,
            fte,err,t1,t2,
            input_core_dims=[("T",),("S",),(),()],
            output_core_dims=[("probability","T")],
            vectorize=True,dask="parallelized",output_dtypes=[float],
        kwargs={"positive_support": bool(positive_support)},
            dask_gufunc_kwargs={"output_sizes":{"probability":3},"allow_rechunk":True},
        ).assign_coords(probability=("probability",["PB","PN","PA"]))
        prob=prob.transpose("probability","T","Y","X").clip(min=0,max=1)
        s=prob.sum("probability",skipna=False)
        prob=xr.where(s>0,prob/s,np.nan)
        probs.append(prob)
    if not probs:
        raise ValueError("No valid fold available for strict OOS probability dressing.")
    out=xr.concat(probs,dim="T").sortby("T").reindex(T=obs["T"])
    mask=xr.where(np.isfinite(obs).sum("T")>=3,1.0,np.nan)
    return (out*mask).transpose("probability","T","Y","X")

def _tercile_logloss_scorer(estimator, X, y):
    raw = estimator.predict_proba(X)
    probability = np.zeros((len(y), 3), dtype=float)
    for index, category in enumerate(estimator.classes_):
        category = int(category)
        if category not in (0, 1, 2):
            raise ValueError("Classifier targets must be tercile labels 0, 1, 2.")
        probability[:, category] = raw[:, index]
    return -log_loss(y, probability, labels=[0, 1, 2])


def _effective_sample_warning(n_years, n_features, learner):
    """Warn when flexible ML is being asked to learn from very few years."""
    if int(n_years) < max(12, 2 * int(n_features)):
        warnings.warn(
            f"{learner}: only {n_years} independent years for {n_features} predictors. "
            "Treat skill as provisional and require nested/outer year-wise validation.",
            RuntimeWarning, stacklevel=2,
        )


def _training_year_signature(da):
    """Stable signature used to detect cluster maps fitted outside a CV fold."""
    if not isinstance(da, xr.DataArray) or "T" not in da.coords:
        return ""
    y = _calendar_year_groups(da["T"])
    return ",".join(str(int(v)) if np.issubdtype(np.asarray(y).dtype, np.integer) else str(v)
                    for v in np.unique(y))


def _stored_training_signature(da):
    """Training-year signature stored on a cluster/covariate map.

    ``was_mme`` writes ``training_years``; ``was_scientific_utils.climate_cluster_map``
    (used by was_crossfit, was_linear_models, was_machine_learning) writes
    ``training_times``.  Both are accepted so a map produced by either helper is
    recognised instead of silently triggering a full re-tuning.
    """
    attrs = getattr(da, "attrs", {}) or {}
    stored = str(attrs.get("training_years", "") or "")
    if stored:
        return stored
    times = str(attrs.get("training_times", "") or "")
    if not times:
        return ""
    try:
        years = np.asarray(times.split(","), dtype="datetime64[ns]").astype("datetime64[Y]").astype(int) + 1970
        return ",".join(str(int(v)) for v in np.unique(years))
    except (ValueError, TypeError):
        return ""


def _cluster_matches_training(cluster_da, y_train):
    if cluster_da is None:
        return False
    stored = _stored_training_signature(cluster_da)
    return bool(stored) and stored == _training_year_signature(y_train)


def _looks_standardized(da, tol=0.05):
    """Heuristic: per-pixel anomalies with mean≈0 and std≈1 on most cells."""
    if not isinstance(da, xr.DataArray) or "T" not in da.dims:
        return False
    if "M" in da.dims:
        da = da.isel(M=0, drop=True)
    mean = np.asarray(da.mean("T", skipna=True).values, dtype=float).ravel()
    std = np.asarray(da.std("T", skipna=True).values, dtype=float).ravel()
    good = np.isfinite(mean) & np.isfinite(std) & (std > 0)
    if good.sum() < 4:
        return False
    return bool(np.nanmedian(np.abs(mean[good])) < tol and
                np.nanmedian(np.abs(std[good] - 1.0)) < tol)


def _resolve_regional_params(model, X_train, y_train, clim_year_start, clim_year_end,
                             best_params, cluster_da):
    """Reuse supplied (best_params, cluster_da) or re-tune, never silently.

    The pair is accepted when the cluster map carries a training signature that
    matches ``y_train`` (``training_years`` or ``training_times``).  When a pair
    was supplied but rejected, a warning says why instead of the old silent
    re-tuning, which also hid the cost of an unexpected HPO inside forecasts.
    """
    if best_params is not None and cluster_da is not None and _cluster_matches_training(cluster_da, y_train):
        return best_params, cluster_da
    if best_params is not None or cluster_da is not None:
        stored = _stored_training_signature(cluster_da) if cluster_da is not None else ""
        warnings.warn(
            f"{type(model).__name__}: supplied best_params/cluster_da were discarded "
            f"(cluster map training years {stored or 'missing'!r} != y_train "
            f"{_training_year_signature(y_train)!r}); hyperparameters are re-tuned.",
            RuntimeWarning, stacklevel=3)
    return model.compute_hyperparameters(X_train, y_train, clim_year_start, clim_year_end)


def _build_climate_feature_clusters(Predictand, n_clusters, random_state=42):
    """Build a stable (Y, X) climate-zone map from temporal summary features.

    The four features are the temporal mean, standard deviation, lower tercile
    and upper tercile.  Features are standardized before KMeans so the spread
    feature is not dominated by the variables expressed in physical units.
    Missing cells remain NaN, and the effective cluster count is automatically
    reduced when the valid domain contains fewer cells than requested.
    """
    if not isinstance(Predictand, xr.DataArray):
        raise TypeError("Predictand must be an xarray.DataArray.")
    if "M" in Predictand.dims:
        Predictand = Predictand.isel(M=0, drop=True)
    missing = {"T", "Y", "X"} - set(Predictand.dims)
    if missing:
        raise ValueError(f"Predictand is missing dimensions {sorted(missing)}.")
    Predictand = Predictand.transpose("T", "Y", "X")
    if _looks_standardized(Predictand):
        # (mean, std, q33, q67) of standardized anomalies are ~(0, 1, -.43, .43)
        # on every cell: KMeans would then partition pure sampling noise.
        warnings.warn(
            "Climate zones are being built from a predictand that looks already "
            "standardized (mean~0, std~1 per pixel); pass the RAW predictand to "
            "compute_hyperparameters/compute_model so the zones are meaningful.",
            RuntimeWarning, stacklevel=2)

    q33 = Predictand.quantile(1.0 / 3.0, dim="T", skipna=True).drop_vars(
        "quantile", errors="ignore"
    )
    q67 = Predictand.quantile(2.0 / 3.0, dim="T", skipna=True).drop_vars(
        "quantile", errors="ignore"
    )
    features = xr.Dataset({
        "mean": Predictand.mean("T", skipna=True),
        "std": Predictand.std("T", skipna=True),
        "q33": q33,
        "q67": q67,
    }).to_array("feature").transpose("feature", "Y", "X")

    stacked = features.stack(cell=("Y", "X")).transpose("cell", "feature")
    values = np.asarray(stacked.values, dtype=float)
    valid = np.all(np.isfinite(values), axis=1)
    labels = np.full(values.shape[0], np.nan, dtype=float)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return xr.DataArray(
            labels,
            dims=("cell",),
            coords={"cell": stacked["cell"]},
        ).unstack("cell").transpose("Y", "X")

    requested = max(1, int(n_clusters))
    effective = min(requested, n_valid)
    Xv = values[valid]
    centre = np.nanmean(Xv, axis=0)
    scale = np.nanstd(Xv, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, 1.0)
    Xv = (Xv - centre) / scale

    if effective == 1:
        labels[valid] = 0.0
    else:
        km = KMeans(n_clusters=effective, random_state=random_state, n_init=10)
        labels[valid] = km.fit_predict(Xv).astype(float)

    result = xr.DataArray(
        labels,
        dims=("cell",),
        coords={"cell": stacked["cell"]},
        name="cluster",
    ).unstack("cell").transpose("Y", "X")
    result = result.where(Predictand.notnull().any("T"))
    result.attrs.update({
        "requested_clusters": requested,
        "effective_clusters": effective,
        "features": "mean,std,q33,q67",
        "training_years": _training_year_signature(Predictand),
    })
    return result


# =============================================================================
# Spatial variability of the pooled ML MMEs (RF / XGB / hpELM / MLP / sklearn)
#
#   spatial_mode = "zones"  one model per KMeans climate zone (historical).
#                  "soft"   ONE model on the whole domain; the zones are replaced
#                           by continuous, fold-local spatial covariates appended
#                           to the GCM channels (Dabernig et al. 2017 "SAMOS";
#                           the static channels of WAS_mme_UNet). No seams, no
#                           unstable zone boundaries, same computational cost.
#                  "both"   per-zone models AND the covariates.
#
# Covariates are built from the RAW training predictand only (climatology of the
# fold), z-scored across pixels, so they are constant in time and leak nothing.
# =============================================================================
SPATIAL_MODES = ("zones", "soft", "both")
DEFAULT_COVARIATES = ("clim_mean", "clim_logstd", "clim_asym", "dry_frac")
STATIC_CHANNEL_LABEL = "static"


def _init_spatial_options(model, spatial_mode="zones", covariate_names=None,
                          static_fields=None, coord_step=None, zones_as_features=False,
                          predictor_smoothing=0, one_se_rule=False,
                          amplitude_calibration=True, amplitude_calibration_inner_folds=3):
    """Attach the spatial/selection options shared by every pooled ML MME.

    ``amplitude_calibration`` (on by default) rescales the prediction with a
    single domain-wide slope fitted on the training fold; see
    ``_calibrate_amplitude``. Pass False to restore the raw, uncalibrated
    output of releases before 2026-09-20.
    """
    mode = str(spatial_mode or "zones").lower()
    if mode not in SPATIAL_MODES:
        raise ValueError(f"spatial_mode must be one of {SPATIAL_MODES}, got {spatial_mode!r}.")
    model.spatial_mode = mode
    model.covariate_names = tuple(covariate_names) if covariate_names else DEFAULT_COVARIATES
    if static_fields is not None and not isinstance(static_fields, dict):
        raise TypeError("static_fields must be a dict {name: DataArray(Y, X)}.")
    model.static_fields = static_fields
    model.coord_step = None if coord_step in (None, 0) else float(coord_step)
    model.zones_as_features = bool(zones_as_features)
    model.predictor_smoothing = max(0, int(predictor_smoothing or 0))
    model.one_se_rule = bool(one_se_rule)
    # True/'global' : une pente par pli ; 'spread' : une pente par annee,
    # pilotee par le rapport signal/dispersion de l'ensemble ; False : brut.
    mode = _normalise_amplitude_mode(amplitude_calibration)
    model.amplitude_calibration = mode is not None
    model.amplitude_calibration_mode = mode or "global"
    # >= 2 : pente estimee sur des predictions HORS echantillon des annees
    # d'entrainement (validation croisee interne par blocs, cout x(1+k)) ;
    # 0 : pente en echantillon, biaisee vers 1 pour tout apprenant qui ajuste
    # ses donnees (mesure : MSSS -0.017 contre +0.011 pour le MLP).
    model.amplitude_calibration_inner_folds = max(0, int(amplitude_calibration_inner_folds or 0))
    model.amplitude_calibration_ = None
    model.covariates_ = None


def _single_zone_map(Predictand):
    """One label (0) on every cell observed at least once; NaN elsewhere."""
    if "M" in Predictand.dims:
        Predictand = Predictand.isel(M=0, drop=True)
    valid = Predictand.transpose("T", "Y", "X").notnull().any("T")
    result = xr.where(valid, 0.0, np.nan).rename("cluster")
    result.attrs.update({
        "requested_clusters": 1, "effective_clusters": 1, "features": "none",
        "training_years": _training_year_signature(Predictand), "spatial_mode": "soft",
    })
    return result


def _regionalisation(model, Predictand):
    """Zone map of the model's spatial mode, built from the RAW predictand."""
    mode = getattr(model, "spatial_mode", "zones")
    if mode == "soft":
        return _single_zone_map(Predictand)
    clusters = _build_climate_feature_clusters(
        Predictand, model.n_clusters, getattr(model, "random_state", 42))
    clusters.attrs["spatial_mode"] = mode
    return clusters


def _spatial_covariates(y_train, clim_start=None, clim_end=None, names=DEFAULT_COVARIATES,
                        static=None, cluster_da=None, coord_step=None):
    """(M, Y, X) spatial covariates: fold climatology + static fields.

    * ``clim_mean``, ``clim_logstd``: location/scale of the raw predictand,
    * ``clim_asym``: (q67 - mean) - (mean - q33), in local standard deviations,
    * ``dry_frac``: fraction of seasons at or below 5 % of the local mean
      (dropped automatically when constant, e.g. for temperature),
    * ``lat``/``lon`` coarsened to ``coord_step`` degrees (off by default:
      raw coordinates let trees carve latitude bands, Meyer et al. 2019),
    * ``static_<name>``: user fields (elevation, distance to coast, ...),
    * ``zone_<k>``: optional one-hot membership of the KMeans zones.

    Each channel is z-scored ACROSS PIXELS (they are constant in time) and
    missing cells are set to 0 (a neutral value; those cells carry no target).
    The result carries the ``training_years`` signature of ``y_train``.
    """
    if "M" in y_train.dims:
        y_train = y_train.isel(M=0, drop=True)
    y_train = y_train.transpose("T", "Y", "X")
    if _looks_standardized(y_train):
        warnings.warn(
            "Spatial covariates are being built from a predictand that looks already "
            "standardized; they must come from the RAW training predictand.",
            RuntimeWarning, stacklevel=2)
    ref = y_train
    if clim_start is not None and clim_end is not None:
        window = y_train.sel(T=slice(str(clim_start), str(clim_end)))
        if window.sizes.get("T", 0) >= 3:
            ref = window
    names = tuple(names or ())
    mu = ref.mean("T", skipna=True)
    sd = ref.std("T", skipna=True)
    sd = sd.where(np.isfinite(sd) & (sd > 0))
    feats = {}
    if "clim_mean" in names:
        feats["clim_mean"] = mu
    if "clim_logstd" in names:
        feats["clim_logstd"] = np.log(sd)
    if "clim_asym" in names:
        q = ref.quantile([1.0 / 3.0, 2.0 / 3.0], dim="T", skipna=True)
        q1 = q.isel(quantile=0, drop=True).drop_vars("quantile", errors="ignore")
        q2 = q.isel(quantile=1, drop=True).drop_vars("quantile", errors="ignore")
        feats["clim_asym"] = ((q2 - mu) - (mu - q1)) / sd
    if "dry_frac" in names:
        feats["dry_frac"] = (ref <= 0.05 * mu.clip(min=0.0)).where(np.isfinite(ref)).mean("T", skipna=True)
    if coord_step:
        step = float(coord_step)
        feats["lat"] = (np.floor(mu["Y"] / step) * step).broadcast_like(mu)
        feats["lon"] = (np.floor(mu["X"] / step) * step).broadcast_like(mu)
    if static:
        for key, field in static.items():
            if not isinstance(field, xr.DataArray) or set(field.dims) != {"Y", "X"}:
                raise ValueError(f"static field {key!r} must be a DataArray with dims (Y, X).")
            field = field.transpose("Y", "X")
            if not (field["Y"].equals(mu["Y"]) and field["X"].equals(mu["X"])):
                field = field.interp_like(mu, method="nearest")
            feats[f"static_{key}"] = field
    if cluster_da is not None:
        labels = np.asarray(cluster_da.values, dtype=float)
        for c in np.unique(labels[np.isfinite(labels)]):
            feats[f"zone_{int(c)}"] = (cluster_da == c).astype(float)
    if not feats:
        return None
    cov = xr.concat([f.astype(float) for f in feats.values()],
                    dim=pd.Index(list(feats), name="M"))
    cov = cov.where(np.isfinite(mu))
    centre = cov.mean(("Y", "X"), skipna=True)
    scale = cov.std(("Y", "X"), skipna=True)
    finite_scale = np.isfinite(scale.values) & (scale.values > 1e-12)
    cov = cov.isel(M=np.flatnonzero(finite_scale))            # drop constant channels
    centre = centre.isel(M=np.flatnonzero(finite_scale))
    scale = scale.isel(M=np.flatnonzero(finite_scale))
    if cov.sizes["M"] == 0:
        return None
    z = ((cov - centre) / scale).fillna(0.0).transpose("M", "Y", "X")
    z.name = "spatial_covariates"
    z.attrs.update({
        "training_years": _training_year_signature(y_train),
        "covariates": ",".join(str(v) for v in z["M"].values),
        "standardization": "z-score across pixels; NaN -> 0",
    })
    return z


def _smoothed_predictor_channels(X, window):
    """Regional means (window x window cells) of every GCM channel.

    The seasonal predictable signal is large scale while the grid-point GCM
    value is noisy: smoothed copies give the learner the regional context.
    """
    window = int(window)
    if window <= 1:
        return None
    sm = X.transpose("T", "M", "Y", "X").rolling(Y=window, X=window, center=True,
                                                 min_periods=1).mean()
    sm = sm.assign_coords(M=[f"{m}__smooth{window}" for m in X["M"].values])
    return sm


def _augment_predictors(X, cov=None, smoothing=0):
    """Concatenate covariates (constant in T) and smoothed GCM channels along M."""
    X = X.transpose("T", "M", "Y", "X")
    parts = [X]
    smoothed = _smoothed_predictor_channels(X, smoothing) if smoothing else None
    if smoothed is not None:
        parts.append(smoothed)
    if cov is not None:
        extra = cov.transpose("M", "Y", "X").expand_dims(T=X.sizes["T"]).assign_coords(T=X["T"])
        extra = extra.transpose("T", "M", "Y", "X")
        for name in X.coords:
            if name != "M" and X.coords[name].dims == ("M",):
                extra = extra.assign_coords({name: ("M", [STATIC_CHANNEL_LABEL] * extra.sizes["M"])})
        parts.append(extra)
    if len(parts) == 1:
        return X
    # Channel labels of mixed dtype (ints/str/pandas string) cannot be
    # concatenated by xarray: use one plain object index for every part.
    parts = [p.assign_coords(M=np.asarray(p["M"].values, dtype=object)) for p in parts]
    out = xr.concat(parts, dim="M", coords="minimal", compat="override")
    out.attrs = dict(X.attrs)
    if cov is not None:
        out.attrs["spatial_covariates"] = cov.attrs.get("covariates", "")
    if smoothed is not None:
        out.attrs["predictor_smoothing"] = int(smoothing)
    return out


def _model_covariates(model, y_raw, clim_start=None, clim_end=None):
    """Covariates requested by the model's spatial mode, from the RAW predictand."""
    mode = getattr(model, "spatial_mode", "zones")
    if mode == "zones":
        return None
    zones = None
    if mode == "soft" and getattr(model, "zones_as_features", False):
        zones = _build_climate_feature_clusters(y_raw, model.n_clusters, getattr(model, "random_state", 42))
    return _spatial_covariates(
        y_raw, clim_start, clim_end, names=getattr(model, "covariate_names", DEFAULT_COVARIATES),
        static=getattr(model, "static_fields", None), cluster_da=zones,
        coord_step=getattr(model, "coord_step", None))


def _prepare_channels(model, X_train, X_test, y_train, covariates=None,
                      clim_start=None, clim_end=None):
    """Predictor channels actually seen by the learner (GCM + spatial extras).

    ``covariates`` come from the caller (WAS_Cross_Validator passes fold-local
    ones built from the raw training predictand); otherwise the map stored by
    ``compute_hyperparameters`` is reused when it was fitted on the same years.
    """
    mode = getattr(model, "spatial_mode", "zones")
    smoothing = getattr(model, "predictor_smoothing", 0)
    cov = None
    if mode != "zones":
        cov = covariates
        if cov is None:
            stored = getattr(model, "covariates_", None)
            if stored is not None and _cluster_matches_training(stored, y_train):
                cov = stored
        if cov is None:
            if _looks_standardized(y_train):
                raise ValueError(
                    f"{type(model).__name__}: spatial_mode={mode!r} needs covariates built from the "
                    "RAW predictand; pass covariates=... (WAS_Cross_Validator does it) or call "
                    "compute_hyperparameters on raw data before compute_model.")
            cov = _model_covariates(model, y_train, clim_start, clim_end)
    if cov is None and not smoothing:
        return X_train, X_test
    return (_augment_predictors(X_train, cov, smoothing),
            _augment_predictors(X_test, cov, smoothing))


def _regional_params_from(model, best_params, Predictand):
    """Reuse tuned hyperparameters on a NEW training sample.

    The zone map is refitted on ``Predictand`` (RAW training predictand of the
    new sample) while the hyperparameters are kept: this is what
    ``probability_inner_retune=False`` means for the calibration folds. In
    strict regional tuning every zone shares one candidate; a legacy
    per-zone dict falls back to its first entry for zones it does not list.
    """
    clusters = _regionalisation(model, Predictand)
    if isinstance(best_params, dict) and best_params and all(
            isinstance(v, dict) for v in best_params.values()):
        per, shared = best_params, next(iter(best_params.values()))
    else:
        per, shared = {}, dict(best_params or {})
    labels = np.unique(clusters.values[np.isfinite(clusters.values)])
    params = {int(c): dict(per.get(int(c), shared)) for c in labels}
    clusters.attrs['hyperparameter_scope'] = 'reused_from_outer_training_fold'
    return dict(best_params=params, cluster_da=clusters)


def _thin_training_rows(train_sel, n_time, n_lat, n_lon, stride):
    """Keep one training cell out of ``stride`` in BOTH Y and X (rows stay (T,Y,X)).

    The previous ``[::stride]`` on the flattened index thinned along X only
    (reduction 1/stride instead of 1/stride^2, and never the tuning sample).
    """
    stride = int(stride or 1)
    if stride <= 1:
        return train_sel
    keep = np.zeros((int(n_lat), int(n_lon)), dtype=bool)
    keep[::stride, ::stride] = True
    return train_sel & np.broadcast_to(keep[None], (int(n_time), int(n_lat), int(n_lon))).ravel()


# ---------------------------------------------------------------------------
# Global amplitude calibration of pooled ML predictions
# ---------------------------------------------------------------------------
# Validated on 17 GCM hindcasts (1993-2016, MJJ, West Africa, 0.1 deg) with
# 8 blocked 3-year folds. Raw "zones" predictions carry far more anomaly
# amplitude than their correlation justifies, which is catastrophic for the
# mean squared error: the MLP scored r = 0.198 but MSSS = -0.522. Rescaling
# with a single domain-wide slope fitted on the training fold kept r = 0.187
# and lifted MSSS to +0.011 -- the only positive MSSS obtained on that case.
#
# The slope MUST be global. A per-pixel slope estimated on ~20 years has a
# standard error near 0.22 while the true correlation is around 0.1, so it is
# mostly noise and wrong-signed on roughly half the grid: applied out of
# sample it destroyed the signal (XGBoost r fell 0.131 -> 0.035, losing
# significance). A single number pooled over all pixels and years is stable,
# and amplitude is a level defect rather than a geographical one.

# Nombre minimal de cellules de validation conservees apres amincissement du
# reglage regional : en deca, on repasse sur la grille pleine plutot que de
# noter un candidat sur un echantillon vide.
_MIN_TUNING_VALID_CELLS = 30

AMPLITUDE_SLOPE_BOUNDS = (0.0, 2.0)
DEGENERATE_INSAMPLE_R = 0.05
DEGENERATE_AMPLITUDE = 0.5


def _pooled_grid_prediction(models, X, cluster_da):
    """Predict a full ``(T, Y, X)`` grid with one fitted estimator per cluster.

    Mirrors the scatter used by the pooled learners' ``compute_model`` so the
    in-sample prediction is produced by exactly the same path as the test one.
    """
    n_time = int(X.sizes["T"])
    n_lat = int(X.sizes["Y"])
    n_lon = int(X.sizes["X"])
    flat = np.full((n_time * n_lat * n_lon,), np.nan)
    rows = _cluster_row_index(cluster_da, n_time)
    for label, estimator in (models or {}).items():
        if estimator is None:
            continue
        block, positions = _stack_rows(X, rows == label, core="M")
        if positions.size == 0:
            continue
        keep = np.all(np.isfinite(block), axis=1)
        if not keep.any():
            continue
        flat[positions[keep]] = np.asarray(estimator.predict(block[keep])).ravel()
    return xr.DataArray(flat.reshape(n_time, n_lat, n_lon), dims=("T", "Y", "X"),
                        coords={"T": X["T"], "Y": X["Y"], "X": X["X"]})


def _amplitude_slope(prediction, target, bounds=AMPLITUDE_SLOPE_BOUNDS):
    """Domain-wide regression slope of observed on predicted anomalies.

    Anomalies are taken about each field's own training mean, so the slope is
    invariant to a constant bias and to the scale the caller works in.
    Returns ``(slope, in-sample correlation, predicted/observed amplitude)``;
    the two diagnostics feed the degeneracy guard.
    """
    pa = prediction - prediction.mean("T")
    oa = target - target.mean("T")
    finite = np.isfinite(pa) & np.isfinite(oa)
    pa = pa.where(finite)
    oa = oa.where(finite)

    cov = float((pa * oa).sum(skipna=True))
    var = float((pa ** 2).sum(skipna=True))
    var_obs = float((oa ** 2).sum(skipna=True))
    if not np.isfinite(cov) or not np.isfinite(var) or var <= 1e-12:
        # a prediction with no temporal variance carries no information:
        # fall back to the training climatology
        return 0.0, 0.0, 0.0
    slope = float(np.clip(cov / var, bounds[0], bounds[1]))
    correlation = cov / np.sqrt(var * var_obs) if var_obs > 1e-12 else 0.0
    amplitude = np.sqrt(var / var_obs) if var_obs > 1e-12 else 0.0
    return slope, float(correlation), float(amplitude)


# ---------------------------------------------------------------------------
# Spread-conditioned amplitude: the year's ensemble agreement modulates the slope
# ---------------------------------------------------------------------------
# A forecaster trusts a seasonal signal more when the ensemble agrees on it.
# The global slope above applies the same shrinkage to every year; here the
# slope becomes a function of the year's signal-to-spread ratio z:
#
#     b(t) = b0 + b1 * (z_t / median(z_train) - 1),     b1 >= 0
#
# fitted at the YEAR level (one optimal slope per training year, regressed on
# z with n = number of training years), so that the standard error of b1 is
# honest. b1 is then shrunk towards 0 by its own t statistic (positive-part
# James-Stein): with no spread-skill relation the method degrades gracefully
# to the global slope, never below it in expectation.
#
# Spread is read from the predictor cube in two ways:
#   * WASS2S member-statistics mode (ensemble_mean=None): channels carry the
#     coordinate ``ensemble_statistic``; signal = mean of the ``mean`` channels,
#     spread^2 = mean of the ``std`` channels squared + variance of the model
#     means (law of total variance of the grand ensemble);
#   * legacy mode (one channel per model mean): signal = mean over M,
#     spread = inter-model standard deviation, a proxy for member spread.

AMPLITUDE_CALIBRATION_MODES = ("global", "spread")
SPREAD_Z_BOUNDS = (0.0, 3.0)          # z_test / median(z_train), extrapolation guard
_COVARIATE_LABELS = set(DEFAULT_COVARIATES) | {"lat", "lon"}


def _normalise_amplitude_mode(value):
    """True -> 'global'; False/None -> None; 'global'/'spread' pass through."""
    if value is None or value is False:
        return None
    if value is True:
        return "global"
    mode = str(value).strip().lower()
    if mode in ("", "off", "none", "false", "0"):
        return None
    if mode in ("on", "true", "1"):
        return "global"
    if mode not in AMPLITUDE_CALIBRATION_MODES:
        raise ValueError(f"amplitude_calibration must be one of {AMPLITUDE_CALIBRATION_MODES}, "
                         f"True/False, got {value!r}.")
    return mode


def _gcm_channels(X):
    """Keep the GCM channels of a predictor cube, dropping covariates.

    ``_augment_predictors`` appends spatial covariates (``clim_mean``,
    ``static_<name>``, ``zone_<k>``, ``lat``/``lon``) and smoothed copies
    (``<model>__smooth<w>``) along M. None of them carries ensemble spread.
    """
    if "M" not in X.dims:
        return X
    labels = [str(v) for v in np.asarray(X["M"].values)]
    keep = [not (lab in _COVARIATE_LABELS or lab.startswith("static_")
                 or lab.startswith("zone_") or "__smooth" in lab)
            for lab in labels]
    if all(keep):
        return X
    return X.isel(M=np.flatnonzero(keep))


def _ensemble_signal_spread(X):
    """``(signal, spread)`` as ``(T, Y, X)`` fields from a ``(T, M, Y, X)`` cube."""
    X = _gcm_channels(X).transpose("T", "M", "Y", "X")
    stats = X.coords.get("ensemble_statistic")
    if stats is not None:
        labels = np.asarray(stats.values).astype(str)
        means = X.isel(M=np.flatnonzero(labels == "mean"))
        stds = X.isel(M=np.flatnonzero(labels == "std"))
        if means.sizes["M"] == 0:
            raise ValueError("member-statistics cube without any 'mean' channel.")
        signal = means.mean("M", skipna=True)
        within = (stds ** 2).mean("M", skipna=True) if stds.sizes["M"] else 0.0
        between = means.var("M", skipna=True) if means.sizes["M"] > 1 else 0.0
        spread = np.sqrt(within + between)
    else:
        if X.sizes["M"] < 2:
            raise ValueError("spread-conditioned calibration needs at least two channels.")
        signal = X.mean("M", skipna=True)
        spread = X.std("M", skipna=True)
    return signal, spread


def _signal_to_spread_ratio(X, signal_clim=None):
    """Domain-wide ratio per year: RMS of the ensemble-mean anomaly over the
    RMS of the ensemble spread. Returns ``(z (T,), signal climatology (Y,X))``.

    One number per year keeps the modulation stable on 20-30 years; a
    per-pixel ratio would inherit the per-pixel noise this module is built to
    avoid.
    """
    signal, spread = _ensemble_signal_spread(X)
    if signal_clim is None:
        signal_clim = signal.mean("T")
    anom = signal - signal_clim
    finite = np.isfinite(anom) & np.isfinite(spread)
    num = np.sqrt((anom.where(finite) ** 2).mean(("Y", "X"), skipna=True))
    den = np.sqrt((spread.where(finite) ** 2).mean(("Y", "X"), skipna=True))
    z = (num / den.where(den > 1e-12)).fillna(0.0)
    return z, signal_clim


def _yearly_slopes(pa, oa):
    """Optimal global slope of each year and its weight sum(pa^2)."""
    finite = np.isfinite(pa) & np.isfinite(oa)
    cov = (pa.where(finite) * oa.where(finite)).sum(("Y", "X"), skipna=True)
    var = (pa.where(finite) ** 2).sum(("Y", "X"), skipna=True)
    slope = (cov / var.where(var > 1e-12))
    return slope, var


def _spread_conditioned_slope(pa_train, oa_train, z_train, z_test,
                              bounds=AMPLITUDE_SLOPE_BOUNDS):
    """Fit ``b(z)`` on the training years, return the slope of each test year.

    Weighted least squares of the yearly slope on ``(z_norm - 1)`` with weights
    sum(pa^2): years with more predicted signal pin the line down. ``b1`` is
    forced non-negative (agreement never *reduces* confidence) and shrunk by
    ``max(0, 1 - 1/t^2)``: a sensitivity that is not distinguishable from zero
    on the training years is not used.
    """
    b_year, w_year = _yearly_slopes(pa_train, oa_train)
    b = np.asarray(b_year.values, dtype=float)
    w = np.asarray(w_year.values, dtype=float)
    z = np.asarray(z_train.values, dtype=float)
    ok = np.isfinite(b) & np.isfinite(w) & np.isfinite(z) & (w > 0)
    scale = float(np.median(z[ok])) if ok.any() and np.median(z[ok]) > 1e-12 else 1.0
    x = z / scale - 1.0

    diag = dict(b0=np.nan, b1=np.nan, b1_raw=np.nan, b1_t=np.nan,
                z_scale=scale, n_years=int(ok.sum()))
    if ok.sum() < 5:
        # too few years for a two-parameter line: fall back to the pooled slope
        b0 = float(np.clip(np.sum(w[ok] * b[ok]) / np.sum(w[ok]), *bounds)) if ok.any() else 0.0
        diag.update(b0=b0, b1=0.0, b1_raw=0.0, b1_t=0.0, fallback="too_few_years")
        zt = np.clip(np.asarray(z_test.values, dtype=float) / scale, *SPREAD_Z_BOUNDS)
        return np.full(zt.shape, b0), diag

    bw, ww, xw = b[ok], w[ok], x[ok]
    xbar = np.sum(ww * xw) / np.sum(ww)
    bbar = np.sum(ww * bw) / np.sum(ww)
    sxx = np.sum(ww * (xw - xbar) ** 2)
    sxy = np.sum(ww * (xw - xbar) * (bw - bbar))
    b1_raw = sxy / sxx if sxx > 1e-12 else 0.0
    b0 = bbar - b1_raw * xbar

    resid = bw - (b0 + b1_raw * xw)
    dof = max(int(ok.sum()) - 2, 1)
    s2 = np.sum(ww * resid ** 2) / dof
    se = np.sqrt(s2 / sxx) if sxx > 1e-12 else np.inf
    t = b1_raw / se if np.isfinite(se) and se > 0 else 0.0
    shrink = max(0.0, 1.0 - 1.0 / (t * t)) if abs(t) > 1e-12 else 0.0
    b1 = max(0.0, b1_raw) * shrink
    # the intercept is re-centred so that the mean training slope is preserved
    b0 = bbar - b1 * xbar

    diag.update(b0=float(b0), b1=float(b1), b1_raw=float(b1_raw), b1_t=float(t))
    zt = np.clip(np.asarray(z_test.values, dtype=float) / scale, *SPREAD_Z_BOUNDS)
    slopes = np.clip(b0 + b1 * (zt - 1.0), *bounds)
    return slopes, diag


def _blocked_inner_predictions(model, models_by_cluster, raw_X_train, y_train,
                               best_params, clim_year_start, clim_year_end, n_folds):
    """Out-of-sample predictions of the training years by blocked inner CV.

    The learner is refitted ``n_folds`` times on contiguous year blocks with
    the OUTER hyperparameters (zone map refitted per block, hyperparameters
    kept -- ``_regional_params_from``), calibration disabled, and each block
    is predicted by the fit that did not see it.

    Why this matters: a slope fitted on in-sample predictions is biased
    towards 1 (or above) for any learner that fits its training data, so the
    calibration under-shrinks. Measured on 17 GCM hindcasts (MJJ, 0.1 deg):
    with in-sample slopes the MLP kept 36 % of the observed amplitude and
    scored MSSS -0.017; XGBoost got a slope of 1.2 and MSSS -0.102. The
    out-of-sample slope is the honest one.

    Returns ``(pa, oa)``: predicted and observed ANOMALIES, each block being
    centred on its own inner-training climatology -- the prediction on the
    inner fit's own in-sample mean, the observation on the inner-training
    observed mean. Re-centring the recombined blocks globally would inject
    the fold-recombination artefact (a block trained without a wet year
    predicts low while that year observes high), which drives the pooled
    slope negative: the first implementation returned exactly 0.0 on every
    fold because of it.
    """
    n = int(y_train.sizes["T"])
    block = max(1, int(np.ceil(n / max(2, int(n_folds)))))
    saved_models = dict(models_by_cluster)
    saved_flag = model.amplitude_calibration
    model.amplitude_calibration = False
    pa_parts, oa_parts = [], []
    try:
        for start in range(0, n, block):
            test = np.arange(start, min(start + block, n))
            train = np.setdiff1d(np.arange(n), test)
            if train.size < 5:
                continue
            y_in = y_train.isel(T=train)
            params = _regional_params_from(model, best_params, y_in)
            # one fit, one prediction of BOTH the inner-training years (for
            # the fit's own climatology) and the held-out block
            both = model.compute_model(
                raw_X_train.isel(T=train), y_in, raw_X_train.isel(T=np.r_[train, test]),
                clim_year_start=clim_year_start, clim_year_end=clim_year_end, **params)
            both = both.transpose("T", "Y", "X")
            level = both.isel(T=slice(0, train.size)).mean("T")
            pa_parts.append(both.isel(T=slice(train.size, None)) - level)
            oa_parts.append(y_train.isel(T=test) - y_in.mean("T"))
    finally:
        model.amplitude_calibration = saved_flag
        # the inner refits overwrote the per-cluster estimators: restore the
        # outer fit so that later diagnostics / forecast see the right models
        models_by_cluster.clear()
        models_by_cluster.update(saved_models)
    if not pa_parts:
        return None
    pa = xr.concat(pa_parts, dim="T").sortby("T").transpose("T", "Y", "X")
    oa = xr.concat(oa_parts, dim="T").sortby("T").transpose("T", "Y", "X")
    return pa, oa


def _anomaly_slope(pa, oa, bounds=AMPLITUDE_SLOPE_BOUNDS):
    """Pooled slope of observed on predicted anomalies (already centred)."""
    finite = np.isfinite(pa) & np.isfinite(oa)
    cov = float((pa.where(finite) * oa.where(finite)).sum(skipna=True))
    var = float((pa.where(finite) ** 2).sum(skipna=True))
    if not np.isfinite(cov) or not np.isfinite(var) or var <= 1e-12:
        return 0.0
    return float(np.clip(cov / var, bounds[0], bounds[1]))


def _calibrate_amplitude(model, models_by_cluster, X_train, y_train,
                         prediction, cluster_da, X_test=None, raw_predictors=None,
                         best_params=None, clim_year_start=None, clim_year_end=None):
    """Rescale ``prediction`` by a slope fitted on the training fold only.

    ``model.amplitude_calibration_mode`` selects the slope:

    * ``"global"`` -- one slope for the whole fold (default);
    * ``"spread"`` -- a slope per forecast year driven by the year's
      ensemble signal-to-spread ratio (needs ``X_test``; falls back to
      ``"global"`` without it and records the fallback).

    ``model.amplitude_calibration_inner_folds`` (default 3) estimates the
    slope on OUT-OF-SAMPLE predictions of the training years (blocked inner
    CV, see ``_blocked_inner_predictions``); 0 uses the in-sample prediction,
    which is cheaper but biased towards 1.

    Also guards against a degenerate fit -- a learner that does not even
    reproduce its own training data while emitting near-full observed
    variance, i.e. broadcasting noise at full amplitude. hpELM showed exactly
    this on the validation case (in-sample r = 0.003, amplitude 0.99,
    MSSS = -6.82). The slope collapses to ~0 there, which is the intended
    fallback to climatology; the warning makes the event visible.
    """
    insample = _pooled_grid_prediction(models_by_cluster, X_train, cluster_da)
    _, correlation, amplitude = _amplitude_slope(insample, y_train)
    mode = getattr(model, "amplitude_calibration_mode", "global") or "global"
    inner_folds = int(getattr(model, "amplitude_calibration_inner_folds", 3) or 0)

    # anomalies used to fit the slope: out-of-sample when possible, each
    # inner block centred on its own training climatology
    anomalies = None
    basis = "in_sample"
    if inner_folds >= 2 and best_params is not None and raw_predictors is not None:
        anomalies = _blocked_inner_predictions(
            model, models_by_cluster, raw_predictors[0], y_train, best_params,
            clim_year_start, clim_year_end, inner_folds)
        if anomalies is not None:
            basis = f"blocked_inner_cv_{inner_folds}"
    if anomalies is None:
        anomalies = (insample - insample.mean("T"), y_train - y_train.mean("T"))
    pa_ref, oa_ref = anomalies
    slope = _anomaly_slope(pa_ref, oa_ref)

    diagnostics = dict(mode=mode, slope=slope, slope_basis=basis,
                       insample_correlation=correlation, insample_amplitude=amplitude)

    if abs(correlation) < DEGENERATE_INSAMPLE_R and amplitude > DEGENERATE_AMPLITUDE:
        warnings.warn(
            f"{type(model).__name__}: in-sample correlation {correlation:.3f} with "
            f"amplitude {amplitude:.2f} -- the fit does not reproduce its own training "
            f"data yet emits near-full observed variance. Amplitude calibration shrinks "
            f"it to {slope:.3f} (towards climatology).",
            RuntimeWarning, stacklevel=2)

    baseline = y_train.mean("T")
    pa_test = prediction - insample.mean("T")

    if mode == "spread" and X_test is not None and slope > 0:
        try:
            z_train, signal_clim = _signal_to_spread_ratio(X_train)
            z_test, _ = _signal_to_spread_ratio(X_test, signal_clim)
            # yearly slopes on the same (out-of-sample when available) basis
            z_train = z_train.sel(T=pa_ref["T"])
            slopes, fit = _spread_conditioned_slope(pa_ref, oa_ref, z_train, z_test)
            per_year = xr.DataArray(slopes, dims=("T",), coords={"T": prediction["T"]})
            diagnostics.update(fit, z_train=np.asarray(z_train.values, dtype=float).tolist(),
                               z_test=np.asarray(z_test.values, dtype=float).tolist(),
                               slopes_test=np.asarray(slopes, dtype=float).tolist())
            calibrated = baseline + per_year * pa_test
        except ValueError as exc:
            # no usable spread in the cube: keep the global slope, say so
            diagnostics.update(fallback=f"global ({exc})")
            calibrated = baseline + slope * pa_test
    else:
        if mode == "spread":
            diagnostics.update(fallback="global (no X_test or zero slope)")
        calibrated = baseline + slope * pa_test

    model.amplitude_calibration_ = diagnostics
    # baseline est (Y, X) : sans transpose, la diffusion xarray renvoie
    # (Y, X, T) et casse le contrat (T, Y, X) attendu en aval
    return calibrated.transpose(*prediction.dims)


def _maybe_calibrate_amplitude(model, models_by_cluster, X_train, y_train,
                               prediction, cluster_da, X_test=None, raw_predictors=None,
                               best_params=None, clim_year_start=None, clim_year_end=None):
    """Apply the calibration when the learner has it enabled."""
    if not getattr(model, "amplitude_calibration", False):
        return prediction
    return _calibrate_amplitude(model, models_by_cluster, X_train, y_train,
                                prediction, cluster_da, X_test=X_test,
                                raw_predictors=raw_predictors, best_params=best_params,
                                clim_year_start=clim_year_start, clim_year_end=clim_year_end)


def _effective_spatial_dof(field):
    """Bretherton et al. (1999) effective spatial degrees of freedom.

    N_eff = (sum lambda)^2 / sum lambda^2 of the spatial covariance, computed
    from the T x T Gram matrix (same non-zero spectrum) on cells complete in
    time. Standardized anomalies are expected.
    """
    if not isinstance(field, xr.DataArray) or "T" not in field.dims:
        return np.nan
    if "M" in field.dims:
        field = field.isel(M=0, drop=True)
    a = np.asarray(field.transpose("T", "Y", "X").values, dtype=float).reshape(field.sizes["T"], -1)
    good = np.isfinite(a).all(axis=0)
    a = a[:, good]
    if a.shape[0] < 2 or a.shape[1] < 2:
        return np.nan
    a = a - a.mean(axis=0, keepdims=True)
    lam = np.linalg.eigvalsh(a @ a.T / a.shape[1])
    lam = lam[lam > 1e-12 * max(lam.max(), 1e-300)]
    if lam.size == 0:
        return np.nan
    return float(lam.sum() ** 2 / np.sum(lam ** 2))


def _sample_size_report(learner, n_rows, n_years, n_features, n_eff_spatial, verbose=True):
    """One line on the true sample size of a pooled fit; warn when it is small."""
    n_eff_total = n_years * n_eff_spatial if np.isfinite(n_eff_spatial) else np.nan
    if verbose:
        print(f"[{learner}] rows={int(n_rows)} years={int(n_years)} features={int(n_features)} "
              f"N_eff_spatial={n_eff_spatial:.1f} N_eff_total~{n_eff_total:.0f} "
              f"(rows/N_eff~{(n_rows / n_eff_total) if np.isfinite(n_eff_total) and n_eff_total > 0 else np.nan:.0f})",
              flush=True)
    if int(n_years) < 12:
        warnings.warn(
            f"{learner}: only {n_years} independent years (N_eff_total~{n_eff_total:.0f} for "
            f"{n_rows} rows). Treat skill as provisional; keep the outer year-wise validation.",
            RuntimeWarning, stacklevel=2)
    return n_eff_total


# Hyperparameters ordered by complexity for the one-standard-error rule:
# +1 = larger value means a more flexible model, -1 = larger means more regularized.
_COMPLEXITY_DIRECTION = {
    'n_estimators': 1, 'max_depth': 1, 'min_samples_split': -1, 'min_samples_leaf': -1,
    'max_features': 1, 'max_samples': 1, 'min_impurity_decrease': -1, 'ccp_alpha': -1,
    'min_weight_fraction_leaf': -1, 'max_leaf_nodes': 1, 'learning_rate': 1,
    'min_child_weight': -1, 'subsample': 1, 'colsample_bytree': 1, 'gamma': -1,
    'reg_alpha': -1, 'reg_lambda': -1, 'neurons': 1, 'norm': -1, 'hidden_layer_sizes': 1,
    'alpha': -1, 'learning_rate_init': 1, 'C': 1, 'epsilon': -1, 'n_components': 1,
    'degree': 1, 'max_terms': 1, 'penalty': -1, 'max_bins': 1, 'l2_regularization': -1,
    'max_iter_no_improve': 0, 'max_iter': 0, 'l1_ratio': 0,
}
_MAX_FEATURES_ALIASES = {'sqrt': 0.5, 'log2': 0.4, 'auto': 1.0, None: 1.0}


def _complexity_value(name, value):
    if value is None:
        return np.inf if name in ('max_depth', 'max_leaf_nodes') else 1.0
    if isinstance(value, (tuple, list)):
        return float(sum(float(v) for v in value))
    if isinstance(value, str):
        return float(_MAX_FEATURES_ALIASES.get(value, np.nan))
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _one_se_select(records, score_key='score', year_key='year_scores'):
    """Most regularized candidate within one standard error of the best score.

    ``records`` are dicts with ``params`` (dict), ``score`` (float, higher is
    better) and ``year_scores`` (per validation year). Complexity is a rank
    aggregate over the known hyperparameters (:data:`_COMPLEXITY_DIRECTION`);
    ties keep the better score. Returns ``(chosen_record, info)``.
    """
    from scipy.stats import rankdata
    best = max(records, key=lambda r: r[score_key])
    years = np.asarray(best.get(year_key) or [], dtype=float)
    years = years[np.isfinite(years)]
    se = float(years.std(ddof=1) / np.sqrt(len(years))) if len(years) > 1 else 0.0
    threshold = best[score_key] - se
    eligible = [r for r in records if r[score_key] >= threshold]
    info = {'rule': 'one_se', 'best_score': float(best[score_key]), 'standard_error': se,
            'n_eligible': len(eligible), 'n_candidates': len(records)}
    if len(eligible) <= 1:
        return best, info
    names = sorted({k for r in eligible for k in r['params']})
    complexity = np.zeros(len(eligible))
    for name in names:
        direction = _COMPLEXITY_DIRECTION.get(name, 0)
        if not direction:
            continue
        vals = np.asarray([_complexity_value(name, r['params'].get(name)) for r in eligible], dtype=float)
        finite = np.isfinite(vals)
        if finite.sum() < 2 or np.nanmin(vals[finite]) == np.nanmax(vals[finite]) and not np.isinf(vals).any():
            continue
        filled = vals.copy()
        top = np.nanmax(vals[finite]) if finite.any() else 0.0
        filled[np.isposinf(vals)] = top + 1.0
        filled[~np.isfinite(filled)] = np.nanmedian(filled[np.isfinite(filled)])
        complexity += direction * rankdata(filled, method='average') / len(eligible)
    order = sorted(range(len(eligible)), key=lambda i: (complexity[i], -eligible[i][score_key]))
    chosen = eligible[order[0]]
    info['chosen_score'] = float(chosen[score_key])
    return chosen, info


def _assign_target_time(da, predictor_for_year, predictant):
    """Assign forecast year and predictand target month to a singleton T axis."""
    if not isinstance(da, xr.DataArray) or "T" not in da.dims or da.sizes["T"] != 1:
        return da
    candidate = None
    if isinstance(predictor_for_year, xr.DataArray):
        for coord_name in ("T", "S"):
            if coord_name in predictor_for_year.coords and predictor_for_year.coords[coord_name].size:
                candidate = predictor_for_year.coords[coord_name].values[0]
                break
    try:
        year = int(np.asarray(candidate).astype("datetime64[Y]").astype(int) + 1970)
        obs_t = predictant["T"].values[0]
        month = int(np.asarray(obs_t).astype("datetime64[M]").astype(int) % 12 + 1)
        target = np.datetime64(f"{year:04d}-{month:02d}-01", "ns")
        return da.assign_coords(T=xr.IndexVariable("T", [target]))
    except (TypeError, ValueError, OverflowError, KeyError, AttributeError):
        return da


def _mean_if_member(da):
    """Return a deterministic (T,Y,X) field from deterministic or member data."""
    if "M" in da.dims:
        da = da.mean("M", skipna=True)
    return da.transpose("T", "Y", "X")

# Suppress the numerical noise of numpy/xarray reductions (all-NaN slices,
# 0/0 on empty pixels, ...) WITHOUT silencing the scientific RuntimeWarnings
# raised explicitly by this module (_effective_sample_warning, empty HPELM
# clusters, standardized-predictand clustering, ...). The previous blanket
# ``filterwarnings('ignore', category=RuntimeWarning)`` muted every one of them
# for the whole process.
for _pattern in (r'invalid value encountered', r'divide by zero encountered',
                 r'overflow encountered', r'Mean of empty slice',
                 r'Degrees of freedom <= 0', r'All-NaN slice encountered',
                 r'underflow encountered'):
    warnings.filterwarnings('ignore', message=_pattern, category=RuntimeWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)


def process_datasets_for_mme(
    rainfall,
    hdcsted=None,
    fcsted=None,
    gcm=False,
    agroparam=False,
    Prob=False,
    hydro=False,
    *,
    dir_to_save_model=None,
    best_models=None,
    scores=None,
    year_start=None,
    year_end=None,
    model=False,
    month_of_initialization=None,
    lead_time=None,
    year_forecast=None,
    score_metric="GROC",
    var="PRCP",
    ensemble_mean="mean",
    ensemble_features=("mean", "std", "q10", "q50", "q90"),
    ensemble_std_ddof=0,
    ensemble_min_members=2,
):
    """Prepare deterministic or probabilistic fields for an MME.

    The function guarantees the following output conventions:

    * deterministic fields: ``(T, M, Y, X)``;
    * probabilistic fields: ``(probability, T, M, Y, X)``;
    * observations: ``(T, M, Y, X)`` with a singleton ``M`` dimension;
    * score fields: one strictly spatial ``(Y, X)`` array per model.

    With ``ensemble_mean='mean'`` (default), M indexes GCM means and existing
    deterministic inputs are accepted. The legacy ``'median'`` mode also remains
    available as one channel per GCM. With ``ensemble_mean=None``, raw members
    are summarized separately for each GCM, time and native-grid cell, BEFORE
    interpolation. M then indexes ``GCM__statistic`` predictor channels. The
    default features are mean, population std (ddof=0), q10, q50 and q90. At
    least two finite members are required by default; missing values remain
    masked. Configure the ordered feature list, ddof and minimum count with
    ``ensemble_features``, ``ensemble_std_ddof``, ``ensemble_min_members``.
    Already averaged inputs are rejected in statistics mode. Hindcast and
    forecast member counts may differ, but output channel schemas must match.

    Statistical channels are regression INPUTS, not physical model forecasts:
    they must not be averaged as a weighted/probabilistic MME. ``Prob=True``
    is therefore incompatible with statistics mode. Score dictionaries remain
    keyed by original GCM names, never duplicated as feature-channel weights.

    In particular, a score carrying an ``M`` dimension is explicitly reduced
    to the requested model before it can be used. This prevents accidental
    xarray broadcasting from reintroducing ``M`` into a deterministic MME.
    """

    from .was_ensemble_features import (
        ensemble_summary_fields, normalise_ensemble_mean, normalise_ensemble_features,
    )

    ensemble_mean = normalise_ensemble_mean(ensemble_mean)
    feature_names = normalise_ensemble_features(ensemble_features)
    statistics_mode = ensemble_mean is None
    if statistics_mode and Prob:
        raise ValueError("ensemble_mean=None creates regression features; Prob=True cannot average distribution statistics as probabilities.")

    def _as_dataarray(obj, label):
        if isinstance(obj, xr.DataArray):
            return obj
        if isinstance(obj, xr.Dataset):
            if len(obj.data_vars) == 1:
                return next(iter(obj.data_vars.values()))
            if "Observation" in obj:
                return obj["Observation"]
            raise ValueError(
                f"{label} is a Dataset with several variables: "
                f"{list(obj.data_vars)}. Select one variable before MME processing."
            )
        raise TypeError(f"{label} must be an xarray DataArray or Dataset, got {type(obj)!r}.")

    def _open_dataarray(path, label):
        ds = xr.open_dataset(path)
        try:
            da = _as_dataarray(ds, label).load()
        finally:
            ds.close()
        return da

    def _normalise_name(name):
        text = str(name).lower().replace(f".{str(var).lower()}", "")
        return "".join(ch for ch in text if ch.isalnum())

    def _select_observation(da):
        da = _as_dataarray(da, "rainfall")
        if "M" in da.dims:
            if da.sizes["M"] != 1:
                warnings.warn(
                    "rainfall contains several M members; the first member is used as observation.",
                    RuntimeWarning,
                )
            da = da.isel(M=0, drop=True)
        if "M" in da.coords and "M" not in da.dims:
            da = da.drop_vars("M")
        missing = {"T", "Y", "X"} - set(da.dims)
        if missing:
            raise ValueError(f"rainfall is missing required dimensions {sorted(missing)}; dims={da.dims}.")
        extras = [d for d in da.dims if d not in ("T", "Y", "X")]
        for dim in extras:
            if da.sizes[dim] == 1:
                da = da.isel({dim: 0}, drop=True)
            else:
                raise ValueError(
                    f"rainfall has unsupported non-singleton dimension {dim!r}: "
                    f"dims={da.dims}, sizes={dict(da.sizes)}."
                )
        return da.transpose("T", "Y", "X")

    def _strip_per_model_member(da, model_name, label):
        if "M" not in da.dims:
            return da
        if da.sizes["M"] == 1:
            return da.isel(M=0, drop=True)
        if "M" in da.coords and model_name in da["M"].values.tolist():
            return da.sel(M=model_name, drop=True)
        warnings.warn(
            f"{label} for {model_name!r} still contains {da.sizes['M']} ensemble members; "
            "their mean is used before concatenating models.",
            RuntimeWarning,
        )
        return da.mean("M", skipna=True)

    def _prepare_field(da, model_name, label, probabilistic=False):
        da = _as_dataarray(da, label).astype(float)
        da = _strip_per_model_member(da, model_name, label)

        allowed = {"T", "Y", "X"}
        if probabilistic:
            allowed.add("probability")

        for dim in list(da.dims):
            if dim not in allowed:
                if da.sizes[dim] == 1:
                    da = da.isel({dim: 0}, drop=True)
                else:
                    raise ValueError(
                        f"{label} for {model_name!r} has unsupported dimension {dim!r}; "
                        f"dims={da.dims}, sizes={dict(da.sizes)}."
                    )

        required = {"T", "Y", "X"} | ({"probability"} if probabilistic else set())
        missing = required - set(da.dims)
        if missing:
            raise ValueError(
                f"{label} for {model_name!r} is missing dimensions {sorted(missing)}; dims={da.dims}."
            )

        if not da.Y.identical(obs.Y) or not da.X.identical(obs.X):
            method = "nearest" if probabilistic or da.attrs.get("ensemble_statistic") == "count" else "linear"
            da = da.interp(
                Y=obs.Y,
                X=obs.X,
                method=method,
                kwargs={"fill_value": "extrapolate"},
            )

        order = ("probability", "T", "Y", "X") if probabilistic else ("T", "Y", "X")
        return da.transpose(*order)

    def _find_path(mapping, model_name, label):
        if not mapping:
            raise ValueError(f"No {label} mapping was supplied.")
        if model_name in mapping:
            return mapping[model_name]
        target = _normalise_name(model_name)
        candidates = [value for key, value in mapping.items() if _normalise_name(key) == target]
        if not candidates:
            candidates = [
                value for value in mapping.values()
                if target[:5] and target[:5] in _normalise_name(value)
            ]
        if len(candidates) != 1:
            raise KeyError(
                f"Unable to identify one {label} file for model {model_name!r}; "
                f"found {len(candidates)} candidates."
            )
        return candidates[0]

    def _score_pool_from_input(score_input):
        if score_input is None:
            return {}
        if score_metric in score_input and isinstance(score_input[score_metric], dict):
            return score_input[score_metric]
        if all(isinstance(v, xr.DataArray) for v in score_input.values()):
            return score_input
        return {}

    def _match_score(model_name, score_pool):
        if model_name in score_pool:
            return score_pool[model_name]

        target = _normalise_name(model_name)
        exact = [(key, value) for key, value in score_pool.items() if _normalise_name(key) == target]
        if len(exact) == 1:
            return exact[0][1]

        prefix = [
            (key, value)
            for key, value in score_pool.items()
            if _normalise_name(key).startswith(target) or target.startswith(_normalise_name(key))
        ]
        if len(prefix) == 1:
            return prefix[0][1]
        if len(prefix) > 1:
            prefix.sort(key=lambda item: abs(len(_normalise_name(item[0])) - len(target)))
            best_distance = abs(len(_normalise_name(prefix[0][0])) - len(target))
            tied = [p for p in prefix if abs(len(_normalise_name(p[0])) - len(target)) == best_distance]
            if len(tied) == 1:
                return tied[0][1]
        return None

    def _prepare_score(score, model_name, model_index):
        if score is None:
            warnings.warn(
                f"No {score_metric} score found for {model_name!r}; a neutral weight of 1 is used.",
                RuntimeWarning,
            )
            return xr.ones_like(mask, dtype=float)

        score = _as_dataarray(score, f"score[{model_name}]").astype(float)

        if "M" in score.dims:
            if score.sizes["M"] == 1:
                score = score.isel(M=0, drop=True)
            elif "M" in score.coords and model_name in score["M"].values.tolist():
                score = score.sel(M=model_name, drop=True)
            elif score.sizes["M"] == len(selected_models):
                score = score.isel(M=model_index, drop=True)
            else:
                raise ValueError(
                    f"Score for {model_name!r} contains an ambiguous M dimension: "
                    f"dims={score.dims}, sizes={dict(score.sizes)}."
                )

        for dim in list(score.dims):
            if dim not in ("Y", "X"):
                if score.sizes[dim] == 1:
                    score = score.isel({dim: 0}, drop=True)
                else:
                    warnings.warn(
                        f"Score for {model_name!r} contains dimension {dim!r}; "
                        "it is averaged before spatial weighting.",
                        RuntimeWarning,
                    )
                    score = score.mean(dim, skipna=True)

        if not {"Y", "X"}.issubset(score.dims):
            if score.ndim == 0:
                score = xr.full_like(mask, float(score.values), dtype=float)
            else:
                raise ValueError(
                    f"Score for {model_name!r} must resolve to (Y, X), got {score.dims}."
                )
        else:
            score = score.transpose("Y", "X").interp(
                Y=obs.Y,
                X=obs.X,
                method="nearest",
                kwargs={"fill_value": "extrapolate"},
            )

        return score.where(mask.notnull())

    obs = _select_observation(rainfall)
    mask = xr.where(obs.notnull().any("T"), 1.0, np.nan).transpose("Y", "X")
    mask.name = None

    hdcsted = {} if hdcsted is None else dict(hdcsted)
    fcsted = {} if fcsted is None else dict(fcsted)

    if best_models is None:
        if gcm:
            best_models = list(hdcsted) if hdcsted else []
        else:
            best_models = [name for name in hdcsted if name in fcsted]
    selected_models = list(dict.fromkeys(best_models))
    if not selected_models:
        raise ValueError("best_models is empty; no model can be prepared for the MME.")

    all_model_hdcst = {}
    all_model_fcst = {}

    for model_name in selected_models:
        if gcm:
            def _load_gcm(mapping, forecast_year, label):
                # Explicit downloader mappings may point to a member-specific
                # cache directory/filename. Do not reconstruct a mean filename.
                if mapping:
                    source = _find_path(mapping, model_name, label)
                    if isinstance(source, (xr.DataArray, xr.Dataset)):
                        return source
                    return _open_dataarray(source, f"{label}[{model_name}]")
                return load_gridded_predictor(
                    dir_to_save_model, model_name, year_start, year_end,
                    model=True, month_of_initialization=month_of_initialization,
                    lead_time=lead_time, year_forecast=forecast_year,
                    preserve_members=True,
                )
            hdcst = _load_gcm(hdcsted, None, "hindcast")
            fcst = _load_gcm(fcsted, year_forecast, "forecast")
        elif agroparam:
            hdcst = _open_dataarray(
                _find_path(hdcsted, model_name, "hindcast"),
                f"hindcast[{model_name}]",
            )
            fcst = _open_dataarray(
                _find_path(fcsted, model_name, "forecast"),
                f"forecast[{model_name}]",
            )
        elif hydro and not isinstance(hdcsted.get(model_name), (xr.DataArray, xr.Dataset)):
            hdcst = _open_dataarray(hdcsted[model_name], f"hindcast[{model_name}]")
            fcst = _open_dataarray(fcsted[model_name], f"forecast[{model_name}]")
        else:
            if model_name not in hdcsted or model_name not in fcsted:
                raise KeyError(
                    f"Model {model_name!r} must exist in both hindcast and forecast dictionaries."
                )
            hdcst = hdcsted[model_name]
            fcst = fcsted[model_name]

        if Prob:
            hdcst = _prepare_field(hdcst, model_name, "hindcast", probabilistic=True)
            fcst = _prepare_field(fcst, model_name, "forecast", probabilistic=True)
        else:
            prepared = []
            for value, label in ((hdcst, "hindcast"), (fcst, "forecast")):
                value = _as_dataarray(value, label)
                # Preserve the old per-model selection for deterministic cubes
                # while member-statistics mode always requires raw member data.
                if not statistics_mode and "M" in value.dims and "M" in value.coords and model_name in value.M.values.tolist():
                    value = value.sel(M=model_name, drop=True)
                summaries = ensemble_summary_fields(
                    value, ensemble_mean=ensemble_mean, features=feature_names,
                    std_ddof=ensemble_std_ddof, min_members=ensemble_min_members,
                )
                fields = [_prepare_field(field, model_name, label) for field in summaries.values()]
                if statistics_mode:
                    channels = [f"{model_name}__{stat}" for stat in summaries]
                    result = xr.concat(fields, dim=xr.IndexVariable("M", channels),
                                       join="exact", coords="minimal", compat="override")
                    result = result.assign_coords(
                        source_model=("M", [str(model_name)] * len(channels)),
                        ensemble_statistic=("M", list(summaries)),
                        feature_units=("M", [str(field.attrs.get("units", "")) for field in fields]),
                    ).transpose("T", "M", "Y", "X")
                else:
                    result = fields[0]
                prepared.append(result)
            hdcst, fcst = prepared
            if statistics_mode and not hdcst.M.identical(fcst.M):
                raise ValueError(f"Hindcast/forecast ensemble feature channels differ for {model_name!r}.")

        # Seasonal stamps may differ within a year, but positional replacement
        # must never pair different years or conceal a reordered archive.
        if hdcst.sizes['T'] != obs.sizes['T'] or not np.array_equal(
                hdcst['T'].dt.year.values, obs['T'].dt.year.values):
            raise ValueError(f'Hindcast years/order differ from observations for {model_name!r}. Align explicitly before MME preparation.')
        if len(np.unique(hdcst['T'].dt.year.values)) != hdcst.sizes['T']:
            raise ValueError('MME preparation requires one seasonal target per year.')
        hdcst = hdcst.assign_coords(T=obs['T'])

        all_model_hdcst[model_name] = hdcst
        all_model_fcst[model_name] = fcst

    # Keep only models successfully loaded in the same explicit order.
    predictor_names = list(all_model_hdcst)
    if not predictor_names:
        raise ValueError(
            f"process_datasets_for_mme: {len(selected_models)} model(s) were "
            "requested but NONE survived loading/_prepare_field — the M-stacked "
            "cube would be empty. Requested: "
            f"{selected_models[:10]}{'...' if len(selected_models) > 10 else ''}. "
            "Check that each member's hindcast/forecast file exists and carries "
            "T/Y/X dims."
        )
    hdcst_list = [all_model_hdcst[name] for name in predictor_names]
    fcst_list = [all_model_fcst[name] for name in predictor_names]

    if Prob:
        hdcst_stack = xr.concat(
            hdcst_list,
            dim=xr.IndexVariable("M", predictor_names),
            join="exact",
            coords="minimal",
            compat="override",
        ).transpose("probability", "T", "M", "Y", "X")
        fcst_stack = xr.concat(
            fcst_list,
            dim=xr.IndexVariable("M", predictor_names),
            join="exact",
            coords="minimal",
            compat="override",
        ).transpose("probability", "T", "M", "Y", "X")

        hdcst_stack = hdcst_stack.where(mask.notnull())
        fcst_stack = fcst_stack.where(mask.notnull())
    elif statistics_mode:
        hdcst_stack = xr.concat(hdcst_list, dim="M", join="exact", coords="minimal", compat="override").transpose("T", "M", "Y", "X")
        fcst_stack = xr.concat(fcst_list, dim="M", join="exact", coords="minimal", compat="override").transpose("T", "M", "Y", "X")
        if len(set(hdcst_stack.M.values.tolist())) != hdcst_stack.sizes["M"]:
            raise ValueError("Ensemble feature channel names are not unique; rename the source GCM identifiers.")
        hdcst_stack = hdcst_stack.where(mask.notnull())
        fcst_stack = fcst_stack.where(mask.notnull())
    else:
        hdcst_stack = xr.concat(
            hdcst_list,
            dim=xr.IndexVariable("M", predictor_names),
            join="exact",
            coords="minimal",
            compat="override",
        ).transpose("T", "M", "Y", "X")
        fcst_stack = xr.concat(
            fcst_list,
            dim=xr.IndexVariable("M", predictor_names),
            join="exact",
            coords="minimal",
            compat="override",
        ).transpose("T", "M", "Y", "X")

        hdcst_stack = hdcst_stack.where(mask.notnull())
        fcst_stack = fcst_stack.where(mask.notnull())

    if not Prob:
        for cube in (hdcst_stack, fcst_stack):
            cube.attrs = dict(cube.attrs)
            cube.attrs.pop("ensemble_statistic", None)
            cube.attrs.update(
                wass2s_ensemble_mode="statistics" if statistics_mode else ensemble_mean,
                ensemble_features=",".join(feature_names) if statistics_mode else ensemble_mean,
                ensemble_std_ddof=int(ensemble_std_ddof),
                ensemble_min_members=int(ensemble_min_members) if statistics_mode else 1,
                ensemble_statistics_grid="native_before_interpolation",
            )
            # Preserve units from each GCM/statistic before concatenation:
            # the first channel may be dimensionless count, and GCM source
            # metadata may differ. Never infer all units from that first field.
            if statistics_mode:
                units = set(cube.feature_units.values.tolist())
                if len(units) == 1 and next(iter(units)):
                    cube.attrs["units"] = next(iter(units))
                else:
                    cube.attrs.pop("units", None)

    obs_m = (
        obs.where(mask.notnull())
        .expand_dims(M=[0])
        .transpose("T", "M", "Y", "X")
    )
    # Missing observations are never imputed for verification. Any estimator
    # imputation must be fitted inside its training fold, not in this assembler.

    score_pool = _score_pool_from_input(scores)
    scores_organized = {
        name: _prepare_score(_match_score(name, score_pool), name, index)
        for index, name in enumerate(predictor_names)
    }

    return hdcst_stack, fcst_stack, obs_m, scores_organized

def myfill(all_model_fcst, obs):

    """
    Fill missing values in forecast data using random samples from observations.

    This function fills NaN values in the forecast data by randomly sampling values from the observed 
    rainfall data along the time dimension.

    Parameters
    ----------
    all_model_fcst : xarray.DataArray
        Forecast data with dimensions (T, M, Y, X) containing possible NaN values.
    obs : xarray.DataArray
        Observed rainfall data with dimensions (T, Y, X) used for filling NaNs.

    Returns
    -------
    da_filled_random : xarray.DataArray
        Forecast data with NaN values filled using random samples from observations.
    """

    # Suppose all_model_hdcst has dimensions: T, M, Y, X
    da = all_model_fcst
    
    T = da.sizes["T"]
    Y = da.sizes["Y"]
    X = da.sizes["X"]
    
    # Create a DataArray of random T indices with shape (T, M, Y, X)
    # so that each element gets its own random index along T
    random_t_indices_full = xr.DataArray(
        np.random.randint(0, T, size=(T, Y, X)),
        dims=["T", "Y", "X"]
    )
    
    # Use vectorized indexing: for each (T, M, Y, X) location,
    # this picks the value at a random T index for that M, Y, X location.
    random_slices_full = obs.isel(T=random_t_indices_full)
    
    # Fill missing values with these randomly selected values
    da_filled_random = da.fillna(random_slices_full)
    return da_filled_random   


class WAS_mme_Weighted:
    """Equal- or skill-weighted deterministic MME with tercile probabilities.

    All public outputs are dimension-safe:

    * :meth:`compute` returns ``(T, Y, X)`` deterministic arrays;
    * :meth:`compute_prob` returns ``(probability, T, Y, X)``;
    * :meth:`forecast` returns the same deterministic/probabilistic conventions.

    Score arrays are reduced to ``(Y, X)`` before weighting, so an accidental
    score member dimension can never propagate into the deterministic result.
    """

    _PROBABILITY_LABELS = ("PB", "PN", "PA")

    def __init__(
        self,
        equal_weighted=False,
        dist_method="nonparam",
        metric="GROC",
        threshold=0.5,
    ):
        self.equal_weighted = bool(equal_weighted)
        self.dist_method = str(dist_method).lower()
        self.metric = str(metric)
        self.threshold = float(threshold)
        self.last_weights_ = None

    @staticmethod
    def _drop_scalar_coordinate(da, coord):
        if coord in da.coords and coord not in da.dims:
            da = da.drop_vars(coord)
        return da

    @classmethod
    def _as_observation(cls, da):
        if not isinstance(da, xr.DataArray):
            raise TypeError(f"Predictant/rainfall must be a DataArray, got {type(da)!r}.")
        if "M" in da.dims:
            if da.sizes["M"] != 1:
                warnings.warn(
                    "Observation contains several M members; the first member is used.",
                    RuntimeWarning,
                )
            da = da.isel(M=0, drop=True)
        da = cls._drop_scalar_coordinate(da, "M")
        return cls._strict_dims(da, ("T", "Y", "X"), "observation")

    @classmethod
    def _strict_dims(cls, da, expected, name):
        if not isinstance(da, xr.DataArray):
            raise TypeError(f"{name} must be a DataArray, got {type(da)!r}.")

        for dim in list(da.dims):
            if dim not in expected:
                if da.sizes[dim] == 1:
                    da = da.isel({dim: 0}, drop=True)
                else:
                    raise ValueError(
                        f"{name} must have dimensions {expected}; got dims={da.dims}, "
                        f"sizes={dict(da.sizes)}. Unexpected dimension: {dim!r}."
                    )

        missing = set(expected) - set(da.dims)
        if missing:
            raise ValueError(f"{name} is missing dimensions {sorted(missing)}; dims={da.dims}.")

        for coord in list(da.coords):
            if coord not in expected and coord not in da.dims:
                da = da.drop_vars(coord)
        return da.transpose(*expected)

    @classmethod
    def _as_ensemble(cls, da, name):
        require_physical_ensemble(da)
        return cls._strict_dims(da, ("T", "M", "Y", "X"), name)

    @classmethod
    def _as_deterministic(cls, da, name):
        return cls._strict_dims(da, ("T", "Y", "X"), name)

    @staticmethod
    def _normalise_forecast_time(forecast, observation):
        if "T" not in forecast.coords or forecast.sizes.get("T", 0) == 0:
            return forecast
        try:
            forecast_year = int(
                forecast["T"].values[0].astype("datetime64[Y]").astype(int) + 1970
            )
            obs_month = int(
                observation["T"].values[0].astype("datetime64[M]").astype(int) % 12 + 1
            )
            new_time = np.datetime64(f"{forecast_year:04d}-{obs_month:02d}-01", "ns")
            if forecast.sizes["T"] == 1:
                forecast = forecast.assign_coords(T=[new_time])
        except (TypeError, ValueError, AttributeError):
            pass
        return forecast

    def transform_score(self, score_array):
        """Convert a score field into non-negative spatial weights."""
        metric = self.metric.lower().replace("_", "").replace("-", "")
        score = score_array.astype(float)

        lower_is_better = {"mae", "rmse", "mse", "crps", "ignorance"}
        higher_is_better = {
            "pearson", "correlation", "groc", "rpss", "kge", "nse", "ioa", "roc"
        }

        if metric == "bias":
            weight = xr.where(np.abs(score) <= self.threshold, 1.0, 0.0)
        elif metric in lower_is_better:
            weight = xr.where(score <= self.threshold, 1.0, 0.0)
        elif metric in higher_is_better:
            weight = xr.where(score >= self.threshold, 1.0, 0.0)
        else:
            weight = score.clip(min=0.0)

        return weight.where(np.isfinite(weight), 0.0)

    @classmethod
    def _prepare_score(cls, score, model_name, obs, model_names, model_index):
        if score is None:
            warnings.warn(
                f"Missing score for {model_name!r}; a neutral weight of 1 is used.",
                RuntimeWarning,
            )
            return xr.ones_like(obs.isel(T=0, drop=True), dtype=float)

        if not isinstance(score, xr.DataArray):
            score = xr.DataArray(score)
        score = score.astype(float)

        if "M" in score.dims:
            if score.sizes["M"] == 1:
                score = score.isel(M=0, drop=True)
            elif "M" in score.coords and model_name in score.M.values.tolist():
                score = score.sel(M=model_name, drop=True)
            elif score.sizes["M"] == len(model_names):
                score = score.isel(M=model_index, drop=True)
            else:
                raise ValueError(
                    f"Score for {model_name!r} has an ambiguous M dimension: "
                    f"dims={score.dims}, sizes={dict(score.sizes)}."
                )

        for dim in list(score.dims):
            if dim not in ("Y", "X"):
                if score.sizes[dim] == 1:
                    score = score.isel({dim: 0}, drop=True)
                else:
                    warnings.warn(
                        f"Score dimension {dim!r} for {model_name!r} is averaged.",
                        RuntimeWarning,
                    )
                    score = score.mean(dim, skipna=True)

        template = obs.isel(T=0, drop=True)
        if score.ndim == 0:
            return xr.full_like(template, float(score.values), dtype=float)
        if not {"Y", "X"}.issubset(score.dims):
            raise ValueError(
                f"Score for {model_name!r} must resolve to (Y, X), got dims={score.dims}."
            )
        return score.transpose("Y", "X").interp(
            Y=obs.Y,
            X=obs.X,
            method="nearest",
            kwargs={"fill_value": "extrapolate"},
        )

    def compute(self, rainfall, hdcst, fcst, scores, complete=False):
        """Compute equal- or skill-weighted deterministic hindcast/forecast."""
        obs = self._as_observation(rainfall)
        hdcst = self._as_ensemble(hdcst, "hdcst")
        fcst = self._as_ensemble(fcst, "fcst")

        if hdcst.sizes["T"] != obs.sizes["T"]:
            raise ValueError(
                f"hdcst and observation must have the same T length; "
                f"got {hdcst.sizes['T']} and {obs.sizes['T']}."
            )
        hdcst = align_predictor_target(hdcst, obs, mode="season_year")
        fcst = self._normalise_forecast_time(fcst, obs)

        hdcst = hdcst.interp(
            Y=obs.Y, X=obs.X, method="nearest", kwargs={"fill_value": "extrapolate"}
        )
        fcst = fcst.interp(
            Y=obs.Y, X=obs.X, method="nearest", kwargs={"fill_value": "extrapolate"}
        )

        model_names = [str(value) for value in hdcst.M.values.tolist()]
        forecast_names = [str(value) for value in fcst.M.values.tolist()]
        if model_names != forecast_names:
            missing = set(model_names) - set(forecast_names)
            if missing:
                raise ValueError(f"Forecast is missing M members: {sorted(missing)}.")
            fcst = fcst.sel(M=hdcst.M)

        mask = xr.where(obs.notnull().any("T"), 1.0, np.nan)
        hdcst = hdcst.where(mask.notnull())
        fcst = fcst.where(mask.notnull())

        unweighted_h = hdcst.mean("M", skipna=True)
        unweighted_f = fcst.mean("M", skipna=True)

        if self.equal_weighted:
            weights = xr.ones_like(hdcst.isel(T=0, drop=True), dtype=float)
        else:
            scores = {} if scores is None else scores
            weight_fields = []
            for index, model_name in enumerate(model_names):
                score = scores.get(model_name)
                score = self._prepare_score(
                    score, model_name, obs, model_names, index
                )
                weight_fields.append(self.transform_score(score))
            weights = xr.concat(
                weight_fields,
                dim=xr.IndexVariable("M", hdcst.M.values),
            ).transpose("M", "Y", "X")

        weights = weights.where(mask.notnull(), 0.0).fillna(0.0)
        self.last_weights_ = weights

        h_weight = weights.where(hdcst.notnull())
        f_weight = weights.where(fcst.notnull())
        h_denom = h_weight.sum("M", skipna=True)
        f_denom = f_weight.sum("M", skipna=True)

        hindcast_det = (hdcst * weights).sum("M", skipna=True) / h_denom
        forecast_det = (fcst * weights).sum("M", skipna=True) / f_denom
        hindcast_det = hindcast_det.where(h_denom > 0)
        forecast_det = forecast_det.where(f_denom > 0)

        if complete:
            hindcast_det = hindcast_det.fillna(unweighted_h)
            forecast_det = forecast_det.fillna(unweighted_f)

        hindcast_det = self._as_deterministic(
            hindcast_det.where(mask.notnull()), "hindcast_det"
        )
        forecast_det = self._as_deterministic(
            forecast_det.where(mask.notnull()), "forecast_det"
        )
        return hindcast_det, forecast_det

    @staticmethod
    def _ppf_terciles_from_code(dist_code, shape, loc, scale):
        if not np.isfinite(dist_code):
            return np.nan, np.nan
        code = int(dist_code)
        try:
            if code == 1:
                return norm.ppf(1.0 / 3.0, loc=loc, scale=scale), norm.ppf(2.0 / 3.0, loc=loc, scale=scale)
            if code == 2:
                return lognorm.ppf(1.0 / 3.0, s=shape, loc=loc, scale=scale), lognorm.ppf(2.0 / 3.0, s=shape, loc=loc, scale=scale)
            if code == 3:
                return expon.ppf(1.0 / 3.0, loc=loc, scale=scale), expon.ppf(2.0 / 3.0, loc=loc, scale=scale)
            if code == 4:
                return gamma_dist.ppf(1.0 / 3.0, a=shape, loc=loc, scale=scale), gamma_dist.ppf(2.0 / 3.0, a=shape, loc=loc, scale=scale)
            if code == 5:
                return weibull_min.ppf(1.0 / 3.0, c=shape, loc=loc, scale=scale), weibull_min.ppf(2.0 / 3.0, c=shape, loc=loc, scale=scale)
            if code == 6:
                return tdist.ppf(1.0 / 3.0, df=shape, loc=loc, scale=scale), tdist.ppf(2.0 / 3.0, df=shape, loc=loc, scale=scale)
            if code == 7:
                return poisson.ppf(1.0 / 3.0, mu=shape, loc=loc), poisson.ppf(2.0 / 3.0, mu=shape, loc=loc)
            if code == 8:
                return nbinom.ppf(1.0 / 3.0, n=shape, p=scale, loc=loc), nbinom.ppf(2.0 / 3.0, n=shape, p=scale, loc=loc)
        except (ValueError, FloatingPointError, TypeError):
            pass
        return np.nan, np.nan

    @staticmethod
    def weibull_shape_solver(k, mean, variance):
        k = float(np.asarray(k).reshape(-1)[0])
        if k <= 0 or mean <= 0 or variance <= 0:
            return np.inf
        g1 = gamma_function(1.0 + 1.0 / k)
        g2 = gamma_function(1.0 + 2.0 / k)
        implied = g2 / (g1 * g1) - 1.0
        observed = variance / (mean * mean)
        return observed - implied

    @classmethod
    def calculate_tercile_probabilities_bestfit(
        cls, best_guess, error_variance, T1, T2, dist_code, dof
    ):
        """Compute PB/PN/PA for one grid cell and all requested times."""
        mean_values = np.asarray(best_guess, dtype=float)
        variance = float(np.asarray(error_variance))
        out = np.full((3, mean_values.size), np.nan, dtype=float)

        if (
            mean_values.size == 0
            or not np.isfinite(variance)
            or variance < 0
            or not np.isfinite(T1)
            or not np.isfinite(T2)
            or not np.isfinite(dist_code)
        ):
            return out

        eps = np.finfo(float).eps
        variance = max(variance, eps)
        code = int(dist_code)

        for index, mean_value in enumerate(mean_values):
            if not np.isfinite(mean_value):
                continue
            try:
                if code == 1:
                    cdf = lambda q: norm.cdf(q, loc=mean_value, scale=np.sqrt(variance))
                elif code == 2:
                    if mean_value <= 0:
                        continue
                    sigma2 = np.log1p(variance / max(mean_value * mean_value, eps))
                    sigma = np.sqrt(max(sigma2, eps))
                    scale_logn = np.exp(np.log(mean_value) - 0.5 * sigma2)
                    cdf = lambda q: lognorm.cdf(q, s=sigma, loc=0.0, scale=scale_logn)
                elif code == 3:
                    scale_exp = np.sqrt(variance)
                    loc_exp = mean_value - scale_exp
                    cdf = lambda q: expon.cdf(q, loc=loc_exp, scale=max(scale_exp, eps))
                elif code == 4:
                    if mean_value <= 0:
                        continue
                    alpha = max(mean_value * mean_value / variance, eps)
                    theta = max(variance / mean_value, eps)
                    cdf = lambda q: gamma_dist.cdf(q, a=alpha, loc=0.0, scale=theta)
                elif code == 5:
                    if mean_value <= 0:
                        continue
                    try:
                        shape = brentq(
                            lambda kval: cls.weibull_shape_solver(kval, mean_value, variance),
                            0.1,
                            100.0,
                            maxiter=100,
                        )
                    except (ValueError, RuntimeError):
                        shape = 2.0
                    scale_weib = mean_value / gamma_function(1.0 + 1.0 / shape)
                    cdf = lambda q: weibull_min.cdf(q, c=shape, loc=0.0, scale=max(scale_weib, eps))
                elif code == 6:
                    df = max(float(dof), 3.0)
                    scale_t = np.sqrt(variance * (df - 2.0) / df)
                    cdf = lambda q: tdist.cdf(q, df=df, loc=mean_value, scale=max(scale_t, eps))
                elif code == 7:
                    mu = max(mean_value, eps)
                    cdf = lambda q: poisson.cdf(q, mu=mu)
                elif code == 8:
                    if mean_value <= 0:
                        continue
                    if variance <= mean_value:
                        mu = max(mean_value, eps)
                        cdf = lambda q: poisson.cdf(q, mu=mu)
                    else:
                        p = np.clip(mean_value / variance, eps, 1.0 - eps)
                        n = max(mean_value * mean_value / (variance - mean_value), eps)
                        cdf = lambda q: nbinom.cdf(q, n=n, p=p)
                else:
                    continue

                c1 = float(cdf(T1))
                c2 = float(cdf(T2))
                probabilities = np.array([c1, c2 - c1, 1.0 - c2], dtype=float)
                probabilities = np.clip(probabilities, 0.0, 1.0)
                total = probabilities.sum()
                if total > 0 and np.isfinite(total):
                    out[:, index] = probabilities / total
            except (ValueError, FloatingPointError, OverflowError, ZeroDivisionError):
                continue

        return out

    @staticmethod
    def calculate_tercile_probabilities_nonparametric(
        best_guess, error_samples, first_tercile, second_tercile
    ):
        return _nonparam_tercile_numba(
            np.asarray(best_guess, dtype=np.float64),
            np.asarray(error_samples, dtype=np.float64),
            float(first_tercile), float(second_tercile),
        )

    @staticmethod
    def _prepare_spatial_parameter(parameter, obs, name):
        if parameter is None:
            return None
        if not isinstance(parameter, xr.DataArray):
            parameter = xr.DataArray(parameter)
        if "M" in parameter.dims:
            if parameter.sizes["M"] == 1:
                parameter = parameter.isel(M=0, drop=True)
            else:
                raise ValueError(f"{name} must not contain a non-singleton M dimension.")
        for dim in list(parameter.dims):
            if dim not in ("Y", "X"):
                if parameter.sizes[dim] == 1:
                    parameter = parameter.isel({dim: 0}, drop=True)
                else:
                    raise ValueError(f"{name} contains unsupported dimension {dim!r}.")
        if parameter.ndim == 0:
            return xr.full_like(obs.isel(T=0, drop=True), float(parameter.values))
        return parameter.transpose("Y", "X").interp(
            Y=obs.Y,
            X=obs.X,
            method="nearest",
            kwargs={"fill_value": "extrapolate"},
        )

    @classmethod
    def _normalise_probability_cube(cls, probability, mask):
        probability = probability.assign_coords(
            probability=("probability", list(cls._PROBABILITY_LABELS))
        ).transpose("probability", "T", "Y", "X")
        probability = probability.clip(min=0.0, max=1.0)
        total = probability.sum("probability", skipna=False)
        probability = probability / total.where(total > 0)
        return probability.where(mask.notnull())

    def _probability_core(
        self,
        obs,
        hindcast_det,
        target_det,
        clim_year_start,
        clim_year_end,
        best_code_da=None,
        best_shape_da=None,
        best_loc_da=None,
        best_scale_da=None,
    ):
        climatology = obs.sel(T=slice(str(clim_year_start), str(clim_year_end)))
        if climatology.sizes.get("T", 0) < 3:
            raise ValueError(
                "At least three observation years are required in the climatology period."
            )

        terciles = climatological_terciles(climatology)[0]
        t1_emp = terciles.isel(quantile=0, drop=True)
        t2_emp = terciles.isel(quantile=1, drop=True)
        errors = obs - hindcast_det
        # Residual dressing must not see the error of the year it dresses. When
        # the target IS the hindcast (same T axis, as in compute_prob) the error
        # of each target year is left out (leave-one-out); the residual variance
        # is the unbiased sample variance (ddof=1) in every case.
        leave_one_out = bool(
            target_det.sizes.get("T", 0) == errors.sizes["T"]
            and np.array_equal(target_det["T"].values, errors["T"].values)
        )
        error_variance = errors.var("T", skipna=True, ddof=1)
        if leave_one_out:
            # Exact ddof=1 variance of the sample without year j:
            # [S - n/(n-1) (e_j - ebar)^2] / (n - 2), S = sum_k (e_k - ebar)^2.
            # A year whose own error is missing keeps the full-sample variance.
            n_errors = errors.count("T")
            deviation2 = (errors - errors.mean("T", skipna=True)) ** 2
            with np.errstate(divide="ignore", invalid="ignore"):
                loo_variance = (deviation2.sum("T", skipna=True)
                                - deviation2 * n_errors / (n_errors - 1)) / (n_errors - 2)
            error_variance = xr.where(deviation2.notnull(), loo_variance,
                                      error_variance).where(n_errors > 2)
        # The forecast target may have a different T coordinate/length from the
        # historical error sample. Give the error sample its own core dimension
        # so xarray does not try to align forecast T with hindcast T.
        errors_for_ufunc = errors.rename(T="T_error")
        dof = max(int(climatology.sizes["T"]) - 1, 3)

        if self.dist_method == "bestfit":
            parameters = (best_code_da, best_shape_da, best_loc_da, best_scale_da)
            if any(value is None for value in parameters):
                raise ValueError(
                    "dist_method='bestfit' requires best_code_da, best_shape_da, "
                    "best_loc_da and best_scale_da."
                )
            code = self._prepare_spatial_parameter(best_code_da, obs, "best_code_da")
            shape = self._prepare_spatial_parameter(best_shape_da, obs, "best_shape_da")
            loc = self._prepare_spatial_parameter(best_loc_da, obs, "best_loc_da")
            scale = self._prepare_spatial_parameter(best_scale_da, obs, "best_scale_da")

            # Category boundaries stay on the observed climatological reference.
            # The selected distribution controls the predictive CDF only.
            if getattr(self, "tercile_reference", "climatology") == "training":
                t1, t2 = _pufunc(
                    self._ppf_terciles_from_code,
                    code,
                    shape,
                    loc,
                    scale,
                    input_core_dims=[[], [], [], []],
                    output_core_dims=[[], []],
                    vectorize=True,
                    dask="parallelized",
                    output_dtypes=[float, float],
                )
            else:
                t1, t2 = t1_emp, t2_emp
            kernel = self.calculate_tercile_probabilities_bestfit
            variance_dims = []
            if leave_one_out:
                variance_dims = ["T"]

                def kernel(best_guess, variances, T1, T2, dist_code, dof,
                           _kernel=self.calculate_tercile_probabilities_bestfit):
                    # One leave-one-out residual variance per dressed year.
                    best_guess = np.asarray(best_guess, dtype=float)
                    variances = np.asarray(variances, dtype=float)
                    out = np.full((3, best_guess.size), np.nan, dtype=float)
                    for j in range(best_guess.size):
                        out[:, j] = _kernel(best_guess[j:j + 1], variances[j],
                                            T1, T2, dist_code, dof)[:, 0]
                    return out

            probability = _pufunc(
                kernel,
                target_det,
                error_variance,
                t1,
                t2,
                code,
                input_core_dims=[["T"], variance_dims, [], [], []],
                output_core_dims=[["probability", "T"]],
                vectorize=True,
                dask="parallelized",
                kwargs={"dof": dof},
                output_dtypes=[float],
                dask_gufunc_kwargs={
                    "output_sizes": {"probability": 3},
                    "allow_rechunk": True,
                },
            )
        elif self.dist_method == "nonparam":
            kernel = self.calculate_tercile_probabilities_nonparametric
            if leave_one_out:

                def kernel(best_guess, error_samples, first_tercile, second_tercile,
                           _kernel=self.calculate_tercile_probabilities_nonparametric):
                    # Year j is dressed with every error but its own: the error
                    # sample keeps its own dimension and the j-th entry is set to
                    # NaN, which _nonparam_tercile_numba ignores.
                    best_guess = np.asarray(best_guess, dtype=np.float64)
                    error_samples = np.asarray(error_samples, dtype=np.float64)
                    out = np.full((3, best_guess.size), np.nan, dtype=float)
                    for j in range(best_guess.size):
                        sample = error_samples.copy()
                        sample[j] = np.nan
                        out[:, j] = _kernel(best_guess[j:j + 1], sample,
                                            first_tercile, second_tercile)[:, 0]
                    return out

            probability = _pufunc(
                kernel,
                target_det,
                errors_for_ufunc,
                t1_emp,
                t2_emp,
                input_core_dims=[["T"], ["T_error"], [], []],
                output_core_dims=[["probability", "T"]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
                dask_gufunc_kwargs={
                    "output_sizes": {"probability": 3},
                    "allow_rechunk": True,
                },
            )
        else:
            raise ValueError(
                f"Invalid dist_method {self.dist_method!r}; expected 'bestfit' or 'nonparam'."
            )

        mask = xr.where(obs.notnull().any("T"), 1.0, np.nan)
        return self._normalise_probability_cube(probability, mask)

    def compute_prob(
        self,
        Predictant,
        clim_year_start,
        clim_year_end,
        hindcast_det,
        best_code_da=None,
        best_shape_da=None,
        best_loc_da=None,
        best_scale_da=None,
    ):
        """Convert deterministic hindcasts into tercile probabilities."""
        obs = self._as_observation(Predictant)
        hindcast_det = self._as_deterministic(hindcast_det, "hindcast_det")
        if hindcast_det.sizes["T"] != obs.sizes["T"]:
            raise ValueError(
                "hindcast_det and Predictant must have the same number of T values."
            )
        hindcast_det = align_predictor_target(hindcast_det, obs, mode="season_year")
        hindcast_det = hindcast_det.interp(
            Y=obs.Y, X=obs.X, method="nearest", kwargs={"fill_value": "extrapolate"}
        )
        return self._probability_core(
            obs,
            hindcast_det,
            hindcast_det,
            clim_year_start,
            clim_year_end,
            best_code_da,
            best_shape_da,
            best_loc_da,
            best_scale_da,
        )

    def forecast(
        self,
        Predictant,
        clim_year_start,
        clim_year_end,
        hindcast_det,
        forecast_det,
        best_code_da=None,
        best_shape_da=None,
        best_loc_da=None,
        best_scale_da=None,
    ):
        """Convert a deterministic forecast into tercile probabilities."""
        obs = self._as_observation(Predictant)
        hindcast_det = self._as_deterministic(hindcast_det, "hindcast_det")
        forecast_det = self._as_deterministic(forecast_det, "forecast_det")

        if hindcast_det.sizes["T"] != obs.sizes["T"]:
            raise ValueError(
                "hindcast_det and Predictant must have the same number of T values."
            )
        hindcast_det = align_predictor_target(hindcast_det, obs, mode="season_year")
        forecast_det = self._normalise_forecast_time(forecast_det, obs)
        hindcast_det = hindcast_det.interp(
            Y=obs.Y, X=obs.X, method="nearest", kwargs={"fill_value": "extrapolate"}
        )
        forecast_det = forecast_det.interp(
            Y=obs.Y, X=obs.X, method="nearest", kwargs={"fill_value": "extrapolate"}
        )

        forecast_prob = self._probability_core(
            obs,
            hindcast_det,
            forecast_det,
            clim_year_start,
            clim_year_end,
            best_code_da,
            best_shape_da,
            best_loc_da,
            best_scale_da,
        )
        mask = xr.where(obs.notnull().any("T"), 1.0, np.nan)
        return forecast_det.where(mask.notnull()), forecast_prob

class WAS_mme_WeightedContinuous(WAS_mme_Weighted):
    """Continuous, shrinkage-stabilised skill-weighted deterministic MME.

    ``WAS_mme_Weighted`` is intentionally preserved unchanged for backward
    compatibility.  This companion avoids hard 0/1 jumps in skill and shrinks
    learned weights toward equal weighting -- a useful safeguard for seasonal
    hindcasts with only a few decades.

    Higher-is-better scores are measured relative to a no-skill reference
    (GROC/ROC: 0.5; RPSS/correlation/KGE/NSE: 0). Lower-is-better scores use a
    smooth inverse loss transform. ``shrinkage``=1 is equal weighting and 0 is
    pure data-derived weighting.
    """
    def __init__(self, dist_method="nonparam", metric="GROC", threshold=0.5,
                 power=1.0, shrinkage=0.20, loss_scale=None, neutral_score=None):
        super().__init__(equal_weighted=False, dist_method=dist_method,
                         metric=metric, threshold=threshold)
        if not (0.0 <= float(shrinkage) <= 1.0):
            raise ValueError("shrinkage must be in [0,1].")
        if float(power) <= 0:
            raise ValueError("power must be > 0.")
        self.power = float(power)
        self.shrinkage = float(shrinkage)
        self.loss_scale = loss_scale
        self.neutral_score = neutral_score

    def transform_score(self, score_array):
        metric = self.metric.lower().replace("_", "").replace("-", "")
        score = score_array.astype(float)
        lower = {"mae", "rmse", "mse", "crps", "ignorance"}
        if metric in lower:
            scale = float(self.loss_scale if self.loss_scale is not None else self.threshold)
            if not np.isfinite(scale) or scale <= 0:
                finite = score.where(np.isfinite(score))
                scale = float(finite.median(skipna=True)) if finite.count() else 1.0
            skill = 1.0 / (1.0 + (score.clip(min=0.0) / max(scale, 1e-12)) ** self.power)
        elif metric == "bias":
            scale = float(self.loss_scale if self.loss_scale is not None else self.threshold)
            skill = 1.0 / (1.0 + (np.abs(score) / max(scale, 1e-12)) ** self.power)
        else:
            if self.neutral_score is not None:
                neutral = float(self.neutral_score)
            elif metric in {"groc", "roc"}:
                neutral = 0.5
            else:
                neutral = 0.0
            denom = max(1.0 - neutral, 1e-12)
            skill = ((score - neutral) / denom).clip(min=0.0) ** self.power
        # Equal-weight shrinkage prevents unstable all-or-nothing maps.
        weight = (1.0 - self.shrinkage) * skill + self.shrinkage
        return weight.where(np.isfinite(weight), self.shrinkage).clip(min=0.0)


class WAS_ProbWeighted(WAS_mme_Weighted):
    """Skill-weighted combination of multi-model tercile probabilities.

    Parameters
    ----------
    metric : str, default="GROC"
        Skill metric represented by ``scores``. The stepwise transformation is
        designed for higher-is-better scores bounded approximately by 0 and 1.
    threshold : float, default=0.5
        Scores less than or equal to this value receive zero weight.
    moderate_break : float, default=0.6
        Upper score boundary of the moderate-skill class.
    high_break : float, default=0.8
        Upper score boundary of the high-skill class.
    moderate_weight : float, default=0.6
        Weight assigned when ``threshold < score <= moderate_break``.
    high_weight : float, default=0.8
        Weight assigned when ``moderate_break < score <= high_break``.
    top_weight : float, default=1.0
        Weight assigned when ``score > high_break``.
    missing_score : {"zero", "one", "raise"}, default="zero"
        Behaviour when no score can be associated with a model. ``"zero"``
        excludes it, ``"one"`` assigns full weight, and ``"raise"`` stops.
    normalize_probabilities : bool, default=True
        Normalize every complete probability vector before and after the MME.
        This also converts percentage inputs summing to 100 into fractions.

    Notes
    -----
    Expected ensemble dimensions are ``(probability, T, M, Y, X)``. Public
    outputs are always ``(probability, T, Y, X)``. A score is always reduced to
    ``(Y, X)`` before weighting, preventing an accidental ``M`` dimension from
    leaking into the result.
    """

    _EXPECTED_PROB_DIMS = ("probability", "T", "M", "Y", "X")
    _OUTPUT_PROB_DIMS = ("probability", "T", "Y", "X")

    def __init__(
        self,
        metric="GROC",
        threshold=0.5,
        moderate_break=0.6,
        high_break=0.8,
        moderate_weight=0.6,
        high_weight=0.8,
        top_weight=1.0,
        missing_score="zero",
        normalize_probabilities=True,
    ):
        super().__init__(
            equal_weighted=False,
            dist_method="nonparam",
            metric=metric,
            threshold=threshold,
        )

        self.moderate_break = float(moderate_break)
        self.high_break = float(high_break)
        self.moderate_weight = float(moderate_weight)
        self.high_weight = float(high_weight)
        self.top_weight = float(top_weight)
        self.missing_score = str(missing_score).lower()
        self.normalize_probabilities = bool(normalize_probabilities)

        if not np.isfinite(self.threshold):
            raise ValueError("threshold must be finite.")
        if not self.moderate_break < self.high_break:
            raise ValueError("moderate_break must be strictly smaller than high_break.")
        if min(self.moderate_weight, self.high_weight, self.top_weight) < 0:
            raise ValueError("All weights must be non-negative.")
        if self.missing_score not in {"zero", "one", "raise"}:
            raise ValueError("missing_score must be 'zero', 'one', or 'raise'.")

        self.last_effective_hindcast_weights_ = None
        self.last_effective_forecast_weights_ = None

    @classmethod
    def _as_probability_ensemble(cls, da, name):
        """Validate and order a probabilistic multi-model DataArray."""
        return cls._strict_dims(da, cls._EXPECTED_PROB_DIMS, name)

    @classmethod
    def _as_probability_output(cls, da, name):
        """Validate and order a probabilistic MME output."""
        return cls._strict_dims(da, cls._OUTPUT_PROB_DIMS, name)

    @staticmethod
    def _canonical_model_name(value):
        """Return a conservative normalized model identifier for matching."""
        return "".join(char for char in str(value).lower() if char.isalnum())

    @classmethod
    def _align_named_axis(cls, reference, other, dim, reference_name, other_name):
        """Reorder ``other[dim]`` to match ``reference[dim]`` by string labels."""
        reference_values = list(reference[dim].values)
        other_values = list(other[dim].values)

        reference_keys = [str(value) for value in reference_values]
        other_lookup = {}
        for value in other_values:
            key = str(value)
            if key in other_lookup:
                raise ValueError(
                    f"{other_name}.{dim} contains duplicate string label {key!r}."
                )
            other_lookup[key] = value

        if len(reference_keys) != len(set(reference_keys)):
            raise ValueError(
                f"{reference_name}.{dim} contains duplicate string labels: "
                f"{reference_keys}."
            )

        missing = [key for key in reference_keys if key not in other_lookup]
        if missing:
            raise ValueError(
                f"{other_name} is missing {dim} labels required by {reference_name}: "
                f"{missing}."
            )

        aligned = other.sel({dim: [other_lookup[key] for key in reference_keys]})
        return aligned.assign_coords({dim: reference[dim].values})

    def _score_pool(self, scores):
        """Accept either a direct model-score mapping or a metric-nested mapping."""
        if scores is None:
            return {}
        if isinstance(scores, xr.DataArray):
            return scores
        if not isinstance(scores, dict):
            raise TypeError(
                "scores must be a dict, a metric-nested dict, a DataArray, or None."
            )

        metric_keys = [self.metric, self.metric.upper(), self.metric.lower()]
        for key in metric_keys:
            candidate = scores.get(key)
            if isinstance(candidate, dict) or isinstance(candidate, xr.DataArray):
                return candidate
        return scores

    def _lookup_score(self, scores, model_name, model_index, model_names):
        """Resolve one model's score without silently selecting an unrelated model."""
        pool = self._score_pool(scores)

        if isinstance(pool, xr.DataArray):
            return pool
        if not pool:
            return None

        if model_name in pool:
            return pool[model_name]

        string_lookup = {str(key): key for key in pool}
        if model_name in string_lookup:
            return pool[string_lookup[model_name]]

        target = self._canonical_model_name(model_name)
        canonical_matches = [
            key
            for key in pool
            if self._canonical_model_name(key) == target
        ]
        if len(canonical_matches) == 1:
            return pool[canonical_matches[0]]
        if len(canonical_matches) > 1:
            raise ValueError(
                f"Ambiguous score keys for model {model_name!r}: "
                f"{[str(key) for key in canonical_matches]}."
            )

        # Prefix matching is only accepted when it identifies exactly one key.
        prefix_matches = [
            key
            for key in pool
            if self._canonical_model_name(key).startswith(target)
            or target.startswith(self._canonical_model_name(key))
        ]
        if len(prefix_matches) == 1:
            return pool[prefix_matches[0]]
        if len(prefix_matches) > 1:
            raise ValueError(
                f"Ambiguous prefix score keys for model {model_name!r}: "
                f"{[str(key) for key in prefix_matches]}."
            )

        return None

    def _missing_score_field(self, obs, model_name):
        template = obs.isel(T=0, drop=True)
        if self.missing_score == "raise":
            raise KeyError(f"No score was found for model {model_name!r}.")
        if self.missing_score == "one":
            warnings.warn(
                f"No score found for {model_name!r}; full weight is used.",
                RuntimeWarning,
            )
            return xr.ones_like(template, dtype=float)

        warnings.warn(
            f"No score found for {model_name!r}; the model is excluded.",
            RuntimeWarning,
        )
        return xr.zeros_like(template, dtype=float)

    def transform_score(self, score_array, threshold=None):
        """Convert a GROC-like skill field into stepwise non-negative weights."""
        cutoff = self.threshold if threshold is None else float(threshold)
        if not np.isfinite(cutoff):
            raise ValueError("threshold must be finite.")

        metric = self.metric.lower().replace("_", "").replace("-", "")
        if metric not in {"groc", "roc", "auc", "rpss", "pearson", "correlation"}:
            warnings.warn(
                f"Stepwise ProbWeighted transformation is being applied to metric "
                f"{self.metric!r}; verify that higher values indicate better skill.",
                RuntimeWarning,
            )

        score = score_array.astype(float)
        finite = np.isfinite(score)
        weight = xr.where(
            ~finite,
            0.0,
            xr.where(
                score <= cutoff,
                0.0,
                xr.where(
                    score <= self.moderate_break,
                    self.moderate_weight,
                    xr.where(
                        score <= self.high_break,
                        self.high_weight,
                        self.top_weight,
                    ),
                ),
            ),
        )
        return weight.clip(min=0.0)

    @staticmethod
    def _normalise_probability_vectors(da):
        """Normalize complete probability vectors and reject incomplete vectors."""
        complete_vector = da.notnull().all("probability")
        nonnegative = da.clip(min=0.0)
        total = nonnegative.sum("probability", skipna=False)
        valid = complete_vector & np.isfinite(total) & (total > 0)
        return (nonnegative / total).where(valid)

    def compute(
        self,
        rainfall,
        hdcst,
        fcst,
        scores,
        threshold=None,
        complete=False,
    ):
        """Compute a skill-weighted MME of tercile probabilities.

        Parameters
        ----------
        rainfall : xarray.DataArray
            Observation field with dimensions ``(T, Y, X)``. A singleton or
            multi-valued ``M`` dimension is accepted; the first member is used.
        hdcst, fcst : xarray.DataArray
            Probabilistic model ensembles with dimensions
            ``(probability, T, M, Y, X)`` in any order.
        scores : dict or xarray.DataArray
            Model skill fields. Every selected score is reduced to ``(Y, X)``.
        threshold : float, optional
            Runtime override of the instance exclusion threshold.
        complete : bool, default=False
            Fill cells with no positively weighted model using the unweighted
            multi-model probability mean.

        Returns
        -------
        hindcast_weighted, forecast_weighted : xarray.DataArray
            Arrays ordered as ``(probability, T, Y, X)``.
        """
        obs = self._as_observation(rainfall)
        hdcst = self._as_probability_ensemble(hdcst, "hdcst")
        fcst = self._as_probability_ensemble(fcst, "fcst")

        if hdcst.sizes["T"] != obs.sizes["T"]:
            raise ValueError(
                "hdcst and rainfall must have the same T length; "
                f"got {hdcst.sizes['T']} and {obs.sizes['T']}."
            )

        hdcst = align_predictor_target(hdcst, obs, mode="season_year")
        fcst = self._normalise_forecast_time(fcst, obs)

        # Forecast model and probability axes must match the hindcast exactly.
        fcst = self._align_named_axis(hdcst, fcst, "M", "hdcst", "fcst")
        fcst = self._align_named_axis(
            hdcst, fcst, "probability", "hdcst", "fcst"
        )

        hdcst = hdcst.interp(
            Y=obs.Y,
            X=obs.X,
            method="nearest",
            kwargs={"fill_value": "extrapolate"},
        )
        fcst = fcst.interp(
            Y=obs.Y,
            X=obs.X,
            method="nearest",
            kwargs={"fill_value": "extrapolate"},
        )

        spatial_mask = xr.where(obs.notnull().any("T"), 1.0, np.nan)
        hdcst = hdcst.where(spatial_mask.notnull())
        fcst = fcst.where(spatial_mask.notnull())

        if self.normalize_probabilities:
            hdcst = self._normalise_probability_vectors(hdcst)
            fcst = self._normalise_probability_vectors(fcst)

        model_values = list(hdcst.M.values)
        model_names = [str(value) for value in model_values]
        if len(model_names) != len(set(model_names)):
            raise ValueError(f"hdcst.M contains duplicate model labels: {model_names}.")

        weight_fields = []
        for index, model_name in enumerate(model_names):
            raw_score = self._lookup_score(
                scores,
                model_name=model_name,
                model_index=index,
                model_names=model_names,
            )
            if raw_score is None:
                score = self._missing_score_field(obs, model_name)
            else:
                score = self._prepare_score(
                    raw_score,
                    model_name=model_name,
                    obs=obs,
                    model_names=model_names,
                    model_index=index,
                )
            weight_fields.append(self.transform_score(score, threshold=threshold))

        weights = xr.concat(
            weight_fields,
            dim=xr.IndexVariable("M", model_values),
        ).transpose("M", "Y", "X")
        weights = weights.where(spatial_mask.notnull(), 0.0).fillna(0.0)
        self.last_weights_ = weights

        # A model is used only when its entire PB/PN/PA vector is available.
        valid_hindcast_vector = hdcst.notnull().all("probability")
        valid_forecast_vector = fcst.notnull().all("probability")

        effective_hindcast_weights = weights.where(valid_hindcast_vector)
        effective_forecast_weights = weights.where(valid_forecast_vector)
        self.last_effective_hindcast_weights_ = effective_hindcast_weights
        self.last_effective_forecast_weights_ = effective_forecast_weights

        hindcast_denominator = effective_hindcast_weights.sum("M", skipna=True)
        forecast_denominator = effective_forecast_weights.sum("M", skipna=True)

        hindcast_weighted = (
            (hdcst * effective_hindcast_weights).sum("M", skipna=True)
            / hindcast_denominator
        ).where(hindcast_denominator > 0)
        forecast_weighted = (
            (fcst * effective_forecast_weights).sum("M", skipna=True)
            / forecast_denominator
        ).where(forecast_denominator > 0)

        if complete:
            hindcast_unweighted = hdcst.where(valid_hindcast_vector).mean(
                "M", skipna=True
            )
            forecast_unweighted = fcst.where(valid_forecast_vector).mean(
                "M", skipna=True
            )
            hindcast_weighted = hindcast_weighted.fillna(hindcast_unweighted)
            forecast_weighted = forecast_weighted.fillna(forecast_unweighted)

        if self.normalize_probabilities:
            hindcast_weighted = self._normalise_probability_vectors(
                hindcast_weighted
            )
            forecast_weighted = self._normalise_probability_vectors(
                forecast_weighted
            )

        hindcast_weighted = self._as_probability_output(
            hindcast_weighted.where(spatial_mask.notnull()),
            "hindcast_weighted",
        )
        forecast_weighted = self._as_probability_output(
            forecast_weighted.where(spatial_mask.notnull()),
            "forecast_weighted",
        )
        return hindcast_weighted, forecast_weighted

class WAS_ProbWeightedContinuous(WAS_ProbWeighted):
    """Continuous/shrunk version of :class:`WAS_ProbWeighted`.

    The historical stepwise class is kept intact. This variant avoids arbitrary
    score bins and therefore changes smoothly when a model skill estimate moves
    from, for example, GROC 0.79 to 0.81.
    """
    def __init__(self, metric="GROC", threshold=0.5, power=1.0,
                 shrinkage=0.20, loss_scale=None, missing_score="zero",
                 normalize_probabilities=True):
        super().__init__(metric=metric, threshold=threshold,
                         missing_score=missing_score,
                         normalize_probabilities=normalize_probabilities)
        self.power = float(power)
        self.shrinkage = float(shrinkage)
        self.loss_scale = loss_scale
        if self.power <= 0 or not (0 <= self.shrinkage <= 1):
            raise ValueError("power must be >0 and shrinkage must be in [0,1].")

    def transform_score(self, score_array, threshold=None):
        metric = self.metric.lower().replace("_", "").replace("-", "")
        score = score_array.astype(float)
        lower = {"mae", "rmse", "mse", "crps", "ignorance"}
        thr = self.threshold if threshold is None else float(threshold)
        if metric in lower or metric == "bias":
            scale = float(self.loss_scale if self.loss_scale is not None else thr)
            x = np.abs(score) if metric == "bias" else score.clip(min=0.0)
            skill = 1.0 / (1.0 + (x / max(scale, 1e-12)) ** self.power)
        else:
            neutral = 0.5 if metric in {"groc", "roc"} else 0.0
            skill = ((score - neutral) / max(1.0-neutral, 1e-12)).clip(min=0.0) ** self.power
        return ((1-self.shrinkage)*skill + self.shrinkage).where(np.isfinite(score), 0.0)


class WAS_ProbWeightedRPS:
    """Proper-score optimized probability blend with LOYO shrinkage selection.

    Weights lie on the simplex and minimize Ranked Probability Score.  L2
    shrinkage toward equal weights is selected by leave-one-year-out CV.  This
    is deliberately global by default because per-gridpoint weights are poorly
    identified with ~20--30 seasonal hindcast years.
    """
    def __init__(self, shrinkage_grid=(0.0, 0.01, 0.03, 0.1, 0.3, 1.0),
                 min_train_years=8, spatial_pooling="global"):
        if spatial_pooling not in {"global", "gridpoint"}:
            raise ValueError("spatial_pooling must be 'global' or 'gridpoint'.")
        self.shrinkage_grid=tuple(float(v) for v in shrinkage_grid)
        if not self.shrinkage_grid or not all(np.isfinite(v) and v >= 0 for v in self.shrinkage_grid):
            raise ValueError("shrinkage_grid must contain finite nonnegative values.")
        self.min_train_years=int(min_train_years)
        self.spatial_pooling=spatial_pooling
        self.weights_=None; self.lambda_=None; self.cv_rps_=None

    @staticmethod
    def _obs_class(obs, y0, y1):
        return WAS_Verification().compute_class(obs, y0, y1).transpose("T","Y","X")

    @staticmethod
    def _rps(P, cls):
        # P (...,3), cls (...)
        c1=P[...,0]; c2=P[...,0]+P[...,1]
        o1=(cls==0).astype(float); o2=(cls<=1).astype(float)
        valid=np.isfinite(cls)&np.isfinite(c1)&np.isfinite(c2)
        if not np.any(valid): return np.inf
        return float(np.mean(((c1-o1)**2+(c2-o2)**2)[valid]))

    @staticmethod
    def _fit_w(P, cls, lam):
        # P: (M,N,3)
        M=P.shape[0]; prior=np.full(M,1/M)
        def obj(w):
            blend=np.tensordot(w,P,axes=(0,0))
            return WAS_ProbWeightedRPS._rps(blend,cls)+float(lam)*np.sum((w-prior)**2)
        cons={"type":"eq","fun":lambda w: np.sum(w)-1.0}
        res=minimize(obj,prior,method="SLSQP",bounds=[(0,1)]*M,constraints=[cons],
                     options={"maxiter":500,"ftol":1e-10})
        return res.x if res.success else prior

    def fit(self, rainfall, hindcast_prob, clim_year_start, clim_year_end, *, obs_class=None):
        """Fit weights against observed categories for the forecast events.

        ``obs_class`` optionally supplies aligned labels 0/1/2 (or NaN),
        for example when upstream cross-validation probabilities refer to
        different training-fold thresholds at each time. When omitted, the
        historical climatological category calculation is retained. Supplying
        classes preserves event definitions; it does not make precomputed
        upstream forecasts an end-to-end nested validation dataset.
        """
        obs=WAS_mme_Weighted._as_observation(rainfall)
        hp=WAS_ProbWeighted._as_probability_ensemble(hindcast_prob,"hindcast_prob")
        hp=align_predictor_target(hp, obs, mode="season_year")
        hp=WAS_ProbWeighted._normalise_probability_vectors(hp)
        if obs_class is None:
            cls=self._obs_class(obs,clim_year_start,clim_year_end)
        else:
            if not isinstance(obs_class, xr.DataArray):
                raise TypeError("obs_class must be an xarray.DataArray with T,Y,X dimensions.")
            if set(obs_class.dims) != {"T", "Y", "X"}:
                raise ValueError("obs_class must have exactly T,Y,X dimensions.")
            cls=obs_class.transpose("T","Y","X")
            obs,cls=xr.align(obs,cls,join="exact")
            invalid=np.isinf(cls) | (np.isfinite(cls) & ~cls.isin([0,1,2]))
            if bool(invalid.any().compute()):
                raise ValueError("obs_class values must be 0, 1, 2 or NaN.")
            cls=cls.where(np.isfinite(obs))
        arr=hp.transpose("M","T","Y","X","probability").values
        hp, cls = xr.align(hp, cls, join="exact")
        cv=cls.values; M,T,ny,nx,_=arr.shape
        years = _calendar_year_groups(obs["T"])
        if len(np.unique(years)) != T:
            raise ValueError("RPS blend currently requires one seasonal target per calendar year.")
        if T < self.min_train_years:
            warnings.warn("Too few years for RPS weighting; equal weights used.",RuntimeWarning)
            self.weights_=xr.DataArray(np.full(M,1/M),dims="M",coords={"M":hp.M})
            self.lambda_=np.nan; return self
        if self.spatial_pooling == "global":
            best=(np.inf,self.shrinkage_grid[0])
            for lam in self.shrinkage_grid:
                vals=[]
                for it in range(T):
                    tr=np.arange(T)!=it
                    P=arr[:,tr].reshape(M,-1,3); C=cv[tr].reshape(-1)
                    good=np.isfinite(C)&np.all(np.isfinite(P),axis=(0,2))
                    if good.sum()<self.min_train_years: continue
                    w=self._fit_w(P[:,good],C[good],lam)
                    Pt=arr[:,it].reshape(M,-1,3); Ct=cv[it].reshape(-1)
                    vals.append(self._rps(np.tensordot(w,Pt,axes=(0,0)),Ct))
                v=float(np.mean(vals)) if vals else np.inf
                if v<best[0]: best=(v,lam)
            self.cv_rps_,self.lambda_=best
            P=arr.reshape(M,-1,3); C=cv.reshape(-1)
            good=np.isfinite(C)&np.all(np.isfinite(P),axis=(0,2))
            w=self._fit_w(P[:,good],C[good],self.lambda_) if good.any() else np.full(M,1/M)
            self.weights_=xr.DataArray(w,dims="M",coords={"M":hp.M},name="RPS_weights")
        else:
            W=np.full((M,ny,nx),1.0/M); L=np.full((ny,nx),np.nan); R=np.full((ny,nx),np.nan)
            for iy in range(ny):
                for ix in range(nx):
                    C=cv[:,iy,ix]; P=arr[:,:,iy,ix,:]
                    valid_year=np.isfinite(C)&np.all(np.isfinite(P),axis=(0,2))
                    if valid_year.sum()<self.min_train_years: continue
                    best=(np.inf,self.shrinkage_grid[0])
                    for lam in self.shrinkage_grid:
                        vals=[]
                        ids=np.where(valid_year)[0]
                        for it in ids:
                            tr=valid_year.copy(); tr[it]=False
                            w=self._fit_w(P[:,tr],C[tr],lam)
                            vals.append(self._rps(np.tensordot(w,P[:,it:it+1],axes=(0,0)),C[it:it+1]))
                        v=float(np.mean(vals)) if vals else np.inf
                        if v<best[0]: best=(v,lam)
                    R[iy,ix],L[iy,ix]=best
                    W[:,iy,ix]=self._fit_w(P[:,valid_year],C[valid_year],best[1])
            self.weights_=xr.DataArray(W,dims=("M","Y","X"),coords={"M":hp.M,"Y":obs.Y,"X":obs.X},name="RPS_weights")
            self.lambda_=xr.DataArray(L,dims=("Y","X"),coords={"Y":obs.Y,"X":obs.X})
            self.cv_rps_=xr.DataArray(R,dims=("Y","X"),coords={"Y":obs.Y,"X":obs.X})
        return self

    def transform(self, prob):
        if self.weights_ is None: raise RuntimeError("Call fit() first.")
        p=WAS_ProbWeighted._as_probability_ensemble(prob,"prob")
        p=WAS_ProbWeighted._normalise_probability_vectors(p)
        w=self.weights_.sel(M=p.M)
        valid=p.notnull().all("probability")
        we=w.where(valid); den=we.sum("M")
        out=(p*we).sum("M",skipna=True)/den
        return WAS_ProbWeighted._normalise_probability_vectors(out).transpose("probability","T","Y","X")

    def compute_model(self, X_train, y_train, X_test, y_test=None,
                      clim_year_start=None, clim_year_end=None, **kwargs):
        """Fit weights on training observations, apply to the held-out forecast.

        Base probabilities must be generated within this outer-training fold;
        a single collection of global OOF predictions does not ensure this.
        """
        X_train = align_predictor_target(X_train, y_train, mode="season_year")
        if clim_year_start is None:
            clim_year_start = int(y_train["T"].dt.year.min())
        if clim_year_end is None:
            clim_year_end = int(y_train["T"].dt.year.max())
        self.fit(y_train, X_train, clim_year_start, clim_year_end)
        return self.transform(X_test)

    def compute(self, rainfall, hdcst, fcst, clim_year_start, clim_year_end):
        """Operational weight fit; the returned fitted hindcast is in-sample."""
        self.fit(rainfall,hdcst,clim_year_start,clim_year_end)
        fitted = self.transform(hdcst)
        fitted.attrs["weight_validation"] = "in_sample; use outer-fold compute_model for evaluation"
        return fitted, self.transform(fcst)


class WAS_Min2009_ProbWeighted:
    """
    Min et al. (2009) Probability-Weighted Multi-Model Ensemble (PMME).

    What this class does
    --------------------
    1) For each model compute tercile probabilities (PB / PN / PA) at every
       grid point from its ensemble forecast using one of three distributions:

       * 'gaussian'  : Gaussian approximation.
                       Boundaries  : T_lo/hi = mu_hcst ± 0.4307 * sigma_hcst
                       Probability : Phi( (T_lo/hi - f_mean) / sigma_FORECAST )
                       sigma_FORECAST = std of the current forecast ensemble.

       * 'lognormal' : Same logic in log-space (better for precipitation).
                       eps_lognormal is added before taking logs.

       * 'empirical' : Category boundaries are the hindcast q33/q66.
                       Probability = fraction of forecast ensemble members
                       below q33 / above q66.

    2) Combine per-model probabilities with weights w_m ∝ sqrt(N_m) (Min 2009
       eq. 1), normalised to sum to 1.

    3) Optionally compute a combined categorical map with a chi-square
       significance test (Min 2009 section 3b, df = 2).

    Assumed dims
    ------------
    forecasts[m] : at minimum dim 'M' (ensemble members); spatial dims optional.
    hindcasts[m] : dims 'T' (years) and 'M' (members); same spatial dims.

    Parameters
    ----------
    distribution : 'gaussian' | 'lognormal' | 'empirical'
    cv_method    : None | 'leave_one_out' | 'rolling_window'
        How to compute climatological mu / sigma from the hindcast pool.
        None -> use the full hindcast pool (no cross-validation).
    rolling_window_size : int
        Half-width of the rolling window when cv_method='rolling_window'.
    n_samples_for_chisq : 'total_ensemble' | 'effective_sample_size' | float
        How to compute N for the chi-square test.
    eps_lognormal : float
        Small offset added before log transform (lognormal path only).
    sigma_floor : float
        Any sigma <= this value is replaced by the full-hindcast raw sigma.
    """

    def __init__(
        self,
        distribution: Literal["gaussian", "lognormal", "empirical"] = "gaussian",
        cv_method: Optional[Literal["leave_one_out", "rolling_window"]] = None,
        rolling_window_size: int = 15,
        n_samples_for_chisq: Literal["total_ensemble", "effective_sample_size"] | float | int = "total_ensemble",
        eps_lognormal: float = 1e-2,
        sigma_floor: float = 1e-12,
    ):
        self.distribution = distribution
        self.cv_method = cv_method
        self.rolling_window_size = int(rolling_window_size)
        self.n_samples_for_chisq = n_samples_for_chisq
        self.eps_lognormal = float(eps_lognormal)
        self.sigma_floor = float(sigma_floor)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _require_dims(da: xr.DataArray, required: Tuple[str, ...], name: str = "DataArray") -> None:
        missing = [d for d in required if d not in da.dims]
        if missing:
            raise ValueError(f"{name}: missing required dims {missing}. Got dims={da.dims}")

    @staticmethod
    def _norm_cdf(z: xr.DataArray) -> xr.DataArray:
        return _pufunc(stats.norm.cdf, z)

    @staticmethod
    def _safe_clip_and_renorm(
        p_bn: xr.DataArray, p_nn: xr.DataArray, p_an: xr.DataArray
    ) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        p_bn = p_bn.clip(0.0, 1.0)
        p_nn = p_nn.clip(0.0, 1.0)
        p_an = p_an.clip(0.0, 1.0)
        total = p_bn + p_nn + p_an
        ok = _pufunc(np.isfinite, total) & (total > 0.0)
        p_bn = xr.where(ok, p_bn / total, np.nan)
        p_nn = xr.where(ok, p_nn / total, np.nan)
        p_an = xr.where(ok, p_an / total, np.nan)
        return p_bn, p_nn, p_an

    def _compute_model_weights(self, ensemble_sizes: Dict[str, int]) -> Dict[str, float]:
        """w_m = sqrt(N_m) / sum_k sqrt(N_k)  [Min 2009 eq. 1]"""
        sqrt_sizes = {m: np.sqrt(float(n)) for m, n in ensemble_sizes.items()}
        tot = float(sum(sqrt_sizes.values()))
        if tot <= 0:
            raise ValueError("Sum of sqrt(ensemble_sizes) must be > 0.")
        return {m: sqrt_sizes[m] / tot for m in ensemble_sizes}

    def _compute_n_for_chisq(
        self, ensemble_sizes: Dict[str, int], model_names: List[str]
    ) -> float:
        total = float(sum(float(ensemble_sizes[m]) for m in model_names))
        if isinstance(self.n_samples_for_chisq, (int, float)):
            return float(self.n_samples_for_chisq)
        if self.n_samples_for_chisq == "effective_sample_size":
            return total / np.sqrt(max(1, len(model_names)))
        return total  # "total_ensemble"

    # ------------------------------------------------------------------
    # cross-validated climatological stats (mu, sigma from hindcasts)
    # ------------------------------------------------------------------

    def _compute_cross_validated_stats(
        self, hindcasts: xr.DataArray
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Return (mu, sigma) computed from the hindcast pool.
        cv_method=None -> full pool.
        cv_method='leave_one_out' / 'rolling_window' -> time-varying stats
        with the current year withheld.
        """
        self._require_dims(hindcasts, ("T", "M"), name="hindcasts")

        if self.cv_method is None:
            mu    = hindcasts.mean(dim=("T", "M"))
            sigma = hindcasts.std(dim=("T", "M"))
            return mu, sigma

        n_times = hindcasts.sizes["T"]

        if self.cv_method == "leave_one_out":
            mu_list, sig_list = [], []
            for i in range(n_times):
                h_train = hindcasts.isel(T=[j for j in range(n_times) if j != i])
                mu_list.append(h_train.mean(dim=("T", "M")))
                sig_list.append(h_train.std(dim=("T", "M")))
            return (
                xr.concat(mu_list,  dim=hindcasts["T"]),
                xr.concat(sig_list, dim=hindcasts["T"]),
            )

        if self.cv_method == "rolling_window":
            w = self.rolling_window_size
            mu_list, sig_list = [], []
            for i in range(n_times):
                start = max(0, i - w // 2)
                end   = min(n_times, i + w // 2 + 1)
                h_win = hindcasts.isel(T=slice(start, end))
                local_i = i - start
                if 0 <= local_i < h_win.sizes["T"]:
                    h_train = h_win.isel(T=[j for j in range(h_win.sizes["T"]) if j != local_i])
                else:
                    h_train = h_win
                mu_list.append(h_train.mean(dim=("T", "M")))
                sig_list.append(h_train.std(dim=("T", "M")))
            return (
                xr.concat(mu_list,  dim=hindcasts["T"]),
                xr.concat(sig_list, dim=hindcasts["T"]),
            )

        raise ValueError(f"Unknown cv_method={self.cv_method!r}")

    def _align_stats(self, mu, sigma, target, hindcasts):
        """Use only the requested training climatology for each target date.

        Operational dates use the entire supplied hindcast training pool;
        dates in that pool retain their leave-one-out/window statistics.
        A degenerate training spread is floored without borrowing the excluded
        observation or ensemble. Undefined training means remain missing.
        """
        if "T" in mu.dims and "T" in target.dims:
            overlap = np.isin(target["T"].values, mu["T"].values)
            full_mu = hindcasts.mean(dim=("T", "M"))
            full_sigma = hindcasts.std(dim=("T", "M"))
            is_operational = xr.DataArray(~overlap, dims="T", coords={"T": target["T"]})
            mu = mu.reindex(T=target["T"])
            sigma = sigma.reindex(T=target["T"])
            mu = xr.where(is_operational, full_mu, mu)
            sigma = xr.where(is_operational, full_sigma, sigma)
        sigma = sigma.where(~np.isfinite(sigma), sigma.clip(min=self.sigma_floor))
        return mu.broadcast_like(target), sigma.broadcast_like(target)

    def _forecast_sigma(
        self, forecasts: xr.DataArray, sigma_hcst: xr.DataArray
    ) -> xr.DataArray:
        """
        Standard deviation of the current forecast ensemble.
        Falls back to sigma_hcst where the forecast ensemble collapses
        (single member or zero spread).
        """
        sig_f = forecasts.std(dim="M")
        # where forecast is degenerate, borrow the hindcast spread
        sig_f = xr.where(sig_f > self.sigma_floor, sig_f, sigma_hcst)
        return sig_f

    # ------------------------------------------------------------------
    # per-model probability computation (CORRECTED)
    # ------------------------------------------------------------------

    def _compute_tercile_probabilities_one_model(
        self,
        forecasts: xr.DataArray,
        hindcasts: xr.DataArray,
        clim_terciles=None,
    ) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        """
        Compute PB / PN / PA for a single model. ``clim_terciles`` supplies
        fixed observed lower/upper bounds in the forecast's physical units.
        Omitting it preserves the historical model-climatology definition.

        Gaussian & lognormal
        --------------------
        Category boundaries come from the HINDCAST climatology:
            T_lower = mu_hcst + z_{1/3} * sigma_hcst   (= mu - 0.4307*sigma)
            T_upper = mu_hcst + z_{2/3} * sigma_hcst   (= mu + 0.4307*sigma)

        Probabilities use the FORECAST ENSEMBLE spread  [BUG 1 FIX]:
            P(BN) = Phi( (T_lower - f_mean) / sigma_FORECAST )
            P(AN) = 1 - Phi( (T_upper - f_mean) / sigma_FORECAST )

        Empirical
        ---------
        Boundaries : q33 / q66 from all hindcast members.
        Probability: fraction of FORECAST MEMBERS below q33 / above q66
                     [BUG 2 FIX — original used ensemble mean → 0/1 only].
        """
        self._require_dims(forecasts, ("M",),        name="forecasts")
        self._require_dims(hindcasts, ("T", "M"),    name="hindcasts")

        # quantile z-scores for theoretical tercile boundaries
        z_lo = float(stats.norm.ppf(1.0 / 3.0))   # ≈ -0.4307
        z_hi = float(stats.norm.ppf(2.0 / 3.0))   # ≈ +0.4307

        f_mean = forecasts.mean(dim="M")

        # ---- Gaussian ------------------------------------------------
        if self.distribution == "gaussian":
            mu, sigma_hcst = self._compute_cross_validated_stats(hindcasts)
            mu, sigma_hcst = self._align_stats(mu, sigma_hcst, f_mean, hindcasts)

            # category boundaries from hindcast climatology
            T_lower = mu + z_lo * sigma_hcst
            T_upper = mu + z_hi * sigma_hcst
            if clim_terciles is not None:
                T_lower, T_upper = clim_terciles

            # 1: use FORECAST ensemble spread, not hindcast sigma
            sigma_f = self._forecast_sigma(forecasts, sigma_hcst)

            p_bn = self._norm_cdf((T_lower - f_mean) / sigma_f)
            p_an = 1.0 - self._norm_cdf((T_upper - f_mean) / sigma_f)
            p_nn = 1.0 - p_bn - p_an
            return self._safe_clip_and_renorm(p_bn, p_nn, p_an)

        # ---- Lognormal -----------------------------------------------
        if self.distribution == "lognormal":
            eps = self.eps_lognormal

            h_log = np.log(xr.where(hindcasts > 0, hindcasts, eps))
            f_log = np.log(xr.where(forecasts  > 0, forecasts,  eps))

            f_log_mean = f_log.mean(dim="M")

            mu_l, sigma_hcst_l = self._compute_cross_validated_stats(h_log)
            mu_l, sigma_hcst_l = self._align_stats(mu_l, sigma_hcst_l, f_log_mean, h_log)

            # category boundaries in log-space
            T_lower_l = mu_l + z_lo * sigma_hcst_l
            T_upper_l = mu_l + z_hi * sigma_hcst_l
            if clim_terciles is not None:
                low, high = clim_terciles
                # A positive lognormal has zero mass at nonpositive bounds.
                T_lower_l = np.log(xr.where(low > 0, low, np.nan)).where(low > 0, -np.inf).where(np.isfinite(low))
                T_upper_l = np.log(xr.where(high > 0, high, np.nan)).where(high > 0, -np.inf).where(np.isfinite(high))

            # 1 (lognormal): use FORECAST log-space spread
            sigma_f_l = self._forecast_sigma(f_log, sigma_hcst_l)

            p_bn = self._norm_cdf((T_lower_l - f_log_mean) / sigma_f_l)
            p_an = 1.0 - self._norm_cdf((T_upper_l - f_log_mean) / sigma_f_l)
            p_nn = 1.0 - p_bn - p_an
            return self._safe_clip_and_renorm(p_bn, p_nn, p_an)

        # ---- Empirical -----------------------------------------------
        if self.distribution == "empirical":
            if clim_terciles is None:
                q33 = hindcasts.quantile(1.0 / 3.0, dim=("T", "M")).drop_vars("quantile", errors="ignore")
                q66 = hindcasts.quantile(2.0 / 3.0, dim=("T", "M")).drop_vars("quantile", errors="ignore")
            else:
                q33, q66 = clim_terciles

            # 2: count ENSEMBLE MEMBERS, not just compare the mean
            valid = np.isfinite(forecasts) & np.isfinite(q33) & np.isfinite(q66)
            p_bn = (forecasts <= q33).where(valid).mean(dim="M").astype(float)
            p_an = (forecasts > q66).where(valid).mean(dim="M").astype(float)
            p_nn = 1.0 - p_bn - p_an
            return self._safe_clip_and_renorm(p_bn, p_nn, p_an)

        raise ValueError(
            f"Unknown distribution={self.distribution!r}. "
            "Choose 'gaussian', 'lognormal', or 'empirical'."
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def compute_pmme_probabilities(
        self,
        forecasts: Dict[str, xr.DataArray],
        hindcasts: Dict[str, xr.DataArray],
        climatology: Optional[xr.DataArray] = None,   # kept for pipeline compat
        ensemble_sizes: Optional[Dict[str, int]] = None,
        strict_models: bool = True,
        clim_terciles=None,
    ) -> Dict[str, xr.DataArray]:
        """
        Compute PMME probabilities (PB / PN / PA) across models.

        Parameters
        ----------
        forecasts      : {model_name: DataArray(M, [spatial...])}
        hindcasts      : {model_name: DataArray(T, M, [spatial...])}
        climatology    : unused (kept for pipeline compatibility)
        ensemble_sizes : {model_name: int}  defaults to forecasts[m].sizes['M']
        strict_models  : if True, require forecasts and hindcasts to have the
                         same keys.
        clim_terciles   : optional (lower, upper) fixed observed climatological
                         bounds, shared by all models; physical forecast units.

        Returns
        -------
        dict with keys 'PB', 'PN', 'PA'  (each a DataArray over spatial dims).
        """
        if strict_models and set(forecasts.keys()) != set(hindcasts.keys()):
            raise ValueError(
                "forecasts and hindcasts must have the same model keys "
                "when strict_models=True."
            )

        model_names = [m for m in forecasts if m in hindcasts]
        if not model_names:
            raise ValueError("No common models between forecasts and hindcasts.")

        if ensemble_sizes is None:
            ensemble_sizes = {m: int(forecasts[m].sizes["M"]) for m in model_names}

        weights = self._compute_model_weights(ensemble_sizes)

        # per-model probabilities
        per_model: Dict[str, Dict[str, xr.DataArray]] = {}
        for m in model_names:
            p_bn, p_nn, p_an = self._compute_tercile_probabilities_one_model(
                forecasts[m], hindcasts[m], clim_terciles=clim_terciles
            )
            per_model[m] = {"PB": p_bn, "PN": p_nn, "PA": p_an}

        # weighted combination
        template = per_model[model_names[0]]["PB"]
        pmme: Dict[str, xr.DataArray] = {}
        for cat in ("PB", "PN", "PA"):
            wsum = xr.zeros_like(template)
            for m in model_names:
                wsum = wsum + per_model[m][cat] * float(weights[m])
            pmme[cat] = wsum

        # final clip & renorm
        pmme["PB"], pmme["PN"], pmme["PA"] = self._safe_clip_and_renorm(
            pmme["PB"], pmme["PN"], pmme["PA"]
        )
        return pmme

    def compute_combined_map(
        self,
        pmme_probs: Dict[str, xr.DataArray],
        ensemble_sizes: Dict[str, int],
        model_names: List[str],
        significance_level: float = 0.05,
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Combined categorical map with chi-square significance test.
        [Min 2009, section 3b]

        Returns
        -------
        combined_map : DataArray
            0 = no significant deviation from equal probabilities,
            1 = PB dominant,  2 = PN dominant,  3 = PA dominant.
        chi_square : DataArray
            chi-square statistic (df = 2).
        """
        for k in ("PB", "PN", "PA"):
            if k not in pmme_probs:
                raise ValueError(f"pmme_probs missing key {k!r}")

        probs_stack = xr.concat(
            [pmme_probs["PB"], pmme_probs["PN"], pmme_probs["PA"]],
            dim=xr.DataArray(["PB", "PN", "PA"], dims="probability"),
        )
        valid_any = probs_stack.notnull().any("probability")

        # argmax on the three categories (fill NaN with -inf for argmax only)
        dominant = probs_stack.fillna(-np.inf).argmax(dim="probability", skipna=False)

        # chi-square statistic: N * sum_i (p_i - 1/3)^2 / (1/3)
        n = self._compute_n_for_chisq(ensemble_sizes, model_names)
        expected = 1.0 / 3.0
        chi_square = n * (
            (pmme_probs["PB"] - expected) ** 2 / expected
            + (pmme_probs["PN"] - expected) ** 2 / expected
            + (pmme_probs["PA"] - expected) ** 2 / expected
        )

        critical = float(stats.chi2.ppf(1.0 - float(significance_level), df=2))
        combined  = xr.where(valid_any & (chi_square > critical), dominant + 1, 0)

        combined.attrs  = {
            "description": "PMME combined forecast (Min et al. 2009 probability-weighted)",
            "values": "0=not significant, 1=PB, 2=PN, 3=PA",
            "significance_level": float(significance_level),
            "chi2_critical_value": critical,
        }
        chi_square.attrs = {
            "description": "Chi-square statistic for PMME categorical significance",
            "df": 2,
            "n_samples_used": float(n),
        }
        return combined, chi_square


class WAS_mme_logistic:
    """
    Multinomial Logistic Regression MME with spatial clustering and three
    hyperparameter-optimization strategies (grid / random / bayesian).

    Corrections relative to the previous version
    ---------------------------------------------
    1. Bayesian objective now scores by NEGATIVE LOG LOSS (a proper score for
       probability forecasts), consistent with the grid/random searches which
       use scoring=_tercile_logloss_scorer. The old code used model.score (accuracy)
       despite the comment, so the three methods optimized different criteria.
    2. Random search no longer samples invalid penalty/solver pairs: l1 is only
       paired with 'saga' (like the grid). RandomizedSearchCV accepts a LIST of
       distributions, mirroring the grid construction.
    3. Spatial clustering is kept as in the original: KMeans directly on the
       predictand values (each cell labelled by its first non-NaN year). Note
       this is an amplitude binning rather than a regime clustering, but it is
       retained here by preference.
    4. Degenerate clusters are guarded: a cluster with <2 observed classes is
       skipped during tuning and fitting (LogisticRegression needs >=2 classes).
    5. predict_proba is mapped onto the fixed [0,1,2] tercile columns via
       model.classes_, so a cluster missing a class no longer raises a shape
       error and never misaligns PB/PN/PA.

    Recommended usage / framework
    -----------------------------
    Call compute_hyperparameters ONCE on the RAW predictors/predictand, then
    pass (best_params, cluster_da) into compute_model so they are not recomputed
    (and so the predictand is not re-classified) inside every CV fold. In
    WAS_Cross_Validator the existing WAS_mme_logistic branch already passes the
    class-coded predictand and the standardized predictor per fold.
    """

    def __init__(self,
                 optimization_method='grid',
                 C_range=[0.1, 1.0, 10.0, 100.0],
                 solver_options=['newton-cg', 'lbfgs', 'sag', 'saga'],
                 random_state=42,
                 cv_folds=5,
                 n_clusters=4,
                 n_iter_search=20,
                 n_trials=50,
                 timeout=None,
                 n_jobs=-1,
                 optuna_n_jobs=None,
                 cv_n_jobs=-1):

        self.optimization_method = optimization_method
        self.C_range = C_range
        self.solver_options = solver_options
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.n_clusters = n_clusters
        self.n_iter_search = n_iter_search
        self.n_trials = n_trials
        self.timeout = timeout
        if n_jobs == 0 or cv_n_jobs == 0 or optuna_n_jobs == 0:
            raise ValueError("n_jobs, cv_n_jobs and optuna_n_jobs cannot be 0.")
        self.n_jobs = int(n_jobs)
        self.cv_n_jobs = int(cv_n_jobs)
        self.optuna_n_jobs = 1 if optuna_n_jobs is None else int(optuna_n_jobs)
        self.models = None
        self.best_params_dict = None
        self.cluster_da = None
        self.study_dict = None

    # ----------------------------------------------------- parameter spaces
    @staticmethod
    def _l2_solvers(solver_options):
        return [s for s in solver_options if s in ('newton-cg', 'lbfgs', 'sag', 'saga')]

    def _create_parameter_grids(self):
        """Grid(s): l2 with all compatible solvers, l1 only with saga."""
        param_grids = [{
            'penalty': ['l2'],
            'solver': self._l2_solvers(self.solver_options),
            'C': self.C_range,
        }]
        if 'saga' in self.solver_options:
            param_grids.append({'penalty': ['l1'], 'solver': ['saga'], 'C': self.C_range})
        return param_grids

    def _create_parameter_distributions(self):
        """
        2: list of distributions so l1 is only ever paired with saga
        (RandomizedSearchCV samples one dict per draw, like GridSearchCV).
        """
        dists = [{
            'penalty': ['l2'],
            'solver': self._l2_solvers(self.solver_options),
            'C': self.C_range,
        }]
        if 'saga' in self.solver_options:
            dists.append({'penalty': ['l1'], 'solver': ['saga'], 'C': self.C_range})
        return dists

    def _cv_splitter(self, groups=None):
        """Folds that never split a year across train and validation.

        The samples fed to the search are pooled over (T, Y, X), so
        KFold(shuffle=True) puts cells from the SAME year on both sides of a
        fold. Rainfall fields are strongly autocorrelated in space, so a
        validation cell is then almost predictable from its own neighbours in
        the training half. Measured on pooled samples with year-correlated
        noise, 18 training years:

            tuning score, KFold(shuffle=True)   -1.0510   (what the code saw)
            tuning score, GroupKFold(year)      -1.1059   (honest)
            true score, 6 held-out years        -1.1483

        The shuffled estimate was optimistic by 0.055 in log loss and pointed
        the C search towards weaker regularisation than the data supports.
        Every other learner in this module already groups by year
        (GroupKFold / LeaveOneGroupOut); the logistic MME was the exception.
        """
        if groups is None:
            return KFold(n_splits=self.cv_folds, shuffle=False)
        n_group = len(np.unique(groups))
        if n_group <= self.cv_folds:
            return LeaveOneGroupOut()
        return GroupKFold(n_splits=min(self.cv_folds, n_group))

    # ----------------------------------------------------- search routines
    def _grid_search_optimization(self, X, y, groups=None):
        model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        cv = self._cv_splitter(groups)
        gs = GridSearchCV(model, param_grid=self._create_parameter_grids(), cv=cv,
                          scoring=_tercile_logloss_scorer, error_score=np.nan, n_jobs=self.n_jobs)
        gs.fit(X, y, groups=groups)
        if getattr(self, "verbose", 1):
            print(f"[logistic/grid] score={gs.best_score_:.4f}, params={gs.best_params_}",
                  flush=True)
        return gs.best_params_

    def _random_search_optimization(self, X, y, groups=None):
        model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        cv = self._cv_splitter(groups)
        rs = RandomizedSearchCV(model, param_distributions=self._create_parameter_distributions(),
                                n_iter=self.n_iter_search, cv=cv, scoring=_tercile_logloss_scorer,
                                random_state=self.random_state, error_score=np.nan, n_jobs=self.n_jobs)
        rs.fit(X, y, groups=groups)
        if getattr(self, "verbose", 1):
            print(f"[logistic/random] score={rs.best_score_:.4f}, params={rs.best_params_}",
                  flush=True)
        return rs.best_params_

    def _bayesian_objective(self, trial, X, y):
        """1: maximize -log_loss (proper score), not accuracy."""
        penalty = trial.suggest_categorical('penalty', ['l2', 'l1'])
        if penalty == 'l1':
            solver = 'saga'
        else:
            l2_solvers = [s for s in self.solver_options if s != 'saga'] or ['lbfgs']  # G: never empty
            solver = trial.suggest_categorical('solver', l2_solvers)

        if isinstance(self.C_range, (list, tuple)) and len(self.C_range) == 2:
            C_min, C_max = min(self.C_range), max(self.C_range)
            C = trial.suggest_float('C', C_min, C_max, log=True)
        else:
            C = trial.suggest_categorical('C', list(self.C_range))

        model = LogisticRegression(penalty=penalty, solver=solver, C=C,
                                   random_state=self.random_state, max_iter=1000)

        labels = np.unique(y)
        # Same reasoning as _cv_splitter: never split a year across folds.
        cv = self._cv_splitter(getattr(self, "_current_groups", None))
        def score_fold(tr, va):
            try:
                fold_model = LogisticRegression(
                    penalty=penalty, solver=solver, C=C,
                    random_state=self.random_state, max_iter=1000)
                fold_model.fit(X[tr], y[tr])
                return _tercile_logloss_scorer(fold_model, X[va], y[va])
            except Exception:
                return np.nan
        splits = list(cv.split(X, y, groups=getattr(self, '_current_groups', None)))
        if JOBLIB_AVAILABLE and self.cv_n_jobs != 1:
            scores = Parallel(n_jobs=self.cv_n_jobs)(
                delayed(score_fold)(tr, va) for tr, va in splits)
        else:
            scores = [score_fold(tr, va) for tr, va in splits]
        scores = np.array(scores, dtype=float)
        valid = scores[np.isfinite(scores)]
        return float(np.mean(valid)) if valid.size else float("-inf")

    def _bayesian_optimization(self, X, y, cluster_id=None):
        if optuna is None:
            raise ImportError("optimization_method='bayesian' requires optuna to be installed.")
        study_name = f'logistic_cluster_{cluster_id}' if cluster_id is not None else 'logistic'
        study = optuna.create_study(study_name=study_name, direction='maximize',
                                    sampler=TPESampler(seed=self.random_state))
        study.optimize(lambda t: self._bayesian_objective(t, X, y),
                       n_trials=self.n_trials, timeout=self.timeout,
                       n_jobs=self.optuna_n_jobs, show_progress_bar=False)
        if cluster_id is not None:
            self.study_dict = self.study_dict or {}
            self.study_dict[cluster_id] = study
        best_params = dict(study.best_params)
        if best_params.get('penalty') == 'l1':
            best_params['solver'] = 'saga'
        if getattr(self, "verbose", 1):
            print(f"[logistic] cluster {cluster_id}: "
                  f"score={study.best_value:.4f}, params={best_params}", flush=True)
        return best_params

    def _best_params_for(self, X_clean_c, y_clean_c, cluster_id=None, groups=None):
        # Stash the groups so the Optuna objective, which takes no extra
        # argument, can build the same year-aware folds as the other searches.
        self._current_groups = groups
        if self.optimization_method == 'grid':
            return self._grid_search_optimization(X_clean_c, y_clean_c, groups)
        if self.optimization_method == 'random':
            return self._random_search_optimization(X_clean_c, y_clean_c, groups)
        if self.optimization_method == 'bayesian':
            return self._bayesian_optimization(X_clean_c, y_clean_c, cluster_id=cluster_id)
        if self.optimization_method == 'none':
            return {'C': 1.0, 'solver': 'lbfgs', 'penalty': 'l2'}
        raise ValueError(f"Unknown optimization method: {self.optimization_method}")

    # ----------------------------------------------------- helpers
    def _cluster_on_values(self, Predictand):
        """Compatibility wrapper using climate-summary features."""
        return _build_climate_feature_clusters(
            Predictand, self.n_clusters, random_state=self.random_state
        )

    @staticmethod
    def _proba_to_terciles(y_prob, classes):
        """
        5: place predict_proba columns onto the fixed [0,1,2] tercile axis
        using model.classes_, so missing classes -> 0 and never misalign.
        """
        out = np.zeros((y_prob.shape[0], 3), dtype=float)
        for j, cls in enumerate(classes):
            ci = int(cls)
            if 0 <= ci <= 2:
                out[:, ci] = y_prob[:, j]
        return out

    # ----------------------------------------------------- hyperparameters
    def compute_hyperparameters(self, Predictors, Predictand,
                                clim_year_start=None, clim_year_end=None, *, target_is_class=False):
        """Tune with fold-local standardization, target encoding and climate zones."""
        return _strict_regional_hyperparameters(
            self, Predictors, Predictand, clim_year_start, clim_year_end, target_is_class=target_is_class)

    def compute_hyperparameters_legacy(self, Predictors, Predictand, clim_year_start, clim_year_end, *, target_is_class=False):
        """
        Compute best hyperparameters per spatial cluster.

        Predictors : (T, M, Y, X) RAW predictor cube.
        Predictand : (T, Y, X) RAW predictand (classes are computed here).
        """
        Predictors = align_predictor_target(Predictors, Predictand, mode="season_year")
        if "M" in Predictand.dims:
            Predictand = Predictand.isel(M=0, drop=True)
        Predictand.name = "varname"

        X_train_std = Predictors if target_is_class else standardize_timeseries(Predictors, clim_year_start, clim_year_end)

        # KMeans directly on predictand values (original approach, kept by preference)
        Cluster = self._cluster_on_values(Predictand)

        verify = WAS_Verification()
        y_train_std = Predictand if target_is_class else verify.compute_class(Predictand, clim_year_start, clim_year_end)

        # align cluster grid with the (classed) predictand grid
        xarray1, xarray2 = xr.align(y_train_std, Cluster, join="outer")
        cluster_da = xarray2
        clusters = np.unique(cluster_da.values)
        clusters = clusters[~np.isnan(clusters)]

        # align time axes (compute_class preserves T order)
        X_train_std['T'] = y_train_std['T']

        best_params_dict = {}
        for c in clusters:
            mask_3d = (cluster_da == c).expand_dims({'T': y_train_std['T']})
            X_c = _stack_masked(X_train_std, mask_3d, core='M')
            y_c = _stack_masked(y_train_std, mask_3d, core=None).ravel()

            # Year label of every pooled (T, Y, X) sample, so the folds can be
            # grouped by year instead of shuffled across it.
            n_time = int(y_train_std.sizes["T"])
            spatial = int(y_c.size // n_time)
            years_all = np.repeat(
                y_train_std["T"].dt.year.values.astype(int), spatial
            )

            bad = np.any(~np.isfinite(X_c), axis=1) | ~np.isfinite(y_c)
            X_clean, y_clean = X_c[~bad], y_c[~bad]
            groups_c = years_all[~bad] if years_all.size == y_c.size else None

            # 4: need samples and at least two classes
            if len(X_clean) == 0 or len(np.unique(y_clean)) < 2:
                continue

            best_params_dict[c] = self._best_params_for(
                X_clean, y_clean, cluster_id=c, groups=groups_c
            )

        self.best_params_dict = best_params_dict
        self.cluster_da = cluster_da
        _report_best_params("logistic", best_params_dict, getattr(self, "verbose", 1))
        return best_params_dict, cluster_da

    # ----------------------------------------------------- shared fit/predict
    def _fit_predict_clusters(self, X_train_std, y_train, X_test_std, best_params, cluster_da):
        """
        Fit one logistic model per cluster on the training fold and predict the
        test fold. Returns (predictions_classes (T,Y,X), predictions_prob (T,Y,X,3)).
        """
        time, lat, lon = X_test_std['T'], X_test_std['Y'], X_test_std['X']
        n_time, n_lat, n_lon = len(time), len(lat), len(lon)

        predictions = np.full((n_time, n_lat, n_lon), np.nan)
        predictions_prob = np.full((n_time, n_lat, n_lon, 3), np.nan)
        self.models = {}

        for c, bp in best_params.items():
            mask_tr = (cluster_da == c).expand_dims({'T': X_train_std['T']})
            mask_te = (cluster_da == c).expand_dims({'T': X_test_std['T']})

            Xtr = X_train_std.where(mask_tr).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            ytr = y_train.where(mask_tr).stack(sample=('T', 'Y', 'X')).values.ravel()
            tr_bad = np.any(~np.isfinite(Xtr), axis=1) | ~np.isfinite(ytr)
            Xtr_clean, ytr_clean = Xtr[~tr_bad], ytr[~tr_bad]

            Xte = X_test_std.where(mask_te).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            te_bad = np.any(~np.isfinite(Xte), axis=1)
            Xte_clean = Xte[~te_bad]

            # 4: skip degenerate clusters instead of crashing
            if len(Xtr_clean) == 0 or len(np.unique(ytr_clean)) < 2 or len(Xte_clean) == 0:
                continue

            model_c = LogisticRegression(**bp, random_state=self.random_state, max_iter=1000)
            model_c.fit(Xtr_clean, ytr_clean)
            self.models[c] = model_c

            y_pred = model_c.predict(Xte_clean)
            y_prob = self._proba_to_terciles(model_c.predict_proba(Xte_clean), model_c.classes_)

            full_pred = np.full(Xte.shape[0], np.nan)
            full_pred[~te_bad] = y_pred
            pred_grid = full_pred.reshape(n_time, n_lat, n_lon)
            predictions = np.where(np.isnan(predictions), pred_grid, predictions)

            full_prob = np.full((Xte.shape[0], 3), np.nan)
            full_prob[~te_bad] = y_prob
            prob_grid = full_prob.reshape(n_time, n_lat, n_lon, 3)
            predictions_prob = np.where(np.isnan(predictions_prob), prob_grid, predictions_prob)

        return predictions, predictions_prob, time, lat, lon

    @staticmethod
    def _wrap_outputs(predictions, predictions_prob, time, lat, lon):
        det = xr.DataArray(predictions, coords={'T': time, 'Y': lat, 'X': lon}, dims=['T', 'Y', 'X'])
        prob = xr.DataArray(predictions_prob,
                            coords={'T': time, 'Y': lat, 'X': lon, 'probability': [0, 1, 2]},
                            dims=['T', 'Y', 'X', 'probability'])
        prob = prob.transpose('probability', 'T', 'Y', 'X').assign_coords(probability=['PB', 'PN', 'PA'])
        return det, prob

    # ----------------------------------------------------- framework contract
    def compute_model(self, X_train, y_train, X_test, y_test, clim_year_start, clim_year_end,
                      best_params=None, cluster_da=None):
        """
        Train per-cluster logistic models and predict the test fold.

        X_train / X_test : standardized predictor cubes (T, M, Y, X).
        y_train          : class-coded predictand (T, Y, X) for training.
        best_params, cluster_da : pass the precomputed ones (recommended). If
                          omitted they are derived here -- but note that requires
                          a RAW predictand for compute_class, so prefer passing
                          precomputed values inside cross-validation.
        """
        X_train = align_predictor_target(X_train, y_train, mode="season_year")
        X_train_std = X_train          # already standardized by the framework
        X_test_std = X_test

        if best_params is None or cluster_da is None or not _cluster_matches_training(cluster_da, y_train):
            best_params, cluster_da = self.compute_hyperparameters(
                X_train, y_train, clim_year_start, clim_year_end, target_is_class=True)

        predictions, predictions_prob, time, lat, lon = self._fit_predict_clusters(
            X_train_std, y_train, X_test_std, best_params, cluster_da)

        return self._wrap_outputs(predictions, predictions_prob, time, lat, lon)

    # ----------------------------------------------------- operational
    def forecast(self, Predictant, clim_year_start, clim_year_end,
                 Predictors_train, Predictor_for_year, best_params=None, cluster_da=None):
        """Operational forecast for a single target year (classes + tercile probabilities)."""
        if "M" in Predictant.dims:
            Predictant_no_m = Predictant.isel(M=0, drop=True)
        else:
            Predictant_no_m = Predictant

        verify = WAS_Verification()
        Predictant_class = verify.compute_class(Predictant_no_m, clim_year_start, clim_year_end)

        mask = xr.where(np.isfinite(Predictant_no_m).sum(dim='T') >= 3, 1.0, np.nan)

        Predictors_train_st = standardize_timeseries(Predictors_train, clim_year_start, clim_year_end)
        clim_slice = slice(str(clim_year_start), str(clim_year_end))
        mean_val = Predictors_train.sel(T=clim_slice).mean(dim='T')
        std_val = Predictors_train.sel(T=clim_slice).std(dim='T')
        Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val

        Predictors_train_st['T'] = Predictant_class['T']

        if best_params is None or cluster_da is None:
            best_params, cluster_da = self.compute_hyperparameters(
                Predictors_train, Predictant_no_m, clim_year_start, clim_year_end)

        predictions, predictions_prob, time, lat, lon = self._fit_predict_clusters(
            Predictors_train_st, Predictant_class, Predictor_for_year_st, best_params, cluster_da)

        forecast_det, forecast_prob = self._wrap_outputs(predictions, predictions_prob, time, lat, lon)
        forecast_det = forecast_det * mask
        forecast_prob = forecast_prob * mask

        # relabel time to the forecast-year target month
        year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970
        month = Predictant_class.isel(T=0).coords['T'].values.astype('datetime64[M]').astype(int) % 12 + 1
        new_T = np.datetime64(f"{year}-{month:02d}-01")
        forecast_det = forecast_det.assign_coords(T=xr.DataArray([new_T], dims=["T"]))
        forecast_det['T'] = forecast_det['T'].astype('datetime64[ns]')
        forecast_prob = forecast_prob.assign_coords(T=xr.DataArray([new_T], dims=["T"]))
        forecast_prob['T'] = forecast_prob['T'].astype('datetime64[ns]')

        return forecast_det, forecast_prob

    def get_optimization_results(self):
        results = {
            'optimization_method': self.optimization_method,
            'best_parameters': self.best_params_dict,
            'cluster_labels': self.cluster_da,
            'models': self.models,
        }
        if self.optimization_method == 'bayesian' and self.study_dict:
            results['optuna_studies'] = self.study_dict
        return results
        
class WAS_mme_ELR:
    """
    Extended Logistic Regression (ELR) multi-model-ensemble tercile forecaster,
    aligned with Wilks (2009, *Meteorol. Appl.* 16, 361-368).

    Single unified equation for every threshold q:

        P(Y <= q | x) = sigmoid( b0 + beta . x + alpha * g(q) )

    What is implemented FROM THE PAPER (vs the previous two-threshold version)
    -------------------------------------------------------------------------
    1. MULTI-THRESHOLD FIT. Wilks fits ONE equation to SEVERAL climatological
       quantiles simultaneously (not just the two needed at output time), which
       constrains alpha much better and yields coherent probabilities at ANY
       threshold -- including thresholds never used in fitting. Here the design
       is stacked over `fit_quantiles` (default a Wilks-style 7-quantile set
       containing the terciles); the tercile probabilities PB/PN/PA are then
       read off the same fitted equation.
    2. GUARANTEED MONOTONE CDF. Monotonicity of P(Y<=q) in q requires alpha>0.
       The damped Newton solver constrains alpha to [alpha_min, inf),
       with a line search and a checked constrained-solver fallback, so pup >= plo holds by
       construction and no crossing can occur at any threshold.
    3. THRESHOLD TRANSFORM g(q). Wilks found g(q) = sqrt(q) clearly superior
       to g(q) = q for (non-negative, raw) precipitation. 'sqrt' is offered
       for raw precip; 'cbrt' (default) is the negative-safe analogue for
       signed predictands; the choice can be tuned by CV (see
       compute_hyperparameters, which also tunes the fitted-quantile set).
    4. PREDICTOR TRANSFORM. Wilks' best single predictor was the SQUARE ROOT
       of the ensemble-mean precipitation. `predictor_transform='sqrt'`
       applies it to EACH raw ensemble member before aggregation/standardization in
       compute_hyperparameters / forecast. (The framework CV branch feeds
       already-standardized predictors to compute_model; for exact
       consistency, apply the same transform to the raw hindcast before the
       framework standardizes it.)
    5. FULL-DISTRIBUTION OUTPUT. The point of the paper: the same fit gives
       the whole predictive distribution. `predictive_quantiles(p_levels)`
       inverts the fitted equation analytically,
           q_p = g^{-1}( (logit(p) - b0 - beta.x) / alpha ),
       returning predictive quantiles (quantile, T, Y, X) after any
       compute_model / forecast call.

    For tercile forecasts the two output thresholds are the climatological
    lower and upper terciles T1 < T2 of the predictand, and

        PB = P(Y <= T1),
        PN = P(Y <= T2) - P(Y <= T1),
        PA = 1 - P(Y <= T2).

    Predictor at each grid cell
    ---------------------------
    The (T, Y, X, M) predictor cube is reduced per cell to:
      predictors='mean'      -> multi-model mean                 (1 feature)  [default]
      predictors='mean_std'  -> multi-model mean and spread      (2 features)
      predictors='models'    -> every model kept as its own feature (M features)

    Thresholds
    ----------
    Output terciles use the declared climatology window; pass
    `clim_terciles=(t1, t2)` (DataArrays on Y, X, same units as y_train) to pin
    every fold to the complete reference boundaries your verification uses
    (WAS_Verification.compute_class). The EXTRA fitting quantiles are always
    taken fold-safe from the training predictand.

    Fitting
    -------
    One equation per cell, all cells fitted simultaneously with a vectorized,
    ridge-stabilized damped-Newton solver with a decreasing objective. Rows with missing
    predictor/predictand are zero-weighted; cells with fewer than `min_obs`
    valid years (or no converged solution) fall back to training-only empirical
    probabilities at the same fixed thresholds. Missing test predictors stay
    missing. With K fitting
    quantiles the stacked-binary objective is a composite likelihood (the K
    indicator responses per year are correlated) -- exactly Wilks' fitting
    strategy. In dry cells several low quantiles may coincide (ties at 0);
    the duplicated rows are harmless, they only re-weight that threshold.

    Parameters
    ----------
    predictors : {'mean', 'mean_std', 'models'}
    threshold_transform : {'identity','cbrt','sqrt'} or callable
        g(q). Default 'cbrt'. 'sqrt' only for non-negative predictands
        (raw precipitation) -- Wilks' empirically best choice there.
    fit_quantiles : sequence of float in (0, 1)
        Climatological quantiles whose thresholds enter the fit. The terciles
        1/3 and 2/3 are always included. Default
        (0.10, 0.25, 1/3, 0.50, 2/3, 0.75, 0.90).
    l2 : float
        Ridge penalty on the coefficients (NOT the intercept), default 1e-3.
    alpha_min : float
        Positivity floor for the threshold coefficient alpha (monotone CDF),
        default 1e-3.
    predictor_transform : {'identity','sqrt','cbrt'} or None
        Transform applied to the RAW predictor before standardization in
        compute_hyperparameters / forecast (Wilks: 'sqrt' for raw precip).
    n_iter, tol : Newton controls (default 50, 1e-7).
    min_obs : minimum valid training years per cell (default 8).
    """

    # named threshold transforms g(q); all NON-DECREASING so that the
    # cumulative forecast p(q)=Pr{V<=q} stays monotone in q.
    _TRANSFORMS = {"identity": None, "cbrt": np.cbrt, "sqrt": np.sqrt}
    # analytic inverses g^{-1}(z) for the quantile function (Wilks Sec. 3:
    # the same equation is solved for q at any cumulative probability p).
    _INVERSES = {"identity": None,
                 "cbrt": lambda z: np.power(z, 3),
                 "sqrt": lambda z: np.square(np.maximum(z, 0.0))}

    def __init__(self, predictors="mean", threshold_transform="cbrt",
                 fit_quantiles=(0.10, 0.25, 1.0 / 3.0, 0.50, 2.0 / 3.0, 0.75, 0.90),
                 l2=1e-3, alpha_min=1e-3, n_iter=50, tol=1e-7, min_obs=8,
                 predictor_transform=None, random_state=42):
        if predictors not in ("mean", "mean_std", "models"):
            raise ValueError("predictors must be 'mean', 'mean_std' or 'models'.")
        self.predictors = predictors
        self.threshold_transform = threshold_transform
        self.fit_quantiles = self._check_quantiles(fit_quantiles)
        self.l2 = float(l2)
        self.alpha_min = float(alpha_min)
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.min_obs = int(min_obs)
        if not np.isfinite(self.l2) or self.l2 < 0:
            raise ValueError("l2 must be finite and non-negative.")
        if not np.isfinite(self.alpha_min) or self.alpha_min <= 0:
            raise ValueError("alpha_min must be finite and strictly positive.")
        if self.n_iter < 1 or not np.isfinite(self.tol) or self.tol <= 0 or self.min_obs < 2:
            raise ValueError("n_iter >= 1, tol > 0 and min_obs >= 2 are required.")
        self._resolve_transform(threshold_transform)
        self._resolve_transform(predictor_transform)
        self.predictor_transform = predictor_transform
        self.random_state = random_state
        # filled by compute_hyperparameters (joint l2 + g(q) + quantile tuning)
        self.best_params = None
        self.cv_scores_ = None
        self.last_crossing_fraction_ = 0.0
        self._last_fit_ = None
        self._last_coords_ = None

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _check_quantiles(qs):
        """Sorted unique fitting quantiles, terciles enforced, all in (0, 1)."""
        qs = np.unique(np.asarray(list(qs) + [1.0 / 3.0, 2.0 / 3.0], dtype=float))
        if np.any(~np.isfinite(qs) | (qs <= 0.0) | (qs >= 1.0)):
            raise ValueError("fit_quantiles must lie strictly inside (0, 1).")
        return qs

    @classmethod
    def _resolve_transform(cls, transform):
        """Map a transform spec (None | name | callable) to a callable or None."""
        if transform is None or transform == "identity":
            return None
        if isinstance(transform, str):
            if transform not in cls._TRANSFORMS:
                raise ValueError(f"unknown threshold_transform '{transform}'.")
            return cls._TRANSFORMS[transform]
        if not callable(transform):
            raise ValueError("A transform must be a supported name or callable.")
        return transform

    @classmethod
    def _resolve_inverse(cls, transform):
        """g^{-1} for the quantile function; named transforms only."""
        if transform is None or transform == "identity":
            return None
        if isinstance(transform, str) and transform in cls._INVERSES:
            return cls._INVERSES[transform]
        raise ValueError("predictive_quantiles needs an invertible named "
                         "transform ('identity', 'cbrt' or 'sqrt').")

    def _apply_g(self, q, transform):
        f = self._resolve_transform(transform)
        return q if f is None else f(q)

    def _g(self, q):
        return self._apply_g(q, self.threshold_transform)

    def _tx_predictor(self, X):
        """Wilks' predictor transform (e.g. sqrt of raw ensemble precip)."""
        f = self._resolve_transform(self.predictor_transform)
        if f is None:
            return X
        if self.predictor_transform == "sqrt":
            X = X.where((X >= 0.0) | X.isnull(), 0.0) if isinstance(X, xr.DataArray) \
                else np.maximum(X, 0.0)
        return f(X)

    def _standardize_predictors(self, train, test, clim_start=None, clim_end=None):
        """Transform raw predictors, then estimate scaling on training dates only."""
        train, test = self._tx_predictor(train), self._tx_predictor(test)
        reference = train
        if clim_start is not None and clim_end is not None:
            window = train.sel(T=slice(str(clim_start), str(clim_end)))
            if window.sizes.get("T", 0) >= 2:
                reference = window
        mean, std = reference.mean("T", skipna=True), reference.std("T", skipna=True)
        std = xr.where(np.isfinite(std) & (std == 0), 1.0, std)
        return (train - mean) / std, (test - mean) / std

    @staticmethod
    def _event_thresholds(y, clim_start=None, clim_end=None, explicit=None):
        if explicit is not None:
            return explicit
        q, _ = climatological_terciles(y, clim_start, clim_end)
        return tuple(q.isel(quantile=i, drop=True) for i in range(2))

    def _features(self, X):
        """Reduce a (T, Y, X[, M]) cube to a (T, Y, X, feat) predictor stack."""
        if "M" in X.dims:
            Xm = X.transpose("T", "Y", "X", "M")
            if self.predictors == "mean":
                feats = [Xm.mean("M", skipna=True)]
            elif self.predictors == "mean_std":
                feats = [Xm.mean("M", skipna=True), Xm.std("M", skipna=True)]
            else:  # 'models'
                feats = [Xm.isel(M=k) for k in range(Xm.sizes["M"])]
        else:
            feats = [X.transpose("T", "Y", "X")]
        return xr.concat(feats, dim="feat").transpose("T", "Y", "X", "feat")

    # ------------------------------------------------------------- core solver
    def _fit_predict(self, Ftr, ytr, Fte, t1, t2, l2=None,
                     transform="__self__", quantiles=None):
        """
        Batched multi-threshold ELR over all cells (Wilks 2009 fit).

        Ftr : (n_train, n_cells, F)   ytr : (n_train, n_cells)
        Fte : (n_test,  n_cells, F)   t1, t2 : (n_cells,) output terciles
        l2        : ridge penalty override (default None -> self.l2).
        transform : g(q) override (default '__self__' -> self.threshold_transform).
        quantiles : fitting-quantile override (default None -> self.fit_quantiles).
        returns probs (3, n_test, n_cells) ordered PB, PN, PA.

        The design stacks one binary block per fitting quantile: K
        climatological thresholds per cell, with the tercile rows pinned to the
        exact (possibly fixed climatological) t1, t2 used at output time.
        alpha is projected onto [alpha_min, inf) each Newton step, so the
        predicted CDF is monotone in q by construction.
        """
        l2 = self.l2 if l2 is None else float(l2)
        tfm = self.threshold_transform if transform == "__self__" else transform
        qs = self.fit_quantiles if quantiles is None else self._check_quantiles(quantiles)
        n_train, n_cells, F = Ftr.shape

        # ---- K training thresholds per cell (fold-safe quantiles of y) ------
        with np.errstate(all="ignore"):
            thr = np.nanquantile(ytr, qs, axis=0)              # (K, n_cells)
        i1 = int(np.argmin(np.abs(qs - 1.0 / 3.0)))
        i2 = int(np.argmin(np.abs(qs - 2.0 / 3.0)))
        thr[i1], thr[i2] = t1, t2      # pin the exact output tercile boundaries
        K = thr.shape[0]
        gthr = self._apply_g(thr, tfm)                          # (K, n_cells)

        if not np.isfinite(l2) or l2 < 0:
            raise ValueError("l2 must be finite and non-negative.")
        if tfm == "sqrt" and np.any(np.isfinite(ytr) & (ytr < 0)):
            raise ValueError("The sqrt ELR threshold transform requires non-negative observations.")
        finite_thr = np.isfinite(thr)
        if np.any(finite_thr & ~np.isfinite(gthr)):
            raise ValueError("The threshold transform must be finite on observed thresholds.")
        order = np.argsort(thr, axis=0)
        sorted_thr = np.take_along_axis(thr, order, axis=0)
        sorted_g = np.take_along_axis(gthr, order, axis=0)
        if np.any((np.diff(sorted_thr, axis=0) >= 0) & (np.diff(sorted_g, axis=0) < 0)):
            raise ValueError("The ELR threshold transform must be non-decreasing.")
        if np.any(np.isfinite(t1) & np.isfinite(t2) & (t1 > t2)):
            raise ValueError("The lower climatological threshold must not exceed the upper one.")

        # Centering eliminates artificial sensitivity to Celsius/Kelvin offsets.
        # Scaling only changes coordinates: alpha's original-unit penalty and
        # lower bound are transformed explicitly, so the objective is unchanged.
        gc = np.nanmean(gthr, axis=0)
        gs = np.nanstd(gthr, axis=0)
        gs = np.where(np.isfinite(gs) & (gs > 0), gs, 1.0)
        gc = np.where(np.isfinite(gc), gc, 0.0)
        zthr = (gthr - gc) / gs

        # ---- extended design: K threshold blocks stacked along rows ---------
        valid_xy = np.isfinite(ytr) & np.isfinite(Ftr).all(axis=2)  # (n_train, n_cells)
        col = np.broadcast_to(zthr[:, None, :], (K, n_train, n_cells))
        Fb = np.broadcast_to(Ftr[None], (K, n_train, n_cells, F))
        A = np.concatenate([Fb, col[..., None],
                            np.ones((K, n_train, n_cells, 1))], axis=3)
        A = np.where(np.isfinite(A), A, 0.0).reshape(K * n_train, n_cells, F + 2)
        Yb = (ytr[None, :, :] <= thr[:, None, :]).astype(float)
        Yb = Yb.reshape(K * n_train, n_cells)
        R = (valid_xy[None] & np.isfinite(thr)[:, None, :]
             & np.isfinite(gthr)[:, None, :])
        R = R.reshape(K * n_train, n_cells).astype(float)

        n_feat = F + 2
        A = np.transpose(A, (1, 0, 2))                          # (n_cells, Kn, n_feat)
        Yb = Yb.T                                               # (n_cells, Kn)
        R = R.T                                                 # (n_cells, Kn)

        pen = np.full((n_cells, n_feat), l2)
        pen[:, F], pen[:, -1] = l2 / gs**2, 0.0
        eye = np.eye(n_feat)
        eyepen = eye[None] * pen[:, None, :]
        lower = self.alpha_min * gs
        beta = np.zeros((n_cells, n_feat))
        beta[:, F] = lower
        nvalid = valid_xy.sum(axis=0)
        fit_cells = ((nvalid >= self.min_obs) & np.isfinite(t1) & np.isfinite(t2)
                     & (np.nanmax(gthr, axis=0) > np.nanmin(gthr, axis=0)))

        def loss(coeff):
            eta = np.einsum("crf,cf->cr", A, coeff)
            return ((R * (np.logaddexp(0.0, eta) - Yb * eta)).sum(axis=1)
                    + 0.5 * (pen * coeff**2).sum(axis=1))

        def gradient(coeff):
            p = expit(np.einsum("crf,cf->cr", A, coeff))
            return np.einsum("crf,cr->cf", A, R * (p - Yb)) + pen * coeff

        def projected_gradient(coeff, grad):
            out = grad.copy()
            bound = (coeff[:, F] <= lower + 1e-12) & (grad[:, F] > 0)
            out[bound, F] = 0.0
            return out

        # Batched damped Newton, with an active-set constraint on alpha and an
        # Armijo line search. No arbitrary coefficient/intercept clipping.
        converged = ~fit_cells
        iterations = np.zeros(n_cells, dtype=int)
        for _ in range(self.n_iter):
            grad = gradient(beta)
            pg = projected_gradient(beta, grad)
            converged |= np.max(np.abs(pg), axis=1) <= self.tol
            active = fit_cells & ~converged
            if not active.any():
                break
            p = expit(np.einsum("crf,cf->cr", A, beta))
            w = R * p * (1.0 - p)
            H = np.einsum("crf,crg->cfg", A * w[:, :, None], A) + eyepen
            H += 1e-10 * eye[None]
            bound = (beta[:, F] <= lower + 1e-12) & (grad[:, F] > 0)
            H[bound, F, :] = 0.0
            H[bound, :, F] = 0.0
            H[bound, F, F] = 1.0
            rhs = -pg
            try:
                direction = np.linalg.solve(H, rhs[..., None])[..., 0]
            except np.linalg.LinAlgError:
                direction = np.einsum("cij,cj->ci", np.linalg.pinv(H), rhs)
            direction[~active] = 0.0
            old = loss(beta)
            length = np.ones(n_cells)
            accepted = ~active
            for _line in range(40):
                candidate = beta + length[:, None] * direction
                candidate[:, F] = np.maximum(candidate[:, F], lower)
                delta = candidate - beta
                improved = loss(candidate) <= old + 1e-4 * (grad * delta).sum(axis=1)
                accepted |= improved & active
                if accepted.all():
                    break
                length[~accepted] *= 0.5
            beta[active & accepted] = candidate[active & accepted]
            iterations[active] += 1

        grad = projected_gradient(beta, gradient(beta))
        converged = fit_cells & (np.max(np.abs(grad), axis=1) <= max(self.tol, 1e-6))
        fallback_solver = fit_cells & ~converged
        # Rare ill-conditioned cells receive an independent constrained solve.
        # A failed solve is never reported as a calibrated probability forecast.
        for cell in np.flatnonzero(fallback_solver):
            ac, yc, rc, pc = A[cell], Yb[cell], R[cell], pen[cell]
            def objective(coeff):
                eta = ac @ coeff
                return float(np.sum(rc * (np.logaddexp(0, eta) - yc * eta))
                             + 0.5 * np.sum(pc * coeff**2))
            def jacobian(coeff):
                return ac.T @ (rc * (expit(ac @ coeff) - yc)) + pc * coeff
            bounds = [(None, None)] * n_feat
            bounds[F] = (float(lower[cell]), None)
            result = minimize(objective, beta[cell], jac=jacobian, bounds=bounds,
                              method="L-BFGS-B", options={"maxiter": 500,
                              "ftol": 1e-14, "gtol": max(self.tol, 1e-8)})
            candidate_gradient = jacobian(result.x)
            if result.x[F] <= lower[cell] + 1e-10 and candidate_gradient[F] > 0:
                candidate_gradient[F] = 0
            ok = (np.all(np.isfinite(result.x)) and np.isfinite(result.fun)
                  and np.max(np.abs(candidate_gradient)) <= max(10*self.tol, 1e-5))
            if ok:
                beta[cell], converged[cell] = result.x, True
            iterations[cell] += int(result.nit)

        # ---- tercile probabilities from the single fitted equation ----------
        bpred, alpha = beta[:, :F], np.maximum(beta[:, F] / gs, self.alpha_min)
        b0 = beta[:, F + 1] - alpha * gc
        g1 = self._apply_g(t1, tfm)
        g2 = self._apply_g(t2, tfm)
        test_valid = np.isfinite(Fte).all(axis=2).T
        Fc = np.transpose(np.where(np.isfinite(Fte), Fte, 0.0), (1, 0, 2))
        lin = np.einsum("ctf,cf->ct", Fc, bpred) + b0[:, None]
        plo = expit(lin + alpha[:, None] * g1[:, None])
        pup = expit(lin + alpha[:, None] * g2[:, None])

        # alpha >= alpha_min > 0 and g non-decreasing => pup >= plo by
        # construction; keep the tracker as a numerical sanity check only.
        crossing = np.isfinite(plo) & np.isfinite(pup) & (pup < plo)
        self.last_crossing_fraction_ = float(np.mean(crossing)) if crossing.size else 0.0

        PB = plo
        PN = np.clip(pup - plo, 0.0, None)
        PA = np.clip(1.0 - pup, 0.0, None)
        tot = PB + PN + PA
        tot = np.where(tot > 0, tot, np.nan)
        probs = np.stack([PB / tot, PN / tot, PA / tot], axis=0)   # (3, n_cells, n_test)

        # Training-only empirical fallback at the SAME fixed events, preserving
        # ties and genuinely dry cells (uniform thirds would be incorrect).
        obs_valid = np.isfinite(ytr)
        counts = obs_valid.sum(axis=0)
        clim_cells = ~converged & np.isfinite(t1) & np.isfinite(t2) & (counts > 0)
        denominator = np.maximum(counts, 1)
        c1 = ((ytr <= t1) & obs_valid).sum(axis=0) / denominator
        c2 = ((ytr <= t2) & obs_valid).sum(axis=0) / denominator
        climatology = np.stack([c1, c2-c1, 1-c2])
        probs[:, clim_cells, :] = climatology[:, clim_cells, None]
        bad = ~(np.isfinite(t1) & np.isfinite(t2)) | (counts == 0)
        probs[:, bad, :] = np.nan
        probs = np.where(test_valid[None], probs, np.nan)
        lin = np.where(test_valid, lin, np.nan)
        self.fit_diagnostics_ = {"converged": converged, "fallback": clim_cells,
                                 "solver_fallback": fallback_solver, "iterations": iterations,
                                 "objective": loss(beta), "valid_training_count": nvalid}

        # stash the fit for full-distribution queries (predictive_quantiles)
        self._last_fit_ = {"lin": lin, "alpha": alpha, "transform": tfm,
                           "clim_cells": clim_cells, "bad": bad}

        return np.transpose(probs, (0, 2, 1))                      # (3, n_test, n_cells)

    # ------------------------------------------------------ framework contract
    def compute_model(self, X_train, y_train, X_test, y_test=None,
                      clim_year_start=None, clim_year_end=None,
                      clim_terciles=None, best_params=None, **kwargs):
        """
        Fit ELR on the training fold and return the test-fold tercile
        probabilities (dims probability, T, Y, X; probability=['PB','PN','PA']).

        X_train / X_test : standardized predictor cubes (T, Y, X[, M]).
        y_train          : predictand (T, Y, X), kept RAW (un-standardized):
                           values let g(q) and its coefficient penalty act in
                           physical units (so g='sqrt'
                           is valid for precipitation, per Wilks).
        clim_terciles    : optional (t1, t2) DataArrays (Y, X) of fixed
                           climatological boundaries in the SAME units as
                           y_train -- pass the same terciles your verification
                           uses (WAS_Verification.compute_class).
        best_params      : optional {'l2':..., 'g':..., 'quantiles':...}.
                           Overrides for this call; otherwise self.l2 /
                           self.threshold_transform / self.fit_quantiles are
                           used (which compute_hyperparameters updates in
                           place).
        """
        X_train = align_predictor_target(X_train, y_train, mode="season_year")
        if "M" in y_train.dims:
            y_train = y_train.isel(M=0, drop=True)
        y_train = y_train.transpose("T", "Y", "X")

        Ftr = self._features(X_train)
        Fte = self._features(X_test)

        ytr_s = y_train.stack(cell=("Y", "X")).transpose("T", "cell")
        Ftr_s = Ftr.stack(cell=("Y", "X")).transpose("T", "cell", "feat")
        Fte_s = Fte.stack(cell=("Y", "X")).transpose("T", "cell", "feat")

        t1da, t2da = self._event_thresholds(
            y_train, clim_year_start, clim_year_end, clim_terciles)
        t1 = t1da.stack(cell=("Y", "X")).transpose("cell").values.astype(float)
        t2 = t2da.stack(cell=("Y", "X")).transpose("cell").values.astype(float)

        l2_use = best_params.get("l2") if best_params else None
        g_use = best_params.get("g", "__self__") if best_params else "__self__"
        q_use = best_params.get("quantiles") if best_params else None
        probs = self._fit_predict(Ftr_s.values, ytr_s.values, Fte_s.values,
                                  t1, t2, l2=l2_use, transform=g_use,
                                  quantiles=q_use)

        # coords for full-distribution queries on this fit
        self._last_coords_ = {"T": Fte_s["T"], "cell": ytr_s["cell"]}

        out = xr.DataArray(
            probs,
            dims=("probability", "T", "cell"),
            coords={"probability": ["PB", "PN", "PA"],
                    "T": Fte_s["T"], "cell": ytr_s["cell"]},
        ).unstack("cell").transpose("probability", "T", "Y", "X")
        out.attrs.update(
            probability_method="extended_logistic_regression",
            elr_converged_cells=int(self.fit_diagnostics_["converged"].sum()),
            elr_empirical_fallback_cells=int(self.fit_diagnostics_["fallback"].sum()),
            elr_fallback_reference="training observations at the fixed climatological thresholds",
        )
        return out

    # ------------------------------------------------- full predictive dist.
    def predictive_quantiles(self, p_levels=(0.10, 0.25, 0.50, 0.75, 0.90)):
        """
        Predictive quantiles of the LAST fitted forecast -- Wilks (2009)'s
        central result: the single fitted equation delivers the full predictive
        distribution, so any quantile is available in closed form,

            q_p = g^{-1}( (logit(p) - b0 - beta.x) / alpha ).

        Call after compute_model / forecast. Requires a named invertible g
        ('identity', 'cbrt', 'sqrt'). With g='sqrt' negative arguments (i.e.
        p below the probability of zero precip) map to q_p = 0.

        Returns
        -------
        DataArray (quantile, T, Y, X) in the predictand's physical units.
        Cells that fell back to climatology or lack thresholds are NaN.
        """
        if self._last_fit_ is None or self._last_coords_ is None:
            raise RuntimeError("call compute_model or forecast first.")
        fit = self._last_fit_
        inv = self._resolve_inverse(fit["transform"])
        p = np.asarray(p_levels, dtype=float)
        if np.any(~np.isfinite(p) | (p <= 0.0) | (p >= 1.0)):
            raise ValueError("p_levels must lie strictly inside (0, 1).")

        logits = np.log(p) - np.log1p(-p)
        z = (logits[:, None, None] - fit["lin"][None]) / fit["alpha"][None, :, None]
        q = z if inv is None else inv(z)                     # (P, n_cells, n_test)
        q[:, fit["clim_cells"] | fit["bad"], :] = np.nan

        out = xr.DataArray(
            np.transpose(q, (0, 2, 1)),
            dims=("quantile", "T", "cell"),
            coords={"quantile": p, "T": self._last_coords_["T"],
                    "cell": self._last_coords_["cell"]},
        ).unstack("cell").transpose("quantile", "T", "Y", "X")
        return out

    # ------------------------------------------------------ hyperparameters
    @staticmethod
    def _year_groups(T):
        """Calendar-year group label per time step (falls back to position)."""
        if np.issubdtype(np.asarray(T.values).dtype, np.datetime64):
            return T.dt.year.values
        return np.arange(T.sizes["T"])

    def _year_splits(self, years, cv_folds, leave_one_year_out, nb_omit=0):
        """Whole-year splits with an optional symmetric calendar-year buffer."""
        if int(nb_omit) != nb_omit or nb_omit < 0:
            raise ValueError("nb_omit must be a non-negative integer.")
        gap = int(np.ceil(nb_omit / 2))
        years = np.asarray(years)
        uniq = np.unique(years)
        n = len(uniq)
        if leave_one_year_out or cv_folds >= n:
            folds = [np.array([u]) for u in uniq]
        elif gap > 0:
            folds = [block for block in np.array_split(uniq, cv_folds) if len(block)]
        else:
            rng = np.random.default_rng(self.random_state)
            folds = [uniq[b] for b in np.array_split(rng.permutation(n), cv_folds)]
        splits = []
        for val_years in folds:
            va = np.isin(years, val_years)
            tr = ~va & (np.abs(years[:, None] - val_years[None, :]) > gap).all(axis=1)
            if tr.sum() >= 2 and va.sum() >= 1:
                splits.append((np.where(tr)[0], np.where(va)[0]))
        return splits

    @staticmethod
    def _score_vs_class(probs, obs_class, kind):
        """RPS or multiclass log-loss of the forecast tercile probabilities
        against the framework's OBSERVED tercile CLASS (0=below, 1=normal,
        2=above) -- the same target WAS_Verification / the logistic class use.

        probs     : (3, n_val, n_cells) ordered PB, PN, PA
        obs_class : (n_val, n_cells) integer-coded tercile category
        """
        PB, PN, PA = probs[0], probs[1], probs[2]
        o = obs_class
        valid = (np.isfinite(o) & np.isfinite(PB) & np.isfinite(PN) & np.isfinite(PA))
        if not valid.any():
            return np.nan
        below = (o == 0)
        middle = (o == 1)
        if kind == "rps":
            # cumulative (half-Brier) form over the two tercile boundaries
            rps = ((PB - below.astype(float)) ** 2
                   + ((PB + PN) - (below | middle).astype(float)) ** 2)
            return float(np.mean(rps[valid]))
        eps = 1e-12  # log_loss
        p_obs = np.where(below, PB, np.where(middle, PN, PA))
        p_obs = np.clip(p_obs, eps, 1.0)
        return float(-np.mean(np.log(p_obs[valid])))

    def compute_hyperparameters(self, Predictor, Predictand,
                                clim_year_start, clim_year_end,
                                clim_terciles=None, l2_grid=None, g_grid=None,
                                quantile_grid=None, score="rps", cv_folds=5,
                                leave_one_year_out=False, nb_omit=0):
        """
        Jointly tune the ridge penalty l2, the threshold transform g(q) AND the
        fitting-quantile set by year-grouped cross-validation. Winners are
        stored in self.l2 / self.threshold_transform / self.fit_quantiles (and
        self.best_params) so the existing CV branch / compute_model use them
        with no further plumbing.

        Why tune g(q) and the quantile set: Wilks (2009) shows the FORM of g(q)
        drives ELR quality (g(q)=sqrt(q) was substantially better than
        g(q)=q for raw precipitation), and fits one equation to several
        climatological quantiles at once. Here the data pick both, among
        non-decreasing g options, so p(q)=Pr{V<=q} stays monotone.

        Observed truth = tercile CLASSES: forecast probabilities are scored
        against WAS_Verification.compute_class (0=below, 1=normal, 2=above),
        the categorical target the RPSS verification uses. RPS by default, or
        multiclass log-loss.

        Standardization: only the PREDICTOR is standardized, within each inner
        training split's climatology window (after predictor_transform, if set).
        The PREDICTAND stays RAW so thresholds and g(q) act in physical units.

        Parameters
        ----------
        l2_grid : sequence of float, optional. Default np.logspace(-4, 1, 6).
        g_grid  : sequence of {'identity','cbrt','sqrt'} or callables, optional.
                  Default ['identity', 'cbrt'] (+ 'sqrt' only when all
                  thresholds are non-negative, i.e. raw precipitation).
        quantile_grid : sequence of quantile tuples, optional. Default
                  [terciles only, self.fit_quantiles] -- i.e. two-threshold
                  vs Wilks-style multi-threshold fit.
        score   : {'rps', 'log_loss'} -- proper score MINIMIZED over folds.
        cv_folds, leave_one_year_out : year-grouped CV controls.
        nb_omit : symmetric exclusion buffer, ceil(nb_omit / 2) calendar
                  years on each side of validation dates. Positive buffers
                  use contiguous validation blocks, avoiding fragmented training.
        clim_terciles : optional (t1, t2) in PHYSICAL units -- same fixed
                  boundaries your CV branch passes to compute_model.

        Returns
        -------
        best_params : dict {'l2':..., 'g':..., 'quantiles':...}
        """
        if score not in ("rps", "log_loss"):
            raise ValueError("score must be 'rps' or 'log_loss'.")
        l2_grid = [float(v) for v in (np.logspace(-4, 1, 6) if l2_grid is None
                                      else l2_grid)]
        if not l2_grid or any(not np.isfinite(v) or v < 0 for v in l2_grid):
            raise ValueError("l2_grid must contain finite non-negative penalties.")
        if int(cv_folds) < 2:
            raise ValueError("cv_folds must be at least 2.")

        if "M" in Predictand.dims:
            Predictand = Predictand.isel(M=0, drop=True)
        Predictand = Predictand.transpose("T", "Y", "X")

        # Events are fixed before inner splits; preprocessing is fitted afresh
        # inside each split. Explicit climatological thresholds take precedence.
        Predictor = align_predictor_target(Predictor, Predictand, mode="season_year")
        t1da, t2da = self._event_thresholds(
            Predictand, clim_year_start, clim_year_end, clim_terciles)
        obs_class = xr.where(Predictand <= t1da, 0, xr.where(Predictand <= t2da, 1, 2)).where(
            np.isfinite(Predictand) & np.isfinite(t1da) & np.isfinite(t2da))
        y_s = Predictand.stack(cell=("Y", "X")).transpose("T", "cell")
        o_s = obs_class.stack(cell=("Y", "X")).transpose("T", "cell")
        Yv, Ov = y_s.values, o_s.values
        t1_fix = t1da.stack(cell=("Y", "X")).transpose("cell").values.astype(float)
        t2_fix = t2da.stack(cell=("Y", "X")).transpose("cell").values.astype(float)

        # ---- g(q) candidates: 'sqrt' needs non-negative thresholds ----------
        observed = Yv[np.isfinite(Yv)]
        can_sqrt = bool(observed.size and np.min(observed) >= 0.0
                        and not np.any(t1_fix < 0.0) and not np.any(t2_fix < 0.0))
        if g_grid is None:
            g_grid = ["identity", "cbrt"] + (["sqrt"] if can_sqrt else [])
        else:
            g_grid = [g for g in g_grid
                      if not (g == "sqrt" and not can_sqrt)]
        if not g_grid:
            g_grid = ["identity"]

        # ---- fitting-quantile candidates ------------------------------------
        if quantile_grid is None:
            quantile_grid = [(1.0 / 3.0, 2.0 / 3.0), tuple(self.fit_quantiles)]
        quantile_grid = [self._check_quantiles(qc) for qc in quantile_grid]
        if not quantile_grid:
            raise ValueError("quantile_grid must contain at least one candidate.")

        years = self._year_groups(Predictand["T"])
        splits = self._year_splits(years, cv_folds, leave_one_year_out, nb_omit)
        if not splits:
            raise ValueError("Not enough years to build a CV split for tuning.")
        prepared = []
        for tr_idx, va_idx in splits:
            train, validation = self._standardize_predictors(
                Predictor.isel(T=tr_idx), Predictor.isel(T=va_idx),
                clim_year_start, clim_year_end)
            ftrain = self._features(train).stack(cell=("Y", "X")).transpose("T", "cell", "feat")
            fvalidation = self._features(validation).stack(cell=("Y", "X")).transpose("T", "cell", "feat")
            prepared.append((tr_idx, va_idx, ftrain.values, fvalidation.values))

        scores = {}
        best_key, best_val = (l2_grid[0], g_grid[0], quantile_grid[0]), np.inf
        for qc in quantile_grid:
            for g in g_grid:
                for l2 in l2_grid:
                    fold_vals = []
                    for tr_idx, va_idx, ftrain, fvalidation in prepared:
                        probs = self._fit_predict(ftrain, Yv[tr_idx],
                                                  fvalidation, t1_fix, t2_fix,
                                                  l2=l2, transform=g,
                                                  quantiles=qc)
                        s = self._score_vs_class(probs, Ov[va_idx], score)
                        if np.isfinite(s):
                            fold_vals.append(s)
                    val = float(np.mean(fold_vals)) if fold_vals else np.inf
                    gname = g if isinstance(g, str) else getattr(g, "__name__", "callable")
                    scores[(gname, l2, tuple(np.round(qc, 4)))] = val
                    if val < best_val:
                        best_val, best_key = val, (l2, g, qc)

        best_l2, best_g, best_qc = best_key
        if not np.isfinite(best_val):
            raise ValueError("ELR tuning has no finite validation scores.")
        self.l2 = float(best_l2)
        self.threshold_transform = best_g
        self.fit_quantiles = best_qc
        self.best_params = {"l2": float(best_l2), "g": best_g,
                            "quantiles": tuple(best_qc)}
        self.cv_scores_ = {"score": score, "values": scores,
                           "best": dict(self.best_params)}
        self._last_fit_ = None
        self._last_coords_ = None
        _report_best_params("elr", self.best_params, getattr(self, "verbose", 1),
                            score=best_val)
        return self.best_params

    # ------------------------------------------------------------- operational
    def forecast(self, Predictant, clim_year_start, clim_year_end,
                 Predictor, Predictor_for_year, clim_terciles=None, **kwargs):
        """
        Operational ELR tercile forecast for a single target year.

        Fits ELR on all available years and returns the tercile probabilities
        for the target year (dims probability, T, Y, X). Only the PREDICTOR is
        transformed (predictor_transform) and standardized (historical
        climatology, same stats applied to the forecast year); the PREDICTAND
        is kept RAW so thresholds and g(q) stay in physical units.

        After this call, `predictive_quantiles(p_levels)` returns the full
        predictive distribution of the target year from the same fit.
        """
        if "M" in Predictant.dims:
            Predictant = Predictant.isel(M=0, drop=True)
        Predictant = Predictant.transpose("T", "Y", "X")
        mask = xr.where(np.isfinite(Predictant).sum(dim='T') >= 3, 1.0, np.nan)

        Predictor = align_predictor_target(Predictor, Predictant, mode="season_year")
        Predictor_st, Predictor_year_st = self._standardize_predictors(
            Predictor, Predictor_for_year, clim_year_start, clim_year_end)

        prob = self.compute_model(
            Predictor_st, Predictant, Predictor_year_st,
            clim_year_start=clim_year_start, clim_year_end=clim_year_end,
            clim_terciles=clim_terciles)

        prob = _assign_target_time(prob, Predictor_for_year, Predictant)

        # keep predictive_quantiles consistent with the relabelled target date
        if self._last_coords_ is not None:
            self._last_coords_ = {"T": prob["T"], "cell": self._last_coords_["cell"]}

        return (prob * mask).transpose("probability", "T", "Y", "X")

class WAS_mme_gaussian_process:
    """
    Gaussian Process Classifier MME (multiclass via One-vs-Rest in sklearn).
    
    Rigorous scientific implementation:
      - Spatial clustering based on historical predictand coherence.
      - Spatio-temporal subsampling.
      - **Grouped Cross-Validation (by Time)** to prevent spatial-autocorrelation data leakage.
    """

    def __init__(
        self,
        random_state=42,
        cv_folds=5,
        n_clusters=4,
        kernel_options=None,
        n_restarts_optimizer_options=(0, 2),
        max_iter_predict_options=(100, 200),
        max_train_samples=1500,  
        hpo_method='grid', 
        n_random_iter=10,
        n_trials=20,
        timeout=None, 
        n_jobs=-1,
        optuna_n_jobs=None,
        cv_n_jobs=-1,
        warm_start=False,
    ):
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.n_clusters = n_clusters

        if kernel_options is None:
            self.kernel_options = [
                C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)),
                C(1.0, (1e-2, 1e2)) * Matern(length_scale=1.0, nu=1.5, length_scale_bounds=(1e-2, 1e2)),
                C(1.0, (1e-2, 1e2)) * Matern(length_scale=1.0, nu=2.5, length_scale_bounds=(1e-2, 1e2)),
            ]
        else:
            self.kernel_options = kernel_options
            
        self.n_restarts_optimizer_options = tuple(n_restarts_optimizer_options)
        self.max_iter_predict_options = tuple(max_iter_predict_options)
        self.max_train_samples = int(max_train_samples)
        
        self.hpo_method = hpo_method.lower()
        if self.hpo_method not in ['grid', 'random', 'bayesian']:
            raise ValueError(f"hpo_method must be 'grid', 'random', or 'bayesian', got {hpo_method}")
        
        self.n_random_iter = n_random_iter
        self.n_trials = n_trials
        self.timeout = timeout
        if n_jobs == 0 or cv_n_jobs == 0 or optuna_n_jobs == 0:
            raise ValueError("n_jobs, cv_n_jobs and optuna_n_jobs cannot be 0.")
        self.n_jobs = int(n_jobs)
        self.cv_n_jobs = int(cv_n_jobs)
        self.optuna_n_jobs = 1 if optuna_n_jobs is None else int(optuna_n_jobs)
        self.warm_start = bool(warm_start)
        
        if self.hpo_method == 'bayesian' and not OPTUNA_AVAILABLE:
            warnings.warn("Optuna not available. Falling back to RandomizedSearchCV.")
            self.hpo_method = 'random'

        self.models = None

    def _subsample(self, X, y, groups, nmax):
        """Reproducible subsampling tracking temporal groups for rigorous CV."""
        if (nmax is None) or (nmax <= 0) or (X.shape[0] <= nmax):
            return X, y, groups
            
        rng = np.random.RandomState(self.random_state)
        idx = rng.choice(X.shape[0], size=nmax, replace=False)
        return X[idx], y[idx], groups[idx]

    def _create_param_grid(self):
        return {
            "kernel": self.kernel_options,
            "n_restarts_optimizer": list(self.n_restarts_optimizer_options),
            "max_iter_predict": list(self.max_iter_predict_options),
        }

    def _optuna_objective(self, trial, X_train, y_train, groups, cv_splitter):
        """Objective function for Optuna."""
        kernel_type = trial.suggest_categorical('kernel_type', ['RBF', 'Matern_1.5', 'Matern_2.5'])
        length_scale = trial.suggest_float('length_scale', 0.1, 10.0, log=True)
        constant_value = trial.suggest_float('constant_value', 0.1, 10.0, log=True)
        
        if kernel_type == 'RBF':
            kernel = C(constant_value) * RBF(length_scale=length_scale)
        elif kernel_type == 'Matern_1.5':
            kernel = C(constant_value) * Matern(length_scale=length_scale, nu=1.5)
        else:
            kernel = C(constant_value) * Matern(length_scale=length_scale, nu=2.5)
        
        kernel = kernel + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-2))
        
        n_restarts = trial.suggest_int('n_restarts_optimizer', min(self.n_restarts_optimizer_options), max(self.n_restarts_optimizer_options))
        max_iter = trial.suggest_int('max_iter_predict', min(self.max_iter_predict_options), max(self.max_iter_predict_options))
        
        model = GaussianProcessClassifier(
            kernel=kernel, n_restarts_optimizer=n_restarts, 
            max_iter_predict=max_iter, random_state=self.random_state,
            warm_start=self.warm_start
        )
        
        def score_fold(train_idx, val_idx):
            X_train_f, X_val_f = X_train[train_idx], X_train[val_idx]
            y_train_f, y_val_f = y_train[train_idx], y_train[val_idx]
            if len(np.unique(y_train_f)) < 2:
                return np.nan
            try:
                fold_model = GaussianProcessClassifier(
                    kernel=kernel, n_restarts_optimizer=n_restarts,
                    max_iter_predict=max_iter, random_state=self.random_state,
                    warm_start=self.warm_start)
                fold_model.fit(X_train_f, y_train_f)
                raw = fold_model.predict_proba(X_val_f)
                proba = np.zeros((raw.shape[0], 3), dtype=float)
                for j, cls in enumerate(fold_model.classes_.astype(int)):
                    if 0 <= cls <= 2:
                        proba[:, cls] = raw[:, j]
                return -log_loss(y_val_f, proba, labels=[0, 1, 2])
            except Exception:
                return -1e5
        splits = list(cv_splitter.split(X_train, y_train, groups=groups))
        if JOBLIB_AVAILABLE and self.cv_n_jobs != 1:
            scores = Parallel(n_jobs=self.cv_n_jobs)(
                delayed(score_fold)(tr, va) for tr, va in splits)
        else:
            scores = [score_fold(tr, va) for tr, va in splits]
        valid = np.asarray(scores, dtype=float)
        valid = valid[np.isfinite(valid)]
        return float(np.mean(valid)) if valid.size else -1e5

    def compute_hyperparameters(self, Predictors, Predictand,
                                clim_year_start=None, clim_year_end=None, *, target_is_class=False):
        """Tune with fold-local standardization, target encoding and climate zones."""
        return _strict_regional_hyperparameters(
            self, Predictors, Predictand, clim_year_start, clim_year_end, target_is_class=target_is_class)

    def compute_hyperparameters_legacy(self, Predictors, Predictand, clim_year_start, clim_year_end, *, target_is_class=False):
        """Computes best hyperparameters. Uses rigorous GroupKFold by Time."""
        Predictors = align_predictor_target(Predictors, Predictand, mode="season_year")
        if "M" in Predictand.dims:
            Predictand = Predictand.isel(M=0, drop=True)
            
        X_train_std = Predictors if target_is_class else standardize_timeseries(Predictors, clim_year_start, clim_year_end)
        
        cluster_da = _build_climate_feature_clusters(
            Predictand, self.n_clusters, random_state=self.random_state
        )
        clusters = np.unique(cluster_da.values)
        clusters = clusters[np.isfinite(clusters)]

        verify = WAS_Verification()
        y_train_std = Predictand if target_is_class else verify.compute_class(Predictand, clim_year_start, clim_year_end)
        X_train_std['T'] = y_train_std['T']

        best_params_dict = {}
        
        # Extract Time values for GroupKFold
        time_values = y_train_std['T'].values
        
        for c in clusters:
            mask_3d = (cluster_da == c).expand_dims({'T': y_train_std['T']})
            
            # Stack keeping coordinate tracking
            stacked_X = X_train_std.where(mask_3d).stack(sample=('T', 'Y', 'X'))
            X_stacked_c = stacked_X.transpose('sample', 'M').values
            y_stacked_c = _stack_masked(y_train_std, mask_3d, core=None).ravel()
            
            # Extract time identifiers for CV Grouping
            samples_coords = stacked_X.sample.values
            time_groups = np.array([s[0] for s in samples_coords])

            nan_mask_c = np.any(~np.isfinite(X_stacked_c), axis=1) | ~np.isfinite(y_stacked_c)
            X_clean_c = X_stacked_c[~nan_mask_c]
            y_clean_c = y_stacked_c[~nan_mask_c].astype(int)
            groups_clean_c = time_groups[~nan_mask_c]

            if len(X_clean_c) == 0 or len(np.unique(y_clean_c)) < 2:
                best_params_dict[c] = None
                continue

            X_clean_c, y_clean_c, groups_clean_c = self._subsample(
                X_clean_c, y_clean_c, groups_clean_c, self.max_train_samples
            )

            # CRITICAL FIX: GroupKFold avoids leaking spatial neighbors at same timestep into validation
            n_group = len(np.unique(groups_clean_c))
            if n_group < 2:
                best_params_dict[c] = None
                continue
            cv_splitter = GroupKFold(n_splits=min(self.cv_folds, n_group))

            if self.hpo_method == 'grid':
                param_grid = self._create_param_grid()
                model = GaussianProcessClassifier(random_state=self.random_state, warm_start=self.warm_start)
                grid_search = GridSearchCV(
                    estimator=model, param_grid=param_grid, cv=cv_splitter,
                    scoring=_tercile_logloss_scorer, n_jobs=self.n_jobs
                )
                grid_search.fit(X_clean_c, y_clean_c, groups=groups_clean_c)
                best_params_dict[c] = grid_search.best_params_
                
            elif self.hpo_method == 'random':
                param_dist = self._create_param_grid()
                model = GaussianProcessClassifier(random_state=self.random_state, warm_start=self.warm_start)
                random_search = RandomizedSearchCV(
                    estimator=model, param_distributions=param_dist, n_iter=self.n_random_iter,
                    cv=cv_splitter, scoring=_tercile_logloss_scorer, random_state=self.random_state, n_jobs=self.n_jobs
                )
                random_search.fit(X_clean_c, y_clean_c, groups=groups_clean_c)
                best_params_dict[c] = random_search.best_params_
                
            elif self.hpo_method == 'bayesian':
                study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
                
                def objective(trial):
                    return self._optuna_objective(trial, X_clean_c, y_clean_c, groups_clean_c, cv_splitter)
                
                study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout, n_jobs=self.optuna_n_jobs, show_progress_bar=False)
                
                if study.best_trial is None:
                    best_params_dict[c] = None
                    continue
                    
                bp = study.best_trial.params
                k_type = bp['kernel_type']
                kernel = C(bp['constant_value']) * (
                    RBF(length_scale=bp['length_scale']) if k_type == 'RBF' else 
                    Matern(length_scale=bp['length_scale'], nu=1.5 if k_type == 'Matern_1.5' else 2.5)
                ) + WhiteKernel(noise_level=1e-5)

                best_params_dict[c] = {
                    'kernel': kernel,
                    'n_restarts_optimizer': bp['n_restarts_optimizer'],
                    'max_iter_predict': bp['max_iter_predict']
                }

        _report_best_params("gp", best_params_dict, getattr(self, "verbose", 1))
        return best_params_dict, cluster_da

    def compute_model(self, X_train, y_train, X_test, y_test, clim_year_start, clim_year_end, best_params=None, cluster_da=None):
        """Fit models on train set and predict on test set."""
        X_train = align_predictor_target(X_train, y_train, mode="season_year")
        # Note: Implementation logic follows existing flow but cleaned up.
        time, lat, lon = X_test["T"], X_test["Y"], X_test["X"]
        n_time, n_lat, n_lon = len(time), len(lat), len(lon)

        if best_params is None or cluster_da is None or not _cluster_matches_training(cluster_da, y_train):
            best_params, cluster_da = self.compute_hyperparameters(X_train, y_train, clim_year_start, clim_year_end, target_is_class=True)

        predictions = np.full((n_time, n_lat, n_lon), np.nan)
        predictions_prob = np.full((n_time, n_lat, n_lon, 3), np.nan)
        self.models = {}

        for c, bp in best_params.items():
            if bp is None:
                continue
                
            mask_3d_train = (cluster_da == c).expand_dims({"T": X_train["T"]})
            mask_3d_test = (cluster_da == c).expand_dims({"T": X_test["T"]})

            X_train_stacked = xr.DataArray(
                _stack_masked(X_train, mask_3d_train, core="M"),
                dims=("sample", "M"),
            )
            y_train_stacked = _stack_masked(y_train, mask_3d_train, core=None).ravel()

            nan_mask_tr = np.any(~np.isfinite(X_train_stacked.values), axis=1) | ~np.isfinite(y_train_stacked)
            X_clean_tr = X_train_stacked.values[~nan_mask_tr]
            y_clean_tr = y_train_stacked[~nan_mask_tr].astype(int)

            if len(X_clean_tr) == 0 or len(np.unique(y_clean_tr)) < 2:
                continue

            # Dummy groups for subsampling outside of CV
            dummy_groups = np.zeros(len(y_clean_tr))
            X_clean_tr, y_clean_tr, _ = self._subsample(X_clean_tr, y_clean_tr, dummy_groups, self.max_train_samples)

            model_c = GaussianProcessClassifier(random_state=self.random_state, warm_start=self.warm_start, **bp)
            model_c.fit(X_clean_tr, y_clean_tr)
            self.models[c] = model_c

            X_test_stacked = xr.DataArray(
                _stack_masked(X_test, mask_3d_test, core="M"),
                dims=("sample", "M"),
            )
            nan_mask_te = np.any(~np.isfinite(X_test_stacked.values), axis=1)
            X_clean_te = X_test_stacked.values[~nan_mask_te]

            if len(X_clean_te) == 0:
                continue
                
            y_pred_c = model_c.predict(X_clean_te)
            y_prob_c = model_c.predict_proba(X_clean_te)

            # Reconstruct deterministic
            full_pred = np.full(X_test_stacked.shape[0], np.nan)
            full_pred[~nan_mask_te] = y_pred_c
            predictions = np.where(np.isnan(predictions), full_pred.reshape(n_time, n_lat, n_lon), predictions)

            # Reconstruct probabilistic mapping strictly to [0,1,2]
            full_prob = np.full((X_test_stacked.shape[0], 3), np.nan)
            cols = model_c.classes_.astype(int)
            tmp = np.zeros((y_prob_c.shape[0], 3))
            for j, cls in enumerate(cols):
                if 0 <= cls <= 2:
                    tmp[:, cls] = y_prob_c[:, j]
            full_prob[~nan_mask_te] = tmp
            predictions_prob = np.where(np.isnan(predictions_prob), full_prob.reshape(n_time, n_lat, n_lon, 3), predictions_prob)

        predicted_da = xr.DataArray(predictions, coords={"T": time, "Y": lat, "X": lon}, dims=["T", "Y", "X"])
        predicted_prob_da = xr.DataArray(predictions_prob, coords={"T": time, "Y": lat, "X": lon, "probability": ["PB", "PN", "PA"]}, dims=["T", "Y", "X", "probability"]).transpose("probability", "T", "Y", "X")

        return predicted_da, predicted_prob_da

    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictors_train, Predictor_for_year, best_params=None, cluster_da=None):
        """Train on entire history, forecast for 1 new time slice."""
        if "M" in Predictant.dims:
            Predictant = Predictant.isel(M=0, drop=True)

        verify = WAS_Verification()
        Predictant_cls = verify.compute_class(Predictant, clim_year_start, clim_year_end)
        
        Predictors_train_st = standardize_timeseries(Predictors_train, clim_year_start, clim_year_end)
        mean_val = Predictors_train.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean(dim="T")
        std_val = Predictors_train.sel(T=slice(str(clim_year_start), str(clim_year_end))).std(dim="T")
        Predictor_for_year_st = (Predictor_for_year - mean_val) / std_val

        Predictors_train_st["T"] = Predictant_cls["T"]

        det, prob = self.compute_model(
            X_train=Predictors_train_st,
            y_train=Predictant_cls,
            X_test=Predictor_for_year_st,
            y_test=None,
            clim_year_start=clim_year_start,
            clim_year_end=clim_year_end,
            best_params=best_params,
            cluster_da=cluster_da,
        )
        det = _assign_target_time(det, Predictor_for_year, Predictant)
        prob = _assign_target_time(prob, Predictor_for_year, Predictant)
        mask = xr.where(Predictant.notnull().any("T"), 1.0, np.nan)
        return det * mask, prob * mask

    def get_hpo_summary(self):
        summary = {
            'hpo_method': self.hpo_method,
            'n_clusters': self.n_clusters,
            'max_train_samples': self.max_train_samples,
            'cv_folds': self.cv_folds,
        }
        if self.hpo_method == 'random':
            summary['n_random_iter'] = self.n_random_iter
        elif self.hpo_method == 'bayesian':
            summary['n_trials'] = self.n_trials
        return summary


#################################################################################
################################################################################
### NGR
##############################################################################
##################################################################################


_SQRT_PI = np.sqrt(np.pi)
_EPS = 1e-8


# ===========================================================================
# 1. Corrected moment formulas
# ===========================================================================

def censored_normal_mean_var(mu, var, dry_threshold=0.0):
    """Mean and variance of Y = max(X, c) with X ~ N(mu, var), c = dry_threshold.

    Correct for ANY threshold (the original code assumed c = 0).
    Validated against 4e6-sample Monte-Carlo to <1e-2 absolute in mean and var
    over mu in [-1, 5], sigma in [1, 2], c in [0, 2].

        E[Y]   = c*Phi(alpha) + mu*(1-Phi(alpha)) + sigma*phi(alpha)
        E[Y^2] = c^2*Phi(alpha) + (mu^2+sigma^2)(1-Phi(alpha))
                 + sigma*(mu+c)*phi(alpha)
        alpha  = (c - mu)/sigma
    """
    mu = np.asarray(mu, float)
    sigma = np.sqrt(np.maximum(np.asarray(var, float), 1e-10))
    c = float(dry_threshold)
    alpha = (c - mu) / sigma
    Phi = norm.cdf(alpha)
    phi = norm.pdf(alpha)
    mean = c * Phi + mu * (1.0 - Phi) + sigma * phi
    e2 = c * c * Phi + (mu**2 + sigma**2) * (1.0 - Phi) + sigma * (mu + c) * phi
    return mean, np.maximum(e2 - mean**2, 0.0)


def truncated_normal_mean_var(mu, sigma, dry_threshold=0.0):
    """Mean/variance of X conditional on X>c, with stable tail arithmetic."""
    return _truncated_normal_mean_var(mu, sigma, dry_threshold)


# ===========================================================================
# 2. Closed-form CRPS (for minimum-CRPS estimation, the EMOS standard)
# ===========================================================================

def crps_gaussian(y, mu, sigma):
    """Analytic CRPS for N(mu, sigma^2) (Gneiting et al. 2005). Validated to
    <1e-4 vs quadrature."""
    sigma = np.maximum(np.asarray(sigma, float), _EPS)
    z = (np.asarray(y, float) - np.asarray(mu, float)) / sigma
    return sigma * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / _SQRT_PI)


def crps_censored_normal(y, mu, sigma, c=0.0):
    """CRPS for Y = max(X, c), X ~ N(mu, sigma^2), observation y >= c.

    Closed form (Jordan, Krüger & Lerch 2019, eq. for `crps_cnorm`, lower=c,
    upper=+inf). Validated against brute-force quadrature to <2e-5.
    """
    sigma = np.maximum(np.asarray(sigma, float), _EPS)
    mu = np.asarray(mu, float)
    z = (np.asarray(y, float) - mu) / sigma
    l = (c - mu) / sigma
    Phi_z, phi_z = norm.cdf(z), norm.pdf(z)
    Phi_l, phi_l = norm.cdf(l), norm.pdf(l)
    Phi_2l = norm.cdf(np.sqrt(2.0) * l)
    return sigma * (
        z * (2.0 * Phi_z - 1.0) + 2.0 * phi_z         # interior (Gaussian) part
        - Phi_l * Phi_l * l - 2.0 * Phi_l * phi_l      # atom at the lower bound c
        - (1.0 / _SQRT_PI) * (1.0 - Phi_2l)            # self-distance correction
    )


def crps_truncated_normal(y, mu, sigma, c=0.0):
    """CRPS for X | X > c, X ~ N(mu, sigma^2), observation y > c.

    Closed form (Thorarinsdottir & Gneiting 2010; `crps_tnorm`, lower=c,
    upper=+inf). Validated against brute-force quadrature to <2e-5.
    """
    sigma = np.maximum(np.asarray(sigma, float), _EPS)
    mu = np.asarray(mu, float)
    z = (np.asarray(y, float) - mu) / sigma
    l = (c - mu) / sigma
    Z = np.maximum(1.0 - norm.cdf(l), 1e-12)
    Phi_z, phi_z = norm.cdf(z), norm.pdf(z)
    Phi_2l = norm.cdf(np.sqrt(2.0) * l)
    return sigma / (Z * Z) * (
        z * Z * (2.0 * Phi_z + Z - 2.0)
        + 2.0 * phi_z * Z
        - (1.0 / _SQRT_PI) * (1.0 - Phi_2l)
    )


# ===========================================================================
# 3. Minimum-CRPS per-gridpoint estimator (drop-in for _fit_one_grid)
# ===========================================================================

# Variance links. EMOS papers use sigma^2 = c + d*S^2 (Gaussian; constrain
# c,d >= 0). A softplus link keeps positivity without hard bounds and is handy
# for the precip families; both are offered.
def _var_linear(c, d, s2):
    return np.maximum(c + d * s2, 1e-10)


def _var_softplus(c, d, s2):
    x = c + d * s2
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0) + 1e-10


_VAR_LINKS = {"linear": _var_linear, "softplus": _var_softplus}
_CRPS_FAMILY = {
    "gaussian": lambda y, mu, sig, c: crps_gaussian(y, mu, sig),
    "censored_normal": crps_censored_normal,
    "truncated_normal": crps_truncated_normal,
}


def fit_one_grid_min_crps(
    fbar,
    svar,
    y,
    family="censored_normal",
    dry_threshold=0.0,
    var_link="linear",
    init=None,
    bounds=None,
):
    """Minimum-CRPS EMOS fit at a single grid point.

    Parameters
    ----------
    fbar, svar, y : 1D arrays over time (ensemble mean, ensemble variance, obs)
    family        : 'gaussian' | 'censored_normal' | 'truncated_normal'
    dry_threshold : censoring/truncation point c (precip families)
    var_link      : 'linear' (sigma^2 = c + d*svar) or 'softplus'

    Returns (params[a,b,c,d], info_dict). For truncated_normal only the wet
    observations (y > c) enter the score, matching the conditional likelihood
    used in Thorarinsdottir & Gneiting (2010).
    """
    fbar = np.asarray(fbar, float)
    svar = np.asarray(svar, float)
    y = np.asarray(y, float)
    good = np.isfinite(fbar) & np.isfinite(svar) & np.isfinite(y)
    fbar, svar, y = fbar[good], svar[good], y[good]
    if y.size < 3:
        return np.array([0.0, 1.0, float(np.nanvar(y) if y.size else 1.0), 0.0]), {
            "success": False, "n_valid": int(y.size), "final_crps": np.nan,
        }

    link = _VAR_LINKS[var_link]
    crps_fn = _CRPS_FAMILY[family]

    if family == "truncated_normal":
        wet = y > dry_threshold
        fb_s, sv_s, y_s = fbar[wet], svar[wet], y[wet]
        if y_s.size < 3:
            fb_s, sv_s, y_s = fbar, svar, y
    else:
        fb_s, sv_s, y_s = fbar, svar, y

    def objective(p):
        a, b, c, d = p
        mu = a + b * fb_s
        sigma = np.sqrt(link(c, d, sv_s))
        val = crps_fn(y_s, mu, sigma, dry_threshold)
        return float(np.mean(val[np.isfinite(val)])) if np.any(np.isfinite(val)) else 1e10

    if init is None:
        # OLS warm start for (a, b); residual variance splits into (c, d)
        A = np.column_stack([np.ones_like(fb_s), fb_s])
        try:
            coef, *_ = np.linalg.lstsq(A, y_s, rcond=None)
            a0, b0 = float(coef[0]), float(coef[1])
        except Exception:
            a0, b0 = 0.0, 1.0
        resid_var = float(np.var(y_s - (a0 + b0 * fb_s))) if y_s.size > 1 else 1.0
        c0 = max(0.5 * resid_var, 1e-4)
        init = np.array([a0, b0, c0, 0.5])

    if bounds is None:
        lo = (1e-6, None) if var_link == "linear" else (None, None)
        bounds = [(None, None), (None, None), lo, (0.0, None)]

    # Reference: raw multi-model mean with climatological spread (a=0,b=1,d=0)
    c_clim = max(float(np.var(y_s)), 1e-4)
    ref = objective([0.0, 1.0, c_clim, 0.0])

    res = minimize(objective, x0=init, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-7})
    if res.success and np.all(np.isfinite(res.x)) and res.fun <= ref:
        return np.asarray(res.x, float), {
            "success": True, "n_valid": int(y.size),
            "final_crps": float(res.fun), "ref_crps": ref,
            "crpss_vs_raw": 1.0 - res.fun / ref if ref > 0 else np.nan,
        }
    # do-no-harm fallback to the raw MME mean
    return np.array([0.0, 1.0, c_clim, 0.0]), {
        "success": False, "n_valid": int(y.size),
        "final_crps": ref, "ref_crps": ref, "crpss_vs_raw": 0.0,
    }


# __all__ = [
#     "censored_normal_mean_var",
#     "truncated_normal_mean_var",
#     "crps_gaussian",
#     "crps_censored_normal",
#     "crps_truncated_normal",
#     "fit_one_grid_min_crps",
# ]


def _crps_gaussian(y, mu, sigma):
    """
    Analytic CRPS for a Gaussian predictive distribution N(mu, sigma^2).
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    eps = np.finfo(float).eps * 100.0
    sigma = np.maximum(sigma, eps)

    z = (y - mu) / sigma
    return sigma * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))


def _pairwise_abs_sum_sorted(x):
    """
    Sum_{i<j} |x_j - x_i| for a 1D array, computed in O(m log m).
    """
    x = np.sort(np.asarray(x, dtype=float))
    m = x.size
    if m < 2:
        return 0.0
    k = np.arange(1, m + 1, dtype=float)
    return float(np.sum((2.0 * k - m - 1.0) * x))


def _crps_ensemble(ens, obs):
    """
    Standard ensemble CRPS:
        mean |X - y| - 1/2 mean |X - X'|
    """
    ens = np.asarray(ens, dtype=float)
    ens = ens[np.isfinite(ens)]
    m = ens.size

    if m == 0:
        return np.nan
    if m == 1:
        return abs(ens[0] - obs)

    term1 = np.mean(np.abs(ens - obs))
    pairwise_sum = _pairwise_abs_sum_sorted(ens)
    term2 = pairwise_sum / (m * m)
    return term1 - term2


def _fair_crps_ensemble(ens, obs):
    """
    Fair ensemble CRPS (finite-ensemble correction):
        mean |X - y| - 1/(m(m-1)) sum_{i<j} |x_i - x_j|
    """
    ens = np.asarray(ens, dtype=float)
    ens = ens[np.isfinite(ens)]
    m = ens.size

    if m == 0:
        return np.nan
    if m == 1:
        return abs(ens[0] - obs)

    term1 = np.mean(np.abs(ens - obs))
    pairwise_sum = _pairwise_abs_sum_sorted(ens)
    term2 = pairwise_sum / (m * (m - 1.0))
    return term1 - term2


class WAS_mme_NGR_Gaussian:
    """
    Canonical Gaussian EMOS / NGR with gridwise fitting.

    Model
    -----
    Y_t | f_t ~ N(mu_t, sigma_t^2)

    mu_t      = a + b * fbar_t
    sigma_t^2 = c + d * s_t^2

    where
        fbar_t = ensemble mean
        s_t^2  = ensemble variance

    This class is intended for predictands that are approximately Gaussian
    after anomaly/standardization preprocessing. For nonnegative or
    zero-inflated variables such as precipitation totals, prefer a truncated,
    censored, gamma, or transformed variant instead of plain Gaussian EMOS.
    """

    def __init__(
        self,
        apply_to="all",
        alpha=0.10,
        test_direction="two-sided",
        bounds=None,
        min_train_years=10,
    ):
        valid_apply = {"all", "sig", "pos", "neg"}
        valid_test = {"two-sided", "greater", "less"}

        if apply_to not in valid_apply:
            raise ValueError(f"apply_to must be one of {valid_apply}")
        if test_direction not in valid_test:
            raise ValueError(f"test_direction must be one of {valid_test}")
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be between 0 and 1")

        self.apply_to = apply_to
        self.min_train_years = max(int(min_train_years), _recommended_min_years(4, floor=8))
        self.alpha = float(alpha)
        self.test_direction = test_direction

        self.bounds = {
            "a": (-np.inf, np.inf),
            "b": (-np.inf, np.inf),
            "c": (0.0, np.inf),
            "d": (0.0, np.inf),
        }
        if bounds is not None:
            self.bounds.update(bounds)

        self.params = None
        self.clim_terciles = None
        self.fitted = False
        self._xarray = False
        self._fit_stats = None
        self.attrs = {}

        self._member_dim = "M"
        self._time_dim = "T"
        self._lat_dim = "Y"
        self._lon_dim = "X"
        self._param_dim = "parameter"
        self._lat_coords = None
        self._lon_coords = None

    def __repr__(self):
        status = "fitted" if self.fitted else "unfitted"
        return (
            f"WAS_mme_NGR_Gaussian(apply_to={self.apply_to!r}, "
            f"alpha={self.alpha}, test_direction={self.test_direction!r}, "
            f"{status})"
        )

    @staticmethod
    def _safe_ens_stats(h):
        """
        h: array (M, T)
        returns mean(T), variance(T), std(T), valid_member_count(T)
        """
        fbar = np.nanmean(h, axis=0)
        counts = np.sum(np.isfinite(h), axis=0)
        var = np.full(h.shape[1], np.nan, dtype=float)

        for t in range(h.shape[1]):
            col = h[:, t]
            col = col[np.isfinite(col)]
            if col.size == 0:
                var[t] = np.nan
            elif col.size == 1:
                var[t] = 0.0
            else:
                var[t] = np.var(col, ddof=1)

        std = np.sqrt(np.maximum(var, 0.0))
        return fbar, var, std, counts

    def _regression_significant(self, r, n, direction=None):
        if direction is None:
            direction = self.test_direction

        if n <= 2 or not np.isfinite(r):
            return False
        if abs(r) >= 1.0 - 1e-12:
            return direction == "two-sided" or (direction == "greater" and r > 0) or (direction == "less" and r < 0)

        t_stat = r * np.sqrt((n - 2.0) / max(1.0 - r * r, 1e-12))
        if direction == "two-sided":
            p = 2.0 * tdist.sf(abs(t_stat), df=n - 2)
        elif direction == "greater":
            p = tdist.sf(t_stat, df=n - 2)
        else:
            p = tdist.cdf(t_stat, df=n - 2)
        return bool(p < self.alpha)

    def _should_calibrate(self, r, n):
        if self.apply_to == "all":
            return True
        if self.apply_to == "sig":
            return self._regression_significant(r, n, self.test_direction)
        if self.apply_to == "pos":
            return (r > 0.0) and self._regression_significant(r, n, "greater")
        if self.apply_to == "neg":
            return (r < 0.0) and self._regression_significant(r, n, "less")
        return True

    def _objective_one_grid(self, pars, fbar, svar, obs):
        a, b, c, d = pars
        mu = a + b * fbar
        sig2 = c + d * svar
        sigma = np.sqrt(np.maximum(sig2, 0.0))
        return float(np.nanmean(_crps_gaussian(obs, mu, sigma)))

    @staticmethod
    def _climatological_pars(y):
        """Climatological fallback [mean(y), 0, var(y), 0]: mu = mean(y_train),
        sigma^2 = var(y_train) (ddof=1), the same convention as NGR_NonGaussian.

        Used when the calibration gate is not passed, the optimiser fails or
        the training sample is too short. The former raw-ensemble fallback
        [0, 1, 0, 1] (mu = fbar, sigma^2 = ensemble spread) is over-confident
        precisely where there is no skill: with a few correlated products the
        spread is far below the climatological variance.
        """
        y = np.asarray(y, dtype=float)
        y = y[np.isfinite(y)]
        if y.size < 2:
            return np.array([np.nan, 0.0, np.nan, 0.0], dtype=float)
        return np.array([float(np.mean(y)), 0.0, float(np.var(y, ddof=1)), 0.0], dtype=float)

    def _fit_one_grid(self, h_gp, o_gp):
        """
        h_gp: (M, T)
        o_gp: (T,)
        """
        valid_time = np.isfinite(o_gp) & np.any(np.isfinite(h_gp), axis=0)
        if valid_time.sum() < self.min_train_years:
            return self._climatological_pars(o_gp), {
                "success": False,
                "n_valid": int(valid_time.sum()),
                "initial_score": np.nan,
                "final_score": np.nan,
                "score_reduction": np.nan,
                "r": np.nan,
                "nit": 0,
            }

        h = h_gp[:, valid_time]
        y = o_gp[valid_time]

        fbar, svar, sstd, counts = self._safe_ens_stats(h)
        mask = np.isfinite(fbar) & np.isfinite(svar) & np.isfinite(y)

        if mask.sum() < self.min_train_years:
            return self._climatological_pars(y), {
                "success": False,
                "n_valid": int(mask.sum()),
                "initial_score": np.nan,
                "final_score": np.nan,
                "score_reduction": np.nan,
                "r": np.nan,
                "nit": 0,
            }

        fbar = fbar[mask]
        svar = svar[mask]
        y = y[mask]

        if np.ptp(fbar) <= np.finfo(float).eps:
            intercept, slope, r = float(np.mean(y)), 0.0, 0.0
        else:
            lm = linregress(fbar, y)
            intercept, slope, r = float(lm.intercept), float(lm.slope), float(lm.rvalue)

        if not self._should_calibrate(r, int(mask.sum())):
            pars = self._climatological_pars(y)
            base_score = self._objective_one_grid(pars, fbar, svar, y)
            return pars, {
                "success": True,
                "n_valid": int(mask.sum()),
                "initial_score": base_score,
                "final_score": base_score,
                "score_reduction": 0.0,
                "r": r,
                "nit": 0,
            }

        resid = y - (intercept + slope * fbar)
        resid_var = float(np.nanvar(resid, ddof=1)) if resid.size > 1 else 1.0
        init = np.array([
            intercept,
            slope,
            max(resid_var * 0.25, 1e-6),
            0.75,
        ])

        bounds = [
            self.bounds["a"],
            self.bounds["b"],
            self.bounds["c"],
            self.bounds["d"],
        ]

        # Reference (and optimiser-failure fallback) is climatology, not the
        # raw ensemble: score_reduction then measures the CRPS gain over it.
        base_pars = self._climatological_pars(y)
        base_score = self._objective_one_grid(base_pars, fbar, svar, y)

        res = minimize(
            self._objective_one_grid,
            x0=init,
            args=(fbar, svar, y),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-7},
        )

        if res.success and np.all(np.isfinite(res.x)):
            final_pars = np.asarray(res.x, dtype=float)
            final_score = float(res.fun)
            nit = int(getattr(res, "nit", 0))
            success = True
        else:
            final_pars = base_pars
            final_score = base_score
            nit = int(getattr(res, "nit", 0))
            success = False

        red = np.nan
        if np.isfinite(base_score) and base_score > 0.0:
            red = (base_score - final_score) / base_score

        return final_pars, {
            "success": success,
            "n_valid": int(mask.sum()),
            "initial_score": base_score,
            "final_score": final_score,
            "score_reduction": red,
            "r": r,
            "nit": nit,
        }

    def fit(
        self,
        hcst_grid,
        obs_grid,
        clim_terciles=False,
        member_dim="M",
        time_dim="T",
        lat_dim="Y",
        lon_dim="X",
    ):
        use_xarray = (
            xr is not None
            and isinstance(hcst_grid, xr.DataArray)
            and isinstance(obs_grid, xr.DataArray)
        )

        self._member_dim = member_dim
        self._time_dim = time_dim
        self._lat_dim = lat_dim
        self._lon_dim = lon_dim

        if use_xarray:
            hcst_grid, obs_grid = xr.align(hcst_grid, obs_grid, join="exact", copy=False)
            hcst = hcst_grid.transpose(member_dim, time_dim, lat_dim, lon_dim).values
            obs = obs_grid.transpose(time_dim, lat_dim, lon_dim).values
            self._lat_coords = hcst_grid.coords[lat_dim]
            self._lon_coords = hcst_grid.coords[lon_dim]
            self.attrs = dict(hcst_grid.attrs)
            self._xarray = True
        else:
            hcst = np.asarray(hcst_grid, dtype=float)
            obs = np.asarray(obs_grid, dtype=float)
            self._xarray = False

        if hcst.ndim != 4:
            raise ValueError("hcst_grid must be 4D with shape (M, T, Y, X)")
        if obs.ndim != 3:
            raise ValueError("obs_grid must be 3D with shape (T, Y, X)")

        m, t, ny, nx = hcst.shape
        if obs.shape != (t, ny, nx):
            raise ValueError(f"obs_grid must have shape {(t, ny, nx)}, got {obs.shape}")

        params = np.full((4, ny, nx), np.nan, dtype=float)
        fit_stats = {
            "success": np.zeros((ny, nx), dtype=bool),
            "n_valid": np.zeros((ny, nx), dtype=int),
            "initial_score": np.full((ny, nx), np.nan, dtype=float),
            "final_score": np.full((ny, nx), np.nan, dtype=float),
            "score_reduction": np.full((ny, nx), np.nan, dtype=float),
            "r": np.full((ny, nx), np.nan, dtype=float),
            "nit": np.zeros((ny, nx), dtype=int),
        }

        for iy in range(ny):
            for ix in range(nx):
                pars, stat = self._fit_one_grid(hcst[:, :, iy, ix], obs[:, iy, ix])
                params[:, iy, ix] = pars
                for key in fit_stats:
                    fit_stats[key][iy, ix] = stat[key]

        if clim_terciles:
            terc = np.nanpercentile(obs, [100.0 / 3.0, 200.0 / 3.0], axis=0)
            if use_xarray:
                self.clim_terciles = xr.DataArray(
                    terc,
                    dims=("tercile", lat_dim, lon_dim),
                    coords={
                        "tercile": ["lower", "upper"],
                        lat_dim: self._lat_coords,
                        lon_dim: self._lon_coords,
                    },
                    name="climatological_terciles",
                )
            else:
                self.clim_terciles = terc

        if use_xarray:
            self.params = xr.DataArray(
                params,
                dims=(self._param_dim, lat_dim, lon_dim),
                coords={
                    self._param_dim: ["a", "b", "c", "d"],
                    lat_dim: self._lat_coords,
                    lon_dim: self._lon_coords,
                },
                name="emos_parameters",
                attrs=self.attrs,
            )
        else:
            self.params = params

        self._fit_stats = fit_stats
        self.fitted = True
        return self

    def _predict_numpy(
        self,
        fcst,
        quantiles=None,
        clim_terciles=False,
        return_synthetic_ensemble=False,
    ):
        if not self.fitted:
            raise RuntimeError("Call fit() before predict().")

        fcst = np.asarray(fcst, dtype=float)
        if fcst.ndim != 4:
            raise ValueError("fcst_grid must be 4D with shape (M, T, Y, X)")

        m, t, ny, nx = fcst.shape
        param_np = self.params.values if hasattr(self.params, "values") else self.params

        if param_np.shape != (4, ny, nx):
            raise ValueError(
                f"Parameter shape must be (4, {ny}, {nx}), got {param_np.shape}"
            )

        mu = np.full((t, ny, nx), np.nan, dtype=float)
        sigma = np.full((t, ny, nx), np.nan, dtype=float)
        synth = np.full((m, t, ny, nx), np.nan, dtype=float) if return_synthetic_ensemble else None

        for iy in range(ny):
            for ix in range(nx):
                h = fcst[:, :, iy, ix]
                fbar, svar, sstd, counts = self._safe_ens_stats(h)
                a, b, c, d = param_np[:, iy, ix]

                mu_gp = a + b * fbar
                sig2_gp = c + d * svar
                sigma_gp = np.sqrt(np.maximum(sig2_gp, 0.0))

                mu[:, iy, ix] = mu_gp
                sigma[:, iy, ix] = sigma_gp

                if return_synthetic_ensemble:
                    eps = np.finfo(float).eps * 100.0
                    sstd_safe = np.maximum(sstd, eps)
                    z = (h - fbar[None, :]) / sstd_safe[None, :]
                    synth[:, :, iy, ix] = mu_gp[None, :] + sigma_gp[None, :] * z

        out = {
            "calibrated_mean": mu,
            "calibrated_std": sigma,
        }

        if quantiles is not None:
            q = np.asarray(quantiles, dtype=float)
            qv = norm.ppf(q)[:, None, None, None] * sigma[None, :, :, :] + mu[None, :, :, :]
            out["calibrated_quantiles"] = qv

        if clim_terciles:
            if self.clim_terciles is None:
                raise RuntimeError("fit(..., clim_terciles=True) or set clim_terciles first.")
            terc = self.clim_terciles.values if hasattr(self.clim_terciles, "values") else self.clim_terciles
            lower = terc[0][None, :, :]
            upper = terc[1][None, :, :]

            scale = np.maximum(sigma, np.finfo(float).eps * 100.0)
            p_below = norm.cdf(lower, loc=mu, scale=scale)
            p_above = norm.sf(upper, loc=mu, scale=scale)
            p_near = 1.0 - p_below - p_above

            probs = np.stack([p_below, p_near, p_above], axis=0)
            probs = np.clip(probs, 0.0, 1.0)
            denom = np.sum(probs, axis=0, keepdims=True)
            denom = np.where(denom <= 0.0, 1.0, denom)
            probs = probs / denom
            out["tercile_probability"] = probs

        if return_synthetic_ensemble:
            out["synthetic_calibrated_ensemble"] = synth

        return out

    def predict(
        self,
        fcst_grid,
        quantiles=None,
        clim_terciles=False,
        return_synthetic_ensemble=False,
        member_dim="M",
        time_dim="T",
        lat_dim="Y",
        lon_dim="X",
    ):
        use_xarray = xr is not None and isinstance(fcst_grid, xr.DataArray)

        if use_xarray:
            fc = fcst_grid.transpose(member_dim, time_dim, lat_dim, lon_dim)
            fc_np = fc.values
            out = self._predict_numpy(
                fc_np,
                quantiles=quantiles,
                clim_terciles=clim_terciles,
                return_synthetic_ensemble=return_synthetic_ensemble,
            )

            ds = xr.Dataset(attrs=dict(fcst_grid.attrs))
            time_coords = fc.coords[time_dim]
            member_coords = fc.coords[member_dim]
            lat_coords = fc.coords[lat_dim]
            lon_coords = fc.coords[lon_dim]

            ds["calibrated_mean"] = xr.DataArray(
                out["calibrated_mean"],
                dims=(time_dim, lat_dim, lon_dim),
                coords={time_dim: time_coords, lat_dim: lat_coords, lon_dim: lon_coords},
            )
            ds["calibrated_std"] = xr.DataArray(
                out["calibrated_std"],
                dims=(time_dim, lat_dim, lon_dim),
                coords={time_dim: time_coords, lat_dim: lat_coords, lon_dim: lon_coords},
            )

            if "calibrated_quantiles" in out:
                ds["calibrated_quantiles"] = xr.DataArray(
                    out["calibrated_quantiles"],
                    dims=("quantile", time_dim, lat_dim, lon_dim),
                    coords={
                        "quantile": list(np.asarray(quantiles, dtype=float)),
                        time_dim: time_coords,
                        lat_dim: lat_coords,
                        lon_dim: lon_coords,
                    },
                )

            if "tercile_probability" in out:
                ds["tercile_probability"] = xr.DataArray(
                    out["tercile_probability"],
                    dims=("probability", time_dim, lat_dim, lon_dim),
                    coords={
                        "probability": ["PB", "PN", "PA"],
                        time_dim: time_coords,
                        lat_dim: lat_coords,
                        lon_dim: lon_coords,
                    },
                )

            if "synthetic_calibrated_ensemble" in out:
                ds["synthetic_calibrated_ensemble"] = xr.DataArray(
                    out["synthetic_calibrated_ensemble"],
                    dims=(member_dim, time_dim, lat_dim, lon_dim),
                    coords={
                        member_dim: member_coords,
                        time_dim: time_coords,
                        lat_dim: lat_coords,
                        lon_dim: lon_coords,
                    },
                )

            return ds

        return self._predict_numpy(
            fcst_grid,
            quantiles=quantiles,
            clim_terciles=clim_terciles,
            return_synthetic_ensemble=return_synthetic_ensemble,
        )

    def transform(self, fcst_grid, **kwargs):
        kwargs = dict(kwargs)
        kwargs["return_synthetic_ensemble"] = True
        out = self.predict(fcst_grid, **kwargs)

        if xr is not None and isinstance(out, xr.Dataset):
            return (
                out["synthetic_calibrated_ensemble"],
                out["calibrated_mean"],
                out["calibrated_std"],
            )

        return (
            out["synthetic_calibrated_ensemble"],
            out["calibrated_mean"],
            out["calibrated_std"],
        )

    def compute_model(
        self,
        X_train,
        y_train,
        X_test,
        obs_for_terciles=None,
        quantiles=None,
        clim_terciles=False,
        member_dim="M",
        time_dim="T",
        lat_dim="Y",
        lon_dim="X",
        return_synthetic_ensemble=False,
    ):
        X_train = align_predictor_target(X_train, y_train, mode="season_year")
        self.fit(
            X_train,
            y_train,
            clim_terciles=bool(clim_terciles or (obs_for_terciles is not None)),
            member_dim=member_dim,
            time_dim=time_dim,
            lat_dim=lat_dim,
            lon_dim=lon_dim,
        )

        if obs_for_terciles is not None:
            if xr is not None and isinstance(obs_for_terciles, xr.DataArray):
                obs_np = obs_for_terciles.transpose(time_dim, lat_dim, lon_dim).values
                terc = np.nanpercentile(obs_np, [100.0 / 3.0, 200.0 / 3.0], axis=0)
                self.clim_terciles = xr.DataArray(
                    terc,
                    dims=("tercile", lat_dim, lon_dim),
                    coords={
                        "tercile": ["lower", "upper"],
                        lat_dim: obs_for_terciles.coords[lat_dim],
                        lon_dim: obs_for_terciles.coords[lon_dim],
                    },
                )
            else:
                obs_np = np.asarray(obs_for_terciles, dtype=float)
                self.clim_terciles = np.nanpercentile(obs_np, [100.0 / 3.0, 200.0 / 3.0], axis=0)

        return self.predict(
            X_test,
            quantiles=quantiles,
            clim_terciles=clim_terciles,
            return_synthetic_ensemble=return_synthetic_ensemble,
            member_dim=member_dim,
            time_dim=time_dim,
            lat_dim=lat_dim,
            lon_dim=lon_dim,
        )

    forecast = predict

    def get_fit_stats(self):
        if not self.fitted or self._fit_stats is None:
            return None

        if self._xarray and self._lat_coords is not None and self._lon_coords is not None:
            out = {}
            for key, arr in self._fit_stats.items():
                out[key] = xr.DataArray(
                    arr,
                    dims=(self._lat_dim, self._lon_dim),
                    coords={self._lat_dim: self._lat_coords, self._lon_dim: self._lon_coords},
                    name=key,
                )
            return out

        return self._fit_stats

    def summary(self):
        if not self.fitted:
            print("Model not fitted.")
            return

        print("WAS_mme_NGR_Gaussian")
        print("=================")
        print(f"apply_to       : {self.apply_to}")
        print(f"alpha          : {self.alpha}")
        print(f"test_direction : {self.test_direction}")

        stats = self.get_fit_stats()
        if stats is None:
            return

        def _as_np(x):
            return x.values if hasattr(x, "values") else x

        success = _as_np(stats["success"])
        reduction = _as_np(stats["score_reduction"])
        nit = _as_np(stats["nit"])

        print()
        print(f"success rate   : {100.0 * np.mean(success):.1f}%")
        if np.any(success):
            print(f"mean score gain: {100.0 * np.nanmean(reduction[success]):.2f}%")
            print(f"mean iterations: {np.nanmean(nit[success]):.1f}")

######################################################################################################################################################################################################################################################################################


def _softplus(x):
    x = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _inverse_softplus(x):
    """Stable inverse of softplus for strictly positive values."""
    x = np.maximum(np.asarray(x, dtype=float), np.finfo(float).tiny)
    return x + np.log(-np.expm1(-x))


def _sigmoid(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _rankdata_average(a):
    a = np.asarray(a, dtype=float)
    n = a.size
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[order[j + 1]] == a[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def _boxcox_forward(y, lmbda, shift=1e-6):
    x = np.asarray(y, dtype=float) + shift
    if np.any(x <= 0):
        raise ValueError("Box-Cox requires y + shift > 0.")
    if abs(lmbda) < 1e-12:
        return np.log(x)
    return (np.power(x, lmbda) - 1.0) / lmbda


def _boxcox_inverse(z, lmbda, shift=1e-6):
    z = np.asarray(z, dtype=float)
    if abs(lmbda) < 1e-12:
        x = np.exp(z)
    else:
        base = 1.0 + lmbda * z
        base = np.maximum(base, 1e-12)
        x = np.power(base, 1.0 / lmbda)
    return np.maximum(x - shift, 0.0)


def _boxcox_log_jacobian(y, lmbda, shift=1e-6):
    x = np.asarray(y, dtype=float) + shift
    if np.any(x <= 0):
        return -np.inf
    if abs(lmbda) < 1e-12:
        return -np.log(x)
    return (lmbda - 1.0) * np.log(x)


def _censored_normal_logpdf(y, mu, sigma, dry_threshold=0.0):
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-8)

    zt = (dry_threshold - mu) / sigma
    out = np.empty_like(np.broadcast_arrays(y, mu, sigma)[0], dtype=float)

    dry = y <= dry_threshold
    if np.any(dry):
        out[dry] = norm.logcdf(zt[dry] if np.ndim(zt) else zt)

    wet = ~dry
    if np.any(wet):
        yw = y[wet]
        muw = mu[wet] if np.ndim(mu) else mu
        sw = sigma[wet] if np.ndim(sigma) else sigma
        out[wet] = norm.logpdf(yw, loc=muw, scale=sw)

    return out


def _truncated_normal_logpdf(y, mu, sigma, dry_threshold=0.0):
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-8)
    alpha = (dry_threshold - np.asarray(mu, dtype=float)) / sigma
    return stats.truncnorm.logpdf(y, alpha, np.inf, loc=mu, scale=sigma)


def _truncated_normal_mean_var(mu, sigma, dry_threshold=0.0):
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-8)
    alpha = (dry_threshold - np.asarray(mu, dtype=float)) / sigma
    mean, var = stats.truncnorm.stats(alpha, np.inf, loc=mu, scale=sigma, moments="mv")
    return mean, np.maximum(var, 0.0)


def _gamma_from_mean_var(mean, var):
    mean = np.maximum(np.asarray(mean, dtype=float), 1e-10)
    var = np.maximum(np.asarray(var, dtype=float), 1e-10)
    shape = mean**2 / var
    scale = var / mean
    return shape, scale


def _gamma_logpdf(y, mean, var, dry_threshold=0.0):
    y = np.asarray(y, dtype=float)
    x = y - dry_threshold
    x = np.maximum(x, 1e-12)
    k, theta = _gamma_from_mean_var(mean, var)
    return gamma_dist.logpdf(x, a=k, scale=theta)


def _fit_occurrence_logistic(wet_frac, fbar, is_dry, l2=1e-6):
    wet_frac = np.asarray(wet_frac, dtype=float)
    fbar = np.asarray(fbar, dtype=float)
    is_dry = np.asarray(is_dry, dtype=float)

    x = np.column_stack(
        [
            np.ones_like(wet_frac),
            wet_frac,
            np.nan_to_num(fbar, nan=np.nanmedian(fbar) if np.any(np.isfinite(fbar)) else 0.0),
        ]
    )

    means = np.zeros(x.shape[1], dtype=float)
    stds = np.ones(x.shape[1], dtype=float)
    for j in range(1, x.shape[1]):
        means[j] = np.nanmean(x[:, j])
        stds[j] = np.nanstd(x[:, j])
        if not np.isfinite(stds[j]) or stds[j] < 1e-10:
            stds[j] = 1.0
        x[:, j] = (x[:, j] - means[j]) / stds[j]

    def obj(beta):
        eta = x @ beta
        p = np.clip(_sigmoid(eta), 1e-10, 1.0 - 1e-10)
        nll = -np.sum(is_dry * np.log(p) + (1.0 - is_dry) * np.log(1.0 - p))
        pen = 0.5 * l2 * np.sum(beta[1:] ** 2)
        return nll + pen

    res = minimize(obj, x0=np.zeros(x.shape[1]), method="BFGS")
    beta = res.x if res.success else np.zeros(x.shape[1])

    return {
        "beta": beta,
        "means": means,
        "stds": stds,
        "success": bool(res.success),
    }


class WAS_mme_NGR_NonGaussian:
    """
    Precipitation-oriented EMOS-style postprocessor with optional predictive families:

    - family='censored_normal'
    - family='truncated_normal'
    - family='gamma'
    - family='transformed_gaussian'

    Notes
    -----
    1) Designed for nonnegative predictands such as precipitation.
    2) The predictand is censored at ``dry_threshold``: Y_c = max(Y, c).
       For families without an intrinsic atom, a hurdle occurrence model
       is used by default:
           P(Y_c = c) = p0
           Y_c | Y_c > c  ~ positive family
       Use :meth:`observation_event_values` for verification observations too.
    3) Fitting uses log-likelihood rather than analytic CRPS, so one coherent
       implementation can cover all families.
    """

    VALID_FAMILIES = {
        "censored_normal",
        "truncated_normal",
        "gamma",
        "transformed_gaussian",
    }
    VALID_OCC = {"auto", "none", "climatology", "logistic"}

    def __init__(
        self,
        family="censored_normal",
        occurrence_model="auto",
        dry_threshold=0.0,
        transform="boxcox",
        transform_shift=1e-6,
        alpha=0.10,
        apply_to="all",
        bounds=None,
        min_train_years=10,
    ):
        if family not in self.VALID_FAMILIES:
            raise ValueError(f"family must be one of {self.VALID_FAMILIES}")
        if occurrence_model not in self.VALID_OCC:
            raise ValueError(f"occurrence_model must be one of {self.VALID_OCC}")
        if apply_to not in {"all", "sig", "pos", "neg"}:
            raise ValueError("apply_to must be one of {'all', 'sig', 'pos', 'neg'}")
        if not np.isfinite(dry_threshold) or float(dry_threshold) < 0.0:
            raise ValueError("dry_threshold must be finite and nonnegative")
        if not np.isfinite(transform_shift) or float(transform_shift) <= 0.0:
            raise ValueError("transform_shift must be finite and strictly positive")
        if transform != "boxcox":
            raise ValueError("transform must be 'boxcox'")
        if not np.isfinite(alpha) or not 0.0 < float(alpha) < 1.0:
            raise ValueError("alpha must lie strictly between 0 and 1")
        if int(min_train_years) != min_train_years or int(min_train_years) < 1:
            raise ValueError("min_train_years must be a positive integer")

        self.family = family
        self.min_train_years = max(int(min_train_years), _recommended_min_years(4, floor=10))
        self.occurrence_model = occurrence_model
        self.dry_threshold = float(dry_threshold)
        self.transform = transform
        self.transform_shift = float(transform_shift)
        self.alpha = float(alpha)
        self.apply_to = apply_to

        self.bounds = {
            "a": (-np.inf, np.inf),
            "b": (-np.inf, np.inf),
            "c": (-np.inf, np.inf),
            "d": (0.0, np.inf),
        }
        if bounds is not None:
            unknown = set(bounds) - set(self.bounds)
            if unknown:
                raise ValueError(f"Unknown NGR parameter bounds: {sorted(unknown)}")
            self.bounds.update(bounds)
        for name, interval in self.bounds.items():
            if len(interval) != 2 or np.any(np.isnan(interval)) or interval[0] > interval[1]:
                raise ValueError(f"Invalid bounds for NGR parameter {name}")

        self.params = None
        self.occ_params = None
        self._occ_feature_means = None
        self._occ_feature_stds = None
        self.transform_param = None
        self.clim_terciles = None
        self.fitted = False
        self._fit_stats = None
        self._xarray = False
        self.attrs = {}

        self._member_dim = "M"
        self._time_dim = "T"
        self._lat_dim = "Y"
        self._lon_dim = "X"
        self._param_dim = "parameter"
        self._lat_coords = None
        self._lon_coords = None

    def __repr__(self):
        status = "fitted" if self.fitted else "unfitted"
        return (
            f"WAS_mme_NGR_NonGaussian(family={self.family!r}, "
            f"occurrence_model={self.occurrence_model!r}, {status})"
        )

    def _needs_hurdle(self):
        if self.family == "censored_normal":
            return False
        if self.occurrence_model == "none":
            return False
        return True

    def observation_event_values(self, values):
        """Map observations to the same censored support as forecast events."""
        array = np.asarray(values, dtype=float)
        if np.any(array[np.isfinite(array)] < 0.0):
            raise ValueError("NonGaussian NGR requires nonnegative observations; use Gaussian NGR for signed targets")
        if isinstance(values, xr.DataArray):
            return values.clip(min=self.dry_threshold)
        return np.maximum(array, self.dry_threshold)

    def _climatological_event_terciles(self, values):
        array = np.asarray(self.observation_event_values(values), dtype=float)
        flat = array.reshape(array.shape[0], -1)
        available = np.any(np.isfinite(flat), axis=0)
        terciles = np.full((2, flat.shape[1]), np.nan)
        if available.any():
            terciles[:, available] = np.nanpercentile(
                flat[:, available], [100.0/3.0, 200.0/3.0], axis=0)
        return terciles.reshape((2,) + array.shape[1:])

    @staticmethod
    def _safe_ens_stats(h, dry_threshold=0.0):
        finite = np.isfinite(h)
        count = finite.sum(axis=0)
        fbar = np.divide(np.where(finite, h, 0.0).sum(axis=0), count,
                         out=np.full(h.shape[1], np.nan), where=count > 0)
        s2 = np.full(h.shape[1], np.nan, dtype=float)
        wet_frac = np.full(h.shape[1], np.nan, dtype=float)
        for t in range(h.shape[1]):
            x = h[:, t]
            good = np.isfinite(x)
            xx = x[good]
            if xx.size == 0:
                s2[t] = np.nan
                wet_frac[t] = np.nan
            elif xx.size == 1:
                s2[t] = 0.0
                wet_frac[t] = np.mean(xx > dry_threshold)
            else:
                s2[t] = np.var(xx, ddof=1)
                wet_frac[t] = np.mean(xx > dry_threshold)
        return fbar, np.maximum(s2, 0.0), wet_frac

    def _regression_significant(self, r, n):
        if n <= 2 or not np.isfinite(r):
            return False
        if abs(r) >= 1.0 - 1e-12:
            return bool(self.apply_to in {"all", "sig"} or
                        (self.apply_to == "pos" and r > 0.0) or
                        (self.apply_to == "neg" and r < 0.0))
        t_stat = r * np.sqrt((n - 2.0) / max(1.0 - r * r, 1e-12))
        if self.apply_to == "sig" or self.apply_to == "all":
            p = 2.0 * tdist.sf(abs(t_stat), df=n - 2)
            return bool(p < self.alpha)
        if self.apply_to == "pos":
            p = tdist.sf(t_stat, df=n - 2)
            return bool((r > 0.0) and (p < self.alpha))
        if self.apply_to == "neg":
            p = tdist.cdf(t_stat, df=n - 2)
            return bool((r < 0.0) and (p < self.alpha))
        return False

    def _should_calibrate(self, fbar, y):
        good = np.isfinite(fbar) & np.isfinite(y)
        if good.sum() < 3:
            return False
        if self.apply_to == "all":
            return True
        if np.ptp(fbar[good]) <= np.finfo(float).eps:
            return False
        lm = linregress(fbar[good], y[good])
        return self._regression_significant(float(lm.rvalue), int(good.sum()))

    def _mu_var_from_pars(self, pars, fbar, s2):
        a, b, c, d = pars
        mu = a + b * fbar
        var = _softplus(c + d * s2) + 1e-10
        return mu, var

    def _pos_nll(self, pars, fbar, s2, y_all, wet_mask, lmbda=None):
        mu, var = self._mu_var_from_pars(pars, fbar, s2)
        sigma = np.sqrt(var)

        if self.family == "censored_normal":
            ll = _censored_normal_logpdf(y_all, mu, sigma, dry_threshold=self.dry_threshold)
            return -float(np.sum(ll)) if np.all(np.isfinite(ll)) else np.inf

        fmu = mu[wet_mask]
        fvar = var[wet_mask]
        fsig = sigma[wet_mask]
        yy = y_all[wet_mask]

        if yy.size == 0:
            return np.inf

        if self.family == "truncated_normal":
            ll = _truncated_normal_logpdf(yy, fmu, fsig, dry_threshold=self.dry_threshold)
            return -float(np.sum(ll)) if np.all(np.isfinite(ll)) else np.inf

        if self.family == "gamma":
            mean_pos = np.maximum(fmu, 1e-8)
            ll = _gamma_logpdf(yy, mean_pos, fvar, dry_threshold=self.dry_threshold)
            return -float(np.sum(ll)) if np.all(np.isfinite(ll)) else np.inf

        if self.family == "transformed_gaussian":
            if self.transform != "boxcox":
                raise NotImplementedError("Only Box-Cox is implemented here.")
            yy_shift = np.maximum(yy - self.dry_threshold, 0.0)
            z = _boxcox_forward(yy_shift, lmbda=lmbda, shift=self.transform_shift)
            lower = float(_boxcox_forward(0.0, lmbda, self.transform_shift))
            upper = -1.0 / lmbda if lmbda < -1e-12 else np.inf
            ll = stats.truncnorm.logpdf(z, (lower-fmu)/fsig, (upper-fmu)/fsig,
                                       loc=fmu, scale=fsig) + _boxcox_log_jacobian(
                yy_shift, lmbda=lmbda, shift=self.transform_shift
            )
            return -float(np.sum(ll)) if np.all(np.isfinite(ll)) else np.inf

        raise ValueError(f"Unknown family {self.family}")

    def _climatological_fit(self, observations, reason, n_valid, lmbda=None):
        """Fit an intercept-only law on training observations in its own units.

        This fallback deliberately ignores regression bounds: it is a distinct
        climatological model, not a successful conditional regression. The
        diagnostics distinguish it from calibration and from unavailable data.
        """
        y = np.asarray(observations, dtype=float)
        y = self.observation_event_values(y[np.isfinite(y)])
        stat = {"success": False, "available": bool(y.size),
                "fallback_reason": reason, "n_valid": int(n_valid),
                "n_observed": int(y.size), "final_nll": np.nan,
                "degenerate_at_dry": False}
        if y.size == 0:
            stat["fallback_reason"] = "no_observations"
            return np.full(4, np.nan), None, np.nan, stat
        wet = y > self.dry_threshold
        occ = ({"kind": "climatology", "p0": float(np.mean(~wet))}
               if self._needs_hurdle() else None)
        if not wet.any():
            stat["degenerate_at_dry"] = True
            stat["fallback_reason"] = "all_dry"
            return np.array([self.dry_threshold, 0.0, float(_inverse_softplus(1e-8)), 0.0]), occ, 0.0, stat

        target = y if self.family == "censored_normal" else y[wet]
        if self.family == "gamma":
            target = target - self.dry_threshold
        if self.family == "transformed_gaussian":
            # A log transform is identifiable even when there are too few wet
            # dates to estimate an additional Box-Cox parameter reliably.
            lmbda = 0.0 if lmbda is None or not np.isfinite(lmbda) else float(lmbda)
            target = _boxcox_forward(target - self.dry_threshold, lmbda, self.transform_shift)
        else:
            lmbda = np.nan
        initial_mean = float(np.mean(target))
        initial_var = max(float(np.var(target)), 1e-8 * max(1.0, initial_mean**2))
        initial = np.array([initial_mean, float(_inverse_softplus(initial_var))])
        zeros = np.zeros(y.size)

        def objective(pair):
            return self._pos_nll(np.array([pair[0], 0.0, pair[1], 0.0]),
                                 zeros, zeros, y, wet, lmbda)

        # Gamma has an inexpensive, well-conditioned intercept-only MLE.
        if self.family == "gamma" and np.ptp(target) > 0.0:
            try:
                shape, _, scale = gamma_dist.fit(target, floc=0.0)
                initial = np.array([shape * scale, float(_inverse_softplus(shape * scale**2))])
            except (ValueError, FloatingPointError):
                pass
        elif np.ptp(target) > 0.0:
            result = minimize(objective, initial, method="L-BFGS-B",
                              options={"maxiter": 300, "ftol": 1e-10, "gtol": 1e-6})
            if result.success and np.all(np.isfinite(result.x)) and np.isfinite(result.fun):
                initial = result.x
        pars = np.array([initial[0], 0.0, initial[1], 0.0])
        stat["final_nll"] = objective(initial)
        return pars, occ, lmbda, stat

    def _fit_one_grid(self, h_gp, o_gp):
        observed = np.asarray(o_gp, dtype=float)
        if np.any(observed[np.isfinite(observed)] < 0.0):
            raise ValueError("NonGaussian NGR requires nonnegative observations; use Gaussian NGR for signed targets")
        if (self.family != "censored_normal" and self.occurrence_model == "none"
                and np.any(observed[np.isfinite(observed)] <= self.dry_threshold)):
            raise ValueError("occurrence_model='none' requires all observations strictly above dry_threshold")
        valid = np.isfinite(observed) & np.any(np.isfinite(h_gp), axis=0)
        n_valid = int(valid.sum())
        if n_valid < self.min_train_years:
            return self._climatological_fit(observed, "insufficient_training", n_valid)

        h = h_gp[:, valid]
        y = self.observation_event_values(observed[valid])
        fbar, s2, wet_frac = self._safe_ens_stats(h, dry_threshold=self.dry_threshold)
        dry = y <= self.dry_threshold
        wet = ~dry
        if not wet.any():
            return self._climatological_fit(observed, "all_dry", n_valid)
        if self.family != "censored_normal" and wet.sum() < max(5, self.min_train_years // 2):
            return self._climatological_fit(observed, "insufficient_wet_training", n_valid)

        lmbda = np.nan
        if self.family == "transformed_gaussian":
            y_pos = y[wet] - self.dry_threshold
            lmbda = 0.0
            if np.ptp(y_pos) > 0.0:
                try:
                    transformed_fit = minimize_scalar(
                        lambda power: -float(stats.boxcox_llf(power, y_pos + self.transform_shift)),
                        bounds=(0.0, 2.0), method="bounded")
                    if transformed_fit.success and np.isfinite(transformed_fit.fun):
                        lmbda = float(transformed_fit.x)
                except (ValueError, FloatingPointError):
                    pass

        if not self._should_calibrate(fbar, y):
            return self._climatological_fit(observed, "signal_gate", n_valid, lmbda)

        occ_model = None
        if self._needs_hurdle():
            # Exact boundary occurrence probabilities avoid failed separated
            # logistic fits and do not manufacture dry events in all-wet data.
            occ_model = {"kind": "climatology", "p0": float(np.mean(dry))}
            if self.occurrence_model in {"auto", "logistic"} and dry.any() and wet.any():
                occ_fit = _fit_occurrence_logistic(wet_frac, fbar, dry.astype(float))
                if occ_fit["success"] and np.all(np.isfinite(occ_fit["beta"])):
                    occ_fit["kind"] = "logistic"
                    occ_model = occ_fit

        wet_target = y if self.family == "censored_normal" else y[wet]
        if self.family == "gamma":
            wet_target = wet_target - self.dry_threshold
        if self.family == "transformed_gaussian":
            wet_target = _boxcox_forward(y[wet] - self.dry_threshold, lmbda, self.transform_shift)
        wet_fbar = fbar if self.family == "censored_normal" else fbar[wet]
        if wet_target.size < 3 or np.ptp(wet_fbar) <= np.finfo(float).eps:
            init_a, init_b = float(np.mean(wet_target)), 0.0
        else:
            lm = linregress(wet_fbar, wet_target)
            init_a, init_b = float(lm.intercept), float(lm.slope)

        init_var = np.var(wet_target - (init_a + init_b * wet_fbar))
        x0 = np.array([init_a, init_b, float(_inverse_softplus(max(init_var, 1e-4))), 0.1])
        bounds = [self.bounds["a"], self.bounds["b"], self.bounds["c"], self.bounds["d"]]
        res = minimize(self._pos_nll, x0=x0, args=(fbar, s2, y, wet, lmbda),
                       method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 600, "ftol": 1e-9, "gtol": 1e-6})
        fitted_ok = bool(res.success and np.all(np.isfinite(res.x)) and np.isfinite(res.fun))
        if not fitted_ok:
            return self._climatological_fit(observed, "optimization_failed", n_valid, lmbda)
        return res.x, occ_model, lmbda, {
            "success": True, "available": True, "fallback_reason": "",
            "n_valid": n_valid, "n_observed": int(np.isfinite(observed).sum()),
            "degenerate_at_dry": False, "final_nll": float(res.fun),
        }

    def fit(
        self,
        hcst_grid,
        obs_grid,
        clim_terciles=False,
        member_dim="M",
        time_dim="T",
        lat_dim="Y",
        lon_dim="X",
    ):
        use_xarray = (
            xr is not None
            and isinstance(hcst_grid, xr.DataArray)
            and isinstance(obs_grid, xr.DataArray)
        )

        self._member_dim = member_dim
        self._time_dim = time_dim
        self._lat_dim = lat_dim
        self._lon_dim = lon_dim

        if use_xarray:
            hcst_grid, obs_grid = xr.align(hcst_grid, obs_grid, join="exact", copy=False)
            hcst = hcst_grid.transpose(member_dim, time_dim, lat_dim, lon_dim).values
            obs = obs_grid.transpose(time_dim, lat_dim, lon_dim).values
            self._lat_coords = hcst_grid.coords[lat_dim]
            self._lon_coords = hcst_grid.coords[lon_dim]
            self.attrs = dict(hcst_grid.attrs)
            self._xarray = True
        else:
            hcst = np.asarray(hcst_grid, dtype=float)
            obs = np.asarray(obs_grid, dtype=float)
            self._xarray = False

        if hcst.ndim != 4:
            raise ValueError("hcst_grid must have shape (M, T, Y, X)")
        if obs.ndim != 3:
            raise ValueError("obs_grid must have shape (T, Y, X)")

        m, t, ny, nx = hcst.shape
        if obs.shape != (t, ny, nx):
            raise ValueError(f"obs_grid must have shape {(t, ny, nx)}, got {obs.shape}")

        params = np.full((4, ny, nx), np.nan, dtype=float)
        occ_params = np.full((3, ny, nx), np.nan, dtype=float)
        occ_feature_means = np.full((2, ny, nx), np.nan, dtype=float)
        occ_feature_stds = np.full((2, ny, nx), np.nan, dtype=float)
        transform_param = np.full((ny, nx), np.nan, dtype=float)

        fit_stats = {
            "success": np.zeros((ny, nx), dtype=bool),
            "available": np.zeros((ny, nx), dtype=bool),
            "fallback_reason": np.full((ny, nx), "", dtype="U32"),
            "degenerate_at_dry": np.zeros((ny, nx), dtype=bool),
            "n_valid": np.zeros((ny, nx), dtype=int),
            "n_observed": np.zeros((ny, nx), dtype=int),
            "final_nll": np.full((ny, nx), np.nan, dtype=float),
        }

        for iy in range(ny):
            for ix in range(nx):
                pars, occ, lmbda, stat = self._fit_one_grid(hcst[:, :, iy, ix], obs[:, iy, ix])
                params[:, iy, ix] = pars
                transform_param[iy, ix] = lmbda
                for key in fit_stats:
                    fit_stats[key][iy, ix] = stat[key]

                if occ is not None:
                    if occ["kind"] == "climatology":
                        occ_params[:, iy, ix] = [occ["p0"], np.nan, np.nan]
                    elif occ["kind"] == "logistic":
                        beta = occ["beta"]
                        occ_params[: len(beta), iy, ix] = beta
                        occ_feature_means[:, iy, ix] = occ["means"][1:]
                        occ_feature_stds[:, iy, ix] = occ["stds"][1:]

        self.params = params
        self.occ_params = occ_params if self._needs_hurdle() else None
        self._occ_feature_means = occ_feature_means if self._needs_hurdle() else None
        self._occ_feature_stds = occ_feature_stds if self._needs_hurdle() else None
        self.transform_param = transform_param if self.family == "transformed_gaussian" else None

        self.clim_terciles = None
        if clim_terciles:
            terc = self._climatological_event_terciles(obs)
            self.clim_terciles = terc

        if use_xarray:
            self.params = xr.DataArray(
                params,
                dims=(self._param_dim, lat_dim, lon_dim),
                coords={
                    self._param_dim: ["a", "b", "c", "d"],
                    lat_dim: self._lat_coords,
                    lon_dim: self._lon_coords,
                },
                name="precip_emos_parameters",
                attrs=self.attrs,
            )
            if self.occ_params is not None:
                self.occ_params = xr.DataArray(
                    occ_params,
                    dims=("occ_parameter", lat_dim, lon_dim),
                    coords={
                        "occ_parameter": ["p0_or_b0", "b1", "b2"],
                        lat_dim: self._lat_coords,
                        lon_dim: self._lon_coords,
                    },
                    name="occurrence_parameters",
                )
            if self.transform_param is not None:
                self.transform_param = xr.DataArray(
                    transform_param,
                    dims=(lat_dim, lon_dim),
                    coords={lat_dim: self._lat_coords, lon_dim: self._lon_coords},
                    name="transform_parameter",
                )
            if self.clim_terciles is not None:
                self.clim_terciles = xr.DataArray(
                    self.clim_terciles,
                    dims=("tercile", lat_dim, lon_dim),
                    coords={
                        "tercile": ["lower", "upper"],
                        lat_dim: self._lat_coords,
                        lon_dim: self._lon_coords,
                    },
                    name="climatological_terciles",
                )

        self._fit_stats = fit_stats
        self.fitted = True
        return self

    def _cdf_positive(self, y, mu, var, family, lmbda=None):
        sigma = np.sqrt(np.maximum(var, 1e-10))
        y = np.asarray(y, dtype=float)

        if family == "censored_normal":
            return np.where(y < self.dry_threshold, 0.0, norm.cdf(y, loc=mu, scale=sigma))

        if family == "truncated_normal":
            alpha = (self.dry_threshold - mu) / sigma
            return stats.truncnorm.cdf(y, alpha, np.inf, loc=mu, scale=sigma)

        if family == "gamma":
            mean_pos = np.maximum(mu, 1e-8)
            k, theta = _gamma_from_mean_var(mean_pos, var)
            return gamma_dist.cdf(np.maximum(y - self.dry_threshold, 0.0), a=k, scale=theta)

        if family == "transformed_gaussian":
            yp = np.maximum(y - self.dry_threshold, 0.0)
            z = _boxcox_forward(yp, lmbda=lmbda, shift=self.transform_shift)
            lower = float(_boxcox_forward(0.0, lmbda, self.transform_shift))
            upper = -1.0 / lmbda if lmbda < -1e-12 else np.inf
            return stats.truncnorm.cdf(z, (lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma)

        raise ValueError(f"Unknown family {family}")

    def _ppf_positive(self, p, mu, var, family, lmbda=None):
        p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0 - 1e-12)
        sigma = np.sqrt(np.maximum(var, 1e-10))

        if family == "censored_normal":
            q = norm.ppf(p, loc=mu, scale=sigma)
            return np.maximum(q, self.dry_threshold)

        if family == "truncated_normal":
            alpha = (self.dry_threshold - mu) / sigma
            return stats.truncnorm.ppf(p, alpha, np.inf, loc=mu, scale=sigma)

        if family == "gamma":
            mean_pos = np.maximum(mu, 1e-8)
            k, theta = _gamma_from_mean_var(mean_pos, var)
            return self.dry_threshold + gamma_dist.ppf(p, a=k, scale=theta)

        if family == "transformed_gaussian":
            lower = float(_boxcox_forward(0.0, lmbda, self.transform_shift))
            upper = -1.0 / lmbda if lmbda < -1e-12 else np.inf
            z = stats.truncnorm.ppf(p, (lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma)
            return self.dry_threshold + _boxcox_inverse(z, lmbda=lmbda, shift=self.transform_shift)

        raise ValueError(f"Unknown family {family}")

    def _positive_mean_var(self, mu, var, family, lmbda=None):
        sigma = np.sqrt(np.maximum(var, 1e-10))

        if family == "censored_normal":
            return censored_normal_mean_var(mu, var, self.dry_threshold)

        if family == "truncated_normal":
            return _truncated_normal_mean_var(mu, sigma, dry_threshold=self.dry_threshold)

        if family == "gamma":
            mean_pos = np.maximum(mu, 1e-8)
            return mean_pos + self.dry_threshold, np.maximum(var, 0.0)

        if family == "transformed_gaussian":
            return self._boxcox_mean_var(mu, var, lmbda)

        raise ValueError(f"Unknown family {family}")

    def _boxcox_mean_var(self, mu, var, lmbda):
        """Moments of the fitted transformed/truncated normal distribution.

        Lambda=0 uses exact truncated-lognormal moments. Positive lambda uses
        adaptive integration in standard-normal coordinates, with bounds around
        each moment integrand's mode (including its right tail). This does not
        replace moments by a finite collection of central quantiles.
        """
        from scipy.integrate import quad

        mu, var = np.broadcast_arrays(np.asarray(mu, dtype=float), np.asarray(var, dtype=float))
        mean = np.full(mu.shape, np.nan)
        variance = np.full(mu.shape, np.nan)
        if lmbda is None or not np.isfinite(lmbda):
            return mean, variance
        if lmbda < 0.0:
            raise ValueError("Predictive Box-Cox moments require the fitted lambda range [0, 2]")
        lower = float(_boxcox_forward(0.0, lmbda, self.transform_shift))
        shift = self.dry_threshold - self.transform_shift
        moment_cache = {}
        for index in np.ndindex(mu.shape):
            location, vv = float(mu[index]), float(var[index])
            if not np.isfinite(location) or not np.isfinite(vv) or vv < 0.0:
                continue
            key = (location, vv)
            if key in moment_cache:
                mean[index], variance[index] = moment_cache[key]
                continue
            sd = np.sqrt(max(vv, 1e-10))
            alpha = (lower - location) / sd
            log_mass = norm.logsf(alpha)

            if alpha > 8.0:
                # A far-tail truncation concentrates the distribution in a
                # layer of width 1/alpha. Integrate its *excess* over the lower
                # bound, avoiding both missed endpoint mass and cancellation
                # between two raw moments close to transform_shift.
                if alpha > 1000.0:
                    inverse_square = 1.0 / alpha**2
                    mills_over_alpha = (1.0 + inverse_square - 2.0 * inverse_square**2
                                        + 10.0 * inverse_square**3)
                    log_density_scale = np.log(mills_over_alpha)
                else:
                    log_density_scale = norm.logpdf(alpha) - log_mass - np.log(alpha)

                def excess_moment(order):
                    if abs(lmbda) < 1e-12:
                        peak = max(alpha, order * sd)
                    else:
                        b = 1.0 + lmbda * location
                        disc = np.sqrt(b*b + 4.0*lmbda*order*sd**2)
                        peak = (2.0*order*sd/(disc+b) if b >= 0.0
                                else (disc-b)/(2.0*lmbda*sd))
                        peak = max(alpha, peak)
                    peak_v = max(float(order), alpha*(peak-alpha))
                    upper_v = max(40.0, alpha*(peak-alpha+12.0))

                    def log_integrand(v):
                        if v <= 0.0:
                            return -np.inf
                        dz = sd * v / alpha
                        if abs(lmbda) < 1e-12:
                            growth = dz
                        else:
                            growth = np.log1p(lmbda*dz / self.transform_shift**lmbda) / lmbda
                        log_excess = (np.log(self.transform_shift) +
                                      (growth if growth > 50.0 else np.log(np.expm1(growth))))
                        return order*log_excess - v - 0.5*(v/alpha)**2 + log_density_scale

                    log_peak = log_integrand(peak_v)
                    points = sorted({v for v in (1.0, 5.0, 10.0, 40.0, peak_v) if 0.0 < v < upper_v})
                    scaled, _ = quad(lambda v: np.exp(log_integrand(v)-log_peak),
                                     0.0, upper_v, points=points,
                                     epsabs=1e-11, epsrel=2e-9, limit=150)
                    return np.exp(log_peak+np.log(scaled)) if scaled > 0.0 else 0.0

                first, second = excess_moment(1), excess_moment(2)
                mean[index] = self.dry_threshold + first
                variance[index] = max(second-first**2, 0.0)
                moment_cache[key] = (mean[index], variance[index])
                continue

            def moment(order):
                if abs(lmbda) < 1e-12:
                    log_value = (order * location + 0.5 * order**2 * sd**2
                                 + norm.logsf(alpha - order * sd) - log_mass)
                    return np.exp(log_value)
                # Solve for the mode of x(z)^order * standard_normal(z).
                b = 1.0 + lmbda * location
                disc = np.sqrt(b * b + 4.0 * lmbda * order * sd**2)
                peak = (2.0 * order * sd / (disc + b) if b >= 0.0
                        else (disc - b) / (2.0 * lmbda * sd))
                peak = max(alpha, peak)
                lo = max(alpha, -12.0)
                hi = max(12.0, peak + 12.0, alpha + 12.0 / max(1.0, alpha))
                stretch = max(1.0, alpha)

                def log_integrand(z):
                    argument = lmbda * (location + sd * z)
                    log_base = np.log1p(argument) if argument > -1.0 else np.log(np.finfo(float).tiny)
                    return (order * log_base / lmbda - 0.5 * z * z
                            - 0.5 * np.log(2.0 * np.pi) - log_mass)

                log_peak = log_integrand(peak)
                scaled, _ = quad(
                    lambda u: np.exp(log_integrand(lo + u / stretch) - log_peak) / stretch,
                    0.0, (hi - lo) * stretch, epsabs=1e-11, epsrel=2e-9,
                    points=[(peak - lo) * stretch], limit=150)
                return np.exp(log_peak + np.log(scaled)) if scaled > 0.0 else 0.0

            first, second = moment(1), moment(2)
            mean[index] = shift + first
            variance[index] = max(second - first**2, 0.0)
            moment_cache[key] = (mean[index], variance[index])
        return mean, variance

    def _predict_numpy(
        self,
        fcst,
        quantiles=None,
        clim_terciles=False,
        return_synthetic_ensemble=False,
    ):
        if not self.fitted:
            raise RuntimeError("Call fit() before predict().")

        fcst = np.asarray(fcst, dtype=float)
        if fcst.ndim != 4:
            raise ValueError("fcst must have shape (M, T, Y, X)")

        m, t, ny, nx = fcst.shape

        param_np = self.params.values if hasattr(self.params, "values") else self.params
        occ_np = None if self.occ_params is None else (
            self.occ_params.values if hasattr(self.occ_params, "values") else self.occ_params
        )
        occ_mean_np = None if self._occ_feature_means is None else self._occ_feature_means
        occ_std_np = None if self._occ_feature_stds is None else self._occ_feature_stds
        lam_np = None if self.transform_param is None else (
            self.transform_param.values if hasattr(self.transform_param, "values") else self.transform_param
        )
        if np.shape(param_np) != (4, ny, nx):
            raise ValueError("Forecast spatial dimensions must match the fitted NGR grid")
        degenerate = None if self._fit_stats is None else self._fit_stats.get("degenerate_at_dry")

        mean_out = np.full((t, ny, nx), np.nan, dtype=float)
        std_out = np.full((t, ny, nx), np.nan, dtype=float)
        probs_out = None
        q_out = None
        synth_out = np.full((m, t, ny, nx), np.nan, dtype=float) if return_synthetic_ensemble else None

        if quantiles is not None:
            q_levels = np.asarray(quantiles, dtype=float)
            if (q_levels.ndim != 1 or np.any(~np.isfinite(q_levels))
                    or np.any((q_levels <= 0.0) | (q_levels >= 1.0))):
                raise ValueError("quantiles must be a finite one-dimensional sequence strictly between 0 and 1")
            q_out = np.full((q_levels.size, t, ny, nx), np.nan, dtype=float)

        if clim_terciles:
            if self.clim_terciles is None:
                raise ValueError("Climatological terciles have not been fitted or supplied")
            probs_out = np.full((3, t, ny, nx), np.nan, dtype=float)
            terc = self.clim_terciles.values if hasattr(self.clim_terciles, "values") else self.clim_terciles

        for iy in range(ny):
            for ix in range(nx):
                f_gp = fcst[:, :, iy, ix]
                fbar, s2, wet_frac = self._safe_ens_stats(f_gp, dry_threshold=self.dry_threshold)
                pars = param_np[:, iy, ix]
                if not np.all(np.isfinite(pars)):
                    continue
                if degenerate is not None and degenerate[iy, ix]:
                    mean_out[:, iy, ix] = self.dry_threshold
                    std_out[:, iy, ix] = 0.0
                    if q_out is not None:
                        q_out[:, :, iy, ix] = self.dry_threshold
                    if synth_out is not None:
                        synth_out[:, :, iy, ix] = np.where(np.isfinite(f_gp), self.dry_threshold, np.nan)
                    if probs_out is not None:
                        lower, upper = terc[:, iy, ix]
                        if np.isfinite(lower) and np.isfinite(upper):
                            f_lower = float(self.dry_threshold <= lower)
                            f_upper = float(self.dry_threshold <= upper)
                            probs_out[:, :, iy, ix] = np.array([f_lower, f_upper-f_lower, 1.0-f_upper])[:, None]
                    continue
                mu, var = self._mu_var_from_pars(pars, fbar, s2)

                lmbda = None if lam_np is None else lam_np[iy, ix]
                p0 = np.zeros_like(fbar)

                if self._needs_hurdle():
                    # A per-grid fallback can be climatological even when the
                    # requested occurrence model was logistic or auto.
                    if np.all(np.isnan(occ_np[1:, iy, ix])):
                        p0[:] = occ_np[0, iy, ix]
                    else:
                        beta = np.where(np.isfinite(occ_np[:, iy, ix]), occ_np[:, iy, ix], 0.0)
                        m1, m2 = occ_mean_np[:, iy, ix]
                        s1, s2s = occ_std_np[:, iy, ix]
                        if not np.isfinite(s1) or s1 < 1e-10:
                            s1 = 1.0
                        if not np.isfinite(s2s) or s2s < 1e-10:
                            s2s = 1.0
                        x1 = (wet_frac - m1) / s1
                        x2 = (np.nan_to_num(fbar, nan=0.0) - m2) / s2s
                        eta = beta[0] + beta[1] * x1 + beta[2] * x2
                        p0[:] = np.clip(_sigmoid(eta), 1e-10, 1.0 - 1e-10)

                if self.family == "censored_normal":
                    sigma = np.sqrt(np.maximum(var, 1e-10))
                    p0 = norm.cdf((self.dry_threshold - mu) / sigma)

                pos_mean, pos_var = self._positive_mean_var(mu, var, self.family, lmbda=lmbda)
                if self.family == "censored_normal":
                    full_mean = pos_mean
                    full_var = pos_var
                else:
                    qwet = 1.0 - np.clip(p0, 0.0, 1.0)
                    full_mean = (1.0 - qwet) * self.dry_threshold + qwet * pos_mean
                    full_var = qwet * pos_var + qwet * (1.0 - qwet) * (pos_mean-self.dry_threshold)**2

                mean_out[:, iy, ix] = full_mean
                std_out[:, iy, ix] = np.sqrt(np.maximum(full_var, 0.0))

                if quantiles is not None:
                    for iq, q in enumerate(q_levels):
                        if self.family == "censored_normal":
                            sigma = np.sqrt(np.maximum(var, 1e-10))
                            q_out[iq, :, iy, ix] = np.where(
                                q <= p0,
                                self.dry_threshold,
                                np.maximum(norm.ppf(q, loc=mu, scale=sigma), self.dry_threshold),
                            )
                        else:
                            r = np.where(q <= p0, np.nan, (q - p0) / np.maximum(1.0 - p0, 1e-12))
                            vals = np.where(
                                q <= p0,
                                self.dry_threshold,
                                self._ppf_positive(r, mu, var, self.family, lmbda=lmbda),
                            )
                            q_out[iq, :, iy, ix] = vals

                if clim_terciles:
                    lower = terc[0, iy, ix]
                    upper = terc[1, iy, ix]

                    if self.family == "censored_normal":
                        p_below = self._cdf_positive(lower, mu, var, self.family, lmbda=lmbda)
                        p_above = 1.0 - self._cdf_positive(upper, mu, var, self.family, lmbda=lmbda)
                    else:
                        f_lower = np.where(
                            lower < self.dry_threshold,
                            0.0,
                            p0 + (1.0 - p0) * self._cdf_positive(lower, mu, var, self.family, lmbda=lmbda),
                        )
                        f_upper = np.where(
                            upper < self.dry_threshold,
                            0.0,
                            p0 + (1.0 - p0) * self._cdf_positive(upper, mu, var, self.family, lmbda=lmbda),
                        )
                        p_below = f_lower
                        p_above = 1.0 - f_upper

                    p_near = 1.0 - p_below - p_above
                    probs = np.stack([p_below, p_near, p_above], axis=0)
                    probs = np.clip(probs, 0.0, 1.0)
                    denom = np.sum(probs, axis=0, keepdims=True)
                    denom = np.where(denom <= 0.0, 1.0, denom)
                    probs_out[:, :, iy, ix] = probs / denom

                if return_synthetic_ensemble:
                    for tt in range(t):
                        raw = f_gp[:, tt]
                        good = np.isfinite(raw)
                        if not np.any(good):
                            continue
                        ranks = _rankdata_average(raw[good])
                        probs = (ranks - 0.5) / good.sum()

                        if self.family == "censored_normal":
                            sigma = np.sqrt(max(var[tt], 1e-10))
                            vals = np.maximum(norm.ppf(probs, loc=mu[tt], scale=sigma), self.dry_threshold)
                        else:
                            vals = np.where(
                                probs <= p0[tt],
                                self.dry_threshold,
                                self._ppf_positive(
                                    (probs - p0[tt]) / max(1.0 - p0[tt], 1e-12),
                                    mu[tt],
                                    var[tt],
                                    self.family,
                                    lmbda=lmbda,
                                ),
                            )
                        tmp = np.full(m, np.nan)
                        tmp[good] = vals
                        synth_out[:, tt, iy, ix] = tmp

        # Missing members are not an observed zero-predictor forecast. Even a
        # climatological fallback does not silently fill unavailable lead times.
        available = np.any(np.isfinite(fcst), axis=0)
        mean_out = np.where(available, mean_out, np.nan)
        std_out = np.where(available, std_out, np.nan)
        if q_out is not None:
            q_out = np.where(available[None, ...], q_out, np.nan)
        if probs_out is not None:
            probs_out = np.where(available[None, ...], probs_out, np.nan)
        out = {
            "calibrated_mean": mean_out,
            "calibrated_std": std_out,
        }
        if q_out is not None:
            out["calibrated_quantiles"] = q_out
        if probs_out is not None:
            out["tercile_probability"] = probs_out
        if synth_out is not None:
            out["synthetic_calibrated_ensemble"] = synth_out
        return out

    def predict(
        self,
        fcst_grid,
        quantiles=None,
        clim_terciles=False,
        return_synthetic_ensemble=False,
        member_dim="M",
        time_dim="T",
        lat_dim="Y",
        lon_dim="X",
    ):
        use_xarray = xr is not None and isinstance(fcst_grid, xr.DataArray)

        if use_xarray:
            fc = fcst_grid.transpose(member_dim, time_dim, lat_dim, lon_dim)
            out = self._predict_numpy(
                fc.values,
                quantiles=quantiles,
                clim_terciles=clim_terciles,
                return_synthetic_ensemble=return_synthetic_ensemble,
            )

            ds = xr.Dataset(attrs=dict(fcst_grid.attrs))
            ds["calibrated_mean"] = xr.DataArray(
                out["calibrated_mean"],
                dims=(time_dim, lat_dim, lon_dim),
                coords={
                    time_dim: fc.coords[time_dim],
                    lat_dim: fc.coords[lat_dim],
                    lon_dim: fc.coords[lon_dim],
                },
            )
            ds["calibrated_std"] = xr.DataArray(
                out["calibrated_std"],
                dims=(time_dim, lat_dim, lon_dim),
                coords={
                    time_dim: fc.coords[time_dim],
                    lat_dim: fc.coords[lat_dim],
                    lon_dim: fc.coords[lon_dim],
                },
            )

            if "calibrated_quantiles" in out:
                ds["calibrated_quantiles"] = xr.DataArray(
                    out["calibrated_quantiles"],
                    dims=("quantile", time_dim, lat_dim, lon_dim),
                    coords={
                        "quantile": list(np.asarray(quantiles, dtype=float)),
                        time_dim: fc.coords[time_dim],
                        lat_dim: fc.coords[lat_dim],
                        lon_dim: fc.coords[lon_dim],
                    },
                )

            if "tercile_probability" in out:
                ds["tercile_probability"] = xr.DataArray(
                    out["tercile_probability"],
                    dims=("probability", time_dim, lat_dim, lon_dim),
                    coords={
                        "probability": ["PB", "PN", "PA"],
                        time_dim: fc.coords[time_dim],
                        lat_dim: fc.coords[lat_dim],
                        lon_dim: fc.coords[lon_dim],
                    },
                )

            if "synthetic_calibrated_ensemble" in out:
                ds["synthetic_calibrated_ensemble"] = xr.DataArray(
                    out["synthetic_calibrated_ensemble"],
                    dims=(member_dim, time_dim, lat_dim, lon_dim),
                    coords={
                        member_dim: fc.coords[member_dim],
                        time_dim: fc.coords[time_dim],
                        lat_dim: fc.coords[lat_dim],
                        lon_dim: fc.coords[lon_dim],
                    },
                )
            return ds

        return self._predict_numpy(
            fcst_grid,
            quantiles=quantiles,
            clim_terciles=clim_terciles,
            return_synthetic_ensemble=return_synthetic_ensemble,
        )

    forecast = predict

    def compute_model(
        self,
        X_train,
        y_train,
        X_test,
        obs_for_terciles=None,
        quantiles=None,
        clim_terciles=False,
        member_dim="M",
        time_dim="T",
        lat_dim="Y",
        lon_dim="X",
        return_synthetic_ensemble=False,
    ):
        X_train = align_predictor_target(X_train, y_train, mode="season_year")
        self.fit(
            X_train,
            y_train,
            clim_terciles=bool(clim_terciles or (obs_for_terciles is not None)),
            member_dim=member_dim,
            time_dim=time_dim,
            lat_dim=lat_dim,
            lon_dim=lon_dim,
        )

        if obs_for_terciles is not None:
            if xr is not None and isinstance(obs_for_terciles, xr.DataArray):
                obs_np = obs_for_terciles.transpose(time_dim, lat_dim, lon_dim).values
            else:
                obs_np = np.asarray(obs_for_terciles, dtype=float)

            terc = self._climatological_event_terciles(obs_np)

            if xr is not None and isinstance(self.clim_terciles, xr.DataArray):
                self.clim_terciles = xr.DataArray(
                    terc,
                    dims=("tercile", lat_dim, lon_dim),
                    coords={
                        "tercile": ["lower", "upper"],
                        lat_dim: obs_for_terciles.coords[lat_dim],
                        lon_dim: obs_for_terciles.coords[lon_dim],
                    },
                )
            else:
                self.clim_terciles = terc

        return self.predict(
            X_test,
            quantiles=quantiles,
            clim_terciles=clim_terciles,
            return_synthetic_ensemble=return_synthetic_ensemble,
            member_dim=member_dim,
            time_dim=time_dim,
            lat_dim=lat_dim,
            lon_dim=lon_dim,
        )

    def get_fit_stats(self):
        if self._fit_stats is None:
            return None
        if self._xarray and self._lat_coords is not None:
            out = {}
            for key, arr in self._fit_stats.items():
                out[key] = xr.DataArray(
                    arr,
                    dims=(self._lat_dim, self._lon_dim),
                    coords={self._lat_dim: self._lat_coords, self._lon_dim: self._lon_coords},
                    name=key,
                )
            return out
        return self._fit_stats

    def summary(self):
        print("WAS_mme_NGR_NonGaussian")
        print("================")
        print(f"family           : {self.family}")
        print(f"occurrence_model : {self.occurrence_model}")
        print(f"dry_threshold    : {self.dry_threshold}")
        if self.family == "transformed_gaussian":
            print(f"transform        : {self.transform}")
        if self._fit_stats is not None:
            succ = self._fit_stats["success"]
            print(f"success rate     : {100.0 * np.mean(succ):.1f}%")


################################################################################ MVA ############################################################


# -----------------------------
# Per-cell helpers for ufuncs
# -----------------------------
@njit(cache=True)
def _mva_train_numba(fc_train, ob_train):
    sm=0.0; n=0
    for i in range(ob_train.size):
        if np.isfinite(ob_train[i]): sm += ob_train[i]; n += 1
    clim_obs=sm/n if n else np.nan
    sf=0.0; nf=0
    for i in range(fc_train.shape[0]):
        for j in range(fc_train.shape[1]):
            v=fc_train[i,j]
            if np.isfinite(v): sf += v; nf += 1
    clim_fcst=sf/nf if nf else np.nan
    ss=0.0
    if nf>1:
        for i in range(fc_train.shape[0]):
            for j in range(fc_train.shape[1]):
                v=fc_train[i,j]
                if np.isfinite(v): ss += (v-clim_fcst)*(v-clim_fcst)
        sigma_e=np.sqrt(ss/(nf-1))
    else: sigma_e=np.nan
    so=0.0
    if n>1:
        for i in range(ob_train.size):
            v=ob_train[i]
            if np.isfinite(v): so += (v-clim_obs)*(v-clim_obs)
        sigma_ref=np.sqrt(so/(n-1))
    else: sigma_ref=np.nan
    return clim_obs,clim_fcst,sigma_e,sigma_ref


@njit(cache=True)
def _mva_apply_numba(fc_new, clim_obs, clim_fcst, sigma_e, sigma_ref):
    out=np.empty_like(fc_new,dtype=np.float64)
    valid=np.isfinite(sigma_e) and sigma_e>0.0 and np.isfinite(sigma_ref) and sigma_ref>=0.0
    scale=sigma_ref/sigma_e if valid else 1.0
    for i in range(fc_new.shape[0]):
        for j in range(fc_new.shape[1]):
            out[i,j]=(fc_new[i,j]-clim_fcst)*scale+clim_obs
    return out


def _train_cell_mva(fc_train: np.ndarray,  # (M, T)
                    ob_train: np.ndarray   # (T,)
                    ) -> np.ndarray:
    """
    Compute MVA parameters for one (Y,X) cell over training period.
    Returns [clim_obs, clim_fcst, sigma_e, sigma_ref]
    """
    if NUMBA_AVAILABLE:
        vals = _mva_train_numba(np.asarray(fc_train, dtype=np.float64),
                                np.asarray(ob_train, dtype=np.float64))
        return np.asarray(vals, dtype=np.float64)
    clim_obs  = np.nanmean(ob_train)
    clim_fcst = np.nanmean(fc_train)
    sigma_e   = np.nanstd(fc_train, ddof=1)
    sigma_ref = np.nanstd(ob_train, ddof=1)
    return np.array([clim_obs, clim_fcst, sigma_e, sigma_ref], dtype=np.float64)


def _apply_cell_mva(fc_new: np.ndarray,      # (M, Tnew) or (M,)
                    clim_obs: float,
                    clim_fcst: float,
                    sigma_e: float,
                    sigma_ref: float) -> np.ndarray:
    """
    Apply MVA to forecast members (Y,X cell). Vectorized over time.
    Fallback (ill-conditioned): climatology-only shift: (F - clim_fcst) + clim_obs
    """
    if fc_new.ndim == 1:
        fc_new = fc_new[:, None]
    if NUMBA_AVAILABLE:
        return _mva_apply_numba(np.asarray(fc_new, dtype=np.float64), float(clim_obs),
                                float(clim_fcst), float(sigma_e), float(sigma_ref))
    if (not np.isfinite(sigma_e)) or (sigma_e <= 0.0) or (not np.isfinite(sigma_ref)) or (sigma_ref < 0.0):
        return (fc_new - clim_fcst + clim_obs).astype(float, copy=False)
    scale = sigma_ref / sigma_e
    out = (fc_new - clim_fcst) * scale + clim_obs
    return out.astype(float, copy=False)


def _loocv_cell_mva(fc_cell: np.ndarray,  # (M, T)
                    ob_cell: np.ndarray   # (T,)
                    ) -> np.ndarray:
    """
    LOOCV over T for one (Y,X) cell.
    For each held-out t, train on T\\{t} and apply to column t.
    """
    M, T = fc_cell.shape
    out = np.full((M, T), np.nan, dtype=float)

    for t in range(T):
        mask = np.ones(T, dtype=bool); mask[t] = False
        fc_tr = fc_cell[:, mask]
        ob_tr = ob_cell[mask]

        clim_obs, clim_fcst, sigma_e, sigma_ref = _train_cell_mva(fc_tr, ob_tr)
        col = fc_cell[:, t]  # (M,)
        out[:, t] = _apply_cell_mva(col, clim_obs, clim_fcst, sigma_e, sigma_ref)[:, 0]
    return out


# -----------------------------
# Probability helpers (shared)
# -----------------------------
def _ppf_terciles_from_code(dist_code, shape, loc, scale):
    """
    Return terciles (T1, T2) from best-fit distribution parameters.
    Codes:
        1: norm, 2: lognorm, 3: expon, 4: gamma, 5: weibull_min,
        6: t,    7: poisson, 8: nbinom
    """
    if (not _HAS_SCIPY) or np.isnan(dist_code):
        return np.nan, np.nan

    code = int(dist_code)
    try:
        if code == 1:
            return (float(norm.ppf(1.0 / 3.0, loc=loc, scale=scale)),
                    float(norm.ppf(2.0 / 3.0, loc=loc, scale=scale)))
        elif code == 2:
            return (float(lognorm.ppf(1.0 / 3.0, s=shape, loc=loc, scale=scale)),
                    float(lognorm.ppf(2.0 / 3.0, s=shape, loc=loc, scale=scale)))
        elif code == 3:
            return (float(expon.ppf(1.0 / 3.0, loc=loc, scale=scale)),
                    float(expon.ppf(2.0 / 3.0, loc=loc, scale=scale)))
        elif code == 4:
            return (float(gamma.ppf(1.0 / 3.0, a=shape, loc=loc, scale=scale)),
                    float(gamma.ppf(2.0 / 3.0, a=shape, loc=loc, scale=scale)))
        elif code == 5:
            return (float(weibull_min.ppf(1.0 / 3.0, c=shape, loc=loc, scale=scale)),
                    float(weibull_min.ppf(2.0 / 3.0, c=shape, loc=loc, scale=scale)))
        elif code == 6:
            return (float(t.ppf(1.0 / 3.0, df=shape, loc=loc, scale=scale)),
                    float(t.ppf(2.0 / 3.0, df=shape, loc=loc, scale=scale)))
        elif code == 7:
            return (float(poisson.ppf(1.0 / 3.0, mu=shape, loc=loc)),
                    float(poisson.ppf(2.0 / 3.0, mu=shape, loc=loc)))
        elif code == 8:
            return (float(nbinom.ppf(1.0 / 3.0, n=shape, p=scale, loc=loc)),
                    float(nbinom.ppf(2.0 / 3.0, n=shape, p=scale, loc=loc)))
    except Exception:
        return np.nan, np.nan
    return np.nan, np.nan


def _weibull_shape_solver(k, M, V):
    """Root of Weibull shape 'k' so that Var/Mean^2 matches V/M^2."""
    if k <= 0:
        return -np.inf
    try:
        g1 = gamma_function(1.0 + 1.0/k)
        g2 = gamma_function(1.0 + 2.0/k)
        implied = (g2 / (g1**2)) - 1.0
        observed = V / (M**2)
        return observed - implied
    except Exception:
        return -np.inf


def _calc_tercile_probs_bestfit(best_guess, error_variance, T1, T2, dist_code, dof):
    """
    Generic tercile probabilities using best-fit family per grid cell.

    Inputs (per grid cell):
    - ``best_guess`` : 1D array over T (hindcast_det or forecast_det)
    - ``T1``, ``T2`` : scalar terciles from climatological best-fit distribution
    - ``dist_code`` : int, as in ``_ppf_terciles_from_code``
    - ``shape``, ``loc``, ``scale`` : scalars from climatology fit

    Strategy:
    - For each time step, build a predictive distribution of the same family:
        * Use ``best_guess[t]`` to adjust mean / location;
        * Keep shape parameters from climatology.
    - Then compute probabilities:
        * P(B) = F(T1)
        * P(N) = F(T2) - F(T1)
        * P(A) = 1 - F(T2)

    Parameters
    ----------
    best_guess : array-like
        Forecast/hindcast best estimates.
    error_variance : float
        Variance of prediction errors.
    T1 : float
        Lower tercile threshold.
    T2 : float
        Upper tercile threshold.
    dist_code : int
        Distribution code (1-8).
    shape : float
        Shape parameter.
    loc : float
        Location parameter.
    scale : float
        Scale parameter.

    Returns
    -------
    array-like
        Probabilities [P(B), P(N), P(A)].

    Notes
    -----
    - Uses :math:`F` as the CDF of the predictive distribution.
    """

    
    if not _HAS_SCIPY:
        n_time = np.asarray(best_guess).size
        return np.full((3, n_time), np.nan, float)

    best_guess = np.asarray(best_guess, float)
    ev = float(error_variance)
    n_time = best_guess.size
    out = np.full((3, n_time), np.nan, float)

    if (np.all(np.isnan(best_guess))
        or np.isnan(dist_code) or np.isnan(T1) or np.isnan(T2)
        or not np.isfinite(ev) or ev < 0.0):
        return out

    code = int(dist_code)

    if code == 1:  # Normal
        sd = np.sqrt(max(0.0, ev))
        out[0, :] = norm.cdf(T1, loc=best_guess, scale=sd)
        out[1, :] = norm.cdf(T2, loc=best_guess, scale=sd) - norm.cdf(T1, loc=best_guess, scale=sd)
        out[2, :] = 1.0 - norm.cdf(T2, loc=best_guess, scale=sd)

    elif code == 2:  # Lognormal
        mu_x = np.maximum(best_guess, 1e-12)
        var_x = np.maximum(ev, 1e-24)
        sigma = np.sqrt(np.log(1.0 + var_x / (mu_x**2)))
        mu = np.log(mu_x) - 0.5 * sigma**2
        out[0, :] = lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
        out[1, :] = lognorm.cdf(T2, s=sigma, scale=np.exp(mu)) - lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
        out[2, :] = 1.0 - lognorm.cdf(T2, s=sigma, scale=np.exp(mu))

    elif code == 3:  # Exponential
        sd = np.sqrt(max(0.0, ev))
        out[0, :] = expon.cdf(T1, loc=best_guess, scale=sd)
        out[1 :] = expon.cdf(T2, loc=best_guess, scale=sd) - expon.cdf(T1, loc=best_guess, scale=sd)
        out[2, :] = 1.0 - expon.cdf(T2, loc=best_guess, scale=sd)

    elif code == 4:  # Gamma
        mu_x = np.maximum(best_guess, 1e-12)
        var_x = np.maximum(ev, 1e-24)
        alpha = mu_x**2 / var_x
        theta = var_x / mu_x
        c1 = gamma.cdf(T1, a=alpha, scale=theta)
        c2 = gamma.cdf(T2, a=alpha, scale=theta)
        out[0, :] = c1
        out[1, :] = np.maximum(0.0, c2 - c1)
        out[2, :] = 1.0 - c2

    elif code == 5:  # Weibull
        var_x = float(np.maximum(ev, 0.0))
        for i in range(n_time):
            M = float(max(best_guess[i], 0.0))
            V = var_x
            if V <= 0.0 or M <= 0.0:
                continue
            try:
                k = float(fsolve(_weibull_shape_solver, 2.0, args=(M, V))[0])
                if k <= 0.0 or not np.isfinite(k):
                    continue
                lam = M / float(gamma_function(1.0 + 1.0/k))
                c1 = weibull_min.cdf(T1, c=k, loc=0.0, scale=lam)
                c2 = weibull_min.cdf(T2, c=k, loc=0.0, scale=lam)
                out[0, i] = c1
                out[1, i] = max(0.0, c2 - c1)
                out[2, i] = 1.0 - c2
            except Exception:
                continue

    elif code == 6:  # Student-t
        if dof is None or dof <= 2:
            return out
        scale = np.sqrt(max(0.0, ev) * (dof - 2.0) / max(dof, 1.0))
        c1 = t.cdf(T1, df=dof, loc=best_guess, scale=scale)
        c2 = t.cdf(T2, df=dof, loc=best_guess, scale=scale)
        out[0, :] = c1
        out[1, :] = np.maximum(0.0, c2 - c1)
        out[2, :] = 1.0 - c2

    elif code == 7:  # Poisson
        mu = np.maximum(best_guess, 0.0)
        c1 = poisson.cdf(T1, mu=mu)
        c2 = poisson.cdf(T2, mu=mu)
        out[0, :] = c1
        out[1, :] = np.maximum(0.0, c2 - c1)
        out[2, :] = 1.0 - c2

    elif code == 8:  # Negative Binomial (overdispersed)
        mu = np.maximum(best_guess, 1e-12)
        V  = np.maximum(ev, 1e-12)
        valid = V > mu
        n = np.where(valid, (mu**2) / (V - mu), np.nan)
        p = np.where(valid, mu / V, np.nan)
        c1 = nbinom.cdf(T1, n=n, p=p)
        c2 = nbinom.cdf(T2, n=n, p=p)
        out[0, :] = c1
        out[1, :] = np.maximum(0.0, c2 - c1)
        out[2, :] = 1.0 - c2

    return out


def _calc_tercile_probs_nonparam(best_guess, error_samples, first_tercile, second_tercile):
    """Non-parametric method using historical error samples."""
    best_guess = np.asarray(best_guess, float)   # (T,)
    dist_err   = np.asarray(error_samples, float)  # (T,)
    n_time = best_guess.size
    pred = np.full((3, n_time), np.nan, float)
    for t in range(n_time):
        x = best_guess[t]
        if not np.isfinite(x):
            continue
        dist = x + dist_err
        dist = dist[np.isfinite(dist)]
        if dist.size == 0:
            continue
        p_below   = float(np.mean(dist <= first_tercile))
        p_between = float(np.mean((dist > first_tercile) & (dist <= second_tercile)))
        p_above   = float(1.0 - (p_below + p_between))
        pred[0, t] = p_below
        pred[1, t] = p_between
        pred[2, t] = p_above
    return pred


# -----------------------------
# Public API (M, T, Y, X)
# -----------------------------
class WAS_mme_MVA:
    """Torralba et al. (2017) mean/variance adjustment for ensemble forecasts."""

    def __init__(self, dist_method: Literal["bestfit", "nonparam"] = "nonparam"):
        self.dist_method = str(dist_method).lower()
        if self.dist_method not in {"bestfit", "nonparam"}:
            raise ValueError("dist_method must be 'bestfit' or 'nonparam'.")
        self.params_: Optional[dict[str, xr.DataArray]] = None

    @staticmethod
    def _prepare_obs(obs):
        if "M" in obs.dims:
            obs = obs.isel(M=0, drop=True)
        if "T" not in obs.dims:
            raise ValueError("obs must include T.")
        return obs.transpose("T", ..., missing_dims="ignore")

    @staticmethod
    def _prepare_ensemble(ensemble, name):
        if "T" not in ensemble.dims:
            raise ValueError(f"{name} must include T.")
        if "M" not in ensemble.dims:
            ensemble = ensemble.expand_dims(M=[0])
        return ensemble.transpose("M", "T", ..., missing_dims="ignore")

    def fit(self, hindcast: xr.DataArray, obs: xr.DataArray) -> "WAS_mme_MVA":
        hindcast = self._prepare_ensemble(hindcast, "hindcast")
        obs = self._prepare_obs(obs)
        hindcast, obs = xr.align(hindcast, obs, join="exact", copy=False)
        if hindcast.sizes.get("T", 0) < 2:
            raise ValueError("MVA requires at least two aligned hindcast years.")

        packed = _pufunc(
            _train_cell_mva,
            hindcast,
            obs,
            input_core_dims=[["M", "T"], ["T"]],
            output_core_dims=[["param"]],
            dask="parallelized",
            vectorize=True,
            output_dtypes=[np.float64],
            dask_gufunc_kwargs={"output_sizes": {"param": 4}, "allow_rechunk": True},
        ).assign_coords(param=["clim_obs", "clim_fcst", "sigma_e", "sigma_ref"])
        self.params_ = {name: packed.sel(param=name, drop=True) for name in packed.param.values}
        return self

    def transform(self, forecast: xr.DataArray) -> xr.DataArray:
        if self.params_ is None:
            raise RuntimeError("Call fit() before transform().")
        forecast = self._prepare_ensemble(forecast, "forecast")
        template = forecast.isel(M=0, T=0, drop=True)
        fields = [self.params_[k].broadcast_like(template) for k in
                  ("clim_obs", "clim_fcst", "sigma_e", "sigma_ref")]

        def _apply(fc, co, cf, se, sr):
            return _apply_cell_mva(fc, float(co), float(cf), float(se), float(sr))

        out = _pufunc(
            _apply,
            forecast,
            *fields,
            input_core_dims=[["M", "T"], [], [], [], []],
            output_core_dims=[["M", "T"]],
            dask="parallelized",
            vectorize=True,
            output_dtypes=[float],
            dask_gufunc_kwargs={"allow_rechunk": True},
        )
        out.name = forecast.name or "calibrated_mva"
        return out.transpose("M", "T", ..., missing_dims="ignore")

    def fit_transform_loocv(self, hindcast: xr.DataArray, obs: xr.DataArray) -> xr.DataArray:
        hindcast = self._prepare_ensemble(hindcast, "hindcast")
        obs = self._prepare_obs(obs)
        hindcast, obs = xr.align(hindcast, obs, join="exact", copy=False)
        if hindcast.sizes.get("T", 0) < 3:
            raise ValueError("MVA LOOCV requires at least three aligned years.")
        if len(np.unique(_calendar_year_groups(hindcast["T"]))) != hindcast.sizes["T"]:
            raise ValueError("MVA standalone LOOCV requires one seasonal target per year; use grouped outer CV for multiple seasons.")
        out = _pufunc(
            _loocv_cell_mva,
            hindcast,
            obs,
            input_core_dims=[["M", "T"], ["T"]],
            output_core_dims=[["M", "T"]],
            dask="parallelized",
            vectorize=True,
            output_dtypes=[float],
            dask_gufunc_kwargs={"allow_rechunk": True},
        )
        out.name = (hindcast.name or "hindcast") + "_mva_loocv"
        return out

    def predict_mean(self, forecast: xr.DataArray) -> xr.DataArray:
        return self.transform(forecast).mean("M", skipna=True).transpose("T", "Y", "X")

    def _probability_helper(self):
        return WAS_mme_Weighted(equal_weighted=True, dist_method=self.dist_method)

    def compute_prob(
        self,
        Predictant,
        clim_year_start,
        clim_year_end,
        hindcast_cross,
        best_code_da=None,
        best_shape_da=None,
        best_loc_da=None,
        best_scale_da=None,
    ):
        hindcast_mean = _mean_if_member(hindcast_cross)
        return self._probability_helper().compute_prob(
            Predictant,
            clim_year_start,
            clim_year_end,
            hindcast_mean,
            best_code_da,
            best_shape_da,
            best_loc_da,
            best_scale_da,
        )

    def forecast(
        self,
        Predictant,
        clim_year_start,
        clim_year_end,
        hindcast_det,
        hindcast_cross,
        forecast_det,
        best_code_da=None,
        best_shape_da=None,
        best_loc_da=None,
        best_scale_da=None,
    ):
        obs = self._prepare_obs(Predictant).transpose("T", "Y", "X")
        self.fit(hindcast_det, obs)
        fc_mean = self.predict_mean(forecast_det)
        fc_mean = _assign_target_time(fc_mean, forecast_det, obs)
        hindcast_mean = _mean_if_member(hindcast_cross)
        return self._probability_helper().forecast(
            obs,
            clim_year_start,
            clim_year_end,
            hindcast_mean,
            fc_mean,
            best_code_da,
            best_shape_da,
            best_loc_da,
            best_scale_da,
        )

    
# ─────────────────────────────────────────────────────────────────────────────
# Module-level helper
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_tercile_probs(pb: float, pn: float, pa: float) -> np.ndarray:
    """
    Clip PB, PN, PA to [0, 1] and renormalise so that they sum exactly to 1.

    This handles small numerical artefacts that can arise from CDF evaluations
    (e.g. pb + pa > 1 due to floating-point error near the threshold values).
    """
    arr = np.clip([pb, pn, pa], 0.0, 1.0)
    s = float(arr.sum())
    return arr / s if np.isfinite(s) and s > 0.0 else np.full(3, np.nan)


def _bma_forecast_members(forecast, training_members, member_dim):
    """Preserve component identity even when forecast members are reordered."""
    expected = np.asarray(training_members)
    index = forecast.get_index(member_dim)
    if not index.is_unique or len(index) != len(expected) or not np.all(np.isin(expected, index)):
        raise ValueError("BMA forecast members must match the unique fitted member labels")
    return forecast.sel({member_dim: expected})


def _bma_spatial_select(array, forecast, lat_dim, lon_dim):
    # Nearest selection carries training coordinates. Reassign the requested
    # coordinates before arithmetic/packaging to avoid xarray intersections.
    coords = {lat_dim: forecast[lat_dim], lon_dim: forecast[lon_dim]}
    return array.sel(coords, method="nearest").assign_coords(coords)


def _bma_terciles(model, forecast, clim_terciles, obs_for_terciles,
                  tercile_q, time_dim, lat_dim, lon_dim):
    if len(tercile_q) != 2 or not 0 < tercile_q[0] <= tercile_q[1] < 1:
        raise ValueError("tercile_q must contain two ordered probabilities inside (0, 1)")
    if clim_terciles is not None:
        if "tercile" not in clim_terciles.dims and "quantile" in clim_terciles.dims:
            clim_terciles = clim_terciles.rename(quantile="tercile")
        model._require_dims(clim_terciles, ["tercile", lat_dim, lon_dim], "clim_terciles")
        if clim_terciles.sizes["tercile"] != 2:
            raise ValueError("clim_terciles must contain exactly two thresholds")
        result = _bma_spatial_select(clim_terciles, forecast, lat_dim, lon_dim)
        result = result.assign_coords(tercile=["lower", "upper"])
        source = "provided_climatological_thresholds"
    else:
        reference = obs_for_terciles if obs_for_terciles is not None else model._obs_grid
        reference = _bma_spatial_select(reference, forecast, lat_dim, lon_dim)
        result = model.compute_gridpoint_terciles_from_obs(
            reference, time_dim=time_dim, lat_dim=lat_dim, lon_dim=lon_dim, q=tercile_q)
        source = "observed_climatology" if obs_for_terciles is not None else "fit_observations"
    if bool((result.isel(tercile=0) > result.isel(tercile=1)).any()):
        raise ValueError("Climatological lower threshold cannot exceed upper threshold")
    result.attrs["tercile_reference"] = source
    return result


def _bma_quantile(cdf, q, component_quantiles):
    """A mixture quantile lies between the component quantiles at the same q."""
    if not np.isfinite(q) or not 0 < q < 1:
        raise ValueError("BMA predictive quantiles must lie strictly inside (0, 1)")
    bounds = np.asarray(component_quantiles, dtype=float)
    lo, hi = float(np.min(bounds)), float(np.max(bounds))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return np.nan
    if lo == hi:
        return lo
    # A zero-weight component can put the mixture root exactly at a bracket
    # endpoint; inverse/CDF roundoff must not invalidate that root.
    if abs(cdf(lo) - q) <= 8*np.finfo(float).eps:
        return lo
    if abs(cdf(hi) - q) <= 8*np.finfo(float).eps:
        return hi
    # Relative tolerance respects very small positive quantiles.
    return float(brentq(lambda value: cdf(value) - q, lo, hi,
                        xtol=np.finfo(float).tiny, rtol=1e-12, maxiter=200))


def _bma_output_metadata(ds):
    ds.attrs.update(probability_calibration="native",
                    residual_calibration_applicable="false",
                    calibration="native BMA predictive mixture fitted on supplied training data")
    for name in ("predictive_mean", "tercile_probability"):
        if name in ds:
            ds[name].attrs.update(ds.attrs)
    if "tercile_probability" in ds:
        ds["tercile_probability"].attrs["tercile_reference"] = ds["tercile_thresholds"].attrs["tercile_reference"]
    return ds


# ─────────────────────────────────────────────────────────────────────────────
class WAS_mme_FastBMA:
    """
    Grid-based Bayesian Model Averaging — SLSQP maximum-likelihood fitting.

    Supported ``model_type`` values
    --------------------------------
    ``'normal'``
        Gaussian BMA (Raftery et al. 2005).  Component: N(f_k, σ²) where f_k
        is the k-th raw ensemble member and σ is a single shared standard
        deviation fitted by maximum likelihood.

    ``'gamma'``
        Gamma mixture for strictly positive quantities (wind speed, etc.).
        Parameterisation: Γ(shape, scale = f_k / shape) so that
        E[Y | k] = f_k.  A single shared shape parameter is estimated.

    ``'gamma0'``
        Zero-adjusted Gamma mixture for precipitation, a pooled-occurrence
        adaptation rather than the exact Sloughter et al. (2007) model.
        The marginal model is

            P(Y = 0)  = 1 − POP
            P(Y > 0)  = POP · Γ-mixture(shape, f_k / shape)

        where POP is estimated via logistic regression on three predictors:
        intercept, ∛ens_mean, and fraction of wet ensemble members.
        Sloughter (2007) instead uses component-specific occurrence models,
        a zero-forecast indicator and gamma-distributed cube-root amounts.

    Tercile thresholds
    ------------------
    Provide either ``clim_terciles`` (dims: ``('tercile', Y, X)``) or
    ``obs_for_terciles`` (dims include ``time_dim, Y, X``); terciles are then
    computed per gridpoint.
    """

    def __init__(self, model_type: str = "normal", tol: float = 1e-3,
                 bias_correction: bool = True, bias_shrinkage: float = 0.25,
                 min_bias_years: int = 8) -> None:
        if model_type not in {"normal", "gamma", "gamma0"}:
            raise NotImplementedError(f"model_type='{model_type}' not supported.")
        self.model_type = model_type
        self.tol        = tol
        self.bias_correction = bool(bias_correction)
        self.bias_shrinkage = float(bias_shrinkage)
        if not (0.0 <= self.bias_shrinkage <= 1.0):
            raise ValueError("bias_shrinkage must be in [0,1].")
        self.min_bias_years = int(min_bias_years)
        if self.min_bias_years < 2 or not np.isfinite(tol) or tol <= 0:
            raise ValueError("min_bias_years must be >= 2 and tol must be positive")
        self.fitted     = False

        self.weights:         Optional[xr.DataArray] = None   # (M, Y, X)
        self.sigma:           Optional[xr.DataArray] = None   # (Y, X) — normal
        self.shape:           Optional[xr.DataArray] = None   # (Y, X) — gamma / gamma0
        self.logistic_params: Optional[xr.DataArray] = None   # (3, Y, X) — gamma0
        self.bias_intercept:   Optional[xr.DataArray] = None   # (M,Y,X)
        self.bias_slope:       Optional[xr.DataArray] = None   # (M,Y,X)

        self._member_dim: Optional[str] = None
        self._time_dim:   Optional[str] = None
        self._lat_dim:    Optional[str] = None
        self._lon_dim:    Optional[str] = None

    # ──────────────────────────────────────────────────────── utilities ──────

    @staticmethod
    def _require_dims(da: xr.DataArray, dims: Sequence[str], name: str) -> None:
        missing = [d for d in dims if d not in da.dims]
        if missing:
            raise ValueError(
                f"{name} is missing required dims: {missing}. Got dims={da.dims}"
            )

    @staticmethod
    def _normalize_weights(w: np.ndarray) -> np.ndarray:
        w = np.clip(np.asarray(w, dtype=float), 0.0, None)
        s = float(w.sum())
        return w / s if (np.isfinite(s) and s > 0.0) else np.full_like(w, 1.0 / w.size)

    @staticmethod
    def compute_gridpoint_terciles_from_obs(
        obs: xr.DataArray,
        *,
        time_dim: str = "T",
        lat_dim:  str = "Y",
        lon_dim:  str = "X",
        q: Tuple[float, float] = (1 / 3, 2 / 3),
    ) -> xr.DataArray:
        """
        Compute per-gridpoint tercile thresholds from a climatological sample.

        Returns
        -------
        xr.DataArray
            dims: ``('tercile', lat_dim, lon_dim)``; tercile ∈ ['lower', 'upper'].
        """
        # FIX #1: original code called WAS_EnsembleBMA._require_dims
        #         (a non-existent class).  Corrected to use this class's method.
        WAS_mme_FastBMA._require_dims(obs, [time_dim, lat_dim, lon_dim], "obs")
        qt = obs.quantile(list(q), dim=time_dim, skipna=True).rename(
            {"quantile": "tercile"}
        )
        return qt.assign_coords(tercile=["lower", "upper"])

    def _align_params_to_forecasts(
        self,
        new_forecasts: xr.DataArray,
        *,
        member_dim: str,
        time_dim:   str,
        lat_dim:    str,
        lon_dim:    str,
    ) -> Tuple[xr.DataArray, xr.DataArray, Optional[xr.DataArray]]:
        """Nearest-neighbour spatial alignment of stored parameters to the forecast grid."""
        self._require_dims(
            new_forecasts, [member_dim, time_dim, lat_dim, lon_dim], "new_forecasts"
        )
        sel = {lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]}
        w_ds = self.weights.sel(sel, method="nearest")

        if self.model_type == "normal":
            return w_ds, self.sigma.sel(sel, method="nearest"), None
        else:
            p1 = self.shape.sel(sel, method="nearest")
            p2 = (
                self.logistic_params.sel(sel, method="nearest")
                if self.model_type == "gamma0"
                else None
            )
            return w_ds, p1, p2

    # ──────────────────────────────────────────────────────────── fit ────────

    def fit(
        self,
        hcst_grid: xr.DataArray,
        obs_grid:  xr.DataArray,
        *,
        member_dim:  str = "M",
        time_dim:    str = "T",
        lat_dim:     str = "Y",
        lon_dim:     str = "X",
        min_samples: int = 12,
    ) -> "WAS_mme_FastBMA":
        """
        Fit BMA parameters at every gridpoint via SLSQP maximum-likelihood.

        Parameters
        ----------
        hcst_grid : dims (member_dim, time_dim, lat_dim, lon_dim)
        obs_grid  : dims (time_dim, lat_dim, lon_dim)
        """
        self._require_dims(hcst_grid, [member_dim, time_dim, lat_dim, lon_dim], "hcst_grid")
        self._require_dims(obs_grid,  [time_dim,   lat_dim,  lon_dim],           "obs_grid")
        hcst_grid, obs_grid = xr.align(hcst_grid, obs_grid, join="exact")
        if hcst_grid.sizes[member_dim] == 0 or not hcst_grid.get_index(member_dim).is_unique:
            raise ValueError("BMA requires nonempty unique member labels")
        if int(min_samples) < 2:
            raise ValueError("min_samples must be >= 2")
        self._obs_grid = obs_grid

        self._member_dim = member_dim
        self._time_dim   = time_dim
        self._lat_dim    = lat_dim
        self._lon_dim    = lon_dim
        # Keep the training observations so predict_probabilistic() can derive
        # tercile thresholds even when the caller omits obs_for_terciles.
        self._obs_grid = obs_grid

        hcst = hcst_grid.transpose(member_dim, time_dim, lat_dim, lon_dim).values
        obs  = obs_grid.transpose(time_dim, lat_dim, lon_dim).values

        n_memb, _, n_lat, n_lon = hcst.shape

        weights_map = np.full((n_memb, n_lat, n_lon), 1.0 / n_memb)
        fit_converged = np.zeros((n_lat, n_lon), dtype=bool)
        param_map_1 = np.full((n_lat, n_lon), np.nan)   # σ (normal) or shape (gamma*)
        # gamma0 is a pooled-occurrence/raw-gamma adaptation, not an exact
        # Sloughter et al. (2007) reproduction (which is component-specific
        # and models cube-root amounts).
        param_map_2 = np.full((3, n_lat, n_lon), np.nan)
        fixed_pop = np.full((n_lat, n_lon), np.nan)
        bias_a_map = np.zeros((n_memb, n_lat, n_lon), dtype=float)
        bias_b_map = np.ones((n_memb, n_lat, n_lon), dtype=float)

        coords = {
            member_dim: hcst_grid.coords[member_dim],
            lat_dim:    hcst_grid.coords[lat_dim],
            lon_dim:    hcst_grid.coords[lon_dim],
        }

        # FIX #2: explicit SLSQP inequality constraint ensuring the implicit
        # last weight w_M = 1 - Σ w_i remains ≥ 0.  Previously this was only
        # softly penalised inside the NLL, which the solver could silently violate.
        weight_sum_le_1 = {
            "type": "ineq",
            "fun": lambda p: 1.0 - np.sum(p[: n_memb - 1]),
        }

        print(f"Fitting {self.model_type} BMA on {n_lat}×{n_lon} grid …")

        for ilat in tqdm(range(n_lat), desc="Training"):
            for ilon in range(n_lon):
                f_raw = hcst[:, :, ilat, ilon]   # (M, T)
                o_raw = obs[:,    ilat, ilon]     # (T,)

                valid = np.isfinite(o_raw) & np.isfinite(f_raw).all(axis=0)
                if int(valid.sum()) < min_samples:
                    continue

                f_data = f_raw[:, valid].copy()   # (M, N)
                o_data = o_raw[valid]      # (N,)

                if self.model_type in {"gamma", "gamma0"} and np.any(o_data < 0):
                    continue  # Negative observations have no support in these distributions.
                if self.model_type == "gamma" and np.any(o_data == 0):
                    warnings.warn("Gamma BMA requires positive observations; use gamma0 for zeros", RuntimeWarning)
                    continue
                occurrence_forecasts = f_data.copy()

                # Raftery-style component bias correction, shrunk toward the
                # identity map to avoid unstable slopes on short hindcasts.
                if self.bias_correction and o_data.size >= self.min_bias_years:
                    for km in range(n_memb):
                        x = f_data[km]
                        good = np.isfinite(x) & np.isfinite(o_data)
                        if self.model_type == "gamma0":
                            good &= o_data > 0  # conditional positive amount, not unconditional rain
                        if good.sum() >= self.min_bias_years:
                            if np.nanstd(x[good]) > 0:
                                lm = linregress(x[good], o_data[good])
                                a = (1.0 - self.bias_shrinkage) * float(lm.intercept)
                                b = self.bias_shrinkage + (1.0 - self.bias_shrinkage) * float(lm.slope)
                            else:
                                a = (1.0 - self.bias_shrinkage) * float(np.mean(o_data[good] - x[good]))
                                b = 1.0
                            if np.isfinite(a) and np.isfinite(b):
                                bias_a_map[km, ilat, ilon] = a
                                bias_b_map[km, ilat, ilon] = b
                                f_data[km] = a + b * f_data[km]

                # ── Normal (Raftery et al. 2005) ─────────────────────────────
                if self.model_type == "normal":

                    def nll_normal(p, _f=f_data, _o=o_data):
                        # Optimize log(sigma), so units and near-zero residuals
                        # cannot produce a huge finite-difference step in sigma.
                        sigma = np.exp(p[-1])
                        all_ws = self._normalize_weights(np.append(p[:-1], 1.0-p[:-1].sum()))
                        z = (_o[None, :] - _f) / sigma
                        logpdf = norm.logpdf(z) - p[-1]
                        logmix = logsumexp(logpdf, b=all_ws[:, None], axis=0)
                        ratio = np.exp(np.clip(logpdf-logmix, -745.0, 700.0))
                        gradient_w = -np.sum(ratio[:-1]-ratio[-1], axis=1)
                        responsibilities = all_ws[:, None] * ratio
                        gradient_sigma = -np.sum(responsibilities*(z*z-1.0))
                        return -float(np.sum(logmix)), np.r_[gradient_w, gradient_sigma]

                    residual_scale = float(np.sqrt(np.mean((o_data-f_data.mean(axis=0))**2)))
                    x0 = np.r_[np.full(n_memb-1, 1.0/n_memb), np.log(max(residual_scale, 1e-3))]
                    bounds = [(0.0, 1.0)]*(n_memb-1)+[(np.log(1e-3), None)]
                    res = minimize(nll_normal, x0, jac=True, method="SLSQP", bounds=bounds,
                                   constraints=[weight_sum_le_1],
                                   options={"maxiter": 500, "ftol": self.tol})
                    feasible = (np.all(np.isfinite(res.x)) and np.isfinite(res.fun)
                                and res.x[:-1].sum() <= 1.0+1e-8)
                    if feasible and res.fun <= nll_normal(x0)[0]+1e-8:
                        weights_map[:, ilat, ilon] = self._normalize_weights(
                            np.r_[res.x[:-1], 1.0-res.x[:-1].sum()])
                        param_map_1[ilat, ilon] = float(np.exp(res.x[-1]))
                        fit_converged[ilat, ilon] = bool(res.success)
                    else:
                        # The feasible initial mixture is retained explicitly;
                        # numerical convergence diagnostics distinguish this fit.
                        param_map_1[ilat, ilon] = max(residual_scale, 1e-3)

                # ── Gamma / Gamma0 ───────────────────────────────────────────
                else:
                    if self.model_type == "gamma0":
                        # Pooled POP regression for the gamma0 adaptation.
                        # Three predictors: 1, ∛ens_mean, fraction of wet members.
                        y_bin    = (o_data > 0).astype(float)
                        ens_mean = occurrence_forecasts.mean(axis=0)
                        x_cbrt   = np.cbrt(np.maximum(ens_mean, 0.0))
                        # FIX #3: add fraction of wet members as second skill predictor
                        x_frac   = (occurrence_forecasts > 0).mean(axis=0).astype(float)
                        X_pop    = np.column_stack(
                            [np.ones(y_bin.size), x_cbrt, x_frac]
                        )

                        if np.all(y_bin == y_bin[0]):
                            fixed_pop[ilat, ilon] = y_bin[0]
                            param_map_2[:, ilat, ilon] = 0.0
                        else:
                            # Stable Bernoulli likelihood; weak slope penalty prevents
                            # unbounded coefficients under separation on short records.
                            def nll_logistic(beta):
                                eta = X_pop @ beta
                                loss = np.sum(np.logaddexp(0.0, eta) - y_bin * eta)
                                grad = X_pop.T @ (expit(eta) - y_bin)
                                return loss + 1e-4 * np.sum(beta[1:] ** 2), grad + np.r_[0., 2e-4 * beta[1:]]
                            initial = np.array([np.log(y_bin.mean() / (1-y_bin.mean())), 0., 0.])
                            res_log = minimize(nll_logistic, initial, jac=True, method="BFGS")
                            beta = res_log.x if np.all(np.isfinite(res_log.x)) and res_log.fun <= nll_logistic(initial)[0] else initial
                            param_map_2[:, ilat, ilon] = beta
                        if fixed_pop[ilat, ilon] == 0.0:
                            param_map_1[ilat, ilon] = 1.0  # amount unused for exact dry point mass
                            fit_converged[ilat, ilon] = True
                            continue

                        mask_pos = (
                            (o_data > 0)
                            & np.isfinite(o_data)
                            & np.isfinite(f_data).all(axis=0)
                        )
                        if int(mask_pos.sum()) < max(5, min_samples // 2):
                            continue
                        o_gamma = o_data[mask_pos]
                        f_gamma = f_data[:, mask_pos]

                    else:  # gamma
                        mask_pos = (
                            (o_data > 0)
                            & np.isfinite(o_data)
                            & np.isfinite(f_data).all(axis=0)
                        )
                        if int(mask_pos.sum()) < min_samples:
                            continue
                        o_gamma = o_data[mask_pos]
                        f_gamma = f_data[:, mask_pos]

                    # Γ(shape, scale = f/shape) ⟹ E[Y|k] = f_k (component mean = forecast)
                    f_gamma = np.maximum(f_gamma, 1e-3)

                    def nll_gamma(p, _f=f_gamma, _o=o_gamma):
                        ws        = p[:-1]
                        shape_val = p[-1]
                        w_last    = 1.0 - ws.sum()
                        if w_last < 0 or shape_val < 0.1 or not np.isfinite(shape_val):
                            return 1e12
                        all_ws = np.append(ws, w_last)
                        logpdfs = sp_gamma.logpdf(_o[None, :], a=shape_val, scale=_f / shape_val)
                        return -float(np.sum(logsumexp(logpdfs, b=all_ws[:, None], axis=0)))

                    x0     = np.append(np.full(n_memb - 1, 1.0 / n_memb), 2.0)
                    bounds = [(0.0, 1.0)] * (n_memb - 1) + [(0.1, 50.0)]

                    try:
                        res = minimize(
                            nll_gamma, x0,
                            method="SLSQP",
                            bounds=bounds,
                            constraints=[weight_sum_le_1],
                            tol=self.tol,
                        )
                        if (np.all(np.isfinite(res.x)) and np.isfinite(res.fun)
                                and res.fun <= nll_gamma(x0) + 1e-8):
                            ws = self._normalize_weights(
                                np.append(res.x[:-1], 1.0 - res.x[:-1].sum())
                            )
                            weights_map[:, ilat, ilon] = ws
                            param_map_1[ilat, ilon]    = float(res.x[-1])
                            fit_converged[ilat, ilon] = bool(res.success)
                        else:
                            param_map_1[ilat, ilon] = 2.0
                    except (ValueError, FloatingPointError):
                        param_map_1[ilat, ilon] = 2.0

        # ── Store parameters as xarray ────────────────────────────────────────
        ll_coords = {lat_dim: coords[lat_dim], lon_dim: coords[lon_dim]}
        self.weights = xr.DataArray(
            weights_map, dims=(member_dim, lat_dim, lon_dim), coords=coords
        )
        self.bias_intercept = xr.DataArray(
            bias_a_map, dims=(member_dim, lat_dim, lon_dim), coords=coords
        )
        self.bias_slope = xr.DataArray(
            bias_b_map, dims=(member_dim, lat_dim, lon_dim), coords=coords
        )

        if self.model_type == "normal":
            self.sigma = xr.DataArray(
                param_map_1, dims=(lat_dim, lon_dim), coords=ll_coords
            )
        else:
            self.shape = xr.DataArray(
                param_map_1, dims=(lat_dim, lon_dim), coords=ll_coords
            )
            if self.model_type == "gamma0":
                # FIX #3 (cont.): coord labels updated to match 3-predictor model
                self.logistic_params = xr.DataArray(
                    param_map_2,
                    dims=("param", lat_dim, lon_dim),
                    coords={
                        "param": ["b0", "b1_cbrt_mean", "b2_frac_wet"],
                        **ll_coords,
                    },
                )

        self.fixed_wet_probability_ = xr.DataArray(fixed_pop, dims=(lat_dim, lon_dim), coords=ll_coords)
        self.fit_converged_ = xr.DataArray(fit_converged, dims=(lat_dim, lon_dim), coords=ll_coords)
        self.fitted = True
        return self

    # ────────────────────────────────────────────────────────── predict ──────

    def predict_probabilistic(
        self,
        new_forecasts:    xr.DataArray,
        *,
        clim_terciles:    Optional[xr.DataArray] = None,
        obs_for_terciles: Optional[xr.DataArray] = None,
        tercile_q:        Tuple[float, float]    = (1 / 3, 2 / 3),
        quantiles:        Sequence[float]         = (0.1, 0.5, 0.9),
        member_dim:       str = "M",
        time_dim:         str = "T",
        lat_dim:          str = "Y",
        lon_dim:          str = "X",
    ) -> xr.Dataset:
        """
        Generate probabilistic forecasts from the fitted BMA mixture.

        Parameters
        ----------
        new_forecasts    : dims (member_dim, time_dim, lat_dim, lon_dim)
        clim_terciles    : dims ('tercile', lat_dim, lon_dim)  [optional]
        obs_for_terciles : dims (time_dim, lat_dim, lon_dim)   [optional]

        Returns
        -------
        xr.Dataset:
          ``predictive_mean``       — (T, Y, X)
          ``predictive_quantiles``  — ('quantile', T, Y, X)
          ``tercile_probability``   — ('probability', T, Y, X)  [if terciles given]
          ``tercile_thresholds``    — ('tercile', Y, X)          [if terciles given]
        """
        if not self.fitted:
            raise ValueError("Call .fit() before .predict_probabilistic().")

        self._require_dims(
            new_forecasts, [member_dim, time_dim, lat_dim, lon_dim], "new_forecasts"
        )
        new_forecasts = _bma_forecast_members(new_forecasts, self.weights[member_dim].values, member_dim)
        if any(not np.isfinite(q) or not 0 < q < 1 for q in quantiles):
            raise ValueError("BMA predictive quantiles must lie strictly inside (0, 1)")
        w_ds, p1_ds, p2_ds = self._align_params_to_forecasts(
            new_forecasts,
            member_dim=member_dim, time_dim=time_dim,
            lat_dim=lat_dim,       lon_dim=lon_dim,
        )

        terciles_da = _bma_terciles(self, new_forecasts, clim_terciles, obs_for_terciles,
                                   tercile_q, time_dim, lat_dim, lon_dim)

        terciles_np = (
            terciles_da.transpose("tercile", lat_dim, lon_dim).values
            if terciles_da is not None else None
        )   # shape (2, n_lat, n_lon) or None

        # ── NumPy extraction ──────────────────────────────────────────────────
        raw_forecasts = new_forecasts.transpose(member_dim, time_dim, lat_dim, lon_dim).values
        fixed_pop = _bma_spatial_select(self.fixed_wet_probability_, new_forecasts, lat_dim, lon_dim).values
        if self.bias_correction and self.bias_intercept is not None:
            sel_bc = {lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]}
            ba = _bma_spatial_select(self.bias_intercept, new_forecasts, lat_dim, lon_dim)
            bb = _bma_spatial_select(self.bias_slope, new_forecasts, lat_dim, lon_dim)
            new_forecasts = ba + bb * new_forecasts
        fcst = new_forecasts.transpose(member_dim, time_dim, lat_dim, lon_dim).values
        ws   = w_ds.transpose(member_dim, lat_dim, lon_dim).values
        p1   = p1_ds.transpose(lat_dim, lon_dim).values
        p2   = (
            p2_ds.transpose("param", lat_dim, lon_dim).values
            if p2_ds is not None else None
        )

        n_memb, n_time, n_lat, n_lon = fcst.shape
        n_quant = len(quantiles)

        out_mean  = np.full((n_time, n_lat, n_lon), np.nan)
        out_quant = np.full((n_quant, n_time, n_lat, n_lon), np.nan)
        out_probs = (
            np.full((3, n_time, n_lat, n_lon), np.nan)
            if terciles_np is not None else None
        )

        print("Predicting …")

        for ilat in tqdm(range(n_lat), desc="Predict"):
            for ilon in range(n_lon):
                param_val = p1[ilat, ilon]
                if not np.isfinite(param_val):
                    continue

                w_loc  = self._normalize_weights(ws[:, ilat, ilon])
                low_th = up_th = np.nan
                if terciles_np is not None:
                    low_th, up_th = terciles_np[:, ilat, ilon]

                # FIX #4: unpack all three POP logistic parameters
                if self.model_type == "gamma0":
                    b0, b1, b2 = p2[:, ilat, ilon]
                    if not (np.isfinite(b0) and np.isfinite(b1) and np.isfinite(b2)):
                        continue

                for t in range(n_time):
                    f_m = fcst[:, t, ilat, ilon]
                    if not np.isfinite(f_m).all():
                        continue

                    do_probs = (
                        out_probs is not None
                        and np.isfinite(low_th)
                        and np.isfinite(up_th)
                    )

                    # ── Normal ───────────────────────────────────────────────
                    if self.model_type == "normal":
                        sigma  = param_val
                        mu_mix = float(np.dot(w_loc, f_m))
                        out_mean[t, ilat, ilon] = mu_mix

                        # FIX #10: capture loop vars via default args to avoid
                        #          accidental late-binding in closures
                        def cdf_mix(x, _w=w_loc, _f=f_m, _s=sigma):
                            return float(np.dot(_w, norm.cdf(x, loc=_f, scale=_s)))

                        for iq, qv in enumerate(quantiles):
                            out_quant[iq, t, ilat, ilon] = _bma_quantile(
                                cdf_mix, qv, norm.ppf(qv, loc=f_m, scale=sigma))

                        if do_probs:
                            pb = cdf_mix(float(low_th))
                            pa = 1.0 - cdf_mix(float(up_th))
                            # FIX #9: normalise all three probabilities
                            out_probs[:, t, ilat, ilon] = _normalize_tercile_probs(
                                pb, 1.0 - pb - pa, pa
                            )

                    # ── Gamma0 (precipitation, Sloughter et al. 2007) ────────
                    elif self.model_type == "gamma0":
                        shape  = param_val
                        f_safe = np.maximum(f_m, 1e-3)
                        scales = f_safe / shape   # Γ(shape, scale) ⟹ mean = f_m

                        # POP: logistic(b0 + b1·∛ens_mean + b2·frac_wet)
                        f_raw = raw_forecasts[:, t, ilat, ilon]
                        f_mean_cbrt = float(np.cbrt(max(float(np.mean(f_raw)), 0.0)))
                        frac_wet = float(np.mean(f_raw > 0))
                        pop = fixed_pop[ilat, ilon]
                        if not np.isfinite(pop):
                            pop = float(expit(b0 + b1 * f_mean_cbrt + b2 * frac_wet))

                        # E[Y] = POP · Σ_k w_k · f_k
                        out_mean[t, ilat, ilon] = pop * float(np.dot(w_loc, f_safe))

                        def cdf_gamma_cond(x, _w=w_loc, _sh=shape, _sc=scales):
                            """CDF of the conditional Gamma part P(Y ≤ x | Y > 0)."""
                            return float(np.dot(_w, sp_gamma.cdf(x, a=_sh, scale=_sc)))

                        def cdf_full(x, _pop=pop):
                            """
                            Full zero-inflated CDF:
                            F(x) = (1−pop) + pop·F_Γ(x) for x ≥ 0, 0 for x < 0.
                            The point mass (1−pop) sits at x = 0.
                            """
                            if x < 0.0:
                                return 0.0
                            return (1.0 - _pop) + _pop * cdf_gamma_cond(x)

                        for iq, qv in enumerate(quantiles):
                            if qv <= 1.0 - pop:
                                # Quantile falls within the point mass at zero
                                out_quant[iq, t, ilat, ilon] = 0.0
                            else:
                                # Invert the conditional Gamma CDF
                                target = (qv - (1.0 - pop)) / max(pop, 1e-12)
                                out_quant[iq, t, ilat, ilon] = _bma_quantile(
                                    cdf_gamma_cond, target, sp_gamma.ppf(target, a=shape, scale=scales))

                        if do_probs:
                            pb = cdf_full(float(low_th))
                            pa = 1.0 - cdf_full(float(up_th))
                            out_probs[:, t, ilat, ilon] = _normalize_tercile_probs(
                                pb, 1.0 - pb - pa, pa
                            )

                    # ── Gamma ─────────────────────────────────────────────────
                    else:
                        shape  = param_val
                        f_safe = np.maximum(f_m, 1e-3)
                        scales = f_safe / shape

                        out_mean[t, ilat, ilon] = float(np.dot(w_loc, f_safe))

                        def cdf_mix_g(x, _w=w_loc, _sh=shape, _sc=scales):
                            return float(np.dot(_w, sp_gamma.cdf(x, a=_sh, scale=_sc)))

                        for iq, qv in enumerate(quantiles):
                            out_quant[iq, t, ilat, ilon] = _bma_quantile(
                                cdf_mix_g, qv, sp_gamma.ppf(qv, a=shape, scale=scales))

                        if do_probs:
                            pb = cdf_mix_g(float(low_th))
                            pa = 1.0 - cdf_mix_g(float(up_th))
                            out_probs[:, t, ilat, ilon] = _normalize_tercile_probs(
                                pb, 1.0 - pb - pa, pa
                            )

        # ── Package output ────────────────────────────────────────────────────
        base_coords = {
            time_dim: new_forecasts[time_dim],
            lat_dim:  new_forecasts[lat_dim],
            lon_dim:  new_forecasts[lon_dim],
        }
        ds = xr.Dataset()
        ds["predictive_mean"] = xr.DataArray(
            out_mean, dims=(time_dim, lat_dim, lon_dim), coords=base_coords
        )
        ds["predictive_quantiles"] = xr.DataArray(
            out_quant,
            dims=("quantile", time_dim, lat_dim, lon_dim),
            coords={**base_coords, "quantile": list(quantiles)},
        )
        if out_probs is not None:
            ds["tercile_probability"] = xr.DataArray(
                out_probs,
                dims=("probability", time_dim, lat_dim, lon_dim),
                coords={**base_coords, "probability": ["PB", "PN", "PA"]},
            )
            ds["tercile_thresholds"] = terciles_da
        return _bma_output_metadata(ds)


# ─────────────────────────────────────────────────────────────────────────────
class WAS_mme_FullBMA:
    """
    Hybrid Bayesian seasonal ensemble postprocessor — per-gridpoint fitting.

    Distribution families (per gridpoint, supplied via ``dist_map``)
    -----------------------------------------------------------------
    Code 1 → ``'normal'``     Gaussian BMA with bias-correction
    Code 4 → ``'gamma'``      Gamma BMA (positive-definite variables)
    Code 2 → ``'lognormal'``  Lognormal BMA (heavy-tailed precipitation, etc.)

    Each component distribution has an intercept ``a_k`` and positive slope
    ``b_k``. Normal/gamma means are linear in the member forecast; lognormal
    log-location is linear in log-forecast (its arithmetic mean also includes
    half the log-variance). Both backends enforce the positive-slope constraint.
    Fast penalized likelihood and full posterior sampling are distinct estimators;
    the fast penalties are not presented as the exact full-backend priors.

    Fitting modes
    -------------
    ``'fast'`` — Penalised L-BFGS-B optimisation (always available).
    ``'full'`` — Full MCMC via PyMC (requires ``pymc`` + ``arviz``).
    ``'auto'`` — Tries MCMC first, falls back to L-BFGS-B on failure.

    Notes
    -----
    - Gridpoints where ``dist_map`` is NaN are skipped.
    - Gamma and lognormal require strictly positive observations. For rainfall
      with dry years use FastBMA(model_type="gamma0"); silently dropping zeros
      would estimate a conditional amount distribution, not the full forecast.
    - ``predict_probabilistic`` spatially aligns stored parameter maps to the
      forecast grid via nearest-neighbour selection, so training and test grids
      need not be identical.
    """

    DIST_CODE_TO_NAME = {1: "normal", 4: "gamma", 2: "lognormal"}
    DIST_NAME_TO_CODE = {"normal": 1, "gamma": 4, "lognormal": 2}
    MODE_CODE         = {"full": 0, "fast": 1, "failed": -1}

    def __init__(
        self,
        mode:           str   = "fast",
        eps:            float = 1e-6,
        tol:            float = 1e-5,
        maxiter:        int   = 300,
        draws:          int   = 1000,
        tune:           int   = 1000,
        chains:         int   = 2,
        target_accept:  float = 0.92,
        random_seed:    int   = 42,
        progressbar:    bool  = True,
        verbose:        bool  = True,
    ) -> None:
        if mode not in {"full", "fast", "auto"}:
            raise ValueError("mode must be one of {'full', 'fast', 'auto'}")

        if not np.isfinite(eps) or eps <= 0 or tol <= 0 or maxiter < 1:
            raise ValueError("eps, tol and maxiter must be positive")
        self.mode          = mode
        self.eps           = eps
        self.tol           = tol
        self.maxiter       = maxiter
        self.draws         = draws
        self.tune          = tune
        self.chains        = chains
        self.target_accept = target_accept
        self.random_seed   = random_seed
        self.progressbar   = progressbar
        self.verbose       = verbose
        self.fitted        = False

        self.weight_map:      Optional[xr.DataArray] = None
        self.intercept_map:   Optional[xr.DataArray] = None
        self.slope_map:       Optional[xr.DataArray] = None
        self.dispersion_map:  Optional[xr.DataArray] = None
        self.family_map_used: Optional[xr.DataArray] = None
        self.fit_mode_map:    Optional[xr.DataArray] = None
        self.fit_status_map:  Optional[xr.DataArray] = None

        # FIX #7: keyed by (lat_coord_value, lon_coord_value) instead of
        #         (training_ilat, training_ilon) so that nearest-neighbour
        #         alignment in predict_probabilistic works correctly when the
        #         forecast grid differs from the training grid.
        self.posterior_: Dict[Tuple[float, float], Dict[str, np.ndarray]] = {}

        self._member_dim:    Optional[str]        = None
        self._time_dim:      Optional[str]        = None
        self._lat_dim:       Optional[str]        = None
        self._lon_dim:       Optional[str]        = None
        self._member_values: Optional[np.ndarray] = None
        self._lat_values:    Optional[np.ndarray] = None
        self._lon_values:    Optional[np.ndarray] = None

    # ──────────────────────────────────────────────────────── utilities ──────

    @staticmethod
    def _require_dims(da: xr.DataArray, dims: Sequence[str], name: str) -> None:
        missing = [d for d in dims if d not in da.dims]
        if missing:
            raise ValueError(
                f"{name} missing required dims {missing}. Got dims={da.dims}"
            )

    @staticmethod
    def _safe_scale(x: np.ndarray, default: float = 1.0) -> float:
        s = np.nanstd(x)
        if not np.isfinite(s) or s <= 0:
            s = np.nanmean(np.abs(x))
        if not np.isfinite(s) or s <= 0:
            s = default
        return float(s)

    @staticmethod
    def _flatten_trace(trace: "az.InferenceData", varname: str) -> np.ndarray:
        arr = trace.posterior[varname].values
        return arr.reshape(arr.shape[0] * arr.shape[1], *arr.shape[2:])

    def _parse_family(self, value: Union[str, int, float, None]) -> Optional[str]:
        if value is None:
            return None
        try:
            if np.isnan(value):
                return None
        except TypeError:
            pass
        if isinstance(value, str):
            fam = value.strip().lower()
            return fam if fam in self.DIST_NAME_TO_CODE else None
        try:
            code = int(value)
        except Exception:
            return None
        return self.DIST_CODE_TO_NAME.get(code, None)

    @staticmethod
    def compute_gridpoint_terciles_from_obs(
        obs: xr.DataArray,
        *,
        time_dim: str = "T",
        lat_dim:  str = "Y",
        lon_dim:  str = "X",
        q: Tuple[float, float] = (1 / 3, 2 / 3),
    ) -> xr.DataArray:
        # FIX: consistent with FastBMA — use this class's own _require_dims
        WAS_mme_FullBMA._require_dims(obs, [time_dim, lat_dim, lon_dim], "obs")
        qt = obs.quantile(list(q), dim=time_dim, skipna=True).rename(
            {"quantile": "tercile"}
        )
        return qt.assign_coords(tercile=["lower", "upper"])

    def _align_terciles_to_forecast_grid(
        self,
        new_forecasts:    xr.DataArray,
        clim_terciles:    Optional[xr.DataArray],
        obs_for_terciles: Optional[xr.DataArray],
        tercile_q:        Tuple[float, float],
        lat_dim:          str,
        lon_dim:          str,
        time_dim:         str,
    ) -> Optional[xr.DataArray]:
        return _bma_terciles(self, new_forecasts, clim_terciles, obs_for_terciles,
                             tercile_q, time_dim, lat_dim, lon_dim)

    # ─────────────────────────────────────────── FAST backend ───────────────

    @staticmethod
    def _unpack_fast_params(
        p: np.ndarray, m: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Unpack the flat parameter vector used by L-BFGS-B:
          p[:m]   — weight logits  (softmax → weights)
          p[m:2m] — intercepts a_k
          p[2m:3m]— log-slopes     (exp → b_k > 0)
          p[3m]   — log-dispersion (exp → σ / shape / σ_log)
        """
        logits_w = p[:m]
        a        = p[m:2 * m]
        b        = np.exp(p[2 * m:3 * m])
        disp     = np.exp(p[3 * m])
        w        = softmax(logits_w)
        return w, a, b, disp

    def _nll_fast_normal(self, p, f_data, y_data, lam_w, lam_a, lam_b, lam_disp, y_scale):
        m, _ = f_data.shape
        w, a, b, sigma = self._unpack_fast_params(p, m)
        mu      = a[:, None] + b[:, None] * f_data
        logpdf = norm.logpdf(y_data[None, :], loc=mu, scale=max(sigma, self.eps))
        nll = -float(np.sum(logsumexp(logpdf, b=w[:, None], axis=0)))
        pen = (
            lam_w    * np.sum((w - 1.0 / m) ** 2)
            + lam_a  * np.sum((a / max(y_scale, self.eps)) ** 2)
            + lam_b  * np.sum(np.log(b) ** 2)           # penalise slopes far from 1
            + lam_disp * (np.log(sigma / max(y_scale, self.eps)) ** 2)
        )
        return nll + pen

    def _nll_fast_gamma(self, p, f_data, y_data, lam_w, lam_a, lam_b, lam_disp, y_scale):
        m, _ = f_data.shape
        w, a, b, shape = self._unpack_fast_params(p, m)
        if np.any(y_data <= 0):
            return 1e20
        mu      = np.maximum(a[:, None] + b[:, None] * f_data, self.eps)
        # Γ(shape, rate=shape/mu) ⟹ mean = mu
        rate    = shape / mu
        logpdf = sp_gamma.logpdf(y_data[None, :], a=shape, scale=mu / shape)
        nll = -float(np.sum(logsumexp(logpdf, b=w[:, None], axis=0)))
        pen = (
            lam_w    * np.sum((w - 1.0 / m) ** 2)
            + lam_a  * np.sum((a / max(y_scale, self.eps)) ** 2)
            + lam_b  * np.sum(np.log(b) ** 2)
            + lam_disp * (np.log(shape) ** 2)
        )
        return nll + pen

    def _nll_fast_lognormal(
        self, p, f_data, y_data, lam_w, lam_a, lam_b, lam_disp, y_scale
    ):
        m, _ = f_data.shape
        w, a, b, sigma_log = self._unpack_fast_params(p, m)
        if np.any(y_data <= 0):
            return 1e20
        # Component: LogNormal(μ_log = a + b·log f, σ_log)
        mu_log  = a[:, None] + b[:, None] * np.log(np.maximum(f_data, self.eps))
        logpdf = norm.logpdf(np.log(y_data)[None, :], loc=mu_log,
                              scale=max(sigma_log, self.eps)) - np.log(y_data)[None, :]
        nll = -float(np.sum(logsumexp(logpdf, b=w[:, None], axis=0)))
        pen = (
            lam_w    * np.sum((w - 1.0 / m) ** 2)
            + lam_a  * np.sum(a ** 2)
            + lam_b  * np.sum(np.log(b) ** 2)
            + lam_disp * (np.log(sigma_log) ** 2)
        )
        return nll + pen

    def _fit_single_grid_fast(
        self,
        f_data: np.ndarray,
        y_data: np.ndarray,
        family: str,
        lam_w:    float,
        lam_a:    float,
        lam_b:    float,
        lam_disp: float,
    ) -> Dict[str, Any]:
        m, _ = f_data.shape
        y_scale = self._safe_scale(y_data, default=1.0)

        if family in {"gamma", "lognormal"}:
            if np.any(y_data <= 0):
                raise ValueError(f"{family} BMA requires strictly positive observations; use FastBMA gamma0 for zeros")
            if y_data.size < 5:
                raise ValueError(f"Insufficient positive samples for '{family}'")

        # FIX #8: for lognormal the dispersion parameter is σ_log (std of log Y),
        #         not the raw-scale std.  Initialise from std(log y) instead.
        if family == "lognormal":
            log_std   = float(np.std(np.log(np.maximum(y_data, self.eps))))
            disp_init = np.log(max(log_std, 0.05))
        elif family == "gamma":
            disp_init = np.log(2.0)  # dimensionless shape, not raw-scale standard deviation
        else:
            disp_init = np.log(max(y_scale, self.eps))

        p0 = np.concatenate([
            np.zeros(m),      # weight logits  → uniform weights after softmax
            np.zeros(m),      # intercepts     → zero bias
            np.zeros(m),      # log-slopes     → slope = 1 (unbiased)
            [disp_init],
        ])

        if family == "normal":
            obj = lambda p: self._nll_fast_normal(
                p, f_data, y_data, lam_w, lam_a, lam_b, lam_disp, y_scale
            )
        elif family == "gamma":
            obj = lambda p: self._nll_fast_gamma(
                p, f_data, y_data, lam_w, lam_a, lam_b, lam_disp, y_scale
            )
        elif family == "lognormal":
            obj = lambda p: self._nll_fast_lognormal(
                p, f_data, y_data, lam_w, lam_a, lam_b, lam_disp, y_scale
            )
        else:
            raise ValueError(f"Unsupported family '{family}'")

        res = minimize(
            obj, p0,
            method="L-BFGS-B",
            bounds=[(-30., 30.)] * m + [(None, None)] * m + [(-12., 12.)] * m
                   + [(np.log(self.eps), 40.)],
            options={"maxiter": self.maxiter, "ftol": self.tol},
        )
        if not np.all(np.isfinite(res.x)) or not np.isfinite(res.fun) or res.fun > obj(p0) + 1e-8:
            raise RuntimeError(f"Invalid BMA optimization: {res.message}")

        w, a, b, disp = self._unpack_fast_params(res.x, m)
        return {"weights": w, "a": a, "b": b, "dispersion": float(disp),
                "family": family, "backend": "fast", "converged": bool(res.success)}

    # ─────────────────────────────────────────── FULL backend ───────────────

    def _fit_single_grid_full(
        self,
        f_data: np.ndarray,
        y_data: np.ndarray,
        family: str,
        alpha0:            float,
        coef_scale_mult:   float,
        sigma_scale_mult:  float,
        shape_scale:       float,
    ) -> Dict[str, Any]:
        if not HAS_PYMC:
            raise RuntimeError("PyMC / ArviZ not available")

        m, _ = f_data.shape
        y_scale = self._safe_scale(y_data, default=1.0)
        if family in {"gamma", "lognormal"} and np.any(y_data <= 0):
            raise ValueError(f"{family} BMA requires strictly positive observations; use FastBMA gamma0 for zeros")

        sample_kw = dict(
            draws=self.draws, tune=self.tune, chains=self.chains,
            target_accept=self.target_accept, random_seed=self.random_seed,
            progressbar=self.progressbar, compute_convergence_checks=True,
            return_inferencedata=True,
        )

        if family == "normal":
            with pm.Model():
                weights = pm.Dirichlet("weights", a=np.full(m, alpha0))
                a       = pm.Normal("a", mu=0.0, sigma=coef_scale_mult * y_scale, shape=m)
                b       = pm.LogNormal("b", mu=0.0, sigma=coef_scale_mult, shape=m)
                sigma   = pm.HalfNormal("sigma", sigma=sigma_scale_mult * y_scale)
                comp    = [pm.Normal.dist(mu=a[k] + b[k] * f_data[k, :], sigma=sigma)
                           for k in range(m)]
                pm.Mixture("y_like", w=weights, comp_dists=comp, observed=y_data)
                trace = pm.sample(**sample_kw)
            return {
                "weights":    self._flatten_trace(trace, "weights"),
                "a":          self._flatten_trace(trace, "a"),
                "b":          self._flatten_trace(trace, "b"),
                "dispersion": self._flatten_trace(trace, "sigma"),
                "family": family, "backend": "full",
            }

        elif family == "gamma":
            valid  = y_data > 0
            y_pos  = y_data[valid];  f_pos = f_data[:, valid]
            if y_pos.size < 5:
                raise ValueError("Insufficient positive samples for gamma")
            with pm.Model():
                weights = pm.Dirichlet("weights", a=np.full(m, alpha0))
                a       = pm.Normal("a", mu=0.0, sigma=coef_scale_mult * y_scale, shape=m)
                b       = pm.LogNormal("b", mu=0.0, sigma=coef_scale_mult, shape=m)
                shape   = pm.HalfNormal("shape", sigma=shape_scale)
                comp = []
                for k in range(m):
                    mu_k   = pm.math.maximum(a[k] + b[k] * f_pos[k, :], self.eps)
                    beta_k = shape / mu_k   # rate = shape/mean  ⟹  E[Y] = shape/rate = mu_k
                    comp.append(pm.Gamma.dist(alpha=shape, beta=beta_k))
                pm.Mixture("y_like", w=weights, comp_dists=comp, observed=y_pos)
                trace = pm.sample(**sample_kw)
            return {
                "weights":    self._flatten_trace(trace, "weights"),
                "a":          self._flatten_trace(trace, "a"),
                "b":          self._flatten_trace(trace, "b"),
                "dispersion": self._flatten_trace(trace, "shape"),
                "family": family, "backend": "full",
            }

        elif family == "lognormal":
            valid  = y_data > 0
            y_pos  = y_data[valid];  f_pos = f_data[:, valid]
            if y_pos.size < 5:
                raise ValueError("Insufficient positive samples for lognormal")
            logy       = np.log(np.maximum(y_pos, self.eps))
            log_scale  = self._safe_scale(logy, default=1.0)
            with pm.Model():
                weights   = pm.Dirichlet("weights", a=np.full(m, alpha0))
                a         = pm.Normal("a", mu=0.0, sigma=coef_scale_mult * log_scale, shape=m)
                b         = pm.LogNormal("b", mu=0.0, sigma=coef_scale_mult, shape=m)
                sigma_log = pm.HalfNormal("sigma_log", sigma=sigma_scale_mult * log_scale)
                comp = []
                for k in range(m):
                    # μ_log,k = a_k + b_k · log(f_k)  ⟹  E[Y|k] = exp(μ_log + σ²/2)
                    mu_log_k = a[k] + b[k] * pm.math.log(
                        pm.math.maximum(f_pos[k, :], self.eps)
                    )
                    comp.append(pm.LogNormal.dist(mu=mu_log_k, sigma=sigma_log))
                pm.Mixture("y_like", w=weights, comp_dists=comp, observed=y_pos)
                trace = pm.sample(**sample_kw)
            return {
                "weights":    self._flatten_trace(trace, "weights"),
                "a":          self._flatten_trace(trace, "a"),
                "b":          self._flatten_trace(trace, "b"),
                "dispersion": self._flatten_trace(trace, "sigma_log"),
                "family": family, "backend": "full",
            }

        raise ValueError(f"Unsupported family '{family}'")

    # ─────────────────────────────────── hybrid single-gridpoint fit ────────

    def _fit_single_grid_hybrid(
        self,
        f_data: np.ndarray,
        y_data: np.ndarray,
        family: str,
        lam_w:           float,
        lam_a:           float,
        lam_b:           float,
        lam_disp:        float,
        alpha0:          float,
        coef_scale_mult: float,
        sigma_scale_mult: float,
        shape_scale:     float,
    ) -> Dict[str, Any]:
        if self.mode == "fast":
            return self._fit_single_grid_fast(f_data, y_data, family, lam_w, lam_a, lam_b, lam_disp)
        if self.mode == "full":
            return self._fit_single_grid_full(f_data, y_data, family, alpha0, coef_scale_mult,
                                               sigma_scale_mult, shape_scale)
        try:
            return self._fit_single_grid_full(f_data, y_data, family, alpha0, coef_scale_mult,
                                               sigma_scale_mult, shape_scale)
        except Exception:
            return self._fit_single_grid_fast(f_data, y_data, family, lam_w, lam_a, lam_b, lam_disp)

    # ──────────────────────────────────────────────────────────── fit ────────

    def fit(
        self,
        hcst_grid: xr.DataArray,
        obs_grid:  xr.DataArray,
        dist_map:  xr.DataArray,
        *,
        member_dim:       str   = "M",
        time_dim:         str   = "T",
        lat_dim:          str   = "Y",
        lon_dim:          str   = "X",
        min_samples:      int   = 12,
        lam_w:            float = 1.0,
        lam_a:            float = 0.1,
        lam_b:            float = 0.1,
        lam_disp:         float = 0.1,
        alpha0:           float = 1.0,
        coef_scale_mult:  float = 1.0,
        sigma_scale_mult: float = 1.0,
        shape_scale:      float = 5.0,
    ) -> "WAS_mme_FullBMA":
        if self.mode == "full" and not HAS_PYMC:
            raise RuntimeError("FullBMA mode='full' requires PyMC and ArviZ; use mode='fast' or 'auto'")
        self._require_dims(hcst_grid, [member_dim, time_dim, lat_dim, lon_dim], "hcst_grid")
        self._require_dims(obs_grid,  [time_dim,   lat_dim,  lon_dim],           "obs_grid")
        hcst_grid, obs_grid = xr.align(hcst_grid, obs_grid, join="exact")
        if hcst_grid.sizes[member_dim] == 0 or not hcst_grid.get_index(member_dim).is_unique:
            raise ValueError("BMA requires nonempty unique member labels")
        if int(min_samples) < 2:
            raise ValueError("min_samples must be >= 2")
        self._obs_grid = obs_grid
        self._require_dims(dist_map,  [lat_dim,    lon_dim],                     "dist_map")
        if min(lam_w, lam_a, lam_b, lam_disp) < 0:
            raise ValueError("BMA regularization penalties must be nonnegative")
        if min(alpha0, coef_scale_mult, sigma_scale_mult, shape_scale) <= 0:
            raise ValueError("BMA prior scales and concentrations must be positive")
        hcst_grid, obs_grid, dist_map = xr.align(hcst_grid, obs_grid, dist_map, join="exact")

        self._member_dim    = member_dim
        self._time_dim      = time_dim
        self._lat_dim       = lat_dim
        self._lon_dim       = lon_dim
        self._member_values = hcst_grid[member_dim].values
        self._lat_values    = hcst_grid[lat_dim].values
        self._lon_values    = hcst_grid[lon_dim].values

        hcst    = hcst_grid.transpose(member_dim, time_dim, lat_dim, lon_dim).values
        obs     = obs_grid.transpose(time_dim, lat_dim, lon_dim).values
        dist_np = dist_map.transpose(lat_dim, lon_dim).values

        n_memb, _, n_lat, n_lon = hcst.shape

        weight_map    = np.full((n_memb, n_lat, n_lon), np.nan)
        intercept_map = np.full((n_memb, n_lat, n_lon), np.nan)
        slope_map     = np.full((n_memb, n_lat, n_lon), np.nan)
        dispersion_map = np.full((n_lat, n_lon), np.nan)
        family_code   = np.full((n_lat, n_lon), np.nan)
        fit_mode      = np.full((n_lat, n_lon), self.MODE_CODE["failed"])
        fit_status    = np.zeros((n_lat, n_lon))
        converged = np.zeros((n_lat, n_lon), dtype=bool)

        self.posterior_.clear()

        if self.verbose:
            print(f"Fitting hybrid Bayesian postprocessor [mode={self.mode}] …")

        for ilat in tqdm(range(n_lat), desc="Training", disable=not self.verbose):
            for ilon in range(n_lon):
                fam = self._parse_family(dist_np[ilat, ilon])
                if fam is None:
                    continue

                f_raw = hcst[:, :, ilat, ilon]
                y_raw = obs[:,    ilat, ilon]
                valid = np.isfinite(y_raw) & np.isfinite(f_raw).all(axis=0)
                if int(valid.sum()) < min_samples:
                    continue
                if fam in {"gamma", "lognormal"} and np.any(y_raw[valid] <= 0):
                    warnings.warn(f"{fam} BMA requires strictly positive observations; use FastBMA gamma0 for zeros", RuntimeWarning)
                    continue

                try:
                    res = self._fit_single_grid_hybrid(
                        f_data=f_raw[:, valid], y_data=y_raw[valid], family=fam,
                        lam_w=lam_w, lam_a=lam_a, lam_b=lam_b, lam_disp=lam_disp,
                        alpha0=alpha0, coef_scale_mult=coef_scale_mult,
                        sigma_scale_mult=sigma_scale_mult, shape_scale=shape_scale,
                    )
                except Exception as exc:
                    if self.verbose:
                        print(f"  Skipping ({ilat},{ilon}) [{fam}]: {exc}")
                    continue

                family_code[ilat, ilon]   = self.DIST_NAME_TO_CODE[fam]
                fit_status[ilat, ilon]    = 1.0
                converged[ilat, ilon] = bool(res.get("converged", False))

                if res["backend"] == "fast":
                    fit_mode[ilat, ilon]       = self.MODE_CODE["fast"]
                    weight_map[:, ilat, ilon]  = res["weights"]
                    intercept_map[:, ilat, ilon] = res["a"]
                    slope_map[:, ilat, ilon]   = res["b"]
                    dispersion_map[ilat, ilon] = float(res["dispersion"])

                elif res["backend"] == "full":
                    fit_mode[ilat, ilon]       = self.MODE_CODE["full"]
                    weight_map[:, ilat, ilon]  = res["weights"].mean(axis=0)
                    intercept_map[:, ilat, ilon] = res["a"].mean(axis=0)
                    slope_map[:, ilat, ilon]   = res["b"].mean(axis=0)
                    dispersion_map[ilat, ilon] = float(np.mean(res["dispersion"]))
                    # FIX #7: store by coordinate value, not array index
                    lat_key = float(self._lat_values[ilat])
                    lon_key = float(self._lon_values[ilon])
                    self.posterior_[(lat_key, lon_key)] = res

        coords_w  = {member_dim: self._member_values,
                     lat_dim:    self._lat_values,
                     lon_dim:    self._lon_values}
        coords_2d = {lat_dim: self._lat_values, lon_dim: self._lon_values}

        self.weight_map      = xr.DataArray(weight_map,    dims=(member_dim, lat_dim, lon_dim), coords=coords_w)
        self.intercept_map   = xr.DataArray(intercept_map, dims=(member_dim, lat_dim, lon_dim), coords=coords_w)
        self.slope_map       = xr.DataArray(slope_map,     dims=(member_dim, lat_dim, lon_dim), coords=coords_w)
        self.dispersion_map  = xr.DataArray(dispersion_map, dims=(lat_dim, lon_dim), coords=coords_2d)
        self.family_map_used = xr.DataArray(family_code,   dims=(lat_dim, lon_dim), coords=coords_2d)
        self.fit_mode_map    = xr.DataArray(fit_mode,      dims=(lat_dim, lon_dim), coords=coords_2d)
        self.fit_status_map  = xr.DataArray(fit_status,    dims=(lat_dim, lon_dim), coords=coords_2d)
        self.fit_converged_ = xr.DataArray(converged, dims=(lat_dim, lon_dim), coords=coords_2d)

        self.fitted = True
        return self

    # ─────────────────────────────────────── predictive helpers ─────────────

    def _mixture_mean_fast(
        self, f_m: np.ndarray, w: np.ndarray,
        a: np.ndarray, b: np.ndarray, disp: float, family: str
    ) -> float:
        if family == "normal":
            return float(np.sum(w * (a + b * f_m)))
        if family == "gamma":
            return float(np.sum(w * np.maximum(a + b * f_m, self.eps)))
        if family == "lognormal":
            # E[Y] = exp(μ_log + σ_log²/2) for each component
            mu_log = a + b * np.log(np.maximum(f_m, self.eps))
            return float(np.sum(w * np.exp(mu_log + 0.5 * disp ** 2)))
        raise ValueError(family)

    def _mixture_cdf_fast(
        self, x: float, f_m: np.ndarray, w: np.ndarray,
        a: np.ndarray, b: np.ndarray, disp: float, family: str
    ) -> float:
        if family == "normal":
            mu   = a + b * f_m
            vals = norm.cdf(x, loc=mu, scale=max(disp, self.eps))
            return float(np.sum(w * vals))
        if family == "gamma":
            mu   = np.maximum(a + b * f_m, self.eps)
            vals = sp_gamma.cdf(x, a=max(disp, self.eps), scale=mu / max(disp, self.eps))
            return float(np.sum(w * vals))
        if family == "lognormal":
            mu_log = a + b * np.log(np.maximum(f_m, self.eps))
            vals   = lognorm.cdf(x, s=max(disp, self.eps), scale=np.exp(mu_log))
            return float(np.sum(w * vals))
        raise ValueError(family)

    def _mixture_quantile_fast(
        self, q: float, f_m: np.ndarray, w: np.ndarray,
        a: np.ndarray, b: np.ndarray, disp: float, family: str
    ) -> float:
        if family == "normal":
            brackets = norm.ppf(q, loc=a + b*f_m, scale=max(disp, self.eps))
        elif family == "gamma":
            mu = np.maximum(a+b*f_m, self.eps)
            brackets = sp_gamma.ppf(q, a=max(disp, self.eps), scale=mu/max(disp, self.eps))
        elif family == "lognormal":
            mu_log = a + b*np.log(np.maximum(f_m, self.eps))
            brackets = lognorm.ppf(q, s=max(disp, self.eps), scale=np.exp(mu_log))
        else:
            raise ValueError(family)
        return _bma_quantile(lambda value: self._mixture_cdf_fast(value, f_m, w, a, b, disp, family), q, brackets)

    def _posterior_predictive(self, f_m, post, family, quantiles, thresholds):
        # Rao-Blackwellization: integrate each predictive component analytically
        # then average across stored posterior draws. No extra predictive RNG.
        w = np.asarray(post["weights"], dtype=float)
        w = w / w.sum(axis=1, keepdims=True)
        a, b = np.asarray(post["a"]), np.asarray(post["b"])
        disp = np.maximum(np.asarray(post["dispersion"])[:, None], self.eps)
        if family == "normal":
            mu = a + b * f_m[None, :]
            dist = norm(loc=mu, scale=disp)
            means = mu
        elif family == "gamma":
            mu = np.maximum(a + b*f_m[None, :], self.eps)
            dist = sp_gamma(a=disp, scale=mu/disp)
            means = mu
        elif family == "lognormal":
            mu = a + b*np.log(np.maximum(f_m[None, :], self.eps))
            dist = lognorm(s=disp, scale=np.exp(mu))
            means = np.exp(mu + disp**2/2.)
        else:
            raise ValueError(family)
        cdf = lambda value: float(np.mean(np.sum(w*dist.cdf(value), axis=1)))
        mean = float(np.mean(np.sum(w*means, axis=1)))
        qs = [_bma_quantile(cdf, q, dist.ppf(q)) for q in quantiles]
        probs = None
        if thresholds is not None and np.all(np.isfinite(thresholds)):
            lower, upper = [cdf(value) for value in thresholds]
            probs = _normalize_tercile_probs(lower, upper-lower, 1-upper)
        return mean, qs, probs

    def _sample_pp_full(
        self, f_m: np.ndarray, post: Dict[str, np.ndarray],
        family: str, rng: np.random.Generator, n_samples: int
    ) -> np.ndarray:
        n_post = post["weights"].shape[0]
        idx    = rng.choice(n_post, size=n_samples, replace=(n_samples > n_post))

        w    = post["weights"][idx, :]
        a    = post["a"][idx, :]
        b    = post["b"][idx, :]
        disp = post["dispersion"][idx]   # shape: (n_samples,)

        u    = rng.random(n_samples)
        cdf  = np.cumsum(w, axis=1)
        comp = (u[:, None] > cdf).sum(axis=1)   # selected component index

        if family == "normal":
            mu     = a + b * f_m[None, :]
            mu_sel = mu[np.arange(n_samples), comp]
            return rng.normal(mu_sel, disp)

        if family == "gamma":
            mu     = np.maximum(a + b * f_m[None, :], self.eps)
            mu_sel = mu[np.arange(n_samples), comp]
            shape  = np.maximum(disp, self.eps)
            rate   = shape / np.maximum(mu_sel, self.eps)
            return rng.gamma(shape=shape, scale=1.0 / rate)

        if family == "lognormal":
            mu_log = a + b * np.log(np.maximum(f_m[None, :], self.eps))
            mu_sel = mu_log[np.arange(n_samples), comp]
            return rng.lognormal(mean=mu_sel, sigma=np.maximum(disp, self.eps))

        raise ValueError(family)

    # ────────────────────────────────────────────────────────── predict ──────

    def predict_probabilistic(
        self,
        new_forecasts:    xr.DataArray,
        *,
        clim_terciles:    Optional[xr.DataArray] = None,
        obs_for_terciles: Optional[xr.DataArray] = None,
        tercile_q:        Tuple[float, float]    = (1 / 3, 2 / 3),
        quantiles:        Sequence[float]         = (0.1, 0.5, 0.9),
        n_pp_samples:     int   = 2000,
        member_dim:       str   = "M",
        time_dim:         str   = "T",
        lat_dim:          str   = "Y",
        lon_dim:          str   = "X",
    ) -> xr.Dataset:
        if not self.fitted:
            raise ValueError("Call .fit() before .predict_probabilistic().")

        self._require_dims(
            new_forecasts, [member_dim, time_dim, lat_dim, lon_dim], "new_forecasts"
        )
        new_forecasts = _bma_forecast_members(new_forecasts, self._member_values, member_dim)
        if any(not np.isfinite(q) or not 0 < q < 1 for q in quantiles):
            raise ValueError("BMA predictive quantiles must lie strictly inside (0, 1)")

        # FIX #6: spatially align all parameter maps to the forecast grid via
        #         nearest-neighbour selection.  Original code used raw numpy
        #         indices, which silently gave wrong results when the forecast
        #         grid differed from the training grid.
        sel = {lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]}

        w_map     = self.weight_map.sel(sel, method="nearest").transpose(member_dim, lat_dim, lon_dim).values
        a_map     = self.intercept_map.sel(sel, method="nearest").transpose(member_dim, lat_dim, lon_dim).values
        b_map     = self.slope_map.sel(sel, method="nearest").transpose(member_dim, lat_dim, lon_dim).values
        d_map     = self.dispersion_map.sel(sel, method="nearest").transpose(lat_dim, lon_dim).values
        fam_map   = self.family_map_used.sel(sel, method="nearest").transpose(lat_dim, lon_dim).values
        mode_map  = self.fit_mode_map.sel(sel, method="nearest").transpose(lat_dim, lon_dim).values
        status_map = self.fit_status_map.sel(sel, method="nearest").transpose(lat_dim, lon_dim).values

        # FIX #7: build nearest-training-coordinate lookup arrays for posterior retrieval
        sel_lat = self.weight_map[lat_dim].sel({lat_dim: new_forecasts[lat_dim]}, method="nearest").values
        sel_lon = self.weight_map[lon_dim].sel({lon_dim: new_forecasts[lon_dim]}, method="nearest").values

        fcst    = new_forecasts.transpose(member_dim, time_dim, lat_dim, lon_dim).values
        _, n_time, n_lat, n_lon = fcst.shape

        terciles_da = self._align_terciles_to_forecast_grid(
            new_forecasts=new_forecasts,
            clim_terciles=clim_terciles,
            obs_for_terciles=obs_for_terciles,
            tercile_q=tercile_q,
            lat_dim=lat_dim, lon_dim=lon_dim, time_dim=time_dim,
        )
        terciles_np = (
            terciles_da.transpose("tercile", lat_dim, lon_dim).values
            if terciles_da is not None else None
        )

        out_mean   = np.full((n_time, n_lat, n_lon), np.nan)
        out_quant  = np.full((len(quantiles), n_time, n_lat, n_lon), np.nan)
        out_probs  = np.full((3, n_time, n_lat, n_lon), np.nan) if terciles_np is not None else None
        out_family = np.full((n_lat, n_lon), np.nan)
        out_mode   = np.full((n_lat, n_lon), np.nan)
        out_status = np.full((n_lat, n_lon), np.nan)

        # n_pp_samples remains accepted for API compatibility; predictive
        # means/CDFs integrate the stored posterior draws analytically.
        if self.verbose:
            print("Predicting with hybrid Bayesian postprocessor …")

        for ilat in tqdm(range(n_lat), desc="Predict", disable=not self.verbose):
            for ilon in range(n_lon):
                if status_map[ilat, ilon] != 1.0:
                    continue
                fam_code = fam_map[ilat, ilon]
                if not np.isfinite(fam_code):
                    continue
                fam = self.DIST_CODE_TO_NAME.get(int(fam_code))
                if fam is None:
                    continue

                mode_code = int(mode_map[ilat, ilon])
                out_family[ilat, ilon] = int(fam_code)
                out_mode[ilat, ilon]   = mode_code
                out_status[ilat, ilon] = 1.0

                low_th = up_th = None
                if terciles_np is not None:
                    low_th, up_th = terciles_np[:, ilat, ilon]

                for t in range(n_time):
                    f_m = fcst[:, t, ilat, ilon]
                    if not np.isfinite(f_m).all():
                        continue

                    if mode_code == self.MODE_CODE["full"]:
                        # FIX #7: look up posterior by training coordinate value
                        lat_key = float(sel_lat[ilat])
                        lon_key = float(sel_lon[ilon])
                        post = self.posterior_.get((lat_key, lon_key))
                        if post is None:
                            continue

                        mean, q_values, probabilities = self._posterior_predictive(
                            f_m, post, fam, quantiles,
                            None if terciles_np is None else (low_th, up_th))
                        out_mean[t, ilat, ilon] = mean
                        out_quant[:, t, ilat, ilon] = q_values
                        if probabilities is not None:
                            out_probs[:, t, ilat, ilon] = probabilities

                    elif mode_code == self.MODE_CODE["fast"]:
                        w    = w_map[:, ilat, ilon]
                        a    = a_map[:, ilat, ilon]
                        b    = b_map[:, ilat, ilon]
                        disp = float(d_map[ilat, ilon])

                        out_mean[t, ilat, ilon] = self._mixture_mean_fast(f_m, w, a, b, disp, fam)
                        for iq, qv in enumerate(quantiles):
                            out_quant[iq, t, ilat, ilon] = self._mixture_quantile_fast(
                                qv, f_m, w, a, b, disp, fam
                            )

                        if terciles_np is not None and np.isfinite(low_th) and np.isfinite(up_th):
                            pb = self._mixture_cdf_fast(float(low_th), f_m, w, a, b, disp, fam)
                            pa = 1.0 - self._mixture_cdf_fast(float(up_th), f_m, w, a, b, disp, fam)
                            # FIX #9: normalise all three probabilities
                            out_probs[:, t, ilat, ilon] = _normalize_tercile_probs(
                                pb, 1.0 - pb - pa, pa
                            )

        coords = {
            time_dim: new_forecasts[time_dim],
            lat_dim:  new_forecasts[lat_dim],
            lon_dim:  new_forecasts[lon_dim],
        }
        ll_coords = {lat_dim: new_forecasts[lat_dim], lon_dim: new_forecasts[lon_dim]}

        ds = xr.Dataset()
        ds["predictive_mean"] = xr.DataArray(
            out_mean, dims=(time_dim, lat_dim, lon_dim), coords=coords
        )
        ds["predictive_quantiles"] = xr.DataArray(
            out_quant,
            dims=("quantile", time_dim, lat_dim, lon_dim),
            coords={**coords, "quantile": list(quantiles)},
        )
        ds["family_used"] = xr.DataArray(
            out_family, dims=(lat_dim, lon_dim), coords=ll_coords,
            attrs={"1": "normal", "4": "gamma", "2": "lognormal"},
        )
        ds["fit_mode_used"] = xr.DataArray(
            out_mode, dims=(lat_dim, lon_dim), coords=ll_coords,
            attrs={"0": "full", "1": "fast", "-1": "failed"},
        )
        ds["fit_status"] = xr.DataArray(
            out_status, dims=(lat_dim, lon_dim), coords=ll_coords,
            attrs={"1": "success", "0": "failed_or_not_fitted"},
        )
        if out_probs is not None:
            ds["tercile_probability"] = xr.DataArray(
                out_probs,
                dims=("probability", time_dim, lat_dim, lon_dim),
                coords={**coords, "probability": ["PB", "PN", "PA"]},
            )
            ds["tercile_thresholds"] = terciles_da
        return _bma_output_metadata(ds)

    # ─────────────────────────────────────── WAS-style convenience wrapper ──

    def compute_model(
        self,
        X_train:   xr.DataArray,
        y_train:   xr.DataArray,
        X_test:    xr.DataArray,
        dist_map:  xr.DataArray,
        *,
        clim_terciles:    Optional[xr.DataArray]  = None,
        obs_for_terciles: Optional[xr.DataArray]  = None,
        tercile_q:        Tuple[float, float]     = (1 / 3, 2 / 3),
        quantiles:        Sequence[float]          = (0.1, 0.5, 0.9),
        n_pp_samples:     int   = 2000,
        member_dim:       str   = "M",
        time_dim:         str   = "T",
        lat_dim:          str   = "Y",
        lon_dim:          str   = "X",
        min_samples:      int   = 12,
        lam_w:            float = 1.0,
        lam_a:            float = 0.1,
        lam_b:            float = 0.1,
        lam_disp:         float = 0.1,
        alpha0:           float = 1.0,
        coef_scale_mult:  float = 1.0,
        sigma_scale_mult: float = 1.0,
        shape_scale:      float = 5.0,
    ) -> xr.Dataset:
        """Fit on (X_train, y_train) then predict on X_test — one-call interface."""
        if time_dim == "T":
            X_train = align_predictor_target(X_train, y_train, mode="season_year")
        self.fit(
            hcst_grid=X_train, obs_grid=y_train, dist_map=dist_map,
            member_dim=member_dim, time_dim=time_dim,
            lat_dim=lat_dim,       lon_dim=lon_dim,
            min_samples=min_samples,
            lam_w=lam_w, lam_a=lam_a, lam_b=lam_b, lam_disp=lam_disp,
            alpha0=alpha0, coef_scale_mult=coef_scale_mult,
            sigma_scale_mult=sigma_scale_mult, shape_scale=shape_scale,
        )
        return self.predict_probabilistic(
            new_forecasts=X_test,
            clim_terciles=clim_terciles,
            obs_for_terciles=obs_for_terciles,
            tercile_q=tercile_q,
            quantiles=quantiles,
            n_pp_samples=n_pp_samples,
            member_dim=member_dim, time_dim=time_dim,
            lat_dim=lat_dim,       lon_dim=lon_dim,
        )

############################################################
### think to allow n_jobs as argument in future version

class WAS_mme_RF:
    """
    Enhanced Random Forest-based Multi-Model Ensemble (MME) forecasting for West Africa.
    
    Parameters
    ----------
    search_method : str, optional
        Hyperparameter optimization method: 'grid', 'random', or 'bayesian' (default: 'random').
    n_estimators_range : list of int, optional
        List of n_estimators values to tune (default is [200, 400, 700]).
    max_depth_range : list of int or None, optional
        List of max depths to tune (default is [4, 6, 8, 12]).
    min_samples_split_range : list of int, optional
        List of minimum samples required to split (default is [10, 20, 40]).
    min_samples_leaf_range : list of int, optional
        List of minimum samples required at leaf node (default is [5, 10, 20]).
    max_features_range : list of str or float, optional
        List of max features to tune (default is [0.4, 0.7, 1.0]).
    bootstrap_range : list of bool, optional
        List of bootstrap options to tune (default is [True, False]).
    max_samples_range : list of float or None, optional
        List of max samples as fraction of dataset (default is [None, 0.7, 0.8]).
    min_impurity_decrease_range : list of float, optional
        List of min impurity decrease values (default is [0.0, 0.001, 0.01]).
    max_leaf_nodes_range : list of int or None, optional
        List of max leaf nodes (default is [None, 50, 100]).
    ccp_alpha_range : list of float, optional
        List of complexity parameters for pruning (default is [0.0, 0.001, 0.01]).
    min_weight_fraction_leaf_range : list of float, optional
        List of minimum weighted fraction of leaves (default is [0.0, 0.1, 0.2]).
    warm_start : bool, optional
        Whether to reuse solution of previous call to fit (default is False).
    random_state : int, optional
        Seed for reproducibility (default is 42).
    dist_method : str, optional
        Distribution method for tercile probabilities ('bestfit', 'nonparam', etc.) (default is 'nonparam').
    n_iter_search : int, optional
        Number of iterations for randomized/bayesian search (default is 20).
    cv_folds : int, optional
        Number of cross-validation folds (default is 5).
    n_clusters : int, optional
        Number of clusters for homogenized zones (default is 6 for West Africa).
    scoring : str, optional
        Scoring metric for hyperparameter optimization (default is 'neg_mean_squared_error').
    verbose : int, optional
        Verbosity level (default is 0).
    n_jobs : int, optional
        Maximum number of parallel workers. ``-1`` uses all available CPUs.
        Hyperparameter searches parallelize the outer CV/trial loop, while
        each inner Random Forest fit uses one worker to avoid oversubscription.
        Final model fitting uses ``n_jobs`` directly (default is 4).
    """

    def __init__(self,
                 search_method: str = 'random',
                 n_estimators_range: Union[List[int], object] = [200, 400, 700],
                 max_depth_range: Union[List[Optional[int]], object] = [4, 6, 8, 12],
                 min_samples_split_range: Union[List[int], object] = [10, 20, 40],
                 min_samples_leaf_range: Union[List[int], object] = [5, 10, 20],
                 max_features_range: List[Union[str, float]] = [0.4, 0.7, 1.0],
                 bootstrap_range: List[bool] = [True, False],
                 max_samples_range: List[Optional[float]] = [None, 0.7, 0.8],
                 min_impurity_decrease_range: List[float] = [0.0, 0.001, 0.01],
                 ccp_alpha_range: List[float] = [0.0, 0.001, 0.01],
                 warm_start: bool = False,
                 random_state: int = 42,
                 dist_method: str = "nonparam",
                 n_iter_search: int = 20,
                 cv_folds: int = 5,
                 n_clusters: int = 6,
                 scoring: str = 'neg_mean_squared_error',
                 leave_one_year_out: bool = False,   # ===== 2 =====
                 verbose: int = 0,
                 n_jobs: int = -1,
                 optuna_n_jobs: Optional[int] = None,
                 cv_n_jobs: int = -1,
                 tuning_model_n_jobs: int = 1,
                 cluster_n_jobs: Optional[int] = None,
                 spatial_subsample: int = 1,
                 spatial_mode: str = "zones",
                 covariate_names: Optional[Sequence[str]] = None,
                 static_fields: Optional[Dict[str, xr.DataArray]] = None,
                 coord_step: Optional[float] = None,
                 zones_as_features: bool = False,
                 predictor_smoothing: int = 0,
                 one_se_rule: bool = False,
                 amplitude_calibration=True,
                 amplitude_calibration_inner_folds: int = 3):

        # Threads used to fit the clusters concurrently (None -> auto).
        self.cluster_n_jobs = cluster_n_jobs
        # Keep every `spatial_subsample`-th grid cell (in Y AND X) when FITTING
        # and when TUNING. 1 disables it. On a 350x171 grid a cluster holds
        # ~287,000 training samples for 10 features, and neighbouring cells are
        # near-duplicates; the fit cost grows as n log n, so a stride of 2 cuts
        # it by roughly 4.5x. It changes the forecasts, so it stays off by default.
        self.spatial_subsample = max(1, int(spatial_subsample))
        # Spatial regionalisation ("zones" | "soft" | "both"), spatial
        # covariates, regional GCM smoothing and the one-SE selection rule.
        _init_spatial_options(self, spatial_mode, covariate_names, static_fields,
                              coord_step, zones_as_features, predictor_smoothing, one_se_rule,
                              amplitude_calibration, amplitude_calibration_inner_folds)
        self.search_method = search_method
        self.n_estimators_range = n_estimators_range
        self.max_depth_range = max_depth_range
        self.min_samples_split_range = min_samples_split_range
        self.min_samples_leaf_range = min_samples_leaf_range
        self.max_features_range = max_features_range
        self.bootstrap_range = bootstrap_range
        self.max_samples_range = max_samples_range
        self.min_impurity_decrease_range = min_impurity_decrease_range
        self.ccp_alpha_range = ccp_alpha_range
        self.warm_start = warm_start
        self.random_state = random_state
        self.dist_method = dist_method
        self.n_iter_search = n_iter_search
        self.cv_folds = cv_folds
        self.n_clusters = n_clusters
        self.scoring = scoring
        self.leave_one_year_out = leave_one_year_out
        self.verbose = verbose
        if n_jobs == 0:
            raise ValueError("n_jobs cannot be 0; use a positive integer or -1.")
        if optuna_n_jobs == 0:
            raise ValueError("optuna_n_jobs cannot be 0.")
        if cv_n_jobs == 0:
            raise ValueError("cv_n_jobs cannot be 0.")
        if tuning_model_n_jobs == 0:
            raise ValueError("tuning_model_n_jobs cannot be 0.")
        self.n_jobs = n_jobs
        self.optuna_n_jobs = 1 if optuna_n_jobs is None else optuna_n_jobs
        self.cv_n_jobs = cv_n_jobs
        self.tuning_model_n_jobs = tuning_model_n_jobs
        self.rf = None
        self.best_params_dict = {}
        self.cluster_da = None

    # ===== 2: year-grouped CV splitter, reused everywhere =====
    def _cv_splitter(self, groups: np.ndarray):
        """
        Return a CV splitter that never puts cells from the same calendar year
        on both sides of the train/test boundary.

        Uses leave-one-year-out when requested or when distinct years are few
        (<= cv_folds); otherwise GroupKFold with cv_folds blocks of years.
        """
        n_groups = len(np.unique(groups))
        if self.leave_one_year_out or n_groups <= self.cv_folds:
            return LeaveOneGroupOut()
        return GroupKFold(n_splits=self.cv_folds)

    def _objective(self, trial, X_train, y_train, groups) -> float:
        """Objective function for Optuna optimization (year-grouped CV)."""
        if isinstance(self.n_estimators_range, list):
            params = {'n_estimators': trial.suggest_categorical('n_estimators', self.n_estimators_range)}
        else:
            params = {'n_estimators': trial.suggest_int('n_estimators', 50, 1000)}

        if isinstance(self.max_depth_range, list):
            max_depth_options = [str(opt) if opt is None else opt for opt in self.max_depth_range]
            max_depth_str = trial.suggest_categorical('max_depth', max_depth_options)
            params['max_depth'] = None if max_depth_str == 'None' else int(max_depth_str)
        else:
            params['max_depth'] = trial.suggest_int('max_depth', 5, 50)

        for param_name, param_range in [
            ('min_samples_split', self.min_samples_split_range),
            ('min_samples_leaf', self.min_samples_leaf_range),
        ]:
            if isinstance(param_range, list):
                params[param_name] = trial.suggest_categorical(param_name, param_range)
            else:
                params[param_name] = trial.suggest_int(param_name, 2, 20)

        max_features_options = [str(opt) if isinstance(opt, (int, float)) else opt for opt in self.max_features_range]
        max_features_str = trial.suggest_categorical('max_features', max_features_options)
        if max_features_str in ['sqrt', 'log2', 'auto', None]:
            params['max_features'] = max_features_str if max_features_str != 'auto' else 'sqrt'
        else:
            params['max_features'] = float(max_features_str)

        params['bootstrap'] = trial.suggest_categorical('bootstrap', self.bootstrap_range)

        if params['bootstrap']:
            if isinstance(self.max_samples_range[0], (int, float)):
                params['max_samples'] = trial.suggest_float('max_samples',
                                                            min(self.max_samples_range),
                                                            max(self.max_samples_range))
            else:
                max_samples_options = [
                    None if opt is None else float(opt) for opt in self.max_samples_range
                ]
                max_samples_value = trial.suggest_categorical('max_samples', max_samples_options)
                params['max_samples'] = None if max_samples_value is None else float(max_samples_value)
        else:
            params['max_samples'] = None

        params['min_impurity_decrease'] = trial.suggest_float('min_impurity_decrease',
                                                             min(self.min_impurity_decrease_range),
                                                             max(self.min_impurity_decrease_range))

        params['ccp_alpha'] = trial.suggest_float('ccp_alpha',
                                                  min(self.ccp_alpha_range),
                                                  max(self.ccp_alpha_range))

        model = RandomForestRegressor(
            **params,
            random_state=self.random_state,
            n_jobs=self.tuning_model_n_jobs,
            warm_start=self.warm_start
        )

        # ===== 2: grouped CV instead of plain KFold =====
        try:
            cv = self._cv_splitter(groups)
            scores = cross_val_score(
                model, X_train, y_train,
                cv=cv, groups=groups,
                scoring=self.scoring,
                n_jobs=self.cv_n_jobs
            )
            return np.mean(scores)
        except Exception as e:
            print(f"TRIAL FAILED: {e}")
            import traceback
            traceback.print_exc()
            return -1e10

    def _grid_search(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Dict:
        """Grid search with year-grouped CV."""
        base = {
            'n_estimators': self.n_estimators_range,
            'max_depth': self.max_depth_range,
            'min_samples_split': self.min_samples_split_range,
            'min_samples_leaf': self.min_samples_leaf_range,
            'max_features': self.max_features_range,
            'min_impurity_decrease': self.min_impurity_decrease_range,
            'ccp_alpha': self.ccp_alpha_range,
        }
        param_grid = []
        if True in self.bootstrap_range:
            param_grid.append({**base, 'bootstrap': [True],
                               'max_samples': list(self.max_samples_range)})
        if False in self.bootstrap_range:
            param_grid.append({**base, 'bootstrap': [False], 'max_samples': [None]})
        if not param_grid:
            raise ValueError("bootstrap_range must contain True and/or False.")
        model = RandomForestRegressor(
            random_state=self.random_state, n_jobs=self.tuning_model_n_jobs
        )
        cv = self._cv_splitter(groups)  # ===== 2 =====
        grid_search = GridSearchCV(
            model, param_grid=param_grid,
            cv=cv, scoring=self.scoring,
            n_jobs=self.n_jobs, verbose=self.verbose
        )
        grid_search.fit(X, y, groups=groups)  # ===== 2 =====
        if self.verbose > 0:
            print(f"Best score: {grid_search.best_score_:.4f}")
            print(f"Best params: {grid_search.best_params_}")
        return grid_search.best_params_

    def _random_search(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Dict:
        """Random search with year-grouped CV."""
        base = {
            'n_estimators': self.n_estimators_range,
            'max_depth': self.max_depth_range,
            'min_samples_split': self.min_samples_split_range,
            'min_samples_leaf': self.min_samples_leaf_range,
            'max_features': self.max_features_range,
            'min_impurity_decrease': self.min_impurity_decrease_range,
            'ccp_alpha': self.ccp_alpha_range,
        }
        param_dist = []
        if True in self.bootstrap_range:
            param_dist.append({**base, 'bootstrap': [True],
                               'max_samples': list(self.max_samples_range)})
        if False in self.bootstrap_range:
            param_dist.append({**base, 'bootstrap': [False], 'max_samples': [None]})
        if not param_dist:
            raise ValueError("bootstrap_range must contain True and/or False.")
        model = RandomForestRegressor(
            random_state=self.random_state, n_jobs=self.tuning_model_n_jobs
        )
        cv = self._cv_splitter(groups)  # ===== 2 =====
        random_search = RandomizedSearchCV(
            model, param_distributions=param_dist,
            n_iter=self.n_iter_search, cv=cv,
            scoring=self.scoring, random_state=self.random_state,
            n_jobs=self.n_jobs, verbose=self.verbose
        )
        random_search.fit(X, y, groups=groups)  # ===== 2 =====
        if self.verbose > 0:
            print(f"Best score: {random_search.best_score_:.4f}")
            print(f"Best params: {random_search.best_params_}")
        return random_search.best_params_

    def _bayesian_search(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Dict:
        """Bayesian optimization (Optuna) with year-grouped CV."""
        study = optuna.create_study(
            direction='maximize' if 'neg_' in self.scoring else 'minimize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        objective_with_data = lambda trial: self._objective(trial, X, y, groups)  # ===== 2 =====
        study.optimize(
            objective_with_data,
            n_trials=self.n_iter_search,
            n_jobs=self.optuna_n_jobs,
            show_progress_bar=self.verbose > 0
        )
        best_params = study.best_params
        sklearn_params = {}
        for key, value in best_params.items():
            if key == 'max_features' and isinstance(value, str):
                if value == 'None':
                    sklearn_params[key] = None
                else:
                    try:
                        sklearn_params[key] = float(value)
                    except ValueError:
                        sklearn_params[key] = value
            elif key == 'max_depth' and value == 'None':
                sklearn_params[key] = None
            elif key == 'max_leaf_nodes' and value == 'None':
                sklearn_params[key] = None
            elif key == 'max_samples' and value == 'None':
                sklearn_params[key] = None
            else:
                sklearn_params[key] = value
        if self.verbose > 0:
            print(f"Best score: {study.best_value:.4f}")
            print(f"Best params: {sklearn_params}")
        return sklearn_params

    def compute_hyperparameters(self, Predictors, Predictand,
                                clim_year_start=None, clim_year_end=None):
        """Tune with fold-local standardization, target encoding and climate zones."""
        return _strict_regional_hyperparameters(
            self, Predictors, Predictand, clim_year_start, clim_year_end)

    def compute_hyperparameters_legacy(self, Predictors: xr.DataArray, Predictand: xr.DataArray,
                                clim_year_start: Optional[int] = None,
                                clim_year_end: Optional[int] = None) -> Tuple[Dict, xr.DataArray]:
        """
        Compute best hyperparameters per zone. Regionalization is KMeans on the
        predictand values (kept by preference); CV is year-grouped (2).

        If clim_year_start/clim_year_end are given, predictors and predictand
        are standardized here. If they are None, the inputs are assumed already
        standardized -- e.g. when called from WAS_Cross_Validator, which
        standardizes before the fold loop. This removes the old hard-coded
        1970-2000 window AND the double-standardization that happened when the
        framework had already standardized the data.
        """
        Predictors = align_predictor_target(Predictors, Predictand, mode="season_year")
        if "M" in Predictand.dims:
            Predictand = Predictand.isel(M=0, drop=True)
        Predictand.name = "varname"

        if clim_year_start is not None and clim_year_end is not None:
            X_train_std = standardize_timeseries(Predictors, clim_year_start, clim_year_end)
            y_train_std = standardize_timeseries(Predictand, clim_year_start, clim_year_end)
        else:
            X_train_std = Predictors
            y_train_std = Predictand
        X_train_std['T'] = y_train_std['T']


        self.cluster_da = _build_climate_feature_clusters(
            Predictand, self.n_clusters, random_state=self.random_state
        )
        clusters = np.unique(self.cluster_da.values)
        clusters = clusters[np.isfinite(clusters)]

        best_params_dict = {}
        for c in clusters:
            c = int(c)
            if self.verbose > 0:
                print(f"\nOptimizing hyperparameters for cluster {c}...")

            mask_3d = (self.cluster_da == c).broadcast_like(X_train_std.isel(M=0))

            X_stacked_c = (
                X_train_std.where(mask_3d)
                .stack(sample=('T', 'Y', 'X')).transpose('sample', 'M')
            )
            y_stacked_c = (
                y_train_std.where(mask_3d)
                .stack(sample=('T', 'Y', 'X'))
            )

            # ===== 2: year groups aligned with the stacked samples =====
            groups_full = X_stacked_c['T'].values.astype('datetime64[Y]').astype(int)
            Xv = X_stacked_c.values
            yv = y_stacked_c.values.ravel()
            nan_mask_c = np.any(~np.isfinite(Xv), axis=1) | ~np.isfinite(yv)
            X_clean_c = Xv[~nan_mask_c]
            y_clean_c = yv[~nan_mask_c]
            groups_c = groups_full[~nan_mask_c]
            # ===== end 2 =====

            n_years_c = len(np.unique(groups_c))
            if len(X_clean_c) < self.cv_folds * 2 or n_years_c < 2:
                if self.verbose > 0:
                    print(f"Cluster {c}: insufficient data "
                          f"({len(X_clean_c)} samples, {n_years_c} years). Skipping.")
                continue

            if self.search_method == 'grid':
                best_params_dict[c] = self._grid_search(X_clean_c, y_clean_c, groups_c)
            elif self.search_method == 'random':
                best_params_dict[c] = self._random_search(X_clean_c, y_clean_c, groups_c)
            elif self.search_method == 'bayesian':
                best_params_dict[c] = self._bayesian_search(X_clean_c, y_clean_c, groups_c)
            else:
                raise ValueError(f"Unknown search_method: {self.search_method}")

        self.best_params_dict = best_params_dict
        return best_params_dict, self.cluster_da

    def _cluster_workers(self):
        """Threads used to fit the clusters concurrently.

        Clusters are independent, so this is the parallel axis that actually
        pays. scikit-learn releases the GIL during tree building, so threads
        are enough and avoid the memory duplication of processes. Controlled by
        ``cluster_n_jobs``; falls back to the number of clusters capped by
        ``n_jobs``.
        """
        requested = getattr(self, "cluster_n_jobs", None)
        if requested is None:
            budget = self.n_jobs if self.n_jobs and self.n_jobs > 0 else _grid_workers()
            requested = max(1, min(int(self.n_clusters), int(budget)))
        return max(1, int(requested))

    def compute_model(self, X_train: xr.DataArray, y_train: xr.DataArray,
                      X_test: xr.DataArray, y_test: Optional[xr.DataArray] = None,
                      clim_year_start: Optional[int] = None,
                      clim_year_end: Optional[int] = None,
                      best_params: Optional[Dict] = None,
                      cluster_da: Optional[xr.DataArray] = None,
                      covariates: Optional[xr.DataArray] = None) -> xr.DataArray:
        """
        Deterministic hindcast using RandomForestRegressor with optimized
        hyperparameters.

        ``covariates`` (M, Y, X) are the fold-local spatial covariates used by
        ``spatial_mode`` "soft"/"both"; WAS_Cross_Validator passes them, and the
        map stored by ``compute_hyperparameters`` is reused otherwise.
        """
        validate_mme_predictor_schema(X_train, X_test)
        X_train = align_predictor_target(X_train, y_train, mode="season_year")
        best_params, cluster_da = _resolve_regional_params(
            self, X_train, y_train, clim_year_start, clim_year_end, best_params, cluster_da)
        _raw_predictors = (X_train, X_test)   # avant covariables : pour la calibration
        X_train, X_test = _prepare_channels(self, X_train, X_test, y_train, covariates,
                                            clim_year_start, clim_year_end)

        time = X_test['T']
        lat = X_test['Y']
        lon = X_test['X']
        n_time = len(time)
        n_lat = len(lat)
        n_lon = len(lon)

        predictions = np.full((n_time * n_lat * n_lon,), np.nan)
        self.rf = {}

        # Row positions of every cluster in the stacked (T, Y, X) axis. The
        # arrays are stacked ONCE per fold and indexed per cluster; the previous
        # code re-materialised the full (T, M, Y, X) array for every cluster.
        train_rows = _cluster_row_index(cluster_da, X_train.sizes['T'])
        test_rows = _cluster_row_index(cluster_da, X_test.sizes['T'])

        active = [c for c in range(self.n_clusters) if c in best_params]

        def _fit_one(c):
            """Fit and predict one cluster. Returns (c, model, positions, values)."""
            bp = best_params[c]

            # Thin the TRAINING rows only (one cell out of `stride` in Y and X).
            # Prediction still covers every cell, so the output grid is unchanged.
            train_sel = _thin_training_rows(train_rows == c, X_train.sizes['T'],
                                            n_lat, n_lon, self.spatial_subsample)

            Xc, _ = _stack_rows(X_train, train_sel, core='M')
            yc, _ = _stack_rows(y_train, train_sel, core=None)
            yc = yc.ravel()
            keep = np.all(np.isfinite(Xc), axis=1) & np.isfinite(yc)
            if not keep.any():
                return c, None, None, None

            Xt, pos_t = _stack_rows(X_test, test_rows == c, core='M')
            keep_t = np.all(np.isfinite(Xt), axis=1)

            rf_c = RandomForestRegressor(
                **{k: v for k, v in bp.items() if k in RandomForestRegressor().get_params()},
                random_state=self.random_state,
                # One thread per model: the parallelism is at the cluster level.
                # Measured on 10 features, n_jobs=-1 inside a single fit gives
                # no speedup (36.7 s against 35.7 s) because the trees are too
                # small; running the clusters concurrently does.
                n_jobs=1 if self._cluster_workers() > 1 else self.n_jobs,
                warm_start=self.warm_start,
            )
            rf_c.fit(Xc[keep], yc[keep])

            if not keep_t.any():
                return c, rf_c, None, None
            return c, rf_c, pos_t[keep_t], rf_c.predict(Xt[keep_t])

        workers = self._cluster_workers()
        if workers > 1 and len(active) > 1:
            results = Parallel(n_jobs=min(workers, len(active)), backend="threading")(
                delayed(_fit_one)(c) for c in active
            )
        else:
            results = [_fit_one(c) for c in active]

        for c, model, positions, values in results:
            if model is not None:
                self.rf[c] = model
            if positions is not None:
                predictions[positions] = values
        predictions = predictions.reshape(n_time, n_lat, n_lon)

        predicted_da = xr.DataArray(
            data=predictions,
            coords={'T': time, 'Y': lat, 'X': lon},
            dims=['T', 'Y', 'X']
        )
        return _maybe_calibrate_amplitude(
            self, self.rf, X_train, y_train, predicted_da, cluster_da, X_test=X_test,
            raw_predictors=_raw_predictors, best_params=best_params,
            clim_year_start=clim_year_start, clim_year_end=clim_year_end)

    # ------------------ Probability Calculation Methods ------------------

    @staticmethod
    def _ppf_terciles_from_code(dist_code, shape, loc, scale):
        """Return tercile thresholds (T1, T2) from best-fit distribution params."""
        if np.isnan(dist_code):
            return np.nan, np.nan

        code = int(dist_code)
        try:
            if code == 1:
                return (norm.ppf(1.0 / 3.0, loc=loc, scale=scale),
                        norm.ppf(2.0 / 3.0, loc=loc, scale=scale))
            elif code == 2:
                return (lognorm.ppf(1.0 / 3.0, s=shape, loc=loc, scale=scale),
                        lognorm.ppf(2.0 / 3.0, s=shape, loc=loc, scale=scale))
            elif code == 3:
                return (expon.ppf(1.0 / 3.0, loc=loc, scale=scale),
                        expon.ppf(2.0 / 3.0, loc=loc, scale=scale))
            elif code == 4:
                return (gamma.ppf(1.0 / 3.0, a=shape, loc=loc, scale=scale),
                        gamma.ppf(2.0 / 3.0, a=shape, loc=loc, scale=scale))
            elif code == 5:
                return (weibull_min.ppf(1.0 / 3.0, c=shape, loc=loc, scale=scale),
                        weibull_min.ppf(2.0 / 3.0, c=shape, loc=loc, scale=scale))
            elif code == 6:
                return (t.ppf(1.0 / 3.0, df=shape, loc=loc, scale=scale),
                        t.ppf(2.0 / 3.0, df=shape, loc=loc, scale=scale))
            elif code == 7:
                return (poisson.ppf(1.0 / 3.0, mu=shape, loc=loc),
                        poisson.ppf(2.0 / 3.0, mu=shape, loc=loc))
            elif code == 8:
                return (nbinom.ppf(1.0 / 3.0, n=shape, p=scale, loc=loc),
                        nbinom.ppf(2.0 / 3.0, n=shape, p=scale, loc=loc))
        except Exception:
            return np.nan, np.nan

        return np.nan, np.nan

    @staticmethod
    def weibull_shape_solver(k, M, V):
        """Root function for the Weibull shape parameter 'k'."""
        if k <= 0:
            return -np.inf
        try:
            g1 = gamma_function(1 + 1/k)
            g2 = gamma_function(1 + 2/k)
            implied_v_over_m_sq = (g2 / (g1**2)) - 1
            observed_v_over_m_sq = V / (M**2)
            return observed_v_over_m_sq - implied_v_over_m_sq
        except ValueError:
            return -np.inf

    @staticmethod
    def calculate_tercile_probabilities_bestfit(best_guess, error_variance, T1, T2, dist_code, dof):
        """Generic tercile probabilities using best-fit family per grid cell."""
        best_guess = np.asarray(best_guess, float)
        error_variance = np.asarray(error_variance, dtype=float)
        n_time = best_guess.size
        out = np.full((3, n_time), np.nan, float)

        if np.all(np.isnan(best_guess)) or np.isnan(dist_code) or np.isnan(T1) or np.isnan(T2) or np.isnan(error_variance):
            return out

        code = int(dist_code)

        if code == 1:  # Normal
            error_std = np.sqrt(error_variance)
            out[0, :] = norm.cdf(T1, loc=best_guess, scale=error_std)
            out[1, :] = norm.cdf(T2, loc=best_guess, scale=error_std) - norm.cdf(T1, loc=best_guess, scale=error_std)
            out[2, :] = 1 - norm.cdf(T2, loc=best_guess, scale=error_std)

        elif code == 2:  # Lognormal
            sigma = np.sqrt(np.log(1 + error_variance / (best_guess**2)))
            mu = np.log(best_guess) - sigma**2 / 2
            out[0, :] = lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
            out[1, :] = lognorm.cdf(T2, s=sigma, scale=np.exp(mu)) - lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
            out[2, :] = 1 - lognorm.cdf(T2, s=sigma, scale=np.exp(mu))

        elif code == 3:  # Exponential
            c1 = expon.cdf(T1, loc=best_guess, scale=np.sqrt(error_variance))
            c2 = expon.cdf(T2, loc=best_guess, scale=np.sqrt(error_variance))
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        elif code == 4:  # Gamma
            alpha = (best_guess ** 2) / error_variance
            theta = error_variance / best_guess
            c1 = gamma.cdf(T1, a=alpha, scale=theta)
            c2 = gamma.cdf(T2, a=alpha, scale=theta)
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        elif code == 5:  # Weibull
            for i in range(n_time):
                M = best_guess[i]
                V = error_variance
                if V <= 0 or M <= 0:
                    out[:, i] = np.nan
                    continue
                k = fsolve(WAS_mme_RF.weibull_shape_solver, 2.0, args=(M, V))[0]
                if k <= 0:
                    out[:, i] = np.nan
                    continue
                lambda_scale = M / gamma_function(1 + 1/k)
                c1 = weibull_min.cdf(T1, c=k, loc=0, scale=lambda_scale)
                c2 = weibull_min.cdf(T2, c=k, loc=0, scale=lambda_scale)
                out[0, i] = c1
                out[1, i] = c2 - c1
                out[2, i] = 1.0 - c2

        elif code == 6:  # Student-t
            if dof <= 2:
                out[:, :] = np.nan
            else:
                loc = best_guess
                scale = np.sqrt(error_variance * (dof - 2) / dof)
                c1 = t.cdf(T1, df=dof, loc=loc, scale=scale)
                c2 = t.cdf(T2, df=dof, loc=loc, scale=scale)
                out[0, :] = c1
                out[1, :] = c2 - c1
                out[2, :] = 1.0 - c2

        elif code == 7:  # Poisson
            mu = best_guess
            c1 = poisson.cdf(T1, mu=mu)
            c2 = poisson.cdf(T2, mu=mu)
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        elif code == 8:  # Negative Binomial
            p = np.where(error_variance > best_guess, best_guess / error_variance, np.nan)
            n = np.where(error_variance > best_guess, (best_guess**2) / (error_variance - best_guess), np.nan)
            c1 = nbinom.cdf(T1, n=n, p=p)
            c2 = nbinom.cdf(T2, n=n, p=p)
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        else:
            raise ValueError(f"Invalid distribution code: {dist_code}")

        # ===== 5: positive-support families are undefined for x <= 0 =====
        # (e.g. when fed standardized anomalies rather than physical rainfall).
        if code in (2, 3, 4, 5, 7, 8):
            bad = ~(best_guess > 0)
            if np.any(bad):
                out[:, bad] = np.nan

        return out

    @staticmethod
    def calculate_tercile_probabilities_nonparametric(best_guess, error_samples, first_tercile, second_tercile):
        return WAS_mme_Weighted.calculate_tercile_probabilities_nonparametric(
            best_guess, error_samples, first_tercile, second_tercile)

    def compute_prob(self, Predictant: xr.DataArray, clim_year_start, clim_year_end,
                     hindcast_det: xr.DataArray, best_code_da: xr.DataArray = None,
                     best_shape_da: xr.DataArray = None, best_loc_da: xr.DataArray = None,
                     best_scale_da: xr.DataArray = None) -> xr.DataArray:
        """
        Tercile probabilities for deterministic hindcasts.

        NOTE (3): for honest, well-calibrated probabilities, pass the
        CROSS-VALIDATED hindcast here as `hindcast_det` (out-of-sample
        residuals). Feeding the in-sample hindcast makes the predictive
        distribution too sharp and the reliability diagram over-confident.
        """
        if "M" in Predictant.dims:
            Predictant = Predictant.isel(M=0, drop=True)

        Predictant = Predictant.transpose("T", "Y", "X")
        mask = xr.where(np.isfinite(Predictant).sum(dim='T') >= 3, 1.0, np.nan)

        clim = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))
        if clim.sizes.get("T", 0) < 3:
            raise ValueError("Not enough years in climatology period for terciles.")

        # Out-of-sample residual variance (hindcast_det should be cross-validated)
        error_variance = ((Predictant - hindcast_det) ** 2).mean(dim="T", skipna=True)
        dof = max(int(clim.sizes["T"]) - 1, 2)

        terciles_emp = climatological_terciles(clim)[0]
        T1_emp = terciles_emp.isel(quantile=0, drop=True)
        T2_emp = terciles_emp.isel(quantile=1, drop=True)

        dm = self.dist_method

        if dm == "bestfit":
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError(
                    "dist_method='bestfit' requires best_code_da, best_shape_da, best_loc_da, best_scale_da."
                )
            # Category boundaries stay on the observed climatological reference.
            # The selected distribution controls the predictive CDF only.
            if getattr(self, "tercile_reference", "climatology") == "training":
                T1, T2 = _pufunc(
                    self._ppf_terciles_from_code,
                    best_code_da, best_shape_da, best_loc_da, best_scale_da,
                    input_core_dims=[(), (), (), ()],
                    output_core_dims=[(), ()],
                    vectorize=True, dask="parallelized",
                    output_dtypes=[float, float],
                )
            else:
                T1, T2 = T1_emp, T2_emp
            hindcast_prob = _pufunc(
                self.calculate_tercile_probabilities_bestfit,
                hindcast_det, error_variance, T1, T2, best_code_da,
                input_core_dims=[("T",), (), (), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True, kwargs={'dof': dof},
                dask="parallelized", output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True},
            )

        elif dm == "nonparam":
            error_samples = Predictant - hindcast_det
            hindcast_prob = _pufunc(
                self.calculate_tercile_probabilities_nonparametric,
                hindcast_det, error_samples, T1_emp, T2_emp,
                input_core_dims=[("T",), ("T",), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True, dask="parallelized", output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True},
            )

        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}")

        hindcast_prob = hindcast_prob.assign_coords(probability=("probability", ["PB", "PN", "PA"]))
        return (hindcast_prob * mask).transpose("probability", "T", "Y", "X")

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det,
                 hindcast_det_cross, Predictor_for_year, best_params=None, cluster_da=None,
                 best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None):
        """Forecast using the Random Forest model with optimized hyperparameters."""
        validate_mme_predictor_schema(hindcast_det, Predictor_for_year)
        if "M" in Predictant.dims:
            Predictant_no_m = Predictant.isel(M=0, drop=True)
        else:
            Predictant_no_m = Predictant

        mask = xr.where(np.isfinite(Predictant_no_m).sum(dim='T') >= 3, 1.0, np.nan).to_numpy()

        hindcast_det_st, Predictor_for_year_st = standardize_mme_predictors(
            hindcast_det, Predictor_for_year, clim_year_start, clim_year_end)
        Predictant_st = standardize_timeseries(Predictant_no_m, clim_year_start, clim_year_end)
        hindcast_det_st['T'] = Predictant_st['T']

        time = Predictor_for_year_st['T']
        lat = Predictor_for_year_st['Y']
        lon = Predictor_for_year_st['X']
        n_time = len(time)
        n_lat = len(lat)
        n_lon = len(lon)

        best_params, cluster_da = _resolve_regional_params(
            self, hindcast_det, Predictant_no_m, clim_year_start, clim_year_end,
            best_params, cluster_da)
        # Spatial covariates / smoothed channels from the RAW predictand climatology.
        hindcast_det_st, Predictor_for_year_st = _prepare_channels(
            self, hindcast_det_st, Predictor_for_year_st, Predictant_no_m, None,
            clim_year_start, clim_year_end)

        predictions = np.full((n_time, n_lat, n_lon), np.nan)
        self.rf = {}

        for c in range(self.n_clusters):
            if c not in best_params:
                continue

            bp = best_params[c]

            mask_3d_train = (cluster_da == c).expand_dims({'T': hindcast_det_st['T']})
            mask_3d_test = (cluster_da == c).expand_dims({'T': Predictor_for_year_st['T']})

            X_train_stacked_c = _stack_masked(hindcast_det_st, mask_3d_train, core='M')
            y_train_stacked_c = _stack_masked(Predictant_st, mask_3d_train, core=None).ravel()

            train_nan_mask = np.any(~np.isfinite(X_train_stacked_c), axis=1) | ~np.isfinite(y_train_stacked_c)
            X_train_clean_c = X_train_stacked_c[~train_nan_mask]
            y_train_clean_c = y_train_stacked_c[~train_nan_mask]

            X_test_stacked_c = _stack_masked(Predictor_for_year_st, mask_3d_test, core='M')
            test_nan_mask = np.any(~np.isfinite(X_test_stacked_c), axis=1)
            X_test_clean_c = X_test_stacked_c[~test_nan_mask]

            if len(X_train_clean_c) == 0:
                continue

            rf_c = RandomForestRegressor(
                **{k: v for k, v in bp.items() if k in RandomForestRegressor().get_params()},
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                warm_start=self.warm_start
            )

            rf_c.fit(X_train_clean_c, y_train_clean_c)
            self.rf[c] = rf_c

            if len(X_test_clean_c) > 0:
                y_pred_c = rf_c.predict(X_test_clean_c)
                result_c = np.full(len(X_test_stacked_c), np.nan)
                result_c[~test_nan_mask] = y_pred_c
                pred_c_reshaped = result_c.reshape(n_time, n_lat, n_lon)
                predictions = np.where(np.isnan(predictions), pred_c_reshaped, predictions)

        result_da = xr.DataArray(
            data=predictions,
            coords={'T': time, 'Y': lat, 'X': lon},
            dims=['T', 'Y', 'X']
        ) * mask

        result_da = reverse_standardize(result_da, Predictant_no_m, clim_year_start, clim_year_end)

        year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970
        T_value_1 = Predictant_no_m.isel(T=0).coords['T'].values
        month_1 = T_value_1.astype('datetime64[M]').astype(int) % 12 + 1
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-01")
        result_da = result_da.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        result_da['T'] = result_da['T'].astype('datetime64[ns]')

        index_start = Predictant_no_m.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant_no_m.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant_no_m.isel(T=slice(index_start, index_end))
        terciles = climatological_terciles(rainfall_for_tercile)[0]
        T1_emp = terciles.isel(quantile=0, drop=True)
        T2_emp = terciles.isel(quantile=1, drop=True)
        error_variance = ((Predictant_no_m - hindcast_det_cross) ** 2).mean(dim="T", skipna=True)
        dof = max(int(rainfall_for_tercile.sizes["T"]) - 1, 2)

        dm = self.dist_method

        if dm == "bestfit":
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError(
                    "dist_method='bestfit' requires best_code_da, best_shape_da, best_loc_da, best_scale_da."
                )
            # Category boundaries stay on the observed climatological reference.
            # The selected distribution controls the predictive CDF only.
            if getattr(self, "tercile_reference", "climatology") == "training":
                T1, T2 = _pufunc(
                    self._ppf_terciles_from_code,
                    best_code_da, best_shape_da, best_loc_da, best_scale_da,
                    input_core_dims=[(), (), (), ()],
                    output_core_dims=[(), ()],
                    vectorize=True, dask="parallelized",
                    output_dtypes=[float, float],
                )
            else:
                T1, T2 = T1_emp, T2_emp
            forecast_prob = _pufunc(
                self.calculate_tercile_probabilities_bestfit,
                result_da, error_variance, T1, T2, best_code_da,
                input_core_dims=[("T",), (), (), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True, dask="parallelized", kwargs={"dof": dof},
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True},
            )

        elif dm == "nonparam":
            # ===== 3: out-of-sample residuals + drop member dim =====
            # Rename the residual time dim so it does not collide with the
            # single forecast-year T in apply_ufunc (else alignment fails).
            error_samples = (Predictant_no_m - hindcast_det_cross).rename({"T": "S"})
            forecast_prob = _pufunc(
                self.calculate_tercile_probabilities_nonparametric,
                result_da, error_samples, T1_emp, T2_emp,
                input_core_dims=[("T",), ("S",), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True, dask="parallelized", output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True},
            )

        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}")

        forecast_prob = forecast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return result_da * mask, mask * forecast_prob.transpose('probability', 'T', 'Y', 'X')


class _XGBRegressorES(XGBRegressor):
    """
    XGBRegressor that supports tunable early stopping inside scikit-learn search.

    XGBoost's `early_stopping_rounds` needs an eval_set, which GridSearchCV /
    RandomizedSearchCV cannot supply per fold. This wrapper carves an internal
    validation split (es_validation_fraction) from the training data and passes
    it as eval_set when early_stopping_rounds is set, so early_stopping_rounds can
    be treated as a normal tunable hyperparameter. With early_stopping_rounds=None
    it behaves exactly like a plain XGBRegressor.
    """
    def __init__(self, es_validation_fraction=0.15, **kwargs):
        self.es_validation_fraction = es_validation_fraction
        super().__init__(**kwargs)

    def get_params(self, deep=True):
        params = super().get_params(deep)
        params['es_validation_fraction'] = self.es_validation_fraction
        return params

    def get_xgb_params(self):
        params = super().get_xgb_params()
        params.pop("es_validation_fraction", None)
        return params

    def fit(self, X, y, *, es_groups=None, **fit_kw):
        """Select boosting rounds on disjoint years, then refit all training rows."""
        esr = getattr(self, "early_stopping_rounds", None)
        self.fitted_sample_count_ = int(len(y))
        if not esr:
            return super().fit(X, y, **fit_kw)
        if es_groups is None:
            raise ValueError("Early stopping on pooled climate rows requires es_groups (calendar years).")
        groups = np.asarray(es_groups)
        if groups.shape != (len(y),):
            raise ValueError("es_groups must have one label per training row.")
        years = np.unique(groups)
        if len(years) < 2:
            raise ValueError("At least two training years are needed for early stopping.")
        n_valid = min(len(years)-1, max(1, int(np.ceil(len(years)*self.es_validation_fraction))))
        validation_years = years[-n_valid:]
        va = np.isin(groups, validation_years)
        tr = ~va
        if "eval_set" in fit_kw:
            raise ValueError("Provide es_groups; an external eval_set could contain the outer test year.")
        X, y = np.asarray(X), np.asarray(y)
        train_kw = dict(fit_kw)
        train_kw["verbose"] = False
        if "sample_weight" in train_kw:
            weight = np.asarray(train_kw["sample_weight"])
            train_kw["sample_weight"] = weight[tr]
            train_kw["sample_weight_eval_set"] = [weight[va]]
        super().fit(X[tr], y[tr], eval_set=[(X[va], y[va])], **train_kw)
        rounds = int(self.best_iteration) + 1
        original_rounds = self.n_estimators
        self.early_stopping_rounds = None
        self.n_estimators = rounds
        try:
            super().fit(X, y, **fit_kw)
        finally:
            self.n_estimators = original_rounds
            self.early_stopping_rounds = esr
        self.selected_n_estimators_ = rounds
        self.es_training_years_ = years[:-n_valid]
        self.es_validation_years_ = validation_years
        return self


class WAS_mme_XGBoosting:
    """
    XGBoost-based Multi-Model Ensemble (MME) forecasting for seasonal rainfall.
    Enhanced with additional regularization hyperparameters and time-series CV.
    
    Parameters
    ----------
    search_method : str, optional
        Hyperparameter optimization method: 'grid', 'random', or 'bayesian' (default: 'random').
    n_estimators_range : list of int or scipy.stats distribution, optional
        List of n_estimators values to tune (optimized for climate data).
    learning_rate_range : list of float or scipy.stats distribution, optional
        List of learning rates to tune (optimized for climate data).
    max_depth_range : list of int or scipy.stats distribution, optional
        List of max depths to tune (optimized for climate data).
    min_child_weight_range : list of float or scipy.stats distribution, optional
        List of minimum child weights to tune (optimized for climate data).
    subsample_range : list of float or scipy.stats distribution, optional
        List of subsample ratios to tune.
    colsample_bytree_range : list of float or scipy.stats distribution, optional
        List of column sampling ratios to tune.
    gamma_range : list of float or scipy.stats distribution, optional
        List of gamma (min_split_loss) values for regularization.
    reg_alpha_range : list of float or scipy.stats distribution, optional
        List of L1 regularization (alpha) values.
    reg_lambda_range : list of float or scipy.stats distribution, optional
        List of L2 regularization (lambda) values.
    random_state : int, optional
        Seed for reproducibility (default is 42).
    dist_method : str, optional
        Distribution method for tercile probabilities ('bestfit', 'nonparam', etc.) (default is 'nonparam').
    n_iter_search : int, optional
        Number of iterations for randomized/bayesian search or points to sample for grid search (default is 10).
    cv_folds : int, optional
        Number of cross-validation folds (default is 3).
    cv_method : str, optional
        Cross-validation method: 'timeseries' or 'kfold' (default: 'timeseries').
        Use 'timeseries' for temporal data to prevent data leakage.
    n_clusters : int, optional
        Number of clusters for homogenized zones (default is 4).
    optuna_n_jobs : int, optional
        Optional Optuna-specific worker count. When ``None``, it inherits
        ``n_jobs`` (default is None).
    optuna_timeout : int, optional
        Timeout in seconds for Optuna optimization (default is None).
    n_jobs : int, optional
        Maximum number of parallel workers. ``-1`` uses all available CPUs.
        Search CV/trials run in parallel at the outer level; inner XGBoost
        fits use one thread during tuning. Final zone models use ``n_jobs``
        directly (default is 4).
    """


    def __init__(self,
                 search_method='random',
                 n_estimators_range=[100, 200, 400],
                 learning_rate_range=[0.01, 0.03, 0.05],
                 max_depth_range=[2, 3, 4],
                 min_child_weight_range=[5, 10, 20],
                 subsample_range=[0.6, 0.8],
                 colsample_bytree_range=[0.6, 0.8, 1.0],
                 gamma_range=[0, 0.1, 1.0],
                 reg_alpha_range=[0.01, 0.1, 1.0],
                 reg_lambda_range=[1.0, 5.0, 10.0],
                 early_stopping_rounds_range=[None],
                 es_validation_fraction=0.15,
                 random_state=42,
                 dist_method="nonparam",
                 n_iter_search=10,
                 cv_folds=3,
                 n_clusters=4,
                 leave_one_year_out=False,          # >>> E
                 optuna_n_jobs=None,
                 optuna_timeout=None,
                 verbose=0,
                 n_jobs=-1,
                 cv_n_jobs=-1,
                 tuning_model_n_jobs=1,
                 cluster_n_jobs: Optional[int] = None,
                 spatial_subsample: int = 1,
                 scoring: str = 'neg_mean_squared_error',
                 spatial_mode: str = "zones",
                 covariate_names: Optional[Sequence[str]] = None,
                 static_fields: Optional[Dict[str, xr.DataArray]] = None,
                 coord_step: Optional[float] = None,
                 zones_as_features: bool = False,
                 predictor_smoothing: int = 0,
                 one_se_rule: bool = False,
                 amplitude_calibration=True,
                 amplitude_calibration_inner_folds: int = 3):
        # Cluster-level parallelism and optional thinning of the
        # TRAINING sample. See _cluster_plan and _stack_rows.
        self.cluster_n_jobs = cluster_n_jobs
        self.spatial_subsample = max(1, int(spatial_subsample))
        # Selection score of the strict regional tuning. XGBoost is trained on
        # the squared error (reg:squarederror) and its prediction is dressed as
        # a conditional MEAN, so the default selection score is the MSE too;
        # the historical MAE selection targeted the conditional median and
        # under-forecast the mean of skewed rainfall.
        self.scoring = scoring
        _init_spatial_options(self, spatial_mode, covariate_names, static_fields,
                              coord_step, zones_as_features, predictor_smoothing, one_se_rule,
                              amplitude_calibration, amplitude_calibration_inner_folds)

        self.search_method = search_method
        self.n_estimators_range = n_estimators_range
        self.learning_rate_range = learning_rate_range
        self.max_depth_range = max_depth_range
        self.min_child_weight_range = min_child_weight_range
        self.subsample_range = subsample_range
        self.colsample_bytree_range = colsample_bytree_range
        self.gamma_range = gamma_range
        self.reg_alpha_range = reg_alpha_range
        self.reg_lambda_range = reg_lambda_range
        # early_stopping_rounds is tuned via _XGBRegressorES (internal eval_set).
        # None disables it; integers enable early stopping on n_estimators.
        self.early_stopping_rounds_range = early_stopping_rounds_range
        self.es_validation_fraction = es_validation_fraction
        self.random_state = random_state
        self.dist_method = dist_method
        self.n_iter_search = n_iter_search
        self.cv_folds = cv_folds
        self.n_clusters = n_clusters
        self.leave_one_year_out = leave_one_year_out
        if n_jobs == 0:
            raise ValueError("n_jobs cannot be 0; use a positive integer or -1.")
        if optuna_n_jobs == 0:
            raise ValueError("optuna_n_jobs cannot be 0.")
        if cv_n_jobs == 0:
            raise ValueError("cv_n_jobs cannot be 0.")
        if tuning_model_n_jobs == 0:
            raise ValueError("tuning_model_n_jobs cannot be 0.")
        self.n_jobs = n_jobs
        self.cv_n_jobs = cv_n_jobs
        self.tuning_model_n_jobs = tuning_model_n_jobs
        self.optuna_n_jobs = 1 if optuna_n_jobs is None else optuna_n_jobs
        self.optuna_timeout = optuna_timeout
        self.verbose = verbose
        self.xgb = {}
        self.best_params_ = None
        self.cluster_da_ = None

    # ===================================================================== CV
    def _cv_splitter(self, groups):
        """
        >>> E: year-grouped CV so the same calendar year never appears on
        both sides of the split. LeaveOneGroupOut when years are few or
        requested, otherwise GroupKFold with cv_folds blocks of years.
        """
        n_groups = len(np.unique(groups))
        if self.leave_one_year_out or n_groups <= self.cv_folds:
            return LeaveOneGroupOut()
        return GroupKFold(n_splits=self.cv_folds)

    # ============================================================ param spaces
    @staticmethod
    def _suggest(trial, name, range_obj, kind='float'):
        if isinstance(range_obj, list):
            return trial.suggest_categorical(name, range_obj)
        if kind == 'int':
            return trial.suggest_int(name, int(range_obj.a), int(range_obj.b))
        if kind == 'float_log':
            return trial.suggest_float(name, range_obj.a, range_obj.b, log=True)
        return trial.suggest_float(name, range_obj.a, range_obj.b)

    def _objective(self, trial, X, y, groups):
        """Optuna objective with year-grouped CV (neg MAE)."""
        params = {
            'n_estimators': self._suggest(trial, 'n_estimators', self.n_estimators_range, 'int'),
            'learning_rate': self._suggest(trial, 'learning_rate', self.learning_rate_range, 'float_log'),
            'max_depth': self._suggest(trial, 'max_depth', self.max_depth_range, 'int'),
            'min_child_weight': self._suggest(trial, 'min_child_weight', self.min_child_weight_range),
            'subsample': self._suggest(trial, 'subsample', self.subsample_range),
            'colsample_bytree': self._suggest(trial, 'colsample_bytree', self.colsample_bytree_range),
            'gamma': self._suggest(trial, 'gamma', self.gamma_range),
            'reg_alpha': self._suggest(trial, 'reg_alpha', self.reg_alpha_range),
            'reg_lambda': self._suggest(trial, 'reg_lambda', self.reg_lambda_range),
        }
        # Optuna categoricals can't hold None -> use 0 as the "off" sentinel.
        es_choices = [0 if v is None else v for v in self.early_stopping_rounds_range]
        esr = trial.suggest_categorical('early_stopping_rounds', es_choices)
        params['early_stopping_rounds'] = None if esr == 0 else esr

        model = _XGBRegressorES(**params, es_validation_fraction=self.es_validation_fraction,
                                eval_metric='rmse', random_state=self.random_state,
                                verbosity=0, n_jobs=self.tuning_model_n_jobs)
        scores = cross_val_score(model, X, y, cv=self._cv_splitter(groups), groups=groups,
                                 scoring='neg_mean_absolute_error',
                                 params={'es_groups': groups}, n_jobs=self.cv_n_jobs)
        return float(np.mean(scores))

    def _param_space(self):
        return {
            'n_estimators': self.n_estimators_range,
            'learning_rate': self.learning_rate_range,
            'max_depth': self.max_depth_range,
            'min_child_weight': self.min_child_weight_range,
            'subsample': self.subsample_range,
            'colsample_bytree': self.colsample_bytree_range,
            'gamma': self.gamma_range,
            'reg_alpha': self.reg_alpha_range,
            'reg_lambda': self.reg_lambda_range,
            'early_stopping_rounds': self.early_stopping_rounds_range,
        }

    def _grid_space(self):
        """Materialize a grid; sample any scipy distribution to a small set."""
        grid = {}
        for name, r in self._param_space().items():
            if isinstance(r, list):
                grid[name] = r
            else:
                s = r.rvs(size=min(4, self.n_iter_search), random_state=self.random_state)
                grid[name] = np.unique(s.astype(float) if 'int' in name else s)
        return grid

    def _grid_search(self, X, y, groups):
        model = _XGBRegressorES(es_validation_fraction=self.es_validation_fraction,
                                eval_metric='rmse', random_state=self.random_state,
                                verbosity=0, n_jobs=self.tuning_model_n_jobs)
        gs = GridSearchCV(model, param_grid=self._grid_space(),
                          cv=self._cv_splitter(groups), scoring='neg_mean_absolute_error',
                          error_score=np.nan, n_jobs=self.n_jobs)
        gs.fit(X, y, groups=groups, es_groups=groups)          # >>> E
        _report_best_params("xgb/grid", gs.best_params_,
                            getattr(self, "verbose", 1), gs.best_score_)
        return gs.best_params_

    def _random_search(self, X, y, groups):
        model = _XGBRegressorES(es_validation_fraction=self.es_validation_fraction,
                                eval_metric='rmse', random_state=self.random_state,
                                verbosity=0, n_jobs=self.tuning_model_n_jobs)
        rs = RandomizedSearchCV(model, param_distributions=self._param_space(),
                                n_iter=self.n_iter_search, cv=self._cv_splitter(groups),
                                scoring='neg_mean_absolute_error', random_state=self.random_state,
                                error_score=np.nan, n_jobs=self.n_jobs)
        rs.fit(X, y, groups=groups, es_groups=groups)          # >>> E
        _report_best_params("xgb/random", rs.best_params_,
                            getattr(self, "verbose", 1), rs.best_score_)
        return rs.best_params_

    def _bayesian_search(self, X, y, groups):
        if optuna is None:
            raise ImportError("search_method='bayesian' requires optuna to be installed.")
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        )
        study.optimize(lambda tr: self._objective(tr, X, y, groups),
                       n_trials=self.n_iter_search, timeout=self.optuna_timeout,
                       n_jobs=self.optuna_n_jobs)
        _report_best_params("xgb/bayesian", study.best_params,
                            getattr(self, "verbose", 1), study.best_value)
        return study.best_params

    # ====================================================== hyperparameters
    def compute_hyperparameters(self, Predictors, Predictand,
                                clim_year_start=None, clim_year_end=None):
        """Tune with fold-local standardization, target encoding and climate zones."""
        return _strict_regional_hyperparameters(
            self, Predictors, Predictand, clim_year_start, clim_year_end)

    def compute_hyperparameters_legacy(self, Predictors, Predictand,
                                clim_year_start=None, clim_year_end=None):
        """
        Best hyperparameters per zone.

        Regionalization: KMeans on the predictand VALUES (kept by preference).
        Standardization is applied here only if clim years are given; when None,
        the inputs are assumed already standardized (framework path).
        """
        Predictors = align_predictor_target(Predictors, Predictand, mode="season_year")
        if "M" in Predictand.dims:
            Predictand = Predictand.isel(M=0, drop=True)
        Predictand.name = "varname"

        if clim_year_start is not None and clim_year_end is not None:
            X_train_std = standardize_timeseries(Predictors, clim_year_start, clim_year_end)
            y_train_std = standardize_timeseries(Predictand, clim_year_start, clim_year_end)
        else:
            X_train_std = Predictors
            y_train_std = Predictand
        X_train_std['T'] = y_train_std['T']

        cluster_da = _build_climate_feature_clusters(
            Predictand, self.n_clusters, random_state=self.random_state
        )
        clusters = np.unique(cluster_da.values)
        clusters = clusters[np.isfinite(clusters)]

        best_params_dict = {}
        for c in clusters:
            c_int = int(c)
            mask_3d = (cluster_da == c).expand_dims({'T': y_train_std['T']})
            Xc = X_train_std.where(mask_3d).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M')
            yc = y_train_std.where(mask_3d).stack(sample=('T', 'Y', 'X'))

            groups_full = Xc['T'].values.astype('datetime64[Y]').astype(int)   # >>> E
            Xv, yv = Xc.values, yc.values.ravel()
            bad = np.any(~np.isfinite(Xv), axis=1) | ~np.isfinite(yv)
            X_clean, y_clean, groups_c = Xv[~bad], yv[~bad], groups_full[~bad]

            n_years = len(np.unique(groups_c))
            _effective_sample_warning(n_years, X_clean.shape[1], "WAS_mme_XGBoosting")
            if len(X_clean) < self.cv_folds * 2 or n_years < 2:
                if self.verbose:
                    print(f"Cluster {c_int}: insufficient data "
                          f"({len(X_clean)} samples, {n_years} years). Skipping.")
                continue

            if self.search_method == 'grid':
                best_params_dict[c_int] = self._grid_search(X_clean, y_clean, groups_c)
            elif self.search_method == 'random':
                best_params_dict[c_int] = self._random_search(X_clean, y_clean, groups_c)
            elif self.search_method == 'bayesian':
                best_params_dict[c_int] = self._bayesian_search(X_clean, y_clean, groups_c)
            else:
                raise ValueError(f"Unknown search_method: {self.search_method}.")

        self.best_params_ = best_params_dict
        self.cluster_da_ = cluster_da
        _report_best_params("xgb", best_params_dict, getattr(self, "verbose", 1))
        return best_params_dict, cluster_da

    # ============================================================ deterministic
    @staticmethod
    def _base_params(bp, random_state, es_validation_fraction=0.15, n_jobs=1):
        base = {
            'n_estimators': bp.get('n_estimators', 200),
            'learning_rate': bp.get('learning_rate', 0.05),
            'max_depth': bp.get('max_depth', 5),
            'min_child_weight': bp.get('min_child_weight', 1),
            'subsample': bp.get('subsample', 1.0),
            'colsample_bytree': bp.get('colsample_bytree', 1.0),
            'early_stopping_rounds': bp.get('early_stopping_rounds', None),
            'es_validation_fraction': es_validation_fraction,
            'eval_metric': 'rmse',
            'random_state': random_state, 'verbosity': 0, 'n_jobs': n_jobs,
        }
        for k in ('gamma', 'reg_alpha', 'reg_lambda'):
            if k in bp:
                base[k] = bp[k]
        return base

    def compute_model(self, X_train, y_train, X_test, y_test=None,
                      clim_year_start=None, clim_year_end=None,
                      best_params=None, cluster_da=None, covariates=None):
        """
        Deterministic hindcast with one XGBRegressor per zone (or one global
        model plus spatial covariates when ``spatial_mode`` is "soft").

        >>> A: clim years are now in the signature and tuning runs in dual
        mode when best_params/cluster_da are not supplied.
        """
        validate_mme_predictor_schema(X_train, X_test)
        X_train = align_predictor_target(X_train, y_train, mode="season_year")
        best_params, cluster_da = _resolve_regional_params(
            self, X_train, y_train, clim_year_start, clim_year_end, best_params, cluster_da)
        _raw_predictors = (X_train, X_test)   # avant covariables : pour la calibration
        X_train, X_test = _prepare_channels(self, X_train, X_test, y_train, covariates,
                                            clim_year_start, clim_year_end)

        time, lat, lon = X_test['T'], X_test['Y'], X_test['X']
        n_time, n_lat, n_lon = len(time), len(lat), len(lon)
        predictions = np.full((n_time * n_lat * n_lon,), np.nan)
        self.xgb = {}

        # Stack once per fold, then index rows per cluster. The previous loop
        # re-materialised the full (T, M, Y, X) array for every cluster and
        # discarded ~4/5 of the rows: 2.91 s against 0.20 s on a 350x171 grid.
        train_rows = _cluster_row_index(cluster_da, X_train.sizes['T'])
        test_rows = _cluster_row_index(cluster_da, X_test.sizes['T'])
        active = [c for c in range(self.n_clusters) if c in best_params]
        # XGBoost genuinely uses threads for histogram building, so the budget
        # left over by the cluster workers is handed to the model.
        workers, threads = _cluster_plan(
            self.n_clusters, self.n_jobs,
            getattr(self, 'cluster_n_jobs', None), model_is_threaded=True,
        )

        def _fit_one(c):
            bp = best_params[c]
            train_sel = _thin_training_rows(train_rows == c, X_train.sizes['T'], n_lat, n_lon,
                                            getattr(self, 'spatial_subsample', 1))

            Xtr, _ = _stack_rows(X_train, train_sel, core='M')
            ytr, _ = _stack_rows(y_train, train_sel, core=None)
            ytr = ytr.ravel()
            keep = np.all(np.isfinite(Xtr), axis=1) & np.isfinite(ytr)
            if not keep.any():
                return c, None, None, None

            Xte, pos = _stack_rows(X_test, test_rows == c, core='M')
            keep_te = np.all(np.isfinite(Xte), axis=1)

            xgb_c = _XGBRegressorES(**self._base_params(bp, self.random_state,
                                                        self.es_validation_fraction,
                                                        n_jobs=threads))
            years = np.repeat(_calendar_year_groups(X_train["T"]), X_train.sizes["Y"]*X_train.sizes["X"])
            xgb_c.fit(Xtr[keep], ytr[keep], es_groups=years[train_sel][keep])
            if not keep_te.any():
                return c, xgb_c, None, None
            return c, xgb_c, pos[keep_te], xgb_c.predict(Xte[keep_te])

        if workers > 1 and len(active) > 1:
            results = Parallel(n_jobs=min(workers, len(active)), backend="threading")(
                delayed(_fit_one)(c) for c in active
            )
        else:
            results = [_fit_one(c) for c in active]

        for c, model, pos, vals in results:
            if model is not None:
                self.xgb[c] = model
            if pos is not None:
                predictions[pos] = vals
        predictions = predictions.reshape(n_time, n_lat, n_lon)

        predicted_da = xr.DataArray(predictions, coords={'T': time, 'Y': lat, 'X': lon},
                                    dims=['T', 'Y', 'X'])
        return _maybe_calibrate_amplitude(
            self, self.xgb, X_train, y_train, predicted_da, cluster_da, X_test=X_test,
            raw_predictors=_raw_predictors, best_params=best_params,
            clim_year_start=clim_year_start, clim_year_end=clim_year_end)

    # ------------------ Probability Calculation Methods ------------------
    @staticmethod
    def _ppf_terciles_from_code(dist_code, shape, loc, scale):
        """Return tercile thresholds (T1, T2) from best-fit distribution params."""
        if np.isnan(dist_code):
            return np.nan, np.nan
        code = int(dist_code)
        try:
            if code == 1:
                return (norm.ppf(1/3, loc=loc, scale=scale),
                        norm.ppf(2/3, loc=loc, scale=scale))
            elif code == 2:
                return (lognorm.ppf(1/3, s=shape, loc=loc, scale=scale),
                        lognorm.ppf(2/3, s=shape, loc=loc, scale=scale))
            elif code == 3:
                return (expon.ppf(1/3, loc=loc, scale=scale),
                        expon.ppf(2/3, loc=loc, scale=scale))
            elif code == 4:
                return (gamma.ppf(1/3, a=shape, loc=loc, scale=scale),
                        gamma.ppf(2/3, a=shape, loc=loc, scale=scale))
            elif code == 5:
                return (weibull_min.ppf(1/3, c=shape, loc=loc, scale=scale),
                        weibull_min.ppf(2/3, c=shape, loc=loc, scale=scale))
            elif code == 6:
                return (t.ppf(1/3, df=shape, loc=loc, scale=scale),
                        t.ppf(2/3, df=shape, loc=loc, scale=scale))
            elif code == 7:
                return (poisson.ppf(1/3, mu=shape, loc=loc),
                        poisson.ppf(2/3, mu=shape, loc=loc))
            elif code == 8:
                return (nbinom.ppf(1/3, n=shape, p=scale, loc=loc),
                        nbinom.ppf(2/3, n=shape, p=scale, loc=loc))
        except Exception:
            return np.nan, np.nan
        return np.nan, np.nan

    @staticmethod
    def weibull_shape_solver(k, M, V):
        """Root function for the Weibull shape parameter k (match V/M^2)."""
        if k <= 0:
            return -np.inf
        try:
            g1 = gamma_function(1 + 1/k)
            g2 = gamma_function(1 + 2/k)
            implied = (g2 / (g1**2)) - 1
            observed = V / (M**2)
            return observed - implied
        except ValueError:
            return -np.inf

    @staticmethod
    def calculate_tercile_probabilities_bestfit(best_guess, error_variance, T1, T2, dist_code, dof):
        """Generic tercile probabilities using best-fit family per grid cell."""
        best_guess = np.asarray(best_guess, float)
        error_variance = np.asarray(error_variance, dtype=float)
        n_time = best_guess.size
        out = np.full((3, n_time), np.nan, float)

        if (np.all(np.isnan(best_guess)) or np.isnan(dist_code)
                or np.isnan(T1) or np.isnan(T2) or np.isnan(error_variance)):
            return out

        code = int(dist_code)

        if code == 1:  # Normal
            sd = np.sqrt(error_variance)
            out[0, :] = norm.cdf(T1, loc=best_guess, scale=sd)
            out[1, :] = norm.cdf(T2, loc=best_guess, scale=sd) - norm.cdf(T1, loc=best_guess, scale=sd)
            out[2, :] = 1 - norm.cdf(T2, loc=best_guess, scale=sd)

        elif code == 2:  # Lognormal
            sigma = np.sqrt(np.log(1 + error_variance / (best_guess**2)))
            mu = np.log(best_guess) - sigma**2 / 2
            out[0, :] = lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
            out[1, :] = lognorm.cdf(T2, s=sigma, scale=np.exp(mu)) - lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
            out[2, :] = 1 - lognorm.cdf(T2, s=sigma, scale=np.exp(mu))

        elif code == 3:  # Exponential
            sc = np.sqrt(error_variance)
            c1 = expon.cdf(T1, loc=best_guess, scale=sc)
            c2 = expon.cdf(T2, loc=best_guess, scale=sc)     # >>> B: was loc_t
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        elif code == 4:  # Gamma
            alpha = (best_guess ** 2) / error_variance
            theta = error_variance / best_guess
            c1 = gamma.cdf(T1, a=alpha, scale=theta)
            c2 = gamma.cdf(T2, a=alpha, scale=theta)
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        elif code == 5:  # Weibull
            for i in range(n_time):
                M, V = best_guess[i], error_variance
                if V <= 0 or M <= 0:
                    out[:, i] = np.nan
                    continue
                k = fsolve(WAS_mme_XGBoosting.weibull_shape_solver, 2.0, args=(M, V))[0]  # >>> C
                if k <= 0:
                    out[:, i] = np.nan
                    continue
                lam = M / gamma_function(1 + 1/k)
                c1 = weibull_min.cdf(T1, c=k, loc=0, scale=lam)
                c2 = weibull_min.cdf(T2, c=k, loc=0, scale=lam)
                out[0, i] = c1
                out[1, i] = c2 - c1
                out[2, i] = 1.0 - c2

        elif code == 6:  # Student-t
            if dof <= 2:
                out[:, :] = np.nan
            else:
                scale = np.sqrt(error_variance * (dof - 2) / dof)
                c1 = t.cdf(T1, df=dof, loc=best_guess, scale=scale)
                c2 = t.cdf(T2, df=dof, loc=best_guess, scale=scale)
                out[0, :] = c1
                out[1, :] = c2 - c1
                out[2, :] = 1.0 - c2

        elif code == 7:  # Poisson
            mu = best_guess
            c1 = poisson.cdf(T1, mu=mu)
            c2 = poisson.cdf(T2, mu=mu)
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        elif code == 8:  # Negative Binomial
            p = np.where(error_variance > best_guess, best_guess / error_variance, np.nan)
            n = np.where(error_variance > best_guess, (best_guess**2) / (error_variance - best_guess), np.nan)
            c1 = nbinom.cdf(T1, n=n, p=p)
            c2 = nbinom.cdf(T2, n=n, p=p)
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        else:
            raise ValueError(f"Invalid distribution code: {dist_code}")

        # >>> F: positive-support families are undefined for non-positive means
        if code in (2, 3, 4, 5, 7, 8):
            bad = ~(best_guess > 0)
            if np.any(bad):
                out[:, bad] = np.nan

        return out

    @staticmethod
    def calculate_tercile_probabilities_nonparametric(best_guess, error_samples, first_tercile, second_tercile):
        return WAS_mme_Weighted.calculate_tercile_probabilities_nonparametric(
            best_guess, error_samples, first_tercile, second_tercile)

    def compute_prob(self, Predictant: xr.DataArray, clim_year_start, clim_year_end,
                     hindcast_det: xr.DataArray, best_code_da=None, best_shape_da=None,
                     best_loc_da=None, best_scale_da=None) -> xr.DataArray:
        """
        Tercile probabilities for deterministic hindcasts.

        For honest calibration pass the CROSS-VALIDATED hindcast as hindcast_det
        (out-of-sample residuals); the in-sample hindcast yields over-confident
        probabilities.
        """
        if "M" in Predictant.dims:
            Predictant = Predictant.isel(M=0, drop=True)
        Predictant = Predictant.transpose("T", "Y", "X")
        mask = xr.where(np.isfinite(Predictant).sum(dim='T') >= 3, 1.0, np.nan)

        clim = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))
        if clim.sizes.get("T", 0) < 3:
            raise ValueError("Not enough years in climatology period for terciles.")

        error_variance = ((Predictant - hindcast_det) ** 2).mean(dim="T", skipna=True)
        dof = max(int(clim.sizes["T"]) - 1, 2)

        terciles_emp = climatological_terciles(clim)[0]        # >>> 1/3, 2/3
        T1_emp = terciles_emp.isel(quantile=0, drop=True)
        T2_emp = terciles_emp.isel(quantile=1, drop=True)

        dm = self.dist_method
        if dm == "bestfit":
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError("dist_method='bestfit' requires best_code_da, best_shape_da, "
                                 "best_loc_da, best_scale_da.")
            # Category boundaries stay on the observed climatological reference.
            # The selected distribution controls the predictive CDF only.
            if getattr(self, "tercile_reference", "climatology") == "training":
                T1, T2 = _pufunc(
                    self._ppf_terciles_from_code,
                    best_code_da, best_shape_da, best_loc_da, best_scale_da,
                    input_core_dims=[(), (), (), ()], output_core_dims=[(), ()],
                    vectorize=True, dask="parallelized", output_dtypes=[float, float])
            else:
                T1, T2 = T1_emp, T2_emp
            hindcast_prob = _pufunc(
                self.calculate_tercile_probabilities_bestfit,
                hindcast_det, error_variance, T1, T2, best_code_da,
                input_core_dims=[("T",), (), (), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True, kwargs={'dof': dof}, dask="parallelized", output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True})

        elif dm == "nonparam":
            error_samples = Predictant - hindcast_det
            hindcast_prob = _pufunc(
                self.calculate_tercile_probabilities_nonparametric,
                hindcast_det, error_samples, T1_emp, T2_emp,
                input_core_dims=[("T",), ("T",), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True, dask="parallelized", output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True})
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}")

        hindcast_prob = hindcast_prob.assign_coords(probability=("probability", ["PB", "PN", "PA"]))
        return (hindcast_prob * mask).transpose("probability", "T", "Y", "X")

    # ============================================================== operational
    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det,
                 hindcast_det_cross, Predictor_for_year, best_params=None, cluster_da=None,
                 best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None):
        """Operational forecast for a target year (deterministic + tercile probabilities)."""
        validate_mme_predictor_schema(hindcast_det, Predictor_for_year)
        if "M" in Predictant.dims:
            Predictant_no_m = Predictant.isel(M=0, drop=True)
        else:
            Predictant_no_m = Predictant
        mask = xr.where(np.isfinite(Predictant_no_m).sum(dim='T') >= 3, 1.0, np.nan).to_numpy()

        hindcast_det_st, Predictor_for_year_st = standardize_mme_predictors(
            hindcast_det, Predictor_for_year, clim_year_start, clim_year_end)
        Predictant_st = standardize_timeseries(Predictant_no_m, clim_year_start, clim_year_end)
        hindcast_det_st['T'] = Predictant_st['T']
        hindcast_det_cross = hindcast_det_cross.assign_coords(T=Predictant_st['T'])

        time, lat, lon = Predictor_for_year_st['T'], Predictor_for_year_st['Y'], Predictor_for_year_st['X']
        n_time, n_lat, n_lon = len(time), len(lat), len(lon)

        best_params, cluster_da = _resolve_regional_params(
            self, hindcast_det, Predictant_no_m, clim_year_start, clim_year_end,
            best_params, cluster_da)
        hindcast_det_st, Predictor_for_year_st = _prepare_channels(
            self, hindcast_det_st, Predictor_for_year_st, Predictant_no_m, None,
            clim_year_start, clim_year_end)

        predictions = np.full((n_time, n_lat, n_lon), np.nan)
        self.xgb = {}

        for c in range(self.n_clusters):
            if c not in best_params:
                continue
            bp = best_params[c]
            m_tr = (cluster_da == c).expand_dims({'T': hindcast_det_st['T']})
            m_te = (cluster_da == c).expand_dims({'T': Predictor_for_year_st['T']})

            Xtr = hindcast_det_st.where(m_tr).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            ytr = Predictant_st.where(m_tr).stack(sample=('T', 'Y', 'X')).values.ravel()
            tr_bad = np.any(~np.isfinite(Xtr), axis=1) | ~np.isfinite(ytr)
            Xtr_c, ytr_c = Xtr[~tr_bad], ytr[~tr_bad]

            Xte = Predictor_for_year_st.where(m_te).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            te_bad = np.any(~np.isfinite(Xte), axis=1)
            Xte_c = Xte[~te_bad]

            if len(Xtr_c) == 0:
                continue

            xgb_c = _XGBRegressorES(**self._base_params(bp, self.random_state,
                                                        self.es_validation_fraction,
                                                        n_jobs=self.n_jobs))
            years = np.repeat(_calendar_year_groups(hindcast_det_st["T"]), hindcast_det_st.sizes["Y"]*hindcast_det_st.sizes["X"])
            xgb_c.fit(Xtr_c, ytr_c, es_groups=years[~tr_bad])
            self.xgb[c] = xgb_c

            if len(Xte_c) > 0:
                y_pred = xgb_c.predict(Xte_c)
                result = np.full(Xte.shape[0], np.nan)
                result[~te_bad] = y_pred
                predictions = np.where(np.isnan(predictions),
                                       result.reshape(n_time, n_lat, n_lon), predictions)

        result_da = xr.DataArray(predictions, coords={'T': time, 'Y': lat, 'X': lon},
                                 dims=['T', 'Y', 'X']) * mask
        result_da = reverse_standardize(result_da, Predictant_no_m, clim_year_start, clim_year_end)

        year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970
        month = Predictant_no_m.isel(T=0).coords['T'].values.astype('datetime64[M]').astype(int) % 12 + 1
        new_T = np.datetime64(f"{year}-{month:02d}-01")
        result_da = result_da.assign_coords(T=xr.DataArray([new_T], dims=["T"]))
        result_da['T'] = result_da['T'].astype('datetime64[ns]')

        index_start = Predictant_no_m.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant_no_m.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant_no_m.isel(T=slice(index_start, index_end))
        terciles = climatological_terciles(rainfall_for_tercile)[0]      # >>> 1/3, 2/3
        T1_emp = terciles.isel(quantile=0, drop=True)
        T2_emp = terciles.isel(quantile=1, drop=True)
        error_variance = ((Predictant_no_m - hindcast_det_cross) ** 2).mean(dim="T", skipna=True)
        dof = max(int(rainfall_for_tercile.sizes["T"]) - 1, 2)

        dm = self.dist_method
        if dm == "bestfit":
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError("dist_method='bestfit' requires best_code_da, best_shape_da, "
                                 "best_loc_da, best_scale_da.")
            # Category boundaries stay on the observed climatological reference.
            # The selected distribution controls the predictive CDF only.
            if getattr(self, "tercile_reference", "climatology") == "training":
                T1, T2 = _pufunc(
                    self._ppf_terciles_from_code,
                    best_code_da, best_shape_da, best_loc_da, best_scale_da,
                    input_core_dims=[(), (), (), ()], output_core_dims=[(), ()],
                    vectorize=True, dask="parallelized", output_dtypes=[float, float])
            else:
                T1, T2 = T1_emp, T2_emp
            forecast_prob = _pufunc(
                self.calculate_tercile_probabilities_bestfit,
                result_da, error_variance, T1, T2, best_code_da,
                input_core_dims=[("T",), (), (), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True, dask="parallelized", kwargs={"dof": dof}, output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True})

        elif dm == "nonparam":
            # >>> D: out-of-sample residuals on the de-M'd predictand.
            # Rename the residual time dim so it does not collide with the
            # single forecast-year T in apply_ufunc (else alignment fails).
            error_samples = (Predictant_no_m - hindcast_det_cross).rename({"T": "S"})
            forecast_prob = _pufunc(
                self.calculate_tercile_probabilities_nonparametric,
                result_da, error_samples, T1_emp, T2_emp,
                input_core_dims=[("T",), ("S",), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True, dask="parallelized", output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True})
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}")

        forecast_prob = forecast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return result_da * mask, mask * forecast_prob.transpose('probability', 'T', 'Y', 'X')



def _hpelm_effective_norm(norm, n_rows, scaling="absolute"):
    """Ridge term added to diag(F'F); see :func:`wass2s.was_elm.effective_norm`.

    ``scaling="absolute"`` is hpelm's historical meaning (``norm`` as is,
    inert on ~1e5-1e6 pooled rows); ``"per_sample"`` is a ridge PER ROW
    (lambda_eff = norm * n_rows), transferable across sample sizes.
    """
    return _elm_effective_norm(norm, n_rows, scaling)


def _hpelm_random_layer(rng, X, neurons, activation, scaling="absolute"):
    """Random hidden layer (kept for callers of the historical helper).

    ``rng`` may be a seed or a ``RandomState`` (an integer seed is then drawn
    from it). The layer is nested in ``neurons``; see
    :func:`wass2s.was_elm.random_layer`.
    """
    seed = int(rng.randint(0, 2 ** 31 - 1)) if hasattr(rng, "randint") else int(rng)
    return _elm_random_layer(seed, X, neurons, activation, scaling)


def _elm_structure(model):
    """``linear_part`` / ``n_seeds`` of a WAS_mme_hpELM (hpelm's model for old pickles)."""
    return dict(linear_part=bool(getattr(model, 'linear_part', False)),
                n_seeds=int(getattr(model, 'n_seeds', 1)))


class HPELMWrapper(ELMRegressor):
    """Historical name of the ELM regressor (hpelm's model: no linear part, one draw).

    Identical predictions to ``hpelm.HPELM`` for the same random layer and
    ``norm`` (``tests/test_elm_gram.py``), without the hpelm dependency.
    """

    def __init__(self, neurons=10, activation='sigm', norm=1.0, random_state=42,
                 norm_scaling='absolute', linear_part=False, n_seeds=1):
        super().__init__(neurons=neurons, activation=activation, norm=norm,
                         random_state=random_state, norm_scaling=norm_scaling,
                         linear_part=linear_part, n_seeds=n_seeds)


class WAS_mme_hpELM:
    """
    Extreme Learning Machine (ELM) Multi-Model Ensemble.

    The ELM is computed by Gram matrices in numpy (:mod:`wass2s.was_elm`); the
    ``hpelm`` package is no longer used (the class keeps its name). A network
    is a ridge regression on ``F = [1 | X | g(X W + B)]``: an unpenalised
    intercept, the input channels (``linear_part=True``) and ``neurons``
    random units. With ``neurons=0`` it IS a ridge regression, so the tuning
    grid contains its own baseline. Predictions are averaged over ``n_seeds``
    random layers. The random layer is nested in ``neurons`` and the ridge only
    moves the diagonal, so the strict regional tuning evaluates the whole
    ``(activation, neurons, norm)`` grid from one pass over each fold
    (``ELMGramBank``) instead of one fit per candidate.

    Parameters
    ----------
    neurons_range : list of int, optional
        Numbers of random units tuned (default [0, 50, 100, 300]; 0 = ridge).
    activation_options : list of str, optional
        Among 'tanh', 'sigm', 'lin', 'rbf_l1', 'rbf_l2', 'rbf_linf' (default ['tanh']).
    norm_range : list of float, optional
        Ridge per row with ``norm_scaling='per_sample'`` (default
        [1e-2, 0.1, 1, 10, 100, 1000]; on the MJJ 0.1 deg evaluation the
        one-SE rule selected 1000 in every fold).
    linear_part : bool, optional
        Include the intercept and the input channels in the regression
        (default True). False is hpelm's historical model.
    n_seeds : int, optional
        Random layers averaged (default 3; 1 is hpelm's behaviour).
    random_state : int, optional
        Seed for reproducibility (default is 42).
    dist_method : str, optional
        Distribution method for tercile probabilities.
    search_method : str, optional
        Hyperparameter optimization method: 'grid', 'random', or 'bayesian' (default: 'random').
    n_iter_search : int, optional
        Number of iterations for randomized search (default is 10).
    cv_folds : int, optional
        Number of cross-validation folds (default is 3).
    n_clusters : int, optional
        Number of clusters for homogenized zones (default is 4).
    n_trials_bayesian : int, optional
        Number of trials for Bayesian optimization (default=50).
    bayesian_sampler : str, optional
        Sampler for Bayesian optimization: 'tpe' or 'random' (default='tpe').
    scoring : str, optional
        Scoring metric for optimization (default='neg_mean_squared_error').
    n_jobs : int, optional
        Maximum number of parallel workers for grid/random search and Optuna.
        ``-1`` uses all available CPUs. Inner CV is serial during parallel
        Optuna trials to prevent nested oversubscription (default is 4).
    """


    def __init__(self,
                 neurons_range=[0, 50, 100, 300],
                 activation_options=['tanh'],
                 norm_range=None,
                 random_state=42,
                 dist_method="nonparam",
                 search_method='random',
                 n_iter_search=100,
                 cv_folds=3,
                 n_clusters=4,
                 leave_one_year_out=False,          # >>> E
                 n_trials_bayesian=50,
                 bayesian_sampler='tpe',
                 scoring='neg_mean_squared_error',
                 verbose=0,
                 n_jobs=-1,
                 optuna_n_jobs=None,
                 cv_n_jobs=-1,
                 cluster_n_jobs: Optional[int] = None,
                 spatial_subsample: int = 1,
                 norm_scaling: str = 'per_sample',
                 spatial_mode: str = "zones",
                 covariate_names: Optional[Sequence[str]] = None,
                 static_fields: Optional[Dict[str, xr.DataArray]] = None,
                 coord_step: Optional[float] = None,
                 zones_as_features: bool = False,
                 predictor_smoothing: int = 0,
                 one_se_rule: bool = False,
                 amplitude_calibration=True,
                 amplitude_calibration_inner_folds: int = 3,
                 linear_part: bool = True,
                 n_seeds: int = 3):
        # Cluster-level parallelism and optional thinning of the
        # TRAINING sample. See _cluster_plan and _stack_rows.
        self.cluster_n_jobs = cluster_n_jobs
        self.spatial_subsample = max(1, int(spatial_subsample))
        _init_spatial_options(self, spatial_mode, covariate_names, static_fields,
                              coord_step, zones_as_features, predictor_smoothing, one_se_rule,
                              amplitude_calibration, amplitude_calibration_inner_folds)
        self.linear_part = bool(linear_part)
        self.n_seeds = max(1, int(n_seeds))
        bad = [a for a in activation_options if a not in ('lin', 'tanh', 'sigm', 'rbf_l1', 'rbf_l2', 'rbf_linf')]
        if bad:
            raise ValueError(f"Unknown ELM activations {bad}.")
        if not self.linear_part and any(int(n) <= 0 for n in (neurons_range if not hasattr(neurons_range, 'support') else [1])):
            raise ValueError("neurons=0 needs linear_part=True (it is then a ridge regression).")
        # norm_scaling: 'per_sample' (ridge per row, default; see
        # _hpelm_effective_norm) or 'absolute' (historical hpelm meaning).
        if str(norm_scaling) not in ('per_sample', 'absolute'):
            raise ValueError("norm_scaling must be 'per_sample' or 'absolute'.")
        self.norm_scaling = str(norm_scaling)
        if norm_range is None:
            norm_range = ([1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0] if self.norm_scaling == 'per_sample'
                          else [1.0, 10.0, 100.0, 1000.0])
        self.neurons_range = neurons_range
        self.activation_options = activation_options
        self.norm_range = norm_range
        if hasattr(norm_range, "support"):
            low, high = norm_range.support()
            if not np.isfinite(low) or low <= 0 or not np.isfinite(high) or high <= 0:
                raise ValueError("norm_range distribution must have strictly positive finite support.")
        else:
            norm_values = np.asarray(list(norm_range), dtype=float)
            if norm_values.size == 0 or np.any(~np.isfinite(norm_values)) or np.any(norm_values <= 0):
                raise ValueError("Every norm_range value must be finite and strictly positive.")
        self.random_state = random_state
        self.dist_method = dist_method
        self.search_method = search_method
        self.n_iter_search = n_iter_search
        self.cv_folds = cv_folds
        self.n_clusters = n_clusters
        self.leave_one_year_out = leave_one_year_out
        self.n_trials_bayesian = n_trials_bayesian
        self.bayesian_sampler = bayesian_sampler
        self.scoring = scoring
        self.verbose = verbose
        if n_jobs == 0:
            raise ValueError("n_jobs cannot be 0; use a positive integer or -1.")
        if optuna_n_jobs == 0:
            raise ValueError("optuna_n_jobs cannot be 0.")
        if cv_n_jobs == 0:
            raise ValueError("cv_n_jobs cannot be 0.")
        self.n_jobs = n_jobs
        self.optuna_n_jobs = 1 if optuna_n_jobs is None else optuna_n_jobs
        self.cv_n_jobs = cv_n_jobs
        self.hpelm = None
        self.best_params_dict = None
        self.bayesian_studies = {}

        valid_methods = ['grid', 'random', 'bayesian']
        if self.search_method not in valid_methods:
            raise ValueError(f"search_method must be one of {valid_methods}, got '{self.search_method}'")

    # ===================================================================== CV
    def _cv_splitter(self, groups):
        """>>> E: year-grouped CV so the same year never straddles a split."""
        n_groups = len(np.unique(groups))
        if self.leave_one_year_out or n_groups <= self.cv_folds:
            return LeaveOneGroupOut()
        return GroupKFold(n_splits=self.cv_folds)

    # ============================================================ param spaces
    @staticmethod
    def _get_bounds(param_range):
        """(min, max, is_distribution) from a list or a scipy distribution."""
        if hasattr(param_range, 'support'):
            low, high = param_range.support()
            return float(low), float(high), True
        return float(min(param_range)), float(max(param_range)), False

    def _create_bayesian_sampler(self):
        if optuna is None:
            return None
        if self.bayesian_sampler == 'random':
            return RandomSampler(seed=self.random_state)
        return TPESampler(seed=self.random_state)

    def _bayesian_objective(self, trial, X, y, groups):
        low_n, high_n, is_dist_n = self._get_bounds(self.neurons_range)
        if is_dist_n:
            neurons = trial.suggest_int('neurons', int(low_n), int(high_n))
        else:
            neurons = trial.suggest_categorical('neurons', [int(n) for n in self.neurons_range])
        activation = trial.suggest_categorical('activation', self.activation_options)
        low_f, high_f, _ = self._get_bounds(self.norm_range)
        norm_ = trial.suggest_float('norm', low_f, high_f, log=True)

        model = HPELMWrapper(neurons=neurons, activation=activation, norm=norm_,
                             random_state=self.random_state,
                             norm_scaling=getattr(self, 'norm_scaling', 'absolute'),
                             **_elm_structure(self))
        scores = cross_val_score(model, X, y, cv=self._cv_splitter(groups), groups=groups,
                                 scoring=self.scoring, n_jobs=self.cv_n_jobs)
        return float(np.mean(scores))

    def _grid_search_optimization(self, X, y, groups):
        """Grid search; sample any scipy distribution to a small grid."""
        param_grid = {'activation': self.activation_options}
        for name, p_range in zip(['neurons', 'norm'], [self.neurons_range, self.norm_range]):
            if hasattr(p_range, 'support'):
                low, high = p_range.support()
                if name == 'norm':
                    param_grid[name] = np.logspace(np.log10(low), np.log10(high), 5).tolist()
                else:
                    param_grid[name] = np.linspace(low, high, 5).astype(int).tolist()
            else:
                param_grid[name] = list(p_range)
        gs = GridSearchCV(HPELMWrapper(random_state=self.random_state,
                                       norm_scaling=getattr(self, 'norm_scaling', 'absolute'),
                                       **_elm_structure(self)),
                          param_grid=param_grid,
                          cv=self._cv_splitter(groups), scoring=self.scoring,
                          n_jobs=self.n_jobs)
        gs.fit(X, y, groups=groups)        # >>> E
        return gs.best_params_

    def _random_search_optimization(self, X, y, groups):
        param_dist = {'neurons': self.neurons_range, 'activation': self.activation_options,
                      'norm': self.norm_range}
        rs = RandomizedSearchCV(HPELMWrapper(random_state=self.random_state,
                                             norm_scaling=getattr(self, 'norm_scaling', 'absolute'),
                                             **_elm_structure(self)),
                                param_distributions=param_dist, n_iter=self.n_iter_search,
                                cv=self._cv_splitter(groups), scoring=self.scoring,
                                random_state=self.random_state, n_jobs=self.n_jobs)
        rs.fit(X, y, groups=groups)        # >>> E
        return rs.best_params_

    def _bayesian_optimization(self, X, y, groups, cluster_id=None):
        if optuna is None:
            raise ImportError("search_method='bayesian' requires optuna to be installed.")
        study = optuna.create_study(
            direction='maximize',
            sampler=self._create_bayesian_sampler(),
            study_name=f"cluster_{cluster_id}" if cluster_id is not None else "global")
        study.optimize(partial(self._bayesian_objective, X=X, y=y, groups=groups),
                       n_trials=self.n_trials_bayesian, n_jobs=self.optuna_n_jobs)
        if cluster_id is not None:
            self.bayesian_studies[cluster_id] = study
        # hpELM's tuning was silent: unlike WAS_mme_RF it printed nothing per
        # cluster, so a run looked stalled between "Cross-validation ongoing"
        # and the next stage. One line per cluster, gated on verbose.
        if self.verbose:
            print(f"[hpelm] cluster {cluster_id}: {len(study.trials)} essais, "
                  f"meilleur score={study.best_value:.4f}, "
                  f"params={study.best_params}", flush=True)
        return study.best_params

    # ====================================================== hyperparameters
    def compute_hyperparameters(self, Predictors, Predictand,
                                clim_year_start=None, clim_year_end=None):
        """Tune with fold-local standardization, target encoding and climate zones."""
        return _strict_regional_hyperparameters(
            self, Predictors, Predictand, clim_year_start, clim_year_end)

    def compute_hyperparameters_legacy(self, Predictors, Predictand,
                                clim_year_start=None, clim_year_end=None):
        """
        Best hyperparameters per zone.

        Regionalization: KMeans on the predictand VALUES (kept by preference).
        Standardization is applied here only if clim years are given (dual mode).
        """
        Predictors = align_predictor_target(Predictors, Predictand, mode="season_year")
        np.random.seed(self.random_state)
        random.seed(self.random_state)

        if "M" in Predictand.dims:
            Predictand = Predictand.isel(M=0, drop=True)
        Predictand.name = "varname"

        if clim_year_start is not None and clim_year_end is not None:
            X_train_std = standardize_timeseries(Predictors, clim_year_start, clim_year_end)
            y_train_std = standardize_timeseries(Predictand, clim_year_start, clim_year_end)
        else:
            X_train_std = Predictors
            y_train_std = Predictand
        X_train_std['T'] = y_train_std['T']

        cluster_da = _build_climate_feature_clusters(
            Predictand, self.n_clusters, random_state=self.random_state
        )
        clusters = np.unique(cluster_da.values)
        clusters = clusters[np.isfinite(clusters)]

        best_params_dict = {}
        for c in clusters:
            c_int = int(c)
            mask_3d = (cluster_da == c).expand_dims({'T': y_train_std['T']})
            Xc = X_train_std.where(mask_3d).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M')
            yc = y_train_std.where(mask_3d).stack(sample=('T', 'Y', 'X'))

            groups_full = Xc['T'].values.astype('datetime64[Y]').astype(int)   # >>> E
            Xv, yv = Xc.values, yc.values.ravel()
            bad = np.any(~np.isfinite(Xv), axis=1) | ~np.isfinite(yv)
            X_clean, y_clean, groups_c = Xv[~bad], yv[~bad], groups_full[~bad]

            n_years = len(np.unique(groups_c))
            _effective_sample_warning(n_years, X_clean.shape[1], "WAS_mme_hpELM")
            if len(X_clean) < self.cv_folds * 2 or n_years < 2:
                if self.verbose:
                    print(f"Cluster {c_int}: insufficient data "
                          f"({len(X_clean)} samples, {n_years} years). Skipping.")
                continue

            if self.search_method == 'grid':
                best_params_dict[c_int] = self._grid_search_optimization(X_clean, y_clean, groups_c)
            elif self.search_method == 'random':
                best_params_dict[c_int] = self._random_search_optimization(X_clean, y_clean, groups_c)
            elif self.search_method == 'bayesian':
                best_params_dict[c_int] = self._bayesian_optimization(X_clean, y_clean, groups_c, cluster_id=c_int)
            else:
                raise ValueError(f"Unknown optimization method: {self.search_method}")

        self.best_params_dict = best_params_dict
        return best_params_dict, cluster_da

    # ============================================================ deterministic
    def _fit_predict_cluster(self, bp, Xtr_c, ytr_c, Xte_c):
        """Train one ELM and predict; returns ``(model, predictions of Xte_c)``.

        ``model`` is an :class:`~wass2s.was_elm.ELMRegressor` (``predict``,
        ``coef_``, ``n_features_in_``). RBF activations draw their centres from
        the training rows and positive widths (hpelm reads W/B as centres and
        widths for 'rbf_*'); an all-NaN prediction is never returned silently.
        """
        activation, n_neurons = bp['activation'], int(bp['neurons'])
        model = HPELMWrapper(neurons=n_neurons, activation=activation, norm=bp['norm'],
                             random_state=self.random_state,
                             norm_scaling=getattr(self, 'norm_scaling', 'absolute'),
                             **_elm_structure(self))
        model.fit(Xtr_c, np.asarray(ytr_c).ravel())
        pred = np.asarray(model.predict(Xte_c)).ravel()

        if Xte_c.shape[0] and not np.any(np.isfinite(pred)):
            warnings.warn(
                f"ELM activation {activation!r} produced no finite prediction "
                f"({n_neurons} neurons, norm={bp['norm']}); this cluster would "
                "have been silently empty.",
                RuntimeWarning, stacklevel=2,
            )
        return model, pred

    def compute_model(self, X_train, y_train, X_test, y_test=None,
                      clim_year_start=None, clim_year_end=None,
                      best_params=None, cluster_da=None, covariates=None):
        """
        Deterministic hindcast with one HPELM per zone (or one global model
        plus spatial covariates when ``spatial_mode`` is "soft").

        >>> A: clim years now in the signature; tuning runs in dual mode when
        best_params/cluster_da are not supplied.
        """
        validate_mme_predictor_schema(X_train, X_test)
        X_train = align_predictor_target(X_train, y_train, mode="season_year")
        best_params, cluster_da = _resolve_regional_params(
            self, X_train, y_train, clim_year_start, clim_year_end, best_params, cluster_da)
        _raw_predictors = (X_train, X_test)   # avant covariables : pour la calibration
        X_train, X_test = _prepare_channels(self, X_train, X_test, y_train, covariates,
                                            clim_year_start, clim_year_end)

        time, lat, lon = X_test['T'], X_test['Y'], X_test['X']
        n_time, n_lat, n_lon = len(time), len(lat), len(lon)
        predictions = np.full((n_time * n_lat * n_lon,), np.nan)
        self.hpelm = {}

        train_rows = _cluster_row_index(cluster_da, X_train.sizes['T'])
        test_rows = _cluster_row_index(cluster_da, X_test.sizes['T'])
        active = [c for c in range(self.n_clusters) if c in best_params]
        # HPELM is a BLAS-bound least-squares solve. With OMP_NUM_THREADS=1 in
        # the launcher, the cluster axis is the only parallelism available.
        workers, _threads = _cluster_plan(
            self.n_clusters, self.n_jobs,
            getattr(self, 'cluster_n_jobs', None), model_is_threaded=False,
        )

        def _fit_one(c):
            bp = best_params[c]
            train_sel = _thin_training_rows(train_rows == c, X_train.sizes['T'], n_lat, n_lon,
                                            getattr(self, 'spatial_subsample', 1))

            Xtr, _ = _stack_rows(X_train, train_sel, core='M')
            ytr, _ = _stack_rows(y_train, train_sel, core=None)
            ytr = ytr.ravel()
            keep = np.all(np.isfinite(Xtr), axis=1) & np.isfinite(ytr)

            Xte, pos = _stack_rows(X_test, test_rows == c, core='M')
            keep_te = np.all(np.isfinite(Xte), axis=1)
            if not keep.any() or not keep_te.any():
                return c, None, None, None

            hp, y_pred = self._fit_predict_cluster(
                bp, Xtr[keep], ytr[keep], Xte[keep_te]
            )
            return c, hp, pos[keep_te], y_pred

        if workers > 1 and len(active) > 1:
            results = Parallel(n_jobs=min(workers, len(active)), backend="threading")(
                delayed(_fit_one)(c) for c in active
            )
        else:
            results = [_fit_one(c) for c in active]

        for c, model, pos, vals in results:
            if model is not None:
                self.hpelm[c] = model
            if pos is not None:
                predictions[pos] = vals
        predictions = predictions.reshape(n_time, n_lat, n_lon)

        predicted_da = xr.DataArray(predictions, coords={'T': time, 'Y': lat, 'X': lon},
                                    dims=['T', 'Y', 'X'])
        return _maybe_calibrate_amplitude(
            self, self.hpelm, X_train, y_train, predicted_da, cluster_da, X_test=X_test,
            raw_predictors=_raw_predictors, best_params=best_params,
            clim_year_start=clim_year_start, clim_year_end=clim_year_end)

    # ------------------ Probability Calculation Methods ------------------
    @staticmethod
    def _ppf_terciles_from_code(dist_code, shape, loc, scale):
        if np.isnan(dist_code):
            return np.nan, np.nan
        code = int(dist_code)
        try:
            if code == 1:
                return (norm.ppf(1/3, loc=loc, scale=scale), norm.ppf(2/3, loc=loc, scale=scale))
            elif code == 2:
                return (lognorm.ppf(1/3, s=shape, loc=loc, scale=scale),
                        lognorm.ppf(2/3, s=shape, loc=loc, scale=scale))
            elif code == 3:
                return (expon.ppf(1/3, loc=loc, scale=scale), expon.ppf(2/3, loc=loc, scale=scale))
            elif code == 4:
                return (gamma.ppf(1/3, a=shape, loc=loc, scale=scale),
                        gamma.ppf(2/3, a=shape, loc=loc, scale=scale))
            elif code == 5:
                return (weibull_min.ppf(1/3, c=shape, loc=loc, scale=scale),
                        weibull_min.ppf(2/3, c=shape, loc=loc, scale=scale))
            elif code == 6:
                return (t.ppf(1/3, df=shape, loc=loc, scale=scale),
                        t.ppf(2/3, df=shape, loc=loc, scale=scale))
            elif code == 7:
                return (poisson.ppf(1/3, mu=shape, loc=loc), poisson.ppf(2/3, mu=shape, loc=loc))
            elif code == 8:
                return (nbinom.ppf(1/3, n=shape, p=scale, loc=loc),
                        nbinom.ppf(2/3, n=shape, p=scale, loc=loc))
        except Exception:
            return np.nan, np.nan
        return np.nan, np.nan

    @staticmethod
    def weibull_shape_solver(k, M, V):
        if k <= 0:
            return -np.inf
        try:
            g1 = gamma_function(1 + 1/k)
            g2 = gamma_function(1 + 2/k)
            return (V / (M**2)) - ((g2 / (g1**2)) - 1)
        except ValueError:
            return -np.inf

    @staticmethod
    def calculate_tercile_probabilities_bestfit(best_guess, error_variance, T1, T2, dist_code, dof):
        best_guess = np.asarray(best_guess, float)
        error_variance = np.asarray(error_variance, dtype=float)
        n_time = best_guess.size
        out = np.full((3, n_time), np.nan, float)

        if (np.all(np.isnan(best_guess)) or np.isnan(dist_code)
                or np.isnan(T1) or np.isnan(T2) or np.isnan(error_variance)):
            return out

        code = int(dist_code)

        if code == 1:
            sd = np.sqrt(error_variance)
            out[0, :] = norm.cdf(T1, loc=best_guess, scale=sd)
            out[1, :] = norm.cdf(T2, loc=best_guess, scale=sd) - norm.cdf(T1, loc=best_guess, scale=sd)
            out[2, :] = 1 - norm.cdf(T2, loc=best_guess, scale=sd)

        elif code == 2:
            sigma = np.sqrt(np.log(1 + error_variance / (best_guess**2)))
            mu = np.log(best_guess) - sigma**2 / 2
            out[0, :] = lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
            out[1, :] = lognorm.cdf(T2, s=sigma, scale=np.exp(mu)) - lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
            out[2, :] = 1 - lognorm.cdf(T2, s=sigma, scale=np.exp(mu))

        elif code == 3:
            sc = np.sqrt(error_variance)
            c1 = expon.cdf(T1, loc=best_guess, scale=sc)
            c2 = expon.cdf(T2, loc=best_guess, scale=sc)        # >>> B: was loc_t
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        elif code == 4:
            alpha = (best_guess ** 2) / error_variance
            theta = error_variance / best_guess
            c1 = gamma.cdf(T1, a=alpha, scale=theta)
            c2 = gamma.cdf(T2, a=alpha, scale=theta)
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        elif code == 5:
            for i in range(n_time):
                M, V = best_guess[i], error_variance
                if V <= 0 or M <= 0:
                    out[:, i] = np.nan
                    continue
                k = fsolve(WAS_mme_hpELM.weibull_shape_solver, 2.0, args=(M, V))[0]  # >>> C
                if k <= 0:
                    out[:, i] = np.nan
                    continue
                lam = M / gamma_function(1 + 1/k)
                c1 = weibull_min.cdf(T1, c=k, loc=0, scale=lam)
                c2 = weibull_min.cdf(T2, c=k, loc=0, scale=lam)
                out[0, i] = c1
                out[1, i] = c2 - c1
                out[2, i] = 1.0 - c2

        elif code == 6:
            if dof <= 2:
                out[:, :] = np.nan
            else:
                scale = np.sqrt(error_variance * (dof - 2) / dof)
                c1 = t.cdf(T1, df=dof, loc=best_guess, scale=scale)
                c2 = t.cdf(T2, df=dof, loc=best_guess, scale=scale)
                out[0, :] = c1
                out[1, :] = c2 - c1
                out[2, :] = 1.0 - c2

        elif code == 7:
            mu = best_guess
            c1 = poisson.cdf(T1, mu=mu)
            c2 = poisson.cdf(T2, mu=mu)
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        elif code == 8:
            p = np.where(error_variance > best_guess, best_guess / error_variance, np.nan)
            n = np.where(error_variance > best_guess, (best_guess**2) / (error_variance - best_guess), np.nan)
            c1 = nbinom.cdf(T1, n=n, p=p)
            c2 = nbinom.cdf(T2, n=n, p=p)
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        else:
            raise ValueError(f"Invalid distribution code: {dist_code}")

        # >>> F: positive-support families undefined for non-positive means
        if code in (2, 3, 4, 5, 7, 8):
            bad = ~(best_guess > 0)
            if np.any(bad):
                out[:, bad] = np.nan

        return out

    @staticmethod
    def calculate_tercile_probabilities_nonparametric(best_guess, error_samples, first_tercile, second_tercile):
        return WAS_mme_Weighted.calculate_tercile_probabilities_nonparametric(
            best_guess, error_samples, first_tercile, second_tercile)

    def compute_prob(self, Predictant: xr.DataArray, clim_year_start, clim_year_end,
                     hindcast_det: xr.DataArray, best_code_da=None, best_shape_da=None,
                     best_loc_da=None, best_scale_da=None) -> xr.DataArray:
        """Tercile probabilities for deterministic hindcasts (pass the CV hindcast for calibration)."""
        if "M" in Predictant.dims:
            Predictant = Predictant.isel(M=0, drop=True)
        Predictant = Predictant.transpose("T", "Y", "X")
        mask = xr.where(np.isfinite(Predictant).sum(dim='T') >= 3, 1.0, np.nan)

        clim = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))
        if clim.sizes.get("T", 0) < 3:
            raise ValueError("Not enough years in climatology period for terciles.")

        error_variance = ((Predictant - hindcast_det) ** 2).mean(dim="T", skipna=True)
        dof = max(int(clim.sizes["T"]) - 1, 2)

        terciles_emp = climatological_terciles(clim)[0]       # >>> 1/3, 2/3
        T1_emp = terciles_emp.isel(quantile=0, drop=True)
        T2_emp = terciles_emp.isel(quantile=1, drop=True)

        dm = self.dist_method
        if dm == "bestfit":
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError("dist_method='bestfit' requires best_code_da, best_shape_da, "
                                 "best_loc_da, best_scale_da.")
            # Category boundaries stay on the observed climatological reference.
            # The selected distribution controls the predictive CDF only.
            if getattr(self, "tercile_reference", "climatology") == "training":
                T1, T2 = _pufunc(
                    self._ppf_terciles_from_code, best_code_da, best_shape_da, best_loc_da, best_scale_da,
                    input_core_dims=[(), (), (), ()], output_core_dims=[(), ()],
                    vectorize=True, dask="parallelized", output_dtypes=[float, float])
            else:
                T1, T2 = T1_emp, T2_emp
            hindcast_prob = _pufunc(
                self.calculate_tercile_probabilities_bestfit,
                hindcast_det, error_variance, T1, T2, best_code_da,
                input_core_dims=[("T",), (), (), (), ()], output_core_dims=[("probability", "T")],
                vectorize=True, kwargs={'dof': dof}, dask="parallelized", output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True})

        elif dm == "nonparam":
            error_samples = Predictant - hindcast_det
            hindcast_prob = _pufunc(
                self.calculate_tercile_probabilities_nonparametric,
                hindcast_det, error_samples, T1_emp, T2_emp,
                input_core_dims=[("T",), ("T",), (), ()], output_core_dims=[("probability", "T")],
                vectorize=True, dask="parallelized", output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True})
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}")

        hindcast_prob = hindcast_prob.assign_coords(probability=("probability", ["PB", "PN", "PA"]))
        return (hindcast_prob * mask).transpose("probability", "T", "Y", "X")

    # ============================================================== operational
    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det,
                 hindcast_det_cross, Predictor_for_year, best_params=None, cluster_da=None,
                 best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None):
        """Operational forecast for a target year (deterministic + tercile probabilities)."""
        validate_mme_predictor_schema(hindcast_det, Predictor_for_year)
        if "M" in Predictant.dims:
            Predictant_no_m = Predictant.isel(M=0, drop=True)
        else:
            Predictant_no_m = Predictant
        mask = xr.where(np.isfinite(Predictant_no_m).sum(dim='T') >= 3, 1.0, np.nan).to_numpy()

        hindcast_det_st, Predictor_for_year_st = standardize_mme_predictors(
            hindcast_det, Predictor_for_year, clim_year_start, clim_year_end)
        Predictant_st = standardize_timeseries(Predictant_no_m, clim_year_start, clim_year_end)
        hindcast_det_st['T'] = Predictant_st['T']

        time, lat, lon = Predictor_for_year_st['T'], Predictor_for_year_st['Y'], Predictor_for_year_st['X']
        n_time, n_lat, n_lon = len(time), len(lat), len(lon)

        best_params, cluster_da = _resolve_regional_params(
            self, hindcast_det, Predictant_no_m, clim_year_start, clim_year_end,
            best_params, cluster_da)
        hindcast_det_st, Predictor_for_year_st = _prepare_channels(
            self, hindcast_det_st, Predictor_for_year_st, Predictant_no_m, None,
            clim_year_start, clim_year_end)

        predictions = np.full((n_time, n_lat, n_lon), np.nan)
        self.hpelm = {}

        for c in range(self.n_clusters):
            if c not in best_params:
                continue
            bp = best_params[c]
            m_tr = (cluster_da == c).expand_dims({'T': hindcast_det_st['T']})
            m_te = (cluster_da == c).expand_dims({'T': Predictor_for_year_st['T']})

            Xtr = hindcast_det_st.where(m_tr).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            ytr = Predictant_st.where(m_tr).stack(sample=('T', 'Y', 'X')).values.ravel()
            tr_bad = np.any(~np.isfinite(Xtr), axis=1) | ~np.isfinite(ytr)
            Xtr_c, ytr_c = Xtr[~tr_bad], ytr[~tr_bad]

            Xte = Predictor_for_year_st.where(m_te).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M').values
            te_bad = np.any(~np.isfinite(Xte), axis=1)
            Xte_c = Xte[~te_bad]

            if len(Xtr_c) == 0 or len(Xte_c) == 0:
                continue

            hp, y_pred = self._fit_predict_cluster(bp, Xtr_c, ytr_c, Xte_c)
            self.hpelm[c] = hp

            result = np.full(Xte.shape[0], np.nan)
            result[~te_bad] = y_pred
            predictions = np.where(np.isnan(predictions),
                                   result.reshape(n_time, n_lat, n_lon), predictions)

        result_da = xr.DataArray(predictions, coords={'T': time, 'Y': lat, 'X': lon},
                                 dims=['T', 'Y', 'X']) * mask
        result_da = reverse_standardize(result_da, Predictant_no_m, clim_year_start, clim_year_end)

        year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970
        month = Predictant_no_m.isel(T=0).coords['T'].values.astype('datetime64[M]').astype(int) % 12 + 1
        new_T = np.datetime64(f"{year}-{month:02d}-01")
        result_da = result_da.assign_coords(T=xr.DataArray([new_T], dims=["T"]))
        result_da['T'] = result_da['T'].astype('datetime64[ns]')

        index_start = Predictant_no_m.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant_no_m.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant_no_m.isel(T=slice(index_start, index_end))
        terciles = climatological_terciles(rainfall_for_tercile)[0]   # >>> 1/3, 2/3
        T1_emp = terciles.isel(quantile=0, drop=True)
        T2_emp = terciles.isel(quantile=1, drop=True)
        error_variance = ((Predictant_no_m - hindcast_det_cross) ** 2).mean(dim="T", skipna=True)
        dof = max(int(rainfall_for_tercile.sizes["T"]) - 1, 2)

        dm = self.dist_method
        if dm == "bestfit":
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError("dist_method='bestfit' requires best_code_da, best_shape_da, "
                                 "best_loc_da, best_scale_da.")
            # Category boundaries stay on the observed climatological reference.
            # The selected distribution controls the predictive CDF only.
            if getattr(self, "tercile_reference", "climatology") == "training":
                T1, T2 = _pufunc(
                    self._ppf_terciles_from_code, best_code_da, best_shape_da, best_loc_da, best_scale_da,
                    input_core_dims=[(), (), (), ()], output_core_dims=[(), ()],
                    vectorize=True, dask="parallelized", output_dtypes=[float, float])
            else:
                T1, T2 = T1_emp, T2_emp
            forecast_prob = _pufunc(
                self.calculate_tercile_probabilities_bestfit,
                result_da, error_variance, T1, T2, best_code_da,
                input_core_dims=[("T",), (), (), (), ()], output_core_dims=[("probability", "T")],
                vectorize=True, dask="parallelized", kwargs={"dof": dof}, output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True})

        elif dm == "nonparam":
            # >>> D: out-of-sample residuals on the de-M'd predictand, with the
            # residual time dim renamed so it doesn't collide with the forecast T.
            error_samples = (Predictant_no_m - hindcast_det_cross).rename({"T": "S"})
            forecast_prob = _pufunc(
                self.calculate_tercile_probabilities_nonparametric,
                result_da, error_samples, T1_emp, T2_emp,
                input_core_dims=[("T",), ("S",), (), ()], output_core_dims=[("probability", "T")],
                vectorize=True, dask="parallelized", output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True})
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}")

        forecast_prob = forecast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return result_da * mask, mask * forecast_prob.transpose('probability', 'T', 'Y', 'X')

    # ------------------ utilities ------------------
    def get_optimization_results(self):
        results = {'best_params': self.best_params_dict, 'search_method': self.search_method,
                   'bayesian_studies': self.bayesian_studies}
        if self.search_method == 'bayesian' and self.bayesian_studies:
            for cid, study in self.bayesian_studies.items():
                results[f'bayesian_trials_cluster_{cid}'] = study.trials_dataframe()
        return results


class WAS_mme_MLP:
    """
    Multi-Layer Perceptron (MLPRegressor) MME forecaster with per-zone
    hyperparameter tuning and tercile-probability dressing. Deterministic
    hindcast via compute_model; tercile probabilities via compute_prob
    ('nonparam' or 'bestfit'). Mirrors the WAS_mme_RF / WAS_mme_XGBoosting
    contract so it plugs into the same_kind_model2 branch of WAS_Cross_Validator.

    Parameters
    ----------
    n_jobs : int, optional
        Maximum number of parallel workers for grid/random search. ``-1``
        uses all available CPUs (default is 4). MLPRegressor itself has no
        ``n_jobs`` parameter, so this controls parallel hyperparameter fits.
    optuna_n_jobs : int, optional
        Optional Optuna-specific worker count. When ``None``, it inherits
        ``n_jobs`` (default is None).
    """

    def __init__(self,
                 search_method='random',
                 hidden_layer_sizes_range=[(2,), (4,), (8,), (4, 2)],
                 activation_options=['relu', 'tanh', 'identity'],
                 learning_rate_init_range=loguniform(0.0001, 0.01),
                 solver_options=['lbfgs', 'adam'],
                 alpha_range=[0.01, 0.1, 1.0, 10.0],
                 max_iter=500,
                 random_state=42,
                 dist_method="nonparam",
                 n_iter_search=10,
                 cv_folds=3,
                 n_clusters=4,
                 early_stopping_options=[False],
                 validation_fraction=0.1,
                 n_iter_no_change=10,
                 leave_one_year_out=False,
                 optuna_n_jobs=None,
                 optuna_timeout=None,
                 n_jobs=-1,
                 cv_n_jobs=-1,
                 cluster_n_jobs: Optional[int] = None,
                 spatial_subsample: int = 1,
                 spatial_mode: str = "zones",
                 covariate_names: Optional[Sequence[str]] = None,
                 static_fields: Optional[Dict[str, xr.DataArray]] = None,
                 coord_step: Optional[float] = None,
                 zones_as_features: bool = False,
                 predictor_smoothing: int = 0,
                 one_se_rule: bool = False,
                 amplitude_calibration=True,
                 amplitude_calibration_inner_folds: int = 3):
        # Cluster-level parallelism and optional thinning of the
        # TRAINING sample. See _cluster_plan and _stack_rows.
        self.cluster_n_jobs = cluster_n_jobs
        self.spatial_subsample = max(1, int(spatial_subsample))
        _init_spatial_options(self, spatial_mode, covariate_names, static_fields,
                              coord_step, zones_as_features, predictor_smoothing, one_se_rule,
                              amplitude_calibration, amplitude_calibration_inner_folds)

        self.search_method = search_method
        self.hidden_layer_sizes_range = hidden_layer_sizes_range
        self.learning_rate_init_range = learning_rate_init_range
        self.activation_options = activation_options
        self.solver_options = solver_options
        self.alpha_range = alpha_range
        self.max_iter = max_iter
        self.random_state = random_state
        self.dist_method = dist_method
        self.n_iter_search = n_iter_search
        self.cv_folds = cv_folds
        self.n_clusters = n_clusters
        # early_stopping is a native MLPRegressor option (sgd/adam only; ignored
        # for lbfgs). Tunable via early_stopping_options; validation_fraction and
        # n_iter_no_change configure it.
        if any(bool(v) for v in early_stopping_options):
            warnings.warn("MLP internal early stopping splits pooled pixels across the same year; use explicit grouped epoch selection. Internal early stopping disabled.", RuntimeWarning)
        self.early_stopping_options = (False,)
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.leave_one_year_out = leave_one_year_out      # E
        if n_jobs == 0:
            raise ValueError("n_jobs cannot be 0; use a positive integer or -1.")
        if optuna_n_jobs == 0:
            raise ValueError("optuna_n_jobs cannot be 0.")
        if cv_n_jobs == 0:
            raise ValueError("cv_n_jobs cannot be 0.")
        self.n_jobs = n_jobs
        self.cv_n_jobs = cv_n_jobs
        self.optuna_n_jobs = 1 if optuna_n_jobs is None else optuna_n_jobs
        self.optuna_timeout = optuna_timeout
        self.mlp = None

    # ----------------------------------------------------- E: grouped CV
    @staticmethod
    def _sample_years(stacked):
        """Calendar-year label per stacked (T,Y,X) sample (falls back to value)."""
        T = stacked['T']
        vals = np.asarray(T.values)
        if np.issubdtype(vals.dtype, np.datetime64):
            return T.dt.year.values
        return vals

    def _cv_splitter(self, groups):
        """Year-grouped splitter: LeaveOneGroupOut (requested / few years) else
        GroupKFold(cv_folds). Whole years are held out -> no spatial leakage."""
        n_groups = len(np.unique(groups))
        if self.leave_one_year_out or self.cv_folds >= n_groups:
            return LeaveOneGroupOut()
        return GroupKFold(n_splits=min(self.cv_folds, n_groups))

    def _objective(self, trial, X_train, y_train, groups, splitter):
        """Optuna objective: year-grouped CV neg-MSE (maximized)."""
        hidden_layer_sizes = trial.suggest_categorical(
            'hidden_layer_sizes', self.hidden_layer_sizes_range)
        activation = trial.suggest_categorical('activation', self.activation_options)
        solver = trial.suggest_categorical('solver', self.solver_options)

        if solver == 'lbfgs':
            learning_rate_init = 0.001
            early_stopping = False
        else:
            if isinstance(self.learning_rate_init_range, list):
                learning_rate_init = trial.suggest_categorical(
                    'learning_rate_init', self.learning_rate_init_range)
            else:
                learning_rate_init = trial.suggest_float(
                    'learning_rate_init', self.learning_rate_init_range.a,
                    self.learning_rate_init_range.b, log=True)
            early_stopping = trial.suggest_categorical(
                'early_stopping', self.early_stopping_options)

        if isinstance(self.alpha_range, list):
            alpha = trial.suggest_categorical('alpha', self.alpha_range)
        else:
            alpha = trial.suggest_float('alpha', self.alpha_range.a,
                                        self.alpha_range.b, log=True)

        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            learning_rate_init=learning_rate_init,
            activation=activation, solver=solver, alpha=alpha,
            early_stopping=early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            max_iter=self.max_iter, random_state=self.random_state)

        scores = cross_val_score(model, X_train, y_train, cv=splitter,
                                 groups=groups, scoring='neg_mean_squared_error',
                                 n_jobs=self.cv_n_jobs)
        return np.mean(scores)

    # ------------------------------------------------------ hyperparameters
    def compute_hyperparameters(self, Predictors, Predictand,
                                clim_year_start=None, clim_year_end=None):
        """Tune with fold-local standardization, target encoding and climate zones."""
        return _strict_regional_hyperparameters(
            self, Predictors, Predictand, clim_year_start, clim_year_end)

    def compute_hyperparameters_legacy(self, Predictors, Predictand,
                                clim_year_start=None, clim_year_end=None):
        """
        Per-zone hyperparameter tuning. KMeans on predictand VALUES (kept).

        A dual-mode: clim years given -> standardize Predictors / Predictand
        internally; both None -> assume already standardized (framework path),
        so no double-standardization when called from compute_model.
        """
        Predictors = align_predictor_target(Predictors, Predictand, mode="season_year")
        if "M" in Predictand.dims:
            Predictand = Predictand.isel(M=0, drop=True)
        Predictand.name = "varname"

        if clim_year_start is not None and clim_year_end is not None:
            X_train_std = standardize_timeseries(Predictors, clim_year_start, clim_year_end)
            y_train_std = standardize_timeseries(Predictand, clim_year_start, clim_year_end)
        else:
            X_train_std = Predictors
            y_train_std = Predictand

        cluster_da = _build_climate_feature_clusters(
            Predictand, self.n_clusters, random_state=self.random_state
        )
        clusters = np.unique(cluster_da.values)
        clusters = clusters[np.isfinite(clusters)]
        X_train_std['T'] = y_train_std['T']

        best_params_dict = {}
        for c in clusters:
            mask_3d = (cluster_da == c).expand_dims({'T': y_train_std['T']})
            Xc = X_train_std.where(mask_3d).stack(sample=('T', 'Y', 'X')).transpose('sample', 'M')
            yc = y_train_std.where(mask_3d).stack(sample=('T', 'Y', 'X'))

            years_all = self._sample_years(Xc)
            Xv = Xc.values
            yv = yc.values.ravel()
            nan_mask_c = np.any(~np.isfinite(Xv), axis=1) | ~np.isfinite(yv)
            X_clean_c = Xv[~nan_mask_c]
            y_clean_c = yv[~nan_mask_c]
            groups_c = years_all[~nan_mask_c]

            if len(X_clean_c) == 0 or len(np.unique(groups_c)) < 2:
                continue
            _effective_sample_warning(len(np.unique(groups_c)), X_clean_c.shape[1], "WAS_mme_MLP")

            splitter = self._cv_splitter(groups_c)   # E

            if self.search_method == 'grid':
                if isinstance(self.learning_rate_init_range, list):
                    lr_values = self.learning_rate_init_range
                else:
                    n_samples = min(5, self.n_iter_search)
                    lr_values = np.unique(self.learning_rate_init_range.rvs(
                        size=n_samples, random_state=self.random_state))
                if isinstance(self.alpha_range, list):
                    alpha_values = self.alpha_range
                else:
                    n_samples = min(5, self.n_iter_search)
                    alpha_values = np.unique(self.alpha_range.rvs(
                        size=n_samples, random_state=self.random_state))
                param_grid = []
                iterative = [s for s in self.solver_options if s in ('adam', 'sgd')]
                if iterative:
                    param_grid.append({
                        'hidden_layer_sizes': self.hidden_layer_sizes_range,
                        'activation': self.activation_options,
                        'solver': iterative,
                        'early_stopping': self.early_stopping_options,
                        'learning_rate_init': lr_values,
                        'alpha': alpha_values,
                    })
                if 'lbfgs' in self.solver_options:
                    param_grid.append({
                        'hidden_layer_sizes': self.hidden_layer_sizes_range,
                        'activation': self.activation_options,
                        'solver': ['lbfgs'],
                        'early_stopping': [False],
                        'learning_rate_init': [0.001],
                        'alpha': alpha_values,
                    })
                if not param_grid:
                    raise ValueError("solver_options contains no supported MLP solver.")

                model = MLPRegressor(max_iter=self.max_iter, random_state=self.random_state,
                                     validation_fraction=self.validation_fraction,
                                     n_iter_no_change=self.n_iter_no_change)
                grid_search = GridSearchCV(model, param_grid=param_grid, cv=splitter,
                                           scoring='neg_mean_squared_error',
                                           error_score=np.nan, n_jobs=self.n_jobs)
                grid_search.fit(X_clean_c, y_clean_c, groups=groups_c)
                best_params_dict[c] = grid_search.best_params_

            elif self.search_method == 'random':
                param_dist = []
                iterative = [s for s in self.solver_options if s in ('adam', 'sgd')]
                if iterative:
                    param_dist.append({
                        'hidden_layer_sizes': self.hidden_layer_sizes_range,
                        'learning_rate_init': self.learning_rate_init_range,
                        'activation': self.activation_options,
                        'solver': iterative,
                        'alpha': self.alpha_range,
                        'early_stopping': self.early_stopping_options,
                    })
                if 'lbfgs' in self.solver_options:
                    param_dist.append({
                        'hidden_layer_sizes': self.hidden_layer_sizes_range,
                        'learning_rate_init': [0.001],
                        'activation': self.activation_options,
                        'solver': ['lbfgs'],
                        'alpha': self.alpha_range,
                        'early_stopping': [False],
                    })
                if not param_dist:
                    raise ValueError("solver_options contains no supported MLP solver.")
                model = MLPRegressor(max_iter=self.max_iter, random_state=self.random_state,
                                     validation_fraction=self.validation_fraction,
                                     n_iter_no_change=self.n_iter_no_change)
                random_search = RandomizedSearchCV(
                    model, param_distributions=param_dist, n_iter=self.n_iter_search,
                    cv=splitter, scoring='neg_mean_squared_error',
                    random_state=self.random_state, error_score=np.nan,
                    n_jobs=self.n_jobs)
                random_search.fit(X_clean_c, y_clean_c, groups=groups_c)
                best_params_dict[c] = random_search.best_params_
    
            elif self.search_method == 'bayesian':
                if not HAS_OPTUNA:
                    raise ImportError("search_method='bayesian' requires optuna.")
            
                study = optuna.create_study(
                    direction='maximize',
                    sampler=optuna.samplers.TPESampler(seed=self.random_state),
                    pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
                )
            
                objective_with_data = lambda trial: self._objective(
                    trial,
                    X_clean_c,
                    y_clean_c,
                    groups_c,
                    splitter
                )
            
                study.optimize(
                    objective_with_data,
                    n_trials=self.n_iter_search,
                    timeout=self.optuna_timeout,
                    n_jobs=self.optuna_n_jobs
                )
            
                bp = study.best_params
                solver = bp['solver']
            
                # Reconstruct parameters that may have been fixed rather than
                # explicitly suggested to Optuna.
                best_params_dict[c] = {
                    'hidden_layer_sizes': bp['hidden_layer_sizes'],
                    'learning_rate_init': (
                        0.001
                        if solver == 'lbfgs'
                        else bp.get('learning_rate_init', 0.001)
                    ),
                    'activation': bp['activation'],
                    'solver': solver,
                    'alpha': bp['alpha'],
                    'early_stopping': (
                        False
                        if solver == 'lbfgs'
                        else bp.get('early_stopping', False)
                    ),
                }
            else:
                raise ValueError(f"Unknown search_method: {self.search_method}. "
                                 "Choose from 'grid', 'random', or 'bayesian'.")    
        # BUG FIX: this return was indented INSIDE the `for c in clusters:`
        # loop, so compute_hyperparameters returned as soon as the FIRST
        # cluster had been tuned. Every other cluster was missing from
        # best_params_dict, and compute_model skips a cluster it has no
        # parameters for -- so on a 5-cluster domain, 4 clusters produced no
        # MLP prediction at all and the member covered ~1/5 of the grid.
        # Dedenting it to the method level tunes every cluster.
        _report_best_params("mlp", best_params_dict, getattr(self, "verbose", 1))
        return best_params_dict, cluster_da

    # ------------------------------------------------------ deterministic fit
    def compute_model(self, X_train, y_train, X_test, y_test=None,
                      clim_year_start=None, clim_year_end=None,
                      best_params=None, cluster_da=None, covariates=None):
        """
        Deterministic hindcast per zone (or one global model plus spatial
        covariates when ``spatial_mode`` is "soft"). Inputs are assumed
        standardized (the same_kind_model2 branch standardizes once). A:
        clim_year_start/end are in the signature, so the best_params=None path
        no longer raises.
        """
        validate_mme_predictor_schema(X_train, X_test)
        X_train = align_predictor_target(X_train, y_train, mode="season_year")
        best_params, cluster_da = _resolve_regional_params(                                   # A
            self, X_train, y_train, clim_year_start, clim_year_end, best_params, cluster_da)
        _raw_predictors = (X_train, X_test)   # avant covariables : pour la calibration
        X_train, X_test = _prepare_channels(self, X_train, X_test, y_train, covariates,
                                            clim_year_start, clim_year_end)
        X_train_std = X_train
        y_train_std = y_train
        X_test_std = X_test

        time = X_test_std['T']; lat = X_test_std['Y']; lon = X_test_std['X']
        n_time, n_lat, n_lon = len(time), len(lat), len(lon)

        predictions = np.full((n_time * n_lat * n_lon,), np.nan)
        self.mlp = {}

        train_rows = _cluster_row_index(cluster_da, X_train_std.sizes['T'])
        test_rows = _cluster_row_index(cluster_da, X_test_std.sizes['T'])
        active = [c for c in range(self.n_clusters) if c in best_params]
        # MLPRegressor exposes no n_jobs, so the cluster axis is the only
        # parallelism available here.
        workers, _threads = _cluster_plan(
            self.n_clusters, self.n_jobs,
            getattr(self, 'cluster_n_jobs', None), model_is_threaded=False,
        )

        def _fit_one(c):
            bp = best_params[c]
            train_sel = _thin_training_rows(train_rows == c, X_train_std.sizes['T'], n_lat, n_lon,
                                            getattr(self, 'spatial_subsample', 1))

            X_tr, _ = _stack_rows(X_train_std, train_sel, core='M')
            y_tr, _ = _stack_rows(y_train_std, train_sel, core=None)
            y_tr = y_tr.ravel()
            keep = np.all(np.isfinite(X_tr), axis=1) & np.isfinite(y_tr)

            X_te, pos = _stack_rows(X_test_std, test_rows == c, core='M')
            keep_te = np.all(np.isfinite(X_te), axis=1)
            if not keep.any() or not keep_te.any():
                return c, None, None, None

            mlp_c = MLPRegressor(
                hidden_layer_sizes=bp['hidden_layer_sizes'],
                learning_rate_init=bp.get('learning_rate_init', 0.001),
                activation=bp['activation'], solver=bp['solver'], alpha=bp['alpha'],
                early_stopping=False,
                validation_fraction=self.validation_fraction,
                n_iter_no_change=self.n_iter_no_change,
                max_iter=self.max_iter, random_state=self.random_state)
            mlp_c.fit(X_tr[keep], y_tr[keep])
            return c, mlp_c, pos[keep_te], mlp_c.predict(X_te[keep_te])

        if workers > 1 and len(active) > 1:
            results = Parallel(n_jobs=min(workers, len(active)), backend="threading")(
                delayed(_fit_one)(c) for c in active
            )
        else:
            results = [_fit_one(c) for c in active]

        for c, model, pos, vals in results:
            if model is not None:
                self.mlp[c] = model
            if pos is not None:
                predictions[pos] = vals
        predictions = predictions.reshape(n_time, n_lat, n_lon)

        predicted_da = xr.DataArray(predictions, coords={'T': time, 'Y': lat, 'X': lon},
                                    dims=['T', 'Y', 'X'])
        return _maybe_calibrate_amplitude(
            self, self.mlp, X_train, y_train, predicted_da, cluster_da, X_test=X_test,
            raw_predictors=_raw_predictors, best_params=best_params,
            clim_year_start=clim_year_start, clim_year_end=clim_year_end)

    # ------------------ Probability calculation methods ------------------
    @staticmethod
    def _ppf_terciles_from_code(dist_code, shape, loc, scale):
        """Tercile thresholds (T1, T2) from best-fit family parameters."""
        if np.isnan(dist_code):
            return np.nan, np.nan
        code = int(dist_code)
        try:
            if code == 1:
                return (norm.ppf(1/3, loc=loc, scale=scale),
                        norm.ppf(2/3, loc=loc, scale=scale))
            elif code == 2:
                return (lognorm.ppf(1/3, s=shape, loc=loc, scale=scale),
                        lognorm.ppf(2/3, s=shape, loc=loc, scale=scale))
            elif code == 3:
                return (expon.ppf(1/3, loc=loc, scale=scale),
                        expon.ppf(2/3, loc=loc, scale=scale))
            elif code == 4:
                return (gamma.ppf(1/3, a=shape, loc=loc, scale=scale),
                        gamma.ppf(2/3, a=shape, loc=loc, scale=scale))
            elif code == 5:
                return (weibull_min.ppf(1/3, c=shape, loc=loc, scale=scale),
                        weibull_min.ppf(2/3, c=shape, loc=loc, scale=scale))
            elif code == 6:
                return (t.ppf(1/3, df=shape, loc=loc, scale=scale),
                        t.ppf(2/3, df=shape, loc=loc, scale=scale))
            elif code == 7:
                return (poisson.ppf(1/3, mu=shape, loc=loc),
                        poisson.ppf(2/3, mu=shape, loc=loc))
            elif code == 8:
                return (nbinom.ppf(1/3, n=shape, p=scale, loc=loc),
                        nbinom.ppf(2/3, n=shape, p=scale, loc=loc))
        except Exception:
            return np.nan, np.nan
        return np.nan, np.nan

    @staticmethod
    def weibull_shape_solver(k, M, V):
        """Root function for the Weibull shape k matching observed V/M^2."""
        if k <= 0:
            return -np.inf
        try:
            g1 = gamma_function(1 + 1/k)
            g2 = gamma_function(1 + 2/k)
            implied = (g2 / (g1**2)) - 1
            observed = V / (M**2)
            return observed - implied
        except ValueError:
            return -np.inf

    @staticmethod
    def calculate_tercile_probabilities_bestfit(best_guess, error_variance, T1, T2, dist_code, dof):
        """Tercile probabilities from a forecast-adjusted best-fit family."""
        best_guess = np.asarray(best_guess, float)
        error_variance = np.asarray(error_variance, dtype=float)
        n_time = best_guess.size
        out = np.full((3, n_time), np.nan, float)

        if (np.all(np.isnan(best_guess)) or np.isnan(dist_code)
                or np.isnan(T1) or np.isnan(T2) or np.isnan(error_variance)):
            return out

        code = int(dist_code)

        if code == 1:  # Normal
            error_std = np.sqrt(error_variance)
            out[0, :] = norm.cdf(T1, loc=best_guess, scale=error_std)
            out[1, :] = norm.cdf(T2, loc=best_guess, scale=error_std) - out[0, :]
            out[2, :] = 1 - norm.cdf(T2, loc=best_guess, scale=error_std)

        elif code == 2:  # Lognormal
            sigma = np.sqrt(np.log(1 + error_variance / (best_guess**2)))
            mu = np.log(best_guess) - sigma**2 / 2
            c1 = lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
            c2 = lognorm.cdf(T2, s=sigma, scale=np.exp(mu))
            out[0, :] = c1; out[1, :] = c2 - c1; out[2, :] = 1 - c2

        elif code == 3:  # Exponential  (B: loc = best_guess, not loc_t)
            scale = np.sqrt(error_variance)
            c1 = expon.cdf(T1, loc=best_guess, scale=scale)
            c2 = expon.cdf(T2, loc=best_guess, scale=scale)
            out[0, :] = c1; out[1, :] = c2 - c1; out[2, :] = 1.0 - c2

        elif code == 4:  # Gamma
            alpha = (best_guess ** 2) / error_variance
            theta = error_variance / best_guess
            c1 = gamma.cdf(T1, a=alpha, scale=theta)
            c2 = gamma.cdf(T2, a=alpha, scale=theta)
            out[0, :] = c1; out[1, :] = c2 - c1; out[2, :] = 1.0 - c2

        elif code == 5:  # Weibull  (C: qualified staticmethod, no prints)
            V = float(error_variance)
            for i in range(n_time):
                M = best_guess[i]
                if not np.isfinite(M) or V <= 0 or M <= 0:
                    continue
                k = fsolve(WAS_mme_MLP.weibull_shape_solver, 2.0, args=(M, V))[0]
                if k <= 0:
                    continue
                lambda_scale = M / gamma_function(1 + 1/k)
                c1 = weibull_min.cdf(T1, c=k, loc=0, scale=lambda_scale)
                c2 = weibull_min.cdf(T2, c=k, loc=0, scale=lambda_scale)
                out[0, i] = c1; out[1, i] = c2 - c1; out[2, i] = 1.0 - c2

        elif code == 6:  # Student-t
            if dof <= 2:
                return out
            loc = best_guess
            scale = np.sqrt(error_variance * (dof - 2) / dof)
            c1 = t.cdf(T1, df=dof, loc=loc, scale=scale)
            c2 = t.cdf(T2, df=dof, loc=loc, scale=scale)
            out[0, :] = c1; out[1, :] = c2 - c1; out[2, :] = 1.0 - c2

        elif code == 7:  # Poisson
            mu = best_guess
            c1 = poisson.cdf(T1, mu=mu)
            c2 = poisson.cdf(T2, mu=mu)
            out[0, :] = c1; out[1, :] = c2 - c1; out[2, :] = 1.0 - c2

        elif code == 8:  # Negative Binomial
            p = np.where(error_variance > best_guess, best_guess / error_variance, np.nan)
            n = np.where(error_variance > best_guess,
                         (best_guess**2) / (error_variance - best_guess), np.nan)
            c1 = nbinom.cdf(T1, n=n, p=p)
            c2 = nbinom.cdf(T2, n=n, p=p)
            out[0, :] = c1; out[1, :] = c2 - c1; out[2, :] = 1.0 - c2

        else:
            raise ValueError("Invalid distribution")

        # F: positive-support families are meaningless for a non-positive mean
        if code in (2, 3, 4, 5, 7, 8):
            invalid = ~(np.isfinite(best_guess) & (best_guess > 0))
            out[:, invalid] = np.nan

        return out

    @staticmethod
    def calculate_tercile_probabilities_nonparametric(best_guess, error_samples, first_tercile, second_tercile):
        return WAS_mme_Weighted.calculate_tercile_probabilities_nonparametric(
            best_guess, error_samples, first_tercile, second_tercile)

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det,
                     best_code_da=None, best_shape_da=None,
                     best_loc_da=None, best_scale_da=None):
        """Tercile probabilities for a (cross-validated) deterministic hindcast."""
        if "M" in Predictant.dims:
            Predictant = Predictant.isel(M=0, drop=True)
        Predictant = Predictant.transpose("T", "Y", "X")

        mask = xr.where(np.isfinite(Predictant).sum(dim='T') >= 3, 1.0, np.nan)
        clim = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))
        if clim.sizes.get("T", 0) < 3:
            raise ValueError("Not enough years in climatology period for terciles.")

        error_variance = ((Predictant - hindcast_det) ** 2).mean(dim="T", skipna=True)
        dof = max(int(clim.sizes["T"]) - 1, 2)

        terciles_emp = climatological_terciles(clim)[0]
        T1_emp = terciles_emp.isel(quantile=0, drop=True)
        T2_emp = terciles_emp.isel(quantile=1, drop=True)

        dm = self.dist_method
        if dm == "bestfit":
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError("dist_method='bestfit' requires best_code_da, "
                                 "best_shape_da, best_loc_da, best_scale_da.")
            # Category boundaries stay on the observed climatological reference.
            # The selected distribution controls the predictive CDF only.
            if getattr(self, "tercile_reference", "climatology") == "training":
                T1, T2 = _pufunc(
                    self._ppf_terciles_from_code,
                    best_code_da, best_shape_da, best_loc_da, best_scale_da,
                    input_core_dims=[(), (), (), ()],
                    output_core_dims=[(), ()], vectorize=True,
                    dask="parallelized", output_dtypes=[float, float])
            else:
                T1, T2 = T1_emp, T2_emp
            hindcast_prob = _pufunc(
                self.calculate_tercile_probabilities_bestfit,
                hindcast_det, error_variance, T1, T2, best_code_da,
                input_core_dims=[("T",), (), (), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True, kwargs={'dof': dof}, dask="parallelized",
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True})

        elif dm == "nonparam":
            error_samples = (Predictant - hindcast_det).rename({"T": "S"})   # D
            hindcast_prob = _pufunc(
                self.calculate_tercile_probabilities_nonparametric,
                hindcast_det, error_samples, T1_emp, T2_emp,
                input_core_dims=[("T",), ("S",), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True, dask="parallelized", output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True})
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}")

        hindcast_prob = hindcast_prob.assign_coords(probability=("probability", ["PB", "PN", "PA"]))
        return (hindcast_prob * mask).transpose("probability", "T", "Y", "X")

    @staticmethod
    def _reshape_and_filter_data(da):
        """Stack (T,Y,X[,M]) -> (n_samples, n_features) and drop NaN rows."""
        da_stacked = da.stack(sample=('T', 'Y', 'X'))
        if 'M' in da.dims:
            da_stacked = da_stacked.transpose('sample', 'M')
        else:
            da_stacked = da_stacked.transpose('sample')
        vals = da_stacked.values
        nan_mask = np.any(np.isnan(vals), axis=1) if vals.ndim > 1 else np.isnan(vals)
        return vals[~nan_mask], nan_mask, vals

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det,
                 hindcast_det_cross, Predictor_for_year, best_params=None,
                 cluster_da=None, best_code_da=None, best_shape_da=None,
                 best_loc_da=None, best_scale_da=None):
        """Deterministic + probabilistic forecast for a single target year."""
        validate_mme_predictor_schema(hindcast_det, Predictor_for_year)
        if "M" in Predictant.dims:
            Predictant_no_m = Predictant.isel(M=0, drop=True)
        else:
            Predictant_no_m = Predictant
        mask = xr.where(np.isfinite(Predictant_no_m).sum(dim='T') >= 3, 1.0, np.nan).to_numpy()

        hindcast_det_st, Predictor_for_year_st = standardize_mme_predictors(
            hindcast_det, Predictor_for_year, clim_year_start, clim_year_end)
        Predictant_st = standardize_timeseries(Predictant_no_m, clim_year_start, clim_year_end)
        hindcast_det_st['T'] = Predictant_st['T']

        time = Predictor_for_year_st['T']; lat = Predictor_for_year_st['Y']; lon = Predictor_for_year_st['X']
        n_time, n_lat, n_lon = len(time), len(lat), len(lon)

        best_params, cluster_da = _resolve_regional_params(
            self, hindcast_det, Predictant_no_m, clim_year_start, clim_year_end,
            best_params, cluster_da)
        hindcast_det_st, Predictor_for_year_st = _prepare_channels(
            self, hindcast_det_st, Predictor_for_year_st, Predictant_no_m, None,
            clim_year_start, clim_year_end)

        predictions = np.full((n_time, n_lat, n_lon), np.nan)
        self.mlp = {}
        for c in range(self.n_clusters):
            if c not in best_params:
                continue
            bp = best_params[c]
            mask_3d_train = (cluster_da == c).expand_dims({'T': hindcast_det_st['T']})
            mask_3d_test = (cluster_da == c).expand_dims({'T': Predictor_for_year_st['T']})

            X_tr = _stack_masked(hindcast_det_st, mask_3d_train, core='M')
            y_tr = _stack_masked(Predictant_st, mask_3d_train, core=None).ravel()
            tr_bad = np.any(~np.isfinite(X_tr), axis=1) | ~np.isfinite(y_tr)
            X_tr_c, y_tr_c = X_tr[~tr_bad], y_tr[~tr_bad]

            X_te = _stack_masked(Predictor_for_year_st, mask_3d_test, core='M')
            te_bad = np.any(~np.isfinite(X_te), axis=1)
            if len(X_tr_c) == 0 or (~te_bad).sum() == 0:
                continue

            mlp_c = MLPRegressor(
                hidden_layer_sizes=bp['hidden_layer_sizes'],
                learning_rate_init=bp.get('learning_rate_init', 0.001),
                activation=bp['activation'], solver=bp['solver'], alpha=bp['alpha'],
                early_stopping=False,
                validation_fraction=self.validation_fraction,
                n_iter_no_change=self.n_iter_no_change,
                max_iter=self.max_iter, random_state=self.random_state)
            mlp_c.fit(X_tr_c, y_tr_c)
            self.mlp[c] = mlp_c

            y_pred_c = mlp_c.predict(X_te[~te_bad])
            result_c = np.full(X_te.shape[0], np.nan)
            result_c[~te_bad] = y_pred_c
            predictions = np.where(np.isnan(predictions),
                                   result_c.reshape(n_time, n_lat, n_lon), predictions)

        result_da = xr.DataArray(predictions, coords={'T': time, 'Y': lat, 'X': lon},
                                 dims=['T', 'Y', 'X']) * mask
        result_da = reverse_standardize(result_da, Predictant_no_m, clim_year_start, clim_year_end)

        year = Predictor_for_year.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970
        month_1 = Predictant_no_m.isel(T=0).coords['T'].values.astype('datetime64[M]').astype(int) % 12 + 1
        new_T_value = np.datetime64(f"{year}-{month_1:02d}-01")
        result_da = result_da.assign_coords(T=xr.DataArray([new_T_value], dims=["T"]))
        result_da['T'] = result_da['T'].astype('datetime64[ns]')

        index_start = Predictant_no_m.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant_no_m.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Predictant_no_m.isel(T=slice(index_start, index_end))
        terciles = climatological_terciles(rainfall_for_tercile)[0]
        T1_emp = terciles.isel(quantile=0, drop=True)
        T2_emp = terciles.isel(quantile=1, drop=True)
        error_variance = ((Predictant_no_m - hindcast_det_cross) ** 2).mean(dim="T", skipna=True)   # D (OOS)
        dof = max(int(rainfall_for_tercile.sizes["T"]) - 1, 2)

        dm = self.dist_method
        if dm == "bestfit":
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError("dist_method='bestfit' requires best_code_da, "
                                 "best_shape_da, best_loc_da, best_scale_da.")
            # Category boundaries stay on the observed climatological reference.
            # The selected distribution controls the predictive CDF only.
            if getattr(self, "tercile_reference", "climatology") == "training":
                T1, T2 = _pufunc(
                    self._ppf_terciles_from_code,
                    best_code_da, best_shape_da, best_loc_da, best_scale_da,
                    input_core_dims=[(), (), (), ()], output_core_dims=[(), ()],
                    vectorize=True, dask="parallelized", output_dtypes=[float, float])
            else:
                T1, T2 = T1_emp, T2_emp
            forecast_prob = _pufunc(
                self.calculate_tercile_probabilities_bestfit,
                result_da, error_variance, T1, T2, best_code_da,
                input_core_dims=[("T",), (), (), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True, dask="parallelized", kwargs={"dof": dof},
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True})

        elif dm == "nonparam":
            error_samples = (Predictant_no_m - hindcast_det_cross).rename({"T": "S"})  # D
            forecast_prob = _pufunc(
                self.calculate_tercile_probabilities_nonparametric,
                result_da, error_samples, T1_emp, T2_emp,
                input_core_dims=[("T",), ("S",), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True, dask="parallelized", output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True})
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}")

        forecast_prob = forecast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return result_da * mask, mask * forecast_prob.transpose('probability', 'T', 'Y', 'X')

class _WAS_mme_SklearnRegressor:
    """Leakage-aware base for low/medium-complexity seasonal ML challengers.

    The key design choice is *year-grouped* tuning while pooling spatial cells
    only to estimate one relationship per homogeneous climate zone.  Pooled
    pixels increase numerical stability but are never treated as independent
    years by CV.
    """
    estimator_name = "base"
    param_grid = {}
    # Probability kernels used by the strict cross-validator: the same ones as
    # compute_prob / forecast, which delegate to WAS_mme_Weighted.
    _ppf_terciles_from_code = staticmethod(_ppf_terciles_from_code)

    @staticmethod
    def calculate_tercile_probabilities_bestfit(best_guess, error_variance, T1, T2, dist_code, dof):
        return WAS_mme_Weighted.calculate_tercile_probabilities_bestfit(
            best_guess, error_variance, T1, T2, dist_code, dof)

    def __init__(self, n_clusters=4, cv_folds=4, leave_one_year_out=False,
                 dist_method="nonparam", random_state=42, scoring="neg_mean_squared_error",
                 n_jobs=-1, min_train_years=12, spatial_subsample=1, spatial_mode="zones",
                 covariate_names=None, static_fields=None, coord_step=None,
                 zones_as_features=False, predictor_smoothing=0, one_se_rule=False,
                 amplitude_calibration=True, amplitude_calibration_inner_folds=3):
        self.n_clusters=int(n_clusters); self.cv_folds=int(cv_folds)
        self.leave_one_year_out=bool(leave_one_year_out); self.dist_method=dist_method
        self.random_state=int(random_state); self.scoring=scoring; self.n_jobs=n_jobs
        self.min_train_years=int(min_train_years); self.best_params_=None; self.cluster_da_=None
        self.spatial_subsample=max(1,int(spatial_subsample))
        _init_spatial_options(self, spatial_mode, covariate_names, static_fields,
                              coord_step, zones_as_features, predictor_smoothing, one_se_rule,
                              amplitude_calibration, amplitude_calibration_inner_folds)

    def _make_estimator(self, **params):
        raise NotImplementedError

    def _cv(self, groups):
        return _year_grouped_splitter(groups,self.cv_folds,self.leave_one_year_out)

    def compute_hyperparameters(self, Predictors, Predictand,
                                clim_year_start=None, clim_year_end=None):
        """Tune with fold-local standardization, target encoding and climate zones."""
        return _strict_regional_hyperparameters(
            self, Predictors, Predictand, clim_year_start, clim_year_end)

    def compute_hyperparameters_legacy(self, Predictors, Predictand, clim_year_start=None, clim_year_end=None):
        Predictors = align_predictor_target(Predictors, Predictand, mode="season_year")
        if "M" in Predictand.dims: Predictand=Predictand.isel(M=0,drop=True)
        # Climate zones are defined from RAW training observations, but learners
        # are tuned in fold-local standardized space when a climatology is given.
        cluster_da=_build_climate_feature_clusters(Predictand,self.n_clusters,self.random_state)
        if clim_year_start is not None and clim_year_end is not None:
            Xfit=standardize_timeseries(Predictors,clim_year_start,clim_year_end)
            yfit=standardize_timeseries(Predictand,clim_year_start,clim_year_end)
        else:
            Xfit=Predictors; yfit=Predictand
        Xfit=Xfit.assign_coords(T=yfit["T"])
        params={}
        for c in np.unique(cluster_da.values[np.isfinite(cluster_da.values)]):
            mask=(cluster_da==c).expand_dims(T=Xfit["T"])
            X=Xfit.where(mask).stack(sample=("T","Y","X")).transpose("sample","M")
            y=yfit.where(mask).stack(sample=("T","Y","X"))
            groups=_calendar_year_groups(X["T"])
            xv=X.values; yv=y.values.ravel(); good=np.all(np.isfinite(xv),axis=1)&np.isfinite(yv)
            xv=xv[good]; yv=yv[good]; groups=np.asarray(groups)[good]
            ny=len(np.unique(groups))
            if ny < self.min_train_years or xv.shape[0] < 20:
                warnings.warn(f"{self.estimator_name} cluster {int(c)} skipped: only {ny} years.",RuntimeWarning)
                continue
            _effective_sample_warning(ny,xv.shape[1],self.estimator_name)
            gs=GridSearchCV(self._make_estimator(),self.param_grid,cv=self._cv(groups),
                            scoring=self.scoring,n_jobs=self.n_jobs,error_score=np.nan)
            gs.fit(xv,yv,groups=groups)
            params[int(c)]=gs.best_params_
        self.best_params_=params; self.cluster_da_=cluster_da
        return params,cluster_da

    def compute_model(self, X_train, y_train, X_test, y_test=None,
                      clim_year_start=None, clim_year_end=None,
                      best_params=None, cluster_da=None, covariates=None):
        validate_mme_predictor_schema(X_train, X_test)
        X_train = align_predictor_target(X_train, y_train, mode="season_year")
        if "M" in y_train.dims: y_train=y_train.isel(M=0,drop=True)
        best_params,cluster_da=_resolve_regional_params(
            self,X_train,y_train,clim_year_start,clim_year_end,best_params,cluster_da)
        _raw_predictors=(X_train,X_test)
        X_train,X_test=_prepare_channels(self,X_train,X_test,y_train,covariates,
                                         clim_year_start,clim_year_end)
        out=np.full((X_test.sizes["T"],X_test.sizes["Y"],X_test.sizes["X"]),np.nan)
        # les estimateurs ajustes sont conserves par cluster pour que la
        # calibration d'amplitude rejoue exactement le meme chemin de prediction
        self.fitted_={}
        for c,bp in best_params.items():
            mtr=(cluster_da==c).expand_dims(T=X_train["T"])
            if self.spatial_subsample>1:
                keep=xr.zeros_like(cluster_da,dtype=bool)
                keep[::self.spatial_subsample,::self.spatial_subsample]=True
                mtr=mtr&keep
            mte=(cluster_da==c).expand_dims(T=X_test["T"])
            Xtr=X_train.where(mtr).stack(sample=("T","Y","X")).transpose("sample","M")
            ytr=y_train.where(mtr).stack(sample=("T","Y","X"))
            Xte=X_test.where(mte).stack(sample=("T","Y","X")).transpose("sample","M")
            a=Xtr.values; b=ytr.values.ravel(); q=Xte.values
            g=np.all(np.isfinite(a),axis=1)&np.isfinite(b); gt=np.all(np.isfinite(q),axis=1)
            if not g.any() or not gt.any(): continue
            est=self._make_estimator(**bp); est.fit(a[g],b[g]); pred=est.predict(q[gt])
            self.fitted_[c]=est
            full=np.full(q.shape[0],np.nan); full[gt]=pred
            cube=full.reshape(X_test.sizes["T"],X_test.sizes["Y"],X_test.sizes["X"])
            out=np.where(np.isfinite(cube),cube,out)
        predicted_da=xr.DataArray(out,dims=("T","Y","X"),
                                  coords={"T":X_test["T"],"Y":X_test["Y"],"X":X_test["X"]})
        return _maybe_calibrate_amplitude(
            self,self.fitted_,X_train,y_train,predicted_da,cluster_da,X_test=X_test,
            raw_predictors=_raw_predictors,best_params=best_params,
            clim_year_start=clim_year_start,clim_year_end=clim_year_end)

    @staticmethod
    def calculate_tercile_probabilities_bestfit(
        best_guess, error_variance, T1, T2, dist_code, dof
    ):
        """Expose the same predictive CDF used by compute_prob and forecast.

        Native cross-validation supplies its declared climatological thresholds
        and training-only residual variance directly to this cellwise kernel.
        """
        return WAS_mme_Weighted.calculate_tercile_probabilities_bestfit(
            best_guess, error_variance, T1, T2, dist_code, dof
        )

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det,
                     best_code_da=None,best_shape_da=None,best_loc_da=None,best_scale_da=None):
        helper=WAS_mme_Weighted(equal_weighted=True,dist_method=self.dist_method)
        return helper.compute_prob(Predictant,clim_year_start,clim_year_end,hindcast_det,
                                   best_code_da,best_shape_da,best_loc_da,best_scale_da)

    def forecast(self, Predictant, clim_year_start, clim_year_end, hindcast_det,
                 hindcast_det_cross, Predictor_for_year, best_params=None, cluster_da=None,
                 best_code_da=None,best_shape_da=None,best_loc_da=None,best_scale_da=None):
        # Contract aligned with RF/XGB/hpELM/MLP: predictor history = hindcast_det,
        # OOS residuals = hindcast_det_cross, target predictors = Predictor_for_year.
        validate_mme_predictor_schema(hindcast_det, Predictor_for_year)
        if "M" in Predictant.dims: Predictant=Predictant.isel(M=0,drop=True)
        Xhist, Xyear = standardize_mme_predictors(
            hindcast_det, Predictor_for_year, clim_year_start, clim_year_end)
        # Zones, hyperparameters and spatial covariates come from the RAW
        # predictand (the standardized target below would give KMeans nothing
        # but sampling noise to partition).
        hindcast_det=align_predictor_target(hindcast_det,Predictant,mode="season_year")
        best_params,cluster_da=_resolve_regional_params(
            self,hindcast_det,Predictant,clim_year_start,clim_year_end,best_params,cluster_da)
        covariates=_model_covariates(self,Predictant,clim_year_start,clim_year_end)
        my=Predictant.sel(T=slice(str(clim_year_start),str(clim_year_end))).mean("T")
        sy=Predictant.sel(T=slice(str(clim_year_start),str(clim_year_end))).std("T").where(lambda z:z>0)
        yst=(Predictant-my)/sy
        predst=self.compute_model(Xhist,yst,Xyear,best_params=best_params,cluster_da=cluster_da,
                                  covariates=covariates)
        pred=predst*sy+my; pred=_assign_target_time(pred,Predictor_for_year,Predictant)
        err=Predictant-hindcast_det_cross.transpose("T","Y","X")
        helper=WAS_mme_Weighted(equal_weighted=True,dist_method=self.dist_method)
        # Probability dressing from genuinely OOS residuals; use helper's internal core.
        prob=helper._probability_core(Predictant,hindcast_det_cross,pred,clim_year_start,clim_year_end,
                                      best_code_da,best_shape_da,best_loc_da,best_scale_da)
        return apply_physical_support(pred, Predictant),prob


class WAS_mme_Ridge(_WAS_mme_SklearnRegressor):
    """Strong low-variance ML baseline; often hard to beat with short hindcasts."""
    estimator_name="Ridge"
    param_grid={"alpha":[0.01,0.1,1.0,10.0,100.0]}
    def _make_estimator(self,**params): return Ridge(**params)


class WAS_mme_ElasticNet(_WAS_mme_SklearnRegressor):
    """Sparse/shrunk multi-model regression for correlated GCM predictors."""
    estimator_name="ElasticNet"
    param_grid={"alpha":[0.001,0.01,0.1,1.0],"l1_ratio":[0.0,0.25,0.5,0.75,1.0]}
    def _make_estimator(self,**params):
        return ElasticNet(max_iter=10000,random_state=self.random_state,**params)



# ===========================================================================
# Kharin et al. (2017): 7-target-season fixed-lead WASS2S adaptation
# One source of truth for direct-module and was_mme public imports.
from .was_mme_kharin2017_v3_7seasons import WAS_mme_Kharin2017


class WAS_mme_PLS(_WAS_mme_SklearnRegressor):
    """Partial Least Squares — built for exactly this regime: correlated
    predictors, few independent years.

    Unlike PCR/CCA, which choose their predictor axes from the predictors'
    own variance alone, PLS chooses them to maximise covariance with the
    predictand directly. On a short, highly-correlated GCM ensemble this is
    typically more stable than a variance-based truncation search, and it
    sidesteps the mode-count selection problem CCA needs a one-SE rule for.
    Input is standardized upstream by the shared training harness, so PLS's
    own internal scaling is disabled to avoid a second, redundant rescaling.
    """
    estimator_name="PLS"
    param_grid={"n_components":[1,2,3,4]}
    def _make_estimator(self,**params):
        return PLSRegression(scale=False, **params)


class WAS_mme_SVR(_WAS_mme_SklearnRegressor):
    """Regularised nonlinear seasonal MME using support-vector regression."""
    estimator_name="SVR"
    param_grid={"C":[0.1,1.0,10.0],"epsilon":[0.05,0.1,0.2],"gamma":["scale",0.01,0.1],
                "kernel":["linear","rbf"]}
    def _make_estimator(self,**params): return SVR(**params)


class _SklearnMARS(RegressorMixin, _MARSCore, BaseEstimator):
    """Thin sklearn-compatible shell around wass2s.was_machine_learning.MARS.

    _MARSCore stores every __init__ argument as a same-named attribute, which
    is exactly what BaseEstimator.get_params()/clone() need — no other
    adaptation is required for GridSearchCV to treat it like any sklearn
    regressor.
    """
    pass


class WAS_mme_MARS(_WAS_mme_SklearnRegressor):
    """Piecewise-linear (hinge-function) seasonal MME.

    Sits between the pure-linear challengers (Ridge/ElasticNet) and the
    high-capacity ones (RF/XGB/MLP/hpELM): MARS can express a modest,
    interpretable non-linearity (a small number of hinge functions, GCV-
    penalised) without the overfitting risk a kernel or a tree ensemble
    carries on a hindcast this short. A reasonable default choice when a
    linear challenger underperforms but the sample does not support a
    high-capacity model.
    """
    estimator_name="MARS"
    param_grid={"max_terms":[5,9,13],"max_degree":[1,2],"penalty":[2.0,3.0,4.0]}
    def _make_estimator(self,**params): return _SklearnMARS(**params)


class WAS_mme_ExtraTrees(_WAS_mme_SklearnRegressor):
    """Highly randomised trees; useful RF challenger with strong regularisation."""
    estimator_name="ExtraTrees"
    param_grid={"n_estimators":[300,600],"max_depth":[4,8,None],
                "min_samples_leaf":[5,10,20],"max_features":[0.5,0.8,1.0]}
    def _make_estimator(self,**params):
        return ExtraTreesRegressor(random_state=self.random_state,n_jobs=1,**params)


class WAS_mme_HistGradientBoosting(_WAS_mme_SklearnRegressor):
    """Regularised histogram boosting; lighter than XGBoost and dependency-free."""
    estimator_name="HistGradientBoosting"
    param_grid={"learning_rate":[0.03,0.05,0.1],"max_leaf_nodes":[7,15,31],
                "l2_regularization":[0.1,1.0,10.0],"min_samples_leaf":[10,20,40]}
    def _make_estimator(self,**params):
        return HistGradientBoostingRegressor(random_state=self.random_state,max_iter=300,early_stopping=False,**params)


class WAS_mme_PCR:
    """
    Principal-component-regression MME baseline.

    Predictor field = the multi-model mean (over M) on the predictand grid.
    EOFs are fitted on the training fold (via the fold-safe WAS_EOF), the
    predictand is regressed on the leading PCs, and the held-out fold is
    projected out-of-sample before applying the regression. The deterministic
    output of ``compute_model`` is in the same space as its training target.
    The native cross-validation driver fits scaling and mode selection within
    each training partition. ``forecast`` applies the same scaling convention
    and calibrates future probabilities from supplied out-of-fold residuals.

    Parameters
    ----------
    n_modes : int
        Number of principal components retained (default 5).
    use_coslat : bool
        Cosine-latitude weighting in the EOF (default True).
    detrend : bool
        Detrend the predictor field inside each fold (default False; the
        framework already standardizes, and the RF baseline does not detrend,
        so leave False for a like-for-like comparison).
    dist_method : str
        'nonparam' (default) or 'bestfit'.
    combine : str
        How the M models enter the EOF:
          'mean' (default) -- predictor field = multi-model mean (one channel).
          'meof'           -- keep every model as its own channel and run a
                              single combined / multivariate EOF over the joint
                              (Y, X, M) feature space. Because the framework has
                              already standardized each (cell, model) to unit
                              variance, the channels share a scale, so the
                              stacked EOF is a proper MEOF (no per-field
                              normalization needed). The leading PCs then encode
                              the joint inter-model variability, and the
                              predictand is regressed on them.
    """

    def __init__(self, n_modes=5, use_coslat=True, combine="mean", detrend=False,
                 dist_method="nonparam", L2norm=False, random_state=42,
                 positive_support=True):
        if combine not in ("mean", "meof"):
            raise ValueError("combine must be 'mean' or 'meof'.")
        self.n_modes = n_modes
        self.use_coslat = use_coslat
        self.combine = combine
        self.detrend = detrend
        self.dist_method = dist_method
        self.L2norm = L2norm
        self.random_state = random_state
        self.positive_support = bool(positive_support)
        self.eof = None
        # Keep the prespecified search range after a full-record finalization.
        # A selected value must not silently shrink the next search space.
        self._mode_search_upper = int(n_modes)

    def _field(self, X):
        """Build the EOF input field from the predictor cube (T, Y, X, M)."""
        if "M" not in X.dims:
            return X
        if self.combine == "meof":
            return X                 # keep models as channels -> combined EOF
        return X.mean("M")           # default: multi-model mean

    def _fit_predict_pcs(self, field_train, y_train, field_test):
        """Fit EOF on train field, regress y_train on PCs, predict the test field."""
        available = np.isfinite(field_test).any([d for d in field_test.dims if d != "T"])
        n_eff = max(1, min(int(self.n_modes), max(1, int(field_train.sizes["T"]) - 2)))
        if n_eff < int(self.n_modes):
            warnings.warn(f"PCR n_modes capped from {self.n_modes} to {n_eff} for this fold.",
                          RuntimeWarning, stacklevel=2)
        eof = WAS_EOF(n_modes=n_eff, use_coslat=self.use_coslat,
                      detrend=self.detrend, standardize=False, L2norm=self.L2norm,
                      random_state=self.random_state)
        self.eof = eof
        from .was_eof import EOFRankError
        try:
            _, s_train, _ = eof.fit(field_train, dim="T")
            s_test = eof.transform(field_test, dim="T")
        except EOFRankError:
            # No varying predictor direction remains. The estimable PCR model
            # is its intercept, not a regression on numerical noise.
            self.rank_zero_fallback_ = True
            mean = y_train.mean("T", skipna=True).where(y_train.notnull().sum("T") >= 3)
            return mean.expand_dims(T=field_test["T"]).where(available).transpose("T", "Y", "X")
        self.rank_zero_fallback_ = False

        S_tr = s_train.transpose("T", "mode").sel(T=field_train["T"]).values
        S_te = s_test.transpose("T", "mode").values
        A_tr = np.column_stack([np.ones(S_tr.shape[0]), S_tr])
        A_te = np.column_stack([np.ones(S_te.shape[0]), S_te])

        ystk = y_train.transpose("T", "Y", "X").sel(T=field_train["T"])
        ystk = ystk.stack(cell=("Y", "X")).transpose("T", "cell")
        Y = ystk.values
        pred = np.full((A_te.shape[0], Y.shape[1]), np.nan)
        design_valid = np.isfinite(A_tr).all(axis=1)
        test_valid = np.isfinite(A_te).all(axis=1)
        # Missing targets are local to their cell; they do not invalidate the
        # entire time series. Share one solve for cells with the same row mask.
        masks = np.isfinite(Y) & design_valid[:, None]
        groups = {}
        for j in range(Y.shape[1]):
            groups.setdefault(masks[:, j].tobytes(), []).append(j)
        for columns in groups.values():
            rows = masks[:, columns[0]]
            if rows.sum() < max(3, A_tr.shape[1]):
                continue
            design = A_tr[rows]
            if np.linalg.matrix_rank(design) < design.shape[1]:
                continue
            coef, *_ = np.linalg.lstsq(design, Y[np.ix_(rows, columns)], rcond=None)
            pred[np.ix_(test_valid, columns)] = A_te[test_valid] @ coef

        out = xr.DataArray(pred, dims=("T", "cell"),
                           coords={"T": field_test["T"], "cell": ystk["cell"]}).unstack("cell")
        return out.where(available).transpose("T", "Y", "X")

    def finalize_truncation(self, X, Y, clim_start=None, clim_end=None, splits=None,
                            n_modes_grid=None, nb_omit=1, inner_max_folds=8,
                            score="mse", one_se=True, enabled=True):
        """Select the operational rank with the procedure evaluated inside CV.

        Inputs are RAW historical fields. Each selection fold refits predictor
        and target scaling and the EOF. Selection uses the same paired one-SE
        rule as the outer-validation and calibration fits. ``enabled=False``
        performs no additional fitting and preserves the prescribed rank.
        """
        if not enabled:
            return None
        from .was_cross_validate import _tune_mme_cca_raw
        if "M" in Y.dims:
            if Y.sizes["M"] != 1:
                raise ValueError("PCR target M dimension must be singleton")
            Y = Y.isel(M=0, drop=True)
        X = align_predictor_target(X, Y, mode="season_year")
        upper = max(1, getattr(self, "_mode_search_upper", int(self.n_modes)))
        modes = list(range(1, upper + 1)) if n_modes_grid is None else sorted(set(map(int, n_modes_grid)))
        if not modes or min(modes) < 1:
            raise ValueError("PCR mode candidates must be positive integers")
        result = _tune_mme_cca_raw(
            self, X, Y.transpose("T", "Y", "X"), clim_start, clim_end, nb_omit,
            {"n_modes_grid": modes, "inner_max_folds": inner_max_folds,
             "score": score, "one_se": one_se}, pcr=True, folds=splits)
        self.mode_selection_ = dict(result)
        self.mode_selection_["preprocessing"] = "training-only scaling and EOF in every selection fold"
        return result

    def tune_modes(self, X, Y, splits, n_modes_grid=None, score="mse", one_se=True,
                   progress=False, clim_year_start=None, clim_year_end=None):
        """Compatibility entry point; pass raw fields, as for ``forecast``.

        With no climatological years, scaling uses each selection fold's full
        training record. This is the same raw selector used by cross-validation.
        """
        return self.finalize_truncation(
            X, Y, clim_year_start, clim_year_end, splits=splits,
            n_modes_grid=n_modes_grid, score=score, one_se=one_se)

    # ------------------ Probability Calculation Methods ------------------
    @staticmethod
    def _ppf_terciles_from_code(dist_code, shape, loc, scale):
        if np.isnan(dist_code):
            return np.nan, np.nan
        code = int(dist_code)
        try:
            if code == 1:
                return norm.ppf(1/3, loc=loc, scale=scale), norm.ppf(2/3, loc=loc, scale=scale)
            if code == 2:
                return lognorm.ppf(1/3, s=shape, loc=loc, scale=scale), lognorm.ppf(2/3, s=shape, loc=loc, scale=scale)
            if code == 3:
                return expon.ppf(1/3, loc=loc, scale=scale), expon.ppf(2/3, loc=loc, scale=scale)
            if code == 4:
                return gamma.ppf(1/3, a=shape, loc=loc, scale=scale), gamma.ppf(2/3, a=shape, loc=loc, scale=scale)
            if code == 5:
                return weibull_min.ppf(1/3, c=shape, loc=loc, scale=scale), weibull_min.ppf(2/3, c=shape, loc=loc, scale=scale)
            if code == 6:
                return t.ppf(1/3, df=shape, loc=loc, scale=scale), t.ppf(2/3, df=shape, loc=loc, scale=scale)
            if code == 7:
                return poisson.ppf(1/3, mu=shape, loc=loc), poisson.ppf(2/3, mu=shape, loc=loc)
            if code == 8:
                return nbinom.ppf(1/3, n=shape, p=scale, loc=loc), nbinom.ppf(2/3, n=shape, p=scale, loc=loc)
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
            return (V / (M ** 2)) - ((g2 / (g1 ** 2)) - 1)
        except Exception:
            return -np.inf

    @staticmethod
    def calculate_tercile_probabilities_bestfit(best_guess, error_variance, T1, T2, dist_code, dof):
        best_guess = np.asarray(best_guess, dtype=float)
        error_variance = float(error_variance)
        n_time = best_guess.size
        out = np.full((3, n_time), np.nan, dtype=float)

        if (np.all(np.isnan(best_guess)) or not np.isfinite(error_variance)
                or error_variance < 0 or not np.isfinite(T1) or not np.isfinite(T2)
                or np.isnan(dist_code)):
            return out
        if error_variance == 0:
            valid = np.isfinite(best_guess)
            out[:, valid] = np.vstack((best_guess[valid] <= T1,
                (best_guess[valid] > T1) & (best_guess[valid] <= T2),
                best_guess[valid] > T2)).astype(float)
            return out

        code = int(dist_code)
        try:
            if code == 1:
                s = np.sqrt(error_variance)
                c1, c2 = norm.cdf(T1, loc=best_guess, scale=s), norm.cdf(T2, loc=best_guess, scale=s)
                out[0, :], out[1, :], out[2, :] = c1, c2 - c1, 1.0 - c2
            elif code == 2:
                valid = best_guess > 0
                sigma = np.full(n_time, np.nan); mu = np.full(n_time, np.nan)
                sigma[valid] = np.sqrt(np.log(1.0 + error_variance / (best_guess[valid] ** 2)))
                mu[valid] = np.log(best_guess[valid]) - 0.5 * sigma[valid] ** 2
                c1, c2 = lognorm.cdf(T1, s=sigma, scale=np.exp(mu)), lognorm.cdf(T2, s=sigma, scale=np.exp(mu))
                out[0, :], out[1, :], out[2, :] = c1, c2 - c1, 1.0 - c2
            elif code == 3:
                scale = np.sqrt(error_variance); loc = best_guess - scale
                c1, c2 = expon.cdf(T1, loc=loc, scale=scale), expon.cdf(T2, loc=loc, scale=scale)
                out[0, :], out[1, :], out[2, :] = c1, c2 - c1, 1.0 - c2
            elif code == 4:
                valid = best_guess > 0
                alpha = np.full(n_time, np.nan); theta = np.full(n_time, np.nan)
                alpha[valid] = (best_guess[valid] ** 2) / error_variance
                theta[valid] = error_variance / best_guess[valid]
                c1, c2 = gamma.cdf(T1, a=alpha, scale=theta), gamma.cdf(T2, a=alpha, scale=theta)
                out[0, :], out[1, :], out[2, :] = c1, c2 - c1, 1.0 - c2
            elif code == 5:
                for i in range(n_time):
                    M, V = best_guess[i], error_variance
                    if not np.isfinite(M) or M <= 0 or V <= 0:
                        continue
                    k = fsolve(WAS_mme_PCR.weibull_shape_solver, 2.0, args=(M, V))[0]
                    if not np.isfinite(k) or k <= 0:
                        continue
                    lam = M / gamma_function(1 + 1 / k)
                    c1, c2 = weibull_min.cdf(T1, c=k, loc=0, scale=lam), weibull_min.cdf(T2, c=k, loc=0, scale=lam)
                    out[0, i], out[1, i], out[2, i] = c1, c2 - c1, 1.0 - c2
            elif code == 6:
                if dof <= 2:
                    return out
                scale = np.sqrt(error_variance * (dof - 2) / dof)
                c1, c2 = t.cdf(T1, df=dof, loc=best_guess, scale=scale), t.cdf(T2, df=dof, loc=best_guess, scale=scale)
                out[0, :], out[1, :], out[2, :] = c1, c2 - c1, 1.0 - c2
            elif code == 7:
                mu = np.where(best_guess >= 0, best_guess, np.nan)
                c1, c2 = poisson.cdf(T1, mu=mu), poisson.cdf(T2, mu=mu)
                out[0, :], out[1, :], out[2, :] = c1, c2 - c1, 1.0 - c2
            elif code == 8:
                valid = error_variance > best_guess
                p = np.where(valid, best_guess / error_variance, np.nan)
                n = np.where(valid, best_guess ** 2 / (error_variance - best_guess), np.nan)
                c1, c2 = nbinom.cdf(T1, n=n, p=p), nbinom.cdf(T2, n=n, p=p)
                out[0, :], out[1, :], out[2, :] = c1, c2 - c1, 1.0 - c2
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
    def calculate_tercile_probabilities_nonparametric(best_guess, error_samples, first_tercile, second_tercile):
        return WAS_mme_Weighted.calculate_tercile_probabilities_nonparametric(
            best_guess, error_samples, first_tercile, second_tercile)

    @staticmethod
    def _normalize_probabilities(prob):
        prob = prob.clip(min=0.0, max=1.0)
        total = prob.sum(dim="probability", skipna=False)
        return xr.where(total > 0, prob / total, np.nan)

    def _tercile_probabilities(self, Predictant, deterministic, clim_year_start, clim_year_end,
                               error_samples, error_variance,
                               best_code_da=None, best_shape_da=None,
                               best_loc_da=None, best_scale_da=None):
        Predictant = Predictant.transpose("T", "Y", "X")
        deterministic = deterministic.transpose("T", "Y", "X")
        mask = xr.where((np.isfinite(Predictant).sum(dim='T') >= 3)
                        & (np.isfinite(error_samples).sum(dim='T') >= 3), 1.0, np.nan)

        clim = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))
        if clim.sizes.get("T", 0) < 3:
            raise ValueError("Not enough years in climatology period for terciles.")
        terc = climatological_terciles(clim)[0]
        T1_emp = terc.isel(quantile=0, drop=True)
        T2_emp = terc.isel(quantile=1, drop=True)
        dof = max(int(Predictant.sizes["T"]) - 1, 3)

        if self.dist_method == "bestfit":
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError("dist_method='bestfit' requires best_code_da, best_shape_da, best_loc_da, best_scale_da.")
            # Category boundaries stay on the observed climatological reference.
            # The selected distribution controls the predictive CDF only.
            T1, T2 = T1_emp, T2_emp
            # Match validation exactly: residual mean is a location correction,
            # while the centered sample variance controls dispersion. The
            # second moment would count bias twice and broaden forecasts.
            error_variance = error_samples.var("T", ddof=1, skipna=True)
            calibrated_mean = apply_physical_support(
                deterministic + error_samples.mean("T", skipna=True), Predictant)
            prob = _pufunc(
                self.calculate_tercile_probabilities_bestfit,
                calibrated_mean, error_variance, T1, T2, best_code_da,
                input_core_dims=[("T",), (), (), (), ()], output_core_dims=[("probability", "T")],
                vectorize=True, kwargs={"dof": dof}, dask="parallelized", output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True})
        elif self.dist_method == "nonparam":
            # residual sample axis renamed so a length-1 forecast and the
            # length-n residual record don't collide on the shared 'T' core dim
            err = error_samples.rename({"T": "__samp"})
            prob = _pufunc(
                self.calculate_tercile_probabilities_nonparametric,
                deterministic, err, T1_emp, T2_emp,
                input_core_dims=[("T",), ("__samp",), (), ()], output_core_dims=[("probability", "T")],
                vectorize=True, dask="parallelized", output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True})
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}")

        prob = prob.assign_coords(probability=("probability", ["PB", "PN", "PA"]))
        prob = self._normalize_probabilities(prob.transpose("probability", "T", "Y", "X"))
        return (prob * mask).transpose("probability", "T", "Y", "X")

    # ----------------------------------------------------- framework contract
    def compute_model(self, X_train, y_train, X_test, y_test=None, **kwargs):
        X_train = align_predictor_target(X_train, y_train, mode="season_year")
        if "M" in y_train.dims:
            y_train = y_train.isel(M=0, drop=True)
        field_train = self._field(X_train)
        field_test = self._field(X_test)
        return self._fit_predict_pcs(field_train, y_train, field_test)

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det,
                     best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None):
        if "M" in Predictant.dims:
            Predictant = Predictant.isel(M=0, drop=True)
        Predictant = Predictant.transpose("T", "Y", "X")
        hindcast_det = hindcast_det.transpose("T", "Y", "X")
        # Calibration pairs require matching verification dates, not merely
        # matching lengths. Partial hindcast records retain only known pairs.
        obs, hindcast = xr.align(Predictant, hindcast_det, join="inner")
        if obs.sizes["T"] < 3:
            raise ValueError("PCR calibration needs at least three matching out-of-fold hindcast dates")
        error_samples = obs - hindcast
        error_variance = error_samples.var(dim="T", ddof=1, skipna=True)

        return self._tercile_probabilities(
            Predictant, hindcast_det, clim_year_start, clim_year_end,
            error_samples, error_variance,
            best_code_da, best_shape_da, best_loc_da, best_scale_da)

    # ------------------------------------------------------------- operational
    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor,
                 hindcast_det, Predictor_for_year,
                 best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None):
        """Operational forecast at the supplied target valid times.

        ``hindcast_det`` must contain independently computed out-of-fold
        predictions in physical units. This calibration is for future times;
        use the cross-validator to score historical probabilistic forecasts.
        """


        if "M" in Predictant.dims:
            Predictant = Predictant.isel(M=0, drop=True)
        Predictant = Predictant.transpose("T", "Y", "X")
        mask = xr.where(np.isfinite(Predictant).sum(dim='T') >= 3, 1.0, np.nan)

        from .was_cross_validate import _fold_standardize
        Predictor = align_predictor_target(Predictor, Predictant, mode="season_year")
        Predictor_st, Predictor_year_st, _, _ = _fold_standardize(
            Predictor, Predictor_for_year, clim_year_start, clim_year_end)
        Predictant_st, _, target_mean, target_scale = _fold_standardize(
            Predictant, Predictant, clim_year_start, clim_year_end)

        field_hist = self._field(Predictor_st)
        field_year = self._field(Predictor_year_st)
        pred_st = self._fit_predict_pcs(field_hist, Predictant_st, field_year)

        forecast_det = pred_st * target_scale + target_mean

        # The caller supplies target valid times before EOF projection, so a
        # detrending transform sees the same calendar as the fitted target.
        # Preserve those times, including multi-date and cross-year forecasts.
        if self.positive_support:
            forecast_det = forecast_det.clip(min=0)

        error_samples = Predictant - hindcast_det.transpose("T", "Y", "X")
        error_variance = error_samples.var(dim="T", ddof=1, skipna=True)
        forecast_prob = self._tercile_probabilities(
            Predictant, forecast_det, clim_year_start, clim_year_end,
            error_samples, error_variance,
            best_code_da, best_shape_da, best_loc_da, best_scale_da)

        return forecast_det * mask, forecast_prob * mask

#################### Please from here to down is just to test ################
###### Not approved scientifically please ####################################
############################Expert knowledge #################################

_PROB_LABELS = ["PB", "PN", "PA"]


# ===========================================================================
#  Low-level scoring & per-point optimisation (pure numpy, validated)
# ===========================================================================

def _rps(P: np.ndarray, O: np.ndarray) -> float:
    """Mean Ranked Probability Score for ordered 3-category forecasts.

    P : (n, 3) forecast probabilities (rows sum to 1)
    O : (n, 3) observed one-hot category
    """
    Pc = np.cumsum(P, axis=-1)
    Oc = np.cumsum(O, axis=-1)
    return float(np.mean(np.sum((Pc - Oc) ** 2, axis=-1)))


def _blend(w: np.ndarray, Pstack: np.ndarray) -> np.ndarray:
    """Convex blend. w:(M,)  Pstack:(M, n, 3) -> (n, 3)."""
    return np.tensordot(w, Pstack, axes=(0, 0))


def _project_bounded_simplex(values, lo, hi):
    """Euclidean projection onto sum(w)=1 and lo<=w<=hi."""
    values, lo, hi = (np.asarray(v, dtype=float) for v in (values, lo, hi))
    if values.shape != lo.shape or lo.shape != hi.shape or values.size == 0:
        raise ValueError("Weight vectors must have the same nonempty shape.")
    if not all(np.all(np.isfinite(v)) for v in (values, lo, hi)):
        raise ValueError("Weights and bounds must be finite.")
    if np.any(lo < 0) or np.any(hi > 1) or np.any(lo > hi) or lo.sum() > 1 + 1e-12 or hi.sum() < 1 - 1e-12:
        raise ValueError("Infeasible probability-weight bounds.")
    left, right = float(np.min(values - hi)), float(np.max(values - lo))
    for _ in range(100):
        shift = (left + right) / 2.0
        if np.clip(values - shift, lo, hi).sum() > 1.0:
            left = shift
        else:
            right = shift
    return np.clip(values - (left + right) / 2.0, lo, hi)


def _fit_weights_one_point(
    Pstack: np.ndarray,         # (M, n, 3) per-model tercile probs (train years)
    Ohot: np.ndarray,           # (n, 3) observed one-hot
    lo: np.ndarray,             # (M,) lower bounds
    hi: np.ndarray,             # (M,) upper bounds
    w_prior: np.ndarray,        # (M,) expert prior / shrinkage target (sums to 1)
    lmbda: float,               # shrinkage strength
) -> Tuple[np.ndarray, float]:
    """Minimise CV-RPS + lambda*||w - w_prior||^2 on the expert-box ∩ simplex."""
    M = Pstack.shape[0]

    def objective(w):
        r = _rps(_blend(w, Pstack), Ohot)
        return r + lmbda * float(np.sum((w - w_prior) ** 2))

    bounds = list(zip(lo, hi))
    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    w0 = _project_bounded_simplex(w_prior, lo, hi)

    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-12})
    w = res.x if (res.success and np.all(np.isfinite(res.x))) else w0
    w = _project_bounded_simplex(w, lo, hi)
    return w, _rps(_blend(w, Pstack), Ohot)


# ===========================================================================
#  Public class
# ===========================================================================

class WAS_mme_ExpertKnowledge:
    """Expert-bounded, RPS-optimal multi-model tercile-probability blend.

    Parameters
    ----------
    expert_bounds : dict {model_name: (lo, hi)} or None
        Hard box constraints on each model's weight. Defaults to (0, 1).
        Example: {"ECMWF": (0.25, 0.6), "DWD": (0.0, 0.1)}.
    expert_prior : dict {model_name: float} or None
        Relative trust weights; normalised internally to sum to 1 and used as
        the shrinkage target (and the warm start). Defaults to equal trust.
    shrinkage : float or "cv"
        L2 shrinkage strength toward `expert_prior`. "cv" selects it by
        leave-one-year-out CV-RPS from `shrinkage_grid`.
    shrinkage_grid : sequence of floats
        Candidate lambdas when shrinkage="cv".
    spatial_pooling : "global" or "gridpoint"
        "global" fits ONE weight vector for the whole domain (robust, the usual
        seasonal choice with few years). "gridpoint" fits weights per grid cell
        (more flexible, needs more years / spatial smoothing).
    min_train_years : int
        Minimum valid years required to fit; otherwise falls back to the prior.
    """

    def __init__(
        self,
        expert_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        expert_prior: Optional[Dict[str, float]] = None,
        shrinkage: Union[float, str] = "cv",
        shrinkage_grid: Sequence[float] = (0.0, 0.02, 0.05, 0.1, 0.2, 0.5),
        spatial_pooling: str = "global",
        min_train_years: int = 8,
    ):
        if spatial_pooling not in ("global", "gridpoint"):
            raise ValueError("spatial_pooling must be 'global' or 'gridpoint'")
        self.expert_bounds = expert_bounds or {}
        self.expert_prior = expert_prior
        self.shrinkage = shrinkage
        self.shrinkage_grid = tuple(float(v) for v in shrinkage_grid)
        if not self.shrinkage_grid or not all(np.isfinite(v) and v >= 0 for v in self.shrinkage_grid):
            raise ValueError("shrinkage_grid must contain finite nonnegative values.")
        self.spatial_pooling = spatial_pooling
        self.min_train_years = int(min_train_years)

        self.model_names_: List[str] = []
        self.weights_: Optional[xr.DataArray] = None     # (model[,Y,X])
        self.lambda_: Optional[float] = None
        self.cv_rps_: Optional[float] = None

    # ---- input normalisation ---------------------------------------------
    @staticmethod
    def _stack_models(model_probs, time_dim, lat_dim, lon_dim, prob_dim):
        """Return (names, array(M,T,3,Y,X)) from a dict or a stacked DataArray."""
        if isinstance(model_probs, dict):
            names = list(model_probs.keys())
            arrs = []
            for n in names:
                a = model_probs[n].transpose(prob_dim, time_dim, lat_dim, lon_dim)
                arrs.append(a)
            arrs = xr.align(*arrs, join="exact")
            da = xr.concat(arrs, dim="model").assign_coords(model=names)
        else:
            da = model_probs
            if "M" in da.dims and "model" not in da.dims:
                da = da.rename(M="model")
            if "model" not in da.dims:
                raise ValueError("stacked DataArray must have a 'model' dim")
            names = [str(m) for m in da["model"].values]
            da = da.transpose("model", prob_dim, time_dim, lat_dim, lon_dim)
        if len(names) == 0 or len(set(names)) != len(names):
            raise ValueError("Probability ensembles require unique nonempty model names.")
        labels = list(da[prob_dim].values)
        if set(labels) == set(_PROB_LABELS):
            da = da.sel({prob_dim: list(_PROB_LABELS)})
        elif labels == [0, 1, 2]:
            da = da.assign_coords({prob_dim: _PROB_LABELS})
        else:
            raise ValueError("Probability categories must be PB, PN, PA or 0, 1, 2.")
        finite = np.isfinite(da)
        bad = finite & ((da < 0) | (da > 1))
        if bool(bad.any()):
            raise ValueError("Probability values must lie in [0,1].")
        total = da.sum(prob_dim, skipna=False)
        if bool((finite.all(prob_dim) & (np.abs(total - 1.0) > 1e-6)).any()):
            raise ValueError("Each probability vector must sum to one.")
        da = da.where(finite.all(prob_dim))
        return names, da

    def _bounds_arrays(self, names):
        lo = np.array([self.expert_bounds.get(n, (0.0, 1.0))[0] for n in names], float)
        hi = np.array([self.expert_bounds.get(n, (0.0, 1.0))[1] for n in names], float)
        if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)) or np.any(lo < 0) or np.any(hi > 1):
            raise ValueError("expert_bounds must be finite and inside [0,1].")
        if np.any(lo > hi):
            raise ValueError("expert_bounds: some lower bound exceeds its upper bound")
        if lo.sum() > 1.0 + 1e-9:
            raise ValueError(f"expert lower bounds sum to {lo.sum():.3f} > 1; infeasible simplex")
        if hi.sum() < 1.0 - 1e-9:
            raise ValueError(f"expert upper bounds sum to {hi.sum():.3f} < 1; infeasible simplex")
        return lo, hi

    def _prior_array(self, names):
        if self.expert_prior is None:
            w = np.full(len(names), 1.0 / len(names))
        else:
            w = np.array([max(self.expert_prior.get(n, 0.0), 0.0) for n in names], float)
            if w.sum() <= 0:
                w = np.full(len(names), 1.0 / len(names))
            w = w / w.sum()
        return w

    # ---- core fit ---------------------------------------------------------
    def _select_lambda(self, Pstack, Ohot, lo, hi, w_prior, groups=None):
        """Select shrinkage by withholding whole calendar years across cells."""
        if self.shrinkage != "cv":
            value = float(self.shrinkage)
            if not np.isfinite(value) or value < 0:
                raise ValueError("shrinkage must be finite and nonnegative.")
            return value
        if groups is None:
            raise ValueError("Calendar-year groups are required for shrinkage CV.")
        groups = np.asarray(groups)
        if groups.shape != (Ohot.shape[0],):
            raise ValueError("One calendar-year label is required per pooled sample.")
        years = np.unique(groups)
        if len(years) < 2:
            return float(max(self.shrinkage_grid))
        best_lam, best_err = self.shrinkage_grid[0], np.inf
        for lam in self.shrinkage_grid:
            errs = []
            for year in years:
                tr, te = groups != year, groups == year
                w, _ = _fit_weights_one_point(Pstack[:, tr], Ohot[tr], lo, hi, w_prior, lam)
                errs.append(_rps(_blend(w, Pstack[:, te]), Ohot[te]))
            err = float(np.mean(errs))
            if err < best_err:
                best_err, best_lam = err, lam
        self.cv_rps_ = best_err
        return best_lam

    def fit(
        self,
        model_probs: Union[Dict[str, xr.DataArray], xr.DataArray],
        obs_class: xr.DataArray,
        time_dim: str = "T",
        lat_dim: str = "Y",
        lon_dim: str = "X",
        prob_dim: str = "probability",
    ) -> "WAS_mme_ExpertKnowledge":
        names, da = self._stack_models(model_probs, time_dim, lat_dim, lon_dim, prob_dim)
        self.model_names_ = names
        lo, hi = self._bounds_arrays(names)
        w_prior = _project_bounded_simplex(self._prior_array(names), lo, hi)
        da, obs_class = xr.align(da, obs_class, join="exact")
        years = _calendar_year_groups(da[time_dim])

        P = da.values  # (M, 3, T, Y, X)
        M, _, T, ny, nx = P.shape
        oc = obs_class.transpose(time_dim, lat_dim, lon_dim).values  # (T, Y, X)

        # one-hot the observed category {0,1,2}
        def onehot(cat):
            o = np.zeros((cat.shape[0], 3))
            m = np.isfinite(cat) & np.isin(cat, [0, 1, 2])
            invalid = np.isfinite(cat) & ~np.isin(cat, [0, 1, 2])
            if np.any(invalid):
                warnings.warn("Observed categories outside {0,1,2} are ignored.", RuntimeWarning)
            idx = np.where(m)[0]
            o[idx, cat[m].astype(int)] = 1.0
            return o, m

        # ---------------- global pooling ----------------
        if self.spatial_pooling == "global":
            # pool all valid (year, gridpoint) samples into one design
            Plist, Olist, Glist = [], [], []
            for iy in range(ny):
                for ix in range(nx):
                    cat = oc[:, iy, ix]
                    o, m = onehot(cat)
                    if len(np.unique(years[m])) < self.min_train_years:
                        continue
                    pcell = np.transpose(P[:, :, m, iy, ix], (0, 2, 1))  # (M, n, 3)
                    if not np.all(np.isfinite(pcell)):
                        continue
                    Plist.append(pcell)
                    Olist.append(o[m])
                    Glist.append(years[m])
            if not Plist:
                w = w_prior.copy(); self.lambda_ = 0.0
            else:
                Pstack = np.concatenate(Plist, axis=1)  # (M, N, 3)
                Ohot = np.concatenate(Olist, axis=0)    # (N, 3)
                self.lambda_ = self._select_lambda(Pstack, Ohot, lo, hi, w_prior, np.concatenate(Glist))
                w, _ = _fit_weights_one_point(Pstack, Ohot, lo, hi, w_prior, self.lambda_)
            self.weights_ = xr.DataArray(w, dims=("model",), coords={"model": names})
            return self

        # ---------------- per-gridpoint ----------------
        W = np.full((M, ny, nx), np.nan)
        # pick a single lambda globally (CV per point is too noisy on few years)
        # use the pooled selection once, then apply per point
        Pp, Oo, Gg = [], [], []
        for iy in range(ny):
            for ix in range(nx):
                cat = oc[:, iy, ix]; o, m = onehot(cat)
                if len(np.unique(years[m])) >= self.min_train_years and np.all(np.isfinite(P[:, :, m, iy, ix])):
                    Pp.append(np.transpose(P[:, :, m, iy, ix], (0, 2, 1))); Oo.append(o[m]); Gg.append(years[m])
        if Pp:
            self.lambda_ = self._select_lambda(np.concatenate(Pp, 1), np.concatenate(Oo, 0),
                                               lo, hi, w_prior, np.concatenate(Gg))
        else:
            self.lambda_ = 0.0
        for iy in range(ny):
            for ix in range(nx):
                cat = oc[:, iy, ix]; o, m = onehot(cat)
                if len(np.unique(years[m])) < self.min_train_years or not np.all(np.isfinite(P[:, :, m, iy, ix])):
                    W[:, iy, ix] = w_prior
                    continue
                pcell = np.transpose(P[:, :, m, iy, ix], (0, 2, 1))
                w, _ = _fit_weights_one_point(pcell, o[m], lo, hi, w_prior, self.lambda_)
                W[:, iy, ix] = w
        self.weights_ = xr.DataArray(
            W, dims=("model", lat_dim, lon_dim),
            coords={"model": names, lat_dim: da[lat_dim], lon_dim: da[lon_dim]},
        )
        return self

    # ---- predict ----------------------------------------------------------
    def predict(
        self,
        model_probs: Union[Dict[str, xr.DataArray], xr.DataArray],
        time_dim: str = "T",
        lat_dim: str = "Y",
        lon_dim: str = "X",
        prob_dim: str = "probability",
    ) -> xr.DataArray:
        """Blend the per-model tercile probabilities with the fitted weights.
        Returns a DataArray with dims (probability, T, Y, X)."""
        if self.weights_ is None:
            raise RuntimeError("Call fit() before predict().")
        names, da = self._stack_models(model_probs, time_dim, lat_dim, lon_dim, prob_dim)
        if names != self.model_names_:
            da = da.sel(model=self.model_names_)
        w = self.weights_
        if "model" not in w.dims:
            raise RuntimeError("fitted weights malformed")
        # broadcast multiply over model, sum
        blended = (da * w).sum("model")
        # renormalise defensively
        s = blended.sum(prob_dim)
        blended = xr.where(s > 0, blended / s, np.nan)
        return blended.assign_coords({prob_dim: _PROB_LABELS}).transpose(
            prob_dim, time_dim, lat_dim, lon_dim
        )

    def compute_model(
        self,
        X_train: Union[Dict[str, xr.DataArray], xr.DataArray],
        obs_class_train: xr.DataArray,
        X_test: Union[Dict[str, xr.DataArray], xr.DataArray],
        time_dim: str = "T",
        lat_dim: str = "Y",
        lon_dim: str = "X",
        prob_dim: str = "probability",
    ) -> xr.DataArray:
        """fit + predict in one call (cross-validator friendly)."""
        self.fit(X_train, obs_class_train, time_dim, lat_dim, lon_dim, prob_dim)
        return self.predict(X_test, time_dim, lat_dim, lon_dim, prob_dim)

    # ---- diagnostics ------------------------------------------------------
    def summary(self):
        print("WAS_mme_ExpertKnowledge")
        print("=======================")
        print(f"pooling   : {self.spatial_pooling}")
        print(f"lambda    : {self.lambda_}")
        if self.cv_rps_ is not None:
            print(f"CV-RPS    : {self.cv_rps_:.4f}")
        if self.weights_ is not None:
            if self.weights_.ndim == 1:
                for n, w in zip(self.model_names_, self.weights_.values):
                    print(f"  w[{n}] = {w:.3f}")
            else:
                wm = self.weights_.mean(dim=[d for d in self.weights_.dims if d != "model"])
                for n, w in zip(self.model_names_, wm.values):
                    print(f"  mean w[{n}] = {float(w):.3f}")


# __all__ = ["WAS_mme_ExpertKnowledge"]


# mme = WAS_mme_ExpertKnowledge(
#     expert_bounds={"ECMWF": (0.25, 0.6), "DWD": (0.0, 0.10)},
#     expert_prior={"ECMWF": 0.4, "UKMO": 0.3, "MF": 0.2, "DWD": 0.1},
#     shrinkage="cv", spatial_pooling="global",
# )
# # model_probs: {name: DataArray(probability=3, T, Y, X)} from your NGR/CSGD/etc.
# # obs_class:   WAS_Verification.compute_class(...) -> {0,1,2} per (T,Y,X)
# prob = mme.compute_model(train_probs, obs_class_train, test_probs)   # (probability, T, Y, X)
# mme.summary()



from wass2s.utils import (
    standardize_timeseries,
    reverse_standardize,
    detrended_data,
    apply_detrend_data,
    extract_leading_eeof_component
)



# ===========================================================================
# CCA-MME truncation selection (CPT-like, adapted from was_cca240826.py)
# ===========================================================================
class MMETruncationResult(dict):
    """Cross-validated CCA truncation with attribute access."""
    __getattr__ = dict.get
    def __repr__(self):
        return (f"MMETruncationResult(n_modes={self.get('n_modes')}, "
                f"n_pca_modes={self.get('n_pca_modes')}, "
                f"score={self.get('score'):.4f}, rule={self.get('rule')!r})")


def _mme_resolve_pca_modes(n_pca_modes):
    """Return independent ``(n_pca_X, n_pca_Y)`` truncations."""
    if isinstance(n_pca_modes, (list, tuple, np.ndarray)):
        seq=list(n_pca_modes)
        if len(seq) != 2:
            raise ValueError("n_pca_modes sequence must contain exactly [X, Y].")
        return seq[0], seq[1]
    return n_pca_modes, n_pca_modes


def _mme_safe_mode_counts(n_modes, n_pca_modes, n_samples):
    """Cap CCA/PCA dimensions by the number of independent training years."""
    if int(n_samples) < 3:
        raise ValueError("CCA-MME requires at least three training years.")
    max_modes=max(1, int(n_samples)-1)
    nx,ny=_mme_resolve_pca_modes(n_pca_modes)
    def cap(v):
        if isinstance(v, str) or isinstance(v, float) and not float(v).is_integer():
            return v
        return max(1,min(int(v),max_modes))
    sx,sy=cap(nx),cap(ny)
    numeric=[v for v in (sx,sy) if isinstance(v,(int,np.integer))]
    ceiling=min([max_modes]+numeric) if numeric else max_modes
    sm=max(1,min(int(n_modes),ceiling))
    return sm,[sx,sy]


def _mme_cca_area_weights(obs):
    lat=obs['Y'].astype(float)
    w=np.cos(np.deg2rad(lat)).clip(min=0.0)
    return w.broadcast_like(obs.isel(T=0,drop=True))


def _mme_cca_aggregate_score(pred, obs, weights, kind='mse'):
    valid=np.isfinite(pred)&np.isfinite(obs)
    if kind=='mse':
        e=xr.where(valid,(pred-obs)**2,np.nan)
        num=(e*weights).sum(dim=('T','Y','X'),skipna=True)
        den=xr.where(valid,weights,np.nan).sum(dim=('T','Y','X'),skipna=True)
        return float(num/den) if float(den)>0 else np.nan
    if kind in ('pearson','spearman'):
        p=xr.where(valid,pred,np.nan); o=xr.where(valid,obs,np.nan)
        if kind=='spearman':
            p=p.rank('T'); o=o.rank('T')
        pa=p-p.mean('T',skipna=True); oa=o-o.mean('T',skipna=True)
        cov=(pa*oa).sum('T',skipna=True)
        den=np.sqrt((pa**2).sum('T',skipna=True)*(oa**2).sum('T',skipna=True))
        corr=xr.where(den>0,cov/den,np.nan)
        w=xr.where(np.isfinite(corr),weights,np.nan)
        sw=w.sum(skipna=True)
        return float((corr*w).sum(skipna=True)/sw) if float(sw)>0 else np.nan
    raise ValueError("score must be 'mse', 'pearson' or 'spearman'.")


def _mme_cca_jackknife_se(partial):
    """Leave-one-fold-out jackknife SE from a vector of partial scores."""
    partial=np.asarray(partial,float)
    partial=partial[np.isfinite(partial)]
    if partial.size<3: return np.nan
    n=partial.size
    return float(np.sqrt((n-1)/n*np.sum((partial-partial.mean())**2)))


def _mme_cca_score_with_jackknife(pred,obs,weights,kind,fold_of):
    """Aggregate score, its marginal jackknife SE, and the partial scores.

    The partial vector is aligned with ``np.unique(fold_of)`` (NaN where a fold
    could not be scored). Because every truncation shares the folds, these are
    comparable term by term, which is what lets tune_mme_cca_truncation compute
    the SE of a PAIRED difference — the right measure for a comparison, unlike
    the marginal SE which is right only for a single score.
    """
    full=_mme_cca_aggregate_score(pred,obs,weights,kind)
    folds=np.unique(fold_of)
    if folds.size<3:
        return full,np.nan,None
    partial=np.full(folds.size,np.nan,float)
    for i,f in enumerate(folds):
        keep=np.where(fold_of!=f)[0]
        if keep.size<2: continue
        partial[i]=_mme_cca_aggregate_score(pred.isel(T=keep),obs.isel(T=keep),weights,kind)
    return full,_mme_cca_jackknife_se(partial),partial


def _mme_cca_preprocess_train_test(template,X_train,Y_train,X_test):
    """Fold-local imputation/masking for CCA; never uses the held-out field."""
    Xtr=template._to_latlon(X_train); Ytr=template._to_latlon(Y_train)
    xref=Xtr.mean('T',skipna=True); yref=Ytr.mean('T',skipna=True)
    Xi=Xtr.fillna(xref); Yi=Ytr.fillna(yref)
    xmask=np.isfinite(Xi).all('T') & (Xi.std('T',skipna=True)>0)
    ymask=np.isfinite(Yi).all('T') & (Yi.std('T',skipna=True)>0)
    # If variability masking is too severe, keep all completely finite features.
    if not bool(xmask.any()): xmask=np.isfinite(Xi).all('T')
    if not bool(ymask.any()): ymask=np.isfinite(Yi).all('T')
    Xi=Xi.where(xmask); Yi=Yi.where(ymask)
    Xte=template._to_latlon(X_test).fillna(xref).where(xmask)
    return Xi,Yi,Xte


def _mme_exact_cca_enabled(use_pca=True, n_pca_modes=None):
    """Exact engine (:mod:`wass2s.was_cca_engine`) unless WAS_CCA_ENGINE=xeofs."""
    if os.environ.get('WAS_CCA_ENGINE','exact').strip().lower()=='xeofs':
        return False
    return exact_cca_supported(use_pca,n_pca_modes)


def _mme_seeded_cca(n_modes,n_pca_modes,standardize,use_coslat,use_pca,random_state=42):
    """xeofs CCA with BOTH PCA preprocessors seeded (xeofs 3.x seeds only the cross SVD)."""
    cca=xe.cross.CCA(n_modes=n_modes,standardize=standardize,use_coslat=use_coslat,
                     use_pca=use_pca,n_pca_modes=n_pca_modes,random_state=random_state)
    for pca in (getattr(cca,'pca1',None),getattr(cca,'pca2',None)):
        if pca is not None and hasattr(pca,'random_state'):
            pca.random_state=random_state
    return cca


def _mme_cca_candidates(Xtr,Ytr,Xte,candidates,standardize,use_coslat,use_pca,random_state=42):
    """Latlon predictions of ``Xte`` for each ``(n_modes, n_pca_modes)``.

    One exact decomposition of each training field serves every candidate
    (the xeofs fallback refits per candidate). A failing candidate yields its
    exception in place of a prediction.
    """
    n=min(Xtr.sizes['T'],Ytr.sizes['T'])
    exact=all(_mme_exact_cca_enabled(use_pca,c[1]) for c in candidates)
    if exact:
        try:
            engine=ExactCCA(Xtr,Ytr,standardize,use_coslat); Zx=engine.bx.project(Xte)
        except Exception as exc:
            return [exc for _ in candidates]
    out=[]
    for n_modes,n_pca in candidates:
        try:
            nm,npca=_mme_safe_mode_counts(n_modes,n_pca,n)
            if exact:
                yp=engine.predict(Xte,nm,npca,Zx=Zx)
            else:
                model=_mme_seeded_cca(nm,npca,standardize,use_coslat,use_pca,random_state).fit(Xtr,Ytr,dim='T')
                yp=model.inverse_transform(model.transform(Xte),model.predict(Xte))[1]
            out.append(yp)
        except Exception as exc:
            out.append(exc)
    return out


def _mme_cca_fit_predict_once(template,cfg,X_train,Y_train,X_test):
    Xtr,Ytr,Xte=_mme_cca_preprocess_train_test(template,X_train,Y_train,X_test)
    yp=_mme_cca_candidates(Xtr,Ytr,Xte,[(cfg['n_modes'],cfg['n_pca_modes'])],
                           cfg.get('standardize',False),cfg.get('use_coslat',True),
                           cfg.get('use_pca',True),getattr(template,'random_state',42))[0]
    if isinstance(yp,Exception): raise yp
    yp['T']=X_test['T']
    return template._from_latlon(yp)


def tune_mme_cca_truncation(X,Y,splits,n_modes_grid=(1,2,3,4),x_pca_grid=(4,6,8,10),
                            y_pca_grid=None,template=None,score='mse',one_se=True,
                            paired_se=True,se_multiplier=2.0,
                            n_jobs=None,progress=True):
    """Select ``(CCA modes, PCA-X, PCA-Y)`` by fold-wise CV + one-SE parsimony.

    X and Y must already be in the space in which CCA is fitted.  Pass
    ``y_pca_grid=None`` to tie X/Y PCA truncation; pass an explicit Y grid for
    the full CPT-like three-dimensional search.
    """
    if template is None:
        raise ValueError('template=WAS_mme_CCA* instance is required.')
    Y=Y.transpose('T','Y','X')
    splits=[(np.asarray(a),np.asarray(b)) for a,b in splits]
    if not splits: raise ValueError('splits is empty')
    yg=tuple(x_pca_grid) if y_pca_grid is None else tuple(y_pca_grid)
    tied=y_pca_grid is None
    combos=[]
    for nm in sorted(set(map(int,n_modes_grid))):
        for nx in sorted(set(map(int,x_pca_grid))):
            ys=(nx,) if tied else yg
            for ny in sorted(set(map(int,ys))):
                if nm<=min(nx,ny): combos.append((nm,nx,ny))
    if not combos: raise ValueError('Empty CCA truncation grid.')
    weights=_mme_cca_area_weights(Y)
    fold_of=np.concatenate([np.full(len(te),i) for i,(_,te) in enumerate(splits)])
    # Split-major: each training split is preprocessed and decomposed once
    # and serves every candidate (the PCA does not depend on the candidate).
    cands=[(nm,[nx,ny]) for nm,nx,ny in combos]
    def one_split(split):
        tr,te=split
        Xtr,Ytr,Xte=_mme_cca_preprocess_train_test(template,X.isel(T=tr),Y.isel(T=tr),X.isel(T=te))
        outs=_mme_cca_candidates(Xtr,Ytr,Xte,cands,template.standardize,template.use_coslat,
                                 template.use_pca,getattr(template,'random_state',42))
        return [o if isinstance(o,Exception) else
                template._from_latlon(o.assign_coords(T=Y['T'].isel(T=te).values)) for o in outs]
    workers=_grid_workers() if n_jobs is None else max(1,int(n_jobs))
    if progress: print(f'  MME-CCA truncation: {len(combos)} candidates x {len(splits)} splits',flush=True)
    if workers>1 and len(splits)>1:
        with ThreadPoolExecutor(max_workers=min(workers,len(splits))) as pool:
            per_split=list(pool.map(one_split,splits))
    else:
        per_split=[one_split(sp) for sp in splits]
    rows=[]
    for j,(nm,nx,ny) in enumerate(combos):
        bad=[res[j] for res in per_split if isinstance(res[j],Exception)]
        if bad:
            rows.append({'n_modes':nm,'n_pca_x':nx,'n_pca_y':ny,'score':np.nan,'se':np.nan,'partial':None,'error':repr(bad[0])})
            continue
        pa=xr.concat([res[j] for res in per_split],dim='T'); oa=Y.sel(T=pa['T'])
        val,se,partial=_mme_cca_score_with_jackknife(pa,oa,weights,score,fold_of)
        rows.append({'n_modes':nm,'n_pca_x':nx,'n_pca_y':ny,'score':val,'se':se,'partial':partial,'error':None})
    finite=[r for r in rows if np.isfinite(r['score'])]
    if not finite: raise RuntimeError('No CCA-MME truncation could be evaluated.')
    lower=score=='mse'; ordered=sorted(finite,key=lambda r:r['score'],reverse=not lower)
    best=ordered[0]; rule='best'
    if one_se and np.isfinite(best['se']):
        # Paired SE: the fold-to-fold variability common to all truncations
        # cancels in the difference with the best candidate. The marginal SE
        # over-inflates the tolerance and collapses the selection onto the
        # grid floor. se_multiplier (2.0, calibrated on 24-year synthetic
        # folds) restores parsimony in the null case without giving up the
        # sensitivity the paired SE buys. See patch_cca_paired_onese.py.
        best_partial=best.get('partial')
        paired=bool(paired_se) and best_partial is not None
        def _tol(cand):
            if not paired: return best['se']
            cp=cand.get('partial')
            if cp is None or len(cp)!=len(best_partial): return best['se']
            s=_mme_cca_jackknife_se(np.asarray(cp)-np.asarray(best_partial))
            if not np.isfinite(s): return 0.0
            return float(se_multiplier)*s
        within=[r for r in finite
                if (r['score']<=best['score']+_tol(r) if lower
                    else r['score']>=best['score']-_tol(r))]
        if within:
            chosen=min(within,key=lambda r:(r['n_modes'],r['n_pca_x']+r['n_pca_y']))
            if chosen is not best: rule='one-se-paired' if paired else 'one-se'
            best=chosen
    return MMETruncationResult(n_modes=int(best['n_modes']),
                               n_pca_modes=[int(best['n_pca_x']),int(best['n_pca_y'])],
                               score=float(best['score']),
                               se=float(best['se']) if np.isfinite(best['se']) else np.nan,
                               metric=score,rule=rule,table=ordered)


def _mme_cca_field_fingerprint(*arrays):
    """Complete value/coordinate signature, independent of dimension order."""
    from wass2s.was_cca import _field_fingerprint
    canonical=[]
    for da in arrays:
        lead=[d for d in ('T','M','Y','X') if d in da.dims]
        da=da.transpose(*lead,*[d for d in da.dims if d not in lead])
        canonical.append(da)
    return _field_fingerprint(*canonical)


def _check_truncation_selection(value):
    """'cv' (nested cross-validated MSE, historical) or 'aic' (Bartlett-corrected AIC)."""
    value = str(value).strip().lower()
    if value not in ("cv", "aic"):
        raise ValueError("truncation_selection must be 'cv' or 'aic'.")
    return value


class _MMECCATruncationMixin:
    """Reusable CCA truncation state/cache shared by standard and EEOF MME CCA.

    The nested cross-validation (``_native_regression_cv``, family
    ``mme_cca``) tunes a fresh copy inside every fold, so the instance itself
    never learns the selection. :meth:`finalize_truncation` makes the final
    selection on the complete record with the same raw procedure and caches
    it for ``forecast``, which otherwise makes that selection itself when
    autotuning is on (WAS_MME_CCA_AUTOTUNE) and nothing is cached for the
    fields it receives.
    """
    def _init_truncation(self):
        self.truncation_=None; self._truncation_cache={}; self._last_effective_n_modes=int(self.n_modes)

    def autotune_enabled(self):
        # An explicit CLI/API setting also governs the operational forecast;
        # None preserves the historical environment-controlled default.
        enabled=getattr(self,'auto_truncation',None)
        return os.environ.get('WAS_MME_CCA_AUTOTUNE','1')!='0' if enabled is None else bool(enabled)

    def default_truncation_grid(self):
        nx,ny=_mme_resolve_pca_modes(self.n_pca_modes)
        def grid(v):
            if not isinstance(v,(int,np.integer)): return (4,6,8,10)
            v=max(2,int(v)); return tuple(sorted({max(2,v//4),max(2,v//2),max(3,3*v//4),v}))
        m=max(1,int(self.n_modes)); mg=tuple(sorted(set(range(1,m+1)))[-4:])
        xg=grid(nx); yg=grid(ny)
        return {'n_modes_grid':mg,'x_pca_grid':xg,'y_pca_grid':yg,
                'score':'mse','one_se':True,'progress':False}

    def truncation_state(self): return self.n_modes,self.n_pca_modes
    def restore_truncation(self,state): self.n_modes,self.n_pca_modes=state; return self
    def apply_truncation(self,result=None):
        result=result or self.truncation_
        if result is None: raise ValueError('No CCA-MME truncation available; call tune() first.')
        self.n_modes=result['n_modes']; self.n_pca_modes=result['n_pca_modes']; return self
    def tune(self,X,Y,splits,apply=True,**kwargs):
        kwargs.setdefault('template',self)
        res=tune_mme_cca_truncation(X,Y,splits,**kwargs); self.truncation_=res
        if apply: self.apply_truncation(res)
        return res
    def remember_truncation(self,key,result=None):
        result=result or self.truncation_
        if result is not None: self._truncation_cache[(key,self._truncation_preprocessing())]=result
    def _truncation_preprocessing(self):
        """A mode choice belongs to its training preprocessing as well as its data."""
        return (type(self).__name__, bool(getattr(self,'detrend',False)),
                bool(getattr(self,'eeof_detrend',False)))
    def recall_truncation(self,key):
        return self._truncation_cache.get((key,self._truncation_preprocessing()))
    def remember_truncation_for(self,Predictor,Predictant,result=None):
        self.remember_truncation(_mme_cca_field_fingerprint(Predictor,Predictant),result)
    def apply_cached_truncation(self,Predictor,Predictant):
        r=self.recall_truncation(_mme_cca_field_fingerprint(Predictor,Predictant))
        if r is not None: self.apply_truncation(r)
        return r
    def finalize_truncation(self,Predictor,Predictant,clim_start,clim_end,splits=None,nb_omit=1,grid=None):
        """Select the truncation on the complete record and remember it for ``forecast``.

        Same raw procedure as the nested cross-validation
        (``was_cross_validate._tune_mme_cca_raw``: per-fold standardization,
        detrending and CCA inside ``compute_model``), scored on ``splits`` —
        by default leave-one-out with the calendar buffer ``ceil(nb_omit/2)``
        of ``WAS_Cross_Validator`` — with ``grid`` (default
        :meth:`default_truncation_grid`, the grid the cross-validation used
        when it was given none). The selection is stored in ``truncation_``
        and cached under the fingerprint of ``(Predictor, Predictant)``;
        ``forecast`` applies it for those fields and restores the constructor
        values afterwards, which are not modified here. Call it after
        ``cross_validate`` with the very fields ``forecast`` will receive.
        """
        from copy import deepcopy
        from wass2s.was_cross_validate import _tune_mme_cca_raw, CustomTimeSeriesSplit
        y=Predictant.isel(M=0,drop=True) if 'M' in Predictant.dims else Predictant
        y=y.transpose('T','Y','X')
        try:
            x=align_predictor_target(Predictor,y,mode='season_year')
        except ValueError:
            if Predictor.sizes.get('T')!=y.sizes['T']: raise
            x=Predictor.assign_coords(T=y['T'])   # positional stamping, as forecast itself does
        if splits is None:
            splits=list(CustomTimeSeriesSplit(n_splits=int(y.sizes['T'])).split(y['T'],nb_omit))
        probe=deepcopy(self)
        if str(getattr(self,'truncation_selection','cv')).lower()=='aic':
            from wass2s.was_cross_validate import _select_mme_cca_aic
            res=_select_mme_cca_aic(probe,x,y,clim_start,clim_end,dict(grid or self.default_truncation_grid()))
            splits=[]
        else:
            res=_tune_mme_cca_raw(probe,x,y,clim_start,clim_end,nb_omit,dict(grid or self.default_truncation_grid()),
                                  folds=splits)
        self.truncation_=res
        self.remember_truncation_for(Predictor,Predictant,res)
        if os.environ.get('WAS_CCA_QUIET_TUNE','0')!='1':
            print(f"  final MME-CCA truncation (full record, {len(splits)} splits): "
                  f"n_modes={res['n_modes']} n_pca_modes={res['n_pca_modes']} "
                  f"{res['metric']}={res['score']:.4f} ({res['rule']})",flush=True)
        return res
    # ------------------------------------------------------------ CCA core
    def _cca_inputs(self, field_train, y_train, field_test):
        """Latlon CCA inputs; NaNs filled with the TRAINING means (no leakage)."""
        Xtr = self._to_latlon(field_train)
        Ytr = self._to_latlon(y_train)
        Xtr_mean = Xtr.mean(dim="T", skipna=True)
        Xtr = Xtr.fillna(Xtr_mean)
        Ytr = Ytr.fillna(Ytr.mean(dim="T", skipna=True))
        Xte = self._to_latlon(field_test).fillna(Xtr_mean)
        return Xtr, Ytr, Xte

    def _fit_predict_cca(self, field_train, y_train, field_test):
        """
        Fit CCA on the training field, predict the test field, return the
        physical-space predictand on the WAS grid (T, Y, X).

        The prediction comes from the exact engine
        (:class:`wass2s.was_cca_engine.ExactCCA`: the xeofs CCA algebra with
        an exact SVD, deterministic); ``cca_model`` is a lazy xeofs object for
        diagnostics only. ``WAS_CCA_ENGINE=xeofs`` restores the historical
        path (``inverse_transform(transform(X), predict(X))[1]``).
        """
        Xtr, Ytr, Xte = self._cca_inputs(field_train, y_train, field_test)
        n = Xtr.sizes.get("T", None)
        self.cca = self._new_cca(n)
        if _mme_exact_cca_enabled(self.use_pca, self.n_pca_modes):
            nm, npca = _mme_safe_mode_counts(self.n_modes, self.n_pca_modes, n)
            engine = ExactCCA(Xtr, Ytr, self.standardize, self.use_coslat)
            self.cca_model = LazyXeofsCCA(partial(self._new_cca, n), Xtr, Ytr, engine=engine)
            y_pred = engine.predict(Xte, nm, npca)
        else:
            self.cca_model = self.cca.fit(Xtr, Ytr, dim="T")
            y_pred = self.cca_model.predict(Xte)
            y_pred = self.cca_model.inverse_transform(
                self.cca_model.transform(Xte), y_pred
            )[1]
        y_pred["T"] = field_test["T"]
        return self._from_latlon(y_pred)

    def compute_model(self, X_train, y_train, X_test, y_test=None, preprocessed=False, **kwargs):
        """Fold prediction. With ``preprocessed=True`` the cross-validator has
        already fitted standardisation/detrending on TRAIN and this method only
        performs CCA. Otherwise the class's own fold preprocessing
        (:meth:`_cca_problem`) is fitted on TRAIN and applied to TEST.
        """
        field_train, y_fit, field_test, post = self._cca_problem(
            X_train, y_train, X_test, preprocessed)
        pred = self._fit_predict_cca(field_train, y_fit, field_test)
        return post(pred) if post is not None else pred

    def cca_candidates(self, X_train, y_train, X_test, candidates, preprocessed=False):
        """``compute_model`` for several ``(n_modes, n_pca_modes)`` at once.

        The candidate-independent work -- multi-model field, detrending (or
        EEOF residual), gap filling and the PCA of both fields -- is done once;
        each candidate then costs a small SVD and a back-projection. Returns a
        list aligned with ``candidates`` (an exception where a candidate
        cannot be fitted). Values equal ``compute_model`` with that
        truncation.
        """
        field_train, y_fit, field_test, post = self._cca_problem(
            X_train, y_train, X_test, preprocessed)
        Xtr, Ytr, Xte = self._cca_inputs(field_train, y_fit, field_test)
        outs = _mme_cca_candidates(Xtr, Ytr, Xte, candidates, self.standardize,
                                   self.use_coslat, self.use_pca,
                                   getattr(self, "random_state", 42))
        res = []
        for out in outs:
            if isinstance(out, Exception):
                res.append(out); continue
            out["T"] = field_test["T"]
            pred = self._from_latlon(out)
            res.append(post(pred) if post is not None else pred)
        return res

    def _truncation_for_forecast(self,Predictor,Predictant,clim_start,clim_end):
        """Cached selection for these fields, else a fresh full-record one under autotuning."""
        if not self.autotune_enabled():
            return None
        r=self.apply_cached_truncation(Predictor,Predictant)
        if r is None:
            try:
                r=self.finalize_truncation(Predictor,Predictant,clim_start,clim_end)
            except Exception as exc:  # an operational forecast must still be issued
                if getattr(self,'truncation_selection','cv')=='aic':
                    raise RuntimeError('MME-CCA AIC selection failed; refusing an unselected forecast.') from exc
                warnings.warn(f'MME-CCA truncation could not be selected on the full record ({exc!r}); '
                              f'forecast uses n_modes={self.n_modes}, n_pca_modes={self.n_pca_modes}.',
                              RuntimeWarning,stacklevel=3)
                return None
            self.apply_truncation(r)
        return r


class WAS_mme_CCA(_MMECCATruncationMixin):
    """
    Canonical Correlation Analysis MME.

    Parameters
    ----------
    n_modes : int
        Number of canonical modes for xeofs.cross.CCA (default 4).
    n_pca_modes : int
        PCA modes kept before CCA when ``use_pca`` is True (default 8).
    standardize : bool
        Passed to xeofs.cross.CCA. Keep False; the framework already
        standardizes (default False).
    use_coslat : bool
        Cosine-latitude weighting inside the CCA (default True).
    use_pca : bool
        Pre-filter each field with PCA before the CCA (default True).
    combine : {'mean', 'meof'}
        How the M models enter the CCA predictor field.
    detrend : bool
        Apply ``detrended_data`` to both predictor and predictand in
        ``forecast``, exactly like ``WAS_CCA_op`` does unconditionally
        (default True). Set False only for ablation comparisons.
    dist_method : {'nonparam', 'bestfit'}
        Tercile-probability strategy (default 'nonparam').
    """

    def __init__(self, n_modes=4, n_pca_modes=8, standardize=False,
                 use_coslat=True, use_pca=True, combine="mean",
                 detrend=True, dist_method="nonparam", positive_support=True, random_state=42,
                 truncation_selection="cv"):
        if xe is None:
            raise ImportError("CCA MME classes require the optional 'xeofs' package.")
        if combine not in ("mean", "meof"):
            raise ValueError("combine must be 'mean' or 'meof'.")
        self.n_modes = n_modes
        self.n_pca_modes = n_pca_modes
        self.standardize = standardize
        self.use_coslat = use_coslat
        self.use_pca = use_pca
        self.combine = combine
        self.detrend = detrend
        self.dist_method = dist_method
        self.positive_support = bool(positive_support)

        self.cca = None
        self.cca_model = None
        self.random_state = random_state
        self.truncation_selection = _check_truncation_selection(truncation_selection)
        self._init_truncation()

    # ------------------------------------------------------------------ utils
    def _new_cca(self, n_samples=None):
        """Fresh fold-local CCA with independent X/Y PCA truncations."""
        if n_samples is None:
            n_samples=max(int(self.n_modes)+2,3)
        nm,npca=_mme_safe_mode_counts(self.n_modes,self.n_pca_modes,n_samples)
        self._last_effective_n_modes=int(nm)
        cca = xe.cross.CCA(
            n_modes=nm,
            standardize=self.standardize,
            use_coslat=self.use_coslat,
            use_pca=self.use_pca,
            n_pca_modes=npca,
            random_state=getattr(self, 'random_state', 42),
        )
        # xeofs versions can seed the cross-SVD without propagating that seed
        # into the two PCA preprocessors. Seed each local preprocessor too.
        for pca in (getattr(cca, 'pca1', None), getattr(cca, 'pca2', None)):
            if pca is not None and hasattr(pca, 'random_state'):
                pca.random_state = getattr(self, 'random_state', 42)
        return cca

    def _field(self, X):
        """Build the CCA predictor field from the predictor cube (T, Y, X, M)."""
        if "M" not in X.dims:
            return X
        if self.combine == "meof":
            return X
        return X.mean("M")

    @staticmethod
    def _to_latlon(da):
        """WAS dims (Y, X[, M]) → xeofs dims (lat, lon[, M]); order (T, lat, lon, ...)."""
        rename = {}
        if "Y" in da.dims:
            rename["Y"] = "lat"
        if "X" in da.dims:
            rename["X"] = "lon"
        da = da.rename(rename)
        lead = [d for d in ("T", "lat", "lon") if d in da.dims]
        rest = [d for d in da.dims if d not in lead]
        return da.transpose(*lead, *rest)

    @staticmethod
    def _from_latlon(da):
        """xeofs dims (lat, lon) → WAS dims (Y, X); order (T, Y, X)."""
        rename = {}
        if "lat" in da.dims:
            rename["lat"] = "Y"
        if "lon" in da.dims:
            rename["lon"] = "X"
        da = da.rename(rename)
        return da.transpose("T", "Y", "X")

    # ----------------------------------------------- probability calculation
    @staticmethod
    def _ppf_terciles_from_code(dist_code, shape, loc, scale):
        if np.isnan(dist_code):
            return np.nan, np.nan
        code = int(dist_code)
        try:
            if code == 1:
                return norm.ppf(1/3, loc=loc, scale=scale), norm.ppf(2/3, loc=loc, scale=scale)
            if code == 2:
                return lognorm.ppf(1/3, s=shape, loc=loc, scale=scale), lognorm.ppf(2/3, s=shape, loc=loc, scale=scale)
            if code == 3:
                return expon.ppf(1/3, loc=loc, scale=scale), expon.ppf(2/3, loc=loc, scale=scale)
            if code == 4:
                return gamma.ppf(1/3, a=shape, loc=loc, scale=scale), gamma.ppf(2/3, a=shape, loc=loc, scale=scale)
            if code == 5:
                return weibull_min.ppf(1/3, c=shape, loc=loc, scale=scale), weibull_min.ppf(2/3, c=shape, loc=loc, scale=scale)
            if code == 6:
                return t.ppf(1/3, df=shape, loc=loc, scale=scale), t.ppf(2/3, df=shape, loc=loc, scale=scale)
            if code == 7:
                return poisson.ppf(1/3, mu=shape, loc=loc), poisson.ppf(2/3, mu=shape, loc=loc)
            if code == 8:
                return nbinom.ppf(1/3, n=shape, p=scale, loc=loc), nbinom.ppf(2/3, n=shape, p=scale, loc=loc)
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
            return (V / (M ** 2)) - ((g2 / (g1 ** 2)) - 1)
        except Exception:
            return -np.inf

    @staticmethod
    def calculate_tercile_probabilities_bestfit(best_guess, error_variance, T1, T2, dist_code, dof):
        best_guess = np.asarray(best_guess, dtype=float)
        error_variance = float(error_variance)
        n_time = best_guess.size
        out = np.full((3, n_time), np.nan, dtype=float)

        if (np.all(np.isnan(best_guess)) or not np.isfinite(error_variance)
                or error_variance <= 0 or not np.isfinite(T1) or not np.isfinite(T2)
                or np.isnan(dist_code)):
            return out

        code = int(dist_code)
        try:
            if code == 1:
                s = np.sqrt(error_variance)
                c1, c2 = norm.cdf(T1, loc=best_guess, scale=s), norm.cdf(T2, loc=best_guess, scale=s)
                out[0, :], out[1, :], out[2, :] = c1, c2 - c1, 1.0 - c2
            elif code == 2:
                valid = best_guess > 0
                sigma = np.full(n_time, np.nan); mu = np.full(n_time, np.nan)
                sigma[valid] = np.sqrt(np.log(1.0 + error_variance / (best_guess[valid] ** 2)))
                mu[valid] = np.log(best_guess[valid]) - 0.5 * sigma[valid] ** 2
                c1, c2 = lognorm.cdf(T1, s=sigma, scale=np.exp(mu)), lognorm.cdf(T2, s=sigma, scale=np.exp(mu))
                out[0, :], out[1, :], out[2, :] = c1, c2 - c1, 1.0 - c2
            elif code == 3:
                scale = np.sqrt(error_variance); loc = best_guess - scale
                c1, c2 = expon.cdf(T1, loc=loc, scale=scale), expon.cdf(T2, loc=loc, scale=scale)
                out[0, :], out[1, :], out[2, :] = c1, c2 - c1, 1.0 - c2
            elif code == 4:
                valid = best_guess > 0
                alpha = np.full(n_time, np.nan); theta = np.full(n_time, np.nan)
                alpha[valid] = (best_guess[valid] ** 2) / error_variance
                theta[valid] = error_variance / best_guess[valid]
                c1, c2 = gamma.cdf(T1, a=alpha, scale=theta), gamma.cdf(T2, a=alpha, scale=theta)
                out[0, :], out[1, :], out[2, :] = c1, c2 - c1, 1.0 - c2
            elif code == 5:
                for i in range(n_time):
                    M, V = best_guess[i], error_variance
                    if not np.isfinite(M) or M <= 0 or V <= 0:
                        continue
                    k = fsolve(WAS_mme_CCA.weibull_shape_solver, 2.0, args=(M, V))[0]
                    if not np.isfinite(k) or k <= 0:
                        continue
                    lam = M / gamma_function(1 + 1 / k)
                    c1, c2 = weibull_min.cdf(T1, c=k, loc=0, scale=lam), weibull_min.cdf(T2, c=k, loc=0, scale=lam)
                    out[0, i], out[1, i], out[2, i] = c1, c2 - c1, 1.0 - c2
            elif code == 6:
                if dof <= 2:
                    return out
                scale = np.sqrt(error_variance * (dof - 2) / dof)
                c1, c2 = t.cdf(T1, df=dof, loc=best_guess, scale=scale), t.cdf(T2, df=dof, loc=best_guess, scale=scale)
                out[0, :], out[1, :], out[2, :] = c1, c2 - c1, 1.0 - c2
            elif code == 7:
                mu = np.where(best_guess >= 0, best_guess, np.nan)
                c1, c2 = poisson.cdf(T1, mu=mu), poisson.cdf(T2, mu=mu)
                out[0, :], out[1, :], out[2, :] = c1, c2 - c1, 1.0 - c2
            elif code == 8:
                valid = error_variance > best_guess
                p = np.where(valid, best_guess / error_variance, np.nan)
                n = np.where(valid, best_guess ** 2 / (error_variance - best_guess), np.nan)
                c1, c2 = nbinom.cdf(T1, n=n, p=p), nbinom.cdf(T2, n=n, p=p)
                out[0, :], out[1, :], out[2, :] = c1, c2 - c1, 1.0 - c2
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
    def calculate_tercile_probabilities_nonparametric(best_guess, error_samples, first_tercile, second_tercile):
        return WAS_mme_Weighted.calculate_tercile_probabilities_nonparametric(
            best_guess, error_samples, first_tercile, second_tercile)

    @staticmethod
    def _normalize_probabilities(prob):
        prob = prob.clip(min=0.0, max=1.0)
        total = prob.sum(dim="probability")
        return xr.where(total > 0, prob / total, np.nan)

    def _tercile_probabilities(self, Predictant, deterministic, clim_year_start, clim_year_end,
                               error_samples, error_variance,
                               best_code_da=None, best_shape_da=None,
                               best_loc_da=None, best_scale_da=None):
        Predictant = Predictant.transpose("T", "Y", "X")
        deterministic = deterministic.transpose("T", "Y", "X")
        mask = xr.where(np.isfinite(Predictant).sum(dim='T') >= 3, 1.0, np.nan)

        clim = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))
        if clim.sizes.get("T", 0) < 3:
            raise ValueError("Not enough years in climatology period for terciles.")
        terc = climatological_terciles(clim)[0]
        T1_emp = terc.isel(quantile=0, drop=True)
        T2_emp = terc.isel(quantile=1, drop=True)
        dof = max(int(clim.sizes["T"]) - int(getattr(self, "_last_effective_n_modes", self.n_modes)) - 1, 3)

        if self.dist_method == "bestfit":
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError("dist_method='bestfit' requires best_code_da, best_shape_da, best_loc_da, best_scale_da.")
            # Category boundaries stay on the observed climatological reference.
            # The selected distribution controls the predictive CDF only.
            if getattr(self, "tercile_reference", "climatology") == "training":
                T1, T2 = _pufunc(
                    self._ppf_terciles_from_code, best_code_da, best_shape_da, best_loc_da, best_scale_da,
                    input_core_dims=[(), (), (), ()], output_core_dims=[(), ()],
                    vectorize=True, dask="parallelized", output_dtypes=[float, float])
            else:
                T1, T2 = T1_emp, T2_emp
            prob = _pufunc(
                self.calculate_tercile_probabilities_bestfit,
                deterministic, error_variance, T1, T2, best_code_da,
                input_core_dims=[("T",), (), (), (), ()], output_core_dims=[("probability", "T")],
                vectorize=True, kwargs={"dof": dof}, dask="parallelized", output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True})
        elif self.dist_method == "nonparam":
            # rename residual sample axis so a length-1 forecast and the
            # length-n residual record don't collide on the shared 'T' core dim
            err = error_samples.rename({"T": "__samp"})
            prob = _pufunc(
                self.calculate_tercile_probabilities_nonparametric,
                deterministic, err, T1_emp, T2_emp,
                input_core_dims=[("T",), ("__samp",), (), ()], output_core_dims=[("probability", "T")],
                vectorize=True, dask="parallelized", output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True})
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}")

        prob = prob.assign_coords(probability=("probability", ["PB", "PN", "PA"]))
        prob = self._normalize_probabilities(prob.transpose("probability", "T", "Y", "X"))
        return (prob * mask).transpose("probability", "T", "Y", "X")

    # ----------------------------------------------------- framework contract
    def _cca_problem(self, X_train, y_train, X_test, preprocessed=False):
        """Candidate-independent fold preprocessing: ``(field_train, y_fit, field_test, post)``.

        Multi-model field and, with ``detrend``, polynomial trends fitted on
        TRAIN (removed from both fields; the predictand trend is added back
        to the prediction by ``post``).
        """
        X_train = align_predictor_target(X_train, y_train, mode="season_year")
        if "M" in y_train.dims:
            y_train = y_train.isel(M=0, drop=True)
        field_train = self._field(X_train)
        field_test = self._field(X_test)
        if preprocessed or not self.detrend:
            return field_train, y_train, field_test, None
        field_train_fit, coeffsX, metaX = _mme_detrend_training(field_train)
        field_test_fit = field_test - apply_detrend_data(field_test, coeffsX, metaX)
        y_train_fit, coeffsY, metaY = _mme_detrend_training(y_train)
        post = lambda pred: pred + apply_detrend_data(pred, coeffsY, metaY)
        return field_train_fit, y_train_fit, field_test_fit, post

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det,
                     best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None):
        if "M" in Predictant.dims:
            Predictant = Predictant.isel(M=0, drop=True)
        Predictant = Predictant.transpose("T", "Y", "X")
        hindcast_det = hindcast_det.transpose("T", "Y", "X")
        if hindcast_det.sizes["T"] == Predictant.sizes["T"]:
            hindcast_det = hindcast_det.assign_coords(T=Predictant["T"])

        error_samples = Predictant - hindcast_det
        error_variance = (error_samples ** 2).mean(dim="T", skipna=True)

        return self._tercile_probabilities(
            Predictant, hindcast_det, clim_year_start, clim_year_end,
            error_samples, error_variance,
            best_code_da, best_shape_da, best_loc_da, best_scale_da)

    # ------------------------------------------------------------- operational
    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor,
                 hindcast_det, Predictor_for_year,
                 best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None):
        """
        Operational CCA forecast for a single target year.

          predictor → optional detrend of the raw multi-model field
          predictand → standardize → optional detrend
          forecast-year predictor → apply the TRAIN trend if detrend=True
          fit CCA on (predictor_dt, predictand_st_dt)
          result → restore the TRAIN trend if detrend=True → reverse_standardize

        Truncation: the selection cached by :meth:`finalize_truncation` for
        these fields (or, under autotuning, a fresh full-record selection) is
        applied for the duration of the call; the constructor values are
        restored afterwards.
        """
        _trunc_state = self.truncation_state()
        try:
            self._truncation_for_forecast(Predictor, Predictant, clim_year_start, clim_year_end)
            return self._forecast_impl(Predictant, clim_year_start, clim_year_end, Predictor,
                                       hindcast_det, Predictor_for_year, best_code_da,
                                       best_shape_da, best_loc_da, best_scale_da)
        finally:
            self.restore_truncation(_trunc_state)

    def _forecast_impl(self, Predictant, clim_year_start, clim_year_end, Predictor,
                       hindcast_det, Predictor_for_year,
                       best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None):
        if "M" in Predictant.dims:
            Predictant = Predictant.isel(M=0, drop=True)
        Predictant = Predictant.transpose("T", "Y", "X")
        mask = xr.where(np.isfinite(Predictant).sum(dim='T') >= 3, 1.0, np.nan)
        
        mean_val = Predictor.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean(dim='T')
        std_val = Predictor.sel(T=slice(str(clim_year_start), str(clim_year_end))).std(dim='T')
        Predictor_for_year = (Predictor_for_year - mean_val) / std_val

        # --- predictor: raw field → polynomial detrend 
        Predictor = standardize_timeseries(Predictor, clim_year_start, clim_year_end)
        
        field_hist = self._field(Predictor)
        field_year = self._field(Predictor_for_year)

        if self.detrend:
            field_hist_fit, coeffsX, metaX = _mme_detrend_training(field_hist)
            # remove the same fitted trend from the forecast-year field 
            field_year_fit = field_year - apply_detrend_data(field_year, coeffsX, metaX)
        else:
            field_hist_fit = field_hist
            field_year_fit = field_year

        # --- predictand: standardize → polynomial detrend 
        Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end)
        if self.detrend:
            Predictant_st_fit, coeffsY, metaY = _mme_detrend_training(Predictant_st)
        else:
            Predictant_st_fit = Predictant_st

        # --- fit CCA and predict 
        pred_st = self._fit_predict_cca(field_hist_fit, Predictant_st_fit, field_year_fit)

        # --- add predictand trend back, then reverse-standardize 
        if self.detrend:
            pred_st = pred_st + apply_detrend_data(pred_st, coeffsY, metaY)

        forecast_det = reverse_standardize(pred_st, Predictant, clim_year_start, clim_year_end)
        if self.positive_support:
            forecast_det = forecast_det.clip(min=0)

        # target time = forecast year + predictand target month
        year = Predictor_for_year.coords["T"].values.astype("datetime64[Y]").astype(int)[0] + 1970
        month = Predictant.isel(T=0).coords["T"].values.astype("datetime64[M]").astype(int) % 12 + 1
        new_T = np.datetime64(f"{year}-{month:02d}-01")
        forecast_det = forecast_det.assign_coords(T=xr.DataArray([new_T], dims=["T"]))
        forecast_det["T"] = forecast_det["T"].astype("datetime64[ns]")

        error_samples = Predictant - hindcast_det.transpose("T", "Y", "X")
        error_variance = (error_samples ** 2).mean(dim="T", skipna=True)
        forecast_prob = self._tercile_probabilities(
            Predictant, forecast_det, clim_year_start, clim_year_end,
            error_samples, error_variance,
            best_code_da, best_shape_da, best_loc_da, best_scale_da)

        return (forecast_det * mask, forecast_prob * mask)


class WAS_mme_CCA_eeof(_MMECCATruncationMixin):
    """
    Canonical Correlation Analysis MME, legacy WAS_CCA ideology (EEOF detrend).

    Parameters
    ----------
    n_modes : int
        Number of canonical modes for xeofs.cross.CCA (default 4).
    n_pca_modes : int
        PCA modes kept before CCA when ``use_pca`` (default 8).
    standardize : bool
        Passed to xeofs.cross.CCA (default False; data are already prepared).
    use_coslat : bool
        Cosine-latitude weighting inside the CCA (default True).
    use_pca : bool
        Pre-filter each field with PCA before the CCA (default True).
    combine : {'mean', 'meof'}
        How the M models enter the CCA predictor field.
        'mean' (default) collapses M to the multi-model mean before any
        processing.
        'meof' keeps M as an extra channel so xeofs sees a joint
        (lat, lon, M) feature space at both the EEOF detrend stage and the
        CCA stage. This is scientifically richer but numerically fragile:
        the internal n_pca_modes=22 of ExtendedEOF may be too small to
        cover the enlarged feature space, and the CCA PCA pre-filter faces
        a far worse sample-to-feature ratio. Spatially coarsen the predictor
        and keep n_pca_modes conservative when using 'meof'.
    eeof_detrend : bool
        Apply the legacy EEOF detrend inside ``forecast`` (default True). Set
        False to fit CCA on the raw standardized fields without EEOF removal.
    dist_method : {'nonparam', 'bestfit'}
        Tercile-probability strategy (default 'nonparam').
    """

    def __init__(self, n_modes=4, n_pca_modes=8, standardize=False,
                 use_coslat=True, use_pca=True, combine="mean",
                 eeof_detrend=False, dist_method="nonparam", positive_support=True,
                 eeof_ref_offset=3, truncation_selection="cv"):
        if xe is None:
            raise ImportError("CCA MME classes require the optional 'xeofs' package.")
        if combine not in ("mean", "meof"):
            raise ValueError("combine must be 'mean' or 'meof'.")
        self.n_modes = n_modes
        self.n_pca_modes = n_pca_modes
        self.standardize = standardize
        self.use_coslat = use_coslat
        self.use_pca = use_pca
        self.combine = combine
        self.eeof_detrend = eeof_detrend
        self.eeof_ref_offset = max(1, int(eeof_ref_offset))
        if self.eeof_detrend:
            warnings.warn(
                "CCA_eeof is a legacy/experimental preprocessing path. Its EEOF "
                "trend cannot be transferred to a future year as cleanly as a "
                "polynomial trend; benchmark it against WAS_mme_CCA before use.",
                RuntimeWarning, stacklevel=2)
        self.dist_method = dist_method
        self.positive_support = bool(positive_support)
        self.truncation_selection = _check_truncation_selection(truncation_selection)

        self.cca = None
        self.cca_model = None
        self._init_truncation()

    # ------------------------------------------------------------------ utils
    def _new_cca(self, n_samples=None):
        """Fresh fold-local CCA with independent X/Y PCA truncations."""
        if n_samples is None:
            n_samples=max(int(self.n_modes)+2,3)
        nm,npca=_mme_safe_mode_counts(self.n_modes,self.n_pca_modes,n_samples)
        self._last_effective_n_modes=int(nm)
        return _mme_seeded_cca(nm, npca, self.standardize, self.use_coslat,
                               self.use_pca, getattr(self, 'random_state', 42))

    def _field(self, X):
        """Build the CCA predictor field from the predictor cube (T, Y, X, M)."""
        if "M" not in X.dims:
            return X
        if self.combine == "meof":
            return X
        return X.mean("M")

    @staticmethod
    def _to_latlon(da):
        """WAS dims (Y, X[, M]) -> xeofs dims (lat, lon[, M]); order (T, lat, lon, ...)."""
        rename = {}
        if "Y" in da.dims:
            rename["Y"] = "lat"
        if "X" in da.dims:
            rename["X"] = "lon"
        da = da.rename(rename)
        lead = [d for d in ("T", "lat", "lon") if d in da.dims]
        rest = [d for d in da.dims if d not in lead]
        return da.transpose(*lead, *rest)

    @staticmethod
    def _from_latlon(da):
        """xeofs dims (lat, lon) -> WAS dims (Y, X); order (T, Y, X)."""
        rename = {}
        if "lat" in da.dims:
            rename["lat"] = "Y"
        if "lon" in da.dims:
            rename["lon"] = "X"
        da = da.rename(rename)
        return da.transpose("T", "Y", "X")

    @staticmethod
    def _eeof_remove(da):
        """
        Legacy EEOF detrend: subtract the leading ExtendedEOF reconstruction.

        Reproduces:
            da - extract_leading_eeof_component(da).fillna(<last valid slice>)
        then fills any residual NaN with 0, exactly as legacy WAS_CCA does.

        combine='meof' warning
        ----------------------
        When ``da`` has shape (T, lat, lon, M), ExtendedEOF internally flattens
        all non-T dimensions into (lat x lon x M) features and applies a PCA
        pre-filter with n_pca_modes=22 (hard-coded inside
        ``extract_leading_eeof_component``).  If n_models x n_cells >> 22 the
        leading reconstructed mode may absorb genuine predictive signal rather
        than only the low-frequency trend.  Coarsen spatially or switch to
        combine='mean' if this is a concern.
        """
        trend = extract_leading_eeof_component(da)
        # the embedding leaves trailing NaNs in T; fill them with the last
        # valid reconstructed time slice (legacy used positional index -3)
        fill_slice = trend.isel(T=-3) if trend.sizes.get("T", 0) >= 3 else trend.isel(T=-1)
        trend = trend.fillna(fill_slice)
        return (da - trend).fillna(0)

    def _eeof_split_train(self, da):
        """Fold-safe EEOF residual plus a persisted component for unseen years.

        The EEOF is estimated on TRAIN only.  Validation/forecast predictors are
        moved to the same residual space by subtracting a persisted training
        component, and the analogous Y component is added back to predictions.
        """
        trend = extract_leading_eeof_component(da)
        idx=max(0, trend.sizes.get("T",1)-self.eeof_ref_offset)
        ref=trend.isel(T=idx,drop=True)
        resid=da-trend.fillna(ref)
        return resid.fillna(0), ref

    # ----------------------------------------------- probability calculation
    @staticmethod
    def _ppf_terciles_from_code(dist_code, shape, loc, scale):
        if np.isnan(dist_code):
            return np.nan, np.nan
        code = int(dist_code)
        try:
            if code == 1:
                return norm.ppf(1/3, loc=loc, scale=scale), norm.ppf(2/3, loc=loc, scale=scale)
            if code == 2:
                return lognorm.ppf(1/3, s=shape, loc=loc, scale=scale), lognorm.ppf(2/3, s=shape, loc=loc, scale=scale)
            if code == 3:
                return expon.ppf(1/3, loc=loc, scale=scale), expon.ppf(2/3, loc=loc, scale=scale)
            if code == 4:
                return gamma.ppf(1/3, a=shape, loc=loc, scale=scale), gamma.ppf(2/3, a=shape, loc=loc, scale=scale)
            if code == 5:
                return weibull_min.ppf(1/3, c=shape, loc=loc, scale=scale), weibull_min.ppf(2/3, c=shape, loc=loc, scale=scale)
            if code == 6:
                return t.ppf(1/3, df=shape, loc=loc, scale=scale), t.ppf(2/3, df=shape, loc=loc, scale=scale)
            if code == 7:
                return poisson.ppf(1/3, mu=shape, loc=loc), poisson.ppf(2/3, mu=shape, loc=loc)
            if code == 8:
                return nbinom.ppf(1/3, n=shape, p=scale, loc=loc), nbinom.ppf(2/3, n=shape, p=scale, loc=loc)
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
            return (V / (M ** 2)) - ((g2 / (g1 ** 2)) - 1)
        except Exception:
            return -np.inf

    @staticmethod
    def calculate_tercile_probabilities_bestfit(best_guess, error_variance, T1, T2, dist_code, dof):
        best_guess = np.asarray(best_guess, dtype=float)
        error_variance = float(error_variance)
        n_time = best_guess.size
        out = np.full((3, n_time), np.nan, dtype=float)

        if (np.all(np.isnan(best_guess)) or not np.isfinite(error_variance)
                or error_variance <= 0 or not np.isfinite(T1) or not np.isfinite(T2)
                or np.isnan(dist_code)):
            return out

        code = int(dist_code)
        try:
            if code == 1:
                s = np.sqrt(error_variance)
                c1, c2 = norm.cdf(T1, loc=best_guess, scale=s), norm.cdf(T2, loc=best_guess, scale=s)
                out[0, :], out[1, :], out[2, :] = c1, c2 - c1, 1.0 - c2
            elif code == 2:
                valid = best_guess > 0
                sigma = np.full(n_time, np.nan); mu = np.full(n_time, np.nan)
                sigma[valid] = np.sqrt(np.log(1.0 + error_variance / (best_guess[valid] ** 2)))
                mu[valid] = np.log(best_guess[valid]) - 0.5 * sigma[valid] ** 2
                c1, c2 = lognorm.cdf(T1, s=sigma, scale=np.exp(mu)), lognorm.cdf(T2, s=sigma, scale=np.exp(mu))
                out[0, :], out[1, :], out[2, :] = c1, c2 - c1, 1.0 - c2
            elif code == 3:
                scale = np.sqrt(error_variance); loc = best_guess - scale
                c1, c2 = expon.cdf(T1, loc=loc, scale=scale), expon.cdf(T2, loc=loc, scale=scale)
                out[0, :], out[1, :], out[2, :] = c1, c2 - c1, 1.0 - c2
            elif code == 4:
                valid = best_guess > 0
                alpha = np.full(n_time, np.nan); theta = np.full(n_time, np.nan)
                alpha[valid] = (best_guess[valid] ** 2) / error_variance
                theta[valid] = error_variance / best_guess[valid]
                c1, c2 = gamma.cdf(T1, a=alpha, scale=theta), gamma.cdf(T2, a=alpha, scale=theta)
                out[0, :], out[1, :], out[2, :] = c1, c2 - c1, 1.0 - c2
            elif code == 5:
                for i in range(n_time):
                    M, V = best_guess[i], error_variance
                    if not np.isfinite(M) or M <= 0 or V <= 0:
                        continue
                    k = fsolve(WAS_mme_CCA_eeof.weibull_shape_solver, 2.0, args=(M, V))[0]
                    if not np.isfinite(k) or k <= 0:
                        continue
                    lam = M / gamma_function(1 + 1 / k)
                    c1, c2 = weibull_min.cdf(T1, c=k, loc=0, scale=lam), weibull_min.cdf(T2, c=k, loc=0, scale=lam)
                    out[0, i], out[1, i], out[2, i] = c1, c2 - c1, 1.0 - c2
            elif code == 6:
                if dof <= 2:
                    return out
                scale = np.sqrt(error_variance * (dof - 2) / dof)
                c1, c2 = t.cdf(T1, df=dof, loc=best_guess, scale=scale), t.cdf(T2, df=dof, loc=best_guess, scale=scale)
                out[0, :], out[1, :], out[2, :] = c1, c2 - c1, 1.0 - c2
            elif code == 7:
                mu = np.where(best_guess >= 0, best_guess, np.nan)
                c1, c2 = poisson.cdf(T1, mu=mu), poisson.cdf(T2, mu=mu)
                out[0, :], out[1, :], out[2, :] = c1, c2 - c1, 1.0 - c2
            elif code == 8:
                valid = error_variance > best_guess
                p = np.where(valid, best_guess / error_variance, np.nan)
                n = np.where(valid, best_guess ** 2 / (error_variance - best_guess), np.nan)
                c1, c2 = nbinom.cdf(T1, n=n, p=p), nbinom.cdf(T2, n=n, p=p)
                out[0, :], out[1, :], out[2, :] = c1, c2 - c1, 1.0 - c2
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
    def calculate_tercile_probabilities_nonparametric(best_guess, error_samples, first_tercile, second_tercile):
        return WAS_mme_Weighted.calculate_tercile_probabilities_nonparametric(
            best_guess, error_samples, first_tercile, second_tercile)

    @staticmethod
    def _normalize_probabilities(prob):
        prob = prob.clip(min=0.0, max=1.0)
        total = prob.sum(dim="probability")
        return xr.where(total > 0, prob / total, np.nan)

    def _tercile_probabilities(self, Predictant, deterministic, clim_year_start, clim_year_end,
                               error_samples, error_variance,
                               best_code_da=None, best_shape_da=None,
                               best_loc_da=None, best_scale_da=None):
        Predictant = Predictant.transpose("T", "Y", "X")
        deterministic = deterministic.transpose("T", "Y", "X")
        mask = xr.where(np.isfinite(Predictant).sum(dim='T') >= 3, 1.0, np.nan)

        clim = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))
        if clim.sizes.get("T", 0) < 3:
            raise ValueError("Not enough years in climatology period for terciles.")
        # 0.30 was not a tercile at all: it cut the lower category at the
        # 30th percentile while the upper one started at the 67th, so the
        # three categories covered 30 / 37 / 33 percent of the climatology
        # instead of a third each. Every category-based score computed
        # against these thresholds was biased.
        terc = climatological_terciles(clim)[0]
        T1_emp = terc.isel(quantile=0, drop=True)
        T2_emp = terc.isel(quantile=1, drop=True)
        dof = max(int(clim.sizes["T"]) - int(getattr(self, "_last_effective_n_modes", self.n_modes)) - 1, 3)

        if self.dist_method == "bestfit":
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError("dist_method='bestfit' requires best_code_da, best_shape_da, best_loc_da, best_scale_da.")
            # Category boundaries stay on the observed climatological reference.
            # The selected distribution controls the predictive CDF only.
            if getattr(self, "tercile_reference", "climatology") == "training":
                T1, T2 = _pufunc(
                    self._ppf_terciles_from_code, best_code_da, best_shape_da, best_loc_da, best_scale_da,
                    input_core_dims=[(), (), (), ()], output_core_dims=[(), ()],
                    vectorize=True, dask="parallelized", output_dtypes=[float, float])
            else:
                T1, T2 = T1_emp, T2_emp
            prob = _pufunc(
                self.calculate_tercile_probabilities_bestfit,
                deterministic, error_variance, T1, T2, best_code_da,
                input_core_dims=[("T",), (), (), (), ()], output_core_dims=[("probability", "T")],
                vectorize=True, kwargs={"dof": dof}, dask="parallelized", output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True})
        elif self.dist_method == "nonparam":
            # rename residual sample axis so a length-1 forecast and the
            # length-n residual record don't collide on the shared 'T' core dim
            err = error_samples.rename({"T": "__samp"})
            prob = _pufunc(
                self.calculate_tercile_probabilities_nonparametric,
                deterministic, err, T1_emp, T2_emp,
                input_core_dims=[("T",), ("__samp",), (), ()], output_core_dims=[("probability", "T")],
                vectorize=True, dask="parallelized", output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True})
        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}")

        prob = prob.assign_coords(probability=("probability", ["PB", "PN", "PA"]))
        prob = self._normalize_probabilities(prob.transpose("probability", "T", "Y", "X"))
        return (prob * mask).transpose("probability", "T", "Y", "X")

    # ----------------------------------------------------- framework contract
    def _cca_problem(self, X_train, y_train, X_test, preprocessed=False):
        """Candidate-independent fold preprocessing with the transferable EEOF residual."""
        X_train = align_predictor_target(X_train, y_train, mode="season_year")
        if "M" in y_train.dims:
            y_train = y_train.isel(M=0, drop=True)
        field_train = self._field(X_train)
        field_test = self._field(X_test)
        if preprocessed or not self.eeof_detrend:
            return field_train, y_train, field_test, None
        Xres, refX = self._eeof_split_train(field_train)
        Yres, refY = self._eeof_split_train(y_train)
        Xtest_res = (field_test - refX).fillna(0)
        return Xres, Yres, Xtest_res, (lambda pred: pred + refY)

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det,
                     best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None):
        if "M" in Predictant.dims:
            Predictant = Predictant.isel(M=0, drop=True)
        Predictant = Predictant.transpose("T", "Y", "X")
        hindcast_det = hindcast_det.transpose("T", "Y", "X")
        if hindcast_det.sizes["T"] == Predictant.sizes["T"]:
            hindcast_det = hindcast_det.assign_coords(T=Predictant["T"])

        error_samples = Predictant - hindcast_det           # out-of-sample (LOYO hindcast)
        error_variance = (error_samples ** 2).mean(dim="T", skipna=True)

        return self._tercile_probabilities(
            Predictant, hindcast_det, clim_year_start, clim_year_end,
            error_samples, error_variance,
            best_code_da, best_shape_da, best_loc_da, best_scale_da)

    # ------------------------------------------------------------- operational
    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor,
                 hindcast_det, Predictor_for_year,
                 best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None):
        """
        Operational CCA forecast for a single target year, legacy WAS_CCA style:

          * predictor field EEOF-detrended (NOT standardized),
          * predictand standardized then EEOF-detrended,
          * forecast-year predictor only gap-filled (NOT detrended),
          * result reverse-standardized with NO trend added back.

        Truncation: as in ``WAS_mme_CCA.forecast`` (cached final selection,
        or a fresh full-record one under autotuning; constructor values
        restored afterwards).
        """
        _trunc_state = self.truncation_state()
        try:
            self._truncation_for_forecast(Predictor, Predictant, clim_year_start, clim_year_end)
            return self._forecast_impl(Predictant, clim_year_start, clim_year_end, Predictor,
                                       hindcast_det, Predictor_for_year, best_code_da,
                                       best_shape_da, best_loc_da, best_scale_da)
        finally:
            self.restore_truncation(_trunc_state)

    def _forecast_impl(self, Predictant, clim_year_start, clim_year_end, Predictor,
                       hindcast_det, Predictor_for_year,
                       best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None):
        if "M" in Predictant.dims:
            Predictant = Predictant.isel(M=0, drop=True)
        Predictant = Predictant.transpose("T", "Y", "X")
        mask = xr.where(np.isfinite(Predictant).sum(dim='T') >= 3, 1.0, np.nan)
        
        mean_val = Predictor.sel(T=slice(str(clim_year_start), str(clim_year_end))).mean(dim='T')
        std_val = Predictor.sel(T=slice(str(clim_year_start), str(clim_year_end))).std(dim='T')
        Predictor_for_year = (Predictor_for_year - mean_val) / std_val

        # ----- predictor/predictand: same fold-transferable EEOF residual space
        Predictor = standardize_timeseries(Predictor, clim_year_start, clim_year_end)
        field_hist = self._field(Predictor)
        Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end)
        field_year = self._field(Predictor_for_year)
        field_year = (field_year.fillna(field_hist.mean(dim="T", skipna=True))
                      .ffill(dim="Y").bfill(dim="Y")
                      .ffill(dim="X").bfill(dim="X").fillna(0))
        if self.eeof_detrend:
            field_hist_fit, refX = self._eeof_split_train(field_hist)
            Predictant_fit, refY = self._eeof_split_train(Predictant_st)
            field_year_fit = (field_year - refX).fillna(0)
        else:
            field_hist_fit = field_hist.fillna(field_hist.mean(dim="T", skipna=True)).fillna(0)
            Predictant_fit = Predictant_st.fillna(0)
            field_year_fit = field_year
        pred_st = self._fit_predict_cca(field_hist_fit, Predictant_fit, field_year_fit)
        if self.eeof_detrend:
            pred_st = pred_st + refY
        forecast_det = reverse_standardize(pred_st, Predictant, clim_year_start, clim_year_end)
        if self.positive_support:
            forecast_det = forecast_det.clip(min=0)

        # target time = forecast year + predictand target month
        year = Predictor_for_year.coords["T"].values.astype("datetime64[Y]").astype(int)[0] + 1970
        month = Predictant.isel(T=0).coords["T"].values.astype("datetime64[M]").astype(int) % 12 + 1
        new_T = np.datetime64(f"{year}-{month:02d}-01")
        forecast_det = forecast_det.assign_coords(T=xr.DataArray([new_T], dims=["T"]))
        forecast_det["T"] = forecast_det["T"].astype("datetime64[ns]")

        error_samples = Predictant - hindcast_det.transpose("T", "Y", "X")
        error_variance = (error_samples ** 2).mean(dim="T", skipna=True)
        forecast_prob = self._tercile_probabilities(
            Predictant, forecast_det, clim_year_start, clim_year_end,
            error_samples, error_variance,
            best_code_da, best_shape_da, best_loc_da, best_scale_da)

        return (forecast_det * mask, forecast_prob * mask)



def _mme_detrend_training(field):
    """Estimate short-fold trends; insufficient local data retains no trend."""
    _, coeffs, meta = detrended_data(field, dim="T", min_valid=3)
    coeffs = coeffs.fillna(0.)
    residual = field - apply_detrend_data(field, coeffs, meta)
    residual.attrs = field.attrs.copy()
    return residual, coeffs, meta


def _regional_tuning_space(model):
    """Declared candidate spaces and fresh estimator factory (no data fitting)."""
    if isinstance(model, _WAS_mme_SklearnRegressor):
        return model.param_grid, model._make_estimator, False
    if isinstance(model, WAS_mme_logistic):
        return model._create_parameter_grids(), lambda **p: LogisticRegression(
            random_state=model.random_state, max_iter=1000, **p), True
    if isinstance(model, WAS_mme_gaussian_process):
        return model._create_param_grid(), lambda **p: GaussianProcessClassifier(
            random_state=model.random_state, warm_start=False, **p), True
    if isinstance(model, WAS_mme_hpELM):
        space = dict(neurons=model.neurons_range, activation=model.activation_options,
                     norm=model.norm_range)
        scaling = getattr(model, 'norm_scaling', 'absolute')
        return space, lambda **p: HPELMWrapper(random_state=model.random_state,
                                               norm_scaling=scaling, **_elm_structure(model),
                                               **p), False
    if isinstance(model, WAS_mme_XGBoosting):
        if any(v is not None and v != 0 for v in model.early_stopping_rounds_range):
            raise ValueError(
                'Strict regional XGBoost tuning requires early_stopping_rounds_range=[None]. '
                'Select n_estimators in the year-grouped search; the historical stopping '
                'subsplit does not refit its per-pixel preprocessing. '
                'compute_hyperparameters_legacy is available for explicit historical comparisons.')
        return model._param_space(), lambda **p: _XGBRegressorES(
            random_state=model.random_state, verbosity=0, n_jobs=model.tuning_model_n_jobs,
            es_validation_fraction=model.es_validation_fraction, **p), False
    if isinstance(model, WAS_mme_RF):
        base = {name: getattr(model, name + '_range') for name in (
            'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf',
            'max_features', 'min_impurity_decrease', 'ccp_alpha')}
        space = []
        if True in model.bootstrap_range:
            space.append(dict(base, bootstrap=[True], max_samples=model.max_samples_range))
        if False in model.bootstrap_range:
            space.append(dict(base, bootstrap=[False], max_samples=[None]))
        return space, lambda **p: RandomForestRegressor(
            random_state=model.random_state, n_jobs=model.tuning_model_n_jobs, **p), False
    if isinstance(model, WAS_mme_MLP):
        base = dict(hidden_layer_sizes=model.hidden_layer_sizes_range,
                    activation=model.activation_options, alpha=model.alpha_range,
                    early_stopping=[False])
        space = []
        iterative = [v for v in model.solver_options if v in ('adam', 'sgd')]
        if iterative:
            space.append(dict(base, solver=iterative, learning_rate_init=model.learning_rate_init_range))
        if 'lbfgs' in model.solver_options:
            space.append(dict(base, solver=['lbfgs'], learning_rate_init=[.001]))
        return space, lambda **p: MLPRegressor(
            max_iter=model.max_iter, random_state=model.random_state, **p), False
    raise TypeError(f'No strict regional tuning space for {type(model).__name__}.')


def _regional_tuning_folds(model, X, y, clim_start=None, clim_end=None, *, target_is_class=False):
    """Fit regionalization and per-pixel transformations within each inner fold."""
    def _fit_standardizer(z, start=None, end=None):
        ref = z.sel(T=slice(str(start), str(end))) if start is not None and end is not None else z
        if ref.sizes['T'] < 2:
            ref = z
        mean, scale = ref.mean('T', skipna=True), ref.std('T', skipna=True)
        return mean, scale.where(np.isfinite(scale) & (scale > 1e-12), 1.)
    years = np.asarray(y['T'].dt.year.values, int)
    if len(np.unique(years)) != len(years) or np.any(np.diff(years) <= 0):
        raise ValueError('Regional MME tuning requires one sorted target season per year.')
    splitter = _year_grouped_splitter(years, model.cv_folds,
                                      getattr(model, 'leave_one_year_out', False))
    _, _, classifier = _regional_tuning_space(model)
    packs = []
    gap = int(getattr(model, 'tuning_gap_years', 0))
    for tr, va in splitter.split(np.zeros((len(years), 1)), groups=years):
        if gap:
            tr = tr[np.min(np.abs(years[tr, None] - years[va]), axis=1) > gap]
        if len(tr) < 2:
            raise ValueError('Too few years in regional tuning after temporal exclusion.')
        xi, yi = X.isel(T=tr), y.isel(T=tr)
        xm, xs = _fit_standardizer(xi, clim_start, clim_end)
        xv = (X.isel(T=va) - xm) / xs
        xi = (xi - xm) / xs
        # Zones and spatial covariates from the RAW inner-training predictand
        # (yi is standardized only below), so nothing of the validation years
        # enters the regionalisation.
        clusters = _regionalisation(model, yi)
        covariates = _model_covariates(model, yi, clim_start, clim_end)
        smoothing = getattr(model, 'predictor_smoothing', 0)
        if covariates is not None or smoothing:
            xi = _augment_predictors(xi, covariates, smoothing)
            xv = _augment_predictors(xv, covariates, smoothing)
        yv = y.isel(T=va)
        if classifier:
            if not target_is_class:
                ref = yi.sel(T=slice(str(clim_start), str(clim_end))) if (
                    clim_start is not None and clim_end is not None) else yi
                if ref.sizes['T'] < 2:
                    ref = yi
                q = ref.quantile([1/3, 2/3], dim='T')
                q1, q2 = q.isel(quantile=0, drop=True), q.isel(quantile=1, drop=True)
                def encode(z):
                    return xr.where(z <= q1, 0., xr.where(z <= q2, 1., 2.)).where(
                        np.isfinite(z) & np.isfinite(q1) & np.isfinite(q2))
                yi, yv = encode(yi), encode(yv)
            ym, ys = None, None
        else:
            ym, ys = _fit_standardizer(yi, clim_start, clim_end)
            yi, yv = (yi-ym)/ys, (yv-ym)/ys
        packs.append(dict(
            train_years=years[tr], validation_years=years[va], clusters=clusters,
            input_mean=xm, input_std=xs, target_mean=ym, target_std=ys,
            Xtrain=xi.transpose('T','Y','X','M').values,
            Xvalid=xv.transpose('T','Y','X','M').values,
            ytrain=yi.transpose('T','Y','X').values,
            yvalid=yv.transpose('T','Y','X').values))
    return packs


def _elm_batched_regional_scores(model, packs, candidates, stride, thin_cells, scoring, results):
    """Strict regional scores of all ELM candidates, one Gram pass per fold and zone.

    Same masks, thinning and per-year scores as the generic ``evaluate`` of
    :func:`_strict_regional_hyperparameters`; the predictions of every
    candidate come from one :class:`~wass2s.was_elm.ELMGramBank`.
    """
    from sklearn.metrics import mean_absolute_error, r2_score
    cands = [dict(c) for c in candidates]
    keys = [(str(c['activation']), int(c['neurons']), float(c['norm'])) for c in cands]
    activations = list(dict.fromkeys(k[0] for k in keys))
    max_neurons = max(k[1] for k in keys)
    seeds = tuple(int(model.random_state) + i for i in range(
        max(1, int(getattr(model, 'n_seeds', 1))) if max_neurons > 0 else 1))
    scaling = getattr(model, 'norm_scaling', 'absolute')
    linear = bool(getattr(model, 'linear_part', False))
    year_scores = [[] for _ in cands]
    errors = [None] * len(cands)
    for pack in packs:
        a, b, q, truth = (pack[k] for k in ('Xtrain', 'ytrain', 'Xvalid', 'yvalid'))
        labels = pack['clusters'].values
        scored = np.isfinite(q).all(-1) & np.broadcast_to(np.isfinite(labels), truth.shape)
        if stride > 1:
            thinned = scored & np.broadcast_to(thin_cells, truth.shape)
            if thinned.sum() >= _MIN_TUNING_VALID_CELLS:
                scored = thinned
        predictions = [np.full(truth.shape, np.nan) for _ in cands]
        for c in np.unique(labels[np.isfinite(labels)]):
            train_mask = (np.broadcast_to(labels == c, b.shape) & np.isfinite(b)
                          & np.isfinite(a).all(-1) & np.broadcast_to(thin_cells, b.shape))
            valid_mask = np.broadcast_to(labels == c, truth.shape) & scored
            if not train_mask.any() or not valid_mask.any():
                continue
            try:
                bank = ELMGramBank(a[train_mask], b[train_mask], activations, max_neurons,
                                   seeds, scaling, linear)
                preds = bank.predict(q[valid_mask], keys)
            except Exception as exc:
                for i in range(len(cands)):
                    errors[i] = errors[i] or f'{type(exc).__name__}: {exc}'
                continue
            for i, pv in enumerate(preds):
                predictions[i][valid_mask] = pv
        for i in range(len(cands)):
            if errors[i]:
                continue
            before = len(year_scores[i])
            for j in range(len(truth)):
                good = np.isfinite(truth[j]) & scored[j]
                if not good.any():
                    continue
                pj, tj = predictions[i][j][good], truth[j][good]
                if not np.isfinite(pj).all():
                    errors[i] = 'ValueError: A candidate failed to predict the common validation sample.'
                    break
                if scoring == 'neg_mean_absolute_error':
                    year_scores[i].append(-mean_absolute_error(tj, pj))
                elif scoring == 'r2':
                    year_scores[i].append(r2_score(tj, pj))
                else:
                    loss = float(np.mean((tj - pj) ** 2))
                    year_scores[i].append(-np.sqrt(loss) if scoring == 'neg_root_mean_squared_error' else -loss)
            if not errors[i] and len(year_scores[i]) - before != len(truth):
                errors[i] = 'ValueError: Every validation year must have a finite common score.'
    for i, params in enumerate(cands):
        ys = year_scores[i]
        ok = errors[i] is None and ys and np.isfinite(ys).all()
        results.append(dict(params=params, score=float(np.mean(ys)) if ok else -np.inf,
                            year_scores=ys, error=None if ok else (errors[i] or 'no finite score')))


def _strict_regional_hyperparameters(model, Predictors, Predictand,
                                     clim_year_start=None, clim_year_end=None,
                                     *, target_is_class=False):
    """Select shared hyperparameters while refitting climate zones in every fold.

    Cluster numbers do not identify the same climate regime across folds. A
    common candidate is therefore evaluated across all fold-local regional
    models, then used for every final region. Final regression coefficients or
    trees remain regional. ``tuning_results_`` contains selection scores, never
    an independent estimate of the final forecast skill.
    """
    from sklearn.model_selection import ParameterGrid, ParameterSampler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    X = align_predictor_target(Predictors, Predictand, mode='season_year')
    y = Predictand.isel(M=0, drop=True) if 'M' in Predictand.dims else Predictand
    if isinstance(model, _WAS_mme_SklearnRegressor) and y.sizes['T'] < model.min_train_years:
        raise ValueError(f'{model.estimator_name} requires at least {model.min_train_years} training years.')
    if 'M' not in X.dims:
        X = X.expand_dims(M=[0])
    spaces, factory, classifier = _regional_tuning_space(model)
    if classifier and target_is_class:
        categories = np.unique(y.values[np.isfinite(y.values)])
        if not np.isin(categories, [0, 1, 2]).all():
            raise ValueError('target_is_class=True requires fixed labels 0, 1, 2.')
    spaces = spaces if isinstance(spaces, list) else [spaces]
    if not spaces:
        raise ValueError('The regional tuning search space is empty.')
    packs = _regional_tuning_folds(model, X, y, clim_year_start, clim_year_end,
                                   target_is_class=target_is_class)
    scoring = getattr(model, 'scoring', 'neg_mean_squared_error')
    if not classifier and scoring not in {
            'neg_mean_squared_error', 'neg_root_mean_squared_error', 'neg_mean_absolute_error', 'r2'}:
        raise ValueError('Strict regional tuning supports MSE, RMSE, MAE and R2 scoring.')
    results = []
    # The tuning sample is thinned exactly like the final fit (spatial_subsample
    # in Y and X), otherwise hyperparameters are selected for a sample
    # stride**2 times larger than the one they are eventually used on.
    stride = int(getattr(model, 'spatial_subsample', 1) or 1)
    thin_cells = np.zeros((y.sizes['Y'], y.sizes['X']), dtype=bool)
    thin_cells[::stride, ::stride] = True
    # Effective sample size of the pooled fit (Bretherton et al. 1999).
    if packs:
        first = packs[0]
        n_rows = int(np.isfinite(first['ytrain']).sum() // max(1, stride ** 2))
        n_feat = int(first['Xtrain'].shape[-1])
        n_years = int(len(first['train_years']))
        n_eff_spatial = _effective_spatial_dof(xr.DataArray(
            first['ytrain'] if not classifier else first['Xtrain'][..., 0],
            dims=('T', 'Y', 'X')))
        model.effective_sample_ = dict(
            rows=n_rows, years=n_years, features=n_feat, n_eff_spatial=n_eff_spatial,
            n_eff_total=_sample_size_report(type(model).__name__, n_rows, n_years, n_feat,
                                            n_eff_spatial, verbose=bool(getattr(model, 'verbose', 0))))

    def evaluate(params):
        fold_scores = []
        try:
            for pack in packs:
                a, b, q, truth = (pack[k] for k in ('Xtrain','ytrain','Xvalid','yvalid'))
                prediction = np.full((*truth.shape, 3) if classifier else truth.shape, np.nan)
                labels = pack['clusters'].values

                # Cellules notees pour ce pli. L'echantillon de validation est
                # aminci comme l'entrainement : sans cela on s'entraine sur un
                # pixel sur stride^2 mais on predit sur la grille pleine a
                # chaque pli interne x cluster x candidat. Mesure sur une grille
                # 170x510 (45 505 pixels, 21 ans) : le reglage coutait ~30 min
                # par pli contre 32 s pour l'ajustement final, soit 25x le cout
                # utile, avec UNE seule combinaison candidate. Les pixels
                # voisins etant fortement correles, le score estime sur
                # l'echantillon aminci suffit a classer des candidats.
                #
                # Le MEME masque sert a la prediction et a la notation plus bas,
                # sinon le scoreur reclame des cellules jamais predites. On
                # n'amincit que si l'echantillon reste suffisant : sur un petit
                # domaine, stride^2 peut le vider.
                scored = np.isfinite(q).all(-1) & np.broadcast_to(
                    np.isfinite(labels), truth.shape)
                if stride > 1:
                    thinned = scored & np.broadcast_to(thin_cells, truth.shape)
                    if thinned.sum() >= _MIN_TUNING_VALID_CELLS:
                        scored = thinned

                for c in np.unique(labels[np.isfinite(labels)]):
                    train_mask = (np.broadcast_to(labels == c, b.shape) & np.isfinite(b)
                                  & np.isfinite(a).all(-1) & np.broadcast_to(thin_cells, b.shape))
                    valid_mask = np.broadcast_to(labels == c, truth.shape) & scored
                    if not train_mask.any() or not valid_mask.any():
                        continue
                    est = factory(**params)
                    train_a, train_b = a[train_mask], b[train_mask]
                    groups = np.broadcast_to(pack['train_years'][:,None,None], b.shape)[train_mask]
                    # GP subsampling remains reproducible and preserves group labels.
                    limit = int(getattr(model, 'max_train_samples', 0) or 0)
                    if limit and len(train_b) > limit:
                        idx = np.random.RandomState(model.random_state).choice(len(train_b), limit, replace=False)
                        train_a, train_b, groups = train_a[idx], train_b[idx], groups[idx]
                    if classifier and len(np.unique(train_b)) == 1:
                        pp = np.zeros((int(valid_mask.sum()), 3)); pp[:,int(train_b[0])] = 1.
                        prediction[valid_mask] = pp
                    else:
                        fit_kwargs = {'es_groups': groups} if isinstance(est, _XGBRegressorES) else {}
                        est.fit(train_a, train_b.astype(int) if classifier else train_b, **fit_kwargs)
                        if classifier:
                            pp = np.zeros((int(valid_mask.sum()), 3))
                            for j, label in enumerate(est.classes_):
                                pp[:,int(label)] = est.predict_proba(q[valid_mask])[:,j]
                            prediction[valid_mask] = pp
                        else:
                            prediction[valid_mask] = np.asarray(est.predict(q[valid_mask])).ravel()
                annual = []
                for j in range(len(truth)):
                    # meme masque que la prediction ci-dessus (amincissement compris)
                    good = np.isfinite(truth[j]) & scored[j]
                    if not good.any():
                        continue
                    pj = prediction[j][good]
                    if not np.isfinite(pj).all():
                        raise ValueError('A candidate failed to predict the common validation sample.')
                    tj = truth[j][good]
                    if classifier:
                        annual.append(-log_loss(tj.astype(int), pj, labels=[0,1,2]))
                    elif scoring == 'neg_mean_absolute_error':
                        annual.append(-mean_absolute_error(tj, pj))
                    elif scoring == 'r2':
                        annual.append(r2_score(tj, pj))
                    else:
                        loss = mean_squared_error(tj, pj)
                        annual.append(-np.sqrt(loss) if scoring == 'neg_root_mean_squared_error' else -loss)
                if len(annual) != len(truth) or not np.isfinite(annual).all():
                    raise ValueError('Every validation year must have a finite common score.')
                fold_scores.extend(annual)
            score = float(np.mean(fold_scores))
            error = None
        except Exception as exc:
            score, error = -np.inf, f'{type(exc).__name__}: {exc}'
        results.append(dict(params=dict(params), score=score, year_scores=fold_scores, error=error))
        return score

    if isinstance(model, WAS_mme_hpELM) and not classifier and getattr(model, 'elm_batched_tuning', True):
        # Every (activation, neurons, norm) candidate from one Gram pass per
        # fold and zone (nested random layer, ridge on the diagonal only).
        # A Bayesian search would be sequential by construction; the full
        # grid costs less here, so 'bayesian' evaluates the grid.
        _method = getattr(model, 'search_method', 'grid').lower()
        if _method == 'random':
            _cands = list(ParameterSampler(spaces, n_iter=int(getattr(model, 'n_iter_search', 20)),
                                           random_state=model.random_state))
        else:
            _material = [{k: list(np.unique(v.ppf(np.linspace(.1, .9, 5)))) if hasattr(v, 'rvs')
                          else list(v) for k, v in space.items()} for space in spaces]
            _cands = list(ParameterGrid(_material))
        _elm_batched_regional_scores(model, packs, _cands, stride, thin_cells, scoring, results)
    else:
        method = getattr(model, 'optimization_method', getattr(model, 'hpo_method',
                         getattr(model, 'search_method', 'grid'))).lower()
        n_iter = int(getattr(model, 'n_iter_search', getattr(model, 'n_random_iter', 20)))
        if method == 'grid':
            material = []
            for space in spaces:
                material.append({k: list(np.unique(v.ppf(np.linspace(.1,.9,5))))
                                 if hasattr(v, 'rvs') else list(v) for k,v in space.items()})
            for candidate in ParameterGrid(material):
                evaluate(candidate)
        elif method == 'random':
            for candidate in ParameterSampler(spaces, n_iter=n_iter, random_state=model.random_state):
                evaluate(candidate)
        elif method == 'bayesian':
            if optuna is None:
                raise ImportError('Bayesian regional tuning requires optuna.')
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=model.random_state))
            def objective(trial):
                si = trial.suggest_int('space', 0, len(spaces)-1) if len(spaces)>1 else 0
                params = {}
                for k,v in spaces[si].items():
                    # Quantile coordinates preserve the declared scipy prior.
                    if hasattr(v, 'rvs'):
                        params[k] = v.ppf(trial.suggest_float(f'{si}:{k}:quantile', 1e-6, 1-1e-6))
                        if np.issubdtype(np.asarray(v.rvs(random_state=model.random_state)).dtype, np.integer):
                            params[k] = int(params[k])
                    else:
                        options = list(v)
                        params[k] = options[trial.suggest_int(f'{si}:{k}:index', 0, len(options)-1)]
                return evaluate(params)
            study.optimize(objective, n_trials=int(getattr(model, 'n_trials_bayesian',
                           getattr(model, 'n_trials', n_iter))),
                           timeout=getattr(model, 'timeout', getattr(model, 'optuna_timeout', None)),
                           n_jobs=1)
            model.strict_tuning_study_ = study
        else:
            raise ValueError(f'Unknown regional search method {method!r}.')
    good = [v for v in results if np.isfinite(v['score'])]
    model.tuning_results_ = results
    model.tuning_diagnostics_ = [{k:v for k,v in p.items() if k not in
        {'Xtrain','Xvalid','ytrain','yvalid'}} for p in packs]
    if not good:
        detail = results[0]['error'] if results else 'No candidate.'
        raise ValueError(f'All strict regional tuning candidates failed. {detail}')
    if getattr(model, 'one_se_rule', False):
        # The mean over ~24 validation years is a noisy selection statistic:
        # take the most regularized candidate within one standard error of
        # the best score (as tune_modes does for PCR/CCA).
        chosen, selection = _one_se_select(good)
    else:
        chosen = max(good, key=lambda v:v['score'])
        selection = {'rule': 'max', 'best_score': float(chosen['score']),
                     'n_candidates': len(good)}
    best = chosen['params']
    model.tuning_selection_ = selection
    if getattr(model, 'verbose', 0):
        print(f"[{type(model).__name__}] selection={selection['rule']} params={best}", flush=True)
    final_clusters = _regionalisation(model, y)
    final_clusters.attrs['hyperparameter_scope'] = 'shared_across_fold_local_regions'
    final_clusters.attrs['inner_preprocessing'] = 'training_only'
    params = {int(c):dict(best) for c in np.unique(final_clusters.values[np.isfinite(final_clusters.values)])}
    model.best_params_ = params
    model.best_params_dict = params
    model.cluster_da_ = final_clusters
    # Spatial covariates of the final fit (raw predictand, same years as the
    # zone map) so compute_model/forecast reuse them without recomputation.
    model.covariates_ = _model_covariates(model, y, clim_year_start, clim_year_end)
    return params, final_clusters
