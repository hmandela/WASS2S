"""Deterministic and probabilistic forecast verification (numba-accelerated).

Provides pixel-wise and domain-averaged scores compatible with the
WASS2S xarray data model (dimensions T, Y, X).

Class
-----
WAS_Verification
    Container for all verification methods.  Scores are applied
    grid-point by grid-point via ``compute_deterministic_score``, or
    computed directly over spatial maps.

Deterministic scores
    Pearson correlation, Spearman correlation, Index of Agreement,
    Nash-Sutcliffe Efficiency, Kling-Gupta Efficiency, MAE, RMSE,
    Taylor diagram.

Probabilistic scores
    GROC (Generalized ROC), RPSS, Ignorance Score, Brier Score,
    Brier Skill Score, Resolution Score.

Tercile utilities
    ``classify`` - assign observations to below / near / above-normal
    tercile classes.
    ``compute_class`` - classify a full DataArray over a climatological
    reference period.
    ``classify_data_into_terciles`` - binary classification with
    pre-computed thresholds.

Acceleration
------------
The per-grid-point kernels are compiled with numba (``@njit``). Climatological
thresholds use the original observation time axis, before forecast-validity
masking. Numerical equivalence is checked against the NumPy reference path.

Two design choices matter and are deliberate:

* ``fastmath`` is **off**.  It would let the compiler reorder floating-point
  reductions, and the resulting differences show up right at the
  ``get_best_models(metric='GROC', threshold=0.6)`` cut, where a borderline
  model would flip in or out of the MME between runs.
* ``error_model="numpy"`` is **on**, so ``x / 0.0`` returns +/-inf rather than
  raising.  A constant observed series must still give ``NSE = -inf``, as the
  pure-Python version did.

Kernels are ``nogil=True``, so the thread pool in ``compute_deterministic_score``
and ``compute_probabilistic_score`` finally scales - the interpreted kernels
held the GIL and capped that speedup.

Environment variables
---------------------
WAS_NCORES
    Worker threads for the grid-parallel path.  Default: ``os.cpu_count()``.
WAS_VERIF_PARALLEL
    Set to ``"0"`` to force the eager (single-threaded) path.
WAS_VERIF_MIN_CELLS
    Grid-size floor below which the eager path is used anyway.  Default 4000.
WAS_VERIF_NUMBA
    Set to ``"0"`` to bypass the compiled kernels entirely and run the
    reference NumPy/scikit-learn implementations (useful for A/B checks).
NUMBA_CACHE_DIR
    Where to write the compilation cache.  Needed when the install prefix is
    read-only, which is the usual case on a cluster.
"""

import os
import calendar
from pathlib import Path

import numpy as np
import xarray as xr

from scipy.stats import pearsonr, norm, linregress
from scipy.stats import lognorm, gamma, expon, weibull_min, t as student_t
from scipy.stats import poisson, nbinom
from scipy.special import gamma as gamma_function
from scipy.optimize import fsolve
from scipy import stats

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors as mcolors
from matplotlib.colors import ListedColormap

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import properscoring
import xskillscore as xs

from wass2s.utils import *


# ==========================================================================
# numba, with a graceful fallback
# ==========================================================================

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:                                         # pragma: no cover
    NUMBA_AVAILABLE = False
    prange = range

    def njit(*args, **kwargs):
        """No-op stand-in so the module imports without numba installed."""
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


def _use_numba():
    """Compiled kernels active?  ``WAS_VERIF_NUMBA=0`` forces the reference path."""
    return NUMBA_AVAILABLE and os.environ.get("WAS_VERIF_NUMBA", "1") != "0"


_JIT = dict(cache=True, nogil=True, fastmath=False, error_model="numpy")


# ==========================================================================
# Thread-pool helpers (unchanged behaviour)
# ==========================================================================

def _verif_num_workers():
    """Worker count for grid-score parallelism: WAS_NCORES if set by the CLI,
    else the CPU count. Clamped to >= 1."""
    try:
        n = int(os.environ.get("WAS_NCORES", "0"))
        if n <= 0:
            n = os.cpu_count() or 1
        return max(1, n)
    except Exception:
        return 1


def _chunk_grid(da, n):
    """Chunk a (..., Y, X) field along Y (then X if Y is small) so per-grid-point
    score cells are spread across `n` worker threads. Returns `da` unchanged if
    it has no Y dimension."""
    if not hasattr(da, "dims") or "Y" not in da.dims:
        return da
    ny = int(da.sizes.get("Y", 1))
    chunks = {"Y": max(1, ny // max(1, n))}
    if ny < n and "X" in da.dims:
        nx = int(da.sizes.get("X", 1))
        chunks["X"] = max(1, nx // max(1, n))
    try:
        return da.chunk(chunks)
    except Exception:
        return da


# ==========================================================================
# Compiled low-level helpers
# ==========================================================================

@njit(**_JIT)
def _nanpercentile2(x, q1, q2):
    """Two percentiles of ``x`` ignoring NaN, matching numpy's default
    ('linear') interpolation.  Returns ``(nan, nan)`` on an all-NaN input."""
    n = 0
    for i in range(x.size):
        if not np.isnan(x[i]):
            n += 1
    if n == 0:
        return np.nan, np.nan

    buf = np.empty(n, np.float64)
    k = 0
    for i in range(x.size):
        if not np.isnan(x[i]):
            buf[k] = x[i]
            k += 1
    buf.sort()

    out = np.empty(2, np.float64)
    qs = np.empty(2, np.float64)
    qs[0] = q1
    qs[1] = q2
    for j in range(2):
        pos = qs[j] / 100.0 * (n - 1)
        lo = int(np.floor(pos))
        hi = int(np.ceil(pos))
        if lo == hi:
            out[j] = buf[lo]
        else:
            out[j] = buf[lo] + (pos - lo) * (buf[hi] - buf[lo])
    return out[0], out[1]


@njit(**_JIT)
def _digitize_right(v, t1, t2):
    """``np.digitize(v, [t1, t2], right=True)`` for a scalar."""
    if v <= t1:
        return 0
    elif v <= t2:
        return 1
    return 2


@njit(**_JIT)
def _digitize_left(v, t1, t2):
    """``np.digitize(v, [t1, t2], right=False)`` for a scalar."""
    if v < t1:
        return 0
    elif v < t2:
        return 1
    return 2


@njit(**_JIT)
def _auc_binary(y_bin, score):
    """AUC of a binary problem via the Mann-Whitney statistic.

    Exactly equal to ``sklearn.metrics.auc(*roc_curve(y_bin, score)[:2])``: the
    trapezoidal area under an ROC curve *is* the normalised rank-sum statistic,
    provided ties are resolved with mid-ranks - which the tie-averaging loop
    below does.  The curve is never built.

    Returns 0.5 when one class is absent, matching ``calculate_groc``.
    """
    n = score.size
    if n == 0:
        return 0.5

    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(n, np.float64)

    i = 0
    while i < n:
        j = i
        while j + 1 < n and score[order[j + 1]] == score[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    rank_sum_pos = 0.0
    n_pos = 0
    for i in range(n):
        if y_bin[i] == 1:
            rank_sum_pos += ranks[i]
            n_pos += 1
    n_neg = n - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


@njit(**_JIT)
def _rankdata_average(x):
    """Mid-ranks of ``x`` (ties averaged), as used by Spearman."""
    n = x.size
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


@njit(**_JIT)
def _pearson_raw(a, b):
    """Pearson r on two clean (finite) 1-D arrays."""
    n = a.size
    if n < 2:
        return np.nan
    ma = 0.0
    mb = 0.0
    for i in range(n):
        ma += a[i]
        mb += b[i]
    ma /= n
    mb /= n

    sab = 0.0
    saa = 0.0
    sbb = 0.0
    for i in range(n):
        da = a[i] - ma
        db = b[i] - mb
        sab += da * db
        saa += da * da
        sbb += db * db
    den = np.sqrt(saa) * np.sqrt(sbb)
    if den == 0.0:
        return np.nan
    return sab / den


@njit(**_JIT)
def _clean_pair(y_true, y_pred):
    """Finite-only copies of two aligned series, plus the retained count."""
    n = y_true.size
    cnt = 0
    for i in range(n):
        if np.isfinite(y_true[i]) and np.isfinite(y_pred[i]):
            cnt += 1
    a = np.empty(cnt, np.float64)
    b = np.empty(cnt, np.float64)
    k = 0
    for i in range(n):
        if np.isfinite(y_true[i]) and np.isfinite(y_pred[i]):
            a[k] = y_true[i]
            b[k] = y_pred[i]
            k += 1
    return a, b, cnt


@njit(**_JIT)
def _clean_prob(y_true, y_probs):
    """Finite-only copies of a series and its (n_classes, T) probabilities."""
    n_classes = y_probs.shape[0]
    nt = y_true.size

    cnt = 0
    for i in range(nt):
        ok = np.isfinite(y_true[i])
        if ok:
            for c in range(n_classes):
                if not np.isfinite(y_probs[c, i]):
                    ok = False
                    break
        if ok:
            cnt += 1

    yc = np.empty(cnt, np.float64)
    pc = np.empty((n_classes, cnt), np.float64)
    k = 0
    for i in range(nt):
        ok = np.isfinite(y_true[i])
        if ok:
            for c in range(n_classes):
                if not np.isfinite(y_probs[c, i]):
                    ok = False
                    break
        if ok:
            yc[k] = y_true[i]
            for c in range(n_classes):
                pc[c, k] = y_probs[c, i]
            k += 1
    return yc, pc, cnt


# ==========================================================================
# Compiled deterministic kernels
# ==========================================================================

@njit(**_JIT)
def _kge_core(y_true, y_pred):
    a, b, cnt = _clean_pair(y_true, y_pred)
    if cnt <= 2:
        return np.nan
    r = _pearson_raw(a, b)
    ma = a.mean()
    mb = b.mean()
    sa = a.std()
    sb = b.std()
    alpha = sb / sa
    beta = mb / ma
    return 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)


@njit(**_JIT)
def _pearson_core(y_true, y_pred):
    a, b, cnt = _clean_pair(y_true, y_pred)
    if cnt <= 2:
        return np.nan
    return _pearson_raw(a, b)


@njit(**_JIT)
def _spearman_core(y_true, y_pred):
    a, b, cnt = _clean_pair(y_true, y_pred)
    if cnt <= 2:
        return np.nan
    return _pearson_raw(_rankdata_average(a), _rankdata_average(b))


@njit(**_JIT)
def _ioa_core(y_true, y_pred):
    a, b, cnt = _clean_pair(y_true, y_pred)
    if cnt <= 2:
        return np.nan
    ma = a.mean()
    num = 0.0
    den = 0.0
    for i in range(cnt):
        num += (b[i] - a[i]) ** 2
        den += (abs(b[i] - ma) + abs(a[i] - ma)) ** 2
    return 1.0 - num / den


@njit(**_JIT)
def _mae_core(y_true, y_pred):
    n = y_true.size
    cnt = 0
    acc = 0.0
    for i in range(n):
        if np.isfinite(y_true[i]) and np.isfinite(y_pred[i]):
            acc += abs(y_true[i] - y_pred[i])
            cnt += 1
    if cnt <= 2:
        return np.nan
    return acc / cnt


@njit(**_JIT)
def _rmse_core(y_true, y_pred):
    n = y_true.size
    cnt = 0
    acc = 0.0
    for i in range(n):
        if np.isfinite(y_true[i]) and np.isfinite(y_pred[i]):
            d = y_true[i] - y_pred[i]
            acc += d * d
            cnt += 1
    if cnt <= 2:
        return np.nan
    return np.sqrt(acc / cnt)


@njit(**_JIT)
def _nse_core(y_true, y_pred):
    a, b, cnt = _clean_pair(y_true, y_pred)
    if cnt <= 2:
        return np.nan
    ma = a.mean()
    num = 0.0
    den = 0.0
    for i in range(cnt):
        num += (a[i] - b[i]) ** 2
        den += (a[i] - ma) ** 2
    # No zero guard: a constant observed series must give -inf, as numpy did.
    return 1.0 - num / den


# ==========================================================================
# Compiled probabilistic kernels
# ==========================================================================

@njit(**_JIT)
def _groc_core(y_true, y_probs, index_start, index_end, n_classes):
    # Terciles come from the climatology slice of the UNMASKED series - that is
    # the order the reference implementation uses; do not "tidy" it, the scores
    # would shift.
    if index_end <= index_start:
        return np.nan

    m = index_end - index_start
    clim = np.empty(m, np.float64)
    for i in range(m):
        clim[i] = y_true[index_start + i]

    n_valid_clim = 0
    for i in range(m):
        if np.isfinite(clim[i]):
            n_valid_clim += 1
    if n_valid_clim < n_classes:
        return np.nan

    p1 = 100.0 / n_classes
    p2 = 100.0 - 100.0 / n_classes
    t1, t2 = _nanpercentile2(clim, 100.0 / 3.0, 200.0 / 3.0)
    if np.isnan(t1) or np.isnan(t2):
        return np.nan

    yc, pc, cnt = _clean_prob(y_true, y_probs)
    if cnt < 2 * n_classes:
        return np.nan

    cls = np.empty(cnt, np.int64)
    for i in range(cnt):
        cls[i] = _digitize_right(yc[i], t1, t2)

    total = 0.0
    ybin = np.empty(cnt, np.int64)
    for c in range(n_classes):
        for i in range(cnt):
            ybin[i] = 1 if cls[i] == c else 0
        total += _auc_binary(ybin, pc[c, :])
    return total / n_classes


@njit(**_JIT)
def _groc_weighted_core(y_true, y_probs, index_start, index_end, n_classes):
    if index_end <= index_start:
        return np.nan
    m = index_end - index_start
    clim = np.empty(m, np.float64)
    for i in range(m):
        clim[i] = y_true[index_start + i]
    n_valid = 0
    for i in range(m):
        if np.isfinite(clim[i]):
            n_valid += 1
    if n_valid < n_classes:
        return np.nan

    t1, t2 = _nanpercentile2(clim, 100.0 / 3.0, 200.0 / 3.0)
    if np.isnan(t1) or np.isnan(t2):
        return np.nan

    yc, pc, cnt = _clean_prob(y_true, y_probs)
    if cnt < 2 * n_classes:
        return np.nan

    cls = np.empty(cnt, np.int64)
    for i in range(cnt):
        cls[i] = _digitize_right(yc[i], t1, t2)

    base = np.zeros(n_classes, np.float64)
    for i in range(cnt):
        base[cls[i]] += 1.0
    tot = 0.0
    for c in range(n_classes):
        base[c] /= cnt
        tot += base[c]
    if tot > 0.0:
        for c in range(n_classes):
            base[c] /= tot
    else:
        for c in range(n_classes):
            base[c] = 1.0 / n_classes

    out = 0.0
    ybin = np.empty(cnt, np.int64)
    for c in range(n_classes):
        for i in range(cnt):
            ybin[i] = 1 if cls[i] == c else 0
        out += base[c] * _auc_binary(ybin, pc[c, :])
    return out


@njit(**_JIT)
def _rpss_core(y_true, y_probs, index_start, index_end):
    n_classes = y_probs.shape[0]
    yc, pc, cnt = _clean_prob(y_true, y_probs)
    if cnt <= 2:
        return np.nan

    # Climatology is selected before filtering forecast/observation pairs.
    lo = index_start
    hi = index_end
    if lo < 0:
        lo = 0
    if hi > y_true.size:
        hi = y_true.size
    if hi <= lo:
        return np.nan
    t1, t2 = _nanpercentile2(y_true[lo:hi], 100.0 / 3.0, 200.0 / 3.0)
    if np.isnan(t1) or np.isnan(t2):
        return np.nan

    rps_f = 0.0
    rps_r = 0.0
    for i in range(cnt):
        c_obs = _digitize_right(yc[i], t1, t2)
        cum_f = 0.0
        cum_o = 0.0
        cum_r = 0.0
        for c in range(n_classes):
            cum_f += pc[c, i]
            cum_o += 1.0 if c == c_obs else 0.0
            cum_r += 1.0 / n_classes
            rps_f += (cum_f - cum_o) ** 2
            rps_r += (cum_r - cum_o) ** 2
    rps_f /= cnt
    rps_r /= cnt
    if rps_r == 0.0:
        return np.nan
    return 1.0 - rps_f / rps_r


@njit(**_JIT)
def _rpss_from_class_core(y_class, y_probs):
    """RPSS when the observations are already tercile labels (0/1/2)."""
    n_classes = y_probs.shape[0]
    yc, pc, cnt = _clean_prob(y_class, y_probs)
    if cnt == 0:
        return np.nan

    rps_f = 0.0
    rps_r = 0.0
    for i in range(cnt):
        c_obs = int(yc[i])
        cum_f = 0.0
        cum_o = 0.0
        cum_r = 0.0
        for c in range(n_classes):
            cum_f += pc[c, i]
            cum_o += 1.0 if c == c_obs else 0.0
            cum_r += 1.0 / n_classes
            rps_f += (cum_f - cum_o) ** 2
            rps_r += (cum_r - cum_o) ** 2
    rps_f /= cnt
    rps_r /= cnt
    if rps_r == 0.0:
        return np.nan
    return 1.0 - rps_f / rps_r


@njit(**_JIT)
def _igs_core(y_true, y_probs, index_start, index_end):
    yc, pc, cnt = _clean_prob(y_true, y_probs)
    if cnt <= 2:
        return np.nan

    lo = index_start
    hi = index_end
    if lo < 0:
        lo = 0
    if hi > y_true.size:
        hi = y_true.size
    if hi <= lo:
        return np.nan
    t1, t2 = _nanpercentile2(y_true[lo:hi], 100.0 / 3.0, 200.0 / 3.0)
    if np.isnan(t1) or np.isnan(t2):
        return np.nan

    acc = 0.0
    for i in range(cnt):
        c_obs = _digitize_right(yc[i], t1, t2)
        p = pc[c_obs, i]
        if p > 0.0:
            acc += -np.log2(p)
        else:
            acc += np.nan
    return acc / cnt


@njit(**_JIT)
def _brier_core(y_true, y_probs, index_start, index_end, k):
    yc, pc, cnt = _clean_prob(y_true, y_probs)
    if cnt < 3:
        return np.nan
    lo = index_start
    hi = index_end
    if lo < 0:
        lo = 0
    if hi > y_true.size:
        hi = y_true.size
    if hi <= lo:
        return np.nan
    t1, t2 = _nanpercentile2(y_true[lo:hi], 100.0 / 3.0, 200.0 / 3.0)
    if np.isnan(t1) or np.isnan(t2):
        return np.nan

    acc = 0.0
    for i in range(cnt):
        o = 1.0 if _digitize_right(yc[i], t1, t2) == k else 0.0
        d = pc[k, i] - o
        acc += d * d
    return acc / cnt


@njit(**_JIT)
def _bss_core(y_true, y_probs, index_start, index_end, k, use_empirical):
    bs = _brier_core(y_true, y_probs, index_start, index_end, k)
    if not np.isfinite(bs):
        return np.nan

    yc, pc, cnt = _clean_prob(y_true, y_probs)
    if cnt < 3:
        return np.nan
    lo = index_start
    hi = index_end
    if lo < 0:
        lo = 0
    if hi > y_true.size:
        hi = y_true.size
    if hi <= lo:
        return np.nan
    t1, t2 = _nanpercentile2(y_true[lo:hi], 100.0 / 3.0, 200.0 / 3.0)
    if np.isnan(t1) or np.isnan(t2):
        return np.nan

    o_all = np.empty(cnt, np.float64)
    for i in range(cnt):
        o_all[i] = 1.0 if _digitize_right(yc[i], t1, t2) == k else 0.0

    if use_empirical:
        s = 0.0
        n_clim = 0
        for i in range(lo, hi):
            if np.isfinite(y_true[i]):
                s += 1.0 if _digitize_right(y_true[i], t1, t2) == k else 0.0
                n_clim += 1
        if n_clim == 0:
            return np.nan
        p_ref = s / n_clim
    else:
        p_ref = 1.0 / 3.0

    bs_ref = 0.0
    for i in range(cnt):
        d = p_ref - o_all[i]
        bs_ref += d * d
    bs_ref /= cnt
    if bs_ref == 0.0:
        return np.nan
    return 1.0 - bs / bs_ref


@njit(**_JIT)
def _res_rel_core(y_true, y_probs, index_start, index_end, bins, want_resolution):
    """Resolution (``want_resolution=True``) or Reliability, Weijs (2010)."""
    nt = y_true.size
    if nt == 0 or y_probs.shape[1] == 0:
        return np.nan
    if index_end <= index_start:
        return np.nan

    m = index_end - index_start
    clim = np.empty(m, np.float64)
    for i in range(m):
        clim[i] = y_true[index_start + i]
    n_valid = 0
    for i in range(m):
        if np.isfinite(clim[i]):
            n_valid += 1
    # Resolution requires >= 5 valid climatology points, Reliability >= 10.
    if want_resolution:
        if n_valid < 5:
            return np.nan
    else:
        if n_valid < 10:
            return np.nan

    t1, t2 = _nanpercentile2(clim, 100.0 / 3.0, 200.0 / 3.0)
    if np.isnan(t1) or np.isnan(t2):
        return np.nan

    yc, pc, cnt = _clean_prob(y_true, y_probs)
    if cnt == 0:
        return np.nan

    n_classes = pc.shape[0]
    cls = np.empty(cnt, np.int64)
    for i in range(cnt):
        cls[i] = _digitize_right(yc[i], t1, t2)

    base = np.zeros(n_classes, np.float64)
    for i in range(cnt):
        base[cls[i]] += 1.0
    for c in range(n_classes):
        base[c] /= cnt

    nb = bins.size - 1
    total = 0.0
    for c in range(n_classes):
        for b in range(nb):
            n_kb = 0
            y_kb = 0.0
            p_kb = 0.0
            for i in range(cnt):
                # np.digitize(p, bins) - 1
                p = pc[c, i]
                idx = 0
                while idx < bins.size and p >= bins[idx]:
                    idx += 1
                idx -= 1
                if idx == b:
                    n_kb += 1
                    if cls[i] == c:
                        y_kb += 1.0
                    p_kb += p
            if n_kb > 0:
                y_kb /= n_kb
                p_kb /= n_kb
                if want_resolution:
                    if 0.0 < y_kb < 1.0 and 0.0 < base[c] < 1.0:
                        term1 = y_kb * np.log2(y_kb / base[c])
                        term2 = (1.0 - y_kb) * np.log2((1.0 - y_kb) / (1.0 - base[c]))
                        total += (n_kb / cnt) * (term1 + term2)
                else:
                    if 0.0 < y_kb < 1.0 and 0.0 < p_kb < 1.0:
                        term1 = y_kb * np.log2(y_kb / p_kb)
                        term2 = (1.0 - y_kb) * np.log2((1.0 - y_kb) / (1.0 - p_kb))
                        total += (n_kb / cnt) * (term1 + term2)
    return total


@njit(**_JIT)
def _tercile_prob_nonparam_core(best_guess, error_samples, first_tercile, second_tercile):
    """Non-parametric tercile probabilities from historical error samples."""
    n_time = best_guess.size
    ns = error_samples.size
    out = np.full((3, n_time), np.nan, np.float64)

    for tt in range(n_time):
        if np.isnan(best_guess[tt]):
            continue
        below = 0.0
        between = 0.0
        n_ok = 0
        for j in range(ns):
            v = best_guess[tt] + error_samples[j]
            if np.isfinite(v):
                n_ok += 1
                if v <= first_tercile:
                    below += 1.0
                elif v <= second_tercile:
                    between += 1.0
        if n_ok == 0:
            continue
        p_below = below / n_ok
        p_between = between / n_ok
        out[0, tt] = p_below
        out[1, tt] = p_between
        out[2, tt] = 1.0 - (p_below + p_between)
    return out


def warmup(n_time=30, verbose=True):
    """Compile every kernel once.

    Call before a long verification loop so the one-off 1-3 s per-kernel
    compilation is not charged to the first model in the loop.  With
    ``cache=True`` this is a no-op from the second run of the process onward.
    """
    import time
    if not _use_numba():
        if verbose:
            print("[WAS_Verification] numba disabled - nothing to warm up")
        return
    rng = np.random.default_rng(0)
    y = rng.normal(size=n_time)
    f = y * 0.6 + rng.normal(size=n_time) * 0.5
    p = rng.dirichlet(np.ones(3), size=n_time).T.copy()
    bins = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    t0 = time.time()
    for fn in (_pearson_core, _spearman_core, _kge_core, _ioa_core,
               _mae_core, _rmse_core, _nse_core):
        fn(y, f)
    _groc_core(y, p, 0, n_time, 3)
    _groc_weighted_core(y, p, 0, n_time, 3)
    _rpss_core(y, p, 0, n_time)
    _rpss_from_class_core(np.zeros(n_time), p)
    _igs_core(y, p, 0, n_time)
    _brier_core(y, p, 0, n_time, 2)
    _bss_core(y, p, 0, n_time, 2, False)
    _res_rel_core(y, p, 0, n_time, bins, True)
    _res_rel_core(y, p, 0, n_time, bins, False)
    _tercile_prob_nonparam_core(f, y - f, -0.4, 0.4)
    if verbose:
        print(f"[WAS_Verification] numba warmup done in {time.time() - t0:.1f}s")


_DEFAULT_BINS = np.array([0.000, 0.025, 0.050, 0.100, 0.150, 0.200,
                          0.250, 0.300, 0.350, 0.400, 0.450, 0.500,
                          0.550, 0.600, 0.650, 0.6700, 0.6750, 0.800,
                          0.850, 0.900, 0.950, 0.975, 1.000])


def _f8(a):
    """Contiguous float64 view - keeps numba from recompiling per memory layout."""
    return np.ascontiguousarray(a, dtype=np.float64)


def _score_at_declared_events(y, p, lower, upper, reference, *, name, options):
    """One-cell score using the producer's exact event boundaries.

    Boundaries may be constant climate terciles or explicitly supplied research
    fold thresholds. Never estimate new quantiles from the scored sample.
    """
    from .was_groc import ordinal_groc
    y, p = np.asarray(y, float), np.asarray(p, float)
    lower, upper, reference = (np.asarray(a, float) for a in (lower, upper, reference))
    valid = (np.isfinite(y) & np.isfinite(lower) & np.isfinite(upper)
             & np.isfinite(p).all(axis=0) & np.isfinite(reference).all(axis=0))
    if valid.sum() < 3:
        return np.nan
    y, p, lower, upper, reference = (y[valid], p[:, valid], lower[valid],
                                    upper[valid], reference[:, valid])
    p = p / p.sum(axis=0)
    reference = reference / reference.sum(axis=0)
    cls = np.where(y <= lower, 0, np.where(y <= upper, 1, 2))
    onehot = np.eye(3)[cls].T
    if name == 'calculate_rpss':
        cf, co, cr = (a.cumsum(axis=0) for a in (p, onehot, reference))
        rf, rr = np.mean(np.sum((cf-co)**2, axis=0)), np.mean(np.sum((cr-co)**2, axis=0))
        return 1-rf/rr if rr > 0 else np.nan
    if name == 'ignorance_score':
        with np.errstate(divide='ignore'):
            return float(np.mean(-np.log2(p[cls, np.arange(len(cls))])))
    if name == 'calculate_groc':
        return ordinal_groc(cls, p)
    if name in {'calculate_macro_auc', 'calculate_groc_weighted'}:
        scores = []
        for k in range(3):
            yy = (cls == k).astype(int)
            if yy.min() == yy.max():
                scores.append(.5)
            else:
                fpr, tpr, _ = roc_curve(yy, p[k])
                scores.append(float(auc(fpr, tpr)))
        weights = onehot.mean(axis=1) if name == 'calculate_groc_weighted' else np.full(3, 1/3)
        return float(np.dot(weights, scores))
    if name in {'brier_score', 'brier_skill_score'}:
        k = WAS_Verification._event_index(options.get('event', 'PA'))
        bs = np.mean((p[k]-onehot[k])**2)
        if name == 'brier_score':
            return float(bs)
        ref_kind = str(options.get('reference', 'empirical')).lower()
        if ref_kind in {'uniform', 'unif', '1/3', '13'}:
            base = np.full(len(cls), 1/3)
        elif ref_kind in {'empirical', 'base', 'clim', 'climatology'}:
            base = reference[k]
        else:
            raise ValueError("reference must be uniform or empirical")
        denom = np.mean((base-onehot[k])**2)
        return float(1-bs/denom) if denom > 0 else np.nan
    if name in {'resolution_score_grid', 'reliability_score_grid'}:
        bins = np.asarray(options.get('bins', _DEFAULT_BINS), float)
        total = 0.
        for k in range(3):
            base_rate = onehot[k].mean()
            bucket = np.digitize(p[k], bins)-1
            for b in range(len(bins)-1):
                chosen = bucket == b
                if not chosen.any():
                    continue
                observed = onehot[k, chosen].mean()
                baseline = base_rate if name == 'resolution_score_grid' else p[k, chosen].mean()
                # Preserve the historic Weijs score convention at degenerate bins.
                if 0 < observed < 1 and 0 < baseline < 1:
                    term = observed*np.log2(observed/baseline)+(1-observed)*np.log2((1-observed)/(1-baseline))
                    total += chosen.mean()*term
        return float(total)
    raise ValueError(f"Explicit tercile thresholds are unsupported for score {name!r}")


class WAS_Verification:
    """
    Verification class for evaluating weather and climate forecasts.

    Provides methods to compute deterministic, probabilistic, and ensemble-based metrics,
    as well as visualization tools for model performance assessment.

    Parameters
    ----------
    dist_method : str, optional
        Distribution method for tercile probability calculations
        ('bestfit', 'nonparam'). Default is 'nonparam'.
    mae_rmse_crps_thrsd : float, optional
        Upper bound of the colour scale for MAE / RMSE / CRPS maps.  Use a value
        commensurate with the predictand: ~50 for onset/cessation dates in days,
        ~10 for dry-spell lengths, ~100 for seasonal rainfall in mm.
    """

    def __init__(self, dist_method="nonparam", mae_rmse_crps_thrsd=100):
        self.dist_method = dist_method
        self.mae_rmse_crps_thrsd = float(mae_rmse_crps_thrsd)

        self.scores = {
            "KGE": ("Kling Gupta Efficiency", -1, 1, "det_score", "RdBu_r", self.kling_gupta_efficiency, ""),
            "Pearson": ("Pearson Correlation", -1, 1, "det_score", "RdBu_r", self.pearson_corr, ""),
            "Spearman": ("Spearman Correlation", -1, 1, "det_score", "RdBu_r", self.spearman_corr, ""),
            "IOA": ("Index Of Agreement", 0, 1, "det_score", "RdBu_r", self.index_of_agreement, ""),
            "MAE": ("Mean Absolute Error", 0, self.mae_rmse_crps_thrsd, "det_score", "viridis", self.mean_absolute_error, "[mm]"),
            "RMSE": ("Root Mean Square Error", 0, self.mae_rmse_crps_thrsd, "det_score", "viridis", self.root_mean_square_error, "[mm]"),
            "NSE": ("Nash Sutcliffe Efficiency", None, 1, "det_score", "RdBu_r", self.nash_sutcliffe_efficiency, ""),
            "TAYLOR_DIAGRAM": ("Taylor Diagram", None, None, "all_grid_det_score", None, self.taylor_diagram),
            "GROC": ("Generalized ordinal discrimination (2AFC)", 0, 1, "prob_score", "RdBu_r", self.calculate_groc, ""),
            "ROC_AUC_macro": ("Macro ROC AUC", 0, 1, "prob_score", "RdBu_r", self.calculate_macro_auc, ""),
            "RPSS": ("Ranked Probability Skill Score", -1, 1, "prob_score", "RdBu_r", self.calculate_rpss, ""),
            "IGS": ("Ignorance Score", 0, None, "prob_score", "RdBu", self.ignorance_score, ""),
            "RES": ("Resolution", 0, None, "prob_score", "RdBu", self.resolution_score_grid, ""),
            "REL": ("Reliability", None, None, "prob_score", None, self.reliability_score_grid, ""),
            "RELIABILITY_DIAGRAM": ("Reliability Diagram", None, None, "all_grid_prob_score", None, self.reliability_diagram, ""),
            "BS": ("Brier Score", 0, 1, "prob_score", "viridis", self.brier_score, ""),
            "BSS": ("Brier Skill Score vs climatology", -1, 1, "prob_score", "RdBu_r", self.brier_skill_score, ""),
            "ROC_CURVE": ("ROC CURVE", None, None, "all_grid_prob_score", None, self.plot_roc_curves, ""),
            "CRPS": ("Continuous Ranked Probability Score with the ensemble distribution", 0, self.mae_rmse_crps_thrsd, "ensemble_score", "RdBu", self.compute_crps, "[mm]")
        }

    def get_scores_metadata(self):
        """
        Retrieve metadata for all available scoring metrics.

        Returns
        -------
        dict
            Dictionary containing score names as keys and tuples of metadata
            (description, min_value, max_value, score_type, colormap, function, unit) as values.
        """
        return self.scores

    # ------------------------
    # Deterministic Metrics
    # ------------------------

    def kling_gupta_efficiency(self, y_true, y_pred):
        """
        Compute Kling-Gupta Efficiency (KGE) metric.

        KGE combines correlation, bias, and variability ratios to assess model performance.

        Returns
        -------
        float
            KGE score, from -Inf to 1 (1 is perfect). np.nan if fewer than 3 valid pairs.
        """
        if _use_numba():
            return _kge_core(_f8(y_true), _f8(y_pred))
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if np.any(mask) and np.sum(mask) > 2:
            a, b = y_true[mask], y_pred[mask]
            r = np.corrcoef(a, b)[0, 1]
            alpha = np.std(b) / np.std(a)
            beta = np.mean(b) / np.mean(a)
            return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        return np.nan

    def pearson_corr(self, y_true, y_pred):
        """
        Compute Pearson Correlation Coefficient.

        Returns
        -------
        float
            Pearson r in [-1, 1]. np.nan if fewer than 3 valid pairs.
        """
        if _use_numba():
            return _pearson_core(_f8(y_true), _f8(y_pred))
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if np.any(mask) and np.sum(mask) > 2:
            return pearsonr(y_true[mask], y_pred[mask])[0]
        return np.nan

    def spearman_corr(self, y_true, y_pred):
        """
        Compute Spearman rank correlation coefficient.

        Returns
        -------
        float
            Spearman's rho in [-1, 1]. np.nan if fewer than 3 valid pairs.
        """
        if _use_numba():
            return _spearman_core(_f8(y_true), _f8(y_pred))
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if np.any(mask) and np.sum(mask) > 2:
            return stats.spearmanr(y_true[mask], y_pred[mask]).correlation
        return np.nan

    def index_of_agreement(self, y_true, y_pred):
        """
        Compute Index of Agreement (IOA).

        Returns
        -------
        float
            IOA in [0, 1] (1 is perfect). np.nan if fewer than 3 valid pairs.
        """
        if _use_numba():
            return _ioa_core(_f8(y_true), _f8(y_pred))
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if np.any(mask) and np.sum(mask) > 2:
            a, b = y_true[mask], y_pred[mask]
            numerator = np.sum((b - a) ** 2)
            denominator = np.sum((np.abs(b - np.mean(a)) + np.abs(a - np.mean(a))) ** 2)
            return 1 - (numerator / denominator)
        return np.nan

    def mean_absolute_error(self, y_true, y_pred):
        """
        Compute Mean Absolute Error (MAE).

        Returns
        -------
        float
            MAE, non-negative. np.nan if fewer than 3 valid pairs.
        """
        if _use_numba():
            return _mae_core(_f8(y_true), _f8(y_pred))
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if np.any(mask) and np.sum(mask) > 2:
            return np.mean(np.abs(y_true[mask] - y_pred[mask]))
        return np.nan

    def root_mean_square_error(self, y_true, y_pred):
        """
        Compute Root Mean Square Error (RMSE).

        Returns
        -------
        float
            RMSE, non-negative. np.nan if fewer than 3 valid pairs.
        """
        if _use_numba():
            return _rmse_core(_f8(y_true), _f8(y_pred))
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if np.any(mask) and np.sum(mask) > 2:
            return np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))
        return np.nan

    def nash_sutcliffe_efficiency(self, y_true, y_pred):
        """
        Compute Nash-Sutcliffe Efficiency (NSE).

        Returns
        -------
        float
            NSE, from -Inf to 1 (1 is perfect). -inf on a constant observed
            series; np.nan if fewer than 3 valid pairs.
        """
        if _use_numba():
            return _nse_core(_f8(y_true), _f8(y_pred))
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if np.any(mask) and np.sum(mask) > 2:
            a, b = y_true[mask], y_pred[mask]
            return 1 - np.sum((a - b) ** 2) / np.sum((a - np.mean(a)) ** 2)
        return np.nan

    def taylor_diagram(
        self,
        obs,
        models,
        *,
        normalize: bool = True,
        correlation: str = "pearson",
        labels: dict | None = None,
        title: str | None = "Taylor diagram",
        figsize=(7, 6),
        savepath: str | None = None,
        hide_rms_numbers: bool = True,
    ):
        """
        Plot a Taylor diagram comparing one or more model series against observations.

        Parameters
        ----------
        obs : xarray.DataArray or np.ndarray
            Observations. If xarray, can be (T), (T,Y,X), etc.
        models : dict[str, xarray.DataArray | np.ndarray]
            Mapping of model name -> forecast array (same shape as obs or broadcastable).
        normalize : bool, optional
            If True (default), normalize std and CRMSD by std(obs).
        correlation : {"pearson","spearman"}, optional
            Correlation metric used on paired, finite values.
        labels : dict[str,str], optional
            Custom display labels for model markers. Defaults to model keys.
        title : str or None, optional
            Figure title.
        figsize : tuple, optional
            Matplotlib figure size.
        savepath : str or None, optional
            If provided, save figure to this path.

        Returns
        -------
        (fig, ax) : matplotlib Figure and Axes
        """
        try:
            import skill_metrics as sm
        except Exception as e:
            raise ImportError("SkillMetrics is required. Install with: pip install SkillMetrics") from e

        def _to_aligned_1d(o, f):
            if hasattr(o, "dims") or hasattr(f, "dims"):
                o_al, f_al = xr.align(o, f, join="inner")
                o_vals = np.asarray(o_al).ravel()
                f_vals = np.asarray(f_al).ravel()
            else:
                o_vals = np.asarray(o).ravel()
                f_vals = np.asarray(f).ravel()
            m = np.isfinite(o_vals) & np.isfinite(f_vals)
            return o_vals[m], f_vals[m]

        model_names = list(models.keys())
        if labels is None:
            labels = {k: k for k in model_names}

        sdev_list, crmsd_list, corr_list, used_names = [], [], [], []
        std_obs_plot = None

        for name in model_names:
            o, f = _to_aligned_1d(obs, models[name])
            if o.size < 3:
                continue
            r = stats.spearmanr(o, f).correlation if correlation.lower() == "spearman" else np.corrcoef(o, f)[0, 1]
            std_o = np.std(o, ddof=0)
            std_f = np.std(f, ddof=0)
            cr = np.sqrt(std_o ** 2 + std_f ** 2 - 2.0 * std_o * std_f * r)
            if normalize and std_o > 0:
                std_o_plot, std_f_plot, cr_plot = 1.0, std_f / std_o, cr / std_o
            else:
                std_o_plot, std_f_plot, cr_plot = std_o, std_f, cr
            if std_obs_plot is None:
                std_obs_plot = std_o_plot
            sdev_list.append(std_f_plot)
            crmsd_list.append(cr_plot)
            corr_list.append(r)
            used_names.append(name)

        if not used_names:
            raise ValueError("No model had >=3 paired finite values w.r.t. observations.")

        sdev = np.concatenate(([std_obs_plot], np.asarray(sdev_list)))
        crmsd = np.concatenate(([0.0], np.asarray(crmsd_list)))
        ccoef = np.concatenate(([1.0], np.asarray(corr_list)))
        marker_labels = ["OBS"] + [labels[n] for n in used_names]
        axis_max = 1.25 * max(np.nanmax(sdev), np.nanmax(crmsd), 1.0)

        fig, ax = plt.subplots(figsize=figsize)
        _ = sm.taylor_diagram(
            sdev, crmsd, ccoef,
            numberPanels=1,
            axisMax=axis_max,
            markerDisplayed="marker",
            markerLabel=marker_labels,
            markerLegend="on",
            styleOBS="-",
            colOBS="k",
            markerObs="*",
            titleOBS="obs",
            labelRMS="CRMSD" if not normalize else "CRMSD (norm.)",
            titleSTD="on",
            titleCOR="on",
            colCOR="tab:blue",
            colSTD="black",
            colRMS="tab:green",
        )

        if hide_rms_numbers:
            green = mcolors.to_rgba("tab:green")
            for txt in fig.findobj(mpl.text.Text):
                s = (txt.get_text() or "").strip()
                if not s:
                    continue
                try:
                    if np.allclose(mcolors.to_rgba(txt.get_color()), green, atol=1e-3):
                        txt.remove()
                except Exception:
                    pass

        if title:
            ax.set_title(title)
        if savepath:
            fig.savefig(savepath, dpi=300, bbox_inches="tight")
        return fig, ax

    def compute_deterministic_score(self, score_func, obs, pred):
        """
        Apply a deterministic scoring function over xarray DataArrays.

        Computes the specified metric across the time dimension for each grid point.

        Parameters
        ----------
        score_func : callable
            Scoring function to apply (e.g., pearson_corr, kling_gupta_efficiency).
        obs : xarray.DataArray
            Observed data with dimensions (T, Y, X).
        pred : xarray.DataArray
            Predicted data with dimensions (T, Y, X).

        Returns
        -------
        xarray.DataArray
            Score values with dimensions (Y, X).
        """
        obs, pred = xr.align(obs, pred)

        def _eager():
            return xr.apply_ufunc(
                score_func, obs, pred,
                input_core_dims=[('T',), ('T',)],
                vectorize=True, dask='parallelized',
                output_core_dims=[()], output_dtypes=['float'],
                dask_gufunc_kwargs={"allow_rechunk": True},
            )

        # Grid-parallel path: chunk over Y/X and run the per-cell score on a
        # thread pool. Identical results to the eager path (each cell is scored
        # independently); threads avoid pickling `self`, and the compiled
        # kernels release the GIL. Falls back to eager on failure or small grids.
        if os.environ.get("WAS_VERIF_PARALLEL", "1") != "0":
            n = _verif_num_workers()
            ncells = int(obs.sizes.get("Y", 1)) * int(obs.sizes.get("X", 1))
            min_cells = int(os.environ.get("WAS_VERIF_MIN_CELLS", "4000"))
            if n > 1 and ncells >= min_cells:
                try:
                    res = xr.apply_ufunc(
                        score_func, _chunk_grid(obs, n), _chunk_grid(pred, n),
                        input_core_dims=[('T',), ('T',)],
                        vectorize=True, dask='parallelized',
                        output_core_dims=[()], output_dtypes=['float'],
                        dask_gufunc_kwargs={"allow_rechunk": True},
                    )
                    return res.compute(scheduler="threads", num_workers=n)
                except Exception:
                    pass
        return _eager()

    # ------------------------
    # Probabilistic Metrics
    # ------------------------

    def classify(self, y, index_start, index_end):
        """
        Classify data into terciles based on climatological thresholds.

        Returns
        -------
        tuple
            (y_class, tercile_33, tercile_67) where y_class is 0 (below-normal),
            1 (near-normal) or 2 (above-normal).
        """
        mask = np.isfinite(y)
        if np.any(mask):
            terciles = np.nanpercentile(y[index_start:index_end], [100.0 / 3.0, 200.0 / 3.0])
            if not np.isfinite(terciles).all():
                return np.full(y.shape[0], np.nan), np.nan, np.nan
            y_class = np.where(mask, np.digitize(y, bins=terciles, right=True), np.nan)
            return y_class, terciles[0], terciles[1]
        return np.full(y.shape[0], np.nan), np.nan, np.nan

    def compute_class(self, Predictant, clim_year_start, clim_year_end):
        """
        Compute tercile class labels for observed data.

        Returns
        -------
        xarray.DataArray
            Classified data (T, Y, X): 0 below-normal, 1 near-normal, 2 above-normal.
        """
        # .get_loc(str(year)) is a pandas partial-string lookup that only works on
        # an exact DatetimeIndex flavour; anything else (cftime, an object-dtype
        # axis produced upstream) raises KeyError(str(year)) and the caller sees
        # only that bare year number, with no indication of what actually failed.
        # .dt.year works on any datetime-like coordinate xarray recognises, which
        # is the same requirement the rest of this pipeline already relies on.
        years = np.asarray(Predictant["T"].dt.year.values, dtype=int)
        index_start = int(np.searchsorted(years, int(clim_year_start), side="left"))
        index_end = int(np.searchsorted(years, int(clim_year_end), side="right"))
        Predictant_class, tercile_33, tercile_67 = xr.apply_ufunc(
            self.classify,
            Predictant,
            input_core_dims=[('T',)],
            kwargs={'index_start': index_start, 'index_end': index_end},
            vectorize=True,
            dask='parallelized',
            output_core_dims=[('T',), (), ()],
            output_dtypes=['float', 'float', 'float']
        )
        return Predictant_class.transpose('T', 'Y', 'X')

    def classify_data_into_terciles(self, y, T1, T2):
        """
        Classify data into terciles based on given thresholds.

        Returns
        -------
        array-like
            0 below-normal, 1 near-normal, 2 above-normal.
        """
        y = np.asarray(y, dtype=float)
        if not np.isfinite([T1,T2]).all() or T1 > T2:
            return np.full(y.shape, np.nan)
        return np.where(np.isfinite(y), np.where(y <= T1,0.,np.where(y <= T2,1.,2.)), np.nan)

    def calculate_groc(self, y_true, y_probs, index_start, index_end, n_classes=3):
        """Generalized ordinal discrimination / 2AFC, Mason & Weigel (2009).

        Eq. 16 compares pairs with different observed categories. Tied forecast
        ordering contributes one half; pairs in the same observed category are
        excluded. Terciles are estimated on the unmasked reference time slice.
        For validation prefer score_fold_probabilities, which takes the exact
        declared thresholds shared by probability generation and verification.
        """
        from .was_groc import ordinal_groc
        y = np.asarray(y_true,dtype=float)
        p = np.asarray(y_probs,dtype=float)
        if n_classes != 3 or p.shape != (3,len(y)):
            raise ValueError('Tercile GROC requires probabilities shaped (3,n_time)')
        if not 0 <= index_start < index_end <= len(y):
            raise ValueError('Invalid climatology bounds')
        clim = y[index_start:index_end]
        if np.isfinite(clim).sum() < 3:
            return np.nan
        thresholds = np.nanpercentile(clim,[100.0/3.0,200.0/3.0])
        classes = np.where(np.isfinite(y),np.digitize(y,thresholds,right=True),np.nan)
        return ordinal_groc(classes,p)

    def calculate_macro_auc(self, y_true, y_probs, index_start, index_end, n_classes=3):
        """
        Macro one-versus-rest ROC AUC (the former GROC implementation).

        This is distinct from generalized ordinal discrimination; the former
        values remain reproducible through this explicitly named method.
        Averages the AUC of each tercile category.  The compiled kernel obtains
        each AUC from the Mann-Whitney rank-sum statistic with mid-ranks for
        ties, which is mathematically identical to the trapezoidal area under
        the ROC curve computed by ``roc_curve`` + ``auc``, without building the
        curve.

        Parameters
        ----------
        y_true : array-like
            Observed values.
        y_probs : array-like
            Forecast probabilities, shape (n_classes, n_samples).
        index_start, index_end : int
            Bounds of the climatology period used for the tercile thresholds.
        n_classes : int, optional
            Number of classes (3 for terciles).

        Returns
        -------
        float
            GROC in [0, 1]; np.nan if insufficient valid data.

        Reference
        ---------
        Weigel, A. P. and Mason, S. J. (2011). The generalized discrimination
        score for ensemble forecasts. Monthly Weather Review, 139(9):3069-3074.
        """
        if n_classes != 3 or np.asarray(y_probs).shape != (3,len(y_true)):
            raise ValueError('Tercile ROC requires probabilities shaped (3, n_time)')
        if not 0 <= index_start < index_end <= len(y_true):
            raise ValueError('Invalid climatology bounds')
        if _use_numba():
            return _groc_core(_f8(y_true), _f8(y_probs),
                              int(index_start), int(index_end), int(n_classes))

        if index_end <= index_start:
            return np.nan
        clim_period = y_true[index_start:index_end]
        valid_clim_data = clim_period[np.isfinite(clim_period)]
        if len(valid_clim_data) < n_classes:
            return np.nan
        percentiles = np.array([100.0 / 3.0, 200.0 / 3.0])
        thresholds = np.nanpercentile(valid_clim_data, percentiles)

        mask = np.isfinite(y_true) & np.isfinite(y_probs).all(axis=0)
        if np.sum(mask) < 2 * n_classes:
            return np.nan
        y_true_clean = y_true[mask]
        y_probs_clean = y_probs[:, mask]
        y_true_class = np.digitize(y_true_clean, bins=thresholds, right=True)

        auc_scores = []
        for i in range(n_classes):
            y_true_binary = (y_true_class == i).astype(int)
            if np.sum(y_true_binary == 1) == 0 or np.sum(y_true_binary == 0) == 0:
                auc_scores.append(0.5)
                continue
            fpr, tpr, _ = roc_curve(y_true_binary, y_probs_clean[i, :])
            auc_scores.append(0.5 if (len(fpr) < 2 or len(tpr) < 2) else auc(fpr, tpr))
        return np.mean(auc_scores)

    def calculate_groc_weighted(self, y_true, y_probs, index_start, index_end, n_classes=3):
        """
        One-versus-rest ROC AUC weighted by observed category frequency.

        Historical compatibility name; not generalized ordinal discrimination.

        Returns
        -------
        float
            Weighted GROC in [0, 1].
        """
        if n_classes != 3 or np.asarray(y_probs).shape != (3,len(y_true)):
            raise ValueError('Tercile ROC requires probabilities shaped (3, n_time)')
        if not 0 <= index_start < index_end <= len(y_true):
            raise ValueError('Invalid climatology bounds')
        if _use_numba():
            return _groc_weighted_core(_f8(y_true), _f8(y_probs),
                                       int(index_start), int(index_end), int(n_classes))

        clim_period = y_true[index_start:index_end]
        valid_clim_data = clim_period[np.isfinite(clim_period)]
        if len(valid_clim_data) < n_classes:
            return np.nan
        percentiles = np.array([100.0 / 3.0, 200.0 / 3.0])
        thresholds = np.nanpercentile(valid_clim_data, percentiles)

        mask = np.isfinite(y_true) & np.isfinite(y_probs).all(axis=0)
        if np.sum(mask) < 2 * n_classes:
            return np.nan
        y_true_clean = y_true[mask]
        y_probs_clean = y_probs[:, mask]
        y_true_class = np.digitize(y_true_clean, bins=thresholds, right=True)

        base_rates = np.array([np.mean(y_true_class == i) for i in range(n_classes)])
        base_rates = (base_rates / np.sum(base_rates) if np.sum(base_rates) > 0
                      else np.ones(n_classes) / n_classes)

        weighted_sum = 0.0
        for i in range(n_classes):
            y_true_binary = (y_true_class == i).astype(int)
            if np.sum(y_true_binary == 1) == 0 or np.sum(y_true_binary == 0) == 0:
                weighted_sum += base_rates[i] * 0.5
            else:
                fpr, tpr, _ = roc_curve(y_true_binary, y_probs_clean[i, :])
                roc_auc = 0.5 if (len(fpr) < 2 or len(tpr) < 2) else auc(fpr, tpr)
                weighted_sum += base_rates[i] * roc_auc
        return weighted_sum

    def calculate_rpss(self, y_true, y_probs, index_start, index_end):
        """
        Compute Ranked Probability Skill Score (RPSS).

        Compares the RPS of the forecast to a uniform (1/3, 1/3, 1/3)
        climatological reference.

        Returns
        -------
        float
            RPSS, from -Inf to 1; np.nan if insufficient valid data.
        """
        if _use_numba():
            return _rpss_core(_f8(y_true), _f8(y_probs), int(index_start), int(index_end))

        encoder = OneHotEncoder(categories=[np.array([0, 1, 2])], sparse_output=False)
        mask = np.isfinite(y_true) & np.isfinite(y_probs).all(axis=0)
        if np.any(mask) and np.sum(mask) > 2:
            y_true_clean = y_true[mask]
            y_probs_clean = y_probs[:, mask]
            terciles = np.nanpercentile(y_true[index_start:index_end], [100.0 / 3.0, 200.0 / 3.0])
            if not np.isfinite(terciles).all():
                return np.nan
            y_true_clean_class = np.digitize(y_true_clean, bins=terciles, right=True)
            one_hot = encoder.fit_transform(y_true_clean_class.reshape(-1, 1))
            cumulative_forecast = np.cumsum(np.swapaxes(y_probs_clean, 0, 1), axis=1)
            climatology = np.full_like(one_hot, 1 / 3)
            cumulative_reference = np.cumsum(climatology, axis=1)
            cumulative_outcome = np.cumsum(one_hot, axis=1)
            rps_forecast = np.mean(np.sum((cumulative_forecast - cumulative_outcome) ** 2, axis=1))
            rps_reference = np.mean(np.sum((cumulative_reference - cumulative_outcome) ** 2, axis=1))
            return 1 - (rps_forecast / rps_reference)
        return np.nan

    def calculate_rpss_(self, y_true, y_probs):
        """
        RPSS for observations that are ALREADY tercile labels (0, 1, 2).

        Used by ``compute_one_year_rpss``; no thresholds are recomputed.

        Returns
        -------
        float
            RPSS, from -Inf to 1.
        """
        if _use_numba():
            return _rpss_from_class_core(_f8(y_true), _f8(y_probs))

        encoder = OneHotEncoder(categories=[[0, 1, 2]], sparse_output=False)
        mask = np.isfinite(y_true) & np.isfinite(y_probs).all(axis=0)
        if np.any(mask):
            y_true_clean = y_true[mask]
            y_probs_clean = y_probs[:, mask]
            one_hot = encoder.fit_transform(y_true_clean.reshape(-1, 1))
            cumulative_forecast = np.cumsum(np.swapaxes(y_probs_clean, 0, 1), axis=1)
            cumulative_outcome = np.cumsum(one_hot, axis=1)
            cumulative_climatology = np.cumsum(np.array([1 / 3, 1 / 3, 1 / 3]))
            rps_forecast = np.mean(np.sum((cumulative_forecast - cumulative_outcome) ** 2, axis=1))
            rps_reference = np.mean(np.sum((cumulative_climatology - cumulative_outcome) ** 2, axis=1))
            return 1 - (rps_forecast / rps_reference)
        return np.nan

    def ignorance_score(self, y_true, y_probs, index_start, index_end):
        """
        Compute Ignorance Score, Weijs (2010).

        Logarithmic loss of the forecast probability assigned to the observed
        category.

        Returns
        -------
        float
            Ignorance score, non-negative; np.nan if insufficient valid data or
            if any observed category received zero probability.
        """
        if _use_numba():
            return _igs_core(_f8(y_true), _f8(y_probs), int(index_start), int(index_end))

        mask = np.isfinite(y_true) & np.isfinite(y_probs).all(axis=0)
        if np.any(mask) and np.sum(mask) > 2:
            y_true_clean = y_true[mask]
            y_probs_clean = y_probs[:, mask]
            terciles = np.nanpercentile(y_true[index_start:index_end], [100.0 / 3.0, 200.0 / 3.0])
            if not np.isfinite(terciles).all():
                return np.nan
            y_true_clean_class = np.digitize(y_true_clean, bins=terciles, right=True)
            n = y_true_clean.shape[0]
            ignorance_sum = 0.0
            for i in range(n):
                prob = y_probs_clean[int(y_true_clean_class[i]), i]
                ignorance_sum += -np.log2(prob) if prob > 0 else np.nan
            return ignorance_sum / n if n > 0 else np.nan
        return np.nan

    @staticmethod
    def _event_index(event):
        """Map 'PB'/'PN'/'PA' (or 0/1/2) to a category index; defaults to PA."""
        return {"PB": 0, "PN": 1, "PA": 2, 0: 0, 1: 1, 2: 2}.get(event, 2)

    def brier_score(self, y_true, y_probs, index_start, index_end, event="PA"):
        """
        Brier Score for a single tercile event (PB / PN / PA).

        Parameters
        ----------
        y_true : 1D array-like
            Observed values for one grid point.
        y_probs : 2D array-like, shape (3, N)
            Forecast probabilities for PB / PN / PA.
        index_start, index_end : int
            Bounds of the calibration slice inside the original observation series.
        event : {"PB","PN","PA",0,1,2}, optional
            Event to score. Default "PA" (above-normal).

        Returns
        -------
        float
            BS in [0, 1]; np.nan if fewer than 3 valid pairs.
        """
        k = self._event_index(event)
        if _use_numba():
            return _brier_core(_f8(y_true), _f8(y_probs),
                               int(index_start), int(index_end), int(k))

        mask = np.isfinite(y_true) & np.isfinite(y_probs).all(axis=0)
        if np.sum(mask) < 3:
            return np.nan
        y = y_true[mask]
        p = y_probs[:, mask]
        T1, T2 = np.nanpercentile(y_true[index_start:index_end], [100.0 / 3.0, 200.0 / 3.0])
        if not np.isfinite([T1, T2]).all():
            return np.nan
        y_class = np.digitize(y, bins=[T1, T2], right=True)
        o = (y_class == k).astype(float)
        return float(np.mean((p[k, :] - o) ** 2))

    def brier_skill_score(self, y_true, y_probs, index_start, index_end,
                          event="PA", reference="uniform"):
        """
        Event-wise Brier Skill Score (BSS) for tercile forecasts.

        Parameters
        ----------
        y_true : 1D array-like
            Observed values for one grid point.
        y_probs : 2D array-like, shape (3, N)
            Forecast probabilities for PB / PN / PA.
        index_start, index_end : int
            Bounds of the calibration slice inside the original observation series.
        event : {"PB","PN","PA",0,1,2}, optional
            Event to score. Default "PA".
        reference : {"uniform","empirical"}, optional
            "uniform": climatology is 1/3.  "empirical": base rate from the
            calibration slice.

        Returns
        -------
        float
            BSS = 1 - BS / BS_ref; np.nan if not computable.

        Notes
        -----
        The previous version called a non-existent ``self.brier_score_event``
        and raised ``AttributeError`` on every call; it now delegates to
        :meth:`brier_score`.
        """
        k = self._event_index(event)
        ref = str(reference).lower()
        if ref in ("uniform", "unif", "1/3", "13"):
            use_empirical = False
        elif ref in ("empirical", "base", "clim", "climatology"):
            use_empirical = True
        else:
            raise ValueError("reference must be 'uniform' or 'empirical'")

        if _use_numba():
            return _bss_core(_f8(y_true), _f8(y_probs),
                             int(index_start), int(index_end), int(k), use_empirical)

        bs = self.brier_score(y_true, y_probs, index_start, index_end, event=event)
        if not np.isfinite(bs):
            return np.nan
        mask = np.isfinite(y_true) & np.isfinite(y_probs).all(axis=0)
        if np.sum(mask) < 3:
            return np.nan
        y = y_true[mask]
        T1, T2 = np.nanpercentile(y_true[index_start:index_end], [100.0 / 3.0, 200.0 / 3.0])
        if not np.isfinite([T1, T2]).all():
            return np.nan
        y_class_all = np.digitize(y, bins=[T1, T2], right=True)
        o_all = (y_class_all == k).astype(float)
        clim = np.asarray(y_true[index_start:index_end], dtype=float)
        clim = clim[np.isfinite(clim)]
        p_ref = float(np.mean(np.digitize(clim, [T1,T2], right=True) == k)) if use_empirical else 1.0 / 3.0
        bs_ref = float(np.mean((p_ref - o_all) ** 2))
        return np.nan if bs_ref == 0 else float(1.0 - bs / bs_ref)

    def resolution_score_grid(self, y_true, y_probs, index_start, index_end,
                              bins=_DEFAULT_BINS):
        """
        Resolution score on a grid point, Weijs (2010).

        Measures how far the conditional observed frequencies depart from the
        climatological base rate.

        Returns
        -------
        float
            Resolution, non-negative; np.nan if insufficient valid data.
        """
        if _use_numba():
            return _res_rel_core(_f8(y_true), _f8(y_probs),
                                 int(index_start), int(index_end), _f8(bins), True)
        return self._res_rel_reference(y_true, y_probs, index_start, index_end,
                                       bins, want_resolution=True)

    def reliability_score_grid(self, y_true, y_probs, index_start, index_end,
                               bins=_DEFAULT_BINS):
        """
        Reliability score on a grid point, Weijs (2010).

        Measures the calibration of forecast probabilities against observed
        frequencies.

        Returns
        -------
        float
            Reliability; np.nan if insufficient valid data.
        """
        if _use_numba():
            return _res_rel_core(_f8(y_true), _f8(y_probs),
                                 int(index_start), int(index_end), _f8(bins), False)
        return self._res_rel_reference(y_true, y_probs, index_start, index_end,
                                       bins, want_resolution=False)

    def _res_rel_reference(self, y_true, y_probs, index_start, index_end,
                           bins, want_resolution):
        """Pure-NumPy fallback shared by resolution_score_grid / reliability_score_grid."""
        if len(y_true) == 0 or y_probs.shape[1] == 0:
            return np.nan
        if index_end <= index_start:
            return np.nan

        clim_period = y_true[index_start:index_end]
        valid_clim_data = clim_period[np.isfinite(clim_period)]
        if len(valid_clim_data) < (5 if want_resolution else 10):
            return np.nan

        terciles = np.nanpercentile(clim_period, [100.0 / 3.0, 200.0 / 3.0])
        mask = np.isfinite(y_true) & np.isfinite(y_probs).all(axis=0)
        if not np.any(mask):
            return np.nan

        y_true_clean = y_true[mask]
        y_probs_clean = y_probs[:, mask]
        y_true_clean_class = np.digitize(y_true_clean, bins=terciles, right=True)
        n_categories, n_instances = y_probs_clean.shape
        y_bar = [np.mean(y_true_clean_class == k) for k in range(n_categories)]

        total = 0.0
        for k in range(n_categories):
            y_probs_clean_k = y_probs_clean[k, :]
            binned_indices = np.digitize(y_probs_clean_k, bins) - 1
            for b in range(len(bins) - 1):
                bin_mask = binned_indices == b
                n_kb = np.sum(bin_mask)
                if n_kb == 0:
                    continue
                y_kb = np.mean(y_true_clean_class[bin_mask] == k)
                p_kb = np.mean(y_probs_clean_k[bin_mask])
                if want_resolution:
                    if 0 < y_kb < 1 and 0 < y_bar[k] < 1:
                        term1 = y_kb * np.log2(y_kb / y_bar[k])
                        term2 = (1 - y_kb) * np.log2((1 - y_kb) / (1 - y_bar[k]))
                        total += (n_kb / n_instances) * (term1 + term2)
                else:
                    if 0 < y_kb < 1 and 0 < p_kb < 1:
                        term1 = y_kb * np.log2(y_kb / p_kb)
                        term2 = (1 - y_kb) * np.log2((1 - y_kb) / (1 - p_kb))
                        total += (n_kb / n_instances) * (term1 + term2)
        return total

    def resolution_and_reliability_per_category_grid(self, y_true, y_probs,
                                                     index_start, index_end,
                                                     bins=_DEFAULT_BINS):
        """
        Resolution and Reliability per category for one grid point, Weijs (2010).

        Returns
        -------
        dict
            Keys: 'total_resolution', 'total_reliability', 'category_resolution',
            'category_reliability', 'category_base_rates'.
        """
        terciles = np.nanpercentile(y_true[index_start:index_end], [100.0 / 3.0, 200.0 / 3.0])
        if not np.isfinite(terciles).all():
            return np.nan
        mask = np.isfinite(y_true) & np.isfinite(y_probs).all(axis=0)
        if not np.any(mask):
            return {
                'total_resolution': np.nan,
                'total_reliability': np.nan,
                'category_resolution': [np.nan, np.nan, np.nan],
                'category_reliability': [np.nan, np.nan, np.nan],
                'category_base_rates': [np.nan, np.nan, np.nan],
            }

        y_true_clean = y_true[mask]
        y_probs_clean = y_probs[:, mask]
        y_true_clean_class = np.digitize(y_true_clean, bins=terciles, right=True)

        n_categories, n_instances = y_probs_clean.shape
        category_resolution = [0.0] * n_categories
        category_reliability = [0.0] * n_categories
        category_base_rates = [np.mean(y_true_clean_class == k) for k in range(n_categories)]

        total_resolution = 0.0
        total_reliability = 0.0
        for k in range(n_categories):
            y_probs_clean_k = y_probs_clean[k, :]
            binned_indices = np.digitize(y_probs_clean_k, bins) - 1
            cat_res_k = 0.0
            cat_rel_k = 0.0
            for b in range(len(bins) - 1):
                bin_mask = binned_indices == b
                n_kb = np.sum(bin_mask)
                if n_kb == 0:
                    continue
                y_kb = np.mean(y_true_clean_class[bin_mask] == k)
                p_kb = np.mean(y_probs_clean_k[bin_mask])
                if 0 < y_kb < 1 and 0 < category_base_rates[k] < 1:
                    cat_res_k += (n_kb / n_instances) * (
                        y_kb * np.log2(y_kb / category_base_rates[k])
                        + (1 - y_kb) * np.log2((1 - y_kb) / (1 - category_base_rates[k]))
                    )
                if 0 < y_kb < 1 and 0 < p_kb < 1:
                    cat_rel_k += (n_kb / n_instances) * (
                        y_kb * np.log2(y_kb / p_kb)
                        + (1 - y_kb) * np.log2((1 - y_kb) / (1 - p_kb))
                    )
            category_resolution[k] = cat_res_k
            category_reliability[k] = cat_rel_k
            total_resolution += cat_res_k
            total_reliability += cat_rel_k

        return {
            'total_resolution': total_resolution,
            'total_reliability': total_reliability,
            'category_resolution': category_resolution,
            'category_reliability': category_reliability,
            'category_base_rates': category_base_rates,
        }

    def resolution_and_reliability_over_all_grid(self, y_true_class, y_probs,
                                                 bins=_DEFAULT_BINS):
        """
        Resolution and Reliability aggregated over all grid points, Weijs (2010).

        Parameters
        ----------
        y_true_class : xarray.DataArray
            Tercile labels (T, Y, X).
        y_probs : xarray.DataArray
            Forecast probabilities (probability, T, Y, X).

        Returns
        -------
        tuple
            (resolution_sum, reliability_sum)
        """
        res = self.resolution_and_reliability_over_all_grid_per_category(
            y_true_class, y_probs, bins)
        return res['total_resolution'], res['total_reliability']

    def resolution_and_reliability_over_all_grid_per_category(self, y_true_class, y_probs,
                                                              bins=_DEFAULT_BINS):
        """
        Per-category Resolution and Reliability aggregated over all grid points.

        Assumes ``y_true_class`` already holds category labels (0, 1, 2).

        Returns
        -------
        dict
            Keys: 'total_resolution', 'total_reliability', 'category_resolution',
            'category_reliability', 'category_base_rates', 'total_instances'.
        """
        y_true_class, y_probs = xr.align(y_true_class, y_probs)
        observed_outcomes = y_true_class.stack(flat_dim=("T", "Y", "X")).values
        predicted_probs = y_probs.stack(flat_dim=("T", "Y", "X")).values

        mask = ~np.isnan(observed_outcomes) & np.isfinite(predicted_probs).all(axis=0)
        observed_classes = observed_outcomes[mask].astype(int)
        predicted_probabilities = predicted_probs[:, mask]

        n_categories, n_instances = predicted_probabilities.shape
        base_rates = [np.mean(observed_classes == k) for k in range(n_categories)]

        total_resolution = 0.0
        total_reliability = 0.0
        category_resolution = [0.0] * n_categories
        category_reliability = [0.0] * n_categories

        for k in range(n_categories):
            y_probs_k = predicted_probabilities[k, :]
            binned_indices = np.digitize(y_probs_k, bins) - 1
            cat_res_k = 0.0
            cat_rel_k = 0.0
            for b in range(len(bins) - 1):
                bin_mask = binned_indices == b
                n_kb = np.sum(bin_mask)
                if n_kb == 0:
                    continue
                y_kb = np.mean(observed_classes[bin_mask] == k)
                p_kb = np.mean(y_probs_k[bin_mask])
                if 0 < y_kb < 1 and 0 < base_rates[k] < 1:
                    cat_res_k += (n_kb / n_instances) * (
                        y_kb * np.log2(y_kb / base_rates[k])
                        + (1 - y_kb) * np.log2((1 - y_kb) / (1 - base_rates[k]))
                    )
                if 0 < y_kb < 1 and 0 < p_kb < 1:
                    cat_rel_k += (n_kb / n_instances) * (
                        y_kb * np.log2(y_kb / p_kb)
                        + (1 - y_kb) * np.log2((1 - y_kb) / (1 - p_kb))
                    )
            category_resolution[k] = cat_res_k
            category_reliability[k] = cat_rel_k
            total_resolution += cat_res_k
            total_reliability += cat_rel_k

        return {
            'total_resolution': total_resolution,
            'total_reliability': total_reliability,
            'category_resolution': category_resolution,
            'category_reliability': category_reliability,
            'category_base_rates': base_rates,
            'total_instances': n_instances,
        }

    def reliability_diagram(self, modelname, dir_to_save_score, y_true_class, y_probs,
                            bins=np.array([0.100, 0.150, 0.200, 0.250, 0.300, 0.350,
                                           0.400, 0.450, 0.500, 0.550, 0.600, 0.650,
                                           0.6700])):
        """
        Plot Reliability Diagrams for probabilistic forecasts.

        Parameters
        ----------
        modelname : str
            Name used in the title and file name.
        dir_to_save_score : str or Path
            Output directory.
        y_true_class : xarray.DataArray
            Tercile labels (T, Y, X).
        y_probs : xarray.DataArray
            Forecast probabilities (probability, T, Y, X).
        bins : array-like, optional
            Probability bins.
        """
        y_true_class, y_probs = xr.align(y_true_class, y_probs)
        observed_outcomes = y_true_class.stack(flat_dim=("T", "Y", "X")).values
        predicted_probs = y_probs.stack(flat_dim=("T", "Y", "X")).values
        mask = ~np.isnan(observed_outcomes) & np.isfinite(predicted_probs).all(axis=0)
        observed_classes = observed_outcomes[mask]
        predicted_probabilities = predicted_probs[:, mask]

        result_all = self.resolution_and_reliability_over_all_grid_per_category(
            y_true_class, y_probs, bins)

        n_bins = len(bins) - 1
        observed_freqs = {0: np.zeros(n_bins), 1: np.zeros(n_bins), 2: np.zeros(n_bins)}
        forecast_counts = {0: np.zeros(n_bins), 1: np.zeros(n_bins), 2: np.zeros(n_bins)}

        # Vectorised over samples (the original looped in Python over every
        # sample x tercile; identical binning, far fewer iterations).
        for tercile in range(3):
            probs_t = predicted_probabilities[tercile, :]
            ok = np.isfinite(probs_t) & np.isfinite(observed_classes)
            idx = np.clip(np.digitize(probs_t[ok], bins) - 1, 0, n_bins - 1)
            hit = observed_classes[ok] == tercile
            forecast_counts[tercile] = np.bincount(idx, minlength=n_bins).astype(float)
            observed_freqs[tercile] = np.bincount(idx, weights=hit.astype(float),
                                                  minlength=n_bins).astype(float)
            observed_freqs[tercile] = np.divide(
                observed_freqs[tercile], forecast_counts[tercile],
                out=np.zeros_like(observed_freqs[tercile]),
                where=forecast_counts[tercile] != 0)

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        titles = ["BELOW AVERAGE", "NEAR AVERAGE", "ABOVE AVERAGE"]
        for idx_t, tercile in enumerate(range(3)):
            ax = axs[idx_t]
            ax.plot(bins[:-1] * 100, observed_freqs[tercile] * 100, 'k-', lw=2,
                    color="black", label="Reliability Curve")
            non_zero_mask = forecast_counts[tercile] > 0
            if np.any(non_zero_mask):
                slope, intercept, _, _, _ = linregress(
                    bins[:-1][non_zero_mask], observed_freqs[tercile][non_zero_mask])
                ax.plot(bins[:-1] * 100, (slope * bins[:-1] + intercept) * 100, 'k--',
                        color="black", lw=1, label="Regression Fit")
            ax.plot([0, 100], [0, 100], 'r:', color="red", lw=1.5, label="Perfect Reliability")

            total_observed = np.sum(observed_classes == tercile)
            total_forecasts = len(observed_classes)
            relative_frequency = (total_observed / total_forecasts) * 100
            ax.axhline(relative_frequency, linestyle='--', color="blue", lw=0.8,
                       label="Relative Frequency")
            ax.axvline(relative_frequency, linestyle='--', color="blue", lw=0.8)

            no_skill_x = np.linspace(0, 100, 100)
            no_skill_y = 0.5 * no_skill_x + relative_frequency / 2
            ax.plot(no_skill_x, no_skill_y, 'b--', color="orange", lw=2, label="No Skill Line")
            ax.fill_between(no_skill_x, no_skill_y, 0,
                            where=(no_skill_x <= relative_frequency), color='gray', alpha=0.2)
            ax.fill_between(no_skill_x, no_skill_y, 100,
                            where=(no_skill_x >= relative_frequency), color='gray', alpha=0.2)

            ax.text(0.05, 0.78, f"REL= {result_all['category_reliability'][idx_t]:.2f}",
                    transform=ax.transAxes, fontsize=12)
            ax.text(0.05, 0.671, f"RES= {result_all['category_resolution'][idx_t]:.2f}",
                    transform=ax.transAxes, fontsize=12)
            ax.bar(bins[:-1] * 100, (forecast_counts[tercile] / total_forecasts) * 100,
                   width=5, color="grey", alpha=0.5, align="edge")
            ax.set_title(titles[tercile])
            ax.set_xlim([0, 100])
            ax.set_ylim([0, 100])
            ax.set_xlabel("Forecast Probability (%)", fontsize=14)
            ax.set_ylabel("Observed Relative Frequency (%)", fontsize=13)
            ax.tick_params(axis="both", labelsize=12)

        fig.suptitle(f"Reliability Diagrams {modelname}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs(f"{dir_to_save_score}", exist_ok=True)
        plt.savefig(f"{dir_to_save_score}/RELIABILITY_{modelname}_.pdf", dpi=300,
                    bbox_inches='tight')
        plt.show()

    def plot_roc_curves(self, modelname, dir_to_save_score, y_true_class, y_probs,
                        n_bootstraps=200, ci=0.95):
        """
        Plot ROC Curves for probabilistic forecasts, one panel per tercile.

        Parameters
        ----------
        modelname : str
            Name used in the title and file name.
        dir_to_save_score : str or Path
            Output directory.
        y_true_class : xarray.DataArray
            Tercile labels (T, Y, X).
        y_probs : xarray.DataArray
            Forecast probabilities (probability, T, Y, X).
        n_bootstraps : int, optional
            Bootstrap resamples for the confidence band. Default 200.
        ci : float, optional
            Confidence level. Default 0.95.
        """
        labels = ["BELOW AVERAGE", "NEAR AVERAGE", "ABOVE AVERAGE"]
        y_true_class, y_probs = xr.align(y_true_class, y_probs)
        observed_outcomes = y_true_class.stack(flat_dim=("T", "Y", "X")).values
        predicted_probs = y_probs.stack(flat_dim=("T", "Y", "X")).values
        mask = ~np.isnan(observed_outcomes) & np.isfinite(predicted_probs).all(axis=0)
        observed_outcomes = observed_outcomes[mask]
        predicted_probs = predicted_probs[:, mask]

        fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
        for i, ax in enumerate(axes):
            binary_labels = (observed_outcomes == i).astype(int)
            fpr, tpr, _ = roc_curve(binary_labels, predicted_probs[i, :])
            roc_auc = auc(fpr, tpr)

            tprs_bootstrap = []
            mean_fpr = np.linspace(0, 1, 100)
            for _ in range(n_bootstraps):
                indices = resample(np.arange(len(predicted_probs[i])), replace=True)
                boot_binary_labels = binary_labels[indices]
                boot_predicted_probs = predicted_probs[i, indices]
                if np.unique(boot_binary_labels).size < 2:
                    continue
                boot_fpr, boot_tpr, _ = roc_curve(boot_binary_labels, boot_predicted_probs)
                interp_tpr = np.interp(mean_fpr, boot_fpr, boot_tpr)
                interp_tpr[0] = 0.0
                tprs_bootstrap.append(interp_tpr)
            if not tprs_bootstrap:
                print(f"No valid bootstrap samples for {labels[i]} category.")
                continue

            ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], 'k--', lw=2, label='No Skill')
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1.5)
                spine.set_visible(True)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_title(f'ROC Curve for {labels[i]} Category', fontsize=12)
            ax.set_xlabel('False Positive Rate', fontsize=14)
            if i == 0:
                ax.set_ylabel('True Positive Rate', fontsize=14)
            ax.legend(fontsize=10, loc="lower right")
            ax.tick_params(axis="both", labelsize=12)
            ax.grid(True)

        fig.suptitle(f"ROC Curves for {modelname}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs(f"{dir_to_save_score}", exist_ok=True)
        plt.savefig(f"{dir_to_save_score}/ROC_{modelname}_.pdf", dpi=300, bbox_inches='tight')
        plt.show()

    # ------------------------
    # Ensemble Metrics
    # ------------------------

    def compute_crps(self, y_true, y_pred, member_dim='number', dim="T"):
        """
        Compute CRPS for ensemble forecasts.

        Parameters
        ----------
        y_true : xarray.DataArray
            Observed data (T, Y, X).
        y_pred : xarray.DataArray
            Ensemble forecast (T, number, Y, X).
        member_dim : str, optional
            Ensemble member dimension. Default 'number'.
        dim : str, optional
            Dimension to reduce. Default 'T'.

        Returns
        -------
        xarray.DataArray
            CRPS (Y, X).
        """
        y_true, y_pred = xr.align(y_true, y_pred)
        return xs.crps_ensemble(y_true, y_pred, member_dim=member_dim, dim=dim)

    # ------------------------
    # Probabilistic Scoring
    # ------------------------

    def score_fold_probabilities(self, obs, probabilities, thresholds, reference_probabilities):
        """Evaluate probabilities using the exact declared climatological events."""
        from .was_crossfit import score_fold_probabilities
        return score_fold_probabilities(obs, probabilities, thresholds, reference_probabilities)

    def compute_probabilistic_score(self, score_func, obs, prob_pred,
                                    clim_year_start, clim_year_end, *,
                                    thresholds=None, reference_probabilities=None,
                                    **score_kwargs):
        """
        Apply a probabilistic scoring function over xarray DataArrays.

        Parameters
        ----------
        score_func : callable
            Probabilistic scoring function (e.g., calculate_rpss, calculate_groc).
        obs : xarray.DataArray
            Observed data (T, Y, X).
        prob_pred : xarray.DataArray
            Forecast probabilities (probability, T, Y, X).
        clim_year_start, clim_year_end : int or str
            Climatology period.
        thresholds : DataArray, optional
            Exact producer event thresholds (quantile,Y,X or quantile,T,Y,X).
            Saved threshold coordinates are also detected automatically. No
            thresholds are re-estimated when these are supplied.
        reference_probabilities : DataArray, optional
            Climatological category frequencies for RPSS/BSS. Defaults to 1/3
            when explicit thresholds have no accompanying reference.

        Returns
        -------
        xarray.DataArray
            Score values (Y, X).
        """
        if obs.get_index('T').has_duplicates or prob_pred.get_index('T').has_duplicates:
            raise ValueError('Duplicate verification times')
        event_coords = ('fold_threshold_lower', 'fold_threshold_upper', 'fold_reference_probability')
        if thresholds is None and all(k in prob_pred.coords for k in event_coords[:2]):
            thresholds = xr.concat([prob_pred.coords[k].drop_vars(
                [v for v in event_coords if v in prob_pred.coords[k].coords])
                for k in event_coords[:2]], dim=xr.IndexVariable('quantile', [.33, .67]))
        if reference_probabilities is None and event_coords[2] in prob_pred.coords:
            reference_probabilities = prob_pred.coords[event_coords[2]].drop_vars(
                [v for v in event_coords if v in prob_pred.coords[event_coords[2]].coords])
        prob_pred = prob_pred.drop_vars([k for k in event_coords if k in prob_pred.coords])
        obs = obs.sortby('T')
        # Keep all observed climatology dates even if a forecast is unavailable.
        prob_pred = prob_pred.reindex(T=obs['T'])
        obs, prob_pred = xr.align(obs, prob_pred, join='exact')
        labels = [str(v) for v in prob_pred['probability'].values]
        if labels != ['PB','PN','PA']:
            raise ValueError('Expected probability labels/order PB, PN, PA')
        invalid = (np.isinf(prob_pred) | (np.isfinite(prob_pred) & ((prob_pred < 0) | (prob_pred > 1)))).any()
        complete = np.isfinite(prob_pred).all('probability')
        partial = np.isfinite(prob_pred).any('probability') & ~complete
        bad_sum = (complete & (abs(prob_pred.sum('probability')-1) > 1e-6)).any()
        if bool(invalid.compute()) or bool(partial.any().compute()) or bool(bad_sum.compute()):
            raise ValueError('Invalid probability simplex; use all-NaN for missing forecasts')
        if thresholds is not None:
            if thresholds.sizes.get('quantile') != 2:
                raise ValueError('Expected exactly two tercile thresholds')
            q = thresholds.drop_vars([k for k in event_coords if k in thresholds.coords])
            if 'T' in q.dims:
                q = q.reindex(T=obs['T'])
            lower, upper = q.isel(quantile=0, drop=True), q.isel(quantile=1, drop=True)
            if bool((lower > upper).any().compute()):
                raise ValueError('Reversed tercile thresholds')
            lower = lower.broadcast_like(obs)
            upper = upper.broadcast_like(obs)
            if reference_probabilities is None:
                reference = xr.full_like(prob_pred, 1/3, dtype=float)
            else:
                reference = reference_probabilities.drop_vars(
                    [k for k in event_coords if k in reference_probabilities.coords])
                if 'T' in reference.dims:
                    reference = reference.reindex(T=obs['T'])
                if list(reference.probability.values) != ['PB', 'PN', 'PA']:
                    raise ValueError('Expected reference probability labels/order PB, PN, PA')
                reference = reference.broadcast_like(prob_pred)
                good = np.isfinite(reference).all('probability')
                partial_ref = np.isfinite(reference).any('probability') & ~good
                invalid_ref = ((reference < 0) | (reference > 1) | np.isinf(reference)).any()
                invalid_ref = invalid_ref | partial_ref.any() | (good & (abs(reference.sum('probability', skipna=False)-1)>1e-6)).any()
                if bool(invalid_ref.compute()):
                    raise ValueError('Invalid reference probability simplex')
            inputs = xr.align(obs, prob_pred, lower, upper, reference, join='exact')
            result = xr.apply_ufunc(_score_at_declared_events, *inputs,
                input_core_dims=[['T'], ['probability','T'], ['T'], ['T'], ['probability','T']],
                output_core_dims=[[]], vectorize=True, dask='parallelized', output_dtypes=[float],
                kwargs={'name': getattr(score_func, '__name__', ''), 'options': score_kwargs},
                dask_gufunc_kwargs={'allow_rechunk': True})
            result.attrs.update(tercile_reference=thresholds.attrs.get('tercile_reference', 'declared'),
                                climatology_start=str(clim_year_start), climatology_end=str(clim_year_end),
                                event_definition='PB: y<=lower; PN: lower<y<=upper; PA: y>upper')
            return result
        years = obs['T'].dt.year.values
        index_start = int(np.searchsorted(years,int(clim_year_start),side='left'))
        index_end = int(np.searchsorted(years,int(clim_year_end),side='right'))
        if index_end <= index_start:
            raise ValueError('No observations in requested climatology period')
        _kw = {'index_start': index_start, 'index_end': index_end, **score_kwargs}

        def _eager():
            return xr.apply_ufunc(
                score_func, obs, prob_pred,
                input_core_dims=[('T',), ('probability', 'T')],
                vectorize=True, kwargs=_kw, dask='parallelized',
                output_core_dims=[()], output_dtypes=['float'],
                dask_gufunc_kwargs={"allow_rechunk": True},
            )

        if os.environ.get("WAS_VERIF_PARALLEL", "1") != "0":
            n = _verif_num_workers()
            ncells = int(obs.sizes.get("Y", 1)) * int(obs.sizes.get("X", 1))
            min_cells = int(os.environ.get("WAS_VERIF_MIN_CELLS", "4000"))
            if n > 1 and ncells >= min_cells:
                try:
                    res = xr.apply_ufunc(
                        score_func, _chunk_grid(obs, n), _chunk_grid(prob_pred, n),
                        input_core_dims=[('T',), ('probability', 'T')],
                        vectorize=True, kwargs=_kw, dask='parallelized',
                        output_core_dims=[()], output_dtypes=['float'],
                        dask_gufunc_kwargs={"allow_rechunk": True},
                    )
                    return res.compute(scheduler="threads", num_workers=n)
                except Exception:
                    pass
        return _eager()

    def plot_model_score(self, model_metric, score, dir_save_score, figure_name="WAS_MLR"):
        """
        Plot a single score on a map.

        Parameters
        ----------
        model_metric : xarray.DataArray
            Score values (Y, X).
        score : str
            Score key (e.g., 'Pearson', 'MAE').
        dir_save_score : str or Path
            Output directory.
        figure_name : str, optional
            File name prefix.
        """
        dir_save_score = Path(dir_save_score)
        dir_save_score.mkdir(parents=True, exist_ok=True)
        score_meta_data = self.scores[score]
        fig, ax = plt.subplots(figsize=(6, 4), subplot_kw={'projection': ccrs.PlateCarree()})
        im = model_metric.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=score_meta_data[4],
            vmin=score_meta_data[1] if score_meta_data[1] is not None else None,
            vmax=score_meta_data[2] if score_meta_data[2] is not None else None,
            add_colorbar=False
        )
        ax.coastlines(resolution='10m')
        ax.add_feature(cfeature.BORDERS, linestyle='--')
        ax.set_title(f"{score} {figure_name}")
        gl = ax.gridlines(draw_labels=True, linewidth=0.1, color='gray', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.5)
        cbar.ax.set_position([ax.get_position().x0, ax.get_position().y0 - 0.1,
                              ax.get_position().width, 0.039])

        name, _, _, _, _, _, unit = self.scores[score]
        cbar.set_label(f"{name} {unit}".rstrip())

        plt.savefig(dir_save_score / f"{figure_name}_{score}.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    def plot_models_score(self, model_metrics, score, dir_save_score):
        """
        Plot several model scores on a grid of maps.

        Parameters
        ----------
        model_metrics : dict
            Model name -> score DataArray (Y, X).
        score : str
            Score key.
        dir_save_score : str or Path
            Output directory.
        """
        dir_save_score = Path(dir_save_score)
        dir_save_score.mkdir(parents=True, exist_ok=True)

        if score not in self.scores:
            raise ValueError(f"Score '{score}' is not defined in the scores dictionary. "
                             f"Available scores: {list(self.scores.keys())}")

        score_meta_data = self.scores[score]
        n_scores = len(model_metrics)
        n_cols = min(3, n_scores)
        n_rows = int(np.ceil(n_scores / n_cols))

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(n_cols * 6, n_rows * 4),
            subplot_kw={'projection': ccrs.PlateCarree()}
        )
        if n_scores == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        vmin = score_meta_data[1]
        vmax = score_meta_data[2]
        if vmin is not None:
            try:
                vmin = float(vmin)
            except (ValueError, TypeError):
                vmin = None
        if vmax is not None:
            try:
                vmax = float(vmax)
            except (ValueError, TypeError):
                vmax = None

        i = -1
        for i, (center, data) in enumerate(model_metrics.items()):
            ax = axes[i]
            im = data.plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap=score_meta_data[4] if score_meta_data[4] else 'viridis',
                vmin=vmin,
                vmax=vmax,
                add_colorbar=False
            )
            ax.coastlines(resolution='10m')
            ax.add_feature(cfeature.BORDERS, linestyle='--')
            ax.set_title(f"{score} {center[:-1]}")
            gl = ax.gridlines(draw_labels=True, linewidth=0.1, color='gray', alpha=0.5)
            gl.top_labels = False
            gl.right_labels = False
            cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.15, shrink=0.7)
            cbar.ax.set_position([
                ax.get_position().x0,
                ax.get_position().y0 - 0.12,
                ax.get_position().width,
                0.03
            ])
            name, _, _, _, _, _, unit = score_meta_data
            label = f"{name}"
            if unit and unit.strip():
                label += f" {unit}"
            cbar.set_label(label)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        plt.savefig(dir_save_score / f"{score}_all_models.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    # ------------------ Probability Calculation Methods ------------------

    @staticmethod
    def _ppf_terciles_from_code(dist_code, shape, loc, scale):
        """
        Tercile thresholds (T1, T2) from best-fit distribution parameters.

        dist_code: 1 norm, 2 lognorm, 3 expon, 4 gamma, 5 weibull_min,
        6 student-t, 7 poisson, 8 nbinom.
        """
        if np.isnan(dist_code):
            return np.nan, np.nan

        code = int(dist_code)
        try:
            if code == 1:
                return (norm.ppf(0.33, loc=loc, scale=scale),
                        norm.ppf(0.67, loc=loc, scale=scale))
            elif code == 2:
                return (lognorm.ppf(0.33, s=shape, loc=loc, scale=scale),
                        lognorm.ppf(0.67, s=shape, loc=loc, scale=scale))
            elif code == 3:
                return (expon.ppf(0.33, loc=loc, scale=scale),
                        expon.ppf(0.67, loc=loc, scale=scale))
            elif code == 4:
                return (gamma.ppf(0.33, a=shape, loc=loc, scale=scale),
                        gamma.ppf(0.67, a=shape, loc=loc, scale=scale))
            elif code == 5:
                return (weibull_min.ppf(0.33, c=shape, loc=loc, scale=scale),
                        weibull_min.ppf(0.67, c=shape, loc=loc, scale=scale))
            elif code == 6:
                return (student_t.ppf(0.33, df=shape, loc=loc, scale=scale),
                        student_t.ppf(0.67, df=shape, loc=loc, scale=scale))
            elif code == 7:
                return (poisson.ppf(0.33, mu=shape, loc=loc),
                        poisson.ppf(0.67, mu=shape, loc=loc))
            elif code == 8:
                return (nbinom.ppf(0.33, n=shape, p=scale, loc=loc),
                        nbinom.ppf(0.67, n=shape, p=scale, loc=loc))
        except Exception:
            return np.nan, np.nan
        return np.nan, np.nan

    @staticmethod
    def weibull_shape_solver(k, M, V):
        """
        Root function for the Weibull shape parameter 'k'.

        Finds k such that the theoretical variance/mean^2 ratio matches the
        observed V/M^2 ratio.
        """
        if k <= 0:
            return -np.inf
        try:
            g1 = gamma_function(1 + 1 / k)
            g2 = gamma_function(1 + 2 / k)
            implied_v_over_m_sq = (g2 / (g1 ** 2)) - 1
            observed_v_over_m_sq = V / (M ** 2)
            return observed_v_over_m_sq - implied_v_over_m_sq
        except ValueError:
            return -np.inf

    @staticmethod
    def calculate_tercile_probabilities_bestfit(best_guess, error_variance, T1, T2,
                                                dist_code, dof):
        """
        Tercile probabilities using the best-fit family of each grid cell.

        For every time step a predictive distribution of the same family is
        built from ``best_guess[t]`` and the climatological error variance, then
        P(B) = F(T1), P(N) = F(T2) - F(T1), P(A) = 1 - F(T2).
        """
        best_guess = np.asarray(best_guess, float)
        error_variance = np.asarray(error_variance, dtype=float)
        n_time = best_guess.size
        out = np.full((3, n_time), np.nan, float)

        if (np.all(np.isnan(best_guess)) or np.isnan(dist_code)
                or np.isnan(T1) or np.isnan(T2) or np.isnan(error_variance)):
            return out

        code = int(dist_code)

        if code == 1:                                   # Normal
            error_std = np.sqrt(error_variance)
            c1 = norm.cdf(T1, loc=best_guess, scale=error_std)
            c2 = norm.cdf(T2, loc=best_guess, scale=error_std)
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        elif code == 2:                                 # Lognormal
            sigma = np.sqrt(np.log(1 + error_variance / (best_guess ** 2)))
            mu = np.log(best_guess) - sigma ** 2 / 2
            c1 = lognorm.cdf(T1, s=sigma, scale=np.exp(mu))
            c2 = lognorm.cdf(T2, s=sigma, scale=np.exp(mu))
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        elif code == 3:                                 # Exponential
            scale = np.sqrt(error_variance)
            c1 = expon.cdf(T1, loc=best_guess, scale=scale)
            c2 = expon.cdf(T2, loc=best_guess, scale=scale)
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        elif code == 4:                                 # Gamma
            alpha = (best_guess ** 2) / error_variance
            theta = error_variance / best_guess
            c1 = gamma.cdf(T1, a=alpha, scale=theta)
            c2 = gamma.cdf(T2, a=alpha, scale=theta)
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        elif code == 5:                                 # Weibull
            V = float(error_variance)
            for i in range(n_time):
                M = best_guess[i]
                if V <= 0 or M <= 0 or not np.isfinite(M):
                    continue
                k = fsolve(WAS_Verification.weibull_shape_solver, 2.0, args=(M, V))[0]
                if k <= 0:
                    continue
                lambda_scale = M / gamma_function(1 + 1 / k)
                c1 = weibull_min.cdf(T1, c=k, loc=0, scale=lambda_scale)
                c2 = weibull_min.cdf(T2, c=k, loc=0, scale=lambda_scale)
                out[0, i] = c1
                out[1, i] = c2 - c1
                out[2, i] = 1.0 - c2

        elif code == 6:                                 # Student-t
            if dof <= 2:
                return out
            scale = np.sqrt(error_variance * (dof - 2) / dof)
            c1 = student_t.cdf(T1, df=dof, loc=best_guess, scale=scale)
            c2 = student_t.cdf(T2, df=dof, loc=best_guess, scale=scale)
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        elif code == 7:                                 # Poisson
            if not np.allclose(best_guess, error_variance, atol=0.5):
                print("Warning: 'error_variance' is not equal to 'best_guess'. "
                      "Poisson assumes mean=variance and is likely inappropriate; "
                      "consider Negative Binomial.")
            c1 = poisson.cdf(T1, mu=best_guess)
            c2 = poisson.cdf(T2, mu=best_guess)
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        elif code == 8:                                 # Negative Binomial
            p = np.where(error_variance > best_guess, best_guess / error_variance, np.nan)
            n = np.where(error_variance > best_guess,
                         (best_guess ** 2) / (error_variance - best_guess), np.nan)
            c1 = nbinom.cdf(T1, n=n, p=p)
            c2 = nbinom.cdf(T2, n=n, p=p)
            out[0, :] = c1
            out[1, :] = c2 - c1
            out[2, :] = 1.0 - c2

        else:
            raise ValueError(f"Invalid distribution code: {code}")

        return out

    @staticmethod
    def calculate_tercile_probabilities_nonparametric(best_guess, error_samples,
                                                      first_tercile, second_tercile):
        """Non-parametric method using historical error samples."""
        if _use_numba():
            return _tercile_prob_nonparam_core(
                _f8(best_guess), _f8(error_samples),
                float(first_tercile), float(second_tercile))

        n_time = len(best_guess)
        pred_prob = np.full((3, n_time), np.nan, dtype=float)
        for tt in range(n_time):
            if np.isnan(best_guess[tt]):
                continue
            dist = best_guess[tt] + error_samples
            dist = dist[np.isfinite(dist)]
            if len(dist) == 0:
                continue
            p_below = np.mean(dist <= first_tercile)
            p_between = np.mean((dist > first_tercile) & (dist <= second_tercile))
            pred_prob[0, tt] = p_below
            pred_prob[1, tt] = p_between
            pred_prob[2, tt] = 1.0 - (p_below + p_between)
        return pred_prob

    def gcm_compute_prob(
        self,
        Predictant: xr.DataArray,
        clim_year_start,
        clim_year_end,
        hindcast_det: xr.DataArray,
        best_code_da: xr.DataArray = None,
        best_shape_da: xr.DataArray = None,
        best_loc_da: xr.DataArray = None,
        best_scale_da: xr.DataArray = None
    ) -> xr.DataArray:
        """
        Compute tercile probabilities for deterministic hindcasts.

        Empirical terciles are fixed on the configured climatology.  The
        selected distribution changes the predictive CDF, never these events.

        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed data (T, Y, X) or (T, Y, X, M).
        clim_year_start, clim_year_end : int or str
            Climatology period (inclusive).
        hindcast_det : xarray.DataArray
            Deterministic hindcast (T, Y, X).
        best_code_da, best_shape_da, best_loc_da, best_scale_da : xarray.DataArray, optional
            Output of ``WAS_TransformData.fit_best_distribution``; required for 'bestfit'.

        Returns
        -------
        xarray.DataArray
            Probabilities (probability=['PB','PN','PA'], T, Y, X).
        """
        if "M" in Predictant.dims:
            Predictant = Predictant.isel(M=0).drop_vars("M").squeeze()

        Predictant = Predictant.transpose("T", "Y", "X")
        mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1.0, np.nan)

        clim = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))
        if clim.sizes.get("T", 0) < 3:
            raise ValueError("Not enough years in climatology period for terciles.")

        error_variance = (Predictant - hindcast_det).var(dim="T")
        dof = max(int(clim.sizes["T"]) - 1, 2)

        from .was_probability_reference import climatological_terciles
        terciles_emp, reference = climatological_terciles(
            Predictant, clim_year_start, clim_year_end
        )
        T1_emp = terciles_emp.isel(quantile=0).drop_vars("quantile")
        T2_emp = terciles_emp.isel(quantile=1).drop_vars("quantile")

        dm = self.dist_method

        if dm == "bestfit":
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError(
                    "dist_method='bestfit' requires best_code_da, best_shape_da, "
                    "best_loc_da, best_scale_da."
                )

            T1, T2 = T1_emp, T2_emp

            hindcast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_bestfit,
                hindcast_det, error_variance, T1, T2, best_code_da,
                input_core_dims=[("T",), (), (), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True,
                kwargs={'dof': dof},
                dask="parallelized",
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3},
                                    "allow_rechunk": True},
            )

        elif dm == "nonparam":
            error_samples = Predictant - hindcast_det
            hindcast_prob = xr.apply_ufunc(
                self.calculate_tercile_probabilities_nonparametric,
                hindcast_det, error_samples, T1_emp, T2_emp,
                input_core_dims=[("T",), ("T",), (), ()],
                output_core_dims=[("probability", "T")],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3},
                                    "allow_rechunk": True},
            )

        else:
            raise ValueError(f"Invalid dist_method: {self.dist_method}")

        hindcast_prob = hindcast_prob.assign_coords(
            probability=("probability", ["PB", "PN", "PA"])
        )
        hindcast_prob = (hindcast_prob * mask).transpose("probability", "T", "Y", "X")
        hindcast_prob = hindcast_prob.assign_coords(
            fold_threshold_lower=T1_emp,
            fold_threshold_upper=T2_emp,
            fold_reference_probability=reference,
        )
        hindcast_prob.attrs.update(terciles_emp.attrs)
        return hindcast_prob

    def gcm_validation_compute(self, models_files_path, Obs, score, month_of_initialization,
                               clim_year_start, clim_year_end, dir_to_save_roc_reliability,
                               lead_time=None, ensemble_mean=None, gridded=True):
        """
        Validate several GCM hindcasts with the requested metric.

        Parameters
        ----------
        models_files_path : dict
            Model identifier -> hindcast NetCDF path.
        Obs : xarray.DataArray
            Observed data (T, Y, X).
        score : str
            Score key (e.g., 'Pearson', 'RPSS', 'ROC_CURVE', 'RELIABILITY_DIAGRAM').
        month_of_initialization : int
            Initialisation month (1-12).
        clim_year_start, clim_year_end : int or str
            Climatology period.
        dir_to_save_roc_reliability : str or Path
            Where ROC / reliability plots are written.
        lead_time : list of int, optional
            Lead times defining the target season, used only for the label.
        ensemble_mean : str, optional
            Ensemble statistic; if None the ensemble mean is taken where required.
        gridded : bool, optional
            Gridded validation (True). Non-gridded is not implemented.

        Returns
        -------
        dict or None
            Model identifier -> score DataArray (Y, X) for det/prob/ensemble
            scores; None for the plot-only scores.

        Notes
        -----
        Observations are interpolated onto each model grid, which is typically
        far coarser than the observation grid.  The resulting cell count often
        falls below ``WAS_VERIF_MIN_CELLS``, so these scores run on the eager
        path even when the thread pool is enabled.
        """
        abb_mont_ini = calendar.month_abbr[int(month_of_initialization)]
        if lead_time is None:
            season = ""
        else:
            season_months = [((int(month_of_initialization) + int(l) - 1) % 12) + 1
                             for l in lead_time]
            season = "".join([calendar.month_abbr[month] for month in season_months])

        x_metric = {}
        if gridded:
            Obs_data_ = Obs
            Obs_data_['T'] = Obs_data_['T'].astype('datetime64[ns]')
            for i in models_files_path.keys():
                score_type = self.scores[score][3]
                model_data_ = xr.open_dataset(models_files_path[i])
                model_data_['T'] = model_data_['T'].astype('datetime64[ns]')
                year_start_ = np.unique(model_data_['T'].dt.year)[0]
                year_end_ = np.unique(model_data_['T'].dt.year)[-1]
                Obs_data = Obs_data_.sel(T=slice(str(year_start_), str(year_end_))).interp(
                    Y=model_data_.Y, X=model_data_.X, method="linear",
                    kwargs={"fill_value": "extrapolate"}
                )

                model_data = model_data_.to_array().drop_vars("variable").squeeze()
                model_data['T'] = Obs_data['T']

                if score_type == "det_score":
                    if 'number' in model_data.dims:
                        model_data = model_data.mean(dim="number")
                    score_result = self.compute_deterministic_score(
                        self.scores[score][5], Obs_data, model_data
                    )
                    x_metric[f"{i}_{abb_mont_ini}Ic_{season}"] = score_result

                elif score_type == "prob_score":
                    if 'number' in model_data.dims:
                        model_data = model_data.mean(dim="number")
                    proba_forecast = self.gcm_compute_prob(
                        Obs_data, clim_year_start, clim_year_end, model_data)
                    score_result = self.compute_probabilistic_score(
                        self.scores[score][5], Obs_data, proba_forecast,
                        clim_year_start, clim_year_end
                    )
                    x_metric[f"{i}_{abb_mont_ini}Ic_{season}"] = score_result

                elif score_type == "ensemble_score":
                    if ensemble_mean is not None:
                        print("Ensemble score does not require an ensemble mean or median.")
                    else:
                        score_result = self.compute_crps(Obs_data, model_data,
                                                         member_dim='number', dim="T")
                        x_metric[f"{i}_{abb_mont_ini}Ic_{season}"] = score_result

                elif score_type == "all_grid_prob_score":
                    if (ensemble_mean is None) or ("number" in model_data.coords):
                        model_data = model_data.mean(dim="number", skipna=True)
                    proba_forecast = self.gcm_compute_prob(
                        Obs_data, clim_year_start, clim_year_end, model_data)
                    obs_class = self.compute_class(Obs_data, clim_year_start, clim_year_end)
                    if score == "ROC_CURVE":
                        self.plot_roc_curves(i, dir_to_save_roc_reliability,
                                             obs_class, proba_forecast,
                                             n_bootstraps=1000, ci=0.95)
                    elif score == "RELIABILITY_DIAGRAM":
                        self.reliability_diagram(i, dir_to_save_roc_reliability,
                                                 obs_class, proba_forecast)
                    else:
                        print(f"Plotting for score {score} is not implemented.")

                elif score_type == "all_grid_det_score":
                    pass
        else:
            print("Non-gridded data validation is not implemented yet.")

        return x_metric if self.scores[score][3] in ["det_score", "prob_score",
                                                     "ensemble_score"] else None

    def weighted_gcm_forecasts(self, Obs, best_models, scores, lead_time, model_dir,
                               clim_year_start, clim_year_end, variable="PRCP"):
        """
        GROC-weighted ensemble forecast from a set of selected GCMs.

        Combines hindcasts and forecasts weighted by GROC, rescales to the
        observed climatology, and returns tercile probabilities.

        Parameters
        ----------
        Obs : xarray.DataArray
            Observed data (T, Y, X).
        best_models : dict
            Model identifiers to include.
        scores : dict
            Score dictionary containing a 'GROC' entry per model.
        lead_time : list of int
            Lead times; only ``lead_time[0]`` is used to build the file names.
        model_dir : str or Path
            Directory holding the hindcast/forecast NetCDF files.
        clim_year_start, clim_year_end : int or str
            Climatology period.
        variable : str, optional
            Variable name in the files. Default 'PRCP'.

        Returns
        -------
        tuple
            (hindcast_det, hindcast_prob, forecast_prob)
        """
        parts = list(best_models.keys())[0].split("_")
        tmp = xr.open_dataset(
            f"{model_dir}/hindcast_{parts[0]}_{variable}_{parts[1]}_{parts[2]}_{lead_time[0]}.nc")
        score_sum = None
        hindcast_det = None
        forecast_det = None

        for model_name in best_models.keys():
            score_array = scores["GROC"][model_name].interp(
                Y=tmp.Y, X=tmp.X, method="nearest", kwargs={"fill_value": "extrapolate"})
            parts = model_name.split("_")
            hindcast_file = f"{model_dir}/hindcast_{parts[0]}_{variable}_{parts[1]}_{parts[2]}_{lead_time[0]}.nc"
            forecast_file = f"{model_dir}/forecast_{parts[0]}_{variable}_{parts[1]}_{parts[2]}_{lead_time[0]}.nc"

            hincast_data = xr.open_dataset(hindcast_file)
            if "number" in hincast_data.coords:
                hincast_data = hincast_data.mean(dim="number", skipna=True)
            hincast_data["T"] = hincast_data["T"].astype("datetime64[ns]")
            hincast_data = hincast_data.interp(
                Y=tmp.Y, X=tmp.X, method="nearest", kwargs={"fill_value": "extrapolate"})

            forecast_data = xr.open_dataset(forecast_file)
            if "number" in forecast_data.coords:
                forecast_data = forecast_data.mean(dim="number", skipna=True)
            forecast_data["T"] = forecast_data["T"].astype("datetime64[ns]")
            forecast_data = forecast_data.interp(
                Y=tmp.Y, X=tmp.X, method="nearest", kwargs={"fill_value": "extrapolate"})

            hincast_weighted = hincast_data * score_array
            forecast_weighted = forecast_data * score_array
            if hindcast_det is None:
                hindcast_det = hincast_weighted
                forecast_det = forecast_weighted
                score_sum = score_array
            else:
                hindcast_det = hindcast_det + hincast_weighted
                forecast_det = forecast_det + forecast_weighted
                score_sum = score_sum + score_array

        hindcast_det = (hindcast_det / score_sum).interp(
            Y=Obs.Y, X=Obs.X, method="nearest", kwargs={"fill_value": "extrapolate"})
        forecast_det = (forecast_det / score_sum).interp(
            Y=Obs.Y, X=Obs.X, method="nearest", kwargs={"fill_value": "extrapolate"})

        f = Obs.mean("T") / hindcast_det.mean("T")
        hindcast_det = hindcast_det * f
        forecast_det = forecast_det * f

        hindcast_det = (hindcast_det.to_array().drop_vars("variable").squeeze()
                        .transpose("T", "Y", "X"))
        forecast_det = (forecast_det.to_array().drop_vars("variable").squeeze("variable")
                        .transpose("T", "Y", "X"))

        hindcast_prob = self.gcm_compute_prob(Obs, clim_year_start, clim_year_end, hindcast_det)
        forecast_prob = (
            self.gcm_compute_prob_forecast(Obs, clim_year_start, clim_year_end,
                                           hindcast_det, forecast_det)
            .squeeze().drop_vars("T")
        )
        return hindcast_det, hindcast_prob, forecast_prob

    @staticmethod
    def classify_percent(p):
        """
        Classify a percentage of normal into five categories.

        Returns
        -------
        int
            1 Well Above (>=150%), 2 Above (>=110%), 3 Near (>=90%),
            4 Below (>=50%), 5 Well Below (<50%).
        """
        if p >= 150:
            return 1
        elif p >= 110:
            return 2
        elif p >= 90:
            return 3
        elif p >= 50:
            return 4
        return 5

    def ratio_to_average(self, predictant, clim_year_start, clim_year_end, year):
        """
        Map one year's data as a percentage of the climatological mean.

        Parameters
        ----------
        predictant : xarray.DataArray
            Observed data (T, Y, X).
        clim_year_start, clim_year_end : int or str
            Climatology period.
        year : int or str
            Year to display.
        """
        clim_slice = slice(str(clim_year_start), str(clim_year_end))
        clim_mean = predictant.sel(T=clim_slice).mean(dim='T')
        ratio = 100 * predictant.sel(T=str(year)) / clim_mean
        mask = xr.where(~np.isnan(predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze()
        mask.name = None
        classified = xr.apply_ufunc(
            self.classify_percent,
            ratio,
            input_core_dims=[('T',)],
            vectorize=True,
            dask='parallelized',
            output_core_dims=[()],
            output_dtypes=['float']
        ) * mask

        cmap = ListedColormap(['#1a9641', '#a6d96a', '#ffffbf', '#fdae61', '#d7191c'])
        labels = ["Well Above Avg", "Above Avg", "Near Avg", "Below Avg", "Well Below Avg"]
        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        classified.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, add_colorbar=False)
        ax.add_feature(cfeature.BORDERS)
        ax.add_feature(cfeature.COASTLINE)
        legend_patches = [mpatches.Patch(color=cmap(i), label=labels[i]) for i in range(5)]
        plt.legend(handles=legend_patches, loc='lower left')
        plt.title("Ratio to Normal [%]")
        plt.show()

    def compute_one_year_rpss_(self, obs, prob_pred, clim_year_start, clim_year_end, year):
        """
        Map the RPSS of a single year (simple version, no skill summary).

        Parameters
        ----------
        obs : xarray.DataArray
            Observed data (T, Y, X).
        prob_pred : xarray.DataArray
            Forecast probabilities (probability, T, Y, X).
        clim_year_start, clim_year_end : int or str
            Climatology period used for the tercile classification.
        year : int or str
            Year to display.
        """
        obs = self.compute_class(obs, clim_year_start, clim_year_end)
        obs = obs.sel(T=str(year))
        prob_pred['T'] = obs['T']
        obs, prob_pred = xr.align(obs, prob_pred)

        A = xr.apply_ufunc(
            self.calculate_rpss_,
            obs, prob_pred,
            input_core_dims=[('T',), ('probability', 'T')],
            vectorize=True,
            dask='parallelized',
            output_core_dims=[()],
            output_dtypes=['float'],
            dask_gufunc_kwargs={"allow_rechunk": True},
        )
        A_ = xr.where(A > 1 + 1e-6, 1, A)

        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        A_.plot(
            ax=ax, transform=ccrs.PlateCarree(), cmap='RdBu_r',
            vmin=-1, vmax=1 + 1e-6,
            cbar_kwargs={'label': 'RPSS', 'shrink': 0.5, 'extend': 'both',
                         'orientation': 'vertical', 'ticks': [-1, -0.5, 0, 0.5, 1]}
        )
        ax.add_feature(cfeature.BORDERS)
        ax.add_feature(cfeature.COASTLINE)
        ax.set_title(f"Ranked Probability Skill Score - {year}")
        plt.tight_layout()
        plt.show()

    def compute_one_year_rpss(self, obs, prob_pred, clim_year_start, clim_year_end, year,
                              shapefile_path=None):
        """
        Map the RPSS of a single year and report the share of skilful grid cells.

        Parameters
        ----------
        obs : xarray.DataArray
            Observed data (T, Y, X).
        prob_pred : xarray.DataArray
            Forecast probabilities (probability, T, Y, X).
        clim_year_start, clim_year_end : int or str
            Climatology period used for the tercile classification.
        year : int or str
            Year to display.
        shapefile_path : str, optional
            Boundary used to clip the map and set its extent.
        """
        obs_class = self.compute_class(obs, clim_year_start, clim_year_end)
        obs_year = obs_class.sel(T=str(year))
        prob_pred['T'] = obs_year['T']
        obs_year, prob_pred = xr.align(obs_year, prob_pred)

        rpss_array = xr.apply_ufunc(
            self.calculate_rpss_,
            obs_year, prob_pred,
            input_core_dims=[('T',), ('probability', 'T')],
            vectorize=True,
            dask='parallelized',
            output_core_dims=[()],
            output_dtypes=['float'],
            dask_gufunc_kwargs={"allow_rechunk": True},
        )
        rpss_clipped = xr.where(rpss_array > 1, 1, rpss_array)

        valid_grids = rpss_clipped.notnull().sum().values
        skillful_grids = (rpss_clipped > 0).sum().values
        skill_percentage = (skillful_grids / valid_grids * 100) if valid_grids > 0 else 0

        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())

        if shapefile_path:
            import geopandas as gpd
            import regionmask
            gdf = gpd.read_file(shapefile_path)
            mask = regionmask.mask_3D_geopandas(gdf, rpss_clipped.X, rpss_clipped.Y)
            rpss_clipped = rpss_clipped.where(mask.any(dim='region'))
            bounds = gdf.total_bounds
            ax.set_extent([bounds[0] - 0.5, bounds[2] + 0.5, bounds[1] - 0.5, bounds[3] + 0.5])

        im = rpss_clipped.plot(
            ax=ax, transform=ccrs.PlateCarree(), cmap='RdBu_r',
            vmin=-1, vmax=1, add_colorbar=False
        )

        cbar = fig.colorbar(im, ax=ax, shrink=0.6, extend='both', orientation='vertical')
        cbar.set_label('Ranked Probability Skill Score (RPSS)', fontsize=10)
        ticks = [-1, -0.5, 0, 0.5, 1]
        cbar.set_ticks(ticks)
        tick_locs = (np.array(ticks) - (-1)) / (1 - (-1))
        cbar.ax.hlines(tick_locs, 0, 1, colors='white', linewidth=1.5)

        ax.add_feature(cfeature.BORDERS, edgecolor='gray')
        ax.add_feature(cfeature.COASTLINE)
        ax.set_title(f"RPSS for {year}\nSkillful Grids (RPSS > 0): {skill_percentage:.1f}%",
                     fontsize=13, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
