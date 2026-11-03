"""Generalized ordinal discrimination for discrete probability forecasts.

Mathematical formulation: Mason and Weigel (2009), equation 16, Monthly
Weather Review 137, 331–349. The authors' ``afc.mp`` routine in the CRAN
``afc`` package provides an independent reference implementation.
This is not macro one-versus-rest ROC AUC. Weigel and Mason (2011) extends
the framework to ensemble forecasts; it is not the discrete-case formula.
"""
from __future__ import annotations

import numpy as np

# Shared with fold probability scoring. Float32 storage/softmax may leave a
# few ulps in the sum; validate first, then normalize only within this bound.
PROBABILITY_SUM_ATOL = 1e-6

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        return lambda function: function


@njit(cache=True)
def _ordinal_pair_score(category, p, suffix, tolerance):
    score, pairs = 0.0, 0
    for i in range(category.size - 1):
        for j in range(i + 1, category.size):
            if category[i] == category[j]:
                continue
            difference = 0.0
            for k in range(p.shape[0]):
                difference += p[k, i] * suffix[k, j] - suffix[k, i] * p[k, j]
            if category[j] < category[i]:
                difference = -difference
            if difference > tolerance:
                score += 1.0
            elif abs(difference) <= tolerance:
                score += 0.5
            pairs += 1
    return score / pairs if pairs else np.nan


def ordinal_groc(categories, probabilities, tie_tolerance=1e-12):
    """Return the ordinal 2AFC/GROC score in [0,1].

    Parameters
    ----------
    categories : array, shape (n,)
        Ordered observed category indices 0,...,K-1. Define these with the
        same thresholds as the supplied forecasts, preferably fitted per fold.
    probabilities : array, shape (K,n)
        Forecast probabilities in increasing category order. Finite columns
        must be in [0,1] and sum to one within absolute tolerance 1e-6.
        Accepted rounding residuals are normalized in float64 before scoring.
    tie_tolerance : float
        Absolute tolerance on the difference between the two order masses.

    Missing observations or incomplete probability columns are omitted.
    Observed-category ties are excluded from the denominator. Forecast-order
    ties, including two identical point masses, receive half credit. No pair
    of distinct observed categories means an undefined score (NaN), not 0.5.
    Runtime is O(n^2*K), with O(n*K) temporary storage.
    """
    category = np.asarray(categories, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if category.ndim != 1 or p.ndim != 2 or p.shape[1] != category.size or p.shape[0] < 2:
        raise ValueError("Provide categories(n,) and probabilities(K,n), with K>=2.")
    if not np.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ValueError("tie_tolerance must be finite and nonnegative.")
    if np.isinf(category).any() or np.isinf(p).any():
        raise ValueError("Infinite observations/probabilities are invalid.")
    good = np.isfinite(category) & np.isfinite(p).all(axis=0)
    category, p = category[good], p[:, good]
    if np.any(category != np.floor(category)) or np.any((category < 0) | (category >= p.shape[0])):
        raise ValueError("Observed category indices must be integers in 0,...,K-1.")
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("Probabilities must lie in [0,1].")
    total = p.sum(axis=0)
    if not np.allclose(total, 1.0, atol=PROBABILITY_SUM_ATOL, rtol=0):
        raise ValueError("Each complete probability column must sum to one.")
    if category.size < 2:
        return np.nan
    p = p / total
    # suffix[k,t] = probability that the forecast category exceeds k.
    suffix = np.cumsum(p[::-1], axis=0)[::-1] - p
    return float(_ordinal_pair_score(category, p, suffix, tie_tolerance))


__all__ = ["ordinal_groc"]
