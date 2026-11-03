"""Vectorised best-fit of marginal distributions (all cells at once).

``WAS_TransformData.fit_best_distribution(mode="grid")`` fits every candidate
family in every grid cell with ``scipy.stats.<family>.fit`` inside
``np.vectorize``: a numerical optimisation per family and per cell. On the
0.1 deg West-Africa grid (~45 000 cells) and the PRCP map {norm, lognorm,
gamma} that is minutes to tens of minutes per call, and the strict
cross-validator calls it once per outer training fold -- it was the dominant
cost of the probabilistic calibration.

For the families used by the CLIs the maximum-likelihood estimates are closed
form or a one-dimensional equation, which vectorises over cells:

* ``norm``    : mean and population standard deviation;
* ``lognorm`` (loc = 0): mean and population s.d. of log(x), scale = exp(mean);
* ``expon``   (loc = 0): scale = mean;
* ``gamma``   (loc = 0): shape solves log(a) - digamma(a) = log(mean) - mean(log x)
  (Minka's initialisation + Newton), scale = mean / a.

Selection rules are those of ``_fit_1d_core``: at least ``min_n`` finite
values, positive-support families only where every value is > 0, AIC with the
same parameter counts, strict ``<`` in map order. Results agree with the scipy
path to the optimiser tolerance (``tests/test_bestfit_fast.py``). Maps with
any other family (t, weibull_min, poisson, nbinom) keep the scipy path.
"""
from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.special import digamma, gammaln, polygamma

__all__ = ["FAST_FAMILIES", "fast_supported", "fit_best_distribution_fast"]

FAST_FAMILIES = {"norm", "lognorm", "expon", "gamma"}


def fast_supported(distribution_map):
    return bool(distribution_map) and set(distribution_map) <= FAST_FAMILIES


def _gamma_shape(s, iterations=50):
    """Solve log(a) - digamma(a) = s (s > 0) for every element."""
    s = np.asarray(s, dtype=float)
    a = (3.0 - s + np.sqrt((s - 3.0) ** 2 + 24.0 * s)) / (12.0 * s)  # Minka
    for _ in range(iterations):
        f = np.log(a) - digamma(a) - s
        step = f / (1.0 / a - polygamma(1, a))
        a_new = np.where(np.isfinite(step), a - step, a)
        a_new = np.where(a_new <= 0, a / 2.0, a_new)
        if np.nanmax(np.abs(a_new - a) / a) < 1e-12:
            a = a_new
            break
        a = a_new
    return a


def fit_best_distribution_fast(data, distribution_map, min_n=15):
    """``(code, shape, loc, scale)`` per cell, as ``fit_best_distribution(mode='grid')``.

    ``data`` is (T, Y, X); the fit uses the finite values of each cell.
    """
    if not fast_supported(distribution_map):
        raise ValueError(f"Fast best-fit supports {sorted(FAST_FAMILIES)} only.")
    data = data.transpose("T", ...)
    v = np.asarray(data.values, dtype=float)
    shape_cells = v.shape[1:]
    v = v.reshape(v.shape[0], -1)
    finite = np.isfinite(v)
    n = finite.sum(0).astype(float)
    ok = n >= min_n
    x = np.where(finite, v, 0.0)
    positive = ok & np.all(np.where(finite, v > 0, True), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = x.sum(0) / n
        var = (np.where(finite, (v - mean) ** 2, 0.0)).sum(0) / n
        logv = np.where(finite & (v > 0), np.log(np.where(v > 0, v, 1.0)), 0.0)
        mlog = logv.sum(0) / n
        vlog = (np.where(finite, (logv - mlog) ** 2, 0.0)).sum(0) / n
        sumlog = logv.sum(0)

    m = v.shape[1]
    best_aic = np.full(m, np.inf)
    out = {k: np.full(m, np.nan) for k in ("code", "shape", "loc", "scale")}
    for name, code in distribution_map.items():
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if name == "norm":
                valid = ok & (var > 0)
                sd = np.sqrt(var)
                logL = -0.5 * n * (np.log(2 * np.pi * var) + 1.0)
                k, shp, loc, scl = 2, np.nan, mean, sd
            elif name == "lognorm":
                valid = positive & (vlog > 0)
                sig = np.sqrt(vlog)
                logL = -sumlog - 0.5 * n * (np.log(2 * np.pi * vlog) + 1.0)
                k, shp, loc, scl = 2, sig, 0.0, np.exp(mlog)
            elif name == "expon":
                valid = positive & (mean > 0)
                logL = -n * np.log(mean) - n
                k, shp, loc, scl = 1, np.nan, 0.0, mean
            elif name == "gamma":
                sgap = np.log(mean) - mlog
                valid = positive & (sgap > 0)
                a = np.where(valid, _gamma_shape(np.where(valid, sgap, 1.0)), np.nan)
                theta = mean / a
                logL = ((a - 1.0) * sumlog - n * mean / theta - n * a * np.log(theta)
                        - n * gammaln(a))
                k, shp, loc, scl = 2, a, 0.0, theta
            else:  # pragma: no cover - guarded by fast_supported
                continue
            aic = 2 * k - 2 * logL
        better = valid & np.isfinite(aic) & (aic < best_aic)
        best_aic = np.where(better, aic, best_aic)
        out["code"] = np.where(better, float(code), out["code"])
        out["shape"] = np.where(better, shp, out["shape"])
        out["loc"] = np.where(better, loc, out["loc"])
        out["scale"] = np.where(better, scl, out["scale"])
    template = data.isel(T=0, drop=True)
    res = [xr.DataArray(out[k].reshape(shape_cells), dims=template.dims,
                        coords={d: template[d] for d in template.dims})
           for k in ("code", "shape", "loc", "scale")]
    return tuple(res)
