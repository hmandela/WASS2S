"""
Bias correction utilities for climate variables.

Two classes are provided:

- ``WAS_Qmap`` : quantile-mapping methods tailored to precipitation
  (non-negative, possibly with many dry days). Supports QUANT, RQUANT,
  SSPLIN, PTF and DIST methods, with optional wet/dry day handling.

- ``WAS_bias_correction`` : bias-correction methods for continuous variables
  such as temperature or wind speed. Supports MEAN, VARSCALE, QUANT, NORM
  DIST and additive/multiplicative QDM methods.

Both classes accept NumPy arrays (1D / 2D / 3D) and 3D ``xarray.DataArray``
inputs, and they propagate / handle NaNs.

This is a corrected version of the original implementation.
"""

import inspect
import warnings
import numpy as np
import xarray as xr
from scipy import stats
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.optimize import minimize
from scipy.stats import norm, lognorm, gamma, weibull_min


# Small constants used to keep CDF values strictly inside (0, 1) when feeding
# them to a quantile function, so that we never hit ``ppf(0) = -inf`` or
# ``ppf(1) = +inf``.
_CDF_EPS = 1e-10


def _dedupe_monotone(x, y):
    """
    Return strictly increasing ``x`` and matching ``y`` by collapsing ties.

    Quantile arrays of precipitation often contain repeated values
    (e.g. many zeros), which break ``scipy.interpolate.interp1d`` and
    ``PchipInterpolator`` (both require strictly increasing x). For each run
    of equal x values we keep the first occurrence and the corresponding y.

    Parameters
    ----------
    x, y : ndarray
        1D arrays of the same length. ``x`` must be (weakly) sorted.

    Returns
    -------
    x_u, y_u : ndarray
        Strictly increasing ``x_u`` and matching ``y_u``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0:
        return x, y
    keep = np.concatenate(([True], np.diff(x) > 0))
    return x[keep], y[keep]


def _filter_nan_pair(o, m):
    """Drop NaN entries independently from each 1D array."""
    o = np.asarray(o, dtype=float)
    m = np.asarray(m, dtype=float)
    o = o[np.isfinite(o)]
    m = m[np.isfinite(m)]
    return o, m


def _validate_bias_arrays(obs, mod, positive=False):
    """Reject mixed labelled/unlabelled inputs and impossible physical values."""
    if isinstance(obs, xr.DataArray) != isinstance(mod, xr.DataArray):
        raise TypeError("obs and mod must both be DataArrays or both be NumPy arrays.")
    if isinstance(obs, xr.DataArray):
        for data in (obs, mod):
            for dim in data.dims:
                index = data.get_index(dim)
                if not index.is_unique or index.hasnans:
                    raise ValueError(f"Bias correction coordinate {dim!r} must be finite and unique.")
    if positive:
        for value in (obs, mod):
            a = np.asarray(value, dtype=float)
            if np.any(a[np.isfinite(a)] < 0):
                raise ValueError("Precipitation quantile mapping requires non-negative values.")


def _qdm_reference_array(reference, fobj):
    """Validate the independent target-model distribution on the fitted grid."""
    if reference is None:
        raise ValueError(
            "QDM requires reference=<target-period model sample>. Supply the model "
            "distribution explicitly; an individual forecast cannot identify it."
        )
    if fobj.get("is_xarray", False):
        if not isinstance(reference, xr.DataArray):
            raise TypeError("QDM reference must be a DataArray on the fitted grid.")
        if set(reference.dims) != {"T", "Y", "X"}:
            raise ValueError("QDM reference must have dimensions T, Y, X.")
        for dim in ("Y", "X"):
            if not reference.get_index(dim).equals(fobj['coords'].indexes[dim]):
                raise ValueError(f"QDM reference coordinate {dim!r} differs from fitted grid.")
        reference = reference.transpose("T", "Y", "X")
    reference = np.asarray(reference, dtype=float)
    if reference.ndim == 1:
        reference = reference[:, None]
    elif reference.ndim == 3:
        reference = reference.reshape(reference.shape[0], -1)
    if reference.ndim != 2 or reference.shape[1] != len(fobj['par']['obs']):
        raise ValueError("QDM reference must match the fitted number of grid cells.")
    if reference.shape[0] < 2:
        raise ValueError("QDM reference requires at least two target-period model samples.")
    return reference


def _fit_qdm(obs, mod, kind="additive", denominator_floor=None, max_ratio=None):
    """Store empirical historical marginals; no target-period data are fitted."""
    kind = str(kind).lower()
    if kind not in {"additive", "multiplicative"}:
        raise ValueError("QDM kind must be 'additive' or 'multiplicative'.")
    for name, value in (("denominator_floor", denominator_floor), ("max_ratio", max_ratio)):
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be a finite positive number or None.")
    if kind == "additive" and (denominator_floor is not None or max_ratio is not None):
        raise ValueError("Ratio regularisation applies only to multiplicative QDM.")
    if kind == "multiplicative":
        _validate_bias_arrays(obs, mod, positive=True)
    par = {"obs": [], "mod": [], "kind": kind,
           "denominator_floor": denominator_floor, "max_ratio": max_ratio}
    for col in range(obs.shape[1]):
        o, m = _filter_nan_pair(obs[:, col], mod[:, col])
        par["obs"].append(np.sort(o))
        par["mod"].append(np.sort(m))
    return {"class": "fitQDM", "par": par,
            "method": "quantile_delta_mapping", "cdf_ties": "mean plotting position",
            "zero_denominator": "NaN for positive targets unless an explicit floor is set"}


def _do_qdm(x, fobj, reference=None):
    """Empirical QDM at target-model ranks, following Cannon et al. (2015).

    For p = F_target(x), additive QDM is Q_obs(p) + x - Q_hist(p),
    multiplicative QDM is Q_obs(p) * x / Q_hist(p). Finite-sample CDFs
    invert the linear empirical quantile function, averaging tied ranks.
    Tails use the boundary rank (0/1), retaining the model delta or ratio.
    Raw zero targets remain zero in multiplicative mode. A positive target
    above a zero historical model quantile is unidentified and stays NaN;
    explicit ratio regularisation changes this default only when requested.
    """
    r = _qdm_reference_array(reference, fobj)
    par = fobj['par']
    if x.shape[1] != len(par['obs']):
        raise ValueError("QDM input does not match the fitted number of grid cells.")
    ratio = par['kind'] == 'multiplicative'
    if ratio:
        _validate_bias_arrays(x, r, positive=True)
    out = np.full_like(x, np.nan, dtype=float)
    for col, (o, m) in enumerate(zip(par['obs'], par['mod'])):
        ri = np.sort(r[np.isfinite(r[:, col]), col])
        if min(len(o), len(m), len(ri)) < 2:
            continue
        valid = np.isfinite(x[:, col])
        xi = x[valid, col]
        unique, first, counts = np.unique(ri, return_index=True, return_counts=True)
        # Mean tied positions in the inverse of numpy's type-7 quantiles.
        probabilities = (first + (counts - 1) / 2) / (len(ri) - 1)
        p = np.interp(xi, unique, probabilities, left=0.0, right=1.0)
        qo = np.quantile(o, p)
        qm = np.quantile(m, p)
        if ratio:
            denominator = qm if par['denominator_floor'] is None else np.maximum(qm, par['denominator_floor'])
            delta = np.divide(xi, denominator, out=np.full_like(xi, np.nan), where=denominator > 0)
            if par['max_ratio'] is not None:
                delta = np.minimum(delta, par['max_ratio'])
            values = qo * delta
            values[xi == 0] = 0.0
        else:
            values = qo + xi - qm
        out[valid, col] = values
    return out


class WAS_Qmap:
    """
    Bias correction methods using quantile mapping techniques, adapted from
    the R ``qmap`` package.

    Provides fitting and application methods for empirical quantile mapping
    (QUANT), robust quantile mapping (RQUANT), smoothing splines (SSPLIN),
    parametric transformations (PTF) and distribution-based methods (DIST).
    Methods optionally handle wet/dry day corrections and are designed for
    precipitation or other non-negative variables.

    Notes
    -----
    - Inputs are expected to be non-negative.
    - For gridded data, computations are performed column-wise (per grid cell).
    - xarray support preserves coordinates and attributes.
    - NaNs in observed / modeled data are filtered out at fit time. NaNs in
      the data passed to ``doQmap`` propagate to NaNs in the output.
    """

    # ------------------------------------------------------------------
    # Top-level dispatch
    # ------------------------------------------------------------------

    @staticmethod
    def fitQmap(obs, mod, method, **kwargs):
        """
        Fit a bias correction model using the specified quantile mapping method.

        Parameters
        ----------
        obs, mod : array_like or xarray.DataArray
            Observed and modeled data of identical shape. Arrays may be 1D
            (time), 2D (time, grid) or 3D (T, Y, X). xarray DataArrays must
            be 3D.
        method : str
            One of ``'QUANT'``, ``'RQUANT'``, ``'SSPLIN'``, ``'PTF'``,
            ``'DIST'`` (case-insensitive).
        **kwargs
            Forwarded to the underlying ``fitQmap*`` method.

        Returns
        -------
        dict
            Fitted object with parameters, ``class`` identifier and metadata.
        """
        _validate_bias_arrays(obs, mod, positive=True)
        if 'wet_day' in kwargs and isinstance(kwargs['wet_day'], np.bool_):
            kwargs['wet_day'] = bool(kwargs['wet_day'])
        is_xarray = isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray)
        time_dim = spatial_dims = coords = attrs = original_dims = None

        if is_xarray:
            if set(obs.dims) != {"T", "Y", "X"} or set(mod.dims) != {"T", "Y", "X"}:
                raise ValueError("xarray inputs must have named dimensions T, Y, X.")
            original_dims = obs.dims
            obs = obs.transpose("T", "Y", "X")
            mod = mod.transpose("T", "Y", "X")
            obs, mod = xr.align(obs, mod, join="exact", copy=False)
            time_dim = "T"
            spatial_dims = ("Y", "X")
            obs_stacked = obs.stack(grid=spatial_dims).transpose(time_dim, 'grid')
            mod_stacked = mod.stack(grid=spatial_dims).transpose(time_dim, 'grid')
            obs_data = obs_stacked.values
            mod_data = mod_stacked.values
            coords = obs.coords
            attrs = obs.attrs
        else:
            obs_data = WAS_Qmap._to_2d(obs)
            mod_data = WAS_Qmap._to_2d(mod)
            if obs_data.shape != mod_data.shape:
                raise ValueError(
                    f"obs and mod must have matching shapes; got "
                    f"{obs_data.shape} and {mod_data.shape}"
                )

        if "qstep" in kwargs and kwargs["qstep"] is not None:
            if not 0 < float(kwargs["qstep"]) <= 1:
                raise ValueError("qstep must be in (0, 1].")
        method_up = method.upper()
        if method_up == 'QDM':
            kwargs.setdefault('kind', 'multiplicative')
            if kwargs['kind'] != 'multiplicative':
                raise ValueError("Use WAS_bias_correction for additive QDM.")
            fobj = _fit_qdm(obs_data, mod_data, **kwargs)
        elif method_up == 'QUANT':
            fobj = WAS_Qmap.fitQmapQUANT(obs_data, mod_data, **kwargs)
        elif method_up == 'RQUANT':
            fobj = WAS_Qmap.fitQmapRQUANT(obs_data, mod_data, **kwargs)
        elif method_up == 'SSPLIN':
            fobj = WAS_Qmap.fitQmapSSPLIN(obs_data, mod_data, **kwargs)
        elif method_up == 'PTF':
            fobj = WAS_Qmap.fitQmapPTF(obs_data, mod_data, **kwargs)
        elif method_up == 'DIST':
            fobj = WAS_Qmap.fitQmapDIST(obs_data, mod_data, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

        fobj['n_grid'] = obs_data.shape[1]
        if is_xarray:
            fobj['is_xarray'] = True
            fobj['time_dim'] = time_dim
            fobj['spatial_dims'] = spatial_dims
            fobj['coords'] = coords
            fobj['attrs'] = attrs
            fobj['original_dims'] = original_dims
        return fobj

    @staticmethod
    def doQmap(x, fobj, **kwargs):
        """
        Apply the fitted bias correction to new data.

        Returns the same array type as ``x`` and preserves its shape.
        """
        if fobj.get('is_xarray', False):
            if not isinstance(x, xr.DataArray):
                raise ValueError(
                    "Input x must be xarray.DataArray when the model was "
                    "fitted from a DataArray"
                )
            time_dim = fobj['time_dim']
            spatial_dims = fobj['spatial_dims']
            if set(x.dims) != {time_dim, *spatial_dims}:
                raise ValueError("Input x must have matching named dimensions.")
            for dim in spatial_dims:
                if not x.get_index(dim).equals(fobj['coords'].indexes[dim]):
                    raise ValueError(f"Input x coordinate {dim!r} differs from the fitted grid.")
            x_stacked = x.stack(grid=spatial_dims).transpose(time_dim, 'grid')
            x_data = x_stacked.values
            corrected_data = WAS_Qmap._doQmap_internal(x_data, fobj, **kwargs)
            corrected_stacked = xr.DataArray(
                corrected_data,
                dims=(time_dim, 'grid'),
                coords={
                    time_dim: x_stacked.coords[time_dim],
                    'grid': x_stacked.coords['grid'],
                },
            )
            corrected = corrected_stacked.unstack('grid').transpose(*x.dims)
            corrected.attrs = dict(fobj['attrs'])
            corrected.name = x.name
            return corrected

        # NumPy path: preserve original ndim instead of using np.squeeze,
        # which can silently collapse legitimate length-1 axes.
        x_arr = np.asarray(x, dtype=float)
        original_ndim = x_arr.ndim
        original_shape = x_arr.shape
        x_data = WAS_Qmap._to_2d(x_arr)
        corrected = WAS_Qmap._doQmap_internal(x_data, fobj, **kwargs)

        if original_ndim == 0:
            return corrected.reshape(()).item()
        if original_ndim == 1:
            return corrected.ravel()
        if original_ndim == 2:
            return corrected
        return corrected.reshape(original_shape)

    @staticmethod
    def _doQmap_internal(x, fobj, **kwargs):
        """Dispatch to the appropriate ``doQmap*`` based on fitted class."""
        if np.any(x[np.isfinite(x)] < 0):
            raise ValueError("Precipitation quantile mapping requires non-negative values.")
        if 'n_grid' in fobj and x.shape[1] != fobj['n_grid']:
            raise ValueError("Input does not match the fitted number of grid cells.")
        cls = fobj['class']
        if cls == 'fitQDM':
            return _do_qdm(x, fobj, **kwargs)
        if cls == 'fitQmapQUANT':
            return WAS_Qmap.doQmapQUANT(x, fobj, **kwargs)
        if cls == 'fitQmapRQUANT':
            return WAS_Qmap.doQmapRQUANT(x, fobj, **kwargs)
        if cls == 'fitQmapSSPLIN':
            return WAS_Qmap.doQmapSSPLIN(x, fobj, **kwargs)
        if cls == 'fitQmapPTF':
            return WAS_Qmap.doQmapPTF(x, fobj, **kwargs)
        if cls == 'fitQmapDIST':
            return WAS_Qmap.doQmapDIST(x, fobj, **kwargs)
        raise ValueError(f"Unknown class: {cls}")

    @staticmethod
    def _to_2d(arr):
        """Reshape a 0/1/2/3-D array into a 2D (time, grid) layout."""
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim == 3:
            arr = arr.reshape(arr.shape[0], -1)
        elif arr.ndim > 3:
            raise ValueError("Numpy array must be 1D, 2D, or 3D")
        return arr

    @staticmethod
    def _wet_day_threshold(obs, mod, wet_day):
        """
        Compute wet day thresholds for observations and model.

        Returns ``(th_mod, th_obs)``. ``obs`` and ``mod`` are assumed to be
        NaN-free.
        """
        if wet_day is False:
            return 0.0, 0.0
        if wet_day is True:
            if len(obs) == 0:
                return 0.0, 0.0
            p_wet = float(np.mean(obs > 0))
            th_mod = (float(np.nextafter(np.min(mod), -np.inf)) if p_wet == 1 else float(np.quantile(mod, 1 - p_wet))) if len(mod) else 0.0
            return th_mod, 0.0
        # numeric threshold
        th_obs = float(wet_day)
        if len(obs) == 0:
            return 0.0, th_obs
        p_wet = float(np.mean(obs >= th_obs))
        th_mod = (float(np.nextafter(np.min(mod), -np.inf)) if p_wet == 1 else float(np.quantile(mod, 1 - p_wet))) if len(mod) else 0.0
        return th_mod, th_obs

    # ------------------------------------------------------------------
    # QUANT
    # ------------------------------------------------------------------

    @staticmethod
    def fitQmapQUANT(obs, mod, wet_day=False, qstep=0.01, nboot=1, random_state=None):
        """Fit empirical quantile mapping with optional bootstrap of obs."""
        rng = np.random.default_rng(random_state)
        n_cols = obs.shape[1]
        n_q = int(1 / qstep) + 1
        probs = np.linspace(0, 1, n_q)

        par = {
            'modq': np.full((n_q, n_cols), np.nan),
            'fitq': np.full((n_q, n_cols), np.nan),
            'wet_day_th_mod': np.zeros(n_cols),
            'wet_day_th_obs': np.zeros(n_cols),
            'all_nan': np.zeros(n_cols, dtype=bool),
            'all_dry': np.zeros(n_cols, dtype=bool),
        }

        for col in range(n_cols):
            # FIX: filter NaNs out before computing thresholds / quantiles.
            o, m = _filter_nan_pair(obs[:, col], mod[:, col])
            par['wet_day_th_mod'][col], par['wet_day_th_obs'][col] = (
                WAS_Qmap._wet_day_threshold(o, m, wet_day)
            )
            th_obs = par['wet_day_th_obs'][col]
            par['all_dry'][col] = bool(o.size >= 2 and m.size >= 2 and np.all(o <= 0 if isinstance(wet_day, (bool, np.bool_)) else o < th_obs))
            if wet_day is True:
                o = o[o > th_obs]
            else:
                o = o[o >= th_obs]
            m = m[m > par['wet_day_th_mod'][col]] if wet_day is not False else m[m >= 0]

            if len(o) < 2 or len(m) < 2:
                par['all_nan'][col] = True
                continue

            par['modq'][:, col] = np.quantile(m, probs)
            if nboot > 1:
                boot_q = np.array([
                    np.quantile(rng.choice(o, len(o), replace=True), probs)
                    for _ in range(nboot)
                ])
                par['fitq'][:, col] = np.mean(boot_q, axis=0)
            else:
                par['fitq'][:, col] = np.quantile(o, probs)

        return {'class': 'fitQmapQUANT', 'par': par, 'wet_day': wet_day}

    @staticmethod
    def doQmapQUANT(x, fobj, type='linear'):
        """
        Apply empirical quantile mapping (QUANT).

        Three regions are handled explicitly:
          - ``x < th_mod``      -> 0 (dry)
          - ``x > modq[-1]``    -> additive shift ``x + (fitq[-1] - modq[-1])``
            (constant-correction extrapolation for extremes)
          - otherwise           -> interpolation between ``modq`` and ``fitq``
        NaN values in ``x`` are preserved as NaN in the output.
        """
        n_cols = x.shape[1]
        corrected = np.full_like(x, np.nan, dtype=float)
        par = fobj['par']
        all_nan = par.get('all_nan', np.zeros(n_cols, dtype=bool))

        for col in range(n_cols):
            if par.get('all_dry', np.zeros(n_cols, dtype=bool))[col]:
                corrected[:, col] = np.where(np.isfinite(x[:, col]), 0.0, np.nan)
                continue
            if all_nan[col]:
                continue
            xi = x[:, col].astype(float, copy=True)
            modq = par['modq'][:, col]
            fitq = par['fitq'][:, col]
            th_mod = par['wet_day_th_mod'][col]

            # FIX: dedupe to satisfy interp1d's strictly-increasing requirement.
            modq_u, fitq_u = _dedupe_monotone(modq, fitq)

            xi_out = np.full_like(xi, np.nan)

            valid = np.isfinite(xi)
            mask_dry = valid & ((xi <= th_mod) if fobj.get('wet_day', False) is not False else (xi < th_mod))
            mask_high = valid & (xi > modq[-1])
            mask_interp = valid & ~mask_dry & ~mask_high

            xi_out[mask_dry] = 0.0
            xi_out[mask_high] = xi[mask_high] + (fitq[-1] - modq[-1])

            if np.any(mask_interp):
                if modq_u.size >= 2:
                    interp_func = interp1d(
                        modq_u, fitq_u, kind=type,
                        bounds_error=False,
                        fill_value=(fitq_u[0], fitq_u[-1]),
                    )
                    xi_out[mask_interp] = interp_func(xi[mask_interp])
                else:
                    xi_out[mask_interp] = fitq_u[0] if fitq_u.size else np.nan

            corrected[:, col] = np.maximum(xi_out, 0.0)
        return corrected

    # ------------------------------------------------------------------
    # RQUANT
    # ------------------------------------------------------------------

    @staticmethod
    def fitQmapRQUANT(obs, mod, wet_day=True, qstep=0.01, nlls=10, nboot=10, random_state=None):
        """Robust quantile mapping with local linear fits."""
        rng = np.random.default_rng(random_state)
        n_cols = obs.shape[1]
        probs = np.linspace(0, 1, int(1 / qstep) + 1)
        n = len(probs)

        par = {
            'modq': np.full((n, n_cols), np.nan),
            'fitq': np.full((n, n_cols), np.nan),
            'slope_bound': np.full((2, n_cols), np.nan),
            'wet_day_th_mod': np.zeros(n_cols),
            'wet_day_th_obs': np.zeros(n_cols),
            'all_nan': np.zeros(n_cols, dtype=bool),
            'all_dry': np.zeros(n_cols, dtype=bool),
        }

        for col in range(n_cols):
            o, m = _filter_nan_pair(obs[:, col], mod[:, col])
            par['wet_day_th_mod'][col], par['wet_day_th_obs'][col] = (
                WAS_Qmap._wet_day_threshold(o, m, wet_day)
            )
            th_obs = par['wet_day_th_obs'][col]
            par['all_dry'][col] = bool(o.size >= 2 and m.size >= 2 and np.all(o <= 0 if isinstance(wet_day, (bool, np.bool_)) else o < th_obs))
            if wet_day is True:
                o = o[o > th_obs]
            else:
                o = o[o >= th_obs]
            m = m[m > par['wet_day_th_mod'][col]] if wet_day is not False else m[m >= 0]

            if len(o) < 2 or len(m) < 2:
                par['all_nan'][col] = True
                continue

            par['modq'][:, col] = np.quantile(m, probs)
            if nboot > 1:
                boot_q = np.array([
                    np.quantile(rng.choice(o, len(o), replace=True), probs)
                    for _ in range(nboot)
                ])
                obsq = np.mean(boot_q, axis=0)
            else:
                obsq = np.quantile(o, probs)

            fitq = np.zeros(n)
            for i in range(n):
                start = max(0, i - nlls // 2)
                end = min(n, i + nlls // 2 + 1)
                X = par['modq'][start:end, col]
                y = obsq[start:end]
                # Need unique X values for polyfit to be meaningful.
                X_u, y_u = _dedupe_monotone(X, y)
                if X_u.size < 2:
                    fitq[i] = y.mean() if len(y) > 0 else obsq[i]
                else:
                    slope, intercept = np.polyfit(X_u, y_u, 1)
                    fitq[i] = slope * par['modq'][i, col] + intercept
            fitq = np.maximum.accumulate(np.maximum(fitq, 0.0))
            par['fitq'][:, col] = fitq

            # FIX: estimate boundary slopes from the first/last `nlls` points
            # rather than only 2, so the slope is consistent with the local
            # fits used everywhere else.
            k = min(nlls, n)
            low_X, low_y = _dedupe_monotone(par['modq'][:k, col], fitq[:k])
            high_X, high_y = _dedupe_monotone(par['modq'][-k:, col], fitq[-k:])
            par['slope_bound'][0, col] = (
                np.polyfit(low_X, low_y, 1)[0] if low_X.size >= 2 else 1.0
            )
            par['slope_bound'][1, col] = (
                np.polyfit(high_X, high_y, 1)[0] if high_X.size >= 2 else 1.0
            )

        return {'class': 'fitQmapRQUANT', 'par': par, 'wet_day': wet_day}

    @staticmethod
    def doQmapRQUANT(x, fobj, type='linear', slope_bound=(0.0, np.inf)):
        """Apply RQUANT correction with slope-bounded extrapolation."""
        n_cols = x.shape[1]
        corrected = np.full_like(x, np.nan, dtype=float)
        par = fobj['par']
        all_nan = par.get('all_nan', np.zeros(n_cols, dtype=bool))

        for col in range(n_cols):
            if par.get('all_dry', np.zeros(n_cols, dtype=bool))[col]:
                corrected[:, col] = np.where(np.isfinite(x[:, col]), 0.0, np.nan)
                continue
            if all_nan[col]:
                continue
            xi = x[:, col].astype(float, copy=True)
            modq = par['modq'][:, col]
            fitq = par['fitq'][:, col]
            th_mod = par['wet_day_th_mod'][col]
            low_slope = float(np.clip(
                par['slope_bound'][0, col], slope_bound[0], slope_bound[1]
            ))
            high_slope = float(np.clip(
                par['slope_bound'][1, col], slope_bound[0], slope_bound[1]
            ))

            modq_u, fitq_u = _dedupe_monotone(modq, fitq)

            xi_out = np.full_like(xi, np.nan)
            valid = np.isfinite(xi)
            mask_dry = valid & ((xi <= th_mod) if fobj.get('wet_day', False) is not False else (xi < th_mod))
            mask_low = valid & (xi < modq[0]) & ~mask_dry
            mask_high = valid & (xi > modq[-1])
            mask_interp = valid & ~mask_dry & ~mask_low & ~mask_high

            xi_out[mask_dry] = 0.0
            xi_out[mask_low] = fitq[0] + low_slope * (xi[mask_low] - modq[0])
            xi_out[mask_high] = fitq[-1] + high_slope * (xi[mask_high] - modq[-1])

            if np.any(mask_interp) and modq_u.size >= 2:
                interp_func = interp1d(
                    modq_u, fitq_u, kind=type,
                    bounds_error=False, fill_value='extrapolate',
                )
                xi_out[mask_interp] = interp_func(xi[mask_interp])
            elif np.any(mask_interp) and modq_u.size == 1:
                xi_out[mask_interp] = fitq_u[0]

            corrected[:, col] = np.maximum(xi_out, 0.0)
        return corrected

    # ------------------------------------------------------------------
    # SSPLIN
    # ------------------------------------------------------------------

    @staticmethod
    def fitQmapSSPLIN(obs, mod, wet_day=False, qstep=0.01):
        """Fit monotone PCHIP quantile interpolation (legacy SSPLIN name; no smoothing penalty)."""
        n_cols = obs.shape[1]
        n_q = int(1 / qstep) + 1
        probs = np.linspace(0, 1, n_q)

        par = {
            'modq': np.full((n_q, n_cols), np.nan),
            'fitq': np.full((n_q, n_cols), np.nan),
            'wet_day_th_mod': np.zeros(n_cols),
            'wet_day_th_obs': np.zeros(n_cols),
            'all_nan': np.zeros(n_cols, dtype=bool),
            'all_dry': np.zeros(n_cols, dtype=bool),
        }
        for col in range(n_cols):
            o, m = _filter_nan_pair(obs[:, col], mod[:, col])
            par['wet_day_th_mod'][col], par['wet_day_th_obs'][col] = (
                WAS_Qmap._wet_day_threshold(o, m, wet_day)
            )
            th_obs = par['wet_day_th_obs'][col]
            par['all_dry'][col] = bool(o.size >= 2 and m.size >= 2 and np.all(o <= 0 if isinstance(wet_day, (bool, np.bool_)) else o < th_obs))
            if wet_day is True:
                o = o[o > th_obs]
            else:
                o = o[o >= th_obs]
            m = m[m > par['wet_day_th_mod'][col]] if wet_day is not False else m[m >= 0]

            if len(o) < 2 or len(m) < 2:
                par['all_nan'][col] = True
                continue

            par['modq'][:, col] = np.quantile(m, probs)
            par['fitq'][:, col] = np.quantile(o, probs)
        return {'class': 'fitQmapSSPLIN', 'par': par, 'wet_day': wet_day}

    @staticmethod
    def doQmapSSPLIN(x, fobj):
        """Apply PCHIP-based quantile mapping correction."""
        n_cols = x.shape[1]
        corrected = np.full_like(x, np.nan, dtype=float)
        par = fobj['par']
        all_nan = par.get('all_nan', np.zeros(n_cols, dtype=bool))

        for col in range(n_cols):
            if par.get('all_dry', np.zeros(n_cols, dtype=bool))[col]:
                corrected[:, col] = np.where(np.isfinite(x[:, col]), 0.0, np.nan)
                continue
            if all_nan[col]:
                continue
            xi = x[:, col].astype(float, copy=True)
            modq = par['modq'][:, col]
            fitq = par['fitq'][:, col]
            th_mod = par['wet_day_th_mod'][col]

            # FIX: PchipInterpolator requires strictly increasing x.
            modq_u, fitq_u = _dedupe_monotone(modq, fitq)

            xi_out = np.full_like(xi, np.nan)
            valid = np.isfinite(xi)
            mask_dry = valid & ((xi <= th_mod) if fobj.get('wet_day', False) is not False else (xi < th_mod))
            mask_interp = valid & ~mask_dry

            xi_out[mask_dry] = 0.0
            if np.any(mask_interp):
                if modq_u.size >= 2:
                    interp_func = PchipInterpolator(modq_u, fitq_u, extrapolate=True)
                    xi_out[mask_interp] = interp_func(np.clip(xi[mask_interp], modq_u[0], modq_u[-1]))
                    high = mask_interp & (xi > modq_u[-1])
                    xi_out[high] = xi[high] + fitq_u[-1] - modq_u[-1]
                else:
                    xi_out[mask_interp] = fitq_u[0] if fitq_u.size else np.nan
            corrected[:, col] = np.maximum(xi_out, 0.0)
        return corrected

    # ------------------------------------------------------------------
    # PTF
    # ------------------------------------------------------------------

    # Built-in transformation functions.
    _PTF_FUNCS = {
        'power':         lambda x, a, b:           a * np.power(x, b),
        'power.x0':      lambda x, a, b, x0:       a * np.power(np.maximum(x - x0, 0.0), b),
        'expasympt':     lambda x, a, b, tau:      (a + b * x) * (1 - np.exp(-x / tau)),
        'expasympt.x0':  lambda x, a, b, x0, tau:  (a + b * (x - x0)) * (1 - np.exp(-(x - x0) / tau)),
        'scale':         lambda x, b:              b * x,
        'linear':        lambda x, a, b:           a + b * x,
    }

    @staticmethod
    def _ptf_n_params(tf):
        """Number of free parameters of a transformation function."""
        sig = inspect.signature(tf)
        params = [
            p for p in sig.parameters.values()
            if p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        # The first positional argument is x.
        return max(0, len(params) - 1)

    @staticmethod
    def _ptf_default_parini(transfun, mq, oq):
        """
        Sensible per-function initial parameters.

        Identity-like starting points are far more robust than ``[1, 1, ...]``,
        which can drive ``(x - x0) ** b`` complex or ``exp(-x / tau)`` to
        underflow.
        """
        tau0 = max(float(np.nanmean(mq)) if mq.size else 1.0, 1.0)
        defaults = {
            'power':        [1.0, 1.0],
            'power.x0':     [1.0, 1.0, 0.0],
            'expasympt':    [0.0, 1.0, tau0],
            'expasympt.x0': [0.0, 1.0, 0.0, tau0],
            'scale':        [1.0],
            'linear':       [0.0, 1.0],
        }
        return defaults.get(transfun)

    @staticmethod
    def fitQmapPTF(obs, mod, transfun='power', parini=None, cost='RSS',
                   wet_day=False, qstep=None):
        """Fit a parametric transformation function (PTF)."""
        n_cols = obs.shape[1]
        par = {
            'transfun': transfun,
            'par': [],
            'wet_day_th_mod': np.zeros(n_cols),
            'wet_day_th_obs': np.zeros(n_cols),
            'all_nan': np.zeros(n_cols, dtype=bool),
            'all_dry': np.zeros(n_cols, dtype=bool),
        }

        if callable(transfun):
            tf = transfun
        else:
            tf = WAS_Qmap._PTF_FUNCS.get(transfun)
            if tf is None:
                raise ValueError(f"Unknown transfun: {transfun}")
        n_params = WAS_Qmap._ptf_n_params(tf)

        if cost not in ('RSS', 'MAE'):
            raise ValueError(f"Unknown cost: {cost}")

        for col in range(n_cols):
            o, m = _filter_nan_pair(obs[:, col], mod[:, col])
            par['wet_day_th_mod'][col], par['wet_day_th_obs'][col] = (
                WAS_Qmap._wet_day_threshold(o, m, wet_day)
            )
            th_obs = par['wet_day_th_obs'][col]
            par['all_dry'][col] = bool(o.size >= 2 and m.size >= 2 and np.all(o <= 0 if isinstance(wet_day, (bool, np.bool_)) else o < th_obs))
            if wet_day is True:
                o = o[o > th_obs]
            else:
                o = o[o >= th_obs]
            m = m[m > par['wet_day_th_mod'][col]] if wet_day is not False else m[m >= 0]

            if len(o) < 2 or len(m) < 2:
                par['par'].append(np.array([0.0] * n_params))
                par['all_nan'][col] = True
                continue

            if qstep is not None:
                probs = np.linspace(0, 1, int(1 / qstep) + 1)
            else:
                probs = np.linspace(0, 1, min(len(m), len(o)))
            mq = np.quantile(m, probs)
            oq = np.quantile(o, probs)

            if parini is None:
                p0 = WAS_Qmap._ptf_default_parini(transfun, mq, oq) \
                     if not callable(transfun) else None
                if p0 is None:
                    p0 = [1.0] * n_params
            else:
                p0 = list(parini)

            def objective(p, mq=mq, oq=oq, tf=tf, cost=cost):
                with np.errstate(all='ignore'):
                    pred = tf(mq, *p)
                if not np.all(np.isfinite(pred)):
                    return 1e30
                if cost == 'RSS':
                    return float(np.sum((oq - pred) ** 2))
                return float(np.sum(np.abs(oq - pred)))

            res = minimize(objective, p0, method='Nelder-Mead')
            with np.errstate(all="ignore"):
                fitted_curve = np.asarray(tf(mq, *res.x), float)
            tolerance = 1e-10 * max(1.0, float(np.max(np.abs(oq))))
            if (not res.success or not np.isfinite(fitted_curve).all()
                    or np.any(np.diff(fitted_curve) < -tolerance)):
                par['all_nan'][col] = True
                warnings.warn(f"PTF fit failed or was non-monotone at grid column {col}.", RuntimeWarning)
            par['par'].append(np.asarray(res.x))

        return {'class': 'fitQmapPTF', 'par': par, 'wet_day': wet_day}

    @staticmethod
    def doQmapPTF(x, fobj):
        """Apply parametric transformation correction."""
        n_cols = x.shape[1]
        corrected = np.full_like(x, np.nan, dtype=float)
        par = fobj['par']
        all_nan = par.get('all_nan', np.zeros(n_cols, dtype=bool))
        transfun = par['transfun']
        if callable(transfun):
            tf = transfun
        else:
            tf = WAS_Qmap._PTF_FUNCS.get(transfun)
            if tf is None:
                raise ValueError(f"Unknown transfun: {transfun}")

        for col in range(n_cols):
            if par.get('all_dry', np.zeros(n_cols, dtype=bool))[col]:
                corrected[:, col] = np.where(np.isfinite(x[:, col]), 0.0, np.nan)
                continue
            if all_nan[col]:
                continue
            xi = x[:, col].astype(float, copy=True)
            params = par['par'][col]
            th_mod = par['wet_day_th_mod'][col]

            xi_out = np.full_like(xi, np.nan)
            valid = np.isfinite(xi)
            mask_dry = valid & ((xi <= th_mod) if fobj.get('wet_day', False) is not False else (xi < th_mod))
            mask_wet = valid & ~mask_dry

            xi_out[mask_dry] = 0.0
            if np.any(mask_wet):
                with np.errstate(all='ignore'):
                    vals = tf(xi[mask_wet], *params)
                vals = np.where(np.isfinite(vals), vals, np.nan)
                xi_out[mask_wet] = vals
            corrected[:, col] = np.maximum(xi_out, 0.0)
        return corrected

    # ------------------------------------------------------------------
    # DIST
    # ------------------------------------------------------------------

    _DIST_MAP_QMAP = {
        'berngamma':   stats.gamma,
        'bernexp':     stats.expon,
        'bernlnorm':   stats.lognorm,
        'bernweibull': stats.weibull_min,
    }

    @staticmethod
    def fitQmapDIST(obs, mod, distr='berngamma', qstep=None, **kwargs):
        """
        Fit distribution-based quantile mapping.

        Uses a Bernoulli for wet/dry occurrence plus a continuous distribution
        for the positive (wet) values.
        """
        n_cols = obs.shape[1]
        par = {'par_o': [], 'par_m': [], 'tfun': [], 'distr': distr,
               'all_nan': np.zeros(n_cols, dtype=bool),
               'all_dry': np.zeros(n_cols, dtype=bool)}

        dist = WAS_Qmap._DIST_MAP_QMAP.get(distr)
        if dist is None:
            raise ValueError(f"Unknown distr: {distr}")

        for col in range(n_cols):
            o, m = _filter_nan_pair(obs[:, col], mod[:, col])

            par['all_dry'][col] = bool(o.size >= 2 and m.size >= 2 and np.all(o == 0))
            # FIX: wet-day probabilities must come from the RAW data,
            # never from a reduced quantile sample.
            p_o = float(np.mean(o > 0)) if o.size else 0.0
            p_m = float(np.mean(m > 0)) if m.size else 0.0

            # Optional: reduce data to quantiles for cheaper distribution
            # fitting. We still always restrict to positive values when
            # fitting the wet part.
            if qstep is not None and o.size and m.size:
                probs = np.linspace(0, 1, int(1 / qstep) + 1)
                o_fit_pool = np.quantile(o, probs)
                m_fit_pool = np.quantile(m, probs)
            else:
                o_fit_pool = o
                m_fit_pool = m

            o_pos = o_fit_pool[o_fit_pool > 0]
            m_pos = m_fit_pool[m_fit_pool > 0]

            par_o = {'prob': p_o, 'params': None}
            par_m = {'prob': p_m, 'params': None}

            try:
                if p_o > 0 and len(o_pos) >= 2:
                    if distr == 'bernlnorm':
                        par_o['params'] = dist.fit(o_pos, floc=0)
                    else:
                        par_o['params'] = dist.fit(o_pos, floc=0)
                else:
                    par_o['prob'] = 0.0
            except Exception:
                par_o['prob'] = 0.0
                par_o['params'] = None

            try:
                if p_m > 0 and len(m_pos) >= 2:
                    if distr == 'bernlnorm':
                        par_m['params'] = dist.fit(m_pos, floc=0)
                    else:
                        par_m['params'] = dist.fit(m_pos, floc=0)
                else:
                    par_m['prob'] = 0.0
            except Exception:
                par_m['prob'] = 0.0
                par_m['params'] = None

            par['par_o'].append(par_o)
            par['par_m'].append(par_m)

            if par_m['params'] is None or par_o['params'] is None:
                par['all_nan'][col] = True

            par['tfun'].append(
                WAS_Qmap._make_dist_tfun(par_o, par_m, dist)
            )

        return {'class': 'fitQmapDIST', 'par': par}

    @staticmethod
    def _make_dist_tfun(par_o, par_m, dist):
        """
        Build a closure that maps modeled values to observed quantiles.

        Defaults are used to bind the current loop values (avoids the classic
        late-binding trap).
        """
        def tfun(val, par_o=par_o, par_m=par_m, dist=dist):
            # FIX: cast to float so np.zeros_like / arithmetic behave well
            # even when callers pass integer arrays.
            val = np.asarray(val, dtype=float)
            out = np.full_like(val, np.nan)

            if par_m['prob'] == 0 or par_m['params'] is None:
                return out
            if par_o['prob'] == 0 or par_o['params'] is None:
                return out

            valid = np.isfinite(val)
            cdf_m = np.zeros_like(val)
            cdf_m[valid] = (
                (1 - par_m['prob'])
                + par_m['prob'] * dist.cdf(val[valid], *par_m['params'])
            )
            # FIX: keep CDFs strictly inside (0, 1) so ppf never returns inf.
            cdf_m = np.clip(cdf_m, _CDF_EPS, 1 - _CDF_EPS)

            mask_wet = valid & (cdf_m > (1 - par_o['prob']))
            out[valid & ~mask_wet] = 0.0
            if np.any(mask_wet):
                u = (cdf_m[mask_wet] - (1 - par_o['prob'])) / par_o['prob']
                u = np.clip(u, _CDF_EPS, 1 - _CDF_EPS)
                out[mask_wet] = dist.ppf(u, *par_o['params'])
            return out

        return tfun

    @staticmethod
    def doQmapDIST(x, fobj):
        """Apply distribution-based bias correction."""
        n_cols = x.shape[1]
        corrected = np.full_like(x, np.nan, dtype=float)
        par = fobj['par']
        all_nan = par.get('all_nan', np.zeros(n_cols, dtype=bool))
        for col in range(n_cols):
            if par.get('all_dry', np.zeros(n_cols, dtype=bool))[col]:
                corrected[:, col] = np.where(np.isfinite(x[:, col]), 0.0, np.nan)
                continue
            if all_nan[col]:
                continue
            tfun = par['tfun'][col]
            corrected[:, col] = tfun(x[:, col])
        return corrected

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def evaluate_bias_correction(
        obs, mod, corrected,
        wet_threshold=0.1,
        extreme_quantiles=(0.95, 0.99),
        time_dim=None,
    ):
        """
        Evaluate bias correction performance.

        Returns a 2-tuple in both code paths so callers can unpack consistently.

        - xarray inputs  -> ``(ds_dry_wet, ds_extreme)`` of xarray Datasets.
        - numpy inputs   -> ``(dict_dry_wet, dict_extreme)``.

        Parameters
        ----------
        time_dim : str, optional
            Name of the time dimension for xarray inputs. If ``None`` it is
            taken from ``obs.dims[0]`` so any naming convention works.
        """
        extreme_quantiles = list(extreme_quantiles)
        is_xarray = (
            isinstance(obs, xr.DataArray)
            and isinstance(mod, xr.DataArray)
            and isinstance(corrected, xr.DataArray)
        )

        if is_xarray:
            # FIX: auto-detect the time dimension; do NOT hard-code 'T'.
            if time_dim is None:
                time_dim = obs.dims[0]

            dry_obs = (obs <= wet_threshold).mean(dim=time_dim)
            dry_mod = (mod <= wet_threshold).mean(dim=time_dim)
            dry_corr = (corrected <= wet_threshold).mean(dim=time_dim)

            wet_obs = (obs > wet_threshold).mean(dim=time_dim)
            wet_mod = (mod > wet_threshold).mean(dim=time_dim)
            wet_corr = (corrected > wet_threshold).mean(dim=time_dim)

            mean_wet_obs = obs.where(obs > wet_threshold).mean(dim=time_dim)
            mean_wet_mod = mod.where(mod > wet_threshold).mean(dim=time_dim)
            mean_wet_corr = corrected.where(corrected > wet_threshold).mean(dim=time_dim)

            ext_obs = obs.quantile(extreme_quantiles, dim=time_dim)
            ext_mod = mod.quantile(extreme_quantiles, dim=time_dim)
            ext_corr = corrected.quantile(extreme_quantiles, dim=time_dim)

            ds_dry_wet = xr.Dataset({
                'dry_fraction_obs': dry_obs,
                'dry_fraction_mod': dry_mod,
                'dry_fraction_corr': dry_corr,
                'wet_fraction_obs': wet_obs,
                'wet_fraction_mod': wet_mod,
                'wet_fraction_corr': wet_corr,
                'mean_wet_obs': mean_wet_obs,
                'mean_wet_mod': mean_wet_mod,
                'mean_wet_corr': mean_wet_corr,
            })
            ds_extreme = xr.Dataset({
                'extreme_quantiles_obs': ext_obs,
                'extreme_quantiles_mod': ext_mod,
                'extreme_quantiles_corr': ext_corr,
            })
            return ds_dry_wet, ds_extreme

        # NumPy path.
        obs_a = np.asarray(obs).ravel()
        mod_a = np.asarray(mod).ravel()
        corr_a = np.asarray(corrected).ravel()

        def _mean_wet(a):
            mask = a > wet_threshold
            return float(np.mean(a[mask])) if np.any(mask) else np.nan

        dry_wet = {
            'dry_fraction_obs':   float(np.mean(obs_a <= wet_threshold)),
            'dry_fraction_mod':   float(np.mean(mod_a <= wet_threshold)),
            'dry_fraction_corr':  float(np.mean(corr_a <= wet_threshold)),
            'wet_fraction_obs':   float(np.mean(obs_a > wet_threshold)),
            'wet_fraction_mod':   float(np.mean(mod_a > wet_threshold)),
            'wet_fraction_corr':  float(np.mean(corr_a > wet_threshold)),
            'mean_wet_obs':       _mean_wet(obs_a),
            'mean_wet_mod':       _mean_wet(mod_a),
            'mean_wet_corr':      _mean_wet(corr_a),
        }
        extreme = {
            'extreme_quantiles_obs':  np.quantile(obs_a, extreme_quantiles),
            'extreme_quantiles_mod':  np.quantile(mod_a, extreme_quantiles),
            'extreme_quantiles_corr': np.quantile(corr_a, extreme_quantiles),
            'extreme_quantiles':      np.asarray(extreme_quantiles),
        }
        return dry_wet, extreme

    # ------------------------------------------------------------------
    # Plotting helpers (matplotlib / cartopy imported lazily)
    # ------------------------------------------------------------------

    @staticmethod
    def _add_basemap(ax, extent=None):
        # Lazy imports keep matplotlib/cartopy optional at module load time.
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.LAKES, linewidth=0.3, edgecolor='k', facecolor='none')
        if extent is not None:
            ax.set_extent(extent, crs=ccrs.PlateCarree())

    @staticmethod
    def _collect_vars(ds, prefix):
        return [v for v in ds.data_vars if v.startswith(prefix)]

    @staticmethod
    def _compute_common_limits(dataarrays, robust=True):
        """Compute shared vmin/vmax across a list of DataArrays."""
        vals = []
        for da in dataarrays:
            arr = np.asarray(da.values).ravel()
            arr = arr[np.isfinite(arr)]
            if arr.size > 0:
                vals.append(arr)
        if not vals:
            return None, None
        vals = np.concatenate(vals)
        if robust:
            return float(np.nanpercentile(vals, 2)), float(np.nanpercentile(vals, 98))
        return float(np.nanmin(vals)), float(np.nanmax(vals))

    @staticmethod
    def plot_fraction_group(ds, group_prefix, extent=None, robust=True, cmap='viridis'):
        """Plot dry/wet fraction variables with one shared colorbar."""
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs

        vars_ = WAS_Qmap._collect_vars(ds, group_prefix)
        if not vars_:
            print(f"No variables found for prefix '{group_prefix}'")
            return

        das = [ds[name] for name in vars_]
        vmin, vmax = WAS_Qmap._compute_common_limits(das, robust=robust)

        n = len(vars_)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            subplot_kw={'projection': ccrs.PlateCarree()},
            figsize=(5 * ncols, 4 * nrows), squeeze=False,
        )
        mappable = None
        i = 0
        for i, name in enumerate(vars_):
            r, c = divmod(i, ncols)
            ax = axes[r][c]
            mappable = ds[name].plot(
                ax=ax, transform=ccrs.PlateCarree(), cmap=cmap,
                vmin=vmin, vmax=vmax, add_colorbar=False,
            )
            WAS_Qmap._add_basemap(ax, extent)
            ax.set_title(name)
        for j in range(i + 1, nrows * ncols):
            r, c = divmod(j, ncols)
            axes[r][c].axis('off')

        cbar = fig.colorbar(mappable, ax=axes, orientation='vertical',
                            fraction=0.025, pad=0.02)
        cbar.set_label(group_prefix)
        fig.suptitle(f"{group_prefix} variables", fontsize=14)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_mean_wet_group(ds, extent=None, robust=True, cmap='viridis'):
        """Plot mean_wet_* variables with one shared colorbar."""
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs

        vars_ = WAS_Qmap._collect_vars(ds, 'mean_wet_')
        if not vars_:
            print("No 'mean_wet_' variables found")
            return

        das = [ds[name] for name in vars_]
        vmin, vmax = WAS_Qmap._compute_common_limits(das, robust=robust)

        n = len(vars_)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            subplot_kw={'projection': ccrs.PlateCarree()},
            figsize=(5 * ncols, 4 * nrows), squeeze=False,
        )
        mappable = None
        i = 0
        for i, name in enumerate(vars_):
            r, c = divmod(i, ncols)
            ax = axes[r][c]
            mappable = ds[name].plot(
                ax=ax, transform=ccrs.PlateCarree(), cmap=cmap,
                vmin=vmin, vmax=vmax, add_colorbar=False,
            )
            WAS_Qmap._add_basemap(ax, extent)
            ax.set_title(name)
        for j in range(i + 1, nrows * ncols):
            r, c = divmod(j, ncols)
            axes[r][c].axis('off')

        cbar = fig.colorbar(mappable, ax=axes, orientation='vertical',
                            fraction=0.025, pad=0.02)
        cbar.set_label('mean_wet')
        fig.suptitle('mean_wet_* variables', fontsize=14)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_extreme_quantiles_group(ds, extent=None, robust=True, cmap='viridis'):
        """Plot extreme_quantiles_* variables faceted by quantile."""
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs

        vars_ = WAS_Qmap._collect_vars(ds, 'extreme_quantiles_')
        if not vars_:
            print("No 'extreme_quantiles_' variables found")
            return

        qvals = ds.coords.get('quantile')
        if qvals is None:
            raise ValueError(
                "Dataset has no 'quantile' coordinate required for "
                "extreme_quantiles_* variables."
            )

        das = [ds[var].sel(quantile=q) for var in vars_ for q in qvals.values]
        vmin, vmax = WAS_Qmap._compute_common_limits(das, robust=robust)

        nrows = len(qvals)
        ncols = len(vars_)

        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            subplot_kw={'projection': ccrs.PlateCarree()},
            figsize=(5 * ncols, 4 * nrows), squeeze=False,
        )
        mappable = None
        for c, var in enumerate(vars_):
            da = ds[var]
            for r, q in enumerate(qvals.values):
                ax = axes[r][c]
                mappable = da.sel(quantile=q).plot(
                    ax=ax, transform=ccrs.PlateCarree(), cmap=cmap,
                    vmin=vmin, vmax=vmax, add_colorbar=False,
                )
                WAS_Qmap._add_basemap(ax, extent)
                ax.set_title(f"{var} – q={q}")

        cbar = fig.colorbar(mappable, ax=axes, orientation='vertical',
                            fraction=0.025, pad=0.02)
        cbar.set_label('extreme quantiles')
        fig.suptitle('extreme_quantiles_* by quantile', fontsize=14)
        plt.tight_layout()
        plt.show()


class WAS_bias_correction:
    """
    Bias correction methods for continuous climate variables (e.g. temperature,
    wind speed).

    Supports mean adjustment, variance scaling, empirical quantile mapping and
    parametric distribution mapping. Inputs may include negative values; for
    strictly positive skewed data such as wind speed, use ``'lognormal'``,
    ``'gamma'`` or ``'weibull'`` distributions.

    Notes
    -----
    - NaNs in observed / modeled data are filtered out at fit time. NaNs in
      ``doBC`` input propagate to NaNs in the output.
    - Grid cells with fewer than 2 valid points in obs or mod are flagged
      (``all_nan``) and yield NaN output during application.
    """

    # ------------------------------------------------------------------
    # Top-level dispatch
    # ------------------------------------------------------------------

    @staticmethod
    def fitBC(obs, mod, method, **kwargs):
        """Fit a bias correction model using the specified method."""
        _validate_bias_arrays(obs, mod)
        is_xarray = isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray)
        time_dim = spatial_dims = coords = attrs = original_dims = None

        if is_xarray:
            if set(obs.dims) != {"T", "Y", "X"} or set(mod.dims) != {"T", "Y", "X"}:
                raise ValueError("xarray inputs must have named dimensions T, Y, X.")
            original_dims = obs.dims
            obs = obs.transpose("T", "Y", "X")
            mod = mod.transpose("T", "Y", "X")
            obs, mod = xr.align(obs, mod, join="exact", copy=False)
            time_dim = "T"
            spatial_dims = ("Y", "X")
            obs_stacked = obs.stack(grid=spatial_dims).transpose(time_dim, 'grid')
            mod_stacked = mod.stack(grid=spatial_dims).transpose(time_dim, 'grid')
            obs_data = obs_stacked.values
            mod_data = mod_stacked.values
            coords = obs.coords
            attrs = obs.attrs
        else:
            obs_data = WAS_bias_correction._to_2d(obs)
            mod_data = WAS_bias_correction._to_2d(mod)
            if obs_data.shape != mod_data.shape:
                raise ValueError(
                    f"obs and mod must have matching shapes; got "
                    f"{obs_data.shape} and {mod_data.shape}"
                )

        if "qstep" in kwargs and kwargs["qstep"] is not None:
            if not 0 < float(kwargs["qstep"]) <= 1:
                raise ValueError("qstep must be in (0, 1].")
        method_up = method.upper()
        if method_up == 'QDM':
            fobj = WAS_bias_correction.fitQDM(obs_data, mod_data, **kwargs)
        elif method_up == 'MEAN':
            fobj = WAS_bias_correction.fitMean(obs_data, mod_data, **kwargs)
        elif method_up == 'VARSCALE':
            fobj = WAS_bias_correction.fitVarscale(obs_data, mod_data, **kwargs)
        elif method_up == 'QUANT':
            fobj = WAS_bias_correction.fitQuant(obs_data, mod_data, **kwargs)
        elif method_up == 'NORM':
            fobj = WAS_bias_correction.fitDist(obs_data, mod_data,
                                               distr='normal', **kwargs)
        elif method_up == 'DIST':
            fobj = WAS_bias_correction.fitDist(obs_data, mod_data, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

        fobj['n_grid'] = obs_data.shape[1]
        if is_xarray:
            fobj['is_xarray'] = True
            fobj['time_dim'] = time_dim
            fobj['spatial_dims'] = spatial_dims
            fobj['coords'] = coords
            fobj['attrs'] = attrs
            fobj['original_dims'] = original_dims
        return fobj

    @staticmethod
    def doBC(x, fobj, **kwargs):
        """Apply the fitted bias correction to new data."""
        if fobj.get('is_xarray', False):
            if not isinstance(x, xr.DataArray):
                raise ValueError(
                    "Input x must be xarray.DataArray when fitted with DataArray"
                )
            time_dim = fobj['time_dim']
            spatial_dims = fobj['spatial_dims']
            if set(x.dims) != {time_dim, *spatial_dims}:
                raise ValueError("Input x must have matching named dimensions.")
            for dim in spatial_dims:
                if not x.get_index(dim).equals(fobj['coords'].indexes[dim]):
                    raise ValueError(f"Input x coordinate {dim!r} differs from the fitted grid.")
            x_stacked = x.stack(grid=spatial_dims).transpose(time_dim, 'grid')
            x_data = x_stacked.values
            corrected_data = WAS_bias_correction._doBC_internal(x_data, fobj, **kwargs)
            corrected_stacked = xr.DataArray(
                corrected_data,
                dims=(time_dim, 'grid'),
                coords={
                    time_dim: x_stacked.coords[time_dim],
                    'grid': x_stacked.coords['grid'],
                },
            )
            corrected = corrected_stacked.unstack('grid').transpose(*x.dims)
            corrected.attrs = dict(fobj['attrs'])
            corrected.name = x.name
            return corrected

        x_arr = np.asarray(x, dtype=float)
        original_ndim = x_arr.ndim
        original_shape = x_arr.shape
        x_data = WAS_bias_correction._to_2d(x_arr)
        corrected = WAS_bias_correction._doBC_internal(x_data, fobj, **kwargs)

        if original_ndim == 0:
            return corrected.reshape(()).item()
        if original_ndim == 1:
            return corrected.ravel()
        if original_ndim == 2:
            return corrected
        return corrected.reshape(original_shape)

    @staticmethod
    def _doBC_internal(x, fobj, **kwargs):
        if 'n_grid' in fobj and x.shape[1] != fobj['n_grid']:
            raise ValueError("Input does not match the fitted number of grid cells.")
        cls = fobj['class']
        if cls == 'fitQDM':
            return WAS_bias_correction.doQDM(x, fobj, **kwargs)
        if cls == 'fitMean':
            return WAS_bias_correction.doMean(x, fobj, **kwargs)
        if cls == 'fitVarscale':
            return WAS_bias_correction.doVarscale(x, fobj, **kwargs)
        if cls == 'fitQuant':
            return WAS_bias_correction.doQuant(x, fobj, **kwargs)
        if cls == 'fitDist':
            return WAS_bias_correction.doDist(x, fobj, **kwargs)
        raise ValueError(f"Unknown class: {cls}")

    @staticmethod
    def _to_2d(arr):
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim == 3:
            arr = arr.reshape(arr.shape[0], -1)
        elif arr.ndim > 3:
            raise ValueError("Numpy array must be 1D, 2D, or 3D")
        return arr

    @staticmethod
    def fitQDM(obs, mod, kind="additive", denominator_floor=None, max_ratio=None):
        """Fit historical empirical marginals for quantile delta mapping.

        Use ``fitBC(obs, mod, "QDM", kind="additive")`` for signed variables
        and ``kind="multiplicative"`` for non-negative ratio variables.
        Application requires ``doBC(x, fitted, reference=target_model_sample)``.
        Neither fitting nor application accepts target observations. Fit on
        the training years only; choose the model-only reference period before
        evaluation. A full projection block may serve as its own reference,
        but this is a transductive projection adjustment, not an operational
        single-date forecasting procedure. Partition by season/month upstream.

        This is deterministic empirical QDM, not precipitation occurrence
        randomisation. Ratio floors/caps are optional approximations and break
        exact relative-change preservation where they activate.
        Reference: Cannon, Sobie & Murdock (2015), doi:10.1175/JCLI-D-14-00754.1.
        """
        return _fit_qdm(obs, mod, kind, denominator_floor, max_ratio)

    @staticmethod
    def doQDM(x, fobj, reference=None):
        """Apply QDM using an explicit target-period model distribution."""
        return _do_qdm(x, fobj, reference)

    # ------------------------------------------------------------------
    # MEAN
    # ------------------------------------------------------------------

    @staticmethod
    def fitMean(obs, mod):
        """Fit mean additive bias correction."""
        n_cols = obs.shape[1]
        par = {'delta': np.full(n_cols, np.nan),
               'all_nan': np.zeros(n_cols, dtype=bool)}
        for col in range(n_cols):
            o, m = _filter_nan_pair(obs[:, col], mod[:, col])
            if len(o) < 2 or len(m) < 2:
                par['all_nan'][col] = True
                continue
            par['delta'][col] = np.mean(o) - np.mean(m)
        return {'class': 'fitMean', 'par': par}

    @staticmethod
    def doMean(x, fobj):
        """Apply mean additive bias correction."""
        n_cols = x.shape[1]
        corrected = np.full_like(x, np.nan, dtype=float)
        par = fobj['par']
        for col in range(n_cols):
            if par['all_nan'][col]:
                continue
            corrected[:, col] = np.where(np.isfinite(x[:, col]), x[:, col] + par['delta'][col], np.nan)
        return corrected

    # ------------------------------------------------------------------
    # VARSCALE
    # ------------------------------------------------------------------

    @staticmethod
    def fitVarscale(obs, mod, std_tol=1e-6):
        """
        Fit variance scaling bias correction.

        If the modeled std is below ``std_tol`` the cell is flagged as
        ``all_nan`` rather than silently substituting ``std_m = 1.0``; that
        used to hide the underlying degeneracy.
        """
        n_cols = obs.shape[1]
        par = {
            'mean_o': np.full(n_cols, np.nan),
            'std_o':  np.full(n_cols, np.nan),
            'mean_m': np.full(n_cols, np.nan),
            'std_m':  np.full(n_cols, np.nan),
            'all_nan': np.zeros(n_cols, dtype=bool),
        }
        for col in range(n_cols):
            o, m = _filter_nan_pair(obs[:, col], mod[:, col])
            if len(o) < 2 or len(m) < 2:
                par['all_nan'][col] = True
                continue
            std_m = float(np.std(m))
            if std_m < std_tol:
                par['all_nan'][col] = True
                continue
            par['mean_o'][col] = float(np.mean(o))
            par['std_o'][col] = float(np.std(o))
            par['mean_m'][col] = float(np.mean(m))
            par['std_m'][col] = std_m
        return {'class': 'fitVarscale', 'par': par}

    @staticmethod
    def doVarscale(x, fobj):
        """Apply variance scaling bias correction."""
        n_cols = x.shape[1]
        corrected = np.full_like(x, np.nan, dtype=float)
        par = fobj['par']
        for col in range(n_cols):
            if par['all_nan'][col]:
                continue
            corrected[:, col] = (
                par['mean_o'][col]
                + (np.where(np.isfinite(x[:, col]), x[:, col], np.nan) - par['mean_m'][col])
                * (par['std_o'][col] / par['std_m'][col])
            )
        return corrected

    # ------------------------------------------------------------------
    # QUANT (non-parametric)
    # ------------------------------------------------------------------

    @staticmethod
    def fitQuant(obs, mod, qstep=0.01, nboot=1, random_state=None):
        """Fit empirical quantile mapping (non-parametric)."""
        rng = np.random.default_rng(random_state)
        n_cols = obs.shape[1]
        nq = int(1 / qstep) + 1
        probs = np.linspace(0, 1, nq)
        par = {
            'modq': np.full((nq, n_cols), np.nan),
            'fitq': np.full((nq, n_cols), np.nan),
            'all_nan': np.zeros(n_cols, dtype=bool),
        }
        for col in range(n_cols):
            o, m = _filter_nan_pair(obs[:, col], mod[:, col])
            if len(o) < 2 or len(m) < 2:
                par['all_nan'][col] = True
                continue
            par['modq'][:, col] = np.quantile(m, probs)
            if nboot > 1:
                boot_q = np.array([
                    np.quantile(rng.choice(o, len(o), replace=True), probs)
                    for _ in range(nboot)
                ])
                par['fitq'][:, col] = np.mean(boot_q, axis=0)
            else:
                par['fitq'][:, col] = np.quantile(o, probs)
        return {'class': 'fitQuant', 'par': par}

    @staticmethod
    def doQuant(x, fobj, type='linear'):
        """Apply empirical quantile mapping correction."""
        n_cols = x.shape[1]
        corrected = np.full_like(x, np.nan, dtype=float)
        par = fobj['par']
        for col in range(n_cols):
            if par['all_nan'][col]:
                continue
            xi = x[:, col].astype(float, copy=True)
            modq = par['modq'][:, col]
            fitq = par['fitq'][:, col]
            # FIX: dedupe for the interp1d strictly-increasing requirement.
            modq_u, fitq_u = _dedupe_monotone(modq, fitq)
            if modq_u.size < 2:
                if modq_u.size == 1:
                    corrected[:, col] = np.where(np.isfinite(xi), fitq_u[0], np.nan)
                continue
            interp_func = interp1d(
                modq_u, fitq_u, kind=type,
                bounds_error=False, fill_value='extrapolate',
            )
            out = np.full_like(xi, np.nan)
            valid = np.isfinite(xi)
            out[valid] = interp_func(xi[valid])
            corrected[:, col] = out
        return corrected

    # ------------------------------------------------------------------
    # DIST (normal / lognormal / gamma / weibull)
    # ------------------------------------------------------------------

    _DIST_MAP_BC = {
        'normal':    norm,
        'lognormal': lognorm,
        'gamma':     gamma,
        'weibull':   weibull_min,
    }

    @staticmethod
    def fitDist(obs, mod, distr='normal'):
        """Fit parametric bias correction assuming a specified distribution."""
        n_cols = obs.shape[1]
        par = {'par_o': [], 'par_m': [], 'distr': distr,
               'all_nan': np.zeros(n_cols, dtype=bool)}
        distr_l = distr.lower()
        dist = WAS_bias_correction._DIST_MAP_BC.get(distr_l)
        if dist is None:
            raise ValueError(f"Unknown distribution: {distr}")

        floc_kwargs = {'floc': 0} if distr_l in ('gamma', 'weibull', 'lognormal') else {}

        for col in range(n_cols):
            o, m = _filter_nan_pair(obs[:, col], mod[:, col])
            if len(o) < 2 or len(m) < 2:
                par['all_nan'][col] = True
                par['par_o'].append(None)
                par['par_m'].append(None)
                continue
            try:
                par_o = dist.fit(o, **floc_kwargs)
                par_m = dist.fit(m, **floc_kwargs)
            except Exception:
                par['all_nan'][col] = True
                par['par_o'].append(None)
                par['par_m'].append(None)
                continue
            par['par_o'].append(par_o)
            par['par_m'].append(par_m)
        return {'class': 'fitDist', 'par': par}

    @staticmethod
    def doDist(x, fobj):
        """Apply parametric distribution bias correction."""
        n_cols = x.shape[1]
        corrected = np.full_like(x, np.nan, dtype=float)
        par = fobj['par']
        distr_l = par['distr'].lower()
        dist = WAS_bias_correction._DIST_MAP_BC.get(distr_l)
        if dist is None:
            raise ValueError(f"Unknown distribution: {par['distr']}")

        for col in range(n_cols):
            if par['all_nan'][col]:
                continue
            par_o = par['par_o'][col]
            par_m = par['par_m'][col]
            xi = x[:, col].astype(float, copy=True)
            out = np.full_like(xi, np.nan)
            valid = np.isfinite(xi)
            cdf = dist.cdf(xi[valid], *par_m)
            # FIX: clip strictly inside (0, 1) so that ppf doesn't return inf
            # at saturated tails.
            cdf = np.clip(cdf, _CDF_EPS, 1 - _CDF_EPS)
            out[valid] = dist.ppf(cdf, *par_o)
            corrected[:, col] = out
        return corrected


# =============================================================================
# Cross-validator adapters (2026-08-26)
# =============================================================================
# These wrappers do not change WAS_Qmap / WAS_bias_correction.  They expose the
# standard WASS2S compute_model/forecast contract so WAS_Cross_Validator can fit
# every correction strictly inside each temporal training fold.

class _WAS_BiasCorrectionCVBase:
    _was_bias_correction_adapter = True
    family = None

    def __init__(self, method, fit_kwargs=None, apply_kwargs=None,
                 member_dim="M", reduce_members="mean", random_state=None):
        self.method=str(method)
        self.fit_kwargs={} if fit_kwargs is None else dict(fit_kwargs)
        self.apply_kwargs={} if apply_kwargs is None else dict(apply_kwargs)
        self.member_dim=str(member_dim)
        if reduce_members not in (None,"mean","median"):
            raise ValueError("reduce_members must be None, 'mean', or 'median'.")
        self.reduce_members=reduce_members
        self.random_state=random_state
        self.fitted_=None

    @staticmethod
    def _target(y):
        if isinstance(y,xr.DataArray) and "M" in y.dims:
            if y.sizes["M"] != 1:
                raise ValueError("Observation must have no ensemble axis or a singleton M.")
            y=y.isel(M=0,drop=True)
        if isinstance(y,xr.DataArray):
            y=y.transpose("T","Y","X")
        return y

    def _fit_one(self, obs, mod):
        return self._fit_dispatch(obs, mod)

    def _fit_dispatch(self, obs, mod):
        raise NotImplementedError

    def _apply_dispatch(self, x, fobj, apply_kwargs=None):
        raise NotImplementedError

    def fit(self, obs, mod):
        obs=self._target(obs)
        if isinstance(mod,xr.DataArray) and self.member_dim in mod.dims:
            fitted={}
            for m in mod[self.member_dim].values:
                fitted[m.item() if hasattr(m,"item") else m]=self._fit_one(
                    obs,mod.sel({self.member_dim:m},drop=True).transpose("T","Y","X"))
            self.fitted_=fitted
        else:
            if isinstance(mod,xr.DataArray): mod=mod.transpose("T","Y","X")
            self.fitted_=self._fit_one(obs,mod)
        return self

    def transform(self, x):
        if self.fitted_ is None:
            raise RuntimeError("Bias correction is not fitted. Call fit() first.")
        if isinstance(x,xr.DataArray) and self.member_dim in x.dims:
            out=[]
            for m in x[self.member_dim].values:
                key=m.item() if hasattr(m,"item") else m
                if key not in self.fitted_:
                    raise KeyError(f"No fitted bias correction for member {key!r}.")
                options = dict(self.apply_kwargs)
                reference = options.get("reference")
                if isinstance(reference, xr.DataArray) and self.member_dim in reference.dims:
                    options["reference"] = reference.sel({self.member_dim: m}, drop=True)
                da=self._apply_dispatch(x.sel({self.member_dim:m},drop=True).transpose("T","Y","X"),self.fitted_[key], options)
                out.append(da.expand_dims({self.member_dim:[m]}))
            ans=xr.concat(out,dim=self.member_dim).transpose("T",self.member_dim,"Y","X")
            if self.reduce_members=="mean": return ans.mean(self.member_dim,skipna=True)
            if self.reduce_members=="median": return ans.median(self.member_dim,skipna=True)
            return ans
        ans=self._apply_dispatch(x.transpose("T","Y","X") if isinstance(x,xr.DataArray) else x,self.fitted_)
        return ans

    def fit_transform(self, obs, mod_train, mod_test=None):
        self.fit(obs,mod_train)
        return self.transform(mod_train if mod_test is None else mod_test)

    def compute_model(self, X_train, y_train, X_test, y_test=None, **kwargs):
        """Fold-safe correction: fit on (y_train, X_train), apply to X_test."""
        self.fit(y_train,X_train)
        out=self.transform(X_test)
        return out

    def forecast(self, Predictant, Predictor, Predictor_for_year, **kwargs):
        """Fit on the historical pair then correct the real forecast year."""
        self.fit(Predictant,Predictor)
        return self.transform(Predictor_for_year)

    def get_params(self):
        return {"family":self.family,"method":self.method,"fit_kwargs":dict(self.fit_kwargs),
                "apply_kwargs":dict(self.apply_kwargs),"member_dim":self.member_dim,
                "reduce_members":self.reduce_members,"random_state":self.random_state}


class WAS_Qmap_CV(_WAS_BiasCorrectionCVBase):
    """Cross-validator-ready precipitation quantile-mapping adapter."""
    family="qmap"
    def __init__(self, method="RQUANT", fit_kwargs=None, apply_kwargs=None,
                 member_dim="M", reduce_members="mean", random_state=None):
        super().__init__(method,fit_kwargs,apply_kwargs,member_dim,reduce_members,random_state)
    def _fit_dispatch(self,obs,mod):
        kwargs = dict(self.fit_kwargs)
        if self.method.upper() in {"QUANT", "RQUANT"}:
            kwargs.setdefault("random_state", self.random_state)
        return WAS_Qmap.fitQmap(obs, mod, self.method, **kwargs)
    def _apply_dispatch(self,x,fobj,apply_kwargs=None):
        return WAS_Qmap.doQmap(x,fobj,**(self.apply_kwargs if apply_kwargs is None else apply_kwargs))


class WAS_bias_correction_CV(_WAS_BiasCorrectionCVBase):
    """Cross-validator-ready continuous-variable bias-correction adapter."""
    family="continuous"
    def __init__(self, method="MEAN", fit_kwargs=None, apply_kwargs=None,
                 member_dim="M", reduce_members="mean", random_state=None):
        super().__init__(method,fit_kwargs,apply_kwargs,member_dim,reduce_members,random_state)
    def _fit_dispatch(self,obs,mod):
        kwargs = dict(self.fit_kwargs)
        if self.method.upper() == "QUANT":
            kwargs.setdefault("random_state", self.random_state)
        return WAS_bias_correction.fitBC(obs, mod, self.method, **kwargs)
    def _apply_dispatch(self,x,fobj,apply_kwargs=None):
        return WAS_bias_correction.doBC(x,fobj,**(self.apply_kwargs if apply_kwargs is None else apply_kwargs))


# More explicit aliases for new notebooks, while retaining the historical names.
WAS_Qmap_Model = WAS_Qmap_CV
WAS_BiasCorrection_Model = WAS_bias_correction_CV
