"""
Bias correction utilities for climate variables.

Two classes are provided:

- ``WAS_Qmap`` : quantile-mapping methods tailored to precipitation
  (non-negative, possibly with many dry days). Supports QUANT, RQUANT,
  SSPLIN, PTF and DIST methods, with optional wet/dry day handling.

- ``WAS_bias_correction`` : bias-correction methods for continuous variables
  such as temperature or wind speed. Supports MEAN, VARSCALE, QUANT, NORM
  and DIST methods.

Both classes accept NumPy arrays (1D / 2D / 3D) and 3D ``xarray.DataArray``
inputs, and they propagate / handle NaNs.

This is a corrected version of the original implementation.
"""

import inspect
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
    o = o[~np.isnan(o)]
    m = m[~np.isnan(m)]
    return o, m


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
        is_xarray = isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray)
        time_dim = spatial_dims = coords = attrs = original_dims = None

        if is_xarray:
            if obs.shape != mod.shape or len(obs.dims) != 3:
                raise ValueError(
                    "xarray DataArrays must be 3D with matching shapes and "
                    "dimensions (T, Y, X)"
                )
            time_dim = obs.dims[0]
            spatial_dims = obs.dims[1:]
            obs_stacked = obs.stack(grid=spatial_dims).transpose(time_dim, 'grid')
            mod_stacked = mod.stack(grid=spatial_dims).transpose(time_dim, 'grid')
            obs_data = obs_stacked.values
            mod_data = mod_stacked.values
            coords = obs.coords
            attrs = obs.attrs
            original_dims = obs.dims
        else:
            obs_data = WAS_Qmap._to_2d(obs)
            mod_data = WAS_Qmap._to_2d(mod)
            if obs_data.shape != mod_data.shape:
                raise ValueError(
                    f"obs and mod must have matching shapes; got "
                    f"{obs_data.shape} and {mod_data.shape}"
                )

        method_up = method.upper()
        if method_up == 'QUANT':
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
            if len(x.dims) != 3 or x.dims[1:] != fobj['spatial_dims']:
                raise ValueError("Input x must have matching spatial dimensions")
            time_dim = fobj['time_dim']
            spatial_dims = fobj['spatial_dims']
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
            corrected = corrected_stacked.unstack('grid')
            corrected.attrs = fobj['attrs']
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
        cls = fobj['class']
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
            th_mod = float(np.quantile(mod, 1 - p_wet)) if len(mod) else 0.0
            return th_mod, 0.0
        # numeric threshold
        th_obs = float(wet_day)
        if len(obs) == 0:
            return 0.0, th_obs
        p_wet = float(np.mean(obs >= th_obs))
        th_mod = float(np.quantile(mod, 1 - p_wet)) if len(mod) else 0.0
        return th_mod, th_obs

    # ------------------------------------------------------------------
    # QUANT
    # ------------------------------------------------------------------

    @staticmethod
    def fitQmapQUANT(obs, mod, wet_day=False, qstep=0.01, nboot=1):
        """Fit empirical quantile mapping with optional bootstrap of obs."""
        n_cols = obs.shape[1]
        n_q = int(1 / qstep) + 1
        probs = np.linspace(0, 1, n_q)

        par = {
            'modq': np.full((n_q, n_cols), np.nan),
            'fitq': np.full((n_q, n_cols), np.nan),
            'wet_day_th_mod': np.zeros(n_cols),
            'wet_day_th_obs': np.zeros(n_cols),
            'all_nan': np.zeros(n_cols, dtype=bool),
        }

        for col in range(n_cols):
            # FIX: filter NaNs out before computing thresholds / quantiles.
            o, m = _filter_nan_pair(obs[:, col], mod[:, col])
            par['wet_day_th_mod'][col], par['wet_day_th_obs'][col] = (
                WAS_Qmap._wet_day_threshold(o, m, wet_day)
            )
            th_obs = par['wet_day_th_obs'][col]
            if wet_day is True:
                o = o[o > th_obs]
            else:
                o = o[o >= th_obs]
            m = m[m >= par['wet_day_th_mod'][col]]

            if len(o) < 2 or len(m) < 2:
                par['all_nan'][col] = True
                continue

            par['modq'][:, col] = np.quantile(m, probs)
            if nboot > 1:
                boot_q = np.array([
                    np.quantile(np.random.choice(o, len(o), replace=True), probs)
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
            if all_nan[col]:
                continue
            xi = x[:, col].astype(float, copy=True)
            modq = par['modq'][:, col]
            fitq = par['fitq'][:, col]
            th_mod = par['wet_day_th_mod'][col]

            # FIX: dedupe to satisfy interp1d's strictly-increasing requirement.
            modq_u, fitq_u = _dedupe_monotone(modq, fitq)

            xi_out = np.full_like(xi, np.nan)

            valid = ~np.isnan(xi)
            mask_dry = valid & (xi < th_mod)
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

            corrected[:, col] = xi_out
        return corrected

    # ------------------------------------------------------------------
    # RQUANT
    # ------------------------------------------------------------------

    @staticmethod
    def fitQmapRQUANT(obs, mod, wet_day=True, qstep=0.01, nlls=10, nboot=10):
        """Robust quantile mapping with local linear fits."""
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
        }

        for col in range(n_cols):
            o, m = _filter_nan_pair(obs[:, col], mod[:, col])
            par['wet_day_th_mod'][col], par['wet_day_th_obs'][col] = (
                WAS_Qmap._wet_day_threshold(o, m, wet_day)
            )
            th_obs = par['wet_day_th_obs'][col]
            if wet_day is True:
                o = o[o > th_obs]
            else:
                o = o[o >= th_obs]
            m = m[m >= par['wet_day_th_mod'][col]]

            if len(o) < 2 or len(m) < 2:
                par['all_nan'][col] = True
                continue

            par['modq'][:, col] = np.quantile(m, probs)
            if nboot > 1:
                boot_q = np.array([
                    np.quantile(np.random.choice(o, len(o), replace=True), probs)
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
            valid = ~np.isnan(xi)
            mask_dry = valid & (xi < th_mod)
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

            corrected[:, col] = xi_out
        return corrected

    # ------------------------------------------------------------------
    # SSPLIN
    # ------------------------------------------------------------------

    @staticmethod
    def fitQmapSSPLIN(obs, mod, wet_day=False, qstep=0.01):
        """Fit quantile mapping using PCHIP smoothing splines."""
        n_cols = obs.shape[1]
        n_q = int(1 / qstep) + 1
        probs = np.linspace(0, 1, n_q)

        par = {
            'modq': np.full((n_q, n_cols), np.nan),
            'fitq': np.full((n_q, n_cols), np.nan),
            'wet_day_th_mod': np.zeros(n_cols),
            'wet_day_th_obs': np.zeros(n_cols),
            'all_nan': np.zeros(n_cols, dtype=bool),
        }
        for col in range(n_cols):
            o, m = _filter_nan_pair(obs[:, col], mod[:, col])
            par['wet_day_th_mod'][col], par['wet_day_th_obs'][col] = (
                WAS_Qmap._wet_day_threshold(o, m, wet_day)
            )
            th_obs = par['wet_day_th_obs'][col]
            if wet_day is True:
                o = o[o > th_obs]
            else:
                o = o[o >= th_obs]
            m = m[m >= par['wet_day_th_mod'][col]]

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
            if all_nan[col]:
                continue
            xi = x[:, col].astype(float, copy=True)
            modq = par['modq'][:, col]
            fitq = par['fitq'][:, col]
            th_mod = par['wet_day_th_mod'][col]

            # FIX: PchipInterpolator requires strictly increasing x.
            modq_u, fitq_u = _dedupe_monotone(modq, fitq)

            xi_out = np.full_like(xi, np.nan)
            valid = ~np.isnan(xi)
            mask_dry = valid & (xi < th_mod)
            mask_interp = valid & ~mask_dry

            xi_out[mask_dry] = 0.0
            if np.any(mask_interp):
                if modq_u.size >= 2:
                    interp_func = PchipInterpolator(modq_u, fitq_u, extrapolate=True)
                    xi_out[mask_interp] = interp_func(xi[mask_interp])
                else:
                    xi_out[mask_interp] = fitq_u[0] if fitq_u.size else np.nan
            corrected[:, col] = xi_out
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
            if wet_day is True:
                o = o[o > th_obs]
            else:
                o = o[o >= th_obs]
            m = m[m >= par['wet_day_th_mod'][col]]

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
            if all_nan[col]:
                continue
            xi = x[:, col].astype(float, copy=True)
            params = par['par'][col]
            th_mod = par['wet_day_th_mod'][col]

            xi_out = np.full_like(xi, np.nan)
            valid = ~np.isnan(xi)
            mask_dry = valid & (xi < th_mod)
            mask_wet = valid & ~mask_dry

            xi_out[mask_dry] = 0.0
            if np.any(mask_wet):
                with np.errstate(all='ignore'):
                    vals = tf(xi[mask_wet], *params)
                vals = np.where(np.isfinite(vals), vals, np.nan)
                xi_out[mask_wet] = vals
            corrected[:, col] = xi_out
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
               'all_nan': np.zeros(n_cols, dtype=bool)}

        dist = WAS_Qmap._DIST_MAP_QMAP.get(distr)
        if dist is None:
            raise ValueError(f"Unknown distr: {distr}")

        for col in range(n_cols):
            o, m = _filter_nan_pair(obs[:, col], mod[:, col])

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
                        par_o['params'] = dist.fit(o_pos)
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
                        par_m['params'] = dist.fit(m_pos)
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

            valid = ~np.isnan(val)
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
        is_xarray = isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray)
        time_dim = spatial_dims = coords = attrs = original_dims = None

        if is_xarray:
            if obs.shape != mod.shape or len(obs.dims) != 3:
                raise ValueError(
                    "xarray DataArrays must be 3D with matching shapes and "
                    "dimensions (T, Y, X)"
                )
            time_dim = obs.dims[0]
            spatial_dims = obs.dims[1:]
            obs_stacked = obs.stack(grid=spatial_dims).transpose(time_dim, 'grid')
            mod_stacked = mod.stack(grid=spatial_dims).transpose(time_dim, 'grid')
            obs_data = obs_stacked.values
            mod_data = mod_stacked.values
            coords = obs.coords
            attrs = obs.attrs
            original_dims = obs.dims
        else:
            obs_data = WAS_bias_correction._to_2d(obs)
            mod_data = WAS_bias_correction._to_2d(mod)
            if obs_data.shape != mod_data.shape:
                raise ValueError(
                    f"obs and mod must have matching shapes; got "
                    f"{obs_data.shape} and {mod_data.shape}"
                )

        method_up = method.upper()
        if method_up == 'MEAN':
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
            if len(x.dims) != 3 or x.dims[1:] != fobj['spatial_dims']:
                raise ValueError("Input x must have matching spatial dimensions")
            time_dim = fobj['time_dim']
            spatial_dims = fobj['spatial_dims']
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
            corrected = corrected_stacked.unstack('grid')
            corrected.attrs = fobj['attrs']
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
        cls = fobj['class']
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
            corrected[:, col] = x[:, col] + par['delta'][col]
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
                + (x[:, col] - par['mean_m'][col])
                * (par['std_o'][col] / par['std_m'][col])
            )
        return corrected

    # ------------------------------------------------------------------
    # QUANT (non-parametric)
    # ------------------------------------------------------------------

    @staticmethod
    def fitQuant(obs, mod, qstep=0.01, nboot=1):
        """Fit empirical quantile mapping (non-parametric)."""
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
                    np.quantile(np.random.choice(o, len(o), replace=True), probs)
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
                    corrected[:, col] = np.where(np.isnan(xi), np.nan, fitq_u[0])
                continue
            interp_func = interp1d(
                modq_u, fitq_u, kind=type,
                bounds_error=False, fill_value='extrapolate',
            )
            out = np.full_like(xi, np.nan)
            valid = ~np.isnan(xi)
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

        floc_kwargs = {'floc': 0} if distr_l in ('gamma', 'weibull') else {}

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
            valid = ~np.isnan(xi)
            cdf = dist.cdf(xi[valid], *par_m)
            # FIX: clip strictly inside (0, 1) so that ppf doesn't return inf
            # at saturated tails.
            cdf = np.clip(cdf, _CDF_EPS, 1 - _CDF_EPS)
            out[valid] = dist.ppf(cdf, *par_o)
            corrected[:, col] = out
        return corrected
