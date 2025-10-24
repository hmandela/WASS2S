import numpy as np
import xarray as xr
from scipy import stats
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.optimize import minimize
from scipy.stats import norm, lognorm, gamma, weibull_min
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

class WAS_Qmap:
    """
    Bias correction methods using quantile mapping techniques, adapted from qmap R package.

    This class provides static methods for fitting and applying various bias correction
    techniques, including empirical quantile mapping (QUANT), robust quantile mapping (RQUANT),
    smoothing splines (SSPLIN), parametric transformations (PTF), and distribution-based
    methods (DIST). The methods support both NumPy arrays (1D, 2D, or 3D) and xarray
    DataArrays (3D: T, Y, X or similar).

    All methods handle wet/dry day corrections optionally and are designed for
    precipitation or similar non-negative variables.

    Notes
    -----
    - Inputs are expected to be non-negative.
    - For gridded data, computations are performed column-wise (per grid cell).
    - xarray support preserves coordinates and attributes.
    """

    @staticmethod
    def fitQmap(obs, mod, method, **kwargs):
        """
        Fit a bias correction model using the specified quantile mapping method.

        Parameters
        ----------
        obs : array_like or xarray.DataArray
            Observed data. If array_like, can be 1D (time), 2D (time, grid), or 3D (T, Y, X).
            If xarray.DataArray, must be 3D with dimensions (T, Y, X).
        mod : array_like or xarray.DataArray
            Modeled data to fit against, same shape as `obs`.
        method : str
            Bias correction method. Options: 'QUANT', 'RQUANT', 'SSPLIN', 'PTF', 'DIST' (case-insensitive).
        **kwargs
            Additional keyword arguments passed to the specific fitting method.

        Returns
        -------
        dict
            Fitted object containing parameters, class identifier, and metadata for applying correction.

        Raises
        ------
        ValueError
            If shapes mismatch, invalid dimensions, or unknown method.

        See Also
        --------
        doQmap : Apply the fitted bias correction to new data.
        """
        is_xarray = isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray)
        original_dims = None
        time_dim = None
        spatial_dims = None
        coords = None
        attrs = None
        if is_xarray:
            if obs.shape != mod.shape or len(obs.dims) != 3:
                raise ValueError("xarray DataArrays must be 3D with matching shapes and dimensions (T, Y, X)")
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
        
        method = method.upper()
        if method == 'QUANT':
            fobj = WAS_Qmap.fitQmapQUANT(obs_data, mod_data, **kwargs)
        elif method == 'RQUANT':
            fobj = WAS_Qmap.fitQmapRQUANT(obs_data, mod_data, **kwargs)
        elif method == 'SSPLIN':
            fobj = WAS_Qmap.fitQmapSSPLIN(obs_data, mod_data, **kwargs)
        elif method == 'PTF':
            fobj = WAS_Qmap.fitQmapPTF(obs_data, mod_data, **kwargs)
        elif method == 'DIST':
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

        Parameters
        ----------
        x : array_like or xarray.DataArray
            New modeled data to correct, same format and shape structure as fitting data.
        fobj : dict
            Fitted object from `fitQmap`.
        **kwargs
            Additional keyword arguments passed to the specific application method.

        Returns
        -------
        array_like or xarray.DataArray
            Bias-corrected data, same type and shape as `x`.

        Raises
        ------
        ValueError
            If input types or dimensions mismatch the fitted object.

        See Also
        --------
        fitQmap : Fit the bias correction model.
        """
        if 'is_xarray' in fobj and fobj['is_xarray']:
            if not isinstance(x, xr.DataArray):
                raise ValueError("Input x must be xarray.DataArray when fitted with DataArray")
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
                coords={time_dim: x_stacked.coords[time_dim], 'grid': x_stacked.coords['grid']}
            )
            corrected = corrected_stacked.unstack('grid')
            corrected.attrs = fobj['attrs']
            return corrected
        else:
            x_data = WAS_Qmap._to_2d(x)
            return np.squeeze(WAS_Qmap._doQmap_internal(x_data, fobj, **kwargs))

    @staticmethod
    def _doQmap_internal(x, fobj, **kwargs):
        """
        Internal helper to apply bias correction based on fitted class.

        Parameters
        ----------
        x : ndarray
            2D array of data to correct (time, grid).
        fobj : dict
            Fitted object.
        **kwargs
            Additional arguments for specific methods.

        Returns
        -------
        ndarray
            Corrected 2D array.

        Raises
        ------
        ValueError
            If unknown fitted class.
        """
        cls = fobj['class']
        if cls == 'fitQmapQUANT':
            return WAS_Qmap.doQmapQUANT(x, fobj, **kwargs)
        elif cls == 'fitQmapRQUANT':
            return WAS_Qmap.doQmapRQUANT(x, fobj, **kwargs)
        elif cls == 'fitQmapSSPLIN':
            return WAS_Qmap.doQmapSSPLIN(x, fobj, **kwargs)
        elif cls == 'fitQmapPTF':
            return WAS_Qmap.doQmapPTF(x, fobj, **kwargs)
        elif cls == 'fitQmapDIST':
            return WAS_Qmap.doQmapDIST(x, fobj, **kwargs)
        else:
            raise ValueError(f"Unknown class: {cls}")

    @staticmethod
    def _to_2d(arr):
        """
        Convert input array to 2D (time, grid) format.

        Parameters
        ----------
        arr : array_like
            Input array (0D to 3D).

        Returns
        -------
        ndarray
            2D array.

        Raises
        ------
        ValueError
            If more than 3 dimensions.
        """
        arr = np.asarray(arr)
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

        Parameters
        ----------
        obs : ndarray
            Observed data column.
        mod : ndarray
            Modeled data column.
        wet_day : bool or float
            If False, no threshold (0). If True, compute based on wet fraction.
            If float, use as observation threshold and compute model accordingly.

        Returns
        -------
        tuple
            (model_threshold, obs_threshold)
        """
        if wet_day is False:
            return 0, 0
        if wet_day is True:
            p_wet = np.mean(obs > 0)
            th_mod = np.quantile(mod, 1 - p_wet)
            th_obs = 0
        else:
            th_obs = wet_day
            obs_th = obs[obs >= th_obs]
            p_wet = len(obs_th) / len(obs)
            th_mod = np.quantile(mod, 1 - p_wet)
        return th_mod, th_obs

    @staticmethod
    def fitQmapQUANT(obs, mod, wet_day=False, qstep=0.01, nboot=1):
        """
        Fit empirical quantile mapping (QUANT) with optional bootstrapping.

        Parameters
        ----------
        obs : ndarray
            2D observed data (time, grid).
        mod : ndarray
            2D modeled data (time, grid).
        wet_day : bool or float, optional
            Wet day handling (default False).
        qstep : float, optional
            Quantile step size (default 0.01).
        nboot : int, optional
            Number of bootstrap samples for observed quantiles (default 1, no bootstrap).

        Returns
        -------
        dict
            Fitted parameters including quantiles and thresholds.
        """
        n_cols = obs.shape[1]
        par = {'modq': np.zeros((int(1/qstep)+1, n_cols)), 'fitq': np.zeros((int(1/qstep)+1, n_cols)),
               'wet_day_th_mod': np.zeros(n_cols), 'wet_day_th_obs': np.zeros(n_cols)}
        probs = np.linspace(0, 1, int(1/qstep)+1)
        for col in range(n_cols):
            o = obs[:, col]
            m = mod[:, col]
            par['wet_day_th_mod'][col], par['wet_day_th_obs'][col] = WAS_Qmap._wet_day_threshold(o, m, wet_day)
            th_obs = par['wet_day_th_obs'][col]
            if wet_day is True:
                o = o[o > th_obs]
            else:
                o = o[o >= th_obs]
            m = m[m >= par['wet_day_th_mod'][col]]
            if len(o) == 0 or len(m) == 0:
                par['modq'][:, col] = np.nan
                par['fitq'][:, col] = np.nan
                continue
            par['modq'][:, col] = np.quantile(m, probs)
            if nboot > 1:
                boot_q = np.array([np.quantile(np.random.choice(o, len(o), replace=True), probs) for _ in range(nboot)])
                par['fitq'][:, col] = np.mean(boot_q, axis=0)
            else:
                par['fitq'][:, col] = np.quantile(o, probs)
        return {'class': 'fitQmapQUANT', 'par': par, 'wet_day': wet_day}

    @staticmethod
    def doQmapQUANT(x, fobj, type='linear'):
        """
        Apply empirical quantile mapping (QUANT) correction.

        Parameters
        ----------
        x : ndarray
            2D data to correct (time, grid).
        fobj : dict
            Fitted object from `fitQmapQUANT`.
        type : str, optional
            Interpolation kind for `interp1d` (default 'linear').

        Returns
        -------
        ndarray
            Corrected data.
        """
        n_cols = x.shape[1]
        corrected = np.zeros_like(x)
        par = fobj['par']
        for col in range(n_cols):
            xi = x[:, col]
            modq = par['modq'][:, col]
            fitq = par['fitq'][:, col]
            th_mod = par['wet_day_th_mod'][col]
            xi_corrected = np.zeros_like(xi)
            mask_below = xi < th_mod
            xi_corrected[mask_below] = 0
            mask_above = ~mask_below
            interp_func = interp1d(modq, fitq, kind=type, bounds_error=False, fill_value=(fitq[0], fitq[-1]))
            xi_corrected[mask_above] = interp_func(xi[mask_above])
            high_mask = xi > modq[-1]
            xi_corrected[high_mask] = xi[high_mask] + (fitq[-1] - modq[-1])
            corrected[:, col] = xi_corrected
        return corrected

    @staticmethod
    def fitQmapRQUANT(obs, mod, wet_day=True, qstep=0.01, nlls=10, nboot=10):
        """
        Fit robust quantile mapping (RQUANT) with local linear fitting.

        Parameters
        ----------
        obs : ndarray
            2D observed data (time, grid).
        mod : ndarray
            2D modeled data (time, grid).
        wet_day : bool or float, optional
            Wet day handling (default True).
        qstep : float, optional
            Quantile step size (default 0.01).
        nlls : int, optional
            Number of local quantiles for linear fit (default 10).
        nboot : int, optional
            Number of bootstrap samples (default 10).

        Returns
        -------
        dict
            Fitted parameters including quantiles, slopes, and thresholds.
        """
        n_cols = obs.shape[1]
        probs = np.linspace(0, 1, int(1/qstep)+1)
        n = len(probs)
        par = {'modq': np.zeros((n, n_cols)), 'fitq': np.zeros((n, n_cols)),
               'slope_bound': np.zeros((2, n_cols)), 'wet_day_th_mod': np.zeros(n_cols), 'wet_day_th_obs': np.zeros(n_cols)}
        for col in range(n_cols):
            o = obs[:, col]
            m = mod[:, col]
            par['wet_day_th_mod'][col], par['wet_day_th_obs'][col] = WAS_Qmap._wet_day_threshold(o, m, wet_day)
            th_obs = par['wet_day_th_obs'][col]
            if wet_day is True:
                o = o[o > th_obs]
            else:
                o = o[o >= th_obs]
            m = m[m >= par['wet_day_th_mod'][col]]
            if len(o) == 0 or len(m) == 0:
                par['modq'][:, col] = np.nan
                par['fitq'][:, col] = np.nan
                continue
            par['modq'][:, col] = np.quantile(m, probs)
            if nboot > 1:
                boot_q = np.array([np.quantile(np.random.choice(o, len(o), replace=True), probs) for _ in range(nboot)])
                obsq = np.mean(boot_q, axis=0)
            else:
                obsq = np.quantile(o, probs)
            fitq = np.zeros(n)
            for i in range(n):
                start = max(0, i - nlls // 2)
                end = min(n, i + nlls // 2 + 1)
                X = par['modq'][start:end, col]
                y = obsq[start:end]
                if len(X) < 2:
                    fitq[i] = y.mean() if len(y) > 0 else obsq[i]
                else:
                    slope, intercept = np.polyfit(X, y, 1)
                    fitq[i] = slope * par['modq'][i, col] + intercept
            par['fitq'][:, col] = fitq
            par['slope_bound'][0, col] = np.polyfit(par['modq'][:2, col], par['fitq'][:2, col], 1)[0]
            par['slope_bound'][1, col] = np.polyfit(par['modq'][-2:, col], par['fitq'][-2:, col], 1)[0]
        return {'class': 'fitQmapRQUANT', 'par': par, 'wet_day': wet_day}

    @staticmethod
    def doQmapRQUANT(x, fobj, type='linear', slope_bound=[0, np.inf]):
        """
        Apply robust quantile mapping (RQUANT) correction.

        Parameters
        ----------
        x : ndarray
            2D data to correct (time, grid).
        fobj : dict
            Fitted object from `fitQmapRQUANT`.
        type : str, optional
            Interpolation kind for `interp1d` (default 'linear').
        slope_bound : list of float, optional
            Bounds for extrapolation slopes [min, max] (default [0, inf]).

        Returns
        -------
        ndarray
            Corrected data.
        """
        n_cols = x.shape[1]
        corrected = np.zeros_like(x)
        par = fobj['par']
        for col in range(n_cols):
            xi = x[:, col]
            modq = par['modq'][:, col]
            fitq = par['fitq'][:, col]
            th_mod = par['wet_day_th_mod'][col]
            low_slope = np.clip(par['slope_bound'][0, col], slope_bound[0], slope_bound[1])
            high_slope = np.clip(par['slope_bound'][1, col], slope_bound[0], slope_bound[1])
            xi_corrected = np.zeros_like(xi)
            mask_below = xi < th_mod
            xi_corrected[mask_below] = 0
            mask_above = ~mask_below
            interp_func = interp1d(modq, fitq, kind=type, bounds_error=False, fill_value='extrapolate')
            xi_corrected[mask_above] = interp_func(xi[mask_above])
            low_mask = xi < modq[0]
            xi_corrected[low_mask] = fitq[0] + low_slope * (xi[low_mask] - modq[0])
            high_mask = xi > modq[-1]
            xi_corrected[high_mask] = fitq[-1] + high_slope * (xi[high_mask] - modq[-1])
            corrected[:, col] = xi_corrected
        return corrected

    @staticmethod
    def fitQmapSSPLIN(obs, mod, wet_day=False, qstep=0.01):
        """
        Fit quantile mapping using PCHIP smoothing splines (SSPLIN).

        Parameters
        ----------
        obs : ndarray
            2D observed data (time, grid).
        mod : ndarray
            2D modeled data (time, grid).
        wet_day : bool or float, optional
            Wet day handling (default False).
        qstep : float, optional
            Quantile step size (default 0.01).

        Returns
        -------
        dict
            Fitted parameters including quantiles and thresholds.
        """
        n_cols = obs.shape[1]
        par = {'modq': np.zeros((int(1/qstep)+1, n_cols)), 'fitq': np.zeros((int(1/qstep)+1, n_cols)),
               'wet_day_th_mod': np.zeros(n_cols), 'wet_day_th_obs': np.zeros(n_cols)}
        probs = np.linspace(0, 1, int(1/qstep)+1)
        for col in range(n_cols):
            o = obs[:, col]
            m = mod[:, col]
            par['wet_day_th_mod'][col], par['wet_day_th_obs'][col] = WAS_Qmap._wet_day_threshold(o, m, wet_day)
            th_obs = par['wet_day_th_obs'][col]
            if wet_day is True:
                o = o[o > th_obs]
            else:
                o = o[o >= th_obs]
            m = m[m >= par['wet_day_th_mod'][col]]
            if len(o) == 0 or len(m) == 0:
                par['modq'][:, col] = np.nan
                par['fitq'][:, col] = np.nan
                continue
            par['modq'][:, col] = np.quantile(m, probs)
            par['fitq'][:, col] = np.quantile(o, probs)
        return {'class': 'fitQmapSSPLIN', 'par': par, 'wet_day': wet_day}

    @staticmethod
    def doQmapSSPLIN(x, fobj):
        """
        Apply quantile mapping correction using PCHIP interpolator.

        Parameters
        ----------
        x : ndarray
            2D data to correct (time, grid).
        fobj : dict
            Fitted object from `fitQmapSSPLIN`.

        Returns
        -------
        ndarray
            Corrected data.
        """
        n_cols = x.shape[1]
        corrected = np.zeros_like(x)
        par = fobj['par']
        for col in range(n_cols):
            xi = x[:, col]
            modq = par['modq'][:, col]
            fitq = par['fitq'][:, col]
            th_mod = par['wet_day_th_mod'][col]
            xi_corrected = np.zeros_like(xi)
            mask_below = xi < th_mod
            xi_corrected[mask_below] = 0
            mask_above = ~mask_below
            interp_func = PchipInterpolator(modq, fitq, extrapolate=True)
            xi_corrected[mask_above] = interp_func(xi[mask_above])
            corrected[:, col] = xi_corrected
        return corrected

    @staticmethod
    def fitQmapPTF(obs, mod, transfun='power', parini=None, cost='RSS', wet_day=False, qstep=None):
        """
        Fit parametric transformation function (PTF) for bias correction.

        Parameters
        ----------
        obs : ndarray
            2D observed data (time, grid).
        mod : ndarray
            2D modeled data (time, grid).
        transfun : str or callable, optional
            Transformation function: 'power', 'power.x0', 'expasympt', 'expasympt.x0', 'scale', 'linear',
            or a custom callable (default 'power').
        parini : list, optional
            Initial parameter guesses for optimization.
        cost : str, optional
            Cost function for optimization: 'RSS' or 'MAE' (default 'RSS').
        wet_day : bool or float, optional
            Wet day handling (default False).
        qstep : float, optional
            If set, fit on quantiles with this step; else fit on sorted data.

        Returns
        -------
        dict
            Fitted parameters including transformation and thresholds.

        Raises
        ------
        ValueError
            If unknown transfun or cost.
        """
        n_cols = obs.shape[1]
        par = {'transfun': transfun, 'par': [], 'wet_day_th_mod': np.zeros(n_cols), 'wet_day_th_obs': np.zeros(n_cols)}
        trans_funcs = {
            'power': lambda x, a, b: a * x**b,
            'power.x0': lambda x, a, b, x0: a * (x - x0)**b,
            'expasympt': lambda x, a, b, tau: (a + b * x) * (1 - np.exp(-x / tau)),
            'expasympt.x0': lambda x, a, b, x0, tau: (a + b * (x - x0)) * (1 - np.exp(-(x - x0) / tau)),
            'scale': lambda x, b: b * x,
            'linear': lambda x, a, b: a + b * x
        }
        if callable(transfun):
            tf = transfun
            n_params = tf.__code__.co_argcount - 1
        else:
            tf = trans_funcs.get(transfun)
            if tf is None:
                raise ValueError("Unknown transfun")
            n_params = tf.__code__.co_argcount - 1
        for col in range(n_cols):
            o = obs[:, col]
            m = mod[:, col]
            par['wet_day_th_mod'][col], par['wet_day_th_obs'][col] = WAS_Qmap._wet_day_threshold(o, m, wet_day)
            th_obs = par['wet_day_th_obs'][col]
            if wet_day is True:
                o = o[o > th_obs]
            else:
                o = o[o >= th_obs]
            m = m[m >= par['wet_day_th_mod'][col]]
            if len(o) == 0 or len(m) == 0:
                par['par'].append([0.0] * n_params)
                continue
            if qstep is not None:
                probs = np.linspace(0, 1, int(1/qstep)+1)
                mq = np.quantile(m, probs)
                oq = np.quantile(o, probs)
            else:
                n_points = min(len(m), len(o))
                probs = np.linspace(0, 1, n_points)
                mq = np.quantile(m, probs)
                oq = np.quantile(o, probs)
            if parini is None:
                parini_guess = [1.0] * n_params
            else:
                parini_guess = parini
            def objective(p):
                pred = tf(mq, *p)
                if cost == 'RSS':
                    return np.sum((oq - pred)**2)
                elif cost == 'MAE':
                    return np.sum(np.abs(oq - pred))
                else:
                    raise ValueError("Unknown cost")
            res = minimize(objective, parini_guess, method='Nelder-Mead')
            par['par'].append(res.x)
        return {'class': 'fitQmapPTF', 'par': par, 'wet_day': wet_day}

    @staticmethod
    def doQmapPTF(x, fobj):
        """
        Apply parametric transformation (PTF) correction.

        Parameters
        ----------
        x : ndarray
            2D data to correct (time, grid).
        fobj : dict
            Fitted object from `fitQmapPTF`.

        Returns
        -------
        ndarray
            Corrected data.

        Raises
        ------
        ValueError
            If unknown transfun in fobj.
        """
        n_cols = x.shape[1]
        corrected = np.zeros_like(x)
        par = fobj['par']
        transfun = par['transfun']
        if callable(transfun):
            tf = transfun
        else:
            trans_funcs = {
                'power': lambda x, a, b: a * x**b,
                'power.x0': lambda x, a, b, x0: a * (x - x0)**b,
                'expasympt': lambda x, a, b, tau: (a + b * x) * (1 - np.exp(-x / tau)),
                'expasympt.x0': lambda x, a, b, x0, tau: (a + b * (x - x0)) * (1 - np.exp(-(x - x0) / tau)),
                'scale': lambda x, b: b * x,
                'linear': lambda x, a, b: a + b * x
            }
            tf = trans_funcs.get(transfun)
            if tf is None:
                raise ValueError("Unknown transfun")
        for col in range(n_cols):
            xi = x[:, col]
            params = par['par'][col]
            th_mod = par['wet_day_th_mod'][col]
            xi_corrected = np.zeros_like(xi)
            mask_below = xi < th_mod
            xi_corrected[mask_below] = 0
            mask_above = ~mask_below
            xi_corrected[mask_above] = tf(xi[mask_above], *params)
            corrected[:, col] = xi_corrected
        return corrected

    @staticmethod
    def fitQmapDIST(obs, mod, distr='berngamma', qstep=None, **kwargs):
        """
        Fit distribution-based quantile mapping (DIST).

        Uses Bernoulli for wet/dry and a continuous distribution for wet values.

        Parameters
        ----------
        obs : ndarray
            2D observed data (time, grid).
        mod : ndarray
            2D modeled data (time, grid).
        distr : str, optional
            Wet distribution: 'berngamma', 'bernexp', 'bernlnorm', 'bernweibull' (default 'berngamma').
        qstep : float, optional
            If set, fit on quantiles; else on full data.
        **kwargs
            Additional arguments (unused).

        Returns
        -------
        dict
            Fitted parameters including distributions and transfer functions.

        Raises
        ------
        ValueError
            If unknown distr.
        """
        n_cols = obs.shape[1]
        par = {'par_o': [], 'par_m': [], 'tfun': [], 'distr': distr}
        dist_map = {
            'berngamma': stats.gamma,
            'bernexp': stats.expon,
            'bernlnorm': stats.lognorm,
            'bernweibull': stats.weibull_min
        }
        dist = dist_map.get(distr)
        if dist is None:
            raise ValueError("Unknown distr")
        for col in range(n_cols):
            o = obs[:, col]
            m = mod[:, col]
            if qstep is not None:
                probs = np.linspace(0, 1, int(1/qstep)+1)
                o = np.quantile(o, probs)
                m = np.quantile(m, probs)
            p_o = np.mean(o > 0)
            p_m = np.mean(m > 0)
            if p_o == 0 or len(o[o > 0]) < 2:
                par_o = {'prob': 0}
            else:
                o_pos = o[o > 0]
                if distr == 'bernlnorm':
                    params_o = dist.fit(o_pos)
                else:
                    params_o = dist.fit(o_pos, floc=0)
                par_o = {'prob': p_o, 'params': params_o}
            if p_m == 0 or len(m[m > 0]) < 2:
                par_m = {'prob': 0}
            else:
                m_pos = m[m > 0]
                if distr == 'bernlnorm':
                    params_m = dist.fit(m_pos)
                else:
                    params_m = dist.fit(m_pos, floc=0)
                par_m = {'prob': p_m, 'params': params_m}
            par['par_o'].append(par_o)
            par['par_m'].append(par_m)
            def tfun(val, par_o=par_o, par_m=par_m, dist=dist):
                if par_m['prob'] == 0:
                    return np.zeros_like(val)
                cdf_m = (1 - par_m['prob']) + par_m['prob'] * dist.cdf(val, *par_m['params'])
                cdf_m = np.clip(cdf_m, 0, 1)
                if par_o['prob'] == 0:
                    return np.zeros_like(cdf_m)
                q_o = np.zeros_like(cdf_m)
                mask_wet = cdf_m > (1 - par_o['prob'])
                u = (cdf_m[mask_wet] - (1 - par_o['prob'])) / par_o['prob']
                q_o[mask_wet] = dist.ppf(u, *par_o['params'])
                return q_o
            par['tfun'].append(tfun)
        return {'class': 'fitQmapDIST', 'par': par}
    
    @staticmethod
    def doQmapDIST(x, fobj):
        """
        Apply distribution-based (DIST) correction.

        Parameters
        ----------
        x : ndarray
            2D data to correct (time, grid).
        fobj : dict
            Fitted object from `fitQmapDIST`.

        Returns
        -------
        ndarray
            Corrected data.
        """
        n_cols = x.shape[1]
        corrected = np.zeros_like(x)
        par = fobj['par']
        for col in range(n_cols):
            xi = x[:, col]
            tfun = par['tfun'][col]
            corrected[:, col] = tfun(xi)
        return corrected

    @staticmethod
    def evaluate_bias_correction(obs, mod, corrected, wet_threshold=0.1, extreme_quantiles=[0.95, 0.99]):
        """
        Evaluate bias correction performance with metrics for dry/wet days and extremes.
        
        Parameters
        ----------
        obs : array_like or xarray.DataArray
            Observed data (numpy array or xarray.DataArray, shape (T, Y, X) or (T,)).
        mod : array_like or xarray.DataArray
            Modeled (uncorrected) data (same shape as obs).
        corrected : array_like or xarray.DataArray
            Bias-corrected data (same shape as obs).
        wet_threshold : float, optional
            Threshold for wet days (default 0.1 mm).
        extreme_quantiles : list of float, optional
            List of quantiles for extremes (default [0.95, 0.99]).
        
        Returns
        -------
        dict or xarray.Dataset
            A dictionary with evaluation metrics, or xarray.Dataset if input is DataArray.
        """
        is_xarray = isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray) and isinstance(corrected, xr.DataArray)
        if is_xarray:
            # Compute metrics along time dimension, return spatial maps
            dry_obs = (obs <= wet_threshold).mean(dim='T')
            dry_mod = (mod <= wet_threshold).mean(dim='T')
            dry_corr = (corrected <= wet_threshold).mean(dim='T')
            
            wet_obs = (obs > wet_threshold).mean(dim='T')
            wet_mod = (mod > wet_threshold).mean(dim='T')
            wet_corr = (corrected > wet_threshold).mean(dim='T')
            
            mean_wet_obs = obs.where(obs > wet_threshold).mean(dim='T')
            mean_wet_mod = mod.where(mod > wet_threshold).mean(dim='T')
            mean_wet_corr = corrected.where(corrected > wet_threshold).mean(dim='T')
            
            ext_obs = obs.quantile(extreme_quantiles, dim='T')
            ext_mod = mod.quantile(extreme_quantiles, dim='T')
            ext_corr = corrected.quantile(extreme_quantiles, dim='T')
            
            ds_dry_wet = xr.Dataset({
                'dry_fraction_obs': dry_obs,
                'dry_fraction_mod': dry_mod,
                'dry_fraction_corr': dry_corr,
                'wet_fraction_obs': wet_obs,
                'wet_fraction_mod': wet_mod,
                'wet_fraction_corr': wet_corr,
                'mean_wet_obs': mean_wet_obs,
                'mean_wet_mod': mean_wet_mod,
                'mean_wet_corr': mean_wet_corr
            })

            ds_extreme = xr.Dataset({
                'extreme_quantiles_obs': ext_obs,
                'extreme_quantiles_mod': ext_mod,
                'extreme_quantiles_corr': ext_corr
            })
            return ds_dry_wet, ds_extreme
        else:
            # For numpy arrays, compute scalar metrics
            obs = np.asarray(obs).flatten()
            mod = np.asarray(mod).flatten()
            corrected = np.asarray(corrected).flatten()
            
            dry_obs = np.mean(obs <= wet_threshold)
            dry_mod = np.mean(mod <= wet_threshold)
            dry_corr = np.mean(corrected <= wet_threshold)
            
            wet_obs = np.mean(obs > wet_threshold)
            wet_mod = np.mean(mod > wet_threshold)
            wet_corr = np.mean(corrected > wet_threshold)
            
            mean_wet_obs = np.mean(obs[obs > wet_threshold]) if np.any(obs > wet_threshold) else np.nan
            mean_wet_mod = np.mean(mod[mod > wet_threshold]) if np.any(mod > wet_threshold) else np.nan
            mean_wet_corr = np.mean(corrected[corrected > wet_threshold]) if np.any(corrected > wet_threshold) else np.nan
            
            ext_obs = np.quantile(obs, extreme_quantiles)
            ext_mod = np.quantile(mod, extreme_quantiles)
            ext_corr = np.quantile(corrected, extreme_quantiles)
            
            return {
                'dry_fraction_obs': dry_obs,
                'dry_fraction_mod': dry_mod,
                'dry_fraction_corr': dry_corr,
                'wet_fraction_obs': wet_obs,
                'wet_fraction_mod': wet_mod,
                'wet_fraction_corr': wet_corr,
                'mean_wet_obs': mean_wet_obs,
                'mean_wet_mod': mean_wet_mod,
                'mean_wet_corr': mean_wet_corr,
                'extreme_quantiles_obs': ext_obs,
                'extreme_quantiles_mod': ext_mod,
                'extreme_quantiles_corr': ext_corr,
                'extreme_quantiles': extreme_quantiles
            }
    @staticmethod
    def _add_basemap(ax, extent=None):
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.LAKES, linewidth=0.3, edgecolor='k', facecolor='none')
        if extent is not None:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
    @staticmethod
    def _collect_vars(ds, prefix):
        return [v for v in ds.data_vars if v.startswith(prefix)]
        
    @staticmethod    
    def plot_fraction_group(ds, group_prefix, extent=None, robust=True):
        """Plots dry/wet fraction groups (e.g., 'dry_fraction_' or 'wet_fraction_')."""
        vars_ = WAS_Qmap._collect_vars(ds, group_prefix)
        if not vars_:
            print(f"No variables found for prefix '{group_prefix}'")
            return
        n = len(vars_)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
    
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            subplot_kw={"projection": ccrs.PlateCarree()},
            figsize=(5*ncols, 4*nrows),
            squeeze=False
        )
        for i, name in enumerate(vars_):
            r, c = divmod(i, ncols)
            ax = axes[r][c]
            da = ds[name]
            im = da.plot(ax=ax, transform=ccrs.PlateCarree(), robust=robust, add_colorbar=True)
            WAS_Qmap._add_basemap(ax, extent)
            ax.set_title(name)
        # hide any empty axes
        for j in range(i+1, nrows*ncols):
            r, c = divmod(j, ncols)
            axes[r][c].axis('off')
        fig.suptitle(f"{group_prefix} variables", fontsize=14)
        plt.tight_layout()
        plt.show()
        
    @staticmethod
    def plot_mean_wet_group(ds, extent=None, robust=True):
        """Plots mean_wet_* variables."""
        vars_ = WAS_Qmap._collect_vars(ds, "mean_wet_")
        if not vars_:
            print("No 'mean_wet_' variables found")
            return
        n = len(vars_)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
    
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            subplot_kw={"projection": ccrs.PlateCarree()},
            figsize=(5*ncols, 4*nrows),
            squeeze=False
        )
        for i, name in enumerate(vars_):
            r, c = divmod(i, ncols)
            ax = axes[r][c]
            da = ds[name]
            da.plot(ax=ax, transform=ccrs.PlateCarree(), robust=robust, add_colorbar=True)
            WAS_Qmap._add_basemap(ax, extent)
            ax.set_title(name)
        for j in range(i+1, nrows*ncols):
            r, c = divmod(j, ncols)
            axes[r][c].axis('off')
        fig.suptitle("mean_wet_* variables", fontsize=14)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_extreme_quantiles_group(ds, extent=None, robust=True):
        """
        Plots extreme_quantiles_* variables, faceting by variable (columns) and quantile (rows).
        Assumes dims ('quantile', 'Y', 'X').
        """
        vars_ = WAS_Qmap._collect_vars(ds, "extreme_quantiles_")
        if not vars_:
            print("No 'extreme_quantiles_' variables found")
            return
    
        qvals = ds.coords.get("quantile")
        if qvals is None:
            raise ValueError("Dataset has no 'quantile' coordinate required for extreme_quantiles_* variables.")
    
        nrows = len(qvals)
        ncols = len(vars_)
    
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            subplot_kw={"projection": ccrs.PlateCarree()},
            figsize=(5*ncols, 4*nrows),
            squeeze=False
        )
    
        for c, var in enumerate(vars_):
            da = ds[var]
            for r, q in enumerate(qvals.values):
                ax = axes[r][c]
                slice_da = da.sel(quantile=q)
                slice_da.plot(ax=ax, transform=ccrs.PlateCarree(), robust=robust, add_colorbar=True)
                WAS_Qmap._add_basemap(ax, extent)
                ttl = f"{var} – q={q}"
                ax.set_title(ttl)
        # Column supertitles
        for c, var in enumerate(vars_):
            axes[0][c].set_title(var, fontsize=12)
        fig.suptitle("extreme_quantiles_* by quantile", fontsize=14)
        plt.tight_layout()
        plt.show()


class WAS_bias_correction:
    """
    Bias correction methods for climate variables such as temperature or wind speed.

    This class provides static methods for fitting and applying bias correction
    techniques suitable for continuous variables that may include negative values
    or skewed positive values, such as mean adjustment, variance scaling, empirical
    quantile mapping (non-parametric), and parametric mapping assuming various
    distributions (normal, lognormal, gamma, weibull). The methods support both
    NumPy arrays (1D, 2D, or 3D) and xarray DataArrays (3D: time, lat, lon or similar).

    Notes
    -----
    - Inputs can be negative and are treated as continuous, but for positive skewed data
      like wind speed, use distributions like 'lognormal', 'gamma', or 'weibull'.
    - No handling for wet/dry days.
    - For gridded data, computations are performed column-wise (per grid cell).
    - xarray support preserves coordinates and attributes.
    - Non-parametric method: 'QUANT' (empirical quantile mapping).
    - Parametric methods: 'NORM' (normal), or 'DIST' with specified distribution.
    - Handles NaNs: Ignores NaNs in fitting by filtering them out; if fewer than 2 valid points
      per grid cell in obs or mod, flags as all_nan and outputs NaNs for that grid in application.
      NaNs in input data during application are propagated as NaNs in output.
    """

    @staticmethod
    def fitBC(obs, mod, method, **kwargs):
        """
        Fit a bias correction model using the specified method.

        Parameters
        ----------
        obs : array_like or xarray.DataArray
            Observed data. If array_like, can be 1D (time), 2D (time, grid), or 3D (time, y, x).
            If xarray.DataArray, must be 3D with dimensions (time, y, x).
        mod : array_like or xarray.DataArray
            Modeled data to fit against, same shape as `obs`.
        method : str
            Bias correction method. Options: 'MEAN', 'VARSCALE', 'QUANT', 'NORM', 'DIST' (case-insensitive).
        **kwargs
            Additional keyword arguments passed to the specific fitting method.
            For 'DIST', include 'distr' (e.g., 'lognormal', 'gamma', 'weibull', default 'normal').

        Returns
        -------
        dict
            Fitted object containing parameters, class identifier, and metadata for applying correction.

        Raises
        ------
        ValueError
            If shapes mismatch, invalid dimensions, or unknown method.

        See Also
        --------
        doBC : Apply the fitted bias correction to new data.
        """
        is_xarray = isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray)
        original_dims = None
        time_dim = None
        spatial_dims = None
        coords = None
        attrs = None
        if is_xarray:
            if obs.shape != mod.shape or len(obs.dims) != 3:
                raise ValueError("xarray DataArrays must be 3D with matching shapes and dimensions (T, Y, X)")
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
        
        method = method.upper()
        if method == 'MEAN':
            fobj = WAS_bias_correction.fitMean(obs_data, mod_data, **kwargs)
        elif method == 'VARSCALE':
            fobj = WAS_bias_correction.fitVarscale(obs_data, mod_data, **kwargs)
        elif method == 'QUANT':
            fobj = WAS_bias_correction.fitQuant(obs_data, mod_data, **kwargs)
        elif method == 'NORM':
            fobj = WAS_bias_correction.fitDist(obs_data, mod_data, distr='normal', **kwargs)
        elif method == 'DIST':
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
        """
        Apply the fitted bias correction to new data.

        Parameters
        ----------
        x : array_like or xarray.DataArray
            New modeled data to correct, same format and shape structure as fitting data.
        fobj : dict
            Fitted object from `fitBC`.
        **kwargs
            Additional keyword arguments passed to the specific application method.

        Returns
        -------
        array_like or xarray.DataArray
            Bias-corrected data, same type and shape as `x`.

        Raises
        ------
        ValueError
            If input types or dimensions mismatch the fitted object.

        See Also
        --------
        fitBC : Fit the bias correction model.
        """
        if 'is_xarray' in fobj and fobj['is_xarray']:
            if not isinstance(x, xr.DataArray):
                raise ValueError("Input x must be xarray.DataArray when fitted with DataArray")
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
                coords={time_dim: x_stacked.coords[time_dim], 'grid': x_stacked.coords['grid']}
            )
            corrected = corrected_stacked.unstack('grid')
            corrected.attrs = fobj['attrs']
            return corrected
        else:
            x_data = WAS_bias_correction._to_2d(x)
            return np.squeeze(WAS_bias_correction._doBC_internal(x_data, fobj, **kwargs))

    @staticmethod
    def _doBC_internal(x, fobj, **kwargs):
        """
        Internal helper to apply bias correction based on fitted class.

        Parameters
        ----------
        x : ndarray
            2D array of data to correct (time, grid).
        fobj : dict
            Fitted object.
        **kwargs
            Additional arguments for specific methods.

        Returns
        -------
        ndarray
            Corrected 2D array.

        Raises
        ------
        ValueError
            If unknown fitted class.
        """
        cls = fobj['class']
        if cls == 'fitMean':
            return WAS_bias_correction.doMean(x, fobj, **kwargs)
        elif cls == 'fitVarscale':
            return WAS_bias_correction.doVarscale(x, fobj, **kwargs)
        elif cls == 'fitQuant':
            return WAS_bias_correction.doQuant(x, fobj, **kwargs)
        elif cls == 'fitDist':
            return WAS_bias_correction.doDist(x, fobj, **kwargs)
        else:
            raise ValueError(f"Unknown class: {cls}")

    @staticmethod
    def _to_2d(arr):
        """
        Convert input array to 2D (time, grid) format.

        Parameters
        ----------
        arr : array_like
            Input array (0D to 3D).

        Returns
        -------
        ndarray
            2D array.

        Raises
        ------
        ValueError
            If more than 3 dimensions.
        """
        arr = np.asarray(arr)
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
    def fitMean(obs, mod):
        """
        Fit mean additive bias correction.

        Parameters
        ----------
        obs : ndarray
            2D observed data (time, grid).
        mod : ndarray
            2D modeled data (time, grid).

        Returns
        -------
        dict
            Fitted parameters including delta (mean difference).
        """
        n_cols = obs.shape[1]
        par = {'delta': np.full(n_cols, np.nan), 'all_nan': np.zeros(n_cols, dtype=bool)}
        for col in range(n_cols):
            o = obs[:, col]
            m = mod[:, col]
            o_valid = o[~np.isnan(o)]
            m_valid = m[~np.isnan(m)]
            if len(o_valid) < 2 or len(m_valid) < 2:
                par['all_nan'][col] = True
                continue
            par['delta'][col] = np.mean(o_valid) - np.mean(m_valid)
        return {'class': 'fitMean', 'par': par}

    @staticmethod
    def doMean(x, fobj):
        """
        Apply mean additive bias correction.

        Parameters
        ----------
        x : ndarray
            2D data to correct (time, grid).
        fobj : dict
            Fitted object from `fitMean`.

        Returns
        -------
        ndarray
            Corrected data.
        """
        n_cols = x.shape[1]
        corrected = np.full_like(x, np.nan)
        par = fobj['par']
        for col in range(n_cols):
            if par['all_nan'][col]:
                continue
            corrected[:, col] = x[:, col] + par['delta'][col]
        return corrected

    @staticmethod
    def fitVarscale(obs, mod):
        """
        Fit variance scaling bias correction.

        Parameters
        ----------
        obs : ndarray
            2D observed data (time, grid).
        mod : ndarray
            2D modeled data (time, grid).

        Returns
        -------
        dict
            Fitted parameters including means and standard deviations.
        """
        n_cols = obs.shape[1]
        par = {'mean_o': np.full(n_cols, np.nan), 'std_o': np.full(n_cols, np.nan),
               'mean_m': np.full(n_cols, np.nan), 'std_m': np.full(n_cols, np.nan),
               'all_nan': np.zeros(n_cols, dtype=bool)}
        for col in range(n_cols):
            o = obs[:, col]
            m = mod[:, col]
            o_valid = o[~np.isnan(o)]
            m_valid = m[~np.isnan(m)]
            if len(o_valid) < 2 or len(m_valid) < 2:
                par['all_nan'][col] = True
                continue
            par['mean_o'][col] = np.mean(o_valid)
            par['std_o'][col] = np.std(o_valid)
            par['mean_m'][col] = np.mean(m_valid)
            std_m = np.std(m_valid)
            par['std_m'][col] = std_m if std_m > 1e-6 else 1.0
        return {'class': 'fitVarscale', 'par': par}

    @staticmethod
    def doVarscale(x, fobj):
        """
        Apply variance scaling bias correction.

        Parameters
        ----------
        x : ndarray
            2D data to correct (time, grid).
        fobj : dict
            Fitted object from `fitVarscale`.

        Returns
        -------
        ndarray
            Corrected data.
        """
        n_cols = x.shape[1]
        corrected = np.full_like(x, np.nan)
        par = fobj['par']
        for col in range(n_cols):
            if par['all_nan'][col]:
                continue
            corrected[:, col] = par['mean_o'][col] + (x[:, col] - par['mean_m'][col]) * (par['std_o'][col] / par['std_m'][col])
        return corrected

    @staticmethod
    def fitQuant(obs, mod, qstep=0.01, nboot=1):
        """
        Fit empirical quantile mapping (non-parametric).

        Parameters
        ----------
        obs : ndarray
            2D observed data (time, grid).
        mod : ndarray
            2D modeled data (time, grid).
        qstep : float, optional
            Quantile step size (default 0.01).
        nboot : int, optional
            Number of bootstrap samples for observed quantiles (default 1, no bootstrap).

        Returns
        -------
        dict
            Fitted parameters including quantiles.
        """
        n_cols = obs.shape[1]
        nq = int(1 / qstep) + 1
        par = {'modq': np.full((nq, n_cols), np.nan), 'fitq': np.full((nq, n_cols), np.nan),
               'all_nan': np.zeros(n_cols, dtype=bool)}
        probs = np.linspace(0, 1, nq)
        for col in range(n_cols):
            o = obs[:, col]
            m = mod[:, col]
            o_valid = o[~np.isnan(o)]
            m_valid = m[~np.isnan(m)]
            if len(o_valid) < 2 or len(m_valid) < 2:
                par['all_nan'][col] = True
                continue
            par['modq'][:, col] = np.quantile(m_valid, probs)
            if nboot > 1:
                boot_q = np.array([np.quantile(np.random.choice(o_valid, len(o_valid), replace=True), probs) for _ in range(nboot)])
                par['fitq'][:, col] = np.mean(boot_q, axis=0)
            else:
                par['fitq'][:, col] = np.quantile(o_valid, probs)
        return {'class': 'fitQuant', 'par': par}

    @staticmethod
    def doQuant(x, fobj, type='linear'):
        """
        Apply empirical quantile mapping correction (non-parametric).

        Parameters
        ----------
        x : ndarray
            2D data to correct (time, grid).
        fobj : dict
            Fitted object from `fitQuant`.
        type : str, optional
            Interpolation kind for `interp1d` (default 'linear').

        Returns
        -------
        ndarray
            Corrected data.
        """
        n_cols = x.shape[1]
        corrected = np.full_like(x, np.nan)
        par = fobj['par']
        for col in range(n_cols):
            if par['all_nan'][col]:
                continue
            xi = x[:, col]
            modq = par['modq'][:, col]
            fitq = par['fitq'][:, col]
            interp_func = interp1d(modq, fitq, kind=type, bounds_error=False, fill_value='extrapolate')
            corrected[:, col] = interp_func(xi)
        return corrected

    @staticmethod
    def fitDist(obs, mod, distr='normal'):
        """
        Fit parametric bias correction assuming a specified distribution.

        Suitable for skewed data like wind speed with 'lognormal', 'gamma', or 'weibull'.

        Parameters
        ----------
        obs : ndarray
            2D observed data (time, grid).
        mod : ndarray
            2D modeled data (time, grid).
        distr : str, optional
            Distribution: 'normal', 'lognormal', 'gamma', 'weibull' (default 'normal').

        Returns
        -------
        dict
            Fitted parameters including distribution parameters.

        Raises
        ------
        ValueError
            If unknown distribution.
        """
        n_cols = obs.shape[1]
        par = {'par_o': [], 'par_m': [], 'distr': distr,
               'all_nan': np.zeros(n_cols, dtype=bool)}
        dist_map = {
            'normal': norm,
            'lognormal': lognorm,
            'gamma': gamma,
            'weibull': weibull_min
        }
        dist = dist_map.get(distr.lower())
        if dist is None:
            raise ValueError(f"Unknown distribution: {distr}")
        for col in range(n_cols):
            o = obs[:, col]
            m = mod[:, col]
            o_valid = o[~np.isnan(o)]
            m_valid = m[~np.isnan(m)]
            if len(o_valid) < 2 or len(m_valid) < 2:
                par['all_nan'][col] = True
                par['par_o'].append(None)
                par['par_m'].append(None)
                continue
            if distr.lower() == 'normal':
                par_o = dist.fit(o_valid)
                par_m = dist.fit(m_valid)
            elif distr.lower() == 'lognormal':
                par_o = dist.fit(o_valid)
                par_m = dist.fit(m_valid)
            elif distr.lower() == 'gamma':
                par_o = dist.fit(o_valid, floc=0)
                par_m = dist.fit(m_valid, floc=0)
            elif distr.lower() == 'weibull':
                par_o = dist.fit(o_valid, floc=0)
                par_m = dist.fit(m_valid, floc=0)
            par['par_o'].append(par_o)
            par['par_m'].append(par_m)
        return {'class': 'fitDist', 'par': par}

    @staticmethod
    def doDist(x, fobj):
        """
        Apply parametric distribution bias correction.

        Parameters
        ----------
        x : ndarray
            2D data to correct (time, grid).
        fobj : dict
            Fitted object from `fitDist`.

        Returns
        -------
        ndarray
            Corrected data.
        """
        n_cols = x.shape[1]
        corrected = np.full_like(x, np.nan)
        par = fobj['par']
        distr = par['distr'].lower()
        dist_map = {
            'normal': norm,
            'lognormal': lognorm,
            'gamma': gamma,
            'weibull': weibull_min
        }
        dist = dist_map.get(distr)
        if dist is None:
            raise ValueError(f"Unknown distribution: {distr}")
        for col in range(n_cols):
            if par['all_nan'][col]:
                continue
            par_o = par['par_o'][col]
            par_m = par['par_m'][col]
            cdf = dist.cdf(x[:, col], *par_m)
            cdf = np.clip(cdf, 0, 1)
            corrected[:, col] = dist.ppf(cdf, *par_o)
        return corrected