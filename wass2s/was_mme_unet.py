# -*- coding: utf-8 -*-
"""Scientific U-Net post-processor for WASS2S multi-model ensembles.

The class :class:`WAS_mme_UNet` is designed as a spatially coherent nonlinear
calibration model for seasonal MME fields.  It follows the public contract used
by WASS2S MME regressors::

    best_params, cluster = model.compute_hyperparameters(Predictor, Predictant,
                                                          clim_start, clim_end)
    pred = model.compute_model(X_train, y_train, X_test, y_test,
                               best_params=best_params, cluster_da=cluster)
    prob = model.compute_prob(Predictant, clim_start, clim_end, hindcast_det)
    fdet, fprob = model.forecast(Predictant, clim_start, clim_end,
                                 Predictors_train, hindcast_det_cross,
                                 Predictor_for_year,
                                 best_params=best_params, cluster_da=cluster)

Scientific safeguards
---------------------
* U-Net decoder channel dimensions are explicit and valid.
* Inputs are dimension-safe: X -> (T,M,Y,X), y -> (T,Y,X).
* xarray coordinates are checked/aligned before converting to NumPy.
* Missing predictor values can be supplied to the network as explicit mask
  channels, rather than silently treating missing == climatological zero.
* GroupNorm is used instead of BatchNorm (more stable for short hindcasts and
  small temporal batches).
* Temporal validation is blocked and never randomly split.
* Optional hyperparameter tuning is performed by inner temporal folds and is
  therefore compatible with nested outer cross-validation.
* Hindcast non-parametric probabilities are leave-one-year-out residual-dressed:
  the residual of year t is excluded when producing probabilities for year t.
* Operational probabilities use only genuinely out-of-sample hindcast residuals.
* Reproducibility controls seed NumPy/Python/PyTorch/CUDA.

Corrections of 15 September 2026
--------------------------------
* Default ``norm='none'``. GroupNorm normalises every feature map over space for each year. A
  spatially uniform predictor anomaly, or a change of its amplitude, is then
  invisible to the network (only padding/mask edges leak it). The network now
  has a 1x1 linear skip from the inputs to the output (``linear_skip=True``)
  and the normalisation can be disabled (``norm='none'``).
* Optional static channels (normalised latitude/longitude and the training
  climatological mean/log-std) let the mapping vary in space.
* ``n_seeds`` trains an ensemble of independently initialised networks and
  averages them; the stopping epoch is selected per seed.
* ``refit_after_stopping=False`` keeps the early-stopped networks (half cost).
* Training batches live on the device and losses are synchronised once per
  epoch; ``mixed_precision`` uses bfloat16 when the GPU supports it.

Important interpretation
------------------------
The U-Net produces a deterministic spatial calibration.  Tercile uncertainty is
estimated separately from cross-validated residuals; the neural network itself
is not a probabilistic neural network.
"""

from __future__ import annotations

import copy
import itertools
import math
import random
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr
from .was_probability_reference import climatological_terciles

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None
    DataLoader = None
    TensorDataset = None
    HAS_TORCH = False

from scipy.optimize import fsolve
from scipy.special import gamma as gamma_function
from scipy.stats import expon, gamma as gamma_dist, lognorm, nbinom, norm, poisson
from scipy.stats import t as student_t
from scipy.stats import weibull_min


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def _set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        else:
            try:
                torch.use_deterministic_algorithms(False)
            except Exception:
                pass
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = False
                torch.backends.cudnn.benchmark = True


def _drop_member(da: xr.DataArray) -> xr.DataArray:
    """Return a strict (T,Y,X) target field."""
    if not isinstance(da, xr.DataArray):
        raise TypeError("Expected an xarray.DataArray.")
    if "M" in da.dims:
        if da.sizes["M"] > 1:
            warnings.warn(
                "Predictand has more than one M member; the first member is used as observation.",
                RuntimeWarning,
                stacklevel=2,
            )
        da = da.isel(M=0, drop=True)
    missing = {"T", "Y", "X"} - set(da.dims)
    if missing:
        raise ValueError(f"Predictand is missing dimensions {sorted(missing)}; dims={da.dims}.")
    extras = [d for d in da.dims if d not in ("T", "Y", "X")]
    for d in extras:
        if da.sizes[d] != 1:
            raise ValueError(f"Unsupported predictand dimension {d!r} with size {da.sizes[d]}.")
        da = da.isel({d: 0}, drop=True)
    return da.transpose("T", "Y", "X")


def _as_predictor(da: xr.DataArray) -> xr.DataArray:
    """Return a strict (T,M,Y,X) predictor cube.

    ``M`` is interpreted as predictor/model channels.  For exchangeable raw
    ensemble members, users should first aggregate members (e.g. ensemble mean
    and optionally spread) rather than assign a persistent identity to member 1,
    member 2, ... .
    """
    if not isinstance(da, xr.DataArray):
        raise TypeError("Predictor must be an xarray.DataArray.")
    missing = {"T", "Y", "X"} - set(da.dims)
    if missing:
        raise ValueError(f"Predictor is missing dimensions {sorted(missing)}; dims={da.dims}.")
    if "M" not in da.dims:
        da = da.expand_dims(M=["predictor_0"])
    extras = [d for d in da.dims if d not in ("T", "M", "Y", "X")]
    for d in extras:
        if da.sizes[d] != 1:
            raise ValueError(f"Unsupported predictor dimension {d!r} with size {da.sizes[d]}.")
        da = da.isel({d: 0}, drop=True)
    return da.transpose("T", "M", "Y", "X")


def _same_coord(a: xr.DataArray, b: xr.DataArray, dim: str) -> bool:
    return (
        dim in a.coords
        and dim in b.coords
        and a.sizes.get(dim) == b.sizes.get(dim)
        and np.array_equal(np.asarray(a[dim].values), np.asarray(b[dim].values))
    )


def _require_same_grid(a: xr.DataArray, b: xr.DataArray, a_name="A", b_name="B") -> None:
    for d in ("Y", "X"):
        if not _same_coord(a, b, d):
            raise ValueError(
                f"{a_name} and {b_name} do not share the exact {d} grid. "
                "Regrid/reindex upstream before U-Net calibration."
            )


def _align_training(X: xr.DataArray, y: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
    """Align training data exactly on T/Y/X while preserving M."""
    X = _as_predictor(X)
    y = _drop_member(y)
    _require_same_grid(X, y, "X_train", "y_train")

    # Annual pairing requires identical ordered verification years, not merely
    # equally sized arrays. A missing year must never shift all later targets.
    from wass2s.was_scientific_utils import align_predictor_target
    X = align_predictor_target(X, y, mode="season_year")
    if X.sizes.get("T", 0) < 5:
        raise ValueError("At least five paired years are required for U-Net fitting.")
    return X.transpose("T", "M", "Y", "X"), y.transpose("T", "Y", "X")


def _clim_subset(da: xr.DataArray, start, end, min_n: int = 2) -> xr.DataArray:
    if start is None or end is None:
        return da
    try:
        q = da.sel(T=slice(str(start), str(end)))
        if q.sizes.get("T", 0) >= min_n:
            return q
    except Exception:
        pass
    return da


def _fit_standardizer(da: xr.DataArray, start=None, end=None) -> Tuple[xr.DataArray, xr.DataArray]:
    clim = _clim_subset(da, start, end, min_n=2)
    mean = clim.mean("T", skipna=True)
    std = clim.std("T", skipna=True)
    std = std.where(np.isfinite(std) & (std > 1.0e-12), 1.0)
    return mean, std


def _standardize_with(da: xr.DataArray, mean: xr.DataArray, std: xr.DataArray) -> xr.DataArray:
    return (da - mean) / std


def _spatial_mask(y: xr.DataArray, min_valid_years: int = 3) -> xr.DataArray:
    return xr.where(np.isfinite(y).sum("T") >= int(min_valid_years), 1.0, np.nan)


def _conditional_nonnegative_clip(pred: xr.DataArray, obs: xr.DataArray) -> xr.DataArray:
    """Use declared physical support, independently of validation values."""
    from wass2s.was_scientific_utils import apply_physical_support
    return apply_physical_support(pred, obs)


def _assign_target_time(da: xr.DataArray, predictor_for_year: xr.DataArray,
                        predictand: xr.DataArray) -> xr.DataArray:
    if "T" not in da.dims or da.sizes["T"] != 1:
        return da
    try:
        candidate = predictor_for_year["T"].values[0]
        year = int(np.asarray(candidate).astype("datetime64[Y]").astype(int) + 1970)
        obs_t = predictand["T"].values[0]
        month = int(np.asarray(obs_t).astype("datetime64[M]").astype(int) % 12 + 1)
        target = np.datetime64(f"{year:04d}-{month:02d}-01", "ns")
        return da.assign_coords(T=xr.IndexVariable("T", [target]))
    except Exception:
        return da


def _time_blocks(n: int, n_folds: int = 3, buffer: int = 0,
                 min_train: int = 6) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Blocked inner CV; validation blocks cover time without random shuffling."""
    n = int(n)
    if n < min_train + 1:
        return []
    k = max(2, min(int(n_folds), max(2, n // 3)))
    idx = np.arange(n)
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for va in np.array_split(idx, k):
        if va.size == 0:
            continue
        lo = max(0, int(va.min()) - int(buffer))
        hi = min(n, int(va.max()) + int(buffer) + 1)
        excluded = np.arange(lo, hi)
        tr = np.setdiff1d(idx, excluded)
        if tr.size >= min_train:
            folds.append((tr, va))
    return folds


# -----------------------------------------------------------------------------
# U-Net architecture
# -----------------------------------------------------------------------------
if HAS_TORCH:

    def _group_count(channels: int, max_groups: int = 8) -> int:
        g = min(int(max_groups), int(channels))
        while g > 1 and channels % g != 0:
            g -= 1
        return max(1, g)


    def _norm_layer(norm: str, channels: int, groups: int):
        if norm == "group":
            return nn.GroupNorm(_group_count(channels, groups), channels)
        if norm == "none":
            return nn.Identity()
        raise ValueError("norm must be 'group' or 'none'.")


    class DoubleConv(nn.Module):
        """Conv -> [GroupNorm] -> SiLU -> Dropout, repeated twice."""

        def __init__(self, in_channels: int, out_channels: int,
                     dropout: float = 0.0, groups: int = 8, norm: str = "group"):
            super().__init__()
            bias = norm == "none"
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=bias),
                _norm_layer(norm, out_channels, groups),
                nn.SiLU(inplace=True),
                nn.Dropout2d(float(dropout)),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=bias),
                _norm_layer(norm, out_channels, groups),
                nn.SiLU(inplace=True),
                nn.Dropout2d(float(dropout)),
            )

        def forward(self, x):
            return self.net(x)


    class Down(nn.Module):
        def __init__(self, in_channels: int, out_channels: int,
                     dropout: float = 0.0, groups: int = 8, norm: str = "group"):
            super().__init__()
            self.net = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels, dropout, groups, norm),
            )

        def forward(self, x):
            return self.net(x)


    class Up(nn.Module):
        """Correct decoder block with explicit decoder + skip channel counts."""

        def __init__(self, decoder_channels: int, skip_channels: int,
                     out_channels: int, bilinear: bool = True,
                     dropout: float = 0.0, groups: int = 8, norm: str = "group"):
            super().__init__()
            self.bilinear = bool(bilinear)
            if self.bilinear:
                self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                up_channels = decoder_channels
            else:
                self.up = nn.ConvTranspose2d(
                    decoder_channels, out_channels, kernel_size=2, stride=2
                )
                up_channels = out_channels
            self.conv = DoubleConv(
                up_channels + skip_channels, out_channels, dropout, groups, norm
            )

        def forward(self, x_decoder, x_skip):
            x_decoder = self.up(x_decoder)
            dy = x_skip.size(2) - x_decoder.size(2)
            dx = x_skip.size(3) - x_decoder.size(3)
            if dx != 0 or dy != 0:
                # Positive differences are padded; negative differences are
                # symmetrically cropped.  In normal MaxPool/Upsample use the
                # differences are 0 or 1.
                if dy < 0:
                    top = (-dy) // 2
                    x_decoder = x_decoder[:, :, top: top + x_skip.size(2), :]
                    dy = 0
                if dx < 0:
                    left = (-dx) // 2
                    x_decoder = x_decoder[:, :, :, left: left + x_skip.size(3)]
                    dx = 0
                if dx > 0 or dy > 0:
                    x_decoder = F.pad(
                        x_decoder,
                        [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2],
                    )
            x = torch.cat([x_skip, x_decoder], dim=1)
            return self.conv(x)


    class UNet(nn.Module):
        """Compact U-Net suited to short seasonal hindcast records."""

        def __init__(self, n_channels: int, n_classes: int = 1,
                     filters: Sequence[int] = (8, 16, 32),
                     bilinear: bool = True, dropout: float = 0.2,
                     norm_groups: int = 8, norm: str = "group",
                     linear_skip: bool = True):
            super().__init__()
            filters = tuple(int(v) for v in filters)
            if len(filters) < 1 or any(v < 1 for v in filters):
                raise ValueError("filters must contain positive integers.")
            self.filters = filters
            self.inc = DoubleConv(n_channels, filters[0], dropout, norm_groups, norm)
            self.downs = nn.ModuleList(
                Down(filters[i - 1], filters[i], dropout, norm_groups, norm)
                for i in range(1, len(filters))
            )
            bottleneck_channels = filters[-1] * 2
            self.bottleneck = Down(
                filters[-1], bottleneck_channels, dropout, norm_groups, norm
            )

            self.ups = nn.ModuleList()
            decoder_channels = bottleneck_channels
            for skip_channels in reversed(filters):
                out_channels = skip_channels
                self.ups.append(
                    Up(
                        decoder_channels,
                        skip_channels,
                        out_channels,
                        bilinear=bilinear,
                        dropout=dropout,
                        groups=norm_groups,
                        norm=norm,
                    )
                )
                decoder_channels = out_channels
            self.outc = nn.Conv2d(filters[0], n_classes, kernel_size=1)
            # Linear path that bypasses every normalisation: the absolute
            # amplitude and the domain-mean of each input channel reach the
            # output (a per-channel MOS baseline, corrected non-linearly).
            self.linear_skip = (nn.Conv2d(n_channels, n_classes, kernel_size=1)
                                if linear_skip else None)

        def forward(self, x):
            skips = [self.inc(x)]
            for down in self.downs:
                skips.append(down(skips[-1]))
            h = self.bottleneck(skips[-1])
            for up, skip in zip(self.ups, reversed(skips)):
                h = up(h, skip)
            out = self.outc(h)
            if self.linear_skip is not None:
                out = out + self.linear_skip(x)
            return out


    class SeedEnsemble(nn.Module):
        """Average of independently initialised networks (same inputs)."""

        def __init__(self, members):
            super().__init__()
            self.members = nn.ModuleList(members)

        def forward(self, x):
            return torch.stack([member(x) for member in self.members], dim=0).mean(0)


# -----------------------------------------------------------------------------
# Probability kernels
# -----------------------------------------------------------------------------
def _normalize_probability_da(prob: xr.DataArray) -> xr.DataArray:
    prob = prob.clip(min=0.0, max=1.0)
    total = prob.sum("probability", skipna=False)
    return xr.where(total > 0, prob / total, np.nan)


def _empirical_prob_loo(pred: np.ndarray, obs: np.ndarray, hind: np.ndarray,
                        t1: float, t2: float) -> np.ndarray:
    """LOO residual dressing for hindcast probabilities at one grid cell."""
    pred = np.asarray(pred, float)
    obs = np.asarray(obs, float)
    hind = np.asarray(hind, float)
    nt = pred.size
    out = np.full((3, nt), np.nan, dtype=float)
    residuals = obs - hind
    for i in range(nt):
        if not np.isfinite(pred[i]) or not np.isfinite(t1) or not np.isfinite(t2):
            continue
        keep = np.isfinite(residuals)
        if i < keep.size:
            keep[i] = False
        e = residuals[keep]
        if e.size < 2:
            continue
        dressed = pred[i] + e
        pb = np.mean(dressed <= t1)
        pn = np.mean((dressed > t1) & (dressed <= t2))
        out[:, i] = (pb, pn, 1.0 - pb - pn)
    return out


def _empirical_prob_forecast(pred: np.ndarray, residuals: np.ndarray,
                             t1: float, t2: float) -> np.ndarray:
    pred = np.asarray(pred, float)
    residuals = np.asarray(residuals, float)
    nt = pred.size
    out = np.full((3, nt), np.nan, dtype=float)
    e = residuals[np.isfinite(residuals)]
    if e.size < 2:
        return out
    for i in range(nt):
        if not np.isfinite(pred[i]) or not np.isfinite(t1) or not np.isfinite(t2):
            continue
        dressed = pred[i] + e
        pb = np.mean(dressed <= t1)
        pn = np.mean((dressed > t1) & (dressed <= t2))
        out[:, i] = (pb, pn, 1.0 - pb - pn)
    return out


def _ppf_terciles_from_code(dist_code, shape, loc, scale):
    if not np.isfinite(dist_code):
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
            return gamma_dist.ppf(1/3, a=shape, loc=loc, scale=scale), gamma_dist.ppf(2/3, a=shape, loc=loc, scale=scale)
        if code == 5:
            return weibull_min.ppf(1/3, c=shape, loc=loc, scale=scale), weibull_min.ppf(2/3, c=shape, loc=loc, scale=scale)
        if code == 6:
            return student_t.ppf(1/3, df=shape, loc=loc, scale=scale), student_t.ppf(2/3, df=shape, loc=loc, scale=scale)
        if code == 7:
            return poisson.ppf(1/3, mu=shape, loc=loc), poisson.ppf(2/3, mu=shape, loc=loc)
        if code == 8:
            return nbinom.ppf(1/3, n=shape, p=scale, loc=loc), nbinom.ppf(2/3, n=shape, p=scale, loc=loc)
    except Exception:
        pass
    return np.nan, np.nan


def _weibull_shape_solver(k, mean, variance):
    if k <= 0 or mean <= 0:
        return np.inf
    g1 = gamma_function(1.0 + 1.0 / k)
    g2 = gamma_function(1.0 + 2.0 / k)
    return (variance / (mean * mean)) - (g2 / (g1 * g1) - 1.0)


def _bestfit_prob_vector(pred: np.ndarray, variances: np.ndarray, t1: float, t2: float,
                         dist_code: float, dof: int) -> np.ndarray:
    """Parametric dressing for one grid cell with time-varying OOS variance."""
    pred = np.asarray(pred, float)
    var = np.asarray(variances, float)
    if var.ndim == 0:
        var = np.full(pred.size, float(var))
    out = np.full((3, pred.size), np.nan, float)
    if not np.isfinite(dist_code) or not np.isfinite(t1) or not np.isfinite(t2):
        return out
    code = int(dist_code)

    for i, (mu, v) in enumerate(zip(pred, var)):
        if not np.isfinite(mu) or not np.isfinite(v) or v <= 0:
            continue
        try:
            if code == 1:
                s = math.sqrt(v)
                c1, c2 = norm.cdf(t1, loc=mu, scale=s), norm.cdf(t2, loc=mu, scale=s)
            elif code == 2:
                if mu <= 0:
                    continue
                sigma = math.sqrt(math.log1p(v / (mu * mu)))
                logmu = math.log(mu) - 0.5 * sigma * sigma
                c1, c2 = lognorm.cdf(t1, s=sigma, scale=math.exp(logmu)), lognorm.cdf(t2, s=sigma, scale=math.exp(logmu))
            elif code == 3:
                s = math.sqrt(v)
                loc = mu - s
                c1, c2 = expon.cdf(t1, loc=loc, scale=s), expon.cdf(t2, loc=loc, scale=s)
            elif code == 4:
                if mu <= 0:
                    continue
                alpha = mu * mu / v
                theta = v / mu
                c1, c2 = gamma_dist.cdf(t1, a=alpha, scale=theta), gamma_dist.cdf(t2, a=alpha, scale=theta)
            elif code == 5:
                if mu <= 0:
                    continue
                k = float(fsolve(_weibull_shape_solver, 2.0, args=(mu, v))[0])
                if not np.isfinite(k) or k <= 0:
                    continue
                lam = mu / gamma_function(1.0 + 1.0 / k)
                c1, c2 = weibull_min.cdf(t1, c=k, scale=lam), weibull_min.cdf(t2, c=k, scale=lam)
            elif code == 6:
                if dof <= 2:
                    continue
                s = math.sqrt(v * (dof - 2) / dof)
                c1, c2 = student_t.cdf(t1, df=dof, loc=mu, scale=s), student_t.cdf(t2, df=dof, loc=mu, scale=s)
            elif code == 7:
                if mu < 0:
                    continue
                c1, c2 = poisson.cdf(t1, mu=mu), poisson.cdf(t2, mu=mu)
            elif code == 8:
                if mu <= 0 or v <= mu:
                    continue
                p = mu / v
                n = mu * mu / (v - mu)
                c1, c2 = nbinom.cdf(t1, n=n, p=p), nbinom.cdf(t2, n=n, p=p)
            else:
                continue
            out[:, i] = (c1, c2 - c1, 1.0 - c2)
        except Exception:
            continue
    return np.clip(out, 0.0, 1.0)


def _loo_mse_variance(obs: xr.DataArray, hind: xr.DataArray) -> xr.DataArray:
    """For each target year, residual MSE estimated from all OTHER years."""
    r2 = (obs - hind) ** 2
    finite = np.isfinite(r2)
    total = r2.where(finite, 0.0).sum("T")
    count = finite.sum("T")
    numerator = total - r2.where(finite, 0.0)
    denominator = count - finite.astype(int)
    return (numerator / denominator.where(denominator > 0)).transpose("T", "Y", "X")


# -----------------------------------------------------------------------------
# Training state
# -----------------------------------------------------------------------------
@dataclass
class _FitResult:
    model: Any
    history: Dict[str, List[float]]
    best_val_loss: float
    epochs_ran: int
    input_channels: int
    best_epoch: int
    training_samples: int
    static: Any = None


# -----------------------------------------------------------------------------
# Main class
# -----------------------------------------------------------------------------
class WAS_mme_UNet:
    """Spatial U-Net calibration for WASS2S MME predictors.

    Parameters
    ----------
    filters : tuple[int]
        Small encoder width. ``(8,16,32)`` is intentionally conservative for
        ~20-30 independent seasonal years.
    dropout : float
        Spatial dropout after each convolution.
    lr : float
        AdamW learning rate.
    weight_decay : float
        L2/AdamW regularisation.
    batch_size : int
        Number of years per temporal batch.
    epochs : int
        Maximum epochs for final/fold fits.
    early_stopping_patience : int
        Early-stopping patience.
    validation_fraction : float
        Final blocked fraction of the training years used for early stopping.
    add_missing_mask : bool
        Append one binary channel per predictor channel indicating finite input.
    loss : {'huber','mse'}
        Huber is the default because seasonal rainfall extremes can dominate MSE.
    dist_method : {'nonparam','bestfit'}
        ``nonparam`` is recommended. Hindcast nonparametric probabilities are
        leave-one-year-out residual dressed.
    tune_hyperparameters : bool
        If True, ``compute_hyperparameters`` performs compact blocked temporal
        inner CV. Keep False when using a prespecified architecture.
    """

    _was_unet = True
    _was_spatial_cnn = True
    _network_name = "UNet"

    def __init__(
        self,
        filters: Tuple[int, ...] = (8, 16, 32),
        dropout: float = 0.20,
        lr: float = 1.0e-3,
        weight_decay: float = 1.0e-4,
        batch_size: int = 4,
        epochs: int = 80,
        early_stopping_patience: int = 10,
        validation_fraction: float = 0.20,
        bilinear: bool = True,
        norm_groups: int = 8,
        add_missing_mask: bool = True,
        fill_value: float = 0.0,
        loss: str = "huber",
        huber_delta: float = 1.0,
        gradient_clip: float = 1.0,
        device: Optional[str] = None,
        dist_method: str = "nonparam",
        random_state: int = 42,
        deterministic: bool = True,
        num_workers: int = 0,
        pin_memory: Optional[bool] = None,
        mixed_precision: bool = False,
        min_valid_years: int = 3,
        tune_hyperparameters: bool = False,
        tuning_filters: Sequence[Tuple[int, ...]] = ((8, 16), (8, 16, 32)),
        tuning_dropouts: Sequence[float] = (0.10, 0.20),
        tuning_lrs: Sequence[float] = (5.0e-4, 1.0e-3),
        tuning_weight_decays: Sequence[float] = (1.0e-4,),
        tuning_folds: int = 3,
        tuning_epochs: int = 30,
        tuning_patience: int = 5,
        max_tuning_candidates: int = 6,
        temporal_buffer: int = 0,
        verbose: int = 0,
        norm: str = "none",
        linear_skip: bool = True,
        add_static_channels: bool = True,
        n_seeds: int = 1,
        refit_after_stopping: bool = True,
        tuning_seeds: int = 1,
    ):
        if not HAS_TORCH:
            raise ImportError("WAS_mme_UNet requires PyTorch (torch).")
        if dist_method not in {"nonparam", "bestfit"}:
            raise ValueError("dist_method must be 'nonparam' or 'bestfit'.")
        if loss not in {"huber", "mse"}:
            raise ValueError("loss must be 'huber' or 'mse'.")
        if norm not in {"group", "none"}:
            raise ValueError("norm must be 'group' or 'none'.")
        if int(n_seeds) < 1 or int(tuning_seeds) < 1:
            raise ValueError("n_seeds and tuning_seeds must be positive.")

        self.filters = tuple(int(v) for v in filters)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.early_stopping_patience = int(early_stopping_patience)
        self.validation_fraction = float(validation_fraction)
        self.bilinear = bool(bilinear)
        self.norm_groups = int(norm_groups)
        self.add_missing_mask = bool(add_missing_mask)
        self.fill_value = float(fill_value)
        self.loss = loss
        self.huber_delta = float(huber_delta)
        self.gradient_clip = float(gradient_clip)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dist_method = dist_method
        self.random_state = int(random_state)
        self.deterministic = bool(deterministic)
        self.num_workers = int(num_workers)
        self.pin_memory = (self.device.startswith("cuda") if pin_memory is None else bool(pin_memory))
        self.mixed_precision = bool(mixed_precision and self.device.startswith("cuda"))
        self.min_valid_years = int(min_valid_years)
        self.tune_hyperparameters = bool(tune_hyperparameters)
        self.tuning_filters = tuple(tuple(int(v) for v in f) for f in tuning_filters)
        self.tuning_dropouts = tuple(float(v) for v in tuning_dropouts)
        self.tuning_lrs = tuple(float(v) for v in tuning_lrs)
        self.tuning_weight_decays = tuple(float(v) for v in tuning_weight_decays)
        self.tuning_folds = int(tuning_folds)
        self.tuning_epochs = int(tuning_epochs)
        self.tuning_patience = int(tuning_patience)
        self.max_tuning_candidates = int(max_tuning_candidates)
        self.temporal_buffer = int(temporal_buffer)
        self.verbose = int(verbose)
        self.norm = str(norm)
        self.linear_skip = bool(linear_skip)
        self.add_static_channels = bool(add_static_channels)
        self.n_seeds = int(n_seeds)
        self.refit_after_stopping = bool(refit_after_stopping)
        self.tuning_seeds = int(tuning_seeds)

        self.model = None
        self.history = None
        self.best_params_: Optional[Dict[str, Any]] = None
        self.tuning_results_: List[Dict[str, Any]] = []
        _set_global_seed(self.random_state, self.deterministic)

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    N_STATIC_CHANNELS = 4

    def _static_channels(self, template: xr.DataArray, clim_mean=None, clim_std=None):
        """Latitude, longitude, climatological mean and log-std (domain z-scores).

        The two climatology channels come from TRAINING observations only; they
        are zero when unavailable so the channel count never changes.
        """
        if not self.add_static_channels:
            return None
        ny, nx = int(template.sizes["Y"]), int(template.sizes["X"])

        def unit(values):
            values = np.asarray(values, dtype=float)
            span = np.nanmax(values) - np.nanmin(values) if values.size else 0.0
            if not np.isfinite(span) or span <= 0:
                return np.zeros_like(values)
            return 2.0 * (values - np.nanmin(values)) / span - 1.0

        lat = np.broadcast_to(unit(template["Y"].values)[:, None], (ny, nx))
        lon = np.broadcast_to(unit(template["X"].values)[None, :], (ny, nx))
        channels = [lat, lon]
        for field, logarithm in ((clim_mean, False), (clim_std, True)):
            if field is None:
                channels.append(np.zeros((ny, nx)))
                continue
            extra = [d for d in field.dims if d not in ("Y", "X")]
            if any(field.sizes[d] != 1 for d in extra):
                raise ValueError("Static climatology must be a (Y, X) field.")
            arr = np.asarray(field.isel({d: 0 for d in extra}).transpose("Y", "X").values,
                             dtype=float)
            if arr.shape != (ny, nx):
                raise ValueError("Static climatology grid differs from the predictor grid.")
            if logarithm:
                with np.errstate(divide="ignore", invalid="ignore"):
                    arr = np.log(np.where(arr > 0, arr, np.nan))
            finite = np.isfinite(arr)
            if finite.sum() > 1 and np.nanstd(arr) > 0:
                arr = (arr - np.nanmean(arr)) / np.nanstd(arr)
            else:
                arr = np.zeros_like(arr)
            channels.append(np.where(finite, arr, 0.0))
        return np.stack(channels).astype(np.float32)

    def _prepare_numpy(self, X: xr.DataArray, y: Optional[xr.DataArray] = None,
                       static: Optional[np.ndarray] = None):
        X = _as_predictor(X)
        x = np.asarray(X.values, dtype=np.float32)  # T,M,Y,X
        finite_x = np.isfinite(x)
        x_filled = np.where(finite_x, x, self.fill_value).astype(np.float32, copy=False)
        if self.add_missing_mask:
            x_filled = np.concatenate(
                [x_filled, finite_x.astype(np.float32)], axis=1
            )
        if static is not None:
            static = np.asarray(static, dtype=np.float32)
            if static.shape[1:] != x_filled.shape[2:]:
                raise ValueError("Static channels do not match the predictor grid.")
            x_filled = np.concatenate(
                [x_filled, np.broadcast_to(static[None], (x_filled.shape[0],) + static.shape)],
                axis=1,
            ).astype(np.float32, copy=False)

        if y is None:
            return x_filled, None, None

        y = _drop_member(y)
        _require_same_grid(X, y, "X", "y")
        if X.sizes["T"] != y.sizes["T"]:
            raise ValueError("X and y must have the same T length after alignment.")
        yy = np.asarray(y.values, dtype=np.float32)
        finite_y = np.isfinite(yy)
        # A pixel/time contributes to the loss only when the target is valid and
        # at least one predictor channel is finite.  Missing predictor channels
        # themselves are represented by explicit mask channels.
        any_x = np.any(finite_x, axis=1)
        mask = finite_y & any_x
        yy = np.where(finite_y, yy, self.fill_value).astype(np.float32, copy=False)
        return x_filled, yy, mask

    def _effective_params(self, best_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        p = {
            "filters": self.filters,
            "dropout": self.dropout,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
        }
        if best_params:
            # Accept the plain dict returned by compute_hyperparameters.  A
            # one-entry {0: {...}} dict is also accepted for compatibility with
            # cluster-based MME APIs.
            if len(best_params) == 1 and 0 in best_params and isinstance(best_params[0], dict):
                best_params = best_params[0]
            for k in p:
                if k in best_params:
                    p[k] = best_params[k]
        p["filters"] = tuple(int(v) for v in p["filters"])
        p["dropout"] = float(p["dropout"])
        p["lr"] = float(p["lr"])
        p["weight_decay"] = float(p["weight_decay"])
        return p

    # ------------------------------------------------------------------
    # Network fitting
    # ------------------------------------------------------------------
    def _validate_grid(self, height: int, width: int, params: Dict[str, Any]) -> None:
        """Validate the spatial reduction used by this architecture."""
        min_side = 2 ** len(tuple(params["filters"]))
        if min(height, width) < min_side:
            raise ValueError(
                f"Grid {height}x{width} is too small for filters={params['filters']}; "
                f"minimum side is {min_side}. Use a shallower U-Net."
            )

    def _build_network(self, input_channels: int, params: Dict[str, Any]):
        """Architecture hook; all fitting and preprocessing remain shared."""
        return UNet(
            n_channels=input_channels, n_classes=1, filters=params["filters"],
            bilinear=self.bilinear, dropout=params["dropout"],
            norm_groups=self.norm_groups, norm=self.norm,
            linear_skip=self.linear_skip,
        )

    def _split_for_early_stopping(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        if n < 6:
            # Still keep one year for validation if possible.
            n_val = 1
        else:
            n_val = max(1, int(round(n * self.validation_fraction)))
            n_val = min(n_val, max(1, n - 5))
        n_train = n - n_val
        return np.arange(n_train), np.arange(n_train, n)

    def _make_loss(self):
        if self.loss == "mse":
            return nn.MSELoss(reduction="none")
        return nn.HuberLoss(delta=self.huber_delta, reduction="none")

    def _training_loss_map(self, model, inputs, target, loss_fn):
        """Unreduced training loss; nested networks may supervise extra heads.

        Validation and prediction use the ordinary forward output, so early
        stopping selects the same output that is used operationally.
        """
        return loss_fn(model(inputs).squeeze(1), target)

    def _fit_arrays(self, X: np.ndarray, y: np.ndarray, mask: np.ndarray,
                    params: Dict[str, Any], epochs: Optional[int] = None,
                    patience: Optional[int] = None,
                    seed_offset: int = 0,
                    split_indices=None, refit: bool = False) -> _FitResult:
        _set_global_seed(self.random_state + int(seed_offset), self.deterministic)
        n, c, h, w = X.shape
        self._validate_grid(h, w, params)

        if refit:
            tr_idx, va_idx = np.arange(n), np.array([], dtype=int)
        else:
            tr_idx, va_idx = (self._split_for_early_stopping(n)
                              if split_indices is None else split_indices)
        Xtr, ytr, mtr = X[tr_idx], y[tr_idx], mask[tr_idx]
        Xva, yva, mva = X[va_idx], y[va_idx], mask[va_idx]

        if not np.any(mtr):
            raise ValueError("No valid training target pixels after masking.")
        if not refit and not np.any(mva):
            raise ValueError("No valid validation target pixels after masking.")

        model = self._build_network(c, params).to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params["lr"],
            weight_decay=params["weight_decay"],
        )
        loss_fn = self._make_loss()
        device = torch.device(self.device)

        # A seasonal hindcast holds a few dozen fields: keep them on the device
        # and index batches directly (no DataLoader, one host sync per epoch).
        def _tensor(a, dtype):
            return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=device)

        Xtr_t, ytr_t, mtr_t = _tensor(Xtr, torch.float32), _tensor(ytr, torch.float32), _tensor(mtr, torch.bool)
        Xva_t, yva_t, mva_t = _tensor(Xva, torch.float32), _tensor(yva, torch.float32), _tensor(mva, torch.bool)
        n_train = int(Xtr_t.shape[0])
        batch = max(1, min(self.batch_size, n_train))
        val_batch = max(1, min(self.batch_size, max(1, int(Xva_t.shape[0]))))
        generator = torch.Generator()
        generator.manual_seed(self.random_state + int(seed_offset))

        max_epochs = int(self.epochs if epochs is None else epochs)
        max_patience = int(self.early_stopping_patience if patience is None else patience)
        best_val = np.inf
        best_state = None
        best_epoch = 0
        patience_count = 0
        history = {"train_loss": [], "val_loss": []}

        amp_dtype = None
        if self.mixed_precision and device.type == "cuda":
            amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

        def _forward(inputs):
            with torch.autocast(device_type=device.type, dtype=amp_dtype or torch.bfloat16,
                                enabled=amp_dtype is not None):
                out = model(inputs).squeeze(1)
            return out.float()

        def _train_loss_map(inputs, target):
            # Subclasses may supervise auxiliary heads during TRAINING only;
            # validation and prediction always use the deployed forward output.
            with torch.autocast(device_type=device.type, dtype=amp_dtype or torch.bfloat16,
                                enabled=amp_dtype is not None):
                raw = self._training_loss_map(model, inputs, target, loss_fn)
            return raw.float()

        for epoch in range(max_epochs):
            model.train()
            train_num = torch.zeros((), device=device)
            train_den = torch.zeros((), device=device)
            order = torch.randperm(n_train, generator=generator).to(device)
            for start_index in range(0, n_train, batch):
                index = order[start_index:start_index + batch]
                xb, yb, mb = Xtr_t[index], ytr_t[index], mtr_t[index]
                optimizer.zero_grad(set_to_none=True)
                raw = _train_loss_map(xb, yb)
                m = mb.float()
                denom = m.sum().clamp_min(1.0)
                loss = (raw * m).sum() / denom
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if self.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if self.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
                    optimizer.step()
                train_num += (raw.detach() * m).sum()
                train_den += denom.detach()

            val_loss = np.nan
            if not refit:
                model.eval()
                val_num = torch.zeros((), device=device)
                val_den = torch.zeros((), device=device)
                with torch.no_grad():
                    for start_index in range(0, int(Xva_t.shape[0]), val_batch):
                        sl = slice(start_index, start_index + val_batch)
                        raw = loss_fn(_forward(Xva_t[sl]), yva_t[sl])
                        m = mva_t[sl].float()
                        val_num += (raw * m).sum()
                        val_den += m.sum()
                val_loss = float((val_num / val_den.clamp_min(1.0)).item())
            train_loss = float((train_num / train_den.clamp_min(1.0)).item())
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if self.verbose > 1:
                print(
                    f"[{self._network_name}] epoch {epoch+1:03d}/{max_epochs} "
                    f"train={train_loss:.6f} val={val_loss:.6f}",
                    flush=True,
                )

            if refit:
                continue
            if np.isfinite(val_loss) and val_loss < best_val - 1.0e-8:
                best_val = val_loss
                best_epoch = epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= max_patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        return _FitResult(
            model=model,
            history=history,
            best_val_loss=float(best_val),
            epochs_ran=len(history["train_loss"]),
            input_channels=c,
            best_epoch=len(history["train_loss"]) if refit else best_epoch,
            training_samples=len(tr_idx),
        )

    def _fit_fields(self, X, y, params, *, epochs=None, patience=None,
                    seed_offset=0, clim_year_start=None, clim_year_end=None,
                    static_clim=None, n_seeds=None):
        """Select the epoch with train-only transforms, then refit every year.

        Every seed starts from its own initialization. The stopping block is used
        only to select that seed's epoch count; with ``refit_after_stopping`` the
        final network has a fresh optimizer and is trained for that count on the
        complete supplied fold, otherwise the early-stopped network is kept.
        Returned transforms belong to the training sample of the kept networks.
        ``static_clim`` = (mean, std) of the TRAINING target in physical units.
        """
        n = X.sizes["T"]
        if n < 3:
            raise ValueError("U-Net stopping/refit requires at least three training years.")
        n_seeds = self.n_seeds if n_seeds is None else int(n_seeds)
        years = np.asarray(X["T"].dt.year.values, int)
        if len(np.unique(years)) != n or np.any(np.diff(years) <= 0):
            raise ValueError("U-Net requires one sorted target season per year.")
        tr, va = self._split_for_early_stopping(n)
        if self.temporal_buffer > 0:
            tr = tr[np.min(np.abs(years[tr, None] - years[va]), axis=1) > self.temporal_buffer]
        if len(tr) < 2:
            raise ValueError("Too few stopping-training years after the temporal buffer.")

        def static_for(sample):
            if not self.add_static_channels:
                return None
            clim = static_clim if static_clim is not None else _fit_standardizer(sample)
            return self._static_channels(X, clim[0], clim[1])

        xm, xs = _fit_standardizer(X.isel(T=tr), clim_year_start, clim_year_end)
        ym, ys = _fit_standardizer(y.isel(T=tr), clim_year_start, clim_year_end)
        stop_static = static_for(y.isel(T=tr))
        a, b, m = self._prepare_numpy(_standardize_with(X, xm, xs),
                                      _standardize_with(y, ym, ys), stop_static)
        # A pixel absent in stopping training is not made trainable by its
        # held-out validation target.
        train_support = np.isfinite(y.isel(T=tr)).sum("T").values >= 1
        m &= train_support[None, ...]
        members, selected, histories, best_losses = [], [], [], []
        for seed in range(n_seeds):
            offset = int(seed_offset) + 7919 * seed
            stop = self._fit_arrays(a, b, m, params, epochs=epochs, patience=patience,
                                    seed_offset=offset, split_indices=(tr, va))
            if stop.best_epoch < 1:
                raise ValueError("U-Net stopping validation did not produce a finite loss.")
            selected.append(int(stop.best_epoch))
            histories.append(stop.history)
            best_losses.append(stop.best_val_loss)
            if self.refit_after_stopping:
                del stop.model
            else:
                members.append(stop.model)
        self.stopping_diagnostics_ = {
            "training_years": years[tr].copy(), "validation_years": years[va].copy(),
            "input_mean": xm.copy(deep=True), "input_std": xs.copy(deep=True),
            "target_mean": ym.copy(deep=True), "target_std": ys.copy(deep=True),
            "selected_epoch": int(round(float(np.median(selected)))),
            "selected_epochs": list(selected), "n_seeds": int(n_seeds),
            "history": histories[0], "histories": histories,
        }
        if self.refit_after_stopping:
            xm, xs = _fit_standardizer(X, clim_year_start, clim_year_end)
            ym, ys = _fit_standardizer(y, clim_year_start, clim_year_end)
            static = static_for(y)
            a, b, m = self._prepare_numpy(_standardize_with(X, xm, xs),
                                         _standardize_with(y, ym, ys), static)
            for seed, epoch_count in enumerate(selected):
                final = self._fit_arrays(a, b, m, params, epochs=epoch_count,
                                         seed_offset=int(seed_offset) + 7919 * seed, refit=True)
                members.append(final.model)
            samples, kept_years = int(n), years.copy()
        else:
            static = stop_static
            samples, kept_years = int(len(tr)), years[tr].copy()
        model = members[0] if len(members) == 1 else SeedEnsemble(members)
        self.refit_diagnostics_ = {"training_years": kept_years,
                                  "epochs": self.stopping_diagnostics_["selected_epoch"],
                                  "epochs_per_seed": list(selected),
                                  "refit": bool(self.refit_after_stopping),
                                  "training_samples": samples}
        result = _FitResult(model=model, history=histories[0],
                            best_val_loss=float(np.mean(best_losses)),
                            epochs_ran=len(histories[0]["train_loss"]),
                            input_channels=int(a.shape[1]),
                            best_epoch=self.stopping_diagnostics_["selected_epoch"],
                            training_samples=samples, static=static)
        return result, (xm, xs, ym, ys)

    def predict_fitted(self, X_test):
        """Predict on the target scale supplied to the last ``compute_model``."""
        if self.model is None or not hasattr(self, "_fitted_standardizers"):
            raise ValueError("Call compute_model before predict_fitted.")
        Xte = _as_predictor(X_test)
        xm, xs, ym, ys = self._fitted_standardizers
        _require_same_grid(Xte, xm, "X_test", "fitted grid")
        if not np.array_equal(Xte.M, xm.M):
            raise ValueError("Predictor channel labels differ from the fitted channels.")
        xx, _, _ = self._prepare_numpy(_standardize_with(Xte, xm, xs), None,
                                       getattr(self, "_fitted_static", None))
        pred = self._predict_arrays(self.model, xx)
        out = xr.DataArray(pred, dims=("T", "Y", "X"),
                           coords={"T": Xte["T"], "Y": Xte.Y, "X": Xte.X},
                           name=f"{self._network_name}_deterministic")
        return (out * ys + ym) * self._fitted_spatial_mask

    def _predict_arrays(self, model, X: np.ndarray) -> np.ndarray:
        model.eval()
        device = torch.device(self.device)
        out: List[np.ndarray] = []
        batch = max(1, min(self.batch_size, len(X)))
        with torch.no_grad():
            for start_index in range(0, len(X), batch):
                xb = torch.as_tensor(np.ascontiguousarray(X[start_index:start_index + batch]),
                                     dtype=torch.float32, device=device)
                out.append(model(xb).squeeze(1).float().cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.empty((0, X.shape[2], X.shape[3]))

    # ------------------------------------------------------------------
    # Optional nested hyperparameter selection
    # ------------------------------------------------------------------
    def _architecture_complexity(self, params):
        return sum(params["filters"])

    def _tuning_candidates(self):
        """Architecture-specific grid with the established capped ordering."""
        candidates = [
            {
                "filters": f,
                "dropout": d,
                "lr": lr,
                "weight_decay": wd,
            }
            for f, d, lr, wd in itertools.product(
                self.tuning_filters,
                self.tuning_dropouts,
                self.tuning_lrs,
                self.tuning_weight_decays,
            )
        ]
        # Deterministic ordering.  A capped budget must still compare several
        # depths; simply taking the first N would spend all eight default trials
        # on the smallest architecture.
        candidates.sort(key=lambda p: (sum(p["filters"]), p["dropout"], p["lr"], p["weight_decay"]))
        if self.max_tuning_candidates > 0:
            groups = {}
            for candidate in candidates:
                groups.setdefault(candidate["filters"], []).append(candidate)
            ordered_groups = [groups[key] for key in sorted(groups, key=sum)]
            selected = []
            while len(selected) < self.max_tuning_candidates and any(ordered_groups):
                for group in ordered_groups:
                    if group and len(selected) < self.max_tuning_candidates:
                        selected.append(group.pop(0))
            candidates = selected

        return candidates

    def compute_hyperparameters(self, Predictors: xr.DataArray, Predictand: xr.DataArray,
                                clim_year_start=None, clim_year_end=None):
        """Return U-Net parameters and ``None`` cluster map.

        When ``tune_hyperparameters=False`` (recommended first scientific
        baseline), the pre-specified architecture is returned without using any
        validation observations for model selection.

        When enabled, candidates are evaluated by blocked temporal inner CV.
        If this method is called inside WASS2S outer CV, the procedure is nested.
        """
        Xraw, yraw = _align_training(Predictors, Predictand)
        default = self._effective_params(None)
        if not self.tune_hyperparameters:
            self.best_params_ = default
            self.tuning_results_ = []
            return default, None

        candidates = self._tuning_candidates()

        folds = _time_blocks(
            Xraw.sizes["T"],
            n_folds=self.tuning_folds,
            buffer=self.temporal_buffer,
            min_train=6,
        )
        if len(folds) < 2:
            warnings.warn(
                f"Too few years for blocked {self._network_name} tuning; using prespecified parameters.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.best_params_ = default
            self.tuning_results_ = []
            return default, None

        results: List[Dict[str, Any]] = []
        for ci, params in enumerate(candidates):
            fold_scores = []
            for fi, (tr, va) in enumerate(folds):
                Xi_raw, Xv_raw = Xraw.isel(T=tr), Xraw.isel(T=va)
                yi_raw, yv_raw = yraw.isel(T=tr), yraw.isel(T=va)
                fit, (xm, xs, ym, ys) = self._fit_fields(
                    Xi_raw, yi_raw, params, epochs=self.tuning_epochs,
                    patience=self.tuning_patience, seed_offset=1000 + fi,
                    clim_year_start=clim_year_start, clim_year_end=clim_year_end,
                    n_seeds=self.tuning_seeds)
                Xv = _standardize_with(Xv_raw, xm, xs)
                yv = _standardize_with(yv_raw, ym, ys)
                xv, _, _ = self._prepare_numpy(Xv, None, fit.static)
                pv = self._predict_arrays(fit.model, xv)
                yv_np = np.asarray(yv.values, dtype=float)
                valid = np.isfinite(yv_np) & np.isfinite(pv) & np.isfinite(Xv.values).any(axis=1)
                if np.any(valid):
                    fold_scores.append(float(np.mean((pv[valid] - yv_np[valid]) ** 2)))
                del fit.model
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()

            score = (float(np.mean(fold_scores))
                     if len(fold_scores) == len(folds) else np.inf)
            rec = dict(params)
            rec.update(score=score, fold_scores=fold_scores)
            results.append(rec)
            if self.verbose:
                print(f"[{self._network_name} tune] {params} -> MSE={score:.6f}", flush=True)

        valid_results = [r for r in results if np.isfinite(r["score"])]
        if not valid_results:
            best = default
        else:
            # Tie-breaking favors the simpler network.
            best_rec = min(valid_results, key=lambda r: (r["score"], self._architecture_complexity(r)))
            best = {k: best_rec[k] for k in default}
        self.best_params_ = best
        self.tuning_results_ = results
        return best, None

    # ------------------------------------------------------------------
    # WASS2S public model contract
    # ------------------------------------------------------------------
    def compute_model(self, X_train: xr.DataArray, y_train: xr.DataArray,
                      X_test: xr.DataArray, y_test: Optional[xr.DataArray] = None,
                      clim_year_start=None, clim_year_end=None,
                      best_params: Optional[Dict[str, Any]] = None,
                      cluster_da=None, static_climatology=None, **kwargs) -> xr.DataArray:
        """Fit the U-Net and predict ``X_test``.

        If climatology years are supplied, X and y are standardized using
        statistics estimated from *X_train/y_train only*.  The returned values
        remain on the standardized target scale, matching the WASS2S MME
        cross-validator contract.  The scientific cross-validator normally
        standardizes each fold before calling this method and passes
        ``clim_year_start=None, clim_year_end=None``.
        """
        del y_test, cluster_da, kwargs
        Xtr, ytr = _align_training(X_train, y_train)
        Xte = _as_predictor(X_test)
        _require_same_grid(Xtr, Xte, "X_train", "X_test")
        if Xtr.sizes["M"] != Xte.sizes["M"]:
            raise ValueError(
                f"Train/test predictor channel count differs: {Xtr.sizes['M']} != {Xte.sizes['M']}."
            )
        if "M" in Xtr.coords and "M" in Xte.coords and not np.array_equal(Xtr["M"], Xte["M"]):
            raise ValueError("Train/test M channel labels differ; channel identity must be stable.")

        if static_climatology is None:
            # Physical-unit climatology of THIS training sample (zero channels
            # if the caller already standardized the target).
            static_climatology = _fit_standardizer(ytr, clim_year_start, clim_year_end)
        if clim_year_start is not None and clim_year_end is not None:
            xm, xs = _fit_standardizer(Xtr, clim_year_start, clim_year_end)
            ym, ys = _fit_standardizer(ytr, clim_year_start, clim_year_end)
            Xtr = _standardize_with(Xtr, xm, xs)
            Xte = _standardize_with(Xte, xm, xs)
            ytr = _standardize_with(ytr, ym, ys)

        params = self._effective_params(best_params)
        fit, transforms = self._fit_fields(Xtr, ytr, params, static_clim=static_climatology)
        self.model = fit.model
        self.history = fit.history
        self.best_params_ = params
        self._fitted_standardizers = transforms
        self._fitted_static = fit.static
        self._fitted_spatial_mask = _spatial_mask(ytr, self.min_valid_years)
        return self.predict_fitted(Xte)

    # ------------------------------------------------------------------
    # Probabilities
    # ------------------------------------------------------------------
    def _empirical_thresholds(self, obs: xr.DataArray, clim_year_start, clim_year_end):
        clim = _clim_subset(obs, clim_year_start, clim_year_end, min_n=3)
        if clim.sizes.get("T", 0) < 3:
            raise ValueError("At least three climatology years are required for terciles.")
        terc = climatological_terciles(clim)[0]
        return terc.isel(quantile=0, drop=True), terc.isel(quantile=1, drop=True), clim

    def compute_prob(self, Predictant: xr.DataArray, clim_year_start, clim_year_end,
                     hindcast_det: xr.DataArray,
                     best_code_da=None, best_shape_da=None,
                     best_loc_da=None, best_scale_da=None, **kwargs) -> xr.DataArray:
        """Hindcast tercile probabilities.

        For ``dist_method='nonparam'`` this is strictly leave-one-year-out with
        respect to residual dressing: residual[t] is never used to construct
        probabilities for hindcast[t].  This removes the subtle self-inclusion
        leakage of the original prototype.
        """
        del kwargs
        obs = _drop_member(Predictant)
        hind = _drop_member(hindcast_det)
        if hind.sizes["T"] == obs.sizes["T"]:
            hind = hind.assign_coords(T=obs["T"])
        obs, hind = xr.align(obs, hind, join="inner")
        mask = _spatial_mask(obs, self.min_valid_years)
        t1_emp, t2_emp, clim = self._empirical_thresholds(obs, clim_year_start, clim_year_end)

        if self.dist_method == "nonparam":
            prob = xr.apply_ufunc(
                _empirical_prob_loo,
                hind,
                obs,
                hind,
                t1_emp,
                t2_emp,
                input_core_dims=[["T"], ["T"], ["T"], [], []],
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
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError(
                    "dist_method='bestfit' requires best_code_da, best_shape_da, "
                    "best_loc_da and best_scale_da. For strictly OOS verification, "
                    "those distribution parameters must themselves be fitted fold-safely."
                )
            # Category boundaries stay on the observed climatological reference.
            # The selected distribution controls the predictive CDF only.
            if getattr(self, "tercile_reference", "climatology") == "training":
                t1, t2 = xr.apply_ufunc(
                    _ppf_terciles_from_code,
                    best_code_da, best_shape_da, best_loc_da, best_scale_da,
                    input_core_dims=[[], [], [], []],
                    output_core_dims=[[], []],
                    vectorize=True,
                    dask="parallelized",
                    output_dtypes=[float, float],
                )
            else:
                t1, t2 = t1_emp, t2_emp
            var_loo = _loo_mse_variance(obs, hind)
            dof = max(int(clim.sizes["T"]) - 1, 2)
            prob = xr.apply_ufunc(
                _bestfit_prob_vector,
                hind,
                var_loo,
                t1,
                t2,
                best_code_da,
                input_core_dims=[["T"], ["T"], [], [], []],
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

        prob = prob.assign_coords(probability=("probability", ["PB", "PN", "PA"]))
        prob = _normalize_probability_da(prob.transpose("probability", "T", "Y", "X"))
        return (prob * mask).transpose("probability", "T", "Y", "X")

    def _forecast_prob(self, obs: xr.DataArray, oof_hindcast: xr.DataArray,
                       forecast_det: xr.DataArray, clim_year_start, clim_year_end,
                       best_code_da=None, best_shape_da=None,
                       best_loc_da=None, best_scale_da=None) -> xr.DataArray:
        obs = _drop_member(obs)
        oof = _drop_member(oof_hindcast)
        if oof.sizes["T"] == obs.sizes["T"]:
            oof = oof.assign_coords(T=obs["T"])
        obs, oof = xr.align(obs, oof, join="inner")
        forecast_det = forecast_det.transpose("T", "Y", "X")
        t1_emp, t2_emp, clim = self._empirical_thresholds(obs, clim_year_start, clim_year_end)

        if self.dist_method == "nonparam":
            residuals = (obs - oof).rename({"T": "S"})
            prob = xr.apply_ufunc(
                _empirical_prob_forecast,
                forecast_det,
                residuals,
                t1_emp,
                t2_emp,
                input_core_dims=[["T"], ["S"], [], []],
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
            if any(v is None for v in (best_code_da, best_shape_da, best_loc_da, best_scale_da)):
                raise ValueError(
                    "dist_method='bestfit' requires the four best-distribution parameter maps."
                )
            # Category boundaries stay on the observed climatological reference.
            # The selected distribution controls the predictive CDF only.
            if getattr(self, "tercile_reference", "climatology") == "training":
                t1, t2 = xr.apply_ufunc(
                    _ppf_terciles_from_code,
                    best_code_da, best_shape_da, best_loc_da, best_scale_da,
                    input_core_dims=[[], [], [], []], output_core_dims=[[], []],
                    vectorize=True, dask="parallelized", output_dtypes=[float, float],
                )
            else:
                t1, t2 = t1_emp, t2_emp
            error_variance = ((obs - oof) ** 2).mean("T", skipna=True)
            # Broadcast the scalar pixel variance along forecast T for the kernel.
            variance = xr.broadcast(forecast_det, error_variance)[1]
            dof = max(int(clim.sizes["T"]) - 1, 2)
            prob = xr.apply_ufunc(
                _bestfit_prob_vector,
                forecast_det,
                variance,
                t1,
                t2,
                best_code_da,
                input_core_dims=[["T"], ["T"], [], [], []],
                output_core_dims=[["probability", "T"]],
                vectorize=True,
                dask="parallelized",
                kwargs={"dof": dof},
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True},
            )
        prob = prob.assign_coords(probability=("probability", ["PB", "PN", "PA"]))
        return _normalize_probability_da(prob.transpose("probability", "T", "Y", "X"))

    # ------------------------------------------------------------------
    # Operational forecast
    # ------------------------------------------------------------------
    def forecast(self, Predictant: xr.DataArray, clim_year_start, clim_year_end,
                 hindcast_det: xr.DataArray, hindcast_det_cross: xr.DataArray,
                 Predictor_for_year: xr.DataArray,
                 best_params: Optional[Dict[str, Any]] = None, cluster_da=None,
                 best_code_da=None, best_shape_da=None,
                 best_loc_da=None, best_scale_da=None, **kwargs):
        """Operational U-Net forecast using the established WASS2S MME contract.

        Parameters
        ----------
        hindcast_det
            Historical MME predictor cube, typically ``all_model_hdcst`` with
            dimensions (T,M,Y,X).  The name is retained only for API compatibility.
        hindcast_det_cross
            Genuinely out-of-fold U-Net deterministic hindcast.  Its residuals
            against observations calibrate operational uncertainty.
        Predictor_for_year
            Target-year MME predictor cube, typically ``all_model_fcst``.
        """
        del cluster_da, kwargs
        obs = _drop_member(Predictant)
        Xhist, obs = _align_training(hindcast_det, obs)
        Xyear = _as_predictor(Predictor_for_year)
        _require_same_grid(Xhist, Xyear, "historical predictor", "forecast predictor")
        if Xhist.sizes["M"] != Xyear.sizes["M"]:
            raise ValueError("Historical and forecast MME channel counts differ.")
        if "M" in Xhist.coords and "M" in Xyear.coords and not np.array_equal(Xhist["M"], Xyear["M"]):
            raise ValueError("Historical and forecast MME channel labels/order differ.")

        # Fold-safe operational preprocessing: climatological statistics are
        # estimated from historical predictors/observations only.
        xm, xs = _fit_standardizer(Xhist, clim_year_start, clim_year_end)
        ym, ys = _fit_standardizer(obs, clim_year_start, clim_year_end)
        Xhist_st = _standardize_with(Xhist, xm, xs)
        Xyear_st = _standardize_with(Xyear, xm, xs)
        y_st = _standardize_with(obs, ym, ys)

        pred_st = self.compute_model(
            Xhist_st,
            y_st,
            Xyear_st,
            y_test=None,
            clim_year_start=None,
            clim_year_end=None,
            best_params=best_params,
            static_climatology=(ym, ys),
        )
        pred = pred_st * ys + ym
        pred = _assign_target_time(pred, Xyear, obs)
        pred = _conditional_nonnegative_clip(pred, obs)
        mask = _spatial_mask(obs, self.min_valid_years)
        pred = pred * mask

        prob = self._forecast_prob(
            obs,
            hindcast_det_cross,
            pred,
            clim_year_start,
            clim_year_end,
            best_code_da,
            best_shape_da,
            best_loc_da,
            best_scale_da,
        ) * mask
        return pred, prob.transpose("probability", "T", "Y", "X")

    # ------------------------------------------------------------------
    # Optional stand-alone strict CV (useful if not yet wired into
    # WAS_Cross_Validator).  This method deliberately does not tune on outer
    # test years.  Hyperparameter tuning, if enabled, occurs inside each fold.
    # ------------------------------------------------------------------
    def cross_validate(self, Predictant: xr.DataArray, Predictor: xr.DataArray,
                       clim_year_start=None, clim_year_end=None,
                       nb_omit: int = 2, strict_oos_probability: bool = True):
        """Produce deterministic and probabilistic OOF hindcasts directly."""
        if strict_oos_probability and self.dist_method != "nonparam":
            raise ValueError("Strict U-Net probabilities require nonparam inner-CV residuals; use WAS_CrossFit for other calibration choices.")
        obs = _drop_member(Predictant)
        X, obs = _align_training(Predictor, obs)
        n = X.sizes["T"]
        if n < 8:
            warnings.warn(
                f"Only {n} years are available; U-Net skill will be highly uncertain.",
                RuntimeWarning,
                stacklevel=2,
            )

        idx = np.arange(n)
        window_len = min(n, max(1, int(nb_omit) + 1))
        dets: List[xr.DataArray] = []
        probs: List[xr.DataArray] = []

        for i in range(n):
            start = i - window_len // 2
            start = max(0, min(start, n - window_len))
            excluded = np.arange(start, start + window_len)
            tr = np.setdiff1d(idx, excluded)
            te = np.array([i])
            if tr.size < 6:
                raise ValueError("Too few outer-training years after temporal exclusion.")

            Xtr_raw, Xte_raw = X.isel(T=tr), X.isel(T=te)
            ytr_raw, yte_raw = obs.isel(T=tr), obs.isel(T=te)
            params, _ = self.compute_hyperparameters(
                Xtr_raw, ytr_raw, clim_year_start, clim_year_end
            )
            xm, xs = _fit_standardizer(Xtr_raw, clim_year_start, clim_year_end)
            ym, ys = _fit_standardizer(ytr_raw, clim_year_start, clim_year_end)
            Xtr = _standardize_with(Xtr_raw, xm, xs)
            Xte = _standardize_with(Xte_raw, xm, xs)
            ytr = _standardize_with(ytr_raw, ym, ys)
            pst = self.compute_model(Xtr, ytr, Xte, best_params=params,
                                     static_climatology=(ym, ys))
            p = _conditional_nonnegative_clip(pst * ys + ym, obs)
            p = p.assign_coords(T=yte_raw["T"])
            dets.append(p)

            if strict_oos_probability and self.dist_method == "nonparam":
                # Create inner OOF residuals using only this outer training fold.
                inner = _time_blocks(len(tr), n_folds=min(5, max(2, len(tr)//4)),
                                     buffer=max(0, int(nb_omit)//2), min_train=6)
                inner_dets = []
                for fi, (itr, iva) in enumerate(inner):
                    Xi_raw, Xv_raw = Xtr_raw.isel(T=itr), Xtr_raw.isel(T=iva)
                    yi_raw, yv_raw = ytr_raw.isel(T=itr), ytr_raw.isel(T=iva)
                    xmi, xsi = _fit_standardizer(Xi_raw, clim_year_start, clim_year_end)
                    ymi, ysi = _fit_standardizer(yi_raw, clim_year_start, clim_year_end)
                    Xi = _standardize_with(Xi_raw, xmi, xsi)
                    Xv = _standardize_with(Xv_raw, xmi, xsi)
                    yi = _standardize_with(yi_raw, ymi, ysi)
                    ipst = self.compute_model(Xi, yi, Xv, best_params=params,
                                              static_climatology=(ymi, ysi))
                    ip = _conditional_nonnegative_clip(ipst * ysi + ymi, obs)
                    ip = ip.assign_coords(T=yv_raw["T"])
                    inner_dets.append(ip)
                if inner_dets:
                    ih = xr.concat(inner_dets, "T").sortby("T").reindex(T=ytr_raw["T"])
                    residuals = (ytr_raw - ih).rename({"T": "S"})
                    t1, t2, _ = self._empirical_thresholds(
                        ytr_raw, clim_year_start, clim_year_end
                    )
                    pr = xr.apply_ufunc(
                        _empirical_prob_forecast,
                        p,
                        residuals,
                        t1,
                        t2,
                        input_core_dims=[["T"], ["S"], [], []],
                        output_core_dims=[["probability", "T"]],
                        vectorize=True,
                        dask="parallelized",
                        output_dtypes=[float],
                        dask_gufunc_kwargs={"output_sizes": {"probability": 3}, "allow_rechunk": True},
                    )
                    pr = pr.assign_coords(probability=("probability", ["PB", "PN", "PA"]))
                    probs.append(_normalize_probability_da(pr.transpose("probability", "T", "Y", "X")))

            if self.verbose:
                print(f"[UNet CV] outer year {i+1}/{n}", flush=True)

        hind = xr.concat(dets, "T").sortby("T").reindex(T=obs["T"])
        hind = _conditional_nonnegative_clip(hind, obs)

        if strict_oos_probability and probs and len(probs) == n:
            hprob = xr.concat(probs, "T").sortby("T").reindex(T=obs["T"])
            hprob = hprob.transpose("probability", "T", "Y", "X")
        else:
            if strict_oos_probability and self.dist_method == "nonparam" and len(probs) != n:
                raise RuntimeError("Strict probabilities unavailable for at least one outer fold; increase training years or reduce the exclusion window. No approximate fallback was used.")
            hprob = self.compute_prob(obs, clim_year_start, clim_year_end, hind)
        return hind, hprob

    def summary(self) -> None:
        p = self._effective_params(self.best_params_)
        print("WAS_mme_UNet — spatial nonlinear MME calibration")
        print("===================================================")
        print(f"filters                 : {p['filters']}")
        print(f"dropout                 : {p['dropout']}")
        print(f"lr                      : {p['lr']}")
        print(f"weight_decay            : {p['weight_decay']}")
        print(f"loss                    : {self.loss}")
        print(f"batch_size              : {self.batch_size}")
        print(f"epochs                  : {self.epochs}")
        print(f"early_stopping_patience : {self.early_stopping_patience}")
        print(f"add_missing_mask        : {self.add_missing_mask}")
        print(f"device                  : {self.device}")
        print(f"dist_method             : {self.dist_method}")
        print(f"tune_hyperparameters    : {self.tune_hyperparameters}")
        print(f"norm / linear_skip      : {self.norm} / {self.linear_skip}")
        print(f"static channels         : {self.add_static_channels}")
        print(f"n_seeds / refit         : {self.n_seeds} / {self.refit_after_stopping}")
        print(f"model fitted            : {self.model is not None}")


# The torch modules exist only when torch imports. Listing them unconditionally
# makes ``from wass2s.was_mme_unet import *`` raise AttributeError without torch.
__all__ = ["WAS_mme_UNet"]
if HAS_TORCH:
    __all__ += ["UNet", "DoubleConv", "Down", "Up", "SeedEnsemble"]


# Keep constructor inputs for fresh model instances in nested cross-validation.
from .was_crossfit_adapters import capture_constructors as _capture_constructors
_capture_constructors(globals())
