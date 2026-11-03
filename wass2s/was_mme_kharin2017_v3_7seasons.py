"""Kharin et al. (2017) seasonal-forecast postprocessing for WASS2S.

Implements the core method in:
V. V. Kharin, W. J. Merryfield, G. J. Boer, and W.-S. Lee (2017),
"A Postprocessing Method for Seasonal Forecasts Using Temporally and Spatially
Smoothed Statistics", Monthly Weather Review, 145, 3545-3561.

WASS2S conventions
------------------
Observations:          (T, Y, X)
Hindcast/forecast MME: (T, M, Y, X)
Outputs deterministic: (T, Y, X)
Outputs probabilistic: (probability, T, Y, X), probability=(PB, PN, PA)

Scientific design used in WASS2S
---------------------------------
Kharin et al. estimate/smooth adjustment parameters across target seasons at a
FIXED forecast lead.  The default WASS2S adaptation implemented here uses seven
consecutive overlapping 3-month target seasons around JAS:

    AMJ, MJJ, JJA, JAS, ASO, SON, OND

all using the same lead window (by default leads 03-05) while the initialization
month moves from January to July.  The temporal coordinate is always the TARGET
season central month, never the initialization month.  The pooled constant case
is therefore labelled C7 (or, generically, C_N), not the exact annual C of the
12-season paper.  A paper-faithful 12-season configuration remains available by
passing all DJF..NDJ target seasons.

Spatial smoothing
-----------------
The paper uses global spherical-harmonic triangular truncation.  WASS2S is
usually run on regional West-African domains, for which a global Tn truncation
is not naturally defined.  Therefore the exact temporal method is implemented,
while spatial smoothing is OFF by default.  An optional normalized Gaussian
regional smoother is provided as an explicit extension, not as a claim of
exact replication of the paper's spectral truncation.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter
from scipy.stats import gamma as sp_gamma
from scipy.stats import norm
from scipy.special import gdtr, gdtrix, ndtri

from .was_probability_reference import climatological_terciles
from .was_probability_calibration import normalize_probability_calibration


def _normal_event_cdf(threshold, mu, sigma):
    """Normal CDF, including the exact point-mass limit with <= events."""
    continuous = xr.apply_ufunc(norm.cdf, (threshold - mu) / sigma.where(sigma > 0),
                               dask="parallelized", output_dtypes=[float])
    result = xr.where(sigma == 0, (threshold >= mu).astype(float), continuous)
    return result.where(np.isfinite(mu) & np.isfinite(sigma) & (sigma >= 0)
                        & threshold.notnull())

try:
    from numba import njit, prange
    _KHARIN_NUMBA = True
except Exception:  # optional acceleration
    _KHARIN_NUMBA = False
    def njit(*args, **kwargs):
        def deco(func):
            return func
        return deco
    def prange(*args):
        return range(*args)


@njit(cache=True, inline="always")
def _kharin_phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


@njit(cache=True, inline="always")
def _kharin_pdf(z):
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


@njit(cache=True, inline="always")
def _kharin_crps_scalar(mu, sigma, y, eps):
    if sigma <= eps:
        return abs(y - mu)
    z = (y - mu) / sigma
    return sigma * (z * (2.0 * _kharin_phi(z) - 1.0) + 2.0 * _kharin_pdf(z) - 1.0 / math.sqrt(math.pi))


@njit(cache=True, parallel=True)
def _fit_ab_season_numba(F, O, SF, qgrid, min_samples, eps):
    nt, nc = F.shape
    A = np.empty(nc, dtype=np.float64); A[:] = np.nan
    B = np.empty(nc, dtype=np.float64); B[:] = np.nan
    for c in prange(nc):
        sf = SF[c]
        if not np.isfinite(sf):
            continue
        n = 0; mf = 0.0; mo = 0.0
        for t in range(nt):
            f = F[t, c]; o = O[t, c]
            if np.isfinite(f) and np.isfinite(o):
                n += 1; mf += f; mo += o
        if n < min_samples or n < 2:
            continue
        mf /= n; mo /= n
        ssf = 0.0; sso = 0.0
        for t in range(nt):
            f = F[t, c]; o = O[t, c]
            if np.isfinite(f) and np.isfinite(o):
                df = f - mf; do = o - mo
                ssf += df * df; sso += do * do
        vf = ssf / (n - 1.0); vo = sso / (n - 1.0)
        if not np.isfinite(vo) or not np.isfinite(vf):
            continue
        if vo == 0:
            A[c] = 0.0; B[c] = 0.0
            continue
        if vf <= eps:
            A[c] = 0.0
            B[c] = np.sqrt(vo / max(sf * sf, eps))
            continue
        scale = np.sqrt(vo / vf)
        best = 1e300; bestq = np.nan
        for iq in range(qgrid.size):
            q = qgrid[iq]
            a = q * scale
            sigma = np.sqrt(max(vo * (1.0 - q * q), 0.0))
            total = 0.0; nn = 0
            for t in range(nt):
                f = F[t, c]; o = O[t, c]
                if np.isfinite(f) and np.isfinite(o):
                    total += _kharin_crps_scalar(a * f, sigma, o, eps)
                    nn += 1
            score = total / nn
            if score < best:
                best = score; bestq = q
        if np.isfinite(bestq):
            A[c] = bestq * scale
            B[c] = np.sqrt(max(vo * (1.0 - bestq * bestq), 0.0) / max(sf * sf, eps))
    return A, B


@njit(cache=True, parallel=True)
def _fit_ab_constant_numba(F, O, SF, qgrid, min_samples, eps):
    nt, nc = F.shape
    A = np.empty(nc, dtype=np.float64); A[:] = np.nan
    B = np.empty(nc, dtype=np.float64); B[:] = np.nan
    for c in prange(nc):
        n = 0; mf = 0.0; mo = 0.0; msf2 = 0.0
        for t in range(nt):
            f = F[t, c]; o = O[t, c]; sf = SF[t, c]
            if np.isfinite(f) and np.isfinite(o) and np.isfinite(sf):
                n += 1; mf += f; mo += o; msf2 += sf * sf
        if n < min_samples or n < 2:
            continue
        mf /= n; mo /= n; msf2 /= n
        ssf = 0.0; sso = 0.0
        for t in range(nt):
            f = F[t, c]; o = O[t, c]; sf = SF[t, c]
            if np.isfinite(f) and np.isfinite(o) and np.isfinite(sf):
                df = f - mf; do = o - mo
                ssf += df * df; sso += do * do
        vf = ssf / (n - 1.0); vo = sso / (n - 1.0)
        if not np.isfinite(vo) or not np.isfinite(vf):
            continue
        if vo == 0:
            A[c] = 0.0; B[c] = 0.0
            continue
        if vf <= eps:
            A[c] = 0.0
            B[c] = np.sqrt(vo / max(msf2, eps))
            continue
        scale = np.sqrt(vo / vf)
        best = 1e300; bestq = np.nan
        for iq in range(qgrid.size):
            q = qgrid[iq]
            a = q * scale
            b = np.sqrt(max(vo * (1.0 - q * q), 0.0) / max(msf2, eps))
            total = 0.0; nn = 0
            for t in range(nt):
                f = F[t, c]; o = O[t, c]; sf = SF[t, c]
                if np.isfinite(f) and np.isfinite(o) and np.isfinite(sf):
                    total += _kharin_crps_scalar(a * f, sf * b, o, eps)
                    nn += 1
            score = total / nn
            if score < best:
                best = score; bestq = q
        if np.isfinite(bestq):
            A[c] = bestq * scale
            B[c] = np.sqrt(max(vo * (1.0 - bestq * bestq), 0.0) / max(msf2, eps))
    return A, B


TemporalMode = Union[str, float, int]


@dataclass
class _KharinFit:
    """Container for parameters fitted on one training sample."""

    det_obs_clim: xr.DataArray       # (season,Y,X)
    det_fc_clim: xr.DataArray        # (season,Y,X)
    det_a: xr.DataArray              # (season,Y,X)

    prob_obs_clim: xr.DataArray      # transformed space, (season,Y,X)
    prob_fc_clim: xr.DataArray       # transformed space, (season,Y,X)
    prob_sf: xr.DataArray            # raw climatological ensemble spread, (season,Y,X)
    prob_a: xr.DataArray             # (season,Y,X)
    prob_b: xr.DataArray             # (season,Y,X)

    gamma_obs_shape: Optional[xr.DataArray] = None
    gamma_obs_scale: Optional[xr.DataArray] = None
    gamma_fc_shape: Optional[xr.DataArray] = None
    gamma_fc_scale: Optional[xr.DataArray] = None
    # Zero-inflated precipitation (15 Sept. 2026): probability of zero and the
    # probability level assigned to a zero value, per season and cell.
    gamma_obs_p0: Optional[xr.DataArray] = None
    gamma_fc_p0: Optional[xr.DataArray] = None
    gamma_obs_zero_u: Optional[xr.DataArray] = None
    gamma_fc_zero_u: Optional[xr.DataArray] = None
    # Physical observed event bounds; shared across CV folds and forecasting.
    reference_terciles: Optional[xr.DataArray] = None
    reference_probabilities: Optional[xr.DataArray] = None


class WAS_mme_Kharin2017:
    """Temporally smoothed regression/CRPS calibration of an MME.

    Deterministic component
    -----------------------
    For each grid point i and season m, the MME ensemble-mean anomaly F is
    rescaled as

        F_tilde = a * F,
        a = Cov(F, O) / Var(F),

    after removing model and observed seasonal climatologies.  The 12 raw
    coefficients may be left unsmoothed ('U'), circularly Gaussian-smoothed
    with a standard deviation in months (e.g. 0.5, 1, 2), or replaced by one
    pooled seasonally invariant coefficient ('C').

    Probabilistic component
    -----------------------
    A parametric predictive normal distribution is calibrated by

        m_f_tilde = a * m_f_anom
        s_f_tilde = b * s_f

    under the variance constraint used by Kharin et al. and with a,b chosen to
    minimize mean CRPS (equivalent to maximizing CRPSS for a fixed reference).
    The raw ensemble spread s_f is estimated from the full training period and
    is constant across years for a given target season.

    For precipitation, ``distribution='gamma'`` first maps observed and model
    climatological precipitation to standard-normal quantiles using fitted
    zero-inflated two-parameter gamma distributions F(x)=p0+(1-p0)G(x); values
    <= ``zero_threshold`` receive the mid-probability p0/2 (as for the SPI),
    instead of the CDF floor that produced z close to -6. Calibrates in transformed
    normal space, and computes tercile probabilities there.  Gamma parameters
    are estimated by the method of moments, which is scalable and fold-safe;
    the paper specifies a fitted two-parameter gamma but does not prescribe a
    fitting estimator in the text.

    Parameters
    ----------
    temporal_smoothing : {'U','C'} or float, default 1.0
        'U' = coefficients estimated independently for each season.
        float = circular Gaussian filter sigma in months.
        'C' = one coefficient estimated from anomalies pooled across seasons.
    distribution : {'normal','gamma','auto'}, default 'auto'
        'auto' uses gamma when ``positive_support=True`` and normal otherwise.
    positive_support : bool, default False
        Set True for precipitation. Deterministic forecasts are then clipped at
        zero after calibration.
    prob_grid_size : int, default 121
        Number of normalized signal-scaling candidates used for CRPS
        minimization. 121 gives spacing 1/120 over [0,1], or 1/60 over [-1,1].
    allow_negative_a : bool, default False
        Probabilistic calibration only. ``False`` is a conservative WASS2S
        amplitude-scaling default (a >= 0); the article does not explicitly
        specify this sign restriction. Set True as a sensitivity
        WASS2S sensitivity extension when anti-correlated ensemble-mean
        anomalies are intentionally allowed to reverse sign. Deterministic
        regression coefficients remain unconstrained, as in Eq. (2).
    min_samples : int, default 8
        Minimum valid training samples for a seasonal parameter estimate.
    tercile_reference : {'climatology', 'training'}, default 'climatology'
        Fixed empirical observed terciles from the requested climate period,
        computed separately for each target season. 'training' retains the
        legacy fold-specific parametric categories for research comparisons.
    probability_calibration : {'outer', 'inner', 'nested'}, default 'outer'
        Common workflow policy, with inner an alias of nested. Kharin's
        probabilities are native CRPS fits on each outer training set, so
        neither setting adds residual calibration or redundant subfolds.
    cv_buffer_years : int, default 0
        Number of neighbouring calendar years excluded on both sides of the
        held-out year. Use 0 to reproduce the paper's leave-one-year-out CV.
    purge_overlapping_seasons : bool, default True
        Additionally exclude training target windows sharing physical months
        with any held-out three-month season. This matters for the 12-season
        annual layout at year boundaries and adds no folds. False retains
        calendar-group-only exclusion for explicit historical comparisons.
    spatial_method : {'none','gaussian'}, default 'none'
        Optional regional Gaussian smoothing of fitted coefficient fields.
        This is a WASS2S extension; it is NOT the spherical-harmonic Tn
        truncation used in the paper.
    spatial_sigma : float or (float,float), default 1.0
        Gaussian sigma in grid cells when spatial_method='gaussian'.
    target_seasons : sequence of season names or central months, default 7-season AMJ..OND
        Consecutive target seasons used for temporal smoothing. The default is
        ("AMJ","MJJ","JJA","JAS","ASO","SON","OND").
    require_complete_window : bool, default True
        Require every configured target season to be present in training data.
    fixed_lead_window : sequence of int, default (3,4,5)
        Common lead offsets used for all target seasons. This metadata is
        validated by :meth:`assemble_fixed_lead_inputs`; the core calibrator
        never changes lead time internally.
    use_numba : bool, default True
        Use parallel Numba kernels for the CRPS grid search when Numba is
        installed. This is strongly recommended on large WASS2S grids.
    """

    _was_kharin2017 = True
    probability_labels = ("PB", "PN", "PA")
    all_seasons = np.arange(1, 13, dtype=int)
    default_7_seasons = ("AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND")

    # Rolling 3-month target seasons. Values are CENTRAL months.
    # Values are CENTRAL months, which are the coordinates on which temporal
    # smoothing must operate.  Initialization month is deliberately not used
    # as the smoothing coordinate.
    rolling_3month_center = {
        "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
        "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
    }

    def __init__(
        self,
        temporal_smoothing: TemporalMode = 1.0,
        distribution: str = "auto",
        positive_support: bool = False,
        prob_grid_size: int = 121,
        allow_negative_a: bool = False,
        min_samples: int = 8,
        cv_buffer_years: int = 0,
        spatial_method: str = "none",
        spatial_sigma: Union[float, Tuple[float, float]] = 1.0,
        target_seasons: Optional[Sequence[Union[str, int]]] = None,
        require_complete_window: bool = True,
        fixed_lead_window: Sequence[int] = (3, 4, 5),
        strict_12_seasons: Optional[bool] = None,
        use_numba: bool = True,
        eps: float = 1e-8,
        zero_threshold: float = 0.0,
        tercile_reference: str = "climatology",
        probability_calibration: str = "outer",
        purge_overlapping_seasons: bool = True,
    ):
        self.temporal_smoothing = self._normalize_temporal_mode(temporal_smoothing)
        self.distribution = str(distribution).lower()
        self.positive_support = bool(positive_support)
        self.prob_grid_size = max(21, int(prob_grid_size))
        if self.prob_grid_size % 2 == 0:
            self.prob_grid_size += 1
        self.allow_negative_a = bool(allow_negative_a)
        self.min_samples = max(3, int(min_samples))
        self.cv_buffer_years = max(0, int(cv_buffer_years))
        self.purge_overlapping_seasons = bool(purge_overlapping_seasons)
        self.spatial_method = str(spatial_method).lower()
        self.spatial_sigma = spatial_sigma
        if target_seasons is None:
            target_seasons = self.default_7_seasons
        self.target_season_names, self.seasons = self._resolve_target_seasons(target_seasons)
        self.require_complete_window = bool(require_complete_window)
        self.fixed_lead_window = tuple(int(v) for v in fixed_lead_window)
        self._validate_lead_window(self.fixed_lead_window)
        # Backward compatibility: strict_12_seasons=True explicitly requests paper mode.
        if strict_12_seasons is True:
            self.target_season_names, self.seasons = self._resolve_target_seasons(tuple(self.rolling_3month_center))
            self.require_complete_window = True
        self.strict_12_seasons = bool(strict_12_seasons) if strict_12_seasons is not None else False
        self.use_numba = bool(use_numba) and _KHARIN_NUMBA
        self.eps = float(eps)
        self.zero_threshold = float(zero_threshold)
        self.tercile_reference = str(tercile_reference).lower()
        self.probability_calibration = normalize_probability_calibration(probability_calibration)
        if self.tercile_reference not in {"climatology", "training"}:
            raise ValueError("tercile_reference must be 'climatology' or 'training'.")
        if not np.isfinite(self.zero_threshold) or self.zero_threshold < 0:
            raise ValueError("zero_threshold must be finite and nonnegative.")
        if not np.isfinite(self.eps) or not 0 < self.eps < 0.5:
            raise ValueError("eps must be finite and between 0 and 0.5.")

        if self.distribution not in {"auto", "normal", "gamma"}:
            raise ValueError("distribution must be 'auto', 'normal', or 'gamma'.")
        if self.spatial_method not in {"none", "gaussian"}:
            raise ValueError("spatial_method must be 'none' or 'gaussian'.")

        self.last_fit_: Optional[_KharinFit] = None
        self.cv_distribution_: Optional[xr.Dataset] = None
        self.cv_raw_distribution_: Optional[xr.Dataset] = None
        self.cv_reference_: Optional[xr.Dataset] = None
        self.cv_transformed_observation_: Optional[xr.DataArray] = None
        self.cv_raw_deterministic_: Optional[xr.DataArray] = None
        self.cv_raw_probability_: Optional[xr.DataArray] = None
        self.attrs: Dict[str, object] = {
            "method": "Kharin2017",
            "paper": "Kharin et al. 2017, Monthly Weather Review 145, 3545-3561",
            "target_seasons": ",".join(self.target_season_names),
            "target_months": ",".join(str(int(v)) for v in self.seasons),
            "fixed_lead_window": ",".join(f"{int(v):02d}" for v in self.fixed_lead_window),
            "constant_case": self.constant_label,
        }

    # ------------------------------------------------------------------
    # Public representation / configuration
    # ------------------------------------------------------------------
    def __repr__(self):
        return (
            "WAS_mme_Kharin2017("
            f"temporal_smoothing={self.temporal_smoothing!r}, "
            f"distribution={self._resolved_distribution()!r}, "
            f"positive_support={self.positive_support}, "
            f"target_seasons={self.target_season_names!r}, "
            f"fixed_lead_window={self.fixed_lead_window!r}, "
            f"spatial_method={self.spatial_method!r})"
        )

    def clone(self, **overrides):
        """Return an unfitted copy with selected constructor attributes changed."""
        # Reconstruct configuration; never copy potentially gigabyte-size CV
        # fields only to discard them, and validate overridden parameters.
        names = (
            "temporal_smoothing", "distribution", "positive_support",
            "prob_grid_size", "allow_negative_a", "min_samples",
            "cv_buffer_years", "spatial_method", "spatial_sigma",
            "require_complete_window", "fixed_lead_window", "use_numba", "eps",
            "zero_threshold", "tercile_reference", "probability_calibration",
            "purge_overlapping_seasons",
        )
        params = {name: getattr(self, name) for name in names}
        params["target_seasons"] = self.target_season_names
        params["strict_12_seasons"] = self.strict_12_seasons
        unknown = set(overrides) - set(params)
        if unknown:
            raise AttributeError(f"Unknown constructor parameters: {sorted(unknown)}")
        params.update(overrides)
        return type(self)(**params)

    @staticmethod
    def _normalize_temporal_mode(mode: TemporalMode):
        if isinstance(mode, str):
            u = mode.strip().upper()
            if u in {"U", "UNSMOOTHED", "NONE", "RAW"}:
                return "U"
            if u in {"C", "CONSTANT", "INVARIANT", "POOLED"}:
                return "C"
            try:
                x = float(mode)
            except ValueError as exc:
                raise ValueError("temporal_smoothing must be 'U', 'C', or a nonnegative float.") from exc
            if not np.isfinite(x) or x < 0:
                raise ValueError("temporal smoothing sigma must be finite and >= 0.")
            return "U" if x == 0 else x
        x = float(mode)
        if not np.isfinite(x) or x < 0:
            raise ValueError("temporal smoothing sigma must be finite and >= 0.")
        return "U" if x == 0 else x

    def _resolved_distribution(self):
        if self.distribution == "auto":
            return "gamma" if self.positive_support else "normal"
        return self.distribution

    @classmethod
    def _resolve_target_seasons(cls, target_seasons):
        names = []
        months = []
        reverse = {v: k for k, v in cls.rolling_3month_center.items()}
        for item in target_seasons:
            if isinstance(item, str):
                name = item.strip().upper()
                if name not in cls.rolling_3month_center:
                    raise ValueError(f"Unknown 3-month target season {item!r}.")
                month = int(cls.rolling_3month_center[name])
            else:
                month = int(item)
                if month not in reverse:
                    raise ValueError(f"Target central month must lie in 1..12, got {item!r}.")
                name = reverse[month]
            names.append(name); months.append(month)
        if not months:
            raise ValueError("target_seasons cannot be empty.")
        if len(set(months)) != len(months):
            raise ValueError("target_seasons contains duplicate central months.")
        # Require a consecutive window on the annual cycle. This is the setting
        # in which adjacent overlapping seasons genuinely borrow information.
        diffs = [((months[i+1] - months[i]) % 12) for i in range(len(months)-1)]
        if any(d != 1 for d in diffs):
            raise ValueError(
                "target_seasons must be consecutive overlapping seasons in calendar order; "
                f"got {names}."
            )
        return tuple(names), np.asarray(months, dtype=int)

    @staticmethod
    def _validate_lead_window(lead_window):
        lead = tuple(int(v) for v in lead_window)
        if len(lead) != 3:
            raise ValueError("fixed_lead_window must contain exactly three lead offsets for a 3-month target season.")
        if lead[1] != lead[0] + 1 or lead[2] != lead[1] + 1:
            raise ValueError(f"fixed_lead_window must be three consecutive offsets, got {lead}.")
        if lead[0] < 0:
            raise ValueError("lead offsets must be non-negative.")
        return lead

    @property
    def constant_label(self):
        return f"C{len(self.seasons)}"

    def temporal_label(self, mode=None):
        mode = self.temporal_smoothing if mode is None else self._normalize_temporal_mode(mode)
        if mode == "C":
            return self.constant_label
        if mode == "U":
            return "U"
        return f"G{float(mode):g}"

    @classmethod
    def paper_12_seasons(cls, **kwargs):
        """Construct the exact 12-target-season temporal configuration of the paper."""
        kwargs = dict(kwargs)
        kwargs["target_seasons"] = tuple(cls.rolling_3month_center)
        kwargs["require_complete_window"] = True
        kwargs.pop("strict_12_seasons", None)
        return cls(**kwargs)

    # ------------------------------------------------------------------
    # Input conventions and seasonal indexing
    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_obs(obs: xr.DataArray) -> xr.DataArray:
        if not isinstance(obs, xr.DataArray):
            raise TypeError("Predictant must be an xarray.DataArray.")
        if "M" in obs.dims:
            if obs.sizes["M"] != 1:
                warnings.warn("Predictant has multiple M members; using M=0 as observation.")
            obs = obs.isel(M=0, drop=True)
        missing = {"T", "Y", "X"} - set(obs.dims)
        if missing:
            raise ValueError(f"Predictant is missing dimensions {sorted(missing)}.")
        for d in list(obs.dims):
            if d not in {"T", "Y", "X"}:
                if obs.sizes[d] == 1:
                    obs = obs.isel({d: 0}, drop=True)
                else:
                    raise ValueError(f"Unsupported Predictant dimension {d!r}.")
        obs = obs.transpose("T", "Y", "X").astype(float)
        return obs.where(np.isfinite(obs))

    @staticmethod
    def _prepare_ensemble(ens: xr.DataArray, name="Predictor") -> xr.DataArray:
        if not isinstance(ens, xr.DataArray):
            raise TypeError(f"{name} must be an xarray.DataArray.")
        missing = {"T", "M", "Y", "X"} - set(ens.dims)
        if missing:
            raise ValueError(f"{name} is missing dimensions {sorted(missing)}.")
        for d in list(ens.dims):
            if d not in {"T", "M", "Y", "X"}:
                if ens.sizes[d] == 1:
                    ens = ens.isel({d: 0}, drop=True)
                else:
                    raise ValueError(f"Unsupported {name} dimension {d!r}.")
        ens = ens.transpose("T", "M", "Y", "X").astype(float)
        if ens.sizes["M"] < 2:
            raise ValueError(f"{name}: Kharin spread calibration requires at least two ensemble members.")
        return ens.where(np.isfinite(ens))

    @staticmethod
    def _months_years(obj: xr.DataArray) -> Tuple[np.ndarray, np.ndarray]:
        """Return target-season month and target year.

        The preferred explicit coordinates are ``season_month`` (1..12) and
        ``season_year``.  They are useful because WASS2S forecast ``T`` can
        represent an initialization date, whereas Kharin temporal smoothing
        is indexed by the TARGET season (for 3-month means, naturally its
        central month).  If those coordinates are absent, ``T.dt.month/year``
        are used.
        """
        da = obj
        if isinstance(da, xr.DataArray) and "T" in da.dims:
            if "season_month" in da.coords:
                months = np.asarray(da["season_month"].values, dtype=int)
            elif "target_month" in da.coords:
                months = np.asarray(da["target_month"].values, dtype=int)
            else:
                try:
                    months = np.asarray(da["T"].dt.month.values, dtype=int)
                except Exception as exc:
                    raise ValueError(
                        "Provide a datetime-like T or a season_month/target_month coordinate."
                    ) from exc

            if "season_year" in da.coords:
                years = np.asarray(da["season_year"].values, dtype=int)
            elif "target_year" in da.coords:
                years = np.asarray(da["target_year"].values, dtype=int)
            else:
                try:
                    years = np.asarray(da["T"].dt.year.values, dtype=int)
                except Exception as exc:
                    raise ValueError(
                        "Provide a datetime-like T or a season_year/target_year coordinate."
                    ) from exc
        else:
            # Backward-compatible path for callers passing the T coordinate.
            try:
                months = np.asarray(da.dt.month.values, dtype=int)
                years = np.asarray(da.dt.year.values, dtype=int)
            except Exception as exc:
                raise ValueError("T must be datetime-like so target month and year can be inferred.") from exc

        if months.shape != years.shape:
            raise ValueError("season month/year coordinates must have the same shape.")
        if months.ndim != 1 or months.size != obj.sizes.get("T", months.size):
            raise ValueError("Target season month/year must have one value per T sample.")
        if np.any((months < 1) | (months > 12)):
            raise ValueError("season_month/target_month values must lie in 1..12.")
        return months, years

    @classmethod
    def assemble_seasonal_inputs(
        cls,
        observations: Dict[str, xr.DataArray],
        hindcasts: Dict[str, xr.DataArray],
        season_center_month: Optional[Dict[str, int]] = None,
        *,
        target_year_from: str = "observation",
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """Assemble separate seasonal WASS2S datasets into one target-season cube.

        This incorporates the most useful practical point from the companion
        notebook: the temporal smoother must use the TARGET/CENTRAL month, not
        the initialization month and not the dictionary/list position.

        Parameters
        ----------
        observations : mapping season -> DataArray(T,Y,X)
        hindcasts : mapping season -> DataArray(T,M,Y,X)
        season_center_month : mapping season -> int, optional
            Defaults to ``rolling_3month_center`` for names DJF..NDJ.
        target_year_from : {'observation','hindcast'}
            Which input supplies the target-year ordering. Arrays are paired
            by year labels, never by position. For an initialization date in
            the previous calendar year, supply explicit season_year labels.

        Returns
        -------
        obs_all, hdcst_all
            Concatenated arrays sorted by target ``T`` with auxiliary
            ``season_month``, ``season_year`` and ``season_name`` coordinates.
        """
        centers = dict(cls.rolling_3month_center if season_center_month is None else season_center_month)
        keys = [k for k in observations if k in hindcasts]
        if not keys:
            raise ValueError("No common season keys between observations and hindcasts.")
        unknown = [k for k in keys if k not in centers]
        if unknown:
            raise ValueError(f"Missing center-month definitions for seasons: {unknown}")
        if target_year_from not in {"observation", "hindcast"}:
            raise ValueError("target_year_from must be 'observation' or 'hindcast'.")

        obs_parts = []
        h_parts = []
        for name in keys:
            o = cls._prepare_obs(observations[name])
            h = cls._prepare_ensemble(hindcasts[name], f"hindcast[{name}]")
            if o.sizes["T"] != h.sizes["T"]:
                raise ValueError(
                    f"Season {name}: observation/hindcast T lengths differ "
                    f"({o.sizes['T']} vs {h.sizes['T']})."
                )
            _, oy = cls._months_years(o)
            _, hy = cls._months_years(h)
            oi, hi = pd.Index(oy), pd.Index(hy)
            if oi.has_duplicates or hi.has_duplicates or set(oy) != set(hy):
                raise ValueError(
                    f"Season {name}: unique matching target years are required; "
                    "set season_year for hindcasts initialized in the previous year."
                )
            if target_year_from == "observation":
                h = h.isel(T=hi.get_indexer(oi))
            else:
                o = o.isel(T=oi.get_indexer(hi))
            if not h.Y.identical(o.Y) or not h.X.identical(o.X):
                h = h.interp(Y=o.Y, X=o.X, method="nearest")

            src = o if target_year_from == "observation" else h
            try:
                _, years = cls._months_years(src)
            except Exception as exc:
                raise ValueError(f"Season {name}: source T must be datetime-like.") from exc
            cm = int(centers[name])
            target_t = pd.to_datetime([f"{int(y):04d}-{cm:02d}-01" for y in years])
            sm = xr.DataArray(np.full(len(years), cm, dtype=int), dims="T", coords={"T": target_t})
            sy = xr.DataArray(years.astype(int), dims="T", coords={"T": target_t})
            sn = xr.DataArray(np.asarray([name] * len(years), dtype=object), dims="T", coords={"T": target_t})

            o = o.assign_coords(T=target_t, season_month=sm, season_year=sy, season_name=sn)
            h = h.assign_coords(T=target_t, season_month=sm, season_year=sy, season_name=sn)
            obs_parts.append(o)
            h_parts.append(h)

        obs_all = xr.concat(obs_parts, dim="T").sortby("T").transpose("T", "Y", "X")
        h_all = xr.concat(h_parts, dim="T").sortby("T").transpose("T", "M", "Y", "X")
        if obs_all.get_index("T").duplicated().any():
            raise ValueError(
                "Duplicate target T values after seasonal assembly. Check target-year conventions "
                "and center-month definitions."
            )
        return obs_all, h_all

    @classmethod
    def assemble_fixed_lead_inputs(
        cls,
        observations: Dict[str, xr.DataArray],
        hindcasts: Dict[str, xr.DataArray],
        *,
        target_seasons: Sequence[str] = default_7_seasons,
        initialization_months: Optional[Dict[str, int]] = None,
        lead_window: Sequence[int] = (3, 4, 5),
        target_year_from: str = "observation",
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """Assemble the WASS2S fixed-lead / moving-initialization experiment.

        The default design is exactly::

            IC Jan + leads 03-05 -> AMJ
            IC Feb + leads 03-05 -> MJJ
            IC Mar + leads 03-05 -> JJA
            IC Apr + leads 03-05 -> JAS
            IC May + leads 03-05 -> ASO
            IC Jun + leads 03-05 -> SON
            IC Jul + leads 03-05 -> OND

        Lead offsets are interpreted as calendar-month offsets from the
        initialization month (lead 03 from January -> April).  The function
        validates the mapping before concatenating anything, so a notebook
        cannot silently change lead time while pretending to compare seasons.
        """
        names, months = cls._resolve_target_seasons(target_seasons)
        lead = cls._validate_lead_window(tuple(int(v) for v in lead_window))
        if initialization_months is None:
            # init + centre lead = target central month
            centre_lead = lead[1]
            initialization_months = {
                name: ((int(cm) - 1 - centre_lead) % 12) + 1
                for name, cm in zip(names, months)
            }
        else:
            initialization_months = {str(k).upper(): int(v) for k, v in initialization_months.items()}

        errors = []
        for name, cm in zip(names, months):
            if name not in initialization_months:
                errors.append(f"{name}: missing initialization month")
                continue
            init = int(initialization_months[name])
            if not 1 <= init <= 12:
                errors.append(f"{name}: invalid initialization month {init}")
                continue
            target_months = tuple(((init - 1 + l) % 12) + 1 for l in lead)
            expected_center = int(cm)
            if target_months[1] != expected_center:
                errors.append(
                    f"{name}: IC={init:02d}, lead={lead} gives central month "
                    f"{target_months[1]:02d}, expected {expected_center:02d}"
                )
        if errors:
            raise ValueError("Invalid fixed-lead seasonal design:\n  - " + "\n  - ".join(errors))

        obs_sel = {name: observations[name] for name in names if name in observations}
        h_sel = {name: hindcasts[name] for name in names if name in hindcasts}
        missing = [name for name in names if name not in obs_sel or name not in h_sel]
        if missing:
            raise ValueError(f"Missing configured target seasons: {missing}")
        centers = {name: int(cm) for name, cm in zip(names, months)}
        obs_all, h_all = cls.assemble_seasonal_inputs(
            obs_sel, h_sel, centers, target_year_from=target_year_from
        )
        # Traceability coordinates; these are not used as the smoothing axis.
        sn = np.asarray(obs_all["season_name"].values).astype(str)
        init = np.asarray([initialization_months[s] for s in sn], dtype=int)
        l0 = np.full(init.shape, lead[0], dtype=int)
        l1 = np.full(init.shape, lead[1], dtype=int)
        l2 = np.full(init.shape, lead[2], dtype=int)
        coords = {
            "initialization_month": ("T", init),
            "lead_start": ("T", l0),
            "lead_center": ("T", l1),
            "lead_end": ("T", l2),
        }
        obs_all = obs_all.assign_coords(**coords)
        h_all = h_all.assign_coords(**coords)
        for da in (obs_all, h_all):
            da.attrs["kharin_design"] = "fixed_lead_moving_initialization"
            da.attrs["kharin_target_seasons"] = ",".join(names)
            da.attrs["kharin_fixed_lead_window"] = ",".join(f"{v:02d}" for v in lead)
        return obs_all, h_all

    def _check_fixed_lead_metadata(self, da, name="input"):
        """Validate fixed-lead traceability coordinates when they are present."""
        expected = tuple(int(v) for v in self.fixed_lead_window)
        for coord, value in zip(("lead_start", "lead_center", "lead_end"), expected):
            if coord in da.coords:
                vals = np.unique(np.asarray(da[coord].values, dtype=float).ravel())
                if vals.size != 1 or not np.isfinite(vals[0]) or vals[0] != value:
                    raise ValueError(
                        f"{name} violates fixed lead {expected}: coordinate {coord} has {vals.tolist()}."
                    )
        attr = da.attrs.get("kharin_fixed_lead_window")
        if attr is not None:
            try:
                got = tuple(int(str(v).strip()) for v in str(attr).split(","))
            except Exception:
                raise ValueError(f"{name}: invalid kharin_fixed_lead_window metadata.")
            if got != expected:
                raise ValueError(
                    f"{name} metadata says lead window {got}, but model is configured for {expected}."
                )

    def _check_season_coverage(self, months):
        u = np.unique(np.asarray(months, dtype=int))
        expected = np.asarray(self.seasons, dtype=int)
        missing = expected[~np.isin(expected, u)]
        extra = u[~np.isin(u, expected)]
        if missing.size:
            msg = (
                f"Kharin2017 configured target window {self.target_season_names} "
                f"requires central months {expected.tolist()}, but is missing {missing.tolist()}."
            )
            if self.require_complete_window:
                raise ValueError(msg)
            warnings.warn(msg, RuntimeWarning, stacklevel=3)
        if extra.size:
            raise ValueError(
                f"Input contains target months {extra.tolist()} outside configured window "
                f"{expected.tolist()}. Subset the input explicitly or configure target_seasons accordingly."
            )
        if len(expected) == 1:
            warnings.warn(
                "Only one target season is configured; temporal smoothing cannot borrow information across seasons.",
                RuntimeWarning, stacklevel=3,
            )

    def _align_training(self, obs, hindcast):
        obs = self._prepare_obs(obs)
        h = self._prepare_ensemble(hindcast, "hindcast")
        if h.sizes["T"] != obs.sizes["T"]:
            raise ValueError("hindcast and Predictant must have the same T length.")
        om, oy = self._months_years(obs)
        hm, hy = self._months_years(h)
        ok = pd.MultiIndex.from_arrays([oy, om])
        hk = pd.MultiIndex.from_arrays([hy, hm])
        if ok.has_duplicates or hk.has_duplicates:
            raise ValueError("Each target season/year must occur exactly once.")
        order = hk.get_indexer(ok)
        if np.any(order < 0):
            raise ValueError(
                "hindcast and observation target season/year labels differ. "
                "Supply season_month and season_year when T is an initialization date; "
                "positional time relabeling is not scientifically safe."
            )
        h = h.isel(T=order).assign_coords(T=obs["T"])
        if not h.Y.identical(obs.Y) or not h.X.identical(obs.X):
            h = h.interp(Y=obs.Y, X=obs.X, method="nearest")
        self._check_fixed_lead_metadata(h, "hindcast")
        months, _ = self._months_years(obs)
        self._check_season_coverage(months)
        return obs, h

    def _align_forecast(self, forecast, obs, training_months):
        f = self._prepare_ensemble(forecast, "forecast")
        if not f.Y.identical(obs.Y) or not f.X.identical(obs.X):
            f = f.interp(Y=obs.Y, X=obs.X, method="nearest")
        self._check_fixed_lead_metadata(f, "forecast")
        fmonths, _ = self._months_years(f)
        utrain = np.unique(training_months)
        if np.any(~np.isin(fmonths, utrain)):
            missing = np.unique(fmonths[~np.isin(fmonths, utrain)])
            raise ValueError(
                f"Forecast contains target month(s) {missing.tolist()} absent from training data."
            )
        return f, fmonths

    # ------------------------------------------------------------------
    # Seasonal climatologies and smoothing
    # ------------------------------------------------------------------
    def _seasonal_mean(self, da: xr.DataArray, months: np.ndarray, dims="T"):
        season = xr.DataArray(months, dims=("T",), coords={"T": da["T"]}, name="_season")
        out = da.assign_coords(_season=season).groupby("_season").mean(dims, skipna=True)
        out = out.rename({"_season": "season"}).reindex(season=self.seasons)
        return out

    def _seasonal_member_spread(self, h: xr.DataArray, months: np.ndarray):
        # Ensemble variance is evaluated for every initialization/year, then
        # averaged over the full training period for each target season.
        v = h.var("M", skipna=True, ddof=1)
        sf2 = self._seasonal_mean(v, months, dims="T")
        # Use the SAME floor in fitting and application. Otherwise identical
        # members give b>0 during fitting but b*0=0 at prediction, collapsing
        # an uncertain calibrated forecast to an almost point mass.
        return np.sqrt(sf2.clip(min=self.eps))

    def _select_season(self, seasonal: xr.DataArray, months: np.ndarray, Tcoord):
        idx = xr.DataArray(months, dims=("T",), coords={"T": Tcoord})
        out = seasonal.sel(season=idx)
        if "season" in out.coords and "season" not in out.dims:
            out = out.drop_vars("season")
        return out

    def _smooth_temporal_window(self, field: xr.DataArray, sigma: float) -> xr.DataArray:
        """Gaussian smoothing over only the actually configured target seasons.

        Weights use circular calendar-month distance, but missing/unconfigured
        seasons never contribute. This is crucial for the 7-season AMJ..OND
        adaptation: no winter coefficient is implicitly invented by a 12-slot
        convolution.
        """
        sigma = float(sigma)
        if sigma <= 0:
            return field
        f = field.transpose("season", "Y", "X").reindex(season=self.seasons)
        smonth = np.asarray(f["season"].values, dtype=float)
        d = np.abs(smonth[:, None] - smonth[None, :])
        d = np.minimum(d, 12.0 - d)
        W = np.exp(-0.5 * (d / sigma) ** 2)
        w = xr.DataArray(
            W, dims=("target_season", "source_season"),
            coords={"target_season": f["season"].values, "source_season": f["season"].values},
        )
        src = f.rename({"season": "source_season"})
        valid = np.isfinite(src)
        num = (src.fillna(0.0) * valid.astype(float) * w).sum("source_season", skipna=True)
        den = (valid.astype(float) * w).sum("source_season", skipna=True)
        out = (num / den.where(den > self.eps)).rename({"target_season": "season"})
        out = out.transpose("season", "Y", "X")
        out.name = field.name
        return out

    def _smooth_spatial(self, field: xr.DataArray) -> xr.DataArray:
        if self.spatial_method == "none":
            return field
        sig = self.spatial_sigma
        if np.isscalar(sig):
            sy = sx = float(sig)
        else:
            sy, sx = map(float, sig)
        arr = np.asarray(field.transpose("season", "Y", "X").values, dtype=float)
        w = np.isfinite(arr).astype(float)
        z = np.where(np.isfinite(arr), arr, 0.0)
        sigma = (0.0, sy, sx)
        num = gaussian_filter(z * w, sigma=sigma, mode=("nearest", "nearest", "nearest"))
        den = gaussian_filter(w, sigma=sigma, mode=("nearest", "nearest", "nearest"))
        sm = np.where(den > self.eps, num / den, np.nan)
        return xr.DataArray(sm, coords=field.transpose("season", "Y", "X").coords,
                            dims=("season", "Y", "X"), name=field.name)

    def _smooth_coefficients(self, field: xr.DataArray) -> xr.DataArray:
        if isinstance(self.temporal_smoothing, float):
            field = self._smooth_temporal_window(field, self.temporal_smoothing)
        field = self._smooth_spatial(field)
        return field

    # ------------------------------------------------------------------
    # Deterministic regression calibration: eqs. (1)-(2)
    # ------------------------------------------------------------------
    def _fit_deterministic(self, h, obs, months):
        fbar = h.mean("M", skipna=True)
        paired = np.isfinite(fbar) & np.isfinite(obs)
        fbar = fbar.where(paired)
        obs = obs.where(paired)
        mu_f = self._seasonal_mean(fbar, months)
        mu_o = self._seasonal_mean(obs, months)
        f_anom = fbar - self._select_season(mu_f, months, fbar["T"])
        o_anom = obs - self._select_season(mu_o, months, obs["T"])

        mode = self.temporal_smoothing
        season_coord = xr.DataArray(months, dims="T", coords={"T": obs["T"]}, name="_season")

        if mode == "C":
            valid = np.isfinite(f_anom) & np.isfinite(o_anom)
            n = valid.sum("T")
            num = (f_anom * o_anom).where(valid).sum("T", skipna=True)
            den = (f_anom * f_anom).where(valid).sum("T", skipna=True)
            a0 = (num / den.where(np.abs(den) > self.eps)).where(n >= self.min_samples)
            a0 = xr.where((den <= self.eps) & (n >= self.min_samples), 0.0, a0)
            a = a0.expand_dims(season=self.seasons).transpose("season", "Y", "X")
            a = self._smooth_spatial(a)
        else:
            valid = np.isfinite(f_anom) & np.isfinite(o_anom)
            num = (f_anom * o_anom).where(valid).assign_coords(_season=season_coord).groupby("_season").sum("T", skipna=True)
            den = (f_anom * f_anom).where(valid).assign_coords(_season=season_coord).groupby("_season").sum("T", skipna=True)
            n = valid.assign_coords(_season=season_coord).groupby("_season").sum("T")
            a = (num / den.where(np.abs(den) > self.eps)).where(n >= self.min_samples)
            a = xr.where((den <= self.eps) & (n >= self.min_samples), 0.0, a)
            a = a.rename({"_season": "season"}).reindex(season=self.seasons)
            a = self._smooth_coefficients(a)

        a.name = "a_deterministic"
        return mu_o, mu_f, a

    def _apply_deterministic(self, fit: _KharinFit, ens: xr.DataArray, months):
        fbar = ens.mean("M", skipna=True)
        mu_f = self._select_season(fit.det_fc_clim, months, ens["T"])
        mu_o = self._select_season(fit.det_obs_clim, months, ens["T"])
        a = self._select_season(fit.det_a, months, ens["T"])
        pred = mu_o + a * (fbar - mu_f)
        pred = pred.transpose("T", "Y", "X")
        if self.positive_support:
            pred = pred.clip(min=0.0)
        return pred

    def _apply_raw_deterministic(self, fit: _KharinFit, ens: xr.DataArray, months):
        """Bias-corrected but otherwise unscaled ensemble-mean baseline.

        This corresponds to the article's unadjusted anomaly forecast: model
        seasonal climatology is removed, the anomaly is transferred to the
        observed climatology, and the regression multiplier is fixed to 1.
        """
        fbar = ens.mean("M", skipna=True)
        mu_f = self._select_season(fit.det_fc_clim, months, ens["T"])
        mu_o = self._select_season(fit.det_obs_clim, months, ens["T"])
        pred = (mu_o + (fbar - mu_f)).transpose("T", "Y", "X")
        if self.positive_support:
            pred = pred.clip(min=0.0)
        return pred

    # ------------------------------------------------------------------
    # Gamma transform for precipitation
    # ------------------------------------------------------------------
    def _gamma_moments(self, data, months, dims):
        """Zero-inflated gamma by moments of the positive values, per season.

        Returns shape, scale, p0 (probability of <= zero_threshold) and the
        probability level assigned to a zero (p0/2, or 1/(2(n+1)) if no zero
        was observed in training). Moments need ``min_samples`` positives.
        ``shape == 0`` is an explicit degenerate-marginal flag, never a gamma
        parameter: the positive atom is at ``scale``. It also represents an
        entirely dry sample with p0=1 and scale=0, without artificial variance.
        """
        shapes, scales, zeros, levels = [], [], [], []
        for s in self.seasons:
            x = data.isel(T=np.where(months == s)[0])
            finite = np.isfinite(x)
            n = finite.sum(dims)
            wet = x.where(x > self.zero_threshold)
            n_pos = np.isfinite(wet).sum(dims)
            p0 = ((finite & (x <= self.zero_threshold)).sum(dims) / n.where(n > 0))
            mu = wet.mean(dims, skipna=True)
            va = wet.var(dims, skipna=True, ddof=1)
            ok = (mu > self.eps) & (va > self.eps) & (n_pos >= self.min_samples)
            constant = (va == 0) & (n_pos >= self.min_samples)
            dry = (n_pos == 0) & (n >= self.min_samples)
            degenerate = constant | dry
            shapes.append(xr.where(degenerate, 0., (mu * mu / va.where(va > self.eps)).where(ok)))
            scales.append(xr.where(degenerate, xr.where(dry, 0., mu),
                                   (va / mu.where(mu > self.eps)).where(ok)))
            zeros.append(p0.where(ok | degenerate))
            levels.append(xr.where(p0 > 0, 0.5 * p0, 0.5 / (n + 1.0)).where(ok | degenerate))

        def stack(parts):
            return xr.concat(parts, dim=xr.IndexVariable("season", self.seasons)).transpose("season", "Y", "X")
        return stack(shapes), stack(scales), stack(zeros), stack(levels)

    def _gamma_moments_obs(self, obs, months):
        return self._gamma_moments(obs, months, "T")

    def _gamma_moments_fc(self, h, months):
        return self._gamma_moments(h, months, ("T", "M"))

    def _gamma_cdf(self, x, shape, scale, p0):
        """Mixed CDF P(X <= x) in physical units (x may be an event threshold)."""
        cont = xr.apply_ufunc(lambda v, a, sc: gdtr(1.0 / sc, a, np.maximum(v, 0.0)),
                              x, shape.where(shape > 0), scale.where(shape > 0),
                              dask="parallelized", output_dtypes=[float])
        cont = xr.where(shape == 0, (x >= scale).astype(float), cont)
        p0 = 0.0 if p0 is None else p0
        u = xr.where(x > self.zero_threshold, p0 + (1.0 - p0) * cont, p0)
        u = xr.where(x < 0, 0., u)
        return u.where(np.isfinite(x) & np.isfinite(shape) & np.isfinite(scale))

    def _gamma_to_normal(self, da, months, shape_s, scale_s, p0_s=None, zero_u_s=None):
        if bool((da < 0).any()):
            raise ValueError("Gamma precipitation calibration requires nonnegative input.")
        shape = self._select_season(shape_s, months, da["T"])
        scale = self._select_season(scale_s, months, da["T"])
        p0 = 0.0 if p0_s is None else self._select_season(p0_s, months, da["T"])
        cont = xr.apply_ufunc(lambda v, a, sc: gdtr(1.0 / sc, a, np.maximum(v, 0.0)),
                              da, shape.where(shape > 0), scale.where(shape > 0),
                              dask="parallelized", output_dtypes=[float])
        # Mid-CDF at a positive atom, matching the treatment of the zero atom.
        cont = xr.where(shape == 0, xr.where(da == scale, .5, (da > scale).astype(float)), cont)
        u = p0 + (1.0 - p0) * cont
        if zero_u_s is not None:
            u = xr.where(da > self.zero_threshold, u, self._select_season(zero_u_s, months, da["T"]))
        u = u.where(np.isfinite(da) & np.isfinite(shape) & np.isfinite(scale))
        u = u.clip(min=1e-6, max=1.0 - 1e-6)
        return xr.apply_ufunc(ndtri, u, dask="parallelized", output_dtypes=[float])

    # ------------------------------------------------------------------
    # CRPS calibration of mean/spread coefficients
    # ------------------------------------------------------------------
    @staticmethod
    def _normal_crps_np(mu, sigma, y, eps=1e-12):
        sigma = np.asarray(sigma, dtype=float)
        mu = np.asarray(mu, dtype=float)
        y = np.asarray(y, dtype=float)
        safe = np.maximum(sigma, eps)
        z = (y - mu) / safe
        out = safe * (
            z * (2.0 * norm.cdf(z) - 1.0)
            + 2.0 * norm.pdf(z)
            - 1.0 / math.sqrt(math.pi)
        )
        deterministic = sigma <= eps
        if np.any(deterministic):
            out = np.where(deterministic, np.abs(y - mu), out)
        return out

    def _q_grid(self):
        lo = -1.0 if self.allow_negative_a else 0.0
        return np.linspace(lo, 1.0, self.prob_grid_size)

    def _fit_ab_one_season_np(self, f_anom, o_anom, sf):
        """CRPS-optimal a,b for one season, vectorized over flattened cells."""
        f_anom = np.asarray(f_anom, dtype=np.float64)
        o_anom = np.asarray(o_anom, dtype=np.float64)
        sf = np.asarray(sf, dtype=np.float64)
        if self.use_numba:
            return _fit_ab_season_numba(
                f_anom, o_anom, sf, np.asarray(self._q_grid(), dtype=np.float64),
                self.min_samples, self.eps,
            )
        # arrays f,o: (n, cell); sf: (cell,)
        valid = np.isfinite(f_anom) & np.isfinite(o_anom) & np.isfinite(sf)[None, :]
        nvalid = valid.sum(axis=0)
        f = np.where(valid, f_anom, np.nan)
        o = np.where(valid, o_anom, np.nan)
        vf = np.nanvar(f, axis=0, ddof=1)
        vo = np.nanvar(o, axis=0, ddof=1)
        sf2 = sf * sf

        scale = np.sqrt(np.maximum(vo, 0.0) / np.maximum(vf, self.eps))
        best = np.full(vf.shape, np.inf)
        best_q = np.full(vf.shape, np.nan)

        for q in self._q_grid():
            a = q * scale
            sigma = np.sqrt(np.maximum(vo * (1.0 - q * q), 0.0))
            mu = f * a[None, :]
            crps = self._normal_crps_np(mu, sigma[None, :], o, eps=self.eps)
            score = np.nanmean(np.where(valid, crps, np.nan), axis=0)
            take = np.isfinite(score) & (score < best)
            best[take] = score[take]
            best_q[take] = q

        a = best_q * scale
        b = np.sqrt(
            np.maximum(vo * (1.0 - best_q * best_q), 0.0)
            / np.maximum(sf2, self.eps)
        )
        bad = (nvalid < self.min_samples) | ~np.isfinite(vf) | ~np.isfinite(vo) | ~np.isfinite(sf)
        # When the ensemble-mean anomaly has essentially no variance, a=0 is
        # the only identifiable choice; spread is calibrated to observed var.
        flat_f = vf <= self.eps
        a = np.where(flat_f & ~bad, 0.0, a)
        b = np.where(flat_f & ~bad, np.sqrt(vo / np.maximum(sf2, self.eps)), b)
        a = np.where((vo == 0) & ~bad, 0., a)
        b = np.where((vo == 0) & ~bad, 0., b)
        a[bad] = np.nan
        b[bad] = np.nan
        return a, b

    def _fit_ab_constant_np(self, f_anom, o_anom, sf_t):
        """Pooled, seasonally invariant CRPS calibration over all seasons."""
        f_anom = np.asarray(f_anom, dtype=np.float64)
        o_anom = np.asarray(o_anom, dtype=np.float64)
        sf_t = np.asarray(sf_t, dtype=np.float64)
        if self.use_numba:
            return _fit_ab_constant_numba(
                f_anom, o_anom, sf_t, np.asarray(self._q_grid(), dtype=np.float64),
                self.min_samples, self.eps,
            )
        valid = np.isfinite(f_anom) & np.isfinite(o_anom) & np.isfinite(sf_t)
        nvalid = valid.sum(axis=0)
        f = np.where(valid, f_anom, np.nan)
        o = np.where(valid, o_anom, np.nan)
        s = np.where(valid, sf_t, np.nan)
        vf = np.nanvar(f, axis=0, ddof=1)
        vo = np.nanvar(o, axis=0, ddof=1)
        msf2 = np.nanmean(s * s, axis=0)
        scale = np.sqrt(np.maximum(vo, 0.0) / np.maximum(vf, self.eps))
        best = np.full(vf.shape, np.inf)
        best_q = np.full(vf.shape, np.nan)

        for q in self._q_grid():
            a = q * scale
            b = np.sqrt(
                np.maximum(vo * (1.0 - q * q), 0.0)
                / np.maximum(msf2, self.eps)
            )
            mu = f * a[None, :]
            sigma = s * b[None, :]
            crps = self._normal_crps_np(mu, sigma, o, eps=self.eps)
            score = np.nanmean(np.where(valid, crps, np.nan), axis=0)
            take = np.isfinite(score) & (score < best)
            best[take] = score[take]
            best_q[take] = q

        a = best_q * scale
        b = np.sqrt(
            np.maximum(vo * (1.0 - best_q * best_q), 0.0)
            / np.maximum(msf2, self.eps)
        )
        bad = (nvalid < self.min_samples) | ~np.isfinite(vf) | ~np.isfinite(vo)
        flat_f = vf <= self.eps
        a = np.where(flat_f & ~bad, 0.0, a)
        b = np.where(flat_f & ~bad, np.sqrt(vo / np.maximum(msf2, self.eps)), b)
        a = np.where((vo == 0) & ~bad, 0., a)
        b = np.where((vo == 0) & ~bad, 0., b)
        a[bad] = np.nan
        b[bad] = np.nan
        return a, b

    def _fit_probability(self, h, obs, months):
        distribution = self._resolved_distribution()
        gamma_obs_shape = gamma_obs_scale = gamma_fc_shape = gamma_fc_scale = None
        gamma_obs_p0 = gamma_obs_zero_u = gamma_fc_p0 = gamma_fc_zero_u = None

        if distribution == "gamma":
            gamma_obs_shape, gamma_obs_scale, gamma_obs_p0, gamma_obs_zero_u = self._gamma_moments_obs(obs, months)
            gamma_fc_shape, gamma_fc_scale, gamma_fc_p0, gamma_fc_zero_u = self._gamma_moments_fc(h, months)
            h_t = self._gamma_to_normal(h, months, gamma_fc_shape, gamma_fc_scale, gamma_fc_p0, gamma_fc_zero_u)
            o_t = self._gamma_to_normal(obs, months, gamma_obs_shape, gamma_obs_scale, gamma_obs_p0, gamma_obs_zero_u)
        else:
            h_t, o_t = h, obs

        fbar = h_t.mean("M", skipna=True)
        paired = np.isfinite(fbar) & np.isfinite(o_t)
        fbar = fbar.where(paired)
        o_t = o_t.where(paired)
        h_t = h_t.where(paired)
        mu_f = self._seasonal_mean(fbar, months)
        mu_o = self._seasonal_mean(o_t, months)
        sf = self._seasonal_member_spread(h_t, months)
        f_anom = fbar - self._select_season(mu_f, months, fbar["T"])
        o_anom = o_t - self._select_season(mu_o, months, o_t["T"])

        ny, nx = obs.sizes["Y"], obs.sizes["X"]
        mode = self.temporal_smoothing

        if mode == "C":
            sf_t = self._select_season(sf, months, obs["T"])
            a0, b0 = self._fit_ab_constant_np(
                np.asarray(f_anom.values).reshape(f_anom.sizes["T"], -1),
                np.asarray(o_anom.values).reshape(o_anom.sizes["T"], -1),
                np.asarray(sf_t.values).reshape(sf_t.sizes["T"], -1),
            )
            a = xr.DataArray(a0.reshape(ny, nx), coords={"Y": obs.Y, "X": obs.X}, dims=("Y", "X"))
            b = xr.DataArray(b0.reshape(ny, nx), coords={"Y": obs.Y, "X": obs.X}, dims=("Y", "X"))
            a = a.expand_dims(season=self.seasons).transpose("season", "Y", "X")
            b = b.expand_dims(season=self.seasons).transpose("season", "Y", "X")
            a, b = self._smooth_spatial(a), self._smooth_spatial(b)
        else:
            alist, blist = [], []
            F = np.asarray(f_anom.values)
            O = np.asarray(o_anom.values)
            SF = np.asarray(sf.values)
            for si, s in enumerate(self.seasons):
                idx = np.where(months == s)[0]
                if idx.size == 0:
                    aa = np.full((ny, nx), np.nan)
                    bb = np.full((ny, nx), np.nan)
                else:
                    aa, bb = self._fit_ab_one_season_np(
                        F[idx].reshape(idx.size, -1),
                        O[idx].reshape(idx.size, -1),
                        SF[si].reshape(-1),
                    )
                    aa, bb = aa.reshape(ny, nx), bb.reshape(ny, nx)
                alist.append(xr.DataArray(aa, coords={"Y": obs.Y, "X": obs.X}, dims=("Y", "X")))
                blist.append(xr.DataArray(bb, coords={"Y": obs.Y, "X": obs.X}, dims=("Y", "X")))
            a = xr.concat(alist, dim=xr.IndexVariable("season", self.seasons)).transpose("season", "Y", "X")
            b = xr.concat(blist, dim=xr.IndexVariable("season", self.seasons)).transpose("season", "Y", "X")
            a, b = self._smooth_coefficients(a), self._smooth_coefficients(b)

        a.name = "a_probability"
        b.name = "b_probability"
        # expand_dims in the constant C mode may expose a read-only NumPy
        # broadcast view. The marginal adjustments below must own their data.
        a, b = a.copy(deep=True), b.copy(deep=True)
        # Smoothing neighbouring coefficients must not create uncertainty in
        # an observed marginal that is exactly constant in this training set.
        for season in self.seasons:
            sample = o_t.isel(T=np.flatnonzero(months == season))
            constant = (sample.count("T") >= self.min_samples) & (sample.max("T") == sample.min("T"))
            a.loc[dict(season=season)] = xr.where(constant, 0., a.sel(season=season))
            b.loc[dict(season=season)] = xr.where(constant, 0., b.sel(season=season))
        sf.name = "raw_spread"
        return (
            mu_o, mu_f, sf, a, b,
            gamma_obs_shape, gamma_obs_scale, gamma_fc_shape, gamma_fc_scale,
            gamma_obs_p0, gamma_obs_zero_u, gamma_fc_p0, gamma_fc_zero_u,
        )

    def _apply_probability(self, fit: _KharinFit, ens: xr.DataArray, months):
        distribution = self._resolved_distribution()
        if distribution == "gamma":
            h_t = self._gamma_to_normal(ens, months, fit.gamma_fc_shape, fit.gamma_fc_scale,
                                        fit.gamma_fc_p0, fit.gamma_fc_zero_u)
            threshold1, threshold2 = self._normal_tercile_levels(fit, ens, months)
            # In gamma-normal space the fitted observed gamma CDF maps its
            # parametric terciles exactly to standard-normal terciles.
        else:
            h_t = ens
            muo = self._select_season(fit.prob_obs_clim, months, ens["T"])
            # Observed normal-climatology spread for tercile boundaries.  We
            # infer it from the calibrated/reference objects below by using the
            # training transformed observations stored in prob_obs_clim plus
            # the variance relation. The fit object stores no explicit obs sd,
            # so reconstruct from a,b,sf and signal variance is not unique.
            # Instead the caller supplies normal thresholds from training obs
            # through temporary attrs set in _fit().
            obs_sd_s = fit.prob_obs_clim.attrs.get("_obs_sd_da")
            if obs_sd_s is None:
                raise RuntimeError("Internal normal-climatology spread is unavailable.")
            sdo = self._select_season(obs_sd_s, months, ens["T"])
            threshold1 = muo + sdo * norm.ppf(1.0 / 3.0)
            threshold2 = muo + sdo * norm.ppf(2.0 / 3.0)

        if fit.reference_terciles is not None:
            threshold1, threshold2 = self._fixed_event_levels(fit, ens, months)

        fbar = h_t.mean("M", skipna=True)
        mu_f = self._select_season(fit.prob_fc_clim, months, ens["T"])
        mu_o = self._select_season(fit.prob_obs_clim, months, ens["T"])
        sf = self._select_season(fit.prob_sf, months, ens["T"])
        a = self._select_season(fit.prob_a, months, ens["T"])
        b = self._select_season(fit.prob_b, months, ens["T"])

        mu = mu_o + a * (fbar - mu_f)
        sigma = (b * sf).clip(min=0.)
        pb = _normal_event_cdf(threshold1, mu, sigma)
        c2 = _normal_event_cdf(threshold2, mu, sigma)
        pn = (c2 - pb).clip(min=0.0, max=1.0)
        pa = (1.0 - c2).clip(min=0.0, max=1.0)
        p = xr.concat([pb, pn, pa], dim=xr.IndexVariable("probability", list(self.probability_labels)))
        s = p.sum("probability", skipna=True)
        p = (p / s.where(s > self.eps)).transpose("probability", "T", "Y", "X")
        p = self._attach_tercile_bounds(p, fit, months, threshold1, threshold2)
        return p, mu.transpose("T", "Y", "X"), sigma.transpose("T", "Y", "X")

    def _climatology_reference(self, obs, clim_start, clim_end):
        """Fixed observed event bounds/frequencies, independently by target season.

        Explicit ``season_year``/``target_year`` coordinates take precedence
        over the timestamp year, as in the grouped validation itself.
        """
        if (clim_start is None) != (clim_end is None):
            raise ValueError("Provide both climatology bounds or neither.")
        if clim_start is not None and int(clim_start) > int(clim_end):
            raise ValueError("Reversed climatology bounds.")
        months, years = self._months_years(obs)
        selected = np.ones(years.size, dtype=bool)
        if clim_start is not None:
            selected = (years >= int(clim_start)) & (years <= int(clim_end))
        thresholds, references = [], []
        for season in self.seasons:
            sample = obs.isel(T=np.flatnonzero(selected & (months == season)))
            try:
                q, reference = climatological_terciles(sample)
            except ValueError as exc:
                raise ValueError(f"Target season {season}: {exc}") from exc
            thresholds.append(q)
            references.append(reference)
        q = xr.concat(thresholds, dim=xr.IndexVariable("season", self.seasons))
        reference = xr.concat(references, dim=xr.IndexVariable("season", self.seasons))
        q.attrs.update(
            climatology_start=int(years[selected].min()) if clim_start is None else int(clim_start),
            climatology_end=int(years[selected].max()) if clim_end is None else int(clim_end),
            climatology_first_year_used=int(years[selected].min()),
            climatology_last_year_used=int(years[selected].max()),
            climatology_year_coordinate="target_year",
            reference_grouping="target_season",
        )
        reference.attrs.update(q.attrs)
        return q, reference

    def _reference_observations(self, training_obs, supplied=None):
        """Validate a separate fixed-event reference without fitting on it.

        The external archive may span more years than the calibration sample,
        but must describe the same physical grid and configured target seasons.
        It is used only to define observed event bounds and frequencies.
        """
        if supplied is None:
            return training_obs
        reference = self._prepare_obs(supplied)
        for dim in ("Y", "X"):
            if not reference.get_index(dim).equals(training_obs.get_index(dim)):
                raise ValueError("obs_for_terciles must use exactly the calibration observation grid.")
        if reference.get_index("T").has_duplicates or reference.get_index("T").hasnans:
            raise ValueError("obs_for_terciles requires unique finite T coordinates.")
        months, years = self._months_years(reference)
        if pd.MultiIndex.from_arrays([years, months]).has_duplicates:
            raise ValueError("obs_for_terciles requires one observation per target season/year.")
        self._check_season_coverage(months)
        return reference

    def _fixed_event_levels(self, fit, ens, months):
        """Map fixed physical boundaries into the fitted predictive space.

        Gamma CDF values use P(X <= bound), including the full atom at zero;
        the mid-probability transform used for training samples is unsuitable
        for event boundaries. Endpoints 0/1 map to infinities exactly.
        """
        bounds = self._select_season(fit.reference_terciles, months, ens["T"])
        values = [bounds.isel(quantile=i, drop=True) for i in range(2)]
        if self._resolved_distribution() != "gamma":
            return values
        shape = self._select_season(fit.gamma_obs_shape, months, ens["T"])
        scale = self._select_season(fit.gamma_obs_scale, months, ens["T"])
        p0 = (None if fit.gamma_obs_p0 is None else
              self._select_season(fit.gamma_obs_p0, months, ens["T"]))
        return [xr.apply_ufunc(ndtri, self._gamma_cdf(value, shape, scale, p0)
                              .where(np.isfinite(value)).clip(min=0., max=1.),
                              dask="parallelized", output_dtypes=[float])
                for value in values]

    def _normal_tercile_levels(self, fit, ens, months):
        """Transformed tercile boundaries of the zero-inflated observed law.

        If P(X=0) >= k/3 the k-th tercile is the zero value; its event level is
        then p0 rather than k/3.
        """
        template = xr.zeros_like(ens.isel(M=0, drop=True))
        p0 = (0.0 if fit.gamma_obs_p0 is None
              else self._select_season(fit.gamma_obs_p0, months, ens["T"]))
        levels = []
        for k in (1.0 / 3.0, 2.0 / 3.0):
            u = xr.where(p0 >= k, p0, k) + template
            levels.append(xr.apply_ufunc(ndtri, u.clip(min=0., max=1.),
                                         dask="parallelized", output_dtypes=[float]))
        return levels[0], levels[1]

    def _attach_tercile_bounds(self, p, fit, months, threshold1, threshold2):
        """Keep physical event definitions and their reference frequencies."""
        if fit.reference_terciles is not None:
            bounds = self._select_season(fit.reference_terciles, months, p["T"])
            reference = self._select_season(fit.reference_probabilities, months, p["T"])
            p = p.assign_coords(
                tercile_lower=bounds.isel(quantile=0, drop=True),
                tercile_upper=bounds.isel(quantile=1, drop=True),
                reference_probability=reference,
            )
            p.attrs.update(fit.reference_terciles.attrs)
            p.attrs["tercile_definition"] = "empirical_observed_fixed_climatology_per_season"
            return p
        if self._resolved_distribution() == "gamma":
            shape = self._select_season(fit.gamma_obs_shape, months, p["T"])
            scale = self._select_season(fit.gamma_obs_scale, months, p["T"])
            p0 = (xr.zeros_like(shape) if fit.gamma_obs_p0 is None
                  else self._select_season(fit.gamma_obs_p0, months, p["T"]))
            bounds = []
            for k in (1.0 / 3.0, 2.0 / 3.0):
                level = ((k - p0) / (1.0 - p0)).clip(min=0.0, max=1.0)
                value = xr.apply_ufunc(lambda a, sc, q: gdtrix(1.0 / sc, a, q),
                                       shape.where(shape > 0), scale.where(shape > 0), level,
                                       dask="parallelized", output_dtypes=[float])
                value = xr.where(shape == 0, scale, value)
                bounds.append(xr.where(p0 >= k, 0.0, value).where(np.isfinite(shape)))
            threshold1, threshold2 = bounds
        p = p.assign_coords(tercile_lower=threshold1, tercile_upper=threshold2)
        p.attrs["tercile_definition"] = "parametric_observed_training_climatology"
        p.attrs["tercile_reference"] = "training"
        p.attrs["category_definition"] = "PB: y <= low; PN: low < y <= high; PA: y > high"
        return p

    def _apply_raw_probability(self, fit: _KharinFit, ens: xr.DataArray, months):
        """Return the paper-style uncalibrated parametric probability forecast.

        The model anomaly multiplier and spread multiplier are both fixed at
        one, while forecast bias is removed by expressing the ensemble-mean
        anomaly relative to the observed climatology.  The climatological
        spread is the training-period ensemble spread for the target season.
        """
        distribution = self._resolved_distribution()
        if distribution == "gamma":
            h_t = self._gamma_to_normal(ens, months, fit.gamma_fc_shape, fit.gamma_fc_scale,
                                        fit.gamma_fc_p0, fit.gamma_fc_zero_u)
            threshold1, threshold2 = self._normal_tercile_levels(fit, ens, months)
        else:
            h_t = ens
            muo = self._select_season(fit.prob_obs_clim, months, ens["T"])
            obs_sd_s = fit.prob_obs_clim.attrs.get("_obs_sd_da")
            if obs_sd_s is None:
                raise RuntimeError("Internal normal-climatology spread is unavailable.")
            sdo = self._select_season(obs_sd_s, months, ens["T"])
            threshold1 = muo + sdo * norm.ppf(1.0 / 3.0)
            threshold2 = muo + sdo * norm.ppf(2.0 / 3.0)

        if fit.reference_terciles is not None:
            threshold1, threshold2 = self._fixed_event_levels(fit, ens, months)

        fbar = h_t.mean("M", skipna=True)
        mu_f = self._select_season(fit.prob_fc_clim, months, ens["T"])
        mu_o = self._select_season(fit.prob_obs_clim, months, ens["T"])
        sf = self._select_season(fit.prob_sf, months, ens["T"]).clip(min=self.eps)
        mu = mu_o + (fbar - mu_f)
        sigma = sf
        z1 = (threshold1 - mu) / sigma
        z2 = (threshold2 - mu) / sigma
        pb = xr.apply_ufunc(norm.cdf, z1, dask="parallelized", output_dtypes=[float])
        c2 = xr.apply_ufunc(norm.cdf, z2, dask="parallelized", output_dtypes=[float])
        pn = (c2 - pb).clip(min=0.0, max=1.0)
        pa = (1.0 - c2).clip(min=0.0, max=1.0)
        p = xr.concat(
            [pb, pn, pa],
            dim=xr.IndexVariable("probability", list(self.probability_labels)),
        )
        total = p.sum("probability", skipna=True)
        p = (p / total.where(total > self.eps)).transpose("probability", "T", "Y", "X")
        p = self._attach_tercile_bounds(p, fit, months, threshold1, threshold2)
        return p, mu.transpose("T", "Y", "X"), sigma.transpose("T", "Y", "X")

    # ------------------------------------------------------------------
    # Fitting and application
    # ------------------------------------------------------------------
    def _fit(self, hindcast, obs, months):
        # Seven overlapping seasons are not seven independent training years.
        # With one row per year/season, a per-season count enforces the stated
        # minimum history for climatologies even in pooled C7 calibration.
        paired = obs.notnull() & hindcast.notnull().any("M")
        counts = self._seasonal_mean(paired.astype(float), months) * 0
        for s in self.seasons:
            counts.loc[dict(season=s)] = paired.isel(T=np.flatnonzero(months == s)).sum("T")
        det_obs_clim, det_fc_clim, det_a = self._fit_deterministic(hindcast, obs, months)
        (
            p_obs_clim, p_fc_clim, p_sf, p_a, p_b,
            gos, gosc, gfs, gfsc, gop0, gozu, gfp0, gfzu,
        ) = self._fit_probability(hindcast, obs, months)
        sufficient = counts >= self.min_samples
        det_a = det_a.where(sufficient)
        p_a = p_a.where(sufficient)
        p_b = p_b.where(sufficient)

        # For normal variables, retain the observed climatological standard
        # deviation for parametric tercile thresholds.  xarray attrs cannot
        # serialize a DataArray, but this is an in-memory fit container only.
        if self._resolved_distribution() == "normal":
            osd = []
            for s in self.seasons:
                x = obs.isel(T=np.where(months == s)[0])
                osd.append(x.std("T", skipna=True, ddof=1))
            osd = xr.concat(osd, dim=xr.IndexVariable("season", self.seasons)).transpose("season", "Y", "X")
            p_obs_clim.attrs["_obs_sd_da"] = osd

        fit = _KharinFit(
            det_obs_clim=det_obs_clim,
            det_fc_clim=det_fc_clim,
            det_a=det_a,
            prob_obs_clim=p_obs_clim,
            prob_fc_clim=p_fc_clim,
            prob_sf=p_sf,
            prob_a=p_a,
            prob_b=p_b,
            gamma_obs_shape=gos,
            gamma_obs_scale=gosc,
            gamma_fc_shape=gfs,
            gamma_fc_scale=gfsc,
            gamma_obs_p0=gop0,
            gamma_fc_p0=gfp0,
            gamma_obs_zero_u=gozu,
            gamma_fc_zero_u=gfzu,
        )
        return fit

    def fit_deterministic(self, hindcast, obs, months):
        """Deterministic regression calibration only (same masks as ``_fit``)."""
        from types import SimpleNamespace
        paired = obs.notnull() & hindcast.notnull().any("M")
        counts = []
        for s in self.seasons:
            counts.append(paired.isel(T=np.flatnonzero(months == s)).sum("T"))
        counts = xr.concat(counts, dim=xr.IndexVariable("season", self.seasons))
        mu_o, mu_f, a = self._fit_deterministic(hindcast, obs, months)
        return SimpleNamespace(det_obs_clim=mu_o, det_fc_clim=mu_f,
                               det_a=a.where(counts >= self.min_samples))

    def compute_model(self, X_train, y_train, X_test, y_test=None, **kwargs):
        """Fit on one fold and return deterministic calibrated predictions.

        This method is provided for direct use, but the recommended WASS2S path
        is the dedicated ``WAS_Cross_Validator`` adapter supplied with this
        class because probabilistic calibration must also be fitted inside each
        outer fold.
        """
        obs, h = self._align_training(y_train, X_train)
        months, _ = self._months_years(obs)
        fit = self.fit_deterministic(h, obs, months)
        f, fmonths = self._align_forecast(X_test, obs, months)
        return self._apply_deterministic(fit, f, fmonths)

    def cross_validate(
        self,
        Predictant: xr.DataArray,
        Predictor: xr.DataArray,
        clim_year_start=None,
        clim_year_end=None,
        cv_buffer_years: Optional[int] = None,
        tercile_reference: Optional[str] = None,
        probability_calibration: Optional[str] = None,
        obs_for_terciles: Optional[xr.DataArray] = None,
    ):
        """Grouped leave-one-year-out CV with fixed climatological events.

        Model climatologies, regression/gamma parameters, spread and CRPS
        calibration remain fitted only to each fold's training years. All
        seasons of the held-out year are withheld together. Fixed observed
        terciles define the same events for every fold and operational forecast.
        ``obs_for_terciles`` may supply the full observed reference archive when
        its climate period extends beyond the hindcast fitting years. It never
        participates in climatology/CRPS parameter fitting.
        ``outer`` (default) and ``inner``/``nested`` select the common workflow
        policy. Here probabilities are native CRPS distributions, already
        trained within each outer fold; residual-calibration subfolds are not
        applicable and are not manufactured for either policy.
        """
        obs, h = self._align_training(Predictant, Predictor)
        calibration = normalize_probability_calibration(
            self.probability_calibration if probability_calibration is None else probability_calibration)
        months, years = self._months_years(obs)
        buffer = self.cv_buffer_years if cv_buffer_years is None else max(0, int(cv_buffer_years))
        unique_years = np.unique(years)
        physical_centres = None
        if self.purge_overlapping_seasons:
            # T normally names the physical target centre. With explicit
            # target coordinates it may instead name initialization. Never
            # mistake season_year's winter validation group for a calendar year.
            physical_year = None
            if "target_year" in obs.coords:
                value = np.asarray(obs["target_year"].values)
                if (value.shape != years.shape or not np.isfinite(value).all()
                        or np.any(value.astype(int) != value)):
                    raise ValueError("Overlap purging requires finite integer target_year values, one per T.")
                physical_year = value.astype(int)
            try:
                timestamp_month = np.asarray(obs["T"].dt.month.values, dtype=int)
                timestamp_year = np.asarray(obs["T"].dt.year.values, dtype=int)
            except (AttributeError, TypeError, ValueError):
                if physical_year is None:
                    raise ValueError("Overlap purging requires actual target-centre datetime T or explicit target_year; otherwise set purge_overlapping_seasons=False.")
            else:
                if physical_year is None:
                    shifted = timestamp_month - 1 + self.fixed_lead_window[1]
                    shifted_month = shifted % 12 + 1
                    matching = timestamp_month == months
                    if np.any(~matching & (shifted_month != months)):
                        raise ValueError("Cannot establish physical target dates for overlap purging: provide actual target-centre dates or target_year, or set purge_overlapping_seasons=False.")
                    physical_year = np.where(matching, timestamp_year, timestamp_year + shifted // 12)
            physical_centres = physical_year * 12 + months - 1
        purged_count = 0
        mode = self.tercile_reference if tercile_reference is None else str(tercile_reference).lower()
        if mode not in {"climatology", "training"}:
            raise ValueError("tercile_reference must be 'climatology' or 'training'.")
        if obs_for_terciles is not None and mode != "climatology":
            raise ValueError('obs_for_terciles requires tercile_reference="climatology".')
        reference_obs = self._reference_observations(obs, obs_for_terciles)
        fixed_reference = (self._climatology_reference(reference_obs, clim_year_start, clim_year_end)
                           if mode == "climatology" else None)

        def _empty_obs_field():
            return xr.DataArray(
                np.full((obs.sizes["T"], obs.sizes["Y"], obs.sizes["X"]), np.nan, dtype=float),
                coords={k: v for k, v in obs.coords.items() if set(v.dims) <= {"T", "Y", "X"}},
                dims=("T", "Y", "X"),
            )

        det = _empty_obs_field()
        raw_det = _empty_obs_field()
        prob = xr.DataArray(
            np.full((3, obs.sizes["T"], obs.sizes["Y"], obs.sizes["X"]), np.nan),
            coords={**det.coords, "probability": list(self.probability_labels)},
            dims=("probability", "T", "Y", "X"),
        )
        raw_prob = xr.full_like(prob, np.nan, dtype=float)
        cv_mu = _empty_obs_field()
        cv_sigma = _empty_obs_field()
        cv_raw_mu = _empty_obs_field()
        cv_raw_sigma = _empty_obs_field()
        cv_ref_mu = _empty_obs_field()
        cv_ref_sigma = _empty_obs_field()
        cv_obs_t = _empty_obs_field()
        cv_lower = _empty_obs_field()
        cv_upper = _empty_obs_field()

        for y in unique_years:
            test_idx = np.where(years == y)[0]
            excluded = np.arange(y - buffer, y + buffer + 1, dtype=int)
            train_idx = np.where(~np.isin(years, excluded))[0]
            if physical_centres is not None:
                held_months = (physical_centres[test_idx, None] + np.array([-1, 0, 1])).ravel()
                shared = np.isin(physical_centres[train_idx, None] + np.array([-1, 0, 1]), held_months).any(axis=1)
                purged_count += int(shared.sum())
                train_idx = train_idx[~shared]
            if train_idx.size < self.min_samples:
                continue

            otr = obs.isel(T=train_idx)
            htr = h.isel(T=train_idx)
            mtr = months[train_idx]
            fit = self._fit(htr, otr, mtr)
            if fixed_reference is not None:
                fit.reference_terciles, fit.reference_probabilities = fixed_reference

            hte = h.isel(T=test_idx)
            mte = months[test_idx]
            dte = self._apply_deterministic(fit, hte, mte)
            rdte = self._apply_raw_deterministic(fit, hte, mte)
            pte, mute, ste = self._apply_probability(fit, hte, mte)
            rpte, rmute, rste = self._apply_raw_probability(fit, hte, mte)
            cv_lower.loc[dict(T=obs["T"].isel(T=test_idx))] = pte.tercile_lower
            cv_upper.loc[dict(T=obs["T"].isel(T=test_idx))] = pte.tercile_upper

            det.loc[dict(T=obs["T"].isel(T=test_idx))] = dte
            raw_det.loc[dict(T=obs["T"].isel(T=test_idx))] = rdte
            prob.loc[dict(T=obs["T"].isel(T=test_idx))] = pte
            raw_prob.loc[dict(T=obs["T"].isel(T=test_idx))] = rpte
            cv_mu.loc[dict(T=obs["T"].isel(T=test_idx))] = mute
            cv_sigma.loc[dict(T=obs["T"].isel(T=test_idx))] = ste
            cv_raw_mu.loc[dict(T=obs["T"].isel(T=test_idx))] = rmute
            cv_raw_sigma.loc[dict(T=obs["T"].isel(T=test_idx))] = rste

            # Reference and transformed observations used to report CRPSS in
            # exactly the space in which the calibration was optimized.
            if self._resolved_distribution() == "gamma":
                ote = self._gamma_to_normal(
                    obs.isel(T=test_idx), mte,
                    fit.gamma_obs_shape, fit.gamma_obs_scale,
                    fit.gamma_obs_p0, fit.gamma_obs_zero_u,
                )
                rmu = xr.zeros_like(ote)
                rsd = xr.ones_like(ote)
                shape = self._select_season(fit.gamma_obs_shape, mte, ote["T"])
                pzero = self._select_season(fit.gamma_obs_p0, mte, ote["T"])
                # A wholly dry or constant-positive observed marginal is a
                # point mass, including in transformed space. Its reference
                # CRPS is zero, so skill relative to it is undefined rather
                # than an artificial perfect score against N(0,1).
                single_atom = (shape == 0) & ((pzero == 0) | (pzero == 1))
                rsd = xr.where(single_atom, 0., rsd)
            else:
                ote = obs.isel(T=test_idx)
                rmu = self._select_season(fit.prob_obs_clim, mte, ote["T"])
                osd = fit.prob_obs_clim.attrs["_obs_sd_da"]
                rsd = self._select_season(osd, mte, ote["T"])
            cv_obs_t.loc[dict(T=obs["T"].isel(T=test_idx))] = ote
            cv_ref_mu.loc[dict(T=obs["T"].isel(T=test_idx))] = rmu
            cv_ref_sigma.loc[dict(T=obs["T"].isel(T=test_idx))] = rsd

        mask = xr.where(obs.notnull().any("T"), 1.0, np.nan)
        det = det.transpose("T", "Y", "X") * mask
        raw_det = raw_det.transpose("T", "Y", "X") * mask
        if self.positive_support:
            det = det.clip(min=0.0)
            raw_det = raw_det.clip(min=0.0)
        prob = prob.transpose("probability", "T", "Y", "X") * mask
        raw_prob = raw_prob.transpose("probability", "T", "Y", "X") * mask
        prob = prob.assign_coords(tercile_lower=cv_lower, tercile_upper=cv_upper)
        raw_prob = raw_prob.assign_coords(tercile_lower=cv_lower, tercile_upper=cv_upper)
        for p in (prob, raw_prob):
            p.attrs["tercile_reference"] = mode
            p.attrs["tercile_definition"] = "parametric_observed_training_climatology"
            p.attrs["cross_validation"] = "target_year_grouped"
            p.attrs["cv_overlap_purge_policy"] = ("exclude shared physical target months" if self.purge_overlapping_seasons
                                                   else "calendar groups only; explicit legacy comparison")
            p.attrs["cv_overlap_purged_training_samples"] = int(purged_count)
            p.attrs.update(self._calibration_metadata(calibration))
            if fixed_reference is not None:
                p.attrs.update(fixed_reference[0].attrs)
                p.attrs["tercile_definition"] = "empirical_observed_fixed_climatology_per_season"
        if fixed_reference is not None:
            reference = self._select_season(fixed_reference[1], months, obs["T"])
            prob = prob.assign_coords(reference_probability=reference)
            raw_prob = raw_prob.assign_coords(reference_probability=reference)

        self.cv_distribution_ = xr.Dataset({"mu": cv_mu * mask, "sigma": cv_sigma * mask})
        self.cv_raw_distribution_ = xr.Dataset({"mu": cv_raw_mu * mask, "sigma": cv_raw_sigma * mask})
        self.cv_reference_ = xr.Dataset({"mu": cv_ref_mu * mask, "sigma": cv_ref_sigma * mask})
        self.cv_transformed_observation_ = cv_obs_t * mask
        self.cv_raw_deterministic_ = raw_det
        self.cv_raw_probability_ = raw_prob
        return det, prob

    def forecast(
        self,
        Predictant: xr.DataArray,
        clim_year_start,
        clim_year_end,
        hindcast: xr.DataArray,
        forecast: xr.DataArray,
        **kwargs,
    ):
        """Fit on the complete hindcast period and calibrate a new MME forecast."""
        calibration = normalize_probability_calibration(kwargs.get("probability_calibration", self.probability_calibration))
        obs, h = self._align_training(Predictant, hindcast)
        months, _ = self._months_years(obs)
        fit = self._fit(h, obs, months)
        mode = str(kwargs.get("tercile_reference", self.tercile_reference)).lower()
        if mode not in {"climatology", "training"}:
            raise ValueError("tercile_reference must be 'climatology' or 'training'.")
        supplied_reference = kwargs.get("obs_for_terciles")
        if supplied_reference is not None and mode != "climatology":
            raise ValueError('obs_for_terciles requires tercile_reference="climatology".')
        if mode == "climatology":
            fit.reference_terciles, fit.reference_probabilities = self._climatology_reference(
                self._reference_observations(obs, supplied_reference), clim_year_start, clim_year_end)
        f, fmonths = self._align_forecast(forecast, obs, months)
        det = self._apply_deterministic(fit, f, fmonths)
        prob, _, _ = self._apply_probability(fit, f, fmonths)
        prob.attrs.update(self._calibration_metadata(calibration))
        self.last_fit_ = fit
        mask = xr.where(obs.notnull().any("T"), 1.0, np.nan)
        return det * mask, prob * mask

    @staticmethod
    def _calibration_metadata(requested):
        return dict(probability_calibration_requested=requested,
                    probability_calibration="native",
                    probability_calibration_effective="native_training_fit",
                    residual_calibration="not_applicable_native_distribution")

    # ------------------------------------------------------------------
    # Evaluation diagnostics
    # ------------------------------------------------------------------
    @staticmethod
    def _season_values(da: xr.DataArray) -> np.ndarray:
        if "season_month" in da.coords:
            return np.asarray(da["season_month"].values, dtype=int)
        if "target_month" in da.coords:
            return np.asarray(da["target_month"].values, dtype=int)
        try:
            return np.asarray(da["T"].dt.month.values, dtype=int)
        except Exception:
            # A non-datetime T axis can still be evaluated as one season.
            return np.ones(da.sizes["T"], dtype=int)

    @classmethod
    def _msss_by_season(cls, obs, fc):
        """Eq. (3), using the same finite pairs for both variances."""
        months = cls._season_values(obs)
        scores = []
        for s in np.unique(months):
            idx = np.where(months == s)[0]
            o, f = xr.align(obs.isel(T=idx), fc.isel(T=idx), join="exact")
            paired = np.isfinite(o) & np.isfinite(f)
            enough = paired.sum("T") >= 2
            o, f = o.where(paired), f.where(paired)
            # Article definition: Var(F-O), not MSE(F-O).
            vare = (f - o).var("T", skipna=True, ddof=0).where(enough)
            varo = o.var("T", skipna=True, ddof=0).where(enough)
            scores.append((1.0 - vare / varo.where(varo > 0)).expand_dims(season=[int(s)]))
        result = xr.concat(scores, dim="season").mean("season", skipna=True)
        result.attrs.update(
            score_definition="1 - Var(forecast - observation) / Var(observation); not MSE",
            valid_cases="same finite observation/forecast pairs for both variances; minimum two per cell and season")
        return result

    @classmethod
    def _domain_msss_by_season(cls, obs, fc):
        """Regional MSSS as in the paper: spatially average variances first."""
        months = cls._season_values(obs)
        values = []
        for s in np.unique(months):
            idx = np.where(months == s)[0]
            o, f = xr.align(obs.isel(T=idx), fc.isel(T=idx), join="exact")
            paired = np.isfinite(o) & np.isfinite(f)
            enough = paired.sum("T") >= 2
            o, f = o.where(paired), f.where(paired)
            vare = (f - o).var("T", skipna=True, ddof=0).where(enough)
            varo = o.var("T", skipna=True, ddof=0).where(enough)
            spatial = np.isfinite(vare) & np.isfinite(varo)
            vare, varo = vare.where(spatial), varo.where(spatial)
            num = vare.mean(("Y", "X"), skipna=True)
            den = varo.mean(("Y", "X"), skipna=True)
            if float(den) > 0:
                values.append(1.0 - float(num / den))
        return float(np.nanmean(values)) if values else np.nan

    @staticmethod
    def _pearson(obs, fc):
        valid = np.isfinite(obs) & np.isfinite(fc)
        obs, fc = obs.where(valid), fc.where(valid)
        oa = obs - obs.mean("T", skipna=True)
        fa = fc - fc.mean("T", skipna=True)
        cov = (oa * fa).mean("T", skipna=True)
        return cov / np.sqrt((oa * oa).mean("T", skipna=True) * (fa * fa).mean("T", skipna=True))

    @classmethod
    def _tercile_thresholds_by_time(cls, obs, clim_start=None, clim_end=None):
        """Observed terciles for each target month, broadcast back to T."""
        clim = obs
        if clim_start is not None and clim_end is not None:
            q = obs.sel(T=slice(str(clim_start), str(clim_end)))
            if q.sizes.get("T", 0) >= 3:
                clim = q
        months_all = cls._season_values(obs)
        months_clim = cls._season_values(clim)
        t1 = xr.DataArray(
            np.full((obs.sizes["T"], obs.sizes["Y"], obs.sizes["X"]), np.nan, dtype=float),
            coords={"T": obs["T"], "Y": obs["Y"], "X": obs["X"]},
            dims=("T", "Y", "X"),
        )
        t2 = t1.copy(deep=True)
        for s in np.unique(months_all):
            ci = np.where(months_clim == s)[0]
            oi = np.where(months_all == s)[0]
            if ci.size < 3 or oi.size == 0:
                continue
            qq = clim.isel(T=ci).quantile([1.0 / 3.0, 2.0 / 3.0], dim="T", skipna=True)
            q1 = qq.isel(quantile=0, drop=True)
            q2 = qq.isel(quantile=1, drop=True)
            t1.loc[dict(T=obs["T"].isel(T=oi))] = q1
            t2.loc[dict(T=obs["T"].isel(T=oi))] = q2
        return t1, t2

    @classmethod
    def _rps_components(cls, obs, prob, clim_start=None, clim_end=None):
        t1, t2 = cls._probability_event_bounds(obs, prob, clim_start, clim_end)
        cat = xr.where(obs <= t1, 0, xr.where(obs <= t2, 1, 2)).where(np.isfinite(obs) & np.isfinite(t1) & np.isfinite(t2))
        ob1 = xr.where(cat == 0, 1.0, 0.0).where(np.isfinite(cat))
        ob2 = xr.where(cat <= 1, 1.0, 0.0).where(np.isfinite(cat))
        fc1 = prob.sel(probability="PB")
        fc2 = prob.sel(probability="PB") + prob.sel(probability="PN")
        rps = (fc1 - ob1) ** 2 + (fc2 - ob2) ** 2
        if "reference_probability" in prob.coords:
            reference = prob.reference_probability
            ref1 = reference.sel(probability="PB")
            ref2 = ref1 + reference.sel(probability="PN")
        else:
            ref1, ref2 = 1.0 / 3.0, 2.0 / 3.0
        rps_ref = (ref1 - ob1) ** 2 + (ref2 - ob2) ** 2
        # Score/reference must use exactly the same valid cases.
        valid = np.isfinite(rps) & np.isfinite(rps_ref)
        return rps.where(valid), rps_ref.where(valid)

    @classmethod
    def _probability_event_bounds(cls, obs, probability, clim_start=None, clim_end=None):
        if "tercile_lower" in probability.coords and "tercile_upper" in probability.coords:
            return probability.tercile_lower, probability.tercile_upper
        raise ValueError(
            "Kharin probability evaluation requires tercile_lower and tercile_upper "
            "coordinates from forecast/cross_validate. Reconstructing thresholds "
            "from verification observations would change the forecast events."
        )

    @classmethod
    def _rpss(cls, obs, prob, clim_start=None, clim_end=None):
        rps, ref = cls._rps_components(obs, prob, clim_start, clim_end)
        return 1.0 - rps.mean("T", skipna=True) / ref.mean("T", skipna=True)

    @classmethod
    def _domain_rpss(cls, obs, prob, clim_start=None, clim_end=None):
        rps, ref = cls._rps_components(obs, prob, clim_start, clim_end)
        num = rps.mean(("T", "Y", "X"), skipna=True)
        den = ref.mean(("T", "Y", "X"), skipna=True)
        return float(1.0 - num / den) if float(den) > 0 else np.nan

    @classmethod
    def _seasonal_crpss(cls, crps, ref):
        valid = np.isfinite(crps) & np.isfinite(ref)
        crps, ref = crps.where(valid), ref.where(valid)
        months = cls._season_values(crps)
        fields = []
        domain = []
        for s in np.unique(months):
            idx = np.where(months == s)[0]
            a = crps.isel(T=idx).mean("T", skipna=True)
            r = ref.isel(T=idx).mean("T", skipna=True)
            fields.append((1.0 - a / r.where(r > 0)).expand_dims(season=[int(s)]))
            ar = a.mean(("Y", "X"), skipna=True)
            rr = r.mean(("Y", "X"), skipna=True)
            if float(rr) > 0:
                domain.append(1.0 - float(ar / rr))
        field = xr.concat(fields, dim="season").mean("season", skipna=True)
        return field, (float(np.nanmean(domain)) if domain else np.nan)

    def evaluate(
        self,
        Predictant: xr.DataArray,
        raw_hindcast: xr.DataArray,
        calibrated_hindcast: xr.DataArray,
        calibrated_prob: Optional[xr.DataArray] = None,
        clim_year_start=None,
        clim_year_end=None,
    ) -> Dict[str, object]:
        """Evaluate raw versus calibrated hindcasts with paper-consistent scores.

        The deterministic reference is the *bias-corrected raw anomaly forecast*
        (a=1), not the model field in its native climatology.  After
        :meth:`cross_validate`, that reference is available fold-by-fold through
        ``cv_raw_deterministic_``.  MSSS follows Eq. (3), using error variance,
        and regional MSSS spatially averages numerator/denominator before the
        ratio as described in section 4a of Kharin et al. (2017).

        CRPSS is likewise formed season-by-season, with regional CRPS and
        reference CRPS spatially averaged before taking the ratio. For gamma
        precipitation the current implementation reports CRPSS in the
        gamma-to-normal calibration space and labels it explicitly; RPSS is in
        the original tercile categories and is directly comparable across
        distributions.
        """
        obs, h = self._align_training(Predictant, raw_hindcast)
        cal = self._prepare_obs(calibrated_hindcast).reindex(T=obs["T"])

        if self.cv_raw_deterministic_ is not None and self.cv_raw_deterministic_.sizes.get("T") == obs.sizes["T"]:
            raw = self.cv_raw_deterministic_.reindex(T=obs["T"])
        else:
            months, _ = self._months_years(obs)
            fit_full = self._fit(h, obs, months)
            raw = self._apply_raw_deterministic(fit_full, h, months)

        msss_raw = self._msss_by_season(obs, raw)
        msss_cal = self._msss_by_season(obs, cal)
        mae_raw = np.abs(raw - obs).mean("T", skipna=True)
        mae_cal = np.abs(cal - obs).mean("T", skipna=True)
        corr_raw = self._pearson(obs, raw)
        corr_cal = self._pearson(obs, cal)
        domain_msss_raw = self._domain_msss_by_season(obs, raw)
        domain_msss_cal = self._domain_msss_by_season(obs, cal)

        out: Dict[str, object] = {
            "score_metadata": {
                "MSSS_definition": "1 - Var(forecast - observation) / Var(observation); not MSE",
                "MSSS_valid_cases": "paired finite cases, minimum two per cell and season; common spatial support for regional variance means",
            },
            "MSSS_raw": msss_raw,
            "MSSS_calibrated": msss_cal,
            "MSSS_gain": msss_cal - msss_raw,
            "MAE_raw": mae_raw,
            "MAE_calibrated": mae_cal,
            "Pearson_raw": corr_raw,
            "Pearson_calibrated": corr_cal,
            "domain_mean": {
                "MSSS_raw": domain_msss_raw,
                "MSSS_calibrated": domain_msss_cal,
                "MSSS_gain": domain_msss_cal - domain_msss_raw,
                "MAE_raw": float(mae_raw.mean(("Y", "X"), skipna=True)),
                "MAE_calibrated": float(mae_cal.mean(("Y", "X"), skipna=True)),
                "Pearson_raw": float(corr_raw.mean(("Y", "X"), skipna=True)),
                "Pearson_calibrated": float(corr_cal.mean(("Y", "X"), skipna=True)),
            },
        }

        if calibrated_prob is not None:
            rp_cal = self._rpss(obs, calibrated_prob, clim_year_start, clim_year_end)
            out["RPSS_calibrated"] = rp_cal
            out["domain_mean"]["RPSS_calibrated"] = self._domain_rpss(
                obs, calibrated_prob, clim_year_start, clim_year_end
            )
            if self.cv_raw_probability_ is not None:
                rp_raw = self._rpss(obs, self.cv_raw_probability_, clim_year_start, clim_year_end)
                out["RPSS_raw"] = rp_raw
                out["RPSS_gain"] = rp_cal - rp_raw
                dr = self._domain_rpss(obs, self.cv_raw_probability_, clim_year_start, clim_year_end)
                dc = out["domain_mean"]["RPSS_calibrated"]
                out["domain_mean"]["RPSS_raw"] = dr
                out["domain_mean"]["RPSS_gain"] = dc - dr

        if (
            self.cv_distribution_ is not None
            and self.cv_reference_ is not None
            and self.cv_transformed_observation_ is not None
        ):
            mu = self.cv_distribution_["mu"]
            sd = self.cv_distribution_["sigma"]
            y = self.cv_transformed_observation_
            rmu = self.cv_reference_["mu"]
            rsd = self.cv_reference_["sigma"]
            crps = xr.apply_ufunc(
                self._normal_crps_np, mu, sd, y,
                input_core_dims=[[], [], []], output_core_dims=[[]],
                vectorize=True, dask="parallelized", output_dtypes=[float],
                kwargs={"eps": self.eps},
            )
            crps_ref = xr.apply_ufunc(
                self._normal_crps_np, rmu, rsd, y,
                input_core_dims=[[], [], []], output_core_dims=[[]],
                vectorize=True, dask="parallelized", output_dtypes=[float],
                kwargs={"eps": self.eps},
            )
            crpss, dcrpss = self._seasonal_crpss(crps, crps_ref)
            key = "CRPSS_calibrated" if self._resolved_distribution() == "normal" else "CRPSS_transformed_gamma"
            out[key] = crpss
            out["domain_mean"][key] = dcrpss

            if self.cv_raw_distribution_ is not None:
                r0mu = self.cv_raw_distribution_["mu"]
                r0sd = self.cv_raw_distribution_["sigma"]
                crps_raw = xr.apply_ufunc(
                    self._normal_crps_np, r0mu, r0sd, y,
                    input_core_dims=[[], [], []], output_core_dims=[[]],
                    vectorize=True, dask="parallelized", output_dtypes=[float],
                    kwargs={"eps": self.eps},
                )
                crpss_raw, dcrpss_raw = self._seasonal_crpss(crps_raw, crps_ref)
                raw_key = key.replace("calibrated", "raw") if "calibrated" in key else key + "_raw"
                gain_key = key.replace("calibrated", "gain") if "calibrated" in key else key + "_gain"
                out[raw_key] = crpss_raw
                out[gain_key] = crpss - crpss_raw
                out["domain_mean"][raw_key] = dcrpss_raw
                out["domain_mean"][gain_key] = dcrpss - dcrpss_raw

        return out

    def attributes_diagram_data(
        self,
        Predictant: xr.DataArray,
        probability: xr.DataArray,
        clim_year_start=None,
        clim_year_end=None,
        n_bins: int = 10,
    ) -> Dict[str, object]:
        """Return reliability-bin data and Murphy Brier decomposition.

        The aggregation is over all valid T/Y/X samples, matching the paper's
        use of pooled samples to obtain stable attributes diagrams.  Returned
        ``table`` contains one row per probability bin and tercile category;
        ``decomposition`` contains reliability, resolution, uncertainty and
        Brier score for PB, PN and PA. With continuous probabilities grouped
        into bins the classical three terms describe the binned forecasts,
        not necessarily their exact original Brier score. The returned
        ``BS_decomposition_residual`` closes that binning identity explicitly.
        """
        try:
            integer_bins = int(n_bins)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("n_bins must be a positive integer.") from exc
        if isinstance(n_bins, (bool, np.bool_)) or integer_bins != n_bins or integer_bins < 1:
            raise ValueError("n_bins must be a positive integer.")
        obs = self._prepare_obs(Predictant)
        p = probability.transpose("probability", "T", "Y", "X").reindex(T=obs["T"])
        t1, t2 = self._probability_event_bounds(obs, p, clim_year_start, clim_year_end)
        category = xr.where(obs <= t1, 0, xr.where(obs <= t2, 1, 2)).where(np.isfinite(t1) & np.isfinite(t2) & np.isfinite(obs))
        edges = np.linspace(0.0, 1.0, integer_bins + 1)
        rows = []
        decom = {}
        for j, label in enumerate(self.probability_labels):
            pf = np.asarray(p.sel(probability=label).values, dtype=float).ravel()
            ob = np.asarray((category == j).where(np.isfinite(category)).values, dtype=float).ravel()
            valid = np.isfinite(pf) & np.isfinite(ob)
            pf, ob = pf[valid], ob[valid]
            if pf.size == 0:
                continue
            o_bar = float(np.mean(ob))
            bs = float(np.mean((pf - ob) ** 2))
            rel = 0.0
            res = 0.0
            for k in range(len(edges) - 1):
                lo, hi = edges[k], edges[k + 1]
                sel = (pf >= lo) & (pf < hi if k < len(edges) - 2 else pf <= hi)
                nk = int(sel.sum())
                if nk == 0:
                    rows.append({"category": label, "bin": k, "bin_left": lo, "bin_right": hi,
                                 "count": 0, "forecast_probability": np.nan, "observed_frequency": np.nan})
                    continue
                pk = float(np.mean(pf[sel]))
                ok = float(np.mean(ob[sel]))
                wk = nk / pf.size
                rel += wk * (pk - ok) ** 2
                res += wk * (ok - o_bar) ** 2
                rows.append({"category": label, "bin": k, "bin_left": lo, "bin_right": hi,
                             "count": nk, "forecast_probability": pk, "observed_frequency": ok})
            decom[label] = {
                "BS": bs,
                "BS_binned": float(rel - res + o_bar * (1.0 - o_bar)),
                "BS_decomposition_residual": float(bs - (rel - res + o_bar * (1.0 - o_bar))),
                "BS_reliability": float(rel),
                "BS_resolution": float(res),
                "BS_uncertainty": float(o_bar * (1.0 - o_bar)),
                "climatological_frequency": o_bar,
                "n": int(pf.size),
            }
        return {"table": pd.DataFrame(rows), "decomposition": decom,
                "metadata": {
                    "decomposition": "Murphy terms for binned forecast probabilities",
                    "identity": "BS = BS_reliability - BS_resolution + BS_uncertainty + BS_decomposition_residual",
                    "n_bins": integer_bins,
                }}

    @staticmethod
    def _domain_pearson_for_subset(obs, fc):
        oa = obs - obs.mean("T", skipna=True)
        fa = fc - fc.mean("T", skipna=True)
        cov = (oa * fa).mean("T", skipna=True)
        den = np.sqrt((oa * oa).mean("T", skipna=True) * (fa * fa).mean("T", skipna=True))
        r = cov / den.where(den > 0)
        return float(r.mean(("Y", "X"), skipna=True))

    def evaluate_by_season(
        self, Predictant, calibrated_hindcast, calibrated_prob=None,
        clim_year_start=None, clim_year_end=None,
    ) -> pd.DataFrame:
        """Domain scores for every configured target season plus WINDOW_MEAN.

        This is the recommended diagnostic for the 7-season adaptation because
        it exposes whether borrowing adjacent seasons actually improves the
        central target JAS, rather than hiding that effect in a window average.
        """
        obs = self._prepare_obs(Predictant)
        cal = self._prepare_obs(calibrated_hindcast).reindex(T=obs["T"])
        raw = self.cv_raw_deterministic_
        if raw is None or raw.sizes.get("T") != obs.sizes.get("T"):
            raise RuntimeError("evaluate_by_season should be called after cross_validate().")
        raw = raw.reindex(T=obs["T"])
        months = self._season_values(obs)
        rows = []
        reverse = {v: k for k, v in self.rolling_3month_center.items()}

        for s in self.seasons:
            idx = np.where(months == int(s))[0]
            if idx.size == 0:
                continue
            o = obs.isel(T=idx); r = raw.isel(T=idx); c = cal.isel(T=idx)
            vare_r = (r-o).var("T", skipna=True, ddof=0).mean(("Y","X"), skipna=True)
            vare_c = (c-o).var("T", skipna=True, ddof=0).mean(("Y","X"), skipna=True)
            varo = o.var("T", skipna=True, ddof=0).mean(("Y","X"), skipna=True)
            msss_r = 1.0-float(vare_r/varo) if float(varo)>0 else np.nan
            msss_c = 1.0-float(vare_c/varo) if float(varo)>0 else np.nan
            row = {
                "season": reverse[int(s)], "season_month": int(s),
                "MSSS_raw": msss_r, "MSSS_calibrated": msss_c,
                "MSSS_gain": msss_c-msss_r,
                "MAE_raw": float(np.abs(r-o).mean(("T","Y","X"), skipna=True)),
                "MAE_calibrated": float(np.abs(c-o).mean(("T","Y","X"), skipna=True)),
                "Pearson_raw": self._domain_pearson_for_subset(o,r),
                "Pearson_calibrated": self._domain_pearson_for_subset(o,c),
            }
            if calibrated_prob is not None and self.cv_raw_probability_ is not None:
                pcal = calibrated_prob.isel(T=idx)
                praw = self.cv_raw_probability_.isel(T=idx)
                row["RPSS_raw"] = self._domain_rpss(o, praw, clim_year_start, clim_year_end)
                row["RPSS_calibrated"] = self._domain_rpss(o, pcal, clim_year_start, clim_year_end)
                row["RPSS_gain"] = row["RPSS_calibrated"] - row["RPSS_raw"]
            if self.cv_distribution_ is not None and self.cv_reference_ is not None and self.cv_transformed_observation_ is not None:
                mu=self.cv_distribution_["mu"].isel(T=idx); sd=self.cv_distribution_["sigma"].isel(T=idx)
                y=self.cv_transformed_observation_.isel(T=idx)
                rmu=self.cv_reference_["mu"].isel(T=idx); rsd=self.cv_reference_["sigma"].isel(T=idx)
                crps=xr.apply_ufunc(self._normal_crps_np,mu,sd,y,input_core_dims=[[],[],[]],output_core_dims=[[]],vectorize=True,dask="parallelized",output_dtypes=[float],kwargs={"eps":self.eps})
                cref=xr.apply_ufunc(self._normal_crps_np,rmu,rsd,y,input_core_dims=[[],[],[]],output_core_dims=[[]],vectorize=True,dask="parallelized",output_dtypes=[float],kwargs={"eps":self.eps})
                num=crps.mean(("T","Y","X"),skipna=True); den=cref.mean(("T","Y","X"),skipna=True)
                row["CRPSS_calibrated" if self._resolved_distribution()=="normal" else "CRPSS_transformed_gamma"] = 1.0-float(num/den) if float(den)>0 else np.nan
            rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            numeric=[c for c in df.columns if c not in {"season","season_month"}]
            meanrow={"season":"WINDOW_MEAN","season_month":np.nan}
            meanrow.update({c:float(df[c].mean(skipna=True)) for c in numeric})
            df=pd.concat([df,pd.DataFrame([meanrow])],ignore_index=True)
        return df

    def benchmark_temporal_smoothing_by_season(
        self, Predictant, Predictor,
        temporal_grid: Sequence[TemporalMode] = ("U",0.5,1.0,1.5,2.0,"C"),
        clim_year_start=None, clim_year_end=None, include_probability=True,
    ) -> pd.DataFrame:
        """Long-form evaluation table: smoothing configuration × target season."""
        rows=[]
        for mode in temporal_grid:
            m=self.clone(temporal_smoothing=self._normalize_temporal_mode(mode))
            det,prob=m.cross_validate(Predictant,Predictor,clim_year_start,clim_year_end)
            tab=m.evaluate_by_season(Predictant,det,prob if include_probability else None,clim_year_start,clim_year_end)
            tab.insert(0,"temporal_smoothing",m.temporal_label())
            rows.append(tab)
        return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()

    def benchmark_temporal_smoothing(
        self,
        Predictant: xr.DataArray,
        Predictor: xr.DataArray,
        temporal_grid: Sequence[TemporalMode] = ("U", 0.5, 1.0, 1.5, 2.0, "C"),
        clim_year_start=None,
        clim_year_end=None,
        include_probability: bool = True,
    ) -> pd.DataFrame:
        """Evaluate U -> Gaussian smoothing -> pooled C_N for the configured window.

        With the default AMJ..OND window the pooled case is reported as C7.
        Returns one row per temporal smoothing choice. Spatial smoothing is held
        at the current class setting so the temporal effect is isolated.
        """
        rows = []
        for mode in temporal_grid:
            m = self.clone(temporal_smoothing=self._normalize_temporal_mode(mode))
            det, prob = m.cross_validate(Predictant, Predictor, clim_year_start, clim_year_end)
            ev = m.evaluate(
                Predictant, Predictor, det,
                prob if include_probability else None,
                clim_year_start, clim_year_end,
            )
            row = {"temporal_smoothing": m.temporal_label()}
            row.update(ev["domain_mean"])
            rows.append(row)
        return pd.DataFrame(rows)
