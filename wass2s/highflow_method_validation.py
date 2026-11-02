"""Sensitivity diagnostics for high-flow climatological windows.

K=2M+1 is a complexity heuristic, not a constraint on alternating slopes.
The reported Gaussian working BIC and n/window-length sensitivity calculation
are NOT a confidence bracket or a validated effective sample-size correction.
The maximum second-difference elbow is NOT Lavielle's published selection rule.

Cumulative anomalies are adapted here from rainfall season diagnostics
(Liebmann & Marengo, 2001; Dunning, Black & Allan, 2016). They are a complementary
criterion using the SAME observations, not independent hydrological validation.
Global extrema do not necessarily identify an envelope of all seasonal modes.
Tapering changes the correction near the edges and can change seasonal means.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from .highwater_season_predictands_modeaware import (
        HighWaterConfig,
        _segment_sse_cost_matrix,
        fit_piecewise_constant_slopes,
        _choose_n_segments_from_modes,
        _rotpos_to_doy,
        _doy_to_monthday,
        _monthday_to_doy,
        fit_station_highwater_regime,
    )
except ImportError:  # pragma: no cover - flat layout
    from highwater_season_predictands_modeaware import (
        HighWaterConfig,
        _segment_sse_cost_matrix,
        fit_piecewise_constant_slopes,
        _choose_n_segments_from_modes,
        _rotpos_to_doy,
        _doy_to_monthday,
        _monthday_to_doy,
        fit_station_highwater_regime,
    )

__all__ = [
    "circular_day_difference",
    "circular_day_mean",
    "circular_day_mae",
    "segment_number_diagnostic",
    "network_segment_number_summary",
    "liebmann_window_bounds",
    "cumulative_inflection_points",
    "cross_check_windows_liebmann",
    "taper_correction_factor",
    "apply_tapered_factor",
]

DAYS_IN_YEAR = 365


# ---------------------------------------------------------------------------
# Circular date arithmetic
# ---------------------------------------------------------------------------

def circular_day_difference(a, b, period: int = DAYS_IN_YEAR):
    """Signed circular difference a - b, wrapped to [-period/2, +period/2).

    Positive means ``a`` is later than ``b`` by the shorter way round the
    year.  This is the only correct way to compare two dates-of-year, and the
    only one used in this package for timing biases.
    """
    d = (np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) % period
    return np.where(d > period / 2.0, d - period, d)


def circular_day_mean(days: Sequence[float], period: int=DAYS_IN_YEAR) -> float:
    """Mean day in [1,period+1); NaN for an undefined/antipodal resultant."""
    x=np.asarray(days,float);x=x[np.isfinite(x)]
    if not len(x):return np.nan
    z=np.mean(np.exp(2j*np.pi*(x-1)/period))
    if abs(z)<1e-12:return np.nan
    return float(1+(np.angle(z)%(2*np.pi))*period/(2*np.pi))


def circular_day_mae(pred, obs, period: int = DAYS_IN_YEAR) -> float:
    """Mean absolute circular error in days."""
    d = circular_day_difference(pred, obs, period=period)
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    return float(np.mean(np.abs(d))) if d.size else float("nan")


# ---------------------------------------------------------------------------
# 1. Segment-number justification
# ---------------------------------------------------------------------------

def segment_number_diagnostic(
    parde_rotated: np.ndarray,
    config: HighWaterConfig,
    n_modes: int,
    k_max: Optional[int] = None,
) -> dict[str, Any]:
    """Compare heuristic 2M+1 with working penalised contrasts and an elbow.

    The n_eff=n/smoothing_window sensitivity is heuristic. Its extrema do not
    bound the correct complexity. Include K=1/2 so no-season/simple cases can
    be diagnosed. Peak counts are prominence-screened, not significance-tested.
    """
    y = np.asarray(parde_rotated, dtype=float)
    n = len(y)
    L = int(config.min_segment_length_days)
    k_hydro = _choose_n_segments_from_modes(int(n_modes), config)
    k_feasible = max(3, n // max(L, 1))
    k_max = int(k_max) if k_max is not None else int(min(k_feasible, 15))
    k_max = max(k_max, k_hydro)

    cost = _segment_sse_cost_matrix(y)

    # Working sensitivity n/w, not an independence result or a confidence
    # bound: smoothing can leave much longer autocorrelation in discharge.
    w = max(int(config.smoothing_window_days), 1)
    n_eff = max(float(n) / float(w), 10.0)

    rows = []
    for k in range(1, k_max + 1):
        if n < k * L:
            continue
        try:
            fit = fit_piecewise_constant_slopes(y, k, L, cost_matrix=cost)
        except ValueError:
            continue
        sse = float(fit["sse"])
        p = 2 * k - 1
        sse_safe = max(sse, 1e-12)
        rows.append({
            "n_segments": k,
            "sse": sse,
            "bic_n": n * np.log(sse_safe / n) + p * np.log(n),
            "bic_neff": n_eff * np.log(sse_safe / n) + p * np.log(n_eff),
        })

    table = pd.DataFrame(rows)
    if table.empty:
        return {
            "table": table,
            "k_hydrological": k_hydro,
            "k_bic_n": np.nan,
            "k_bic_neff": np.nan,
            "k_elbow_second_difference": np.nan,
            "n_effective": n_eff,
        }

    # Maximum second-difference elbow; not the Lavielle threshold rule.
    sse = table["sse"].to_numpy(dtype=float)
    ks = table["n_segments"].to_numpy(dtype=int)
    rng = sse.max() - sse.min()
    jn = (sse - sse.min()) / rng if rng > 0 else np.zeros_like(sse)
    k_lav = np.nan
    if len(jn) >= 3:
        # Second difference of the normalised curve: the elbow is where the
        # marginal gain collapses.
        d2 = jn[:-2] - 2.0 * jn[1:-1] + jn[2:]
        k_lav = int(ks[1:-1][int(np.argmax(d2))])

    table["is_hydrological_choice"] = table["n_segments"] == k_hydro

    return {
        "table": table,
        "k_hydrological": int(k_hydro),
        "k_bic_n": int(table.loc[table["bic_n"].idxmin(), "n_segments"]),
        "k_bic_neff": int(table.loc[table["bic_neff"].idxmin(), "n_segments"]),
        "k_elbow_second_difference": k_lav,
        "n_effective": float(n_eff),
        "criterion_status": "working Gaussian criteria; effective sample size heuristic",
        "elbow_definition": "maximum second difference of normalized SSE, not Lavielle",
    }


def network_segment_number_summary(
    stations: dict[str, pd.Series],
    config: HighWaterConfig,
    reference_start=None,
    reference_end=None,
) -> pd.DataFrame:
    """Run ``segment_number_diagnostic`` over a whole network.

    Returns one row per station with the hydrological choice, the penalised
    optima and their agreement, so the 2M+1 rule can be defended (or revised)
    on evidence rather than on assertion.
    """
    rows = []
    for name, series in stations.items():
        rec: dict[str, Any] = {"station": name}
        try:
            regime = fit_station_highwater_regime(
                series, config,
                reference_start=reference_start,
                reference_end=reference_end,
            )
            diag = segment_number_diagnostic(
                regime["parde_rotated"], config, regime["n_modes"]
            )
            rec.update({
                "status": "OK",
                "modality": regime["modality"],
                "n_modes": int(regime["n_modes"]),
                "k_hydrological": diag["k_hydrological"],
                "k_bic_n": diag["k_bic_n"],
                "k_bic_neff": diag["k_bic_neff"],
                "k_elbow_second_difference": diag["k_elbow_second_difference"],
                "agrees_bic_neff": diag["k_hydrological"] == diag["k_bic_neff"],
                "agrees_elbow": diag["k_hydrological"] == diag["k_elbow_second_difference"],
            })
        except Exception as exc:  # noqa: BLE001 - diagnostic must not abort
            rec.update({"status": "FAILED", "error": str(exc)})
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Independent cross-check of the bounds
# ---------------------------------------------------------------------------

def liebmann_window_bounds(parde_rotated: np.ndarray, anchor_doy: int,threshold: float=1.) -> dict[str,Any]:
    """Legacy name for a discharge cumulative-anomaly diagnostic.

    Adapted from rainfall work, NOT an exact implementation/validation of the
    rainfall authors' hydrological methodology. Selects a global min-to-max arc;
    this need not equal either the dominant mode or the envelope of all modes.
    """
    p=np.asarray(parde_rotated,float)
    if len(p)!=365 or not np.isfinite(p).all():raise ValueError('Expected finite 365-day curve.')
    g=np.r_[0.,np.cumsum(p-threshold)]
    amp=float(np.ptp(g))
    if amp<1e-10 or np.ptp(p)<1e-10:
        return dict(status='NO_ANOMALY_SEASON',departure_amplitude=amp,start_doy=np.nan,end_doy=np.nan,duration_days=np.nan)
    # Include the origin boundary g(0). The first day after a cumulative
    # minimum is the candidate onset, not a clipped index at the year's end.
    imin=int(np.argmin(g[:-1]));imax=int(np.argmax(g[:-1]))
    start=imin+1;end=imax if imax else 365
    sd=_rotpos_to_doy(start,anchor_doy);ed=_rotpos_to_doy(end,anchor_doy)
    return dict(status='OK',start_pos=start,end_pos=end,start_doy=sd,end_doy=ed,
                start_month_day=_doy_to_monthday(sd),end_month_day=_doy_to_monthday(ed),
                duration_days=(end-start)%365+1,departure_amplitude=amp,
                method='discharge cumulative-anomaly adaptation',interpretation='global min-to-max arc, not necessarily multimodal envelope')


def cumulative_inflection_points(
    parde_rotated: np.ndarray,
    anchor_doy: int,
    smooth_days: int = 0,
) -> pd.DataFrame:
    """Inflection points of the cumulative mass curve, i.e. where C'' = 0.

    Provided to make explicit - and verifiable - that these are the
    hydrograph EXTREMA (peaks and inter-modal troughs), not the season
    bounds.  Each row is labelled ``PEAK`` or ``TROUGH`` according to the
    sign of C'' before the crossing.

    The ``spurious`` column flags sign changes whose local amplitude is a
    negligible fraction of the largest one; on an unsmoothed climatology
    these dominate the dry season and are pure sampling noise, which is the
    practical reason a second-derivative criterion cannot be used for
    operational bound detection.
    """
    p = np.asarray(parde_rotated, dtype=float)
    if smooth_days and smooth_days > 1:
        k = int(smooth_days) | 1
        half = k // 2
        ext = np.concatenate([p[-half:], p, p[:half]])
        p = np.convolve(ext, np.ones(k) / k, mode="valid")

    # C' proportional to P, so C'' proportional to dP/dd.
    c2 = np.gradient(p)
    amp = np.max(np.abs(c2)) if np.max(np.abs(c2)) > 0 else 1.0

    rows = []
    n = len(p)
    for i in range(n - 1):
        if c2[i] == 0 or c2[i] * c2[i + 1] >= 0:
            continue
        pos = i + 2
        doy = _rotpos_to_doy(pos, anchor_doy)
        local = max(abs(c2[i]), abs(c2[i + 1]))
        rows.append({
            "rot_pos": pos,
            "doy": doy,
            "month_day": _doy_to_monthday(doy),
            "parde": float(p[min(i + 1, n - 1)]),
            "kind": "PEAK" if c2[i] > 0 else "TROUGH",
            "relative_amplitude": float(local / amp),
            "spurious": bool(local / amp < 0.05),
        })
    return pd.DataFrame(rows)


def cross_check_windows_liebmann(regimes,threshold=None):
    """Compare only homologous single-season objects; flag multimodal ambiguity.

    It is not valid to assert that global cumulative-anomaly extrema always
    bound every high-flow mode. No automatic envelope comparison is made.
    """
    rows=[]
    for name,reg in regimes.items():
        windows=reg.get('windows',[])
        if not windows:continue
        r=dict(station=name,modality=reg.get('modality'),n_seasons=len(windows))
        if len(windows)!=1 or reg.get('n_modes',0)>1:
            rows.append(dict(r,status='MULTIMODAL_NOT_DIRECTLY_COMPARABLE'));continue
        lb=liebmann_window_bounds(reg['parde_rotated'],reg['anchor_doy'],1. if threshold is None else threshold)
        if lb['status']!='OK':rows.append(dict(r,status=lb['status']));continue
        w=windows[0];ds=float(circular_day_difference(w['start_doy'],lb['start_doy']));de=float(circular_day_difference(w['end_doy'],lb['end_doy']))
        rows.append(dict(r,status='OK',comparison='complementary_not_independent',
              seg_start_month_day=w['start_month_day'],seg_end_month_day=w['end_month_day'],
              anomaly_start_month_day=lb['start_month_day'],anomaly_end_month_day=lb['end_month_day'],
              start_offset_days=ds,end_offset_days=de,duration_offset_days=w['duration_days']-lb['duration_days']))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Edge continuity of corrected daily series
# ---------------------------------------------------------------------------

def taper_correction_factor(times: pd.DatetimeIndex, start_month_day: str,
                            end_month_day: str, factor: float, taper_days: int=10) -> np.ndarray:
    """Oriented circular arc with a geometric ramp on each edge.

    Membership works for ANY duration 1..365, including 300-day windows. Zero
    taper is an exact hard mask. With positive taper the edge weight is 1/2;
    it reaches 1/0 after taper_days inside/outside. This changes edge values
    and does NOT guarantee exact in-window mean-ratio calibration afterward.
    """
    if not np.isfinite(factor) or factor<=0:raise ValueError('factor must be finite and positive.')
    if taper_days<0:raise ValueError('taper_days must be nonnegative.')
    times=pd.DatetimeIndex(times);doy=np.array([_canonical_doy_scalar(t) for t in times],float)
    s=_monthday_to_doy(start_month_day);e=_monthday_to_doy(end_month_day)
    duration=(e-s)%365+1;offset=(doy-s)%365;inside=offset<duration
    if taper_days==0 or duration==365:return np.exp(inside.astype(float)*np.log(factor))
    # Distances to nearest edge along the INSIDE/OUTSIDE arcs separately.
    # Shortest signed circular difference is inappropriate for arcs > half-year.
    depth=np.minimum(offset,duration-1-offset)
    outside_distance=np.minimum(offset-(duration-1),365-offset)
    signed=np.where(inside,depth,-outside_distance)
    weight=np.clip(.5+signed/(2*float(taper_days)),0,1)
    return np.exp(weight*np.log(factor))


def _canonical_doy_scalar(ts: pd.Timestamp) -> int:
    """Day of year ignoring 29 February, so 1 March is always day 60."""
    doy = int(ts.dayofyear)
    if ts.is_leap_year and doy > 59:
        doy -= 1
    return max(1, min(doy, DAYS_IN_YEAR))


def apply_tapered_factor(
    discharge: pd.Series,
    start_month_day: str,
    end_month_day: str,
    factor: float,
    taper_days: int = 10,
) -> pd.Series:
    """Apply a multiplicative correction with smooth edges."""
    w = taper_correction_factor(
        discharge.index, start_month_day, end_month_day, factor, taper_days
    )
    return discharge * w
