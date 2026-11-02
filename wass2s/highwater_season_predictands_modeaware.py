#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Objective identification of climatological high-water seasons and construction
of annual seasonal-discharge predictands.

Designed for West and Central African river regimes, including:
    * unimodal, bimodal and multimodal hydrographs;
    * one or several distinct high-water seasons;
    * seasons crossing 31 December;
    * missing observations and incomplete station records;
    * bootstrap uncertainty on season boundaries;
    * seasonal mean high-water discharge and seasonal maximum discharge;
    * optional mode-specific targets (MODE1, MODE2, ...), even when several
      peaks belong to one continuous high-water season;
    * a CPT/WASS2S-like input CSV format with LON/LAT/ELEV metadata rows.

Scientific idea
---------------
For a 365-day daily climatology Qbar(d), define the normalized cumulative
runoff mass curve

    C(d) = sum_{i<=d} Qbar(i) / sum_{i=1}^{365} Qbar(i).

Its daily increment is

    dC(d) = Qbar(d) / sum(Qbar),

and the daily Pardé coefficient is

    P(d) = Qbar(d) / Qannual = 365 * dC(d).

Therefore, changes in the slope of C(d) are changes in the seasonal flow
regime.  Instead of fitting independent straight lines to C(d), this module
segments P(d) (the slope domain) into piecewise-constant regimes and then
integrates those fitted slopes to reconstruct C(d).  The reconstructed
piecewise cumulative curve is continuous by construction.

Important distinction
---------------------
"Modes" are distinct climatological peaks in the hydrograph.
"High-water seasons" are contiguous periods whose fitted regime has a mean
Pardé coefficient >= a configurable threshold (1.0 by default).

Thus a hydrograph can be bimodal but have only one continuous high-water
season when the inter-peak valley remains above the annual mean discharge.

Optional mode-specific behaviour
--------------------------------
When ``mode_specific_outputs=True``, each prominence-selected climatological peak is
also assigned a chronological mode window:

    MODE1 = first prominence-selected peak after the hydrological-year origin
    MODE2 = second prominence-selected peak
    ...

If two or more modes occur inside the SAME continuous high-water season, that
season is split at the climatological local minimum between adjacent peaks.
This allows, for example, a bimodal-but-continuously-high river to retain one
high-water season (HW1) while also producing MODE1 and MODE2 forecast targets.

The chronological MODE number is deliberately kept separate from importance:
the climatologically largest-volume mode is additionally flagged MAIN_MODE.
This prevents the meaning of MODE1/MODE2 from changing between stations.

The detected climatological windows are then kept fixed and applied to every
hydrological year to build forecast predictands without fabricating missing
raw observations.

Dependencies
------------
    numpy, pandas, scipy, matplotlib

Example
-------
    from highwater_season_predictands_modeaware import HighWaterConfig, fit_station_highwater_regime
    regime = fit_station_highwater_regime(daily_discharge, HighWaterConfig())

No command-line interface or import-time execution is provided.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


try:
    from .highflow_utils import (clean_daily_series, canonical_doy, doy_to_md,
        window_dates, union_window_dates, hydrological_year_bounds,
        coverage_statistics, match_windows, window_day_set)
except ImportError:
    from highflow_utils import (clean_daily_series, canonical_doy, doy_to_md,
        window_dates, union_window_dates, hydrological_year_bounds,
        coverage_statistics, match_windows, window_day_set)

# =============================================================================
# 1. Configuration
# =============================================================================


@dataclass
class HighWaterConfig:
    """Configuration of the climatological high-water detection.

    The defaults are intentionally conservative.  They are not universal
    hydrological constants; they should be subjected to sensitivity analysis.
    """

    # --- Raw-data quality control ------------------------------------------------
    # Negative discharge is physically invalid for ordinary river discharge and
    # is converted to NaN. Large positive values are NOT automatically removed,
    # because genuine floods must not be treated as statistical outliers.
    negative_to_nan: bool = True
    missing_values: tuple[float, ...] = (-99.0, -999.0)

    # Before constructing a station climatology, retain only calendar years with
    # at least this fraction of valid daily observations. This is essential for
    # multi-station files whose common date axis is longer than an individual
    # gauge record. A year containing only NaNs must not count as a reference year.
    min_reference_year_coverage: float = 0.50

    # Minimum number of usable years needed to define a climatological regime.
    min_reference_years: int = 5

    # Minimum fraction of the retained usable reference years that must contribute
    # to a given month-day before its daily climatological mean is accepted.
    min_climatology_year_fraction: float = 0.60

    # Maximum number of consecutive missing days in the *climatology* that may be
    # interpolated. Raw annual observations are NEVER interpolated for the final
    # mean/max predictands.
    max_climatology_gap_days: int = 5

    # Circular centered smoothing of the daily climatology. An odd 7- or 15-day
    # window is generally reasonable for daily discharge climatology. Circular
    # smoothing avoids an artificial discontinuity at 31 Dec / 1 Jan.
    smoothing_window_days: int = 7

    # --- Hydrological-year definition -------------------------------------------
    # "auto" rotates the year to start in the climatological low-flow trough.
    # Alternatively provide e.g. "05-01" to force 1 May.
    hydrological_year_start: str = "auto"

    # Window used to identify the broad low-flow trough when the start is auto.
    # A longer window prevents a single anomalously low climatological day from
    # setting the hydrological-year origin.
    lowflow_anchor_window_days: int = 31

    # --- Mode detection ----------------------------------------------------------
    # A climatological mode must exceed this Pardé coefficient. P=1 means mean
    # annual discharge. Set slightly below 1 only if a weak secondary mode is
    # scientifically important.
    mode_min_parde: float = 1.0

    # Minimum peak separation on the circular climatological hydrograph.
    min_mode_separation_days: int = 30

    # Peak prominence is max(absolute, fraction * Parde_range). This combination
    # adapts to rivers with different amplitudes while suppressing tiny wiggles.
    mode_prominence_parde: float = 0.15
    mode_prominence_fraction: float = 0.10

    # Maximum number of climatological modes retained. The method supports more
    # than two modes; this cap mainly prevents noisy records from creating an
    # excessive number of regimes.
    max_modes: int = 5

    # --- Piecewise regime segmentation ------------------------------------------
    # For M detected modes, the default physical model uses 2*M+1 segments:
    # low - high - valley - high - ... - low. The dynamic-programming fit is
    # globally optimal for that number of slope regimes.
    min_segment_length_days: int = 15

    # A fitted slope regime is high-water when its mean Pardé coefficient is at
    # least this value. P=1 is especially interpretable: mean segment discharge
    # is above the annual-mean discharge.
    highwater_parde_threshold: float = 1.0

    # Classify a high-water season as "minor" if it contributes less than this
    # fraction of the climatological annual discharge volume. It is not deleted.
    minor_season_volume_fraction: float = 0.10

    # Explicit opt-in only: a no-mode fallback is descriptive, not a detected
    # high-water season. Flat hydrographs never receive an invented season.
    allow_weak_season_fallback: bool = False

    # --- Annual predictand construction -----------------------------------------
    # Minimum fraction of daily observations required within a climatological
    # high-water window to calculate its annual mean discharge.
    min_valid_fraction_mean: float = 0.80

    # Maxima are more vulnerable to missing days, so a stricter default is used.
    min_valid_fraction_max: float = 0.90

    # Hydrological years with less than this overall coverage are retained in the
    # output but flagged. Window-specific coverage still controls each target.
    min_hydrological_year_fraction: float = 0.70

    # --- Bootstrap uncertainty ---------------------------------------------------
    # Number of hydrological-year block-bootstrap replicates. 0 disables it.
    n_bootstrap: int = 200
    bootstrap_ci: float = 0.95
    random_state: int = 42

    # Re-detecting mode count inside every bootstrap replicate is more complete
    # but can cause label switching and is slower. False gives confidence limits
    # conditional on the full-sample mode structure; True includes uncertainty in
    # modality as well.
    bootstrap_reselect_modes: bool = False
    bootstrap_block_type: str = "hydrological_year"
    # Alternative: calendar_year reproduces the reference-year eligibility rule.
    # Hydrological blocks preserve cross-December seasonal dependence; their
    # intervals are explicitly conditional on this block eligibility and anchor.

    # Minimum Jaccard overlap for matching a bootstrap high-water window to a
    # reference window.
    bootstrap_min_overlap: float = 0.20

    # --- Optional mode-specific predictands ------------------------------------
    # False preserves the original behaviour exactly: annual targets are built
    # only from contiguous high-water seasons (HW1, HW2, ...). When True, the
    # code additionally builds MODE1, MODE2, ... windows and annual mean/max
    # targets for each prominence-selected climatological hydrograph mode.
    mode_specific_outputs: bool = False

    # Normally only modes lying inside a fitted high-water season are used for
    # high-flow mode targets. This is the hydrologically cleanest definition.
    # If True, a prominence-selected peak that is not captured by a fitted high-water
    # season is assigned a fallback window based on contiguous daily smoothed
    # Pardé values >= highwater_parde_threshold. Such windows are explicitly
    # flagged as fallback_mode_window=True.
    include_unassigned_modes: bool = False


# =============================================================================
# 2. Input reader for the user's discharge format
# =============================================================================


@dataclass
class PictureDischargeData:
    """Parsed discharge observations and station metadata."""

    discharge: pd.DataFrame  # DatetimeIndex x stations
    metadata: pd.DataFrame   # index=station; columns LON, LAT, ELEV


def read_picture_discharge_csv(path: str | Path) -> PictureDischargeData:
    """Read the supplied CPT/picture-like discharge CSV.

    Expected layout
    ---------------
        ID, station_1, station_2, ...
        LON, ...
        LAT, ...
        DAILY/ELEV, ...
        YYYYMMDD, Q, Q, ...
        ...

    The third metadata row may contain elevations. In the user's sample it is
    currently empty, which is accepted.
    """

    raw = pd.read_csv(path)
    if raw.shape[1] < 2:
        raise ValueError("The input must contain ID plus at least one station column.")
    if str(raw.columns[0]).strip().upper() != "ID":
        raise ValueError("The first column must be named 'ID'.")

    id_values = raw.iloc[:, 0].astype(str).str.strip()

    # Locate metadata rows by label rather than assuming exact row numbers.
    def _find_row(label: str) -> Optional[int]:
        hit = np.flatnonzero(id_values.str.upper().eq(label.upper()).to_numpy())
        return int(hit[0]) if len(hit) else None

    lon_row = _find_row("LON")
    lat_row = _find_row("LAT")

    # Accept either DAILY/ELEV or ELEV as the elevation marker.
    elev_hits = np.flatnonzero(
        id_values.str.upper().isin(["DAILY/ELEV", "ELEV"]).to_numpy()
    )
    elev_row = int(elev_hits[0]) if len(elev_hits) else None

    if lon_row is None or lat_row is None:
        raise ValueError("Could not find LON and LAT metadata rows.")

    stations = list(raw.columns[1:])
    metadata = pd.DataFrame(index=stations, columns=["LON", "LAT", "ELEV"], dtype=float)
    metadata.index.name = "station"
    metadata["LON"] = pd.to_numeric(raw.iloc[lon_row, 1:], errors="coerce").to_numpy()
    metadata["LAT"] = pd.to_numeric(raw.iloc[lat_row, 1:], errors="coerce").to_numpy()
    if elev_row is not None:
        metadata["ELEV"] = pd.to_numeric(raw.iloc[elev_row, 1:], errors="coerce").to_numpy()

    # Daily data start after the last identified metadata row. We still parse the
    # ID as YYYYMMDD and discard anything that is not a valid date, which makes
    # the reader robust to additional header rows.
    first_possible = max(x for x in [lon_row, lat_row, elev_row] if x is not None) + 1
    data = raw.iloc[first_possible:].copy()
    dates = pd.to_datetime(data.iloc[:, 0].astype(str), format="%Y%m%d", errors="coerce")
    valid_date = dates.notna().to_numpy()
    data = data.loc[valid_date]
    dates = dates.loc[valid_date]

    q = data.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    q.index = pd.DatetimeIndex(dates.to_numpy(), name="T")
    q.columns = stations
    q = q.sort_index()

    # Duplicate dates are aggregated by mean instead of silently retaining an
    # arbitrary duplicate row.
    if q.index.has_duplicates:
        warnings.warn("Duplicate dates found; duplicate daily rows are averaged.")
        q = q.groupby(level=0).mean()

    return PictureDischargeData(discharge=q, metadata=metadata)


# =============================================================================
# 3. Calendar helpers: use a canonical non-leap year
# =============================================================================


_CANONICAL_YEAR = 2001  # Non-leap year; Jan 1 -> DOY 1, Dec 31 -> DOY 365.


def _remove_feb29(series: pd.Series) -> pd.Series:
    """Remove 29 February so all hydrological years use 365 climatological days."""
    mask = ~((series.index.month == 2) & (series.index.day == 29))
    return series.loc[mask]


def _canonical_doy(index: pd.DatetimeIndex) -> np.ndarray:
    """Canonical month-day labels (29 February -> 59); not an aggregation axis."""
    return canonical_doy(index)


def _doy_to_monthday(doy: int) -> str:
    return doy_to_md(doy)


def _monthday_to_doy(monthday: str) -> int:
    try:
        dt = pd.Timestamp(f"{_CANONICAL_YEAR}-{monthday}")
    except Exception as exc:
        raise ValueError(f"Invalid month-day '{monthday}', expected MM-DD.") from exc
    return int(dt.dayofyear)


def _rotate(values: np.ndarray, anchor_doy: int) -> np.ndarray:
    """Rotate 365 values so rotated position 1 corresponds to anchor_doy."""
    a = int(anchor_doy) - 1
    return np.concatenate([values[a:], values[:a]])


def _rotpos_to_doy(rot_pos: int, anchor_doy: int) -> int:
    """Convert 1-based hydrological position to canonical calendar DOY."""
    return ((int(anchor_doy) + int(rot_pos) - 2) % 365) + 1


def _doy_to_rotpos(doy: int, anchor_doy: int) -> int:
    """Convert canonical calendar DOY to 1-based hydrological position."""
    return ((int(doy) - int(anchor_doy)) % 365) + 1


# =============================================================================
# 4. Climatology and robust circular smoothing
# =============================================================================


def _circular_rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Centered circular moving mean; preserves the annual mean when finite."""
    values = np.asarray(values, dtype=float)
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        raise ValueError("smoothing_window_days must be odd.")
    if window > len(values):
        raise ValueError("Smoothing window cannot exceed the climatological year.")

    half = window // 2
    extended = np.concatenate([values[-half:], values, values[:half]])
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(extended, kernel, mode="valid")


def _limited_circular_interpolate(values: np.ndarray, max_gap: int) -> np.ndarray:
    """Interpolate only short missing gaps on a circular 365-day climatology.

    This is deliberately applied only to the *climatology*. We never interpolate
    raw annual data before calculating annual Qmean or Qmax, because doing so can
    manufacture or suppress hydrological extremes.
    """
    x = np.asarray(values, dtype=float).copy()
    n = len(x)
    if not np.isnan(x).any() or max_gap <= 0:
        return x
    if np.isnan(x).all():
        return x

    # Triplicate the year so interpolation at Jan/Dec can use values from the
    # opposite side of the calendar, then keep the central copy.
    ext = pd.Series(np.tile(x, 3), dtype=float)
    ext = ext.interpolate(
        method="linear",
        limit=int(max_gap),
        limit_area="inside",
    )
    return ext.iloc[n : 2 * n].to_numpy(dtype=float)


def prepare_series(series: pd.Series, config: HighWaterConfig) -> pd.Series:
    """Daily QC; retain Gregorian leap days for all subsequent aggregates."""
    return clean_daily_series(series, negative_to_nan=config.negative_to_nan,
                              missing_values=config.missing_values)


def build_daily_climatology(
    series: pd.Series,
    config: HighWaterConfig,
    reference_start: Optional[str | pd.Timestamp] = None,
    reference_end: Optional[str | pd.Timestamp] = None,
) -> pd.DataFrame:
    """Construct quality-controlled mean daily climatology (365 days).

    The arithmetic mean is the default because the cumulative sum of the mean
    hydrograph retains the expected runoff-volume interpretation. A median
    climatology would be more robust to extremes but would no longer represent
    mean runoff volume in the same way.
    """
    for name in ('min_reference_year_coverage','min_climatology_year_fraction','min_valid_fraction_mean','min_valid_fraction_max'):
        if not 0 < getattr(config,name) <= 1: raise ValueError(f'{name} must lie in (0,1].')
    if config.min_reference_years < 1 or config.min_segment_length_days < 1:
        raise ValueError('Reference years and segment length must be positive.')
    q = prepare_series(series, config)

    if reference_start is not None:
        q = q.loc[pd.Timestamp(reference_start):]
    if reference_end is not None:
        q = q.loc[:pd.Timestamp(reference_end)]
    if q.empty:
        raise ValueError("No data remain in the requested reference period.")

    q = _remove_feb29(q)
    # Pad partial boundary years before calculating eligibility. Missing rows
    # cannot improve coverage: every reference calendar year has 365 days.
    full_index = pd.date_range(f"{q.index.min().year}-01-01", f"{q.index.max().year}-12-31")
    full_index = full_index[~((full_index.month == 2) & (full_index.day == 29))]
    q = q.reindex(full_index)
    # The common file may span years during which this particular station has no
    # observations. Determine station-specific usable reference years first.
    year_coverage = q.notna().groupby(q.index.year).mean()
    usable_years = year_coverage[
        year_coverage >= config.min_reference_year_coverage
    ].index.to_numpy(dtype=int)

    if len(usable_years) < config.min_reference_years:
        raise ValueError(
            f"Only {len(usable_years)} usable reference years meet the "
            f"{config.min_reference_year_coverage:.0%} annual-coverage criterion; "
            f"at least {config.min_reference_years} are required."
        )

    q = q[q.index.year.isin(usable_years)]
    years = np.sort(usable_years)
    n_years = len(years)

    doy = _canonical_doy(q.index)
    frame = pd.DataFrame({"Q": q.to_numpy(), "doy": doy}, index=q.index)
    grouped = frame.groupby("doy")["Q"]

    clim_mean = grouped.mean().reindex(np.arange(1, 366))
    valid_count = grouped.count().reindex(np.arange(1, 366), fill_value=0)
    valid_fraction = valid_count / float(n_years)

    # Days supported by too few years are intentionally treated as missing.
    accepted = clim_mean.where(valid_fraction >= config.min_climatology_year_fraction)
    interpolated = _limited_circular_interpolate(
        accepted.to_numpy(dtype=float), config.max_climatology_gap_days
    )

    # Long gaps remaining after limited interpolation are scientifically unsafe:
    # the season would be inferred from invented information.
    if not np.all(np.isfinite(interpolated)):
        bad = int(np.sum(~np.isfinite(interpolated)))
        raise ValueError(
            f"Daily climatology retains {bad} unsupported days after short-gap "
            "interpolation. Lower min_climatology_year_fraction only if justified, "
            "or improve the observational record."
        )

    if np.nanmax(interpolated) <= 0 or np.nansum(interpolated) <= 0:
        raise ValueError("Climatological discharge is zero or invalid.")

    smoothed = _circular_rolling_mean(interpolated, config.smoothing_window_days)

    out = pd.DataFrame(
        {
            "doy": np.arange(1, 366),
            "month_day": [_doy_to_monthday(d) for d in range(1, 366)],
            "q_clim": interpolated,
            "q_smooth": smoothed,
            "n_valid_years": valid_count.to_numpy(dtype=int),
            "valid_year_fraction": valid_fraction.to_numpy(dtype=float),
        }
    ).set_index("doy")
    out.attrs["n_reference_years"] = n_years
    out.attrs["reference_years"] = years.tolist()
    return out


# =============================================================================
# 5. Hydrological-year anchor and circular seasonality diagnostics
# =============================================================================


def choose_hydrological_anchor(
    climatology: pd.DataFrame,
    config: HighWaterConfig,
) -> int:
    """Choose the hydrological-year start as a broad low-flow trough.

    Starting the hydrological year in a climatological low-flow period prevents
    a high-water season crossing 31 December from being split artificially at
    the calendar-year boundary.
    """
    if config.hydrological_year_start.lower() != "auto":
        return _monthday_to_doy(config.hydrological_year_start)

    q = climatology["q_smooth"].to_numpy(dtype=float)
    low = _circular_rolling_mean(q, config.lowflow_anchor_window_days)
    return int(np.argmin(low)) + 1


def circular_seasonality_metrics(q_calendar: np.ndarray) -> dict[str, float]:
    """Flow-weighted circular timing and concentration diagnostics.

    The resultant length R ranges from 0 to 1. Low R can indicate weak
    seasonality or opposing/multiple modes, so it is diagnostic rather than a
    standalone modality classifier.
    """
    q = np.asarray(q_calendar, dtype=float)
    q = np.clip(q, 0.0, None)
    if q.sum() <= 0:
        return {"center_of_mass_doy": np.nan, "circular_concentration": np.nan}

    theta = 2.0 * np.pi * np.arange(365) / 365.0
    z = np.sum(q * np.exp(1j * theta)) / np.sum(q)
    angle = float(np.angle(z))
    if angle < 0:
        angle += 2.0 * np.pi
    # angle 0 corresponds to canonical day 1.
    com_doy = int(round(angle / (2.0 * np.pi) * 365.0)) + 1
    if com_doy > 365:
        com_doy -= 365
    return {
        "center_of_mass_doy": float(com_doy),
        "circular_concentration": float(abs(z)),
    }


# =============================================================================
# 6. Peak/mode detection
# =============================================================================


def detect_climatological_modes(parde_rotated: np.ndarray, config: HighWaterConfig) -> dict[str, Any]:
    """Circular prominence-based peak screening (NOT a significance test)."""
    p=np.asarray(parde_rotated,float);n=len(p)
    if n!=365 or not np.isfinite(p).all():raise ValueError('Expected a finite 365-day Pardé curve.')
    dynamic_range=float(np.ptp(p))
    prominence=max(config.mode_prominence_parde,config.mode_prominence_fraction*dynamic_range)
    ext=np.tile(p,3)
    peaks,props=find_peaks(ext,height=config.mode_min_parde,prominence=prominence)
    valid=(peaks>=n)&(peaks<2*n)
    peaks=peaks[valid]-n;prom=np.asarray(props.get('prominences',[]))[valid]
    chosen=[]
    for j in np.argsort(-prom,kind='stable'):
        if all(min(abs(int(peaks[j])-int(peaks[k])),n-abs(int(peaks[j])-int(peaks[k])))>=config.min_mode_separation_days for k in chosen):
            chosen.append(int(j))
        if len(chosen)>=config.max_modes:break
    chosen=sorted(chosen,key=lambda j:peaks[j]);pk=peaks[chosen]
    return dict(peak_positions=(pk+1).astype(int).tolist(),peak_parde=p[pk].tolist(),
                prominences=prom[chosen].tolist(),n_modes=len(pk),prominence_threshold=float(prominence),
                selection_interpretation='heuristic prominence screening, not a statistical significance test')


# =============================================================================
# 7. Globally optimal piecewise slope segmentation
# =============================================================================


def _segment_sse_cost_matrix(y: np.ndarray) -> np.ndarray:
    """Cost[i,j] = SSE of a constant mean fitted to y[i:j].

    Segmenting the Pardé/slope series into constant regimes is equivalent to a
    continuous piecewise-linear approximation of the cumulative mass curve
    after integration.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    ps = np.concatenate([[0.0], np.cumsum(y)])
    pss = np.concatenate([[0.0], np.cumsum(y * y)])
    cost = np.full((n + 1, n + 1), np.inf, dtype=float)

    for i in range(n):
        j = np.arange(i + 1, n + 1)
        m = j - i
        sy = ps[j] - ps[i]
        syy = pss[j] - pss[i]
        sse = syy - (sy * sy) / m
        cost[i, j] = np.maximum(sse, 0.0)
    return cost


def fit_piecewise_constant_slopes(
    parde_rotated: np.ndarray,
    n_segments: int,
    min_segment_length: int,
    cost_matrix: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Globally optimize contiguous slope regimes by dynamic programming."""
    y = np.asarray(parde_rotated, dtype=float)
    n = len(y)
    k = int(n_segments)
    L = int(min_segment_length)

    if k < 1:
        raise ValueError("n_segments must be >= 1.")
    if n < k * L:
        raise ValueError(
            f"Cannot fit {k} segments of at least {L} days to {n} days."
        )
    if not np.all(np.isfinite(y)):
        raise ValueError("Pardé series contains non-finite values.")

    cost = _segment_sse_cost_matrix(y) if cost_matrix is None else cost_matrix

    dp = np.full((k + 1, n + 1), np.inf, dtype=float)
    prev = np.full((k + 1, n + 1), -1, dtype=int)
    dp[0, 0] = 0.0

    for kk in range(1, k + 1):
        for end in range(kk * L, n + 1):
            starts = np.arange((kk - 1) * L, end - L + 1)
            candidate = dp[kk - 1, starts] + cost[starts, end]
            j = int(np.argmin(candidate))
            dp[kk, end] = float(candidate[j])
            prev[kk, end] = int(starts[j])

    if not np.isfinite(dp[k, n]):
        raise ValueError("No valid piecewise segmentation was found.")

    # Backtrack globally optimal breakpoints.
    boundaries = [n]
    end = n
    for kk in range(k, 0, -1):
        start = int(prev[kk, end])
        boundaries.append(start)
        end = start
    boundaries = sorted(boundaries)

    fitted = np.empty(n, dtype=float)
    segments: list[dict[str, Any]] = []
    for sid, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]), start=1):
        mean_parde = float(np.mean(y[start:end]))
        fitted[start:end] = mean_parde
        segments.append(
            {
                "segment_id": sid,
                "start_pos": int(start + 1),
                "end_pos": int(end),
                "duration_days": int(end - start),
                "mean_parde": mean_parde,
                # normalized cumulative-curve slope per day
                "normalized_cumulative_slope": mean_parde / 365.0,
            }
        )

    # Since each segment mean preserves the sum over that segment, sum(fitted)
    # equals sum(original Pardé)=365 (apart from floating error). Therefore the
    # integrated fitted mass curve ends at exactly ~1 and is continuous.
    fitted_cumulative = np.cumsum(fitted) / np.sum(fitted)

    return {
        "n_segments": k,
        "sse": float(dp[k, n]),
        "segments": segments,
        "fitted_parde": fitted,
        "fitted_cumulative": fitted_cumulative,
        "break_positions": [int(x) for x in boundaries[1:-1]],
    }


def _choose_n_segments_from_modes(n_modes: int, config: HighWaterConfig) -> int:
    """Heuristic complexity M -> 2M+1; does not impose slope alternation."""
    if n_modes <= 0:
        return 3
    requested = 2 * int(n_modes) + 1
    maximum = 365 // int(config.min_segment_length_days)
    return max(3, min(requested, maximum))


# =============================================================================
# 8. Convert fitted slope regimes into high-water seasons
# =============================================================================


def _build_highwater_windows(fit, q_rotated, parde_rotated, anchor_doy, mode_info, config):
    """Join contiguous high regimes, including across the hydrological anchor.

    No hydrological season is invented for a constant/no-mode curve. An optional
    weak-season fallback is flagged and downstream selectors reject it by default.
    """
    q=np.asarray(q_rotated,float);p=np.asarray(parde_rotated,float);segments=fit['segments']
    no_mode=not mode_info.get('n_modes',0)
    if np.ptp(p)<1e-10 or (no_mode and not config.allow_weak_season_fallback):return []
    high=[s['mean_parde']>=config.highwater_parde_threshold for s in segments]
    spans=[];i=0
    while i<len(segments):
        if not high[i]:i+=1;continue
        j=i
        while j+1<len(segments) and high[j+1]:j+=1
        spans.append((int(segments[i]['start_pos']),int(segments[j]['end_pos'])))
        i=j+1
    fallback=False
    if not spans and config.allow_weak_season_fallback:
        best=max(segments,key=lambda s:s['mean_parde']);spans=[(best['start_pos'],best['end_pos'])];fallback=True
    if len(spans)>1 and spans[0][0]==1 and spans[-1][1]==365:
        a=spans.pop(0);b=spans.pop(-1);spans.append((b[0],a[1]+365))
    spans.sort();out=[];peaks=np.asarray(mode_info.get('peak_positions',[]),dtype=int)
    for sid,(start,end) in enumerate(spans,1):
        ix=np.arange(start-1,end)%365;qwin=q[ix];pwin=p[ix]
        inside=[int(k) for k in peaks if ((k-start)%365)<=end-start]
        pp=start+int(np.argmax(qwin));sd=_rotpos_to_doy(start,anchor_doy);ed=_rotpos_to_doy(end,anchor_doy)
        pdoy=_rotpos_to_doy(pp,anchor_doy);vol=float(qwin.sum());fraction=vol/float(q.sum())
        out.append(dict(season_id=sid,start_pos=start,end_pos=end,duration_days=end-start+1,
             start_doy=sd,end_doy=ed,peak_doy=pdoy,start_month_day=_doy_to_monthday(sd),
             end_month_day=_doy_to_monthday(ed),peak_month_day=_doy_to_monthday(pdoy),
             crosses_calendar_year=bool(sd>ed),crosses_hydrological_anchor=bool(end>365),
             mean_parde=float(pwin.mean()),max_parde=float(pwin.max()),
             climatological_mean_q=float(qwin.mean()),climatological_max_q=float(qwin.max()),
             climatological_volume_q_day=vol,climatological_volume_m3=vol*86400,
             annual_volume_fraction=fraction,importance='minor' if fraction<config.minor_season_volume_fraction else 'major',
             n_modes_inside=len(inside),mode_positions=inside,weak_season_fallback=bool(fallback or no_mode)))
    for rank,j in enumerate(sorted(range(len(out)),key=lambda j:-out[j]['annual_volume_fraction']),1):
        out[j].update(volume_rank=rank,is_main_season=rank==1)
    return out


def _threshold_component_around_peak(
    parde_rotated: np.ndarray,
    peak_pos: int,
    threshold: float,
) -> Optional[tuple[int, int]]:
    """Return the contiguous daily Pardé>=threshold component around a peak.

    Parameters
    ----------
    parde_rotated : ndarray
        Smoothed daily Pardé coefficients on the rotated hydrological year.
    peak_pos : int
        1-based hydrological position of the climatological peak.
    threshold : float
        Daily Pardé threshold used only for the optional fallback mode window.

    Notes
    -----
    This is *not* the primary high-water-season definition. The primary season
    still comes from the piecewise fitted slope regimes. This helper exists only
    when ``include_unassigned_modes=True`` and a prominence-selected climatological mode
    lies outside all fitted high-water seasons.

    Because the year is rotated to start in a broad climatological low-flow
    trough, a genuine high-flow component should not wrap across array ends.
    """
    p = np.asarray(parde_rotated, dtype=float)
    idx = int(peak_pos) - 1
    if idx < 0 or idx >= len(p) or not np.isfinite(p[idx]) or p[idx] < threshold:
        return None

    left = idx
    while left > 0 and np.isfinite(p[left - 1]) and p[left - 1] >= threshold:
        left -= 1

    right = idx
    while (
        right < len(p) - 1
        and np.isfinite(p[right + 1])
        and p[right + 1] >= threshold
    ):
        right += 1

    return int(left + 1), int(right + 1)  # convert to 1-based positions


def _build_mode_specific_windows(
    highwater_windows: list[dict[str, Any]],
    q_rotated: np.ndarray,
    parde_rotated: np.ndarray,
    anchor_doy: int,
    mode_info: dict[str, Any],
    config: HighWaterConfig,
) -> list[dict[str, Any]]:
    """Build chronological MODE1, MODE2, ... high-flow windows.

    Scientific logic
    ----------------
    A *mode* is a prominence-selected climatological peak. A *high-water season* is a
    contiguous fitted regime above the selected Pardé threshold. These are not
    always the same object.

    1. If a high-water season contains exactly one prominence-selected peak, the mode
       window equals that high-water season.
    2. If one continuous high-water season contains several prominence-selected peaks,
       the season is partitioned at the local climatological minimum between
       each adjacent pair of peaks. This yields MODE1/MODE2/... targets without
       falsely claiming that the river has several separate high-water seasons.
    3. If requested, an otherwise unassigned prominence-selected mode can receive a
       fallback window from the contiguous *daily smoothed* Pardé>=threshold
       component around its peak. This fallback is flagged explicitly.

    Mode IDs are chronological in the rotated hydrological year. Importance is
    stored separately through ``volume_rank`` and ``is_main_mode``.
    """
    if not config.mode_specific_outputs:
        return []

    q = np.asarray(q_rotated, dtype=float)
    p = np.asarray(parde_rotated, dtype=float)
    total_q = float(np.sum(q))
    all_peaks = sorted(int(x) for x in mode_info.get("peak_positions", []))

    candidates: list[dict[str, Any]] = []
    assigned_peaks: set[int] = set()

    for season in highwater_windows:
        season_start = int(season["start_pos"])
        season_end = int(season["end_pos"])
        peaks = sorted(
            season_start + ((p0-season_start) % 365)
            for p0 in all_peaks if ((p0-season_start) % 365) <= season_end-season_start
        )
        if not peaks:
            # This can occur for a weak-season fallback or under a deliberately
            # strict mode-prominence criterion. The high-water season remains
            # valid, but there is no mode-specific target to define from it.
            continue

        assigned_peaks.update((pk-1)%365+1 for pk in peaks)

        # Find one climatological valley between every adjacent pair of modes.
        # The search is strictly BETWEEN the peaks, never including the peaks
        # themselves, so a boundary cannot collapse onto a local maximum.
        valleys: list[int] = []
        for left_peak, right_peak in zip(peaks[:-1], peaks[1:]):
            if right_peak - left_peak <= 1:
                valley_pos = int((left_peak + right_peak) // 2)
            else:
                # 1-based positions left_peak+1 ... right_peak-1 correspond to
                # zero-based indexes left_peak ... right_peak-2.
                interior = q[np.arange(left_peak, right_peak - 1) % 365]
                if len(interior) == 0 or not np.any(np.isfinite(interior)):
                    valley_pos = int((left_peak + right_peak) // 2)
                else:
                    valley_pos = int(left_peak + np.nanargmin(interior) + 1)
            valleys.append(valley_pos)

        # Partition the parent high-water season without overlap or gaps:
        #   first mode  : season_start ... valley_1
        #   middle mode : valley_{j-1}+1 ... valley_j
        #   last mode   : valley_last+1 ... season_end
        starts = [season_start] + [v + 1 for v in valleys]
        ends = valleys + [season_end]

        for local_index, (peak_pos, start, end) in enumerate(
            zip(peaks, starts, ends), start=1
        ):
            # Defensive check: the peak must remain inside its own partition.
            # If numerical/pathological geometry violates this, contract/expand
            # the window minimally around the peak rather than silently failing.
            start = min(int(start), int(peak_pos))
            end = max(int(end), int(peak_pos))

            candidates.append(
                {
                    "peak_pos": int(peak_pos),
                    "start_pos": int(start),
                    "end_pos": int(end),
                    "parent_season_id": int(season["season_id"]),
                    "parent_season_n_modes": int(len(peaks)),
                    "local_mode_index_in_parent": int(local_index),
                    "split_from_multimode_season": bool(len(peaks) > 1),
                    "fallback_mode_window": False,
                    "boundary_method": (
                        "intermodal_climatological_valley"
                        if len(peaks) > 1
                        else "parent_highwater_season"
                    ),
                }
            )

    # Optional fallback for a prominence-selected peak not captured by any fitted
    # high-water season. Disabled by default because a "high-flow mode" should
    # normally live inside an objectively detected high-water period.
    if config.include_unassigned_modes:
        for peak_pos in all_peaks:
            if peak_pos in assigned_peaks:
                continue
            component = _threshold_component_around_peak(
                p,
                peak_pos,
                config.highwater_parde_threshold,
            )
            if component is None:
                continue
            start, end = component
            candidates.append(
                {
                    "peak_pos": int(peak_pos),
                    "start_pos": int(start),
                    "end_pos": int(end),
                    "parent_season_id": None,
                    "parent_season_n_modes": 0,
                    "local_mode_index_in_parent": None,
                    "split_from_multimode_season": False,
                    "fallback_mode_window": True,
                    "boundary_method": "daily_parde_threshold_fallback",
                }
            )

    # Stable chronological numbering is defined AFTER all candidates are built.
    candidates.sort(key=lambda d: (int(d["peak_pos"])-1)%365)
    mode_windows: list[dict[str, Any]] = []

    for mode_id, cand in enumerate(candidates, start=1):
        start = int(cand["start_pos"])
        end = int(cand["end_pos"])
        peak_pos = int(cand["peak_pos"])
        ix = np.arange(start - 1, end) % 365
        qwin = q[ix]
        pwin = p[ix]

        start_doy = _rotpos_to_doy(start, anchor_doy)
        end_doy = _rotpos_to_doy(end, anchor_doy)
        peak_doy = _rotpos_to_doy(peak_pos, anchor_doy)
        volume = float(np.sum(qwin))

        record = {
            **cand,
            "mode_id": int(mode_id),
            "mode_name": f"MODE{mode_id}",
            "duration_days": int(end - start + 1),
            "start_doy": int(start_doy),
            "end_doy": int(end_doy),
            "peak_doy": int(peak_doy),
            "start_month_day": _doy_to_monthday(start_doy),
            "end_month_day": _doy_to_monthday(end_doy),
            "peak_month_day": _doy_to_monthday(peak_doy),
            "crosses_calendar_year": bool(start_doy > end_doy),
            "mean_parde": float(np.mean(pwin)),
            "max_parde": float(np.max(pwin)),
            "climatological_mean_q": float(np.mean(qwin)),
            "climatological_max_q": float(np.max(qwin)),
            "climatological_volume_q_day": volume,
            "climatological_volume_m3": float(volume * 86400.0),
            "annual_volume_fraction": float(volume / total_q),
        }

        parent_id = cand.get("parent_season_id")
        if parent_id is not None:
            parent = next(
                (w for w in highwater_windows if int(w["season_id"]) == int(parent_id)),
                None,
            )
            parent_volume = (
                float(parent["climatological_volume_q_day"])
                if parent is not None
                else np.nan
            )
            record["fraction_of_parent_season_volume"] = (
                float(volume / parent_volume)
                if np.isfinite(parent_volume) and parent_volume > 0
                else np.nan
            )
        else:
            record["fraction_of_parent_season_volume"] = np.nan

        mode_windows.append(record)

    # Importance ranking is independent of chronological mode number.
    ranking = sorted(
        range(len(mode_windows)),
        key=lambda j: mode_windows[j]["annual_volume_fraction"],
        reverse=True,
    )
    for rank, j in enumerate(ranking, start=1):
        mode_windows[j]["volume_rank"] = int(rank)
        mode_windows[j]["is_main_mode"] = bool(rank == 1)
        mode_windows[j]["importance"] = (
            "minor"
            if mode_windows[j]["annual_volume_fraction"]
            < config.minor_season_volume_fraction
            else "major"
        )

    # Enrich parent high-water windows with the chronological MODE IDs that lie
    # inside them. This is convenient in CSV diagnostics and plotting.
    for season in highwater_windows:
        season["mode_ids"] = [
            int(m["mode_id"])
            for m in mode_windows
            if m.get("parent_season_id") is not None
            and int(m["parent_season_id"]) == int(season["season_id"])
        ]

    return mode_windows


# =============================================================================
# 9. Complete station-level regime fit
# =============================================================================


def fit_station_highwater_regime(
    series: pd.Series,
    config: HighWaterConfig,
    reference_start: Optional[str | pd.Timestamp] = None,
    reference_end: Optional[str | pd.Timestamp] = None,
    fixed_anchor_doy: Optional[int] = None,
    fixed_n_modes: Optional[int] = None,
) -> dict[str, Any]:
    """Fit the climatological high-water structure for one station."""
    clim = build_daily_climatology(
        series,
        config,
        reference_start=reference_start,
        reference_end=reference_end,
    )

    anchor_doy = (
        int(fixed_anchor_doy)
        if fixed_anchor_doy is not None
        else choose_hydrological_anchor(clim, config)
    )

    q_calendar = clim["q_smooth"].to_numpy(dtype=float)
    q_rot = _rotate(q_calendar, anchor_doy)
    annual_mean = float(np.mean(q_rot))
    parde_rot = q_rot / annual_mean
    normalized_increment = q_rot / float(np.sum(q_rot))
    cumulative = np.cumsum(normalized_increment)

    mode_info = detect_climatological_modes(parde_rot, config)
    if fixed_n_modes is not None:
        n_modes_for_complexity = int(fixed_n_modes)
    else:
        n_modes_for_complexity = int(mode_info["n_modes"])

    n_segments = _choose_n_segments_from_modes(n_modes_for_complexity, config)
    fit = fit_piecewise_constant_slopes(
        parde_rot,
        n_segments=n_segments,
        min_segment_length=config.min_segment_length_days,
    )

    windows = _build_highwater_windows(
        fit=fit,
        q_rotated=q_rot,
        parde_rotated=parde_rot,
        anchor_doy=anchor_doy,
        mode_info=mode_info,
        config=config,
    )

    mode_windows = _build_mode_specific_windows(
        highwater_windows=windows,
        q_rotated=q_rot,
        parde_rotated=parde_rot,
        anchor_doy=anchor_doy,
        mode_info=mode_info,
        config=config,
    )

    circular = circular_seasonality_metrics(q_calendar)
    n_modes = int(mode_info["n_modes"])
    if n_modes == 0:
        modality = "weak_or_undefined"
    elif n_modes == 1:
        modality = "unimodal"
    elif n_modes == 2:
        modality = "bimodal"
    else:
        modality = f"multimodal_{n_modes}"

    return {
        "anchor_doy": int(anchor_doy),
        "anchor_month_day": _doy_to_monthday(anchor_doy),
        "annual_mean_q": annual_mean,
        "climatology": clim,
        "reference_start": str(reference_start) if reference_start is not None else None,
        "reference_end": str(reference_end) if reference_end is not None else None,
        "reference_years": clim.attrs.get("reference_years", []),
        "regime_status": "OK" if windows else "NO_IDENTIFIABLE_SEASON",
        "complexity_rule": "2M+1 heuristic; alternation is not constrained",

        "q_rotated": q_rot,
        "parde_rotated": parde_rot,
        "normalized_increment_rotated": normalized_increment,
        "cumulative_rotated": cumulative,
        "mode_info": mode_info,
        "modality": modality,
        "n_modes": n_modes,
        "n_highwater_seasons": int(len(windows)),
        "n_mode_windows": int(len(mode_windows)),
        "piecewise_fit": fit,
        "windows": windows,
        "mode_windows": mode_windows,
        **circular,
    }


# =============================================================================
# 10. Hydrological-year mapping and annual forecast predictands
# =============================================================================


def _assign_hydrological_coordinates(
    index: pd.DatetimeIndex,
    anchor_doy: int,
) -> pd.DataFrame:
    """Assign hydrological-year label and rotated day 1..365 to real dates."""
    doy = _canonical_doy(index)
    rot = ((_canonical_doy(index) - int(anchor_doy)) % 365) + 1

    # A date belongs to the hydrological year that began at the most recent
    # anchor month-day. This works for arbitrary anchor dates and cross-year
    # high-water seasons.
    anchor_md = _doy_to_monthday(anchor_doy)
    anchor_month, anchor_day = [int(x) for x in anchor_md.split("-")]
    after_anchor = (index.month > anchor_month) | (
        (index.month == anchor_month) & (index.day >= anchor_day)
    )
    hydro_year = index.year - (~after_anchor).astype(int)

    return pd.DataFrame(
        {
            "hydro_year": hydro_year.astype(int),
            "rot_pos": rot.astype(int),
            "calendar_doy": doy.astype(int),
        },
        index=index,
    )


def compute_annual_predictands(series: pd.Series, regime: dict[str, Any], config: HighWaterConfig) -> pd.DataFrame:
    """Fixed-window annual targets on the actual Gregorian calendar.

    A year labels the hydrological year of each window's START. Cross-anchor
    windows are allowed and extend into the next hydrological year. Feb 29 is
    retained. ALL uses a UNION of expected dates and the same coverage rules,
    never a concatenation of just the available values. Missingness is not
    interpolated, and an accepted partial mean remains an available-day mean.
    """
    q=prepare_series(series,config)
    if q.empty:return pd.DataFrame(columns=['hydro_year'])
    anchor=int(regime['anchor_doy']);coord=_assign_hydrological_coordinates(q.index,anchor)
    records=[]
    for year in sorted(coord.hydro_year.unique()):
        hy_dates=pd.date_range(*hydrological_year_bounds(int(year),anchor),freq='D')
        frac=q.reindex(hy_dates).notna().sum()/len(hy_dates)
        rec=dict(hydro_year=int(year),hydro_year_valid_fraction=float(frac),
                 hydro_year_ok=bool(frac>=config.min_hydrological_year_fraction),
                 hydro_year_n_expected=len(hy_dates))
        families=[('HW',regime.get('windows',[]),'season_id','is_main_season')]
        if config.mode_specific_outputs:families.append(('MODE',regime.get('mode_windows',[]),'mode_id','is_main_mode'))
        for kind,wins,idkey,mainkey in families:
            main_mean=main_max=np.nan
            for win in wins:
                prefix=f'{kind}{int(win[idkey])}'
                dates=window_dates(int(year),anchor,win)
                stats=coverage_statistics(q,dates,config.min_valid_fraction_mean,config.min_valid_fraction_max)
                rec.update({f'{prefix}_{k}':v for k,v in stats.items()})
                rec[f'{prefix}_START_DATE']=dates[0];rec[f'{prefix}_END_DATE']=dates[-1]
                if win.get(mainkey,False):main_mean,main_max=stats['Q_MEAN'],stats['Q_MAX']
            dates=union_window_dates(int(year),anchor,wins)
            stats=coverage_statistics(q,dates,config.min_valid_fraction_mean,config.min_valid_fraction_max)
            rec[f'Q_{kind}_MEAN_MAIN']=main_mean;rec[f'Q_{kind}_MAX_MAIN']=main_max
            rec[f'Q_{kind}_MEAN_ALL']=stats['Q_MEAN'];rec[f'Q_{kind}_MAX_ALL']=stats['Q_MAX']
            for key in ('N_VALID','N_EXPECTED','VALID_FRACTION','MEAN_OK','MAX_OK'):
                rec[f'{kind}_ALL_{key}']=stats[key]
        records.append(rec)
    out=pd.DataFrame(records)
    out.attrs['calendar']='Gregorian aggregation; 365-day climatological boundaries'
    out.attrs['year_label']='Hydrological year containing each window start'
    return out.sort_values('hydro_year').reset_index(drop=True)


# =============================================================================
# 11. Hydrological-year block bootstrap for boundary uncertainty
# =============================================================================


def _hydrological_year_matrix(
    series: pd.Series,
    anchor_doy: int,
    config: HighWaterConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return matrix [hydrological_year, rotated_day] for block bootstrap."""
    q = _remove_feb29(prepare_series(series, config))
    coord = _assign_hydrological_coordinates(q.index, anchor_doy)
    frame = pd.DataFrame({"Q": q.to_numpy(dtype=float)}, index=q.index).join(coord)

    years = np.sort(frame["hydro_year"].unique())
    mat = np.full((len(years), 365), np.nan, dtype=float)
    retained_years = []
    rows = []

    for hy in years:
        g = frame[frame["hydro_year"] == hy]
        row = np.full(365, np.nan, dtype=float)
        # There is normally one value per rotated day after Feb 29 removal.
        for pos, vals in g.groupby("rot_pos")["Q"]:
            row[int(pos) - 1] = float(vals.mean())
        coverage = np.mean(np.isfinite(row))
        if coverage >= config.min_hydrological_year_fraction:
            retained_years.append(int(hy))
            rows.append(row)

    if not rows:
        return np.empty((0, 365)), np.array([], dtype=int)
    return np.vstack(rows), np.asarray(retained_years, dtype=int)


def _fit_from_rotated_climatology(
    q_rot: np.ndarray,
    anchor_doy: int,
    config: HighWaterConfig,
    fixed_n_modes: Optional[int] = None,
) -> dict[str, Any]:
    """Internal fast fit used by bootstrap replicates."""
    q = np.asarray(q_rot, dtype=float)
    q = _limited_circular_interpolate(q, config.max_climatology_gap_days)
    if not np.all(np.isfinite(q)) or np.sum(q) <= 0:
        raise ValueError("Invalid bootstrap climatology.")

    q = _circular_rolling_mean(q, config.smoothing_window_days)
    parde = q / np.mean(q)
    mode_info = detect_climatological_modes(parde, config)
    n_modes_for_complexity = (
        int(mode_info["n_modes"]) if fixed_n_modes is None else int(fixed_n_modes)
    )
    n_segments = _choose_n_segments_from_modes(n_modes_for_complexity, config)
    fit = fit_piecewise_constant_slopes(
        parde, n_segments, config.min_segment_length_days
    )
    windows = _build_highwater_windows(
        fit, q, parde, anchor_doy, mode_info, config
    )
    mode_windows = _build_mode_specific_windows(
        highwater_windows=windows,
        q_rotated=q,
        parde_rotated=parde,
        anchor_doy=anchor_doy,
        mode_info=mode_info,
        config=config,
    )
    return {
        "windows": windows,
        "mode_windows": mode_windows,
        "mode_info": mode_info,
        "fit": fit,
    }


def _window_day_set(window: dict[str, Any]) -> set[int]:
    """Circular day set, including unwrapped windows ending after day 365."""
    return window_day_set(window)


def bootstrap_window_uncertainty(series: pd.Series, regime: dict[str, Any], config: HighWaterConfig) -> list[dict[str, Any]]:
    """Block-bootstrap boundary intervals with one-to-one circular matching.

    Intervals describe uncertainty in the CLIMATOLOGICAL boundary estimator,
    not interannual onset variability or a forecast for a future season.
    Anchor is fixed. With bootstrap_reselect_modes=False, ONLY the complexity
    K is fixed (peaks are still screened); this conditioning is reported.
    Hydrological blocks preserve cross-calendar seasonal dependence but have
    a different year-eligibility rule. Calendar blocks use exactly the retained
    reference years and daily support threshold of the reference estimator.
    """
    refs=[dict(w) for w in regime.get('windows',[])];mrefs=[dict(w) for w in regime.get('mode_windows',[])]
    if config.n_bootstrap<=0 or not refs:return refs
    q=prepare_series(series,config)
    a,b=regime.get('reference_start'),regime.get('reference_end')
    if a is not None:q=q.loc[pd.Timestamp(a):]
    if b is not None:q=q.loc[:pd.Timestamp(b)]
    years=regime.get('reference_years',[])
    if years:q=q[q.index.year.isin(years)]
    anchor=regime['anchor_doy']
    if config.bootstrap_block_type=='calendar_year':
        q=_remove_feb29(q);rows=[];kept=[]
        for y in sorted(q.index.year.unique()):
            idx=pd.date_range(f'{y}-01-01',f'{y}-12-31');idx=idx[~((idx.month==2)&(idx.day==29))]
            v=q.reindex(idx).to_numpy()
            if np.isfinite(v).mean()>=config.min_reference_year_coverage:rows.append(_rotate(v,anchor));kept.append(int(y))
        mat=np.array(rows);block_years=np.array(kept)
    elif config.bootstrap_block_type=='hydrological_year':
        mat,block_years=_hydrological_year_matrix(q,anchor,config)
    else:raise ValueError("bootstrap_block_type must be 'calendar_year' or 'hydrological_year'.")
    if len(mat)<config.min_reference_years:
        for w in refs+mrefs:w.update(bootstrap_status='INSUFFICIENT_BLOCKS',bootstrap_match_fraction=np.nan)
        regime['mode_windows']=mrefs;return refs
    rng=np.random.default_rng(config.random_state);n=len(mat);samples=[[] for _ in refs];msamples=[[] for _ in mrefs]
    counts=[];success=0;fixed=None if config.bootstrap_reselect_modes else regime['n_modes']
    for _ in range(config.n_bootstrap):
        block=mat[rng.integers(0,n,size=n)]
        count=np.isfinite(block).sum(axis=0);total=np.nansum(block,axis=0)
        qb=np.divide(total,count,out=np.full(365,np.nan),where=count>0)
        qb[count/n<config.min_climatology_year_fraction]=np.nan
        try:boot=_fit_from_rotated_climatology(qb,anchor,config,fixed_n_modes=fixed)
        except ValueError:continue
        success+=1;counts.append(boot['mode_info']['n_modes'])
        for reference,candidates,store in [(refs,boot['windows'],samples),(mrefs,boot.get('mode_windows',[]),msamples)]:
            for i,j,overlap in match_windows(reference,candidates,min_overlap=config.bootstrap_min_overlap):
                c,r=candidates[j],reference[i]
                ds=(c['start_doy']-r['start_doy']+182.5)%365-182.5
                de=(c['end_doy']-r['end_doy']+182.5)%365-182.5
                store[i].append((ds,de))
    alpha=(1-config.bootstrap_ci)/2
    frequency={int(v):int(counts.count(v)) for v in set(counts)}
    for reference,store in [(refs,samples),(mrefs,msamples)]:
        for i,w in enumerate(reference):
            w.update(bootstrap_status='OK' if store[i] else 'NO_MATCH',bootstrap_match_fraction=len(store[i])/config.n_bootstrap,
                     bootstrap_success_fraction=success/config.n_bootstrap,bootstrap_block_type=config.bootstrap_block_type,
                     bootstrap_n_blocks=n,bootstrap_block_years=json.dumps(block_years.tolist()),
                     bootstrap_conditioning='fixed anchor'+('; fixed K, peaks re-screened' if fixed is not None else '; K reselected'),
                     bootstrap_mode_count_frequency=json.dumps(frequency))
            if not store[i]:continue
            low,high=np.quantile(np.asarray(store[i]),[alpha,1-alpha],axis=0)
            for dim,k in [('start',0),('end',1)]:
                lo=int(w[f'{dim}_pos'])+int(np.floor(low[k]));hi=int(w[f'{dim}_pos'])+int(np.ceil(high[k]))
                w[f'{dim}_pos_ci_low']=lo;w[f'{dim}_pos_ci_high']=hi
                w[f'{dim}_ci_low_month_day']=_doy_to_monthday(_rotpos_to_doy(lo,anchor))
                w[f'{dim}_ci_high_month_day']=_doy_to_monthday(_rotpos_to_doy(hi,anchor))
    regime['mode_windows']=mrefs
    return refs


# =============================================================================
# 12. Sensitivity analysis of methodological choices
# =============================================================================


def sensitivity_analysis_station(
    series: pd.Series,
    base_config: HighWaterConfig,
    min_segment_lengths: Iterable[int] = (10, 15, 20, 25, 30),
    smoothing_windows: Iterable[int] = (5, 7, 15),
    highwater_parde_thresholds: Iterable[float] = (0.9, 1.0, 1.1),
    reference_start: Optional[str | pd.Timestamp] = None,
    reference_end: Optional[str | pd.Timestamp] = None,
) -> pd.DataFrame:
    """Evaluate robustness of detected seasons to key tuning parameters.

    This function is intended for the methodological sensitivity table/figure
    of a scientific analysis.  Bootstrap is disabled inside the grid because
    structural sensitivity and sampling uncertainty answer different questions.

    Interpretation
    --------------
    Stable start/end dates across this grid strengthen confidence that the
    detected high-water season is a property of the river regime rather than a
    consequence of one arbitrary smoothing, minimum-duration, or Pardé setting.
    """
    from dataclasses import replace

    records: list[dict[str, Any]] = []
    for smooth in smoothing_windows:
        if int(smooth) % 2 == 0:
            raise ValueError("All sensitivity smoothing windows must be odd.")
        for minseg in min_segment_lengths:
            for threshold in highwater_parde_thresholds:
                cfg = replace(
                    base_config,
                    smoothing_window_days=int(smooth),
                    min_segment_length_days=int(minseg),
                    highwater_parde_threshold=float(threshold),
                    n_bootstrap=0,
                )
                try:
                    reg = fit_station_highwater_regime(
                        series,
                        cfg,
                        reference_start=reference_start,
                        reference_end=reference_end,
                    )
                    for w in reg["windows"]:
                        records.append(
                            {
                                "smoothing_days": int(smooth),
                                "min_segment_days": int(minseg),
                                "highwater_parde_threshold": float(threshold),
                                "modality": reg["modality"],
                                "n_modes": reg["n_modes"],
                                "n_highwater_seasons": reg["n_highwater_seasons"],
                                "season_id": w["season_id"],
                                "start_pos": w["start_pos"],
                                "end_pos": w["end_pos"],
                                "start_month_day": w["start_month_day"],
                                "end_month_day": w["end_month_day"],
                                "duration_days": w["duration_days"],
                                "annual_volume_fraction": w["annual_volume_fraction"],
                            }
                        )
                except Exception as exc:
                    records.append(
                        {
                            "smoothing_days": int(smooth),
                            "min_segment_days": int(minseg),
                            "highwater_parde_threshold": float(threshold),
                            "status": f"failed: {exc}",
                        }
                    )
    return pd.DataFrame.from_records(records)


# =============================================================================
# 13. Process all stations
# =============================================================================


def process_all_stations(
    data: PictureDischargeData,
    config: HighWaterConfig,
    reference_start: Optional[str | pd.Timestamp] = None,
    reference_end: Optional[str | pd.Timestamp] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    """Fit regimes and annual predictands for every station.

    Returns
    -------
    station_summary : one row per station
    season_definitions : one row per station/high-water season
    annual_predictands : long table with station x hydrological year
    regimes : full diagnostic objects, useful for plotting
    """
    summaries = []
    seasons = []
    annual_tables = []
    regimes: dict[str, dict[str, Any]] = {}

    for station in data.discharge.columns:
        series = data.discharge[station]
        meta = data.metadata.loc[station]

        try:
            regime = fit_station_highwater_regime(
                series,
                config,
                reference_start=reference_start,
                reference_end=reference_end,
            )
            regime["windows"] = bootstrap_window_uncertainty(series, regime, config)
            regimes[station] = regime

            prepared = prepare_series(series, config)
            overall_valid = float(prepared.notna().mean()) if len(prepared) else np.nan

            summaries.append(
                {
                    "station": station,
                    "LON": meta.get("LON", np.nan),
                    "LAT": meta.get("LAT", np.nan),
                    "ELEV": meta.get("ELEV", np.nan),
                    "status": "ok",
                    "record_start": prepared.index.min(),
                    "record_end": prepared.index.max(),
                    "overall_daily_valid_fraction": overall_valid,
                    "anchor_doy": regime["anchor_doy"],
                    "anchor_month_day": regime["anchor_month_day"],
                    "modality": regime["modality"],
                    "n_modes": regime["n_modes"],
                    "n_highwater_seasons": regime["n_highwater_seasons"],
                    "n_mode_windows": regime.get("n_mode_windows", 0),
                    "annual_mean_q": regime["annual_mean_q"],
                    "center_of_mass_doy": regime["center_of_mass_doy"],
                    "center_of_mass_month_day": _doy_to_monthday(
                        int(regime["center_of_mass_doy"])
                    ) if np.isfinite(regime["center_of_mass_doy"]) else None,
                    "circular_concentration": regime["circular_concentration"],
                    "n_piecewise_segments": regime["piecewise_fit"]["n_segments"],
                    "piecewise_sse": regime["piecewise_fit"]["sse"],
                }
            )

            for w in regime["windows"]:
                row = {
                    "station": station,
                    "LON": meta.get("LON", np.nan),
                    "LAT": meta.get("LAT", np.nan),
                    "ELEV": meta.get("ELEV", np.nan),
                    "modality": regime["modality"],
                    "n_modes_total": regime["n_modes"],
                    **{
                        k: v
                        for k, v in w.items()
                        if k not in {"mode_positions", "mode_ids"}
                    },
                    "mode_positions": json.dumps(w.get("mode_positions", [])),
                    "mode_ids": json.dumps(w.get("mode_ids", [])),
                }
                seasons.append(row)

            annual = compute_annual_predictands(series, regime, config)
            annual.insert(0, "station", station)
            annual.insert(1, "LON", meta.get("LON", np.nan))
            annual.insert(2, "LAT", meta.get("LAT", np.nan))
            annual.insert(3, "ELEV", meta.get("ELEV", np.nan))
            annual_tables.append(annual)

        except Exception as exc:
            summaries.append(
                {
                    "station": station,
                    "LON": meta.get("LON", np.nan),
                    "LAT": meta.get("LAT", np.nan),
                    "ELEV": meta.get("ELEV", np.nan),
                    "status": f"failed: {exc}",
                }
            )

    station_summary = pd.DataFrame(summaries)
    season_definitions = pd.DataFrame(seasons)
    annual_predictands = (
        pd.concat(annual_tables, ignore_index=True) if annual_tables else pd.DataFrame()
    )
    return station_summary, season_definitions, annual_predictands, regimes


# =============================================================================
# 14. WASS2S/CPT-like annual target export
# =============================================================================


def annual_target_to_picture_format(
    annual_predictands: pd.DataFrame,
    metadata: pd.DataFrame,
    target_column: str,
    date_month_day: str = "04-01",
) -> pd.DataFrame:
    """Convert one annual target to the user's picture/CPT-like wide format.

    Each hydrological year is represented by YYYYMMDD. The default 04-01 is only
    a conventional annual timestamp; it does NOT redefine the detected season.
    Change it if WASS2S expects another target timestamp.
    """
    if target_column not in annual_predictands.columns:
        raise KeyError(f"Target column '{target_column}' not found.")

    stations = [s for s in metadata.index if s in annual_predictands["station"].unique()]
    rows = []

    rows.append(["LON"] + [metadata.loc[s, "LON"] for s in stations])
    rows.append(["LAT"] + [metadata.loc[s, "LAT"] for s in stations])
    rows.append(["DAILY/ELEV"] + [metadata.loc[s, "ELEV"] for s in stations])

    pivot = annual_predictands.pivot(index="hydro_year", columns="station", values=target_column)
    pivot = pivot.reindex(columns=stations)

    mm, dd = [int(x) for x in date_month_day.split("-")]
    for year, vals in pivot.sort_index().iterrows():
        date_id = f"{int(year):04d}{mm:02d}{dd:02d}"
        rows.append([date_id] + vals.tolist())

    return pd.DataFrame(rows, columns=["ID"] + stations)


# =============================================================================
# 15. Diagnostic plotting
# =============================================================================


def plot_station_diagnostic(
    station: str,
    regime: dict[str, Any],
    output_path: Optional[str | Path] = None,
):
    """Create a 3-panel scientific diagnostic figure for one station."""
    import matplotlib.pyplot as plt

    q = regime["q_rotated"]
    p = regime["parde_rotated"]
    cum = regime["cumulative_rotated"]
    fit_p = regime["piecewise_fit"]["fitted_parde"]
    fit_cum = regime["piecewise_fit"]["fitted_cumulative"]
    x = np.arange(1, 366)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(x, q, linewidth=1.7, label="Smoothed daily climatology")
    axes[0].set_ylabel("Discharge")
    axes[0].set_title(
        f"{station} — {regime['modality']}; "
        f"{regime['n_highwater_seasons']} high-water season(s); "
        f"{regime.get('n_mode_windows', 0)} mode window(s)"
    )

    axes[1].plot(x, p, linewidth=1.5, label="Daily Pardé coefficient")
    axes[1].step(x, fit_p, where="mid", linewidth=1.7, label="Piecewise slope regimes")
    axes[1].axhline(1.0, linestyle="--", linewidth=1.0, label="Annual mean (P=1)")
    axes[1].set_ylabel("Pardé coefficient")

    axes[2].plot(x, cum, linewidth=1.7, label="Normalized cumulative runoff")
    axes[2].plot(x, fit_cum, linestyle="--", linewidth=1.5, label="Continuous piecewise fit")
    axes[2].set_ylabel("Cumulative fraction")
    axes[2].set_xlabel(
        f"Hydrological day (day 1 = {regime['anchor_month_day']})"
    )

    # Highlight high-water windows on all panels using default Matplotlib style.
    for w in regime["windows"]:
        for ax in axes:
            ax.axvspan(w["start_pos"], w["end_pos"], alpha=0.12)
        axes[0].axvline(w["start_pos"], linestyle=":", linewidth=0.8)
        axes[0].axvline(w["end_pos"], linestyle=":", linewidth=0.8)

    for pos in regime["mode_info"]["peak_positions"]:
        axes[0].axvline(pos, linestyle="--", linewidth=0.8)

    # If mode-specific behaviour is enabled, show the valley-derived boundaries
    # and label each chronological mode. These are different from the shaded
    # high-water-season windows when one season contains several modes.
    for m in regime.get("mode_windows", []):
        axes[0].axvline(m["start_pos"], linestyle="-.", linewidth=0.7)
        axes[0].axvline(m["end_pos"], linestyle="-.", linewidth=0.7)
        axes[0].text(
            m["peak_pos"],
            q[int(m["peak_pos"]) - 1],
            m["mode_name"],
            fontsize=8,
            ha="center",
            va="bottom",
        )

    for ax in axes:
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
    return fig


# =============================================================================
# 16. Output writer and command-line interface
# =============================================================================


def write_results(
    output_dir: str | Path,
    input_data: PictureDischargeData,
    config: HighWaterConfig,
    station_summary: pd.DataFrame,
    season_definitions: pd.DataFrame,
    annual_predictands: pd.DataFrame,
    regimes: dict[str, dict[str, Any]],
    make_figures: bool = True,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    station_summary.to_csv(out / "station_regime_summary.csv", index=False)
    season_definitions.to_csv(out / "highwater_season_definitions.csv", index=False)
    annual_predictands.to_csv(out / "annual_highwater_predictands_long.csv", index=False)

    # Optional chronological MODE1/MODE2/... definitions. Keeping these in a
    # separate table avoids conflating "high-water season" with "hydrograph
    # mode" when one continuous season contains several peaks.
    if config.mode_specific_outputs:
        mode_rows: list[dict[str, Any]] = []
        for station, regime in regimes.items():
            meta = input_data.metadata.loc[station]
            for m in regime.get("mode_windows", []):
                mode_rows.append(
                    {
                        "station": station,
                        "LON": meta.get("LON", np.nan),
                        "LAT": meta.get("LAT", np.nan),
                        "ELEV": meta.get("ELEV", np.nan),
                        "modality": regime.get("modality"),
                        "n_modes_total": regime.get("n_modes"),
                        **m,
                    }
                )
        pd.DataFrame(mode_rows).to_csv(
            out / "highflow_mode_definitions.csv", index=False
        )

    with open(out / "highwater_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    # Generic main-season and all-high-water targets.
    for target in [
        "Q_HW_MEAN_MAIN",
        "Q_HW_MAX_MAIN",
        "Q_HW_MEAN_ALL",
        "Q_HW_MAX_ALL",
    ]:
        if target in annual_predictands.columns:
            pic = annual_target_to_picture_format(
                annual_predictands, input_data.metadata, target
            )
            pic.to_csv(out / f"{target}_picture.csv", index=False)

    # Optional main/all mode-specific targets.
    if config.mode_specific_outputs:
        for target in [
            "Q_MODE_MEAN_MAIN",
            "Q_MODE_MAX_MAIN",
            "Q_MODE_MEAN_ALL",
            "Q_MODE_MAX_ALL",
        ]:
            if target in annual_predictands.columns:
                pic = annual_target_to_picture_format(
                    annual_predictands, input_data.metadata, target
                )
                pic.to_csv(out / f"{target}_picture.csv", index=False)

    # Also export each detected season when present in the union of station
    # outputs. Different stations may have different numbers of seasons.
    hw_mean_cols = sorted(
        c for c in annual_predictands.columns if c.startswith("HW") and c.endswith("_Q_MEAN")
    )
    hw_max_cols = sorted(
        c for c in annual_predictands.columns if c.startswith("HW") and c.endswith("_Q_MAX")
    )
    for target in hw_mean_cols + hw_max_cols:
        pic = annual_target_to_picture_format(
            annual_predictands, input_data.metadata, target
        )
        pic.to_csv(out / f"{target}_picture.csv", index=False)

    # Export chronological MODE1_Q_MEAN/MODE1_Q_MAX/... when requested.
    if config.mode_specific_outputs:
        mode_mean_cols = sorted(
            c
            for c in annual_predictands.columns
            if c.startswith("MODE") and c.endswith("_Q_MEAN")
        )
        mode_max_cols = sorted(
            c
            for c in annual_predictands.columns
            if c.startswith("MODE") and c.endswith("_Q_MAX")
        )
        for target in mode_mean_cols + mode_max_cols:
            pic = annual_target_to_picture_format(
                annual_predictands, input_data.metadata, target
            )
            pic.to_csv(out / f"{target}_picture.csv", index=False)

    if make_figures:
        figdir = out / "diagnostic_figures"
        for station, regime in regimes.items():
            safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in station)
            plot_station_diagnostic(station, regime, figdir / f"{safe}.png")


__all__ = [
    'HighWaterConfig',
    'PictureDischargeData',
    'read_picture_discharge_csv',
    'prepare_series',
    'build_daily_climatology',
    'choose_hydrological_anchor',
    'circular_seasonality_metrics',
    'detect_climatological_modes',
    'fit_piecewise_constant_slopes',
    'fit_station_highwater_regime',
    'compute_annual_predictands',
    'bootstrap_window_uncertainty',
    'sensitivity_analysis_station',
    'process_all_stations',
    'annual_target_to_picture_format',
    'plot_station_diagnostic',
    'write_results',
]
