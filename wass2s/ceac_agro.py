"""
CEAC agroclimatic indicators -- corrected, memory-safe, Numba-accelerated.

Key design choices
------------------
1. Gridded calculations are performed year by year; no multi-decadal shifted
   (T, Y, X) cube is built.
2. Zones 1--5 use a Jan-01 analysis anchor.
3. Zones 6--9 use a Jul-15 analysis anchor. This is deliberately earlier than
   the old Aug-01 anchor because zone 7 legitimately starts its onset search on
   Jul-15. Dates earlier than the anchor in calendar order are mapped to the
   following civil year.
4. Gridded pixel loops are compiled with Numba and parallelised with prange.
5. Onset / cessation are stored as *extended day-of-year* relative to Jan-01
   of the agricultural start year. Therefore dates in the next civil year are
   > 365/366. No arbitrary SHIFT_OFFSET is used.
6. If an onset/cessation criterion is not met inside its search window, NaN is
   returned by default. No random date is fabricated.

Public class names and principal method signatures are kept compatible with
previous WASS2S CEAC code.
"""

from __future__ import annotations

import datetime as dt
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import xarray as xr

try:
    import numba
    from numba import njit, prange
    HAS_NUMBA = True
except Exception:  # pragma: no cover
    numba = None
    HAS_NUMBA = False
    prange = range

    def njit(*args, **kwargs):
        return lambda function: function


# =============================================================================
# Criteria
# =============================================================================

DEFAULT_CRITERIA = {
    1: {"start_search": "05-01", "start_search2": "09-01", "date_dry_soil": "01-01", "cumulative": 15, "number_dry_days": 15, "thrd_rain_day": 0.85, "end_search": "08-30",  "end_search2": "10-30", "nbjour": 35, "ETP": 5.0, "Cap_ret_maxi": 70},
    2: {"start_search": "03-15", "start_search2": "09-01", "date_dry_soil": "01-01", "cumulative": 20, "number_dry_days": 10, "thrd_rain_day": 0.85, "end_search": "08-01",  "end_search2": "11-01", "nbjour": 40, "ETP": 5.0, "Cap_ret_maxi": 70},
    3: {"start_search": "02-01", "start_search2": "10-01", "date_dry_soil": "01-01", "cumulative": 20, "number_dry_days": 10, "thrd_rain_day": 0.85, "end_search": "05-15", "end_search2": "12-30", "nbjour": 45, "ETP": 5.0, "Cap_ret_maxi": 70},
    4: {"start_search": "01-01", "start_search2": "11-01", "date_dry_soil": "01-01", "cumulative": 20, "number_dry_days": 7,  "thrd_rain_day": 0.85, "end_search": "04-01", "end_search2": "12-30", "nbjour": 50, "ETP": 5.0, "Cap_ret_maxi": 80},
    5: {"start_search": "01-01", "start_search2": "06-01", "date_dry_soil": "01-01", "cumulative": 25, "number_dry_days": 7,  "thrd_rain_day": 0.85, "end_search": "03-10", "end_search2": "08-10", "nbjour": 50, "ETP": 5.0, "Cap_ret_maxi": 60},
    6: {"start_search": "08-01", "start_search2": "04-20", "date_dry_soil": "08-01", "cumulative": 25, "number_dry_days": 7,  "thrd_rain_day": 0.85, "end_search": "10-18", "end_search2": "07-15", "nbjour": 50, "ETP": 5.0, "Cap_ret_maxi": 70},
    7: {"start_search": "07-15", "start_search2": "03-01", "date_dry_soil": "08-01", "cumulative": 20, "number_dry_days": 10, "thrd_rain_day": 0.85, "end_search": "10-20", "end_search2": "06-20", "nbjour": 50, "ETP": 5.0, "Cap_ret_maxi": 70},
    8: {"start_search": "09-01", "start_search2": "03-01", "date_dry_soil": "08-01", "cumulative": 20, "number_dry_days": 10, "thrd_rain_day": 0.85, "end_search": "12-01", "end_search2": "05-15", "nbjour": 40, "ETP": 5.0, "Cap_ret_maxi": 70},
    9: {"start_search": "10-01", "start_search2": "03-15", "date_dry_soil": "08-01", "cumulative": 20, "number_dry_days": 15, "thrd_rain_day": 0.85, "end_search": "01-31", "end_search2": "05-15", "nbjour": 30, "ETP": 5.0, "Cap_ret_maxi": 60},
}

# Analysis anchors.  The south anchor is Jul-15 because zone 7 starts its onset
# search on Jul-15.  This makes all zone 6--9 search windows chronological.
NORTH_ANCHOR = "01-01"
SOUTH_ANCHOR = "07-15"

_DATE_KEYS = ("start_search", "end_search", "date_dry_soil", "start_search2", "end_search2")


# =============================================================================
# Numba kernels
# =============================================================================

def _onset_1d_reference(x, start_idx, cumulative, dry_len, dry_threshold, end_idx):
    """First rainy day of a 1--3 day event, confirmed through d + 30.

    A missing/invalid earlier candidate censors the first-event date. This is
    a retrospective indicator: the onset is not available operationally until
    thirty subsequent daily observations are available.
    """
    if not (np.isfinite(start_idx) and np.isfinite(end_idx) and
            np.isfinite(cumulative) and np.isfinite(dry_len) and
            np.isfinite(dry_threshold)):
        return np.nan
    n = x.shape[0]
    s, e = int(start_idx), int(end_idx)
    if (s != start_idx or e != end_idx or s < 0 or e < s or s >= n or
            cumulative <= 0 or dry_len < 1 or dry_len != int(dry_len) or
            dry_threshold <= 0):
        return np.nan
    # An observed onset before a truncated search end remains identifiable;
    # a candidate itself always needs all thirty confirming days.
    e = min(e, n - 1)
    for d in range(s, e + 1):
        if not np.isfinite(x[d]) or x[d] < 0:
            return np.nan
        if x[d] < dry_threshold:
            continue
        total = 0.0
        qualifies = False
        for kk in range(3):
            j = d + kk
            if j >= n or not np.isfinite(x[j]) or x[j] < 0:
                return np.nan
            total += x[j]
            if total >= cumulative:
                qualifies = True
                break
        if not qualifies:
            continue
        if d + 30 >= n:
            return np.nan
        run = 0
        false_start = False
        for j in range(d + 1, d + 31):
            if not np.isfinite(x[j]) or x[j] < 0:
                return np.nan
            if x[j] < dry_threshold:
                run += 1
                if run >= dry_len:
                    false_start = True
            else:
                run = 0
        if not false_start:
            return float(d)
    return np.nan


def _cessation_1d_reference(x, dry_soil_idx, search_start_idx, etp, capacity,
                           search_end_idx, fallback_to_end):
    """First zero of R[t] = clip(R[t-1] + P[t] - ETP, 0, capacity).

    R[dry_soil_idx-1] = 0; both search endpoints are inclusive. Missing forcing
    before the first zero censors the result. Fallback is permitted only when
    the requested end date was observed and the whole balance was known.
    """
    if not (np.isfinite(dry_soil_idx) and np.isfinite(search_start_idx) and
            np.isfinite(search_end_idx) and np.isfinite(etp) and np.isfinite(capacity)):
        return np.nan
    d0, s, e = int(dry_soil_idx), int(search_start_idx), int(search_end_idx)
    n = x.shape[0]
    if (d0 != dry_soil_idx or s != search_start_idx or e != search_end_idx or
            d0 < 0 or s < d0 or e < s or s >= n or etp < 0 or capacity <= 0):
        return np.nan
    ru = 0.0
    for k in range(d0, min(e + 1, n)):
        v = x[k]
        if not np.isfinite(v) or v < 0:
            return np.nan
        ru = max(0.0, min(capacity, ru + v - etp))
        if k >= s and ru <= 0:
            return float(k)
    if fallback_to_end and e < n:
        return float(e)
    return np.nan


def _longest_dry_run_reference(x, start, stop, threshold):
    """Exact longest run on [start, stop); an incomplete interval is unknown."""
    n = x.shape[0]
    if (not np.isfinite(start) or not np.isfinite(stop) or not np.isfinite(threshold) or
            start != int(start) or stop != int(stop) or
            start < 0 or stop > n or stop <= start or threshold <= 0):
        return np.nan
    run = best = 0
    for k in range(int(start), int(stop)):
        v = x[k]
        if not np.isfinite(v) or v < 0:
            return np.nan
        if v < threshold:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return float(best)


def _count_spells_reference(x, onset, cessation, min_len, threshold, wet):
    """Count maximal runs of >= min_len days on [onset, cessation], inclusive."""
    if not (np.isfinite(onset) and np.isfinite(cessation) and np.isfinite(min_len)
            and np.isfinite(threshold)):
        return np.nan
    o, c = int(onset), int(cessation)
    if (o != onset or c != cessation or o < 0 or c < o or c >= x.shape[0] or
            min_len < 1 or min_len != int(min_len) or threshold <= 0):
        return np.nan
    count = run = 0
    for k in range(o, c + 1):
        v = x[k]
        if not np.isfinite(v) or v < 0:
            return np.nan
        meets = v >= threshold if wet else v < threshold
        if meets:
            run += 1
        else:
            if run >= min_len:
                count += 1
            run = 0
    if run >= min_len:
        count += 1
    return float(count)


# The exact same scalar source powers the Python reference and JIT backend.
_onset_1d_numba = njit(cache=True)(_onset_1d_reference)
_cessation_1d_numba = njit(cache=True)(_cessation_1d_reference)
_longest_dry_run_numba = njit(cache=True)(_longest_dry_run_reference)
_count_spells_numba = njit(cache=True)(_count_spells_reference)


@njit(parallel=True, cache=True)
def _onset_block_numba(data2d, start_idx, cumulative, dry_len, dry_thr, end_idx):
    out = np.full(data2d.shape[0], np.nan, dtype=np.float64)
    for p in prange(data2d.shape[0]):
        out[p] = _onset_1d_numba(data2d[p], start_idx[p], cumulative[p],
                                dry_len[p], dry_thr[p], end_idx[p])
    return out


@njit(parallel=True, cache=True)
def _cessation_block_numba(data2d, dry0, start2, etp, cap, end2, fallback_to_end):
    out = np.full(data2d.shape[0], np.nan, dtype=np.float64)
    for p in prange(data2d.shape[0]):
        out[p] = _cessation_1d_numba(data2d[p], dry0[p], start2[p], etp[p], cap[p], end2[p], fallback_to_end)
    return out


@njit(parallel=True, cache=True)
def _onset_dryspell_block_numba(data2d, start_idx, cumulative, dry_len, dry_thr, end_idx, nbjour):
    out = np.full(data2d.shape[0], np.nan, dtype=np.float64)
    for p in prange(data2d.shape[0]):
        onset = _onset_1d_numba(data2d[p], start_idx[p], cumulative[p], dry_len[p], dry_thr[p], end_idx[p])
        if np.isfinite(onset) and np.isfinite(nbjour[p]) and nbjour[p] >= 1 and nbjour[p] == int(nbjour[p]):
            out[p] = _longest_dry_run_numba(data2d[p], onset + 1, onset + 1 + nbjour[p], dry_thr[p])
    return out


@njit(parallel=True, cache=True)
def _cessation_dryspell_block_numba(data2d, start1, cumulative, dry_len, dry_thr, end1,
                                  start2, dry0, etp, cap, end2, nbjour, fallback_to_end):
    out = np.full(data2d.shape[0], np.nan, dtype=np.float64)
    for p in prange(data2d.shape[0]):
        onset = _onset_1d_numba(data2d[p], start1[p], cumulative[p], dry_len[p], dry_thr[p], end1[p])
        cessation = _cessation_1d_numba(data2d[p], dry0[p], start2[p], etp[p], cap[p], end2[p], fallback_to_end)
        if np.isfinite(onset) and np.isfinite(cessation) and np.isfinite(nbjour[p]) and nbjour[p] >= 1 and nbjour[p] == int(nbjour[p]):
            out[p] = _longest_dry_run_numba(data2d[p], onset + nbjour[p], cessation, dry_thr[p])
    return out


@njit(parallel=True, cache=True)
def _count_dry_block_numba(data2d, onset_idx, cessation_idx, min_len, threshold):
    out = np.full(data2d.shape[0], np.nan, dtype=np.float64)
    for p in prange(data2d.shape[0]):
        out[p] = _count_spells_numba(data2d[p], onset_idx[p], cessation_idx[p], min_len, threshold, False)
    return out


@njit(parallel=True, cache=True)
def _count_wet_block_numba(data2d, onset_idx, cessation_idx, min_len, threshold):
    out = np.full(data2d.shape[0], np.nan, dtype=np.float64)
    for p in prange(data2d.shape[0]):
        out[p] = _count_spells_numba(data2d[p], onset_idx[p], cessation_idx[p], min_len, threshold, True)
    return out


# =============================================================================
# Base class
# =============================================================================

class CAF_AgroClimateBase:
    """Shared helpers for retrospective CEAC station and gridded indicators.

    Rainfall and ETP must be daily depths in mm, and soil capacity in mm.
    ``no_event_policy='end'`` is an explicit right-censoring convention for
    cessation; it never fills dates lost because rainfall is unknown.
    """

    # Kept only so old external code importing the attribute does not crash.
    # It is NOT used by the corrected implementation.
    SHIFT_OFFSET = 244

    def __init__(self, user_criteria=None, no_event_policy="nan"):
        self.criteria = deepcopy(user_criteria if user_criteria is not None else DEFAULT_CRITERIA)
        if no_event_policy not in ("nan", "end"):
            raise ValueError("no_event_policy must be 'nan' or 'end'.")
        self.no_event_policy = no_event_policy
        self.validate_criteria()

    # ------------------------------------------------------------------
    # Calendar / zone handling
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_mmdd(mmdd):
        m, d = map(int, str(mmdd).split("-"))
        return m, d

    def _is_shifted(self, z):
        return not pd.isna(z) and int(z) >= 6

    def analysis_anchor(self, z):
        return SOUTH_ANCHOR if self._is_shifted(z) else NORTH_ANCHOR

    def _anchor_date(self, year, z):
        m, d = self._parse_mmdd(self.analysis_anchor(z))
        return dt.date(int(year), m, d)

    def _resolved_criterion_dates(self, year, z):
        """Resolve all date criteria into a chronological agricultural-year timeline."""
        year = int(year)
        z = int(z)
        c = self.criteria[z]
        anchor = self._anchor_date(year, z)

        def base(mmdd):
            m, d = self._parse_mmdd(mmdd)
            candidate = dt.date(year, m, d)
            if candidate < anchor:
                candidate = dt.date(year + 1, m, d)
            return candidate

        # Onset window: start is the first possible onset date; end must not
        # precede it, so roll end into the next civil year when required.
        start1 = base(c["start_search"])
        end1 = base(c["end_search"])
        if end1 < start1:
            end1 = dt.date(end1.year + 1, end1.month, end1.day)

        # Soil-water/cessation phase: dry-soil date is the phase origin.
        # start_search2 and end_search2 are rolled forward until chronology is
        # respected. This correctly maps zone-6 end_search2=Jul-15 to the NEXT
        # Jul-15, while zone-7 start_search=Jul-15 stays on the CURRENT Jul-15.
        dry0 = base(c["date_dry_soil"])
        start2 = base(c["start_search2"])
        if start2 < dry0:
            start2 = dt.date(start2.year + 1, start2.month, start2.day)
        end2 = base(c["end_search2"])
        while end2 < start2:
            end2 = dt.date(end2.year + 1, end2.month, end2.day)

        return {
            "start_search": start1,
            "end_search": end1,
            "date_dry_soil": dry0,
            "start_search2": start2,
            "end_search2": end2,
        }

    def _criterion_date(self, year, mmdd, z, key=None):
        """Resolve a criterion date; ``key`` is recommended for unambiguous use."""
        resolved = self._resolved_criterion_dates(year, z)
        if key is not None:
            return resolved[key]

        # Backward-compatible lookup by the mm-dd value used by old callers.
        matches = [k for k in _DATE_KEYS if self.criteria[int(z)][k] == mmdd]
        if not matches:
            raise KeyError(f"{mmdd!r} is not a date criterion for zone {z}")
        dates = {resolved[k] for k in matches}
        if len(dates) != 1:
            raise ValueError(
                f"Ambiguous criterion date {mmdd!r} for zone {z}; pass key explicitly"
            )
        return dates.pop()

    def get_index_for_station(self, year, mm_dd, z, key=None):
        """0-based index of a real criterion date in the zone analysis window."""
        return (self._criterion_date(year, mm_dd, z, key=key) - self._anchor_date(year, z)).days

    @staticmethod
    def day_of_year(y, mm_dd):
        m, d = map(int, str(mm_dd).split("-"))
        date = dt.date(int(y), m, d)
        return (date - dt.date(int(y), 1, 1)).days + 1

    def season_index_to_extended_doy(self, index, year, z):
        if not np.isfinite(index):
            return np.nan
        actual = self._anchor_date(year, z) + dt.timedelta(days=int(round(float(index))))
        return float((actual - dt.date(int(year), 1, 1)).days + 1)

    def extended_doy_to_season_index(self, value, year, z):
        if not np.isfinite(value):
            return np.nan
        actual = dt.date(int(year), 1, 1) + dt.timedelta(days=int(round(float(value))) - 1)
        return float((actual - self._anchor_date(year, z)).days)

    @staticmethod
    def extended_doy_to_date(value, year):
        if not np.isfinite(value):
            return pd.NaT
        return pd.Timestamp(dt.date(int(year), 1, 1) + dt.timedelta(days=int(round(float(value))) - 1))

    def output_format_value(self, v, z, year=None):
        """
        Convert seasonal index to extended day-of-year.

        ``year`` is required in the corrected implementation because arbitrary
        offsets are not valid across leap years or across zones.
        """
        if pd.isna(v):
            return np.nan
        if year is None:
            raise ValueError("output_format_value(...): year is required in corrected CEAC module.")
        return self.season_index_to_extended_doy(v, year, z)

    def revert_to_index(self, v, z, year=None):
        if pd.isna(v):
            return np.nan
        if year is None:
            raise ValueError("revert_to_index(...): year is required in corrected CEAC module.")
        return int(round(self.extended_doy_to_season_index(v, year, z)))

    def validate_criteria(self):
        """Fail fast if any zone has impossible onset/cessation chronology."""
        errors = []
        dummy_year = 2001  # non-leap reference is enough for ordering checks
        required = set(_DATE_KEYS) | {"cumulative", "number_dry_days", "thrd_rain_day", "nbjour", "ETP", "Cap_ret_maxi"}

        for z, c in sorted(self.criteria.items()):
            if not isinstance(z, (int, np.integer)) or not 1 <= z <= 9:
                errors.append(f"zone {z}: expected an integer zone in 1..9")
                continue
            missing = required - set(c)
            if missing:
                errors.append(f"zone {z}: missing keys {sorted(missing)}")
                continue
            try:
                s1 = self.get_index_for_station(dummy_year, c["start_search"], z, key="start_search")
                e1 = self.get_index_for_station(dummy_year, c["end_search"], z, key="end_search")
                d0 = self.get_index_for_station(dummy_year, c["date_dry_soil"], z, key="date_dry_soil")
                s2 = self.get_index_for_station(dummy_year, c["start_search2"], z, key="start_search2")
                e2 = self.get_index_for_station(dummy_year, c["end_search2"], z, key="end_search2")
            except Exception as exc:
                errors.append(f"zone {z}: invalid date criterion ({exc})")
                continue

            if s1 > e1:
                errors.append(f"zone {z}: start_search index {s1} > end_search index {e1}")
            if d0 > s2:
                errors.append(f"zone {z}: date_dry_soil index {d0} > start_search2 index {s2}")
            if s2 > e2:
                errors.append(f"zone {z}: start_search2 index {s2} > end_search2 index {e2}")
            numeric = ("cumulative", "number_dry_days", "nbjour", "Cap_ret_maxi", "ETP", "thrd_rain_day")
            try:
                if not all(np.isfinite(c[key]) for key in numeric):
                    raise ValueError("numeric criteria must be finite")
                if any(c[key] <= 0 for key in numeric if key != "ETP") or c["ETP"] < 0:
                    raise ValueError("numeric criteria must be positive (ETP may be zero)")
                if any(c[key] != int(c[key]) for key in ("number_dry_days", "nbjour")):
                    raise ValueError("number_dry_days and nbjour must be whole days")
            except (TypeError, ValueError) as exc:
                errors.append(f"zone {z}: {exc}")

        if errors:
            raise ValueError("Invalid CEAC criteria:\n  - " + "\n  - ".join(errors))
        return True

    # ------------------------------------------------------------------
    # Grid preparation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_daily_grid(daily_data):
        if not isinstance(daily_data, xr.DataArray):
            raise TypeError("daily_data must be an xarray.DataArray")
        if set(daily_data.dims) != {"T", "Y", "X"}:
            raise ValueError("daily_data must have exactly T, Y and X dimensions")
        if not np.issubdtype(daily_data["T"].dtype, np.datetime64):
            raise TypeError("CEAC currently requires a Gregorian datetime64 daily calendar")
        dates = pd.DatetimeIndex(daily_data["T"].values)
        if not len(dates) or dates.hasnans or not dates.is_unique:
            raise ValueError("daily T coordinates must be nonempty, finite and unique")
        if not dates.equals(dates.normalize()):
            raise ValueError("daily T coordinates must be midnight dates; aggregate subdaily data first")
        for dim in ("Y", "X"):
            values = np.asarray(daily_data[dim].values, dtype=float)
            if not values.size or not np.isfinite(values).all() or len(np.unique(values)) != len(values):
                raise ValueError(f"{dim} coordinates must be nonempty, finite and unique")
        return daily_data.transpose("T", "Y", "X").sortby("T")

    @staticmethod
    def _spatial_grid(field):
        if not isinstance(field, xr.DataArray):
            raise TypeError("spatial field must be an xarray.DataArray")
        extra = set(field.dims) - {"Y", "X"}
        for dim in extra:
            if field.sizes[dim] != 1:
                raise ValueError("spatial field must have Y and X plus singleton extra dimensions")
            field = field.isel({dim: 0}, drop=True)
        if set(field.dims) != {"Y", "X"}:
            raise ValueError("spatial field must retain Y and X dimensions, including size-one axes")
        for dim in ("Y", "X"):
            vals = np.asarray(field[dim].values, dtype=float)
            if not vals.size or not np.isfinite(vals).all() or np.unique(vals).size != vals.size:
                raise ValueError(f"spatial field has invalid {dim} coordinates")
        return field.transpose("Y", "X").sortby(["Y", "X"])

    @classmethod
    def _align_spatial(cls, field, template):
        field = cls._spatial_grid(field)
        out = field.reindex(Y=template.Y, X=template.X, method="nearest")
        # Nearest categorical sampling is allowed only inside the source extent.
        inside_y = (template.Y >= field.Y.min()) & (template.Y <= field.Y.max())
        inside_x = (template.X >= field.X.min()) & (template.X <= field.X.max())
        return out.where(inside_y & inside_x).transpose("Y", "X")

    def _prepare_mask(self, daily_data, map_reclassified):
        mask = self._align_spatial(map_reclassified, daily_data.isel(T=0, drop=True))
        vals = np.asarray(mask.values, dtype=float)
        valid = np.isfinite(vals) & np.isin(vals, np.asarray(sorted(self.criteria), dtype=float))
        if not np.any(valid):
            raise ValueError("No valid CEAC zones overlap the daily grid")
        return xr.DataArray(np.where(valid, vals, np.nan), coords=mask.coords,
                            dims=mask.dims, name="zone")

    def _window_bounds(self, year, shifted):
        year = int(year)
        anchor = SOUTH_ANCHOR if shifted else NORTH_ANCHOR
        m, d = self._parse_mmdd(anchor)
        start = pd.Timestamp(year=year, month=m, day=d)
        end = pd.Timestamp(year=year + 1, month=m, day=d) if shifted else pd.Timestamp(year=year, month=12, day=31)
        # Custom late onset windows need their whole retrospective confirmation
        # and dry-spell interval, even across the civil-year boundary.
        for z, criteria in self.criteria.items():
            if self._is_shifted(z) == bool(shifted):
                dates = self._resolved_criterion_dates(year, z)
                end = max(end, pd.Timestamp(dates["end_search2"]),
                          pd.Timestamp(dates["end_search"]) + pd.Timedelta(days=max(30, int(criteria["nbjour"]))))
        return start, end

    def _group_available(self, daily_data, year, shifted):
        start, end = self._window_bounds(year, shifted)
        dates = pd.DatetimeIndex(daily_data["T"].values)
        return start >= dates[0] and end <= dates[-1]

    def _valid_years(self, daily_data, mask):
        dates = pd.DatetimeIndex(daily_data["T"].values)
        groups = [shifted for shifted in (False, True)
                  if np.any(self._zone_group_mask_flat(mask, shifted))]
        years = [year for year in range(dates[0].year - 1, dates[-1].year + 1)
                 if any(self._group_available(daily_data, year, shifted) for shifted in groups)]
        if not years:
            raise ValueError("No complete CEAC agricultural year in daily_data")
        return np.asarray(years, dtype=int)

    @staticmethod
    def _configure_numba_threads(nb_cores):
        if not np.isfinite(nb_cores) or nb_cores < 1 or nb_cores != int(nb_cores):
            raise ValueError("nb_cores must be a positive integer")
        if not HAS_NUMBA:
            return 1
        n = min(int(nb_cores), int(numba.config.NUMBA_NUM_THREADS))
        numba.set_num_threads(n)
        return n

    def _load_window_2d(self, daily_data, start, end):
        """Load one analysis window as contiguous (ncell, T) float64 array."""
        expected = pd.date_range(start, end, freq="D")
        sub = daily_data.sel(T=slice(start, end))
        dates = pd.DatetimeIndex(sub["T"].values)
        if len(dates) != len(expected) or not dates.equals(expected):
            sub = sub.reindex(T=expected)

        # Dask (if present) is used only here for lazy IO, not for pixel kernels.
        data = sub.transpose("Y", "X", "T").data
        if hasattr(data, "compute"):
            data = data.compute()
        arr3 = np.ascontiguousarray(np.asarray(data, dtype=np.float64))
        ny, nx, nt = arr3.shape
        return arr3.reshape(ny * nx, nt), ny, nx

    def _parameter_flat(self, mask, year, key, shifted):
        zflat = np.asarray(mask.values).reshape(-1)
        out = np.full(zflat.size, np.nan, dtype=np.float64)
        for z in self.criteria:
            if (z >= 6) != bool(shifted):
                continue
            sel = zflat == z
            if not np.any(sel):
                continue
            val = self.criteria[z][key]
            if key in _DATE_KEYS:
                val = self.get_index_for_station(year, val, z, key=key)
            out[sel] = float(val)
        return out

    def _zone_group_mask_flat(self, mask, shifted):
        z = np.asarray(mask.values).reshape(-1)
        if shifted:
            return np.isfinite(z) & (z >= 6)
        return np.isfinite(z) & (z <= 5)

    def _date_status_flat(self, data2d, result, mask, year, shifted, kind):
        """Observed=1, complete/no event=0, unknown=-1, unavailable zone/year=-2."""
        status = np.full(result.shape, -2, dtype=np.int8)
        zones = np.asarray(mask.values).reshape(-1)
        for z, criteria in self.criteria.items():
            if self._is_shifted(z) != bool(shifted):
                continue
            selected = zones == z
            status[selected & np.isfinite(result)] = 1
            unresolved = selected & ~np.isfinite(result)
            if not np.any(unresolved):
                continue
            first = "start_search" if kind == "onset" else "date_dry_soil"
            last = "end_search" if kind == "onset" else "end_search2"
            a = self.get_index_for_station(year, criteria[first], z, key=first)
            b = self.get_index_for_station(year, criteria[last], z, key=last) + (30 if kind == "onset" else 0)
            status[unresolved] = -1
            if 0 <= a <= b < data2d.shape[1]:
                rain = data2d[unresolved, a:b + 1]
                complete = np.all(np.isfinite(rain) & (rain >= 0), axis=1)
                status[np.flatnonzero(unresolved)[complete]] = 0
        return status

    @staticmethod
    def _with_date_status(result, status):
        result = result.assign_coords(event_status=(("T", "Y", "X"), status[None, :, :]))
        result.event_status.attrs.update({
            "flag_values": [-2, -1, 0, 1],
            "flag_meanings": "unavailable_zone_year unknown_forcing no_event observed_event",
            "description": "0 includes an explicit end-coded right-censored cessation",
        })
        return result

    def _indices_to_extended_doy_2d(self, idx2d, mask, year):
        out = np.full(idx2d.shape, np.nan, dtype=np.float32)
        zvals = np.asarray(mask.values)
        for z in self.criteria:
            sel = zvals == z
            if not np.any(sel):
                continue
            anchor = self._anchor_date(year, z)
            base = (anchor - dt.date(int(year), 1, 1)).days
            vals = idx2d[sel]
            good = np.isfinite(vals)
            enc = np.full(vals.shape, np.nan, dtype=np.float32)
            enc[good] = base + vals[good] + 1.0
            out[sel] = enc
        return out

    def _encoded_to_indices_2d(self, encoded, mask, year):
        arr = np.asarray(encoded, dtype=np.float32)
        out = np.full(arr.shape, np.nan, dtype=np.float32)
        zvals = np.asarray(mask.values)
        for z in self.criteria:
            sel = zvals == z
            if not np.any(sel):
                continue
            anchor = self._anchor_date(year, z)
            base = (anchor - dt.date(int(year), 1, 1)).days
            vals = arr[sel]
            good = np.isfinite(vals)
            dec = np.full(vals.shape, np.nan, dtype=np.float32)
            dec[good] = vals[good] - base - 1.0
            out[sel] = dec
        return out

    def _result_da(self, values2d, mask, year, name):
        return xr.DataArray(
            values2d.astype(np.float32, copy=False),
            coords={"Y": mask["Y"], "X": mask["X"]},
            dims=("Y", "X"),
            name=name,
        ).expand_dims(T=[pd.Timestamp(f"{int(year)}-01-01")])

    def _finalize(self, annual, name, attrs=None):
        out = xr.concat(annual, dim="T").transpose("T", "Y", "X")
        out.name = name
        if attrs:
            out.attrs.update(attrs)
        out.attrs.update({
            "calendar": "proleptic_gregorian",
            "engine": "numba" if HAS_NUMBA else "python",
            "missing_policy": "unknown forcings censor exact events and spell statistics",
            "rainfall_units_required": "mm/day (daily accumulated depth)",
            "onset_confirmation_days": 30,
            "onset_availability": "retrospective: earliest 30 days after dated onset",
            "zone_criteria": repr(self.criteria),
        })
        return out

    def format_grid_output(self, res_xr, mask, year=None):
        """Correct replacement for the old +244 formatting."""
        if "T" in res_xr.dims:
            pieces = []
            for i in range(res_xr.sizes["T"]):
                r = res_xr.isel(T=i)
                t = pd.Timestamp(res_xr["T"].values[i])
                y = int(t.year)
                v = self._indices_to_extended_doy_2d(np.asarray(r.values), mask, y)
                pieces.append(self._result_da(v, mask, y, res_xr.name or "indicator"))
            return xr.concat(pieces, dim="T")
        if year is None:
            raise ValueError("year is required when formatting a 2-D field")
        v = self._indices_to_extended_doy_2d(np.asarray(res_xr.values), mask, int(year))
        return xr.DataArray(v, coords=mask.coords, dims=mask.dims, name=res_xr.name)

    def revert_grid_index(self, res_xr, mask, year=None):
        """Convert extended-DOY date fields back to analysis-window indices."""
        if "T" in res_xr.dims:
            pieces = []
            for i in range(res_xr.sizes["T"]):
                r = res_xr.isel(T=i)
                t = pd.Timestamp(res_xr["T"].values[i])
                y = int(t.year)
                v = self._encoded_to_indices_2d(np.asarray(r.values), mask, y)
                pieces.append(self._result_da(v, mask, y, res_xr.name or "indicator"))
            return xr.concat(pieces, dim="T")
        if year is None:
            raise ValueError("year is required when reverting a 2-D field")
        v = self._encoded_to_indices_2d(np.asarray(res_xr.values), mask, int(year))
        return xr.DataArray(v, coords=mask.coords, dims=mask.dims, name=res_xr.name)

    # ------------------------------------------------------------------
    # Compatibility shift helper (not used by compute methods)
    # ------------------------------------------------------------------

    def shift_gridded_data(self, daily_data, map_reclassified):
        warnings.warn(
            "shift_gridded_data() is deprecated. Corrected compute() methods no longer build a shifted historical cube.",
            DeprecationWarning,
            stacklevel=2,
        )
        daily_data = self._validate_daily_grid(daily_data)
        mask = self._prepare_mask(daily_data, map_reclassified)
        years = self._valid_years(daily_data, mask)
        return daily_data, mask, years

    # ------------------------------------------------------------------
    # Station table helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_cdt(df_raw):
        df = df_raw.copy()
        if "ID" in df.columns or str(df.columns[0]).upper() == "ID":
            header = pd.DataFrame([df.columns])
            header.columns = range(df.shape[1])
            df.columns = range(df.shape[1])
            df = pd.concat([header, df], ignore_index=True)
        return df

    def transform_and_shift_cdt(self, df_raw, map_reclassified):
        df = self._normalize_cdt(df_raw)
        if df.shape[0] < 5 or df.shape[1] < 2:
            raise ValueError("CDT input requires four metadata rows and daily station data")
        ids_s = pd.Series(df.iloc[0, 1:].astype(str).values)
        if ids_s.duplicated().any():
            raise ValueError("CDT station identifiers must be unique")
        ids = ids_s.values
        df.iloc[0, 1:] = ids
        lons = pd.to_numeric(df.iloc[1, 1:], errors="coerce").to_numpy(float)
        lats = pd.to_numeric(df.iloc[2, 1:], errors="coerce").to_numpy(float)
        dates0 = pd.to_datetime(df.iloc[4:, 0], format="%Y%m%d", errors="raise")
        good = ~dates0.isna()
        if not good.all():
            raise ValueError("CDT rainfall rows must have an explicit calendar date")
        dates = pd.DatetimeIndex(dates0[good])
        if not len(dates) or not dates.is_unique:
            raise ValueError("CDT daily dates must be nonempty and unique")
        vals = df.iloc[4:, 1:].loc[good].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=True)
        vals[(vals < 0) | ~np.isfinite(vals)] = np.nan
        da = xr.DataArray(vals, coords={"T": ("T", dates.to_numpy()),
                                      "station": ("station", np.asarray(ids, dtype=str))},
                          dims=("T", "station")).sortby("T")

        zmap = self._spatial_grid(map_reclassified)

        tmin = pd.Timestamp(da["T"].values[0])
        tmax = pd.Timestamp(da["T"].values[-1])
        stn_zones = {}
        frames = []

        for i, stn in enumerate(ids):
            try:
                if not (float(zmap.X.min()) <= lons[i] <= float(zmap.X.max()) and
                        float(zmap.Y.min()) <= lats[i] <= float(zmap.Y.max())):
                    raise ValueError("station lies outside zone map extent")
                z = float(zmap.sel(X=lons[i], Y=lats[i], method="nearest").values.item())
            except Exception:
                z = np.nan
            stn_zones[stn] = z
            if not np.isfinite(z) or z not in self.criteria:
                continue
            z = int(z)
            shifted = z >= 6
            series = da.isel(station=i)
            for year in range(tmin.year - 1, tmax.year + 1):
                start, end = self._window_bounds(year, shifted)
                if start < tmin or end > tmax:
                    continue
                expected = pd.date_range(start, end, freq="D")
                s = series.sel(T=slice(start, end)).reindex(T=expected)
                frames.append(pd.DataFrame({
                    "DATE": expected,
                    "STATION": stn,
                    "VALUE": np.asarray(s.values, dtype=float),
                    "LAT": lats[i], "LON": lons[i], "zonename": z, "year": year,
                }))

        long = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["DATE", "STATION", "VALUE", "LAT", "LON", "zonename", "year"]
        )
        return long, stn_zones, df.iloc[:4, :]

    def _parse_cpt_to_long(self, df_cpt, val_name):
        first = df_cpt.columns[0]
        lats = df_cpt.iloc[0, 1:].values
        lons = df_cpt.iloc[1, 1:].values
        cols = df_cpt.columns[1:].tolist()
        df = df_cpt.iloc[2:].copy().reset_index(drop=True).rename(columns={first: "year"})
        df = df.melt(id_vars=["year"], var_name="station", value_name=val_name)
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df[val_name] = pd.to_numeric(df[val_name], errors="coerce").replace([-99.0, -999.0], np.nan)
        if df.duplicated(["station", "year"]).any():
            raise ValueError("CPT input requires exactly one value per station and year")
        df["lat"] = df["station"].map(dict(zip(cols, lats)))
        df["lon"] = df["station"].map(dict(zip(cols, lons)))
        return df

    @staticmethod
    def build_cpt_output(res_df, val_col):
        if res_df.empty:
            return pd.DataFrame(columns=["STATION"])
        res_df = res_df.copy()
        res_df[val_col] = res_df[val_col].fillna(-999.0)
        piv = res_df.pivot(index="year", columns="station", values=val_col)
        meta = res_df.groupby("station")[["lat", "lon"]].first()
        lat_row = pd.DataFrame([meta.loc[piv.columns, "lat"].tolist()], columns=piv.columns, index=["LAT"])
        lon_row = pd.DataFrame([meta.loc[piv.columns, "lon"].tolist()], columns=piv.columns, index=["LON"])
        out = pd.concat([lat_row, lon_row, piv]).reset_index().rename(columns={"index": "STATION"})
        out.columns.name = None
        return out

    def _select_annual_field(self, da, mask, year):
        if "T" in da.dims:
            if np.issubdtype(da["T"].dtype, np.datetime64):
                indices = np.flatnonzero(da["T"].dt.year.values == int(year))
            else:
                indices = np.flatnonzero(np.asarray(da["T"].values).astype(str) == str(int(year)))
            if len(indices) != 1:
                raise ValueError(f"Expected one field for year {year}, found {len(indices)}")
            da = da.isel(T=int(indices[0]), drop=True)
        return self._align_spatial(da, mask)

    def diagnose_date_indicator(self, indicator, map_reclassified, kind="onset"):
        """
        Quality-control table for encoded onset/cessation fields.

        Returns, by year and climate zone, the number of valid/missing pixels,
        min/median/max decoded date and the number of values outside the zone's
        configured search window. With outputs produced by this module,
        ``n_outside_window`` should always be zero.
        """
        if kind not in ("onset", "cessation"):
            raise ValueError("kind must be 'onset' or 'cessation'")
        if "T" not in indicator.dims:
            raise ValueError("indicator must have annual T dimension")

        # Build a 2-D spatial template from the indicator itself.
        template = indicator.isel(T=0, drop=True)
        mask = self._align_spatial(map_reclassified, template)
        zvals = np.asarray(mask.values)

        rows = []
        for it in range(indicator.sizes["T"]):
            field = indicator.isel(T=it).transpose("Y", "X")
            t = pd.Timestamp(indicator["T"].values[it])
            year = int(t.year)
            vals = np.asarray(field.values, dtype=float)

            for z in sorted(self.criteria):
                sel = zvals == z
                n_total = int(np.count_nonzero(sel))
                if n_total == 0:
                    continue
                v = vals[sel]
                good = np.isfinite(v)
                vg = v[good]

                rr = self._resolved_criterion_dates(year, z)
                if kind == "onset":
                    dmin, dmax = rr["start_search"], rr["end_search"]
                else:
                    dmin, dmax = rr["start_search2"], rr["end_search2"]

                dates = [self.extended_doy_to_date(x, year) for x in vg]
                outside = sum((d.date() < dmin or d.date() > dmax) for d in dates)

                rows.append({
                    "year": year,
                    "zone": z,
                    "search_start": pd.Timestamp(dmin),
                    "search_end": pd.Timestamp(dmax),
                    "n_pixels": n_total,
                    "n_valid": int(good.sum()),
                    "n_missing": int(n_total - good.sum()),
                    "valid_fraction": float(good.mean()) if n_total else np.nan,
                    "min_date": min(dates) if dates else pd.NaT,
                    "median_date": (
                        pd.Timestamp(int(np.median(np.asarray(dates, dtype="datetime64[ns]").astype("int64"))))
                        if dates else pd.NaT
                    ),
                    "max_date": max(dates) if dates else pd.NaT,
                    "n_outside_window": int(outside),
                })

        return pd.DataFrame(rows)


# =============================================================================
# Onset
# =============================================================================

class CEAC_compute_onset(CAF_AgroClimateBase):

    @staticmethod
    def onset_function(x, idebut, cumul, nbsec, jour_pluvieux, irch_fin):
        """Python reference; 0-based search indices, inclusive search endpoints."""
        return _onset_1d_reference(np.asarray(x, dtype=float), idebut, cumul, nbsec, jour_pluvieux, irch_fin)

    def compute(self, daily_data, map_rec, nb_cores):
        daily_data = self._validate_daily_grid(daily_data)
        mask = self._prepare_mask(daily_data, map_rec)
        years = self._valid_years(daily_data, mask)
        self._configure_numba_threads(nb_cores)
        annual = []

        for year in years:
            idx2d = np.full(mask.shape, np.nan, dtype=np.float32)
            status2d = np.full(mask.shape, -2, dtype=np.int8)
            for shifted in (False, True):
                group = self._zone_group_mask_flat(mask, shifted)
                if not np.any(group) or not self._group_available(daily_data, year, shifted):
                    continue
                start, end = self._window_bounds(year, shifted)
                data2d, ny, nx = self._load_window_2d(daily_data, start, end)
                out = _onset_block_numba(
                    data2d,
                    self._parameter_flat(mask, year, "start_search", shifted),
                    self._parameter_flat(mask, year, "cumulative", shifted),
                    self._parameter_flat(mask, year, "number_dry_days", shifted),
                    self._parameter_flat(mask, year, "thrd_rain_day", shifted),
                    self._parameter_flat(mask, year, "end_search", shifted),
                )
                flat = idx2d.reshape(-1)
                flat[group] = out[group]
                statuses = self._date_status_flat(data2d, out, mask, year, shifted, "onset")
                status2d.reshape(-1)[group] = statuses[group]
                del data2d, out
            encoded = self._indices_to_extended_doy_2d(idx2d, mask, year)
            annual.append(self._with_date_status(self._result_da(encoded, mask, year, "Onset"), status2d))

        return self._finalize(annual, "Onset", {
            "units": "extended_day_of_year",
            "encoding_note": "1=Jan-01 of agricultural start year; values > days_in_year refer to next civil year",
            "south_analysis_anchor": SOUTH_ANCHOR,
            "no_event_policy": "nan",
        })

    def compute_insitu(self, daily_df_raw, map_rec):
        long, zones, _ = self.transform_and_shift_cdt(daily_df_raw, map_rec)
        rows = []
        for (stn, y), g in long.groupby(["STATION", "year"], sort=True):
            z = zones.get(stn, np.nan)
            if not np.isfinite(z) or int(z) not in self.criteria:
                v = np.nan
            else:
                z = int(z); c = self.criteria[z]
                idx = self.onset_function(g.VALUE.to_numpy(), self.get_index_for_station(y, c["start_search"], z, key="start_search"), c["cumulative"], c["number_dry_days"], c["thrd_rain_day"], self.get_index_for_station(y, c["end_search"], z, key="end_search"))
                v = self.season_index_to_extended_doy(idx, y, z)
            rows.append({"year": y, "station": stn, "lat": g.LAT.iloc[0], "lon": g.LON.iloc[0], "onset": v})
        return self.build_cpt_output(pd.DataFrame(rows), "onset")


# =============================================================================
# Cessation
# =============================================================================

class CEAC_compute_cessation(CAF_AgroClimateBase):

    @staticmethod
    def cessation_function(x, ijour, idebut, ETP, Cap, irch_fin, fallback_to_end=False):
        return _cessation_1d_reference(np.asarray(x, dtype=float), ijour, idebut, ETP, Cap, irch_fin, fallback_to_end)

    def compute(self, daily_data, map_rec, nb_cores):
        daily_data = self._validate_daily_grid(daily_data)
        mask = self._prepare_mask(daily_data, map_rec)
        years = self._valid_years(daily_data, mask)
        self._configure_numba_threads(nb_cores)
        annual=[]
        fallback = self.no_event_policy == "end"
        for year in years:
            idx2d=np.full(mask.shape,np.nan,dtype=np.float32)
            status2d = np.full(mask.shape, -2, dtype=np.int8)
            for shifted in (False,True):
                group=self._zone_group_mask_flat(mask,shifted)
                if not np.any(group) or not self._group_available(daily_data, year, shifted): continue
                start,end=self._window_bounds(year,shifted)
                data2d,ny,nx=self._load_window_2d(daily_data,start,end)
                out=_cessation_block_numba(
                    data2d,
                    self._parameter_flat(mask,year,"date_dry_soil",shifted),
                    self._parameter_flat(mask,year,"start_search2",shifted),
                    self._parameter_flat(mask,year,"ETP",shifted),
                    self._parameter_flat(mask,year,"Cap_ret_maxi",shifted),
                    self._parameter_flat(mask,year,"end_search2",shifted),
                    False,
                )
                statuses = self._date_status_flat(data2d, out, mask, year, shifted, "cessation")
                if fallback:
                    end_dates = self._parameter_flat(mask, year, "end_search2", shifted)
                    out[statuses == 0] = end_dates[statuses == 0]
                status2d.reshape(-1)[group] = statuses[group]
                idx2d.reshape(-1)[group]=out[group]
                del data2d,out
            enc=self._indices_to_extended_doy_2d(idx2d,mask,year)
            annual.append(self._with_date_status(self._result_da(enc,mask,year,"Cessation"), status2d))
        return self._finalize(annual,"Cessation",{
            "units":"extended_day_of_year","south_analysis_anchor":SOUTH_ANCHOR,"no_event_policy":self.no_event_policy,
        })

    def compute_insitu(self,daily_df_raw,map_rec):
        long,zones,_=self.transform_and_shift_cdt(daily_df_raw,map_rec); rows=[]
        for (stn,y),g in long.groupby(["STATION","year"],sort=True):
            z=zones.get(stn,np.nan)
            if not np.isfinite(z) or int(z) not in self.criteria: v=np.nan
            else:
                z=int(z); c=self.criteria[z]
                idx=self.cessation_function(g.VALUE.to_numpy(),self.get_index_for_station(y, c["date_dry_soil"], z, key="date_dry_soil"),self.get_index_for_station(y, c["start_search2"], z, key="start_search2"),c["ETP"],c["Cap_ret_maxi"],self.get_index_for_station(y, c["end_search2"], z, key="end_search2"),self.no_event_policy=="end")
                v=self.season_index_to_extended_doy(idx,y,z)
            rows.append({"year":y,"station":stn,"lat":g.LAT.iloc[0],"lon":g.LON.iloc[0],"cessation":v})
        return self.build_cpt_output(pd.DataFrame(rows),"cessation")


# =============================================================================
# Onset dry-spell length
# =============================================================================

class CEAC_compute_onset_dry_spell(CAF_AgroClimateBase):

    @staticmethod
    def ds_onset_func(x, idebut, cumul, nbsec, jp, irch_fin, nbjour):
        x = np.asarray(x, dtype=float)
        onset = _onset_1d_reference(x, idebut, cumul, nbsec, jp, irch_fin)
        if not np.isfinite(onset) or not np.isfinite(nbjour) or nbjour < 1 or nbjour != int(nbjour):
            return np.nan
        return _longest_dry_run_reference(x, onset + 1, onset + 1 + nbjour, jp)

    def compute(self,daily_data,map_rec,nb_cores):
        daily_data=self._validate_daily_grid(daily_data); mask=self._prepare_mask(daily_data,map_rec); years=self._valid_years(daily_data,mask); self._configure_numba_threads(nb_cores); annual=[]
        for year in years:
            val2d=np.full(mask.shape,np.nan,np.float32)
            for shifted in (False,True):
                group=self._zone_group_mask_flat(mask,shifted)
                if not np.any(group) or not self._group_available(daily_data, year, shifted): continue
                start,end=self._window_bounds(year,shifted); data2d,ny,nx=self._load_window_2d(daily_data,start,end)
                out=_onset_dryspell_block_numba(data2d,
                    self._parameter_flat(mask,year,"start_search",shifted),self._parameter_flat(mask,year,"cumulative",shifted),self._parameter_flat(mask,year,"number_dry_days",shifted),self._parameter_flat(mask,year,"thrd_rain_day",shifted),self._parameter_flat(mask,year,"end_search",shifted),self._parameter_flat(mask,year,"nbjour",shifted))
                val2d.reshape(-1)[group]=out[group]; del data2d,out
            annual.append(self._result_da(val2d,mask,year,"Onset_dryspell"))
        return self._finalize(annual,"Onset_dryspell",{"units":"days","definition":"longest dry spell during nbjour days after onset"})

    def compute_insitu(self,daily_df_raw,map_rec):
        long,zones,_=self.transform_and_shift_cdt(daily_df_raw,map_rec); rows=[]
        for (stn,y),g in long.groupby(["STATION","year"],sort=True):
            z=zones.get(stn,np.nan)
            if not np.isfinite(z) or int(z) not in self.criteria: v=np.nan
            else:
                z=int(z); c=self.criteria[z]
                v=self.ds_onset_func(g.VALUE.to_numpy(),self.get_index_for_station(y, c["start_search"], z, key="start_search"),c["cumulative"],c["number_dry_days"],c["thrd_rain_day"],self.get_index_for_station(y, c["end_search"], z, key="end_search"),c["nbjour"])
            rows.append({"year":y,"station":stn,"lat":g.LAT.iloc[0],"lon":g.LON.iloc[0],"onsetdryspell":v})
        return self.build_cpt_output(pd.DataFrame(rows),"onsetdryspell")


# =============================================================================
# Cessation dry-spell length
# =============================================================================

class CEAC_compute_cessation_dry_spell(CAF_AgroClimateBase):

    @staticmethod
    def ds_cess_func(x, id1, cum, nbs, jp, ir1, id2, ijd, ETP, Cap, ir2, nbj, fallback_to_end=False):
        x = np.asarray(x, dtype=float)
        onset = _onset_1d_reference(x, id1, cum, nbs, jp, ir1)
        cessation = _cessation_1d_reference(x, ijd, id2, ETP, Cap, ir2, fallback_to_end)
        if not np.isfinite(onset) or not np.isfinite(cessation) or not np.isfinite(nbj) or nbj < 1 or nbj != int(nbj):
            return np.nan
        return _longest_dry_run_reference(x, onset + nbj, cessation, jp)

    def compute(self,daily_data,map_rec,nb_cores):
        daily_data=self._validate_daily_grid(daily_data); mask=self._prepare_mask(daily_data,map_rec); years=self._valid_years(daily_data,mask); self._configure_numba_threads(nb_cores); annual=[]; fallback=self.no_event_policy=="end"
        for year in years:
            val2d=np.full(mask.shape,np.nan,np.float32)
            for shifted in (False,True):
                group=self._zone_group_mask_flat(mask,shifted)
                if not np.any(group) or not self._group_available(daily_data, year, shifted): continue
                start,end=self._window_bounds(year,shifted); data2d,ny,nx=self._load_window_2d(daily_data,start,end)
                out=_cessation_dryspell_block_numba(data2d,
                    self._parameter_flat(mask,year,"start_search",shifted),self._parameter_flat(mask,year,"cumulative",shifted),self._parameter_flat(mask,year,"number_dry_days",shifted),self._parameter_flat(mask,year,"thrd_rain_day",shifted),self._parameter_flat(mask,year,"end_search",shifted),self._parameter_flat(mask,year,"start_search2",shifted),self._parameter_flat(mask,year,"date_dry_soil",shifted),self._parameter_flat(mask,year,"ETP",shifted),self._parameter_flat(mask,year,"Cap_ret_maxi",shifted),self._parameter_flat(mask,year,"end_search2",shifted),self._parameter_flat(mask,year,"nbjour",shifted),fallback)
                val2d.reshape(-1)[group]=out[group]; del data2d,out
            annual.append(self._result_da(val2d,mask,year,"Cessation_dryspell"))
        return self._finalize(annual,"Cessation_dryspell",{"units":"days"})

    def compute_insitu(self,daily_df_raw,map_rec):
        long,zones,_=self.transform_and_shift_cdt(daily_df_raw,map_rec); rows=[]
        for (stn,y),g in long.groupby(["STATION","year"],sort=True):
            z=zones.get(stn,np.nan)
            if not np.isfinite(z) or int(z) not in self.criteria: v=np.nan
            else:
                z=int(z); c=self.criteria[z]
                v=self.ds_cess_func(g.VALUE.to_numpy(),self.get_index_for_station(y, c["start_search"], z, key="start_search"),c["cumulative"],c["number_dry_days"],c["thrd_rain_day"],self.get_index_for_station(y, c["end_search"], z, key="end_search"),self.get_index_for_station(y, c["start_search2"], z, key="start_search2"),self.get_index_for_station(y, c["date_dry_soil"], z, key="date_dry_soil"),c["ETP"],c["Cap_ret_maxi"],self.get_index_for_station(y, c["end_search2"], z, key="end_search2"),c["nbjour"],self.no_event_policy=="end")
            rows.append({"year":y,"station":stn,"lat":g.LAT.iloc[0],"lon":g.LON.iloc[0],"cessation_dryspell":v})
        return self.build_cpt_output(pd.DataFrame(rows),"cessation_dryspell")


# =============================================================================
# Spell counts
# =============================================================================

class CEAC_count_dry_spells(CAF_AgroClimateBase):

    @staticmethod
    def count_dry_spells(x, onset, cessation, d_len, thresh):
        return _count_spells_reference(np.asarray(x, dtype=float), onset, cessation, d_len, thresh, False)

    def compute(self,daily_data,on_da,cess_da,map_rec,d_len,thresh,nb_cores):
        daily_data=self._validate_daily_grid(daily_data); mask=self._prepare_mask(daily_data,map_rec); years=self._valid_years(daily_data,mask); self._configure_numba_threads(nb_cores); annual=[]
        for year in years:
            on_field=self._select_annual_field(on_da,mask,year); ce_field=self._select_annual_field(cess_da,mask,year)
            on_idx=self._encoded_to_indices_2d(on_field.values,mask,year).reshape(-1); ce_idx=self._encoded_to_indices_2d(ce_field.values,mask,year).reshape(-1)
            val2d=np.full(mask.shape,np.nan,np.float32)
            for shifted in (False,True):
                group=self._zone_group_mask_flat(mask,shifted)
                if not np.any(group) or not self._group_available(daily_data, year, shifted): continue
                start,end=self._window_bounds(year,shifted); data2d,ny,nx=self._load_window_2d(daily_data,start,end)
                out=_count_dry_block_numba(data2d,on_idx,ce_idx,float(d_len),float(thresh)); val2d.reshape(-1)[group]=out[group]; del data2d,out
            annual.append(self._result_da(val2d,mask,year,"Count_dryspell"))
        return self._finalize(annual,"Count_dryspell",{"units":"count","definition":f"number of dry spells of at least {d_len} consecutive days"})

    def compute_insitu(self,daily_raw,on_cpt,cess_cpt,map_rec,d_len,thresh=1.0):
        long,zones,_=self.transform_and_shift_cdt(daily_raw,map_rec); m=pd.merge(self._parse_cpt_to_long(on_cpt,"o"),self._parse_cpt_to_long(cess_cpt,"c"),on=["station","year"],suffixes=("_o","_c")); rows=[]
        for (stn,y),g in long.groupby(["STATION","year"],sort=True):
            sub=m[(m.station==stn)&(m.year==y)]; z=zones.get(stn,np.nan)
            if not np.isfinite(z) or sub.empty: v=np.nan
            else:
                z=int(z); oi=self.extended_doy_to_season_index(sub.o.values[0],y,z); ci=self.extended_doy_to_season_index(sub.c.values[0],y,z); v=self.count_dry_spells(g.VALUE.to_numpy(),oi,ci,d_len,thresh)
            rows.append({"year":y,"station":stn,"lat":g.LAT.iloc[0],"lon":g.LON.iloc[0],"dry_spells":v})
        return self.build_cpt_output(pd.DataFrame(rows),"dry_spells")


class CEAC_count_wet_spells(CAF_AgroClimateBase):

    @staticmethod
    def count_wet_spells(x, onset, cessation, w_len, thresh):
        return _count_spells_reference(np.asarray(x, dtype=float), onset, cessation, w_len, thresh, True)

    def compute(self,daily_data,on_da,cess_da,map_rec,w_len,thresh,nb_cores):
        daily_data=self._validate_daily_grid(daily_data); mask=self._prepare_mask(daily_data,map_rec); years=self._valid_years(daily_data,mask); self._configure_numba_threads(nb_cores); annual=[]
        for year in years:
            on_field=self._select_annual_field(on_da,mask,year); ce_field=self._select_annual_field(cess_da,mask,year)
            on_idx=self._encoded_to_indices_2d(on_field.values,mask,year).reshape(-1); ce_idx=self._encoded_to_indices_2d(ce_field.values,mask,year).reshape(-1)
            val2d=np.full(mask.shape,np.nan,np.float32)
            for shifted in (False,True):
                group=self._zone_group_mask_flat(mask,shifted)
                if not np.any(group) or not self._group_available(daily_data, year, shifted): continue
                start,end=self._window_bounds(year,shifted); data2d,ny,nx=self._load_window_2d(daily_data,start,end)
                out=_count_wet_block_numba(data2d,on_idx,ce_idx,float(w_len),float(thresh)); val2d.reshape(-1)[group]=out[group]; del data2d,out
            annual.append(self._result_da(val2d,mask,year,"Count_wetspell"))
        return self._finalize(annual,"Count_wetspell",{"units":"count","definition":f"number of wet spells of at least {w_len} consecutive days"})

    def compute_insitu(self,daily_raw,on_cpt,cess_cpt,map_rec,w_len,thresh=1.0):
        long,zones,_=self.transform_and_shift_cdt(daily_raw,map_rec); m=pd.merge(self._parse_cpt_to_long(on_cpt,"o"),self._parse_cpt_to_long(cess_cpt,"c"),on=["station","year"],suffixes=("_o","_c")); rows=[]
        for (stn,y),g in long.groupby(["STATION","year"],sort=True):
            sub=m[(m.station==stn)&(m.year==y)]; z=zones.get(stn,np.nan)
            if not np.isfinite(z) or sub.empty: v=np.nan
            else:
                z=int(z); oi=self.extended_doy_to_season_index(sub.o.values[0],y,z); ci=self.extended_doy_to_season_index(sub.c.values[0],y,z); v=self.count_wet_spells(g.VALUE.to_numpy(),oi,ci,w_len,thresh)
            rows.append({"year":y,"station":stn,"lat":g.LAT.iloc[0],"lon":g.LON.iloc[0],"wet_spells":v})
        return self.build_cpt_output(pd.DataFrame(rows),"wet_spells")


__all__ = [
    "DEFAULT_CRITERIA", "NORTH_ANCHOR", "SOUTH_ANCHOR",
    "CAF_AgroClimateBase", "CEAC_compute_onset", "CEAC_compute_cessation",
    "CEAC_compute_onset_dry_spell", "CEAC_compute_cessation_dry_spell",
    "CEAC_count_dry_spells", "CEAC_count_wet_spells",
]
