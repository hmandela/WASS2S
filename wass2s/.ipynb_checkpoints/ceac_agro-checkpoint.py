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
from typing import Optional

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

if HAS_NUMBA:

    @njit(cache=True)
    def _has_dry_run_numba(x, start, stop, dry_len, threshold):
        """True if >= dry_len consecutive finite dry days occur in [start, stop)."""
        if dry_len <= 0:
            return False
        run = 0
        n = x.shape[0]
        if start < 0:
            start = 0
        if stop > n:
            stop = n
        for k in range(start, stop):
            v = x[k]
            if np.isfinite(v) and v < threshold:
                run += 1
                if run >= dry_len:
                    return True
            else:
                run = 0
        return False


    @njit(cache=True)
    def _longest_dry_run_numba(x, start, stop, threshold):
        """Longest consecutive dry run in [start, stop). Missing values break runs."""
        n = x.shape[0]
        if start < 0:
            start = 0
        if stop > n:
            stop = n
        if stop <= start:
            return 0
        cur = 0
        best = 0
        for k in range(start, stop):
            v = x[k]
            if np.isfinite(v) and v < threshold:
                cur += 1
                if cur > best:
                    best = cur
            else:
                cur = 0
        return best


    @njit(cache=True)
    def _onset_1d_numba(x, start_idx, cumulative, dry_len, dry_threshold, end_idx):
        """
        First qualifying onset day.

        A candidate day d qualifies when rainfall accumulated from d over 1, 2
        or 3 consecutive days reaches ``cumulative`` and it is not followed by
        a dry spell of at least ``dry_len`` days during the next 30 days.
        """
        n = x.shape[0]
        if n < 1:
            return np.nan

        s = int(start_idx)
        e = int(end_idx)
        if s < 0:
            s = 0
        if e >= n:
            e = n - 1
        if e < s:
            return np.nan

        for d in range(s, e + 1):
            # The onset event must actually START on a rainy day. Without this
            # guard, a 3-day window [0, 0, heavy_rain] would incorrectly date
            # onset two days before the rain.
            if not np.isfinite(x[d]) or x[d] < dry_threshold:
                continue

            total = 0.0
            qualifies = False
            valid_event = True

            # 1--3 day wet-event accumulation, starting at d.
            for kk in range(3):
                j = d + kk
                if j >= n:
                    break
                v = x[j]
                if not np.isfinite(v):
                    valid_event = False
                    break
                total += v
                if total >= cumulative:
                    qualifies = True
                    break

            if not valid_event or not qualifies:
                continue

            # False-start check: exactly dry_len or longer is disqualifying.
            dry_stop = d + 31  # d+1 ... d+30 inclusive
            if _has_dry_run_numba(x, d + 1, dry_stop, int(dry_len), dry_threshold):
                continue

            return float(d)

        return np.nan


    @njit(cache=True)
    def _cessation_1d_numba(x, dry_soil_idx, search_start_idx, etp, capacity, search_end_idx, fallback_to_end):
        """Simple CEAC water-balance cessation search."""
        n = x.shape[0]
        if n < 1:
            return np.nan

        d0 = int(dry_soil_idx)
        s = int(search_start_idx)
        e = int(search_end_idx)
        if d0 < 0:
            d0 = 0
        if s < 0:
            s = 0
        if e >= n:
            e = n - 1
        if d0 >= n or s >= n or e < s or s < d0:
            return np.nan

        ru = 0.0
        for k in range(d0, s + 1):
            v = x[k]
            if np.isfinite(v):
                ru = ru + v - etp
                if ru < 0.0:
                    ru = 0.0
                if ru > capacity:
                    ru = capacity

        # If the reservoir is already exhausted at the first allowed date,
        # cessation is that date (fixes the old +1-day behaviour).
        if ru <= 0.0:
            return float(s)

        for k in range(s + 1, e + 1):
            v = x[k]
            if not np.isfinite(v):
                continue
            ru = ru + v - etp
            if ru < 0.0:
                ru = 0.0
            if ru > capacity:
                ru = capacity
            if ru <= 0.0:
                return float(k)

        if fallback_to_end:
            return float(e)
        return np.nan


    @njit(parallel=True, cache=True)
    def _onset_block_numba(data2d, start_idx, cumulative, dry_len, dry_thr, end_idx):
        ncell = data2d.shape[0]
        out = np.empty(ncell, dtype=np.float32)
        out[:] = np.nan
        for p in prange(ncell):
            if not np.isfinite(start_idx[p]) or not np.isfinite(end_idx[p]):
                continue
            out[p] = _onset_1d_numba(
                data2d[p], start_idx[p], cumulative[p], dry_len[p], dry_thr[p], end_idx[p]
            )
        return out


    @njit(parallel=True, cache=True)
    def _cessation_block_numba(data2d, dry0, start2, etp, cap, end2, fallback_to_end):
        ncell = data2d.shape[0]
        out = np.empty(ncell, dtype=np.float32)
        out[:] = np.nan
        for p in prange(ncell):
            if not np.isfinite(dry0[p]) or not np.isfinite(start2[p]) or not np.isfinite(end2[p]):
                continue
            out[p] = _cessation_1d_numba(
                data2d[p], dry0[p], start2[p], etp[p], cap[p], end2[p], fallback_to_end
            )
        return out


    @njit(parallel=True, cache=True)
    def _onset_dryspell_block_numba(data2d, start_idx, cumulative, dry_len, dry_thr, end_idx, nbjour):
        ncell = data2d.shape[0]
        out = np.empty(ncell, dtype=np.float32)
        out[:] = np.nan
        for p in prange(ncell):
            if not np.isfinite(start_idx[p]) or not np.isfinite(end_idx[p]):
                continue
            onset = _onset_1d_numba(
                data2d[p], start_idx[p], cumulative[p], dry_len[p], dry_thr[p], end_idx[p]
            )
            if not np.isfinite(onset):
                continue
            o = int(onset)
            # nbjour days AFTER onset (not nbjour+1 including onset).
            out[p] = float(_longest_dry_run_numba(data2d[p], o + 1, o + 1 + int(nbjour[p]), dry_thr[p]))
        return out


    @njit(parallel=True, cache=True)
    def _cessation_dryspell_block_numba(data2d, start1, cumulative, dry_len, dry_thr, end1,
                                         start2, dry0, etp, cap, end2, nbjour, fallback_to_end):
        ncell = data2d.shape[0]
        out = np.empty(ncell, dtype=np.float32)
        out[:] = np.nan
        for p in prange(ncell):
            if not np.isfinite(start1[p]) or not np.isfinite(end1[p]):
                continue
            onset = _onset_1d_numba(
                data2d[p], start1[p], cumulative[p], dry_len[p], dry_thr[p], end1[p]
            )
            if not np.isfinite(onset):
                continue
            cessation = _cessation_1d_numba(
                data2d[p], dry0[p], start2[p], etp[p], cap[p], end2[p], fallback_to_end
            )
            if not np.isfinite(cessation):
                continue
            a = int(onset) + int(nbjour[p])
            b = int(cessation)
            if b <= a:
                continue
            # All-dry intervals correctly return their full length (old code returned NaN).
            out[p] = float(_longest_dry_run_numba(data2d[p], a, b, dry_thr[p]))
        return out


    @njit(parallel=True, cache=True)
    def _count_dry_block_numba(data2d, onset_idx, cessation_idx, min_len, threshold):
        ncell = data2d.shape[0]
        out = np.empty(ncell, dtype=np.float32)
        out[:] = np.nan
        for p in prange(ncell):
            if not np.isfinite(onset_idx[p]) or not np.isfinite(cessation_idx[p]):
                continue
            o = int(onset_idx[p])
            c = int(cessation_idx[p])
            n = data2d.shape[1]
            if o < 0 or c < o or o >= n:
                continue
            if c >= n:
                c = n - 1
            count = 0
            run = 0
            for k in range(o, c + 1):
                v = data2d[p, k]
                if np.isfinite(v) and v < threshold:
                    run += 1
                else:
                    if run >= min_len:
                        count += 1
                    run = 0
            if run >= min_len:
                count += 1
            out[p] = float(count)
        return out


    @njit(parallel=True, cache=True)
    def _count_wet_block_numba(data2d, onset_idx, cessation_idx, min_len, threshold):
        ncell = data2d.shape[0]
        out = np.empty(ncell, dtype=np.float32)
        out[:] = np.nan
        for p in prange(ncell):
            if not np.isfinite(onset_idx[p]) or not np.isfinite(cessation_idx[p]):
                continue
            o = int(onset_idx[p])
            c = int(cessation_idx[p])
            n = data2d.shape[1]
            if o < 0 or c < o or o >= n:
                continue
            if c >= n:
                c = n - 1
            count = 0
            run = 0
            for k in range(o, c + 1):
                v = data2d[p, k]
                if np.isfinite(v) and v >= threshold:
                    run += 1
                else:
                    if run >= min_len:
                        count += 1
                    run = 0
            if run >= min_len:
                count += 1
            out[p] = float(count)
        return out


# =============================================================================
# Base class
# =============================================================================

class CAF_AgroClimateBase:
    """Shared helpers for CEAC station and gridded indicators."""

    # Kept only so old external code importing the attribute does not crash.
    # It is NOT used by the corrected implementation.
    SHIFT_OFFSET = 244

    def __init__(self, user_criteria=None, no_event_policy="nan"):
        self.criteria = user_criteria if user_criteria is not None else DEFAULT_CRITERIA
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
            missing = required - set(c)
            if missing:
                errors.append(f"zone {z}: missing keys {sorted(missing)}")
                continue
            try:
                s1 = self.get_index_for_station(dummy_year, c["start_search"], z)
                e1 = self.get_index_for_station(dummy_year, c["end_search"], z)
                d0 = self.get_index_for_station(dummy_year, c["date_dry_soil"], z)
                s2 = self.get_index_for_station(dummy_year, c["start_search2"], z)
                e2 = self.get_index_for_station(dummy_year, c["end_search2"], z)
            except Exception as exc:
                errors.append(f"zone {z}: invalid date criterion ({exc})")
                continue

            if s1 > e1:
                errors.append(f"zone {z}: start_search index {s1} > end_search index {e1}")
            if d0 > s2:
                errors.append(f"zone {z}: date_dry_soil index {d0} > start_search2 index {s2}")
            if s2 > e2:
                errors.append(f"zone {z}: start_search2 index {s2} > end_search2 index {e2}")
            if c["cumulative"] <= 0 or c["number_dry_days"] <= 0 or c["nbjour"] <= 0:
                errors.append(f"zone {z}: cumulative/number_dry_days/nbjour must be > 0")
            if c["Cap_ret_maxi"] <= 0 or c["ETP"] < 0:
                errors.append(f"zone {z}: invalid ETP/Cap_ret_maxi")

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
        missing = {"T", "Y", "X"} - set(daily_data.dims)
        if missing:
            raise ValueError(f"daily_data missing dimensions {sorted(missing)}")
        if not np.issubdtype(daily_data["T"].dtype, np.datetime64):
            raise TypeError("daily_data['T'] must be datetime64")
        da = daily_data.transpose("T", "Y", "X")
        if not da.get_index("T").is_monotonic_increasing:
            da = da.sortby("T")
        return da

    def _prepare_mask(self, daily_data, map_reclassified):
        if not isinstance(map_reclassified, xr.DataArray):
            raise TypeError("map_reclassified must be an xarray.DataArray")
        mask = map_reclassified.squeeze(drop=True)
        if "T" in mask.dims:
            if mask.sizes["T"] != 1:
                raise ValueError("map_reclassified must be 2-D (Y,X)")
            mask = mask.isel(T=0, drop=True)
        if not {"Y", "X"}.issubset(mask.dims):
            raise ValueError("map_reclassified must contain Y and X dimensions")
        mask = mask.reindex_like(daily_data.isel(T=0, drop=True), method="nearest").transpose("Y", "X")
        vals = np.asarray(mask.values, dtype=float)
        valid = np.isfinite(vals) & np.isin(vals.astype(np.int16, copy=False), np.asarray(sorted(self.criteria), dtype=np.int16))
        cleaned = np.where(valid, vals, np.nan).astype(np.float32)
        return xr.DataArray(cleaned, coords=mask.coords, dims=mask.dims, name="zone")

    def _window_bounds(self, year, shifted):
        year = int(year)
        anchor = SOUTH_ANCHOR if shifted else NORTH_ANCHOR
        m, d = self._parse_mmdd(anchor)
        start = pd.Timestamp(year=year, month=m, day=d)
        # Include the same anchor date in the following year. Zone 6 uses Jul-15
        # as end_search2, so the southern analysis window must include it.
        end = pd.Timestamp(year=year + 1, month=m, day=d) if shifted else pd.Timestamp(year=year, month=12, day=31)
        return start, end

    def _valid_years(self, daily_data, mask):
        tmin = pd.Timestamp(daily_data["T"].values[0])
        tmax = pd.Timestamp(daily_data["T"].values[-1])
        zvals = np.asarray(mask.values)
        has_north = np.any(np.isfinite(zvals) & (zvals <= 5))
        has_south = np.any(np.isfinite(zvals) & (zvals >= 6))
        years = []
        for y in range(tmin.year - 1, tmax.year + 1):
            ok = True
            if has_north:
                s, e = self._window_bounds(y, False)
                ok = ok and (s >= tmin and e <= tmax)
            if has_south:
                s, e = self._window_bounds(y, True)
                ok = ok and (s >= tmin and e <= tmax)
            if ok:
                years.append(y)
        if not years:
            raise ValueError("No complete CEAC agricultural year in daily_data")
        return np.asarray(years, dtype=int)

    @staticmethod
    def _configure_numba_threads(nb_cores):
        if not HAS_NUMBA:
            raise ImportError("Numba is required for gridded CEAC calculations. Install with: conda install -c conda-forge numba")
        n = max(1, int(nb_cores))
        maxn = int(numba.config.NUMBA_NUM_THREADS)
        n = min(n, maxn)
        numba.set_num_threads(n)
        return n

    def _load_window_2d(self, daily_data, start, end):
        """Load one analysis window as contiguous (ncell, T) float32 array."""
        expected = pd.date_range(start, end, freq="D")
        sub = daily_data.sel(T=slice(start, end))
        dates = pd.DatetimeIndex(sub["T"].values)
        if len(dates) != len(expected) or not dates.equals(expected):
            sub = sub.reindex(T=expected)

        # Dask (if present) is used only here for lazy IO, not for pixel kernels.
        data = sub.transpose("Y", "X", "T").data
        if hasattr(data, "compute"):
            data = data.compute()
        arr3 = np.ascontiguousarray(np.asarray(data, dtype=np.float32))
        ny, nx, nt = arr3.shape
        return arr3.reshape(ny * nx, nt), ny, nx

    def _parameter_flat(self, mask, year, key, shifted):
        zflat = np.asarray(mask.values).reshape(-1)
        out = np.full(zflat.size, np.nan, dtype=np.float32)
        for z in self.criteria:
            if (z >= 6) != bool(shifted):
                continue
            sel = zflat == z
            if not np.any(sel):
                continue
            val = self.criteria[z][key]
            if key in _DATE_KEYS:
                val = self.get_index_for_station(year, val, z, key=key)
            out[sel] = np.float32(val)
        return out

    def _zone_group_mask_flat(self, mask, shifted):
        z = np.asarray(mask.values).reshape(-1)
        if shifted:
            return np.isfinite(z) & (z >= 6)
        return np.isfinite(z) & (z <= 5)

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
        ids_s = pd.Series(df.iloc[0, 1:].astype(str).values)
        ids = ids_s.where(~ids_s.duplicated(), ids_s + "_" + ids_s.groupby(ids_s).cumcount().astype(str)).values
        df.iloc[0, 1:] = ids
        lons = pd.to_numeric(df.iloc[1, 1:], errors="coerce").to_numpy(float)
        lats = pd.to_numeric(df.iloc[2, 1:], errors="coerce").to_numpy(float)
        dates0 = pd.to_datetime(df.iloc[4:, 0], format="%Y%m%d", errors="coerce")
        good = ~dates0.isna()
        dates = pd.DatetimeIndex(dates0[good])
        vals = df.iloc[4:, 1:].loc[good].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        vals[vals == -99.0] = np.nan
        da = xr.DataArray(vals, coords={"T": dates, "station": ids}, dims=("T", "station")).sortby("T")

        zmap = map_reclassified.squeeze(drop=True)
        if "T" in zmap.dims:
            zmap = zmap.isel(T=0, drop=True)

        tmin = pd.Timestamp(da["T"].values[0])
        tmax = pd.Timestamp(da["T"].values[-1])
        stn_zones = {}
        frames = []

        for i, stn in enumerate(ids):
            try:
                z = float(zmap.sel(X=lons[i], Y=lats[i], method="nearest").values.item())
            except Exception:
                z = np.nan
            stn_zones[stn] = z
            if not np.isfinite(z) or int(z) not in self.criteria:
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
                v = da.sel(T=str(int(year)))
            else:
                try:
                    v = da.sel(T=int(year))
                except Exception:
                    v = da.sel(T=str(int(year)))
            if v.sizes.get("T", 1) != 1:
                raise ValueError(f"Expected one field for year {year}, found {v.sizes.get('T', 0)}")
            v = v.squeeze(drop=True)
        else:
            v = da.squeeze(drop=True)
        return v.reindex_like(mask, method="nearest")

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
        mask = map_reclassified.squeeze(drop=True)
        if "T" in mask.dims:
            mask = mask.isel(T=0, drop=True)
        mask = mask.reindex_like(template, method="nearest").transpose("Y", "X")
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
        """Python reference implementation matching the corrected Numba kernel."""
        x = np.asarray(x, dtype=float)
        if not (np.isfinite(idebut) and np.isfinite(cumul) and np.isfinite(nbsec) and np.isfinite(jour_pluvieux) and np.isfinite(irch_fin)):
            return np.nan
        s = max(0, int(idebut)); e = min(len(x)-1, int(irch_fin))
        if e < s:
            return np.nan
        for d in range(s, e+1):
            if not np.isfinite(x[d]) or x[d] < jour_pluvieux:
                continue
            total = 0.0; q = False
            for kk in range(3):
                j = d + kk
                if j >= len(x) or not np.isfinite(x[j]):
                    break
                total += x[j]
                if total >= cumul:
                    q = True; break
            if not q:
                continue
            run = 0; bad = False
            for j in range(d+1, min(d+31, len(x))):
                if np.isfinite(x[j]) and x[j] < jour_pluvieux:
                    run += 1
                    if run >= int(nbsec):
                        bad = True; break
                else:
                    run = 0
            if not bad:
                return float(d)
        return np.nan

    def compute(self, daily_data, map_rec, nb_cores):
        daily_data = self._validate_daily_grid(daily_data)
        mask = self._prepare_mask(daily_data, map_rec)
        years = self._valid_years(daily_data, mask)
        self._configure_numba_threads(nb_cores)
        annual = []

        for year in years:
            idx2d = np.full(mask.shape, np.nan, dtype=np.float32)
            for shifted in (False, True):
                group = self._zone_group_mask_flat(mask, shifted)
                if not np.any(group):
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
                del data2d, out
            encoded = self._indices_to_extended_doy_2d(idx2d, mask, year)
            annual.append(self._result_da(encoded, mask, year, "Onset"))

        return self._finalize(annual, "Onset", {
            "units": "extended_day_of_year",
            "encoding_note": "1=Jan-01 of agricultural start year; values > days_in_year refer to next civil year",
            "south_analysis_anchor": SOUTH_ANCHOR,
            "no_event_policy": self.no_event_policy,
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
                idx = self.onset_function(g.VALUE.to_numpy(), self.get_index_for_station(y,c["start_search"],z), c["cumulative"], c["number_dry_days"], c["thrd_rain_day"], self.get_index_for_station(y,c["end_search"],z))
                v = self.season_index_to_extended_doy(idx, y, z)
            rows.append({"year": y, "station": stn, "lat": g.LAT.iloc[0], "lon": g.LON.iloc[0], "onset": v})
        return self.build_cpt_output(pd.DataFrame(rows), "onset")


# =============================================================================
# Cessation
# =============================================================================

class CEAC_compute_cessation(CAF_AgroClimateBase):

    @staticmethod
    def cessation_function(x, ijour, idebut, ETP, Cap, irch_fin, fallback_to_end=False):
        x = np.asarray(x, dtype=float)
        if not all(np.isfinite(v) for v in (ijour, idebut, ETP, Cap, irch_fin)):
            return np.nan
        d0=max(0,int(ijour)); s=max(0,int(idebut)); e=min(len(x)-1,int(irch_fin))
        if d0>=len(x) or s>=len(x) or e<s or s<d0:
            return np.nan
        ru=0.0
        for k in range(d0,s+1):
            if np.isfinite(x[k]):
                ru=max(0.0,min(ru+x[k]-ETP,Cap))
        if ru<=0:
            return float(s)
        for k in range(s+1,e+1):
            if not np.isfinite(x[k]):
                continue
            ru=max(0.0,min(ru+x[k]-ETP,Cap))
            if ru<=0:
                return float(k)
        return float(e) if fallback_to_end else np.nan

    def compute(self, daily_data, map_rec, nb_cores):
        daily_data = self._validate_daily_grid(daily_data)
        mask = self._prepare_mask(daily_data, map_rec)
        years = self._valid_years(daily_data, mask)
        self._configure_numba_threads(nb_cores)
        annual=[]
        fallback = self.no_event_policy == "end"
        for year in years:
            idx2d=np.full(mask.shape,np.nan,dtype=np.float32)
            for shifted in (False,True):
                group=self._zone_group_mask_flat(mask,shifted)
                if not np.any(group): continue
                start,end=self._window_bounds(year,shifted)
                data2d,ny,nx=self._load_window_2d(daily_data,start,end)
                out=_cessation_block_numba(
                    data2d,
                    self._parameter_flat(mask,year,"date_dry_soil",shifted),
                    self._parameter_flat(mask,year,"start_search2",shifted),
                    self._parameter_flat(mask,year,"ETP",shifted),
                    self._parameter_flat(mask,year,"Cap_ret_maxi",shifted),
                    self._parameter_flat(mask,year,"end_search2",shifted),
                    fallback,
                )
                idx2d.reshape(-1)[group]=out[group]
                del data2d,out
            enc=self._indices_to_extended_doy_2d(idx2d,mask,year)
            annual.append(self._result_da(enc,mask,year,"Cessation"))
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
                idx=self.cessation_function(g.VALUE.to_numpy(),self.get_index_for_station(y,c["date_dry_soil"],z),self.get_index_for_station(y,c["start_search2"],z),c["ETP"],c["Cap_ret_maxi"],self.get_index_for_station(y,c["end_search2"],z),self.no_event_policy=="end")
                v=self.season_index_to_extended_doy(idx,y,z)
            rows.append({"year":y,"station":stn,"lat":g.LAT.iloc[0],"lon":g.LON.iloc[0],"cessation":v})
        return self.build_cpt_output(pd.DataFrame(rows),"cessation")


# =============================================================================
# Onset dry-spell length
# =============================================================================

class CEAC_compute_onset_dry_spell(CAF_AgroClimateBase):

    @staticmethod
    def ds_onset_func(x,idebut,cumul,nbsec,jp,irch_fin,nbjour):
        onset=CEAC_compute_onset.onset_function(x,idebut,cumul,nbsec,jp,irch_fin)
        if not np.isfinite(onset): return np.nan
        x=np.asarray(x,float); a=int(onset)+1; b=min(len(x),a+int(nbjour)); cur=best=0
        for v in x[a:b]:
            if np.isfinite(v) and v<jp: cur+=1; best=max(best,cur)
            else: cur=0
        return float(best)

    def compute(self,daily_data,map_rec,nb_cores):
        daily_data=self._validate_daily_grid(daily_data); mask=self._prepare_mask(daily_data,map_rec); years=self._valid_years(daily_data,mask); self._configure_numba_threads(nb_cores); annual=[]
        for year in years:
            val2d=np.full(mask.shape,np.nan,np.float32)
            for shifted in (False,True):
                group=self._zone_group_mask_flat(mask,shifted)
                if not np.any(group): continue
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
                v=self.ds_onset_func(g.VALUE.to_numpy(),self.get_index_for_station(y,c["start_search"],z),c["cumulative"],c["number_dry_days"],c["thrd_rain_day"],self.get_index_for_station(y,c["end_search"],z),c["nbjour"])
            rows.append({"year":y,"station":stn,"lat":g.LAT.iloc[0],"lon":g.LON.iloc[0],"onsetdryspell":v})
        return self.build_cpt_output(pd.DataFrame(rows),"onsetdryspell")


# =============================================================================
# Cessation dry-spell length
# =============================================================================

class CEAC_compute_cessation_dry_spell(CAF_AgroClimateBase):

    @staticmethod
    def ds_cess_func(x,id1,cum,nbs,jp,ir1,id2,ijd,ETP,Cap,ir2,nbj,fallback_to_end=False):
        onset=CEAC_compute_onset.onset_function(x,id1,cum,nbs,jp,ir1)
        if not np.isfinite(onset): return np.nan
        cess=CEAC_compute_cessation.cessation_function(x,ijd,id2,ETP,Cap,ir2,fallback_to_end)
        if not np.isfinite(cess): return np.nan
        x=np.asarray(x,float); a=int(onset)+int(nbj); b=int(cess)
        if b<=a: return np.nan
        cur=best=0
        for v in x[a:b]:
            if np.isfinite(v) and v<jp: cur+=1; best=max(best,cur)
            else: cur=0
        return float(best)

    def compute(self,daily_data,map_rec,nb_cores):
        daily_data=self._validate_daily_grid(daily_data); mask=self._prepare_mask(daily_data,map_rec); years=self._valid_years(daily_data,mask); self._configure_numba_threads(nb_cores); annual=[]; fallback=self.no_event_policy=="end"
        for year in years:
            val2d=np.full(mask.shape,np.nan,np.float32)
            for shifted in (False,True):
                group=self._zone_group_mask_flat(mask,shifted)
                if not np.any(group): continue
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
                v=self.ds_cess_func(g.VALUE.to_numpy(),self.get_index_for_station(y,c["start_search"],z),c["cumulative"],c["number_dry_days"],c["thrd_rain_day"],self.get_index_for_station(y,c["end_search"],z),self.get_index_for_station(y,c["start_search2"],z),self.get_index_for_station(y,c["date_dry_soil"],z),c["ETP"],c["Cap_ret_maxi"],self.get_index_for_station(y,c["end_search2"],z),c["nbjour"],self.no_event_policy=="end")
            rows.append({"year":y,"station":stn,"lat":g.LAT.iloc[0],"lon":g.LON.iloc[0],"cessation_dryspell":v})
        return self.build_cpt_output(pd.DataFrame(rows),"cessation_dryspell")


# =============================================================================
# Spell counts
# =============================================================================

class CEAC_count_dry_spells(CAF_AgroClimateBase):

    @staticmethod
    def count_dry_spells(x,onset,cessation,d_len,thresh):
        x=np.asarray(x,float)
        if not np.isfinite(onset) or not np.isfinite(cessation): return np.nan
        o=int(onset); c=min(int(cessation),len(x)-1)
        if o<0 or c<o or o>=len(x): return np.nan
        count=run=0
        for v in x[o:c+1]:
            if np.isfinite(v) and v<thresh: run+=1
            else:
                if run>=int(d_len): count+=1
                run=0
        if run>=int(d_len): count+=1
        return float(count)

    def compute(self,daily_data,on_da,cess_da,map_rec,d_len,thresh,nb_cores):
        daily_data=self._validate_daily_grid(daily_data); mask=self._prepare_mask(daily_data,map_rec); years=self._valid_years(daily_data,mask); self._configure_numba_threads(nb_cores); annual=[]
        for year in years:
            on_field=self._select_annual_field(on_da,mask,year); ce_field=self._select_annual_field(cess_da,mask,year)
            on_idx=self._encoded_to_indices_2d(on_field.values,mask,year).reshape(-1); ce_idx=self._encoded_to_indices_2d(ce_field.values,mask,year).reshape(-1)
            val2d=np.full(mask.shape,np.nan,np.float32)
            for shifted in (False,True):
                group=self._zone_group_mask_flat(mask,shifted)
                if not np.any(group): continue
                start,end=self._window_bounds(year,shifted); data2d,ny,nx=self._load_window_2d(daily_data,start,end)
                out=_count_dry_block_numba(data2d,on_idx,ce_idx,int(d_len),float(thresh)); val2d.reshape(-1)[group]=out[group]; del data2d,out
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
    def count_wet_spells(x,onset,cessation,w_len,thresh):
        x=np.asarray(x,float)
        if not np.isfinite(onset) or not np.isfinite(cessation): return np.nan
        o=int(onset); c=min(int(cessation),len(x)-1)
        if o<0 or c<o or o>=len(x): return np.nan
        count=run=0
        for v in x[o:c+1]:
            if np.isfinite(v) and v>=thresh: run+=1
            else:
                if run>=int(w_len): count+=1
                run=0
        if run>=int(w_len): count+=1
        return float(count)

    def compute(self,daily_data,on_da,cess_da,map_rec,w_len,thresh,nb_cores):
        daily_data=self._validate_daily_grid(daily_data); mask=self._prepare_mask(daily_data,map_rec); years=self._valid_years(daily_data,mask); self._configure_numba_threads(nb_cores); annual=[]
        for year in years:
            on_field=self._select_annual_field(on_da,mask,year); ce_field=self._select_annual_field(cess_da,mask,year)
            on_idx=self._encoded_to_indices_2d(on_field.values,mask,year).reshape(-1); ce_idx=self._encoded_to_indices_2d(ce_field.values,mask,year).reshape(-1)
            val2d=np.full(mask.shape,np.nan,np.float32)
            for shifted in (False,True):
                group=self._zone_group_mask_flat(mask,shifted)
                if not np.any(group): continue
                start,end=self._window_bounds(year,shifted); data2d,ny,nx=self._load_window_2d(daily_data,start,end)
                out=_count_wet_block_numba(data2d,on_idx,ce_idx,int(w_len),float(thresh)); val2d.reshape(-1)[group]=out[group]; del data2d,out
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

# """
# CEAC agroclimatic indicators: memory-safe implementation.

# This module computes onset, cessation, onset-related dry spells,
# cessation-related dry spells, and counts of dry/wet spells for:

# 1. Station data in CPT/CDT-like tabular format.
# 2. Gridded daily xarray.DataArray data with dimensions (T, Y, X).

# Design
# ------
# The key design goal is to avoid constructing a multi-decadal shifted
# (T, Y, X) cube in memory. Gridded calculations are performed one
# agricultural year at a time and the two seasonal regimes are treated
# separately:

# * zones 1..5:  Jan 01 (year y) -> Dec 31 (year y)
# * zones 6..9:  Aug 01 (year y) -> Jul 31 (year y+1)

# The two regimes are merged only after each indicator has been reduced to
# a 2-D (Y, X) field. This also avoids leap-year alignment errors between
# calendar-year and Aug-Jul seasons.

# Compatibility
# -------------
# The public class names and main method signatures from the previous
# implementation are retained:

#     CAF_AgroClimateBase
#     CEAC_compute_onset
#     CEAC_compute_cessation
#     CEAC_compute_onset_dry_spell
#     CEAC_compute_cessation_dry_spell
#     CEAC_count_dry_spells
#     CEAC_count_wet_spells

# All DEFAULT_CRITERIA dates are REAL calendar dates. For zones >= 6,
# get_index_for_station() internally converts real dates to indices in an
# Aug-Jul agricultural year.
# """

# from __future__ import annotations

# import atexit
# import contextlib
# import datetime as _dt
# import random
# import warnings
# from typing import Callable, Iterable, Optional

# import numpy as np
# import pandas as pd
# import xarray as xr

# try:
#     import dask.array as _dask_array  # noqa: F401
#     HAS_DASK = True
# except Exception:
#     HAS_DASK = False


# # =============================================================================
# # Dask client management
# # =============================================================================

# _SHARED_CLIENT = None


# def _bounded_close(client) -> None:
#     """Close a Dask client/cluster without allowing teardown to hang."""
#     if client is None:
#         return

#     cluster = getattr(client, "cluster", None)

#     with contextlib.suppress(Exception):
#         client.close(timeout=10)

#     if cluster is not None:
#         with contextlib.suppress(Exception):
#             cluster.close(timeout=10)


# def _safe_close_client(client) -> None:
#     """
#     Compatibility no-op.

#     Clients returned by _get_compute_client() are process-wide and reused.
#     They are intentionally not closed after every annual computation.
#     """
#     return None


# def close_shared_client() -> None:
#     """Explicitly close the client created by this module, if any."""
#     global _SHARED_CLIENT

#     if _SHARED_CLIENT is not None:
#         _bounded_close(_SHARED_CLIENT)
#         _SHARED_CLIENT = None


# def _get_compute_client(nb_cores):
#     """
#     Return an existing Dask distributed client or lazily create one.

#     If distributed cannot be initialized, return None. In that case,
#     xarray/dask ``.compute()`` uses the active/default scheduler.

#     Notes
#     -----
#     ``memory_limit="auto"`` is intentionally used instead of disabling
#     worker memory management.
#     """
#     global _SHARED_CLIENT

#     if nb_cores is None:
#         nb_cores = 1

#     nb_cores = int(nb_cores)
#     if nb_cores <= 0:
#         raise ValueError("nb_cores must be a positive integer.")

#     try:
#         from dask.distributed import Client, LocalCluster, get_client

#         try:
#             return get_client()
#         except Exception:
#             pass

#         if _SHARED_CLIENT is not None:
#             try:
#                 if _SHARED_CLIENT.status == "running":
#                     return _SHARED_CLIENT
#             except Exception:
#                 _SHARED_CLIENT = None

#         cluster = LocalCluster(
#             n_workers=nb_cores,
#             threads_per_worker=1,
#             processes=True,
#             dashboard_address=None,
#             memory_limit="auto",
#         )

#         _SHARED_CLIENT = Client(cluster)
#         return _SHARED_CLIENT

#     except Exception as exc:
#         warnings.warn(
#             f"Could not create a distributed Dask client ({exc!r}). "
#             "Falling back to the active/default Dask scheduler.",
#             RuntimeWarning,
#         )
#         return None


# atexit.register(close_shared_client)


# # =============================================================================
# # Default CEAC criteria
# # =============================================================================

# DEFAULT_CRITERIA = {
#     1: {
#         "start_search": "05-01",
#         "start_search2": "09-01",
#         "date_dry_soil": "01-01",
#         "cumulative": 15,
#         "number_dry_days": 15,
#         "thrd_rain_day": 0.85,
#         "end_search": "08-30",
#         "end_search2": "10-30",
#         "nbjour": 35,
#         "ETP": 5.0,
#         "Cap_ret_maxi": 70,
#     },
#     2: {
#         "start_search": "03-15",
#         "start_search2": "09-01",
#         "date_dry_soil": "01-01",
#         "cumulative": 20,
#         "number_dry_days": 10,
#         "thrd_rain_day": 0.85,
#         "end_search": "08-01",
#         "end_search2": "11-01",
#         "nbjour": 40,
#         "ETP": 5.0,
#         "Cap_ret_maxi": 70,
#     },
#     3: {
#         "start_search": "02-01",
#         "start_search2": "10-01",
#         "date_dry_soil": "01-01",
#         "cumulative": 20,
#         "number_dry_days": 10,
#         "thrd_rain_day": 0.85,
#         "end_search": "05-15",
#         "end_search2": "12-30",
#         "nbjour": 45,
#         "ETP": 5.0,
#         "Cap_ret_maxi": 70,
#     },
#     4: {
#         "start_search": "01-01",
#         "start_search2": "11-01",
#         "date_dry_soil": "01-01",
#         "cumulative": 20,
#         "number_dry_days": 7,
#         "thrd_rain_day": 0.85,
#         "end_search": "04-01",
#         "end_search2": "12-30",
#         "nbjour": 50,
#         "ETP": 5.0,
#         "Cap_ret_maxi": 80,
#     },
#     5: {
#         "start_search": "01-01",
#         "start_search2": "06-01",
#         "date_dry_soil": "01-01",
#         "cumulative": 25,
#         "number_dry_days": 7,
#         "thrd_rain_day": 0.85,
#         "end_search": "03-10",
#         "end_search2": "08-10",
#         "nbjour": 50,
#         "ETP": 4.0,
#         "Cap_ret_maxi": 60,
#     },
#     # Austral zones: real dates; the Aug-Jul shift is handled internally.
#     6: {
#         "start_search": "08-01",
#         "start_search2": "04-20",
#         "date_dry_soil": "08-01",
#         "cumulative": 25,
#         "number_dry_days": 7,
#         "thrd_rain_day": 0.85,
#         "end_search": "10-18",
#         "end_search2": "07-15",
#         "nbjour": 50,
#         "ETP": 6.0,
#         "Cap_ret_maxi": 70,
#     },
#     7: {
#         "start_search": "07-15",
#         "start_search2": "03-01",
#         "date_dry_soil": "08-01",
#         "cumulative": 20,
#         "number_dry_days": 10,
#         "thrd_rain_day": 0.85,
#         "end_search": "10-20",
#         "end_search2": "06-20",
#         "nbjour": 50,
#         "ETP": 6.0,
#         "Cap_ret_maxi": 70,
#     },
#     8: {
#         "start_search": "09-01",
#         "start_search2": "03-01",
#         "date_dry_soil": "08-01",
#         "cumulative": 20,
#         "number_dry_days": 10,
#         "thrd_rain_day": 0.85,
#         "end_search": "12-01",
#         "end_search2": "05-15",
#         "nbjour": 40,
#         "ETP": 4.0,
#         "Cap_ret_maxi": 70,
#     },
#     9: {
#         "start_search": "10-01",
#         "start_search2": "03-15",
#         "date_dry_soil": "08-01",
#         "cumulative": 20,
#         "number_dry_days": 15,
#         "thrd_rain_day": 0.85,
#         "end_search": "01-31",
#         "end_search2": "05-15",
#         "nbjour": 30,
#         "ETP": 5.0,
#         "Cap_ret_maxi": 60,
#     },
# }


# # =============================================================================
# # Base class
# # =============================================================================

# class CAF_AgroClimateBase:
#     """
#     Shared CEAC helpers for station and gridded calculations.

#     Parameters
#     ----------
#     user_criteria : dict, optional
#         CEAC criteria dictionary. If None, DEFAULT_CRITERIA is used.
#     """

#     SHIFT_OFFSET = 244

#     def __init__(self, user_criteria=None):
#         self.criteria = (
#             user_criteria if user_criteria is not None else DEFAULT_CRITERIA
#         )

#     # ------------------------------------------------------------------
#     # General validation / date helpers
#     # ------------------------------------------------------------------

#     def _is_shifted(self, z) -> bool:
#         """True for valid austral zones (zone id >= 6)."""
#         return not pd.isna(z) and int(z) >= 6

#     @staticmethod
#     def day_of_year(y, mm_dd):
#         dt = _dt.datetime.strptime(f"{int(y)}-{mm_dd}", "%Y-%m-%d").date()
#         return (dt - _dt.date(int(y), 1, 1)).days + 1

#     def get_index_for_station(self, year, mm_dd, z):
#         """
#         Convert a REAL calendar date to a 0-based seasonal-array index.

#         Zones <= 5
#             Seasonal array = Jan 01 -> Dec 31 of ``year``.

#         Zones >= 6
#             Seasonal array = Aug 01 of ``year`` -> Jul 31 of ``year+1``.
#             Dates with month < 8 therefore belong to the following civil year.
#         """
#         year = int(year)
#         shifted = self._is_shifted(z)

#         base_date = (
#             _dt.date(year, 8, 1)
#             if shifted
#             else _dt.date(year, 1, 1)
#         )

#         month, day = map(int, str(mm_dd).split("-"))
#         target_year = year + 1 if (shifted and month < 8) else year
#         target_date = _dt.date(target_year, month, day)

#         return (target_date - base_date).days

#     def output_format_value(self, v, z):
#         if pd.isna(v):
#             return np.nan
#         return v + self.SHIFT_OFFSET if self._is_shifted(z) else v + 1

#     def revert_to_index(self, v, z):
#         if pd.isna(v):
#             return np.nan
#         return int(v - self.SHIFT_OFFSET) if self._is_shifted(z) else int(v - 1)

#     # ------------------------------------------------------------------
#     # Gridded spatial/time helpers
#     # ------------------------------------------------------------------

#     @staticmethod
#     def _validate_daily_grid(daily_data):
#         if not isinstance(daily_data, xr.DataArray):
#             raise TypeError("daily_data must be an xarray.DataArray.")

#         missing = {"T", "Y", "X"} - set(daily_data.dims)
#         if missing:
#             raise ValueError(
#                 f"daily_data must contain dimensions ('T','Y','X'); "
#                 f"missing={sorted(missing)}."
#             )

#         if not np.issubdtype(daily_data["T"].dtype, np.datetime64):
#             raise TypeError(
#                 "daily_data['T'] must be datetime64 for calendar/season handling."
#             )

#         if daily_data.sizes["T"] == 0:
#             raise ValueError("daily_data contains no time samples.")

#         if not daily_data.get_index("T").is_monotonic_increasing:
#             daily_data = daily_data.sortby("T")

#         return daily_data.transpose("T", "Y", "X")

#     def _prepare_mask(self, daily_data, map_reclassified):
#         """
#         Align the zone map to the 2-D rainfall grid only.

#         Crucially, the mask is NOT broadcast/reindexed over the full T axis.
#         """
#         if not isinstance(map_reclassified, xr.DataArray):
#             raise TypeError("map_reclassified must be an xarray.DataArray.")

#         mask = map_reclassified

#         # Drop only NON-spatial singleton dimensions. Never squeeze Y/X, because
#         # one-cell transects/domains are valid.
#         for dim in list(mask.dims):
#             if dim in ("Y", "X"):
#                 continue
#             if mask.sizes[dim] != 1:
#                 raise ValueError(
#                     "map_reclassified must be a 2-D (Y, X) field; "
#                     f"unexpected non-singleton dimension {dim!r}."
#                 )
#             mask = mask.isel({dim: 0}, drop=True)

#         if not {"Y", "X"}.issubset(mask.dims):
#             raise ValueError("map_reclassified must contain Y and X dimensions.")

#         template = daily_data.isel(T=0, drop=True)
#         mask = mask.reindex_like(template, method="nearest")
#         mask = mask.transpose("Y", "X")

#         return mask

#     def _mask_has_regime(self, mask, shifted):
#         vals = np.asarray(mask.values)
#         finite = vals[np.isfinite(vals)]
#         if finite.size == 0:
#             return False

#         valid = np.array(
#             [int(v) in self.criteria for v in finite],
#             dtype=bool,
#         )
#         finite = finite[valid]
#         if finite.size == 0:
#             return False

#         if shifted:
#             return bool(np.any(finite >= 6))
#         return bool(np.any(finite <= 5))

#     @staticmethod
#     def _season_bounds(year, shifted):
#         year = int(year)
#         if shifted:
#             return (
#                 pd.Timestamp(year=year, month=8, day=1),
#                 pd.Timestamp(year=year + 1, month=7, day=31),
#             )
#         return (
#             pd.Timestamp(year=year, month=1, day=1),
#             pd.Timestamp(year=year, month=12, day=31),
#         )

#     def _valid_season_years(self, daily_data, mask):
#         """
#         Return years for which all regimes actually present in ``mask`` have
#         complete boundary coverage in the input time axis.

#         Missing days inside a covered season are inserted as NaN later.
#         """
#         tmin = pd.Timestamp(daily_data["T"].values[0])
#         tmax = pd.Timestamp(daily_data["T"].values[-1])

#         has_north = self._mask_has_regime(mask, shifted=False)
#         has_south = self._mask_has_regime(mask, shifted=True)

#         if not has_north and not has_south:
#             raise ValueError("No valid CEAC zones found in map_reclassified.")

#         candidates = range(tmin.year - 1, tmax.year + 1)
#         years = []

#         for y in candidates:
#             ok = True

#             if has_north:
#                 s, e = self._season_bounds(y, shifted=False)
#                 ok = ok and (s >= tmin and e <= tmax)

#             if has_south:
#                 s, e = self._season_bounds(y, shifted=True)
#                 ok = ok and (s >= tmin and e <= tmax)

#             if ok:
#                 years.append(int(y))

#         if not years:
#             raise ValueError(
#                 "No complete agricultural year is covered by daily_data "
#                 "for the zone regimes present in the mask."
#             )

#         return np.asarray(years, dtype=int)

#     def _season_for_year(self, daily_data, year, shifted):
#         """
#         Return one complete daily seasonal cube for one regime/year.

#         The selection is reindexed to a complete daily date range, so missing
#         dates become NaN instead of silently changing day indices.
#         """
#         start, end = self._season_bounds(year, shifted)
#         expected = pd.date_range(start, end, freq="D")

#         season = daily_data.sel(T=slice(start, end))
#         season = season.reindex(T=expected)

#         return season.transpose("T", "Y", "X")

#     @staticmethod
#     def _spatial_chunks(da, nb_cores):
#         """
#         Choose approximately ``nb_cores`` spatial chunks, not nb_cores**2.
#         """
#         n_workers = max(1, int(nb_cores))
#         n_side = max(1, int(np.ceil(np.sqrt(n_workers))))

#         ny = int(da.sizes["Y"])
#         nx = int(da.sizes["X"])

#         cy = max(1, int(np.ceil(ny / n_side)))
#         cx = max(1, int(np.ceil(nx / n_side)))

#         return {"Y": cy, "X": cx}

#     @staticmethod
#     def _blank_2d(mask, dtype=np.float32):
#         return xr.full_like(mask, np.nan, dtype=dtype)

#     @staticmethod
#     def _maybe_chunk(obj, chunks):
#         """Chunk with Dask when available; otherwise return the object unchanged."""
#         if HAS_DASK:
#             return obj.chunk(chunks)
#         return obj

#     def _combine_regimes(self, north, south, mask):
#         """Merge already-reduced 2-D results; no 3-D historical allocation."""
#         if north is None:
#             north = self._blank_2d(mask)
#         if south is None:
#             south = self._blank_2d(mask)

#         return xr.where(
#             mask <= 5,
#             north,
#             xr.where(mask >= 6, south, np.nan),
#         )

#     def _map_criteria(self, mask, key, year):
#         """
#         Broadcast one CEAC criterion over a 2-D zone map.

#         Date criteria are converted to zero-based indices in the relevant
#         seasonal array using get_index_for_station().
#         """
#         is_date = ("search" in key or "date" in key)

#         def _safe_get(z):
#             if pd.isna(z):
#                 return np.nan

#             z = int(z)
#             if z not in self.criteria:
#                 return np.nan

#             value = self.criteria[z][key]
#             if is_date:
#                 return self.get_index_for_station(int(year), value, z)
#             return value

#         values = np.vectorize(_safe_get, otypes=[float])(mask.values)

#         return xr.DataArray(
#             values,
#             coords=mask.coords,
#             dims=mask.dims,
#             name=key,
#         )

#     def _compute_grid_indicator_year(
#         self,
#         daily_data,
#         mask,
#         year,
#         nb_cores,
#         func,
#         criterion_keys,
#         *,
#         kwargs=None,
#         output_dtype=np.float32,
#     ):
#         """
#         Compute a CEAC indicator for one agricultural year.

#         North/calendar and south/Aug-Jul regimes are evaluated independently.
#         They are merged only after the T core dimension has been reduced.
#         """
#         kwargs = {} if kwargs is None else dict(kwargs)
#         spatial_chunks = self._spatial_chunks(daily_data, nb_cores)

#         regime_results = {}

#         for shifted, label in ((False, "north"), (True, "south")):
#             if not self._mask_has_regime(mask, shifted):
#                 regime_results[label] = None
#                 continue

#             season = self._season_for_year(daily_data, year, shifted)

#             zone_condition = (mask >= 6) if shifted else (mask <= 5)

#             # Mask only this one annual/seasonal cube; this stays lazy after chunk().
#             season = season.where(zone_condition)
#             season = self._maybe_chunk(
#                 season,
#                 {
#                     "T": -1,  # T is a gufunc core dimension: exactly one chunk.
#                     **spatial_chunks,
#                 },
#             )

#             args = [season]

#             for key in criterion_keys:
#                 arr = self._map_criteria(mask, key, year).where(zone_condition)
#                 args.append(self._maybe_chunk(arr, spatial_chunks))

#             res = xr.apply_ufunc(
#                 func,
#                 *args,
#                 input_core_dims=[("T",)] + [()] * len(criterion_keys),
#                 output_core_dims=[()],
#                 vectorize=True,
#                 dask="parallelized",
#                 output_dtypes=[output_dtype],
#                 kwargs=kwargs,
#             ).compute()

#             regime_results[label] = res.astype(output_dtype, copy=False)

#         return self._combine_regimes(
#             regime_results.get("north"),
#             regime_results.get("south"),
#             mask,
#         )

#     def _final_reference_date(self, mask):
#         """Retain the historical output-date convention based on minimum zone."""
#         unique_zone = np.unique(np.asarray(mask.values))
#         unique_zone = unique_zone[np.isfinite(unique_zone)]

#         unique_zone = np.asarray(
#             [z for z in unique_zone if int(z) in self.criteria],
#             dtype=float,
#         )

#         if unique_zone.size == 0:
#             raise ValueError("No valid zones found in the mask.")

#         zone_id = int(np.min(unique_zone))
#         return self.criteria[zone_id]["start_search"]

#     def _finalize_annual_grid(self, out, years, mask, name, format_index=False):
#         if len(out) != len(years):
#             raise RuntimeError("Internal error: output/year length mismatch.")

#         final = xr.concat(out, dim=pd.Index(years, name="T"))

#         if format_index:
#             final = self.format_grid_output(final, mask)

#         ref_mmdd = self._final_reference_date(mask)
#         final["T"] = pd.to_datetime([f"{int(y)}-{ref_mmdd}" for y in years])
#         final = final.transpose("T", "Y", "X")
#         final.name = name

#         return final

#     # ------------------------------------------------------------------
#     # Backward-compatible shift_gridded_data
#     # ------------------------------------------------------------------

#     def shift_gridded_data(self, daily_data, map_reclassified):
#         """
#         Backward-compatible LAZY shifted cube constructor.

#         New CEAC compute() methods do NOT use this method. It is retained only
#         for external code that still calls it directly.

#         Because calendar-year and Aug-Jul seasons can have different leap-day
#         placement, the compatibility cube uses positional truncation as the old
#         implementation did. For scientifically robust calculations, use the
#         indicator compute() methods, which treat regimes separately.
#         """
#         daily_data = self._validate_daily_grid(daily_data)
#         mask = self._prepare_mask(daily_data, map_reclassified)
#         years = self._valid_season_years(daily_data, mask)

#         # Chunk BEFORE any large where/merge operation.
#         spatial_chunks = self._spatial_chunks(daily_data, max(1, min(4, len(years))))
#         daily_lazy = self._maybe_chunk(
#             daily_data, {"T": 366, **spatial_chunks}
#         )

#         y1 = int(years[0])
#         y2 = int(years[-1])

#         le5 = daily_lazy.sel(
#             T=slice(f"{y1}-01-01", f"{y2}-12-31")
#         )
#         gt6 = daily_lazy.sel(
#             T=slice(f"{y1}-08-01", f"{y2 + 1}-07-31")
#         )

#         m_len = min(le5.sizes["T"], gt6.sizes["T"])
#         le5 = le5.isel(T=slice(0, m_len))
#         gt6 = gt6.isel(T=slice(0, m_len)).assign_coords(T=le5["T"])

#         shifted = xr.where(
#             mask <= 5,
#             le5,
#             xr.where(mask >= 6, gt6, np.nan),
#         )

#         return shifted, mask, years

#     def format_grid_output(self, res_xr, mask):
#         return xr.where(
#             mask >= 6,
#             res_xr + self.SHIFT_OFFSET,
#             res_xr + 1,
#         )

#     def revert_grid_index(self, res_xr, mask):
#         return xr.where(
#             mask >= 6,
#             res_xr - self.SHIFT_OFFSET,
#             res_xr - 1,
#         )

#     # ------------------------------------------------------------------
#     # Station / CPT-CDT path
#     # ------------------------------------------------------------------

#     @staticmethod
#     def _read_cdt_table(df_raw):
#         """
#         Normalize the historical CDT-like input layout while preserving the
#         original convention used by the previous module.
#         """
#         df = df_raw.copy()

#         if "ID" in df.columns or str(df.columns[0]).upper() == "ID":
#             header_row = pd.DataFrame([df.columns])
#             header_row.columns = range(df.shape[1])
#             df.columns = range(df.shape[1])
#             df = pd.concat([header_row, df], ignore_index=True)

#         return df

#     def _station_valid_years(self, tmin, tmax, shifted):
#         candidates = range(tmin.year - 1, tmax.year + 1)
#         years = []

#         for y in candidates:
#             start, end = self._season_bounds(y, shifted)
#             if start >= tmin and end <= tmax:
#                 years.append(int(y))

#         return years

#     def transform_and_shift_cdt(self, df_raw, map_reclassified):
#         """
#         Convert CPT/CDT-like daily station data to a long seasonal table.

#         Unlike the old implementation, austral Aug-Jul data are explicitly
#         tagged with their agricultural start year.
#         """
#         df = self._read_cdt_table(df_raw)

#         if df.shape[0] < 5 or df.shape[1] < 2:
#             raise ValueError("daily station table is too small or malformed.")

#         station_ids = pd.Series(df.iloc[0, 1:].values.astype(str))
#         station_ids = station_ids.where(
#             ~station_ids.duplicated(),
#             station_ids
#             + "_"
#             + station_ids.groupby(station_ids).cumcount().astype(str),
#         )
#         ids = station_ids.values
#         df.iloc[0, 1:] = ids

#         lons = pd.to_numeric(df.iloc[1, 1:], errors="coerce").to_numpy(float)
#         lats = pd.to_numeric(df.iloc[2, 1:], errors="coerce").to_numpy(float)

#         dates = pd.to_datetime(
#             df.iloc[4:, 0],
#             format="%Y%m%d",
#             errors="coerce",
#         )

#         valid_time = ~dates.isna()
#         dates = pd.DatetimeIndex(dates[valid_time])

#         values = (
#             df.iloc[4:, 1:]
#             .loc[valid_time]
#             .apply(pd.to_numeric, errors="coerce")
#             .to_numpy(float)
#         )

#         values[values == -99.0] = np.nan

#         da = xr.DataArray(
#             values,
#             coords={"T": dates, "station": ids},
#             dims=("T", "station"),
#         ).sortby("T")

#         if da.sizes["T"] == 0:
#             raise ValueError("No valid daily dates found in station input.")

#         # Use a 2-D zone map for station lookup.
#         zone_map = map_reclassified
#         for dim in list(zone_map.dims):
#             if dim in ("Y", "X"):
#                 continue
#             if zone_map.sizes[dim] != 1:
#                 raise ValueError(
#                     "map_reclassified must be spatial (Y, X) for station lookup."
#                 )
#             zone_map = zone_map.isel({dim: 0}, drop=True)

#         stn_zones = {}
#         frames = []

#         tmin = pd.Timestamp(da["T"].values[0])
#         tmax = pd.Timestamp(da["T"].values[-1])

#         for i, stn in enumerate(ids):
#             try:
#                 z = (
#                     zone_map
#                     .sel(X=lons[i], Y=lats[i], method="nearest")
#                     .values
#                     .item()
#                 )
#             except Exception:
#                 z = np.nan

#             stn_zones[stn] = z

#             if pd.isna(z) or int(z) not in self.criteria:
#                 continue

#             shifted = self._is_shifted(z)
#             years = self._station_valid_years(tmin, tmax, shifted)
#             series = da.isel(station=i)

#             for year in years:
#                 start, end = self._season_bounds(year, shifted)
#                 expected = pd.date_range(start, end, freq="D")

#                 vals = (
#                     series.sel(T=slice(start, end))
#                     .reindex(T=expected)
#                     .values
#                 )

#                 frames.append(
#                     pd.DataFrame(
#                         {
#                             "DATE": expected,
#                             "STATION": stn,
#                             "VALUE": vals,
#                             "LAT": lats[i],
#                             "LON": lons[i],
#                             "zonename": z,
#                             "year": int(year),
#                         }
#                     )
#                 )

#         if frames:
#             df_long = pd.concat(frames, ignore_index=True)
#         else:
#             df_long = pd.DataFrame(
#                 columns=[
#                     "DATE", "STATION", "VALUE", "LAT",
#                     "LON", "zonename", "year",
#                 ]
#             )

#         return df_long, stn_zones, df.iloc[:4, :]

#     def _parse_cpt_to_long(self, df_cpt, val_name):
#         if df_cpt.shape[0] < 2 or df_cpt.shape[1] < 2:
#             raise ValueError("CPT indicator table is too small or malformed.")

#         lats = df_cpt.iloc[0, 1:].values
#         lons = df_cpt.iloc[1, 1:].values
#         cols = df_cpt.columns[1:].tolist()

#         first_col = df_cpt.columns[0]
#         df = (
#             df_cpt.iloc[2:]
#             .copy()
#             .reset_index(drop=True)
#             .rename(columns={first_col: "year"})
#         )

#         df = df.melt(
#             id_vars=["year"],
#             var_name="station",
#             value_name=val_name,
#         )

#         df["year"] = pd.to_numeric(df["year"], errors="coerce")

#         lat_map = dict(zip(cols, lats))
#         lon_map = dict(zip(cols, lons))
#         df["lat"] = df["station"].map(lat_map)
#         df["lon"] = df["station"].map(lon_map)

#         return df

#     def build_cpt_output(self, res_df, val_col):
#         if res_df.empty:
#             return pd.DataFrame(columns=["STATION"])

#         res_df = res_df.copy()
#         res_df[val_col] = res_df[val_col].fillna(-999.0)

#         piv = res_df.pivot(
#             index="year",
#             columns="station",
#             values=val_col,
#         )

#         meta = res_df.groupby("station")[["lat", "lon"]].first()

#         lats = meta.loc[piv.columns, "lat"].tolist()
#         lons = meta.loc[piv.columns, "lon"].tolist()

#         lat_row = pd.DataFrame(
#             [lats], columns=piv.columns, index=["LAT"]
#         )
#         lon_row = pd.DataFrame(
#             [lons], columns=piv.columns, index=["LON"]
#         )

#         final = (
#             pd.concat([lat_row, lon_row, piv])
#             .reset_index()
#             .rename(columns={"index": "STATION"})
#         )
#         final.columns.name = None

#         return final

#     # ------------------------------------------------------------------
#     # Annual indicator helpers
#     # ------------------------------------------------------------------

#     def _prepare_grid_compute(self, daily_data, map_rec, nb_cores):
#         daily_data = self._validate_daily_grid(daily_data)
#         mask = self._prepare_mask(daily_data, map_rec)
#         years = self._valid_season_years(daily_data, mask)

#         # Warn if the user already loaded a very large NumPy-backed cube.
#         if daily_data.chunks is None and daily_data.nbytes > 1_000_000_000:
#             warnings.warn(
#                 "daily_data is already an in-memory NumPy-backed array larger "
#                 "than 1 GB. The CEAC algorithm will avoid additional historical "
#                 "3-D copies, but for best memory behavior open the source with "
#                 "xr.open_dataarray(..., chunks=...) or xr.open_dataset(..., chunks=...).",
#                 RuntimeWarning,
#             )

#         _get_compute_client(nb_cores)
#         return daily_data, mask, years

#     @staticmethod
#     def _select_annual_indicator(indicator, year):
#         if "T" not in indicator.dims:
#             return indicator.squeeze(drop=True)

#         if np.issubdtype(indicator["T"].dtype, np.datetime64):
#             out = indicator.sel(T=str(int(year)))
#         else:
#             try:
#                 out = indicator.sel(T=int(year))
#             except Exception:
#                 out = indicator.sel(T=str(int(year)))

#         if out.sizes.get("T", 1) != 1:
#             raise ValueError(
#                 f"Expected exactly one annual indicator value for year {year}, "
#                 f"found {out.sizes.get('T', 0)}."
#             )

#         return out.squeeze(drop=True)

#     def _prepare_annual_index_field(self, indicator, mask, year):
#         field = self._select_annual_indicator(indicator, year)
#         field = field.reindex_like(mask, method="nearest")
#         return self.revert_grid_index(field, mask)

#     def _compute_grid_spell_year(
#         self,
#         daily_data,
#         mask,
#         year,
#         onset_field,
#         cessation_field,
#         nb_cores,
#         func,
#         *,
#         kwargs,
#         output_dtype=np.float32,
#     ):
#         spatial_chunks = self._spatial_chunks(daily_data, nb_cores)
#         regime_results = {}

#         for shifted, label in ((False, "north"), (True, "south")):
#             if not self._mask_has_regime(mask, shifted):
#                 regime_results[label] = None
#                 continue

#             zone_condition = (mask >= 6) if shifted else (mask <= 5)

#             season = self._season_for_year(
#                 daily_data, year, shifted
#             ).where(zone_condition)

#             season = self._maybe_chunk(
#                 season, {"T": -1, **spatial_chunks}
#             )

#             o = self._maybe_chunk(onset_field.where(zone_condition), spatial_chunks)
#             c = self._maybe_chunk(cessation_field.where(zone_condition), spatial_chunks)

#             res = xr.apply_ufunc(
#                 func,
#                 season,
#                 o,
#                 c,
#                 input_core_dims=[("T",), (), ()],
#                 output_core_dims=[()],
#                 vectorize=True,
#                 dask="parallelized",
#                 output_dtypes=[output_dtype],
#                 kwargs=kwargs,
#             ).compute()

#             regime_results[label] = res.astype(output_dtype, copy=False)

#         return self._combine_regimes(
#             regime_results.get("north"),
#             regime_results.get("south"),
#             mask,
#         )


# # =============================================================================
# # Onset
# # =============================================================================

# class CEAC_compute_onset(CAF_AgroClimateBase):

#     @staticmethod
#     def onset_function(
#         x,
#         idebut,
#         cumul,
#         nbsec,
#         jour_pluvieux,
#         irch_fin,
#     ):
#         if not (
#             np.any(np.isfinite(x))
#             and np.isfinite(idebut)
#             and np.isfinite(cumul)
#             and np.isfinite(nbsec)
#             and np.isfinite(jour_pluvieux)
#             and np.isfinite(irch_fin)
#         ):
#             return np.nan

#         x = np.asarray(x)
#         idebut = int(idebut)
#         nbsec = int(nbsec)
#         irch_fin = int(irch_fin)

#         if len(x) < 3:
#             return np.nan

#         if idebut < -1:
#             idebut = -1

#         if idebut >= len(x) - 1:
#             return np.nan

#         irch_fin = min(irch_fin, len(x) - 2)

#         if irch_fin <= idebut:
#             return np.nan

#         idate = idebut
#         trouv = 0

#         while True:
#             idate += 1

#             if idate >= len(x) - 1:
#                 return np.nan

#             if (
#                 pd.isna(x[idate - 1])
#                 or pd.isna(x[idate])
#                 or pd.isna(x[idate + 1])
#             ):
#                 return np.nan

#             if idate > irch_fin:
#                 return random.randint(
#                     max(0, irch_fin - 5),
#                     max(0, irch_fin),
#                 )

#             c1 = x[idate - 1]
#             c2 = x[idate - 1] + x[idate]
#             c3 = x[idate - 1] + x[idate] + x[idate + 1]

#             if c1 >= cumul or c2 >= cumul or c3 >= cumul:
#                 arr = np.array(
#                     [x[idate - 1], x[idate], x[idate + 1]]
#                 )
#                 ideb = [idate - 1, idate, idate + 1][np.argmax(arr)]
#                 trouv = 1

#                 pluie30 = (
#                     x[ideb: ideb + 31]
#                     if ideb + 30 < len(x)
#                     else x[ideb:]
#                 )

#                 isec = 0

#                 while True:
#                     isec += 1

#                     if isec + nbsec >= len(pluie30):
#                         break

#                     if (
#                         np.sum(
#                             pluie30[isec: isec + nbsec + 1]
#                             < jour_pluvieux
#                         )
#                         == (nbsec + 1)
#                     ):
#                         trouv = 0
#                         break

#                     if isec == (30 - nbsec):
#                         break

#             if trouv == 1:
#                 return float(ideb)

#     def compute_insitu(self, daily_df_raw, map_rec):
#         df_long, zones, _ = self.transform_and_shift_cdt(
#             daily_df_raw, map_rec
#         )

#         res = []

#         for (stn, y), group in df_long.groupby(
#             ["STATION", "year"], sort=True
#         ):
#             z = zones.get(stn, np.nan)

#             if pd.isna(z) or int(z) not in self.criteria:
#                 v = np.nan
#             else:
#                 z = int(z)
#                 c = self.criteria[z]
#                 v = self.onset_function(
#                     group["VALUE"].to_numpy(),
#                     self.get_index_for_station(y, c["start_search"], z),
#                     c["cumulative"],
#                     c["number_dry_days"],
#                     c["thrd_rain_day"],
#                     self.get_index_for_station(y, c["end_search"], z),
#                 )
#                 v = self.output_format_value(v, z)

#             res.append(
#                 {
#                     "year": int(y),
#                     "station": stn,
#                     "lat": group["LAT"].iloc[0],
#                     "lon": group["LON"].iloc[0],
#                     "onset": v,
#                 }
#             )

#         return self.build_cpt_output(pd.DataFrame(res), "onset")

#     def compute(self, daily_data, map_rec, nb_cores):
#         daily_data, mask, years = self._prepare_grid_compute(
#             daily_data, map_rec, nb_cores
#         )

#         out = []

#         for year in years:
#             res = self._compute_grid_indicator_year(
#                 daily_data,
#                 mask,
#                 int(year),
#                 nb_cores,
#                 self.onset_function,
#                 (
#                     "start_search",
#                     "cumulative",
#                     "number_dry_days",
#                     "thrd_rain_day",
#                     "end_search",
#                 ),
#             )
#             out.append(res)

#         return self._finalize_annual_grid(
#             out,
#             years,
#             mask,
#             name="Onset",
#             format_index=True,
#         )


# # =============================================================================
# # Cessation
# # =============================================================================

# class CEAC_compute_cessation(CAF_AgroClimateBase):

#     @staticmethod
#     def cessation_function(
#         x,
#         ijour,
#         idebut,
#         ETP,
#         Cap,
#         irch_fin,
#     ):
#         if not (
#             np.isfinite(x).any()
#             and np.isfinite(idebut)
#             and np.isfinite(ijour)
#             and np.isfinite(ETP)
#             and np.isfinite(Cap)
#             and np.isfinite(irch_fin)
#         ):
#             return np.nan

#         x = np.asarray(x)
#         n = len(x)

#         ijour = max(0, int(ijour))
#         ifin = int(idebut)
#         irch_fin = min(int(irch_fin), n - 1)

#         if n == 0 or ijour >= n or ifin < 0:
#             return np.nan

#         ifin = min(ifin, n - 1)

#         if ijour > ifin or irch_fin < ifin:
#             return np.nan

#         ru = 0.0

#         for k in range(ijour, ifin + 1):
#             if np.isfinite(x[k]):
#                 ru = max(0.0, min(ru + x[k] - ETP, Cap))

#         while ifin < irch_fin:
#             ifin += 1

#             if ifin >= n:
#                 break

#             if not np.isfinite(x[ifin]):
#                 continue

#             ru = max(0.0, min(ru + x[ifin] - ETP, Cap))

#             if ru <= 0:
#                 break

#         if ifin <= irch_fin:
#             return float(ifin)

#         return float(
#             random.randint(
#                 max(0, irch_fin - 5),
#                 max(0, irch_fin),
#             )
#         )

#     def compute_insitu(self, daily_df_raw, map_rec):
#         df_long, zones, _ = self.transform_and_shift_cdt(
#             daily_df_raw, map_rec
#         )

#         res = []

#         for (stn, y), group in df_long.groupby(
#             ["STATION", "year"], sort=True
#         ):
#             z = zones.get(stn, np.nan)

#             if pd.isna(z) or int(z) not in self.criteria:
#                 v = np.nan
#             else:
#                 z = int(z)
#                 c = self.criteria[z]

#                 v = self.cessation_function(
#                     group["VALUE"].to_numpy(),
#                     self.get_index_for_station(
#                         y, c["date_dry_soil"], z
#                     ),
#                     self.get_index_for_station(
#                         y, c["start_search2"], z
#                     ),
#                     c["ETP"],
#                     c["Cap_ret_maxi"],
#                     self.get_index_for_station(
#                         y, c["end_search2"], z
#                     ),
#                 )

#                 v = self.output_format_value(v, z)

#             res.append(
#                 {
#                     "year": int(y),
#                     "station": stn,
#                     "lat": group["LAT"].iloc[0],
#                     "lon": group["LON"].iloc[0],
#                     "cessation": v,
#                 }
#             )

#         return self.build_cpt_output(
#             pd.DataFrame(res), "cessation"
#         )

#     def compute(self, daily_data, map_rec, nb_cores):
#         daily_data, mask, years = self._prepare_grid_compute(
#             daily_data, map_rec, nb_cores
#         )

#         out = []

#         for year in years:
#             res = self._compute_grid_indicator_year(
#                 daily_data,
#                 mask,
#                 int(year),
#                 nb_cores,
#                 self.cessation_function,
#                 (
#                     "date_dry_soil",
#                     "start_search2",
#                     "ETP",
#                     "Cap_ret_maxi",
#                     "end_search2",
#                 ),
#             )
#             out.append(res)

#         return self._finalize_annual_grid(
#             out,
#             years,
#             mask,
#             name="Cessation",
#             format_index=True,
#         )


# # =============================================================================
# # Dry spell after onset
# # =============================================================================

# class CEAC_compute_onset_dry_spell(CAF_AgroClimateBase):

#     @staticmethod
#     def ds_onset_func(
#         x,
#         idebut,
#         cumul,
#         nbsec,
#         jp,
#         irch_fin,
#         nbjour,
#     ):
#         if not (
#             np.any(np.isfinite(x))
#             and np.isfinite(idebut)
#             and np.isfinite(nbjour)
#         ):
#             return np.nan

#         deb = CEAC_compute_onset.onset_function(
#             x,
#             idebut,
#             cumul,
#             nbsec,
#             jp,
#             irch_fin,
#         )

#         if np.isnan(deb):
#             return np.nan

#         start = int(deb)
#         stop = min(start + int(nbjour) + 1, len(x))
#         p = np.asarray(x)[start:stop]

#         if p.size == 0:
#             return np.nan

#         r = np.where(p > jp)[0]
#         d1 = np.array([0] + list(r))
#         d2 = np.array(list(r) + [len(p)])

#         return float(np.max(d2 - d1) - 1)

#     def compute_insitu(self, daily_df_raw, map_rec):
#         df_long, zones, _ = self.transform_and_shift_cdt(
#             daily_df_raw, map_rec
#         )

#         res = []

#         for (stn, y), group in df_long.groupby(
#             ["STATION", "year"], sort=True
#         ):
#             z = zones.get(stn, np.nan)

#             if pd.isna(z) or int(z) not in self.criteria:
#                 v = np.nan
#             else:
#                 z = int(z)
#                 c = self.criteria[z]

#                 v = self.ds_onset_func(
#                     group["VALUE"].to_numpy(),
#                     self.get_index_for_station(
#                         y, c["start_search"], z
#                     ),
#                     c["cumulative"],
#                     c["number_dry_days"],
#                     c["thrd_rain_day"],
#                     self.get_index_for_station(
#                         y, c["end_search"], z
#                     ),
#                     c["nbjour"],
#                 )

#             res.append(
#                 {
#                     "year": int(y),
#                     "station": stn,
#                     "lat": group["LAT"].iloc[0],
#                     "lon": group["LON"].iloc[0],
#                     "onsetdryspell": v,
#                 }
#             )

#         return self.build_cpt_output(
#             pd.DataFrame(res), "onsetdryspell"
#         )

#     def compute(self, daily_data, map_rec, nb_cores):
#         daily_data, mask, years = self._prepare_grid_compute(
#             daily_data, map_rec, nb_cores
#         )

#         out = []

#         for year in years:
#             res = self._compute_grid_indicator_year(
#                 daily_data,
#                 mask,
#                 int(year),
#                 nb_cores,
#                 self.ds_onset_func,
#                 (
#                     "start_search",
#                     "cumulative",
#                     "number_dry_days",
#                     "thrd_rain_day",
#                     "end_search",
#                     "nbjour",
#                 ),
#             )
#             out.append(res)

#         return self._finalize_annual_grid(
#             out,
#             years,
#             mask,
#             name="Onset_dryspell",
#             format_index=False,
#         )


# # =============================================================================
# # Dry spell between post-onset window and cessation
# # =============================================================================

# class CEAC_compute_cessation_dry_spell(CAF_AgroClimateBase):

#     @staticmethod
#     def ds_cess_func(
#         x,
#         id1,
#         cum,
#         nbs,
#         jp,
#         ir1,
#         id2,
#         ijd,
#         ETP,
#         Cap,
#         ir2,
#         nbj,
#     ):
#         if not (
#             np.any(np.isfinite(x))
#             and np.isfinite(id1)
#             and np.isfinite(nbj)
#         ):
#             return np.nan

#         deb = CEAC_compute_onset.onset_function(
#             x, id1, cum, nbs, jp, ir1
#         )

#         if pd.isna(deb):
#             return np.nan

#         fin = CEAC_compute_cessation.cessation_function(
#             x, ijd, id2, ETP, Cap, ir2
#         )

#         if pd.isna(fin):
#             return np.nan

#         start = int(deb + nbj)
#         stop = int(fin)

#         if (stop - start) <= 0 or start >= len(x):
#             return np.nan

#         p = np.asarray(x)[start:stop]
#         r = np.where(p > jp)[0]

#         if len(r) == 0:
#             return np.nan

#         return float(
#             np.max(
#                 np.array(list(r) + [len(p)])
#                 - np.array([0] + list(r))
#             )
#             - 1
#         )

#     def compute_insitu(self, daily_df_raw, map_rec):
#         df_long, zones, _ = self.transform_and_shift_cdt(
#             daily_df_raw, map_rec
#         )

#         res = []

#         for (stn, y), group in df_long.groupby(
#             ["STATION", "year"], sort=True
#         ):
#             z = zones.get(stn, np.nan)

#             if pd.isna(z) or int(z) not in self.criteria:
#                 v = np.nan
#             else:
#                 z = int(z)
#                 c = self.criteria[z]

#                 v = self.ds_cess_func(
#                     group["VALUE"].to_numpy(),
#                     self.get_index_for_station(
#                         y, c["start_search"], z
#                     ),
#                     c["cumulative"],
#                     c["number_dry_days"],
#                     c["thrd_rain_day"],
#                     self.get_index_for_station(
#                         y, c["end_search"], z
#                     ),
#                     self.get_index_for_station(
#                         y, c["start_search2"], z
#                     ),
#                     self.get_index_for_station(
#                         y, c["date_dry_soil"], z
#                     ),
#                     c["ETP"],
#                     c["Cap_ret_maxi"],
#                     self.get_index_for_station(
#                         y, c["end_search2"], z
#                     ),
#                     c["nbjour"],
#                 )

#             res.append(
#                 {
#                     "year": int(y),
#                     "station": stn,
#                     "lat": group["LAT"].iloc[0],
#                     "lon": group["LON"].iloc[0],
#                     "cessation_dryspell": v,
#                 }
#             )

#         return self.build_cpt_output(
#             pd.DataFrame(res), "cessation_dryspell"
#         )

#     def compute(self, daily_data, map_rec, nb_cores):
#         daily_data, mask, years = self._prepare_grid_compute(
#             daily_data, map_rec, nb_cores
#         )

#         out = []

#         for year in years:
#             res = self._compute_grid_indicator_year(
#                 daily_data,
#                 mask,
#                 int(year),
#                 nb_cores,
#                 self.ds_cess_func,
#                 (
#                     "start_search",
#                     "cumulative",
#                     "number_dry_days",
#                     "thrd_rain_day",
#                     "end_search",
#                     "start_search2",
#                     "date_dry_soil",
#                     "ETP",
#                     "Cap_ret_maxi",
#                     "end_search2",
#                     "nbjour",
#                 ),
#             )
#             out.append(res)

#         return self._finalize_annual_grid(
#             out,
#             years,
#             mask,
#             name="Cessation_dryspell",
#             format_index=False,
#         )


# # =============================================================================
# # Dry-spell counts
# # =============================================================================

# class CEAC_count_dry_spells(CAF_AgroClimateBase):

#     @staticmethod
#     def count_dry_spells(
#         x,
#         onset,
#         cessation,
#         d_len,
#         thresh,
#     ):
#         if not (
#             np.isfinite(x).any()
#             and np.isfinite(onset)
#             and np.isfinite(cessation)
#         ):
#             return np.nan

#         x = np.asarray(x)
#         o = int(onset)
#         c = int(cessation)

#         if o < 0 or c < 0 or o >= len(x):
#             return np.nan

#         c = min(c, len(x) - 1)

#         if c < o:
#             return np.nan

#         count = 0
#         cur = 0

#         # Preserve the previous module's EXACT-length semantics.
#         for day in range(o, c + 1):
#             if np.isfinite(x[day]) and x[day] < thresh:
#                 cur += 1
#             else:
#                 if cur == int(d_len):
#                     count += 1
#                 cur = 0

#         if cur == int(d_len):
#             count += 1

#         return float(count)

#     def compute_insitu(
#         self,
#         daily_raw,
#         on_cpt,
#         cess_cpt,
#         map_rec,
#         d_len,
#         thresh=1.0,
#     ):
#         df_long, zones, _ = self.transform_and_shift_cdt(
#             daily_raw, map_rec
#         )

#         m = pd.merge(
#             self._parse_cpt_to_long(on_cpt, "o"),
#             self._parse_cpt_to_long(cess_cpt, "c"),
#             on=["station", "year"],
#             suffixes=("_o", "_c"),
#         )

#         res = []

#         for (stn, y), group in df_long.groupby(
#             ["STATION", "year"], sort=True
#         ):
#             sub = m[
#                 (m["station"] == stn)
#                 & (m["year"] == y)
#             ]

#             z = zones.get(stn, np.nan)

#             if pd.isna(z) or sub.empty:
#                 v = np.nan
#             else:
#                 z = int(z)

#                 o_idx = self.revert_to_index(
#                     sub["o"].values[0], z
#                 )
#                 c_idx = self.revert_to_index(
#                     sub["c"].values[0], z
#                 )

#                 v = self.count_dry_spells(
#                     group["VALUE"].to_numpy(),
#                     o_idx,
#                     c_idx,
#                     d_len,
#                     thresh,
#                 )

#             lat_val = (
#                 sub["lat_o"].values[0]
#                 if not sub.empty
#                 else group["LAT"].iloc[0]
#             )
#             lon_val = (
#                 sub["lon_o"].values[0]
#                 if not sub.empty
#                 else group["LON"].iloc[0]
#             )

#             res.append(
#                 {
#                     "year": int(y),
#                     "station": stn,
#                     "lat": lat_val,
#                     "lon": lon_val,
#                     "dry_spells": v,
#                 }
#             )

#         return self.build_cpt_output(
#             pd.DataFrame(res), "dry_spells"
#         )

#     def compute(
#         self,
#         daily_data,
#         on_da,
#         cess_da,
#         map_rec,
#         d_len,
#         thresh,
#         nb_cores,
#     ):
#         daily_data, mask, years = self._prepare_grid_compute(
#             daily_data, map_rec, nb_cores
#         )

#         out = []

#         for year in years:
#             onset_field = self._prepare_annual_index_field(
#                 on_da, mask, int(year)
#             )
#             cessation_field = self._prepare_annual_index_field(
#                 cess_da, mask, int(year)
#             )

#             res = self._compute_grid_spell_year(
#                 daily_data,
#                 mask,
#                 int(year),
#                 onset_field,
#                 cessation_field,
#                 nb_cores,
#                 self.count_dry_spells,
#                 kwargs={
#                     "d_len": d_len,
#                     "thresh": thresh,
#                 },
#             )

#             out.append(res)

#         return self._finalize_annual_grid(
#             out,
#             years,
#             mask,
#             name="Count_dryspell",
#             format_index=False,
#         )


# # =============================================================================
# # Wet-spell counts
# # =============================================================================

# class CEAC_count_wet_spells(CAF_AgroClimateBase):

#     @staticmethod
#     def count_wet_spells(
#         x,
#         onset,
#         cessation,
#         w_len,
#         thresh,
#     ):
#         if not (
#             np.isfinite(x).any()
#             and np.isfinite(onset)
#             and np.isfinite(cessation)
#         ):
#             return np.nan

#         x = np.asarray(x)
#         o = int(onset)
#         c = int(cessation)

#         if o < 0 or c < 0 or o >= len(x):
#             return np.nan

#         c = min(c, len(x) - 1)

#         if c < o:
#             return np.nan

#         count = 0
#         cur = 0

#         # Preserve the previous module's EXACT-length semantics.
#         for day in range(o, c + 1):
#             if np.isfinite(x[day]) and x[day] >= thresh:
#                 cur += 1
#             else:
#                 if cur == int(w_len):
#                     count += 1
#                 cur = 0

#         if cur == int(w_len):
#             count += 1

#         return float(count)

#     def compute_insitu(
#         self,
#         daily_raw,
#         on_cpt,
#         cess_cpt,
#         map_rec,
#         w_len,
#         thresh=1.0,
#     ):
#         df_long, zones, _ = self.transform_and_shift_cdt(
#             daily_raw, map_rec
#         )

#         m = pd.merge(
#             self._parse_cpt_to_long(on_cpt, "o"),
#             self._parse_cpt_to_long(cess_cpt, "c"),
#             on=["station", "year"],
#             suffixes=("_o", "_c"),
#         )

#         res = []

#         for (stn, y), group in df_long.groupby(
#             ["STATION", "year"], sort=True
#         ):
#             sub = m[
#                 (m["station"] == stn)
#                 & (m["year"] == y)
#             ]

#             z = zones.get(stn, np.nan)

#             if pd.isna(z) or sub.empty:
#                 v = np.nan
#             else:
#                 z = int(z)

#                 o_idx = self.revert_to_index(
#                     sub["o"].values[0], z
#                 )
#                 c_idx = self.revert_to_index(
#                     sub["c"].values[0], z
#                 )

#                 v = self.count_wet_spells(
#                     group["VALUE"].to_numpy(),
#                     o_idx,
#                     c_idx,
#                     w_len,
#                     thresh,
#                 )

#             lat_val = (
#                 sub["lat_o"].values[0]
#                 if not sub.empty
#                 else group["LAT"].iloc[0]
#             )
#             lon_val = (
#                 sub["lon_o"].values[0]
#                 if not sub.empty
#                 else group["LON"].iloc[0]
#             )

#             res.append(
#                 {
#                     "year": int(y),
#                     "station": stn,
#                     "lat": lat_val,
#                     "lon": lon_val,
#                     "wet_spells": v,
#                 }
#             )

#         return self.build_cpt_output(
#             pd.DataFrame(res), "wet_spells"
#         )

#     def compute(
#         self,
#         daily_data,
#         on_da,
#         cess_da,
#         map_rec,
#         w_len,
#         thresh,
#         nb_cores,
#     ):
#         daily_data, mask, years = self._prepare_grid_compute(
#             daily_data, map_rec, nb_cores
#         )

#         out = []

#         for year in years:
#             onset_field = self._prepare_annual_index_field(
#                 on_da, mask, int(year)
#             )
#             cessation_field = self._prepare_annual_index_field(
#                 cess_da, mask, int(year)
#             )

#             res = self._compute_grid_spell_year(
#                 daily_data,
#                 mask,
#                 int(year),
#                 onset_field,
#                 cessation_field,
#                 nb_cores,
#                 self.count_wet_spells,
#                 kwargs={
#                     "w_len": w_len,
#                     "thresh": thresh,
#                 },
#             )

#             out.append(res)

#         return self._finalize_annual_grid(
#             out,
#             years,
#             mask,
#             name="Count_wetspell",
#             format_index=False,
#         )


# __all__ = [
#     "DEFAULT_CRITERIA",
#     "CAF_AgroClimateBase",
#     "CEAC_compute_onset",
#     "CEAC_compute_cessation",
#     "CEAC_compute_onset_dry_spell",
#     "CEAC_compute_cessation_dry_spell",
#     "CEAC_count_dry_spells",
#     "CEAC_count_wet_spells",
#     "close_shared_client",
# ]




# """CEAC agroclimatic indicator computation.

# Refactored, composable implementation of onset, cessation, and spell
# counting for both station (CPT/CDT format) and gridded data.

# Base class
# ----------
# CAF_AgroClimateBase
#     Shared helpers for date-index arithmetic, zone-specific criteria
#     mapping, CPT/CDT input parsing, and output formatting.

# Indicator classes
# -----------------
# CEAC_compute_onset
#     Season onset detection.
# CEAC_compute_cessation
#     Season cessation detection.
# CEAC_compute_onset_dry_spell
#     Onset with a post-onset dry-spell check.
# CEAC_compute_cessation_dry_spell
#     Cessation with an additional dry-spell criterion.
# CEAC_count_dry_spells
#     Count of dry spells of at least *d_len* days within the growing season.
# CEAC_count_wet_spells
#     Count of wet spells of at least *w_len* days within the growing season.

# Referential note
# ----------------
# All criteria dates in DEFAULT_CRITERIA are REAL calendar dates. The user
# never needs to know about the internal seasonal shift applied to the
# austral zones (>= 6), whose growing season straddles two civil years.
# That shift is handled entirely inside get_index_for_station (station path)
# and _map_criteria (gridded path), which both convert a real date into an
# index in the (already) shifted series in the exact same way.
# """
# import pandas as pd
# import xarray as xr
# import numpy as np
# import random
# import datetime
# from dask.distributed import Client

# # ---------------------------------------------------------------------------
# # Shared Dask client: created once per process and reused across ALL calls,
# # instead of spinning up (and tearing down) a fresh LocalCluster on every
# # compute() — which caused nanny-join hangs at close() and worker
# # oversubscription. Closed once, at interpreter exit. Reuses (via get_client)
# # any client already created by another wass2s module, so the whole process
# # shares a single cluster.
# # ---------------------------------------------------------------------------
# _SHARED_CLIENT = None


# def _bounded_close(client):
#     """Close a client (and its LocalCluster) with a bounded timeout, swallowing
#     teardown errors. Used only at interpreter exit."""
#     if client is None:
#         return
#     import contextlib
#     cluster = getattr(client, "cluster", None)
#     with contextlib.suppress(Exception):
#         client.close(timeout=10)
#     if cluster is not None:
#         with contextlib.suppress(Exception):
#             cluster.close(timeout=10)


# def _safe_close_client(client):
#     """Intentional no-op. Every client here comes from _get_compute_client and
#     is process-wide and reused; it must NOT be torn down per call. The shared
#     client is closed exactly once, at interpreter exit, by the atexit hook in
#     _get_compute_client (or reaped by the OS/Slurm cgroup on os._exit)."""
#     return


# def _get_compute_client(nb_cores):
#     """Return a process-wide Dask client, created lazily on first use and reused
#     afterwards. Returns None if a distributed client cannot be created, in which
#     case .compute() falls back to dask's default (threaded) scheduler."""
#     global _SHARED_CLIENT
#     try:
#         from dask.distributed import Client, LocalCluster, get_client
#         try:
#             return get_client()
#         except Exception:
#             pass
#         if _SHARED_CLIENT is not None:
#             try:
#                 if _SHARED_CLIENT.status == "running":
#                     return _SHARED_CLIENT
#             except Exception:
#                 _SHARED_CLIENT = None
#         cluster = LocalCluster(
#             n_workers=max(1, int(nb_cores)),
#             threads_per_worker=1,
#             processes=True,
#             dashboard_address=None,
#             memory_limit=0,
#         )
#         _SHARED_CLIENT = Client(cluster)
#         import atexit
#         atexit.register(_bounded_close, _SHARED_CLIENT)
#         return _SHARED_CLIENT
#     except Exception:
#         return None

# import warnings

# warnings.filterwarnings('ignore')

# # All dates below are REAL calendar dates. Zones >= 6 (austral) have a season
# # spanning two civil years; the shift that folds it into a single contiguous
# # block is applied internally, not encoded here.
# DEFAULT_CRITERIA = {
#     1: {"start_search": "05-01", "start_search2": "09-01", "date_dry_soil": "01-01", "cumulative": 15, "number_dry_days": 15, "thrd_rain_day": 0.85, "end_search": "08-30",  "end_search2": "10-30", "nbjour": 35, "ETP": 5.0, "Cap_ret_maxi": 70},
#     2: {"start_search": "03-15", "start_search2": "09-01", "date_dry_soil": "01-01", "cumulative": 20, "number_dry_days": 10, "thrd_rain_day": 0.85, "end_search": "08-01",  "end_search2": "11-01", "nbjour": 40, "ETP": 5.0, "Cap_ret_maxi": 70},
#     3: {"start_search": "02-01", "start_search2": "10-01", "date_dry_soil": "01-01", "cumulative": 20, "number_dry_days": 10, "thrd_rain_day": 0.85, "end_search": "05-15", "end_search2": "12-30", "nbjour": 45, "ETP": 5.0, "Cap_ret_maxi": 70},
#     4: {"start_search": "01-01", "start_search2": "11-01", "date_dry_soil": "01-01", "cumulative": 20, "number_dry_days": 7, "thrd_rain_day": 0.85, "end_search": "04-01", "end_search2": "12-30", "nbjour": 50, "ETP": 5.0, "Cap_ret_maxi": 80},
#     5: {"start_search": "01-01", "start_search2": "06-01", "date_dry_soil": "01-01", "cumulative": 25, "number_dry_days": 7, "thrd_rain_day": 0.85, "end_search": "03-10", "end_search2": "08-10", "nbjour": 50, "ETP": 4.0, "Cap_ret_maxi": 60},
#     # --- Zones australes (dates réelles ; décalage géré en interne) ---
#     6: {"start_search": "08-01", "start_search2": "04-20", "date_dry_soil": "08-01", "cumulative": 25, "number_dry_days": 7, "thrd_rain_day": 0.85, "end_search": "10-18",  "end_search2": "07-15", "nbjour": 50, "ETP": 6.0, "Cap_ret_maxi": 70},
#     7: {"start_search": "07-15", "start_search2": "03-01", "date_dry_soil": "08-01", "cumulative": 20, "number_dry_days": 10, "thrd_rain_day": 0.85, "end_search": "10-20",  "end_search2": "06-20", "nbjour": 50, "ETP": 6.0, "Cap_ret_maxi": 70},
#     8: {"start_search": "09-01", "start_search2": "03-01", "date_dry_soil": "08-01", "cumulative": 20, "number_dry_days": 10, "thrd_rain_day": 0.85, "end_search": "12-01",  "end_search2": "05-15", "nbjour": 40, "ETP": 4.0, "Cap_ret_maxi": 70},
#     9: {"start_search": "10-01", "start_search2": "03-15", "date_dry_soil": "08-01", "cumulative": 20, "number_dry_days": 15, "thrd_rain_day": 0.85, "end_search": "01-31",  "end_search2": "05-15", "nbjour": 30, "ETP": 5.0, "Cap_ret_maxi": 60},
# }

# class CAF_AgroClimateBase:
#     SHIFT_OFFSET = 244 # L'offset pour marquer l'extension

#     def __init__(self, user_criteria=None):
#         self.criteria = user_criteria if user_criteria else DEFAULT_CRITERIA

#     def _is_shifted(self, z):
#         # zone du shift
#         return not pd.isna(z) and int(z) >= 6 

#     @staticmethod
#     def day_of_year(y, mm_dd):
#         dt = datetime.datetime.strptime(f"{int(y)}-{mm_dd}", "%Y-%m-%d").date()
#         return (dt - datetime.date(int(y), 1, 1)).days + 1

#     def get_index_for_station(self, year, mm_dd, z):
#         """Convert a REAL calendar date (mm_dd) into a 0-based index in the
#         (already) shifted series for zone *z*. Shifted zones use an Aug-1 base
#         and roll dates with month < 8 into the following civil year; non-shifted
#         zones use a Jan-1 base (index == day_of_year - 1). Single source of
#         truth for both the station and the gridded path."""
#         year = int(year)
#         shifted = self._is_shifted(z)
#         base_date = datetime.date(year, 8, 1) if shifted else datetime.date(year, 1, 1)
#         target_month, target_day = map(int, mm_dd.split('-'))
#         target_year = year + 1 if (shifted and target_month < 8) else year
#         target_date = datetime.date(target_year, target_month, target_day)
#         return (target_date - base_date).days

#     def output_format_value(self, v, z):
#         if np.isnan(v): return np.nan
#         return v + self.SHIFT_OFFSET if self._is_shifted(z) else v + 1

#     def revert_to_index(self, v, z):
#         if np.isnan(v): return np.nan
#         return int(v - self.SHIFT_OFFSET) if self._is_shifted(z) else int(v - 1)

#     def _map_criteria(self, mask, key, year):
#         """Broadcast a criteria *key* across the zone mask for a given *year*.

#         For date-valued keys, the returned value is the 0-based index of that
#         REAL date in the (already) shifted series — computed by the exact same
#         get_index_for_station used on the station path, so both paths agree and
#         the shift stays invisible to the criteria dictionary. Non-date keys are
#         passed through unchanged.
#         """
#         is_date = ('search' in key or 'date' in key)
#         def _safe_get(z):
#             if np.isnan(z) or int(z) not in self.criteria:
#                 return np.nan
#             z = int(z)
#             v = self.criteria[z][key]
#             return self.get_index_for_station(int(year), v, z) if is_date else v
#         return xr.DataArray(np.vectorize(_safe_get, otypes=[float])(mask.values), coords=mask.coords)

#     def shift_gridded_data(self, daily_data, map_reclassified):
#         mask = map_reclassified.reindex_like(daily_data, method='nearest')
#         y1, y2 = int(daily_data['T'].dt.year.min()), int(daily_data['T'].dt.year.max())
        
#         #  mask <= 5 vs mask >= 6
#         le5 = daily_data.sel(T=slice(f"{y1}", f"{y2-1}")).where(mask <= 5)
#         gt6 = daily_data.sel(T=slice(f"{y1}-08-01", f"{y2}-07-31")).where(mask >= 6)
        
#         m_len = min(len(le5['T']), len(gt6['T']))
#         le5, gt6 = le5.isel(T=slice(0, m_len)), gt6.isel(T=slice(0, m_len))
#         gt6 = gt6.assign_coords(T=le5['T'].values)
#         return le5.combine_first(gt6), mask, np.unique(le5['T'].dt.year.to_numpy())

#     def format_grid_output(self, res_xr, mask):
#         return xr.where(mask >= 6, res_xr + self.SHIFT_OFFSET, res_xr + 1)

#     def revert_grid_index(self, res_xr, mask):
#         return xr.where(mask >= 6, res_xr - self.SHIFT_OFFSET, res_xr - 1)

#     def transform_and_shift_cdt(self, df_raw, map_reclassified):
#         if "ID" in df_raw.columns or str(df_raw.columns[0]).upper() == "ID":
#             header_row = pd.DataFrame([df_raw.columns])
#             header_row.columns = range(df_raw.shape[1])
#             df_raw.columns = range(df_raw.shape[1])
#             df_raw = pd.concat([header_row, df_raw], ignore_index=True)

#         s_ids = pd.Series(df_raw.iloc[0, 1:].values.astype(str))
#         ids = s_ids.where(~s_ids.duplicated(), s_ids + "_" + s_ids.groupby(s_ids).cumcount().astype(str)).values
#         df_raw.iloc[0, 1:] = ids 
        
#         lons, lats = df_raw.iloc[1, 1:].astype(float).values, df_raw.iloc[2, 1:].astype(float).values
#         dates = pd.to_datetime(df_raw.iloc[4:, 0], format='%Y%m%d')
#         da = xr.DataArray(df_raw.iloc[4:, 1:].astype(float).values, coords={'T': dates, 'station': ids}, dims=['T', 'station'])
#         y1, y2 = dates.min().year, dates.max().year
        
#         stn_zones, series_lst = {}, []
#         for i, stn in enumerate(ids):
#             try: z = map_reclassified.sel(X=lons[i], Y=lats[i], method='nearest').values.item()
#             except: z = np.nan
#             stn_zones[stn] = z
#             s = da.isel(station=i)
#             # Construction des séries continues par station
#             if self._is_shifted(z): series_lst.append(s.sel(T=slice(f"{y1}-08-01", f"{y2}-07-31")))
#             else: series_lst.append(s.sel(T=slice(f"{y1}-01-01", f"{y2-1}-12-31")))
                
#         m_len = min(len(s) for s in series_lst)
#         std_dt_vals = dates.iloc[:m_len].values  
        
#         arr_2d = np.column_stack([s.values[:m_len] for s in series_lst])
#         df_shifted = pd.DataFrame(arr_2d, index=std_dt_vals, columns=ids)
        
#         df_long = df_shifted.reset_index().melt(id_vars="index", var_name="STATION", value_name="VALUE")
#         df_long.rename(columns={"index": "DATE"}, inplace=True)
        
#         meta_df = pd.DataFrame({"STATION": ids, "LON": lons, "LAT": lats, "zonename": [stn_zones[s] for s in ids]})
#         df_long = df_long.merge(meta_df, on="STATION")
#         df_long["VALUE"] = df_long["VALUE"].replace(-99.0, np.nan)
#         df_long["year"] = df_long["DATE"].dt.year
        
#         return df_long, stn_zones, df_raw.iloc[:4, :]

#     def _parse_cpt_to_long(self, df_cpt, val_name):
#         lats, lons = df_cpt.iloc[0, 1:].values, df_cpt.iloc[1, 1:].values
#         cols = df_cpt.columns[1:].tolist()
#         df = df_cpt.iloc[2:].copy().reset_index(drop=True).rename(columns={"STATION": "year"})
#         df = df.melt(id_vars=["year"], var_name="station", value_name=val_name)
#         df["year"] = pd.to_numeric(df["year"], errors="coerce")
#         df["lat"], df["lon"] = df["station"].map(dict(zip(cols, lats))), df["station"].map(dict(zip(cols, lons)))
#         return df

#     def build_cpt_output(self, res_df, val_col):
#         res_df[val_col] = res_df[val_col].fillna(-999.0)
#         piv = res_df.pivot(index="year", columns="station", values=val_col)
#         meta = res_df.groupby("station")[["lat", "lon"]].first()
#         lats, lons = meta.loc[piv.columns, "lat"].tolist(), meta.loc[piv.columns, "lon"].tolist()
#         lat_row = pd.DataFrame([lats], columns=piv.columns, index=["LAT"])
#         lon_row = pd.DataFrame([lons], columns=piv.columns, index=["LON"])
#         final = pd.concat([lat_row, lon_row, piv]).reset_index().rename(columns={"index": "STATION"})
#         final.columns.name = None 
#         return final


# class CEAC_compute_onset(CAF_AgroClimateBase):
#     @staticmethod
#     def onset_function(x, idebut, cumul, nbsec, jour_pluvieux, irch_fin):
#         if not (np.any(np.isfinite(x)) and np.isfinite(idebut) and np.isfinite(nbsec) and np.isfinite(irch_fin)): return np.nan
#         idebut, nbsec, irch_fin = int(idebut), int(nbsec), int(irch_fin)
#         idate, trouv = idebut, 0
#         while True:
#             idate += 1
#             if idate >= len(x)-1 or pd.isna(x[idate-1]) or pd.isna(x[idate]) or pd.isna(x[idate+1]): return np.nan
#             if idate > irch_fin: return random.randint(irch_fin - 5, irch_fin)
#             c1, c2, c3 = x[idate-1], x[idate-1]+x[idate], x[idate-1]+x[idate]+x[idate+1]
#             if c1 >= cumul or c2 >= cumul or c3 >= cumul:
#                 arr = np.array([x[idate-1], x[idate], x[idate+1]])
#                 ideb = [idate-1, idate, idate+1][np.argmax(arr)]
#                 trouv = 1
#                 pluie30 = x[ideb:ideb+31] if ideb+30 < len(x) else x[ideb:]
#                 isec = 0
#                 while True:
#                     isec += 1
#                     if isec+nbsec >= len(pluie30): break
#                     if np.sum(pluie30[isec:isec+nbsec+1] < jour_pluvieux) == (nbsec + 1): trouv = 0; break
#                     if isec == (30 - nbsec): break
#             if trouv == 1: return ideb
#         return np.nan

#     def compute_insitu(self, daily_df_raw, map_rec):
#         df_long, zones, _ = self.transform_and_shift_cdt(daily_df_raw, map_rec)
#         res = []
#         for (stn, y), group in df_long.groupby(["STATION", "year"]):
#             z = zones[stn]
#             if pd.isna(z) or int(z) not in self.criteria: v = np.nan
#             else:
#                 z = int(z)
#                 c = self.criteria[z]
#                 v = self.onset_function(group["VALUE"].values, self.get_index_for_station(y, c["start_search"], z), c["cumulative"], c["number_dry_days"], c["thrd_rain_day"], self.get_index_for_station(y, c["end_search"], z))
#                 v = self.output_format_value(v, z)
#             res.append({"year": y, "station": stn, "lat": group["LAT"].iloc[0], "lon": group["LON"].iloc[0], "onset": v})
#         return self.build_cpt_output(pd.DataFrame(res), "onset")

#     def compute(self, daily_data, map_rec, nb_cores):
#         shifted, mask, years = self.shift_gridded_data(daily_data, map_rec)
#         cx, cy = int(np.round(len(shifted.X)/nb_cores)), int(np.round(len(shifted.Y)/nb_cores))
#         out = []
#         for y in years:
#             yd, mk = shifted.sel(T=str(y)), lambda k: self._map_criteria(mask, k, y)
#             client = _get_compute_client(nb_cores)
#             res = xr.apply_ufunc(
#                 self.onset_function, yd.chunk({'Y': cy, 'X': cx}),
#                 mk("start_search").chunk({'Y': cy, 'X': cx}), mk("cumulative").chunk({'Y': cy, 'X': cx}),
#                 mk("number_dry_days").chunk({'Y': cy, 'X': cx}), mk("thrd_rain_day").chunk({'Y': cy, 'X': cx}),
#                 mk("end_search").chunk({'Y': cy, 'X': cx}),
#                 input_core_dims=[('T',)]+[()]*5, vectorize=True, dask='parallelized', output_dtypes=['float']
#             ).compute()
#             _safe_close_client(client)
#             out.append(res)
            
#         unique_zone = np.unique(mask.to_numpy())
#         unique_zone = unique_zone[~np.isnan(unique_zone)]
#         # Choose a date to store results
#         if unique_zone.size == 0:
#             raise ValueError("No valid zones found in the mask.")
#         else:
#             # Use zone in low latitude
#             zone_id_to_use = int(np.min(unique_zone))
        
#         start_search_str = self.criteria[zone_id_to_use]["start_search"]        
#         final = self.format_grid_output(xr.concat(out, dim=pd.Index(years, name="T")), mask)
#         final['T'] = pd.to_datetime([f"{y}-{start_search_str}" for y in years])
#         final.name = "Onset"
#         return final


# class CEAC_compute_cessation(CAF_AgroClimateBase):
#     @staticmethod
#     def cessation_function(x, ijour, idebut, ETP, Cap, irch_fin):
#         if not (np.isfinite(x).any() and np.isfinite(idebut) and np.isfinite(ijour) and np.isfinite(ETP) and np.isfinite(Cap) and np.isfinite(irch_fin)): return np.nan
#         ru, ifin = 0, int(idebut)
#         for k in range(int(ijour), ifin + 1):
#             if not pd.isna(x[k]): ru = max(0, min(ru + x[k] - ETP, Cap))
#         while ifin < int(irch_fin):
#             ifin += 1
#             if ifin >= len(x) or pd.isna(x[ifin]): continue
#             ru = max(0, min(ru + x[ifin] - ETP, Cap))
#             if ru <= 0: break
#         return ifin if ifin <= int(irch_fin) else random.randint(int(irch_fin) - 5, int(irch_fin))

#     def compute_insitu(self, daily_df_raw, map_rec):
#         df_long, zones, _ = self.transform_and_shift_cdt(daily_df_raw, map_rec)
#         res = []
#         for (stn, y), group in df_long.groupby(["STATION", "year"]):
#             z = zones[stn]
#             if pd.isna(z) or int(z) not in self.criteria: v = np.nan
#             else:
#                 z = int(z)
#                 c = self.criteria[z]
#                 v = self.cessation_function(group["VALUE"].values, self.get_index_for_station(y, c["date_dry_soil"], z), self.get_index_for_station(y, c["start_search2"], z), c["ETP"], c["Cap_ret_maxi"], self.get_index_for_station(y, c["end_search2"], z))
#                 v = self.output_format_value(v, z)
#             res.append({"year": y, "station": stn, "lat": group["LAT"].iloc[0], "lon": group["LON"].iloc[0], "cessation": v})
#         return self.build_cpt_output(pd.DataFrame(res), "cessation")

#     def compute(self, daily_data, map_rec, nb_cores):
#         shifted, mask, years = self.shift_gridded_data(daily_data, map_rec)
#         cx, cy = int(np.round(len(shifted.X)/nb_cores)), int(np.round(len(shifted.Y)/nb_cores))
#         out = []
#         for y in years:
#             yd, mk = shifted.sel(T=str(y)), lambda k: self._map_criteria(mask, k, y)
#             client = _get_compute_client(nb_cores)
#             res = xr.apply_ufunc(
#                 self.cessation_function, yd.chunk({'Y': cy, 'X': cx}),
#                 mk("date_dry_soil").chunk({'Y': cy, 'X': cx}), mk("start_search2").chunk({'Y': cy, 'X': cx}),
#                 mk("ETP").chunk({'Y': cy, 'X': cx}), mk("Cap_ret_maxi").chunk({'Y': cy, 'X': cx}),
#                 mk("end_search2").chunk({'Y': cy, 'X': cx}),
#                 input_core_dims=[('T',)]+[()]*5, vectorize=True, dask='parallelized', output_dtypes=['float']
#             ).compute()
#             _safe_close_client(client)
#             out.append(res)

#         unique_zone = np.unique(mask.to_numpy())
#         unique_zone = unique_zone[~np.isnan(unique_zone)]
#         # Choose a date to store results
#         if unique_zone.size == 0:
#             raise ValueError("No valid zones found in the mask.")
#         else:
#             # Use zone in low latitude
#             zone_id_to_use = int(np.min(unique_zone))
        
#         start_search_str = self.criteria[zone_id_to_use]["start_search"]
        
#         final = self.format_grid_output(xr.concat(out, dim=pd.Index(years, name="T")), mask)
#         final['T'] = pd.to_datetime([f"{y}-{start_search_str}" for y in years])
#         final.name = "Cessation"
#         return final


# class CEAC_compute_onset_dry_spell(CAF_AgroClimateBase):
#     @staticmethod
#     def ds_onset_func(x, idebut, cumul, nbsec, jp, irch_fin, nbjour):
#         if not (np.any(np.isfinite(x)) and np.isfinite(idebut)): return np.nan
#         deb = CEAC_compute_onset.onset_function(x, idebut, cumul, nbsec, jp, irch_fin)
#         if not np.isnan(deb):
#             p = x[int(deb) : min(int(deb) + int(nbjour) + 1, len(x))]
#             r = np.where(p > jp)[0]
#             d1, d2 = np.array([0] + list(r)), np.array(list(r) + [len(p)])
#             return np.max(d2 - d1) - 1
#         return np.nan

#     def compute_insitu(self, daily_df_raw, map_rec):
#         df_long, zones, _ = self.transform_and_shift_cdt(daily_df_raw, map_rec)
#         res = []
#         for (stn, y), group in df_long.groupby(["STATION", "year"]):
#             z = zones[stn]
#             if pd.isna(z) or int(z) not in self.criteria: v = np.nan
#             else:
#                 z = int(z)
#                 c = self.criteria[z]
#                 v = self.ds_onset_func(group["VALUE"].values, self.get_index_for_station(y, c["start_search"], z), c["cumulative"], c["number_dry_days"], c["thrd_rain_day"], self.get_index_for_station(y, c["end_search"], z), c["nbjour"])
#             res.append({"year": y, "station": stn, "lat": group["LAT"].iloc[0], "lon": group["LON"].iloc[0], "onsetdryspell": v})
#         return self.build_cpt_output(pd.DataFrame(res), "onsetdryspell")

#     def compute(self, daily_data, map_rec, nb_cores):
#         shifted, mask, years = self.shift_gridded_data(daily_data, map_rec)
#         cx, cy = int(np.round(len(shifted.X)/nb_cores)), int(np.round(len(shifted.Y)/nb_cores))
#         out = []
#         for y in years:
#             yd, mk = shifted.sel(T=str(y)), lambda k: self._map_criteria(mask, k, y)
#             client = _get_compute_client(nb_cores)
#             res = xr.apply_ufunc(
#                 self.ds_onset_func, yd.chunk({'Y':cy,'X':cx}),
#                 mk("start_search").chunk({'Y':cy,'X':cx}), mk("cumulative").chunk({'Y':cy,'X':cx}),
#                 mk("number_dry_days").chunk({'Y':cy,'X':cx}), mk("thrd_rain_day").chunk({'Y':cy,'X':cx}),
#                 mk("end_search").chunk({'Y':cy,'X':cx}), mk("nbjour").chunk({'Y':cy,'X':cx}),
#                 input_core_dims=[('T',)]+[()]*6, vectorize=True, dask='parallelized', output_dtypes=['float']
#             ).compute()
#             _safe_close_client(client)
#             out.append(res)

#         unique_zone = np.unique(mask.to_numpy())
#         unique_zone = unique_zone[~np.isnan(unique_zone)]
#         # Choose a date to store results
#         if unique_zone.size == 0:
#             raise ValueError("No valid zones found in the mask.")
#         else:
#             # Use zone in low latitude
#             zone_id_to_use = int(np.min(unique_zone))
#         start_search_str = self.criteria[zone_id_to_use]["start_search"]
        
#         final = xr.concat(out, dim=pd.Index(years, name="T"))
#         final['T'] = pd.to_datetime([f"{y}-{start_search_str}" for y in years])
#         final.name = "Onset_dryspell"
#         return final


# class CEAC_compute_cessation_dry_spell(CAF_AgroClimateBase):
#     @staticmethod
#     def ds_cess_func(x, id1, cum, nbs, jp, ir1, id2, ijd, ETP, Cap, ir2, nbj):
#         if not (np.any(np.isfinite(x)) and np.isfinite(id1)): return np.nan
#         deb = CEAC_compute_onset.onset_function(x, id1, cum, nbs, jp, ir1)
#         if pd.isna(deb): return np.nan
#         fin = CEAC_compute_cessation.cessation_function(x, ijd, id2, ETP, Cap, ir2)
#         if not np.isnan(fin) and (fin - (deb + nbj)) > 0 and (deb + nbj) < len(x):
#             p = x[int(deb + nbj):int(fin)]
#             r = np.where(p > jp)[0]
#             if len(r) == 0: return np.nan
#             return np.max(np.array(list(r) + [len(p)]) - np.array([0] + list(r))) - 1
#         return np.nan

#     def compute_insitu(self, daily_df_raw, map_rec):
#         df_long, zones, _ = self.transform_and_shift_cdt(daily_df_raw, map_rec)
#         res = []
#         for (stn, y), group in df_long.groupby(["STATION", "year"]):
#             z = zones[stn]
#             if pd.isna(z) or int(z) not in self.criteria: v = np.nan
#             else:
#                 z = int(z)
#                 c = self.criteria[z]
#                 v = self.ds_cess_func(group["VALUE"].values, self.get_index_for_station(y, c["start_search"], z), c["cumulative"], c["number_dry_days"], c["thrd_rain_day"], self.get_index_for_station(y, c["end_search"], z), self.get_index_for_station(y, c["start_search2"], z), self.get_index_for_station(y, c["date_dry_soil"], z), c["ETP"], c["Cap_ret_maxi"], self.get_index_for_station(y, c["end_search2"], z), c["nbjour"])
#             res.append({"year": y, "station": stn, "lat": group["LAT"].iloc[0], "lon": group["LON"].iloc[0], "cessation_dryspell": v})
#         return self.build_cpt_output(pd.DataFrame(res), "cessation_dryspell")

#     def compute(self, daily_data, map_rec, nb_cores):
#         shifted, mask, years = self.shift_gridded_data(daily_data, map_rec)
#         cx, cy = int(np.round(len(shifted.X)/nb_cores)), int(np.round(len(shifted.Y)/nb_cores))
#         out = []
#         for y in years:
#             yd, mk = shifted.sel(T=str(y)), lambda k: self._map_criteria(mask, k, y)
#             client = _get_compute_client(nb_cores)
#             res = xr.apply_ufunc(
#                 self.ds_cess_func, yd.chunk({'Y':cy,'X':cx}),
#                 mk("start_search").chunk({'Y':cy,'X':cx}), mk("cumulative").chunk({'Y':cy,'X':cx}),
#                 mk("number_dry_days").chunk({'Y':cy,'X':cx}), mk("thrd_rain_day").chunk({'Y':cy,'X':cx}),
#                 mk("end_search").chunk({'Y':cy,'X':cx}), mk("start_search2").chunk({'Y':cy,'X':cx}),
#                 mk("date_dry_soil").chunk({'Y':cy,'X':cx}), mk("ETP").chunk({'Y':cy,'X':cx}),
#                 mk("Cap_ret_maxi").chunk({'Y':cy,'X':cx}), mk("end_search2").chunk({'Y':cy,'X':cx}),
#                 mk("nbjour").chunk({'Y':cy,'X':cx}),
#                 input_core_dims=[('T',)]+[()]*11, vectorize=True, dask='parallelized', output_dtypes=['float']
#             ).compute()
#             _safe_close_client(client)
#             out.append(res)
#         unique_zone = np.unique(mask.to_numpy())
#         unique_zone = unique_zone[~np.isnan(unique_zone)]
#         # Choose a date to store results
#         if unique_zone.size == 0:
#             raise ValueError("No valid zones found in the mask.")
#         else:
#             # Use zone in low latitude
#             zone_id_to_use = int(np.min(unique_zone))
#         start_search_str = self.criteria[zone_id_to_use]["start_search"]
            
#         final = xr.concat(out, dim=pd.Index(years, name="T"))
#         final['T'] = pd.to_datetime([f"{y}-{start_search_str}" for y in years])
#         final.name = "Cessation_dryspell"
#         return final


# class CEAC_count_dry_spells(CAF_AgroClimateBase):
#     @staticmethod
#     def count_dry_spells(x, onset, cessation, d_len, thresh):
#         if not (np.isfinite(x).any() and np.isfinite(onset) and np.isfinite(cessation)): return np.nan
#         o, c = int(onset), int(cessation)
#         if o < 0 or c < 0 or o >= len(x): return np.nan
#         c = min(c, len(x) - 1)
#         count, cur = 0, 0
#         for day in range(o, c + 1):
#             if x[day] < thresh: cur += 1
#             else:
#                 if cur == d_len: count += 1
#                 cur = 0
#         if cur == d_len: count += 1
#         return count

#     def compute_insitu(self, daily_raw, on_cpt, cess_cpt, map_rec, d_len, thresh=1.0):
#         df_long, zones, _ = self.transform_and_shift_cdt(daily_raw, map_rec)
#         m = pd.merge(self._parse_cpt_to_long(on_cpt, "o"), self._parse_cpt_to_long(cess_cpt, "c"), on=["station", "year"], suffixes=('_o','_c'))
#         res = []
#         for (stn, y), group in df_long.groupby(["STATION", "year"]):
#             sub = m[(m["station"] == stn) & (m["year"] == y)]
#             z = zones[stn]
#             if pd.isna(z) or sub.empty: v = np.nan
#             else:
#                 z = int(z)
#                 o_idx = self.revert_to_index(sub["o"].values[0], z)
#                 c_idx = self.revert_to_index(sub["c"].values[0], z)
#                 v = self.count_dry_spells(group["VALUE"].values, o_idx, c_idx, d_len, thresh)
                
#             lat_val = sub["lat_o"].values[0] if not sub.empty else group["LAT"].iloc[0]
#             lon_val = sub["lon_o"].values[0] if not sub.empty else group["LON"].iloc[0]
#             res.append({"year": y, "station": stn, "lat": lat_val, "lon": lon_val, "dry_spells": v})
#         return self.build_cpt_output(pd.DataFrame(res), "dry_spells")

#     def compute(self, daily_data, on_da, cess_da, map_rec, d_len, thresh, nb_cores):
#         shifted, mask, years = self.shift_gridded_data(daily_data, map_rec)
#         on_rel, cess_rel = self.revert_grid_index(on_da.reindex_like(mask, method='nearest'), mask), self.revert_grid_index(cess_da.reindex_like(mask, method='nearest'), mask)
#         cx, cy = int(np.round(len(shifted.X)/nb_cores)), int(np.round(len(shifted.Y)/nb_cores))
#         out = []
#         for y in years:
#             yd, o_y, c_y = shifted.sel(T=str(y)), on_rel.sel(T=str(y)).squeeze(), cess_rel.sel(T=str(y)).squeeze()
#             client = _get_compute_client(nb_cores)
#             res = xr.apply_ufunc(
#                 self.count_dry_spells, yd.chunk({'Y':cy,'X':cx}), o_y.chunk({'Y':cy,'X':cx}), c_y.chunk({'Y':cy,'X':cx}),
#                 input_core_dims=[('T',),(),()], vectorize=True, kwargs={'d_len': d_len, 'thresh': thresh}, dask='parallelized', output_dtypes=['float']
#             ).compute()
#             _safe_close_client(client)
#             out.append(res)
#         unique_zone = np.unique(mask.to_numpy())
#         unique_zone = unique_zone[~np.isnan(unique_zone)]
#         # Choose a date to store results
#         if unique_zone.size == 0:
#             raise ValueError("No valid zones found in the mask.")
#         else:
#             # Use zone in low latitude
#             zone_id_to_use = int(np.min(unique_zone))
#         start_search_str = self.criteria[zone_id_to_use]["start_search"]
#         final = xr.concat(out, dim=pd.Index(years, name="T"))
#         final['T'] = pd.to_datetime([f"{y}-{start_search_str}" for y in years])
#         final.name = "Count_dryspell"
#         return final


# class CEAC_count_wet_spells(CAF_AgroClimateBase):
#     @staticmethod
#     def count_wet_spells(x, onset, cessation, w_len, thresh):
#         if not (np.isfinite(x).any() and np.isfinite(onset) and np.isfinite(cessation)): return np.nan
#         o, c = int(onset), int(cessation)
#         if o < 0 or c < 0 or o >= len(x): return np.nan
#         c = min(c, len(x) - 1)
#         count, cur = 0, 0
#         for day in range(o, c + 1):
#             if x[day] >= thresh: cur += 1
#             else:
#                 if cur == w_len: count += 1
#                 cur = 0
#         if cur == w_len: count += 1
#         return count

#     def compute_insitu(self, daily_raw, on_cpt, cess_cpt, map_rec, w_len, thresh=1.0):
#         df_long, zones, _ = self.transform_and_shift_cdt(daily_raw, map_rec)
#         m = pd.merge(self._parse_cpt_to_long(on_cpt, "o"), self._parse_cpt_to_long(cess_cpt, "c"), on=["station", "year"], suffixes=('_o','_c'))
#         res = []
#         for (stn, y), group in df_long.groupby(["STATION", "year"]):
#             sub = m[(m["station"] == stn) & (m["year"] == y)]
#             z = zones[stn]
#             if pd.isna(z) or sub.empty: v = np.nan
#             else:
#                 z = int(z)
#                 o_idx = self.revert_to_index(sub["o"].values[0], z)
#                 c_idx = self.revert_to_index(sub["c"].values[0], z)
#                 v = self.count_wet_spells(group["VALUE"].values, o_idx, c_idx, w_len, thresh)
            
#             lat_val = sub["lat_o"].values[0] if not sub.empty else group["LAT"].iloc[0]
#             lon_val = sub["lon_o"].values[0] if not sub.empty else group["LON"].iloc[0]
#             res.append({"year": y, "station": stn, "lat": lat_val, "lon": lon_val, "wet_spells": v})
#         return self.build_cpt_output(pd.DataFrame(res), "wet_spells")

#     def compute(self, daily_data, on_da, cess_da, map_rec, w_len, thresh, nb_cores):
#         shifted, mask, years = self.shift_gridded_data(daily_data, map_rec)
#         on_rel, cess_rel = self.revert_grid_index(on_da.reindex_like(mask, method='nearest'), mask), self.revert_grid_index(cess_da.reindex_like(mask, method='nearest'), mask)
#         cx, cy = int(np.round(len(shifted.X)/nb_cores)), int(np.round(len(shifted.Y)/nb_cores))
#         out = []
#         for y in years:
#             yd, o_y, c_y = shifted.sel(T=str(y)), on_rel.sel(T=str(y)).squeeze(), cess_rel.sel(T=str(y)).squeeze()
#             client = _get_compute_client(nb_cores)
#             res = xr.apply_ufunc(
#                 self.count_wet_spells, yd.chunk({'Y':cy,'X':cx}), o_y.chunk({'Y':cy,'X':cx}), c_y.chunk({'Y':cy,'X':cx}),
#                 input_core_dims=[('T',),(),()], vectorize=True, kwargs={'w_len': w_len, 'thresh': thresh}, dask='parallelized', output_dtypes=['float']
#             ).compute()
#             _safe_close_client(client)
#             out.append(res)
#         unique_zone = np.unique(mask.to_numpy())
#         unique_zone = unique_zone[~np.isnan(unique_zone)]
#         # Choose a date to store results
#         if unique_zone.size == 0:
#             raise ValueError("No valid zones found in the mask.")
#         else:
#             # Use zone in low latitude
#             zone_id_to_use = int(np.min(unique_zone))
#         start_search_str = self.criteria[zone_id_to_use]["start_search"]
#         final = xr.concat(out, dim=pd.Index(years, name="T"))
#         final['T'] = pd.to_datetime([f"{y}-{start_search_str}" for y in years])
#         final.name = "Count_wetspell"
#         return final












































































# """CEAC agroclimatic indicator computation.

# Refactored, composable implementation of onset, cessation, and spell
# counting for both station (CPT/CDT format) and gridded data.

# Base class
# ----------
# CAF_AgroClimateBase
#     Shared helpers for date-index arithmetic, zone-specific criteria
#     mapping, CPT/CDT input parsing, and output formatting.

# Indicator classes
# -----------------
# CEAC_compute_onset
#     Season onset detection.
# CEAC_compute_cessation
#     Season cessation detection.
# CEAC_compute_onset_dry_spell
#     Onset with a post-onset dry-spell check.
# CEAC_compute_cessation_dry_spell
#     Cessation with an additional dry-spell criterion.
# CEAC_count_dry_spells
#     Count of dry spells of at least *d_len* days within the growing season.
# CEAC_count_wet_spells
#     Count of wet spells of at least *w_len* days within the growing season.
# """
# import pandas as pd
# import xarray as xr
# import numpy as np
# import random
# import datetime
# from dask.distributed import Client

# # ---------------------------------------------------------------------------
# # Shared Dask client: created once per process and reused across ALL calls,
# # instead of spinning up (and tearing down) a fresh LocalCluster on every
# # compute() — which caused nanny-join hangs at close() and worker
# # oversubscription. Closed once, at interpreter exit. Reuses (via get_client)
# # any client already created by another wass2s module, so the whole process
# # shares a single cluster.
# # ---------------------------------------------------------------------------
# _SHARED_CLIENT = None


# def _bounded_close(client):
#     """Close a client (and its LocalCluster) with a bounded timeout, swallowing
#     teardown errors. Used only at interpreter exit."""
#     if client is None:
#         return
#     import contextlib
#     cluster = getattr(client, "cluster", None)
#     with contextlib.suppress(Exception):
#         client.close(timeout=10)
#     if cluster is not None:
#         with contextlib.suppress(Exception):
#             cluster.close(timeout=10)


# def _safe_close_client(client):
#     """Intentional no-op. Every client here comes from _get_compute_client and
#     is process-wide and reused; it must NOT be torn down per call. The shared
#     client is closed exactly once, at interpreter exit, by the atexit hook in
#     _get_compute_client (or reaped by the OS/Slurm cgroup on os._exit)."""
#     return


# def _get_compute_client(nb_cores):
#     """Return a process-wide Dask client, created lazily on first use and reused
#     afterwards. Returns None if a distributed client cannot be created, in which
#     case .compute() falls back to dask's default (threaded) scheduler."""
#     global _SHARED_CLIENT
#     try:
#         from dask.distributed import Client, LocalCluster, get_client
#         try:
#             return get_client()
#         except Exception:
#             pass
#         if _SHARED_CLIENT is not None:
#             try:
#                 if _SHARED_CLIENT.status == "running":
#                     return _SHARED_CLIENT
#             except Exception:
#                 _SHARED_CLIENT = None
#         cluster = LocalCluster(
#             n_workers=max(1, int(nb_cores)),
#             threads_per_worker=1,
#             processes=True,
#             dashboard_address=None,
#             memory_limit=0,
#         )
#         _SHARED_CLIENT = Client(cluster)
#         import atexit
#         atexit.register(_bounded_close, _SHARED_CLIENT)
#         return _SHARED_CLIENT
#     except Exception:
#         return None

# import warnings

# warnings.filterwarnings('ignore')

# DEFAULT_CRITERIA = {
#     1: {"start_search": "05-01", "start_search2": "09-01", "date_dry_soil": "01-01", "cumulative": 15, "number_dry_days": 15, "thrd_rain_day": 0.85, "end_search": "08-30",  "end_search2": "10-30", "nbjour": 35, "ETP": 5.0, "Cap_ret_maxi": 70},
#     2: {"start_search": "03-15", "start_search2": "09-01", "date_dry_soil": "01-01", "cumulative": 20, "number_dry_days": 10, "thrd_rain_day": 0.85, "end_search": "08-01",  "end_search2": "11-01", "nbjour": 40, "ETP": 5.0, "Cap_ret_maxi": 70},
#     3: {"start_search": "02-01", "start_search2": "10-01", "date_dry_soil": "01-01", "cumulative": 20, "number_dry_days": 10, "thrd_rain_day": 0.85, "end_search": "05-15", "end_search2": "12-30", "nbjour": 45, "ETP": 5.0, "Cap_ret_maxi": 70},
#     4: {"start_search": "01-01", "start_search2": "11-01", "date_dry_soil": "01-01", "cumulative": 20, "number_dry_days": 7, "thrd_rain_day": 0.85, "end_search": "04-01", "end_search2": "12-30", "nbjour": 50, "ETP": 5.0, "Cap_ret_maxi": 80},
#     5: {"start_search": "01-01", "start_search2": "06-01", "date_dry_soil": "01-01", "cumulative": 25, "number_dry_days": 7, "thrd_rain_day": 0.85, "end_search": "03-10", "end_search2": "08-10", "nbjour": 50, "ETP": 4.0, "Cap_ret_maxi": 60},
#     6: {"start_search": "02-01", "start_search2": "10-15", "date_dry_soil": "01-01", "cumulative": 20, "number_dry_days": 7, "thrd_rain_day": 0.85, "end_search": "03-20",  "end_search2": "12-15", "nbjour": 50, "ETP": 4.0, "Cap_ret_maxi": 70},
#     7: {"start_search": "02-01", "start_search2": "09-01", "date_dry_soil": "01-01", "cumulative": 20, "number_dry_days": 10, "thrd_rain_day": 0.85, "end_search": "03-20",  "end_search2": "12-15", "nbjour": 50, "ETP": 4.0, "Cap_ret_maxi": 70},
#     8: {"start_search": "03-01", "start_search2": "08-01", "date_dry_soil": "01-01", "cumulative": 20, "number_dry_days": 10, "thrd_rain_day": 0.85, "end_search": "04-20",  "end_search2": "10-15", "nbjour": 40, "ETP": 4.0, "Cap_ret_maxi": 70},
#     9: {"start_search": "05-01", "start_search2": "08-15", "date_dry_soil": "01-01", "cumulative": 20, "number_dry_days": 10, "thrd_rain_day": 0.85, "end_search": "06-20",  "end_search2": "10-15", "nbjour": 30, "ETP": 4.0, "Cap_ret_maxi": 70},
# }

# class CAF_AgroClimateBase:
#     SHIFT_OFFSET = 244 # L'offset pour marquer l'extension

#     def __init__(self, user_criteria=None):
#         self.criteria = user_criteria if user_criteria else DEFAULT_CRITERIA

#     def _is_shifted(self, z):
#         # zone du shift
#         return not pd.isna(z) and int(z) >= 6 

#     @staticmethod
#     def day_of_year(y, mm_dd):
#         dt = datetime.datetime.strptime(f"{int(y)}-{mm_dd}", "%Y-%m-%d").date()
#         return (dt - datetime.date(int(y), 1, 1)).days + 1

#     def get_index_for_station(self, year, mm_dd, z):
#         shifted = self._is_shifted(z)
#         base_date = datetime.date(year, 8, 1) if shifted else datetime.date(year, 1, 1)
#         target_month, target_day = map(int, mm_dd.split('-'))
#         target_year = year + 1 if (shifted and target_month < 8) else year
#         target_date = datetime.date(target_year, target_month, target_day)
#         return (target_date - base_date).days

#     def output_format_value(self, v, z):
#         if np.isnan(v): return np.nan
#         return v + self.SHIFT_OFFSET if self._is_shifted(z) else v + 1

#     def revert_to_index(self, v, z):
#         if np.isnan(v): return np.nan
#         return int(v - self.SHIFT_OFFSET) if self._is_shifted(z) else int(v - 1)

#     def _map_criteria(self, mask, key, year):
#         def _safe_get(z):
#             if np.isnan(z) or int(z) not in self.criteria: return np.nan
#             v = self.criteria[int(z)][key]
#             return self.day_of_year(year, v) if 'search' in key or 'date' in key else v
#         return xr.DataArray(np.vectorize(_safe_get, otypes=[float])(mask.values), coords=mask.coords)

#     def shift_gridded_data(self, daily_data, map_reclassified):
#         mask = map_reclassified.reindex_like(daily_data, method='nearest')
#         y1, y2 = int(daily_data['T'].dt.year.min()), int(daily_data['T'].dt.year.max())
        
#         #  mask <= 5 vs mask >= 6
#         le5 = daily_data.sel(T=slice(f"{y1}", f"{y2-1}")).where(mask <= 5)
#         gt6 = daily_data.sel(T=slice(f"{y1}-08-01", f"{y2}-07-31")).where(mask >= 6)
        
#         m_len = min(len(le5['T']), len(gt6['T']))
#         le5, gt6 = le5.isel(T=slice(0, m_len)), gt6.isel(T=slice(0, m_len))
#         gt6 = gt6.assign_coords(T=le5['T'].values)
#         return le5.combine_first(gt6), mask, np.unique(le5['T'].dt.year.to_numpy())

#     def format_grid_output(self, res_xr, mask):
#         return xr.where(mask >= 6, res_xr + self.SHIFT_OFFSET, res_xr + 1)

#     def revert_grid_index(self, res_xr, mask):
#         return xr.where(mask >= 6, res_xr - self.SHIFT_OFFSET, res_xr - 1)

#     def transform_and_shift_cdt(self, df_raw, map_reclassified):
#         if "ID" in df_raw.columns or str(df_raw.columns[0]).upper() == "ID":
#             header_row = pd.DataFrame([df_raw.columns])
#             header_row.columns = range(df_raw.shape[1])
#             df_raw.columns = range(df_raw.shape[1])
#             df_raw = pd.concat([header_row, df_raw], ignore_index=True)

#         s_ids = pd.Series(df_raw.iloc[0, 1:].values.astype(str))
#         ids = s_ids.where(~s_ids.duplicated(), s_ids + "_" + s_ids.groupby(s_ids).cumcount().astype(str)).values
#         df_raw.iloc[0, 1:] = ids 
        
#         lons, lats = df_raw.iloc[1, 1:].astype(float).values, df_raw.iloc[2, 1:].astype(float).values
#         dates = pd.to_datetime(df_raw.iloc[4:, 0], format='%Y%m%d')
#         da = xr.DataArray(df_raw.iloc[4:, 1:].astype(float).values, coords={'T': dates, 'station': ids}, dims=['T', 'station'])
#         y1, y2 = dates.min().year, dates.max().year
        
#         stn_zones, series_lst = {}, []
#         for i, stn in enumerate(ids):
#             try: z = map_reclassified.sel(X=lons[i], Y=lats[i], method='nearest').values.item()
#             except: z = np.nan
#             stn_zones[stn] = z
#             s = da.isel(station=i)
#             # Construction des séries continues par station
#             if self._is_shifted(z): series_lst.append(s.sel(T=slice(f"{y1}-08-01", f"{y2}-07-31")))
#             else: series_lst.append(s.sel(T=slice(f"{y1}-01-01", f"{y2-1}-12-31")))
                
#         m_len = min(len(s) for s in series_lst)
#         std_dt_vals = dates.iloc[:m_len].values  
        
#         arr_2d = np.column_stack([s.values[:m_len] for s in series_lst])
#         df_shifted = pd.DataFrame(arr_2d, index=std_dt_vals, columns=ids)
        
#         df_long = df_shifted.reset_index().melt(id_vars="index", var_name="STATION", value_name="VALUE")
#         df_long.rename(columns={"index": "DATE"}, inplace=True)
        
#         meta_df = pd.DataFrame({"STATION": ids, "LON": lons, "LAT": lats, "zonename": [stn_zones[s] for s in ids]})
#         df_long = df_long.merge(meta_df, on="STATION")
#         df_long["VALUE"] = df_long["VALUE"].replace(-99.0, np.nan)
#         df_long["year"] = df_long["DATE"].dt.year
        
#         return df_long, stn_zones, df_raw.iloc[:4, :]

#     def _parse_cpt_to_long(self, df_cpt, val_name):
#         lats, lons = df_cpt.iloc[0, 1:].values, df_cpt.iloc[1, 1:].values
#         cols = df_cpt.columns[1:].tolist()
#         df = df_cpt.iloc[2:].copy().reset_index(drop=True).rename(columns={"STATION": "year"})
#         df = df.melt(id_vars=["year"], var_name="station", value_name=val_name)
#         df["year"] = pd.to_numeric(df["year"], errors="coerce")
#         df["lat"], df["lon"] = df["station"].map(dict(zip(cols, lats))), df["station"].map(dict(zip(cols, lons)))
#         return df

#     def build_cpt_output(self, res_df, val_col):
#         res_df[val_col] = res_df[val_col].fillna(-999.0)
#         piv = res_df.pivot(index="year", columns="station", values=val_col)
#         meta = res_df.groupby("station")[["lat", "lon"]].first()
#         lats, lons = meta.loc[piv.columns, "lat"].tolist(), meta.loc[piv.columns, "lon"].tolist()
#         lat_row = pd.DataFrame([lats], columns=piv.columns, index=["LAT"])
#         lon_row = pd.DataFrame([lons], columns=piv.columns, index=["LON"])
#         final = pd.concat([lat_row, lon_row, piv]).reset_index().rename(columns={"index": "STATION"})
#         final.columns.name = None 
#         return final


# class CEAC_compute_onset(CAF_AgroClimateBase):
#     @staticmethod
#     def onset_function(x, idebut, cumul, nbsec, jour_pluvieux, irch_fin):
#         if not (np.any(np.isfinite(x)) and np.isfinite(idebut) and np.isfinite(nbsec) and np.isfinite(irch_fin)): return np.nan
#         idebut, nbsec, irch_fin = int(idebut), int(nbsec), int(irch_fin)
#         idate, trouv = idebut, 0
#         while True:
#             idate += 1
#             if idate >= len(x)-1 or pd.isna(x[idate-1]) or pd.isna(x[idate]) or pd.isna(x[idate+1]): return np.nan
#             if idate > irch_fin: return random.randint(irch_fin - 5, irch_fin)
#             c1, c2, c3 = x[idate-1], x[idate-1]+x[idate], x[idate-1]+x[idate]+x[idate+1]
#             if c1 >= cumul or c2 >= cumul or c3 >= cumul:
#                 arr = np.array([x[idate-1], x[idate], x[idate+1]])
#                 ideb = [idate-1, idate, idate+1][np.argmax(arr)]
#                 trouv = 1
#                 pluie30 = x[ideb:ideb+31] if ideb+30 < len(x) else x[ideb:]
#                 isec = 0
#                 while True:
#                     isec += 1
#                     if isec+nbsec >= len(pluie30): break
#                     if np.sum(pluie30[isec:isec+nbsec+1] < jour_pluvieux) == (nbsec + 1): trouv = 0; break
#                     if isec == (30 - nbsec): break
#             if trouv == 1: return ideb
#         return np.nan

#     def compute_insitu(self, daily_df_raw, map_rec):
#         df_long, zones, _ = self.transform_and_shift_cdt(daily_df_raw, map_rec)
#         res = []
#         for (stn, y), group in df_long.groupby(["STATION", "year"]):
#             z = zones[stn]
#             if pd.isna(z) or int(z) not in self.criteria: v = np.nan
#             else:
#                 z = int(z)
#                 c = self.criteria[z]
#                 v = self.onset_function(group["VALUE"].values, self.get_index_for_station(y, c["start_search"], z), c["cumulative"], c["number_dry_days"], c["thrd_rain_day"], self.get_index_for_station(y, c["end_search"], z))
#                 v = self.output_format_value(v, z)
#             res.append({"year": y, "station": stn, "lat": group["LAT"].iloc[0], "lon": group["LON"].iloc[0], "onset": v})
#         return self.build_cpt_output(pd.DataFrame(res), "onset")

#     def compute(self, daily_data, map_rec, nb_cores):
#         shifted, mask, years = self.shift_gridded_data(daily_data, map_rec)
#         cx, cy = int(np.round(len(shifted.X)/nb_cores)), int(np.round(len(shifted.Y)/nb_cores))
#         out = []
#         for y in years:
#             yd, mk = shifted.sel(T=str(y)), lambda k: self._map_criteria(mask, k, y)
#             client = _get_compute_client(nb_cores)
#             res = xr.apply_ufunc(
#                 self.onset_function, yd.chunk({'Y': cy, 'X': cx}),
#                 mk("start_search").chunk({'Y': cy, 'X': cx}), mk("cumulative").chunk({'Y': cy, 'X': cx}),
#                 mk("number_dry_days").chunk({'Y': cy, 'X': cx}), mk("thrd_rain_day").chunk({'Y': cy, 'X': cx}),
#                 mk("end_search").chunk({'Y': cy, 'X': cx}),
#                 input_core_dims=[('T',)]+[()]*5, vectorize=True, dask='parallelized', output_dtypes=['float']
#             ).compute()
#             _safe_close_client(client)
#             out.append(res)
            
#         unique_zone = np.unique(mask.to_numpy())
#         unique_zone = unique_zone[~np.isnan(unique_zone)]
#         # Choose a date to store results
#         if unique_zone.size == 0:
#             raise ValueError("No valid zones found in the mask.")
#         else:
#             # Use zone in low latitude
#             zone_id_to_use = int(np.min(unique_zone))
        
#         start_search_str = self.criteria[zone_id_to_use]["start_search"]        
#         final = self.format_grid_output(xr.concat(out, dim=pd.Index(years, name="T")), mask)
#         final['T'] = pd.to_datetime([f"{y}-{start_search_str}" for y in years])
#         final.name = "Onset"
#         return final


# class CEAC_compute_cessation(CAF_AgroClimateBase):
#     @staticmethod
#     def cessation_function(x, ijour, idebut, ETP, Cap, irch_fin):
#         if not (np.isfinite(x).any() and np.isfinite(idebut) and np.isfinite(ijour) and np.isfinite(ETP) and np.isfinite(Cap) and np.isfinite(irch_fin)): return np.nan
#         ru, ifin = 0, int(idebut)
#         for k in range(int(ijour), ifin + 1):
#             if not pd.isna(x[k]): ru = max(0, min(ru + x[k] - ETP, Cap))
#         while ifin < int(irch_fin):
#             ifin += 1
#             if ifin >= len(x) or pd.isna(x[ifin]): continue
#             ru = max(0, min(ru + x[ifin] - ETP, Cap))
#             if ru <= 0: break
#         return ifin if ifin <= int(irch_fin) else random.randint(int(irch_fin) - 5, int(irch_fin))

#     def compute_insitu(self, daily_df_raw, map_rec):
#         df_long, zones, _ = self.transform_and_shift_cdt(daily_df_raw, map_rec)
#         res = []
#         for (stn, y), group in df_long.groupby(["STATION", "year"]):
#             z = zones[stn]
#             if pd.isna(z) or int(z) not in self.criteria: v = np.nan
#             else:
#                 z = int(z)
#                 c = self.criteria[z]
#                 v = self.cessation_function(group["VALUE"].values, self.get_index_for_station(y, c["date_dry_soil"], z), self.get_index_for_station(y, c["start_search2"], z), c["ETP"], c["Cap_ret_maxi"], self.get_index_for_station(y, c["end_search2"], z))
#                 v = self.output_format_value(v, z)
#             res.append({"year": y, "station": stn, "lat": group["LAT"].iloc[0], "lon": group["LON"].iloc[0], "cessation": v})
#         return self.build_cpt_output(pd.DataFrame(res), "cessation")

#     def compute(self, daily_data, map_rec, nb_cores):
#         shifted, mask, years = self.shift_gridded_data(daily_data, map_rec)
#         cx, cy = int(np.round(len(shifted.X)/nb_cores)), int(np.round(len(shifted.Y)/nb_cores))
#         out = []
#         for y in years:
#             yd, mk = shifted.sel(T=str(y)), lambda k: self._map_criteria(mask, k, y)
#             client = _get_compute_client(nb_cores)
#             res = xr.apply_ufunc(
#                 self.cessation_function, yd.chunk({'Y': cy, 'X': cx}),
#                 mk("date_dry_soil").chunk({'Y': cy, 'X': cx}), mk("start_search2").chunk({'Y': cy, 'X': cx}),
#                 mk("ETP").chunk({'Y': cy, 'X': cx}), mk("Cap_ret_maxi").chunk({'Y': cy, 'X': cx}),
#                 mk("end_search2").chunk({'Y': cy, 'X': cx}),
#                 input_core_dims=[('T',)]+[()]*5, vectorize=True, dask='parallelized', output_dtypes=['float']
#             ).compute()
#             _safe_close_client(client)
#             out.append(res)

#         unique_zone = np.unique(mask.to_numpy())
#         unique_zone = unique_zone[~np.isnan(unique_zone)]
#         # Choose a date to store results
#         if unique_zone.size == 0:
#             raise ValueError("No valid zones found in the mask.")
#         else:
#             # Use zone in low latitude
#             zone_id_to_use = int(np.min(unique_zone))
        
#         start_search_str = self.criteria[zone_id_to_use]["start_search"]
        
#         final = self.format_grid_output(xr.concat(out, dim=pd.Index(years, name="T")), mask)
#         final['T'] = pd.to_datetime([f"{y}-{start_search_str}" for y in years])
#         final.name = "Cessation"
#         return final


# class CEAC_compute_onset_dry_spell(CAF_AgroClimateBase):
#     @staticmethod
#     def ds_onset_func(x, idebut, cumul, nbsec, jp, irch_fin, nbjour):
#         if not (np.any(np.isfinite(x)) and np.isfinite(idebut)): return np.nan
#         deb = CEAC_compute_onset.onset_function(x, idebut, cumul, nbsec, jp, irch_fin)
#         if not np.isnan(deb):
#             p = x[int(deb) : min(int(deb) + int(nbjour) + 1, len(x))]
#             r = np.where(p > jp)[0]
#             d1, d2 = np.array([0] + list(r)), np.array(list(r) + [len(p)])
#             return np.max(d2 - d1) - 1
#         return np.nan

#     def compute_insitu(self, daily_df_raw, map_rec):
#         df_long, zones, _ = self.transform_and_shift_cdt(daily_df_raw, map_rec)
#         res = []
#         for (stn, y), group in df_long.groupby(["STATION", "year"]):
#             z = zones[stn]
#             if pd.isna(z) or int(z) not in self.criteria: v = np.nan
#             else:
#                 z = int(z)
#                 c = self.criteria[z]
#                 v = self.ds_onset_func(group["VALUE"].values, self.get_index_for_station(y, c["start_search"], z), c["cumulative"], c["number_dry_days"], c["thrd_rain_day"], self.get_index_for_station(y, c["end_search"], z), c["nbjour"])
#             res.append({"year": y, "station": stn, "lat": group["LAT"].iloc[0], "lon": group["LON"].iloc[0], "onsetdryspell": v})
#         return self.build_cpt_output(pd.DataFrame(res), "onsetdryspell")

#     def compute(self, daily_data, map_rec, nb_cores):
#         shifted, mask, years = self.shift_gridded_data(daily_data, map_rec)
#         cx, cy = int(np.round(len(shifted.X)/nb_cores)), int(np.round(len(shifted.Y)/nb_cores))
#         out = []
#         for y in years:
#             yd, mk = shifted.sel(T=str(y)), lambda k: self._map_criteria(mask, k, y)
#             client = _get_compute_client(nb_cores)
#             res = xr.apply_ufunc(
#                 self.ds_onset_func, yd.chunk({'Y':cy,'X':cx}),
#                 mk("start_search").chunk({'Y':cy,'X':cx}), mk("cumulative").chunk({'Y':cy,'X':cx}),
#                 mk("number_dry_days").chunk({'Y':cy,'X':cx}), mk("thrd_rain_day").chunk({'Y':cy,'X':cx}),
#                 mk("end_search").chunk({'Y':cy,'X':cx}), mk("nbjour").chunk({'Y':cy,'X':cx}),
#                 input_core_dims=[('T',)]+[()]*6, vectorize=True, dask='parallelized', output_dtypes=['float']
#             ).compute()
#             _safe_close_client(client)
#             out.append(res)

#         unique_zone = np.unique(mask.to_numpy())
#         unique_zone = unique_zone[~np.isnan(unique_zone)]
#         # Choose a date to store results
#         if unique_zone.size == 0:
#             raise ValueError("No valid zones found in the mask.")
#         else:
#             # Use zone in low latitude
#             zone_id_to_use = int(np.min(unique_zone))
#         start_search_str = self.criteria[zone_id_to_use]["start_search"]
        
#         final = xr.concat(out, dim=pd.Index(years, name="T"))
#         final['T'] = pd.to_datetime([f"{y}-{start_search_str}" for y in years])
#         final.name = "Onset_dryspell"
#         return final


# class CEAC_compute_cessation_dry_spell(CAF_AgroClimateBase):
#     @staticmethod
#     def ds_cess_func(x, id1, cum, nbs, jp, ir1, id2, ijd, ETP, Cap, ir2, nbj):
#         if not (np.any(np.isfinite(x)) and np.isfinite(id1)): return np.nan
#         deb = CEAC_compute_onset.onset_function(x, id1, cum, nbs, jp, ir1)
#         if pd.isna(deb): return np.nan
#         fin = CEAC_compute_cessation.cessation_function(x, ijd, id2, ETP, Cap, ir2)
#         if not np.isnan(fin) and (fin - (deb + nbj)) > 0 and (deb + nbj) < len(x):
#             p = x[int(deb + nbj):int(fin)]
#             r = np.where(p > jp)[0]
#             if len(r) == 0: return np.nan
#             return np.max(np.array(list(r) + [len(p)]) - np.array([0] + list(r))) - 1
#         return np.nan

#     def compute_insitu(self, daily_df_raw, map_rec):
#         df_long, zones, _ = self.transform_and_shift_cdt(daily_df_raw, map_rec)
#         res = []
#         for (stn, y), group in df_long.groupby(["STATION", "year"]):
#             z = zones[stn]
#             if pd.isna(z) or int(z) not in self.criteria: v = np.nan
#             else:
#                 z = int(z)
#                 c = self.criteria[z]
#                 v = self.ds_cess_func(group["VALUE"].values, self.get_index_for_station(y, c["start_search"], z), c["cumulative"], c["number_dry_days"], c["thrd_rain_day"], self.get_index_for_station(y, c["end_search"], z), self.get_index_for_station(y, c["start_search2"], z), self.get_index_for_station(y, c["date_dry_soil"], z), c["ETP"], c["Cap_ret_maxi"], self.get_index_for_station(y, c["end_search2"], z), c["nbjour"])
#             res.append({"year": y, "station": stn, "lat": group["LAT"].iloc[0], "lon": group["LON"].iloc[0], "cessation_dryspell": v})
#         return self.build_cpt_output(pd.DataFrame(res), "cessation_dryspell")

#     def compute(self, daily_data, map_rec, nb_cores):
#         shifted, mask, years = self.shift_gridded_data(daily_data, map_rec)
#         cx, cy = int(np.round(len(shifted.X)/nb_cores)), int(np.round(len(shifted.Y)/nb_cores))
#         out = []
#         for y in years:
#             yd, mk = shifted.sel(T=str(y)), lambda k: self._map_criteria(mask, k, y)
#             client = _get_compute_client(nb_cores)
#             res = xr.apply_ufunc(
#                 self.ds_cess_func, yd.chunk({'Y':cy,'X':cx}),
#                 mk("start_search").chunk({'Y':cy,'X':cx}), mk("cumulative").chunk({'Y':cy,'X':cx}),
#                 mk("number_dry_days").chunk({'Y':cy,'X':cx}), mk("thrd_rain_day").chunk({'Y':cy,'X':cx}),
#                 mk("end_search").chunk({'Y':cy,'X':cx}), mk("start_search2").chunk({'Y':cy,'X':cx}),
#                 mk("date_dry_soil").chunk({'Y':cy,'X':cx}), mk("ETP").chunk({'Y':cy,'X':cx}),
#                 mk("Cap_ret_maxi").chunk({'Y':cy,'X':cx}), mk("end_search2").chunk({'Y':cy,'X':cx}),
#                 mk("nbjour").chunk({'Y':cy,'X':cx}),
#                 input_core_dims=[('T',)]+[()]*11, vectorize=True, dask='parallelized', output_dtypes=['float']
#             ).compute()
#             _safe_close_client(client)
#             out.append(res)
#         unique_zone = np.unique(mask.to_numpy())
#         unique_zone = unique_zone[~np.isnan(unique_zone)]
#         # Choose a date to store results
#         if unique_zone.size == 0:
#             raise ValueError("No valid zones found in the mask.")
#         else:
#             # Use zone in low latitude
#             zone_id_to_use = int(np.min(unique_zone))
#         start_search_str = self.criteria[zone_id_to_use]["start_search"]
            
#         final = xr.concat(out, dim=pd.Index(years, name="T"))
#         final['T'] = pd.to_datetime([f"{y}-{start_search_str}" for y in years])
#         final.name = "Cessation_dryspell"
#         return final


# class CEAC_count_dry_spells(CAF_AgroClimateBase):
#     @staticmethod
#     def count_dry_spells(x, onset, cessation, d_len, thresh):
#         if not (np.isfinite(x).any() and np.isfinite(onset) and np.isfinite(cessation)): return np.nan
#         o, c = int(onset), int(cessation)
#         if o < 0 or c < 0 or o >= len(x): return np.nan
#         c = min(c, len(x) - 1)
#         count, cur = 0, 0
#         for day in range(o, c + 1):
#             if x[day] < thresh: cur += 1
#             else:
#                 if cur == d_len: count += 1
#                 cur = 0
#         if cur == d_len: count += 1
#         return count

#     def compute_insitu(self, daily_raw, on_cpt, cess_cpt, map_rec, d_len, thresh=1.0):
#         df_long, zones, _ = self.transform_and_shift_cdt(daily_raw, map_rec)
#         m = pd.merge(self._parse_cpt_to_long(on_cpt, "o"), self._parse_cpt_to_long(cess_cpt, "c"), on=["station", "year"], suffixes=('_o','_c'))
#         res = []
#         for (stn, y), group in df_long.groupby(["STATION", "year"]):
#             sub = m[(m["station"] == stn) & (m["year"] == y)]
#             z = zones[stn]
#             if pd.isna(z) or sub.empty: v = np.nan
#             else:
#                 z = int(z)
#                 o_idx = self.revert_to_index(sub["o"].values[0], z)
#                 c_idx = self.revert_to_index(sub["c"].values[0], z)
#                 v = self.count_dry_spells(group["VALUE"].values, o_idx, c_idx, d_len, thresh)
                
#             lat_val = sub["lat_o"].values[0] if not sub.empty else group["LAT"].iloc[0]
#             lon_val = sub["lon_o"].values[0] if not sub.empty else group["LON"].iloc[0]
#             res.append({"year": y, "station": stn, "lat": lat_val, "lon": lon_val, "dry_spells": v})
#         return self.build_cpt_output(pd.DataFrame(res), "dry_spells")

#     def compute(self, daily_data, on_da, cess_da, map_rec, d_len, thresh, nb_cores):
#         shifted, mask, years = self.shift_gridded_data(daily_data, map_rec)
#         on_rel, cess_rel = self.revert_grid_index(on_da.reindex_like(mask, method='nearest'), mask), self.revert_grid_index(cess_da.reindex_like(mask, method='nearest'), mask)
#         cx, cy = int(np.round(len(shifted.X)/nb_cores)), int(np.round(len(shifted.Y)/nb_cores))
#         out = []
#         for y in years:
#             yd, o_y, c_y = shifted.sel(T=str(y)), on_rel.sel(T=str(y)).squeeze(), cess_rel.sel(T=str(y)).squeeze()
#             client = _get_compute_client(nb_cores)
#             res = xr.apply_ufunc(
#                 self.count_dry_spells, yd.chunk({'Y':cy,'X':cx}), o_y.chunk({'Y':cy,'X':cx}), c_y.chunk({'Y':cy,'X':cx}),
#                 input_core_dims=[('T',),(),()], vectorize=True, kwargs={'d_len': d_len, 'thresh': thresh}, dask='parallelized', output_dtypes=['float']
#             ).compute()
#             _safe_close_client(client)
#             out.append(res)
#         unique_zone = np.unique(mask.to_numpy())
#         unique_zone = unique_zone[~np.isnan(unique_zone)]
#         # Choose a date to store results
#         if unique_zone.size == 0:
#             raise ValueError("No valid zones found in the mask.")
#         else:
#             # Use zone in low latitude
#             zone_id_to_use = int(np.min(unique_zone))
#         start_search_str = self.criteria[zone_id_to_use]["start_search"]
#         final = xr.concat(out, dim=pd.Index(years, name="T"))
#         final['T'] = pd.to_datetime([f"{y}-{start_search_str}" for y in years])
#         final.name = "Count_dryspell"
#         return final


# class CEAC_count_wet_spells(CAF_AgroClimateBase):
#     @staticmethod
#     def count_wet_spells(x, onset, cessation, w_len, thresh):
#         if not (np.isfinite(x).any() and np.isfinite(onset) and np.isfinite(cessation)): return np.nan
#         o, c = int(onset), int(cessation)
#         if o < 0 or c < 0 or o >= len(x): return np.nan
#         c = min(c, len(x) - 1)
#         count, cur = 0, 0
#         for day in range(o, c + 1):
#             if x[day] >= thresh: cur += 1
#             else:
#                 if cur == w_len: count += 1
#                 cur = 0
#         if cur == w_len: count += 1
#         return count

#     def compute_insitu(self, daily_raw, on_cpt, cess_cpt, map_rec, w_len, thresh=1.0):
#         df_long, zones, _ = self.transform_and_shift_cdt(daily_raw, map_rec)
#         m = pd.merge(self._parse_cpt_to_long(on_cpt, "o"), self._parse_cpt_to_long(cess_cpt, "c"), on=["station", "year"], suffixes=('_o','_c'))
#         res = []
#         for (stn, y), group in df_long.groupby(["STATION", "year"]):
#             sub = m[(m["station"] == stn) & (m["year"] == y)]
#             z = zones[stn]
#             if pd.isna(z) or sub.empty: v = np.nan
#             else:
#                 z = int(z)
#                 o_idx = self.revert_to_index(sub["o"].values[0], z)
#                 c_idx = self.revert_to_index(sub["c"].values[0], z)
#                 v = self.count_wet_spells(group["VALUE"].values, o_idx, c_idx, w_len, thresh)
            
#             lat_val = sub["lat_o"].values[0] if not sub.empty else group["LAT"].iloc[0]
#             lon_val = sub["lon_o"].values[0] if not sub.empty else group["LON"].iloc[0]
#             res.append({"year": y, "station": stn, "lat": lat_val, "lon": lon_val, "wet_spells": v})
#         return self.build_cpt_output(pd.DataFrame(res), "wet_spells")

#     def compute(self, daily_data, on_da, cess_da, map_rec, w_len, thresh, nb_cores):
#         shifted, mask, years = self.shift_gridded_data(daily_data, map_rec)
#         on_rel, cess_rel = self.revert_grid_index(on_da.reindex_like(mask, method='nearest'), mask), self.revert_grid_index(cess_da.reindex_like(mask, method='nearest'), mask)
#         cx, cy = int(np.round(len(shifted.X)/nb_cores)), int(np.round(len(shifted.Y)/nb_cores))
#         out = []
#         for y in years:
#             yd, o_y, c_y = shifted.sel(T=str(y)), on_rel.sel(T=str(y)).squeeze(), cess_rel.sel(T=str(y)).squeeze()
#             client = _get_compute_client(nb_cores)
#             res = xr.apply_ufunc(
#                 self.count_wet_spells, yd.chunk({'Y':cy,'X':cx}), o_y.chunk({'Y':cy,'X':cx}), c_y.chunk({'Y':cy,'X':cx}),
#                 input_core_dims=[('T',),(),()], vectorize=True, kwargs={'w_len': w_len, 'thresh': thresh}, dask='parallelized', output_dtypes=['float']
#             ).compute()
#             _safe_close_client(client)
#             out.append(res)
#         unique_zone = np.unique(mask.to_numpy())
#         unique_zone = unique_zone[~np.isnan(unique_zone)]
#         # Choose a date to store results
#         if unique_zone.size == 0:
#             raise ValueError("No valid zones found in the mask.")
#         else:
#             # Use zone in low latitude
#             zone_id_to_use = int(np.min(unique_zone))
#         start_search_str = self.criteria[zone_id_to_use]["start_search"]
#         final = xr.concat(out, dim=pd.Index(years, name="T"))
#         final['T'] = pd.to_datetime([f"{y}-{start_search_str}" for y in years])
#         final.name = "Count_wetspell"
#         return final
