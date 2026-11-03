"""Data download utilities for ERA5, CHIRPS, TAMSAT, and NMME.

Provides a unified interface for downloading and pre-processing gridded
seasonal hindcast and reanalysis datasets.

Class
-----
WAS_Download
    Central download manager.  Key methods:

    ``WAS_Download_Models``
        NMME and C3S seasonal forecast / hindcast data.
    ``WAS_Download_Reanalysis`` / ``WAS_Download_Reanalysis_``
        ERA5 single-level and pressure-level reanalysis (via CDS).
    ``WAS_Download_ERA5Land`` / ``WAS_Download_ERA5Land_daily``
        ERA5-Land daily and monthly data.
    ``WAS_Download_CHIRPSv3_Seasonal`` / ``WAS_Download_CHIRPSv3_Daily``
        CHIRPS v3 precipitation (seasonal aggregates and daily fields).
    ``WAS_Download_TAMSAT_Seasonal`` / ``WAS_Download_TAMSAT_Daily``
        TAMSAT African rainfall estimates.
    ``WAS_Download_AgroIndicators`` / ``WAS_Download_AgroIndicators_daily``
        NMME-derived agroclimatic indicator downloads.

Standalone function
-------------------
plot_map
    Quick Cartopy map of a geographic extent.
"""
from __future__ import annotations
import logging
import os
import cdsapi
import urllib3
import calendar
from calendar import month_abbr
import xarray as xr
import zipfile
import io
import pandas as pd
from pathlib import Path
import xarray as xr
from datetime import timedelta
from datetime import date
from datetime import datetime
import gc
from dask.diagnostics import ProgressBar
import cdsapi
import netCDF4
import h5netcdf
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import requests
from tqdm import tqdm
from wass2s.utils import *
import rioxarray as rioxr
import datetime as dt
import time as _time
import urllib.request
from typing import List, Tuple, Sequence, Optional
# Suppress warnings for urllib3 to avoid SSL certificate verification errors
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.getLogger("cdsapi").setLevel(logging.ERROR)


def _download_atomic(url, path, max_retries=3, retry_delay=5, timeout=180):
    """
    Download `url` to `path` through a temporary `<path>.part` file, renamed
    atomically on success. Consequence: if `path` exists, it is complete.
    Returns True on success, False after exhausting retries.
    """

    tmp = path.parent / (path.name + ".part")
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[DL ] Attempt {attempt}/{max_retries}: {url}")
            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                with open(tmp, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            os.replace(tmp, path)  # atomic on POSIX: path never half-written
            return True
        except Exception as exc:
            print(f"[ERR] Attempt {attempt}/{max_retries} failed: {exc}")
            if tmp.exists():
                os.remove(tmp)
            if attempt < max_retries:
                _time.sleep(retry_delay)
    return False





def _require_c3s_processing_version(path):
    """Reject ambiguous pre-correction caches without silently downloading."""
    with xr.open_dataset(path) as cached:
        version = cached.attrs.get("wass2s_processing_version")
    if version != "calendar-rates-20260911":
        raise ValueError(
            f"C3S cache {path} lacks the current lead/calendar/unit provenance. "
            "It was not re-downloaded. Rebuild explicitly with force_download=True "
            "after preserving any required original; old values cannot be relabelled safely."
        )


def _c3s_monthly_mean(data):
    """Seasonal mean of monthly mean fields, weighted by calendar-day counts."""
    from calendar import monthrange
    if "forecastMonth" not in data.dims:
        raise ValueError("Monthly C3S aggregation needs forecastMonth")
    if "valid_time" in data.coords and "forecastMonth" in data.valid_time.dims:
        days = data.valid_time.dt.days_in_month
    else:
        name = next((k for k in ("forecast_reference_time", "indexing_time", "time") if k in data.coords), None)
        if name is None:
            raise ValueError("Monthly means need valid_time or a forecast initialization")
        if data[name].dt.calendar not in {"standard", "gregorian", "proleptic_gregorian"}:
            raise ValueError("Non-Gregorian calendars require explicit valid_time")
        leads = np.asarray(data.forecastMonth.values)
        if not np.all(np.isfinite(leads)) or not np.all(leads == leads.astype(int)) or np.any(leads < 1):
            raise ValueError("CDS forecastMonth must contain positive integers")
        rows = []
        for value in np.atleast_1d(data[name].values):
            ref = pd.Timestamp(value)
            row = []
            for lead in leads.astype(int):
                year, month0 = divmod(ref.year * 12 + ref.month - 1 + lead - 1, 12)
                row.append(monthrange(year, month0 + 1)[1])
            rows.append(row)
        if data[name].ndim == 0:
            days = xr.DataArray(rows[0], dims="forecastMonth", coords={"forecastMonth":data.forecastMonth})
        else:
            days = xr.DataArray(rows, dims=(name,"forecastMonth"), coords={name:data[name],"forecastMonth":data.forecastMonth})
    out = (data * days).sum("forecastMonth", skipna=False, keep_attrs=True) / days.sum("forecastMonth")
    out.attrs = dict(data.attrs, aggregation="calendar-day-weighted mean of monthly means")
    if isinstance(out, xr.Dataset):
        for name in out.data_vars:
            out[name].attrs = dict(data[name].attrs)
    return out


def _save_netcdf_atomic(dataset, path):
    """Materialize before replacing a cache; failed writes leave it intact."""
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_path = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=path.parent)
    os.close(descriptor)
    try:
        dataset.load().to_netcdf(temp_path)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _remove_quiet(path):
    import os

    try:
        if path.exists():
            os.remove(path)
    except OSError as exc:
        print(f"[WARN] Could not remove {path}: {exc}")




class _CCSRUnavailableError(RuntimeError):
    """The selected CCSR model, variable, or complete season is unavailable."""


def _ccsr_model_spec(model_id, cfsv2_sample="pentad_samples"):
    """Resolve WASS2S identifiers to explicit CCSR products and member labels.

    CanSIPS-IC4 follows the official ECCC member ordering in both archives:
    GEM5.2-NEMO = 1..20; CanESM5 = 21..40.  Selection is by label, not position.
    """
    if cfsv2_sample not in ("pentad_samples", "early_month_samples"):
        raise ValueError("ccsr_cfsv2_sample must be 'pentad_samples' or 'early_month_samples'.")
    models = {
        "CFSV2_1": (f"NOAA-NCEP/CFSv2/{cfsv2_sample}", "CFSv2", None, False),
        "CMC1_1": ("ECCC/CanSIPS-IC4", "CanESM5", tuple(range(21, 41)), False),
        "CMC2_1": ("ECCC/CanSIPS-IC4", "GEM5.2-NEMO", tuple(range(1, 21)), False),
        "NASA_1": ("NASA-GMAO/GEOSS2S", "GEOSS2S", None, False),
        "GFDL_1": ("NOAA-GFDL/SPEAR", "SPEAR", None, False),
        "NCAR_CCSM4_1": ("COLA-RSMAS/CCSM4", "CCSM4", None, True),
        "NCAR_CESM1_1": ("COLA-RSMAS/CESM1", "CESM1", None, True),
    }
    if model_id not in models:
        raise _CCSRUnavailableError(
            f"No member-resolved CCSR product is mapped to {model_id}. "
            "NMME_1 is a multi-model mean, not an individual member archive."
        )
    path, version, members, combined = models[model_id]
    return {"path": path, "model": version, "members": members, "combined": combined}


def _ccsr_is_not_found(exc):
    """Only a proven missing endpoint is an availability failure."""
    import re
    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) == 404:
        return True
    text = str(exc)
    return bool(
        re.search(r"\b(?:HTTP(?: Error)?[: /]*|status(?: code)?[: =]*)404\b", text, re.I)
        or "NetCDF: file not found" in text
    )


def _ccsr_gregorian_dates(coordinate, name):
    """Decode only Gregorian calendars; do not reinterpret 360-day dates."""
    import numpy as np
    import pandas as pd
    calendar_name = coordinate.encoding.get("calendar", coordinate.attrs.get("calendar", "standard"))
    try:
        calendar_name = coordinate.dt.calendar
    except (AttributeError, TypeError):
        pass
    if calendar_name not in ("standard", "gregorian", "proleptic_gregorian"):
        raise ValueError(f"CCSR {name} calendar {calendar_name!r} is not Gregorian.")
    values = np.asarray(coordinate.values)
    if np.issubdtype(values.dtype, np.number):
        raise ValueError(f"CCSR {name} dates must be CF-decoded before seasonal aggregation.")
    result = []
    for value in values.reshape(-1):
        if hasattr(value, "year") and not isinstance(value, np.datetime64):
            result.append(pd.Timestamp(value.year, value.month, value.day,
                                       getattr(value, "hour", 0), getattr(value, "minute", 0),
                                       getattr(value, "second", 0)).to_datetime64())
        else:
            result.append(pd.Timestamp(value).to_datetime64())
    dates = np.asarray(result, dtype="datetime64[ns]").reshape(values.shape)
    if np.isnat(dates).any():
        raise ValueError(f"CCSR {name} contains missing dates.")
    return dates


def _ccsr_geographic_indices(dataset, area):
    """Return bounded coordinate indexes, including a dateline-crossing box."""
    import numpy as np
    bounds = np.asarray(area, dtype=float)
    if bounds.shape != (4,) or not np.isfinite(bounds).all():
        raise ValueError("area must be finite [North, West, South, East].")
    north, west, south, east = bounds
    if not (-90 <= south <= north <= 90):
        raise ValueError("area must satisfy -90 <= South <= North <= 90.")
    for dimension in ("Y", "X"):
        if dimension not in dataset.coords or dataset[dimension].dims != (dimension,):
            raise ValueError(f"CCSR requires a one-dimensional {dimension} coordinate.")
    y = np.asarray(dataset.Y.values, dtype=float)
    x = ((np.asarray(dataset.X.values, dtype=float) + 180) % 360) - 180
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("CCSR spatial coordinates contain non-finite values.")
    yindices = np.flatnonzero((y >= south) & (y <= north))
    if abs(east - west) >= 360:
        xindices = np.arange(x.size)
    else:
        west = (west + 180) % 360 - 180
        east = (east + 180) % 360 - 180
        inside = ((x >= west) & (x <= east)) if west <= east else ((x >= west) | (x <= east))
        xindices = np.flatnonzero(inside)
    if not xindices.size or not yindices.size:
        raise ValueError("The requested area contains no CCSR grid cell.")
    # Preserve a single representative if a source repeats a cyclic meridian.
    _, unique = np.unique(x[xindices], return_index=True)
    xindices = xindices[np.sort(unique)]
    return {"Y": yindices, "X": xindices}


def _ccsr_seasonal_member_values(selected, variable, remote_variable, init_date, leads):
    """Aggregate complete monthly series for each individual member."""
    import re
    import numpy as np
    import pandas as pd
    import xarray as xr
    if "target" not in selected:
        raise ValueError("CCSR monthly data need the target(S,L) coordinate.")
    target = selected["target"]
    if set(target.dims) != {"S", "L"}:
        raise ValueError("CCSR target must have dimensions S,L.")
    actual = _ccsr_gregorian_dates(target.transpose("S", "L"), "target")
    start = pd.Timestamp(init_date)
    expected = np.asarray([
        (start + pd.DateOffset(months=int(lead))).to_datetime64()
        for lead in leads
    ], dtype="datetime64[ns]")[None, :]
    if not np.array_equal(actual, expected):
        raise ValueError("CCSR target dates disagree with initialization plus zero-based lead months.")
    ends = np.asarray([
        (pd.Timestamp(value) + pd.offsets.MonthBegin(1)).to_datetime64()
        for value in expected[0]
    ], dtype="datetime64[ns]")[None, :]
    if "target_bnds" in selected:
        bounds = selected["target_bnds"]
        extra = [dimension for dimension in bounds.dims if dimension not in ("S", "L")]
        if len(extra) != 1 or bounds.sizes[extra[0]] != 2:
            raise ValueError("CCSR target_bnds must contain two bounds per S,L.")
        dates = _ccsr_gregorian_dates(bounds.transpose("S", "L", extra[0]), "target_bnds")
        if not np.array_equal(dates[:, :, 0], expected) or not np.array_equal(dates[:, :, 1], ends):
            raise ValueError("CCSR target_bnds must cover complete Gregorian target months.")
    days = xr.DataArray(
        ((ends - expected) / np.timedelta64(1, "D")).astype(float),
        dims=("S", "L"), coords={"S": selected.S, "L": selected.L},
    )
    data = selected[remote_variable]
    if set(data.dims) != {"S", "M", "L", "Y", "X"}:
        raise ValueError(f"CCSR {remote_variable} must have dimensions S,M,L,Y,X; found {data.dims}.")
    if np.isinf(np.asarray(data.values)).any():
        raise ValueError("CCSR contains infinite data values.")
    original_units = str(data.attrs.get("units", "")).strip()
    units = re.sub(r"[\s_\^*]", "", original_units.lower()).replace("−", "-")
    units = units.replace("per", "/")
    if variable == "PRCP":
        if units in ("mm/day", "mmday-1", "mm/d", "mmd-1", "kgm-2day-1", "kgm-2d-1"):
            daily = data
        elif units in ("mm/s", "mms-1", "kgm-2s-1", "kg/m2/s"):
            daily = data * 86400.0
        elif units in ("m/day", "mday-1", "m/d", "md-1"):
            daily = data * 1000.0
        elif units in ("m/s", "ms-1"):
            daily = data * 86400000.0
        elif units in ("mm/month", "mmmonth-1"):
            daily = data / days
        elif units == "mm" and re.search(r"\btime\s*:\s*sum\b", str(data.attrs.get("cell_methods", ""))):
            daily = data / days
        else:
            raise ValueError(f"Unsupported or ambiguous CCSR precipitation units {original_units!r}.")
        result = (daily * days).sum("L", skipna=False)
        output_units = "mm"
        aggregation = "sum of complete monthly precipitation amounts per ensemble member"
    else:
        if units in ("k", "kelvin", "degk", "degreekelvin", "degreeskelvin"):
            data = data - 273.15
        elif units not in ("c", "°c", "degc", "celsius", "degreecelsius", "degreescelsius", "degreec", "degreesc"):
            raise ValueError(f"Unsupported CCSR temperature units {original_units!r}.")
        result = (data * days).sum("L", skipna=False) / days.sum("L")
        output_units = "degC"
        aggregation = "calendar-day-weighted mean of complete monthly means per ensemble member"
    result = result.rename({"S": "T"}).rename(variable)
    # Monthly auxiliary coordinates do not describe an aggregated season.
    result = result.drop_vars([name for name in result.coords if name not in ("T", "M", "Y", "X")])
    result = result.assign_coords(X=((result.X + 180) % 360) - 180).sortby("X").sortby("Y")
    result = result.transpose("T", "M", "Y", "X")
    result.attrs = dict(units=output_units, source_units=original_units,
                        source_variable=remote_variable, aggregation=aggregation)
    return result


def _ccsr_download_season(model_id, variable, years, init_month, leads, area, ensemble_mean,
                          *, base_url="https://forecast.ccsr.columbia.edu/data/NMME",
                          cfsv2_sample="pentad_samples", opener=None, engine="netcdf4"):
    """Download bounded monthly CCSR slices and return WASS2S seasonal data.

    ``leads`` contains zero-based month offsets: April + [3,4,5] = JAS.
    Hindcasts are preferred; the forecast archive fills only missing requested
    initializations.  A requested year with no finite complete member season is
    unavailable, never a silently shortened time series.  Spatially incomplete
    cells remain NaN.  Ensemble reduction follows per-member seasonal aggregation.
    """
    import numpy as np
    import pandas as pd
    import xarray as xr
    if isinstance(ensemble_mean, str):
        ensemble_mean = ensemble_mean.strip().lower()
        if ensemble_mean == "none":
            ensemble_mean = None
    if ensemble_mean not in (None, "mean", "median"):
        raise ValueError("ensemble_mean must be None, 'mean', or 'median'.")
    if engine not in ("netcdf4", "pydap"):
        raise ValueError("ccsr_engine must be 'netcdf4' or 'pydap'.")
    spec = _ccsr_model_spec(model_id, cfsv2_sample)
    remote_variable = {"PRCP": "pr", "TEMP": "tas", "TMAX": "tasmax", "TMIN": "tasmin", "SST": "tos"}.get(variable)
    if remote_variable is None:
        raise _CCSRUnavailableError(f"CCSR variable {variable!r} is not mapped.")
    try:
        requested_years = np.asarray(list(years), dtype=float)
        requested_leads = np.asarray(list(leads), dtype=float)
        month = float(init_month)
    except (TypeError, ValueError) as exc:
        raise ValueError("CCSR years, initialization month, and leads must be integers.") from exc
    if (requested_years.ndim != 1 or requested_years.size == 0
            or not np.isfinite(requested_years).all()
            or not np.equal(requested_years, np.floor(requested_years)).all()
            or (requested_years < 1).any()):
        raise ValueError("CCSR years must be a non-empty sequence of positive integers.")
    if (requested_leads.ndim != 1 or requested_leads.size == 0
            or not np.isfinite(requested_leads).all()
            or not np.equal(requested_leads, np.floor(requested_leads)).all()
            or (requested_leads < 0).any()
            or len(np.unique(requested_leads)) != len(requested_leads)):
        raise ValueError("CCSR leads must contain distinct non-negative integer month offsets.")
    if not np.isfinite(month) or month != int(month) or not 1 <= month <= 12:
        raise ValueError("CCSR initialization month must be an integer from 1 to 12.")
    requested_years = sorted(set(requested_years.astype(int).tolist()))
    requested_leads = np.sort(requested_leads.astype(int))
    month = int(month)
    open_dataset = xr.open_dataset if opener is None else opener
    partitions = ("combined",) if spec["combined"] else ("hindcast", "forecast")
    seasons, origins, urls = {}, {}, []
    for partition in partitions:
        pending = [year for year in requested_years if year not in seasons]
        if not pending:
            break
        suffix = remote_variable if partition == "combined" else f"{partition}/{remote_variable}"
        url = f"{str(base_url).rstrip('/')}/{spec['path']}/{suffix}"
        opened = None
        try:
            opened = open_dataset(url, decode_times=True, engine=engine)
            if remote_variable not in opened:
                raise ValueError(f"CCSR endpoint {url} does not contain {remote_variable}.")
            for dimension in ("S", "L", "M", "Y", "X"):
                if dimension not in opened.coords or opened[dimension].dims != (dimension,):
                    raise ValueError(f"CCSR requires coordinate {dimension}({dimension}).")
            dates = _ccsr_gregorian_dates(opened.S, "S")
            index = pd.DatetimeIndex(dates)
            if index.has_duplicates or not ((index.day == 1) & (index.hour == 0)
                                             & (index.minute == 0) & (index.second == 0)).all():
                raise ValueError("CCSR S must contain unique first-of-month initializations.")
            if not np.isfinite(np.asarray(opened.M.values, dtype=float)).all() or len(np.unique(opened.M.values)) != opened.sizes["M"]:
                raise ValueError("CCSR ensemble member labels must be finite and unique.")
            selected_years = [year for year in pending if pd.Timestamp(year, month, 1) in index]
            if not selected_years:
                continue
            available_leads = np.asarray(opened.L.values)
            if len(np.unique(available_leads)) != len(available_leads):
                raise ValueError("CCSR lead labels must be unique.")
            if not np.isin(requested_leads, available_leads).all():
                raise _CCSRUnavailableError(f"CCSR {model_id} has no requested leads {requested_leads.tolist()} in {partition}.")
            selection = _ccsr_geographic_indices(opened, area)
            selection["L"] = [int(np.flatnonzero(available_leads == lead)[0]) for lead in requested_leads]
            if spec["members"] is not None:
                expected_members = np.arange(1, 41)
                if not np.array_equal(np.sort(np.asarray(opened.M.values)), expected_members):
                    raise ValueError("CCSR CanSIPS-IC4 member labels must be exactly 1..40 before selecting its component model.")
                selection["M"] = [int(np.flatnonzero(opened.M.values == member)[0]) for member in spec["members"]]
            for year in selected_years:
                initialization = pd.Timestamp(year, month, 1)
                selection["S"] = [int(index.get_loc(initialization))]
                # Restrict time, lead, members, and geography before data transfer.
                subset = opened.isel(selection).load()
                season = _ccsr_seasonal_member_values(subset, variable, remote_variable,
                                                      initialization, requested_leads)
                if not np.isfinite(np.asarray(season.values)).any():
                    continue
                seasons[year] = season
                origins[year] = partition
                if url not in urls:
                    urls.append(url)
        except Exception as exc:
            if not _ccsr_is_not_found(exc):
                raise
        finally:
            if opened is not None:
                opened.close()
    missing = [year for year in requested_years if year not in seasons]
    if missing:
        raise _CCSRUnavailableError(
            f"CCSR {model_id}.{variable} has no complete finite season for initialization "
            f"month {month:02d}, years {missing}, leads {requested_leads.tolist()}. "
            "No partial archive was returned."
        )
    # Archive partitions must share a spatial grid: only the member axis may
    # be enlarged during concatenation.  An outer spatial union would silently
    # invent missing cells and could change regional statistics.
    first_season = seasons[requested_years[0]]
    for year in requested_years[1:]:
        for dimension in ("Y", "X"):
            if not np.array_equal(seasons[year][dimension].values,
                                  first_season[dimension].values):
                raise ValueError(
                    f"CCSR {model_id} spatial {dimension} coordinates differ between "
                    "requested initializations; align the source grids explicitly."
                )
    # Different archive partitions may expose different member counts.  Keep
    # actual labels and mark absent members with NaN; never replicate members.
    combined = xr.concat([seasons[year] for year in requested_years], dim="T", join="outer")
    combined = combined.assign_coords(source_partition=("T", [origins[year] for year in requested_years]))
    if ensemble_mean is not None:
        combined = getattr(combined, ensemble_mean)("M", skipna=True, keep_attrs=True)
    result = combined.to_dataset(name=variable)
    result.attrs = {
        "wass2s_nmme_source": "ccsr",
        "wass2s_ccsr_processing_version": "ccsr-calendar-members-20260916",
        "wass2s_model_id": model_id,
        "wass2s_source_model": spec["model"],
        "wass2s_source_product": spec["path"],
        "wass2s_source_urls": "\n".join(urls),
        "wass2s_ensemble_statistic": "members" if ensemble_mean is None else ensemble_mean,
        "wass2s_initialization_month": month,
        "wass2s_lead_month_offsets": ",".join(str(value) for value in requested_leads),
        "wass2s_requested_years": ",".join(str(value) for value in requested_years),
        "wass2s_requested_area": ",".join(str(float(value)) for value in area),
        "wass2s_calendar": "proleptic_gregorian",
        "wass2s_missing_month_policy": "complete season required per member and grid cell",
    }
    if spec["members"] is not None:
        result.attrs["wass2s_source_member_labels"] = ",".join(map(str, spec["members"]))
    return result


def _ccsr_request_key(model_id, variable, years, month, leads, area, base_url, sampling):
    """Canonical request provenance: filenames alone omit years and geography."""
    import json
    import numpy as np

    def integers(values, name, minimum):
        numeric = np.asarray(list(values), dtype=float)
        if (numeric.ndim != 1 or not numeric.size or not np.isfinite(numeric).all()
                or (numeric != np.floor(numeric)).any() or (numeric < minimum).any()
                or len(np.unique(numeric)) != len(numeric)):
            raise ValueError(f"CCSR {name} must contain distinct integers >= {minimum}.")
        return sorted(numeric.astype(int).tolist())

    init_month = integers([month], "initialization month", 1)[0]
    if init_month > 12:
        raise ValueError("CCSR initialization month must be in 1..12.")
    if sampling not in ("pentad_samples", "early_month_samples"):
        raise ValueError("ccsr_cfsv2_sample must be 'pentad_samples' or 'early_month_samples'.")
    bounds = np.asarray(area, dtype=float)
    if bounds.shape != (4,) or not np.isfinite(bounds).all() or not (-90 <= bounds[2] <= bounds[0] <= 90):
        raise ValueError("CCSR area must be finite [North, West, South, East], with -90 <= South <= North <= 90.")
    return json.dumps({
        "version": "ccsr-members-calendar-20260916", "model": model_id,
        "variable": variable, "years": integers(years, "years", 1),
        "month": init_month, "leads": integers(leads, "leads", 0),
        "area": bounds.tolist(), "base_url": str(base_url).rstrip("/"),
        "sampling": sampling if model_id == "CFSV2_1" else None,
    }, sort_keys=True, separators=(",", ":"))


def _ccsr_cache_matches(path, request_key, statistic):
    """Reuse only an identical CCSR request with the requested member semantics."""
    import json
    import warnings
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import xarray as xr
    if not Path(path).is_file():
        return False
    request = json.loads(request_key)
    try:
        with xr.open_dataset(path) as cached:
            variable = cached.get(request["variable"])
            dims = ("T", "M", "Y", "X") if statistic == "members" else ("T", "Y", "X")
            expected_dates = pd.to_datetime([f"{year:04d}-{request['month']:02d}-01" for year in request["years"]])
            matches = (
                cached.attrs.get("wass2s_ccsr_request") == request_key
                and cached.attrs.get("wass2s_processing_version") == request["version"]
                and cached.attrs.get("wass2s_nmme_source") == "ccsr"
                and cached.attrs.get("wass2s_ensemble_statistic") == statistic
                and variable is not None and variable.dims == dims
                and all(cached.sizes[dim] > 0 for dim in dims)
                and np.array_equal(cached.T.values, expected_dates.values)
            )
            if matches:
                return True
    except (OSError, ValueError, KeyError):
        pass
    warnings.warn(f"Ignoring incompatible CCSR cache {path}: source, request or ensemble representation differs.", RuntimeWarning)
    return False


def _as_tuple(value, default):
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _select_variable(ds, candidates):
    candidates = _as_tuple(candidates, default=())
    for name in candidates:
        if name in ds.data_vars:
            return name
    raise KeyError(f"None of {candidates} found. Available variables: {list(ds.data_vars)}")


def _standardize_latlon_da(da, area=None):
    rename = {}
    if "lon" in da.dims or "lon" in da.coords:
        rename["lon"] = "X"
    if "longitude" in da.dims or "longitude" in da.coords:
        rename["longitude"] = "X"
    if "lat" in da.dims or "lat" in da.coords:
        rename["lat"] = "Y"
    if "latitude" in da.dims or "latitude" in da.coords:
        rename["latitude"] = "Y"
    da = da.rename({old: new for old, new in rename.items() if old in da.dims or old in da.coords})

    if "X" in da.coords and float(da["X"].max()) > 180:
        da = da.assign_coords(X=(((da["X"] + 180) % 360) - 180)).sortby("X")
    if "Y" in da.coords:
        da = da.sortby("Y")
    if "X" in da.coords:
        da = da.sortby("X")

    if area is not None:
        north, west, south, east = area
        da = da.sel(X=slice(west, east), Y=slice(south, north))

    drop = [name for name in ["spatial_ref", "crs"] if name in da.coords]
    if drop:
        da = da.drop_vars(drop, errors="ignore")
    return da

# Helpers used only by the complementary downloaders.
def _was_cache_key(**identity):
    """Deterministic configuration key; never pass tokens/passwords here."""
    import hashlib
    import json
    return hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _was_complete_monthly_axis(out, year_start, year_end, closed_only=True):
    """Expose absent edge months too; a missing month is never silently omitted."""
    first = pd.Timestamp(int(year_start), 1, 1)
    last = pd.Timestamp(int(year_end), 12, 1)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    last_allowed = now.normalize().replace(day=1)
    if closed_only:
        last_allowed -= pd.offsets.MonthBegin(1)
    last = min(last, last_allowed)
    if first > last:
        raise ValueError("No eligible calendar month in the requested period.")
    times = pd.date_range(first, last, freq="MS")
    out = out.reindex(T=times)
    out["N_VALID_DAYS"] = out["N_VALID_DAYS"].fillna(0).astype("int16")
    out["DAILY_COVERAGE"] = out["DAILY_COVERAGE"].fillna(0).astype("float32")
    out["EXPECTED_DAYS"] = ("T", times.days_in_month.to_numpy(dtype="int16"))
    return out


def _was_concat_files(files, time_dim="T", max_eager_bytes=512 * 1024**2):
    """Lazy concatenation with Dask; explicit bounded-memory fallback otherwise.

    The caller must close the returned Dataset. No spatial outer join and no
    silent duplicate timestamp removal are allowed.
    """
    import importlib.util
    files = list(files)
    if not files:
        raise ValueError("No files available for concatenation.")
    if importlib.util.find_spec("dask") is not None:
        out = xr.open_mfdataset(files, combine="nested", concat_dim=time_dim,
                               chunks={time_dim: 31}, parallel=False,
                               join="exact", data_vars="minimal", coords="minimal", compat="equals")
        close = out.close
        try:
            out = out.sortby(time_dim)
            if not out.get_index(time_dim).is_unique:
                raise ValueError("Duplicate timestamps across cache files; inspect overlapping source periods.")
            out.set_close(close)
            return out
        except Exception:
            close()
            raise
    total = 0
    for path in files:
        with xr.open_dataset(path) as item:
            total += item.nbytes
    # Concatenation needs additional copies, hence a conservative small ceiling.
    if total > max_eager_bytes:
        raise ImportError(f"Dask is needed to assemble {total / 1024**2:.1f} MiB of data safely. Install dask[array].")
    arrays = []
    for path in files:
        with xr.open_dataset(path) as item:
            arrays.append(item.load())
    out = xr.concat(arrays, dim=time_dim, join="exact", data_vars="minimal", coords="minimal", compat="equals").sortby(time_dim)
    if not out.get_index(time_dim).is_unique:
        raise ValueError("Duplicate timestamps across cache files; inspect overlapping source periods.")
    return out


def _was_monthly_rainfall(ds, threshold=1.0, complete_months_only=True, now=None):
    """Calendar-month sums with full-calendar denominator and per-pixel QC.

    A relaxed threshold accepts an observed PARTIAL sum; no extrapolation to
    the missing days is performed. Duplicate dates are rejected.
    """
    if not 0 < float(threshold) <= 1:
        raise ValueError("min_daily_coverage must be in (0, 1].")
    if not ds.sizes.get("T", 0):
        raise ValueError("Empty daily dataset.")
    times = pd.DatetimeIndex(ds["T"].values)
    if not times.is_unique or not times.equals(times.normalize()):
        raise ValueError("Daily totals require one unique, midnight-labelled date per observation.")
    now = pd.Timestamp.now(tz="UTC").tz_localize(None) if now is None else pd.Timestamp(now)
    if now.tzinfo is not None:
        now = now.tz_convert("UTC").tz_localize(None)
    ds = ds.sortby("T")
    if complete_months_only:
        ds = ds.sel(T=slice(None, now.normalize().replace(day=1) - pd.Timedelta(days=1)))
    if not ds.sizes.get("T", 0):
        raise ValueError("No closed calendar month in the daily data.")
    pr = ds["PRCP"].where(np.isfinite(ds["PRCP"]) & (ds["PRCP"] >= 0))
    total = pr.resample(T="MS").sum(skipna=True, min_count=1)
    n = pr.notnull().resample(T="MS").sum().fillna(0).astype("int16")
    expected = xr.DataArray(pd.DatetimeIndex(total["T"].values).days_in_month.to_numpy(dtype="int16"),
                            dims="T", coords={"T": total["T"]}, name="EXPECTED_DAYS")
    cov = (n / expected).astype("float32")
    total = total.where(cov >= float(threshold)).astype("float32")
    total.attrs.update(units="mm/month", aggregation="sum of available daily totals; not scaled for missing days")
    cov.attrs.update(units="1", long_name="Fraction of calendar days with a finite nonnegative rainfall value")
    result = xr.Dataset(dict(PRCP=total, N_VALID_DAYS=n, DAILY_COVERAGE=cov, EXPECTED_DAYS=expected))
    result.attrs.update(min_daily_coverage=float(threshold), WASS2S_dimensions="T,Y,X", missing_day_scaling="none")
    return result


def _was_permanent_http_error(exc):
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is None:
        status = getattr(exc, "code", None)  # urllib HTTPError
    return status in {400, 401, 403, 404, 405, 410, 422}


def _was_write_netcdf(obj, path, *, encoding=None, compute=True, **kwargs):
    """Atomic replacement; never leave a half-written final NetCDF file.

    netCDF4/h5netcdf compression is used when installed. Small-data/test
    installations can use scipy (no compression); this fallback is announced.
    A failed write leaves any previously valid destination untouched.
    """
    import importlib.util
    import os
    import tempfile
    import warnings
    if not compute:
        from dask import delayed
        return delayed(_was_write_netcdf)(obj, path, encoding=encoding, compute=True, **kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    enc = {k: dict(v) for k, v in (encoding or {}).items()}
    if "engine" not in kwargs:
        if importlib.util.find_spec("netCDF4") is not None:
            kwargs["engine"] = "netcdf4"
        elif importlib.util.find_spec("h5netcdf") is not None:
            kwargs["engine"] = "h5netcdf"
        else:
            kwargs["engine"] = "scipy"
            dropped = {"zlib", "complevel", "shuffle", "fletcher32", "chunksizes", "contiguous", "compression", "compression_opts"}
            if any(dropped.intersection(v) for v in enc.values()):
                warnings.warn("netCDF4/h5netcdf unavailable: writing uncompressed NetCDF with scipy.", RuntimeWarning)
            enc = {k: {a: b for a, b in v.items() if a not in dropped} for k, v in enc.items()}
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=path.parent)
    os.close(fd)
    try:
        obj.to_netcdf(tmp, encoding=enc, **kwargs)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


class WAS_Download:
    def __init__(self):
        """Initialize the WAS_Download class."""
        pass


    def ModelsName(
        self,
        centre={
            "BOM_2": "bom",
            "ECMWF_51": "ecmwf",
            "UKMO_604": "ukmo",
            "UKMO_603": "ukmo",
            "UKMO_605": "ukmo",
            "UKMO_610": "ukmo",
            "METEOFRANCE_8": "meteo_france",
            "METEOFRANCE_9": "meteo_france",
            "DWD_21": "dwd", # month of initialization available for forecast are Jan to Mar
            "DWD_22": "dwd", # month of initialization available for forecast are Apr to __ 
            "CMCC_35": "cmcc",
            "CMCC_4": "cmcc",
            "NCEP_2": "ncep",
            "JMA_3": "jma",
            "JMA_4": "jma",
            "ECCC_4": "eccc",
            "ECCC_5": "eccc",
            # "CFSV2": "CFS",
            # "CMC1": "cmc1",
            # "CMC2": "cmc2",
            # "GFDL": "gfdl",
            # "NASA": "nasa",
            # "NCAR_CCSM4": "ncar",
            # "NMME" : "nmme"
            "CFSV2_1": "cfsv2",
            "CMC1_1": "cmc1",
            "CMC2_1": "cmc2",
            "GFDL_1": "gfdl",
            "NASA_1": "nasa",
            "NCAR_CCSM4_1": "ncar_ccsm4",
            "NCAR_CESM1_1": "ncar_cesm1",
            "NMME_1" : "nmme",
        },
        variables_1={
            "PRCP":  "total_precipitation",
            "TEMP":  "2m_temperature",
            "TDEW": "2m_dewpoint_temperature",
            "TMAX":  "maximum_2m_temperature_in_the_last_24_hours",
            "TMIN":  "minimum_2m_temperature_in_the_last_24_hours",
            "UGRD10":"10m_u_component_of_wind",
            "VGRD10":"10m_v_component_of_wind",
            "SST":   "sea_surface_temperature",
            "SLP": "mean_sea_level_pressure",
            "DSWR": "surface_solar_radiation_downwards",
            "DLWR": "surface_thermal_radiation_downwards",
            "NOLR": "top_net_thermal_radiation",
            "RUNOFF":"mean_surface_runoff_rate"
        },
        variables_2={
            "HUSS_1000": "specific_humidity",
            "HUSS_925": "specific_humidity",
            "HUSS_850": "specific_humidity",
            "UGRD_1000": "u_component_of_wind",
            "UGRD_925": "u_component_of_wind",
            "UGRD_850": "u_component_of_wind",
            "VGRD_1000": "v_component_of_wind",
            "VGRD_925": "v_component_of_wind",
            "VGRD_850": "v_component_of_wind",
        },
    ):
        """
        Generate a combined dictionary of model names and variables. 
        For more information on C3S, browse the `MetaData <https://confluence.ecmwf.int/display/CKB/Description+of+the+C3S+seasonal+multi-system>`_.
        For more information on NMME, browse the `MetaData <https://confluence.ecmwf.int/display/CKB/Description+of+the+C3S+seasonal+multi-system>`_.

        Parameters:
            centre (dict): Mapping of model identifiers to model names.
            variables_1 (dict): Mapping of variable short names to full names for category 1.
            variables_2 (dict): Mapping of variable short names to full names for category 2.

        Returns:
            dict: A combined dictionary with keys as model.variable combinations and values as tuples (model name, variable name).
        """
        combined_dict1 = {
            f"{c}.{v}": (centre[c], variables_1[v]) for c in centre for v in variables_1
        }
        combined_dict2 = {
            f"{c}.{v}": (centre[c], variables_2[v]) for c in centre for v in variables_2
        }
        combined_dict = {**combined_dict1, **combined_dict2}
        return combined_dict

    
    # def ModelsName(
    #     self,
    #     source=None,
    #     variable=None,
    #     product="monthly",
    #     include_pressure=True,
    #     include_wmolc_aliases=False,
    #     return_metadata=False,
    # ):
    #     """
    #     Return valid model-variable combinations supported by the downloader.
    
    #     This method avoids creating invalid combinations. For example, WMOLC models
    #     such as Beijing, CPTEC or Moscow are not paired with RUNOFF because the
    #     current WMOLC direct-download files do not provide runoff.
    
    #     Parameters
    #     ----------
    #     source : str or list of str, optional
    #         Source family to include. Accepted values are:
    #         "c3s", "cds", "nmme", "wmolc", or None for all.
    
    #     variable : str or list of str, optional
    #         Variable short name(s) to keep, for example "PRCP", "RUNOFF",
    #         ["PRCP", "TEMP"], etc. If None, all valid variables are returned.
    
    #     product : {"monthly", "daily"}, default "monthly"
    #         Product context. This matters for a few CDS variables whose native
    #         names differ between monthly and original daily data, especially RUNOFF.
    
    #     include_pressure : bool, default True
    #         If True, include pressure-level variables such as UGRD_850, VGRD_200,
    #         HUSS_925, etc.
    
    #     include_wmolc_aliases : bool, default False
    #         If True, also include keys such as "WMOLC_Beijing.PRCP".
    #         Default is False to avoid duplicate downloads.
    
    #     return_metadata : bool, default False
    #         If False, returns the old style:
    #             {"ECMWF_51.PRCP": ("ecmwf", "total_precipitation")}
    #         If True, returns rich metadata for each key.
    
    #     Returns
    #     -------
    #     dict
    #         Dictionary of valid model-variable combinations.
    #     """

    #     # center_variable = [key for key in downloader.ModelsName().keys() if "RUNOFF" in key]
    #     # center_variable = list(downloader.ModelsName(variable="RUNOFF").keys())
    #     # wmolc_prcp = list(downloader.ModelsName(source="wmolc", variable="PRCP").keys())
    #     # models = downloader.ModelsName(variable="PRCP", return_metadata=True)
        
    #     # for key, meta in models.items():
    #     #     print(key, meta["source"], meta["centre"], meta["native_variable"])

    #     # product="monthly" -> RUNOFF = "mean_surface_runoff_rate"
    #     # product="daily"   -> RUNOFF = "surface_runoff"
        
    #     if product not in {"monthly", "daily"}:
    #         raise ValueError("product must be either 'monthly' or 'daily'.")
    
    #     def _as_set(x):
    #         if x is None:
    #             return None
    #         if isinstance(x, str):
    #             return {x.lower()}
    #         return {str(i).lower() for i in x}
    
    #     def _as_var_set(x):
    #         if x is None:
    #             return None
    #         if isinstance(x, str):
    #             return {x.upper()}
    #         return {str(i).upper() for i in x}
    
    #     source_filter = _as_set(source)
    #     variable_filter = _as_var_set(variable)
    
    #     def _source_allowed(name):
    #         if source_filter is None:
    #             return True
    #         name = name.lower()
    #         if name == "cds":
    #             name = "c3s"
    #         return name in source_filter or ("cds" in source_filter and name == "c3s")
    
    #     def _variable_allowed(v):
    #         return variable_filter is None or v.upper() in variable_filter
    
    #     def _native_name(vmap, product_name):
    #         if isinstance(vmap, dict):
    #             if product_name in vmap:
    #                 return vmap[product_name]
    #             if "monthly" in vmap:
    #                 return vmap["monthly"]
    #             if "daily" in vmap:
    #                 return vmap["daily"]
    #             return next(iter(vmap.values()))
    #         return vmap
    
    #     # ------------------------------------------------------------------
    #     # 1. C3S / CDS seasonal models
    #     # ------------------------------------------------------------------
    #     c3s_models = {
    #         "BOM_2":          {"centre": "bom",          "system": "2"},
    #         "ECMWF_51":       {"centre": "ecmwf",        "system": "51"},
    #         "UKMO_603":       {"centre": "ukmo",         "system": "603"},
    #         "UKMO_604":       {"centre": "ukmo",         "system": "604"},
    #         "UKMO_605":       {"centre": "ukmo",         "system": "605"},
    #         "UKMO_610":       {"centre": "ukmo",         "system": "610"},
    #         "METEOFRANCE_8":  {"centre": "meteo_france", "system": "8"},
    #         "METEOFRANCE_9":  {"centre": "meteo_france", "system": "9"},
    #         "DWD_21":         {"centre": "dwd",          "system": "21"},
    #         "DWD_22":         {"centre": "dwd",          "system": "22"},
    #         "CMCC_4":         {"centre": "cmcc",         "system": "4"},
    #         "CMCC_35":        {"centre": "cmcc",         "system": "35"},
    #         "NCEP_2":         {"centre": "ncep",         "system": "2"},
    #         "JMA_3":          {"centre": "jma",          "system": "3"},
    #         "JMA_4":          {"centre": "jma",          "system": "4"},
    #         "ECCC_4":         {"centre": "eccc",         "system": "4"},
    #         "ECCC_5":         {"centre": "eccc",         "system": "5"},
    #     }
    
    #     c3s_single_variables = {
    #         "PRCP":   "total_precipitation",
    #         "TEMP":   "2m_temperature",
    #         "TDEW":   "2m_dewpoint_temperature",
    #         "TMAX":   "maximum_2m_temperature_in_the_last_24_hours",
    #         "TMIN":   "minimum_2m_temperature_in_the_last_24_hours",
    #         "UGRD10": "10m_u_component_of_wind",
    #         "VGRD10": "10m_v_component_of_wind",
    #         "SST":    "sea_surface_temperature",
    #         "SLP":    "mean_sea_level_pressure",
    #         "DSWR":   "surface_solar_radiation_downwards",
    #         "DLWR":   "surface_thermal_radiation_downwards",
    #         "NOLR":   "top_net_thermal_radiation",
    #         "SRUNOFF": {
    #             "monthly": "mean_surface_runoff_rate",
    #             "daily": "surface_runoff",
    #         },
    #         "RUNOFF": {
    #             "monthly": "mean_runoff_rate",
    #             "daily": "runoff",
    #         },
    #     }
    
    #     c3s_pressure_variables = {
    #         "HUSS_1000": "specific_humidity",
    #         "HUSS_925":  "specific_humidity",
    #         "HUSS_850":  "specific_humidity",
    #         "UGRD_1000": "u_component_of_wind",
    #         "UGRD_925":  "u_component_of_wind",
    #         "UGRD_850":  "u_component_of_wind",
    #         "UGRD_700":  "u_component_of_wind",
    #         "UGRD_600":  "u_component_of_wind",
    #         "UGRD_500":  "u_component_of_wind",
    #         "UGRD_200":  "u_component_of_wind",
    #         "VGRD_1000": "v_component_of_wind",
    #         "VGRD_925":  "v_component_of_wind",
    #         "VGRD_850":  "v_component_of_wind",
    #         "VGRD_700":  "v_component_of_wind",
    #         "VGRD_600":  "v_component_of_wind",
    #         "VGRD_500":  "v_component_of_wind",
    #         "VGRD_200":  "v_component_of_wind",
    #         "HGT_500":   "geopotential",
    #     }
    
    #     # ------------------------------------------------------------------
    #     # 2. NMME models
    #     #    Important: NMME does not provide all C3S variables in your current
    #     #    downloader. Do not create RUNOFF, DSWR, DLWR, etc. here.
    #     # ------------------------------------------------------------------
    #     nmme_models = {
    #         "CFSV2_1":       {"centre": "cfsv2",       "system": "1", "netcdf_model": "CFSv2"},
    #         "CMC1_1":        {"centre": "cmc1",        "system": "1", "netcdf_model": "CanESM5"},
    #         "CMC2_1":        {"centre": "cmc2",        "system": "1", "netcdf_model": "GEM5.2_NEMO"},
    #         "GFDL_1":        {"centre": "gfdl",        "system": "1", "netcdf_model": "GFDL-SPEAR"},
    #         "NASA_1":        {"centre": "nasa",        "system": "1", "netcdf_model": "NASA_GEOS5v2"},
    #         "NCAR_CCSM4_1":  {"centre": "ncar_ccsm4",  "system": "1", "netcdf_model": "NCAR_CCSM4"},
    #         "NCAR_CESM1_1":  {"centre": "ncar_cesm1",  "system": "1", "netcdf_model": "NCAR_CESM1"},
    #         "NMME_1":        {"centre": "nmme",        "system": "1", "netcdf_model": "NMME"},
    #     }
    
    #     nmme_variables = {
    #         "PRCP": {"cpt": "precip", "netcdf": "prate"},
    #         "TEMP": {"cpt": "tmp2m",  "netcdf": "tmp2m"},
    #         "SST":  {"cpt": "sst",    "netcdf": "tmpsfc"},
    #     }
    
    #     # ------------------------------------------------------------------
    #     # 3. WMO Lead Centre direct-download models
    #     #    Only include variables for which your WMOLC downloader has a file
    #     #    naming rule.
    #     # ------------------------------------------------------------------
    #     wmolc_models = {
    #         "Beijing":   {"centre": "Beijing",   "file_prefix": "beijing"},
    #         "CPTEC":     {"centre": "CPTEC",     "file_prefix": "cptec"},
    #         "Offenbach": {"centre": "Offenbach", "file_prefix": "offenbach"},
    #         "Montreal":  {"centre": "Montreal",  "file_prefix": "montreal"},
    #         "Pune":      {"centre": "Pune",      "file_prefix": "pune"},
    #         "Pretoria":  {"centre": "Pretoria",  "file_prefix": "pretoria"},
    #         "Moscow":    {"centre": "Moscow",    "file_prefix": "moscow"},
    #     }
    
    #     wmolc_variables = {
    #         "PRCP": {"file_suffix": "prec", "standard_name": "precipitation"},
    #         "TEMP": {"file_suffix": "t2m",  "standard_name": "2m_temperature"},
    #         "T2M":  {"file_suffix": "t2m",  "standard_name": "2m_temperature"},
    #         "SST":  {"file_suffix": "sst",  "standard_name": "sea_surface_temperature"},
    #         "SLP":  {"file_suffix": "mslp", "standard_name": "mean_sea_level_pressure"},
    #         "MSLP": {"file_suffix": "mslp", "standard_name": "mean_sea_level_pressure"},
    #         "H500": {"file_suffix": "h500", "standard_name": "geopotential_height_500"},
    #         "Z500": {"file_suffix": "h500", "standard_name": "geopotential_height_500"},
    #     }
    
    #     registry = {}
    
    #     def _add(key, metadata):
    #         if return_metadata:
    #             registry[key] = metadata
    #         else:
    #             registry[key] = (
    #                 metadata.get("centre"),
    #                 metadata.get("native_variable"),
    #             )
    
    #     # Add C3S/CDS
    #     if _source_allowed("c3s"):
    #         for model_id, model_meta in c3s_models.items():
    #             for var_code, var_native in c3s_single_variables.items():
    #                 if not _variable_allowed(var_code):
    #                     continue
    
    #                 native = _native_name(var_native, product)
    #                 key = f"{model_id}.{var_code}"
    
    #                 _add(
    #                     key,
    #                     {
    #                         "source": "c3s",
    #                         "model_id": model_id,
    #                         "centre": model_meta["centre"],
    #                         "system": model_meta["system"],
    #                         "variable": var_code,
    #                         "native_variable": native,
    #                         "level_type": "single",
    #                         "product": product,
    #                         "dataset_monthly": "seasonal-monthly-single-levels",
    #                         "dataset_daily": "seasonal-original-single-levels",
    #                     },
    #                 )
    
    #             if include_pressure:
    #                 for var_code, native in c3s_pressure_variables.items():
    #                     if not _variable_allowed(var_code):
    #                         continue
    
    #                     level = var_code.split("_")[-1]
    #                     key = f"{model_id}.{var_code}"
    
    #                     _add(
    #                         key,
    #                         {
    #                             "source": "c3s",
    #                             "model_id": model_id,
    #                             "centre": model_meta["centre"],
    #                             "system": model_meta["system"],
    #                             "variable": var_code,
    #                             "native_variable": native,
    #                             "pressure_level": level,
    #                             "level_type": "pressure",
    #                             "product": product,
    #                             "dataset_monthly": "seasonal-monthly-pressure-levels",
    #                             "dataset_daily": "seasonal-original-pressure-levels",
    #                         },
    #                     )
    
    #     # Add NMME
    #     if _source_allowed("nmme"):
    #         for model_id, model_meta in nmme_models.items():
    #             for var_code, var_meta in nmme_variables.items():
    #                 if not _variable_allowed(var_code):
    #                     continue
    
    #                 key = f"{model_id}.{var_code}"
    
    #                 _add(
    #                     key,
    #                     {
    #                         "source": "nmme",
    #                         "model_id": model_id,
    #                         "centre": model_meta["centre"],
    #                         "system": model_meta["system"],
    #                         "variable": var_code,
    #                         "native_variable": var_meta["cpt"],
    #                         "cpt_variable": var_meta["cpt"],
    #                         "netcdf_variable": var_meta["netcdf"],
    #                         "netcdf_model": model_meta["netcdf_model"],
    #                         "level_type": "single",
    #                         "product": "monthly",
    #                     },
    #                 )
    
    #     # Add WMOLC
    #     if _source_allowed("wmolc"):
    #         for model_id, model_meta in wmolc_models.items():
    #             for var_code, var_meta in wmolc_variables.items():
    #                 if not _variable_allowed(var_code):
    #                     continue
    
    #                 key = f"{model_id}.{var_code}"
    
    #                 _add(
    #                     key,
    #                     {
    #                         "source": "wmolc",
    #                         "model_id": model_id,
    #                         "centre": model_meta["centre"],
    #                         "system": None,
    #                         "variable": var_code,
    #                         "native_variable": var_meta["file_suffix"],
    #                         "file_prefix": model_meta["file_prefix"],
    #                         "file_suffix": var_meta["file_suffix"],
    #                         "standard_name": var_meta["standard_name"],
    #                         "level_type": "single",
    #                         "product": "monthly",
    #                     },
    #                 )
    
    #                 if include_wmolc_aliases:
    #                     alias_key = f"WMOLC_{model_id}.{var_code}"
    #                     _add(
    #                         alias_key,
    #                         {
    #                             "source": "wmolc",
    #                             "model_id": f"WMOLC_{model_id}",
    #                             "centre": model_meta["centre"],
    #                             "system": None,
    #                             "variable": var_code,
    #                             "native_variable": var_meta["file_suffix"],
    #                             "file_prefix": model_meta["file_prefix"],
    #                             "file_suffix": var_meta["file_suffix"],
    #                             "standard_name": var_meta["standard_name"],
    #                             "level_type": "single",
    #                             "product": "monthly",
    #                             "alias_of": key,
    #                         },
    #                     )
    
    #     return dict(sorted(registry.items()))
    

    def ReanalysisName(
        self,
        centre={"ERA5": "reanalysis ERA5", "NOAA": "NOAA ERSST", "ERA5Land": "reanalysis ERA5Land"},
        variables_1={
            "PRCP": "total_precipitation",
            "TEMP": "2m_temperature",
            "UGRD10": "10m_u_component_of_wind",
            "VGRD10": "10m_v_component_of_wind",
            "SST": "sea_surface_temperature",
            "SLP": "mean_sea_level_pressure",
            "DSWR": "surface_solar_radiation_downwards",
            "DLWR": "surface_thermal_radiation_downwards",
            "NOLR": "top_thermal_radiation",
            "RUNOFF": "surface_runoff"
            
        },
        variables_2={
            "HUSS_1000": "specific_humidity",
            "HUSS_925": "specific_humidity",
            "HUSS_850": "specific_humidity",
            "UGRD_1000": "u_component_of_wind",
            "UGRD_925": "u_component_of_wind",
            "UGRD_850": "u_component_of_wind",
            "VGRD_1000": "v_component_of_wind",
            "VGRD_925": "v_component_of_wind",
            "VGRD_850": "v_component_of_wind",
        },
    ):
        """
        Generate a combined dictionary of reanalysis names and variables.

        Parameters:
            centre (dict): Mapping of reanalysis identifiers to reanalysis names.
            variables_1 (dict): Mapping of variable short names to full names for category 1.
            variables_2 (dict): Mapping of variable short names to full names for category 2.

        Returns:
            dict: A combined dictionary with keys as reanalysis.variable combinations and values as tuples (reanalysis name, variable name).
        """
        combined_dict1 = {
            f"{c}.{v}": (centre[c], variables_1[v]) for c in centre for v in variables_1
        }
        combined_dict2 = {
            f"{c}.{v}": (centre[c], variables_2[v]) for c in centre for v in variables_2
        }
        combined_dict = {**combined_dict1, **combined_dict2}
        return combined_dict

    def AgroObsName(
        self,
        variables={
            "AGRO.PRCP": ("precipitation_flux", None),
            "AGRO.TMAX": ("2m_temperature", "24_hour_maximum"),
            "AGRO.TEMP": ("2m_temperature", "24_hour_mean"),
            "AGRO.TMIN": ("2m_temperature", "24_hour_minimum"),
            "AGRO.TMIN": ("2m_temperature", "24_hour_minimum"),
            "AGRO.DSWR": ("solar_radiation_flux", None),
            "AGRO.ETP": ("reference_evapotranspiration", None),
            "AGRO.WFF": ("10m_wind_speed", "24_hour_mean"),
            "AGRO.HUMAX": ("2m_relative_humidity_derived", "24_hour_maximum"),
            "AGRO.HUMIN": ("2m_relative_humidity_derived", "24_hour_minimum"),
        },
    ):
        # 1 W m-2 = 0.0864 MJ m-2 day-1
        """
        Generate a dictionary for agrometeorological observation variables.

        Parameters:
            variables (dict): Mapping of agro variable short names to full names.

        Returns:
            dict: A dictionary mapping agro variables to their corresponding full names.
        """
        return variables
    def download_nmme_txt_with_progress(self, url, file_path, chunk_size=1024):   
        # Check if the URL exists using a HEAD request
        try:
            head = requests.head(url)
            if head.status_code != 200:
                print(f"URL returned status code {head.status_code}. Skipping download.")
                return
        except Exception as e:
            print(f"Error checking URL: {e}. Skipping download.")
            return
    
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(file_path, "wb") as f, tqdm(
            total=total_size, unit="B", unit_scale=True, desc=file_path.name
        ) as progress:
            for data in response.iter_content(chunk_size):
                progress.update(len(data))
                f.write(data)

    def days_in_month(self, year, month):
        a = calendar.monthrange(year, month)[1]
        return a

    def parse_cpt_data_optimized(self, file_path):
        times = []
        times_start = []
        data_list = []
        lons = None
        lats = None
        days_in_month_values = []

        # Read all lines into memory once
        with open(file_path, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('cpt:field'):
                # Parse metadata (e.g., time)
                while i < len(lines) and lines[i].startswith('cpt:'):
                    if 'cpt:T=' in lines[i]:
                        t_str = lines[i].split('cpt:T=')[1].split()[0]
                        t_str = t_str.rstrip(',')
                        year_str, pot_months = t_str.split('-', 1)
                        start_year = int(year_str)
                        if '/' in pot_months:
                            start_str, end_str = pot_months.split("/")
                            start_month = int(start_str)
                            if '-' in end_str:
                                end_year_str, end_month_str = end_str.split('-')
                                end_year = int(end_year_str)
                                end_month = int(end_month_str)
                            else:
                                end_year = start_year
                                end_month = int(end_str)
                            # Generate list of (year, month) pairs
                            months_list = []
                            current_year = start_year
                            current_month = start_month
                            while True:
                                months_list.append((current_year, current_month))
                                if current_year == end_year and current_month == end_month:
                                    break
                                current_month += 1
                                if current_month > 12:
                                    current_month = 1
                                    current_year += 1
                            if len(months_list) != 3:
                                raise ValueError("Expected 3-month season")
                            # Use middle month and its year for time
                            time_year, month = months_list[1]
                            days_in_mon = sum(self.days_in_month(y, m) for y, m in months_list)
                        else:
                            month = int(pot_months)
                            time_year = start_year
                            days_in_mon = self.days_in_month(start_year, month)
                            months_list = [(start_year, month)]

                        days_in_month_values.append(days_in_mon)
                        times.append(datetime.datetime(time_year, month, 1))

                        #### Retrieve init start
                        start_str = lines[i].split('cpt:S=')[1].split()[0]
                        start_str = start_str.rstrip(',')
                        yearstart, monthstart, daystart = start_str.split('-')
                        times_start.append(datetime.datetime(int(yearstart), int(monthstart), 1))
                    i += 1
                # Parse longitudes (assumed to be the next line)
                if i < len(lines):
                    lons = np.array([float(x) for x in lines[i].split()])
                    i += 1
                # Read the next 181 lines as a data block
                if i + 181 <= len(lines):
                    # Join the 181 lines into a single string
                    data_block = '\n'.join(lines[i:i + 181])
                    # Parse the block into a 2D array using np.loadtxt
                    data_array = np.loadtxt(io.StringIO(data_block), dtype=float)
                    if data_array.shape[1] == 361:  # 1 latitude + 360 longitudes
                        # Extract latitudes only once (assuming they’re consistent)
                        if lats is None:
                            lats = data_array[:, 0]
                        # Extract data (excluding latitude column)
                        data = data_array[:, 1:]
                        # Replace missing values (e.g., -999.0) with NaN
                        data[data == -999.0] = np.nan
                        data_list.append(data)
                        i += 181
                    else:
                        raise ValueError("Unexpected number of columns in data block")
                else:
                    break
            else:
                i += 1

        # Stack data into a 3D array (time, latitude, longitude)
        data_3d = np.stack(data_list, axis=0)

        # Create an xarray DataArray for convenient analysis
        da = xr.DataArray(
            data_3d,
            dims=['T', 'Y', 'X'],
            coords={
                'T': times,
                'Y': lats,
                'X': lons
            },
        )
        
        days_in_month_da = xr.DataArray(
            days_in_month_values,
            dims=['T'],
            coords={'T': da['T']}
        )
        return da, days_in_month_da, times_start


    def WAS_Download_Models(
        self,
        dir_to_save,
        center_variable,
        month_of_initialization,
        lead_time,
        year_start_hindcast,
        year_end_hindcast,
        area,
        year_forecast=None,
        ensemble_mean=None,
        force_download=False,
        nmme_source="cpt",  # Options: "ccsr", "netcdf", or "cpt"
        year_init_nmmeNC=2025,
        ccsr_base_url="https://forecast.ccsr.columbia.edu/data/NMME",
        ccsr_cfsv2_sample="pentad_samples",
        ccsr_engine="netcdf4",
    ):
        """
        Download seasonal forecast model data for specified center-variable combinations, initialization month, lead times, and years.

        Parameters:
            dir_to_save (str): Directory to save the downloaded files.
            center_variable (list): List of center-variable identifiers (e.g., ["ECMWF_51.PRCP", "UKMO_602.TEMP"]).
            month_of_initialization (int): Initialization month as an integer (1-12).
            lead_time (list): Zero-based monthly offsets from initialization.
                For April initialization, [3, 4, 5] selects July–September.
            year_start_hindcast (int): Start year for hindcast data.
            year_end_hindcast (int): End year for hindcast data.
            area (list): Bounding box as [North, West, South, East] for clipping.
            year_forecast (int, optional): Forecast year if downloading forecast data. Defaults to None.
            ensemble_mean (str, optional): "mean", "median", or None. None retains
                actual ensemble members. Text "None" is accepted for CLI callers.
                Member files live in ensemble_members/, median files in
                ensemble_median/; mean filenames retain the legacy layout.
            nmme_source (str): "ccsr", "cpt" (legacy default), or "netcdf".
                CCSR downloads monthly members over OPeNDAP, aggregates each
                member over the requested season, then optionally reduces M.
                Its outputs use T (initialization date), M, Y, X, or T, Y, X.
                The two legacy CPC branches supply ensemble MEANS only.
            ccsr_base_url (str): Root of the CCSR NMME OPeNDAP catalogue.
            ccsr_cfsv2_sample (str): "pentad_samples" (default) or
                "early_month_samples". The same sampling must be used for
                hindcast and forecast. Missing archives are not substituted
                with another sampling scheme.
            ccsr_engine (str): "netcdf4" (default) or "pydap". The latter
                requires the optional pydap package. Cached data are independent
                of the transport engine.
            force_download (bool): If True, forces download even if file exists.
        """
        import warnings

        if isinstance(ensemble_mean, str):
            ensemble_mean = ensemble_mean.strip().lower()
            if ensemble_mean == "none":
                ensemble_mean = None
        if ensemble_mean not in (None, "mean", "median"):
            raise ValueError("ensemble_mean must be None, 'mean', or 'median'.")
        nmme_source = str(nmme_source).strip().lower()
        if nmme_source not in ("ccsr", "cpt", "netcdf"):
            raise ValueError("nmme_source must be 'ccsr', 'cpt', or 'netcdf'.")

        member_names = ("number", "member", "members", "M", "realization", "ensemble", "ensmem")
        requested_statistic = "members" if ensemble_mean is None else ensemble_mean
        legacy_directory = Path(dir_to_save)
        if ensemble_mean is None:
            dir_to_save = legacy_directory / "ensemble_members"
        elif ensemble_mean == "median":
            dir_to_save = legacy_directory / "ensemble_median"

        def member_dimension(data):
            # A coordinate by itself is not evidence of member-resolved data.
            dims = {dim for variable in data.data_vars.values() for dim in variable.dims}
            found = [name for name in member_names if name in dims]
            if len(found) > 1:
                raise ValueError(f"Ambiguous ensemble member dimensions: {found}")
            return found[0] if found else None

        def cache_available(path, c3s=False):
            if force_download or not Path(path).is_file():
                return False
            with xr.open_dataset(path) as cached:
                member = member_dimension(cached)
                statistic = cached.attrs.get("wass2s_ensemble_statistic")
                if not c3s and cached.attrs.get("wass2s_nmme_source") == "ccsr" and nmme_source != "ccsr":
                    warnings.warn(f"Ignoring CCSR cache {path} for legacy source {nmme_source!r}.", RuntimeWarning)
                    return False
            compatible = (member is not None) if ensemble_mean is None else (member is None)
            if statistic is not None and statistic != requested_statistic:
                compatible = False
            if not compatible:
                warnings.warn(f"Ignoring incompatible ensemble cache {path}: "
                              f"requested {requested_statistic}.", RuntimeWarning)
                return False
            if c3s:
                _require_c3s_processing_version(path)
            return True

        def reuse_member_archive(output_path, c3s=False):
            # Promote genuine legacy member files, or derive a requested mean /
            # median from a retained member archive. Never fabricate members.
            output_path = Path(output_path)
            raw_target = legacy_directory / "ensemble_members" / output_path.name
            candidates = [raw_target, legacy_directory / output_path.name]
            if force_download:
                return
            for source in dict.fromkeys(candidates):
                if not source.is_file():
                    continue
                with xr.open_dataset(source) as opened:
                    if not c3s and opened.attrs.get("wass2s_nmme_source") == "ccsr" and nmme_source != "ccsr":
                        continue
                    dimension = member_dimension(opened)
                    statistic = opened.attrs.get("wass2s_ensemble_statistic")
                    if dimension is None or statistic in ("mean", "median"):
                        continue
                    if c3s:
                        _require_c3s_processing_version(source)
                    members = opened.load()
                members.attrs["wass2s_ensemble_statistic"] = "members"
                if source != raw_target:
                    _save_netcdf_atomic(members, raw_target)
                if ensemble_mean is not None and not c3s:
                    # Preserve the existing C3S order of monthly aggregation /
                    # reduction. Median and incomplete member-month coverage do
                    # not commute with seasonal aggregation; a separate C3S
                    # request prevents results depending on cache call order.
                    reduced = getattr(members, ensemble_mean)(dim=dimension, keep_attrs=True)
                    reduced.attrs["wass2s_ensemble_statistic"] = ensemble_mean
                    _save_netcdf_atomic(reduced, output_path)
                return

        years = (
            [str(year) for year in range(year_start_hindcast, year_end_hindcast + 1)]
            if year_forecast is None
            else [str(year_forecast)]
        )

        center = [item.split(".")[0] for item in center_variable]
        variables = [item.split(".")[1] for item in center_variable]

        centre = {
            "BOM_2": "bom",
            "ECMWF_51": "ecmwf",
            "UKMO_604": "ukmo", # month of initialization available for forecast are Apr to __
            "UKMO_603": "ukmo", # month of initialization available for forecast are Jan to Mar
            "UKMO_605": "ukmo",
            "UKMO_610": "ukmo",
            "METEOFRANCE_8": "meteo_france",
            "METEOFRANCE_9": "meteo_france", 
            "DWD_21": "dwd", # month of initialization available for forecast are Jan to Mar
            "DWD_22": "dwd", # month of initialization available for forecast are Apr to __ 
            "CMCC_35": "cmcc",
            "CMCC_4": "cmcc",
            "NCEP_2": "ncep",
            "JMA_3": "jma",
            "JMA_4": "jma",
            "ECCC_4": "eccc",
            "ECCC_5": "eccc",
            # "CFSV2": "cfsv2",
            # "CMC1": "cmc1",
            # "CMC2": "cmc2",
            # "GFDL": "gfdl",
            # "NASA": "nasa",
            # "NCAR_CCSM4": "ncar_ccsm4",
            # "NMME" : "nmme"
            "CFSV2_1": "cfsv2",
            "CMC1_1": "cmc1",
            "CMC2_1": "cmc2",
            "GFDL_1": "gfdl",
            "NASA_1": "nasa",
            "NCAR_CCSM4_1": "ncar_ccsm4",
            "NCAR_CESM1_1": "ncar_cesm1",
            "NMME_1" : "nmme"
        }
        
        # Mapping for NMME NetCDF filenames/directories on FTP
        nmme_netcdf_model_map = {
            "CFSV2_1": "CFSv2", "CMC1_1": "CanESM5", "CMC2_1": "GEM5.2_NEMO", 
            "GFDL_1": "GFDL-SPEAR", 
            "NASA_1": "NASA_GEOS5v2", "NCAR_CCSM4_1": "NCAR_CCSM4",
            "NCAR_CESM1_1": "NCAR_CESM1",
            "NMME_1" : "NMME"
        }

        variables_1 = {
            "PRCP": "total_precipitation",
            "TEMP": "2m_temperature",
            "TMAX": "maximum_2m_temperature_in_the_last_24_hours",
            "TMIN": "minimum_2m_temperature_in_the_last_24_hours",
            "UGRD10": "10m_u_component_of_wind",
            "VGRD10": "10m_v_component_of_wind",
            "SST": "sea_surface_temperature",
            "SLP": "mean_sea_level_pressure",
            "DSWR": "surface_solar_radiation_downwards",
            "DLWR": "surface_thermal_radiation_downwards",
            "NOLR": "top_thermal_radiation",
            "RUNOFF":"mean_surface_runoff_rate"
        }

        variables_2 = {
            "HUSS_1000": "specific_humidity",
            "HUSS_925": "specific_humidity",
            "HUSS_850": "specific_humidity",
            "UGRD_1000": "u_component_of_wind",
            "UGRD_925": "u_component_of_wind",
            "UGRD_850": "u_component_of_wind",
            "UGRD_700": "u_component_of_wind",
            "UGRD_200": "u_component_of_wind",
            "VGRD_1000": "v_component_of_wind",
            "VGRD_925": "v_component_of_wind",
            "VGRD_850": "v_component_of_wind",
            "VGRD_700": "v_component_of_wind",
            "VGRD_200": "v_component_of_wind",
        }

        system = {
            "BOM_2": "2",
            "ECMWF_51": "51",
            "UKMO_604": "604",
            "UKMO_603": "603",
            "UKMO_605": "605",
            "UKMO_610": "610",
            "METEOFRANCE_8": "8",
            "METEOFRANCE_9": "9",
            "DWD_21": "21",
            "DWD_22": "22",
            # "DWD_2": "2",
            "CMCC_35": "35",
            "CMCC_4": "4",
            "NCEP_2": "2",
            "JMA_3": "3",
            "JMA_4": "4",
            "ECCC_4": "4",
            "ECCC_5": "5",
            # "CFSV2": "1",
            # "CMC1": "1",
            # "CMC2": "1",
            # "GFDL": "1",
            # "NASA": "1",
            # "NCAR_CCSM4": "1",
            # "NMME" : "1"
            "CFSV2_1": "1",
            "CMC1_1": "1",
            "CMC2_1": "1",
            "GFDL_1": "1",
            "NASA_1": "1",
            "NCAR_CCSM4_1": "1",
            "NCAR_CESM1_1": "1",
            "NMME_1" : "1"

        }
        
        nmme = ["cfsv2", "cmc1", "cmc2", "gfdl",  "nasa", "ncar_ccsm4", "ncar_cesm1", "nmme"]
        
        selected_centre = [centre[k] for k in center]
        selected_system = [system[k] for k in center]
        selected_var = [k for k in variables]

        dir_to_save = Path(dir_to_save)
        os.makedirs(dir_to_save, exist_ok=True)
        
        abb_mont_ini = calendar.month_abbr[int(month_of_initialization)]
        season_months = [((int(month_of_initialization) + int(l) - 1) % 12) + 1 for l in lead_time]
        season = "".join([calendar.month_abbr[month] for month in season_months])
        
        
        store_file_path = {}
        for cent, syst, k in zip(selected_centre, selected_system, selected_var):
            file_prefix = "forecast" if year_forecast else "hindcast"
            file_path = f"{dir_to_save}/{file_prefix}_{cent.replace('_', '')}{syst}_{k}_{abb_mont_ini}Ic_{season}_{lead_time[0]}.nc"
            if cent in nmme and nmme_source == "ccsr":
                if ccsr_engine not in ("netcdf4", "pydap"):
                    raise ValueError("ccsr_engine must be 'netcdf4' or 'pydap'.")
                model_id = f"{cent.upper()}_{syst}"
                request_key = _ccsr_request_key(
                    model_id, k, years, month_of_initialization, lead_time,
                    area, ccsr_base_url, ccsr_cfsv2_sample,
                )
                if not force_download and _ccsr_cache_matches(
                    file_path, request_key, requested_statistic
                ):
                    print(f"{file_path} already exists with matching CCSR provenance. Skipping download.")
                    store_file_path[f"{cent}_{syst}"] = file_path
                    continue
                raw_target = legacy_directory / "ensemble_members" / Path(file_path).name
                raw = None
                if not force_download and _ccsr_cache_matches(raw_target, request_key, "members"):
                    with xr.open_dataset(raw_target) as opened:
                        raw = opened.load()
                if raw is None:
                    try:
                        raw = _ccsr_download_season(
                            model_id, k, years, month_of_initialization,
                            lead_time, area, None, base_url=ccsr_base_url,
                            cfsv2_sample=ccsr_cfsv2_sample,
                            engine=ccsr_engine,
                        )
                    except _CCSRUnavailableError as exc:
                        warnings.warn(f"CCSR {model_id}.{k} unavailable: {exc}", RuntimeWarning)
                        continue
                    raw.attrs["wass2s_ccsr_request"] = request_key
                    raw.attrs["wass2s_processing_version"] = "ccsr-members-calendar-20260916"
                    raw.attrs["wass2s_ensemble_statistic"] = "members"
                    _save_netcdf_atomic(raw, raw_target)
                if ensemble_mean is not None:
                    result = getattr(raw, ensemble_mean)(dim="M", skipna=True, keep_attrs=True)
                    result.attrs["wass2s_ensemble_statistic"] = ensemble_mean
                    _save_netcdf_atomic(result, file_path)
                store_file_path[f"{cent}_{syst}"] = file_path
                print(f"Saved CCSR {model_id}.{k} ({requested_statistic}): {file_path}")
                continue
            if not cache_available(file_path, c3s=cent not in nmme):
                reuse_member_archive(file_path, c3s=cent not in nmme)

            if cent in nmme and ensemble_mean != "mean":
                if cache_available(file_path):
                    store_file_path[f"{cent}_{syst}"] = file_path
                else:
                    warnings.warn(
                        f"NMME {cent}/{syst}/{k} unavailable for ensemble_mean={ensemble_mean!r}: "
                        f"nmme_source={nmme_source!r} supplies ensemble means only. "
                        "Provide a genuine member-resolved seasonal archive in "
                        f"{legacy_directory / 'ensemble_members' / Path(file_path).name}; "
                        "spread and quantiles cannot be recovered from ENSMEAN/CPT means.",
                        RuntimeWarning,
                    )
                continue

            if cent in nmme:

                if nmme_source == "netcdf":
                    nmme_nc_vars = {"PRCP": "prate", "TEMP": "tmp2m", "SST": "tmpsfc"}
                    if k not in nmme_nc_vars:
                        print(f"Skipping {k}: Not currently mapped for NetCDF FTP.")
                        continue
    
                    ftp_var = nmme_nc_vars[k]
                    ftp_model = nmme_netcdf_model_map[f"{cent.upper()}_{syst}"]
                    file_path = f"{dir_to_save}/{file_prefix}_{cent.replace('_', '')}{syst}_{k}_{abb_mont_ini}Ic_{season}_{lead_time[0]}.nc"
                    
                    if cache_available(file_path):
                        print(f"{file_path} already exists. Skipping download.")
                        store_file_path[f"{cent}_{syst}"] = file_path                   
                    else:
                        try:
                            list_ds_years = []
                            # ic_folder = f"{abb_mont_ini.lower()}2025ic" 
    
                            for yr in years:
                                # Files always use 'fcst' on the CPC FTP server regardless of year
                                tag = "fcst" 
                                date_str = f"{yr}{int(month_of_initialization):02d}"
                                fname = f"{ftp_model}.{ftp_var}.{date_str}.ENSMEAN.{tag}.nc"
                                ic_folder = f"{abb_mont_ini.lower()}{year_init_nmmeNC}ic"
                                url = f"https://ftp.cpc.ncep.noaa.gov/International/nmme/netcdf/{ic_folder}/{ftp_model}/{fname}"
                                
                                temp_file = dir_to_save / f"tmp_{fname}"
                                
                                # print(f"Downloading from NMME FTP: {url}")
                                urllib.request.urlretrieve(url, temp_file)
                                
                                # Use decode_times=False to avoid "months since" decoding errors
                                ds_y = xr.open_dataset(temp_file, decode_times=False)
                                
                                # Standardize coords
                                if "lon" in ds_y.coords: ds_y = ds_y.rename({"lon":"X", "lat":"Y"})
                                ds_y = ds_y.assign_coords(X=((ds_y.X + 180) % 360 - 180)).sortby("X").sortby("Y")
                                ds_y = ds_y.sel(X=slice(area[1], area[3]), Y=slice(area[2], area[0]))
                                
                                
                                # Select Lead Time (Target) based on index
                                lead_idx = [int(l) for l in lead_time]
                                ds_y = ds_y.isel(target=lead_idx)
    
                                # Unit Conversions
                                if k == "PRCP":
                                    # mm/s to total mm for the month
                                    start_dt = pd.to_datetime(f"{yr}-{int(month_of_initialization):02d}-01")
                                    days = [(start_dt + pd.DateOffset(months=int(l))).days_in_month for l in lead_time]
                                    # Create alignment DataArray using the undecoded numeric target coordinates
                                    days_da = xr.DataArray(days, coords={"target": ds_y.target}, dims="target")
                                    ds_y = ds_y * 86400 * days_da
                                    ds_y = ds_y.sum(dim="target")
                                    ds_y = ds_y.drop_vars("initial_time")#.to_array().drop_vars("variable")
                                elif k in ["TEMP", "SST"]:
                                    ds_y = ds_y - 273.15 # Kelvin to Celsius
                                    ds_y = ds_y.mean(dim="target")
                                    ds_y = ds_y.drop_vars("initial_time")
                                else:
                                    ds_y = ds_y.mean(dim="target")
                                    ds_y = ds_y.drop_vars("initial_time")
    
                                # Assign the year coordinate T
                                ds_y = ds_y.expand_dims("T").assign_coords(T=[f"{yr}-{int(month_of_initialization):02d}-01"])#.assign_coords(T=[str(yr)])
                                # Rename 'fcst' data variable to k
                                ds_y = ds_y.rename({tag: k})
                                list_ds_years.append(ds_y)
                                
                                ds_y.close()
                                if os.path.exists(temp_file): os.remove(temp_file)
    
                            if list_ds_years:
                                ds_final = xr.concat(list_ds_years, dim="T").sortby("T").transpose('T', 'Y', 'X')
                                ds_final.attrs["wass2s_ensemble_statistic"] = "mean"
                                ds_final.to_netcdf(file_path)
                                store_file_path[f"{cent}_{syst}"] = file_path
                                print(f"Saved: {file_path}")
    
                        except Exception as e:
                            print(f"NMME NetCDF Error for {cent}: {e}")
                else:
                ## to tab
                    file_path = f"{dir_to_save}/{file_prefix}_{cent.replace('_', '')}{syst}_{k}_{abb_mont_ini}Ic_{season}_{lead_time[0]}.nc"
                    init_str = f"{abb_mont_ini}ic"
                    tag = "fcst" if year_forecast else "hcst"
                    k = "precip" if k=="PRCP" else k
                    k = "tmp2m" if k=="TEMP" else k
                    k = "sst" if k=="SST" else k
                    if cache_available(file_path):
                        print(f"{file_path} already exists. Skipping download.")
                        store_file_path[f"{cent}_{syst}"] = file_path                   
                    else:
                        try:
                            # Choose base URL depending on forecast/hindcast and temporal resolution.
                            if len(lead_time) == 3:
                                # Build lead time string using min and max lead time values.
                                lead_str = f"{season_months[0]}-{season_months[-1]}"
                                crosses_year = season_months[0] > season_months[-1]
                                
                                if year_forecast:
                                    base_url = "https://ftp.cpc.ncep.noaa.gov/International/nmme/seasonal_nmme_forecast_in_cpt_format/"
                                    if crosses_year:
                                        year_range = f"{year_forecast}-{year_forecast + 1}"
                                    elif int(month_of_initialization) > season_months[0]:
                                        year_range = f"{year_forecast + 1}-{year_forecast + 1}"
                                    else:
                                        year_range = f"{year_forecast}-{year_forecast}"
                                        
                                    # year_range = f"{year_forecast}-{year_forecast + 1}" if crosses_year else f"{year_forecast}-{year_forecast}"
                                    file_name = f"{cent}_{k}_{tag}_{init_str}_{lead_str}_{year_range}.txt"
                                    full_url = base_url + file_name
                                    file_txt_path = dir_to_save / file_name
                                    try:
                                        if os.path.exists(file_txt_path):
                                            da, number_day, times_start = self.parse_cpt_data_optimized(file_txt_path)
                                        else:
                                            self.download_nmme_txt_with_progress(full_url, file_txt_path)
                                            da, number_day, times_start = self.parse_cpt_data_optimized(file_txt_path)
                                    except:
                                        print(f"failed to download {file_name}")
                                else:
                                    base_url = "https://ftp.cpc.ncep.noaa.gov/International/nmme/seasonal_nmme_hindcast_in_cpt_format/"
    
                                    # CPC archive: seasons that cross year (NDJ/DJF) start in 1991; others in 1992
                                    hind_start = 1991 if crosses_year else 1992
                                    hind_end   = 2021
                                    year_range1 = f"{hind_start}-{hind_end}"
                                    year_range2 = f"{1991}-{2020}"
            
                                    file_name_1 = f"{cent}_{k}_{tag}_{init_str}_{lead_str}_{year_range1}.txt"
                                    file_name_2 = f"{cent}_{k}_{tag}_{init_str}_{lead_str}_{year_range2}.txt"
                                    full_url1 = base_url + file_name_1
                                    full_url2 = base_url + file_name_2
                                    # print(full_url2)
                                    file_txt_path_1 = dir_to_save / file_name_1
                                    file_txt_path_2 = dir_to_save / file_name_2
                                    try:
                                        if os.path.exists(file_txt_path_1):
                                            da, number_day, times_start = self.parse_cpt_data_optimized(file_txt_path_1)
                                        else:
                                            self.download_nmme_txt_with_progress(full_url1, file_txt_path_1)
                                            da, number_day, times_start = self.parse_cpt_data_optimized(file_txt_path_1)
                                    except:
                                        print(f"failed to download {file_name_1}")
    
                                    try:
                                        
                                        if os.path.exists(file_txt_path_2):
                                            da, number_day, times_start = self.parse_cpt_data_optimized(file_txt_path_2)
                                        else:
                                            self.download_nmme_txt_with_progress(full_url2, file_txt_path_2)
                                            da, number_day, times_start = self.parse_cpt_data_optimized(file_txt_path_2)
                                    except:
                                        print(f"failed to download {file_name_2}")

        
                                if k == "precip":
                                    da = da * number_day
                                da = da.assign_coords(T=times_start)
                                if year_forecast:
                                    da = da.sel(T=str(year_forecast))
                                else:
                                    da = da.sel(T=slice(str(year_start_hindcast),str(year_end_hindcast)))
                                ds = da.to_dataset(name=k)
                                ds = ds.isel(Y=slice(None, None, -1))
                                ds = ds.assign_coords(X=((ds.X + 180) % 360 - 180))
                                ds = ds.sortby("X")
                                ds = ds.sel(X=slice(area[1],area[3]),Y=slice(area[2], area[0])).transpose('T', 'Y', 'X') 
    
                                ds.attrs["wass2s_ensemble_statistic"] = "mean"
                                ds.to_netcdf(file_path)
                                print(f"Download finished for {cent} {syst} {k} to {file_path}")
                                ds.close()
                                store_file_path[f"{cent}_{syst}"] = file_path
                                
                            else:
                                if year_forecast:
                                    base_url = "https://ftp.cpc.ncep.noaa.gov/International/nmme/monthly_nmme_forecast_in_cpt_format/"
                                    if int(month_of_initialization) > season_months[0]:
                                        year_range = f"{year_forecast + 1}"
                                    else:
                                        year_range = f"{year_forecast}"
                                else:
                                    base_url = "https://ftp.cpc.ncep.noaa.gov/International/nmme/monthly_nmme_hindcast_in_cpt_format/"
                                    year_range = f"{1992}"
                                all_da = []
                                for i in season_months:
                                    file_name = f"{cent}_{k}_{tag}_{init_str}_{i}_{year_range}.txt"
                                    full_url = base_url + file_name
                                    print(full_url)
                                    file_txt_path = dir_to_save / file_name
                                    
                                    if os.path.exists(file_txt_path):
                                        da_, number_day, times_start = self.parse_cpt_data_optimized(file_txt_path)
                                    else:
                                        self.download_nmme_txt_with_progress(full_url, file_txt_path)
                                        da_, number_day, times_start = self.parse_cpt_data_optimized(file_txt_path)
                                
                                    if k == "precip":
                                        da_ = da_ * number_day
                                                                
                                    
                                    all_da.append(da_)
                                da = xr.concat(all_da, dim="T").sortby("T")
                               
                                if k == "precip":
                                    da = da.resample(T="YE").sum()
                                else:
                                    da = da.resample(T="YE").mean()
                                da = da.assign_coords(T=times_start)    
           
                                if year_forecast:
                                    da = da.sel(T=str(year_forecast))
                                else:
                                    da = da.sel(T=slice(str(year_start_hindcast),str(year_end_hindcast)))
                                ds = da.to_dataset(name=k)
                                ds = ds.isel(Y=slice(None, None, -1))
                                ds = ds.assign_coords(X=((ds.X + 180) % 360 - 180))
                                ds = ds.sortby("X")
                                ds = ds.sel(X=slice(area[1],area[3]),Y=slice(area[2], area[0])).transpose('T', 'Y', 'X') 
                                ds.attrs["wass2s_ensemble_statistic"] = "mean"
                                ds.to_netcdf(file_path)
                                print(f"Download finished for {cent} {syst} {k} to {file_path}")
                                ds.close()
                                store_file_path[f"{cent}_{syst}"] = file_path  
                        except:
                            pass
            else:
                file_path = f"{dir_to_save}/{file_prefix}_{cent.replace('_', '')}{syst}_{k}_{abb_mont_ini}Ic_{season}_{lead_time[0]}.nc"
                if cache_available(file_path, c3s=True):
                    print(f"{file_path} already exists. Skipping download.")
                    store_file_path[f"{cent}_{syst}"] = file_path
                else:                
                    try:
                        from wass2s.was_scientific_utils import c3s_leads_from_offsets
                        cds_leads = c3s_leads_from_offsets(lead_time)
                        if k in variables_2:
                            press_level = k.split("_")[1]
                            dataset = "seasonal-monthly-pressure-levels"
                            request = {
                                "originating_centre": cent,
                                "system": syst,
                                "variable": variables_2[k],
                                "pressure_level": press_level,
                                "product_type": ["monthly_mean"],
                                "year": years,
                                "month": month_of_initialization,
                                "leadtime_month": cds_leads,
                                "data_format": "netcdf",
                                "area": area,
                            }
                        else:
                            dataset = "seasonal-monthly-single-levels"
                            request = {
                                "originating_centre": cent,
                                "system": syst,
                                "variable": variables_1[k],
                                "product_type": ["monthly_mean"],
                                "year": years,
                                "month": month_of_initialization,
                                "leadtime_month": cds_leads,
                                "data_format": "netcdf",
                                "area": area,
                            }
        
                        client = cdsapi.Client()
                        import tempfile
                        fd, raw_path = tempfile.mkstemp(prefix="c3s_", suffix=".raw.nc", dir=dir_to_save)
                        os.close(fd)
                        client.retrieve(dataset, request).download(raw_path)
                        print(f"Downloaded: {file_path}")
    
        
                        # Load the NetCDF file and apply area selection if specified
                        with xr.open_dataset(raw_path) as opened:
                            ds = opened.load()
                        os.unlink(raw_path)
            
                        if k in ["TMIN","TEMP","TMAX","SST"]:
                            ds = ds - 273.15
                            ds = getattr(ds,ensemble_mean)(dim="number") if ensemble_mean != None else ds 
                            ds = _c3s_monthly_mean(ds).isel(latitude=slice(None, None, -1))
                            if "indexing_time" in ds.coords: 
                                ds = ds.rename({"latitude":"lat","longitude":"lon","indexing_time":"time"})
                            else:
                                ds = ds.rename({"latitude":"lat","longitude":"lon","forecast_reference_time":"time"})
                        if k =="PRCP":
                            from wass2s.was_scientific_utils import c3s_monthly_precipitation_total
                            ds = c3s_monthly_precipitation_total(ds)
                            ds = getattr(ds,ensemble_mean)(dim="number", keep_attrs=True) if ensemble_mean is not None else ds
                            ds = ds.isel(latitude=slice(None, None, -1))
                            ds = ds.clip(min=0)  # Keep missing seasonal totals missing.
                            if "indexing_time" in ds.coords: 
                                ds = ds.rename({"latitude":"lat","longitude":"lon","indexing_time":"time"})
                            else:
                                ds = ds.rename({"latitude":"lat","longitude":"lon","forecast_reference_time":"time"})

                        if k == "RUNOFF":
                            from wass2s.was_scientific_utils import c3s_monthly_precipitation_total
                            ds = c3s_monthly_precipitation_total(ds)
                            if ensemble_mean is not None:
                                ds = getattr(ds, ensemble_mean)(dim="number", keep_attrs=True)
                            ds = ds.isel(latitude=slice(None, None, -1)).clip(min=0)
                            time_name = "indexing_time" if "indexing_time" in ds.coords else "forecast_reference_time"
                            ds = ds.rename({"latitude":"lat", "longitude":"lon", time_name:"time"})
                            ds.attrs["quantity"] = "seasonal local runoff depth; not routed river discharge"

                        if k == "SLP":
                            ds = ds/100
                            ds = getattr(ds,ensemble_mean)(dim="number") if ensemble_mean != None else ds 
                            ds = _c3s_monthly_mean(ds).isel(latitude=slice(None, None, -1))
                            if "indexing_time" in ds.coords: 
                                ds = ds.rename({"latitude":"lat","longitude":"lon","indexing_time":"time"})
                            else:
                                ds = ds.rename({"latitude":"lat","longitude":"lon","forecast_reference_time":"time"})

                        if k in ["UGRD10","VGRD10"]:
                            ds = getattr(ds,ensemble_mean)(dim="number") if ensemble_mean != None else ds
                            ds = _c3s_monthly_mean(ds).isel(latitude=slice(None, None, -1))
                            if "indexing_time" in ds.coords: 
                                ds = ds.rename({"latitude":"lat","longitude":"lon","indexing_time":"time"})
                            else:
                                ds = ds.rename({"latitude":"lat","longitude":"lon","forecast_reference_time":"time"})

                        if k in ["DSWR","DLWR", "NOLR"]:
                            ds = getattr(ds,ensemble_mean)(dim="number") if ensemble_mean != None else ds
                            ds = _c3s_monthly_mean(ds).isel(latitude=slice(None, None, -1))
                            if "indexing_time" in ds.coords: 
                                ds = ds.rename({"latitude":"lat","longitude":"lon","indexing_time":"time"})
                            else:
                                ds = ds.rename({"latitude":"lat","longitude":"lon","forecast_reference_time":"time"})

                        if k not in ["TMIN","TEMP","TMAX","SST","UGRD10","VGRD10", "PRCP","RUNOFF","SLP","DSWR","DLWR","NOLR"]:
                            ds = getattr(ds,ensemble_mean)(dim="number") if ensemble_mean != None else ds
                            ds = _c3s_monthly_mean(ds)
                            if "pressure_level" in ds.dims:
                                ds = ds.squeeze("pressure_level", drop=True)
                            ds = ds.isel(latitude=slice(None, None, -1))
                            if "indexing_time" in ds.coords: 
                                ds = ds.rename({"latitude":"lat","longitude":"lon","indexing_time":"time"})
                            else:
                                ds = ds.rename({"latitude":"lat","longitude":"lon","forecast_reference_time":"time"})
            

                            
                        ds = ds.rename({"lon":"X","lat":"Y","time":"T"})    
                        output_path = f"{dir_to_save}/{file_prefix}_{cent.replace('_', '')}{syst}_{k}_{abb_mont_ini}Ic_{season}_{lead_time[0]}.nc"
                        
                        # Save the combined dataset for the center-variable combination
                        if ensemble_mean is None and member_dimension(ds) is None:
                            raise ValueError("C3S returned no member dimension for a full-ensemble request.")
                        ds.attrs["wass2s_processing_version"] = "calendar-rates-20260911"
                        ds.attrs["wass2s_ensemble_statistic"] = requested_statistic
                        _save_netcdf_atomic(ds, output_path)
                        print(f"Download finished, combined dataset for {cent} {syst} {k} to {output_path}")
                        ds.close()
                        store_file_path[f"{cent}_{syst}"] = file_path
                    # except Exception as e:
                    #     if "raw_path" in locals() and os.path.exists(raw_path):
                    #         os.unlink(raw_path)
                    #     raise RuntimeError(f"C3S processing failed for {cent}/{syst}/{k}") from e
                    except Exception as e:
                        print(f"C3S processing failed for {cent}/{syst}/{k}")
        return store_file_path




    def WAS_Download_AgroIndicators_daily(
            self,
            dir_to_save,
            variables,
            year_start,
            year_end,
            area,
            force_download=False,
            max_retries=3,
            retry_delay=5,
        ):
        """
        Download daily agro-meteorological indicators from the Copernicus Data Store (CDS)
        for specified variables and years, with retries for failed downloads.

        Parameters
        ----------
        dir_to_save : str or pathlib.Path
            Directory path where the downloaded NetCDF files will be saved.
            The directory will be created if it does not exist.
        variables : list of str
            List of variable shorthand names to download. Valid options are:
            ["AGRO.PRCP", "AGRO.TMAX", "AGRO.TEMP", "AGRO.TMIN", "AGRO.DSWR",
            "AGRO.ETP", "AGRO.WFF", "AGRO.HUMAX", "AGRO.HUMIN"].
            Each variable corresponds to a CDS variable and optional statistic
            (e.g., "AGRO.PRCP" maps to "precipitation_flux").
        year_start : int
            Start year for the data to download (inclusive).
        year_end : int
            End year for the data to download (inclusive).
        area : list of float
            Bounding box for spatial subsetting in the format [North, West, South, East].
            Example: [50, -10, 40, 10] for a region in Europe.
        force_download : bool, optional
            If True, forces download even if the output file exists. Default is False.
        max_retries : int, optional
            Maximum number of retry attempts for failed downloads. Default is 3.
        retry_delay : int, optional
            Seconds to wait between retry attempts. Default is 5.

        Returns
        -------
        None
            The function saves NetCDF files to `dir_to_save` but does not return a value.
            Output files are named as `Daily_<variable>_<year_start>_<year_end>.nc`.

        Notes
        -----
        - The function downloads data from the CDS dataset "sis-agrometeorological-indicators".
        - Data is downloaded year-by-year as ZIP files containing NetCDF files, which are
          extracted, concatenated, and saved as a single NetCDF file per variable.
        - Temperature variables ("AGRO.TMIN", "AGRO.TEMP", "AGRO.TMAX") are converted
          from Kelvin to Celsius.
        - Solar radiation ("AGRO.DSWR") is converted from J/m^2/day to W/m^2.
        - Coordinates are renamed to "X" (longitude), "Y" (latitude), and "T" (time),
          with latitude flipped to ascending order.
        - The function requires a valid CDS API key configured in `~/.cdsapirc`.
        - Downloads are skipped for a variable if any year's data fails to download
          after `max_retries` attempts to ensure data completeness.
        """
        dir_to_save = Path(dir_to_save)
        dir_to_save.mkdir(parents=True, exist_ok=True)
        days = [f"{day:02}" for day in range(1, 32)]
        months = [f"{month:02}" for month in range(1, 13)]
        version = "2_0"

        variable_mapping = {
            "AGRO.PRCP": ("precipitation_flux", None),
            "AGRO.TMAX": ("2m_temperature", "24_hour_maximum"),
            "AGRO.TEMP": ("2m_temperature", "24_hour_mean"),
            "AGRO.TMIN": ("2m_temperature", "24_hour_minimum"),
            "AGRO.DSWR": ("solar_radiation_flux", None),
            "AGRO.ETP": ("reference_evapotranspiration", None),
            "AGRO.WFF": ("10m_wind_speed", "24_hour_mean"),
            "AGRO.HUMAX": ("2m_relative_humidity_derived", "24_hour_maximum"),
            "AGRO.HUMIN": ("2m_relative_humidity_derived", "24_hour_minimum")
        }

        for var in variables:
            if var not in variable_mapping:
                print(f"Unknown variable: {var}. Skipping.")
                continue

            cds_variable, statistic = variable_mapping[var]
            output_path = dir_to_save / f"Daily_{var.split('.')[1]}_{year_start}_{year_end}.nc"

            if not force_download and output_path.exists():
                print(f"{output_path} already exists. Skipping download.")
                continue

            combined_datasets = []
            all_years_downloaded = True

            for year in range(year_start, year_end + 1):
                zip_file_path = dir_to_save / f"Daily_{var.split('.')[1]}_{year}.zip"
                success = False
                retries = 0

                while retries < max_retries and not success:
                    try:
                        client = cdsapi.Client()
                        dataset = "sis-agrometeorological-indicators"
                        request = {
                            "variable": cds_variable,
                            "year": str(year),
                            "month": months,
                            "day": days,
                            "version": version,
                            "area": area,
                        }
                        if statistic:
                            request["statistic"] = [statistic]

                        print(f"Attempt {retries + 1}/{max_retries}: Downloading {cds_variable} ({statistic}) data for {year}...")
                        client.retrieve(dataset, request).download(str(zip_file_path))
                        print(f"Downloaded: {zip_file_path}")
                        success = True

                    except Exception as e:
                        retries += 1
                        print(f"Attempt {retries}/{max_retries} failed for {cds_variable} ({statistic}) data for {year}: {e}")
                        if retries < max_retries:
                            print(f"Retrying after {retry_delay} seconds...")
                            _time.sleep(retry_delay)
                        if zip_file_path.exists():
                            os.remove(zip_file_path)
                            print(f"Deleted incomplete ZIP file: {zip_file_path}")

                if not success:
                    print(f"Failed to download {cds_variable} ({statistic}) data for {year} after {max_retries} attempts.")
                    all_years_downloaded = False
                    continue

                try:
                    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                        for netcdf_file_name in zip_ref.namelist():
                            with zip_ref.open(netcdf_file_name) as file:
                                ds = xr.open_dataset(io.BytesIO(file.read()))
                                combined_datasets.append(ds)
                except Exception as e:
                    print(f"Failed to extract/process {zip_file_path}: {e}")
                    all_years_downloaded = False
                    if zip_file_path.exists():
                        os.remove(zip_file_path)
                        print(f"Deleted ZIP file due to processing error: {zip_file_path}")
                    continue

                if zip_file_path.exists():
                    os.remove(zip_file_path)
                    print(f"Deleted ZIP file: {zip_file_path}")

            if combined_datasets and all_years_downloaded:
                try:
                    combined_ds = xr.concat(combined_datasets, dim="time").drop_vars('crs')

                    if var in ["AGRO.TMIN", "AGRO.TEMP", "AGRO.TMAX"]:
                        combined_ds = combined_ds - 273.15
                    if var == "AGRO.DSWR":
                        combined_ds = combined_ds / 86400

                    combined_ds = combined_ds.rename({"lon": "X", "lat": "Y", "time": "T"})
                    combined_ds = combined_ds.isel(Y=slice(None, None, -1))

                    combined_ds.to_netcdf(output_path)
                    combined_ds.close()
                    print(f"Combined dataset for {var} saved to {output_path}")
                except Exception as e:
                    print(f"Failed to process or save combined dataset for {var}: {e}")
            else:
                print(f"Skipping save for {var} due to incomplete year downloads.")


    def WAS_Download_Models_Daily(
            self,
            dir_to_save,
            center_variable,
            month_of_initialization,
            day_of_initialization,
            leadtime_hour,
            year_start_hindcast,
            year_end_hindcast,
            area,
            year_forecast=None,
            ensemble_mean=None,
            force_download=False,
            data_format="netcdf",
            output_layout="valid_time",
            runoff_units="m3/s",
        ):
            """
            Download daily seasonal hindcast/forecast data from CDS seasonal-original
            datasets.
    
            Output conventions:
                output_layout="valid_time":
                    T, Y, X when ensemble_mean is "mean" or "median"
                    T, M, Y, X when members are kept
    
                output_layout="init_leadtime":
                    member, T, leadtime, level, Y, X
    
            Notes:
                - In "valid_time" layout, T is the valid forecast date.
                - In "init_leadtime" layout, T is the initialization date/year.
                - Accumulated variables are deaccumulated along leadtime.
                - RUNOFF can be saved as "mm" or "m3/s".
                - Works for both data_format="netcdf" and data_format="grib".
                  CDS NetCDF exposes the initialization time as
                  "forecast_reference_time"/"indexing_time"; cfgrib on GRIB1
                  (DWD, etc.) exposes it as a scalar coord named "time". Both are
                  normalized to a proper "T" dimension here.
            """
            from calendar import month_abbr
            from pathlib import Path
            import gc
            import os
            import time as _time
    
            import cdsapi
            import numpy as np
            import xarray as xr
    
            if isinstance(center_variable, str):
                raise TypeError(
                    "center_variable must be a list, for example "
                    "['ECMWF_51.RUNOFF'] or ['ECMWF_51.RUNOFF', 'UKMO_604.RUNOFF']."
                )
            center_variable = list(center_variable)
    
            if isinstance(leadtime_hour, (str, int, np.integer)):
                leadtime_hour = [str(leadtime_hour)]
            else:
                leadtime_hour = [str(lt) for lt in leadtime_hour]
    
            if ensemble_mean not in [None, "mean", "median"]:
                raise ValueError("ensemble_mean must be None, 'mean', or 'median'.")
    
            if output_layout not in ["valid_time", "init_leadtime"]:
                raise ValueError("output_layout must be 'valid_time' or 'init_leadtime'.")
    
            if runoff_units not in ["mm", "m3/s"]:
                raise ValueError("runoff_units must be 'mm' or 'm3/s'.")
    
            if year_forecast is None:
                years = [str(y) for y in range(year_start_hindcast, year_end_hindcast + 1)]
                file_prefix = "hindcast"
            else:
                years = [str(year_forecast)]
                file_prefix = "forecast"
    
            centre = {
                "BOM_2": "bom",
                "ECMWF_51": "ecmwf",
                "UKMO_603": "ukmo",
                "UKMO_604": "ukmo",
                "UKMO_605": "ukmo",
                "UKMO_610": "ukmo",
                "METEOFRANCE_8": "meteo_france",
                "METEOFRANCE_9": "meteo_france",
                "DWD_21": "dwd",
                "DWD_22": "dwd",
                "CMCC_35": "cmcc",
                "CMCC_4": "cmcc",
                "NCEP_2": "ncep",
                "JMA_3": "jma",
                "JMA_4": "jma",
                "ECCC_4": "eccc",
                "ECCC_5": "eccc",
            }
    
            system = {
                "BOM_2": "2",
                "ECMWF_51": "51",
                "UKMO_603": "603",
                "UKMO_604": "604",
                "UKMO_605": "605",
                "UKMO_610": "610",
                "METEOFRANCE_8": "8",
                "METEOFRANCE_9": "9",
                "DWD_21": "21",
                "DWD_22": "22",
                "CMCC_35": "35",
                "CMCC_4": "4",
                "NCEP_2": "2",
                "JMA_3": "3",
                "JMA_4": "4",
                "ECCC_4": "4",
                "ECCC_5": "5",
            }
    
            variables_1 = {
                "PRCP": "total_precipitation",
                "TEMP": "2m_temperature",
                "TDEW": "2m_dewpoint_temperature",
                "TMAX": "maximum_2m_temperature_in_the_last_24_hours",
                "TMIN": "minimum_2m_temperature_in_the_last_24_hours",
                "UGRD10": "10m_u_component_of_wind",
                "VGRD10": "10m_v_component_of_wind",
                "SST": "sea_surface_temperature",
                "SLP": "mean_sea_level_pressure",
                "DSWR": "surface_solar_radiation_downwards",
                "DLWR": "surface_thermal_radiation_downwards",
                "NOLR": "top_net_thermal_radiation",
                "SRUNOFF": "surface_runoff",
                "RUNOFF": "runoff",
            }
    
            variables_2 = {
                "HUSS_1000": "specific_humidity",
                "HUSS_925": "specific_humidity",
                "HUSS_850": "specific_humidity",
                "UGRD_1000": "u_component_of_wind",
                "UGRD_925": "u_component_of_wind",
                "UGRD_850": "u_component_of_wind",
                "UGRD_700": "u_component_of_wind",
                "UGRD_200": "u_component_of_wind",
                "VGRD_1000": "v_component_of_wind",
                "VGRD_925": "v_component_of_wind",
                "VGRD_850": "v_component_of_wind",
                "VGRD_700": "v_component_of_wind",
                "VGRD_200": "v_component_of_wind",
            }
    
            init_day_dict_jma = {
                "01": 16, "02": 10, "03": 12, "04": 11, "05": 16, "06": 15,
                "07": 15, "08": 14, "09": 13, "10": 13, "11": 12, "12": 12,
            }
            init_day_dict_ncep = {
                "01": 1, "02": 5, "03": 2, "04": 1, "05": 1, "06": 5,
                "07": 5, "08": 4, "09": 3, "10": 3, "11": 2, "12": 2,
            }
    
            def _open_download(path, fmt):
                if fmt == "grib":
                    return xr.open_dataset(
                        path,
                        engine="cfgrib",
                        backend_kwargs={"indexpath": ""},
                    )
                return xr.open_dataset(path)
    
            def _normalize_cds_dataset(ds):
                rename_map = {}
    
                if "number" in ds.dims or "number" in ds.coords:
                    rename_map["number"] = "member"
                if "longitude" in ds.dims or "longitude" in ds.coords:
                    rename_map["longitude"] = "X"
                if "latitude" in ds.dims or "latitude" in ds.coords:
                    rename_map["latitude"] = "Y"
                if "forecast_period" in ds.dims or "forecast_period" in ds.coords:
                    rename_map["forecast_period"] = "leadtime"
                if "step" in ds.dims or "step" in ds.coords:
                    rename_map["step"] = "leadtime"
    
                # --- Initialization / reference-time coordinate ---------------------
                # CDS NetCDF names it "forecast_reference_time" or "indexing_time"
                # (dim or scalar coord). cfgrib on GRIB1 (DWD, etc.) instead exposes
                # it as a SCALAR coord literally named "time" (standard_name =
                # forecast_reference_time), which the dim-only checks would miss ->
                # "T" would never be built and the valid_time layout would collapse.
                # We resolve whichever of these is present to "T".
                ref_name = None
                for cand in ("forecast_reference_time", "indexing_time"):
                    if cand in ds.dims or cand in ds.coords:
                        ref_name = cand
                        break
                if ref_name is None and ("time" in ds.coords or "time" in ds.dims):
                    t = ds["time"] if "time" in ds.variables else None
                    looks_like_ref = (
                        (t is not None and t.attrs.get("standard_name") == "forecast_reference_time")
                        or "step" in ds.dims
                        or "forecast_period" in ds.dims
                        or "step" in rename_map
                        or "forecast_period" in rename_map
                    )
                    if looks_like_ref:
                        ref_name = "time"
                if ref_name is not None:
                    rename_map[ref_name] = "T"
    
                if "isobaricInhPa" in ds.dims or "isobaricInhPa" in ds.coords:
                    rename_map["isobaricInhPa"] = "level"
                if "pressure_level" in ds.dims or "pressure_level" in ds.coords:
                    rename_map["pressure_level"] = "level"
    
                ds = ds.rename({k: v for k, v in rename_map.items() if k in ds})
    
                # Promote a scalar T coord to a length-1 dimension so downstream
                # `"T" in ds.dims` guards (e.g. _rearrange_to_valid_time) pass. This
                # covers forecast_reference_time, indexing_time and the GRIB1 "time"
                # case in one line, using the existing datetime value.
                if "T" in ds.coords and "T" not in ds.dims:
                    ds = ds.expand_dims("T")
    
                if "valid_time" not in ds.coords and "T" in ds.coords and "leadtime" in ds.coords:
                    ds = ds.assign_coords(valid_time=ds["T"] + ds["leadtime"])
    
                drop_vars = [
                    "surface",
                    "heightAboveGround",
                    "meanSea",
                    "entireAtmosphere",
                ]
                ds = ds.drop_vars([name for name in drop_vars if name in ds.variables], errors="ignore")
    
                if "Y" in ds.coords:
                    ds = ds.sortby("Y")
    
                preferred = ["member", "T", "leadtime", "level", "Y", "X"]
                for name in list(ds.data_vars):
                    dims = [dim for dim in preferred if dim in ds[name].dims]
                    dims += [dim for dim in ds[name].dims if dim not in dims]
                    ds[name] = ds[name].transpose(*dims)
    
                return ds
    
            def _deaccumulate(ds, dim="leadtime"):
                if dim not in ds.dims:
                    return ds
                first = ds.isel({dim: slice(0, 1)})
                diff = ds.diff(dim)
                out = xr.concat([first, diff], dim=dim)
                out = out.assign_coords({dim: ds[dim]})
                return out.where(out >= 0, 0)
    
            def _grid_cell_area_m2(ds):
                if "Y" not in ds.coords or "X" not in ds.coords:
                    raise ValueError("RUNOFF conversion to m3/s needs X and Y coordinates.")
                if ds.sizes.get("Y", 0) < 2 or ds.sizes.get("X", 0) < 2:
                    raise ValueError("RUNOFF conversion to m3/s needs at least two X and Y points.")
    
                radius = 6371000.0
                dlat = np.deg2rad(float(abs(ds["Y"].diff("Y").median())))
                dlon = np.deg2rad(float(abs(ds["X"].diff("X").median())))
                area_y = (radius ** 2) * dlat * dlon * np.cos(np.deg2rad(ds["Y"]))
                area_y.attrs["units"] = "m2"
                return area_y
    
            def _leadtime_interval_seconds(ds):
                if "leadtime" not in ds.coords:
                    raise ValueError("RUNOFF conversion to m3/s needs leadtime coordinates.")
    
                lead = ds["leadtime"]
                if np.issubdtype(lead.dtype, np.timedelta64):
                    lead_seconds = (lead / np.timedelta64(1, "s")).astype(float)
                else:
                    # CDS leadtime_hour requests are in hours when decoded as numeric values.
                    lead_seconds = lead.astype(float) * 3600.0
    
                first = lead_seconds.isel(leadtime=slice(0, 1))
                diff = lead_seconds.diff("leadtime")
                seconds = xr.concat([first, diff], dim="leadtime")
                seconds = seconds.assign_coords(leadtime=lead)
                return seconds.where(seconds > 0)
    
            def _apply_units(ds, var_code):
                if var_code in ["TMIN", "TEMP", "TMAX", "SST", "TDEW"]:
                    ds = ds - 273.15
                    for name in ds.data_vars:
                        ds[name].attrs["units"] = "degC"
    
                elif var_code == "SLP":
                    ds = ds / 100.0
                    for name in ds.data_vars:
                        ds[name].attrs["units"] = "hPa"
    
                elif var_code == "PRCP":
                    ds = _deaccumulate(ds) * 1000.0
                    for name in ds.data_vars:
                        ds[name].attrs["units"] = "mm"
    
                elif var_code in ["SRUNOFF", "RUNOFF"]:
                    ds = _deaccumulate(ds)
                    if runoff_units == "mm":
                        ds = ds * 1000.0
                        for name in ds.data_vars:
                            ds[name].attrs["units"] = "mm"
                    else:
                        cell_area = _grid_cell_area_m2(ds)
                        interval_seconds = _leadtime_interval_seconds(ds)
                        ds = (ds * cell_area) / interval_seconds
                        for name in ds.data_vars:
                            ds[name].attrs["units"] = "m3 s-1"
    
                elif var_code in ["DSWR", "DLWR", "NOLR"]:
                    ds = _deaccumulate(ds) / 86400.0
                    for name in ds.data_vars:
                        ds[name].attrs["units"] = "W m-2"
    
                return ds
    
            def _rearrange_to_valid_time(ds):
                if "T" in ds.dims and "leadtime" in ds.dims:
                    if "valid_time" not in ds.coords:
                        ds = ds.assign_coords(valid_time=ds["T"] + ds["leadtime"])
    
                    ds = ds.stack(_sample=("T", "leadtime"))
                    ds = ds.reset_index("_sample")
    
                    valid_time = ds["valid_time"].values
                    if "T" in ds.coords:
                        ds = ds.rename({"T": "init_time"})
    
                    ds = ds.assign_coords(T=("_sample", valid_time))
                    ds = ds.swap_dims({"_sample": "T"})
                    ds = ds.drop_vars("_sample", errors="ignore")
                    ds = ds.sortby("T")
    
                ds = ds.drop_vars(["valid_time", "init_time", "leadtime"], errors="ignore")
    
                if "member" in ds.dims:
                    ds = ds.rename({"member": "M"})
    
                for name in list(ds.data_vars):
                    if "M" in ds[name].dims:
                        preferred = ["M", "T", "level", "Y", "X"]
                    else:
                        preferred = ["T", "level", "Y", "X"]
                    dims = [dim for dim in preferred if dim in ds[name].dims]
                    dims += [dim for dim in ds[name].dims if dim not in dims]
                    ds[name] = ds[name].transpose(*dims)
    
                return ds
    
            dir_to_save = Path(dir_to_save)
            dir_to_save.mkdir(parents=True, exist_ok=True)
    
            month_key = f"{int(month_of_initialization):02}"
            abb_month_ini = month_abbr[int(month_of_initialization)]
            years_str = f"{years[0]}_{years[-1]}" if len(years) > 1 else years[0]
            lead_str = f"{leadtime_hour[0]}-{leadtime_hour[-1]}" if len(leadtime_hour) > 1 else leadtime_hour[0]
    
            store_file_path = {}
            client = cdsapi.Client()
    
            for cv in center_variable:
                try:
                    c, v = cv.split(".", 1)
                    if c not in centre:
                        raise ValueError(f"Unknown centre/system code: {c}")
                    if v not in variables_1 and v not in variables_2:
                        raise ValueError(f"Unknown variable code: {v}")
    
                    cent = centre[c]
                    syst = system[c]
    
                    request_day = int(day_of_initialization)
                    if year_forecast is None and cent == "jma":
                        request_day = init_day_dict_jma[month_key]
                    elif year_forecast is None and cent == "ncep":
                        request_day = init_day_dict_ncep[month_key]
    
                    day_key = f"{int(request_day):02}"
                    unit_tag = "_m3s" if v in ["RUNOFF", "SRUNOFF"] and runoff_units == "m3/s" else ""
                    output_file = (
                        dir_to_save
                        / f"{file_prefix}_{cent}{syst}_{v}_{abb_month_ini}{day_key}_{years_str}_{lead_str}{unit_tag}.nc"
                    )
    
                    if output_file.exists() and not force_download:
                        print(f"{output_file} already exists. Skipping download.")
                        store_file_path[cv] = output_file
                        continue
    
                    fmt = data_format.lower()
                    if fmt not in ["netcdf", "grib"]:
                        raise ValueError("data_format must be 'netcdf' or 'grib'.")
    
                    suffix = "grib" if fmt == "grib" else "nc"
                    temp_file = dir_to_save / f"temp_{cent}{syst}_{v}.{suffix}"
    
                    if v in variables_2:
                        dataset = "seasonal-original-pressure-levels"
                        pressure_level = v.rsplit("_", 1)[1]
                        request = {
                            "originating_centre": cent,
                            "system": syst,
                            "variable": [variables_2[v]],
                            "pressure_level": [pressure_level],
                            "year": years,
                            "month": [month_key],
                            "day": [day_key],
                            "leadtime_hour": [str(lt) for lt in leadtime_hour],
                            "data_format": fmt,
                            "area": area,
                        }
                    else:
                        dataset = "seasonal-original-single-levels"
                        request = {
                            "originating_centre": cent,
                            "system": syst,
                            "variable": [variables_1[v]],
                            "year": years,
                            "month": [month_key],
                            "day": [day_key],
                            "leadtime_hour": [str(lt) for lt in leadtime_hour],
                            "data_format": fmt,
                            "area": area,
                        }
    
                    print(f"Requesting data from '{dataset}' for {cv}...")
                    client.retrieve(dataset, request).download(str(temp_file))
                    print(f"Downloaded: {temp_file}")
    
                    ds = _open_download(temp_file, fmt)
                    ds = _normalize_cds_dataset(ds)
    
                    if ensemble_mean in ["mean", "median"] and "member" in ds.dims:
                        ds = getattr(ds, ensemble_mean)(dim="member")
    
                    ds = _apply_units(ds, v)
                    if output_layout == "valid_time":
                        ds = _rearrange_to_valid_time(ds)
    
                    ds.attrs["source_dataset"] = dataset
                    ds.attrs["originating_centre"] = cent
                    ds.attrs["system"] = syst
                    ds.attrs["variable_code"] = v
                    ds.attrs["download_format"] = fmt
                    ds.attrs["output_layout"] = output_layout
                    if v in ["RUNOFF", "SRUNOFF"]:
                        ds.attrs["runoff_units"] = runoff_units
    
                    encoding = {name: {"zlib": True, "complevel": 4} for name in ds.data_vars}
                    ds.to_netcdf(output_file, encoding=encoding)
                    print(f"Saved processed data to: {output_file}")
    
                    ds.close()
                    store_file_path[cv] = output_file
    
                    if temp_file.exists():
                        os.remove(temp_file)
                        print(f"Deleted temp file: {temp_file}")
    
                    del ds, request
                    gc.collect()
    
                except Exception as exc:
                    print(f"Failed to download data for {cv}: {exc}")
    
                _time.sleep(1)
    
            return store_file_path

    # def WAS_Download_Models_Daily(
    #     self,
    #     dir_to_save,
    #     center_variable,
    #     month_of_initialization,
    #     day_of_initialization,
    #     leadtime_hour,
    #     year_start_hindcast,
    #     year_end_hindcast,
    #     area,
    #     year_forecast=None,
    #     ensemble_mean=None,
    #     force_download=False,
    #     data_format="netcdf",
    #     output_layout="valid_time",
    #     runoff_units="m3/s",
    # ):
    #     """
    #     Download daily seasonal hindcast/forecast data from CDS seasonal-original
    #     datasets and save a clean WAS-style NetCDF.
    
    #     Output conventions:
    #         output_layout="valid_time":
    #             T, Y, X when ensemble_mean is "mean" or "median"
    #             T, M, Y, X when members are kept
    
    #         output_layout="init_leadtime":
    #             member, T, leadtime, level, Y, X
    
    #     Notes:
    #         - In "valid_time" layout, T is the valid forecast date.
    #         - In "init_leadtime" layout, T is the initialization date/year.
    #         - Accumulated variables are deaccumulated along leadtime.
    #         - RUNOFF can be saved as "mm" or "m3/s".
    #     """
    #     from calendar import month_abbr
    #     from pathlib import Path
    #     import gc
    #     import os
    #     import time as _time
    
    #     import cdsapi
    #     import numpy as np
    #     import xarray as xr
    
    #     if isinstance(center_variable, str):
    #         raise TypeError(
    #             "center_variable must be a list, for example "
    #             "['ECMWF_51.RUNOFF'] or ['ECMWF_51.RUNOFF', 'UKMO_604.RUNOFF']."
    #         )
    #     center_variable = list(center_variable)
    
    #     if isinstance(leadtime_hour, (str, int, np.integer)):
    #         leadtime_hour = [str(leadtime_hour)]
    #     else:
    #         leadtime_hour = [str(lt) for lt in leadtime_hour]
    
    #     if ensemble_mean not in [None, "mean", "median"]:
    #         raise ValueError("ensemble_mean must be None, 'mean', or 'median'.")
    
    #     if output_layout not in ["valid_time", "init_leadtime"]:
    #         raise ValueError("output_layout must be 'valid_time' or 'init_leadtime'.")
    
    #     if runoff_units not in ["mm", "m3/s"]:
    #         raise ValueError("runoff_units must be 'mm' or 'm3/s'.")
    
    #     if year_forecast is None:
    #         years = [str(y) for y in range(year_start_hindcast, year_end_hindcast + 1)]
    #         file_prefix = "hindcast"
    #     else:
    #         years = [str(year_forecast)]
    #         file_prefix = "forecast"
    
    #     centre = {
    #         "BOM_2": "bom",
    #         "ECMWF_51": "ecmwf",
    #         "UKMO_603": "ukmo",
    #         "UKMO_604": "ukmo",
    #         "UKMO_605": "ukmo",
    #         "UKMO_610": "ukmo",
    #         "METEOFRANCE_8": "meteo_france",
    #         "METEOFRANCE_9": "meteo_france",
    #         "DWD_21": "dwd",
    #         "DWD_22": "dwd",
    #         "CMCC_35": "cmcc",
    #         "CMCC_4": "cmcc",
    #         "NCEP_2": "ncep",
    #         "JMA_3": "jma",
    #         "JMA_4": "jma",
    #         "ECCC_4": "eccc",
    #         "ECCC_5": "eccc",
    #     }
    
    #     system = {
    #         "BOM_2": "2",
    #         "ECMWF_51": "51",
    #         "UKMO_603": "603",
    #         "UKMO_604": "604",
    #         "UKMO_605": "605",
    #         "UKMO_610": "610",
    #         "METEOFRANCE_8": "8",
    #         "METEOFRANCE_9": "9",
    #         "DWD_21": "21",
    #         "DWD_22": "22",
    #         "CMCC_35": "35",
    #         "CMCC_4": "4",
    #         "NCEP_2": "2",
    #         "JMA_3": "3",
    #         "JMA_4": "4",
    #         "ECCC_4": "4",
    #         "ECCC_5": "5",
    #     }
    
    #     variables_1 = {
    #         "PRCP": "total_precipitation",
    #         "TEMP": "2m_temperature",
    #         "TDEW": "2m_dewpoint_temperature",
    #         "TMAX": "maximum_2m_temperature_in_the_last_24_hours",
    #         "TMIN": "minimum_2m_temperature_in_the_last_24_hours",
    #         "UGRD10": "10m_u_component_of_wind",
    #         "VGRD10": "10m_v_component_of_wind",
    #         "SST": "sea_surface_temperature",
    #         "SLP": "mean_sea_level_pressure",
    #         "DSWR": "surface_solar_radiation_downwards",
    #         "DLWR": "surface_thermal_radiation_downwards",
    #         "NOLR": "top_net_thermal_radiation",
    #         "SRUNOFF": "surface_runoff",
    #         "RUNOFF": "runoff",
    #     }
    
    #     variables_2 = {
    #         "HUSS_1000": "specific_humidity",
    #         "HUSS_925": "specific_humidity",
    #         "HUSS_850": "specific_humidity",
    #         "UGRD_1000": "u_component_of_wind",
    #         "UGRD_925": "u_component_of_wind",
    #         "UGRD_850": "u_component_of_wind",
    #         "UGRD_700": "u_component_of_wind",
    #         "UGRD_200": "u_component_of_wind",
    #         "VGRD_1000": "v_component_of_wind",
    #         "VGRD_925": "v_component_of_wind",
    #         "VGRD_850": "v_component_of_wind",
    #         "VGRD_700": "v_component_of_wind",
    #         "VGRD_200": "v_component_of_wind",
    #     }
    
    #     init_day_dict_jma = {
    #         "01": 16, "02": 10, "03": 12, "04": 11, "05": 16, "06": 15,
    #         "07": 15, "08": 14, "09": 13, "10": 13, "11": 12, "12": 12,
    #     }
    #     init_day_dict_ncep = {
    #         "01": 1, "02": 5, "03": 2, "04": 1, "05": 1, "06": 5,
    #         "07": 5, "08": 4, "09": 3, "10": 3, "11": 2, "12": 2,
    #     }
    
    #     def _open_download(path, fmt):
    #         if fmt == "grib":
    #             return xr.open_dataset(
    #                 path,
    #                 engine="cfgrib",
    #                 backend_kwargs={"indexpath": ""},
    #             )
    #         return xr.open_dataset(path)
    
    #     def _normalize_cds_dataset(ds):
    #         rename_map = {}
    
    #         if "number" in ds.dims or "number" in ds.coords:
    #             rename_map["number"] = "member"
    #         if "longitude" in ds.dims or "longitude" in ds.coords:
    #             rename_map["longitude"] = "X"
    #         if "latitude" in ds.dims or "latitude" in ds.coords:
    #             rename_map["latitude"] = "Y"
    #         if "forecast_period" in ds.dims or "forecast_period" in ds.coords:
    #             rename_map["forecast_period"] = "leadtime"
    #         if "step" in ds.dims or "step" in ds.coords:
    #             rename_map["step"] = "leadtime"
    #         if "forecast_reference_time" in ds.dims:
    #             rename_map["forecast_reference_time"] = "T"
    #         elif "indexing_time" in ds.dims:
    #             rename_map["indexing_time"] = "T"
    #         elif "time" in ds.dims and ("forecast_period" in ds.dims or "step" in ds.dims):
    #             rename_map["time"] = "T"
    #         if "isobaricInhPa" in ds.dims or "isobaricInhPa" in ds.coords:
    #             rename_map["isobaricInhPa"] = "level"
    #         if "pressure_level" in ds.dims or "pressure_level" in ds.coords:
    #             rename_map["pressure_level"] = "level"
    
    #         ds = ds.rename({k: v for k, v in rename_map.items() if k in ds})
    
    #         if "T" not in ds.dims:
    #             if "forecast_reference_time" in ds.coords:
    #                 ds = ds.expand_dims(T=np.atleast_1d(ds["forecast_reference_time"].values))
    #                 ds = ds.drop_vars("forecast_reference_time", errors="ignore")
    #             elif "indexing_time" in ds.coords:
    #                 ds = ds.expand_dims(T=np.atleast_1d(ds["indexing_time"].values))
    #                 ds = ds.drop_vars("indexing_time", errors="ignore")
    
    #         if "valid_time" not in ds.coords and "T" in ds.coords and "leadtime" in ds.coords:
    #             ds = ds.assign_coords(valid_time=ds["T"] + ds["leadtime"])
    
    #         drop_vars = [
    #             "surface",
    #             "heightAboveGround",
    #             "meanSea",
    #             "entireAtmosphere",
    #         ]
    #         ds = ds.drop_vars([name for name in drop_vars if name in ds.variables], errors="ignore")
    
    #         if "Y" in ds.coords:
    #             ds = ds.sortby("Y")
    
    #         preferred = ["member", "T", "leadtime", "level", "Y", "X"]
    #         for name in list(ds.data_vars):
    #             dims = [dim for dim in preferred if dim in ds[name].dims]
    #             dims += [dim for dim in ds[name].dims if dim not in dims]
    #             ds[name] = ds[name].transpose(*dims)
    
    #         return ds
    
    #     def _deaccumulate(ds, dim="leadtime"):
    #         if dim not in ds.dims:
    #             return ds
    #         first = ds.isel({dim: slice(0, 1)})
    #         diff = ds.diff(dim)
    #         out = xr.concat([first, diff], dim=dim)
    #         out = out.assign_coords({dim: ds[dim]})
    #         return out.where(out >= 0, 0)
    
    #     def _grid_cell_area_m2(ds):
    #         if "Y" not in ds.coords or "X" not in ds.coords:
    #             raise ValueError("RUNOFF conversion to m3/s needs X and Y coordinates.")
    #         if ds.sizes.get("Y", 0) < 2 or ds.sizes.get("X", 0) < 2:
    #             raise ValueError("RUNOFF conversion to m3/s needs at least two X and Y points.")
    
    #         radius = 6371000.0
    #         dlat = np.deg2rad(float(abs(ds["Y"].diff("Y").median())))
    #         dlon = np.deg2rad(float(abs(ds["X"].diff("X").median())))
    #         area_y = (radius ** 2) * dlat * dlon * np.cos(np.deg2rad(ds["Y"]))
    #         area_y.attrs["units"] = "m2"
    #         return area_y
    
    #     def _leadtime_interval_seconds(ds):
    #         if "leadtime" not in ds.coords:
    #             raise ValueError("RUNOFF conversion to m3/s needs leadtime coordinates.")
    
    #         lead = ds["leadtime"]
    #         if np.issubdtype(lead.dtype, np.timedelta64):
    #             lead_seconds = (lead / np.timedelta64(1, "s")).astype(float)
    #         else:
    #             # CDS leadtime_hour requests are in hours when decoded as numeric values.
    #             lead_seconds = lead.astype(float) * 3600.0
    
    #         first = lead_seconds.isel(leadtime=slice(0, 1))
    #         diff = lead_seconds.diff("leadtime")
    #         seconds = xr.concat([first, diff], dim="leadtime")
    #         seconds = seconds.assign_coords(leadtime=lead)
    #         return seconds.where(seconds > 0)
    
    #     def _apply_units(ds, var_code):
    #         if var_code in ["TMIN", "TEMP", "TMAX", "SST", "TDEW"]:
    #             ds = ds - 273.15
    #             for name in ds.data_vars:
    #                 ds[name].attrs["units"] = "degC"
    
    #         elif var_code == "SLP":
    #             ds = ds / 100.0
    #             for name in ds.data_vars:
    #                 ds[name].attrs["units"] = "hPa"
    
    #         elif var_code == "PRCP":
    #             ds = _deaccumulate(ds) * 1000.0
    #             for name in ds.data_vars:
    #                 ds[name].attrs["units"] = "mm"
    
    #         elif var_code in ["SRUNOFF", "RUNOFF"]:
    #             ds = _deaccumulate(ds)
    #             if runoff_units == "mm":
    #                 ds = ds * 1000.0
    #                 for name in ds.data_vars:
    #                     ds[name].attrs["units"] = "mm"
    #             else:
    #                 cell_area = _grid_cell_area_m2(ds)
    #                 interval_seconds = _leadtime_interval_seconds(ds)
    #                 ds = (ds * cell_area) / interval_seconds
    #                 for name in ds.data_vars:
    #                     ds[name].attrs["units"] = "m3 s-1"
    
    #         elif var_code in ["DSWR", "DLWR", "NOLR"]:
    #             ds = _deaccumulate(ds) / 86400.0
    #             for name in ds.data_vars:
    #                 ds[name].attrs["units"] = "W m-2"
    
    #         return ds
    
    #     def _rearrange_to_valid_time(ds):
    #         if "T" in ds.dims and "leadtime" in ds.dims:
    #             if "valid_time" not in ds.coords:
    #                 ds = ds.assign_coords(valid_time=ds["T"] + ds["leadtime"])
    
    #             ds = ds.stack(_sample=("T", "leadtime"))
    #             ds = ds.reset_index("_sample")
    
    #             valid_time = ds["valid_time"].values
    #             if "T" in ds.coords:
    #                 ds = ds.rename({"T": "init_time"})
    
    #             ds = ds.assign_coords(T=("_sample", valid_time))
    #             ds = ds.swap_dims({"_sample": "T"})
    #             ds = ds.drop_vars("_sample", errors="ignore")
    #             ds = ds.sortby("T")
    
    #         ds = ds.drop_vars(["valid_time", "init_time", "leadtime"], errors="ignore")
    
    #         if "member" in ds.dims:
    #             ds = ds.rename({"member": "M"})
    
    #         for name in list(ds.data_vars):
    #             if "M" in ds[name].dims:
    #                 preferred = ["M", "T", "level", "Y", "X"]
    #             else:
    #                 preferred = ["T", "level", "Y", "X"]
    #             dims = [dim for dim in preferred if dim in ds[name].dims]
    #             dims += [dim for dim in ds[name].dims if dim not in dims]
    #             ds[name] = ds[name].transpose(*dims)
    
    #         return ds
    
    #     dir_to_save = Path(dir_to_save)
    #     dir_to_save.mkdir(parents=True, exist_ok=True)
    
    #     month_key = f"{int(month_of_initialization):02}"
    #     abb_month_ini = month_abbr[int(month_of_initialization)]
    #     years_str = f"{years[0]}_{years[-1]}" if len(years) > 1 else years[0]
    #     lead_str = f"{leadtime_hour[0]}-{leadtime_hour[-1]}" if len(leadtime_hour) > 1 else leadtime_hour[0]
    
    #     store_file_path = {}
    #     client = cdsapi.Client()
    
    #     for cv in center_variable:
    #         try:
    #             c, v = cv.split(".", 1)
    #             if c not in centre:
    #                 raise ValueError(f"Unknown centre/system code: {c}")
    #             if v not in variables_1 and v not in variables_2:
    #                 raise ValueError(f"Unknown variable code: {v}")
    
    #             cent = centre[c]
    #             syst = system[c]
    
    #             request_day = int(day_of_initialization)
    #             if year_forecast is None and cent == "jma":
    #                 request_day = init_day_dict_jma[month_key]
    #             elif year_forecast is None and cent == "ncep":
    #                 request_day = init_day_dict_ncep[month_key]
    
    #             day_key = f"{int(request_day):02}"
    #             unit_tag = "_m3s" if v in ["RUNOFF", "SRUNOFF"] and runoff_units == "m3/s" else ""
    #             output_file = (
    #                 dir_to_save
    #                 / f"{file_prefix}_{cent}{syst}_{v}_{abb_month_ini}{day_key}_{years_str}_{lead_str}{unit_tag}.nc"
    #             )
    
    #             if output_file.exists() and not force_download:
    #                 print(f"{output_file} already exists. Skipping download.")
    #                 store_file_path[cv] = output_file
    #                 continue
    
    #             fmt = data_format.lower()
    #             if fmt not in ["netcdf", "grib"]:
    #                 raise ValueError("data_format must be 'netcdf' or 'grib'.")
    
    #             suffix = "grib" if fmt == "grib" else "nc"
    #             temp_file = dir_to_save / f"temp_{cent}{syst}_{v}.{suffix}"
    
    #             if v in variables_2:
    #                 dataset = "seasonal-original-pressure-levels"
    #                 pressure_level = v.rsplit("_", 1)[1]
    #                 request = {
    #                     "originating_centre": cent,
    #                     "system": syst,
    #                     "variable": [variables_2[v]],
    #                     "pressure_level": [pressure_level],
    #                     "year": years,
    #                     "month": [month_key],
    #                     "day": [day_key],
    #                     "leadtime_hour": [str(lt) for lt in leadtime_hour],
    #                     "data_format": fmt,
    #                     "area": area,
    #                 }
    #             else:
    #                 dataset = "seasonal-original-single-levels"
    #                 request = {
    #                     "originating_centre": cent,
    #                     "system": syst,
    #                     "variable": [variables_1[v]],
    #                     "year": years,
    #                     "month": [month_key],
    #                     "day": [day_key],
    #                     "leadtime_hour": [str(lt) for lt in leadtime_hour],
    #                     "data_format": fmt,
    #                     "area": area,
    #                 }
    
    #             print(f"Requesting data from '{dataset}' for {cv}...")
    #             client.retrieve(dataset, request).download(str(temp_file))
    #             print(f"Downloaded: {temp_file}")
    
    #             ds = _open_download(temp_file, fmt)
    #             ds = _normalize_cds_dataset(ds)
    
    #             if ensemble_mean in ["mean", "median"] and "member" in ds.dims:
    #                 ds = getattr(ds, ensemble_mean)(dim="member")
    
    #             ds = _apply_units(ds, v)
    #             if output_layout == "valid_time":
    #                 ds = _rearrange_to_valid_time(ds)
    
    #             ds.attrs["source_dataset"] = dataset
    #             ds.attrs["originating_centre"] = cent
    #             ds.attrs["system"] = syst
    #             ds.attrs["variable_code"] = v
    #             ds.attrs["download_format"] = fmt
    #             ds.attrs["output_layout"] = output_layout
    #             if v in ["RUNOFF", "SRUNOFF"]:
    #                 ds.attrs["runoff_units"] = runoff_units
    
    #             encoding = {name: {"zlib": True, "complevel": 4} for name in ds.data_vars}
    #             ds.to_netcdf(output_file, encoding=encoding)
    #             print(f"Saved processed data to: {output_file}")
    
    #             ds.close()
    #             store_file_path[cv] = output_file
    
    #             if temp_file.exists():
    #                 os.remove(temp_file)
    #                 print(f"Deleted temp file: {temp_file}")
    
    #             del ds, request
    #             gc.collect()
    
    #         except Exception as exc:
    #             print(f"Failed to download data for {cv}: {exc}")
    
    #         _time.sleep(1)
    
    #     return store_file_path
    

    # def WAS_Download_Models_Daily(
    #     self,
    #     dir_to_save,
    #     center_variable,         # e.g. ["ECMWF_51.PRCP", "UKMO_603.TEMP", ...]
    #     month_of_initialization, # int: e.g. 2 for February
    #     day_of_initialization,   # int: e.g. 1 for the 1st day
    #     leadtime_hour,           # list of strings: e.g. ["24","48",..., "5160"]
    #     year_start_hindcast,
    #     year_end_hindcast,
    #     area,
    #     year_forecast=None,
    #     ensemble_mean=None,
    #     force_download=False,
    # ):
    #     """
    #     Download daily/sub-daily seasonal forecast model data (original)
    #     using 'seasonal-original-single-levels' from the CDS.
    
    #     Parameters:
    #         dir_to_save (str or Path): Directory to save the downloaded files.
    #         center_variable (list): Each element e.g. "ECMWF_51.PRCP"
    #             - left side of '.' is model (ECMWF_51),
    #             - right side is variable short code (PRCP).
    #         month_of_initialization (int): Initialization month (1-12).
    #         day_of_initialization (int): Initialization day (1-31).
    #         leadtime_hour (list of str): e.g. ["24", "48", ..., "5160"].
    #         year_start_hindcast (int): Start year for hindcast data.
    #         year_end_hindcast (int): End year for hindcast data.
    #         area (list): Bounding box as [North, West, South, East].
    #         year_forecast (int, optional): If provided, downloads that single
    #             forecast year. Otherwise downloads hindcast for the specified range.
    #         ensemble_mean (str, optional): e.g. "mean", "median", or None.
    #         force_download (bool): Force download if True, even if file exists.
    #     """
    
    #     # 1. Determine whether we are downloading hindcast or forecast.
    #     if year_forecast is None:
    #         # Hindcast range
    #         years = [str(y) for y in range(year_start_hindcast, year_end_hindcast + 1)]
    #         file_prefix = "hindcast"
    #     else:
    #         # Single forecast year
    #         years = [str(year_forecast)]
    #         file_prefix = "forecast"
    
    #     # 2. Build standard dictionaries for center/system/variables
    #     centre = {
    #         "BOM_2": "bom",
    #         "ECMWF_51": "ecmwf",
    #         "UKMO_604": "ukmo", # month of initialization available for forecast are Apr to __
    #         "UKMO_603": "ukmo", # month of initialization available for forecast are Jan to Mar
    #         "UKMO_605": "ukmo",
    #         "UKMO_610": "ukmo",
    #         "METEOFRANCE_8": "meteo_france",
    #         "METEOFRANCE_9": "meteo_france",
    #         "DWD_21": "dwd",
    #         "DWD_22": "dwd",
    #         # "DWD_2": "dwd",
    #         "CMCC_35": "cmcc",
    #         "CMCC_4": "cmcc",
    #         "NCEP_2": "ncep",
    #         "JMA_3": "jma",
    #         "JMA_4": "jma",
    #         "ECCC_4": "eccc",
    #         "ECCC_5": "eccc",
    #     }
    
    #     system = {
    #         "BOM_2": "2",
    #         "ECMWF_51": "51",
    #         "UKMO_604": "604",
    #         "UKMO_603": "603",
    #         "UKMO_605": "605",
    #         "UKMO_610": "610",
    #         "METEOFRANCE_8": "8",
    #         "METEOFRANCE_9": "9",
    #         "DWD_21": "21",
    #         "DWD_22": "22",
    #         # "DWD_2": "2",
    #         "CMCC_35": "35",
    #         "CMCC_4": "4",
    #         "NCEP_2": "2",
    #         "JMA_3": "3",
    #         "JMA_4": "4",
    #         "ECCC_4": "4",
    #         "ECCC_5": "5",
    #     }
    
    #     variables_1 = {
    #         "PRCP":  "total_precipitation",
    #         "TEMP":  "2m_temperature",
    #         "TDEW": "2m_dewpoint_temperature",
    #         "TMAX":  "maximum_2m_temperature_in_the_last_24_hours",
    #         "TMIN":  "minimum_2m_temperature_in_the_last_24_hours",
    #         "UGRD10":"10m_u_component_of_wind",
    #         "VGRD10":"10m_v_component_of_wind",
    #         "SST":   "sea_surface_temperature",
    #         "SLP": "mean_sea_level_pressure",
    #         "DSWR": "surface_solar_radiation_downwards",
    #         "DLWR": "surface_thermal_radiation_downwards",
    #         "NOLR": "top_net_thermal_radiation",
    #         "RUNOFF": "surface_runoff"
            
    #     }
    #     variables_2 = {
    #         "HUSS_1000": "specific_humidity",
    #         "HUSS_925": "specific_humidity",
    #         "HUSS_850": "specific_humidity",
    #         "UGRD_1000": "u_component_of_wind",
    #         "UGRD_925": "u_component_of_wind",
    #         "UGRD_850": "u_component_of_wind",
    #         "UGRD_700": "u_component_of_wind",
    #         "UGRD_200": "u_component_of_wind",
    #         "VGRD_1000": "v_component_of_wind",
    #         "VGRD_925": "v_component_of_wind",
    #         "VGRD_850": "v_component_of_wind",
    #         "VGRD_700": "v_component_of_wind",
    #         "VGRD_200": "v_component_of_wind",
    #     }

    #     ### Particularity for day of initialization NCEP and JMA
    #     init_day_dict_jma = {
    #         "01":16, "02":10, "03":12, "04":11, "05":16, "06":15,
    #         "07":15, "08":14, "09":13, "10":13, "11":12, "12":12
    #     }

    #     init_day_dict_ncep = {
    #         "01":1, "02":5, "03":2, "04":1, "05":1, "06":5,
    #         "07":5, "08":4, "09":3, "10":3, "11":2, "12":2
    #     }
        
    
    #     # 3. Ensure the output directory exists
    #     dir_to_save = Path(dir_to_save)
    #     dir_to_save.mkdir(parents=True, exist_ok=True)
    #     store_file_path = {}
    #     client = cdsapi.Client()        
    #     # 4. Loop over each center-variable combination
    #     for cv in center_variable:
    #         # Example: "ECMWF_51.PRCP"
    #         c = cv.split(".")[0]  # e.g. "ECMWF_51"
    #         v = cv.split(".")[1]  # e.g. "PRCP"
    
    #         # Map to the Copernicus naming
    #         cent = centre[c]
    #         syst = system[c]

    #         ### Particularity for day of initialization NCEP and JMA
    #         if cent == "jma" and year_forecast is None:
    #             day_of_initialization = init_day_dict_jma[month_of_initialization]
    #         if cent == "ncep" and year_forecast is None:
    #             day_of_initialization = init_day_dict_ncep[month_of_initialization]

    #         # Build a single output path
    #         abb_mont_ini = month_abbr[int(month_of_initialization)]
            
    #         # E.g. "hindcast_ecmwf51_PRCP_Feb01_1981-2016_24-5160.nc"
    #         years_str = f"{years[0]}_{years[-1]}" if len(years) > 1 else years[0]
    #         lead_str  = f"{leadtime_hour[0]}-{leadtime_hour[-1]}" if len(leadtime_hour) > 1 else leadtime_hour[0]
    
    #         output_file = (
    #             dir_to_save /
    #             f"{file_prefix}_{cent}{syst}_{v}_{abb_mont_ini}01_{years_str}_{lead_str}.nc"
    #         )

    #         # output_file = (
    #         #     dir_to_save /
    #         #     f"{file_prefix}_{cent}{syst}_{v}_{abb_mont_ini}{day_of_initialization}_{years_str}_{lead_str}.nc"
    #         # )
            

    #         if not force_download and output_file.exists():
    #             print(f"{output_file} already exists. Skipping download.")
    #             store_file_path[f"{cent}{syst}"] = output_file
            
    #         else:

    #             try:
    #                 # Temporary file to download
    #                 temp_file = dir_to_save / f"temp_{cent}{syst}_{v}.nc"

    #                 if v in variables_2:
    #                     press_level = v.split("_")[1]        
    #                     # 5. Prepare the request for 'seasonal-original-pressure-levels'
    #                     dataset = "seasonal-original-pressure-levels"
    #                     request = {
    #                         "originating_centre": cent,
    #                         "system": syst,
    #                         "variable": [variables_2[v]],
    #                         "pressure_level": press_level,
    #                         "year": years,  # list of strings
    #                         "month": [f"{int(month_of_initialization):02}"],
    #                         "day":   [f"{int(day_of_initialization):02}"],
    #                         "leadtime_hour": leadtime_hour,  # e.g. ["24","48",..., "5160"]
    #                         "data_format": "netcdf",
    #                         "area": area,   # e.g. [90, -180, -90, 180]
    #                     }
    #                 else:
    #                     dataset = "seasonal-original-single-levels"
    #                     request = {
    #                         "originating_centre": cent,
    #                         "system": syst,
    #                         "variable": [variables_1[v]],
    #                         "year": years,  # list of strings
    #                         "month": [f"{int(month_of_initialization):02}"],
    #                         "day":   [f"{int(day_of_initialization):02}"],
    #                         "leadtime_hour": leadtime_hour,  # e.g. ["24","48",..., "5160"]
    #                         "data_format": "netcdf",
    #                         "area": area,   # e.g. [90, -180, -90, 180]                    
    #                     }
    #                 # print(request, temp_file, dataset, cent, syst, [variables_1[v]], v, years, month_of_initialization, day_of_initialization, leadtime_hour, area)    
    #                 # 6. Download from CDS
    #                 print(f"Requesting data from '{dataset}' for {cv}...")
    #                 client.retrieve(dataset, request).download(str(temp_file))
    #                 print(f"Downloaded: {temp_file}")

    #                 # 7. Post-process with xarray
    #                 ##########################################################
    #                 # Take in account level pressure for some variables in this part
    #                 ##########################################################
                    
    #                 ds = xr.open_dataset(temp_file)
    #                 if 'forecast_reference_time' in ds.coords:                     
    #                     time = (ds['forecast_reference_time'][0] + ds['forecast_period']).data
    #                     ds = ds.isel(forecast_reference_time=0)
    #                     ds['forecast_period'] = time
    #                     ds = ds.rename({"forecast_period":"time"})
    #                     ds = ds.drop_vars(['forecast_reference_time', 'valid_time', 'number'])

    #                 # elif 'forecast_reference_time' in ds.coords:
    #                 #     time = (ds['forecast_reference_time'] + ds['forecast_period']).data
    #                 #     ds = ds.assign_coords(time=(('forecast_reference_time', 'forecast_period'), time))
    #                 #     ds = ds.stack(time=('forecast_reference_time', 'forecast_period'))
    #                 #     ds = ds.drop_vars(['forecast_reference_time', 'forecast_period'])
    #                 #     ds = ds.rename({"valid_time":"time"})

    #                 else:
    #                     time = (ds['indexing_time'][0] + ds['forecast_period']).data
    #                     ds = ds.isel(forecast_reference_time=0)
    #                     ds['forecast_period'] = time
    #                     ds = ds.rename({"forecast_period":"time"})
    #                     ds = ds.drop_vars(['indexing_time', 'valid_time', 'number'])
                        
    #                     # time = (ds['indexing_time']  + ds['forecast_period']).data
    #                     # ds = ds.assign_coords(time=(('indexing_time', 'forecast_period'), time)).isel(indexing_time=0)
    #                     # ds = ds.stack(time=('indexing_time', 'forecast_period'))
    #                     # ds = ds.drop_vars(['indexing_time', 'forecast_period'])
    #                     # ds = ds.rename({"valid_time":"time"})                    
        
    #                 # If there's an ensemble dimension, apply ensemble mean/median if requested
    #                 if ensemble_mean in ["mean", "median"] and "number" in ds.dims:
    #                     ds = getattr(ds, ensemble_mean)(dim="number")
        
    #                 # Flip latitude
    #                 if "latitude" in ds.coords:
    #                     ds = ds.isel(latitude=slice(None, None, -1))

    #                 if v in ["TMIN","TEMP","TMAX","SST", "TDEW"]:
    #                     ds = ds - 273.15
    #                 if v =="SLP":
    #                     ds = ds / 100
    #                 if v =="PRCP":
    #                     ds['time'] = ds['time'].to_index()
    #                     years_ = ds['time'].dt.year
    #                     tampon = []
    #                     for year_ in np.unique(years_):
                            
    #                         # Select the data for the specific year
    #                         yearly_ds = ds.sel(time=ds['time'].dt.year == year_)
                            
    #                         # Calculate differences for the year
    #                         differences = [yearly_ds.isel(time=i) - yearly_ds.isel(time=i-1) for i in range(1, len(yearly_ds['time']))]
    #                         differences = xr.concat(differences, dim="time")
    #                         differences['time'] = yearly_ds['time'].isel(time=slice(1,None))
    #                         tampon.append(differences)
    #                     ds = (xr.concat(tampon, dim="time") * 1000).where(lambda x: x >= 0, other=0)

    #                 if v=="RUNOFF":
    #                     ds['time'] = ds['time'].to_index()
    #                     diffs = ds.groupby('time.year').apply(lambda x: x.diff('time'))
    #                     ds = diffs.where(lambda x: x >= 0, other=0)
    #                     lat = ds.latitude
    #                     lon = ds.longitude
    #                     dlon = np.deg2rad(0.1)
    #                     dlat = np.deg2rad(0.1)
    #                     r = 6371000 # Earth radius in meters
    #                     # Area for each grid cell
    #                     area_= (r ** 2) * dlon * np.cos(np.deg2rad(lat)) * dlat
    #                     # Perform the conversion to m^3/s
    #                     ds = (ds * area_) / 86400                        

    #                 if v in ["DSWR","DLWR","OLR"]:
    #                     ds['time'] = ds['time'].to_index()
    #                     years_ = ds['time'].dt.year
    #                     tampon = []
    #                     for year_ in np.unique(years_):
                            
    #                         # Select the data for the specific year
    #                         yearly_ds = ds.sel(time=ds['time'].dt.year == year_)
                            
    #                         # Calculate differences for the year
    #                         differences = [yearly_ds.isel(time=i) - yearly_ds.isel(time=i-1) for i in range(1, len(yearly_ds['time']))]
    #                         differences = xr.concat(differences, dim="time")
    #                         differences['time'] = yearly_ds['time'].isel(time=slice(1,None))
    #                         tampon.append(differences)
    #                     ds = xr.concat(tampon, dim="time")/(24*60*60)

    #                 # Finally, rename the coords to X, Y, T to match my style
    #                 if "longitude" in ds.coords:
    #                     ds = ds.rename({"longitude": "X"})
    #                 if "latitude" in ds.coords:
    #                     ds = ds.rename({"latitude": "Y"})
    #                 if "time" in ds.coords:
    #                     ds = ds.rename({"time": "T"})
        
    #                 # 8. Save the processed data
    #                 ds.to_netcdf(output_file)
    #                 print(f"Saved processed data to: {output_file}")
    #                 ds.close()
    #                 store_file_path[f"{cent}{syst}"] = output_file
    #                 os.remove(temp_file)
    #                 print(f"Deleted temp file: {temp_file}")
    #                 del request, ds        
    #                 gc.collect()  
    #             except Exception as e:
    #                 print(f"Failed to download data for {cv}: {e}")  
    #         _time.sleep(1)  # Sleep to avoid overwhelming the server
    #     return store_file_path

    def WAS_Download_AgroIndicators(
            self,
            dir_to_save,
            variables,
            year_start,
            year_end,
            area,
            seas=["01", "02", "03"],  # e.g. NDJ = ["11","12","01"]
            force_download=False,
            max_retries=3,
            retry_delay=5,
        ):
        """
        Download agro-meteorological indicators for specified variables, years, and months,
        handling cross-year seasons (e.g., NDJ) with retries for failed downloads.

        Parameters:
            dir_to_save (str): Directory to save the downloaded files.
            variables (list): List of shorthand variables (e.g., ["AGRO.PRCP", "AGRO.TMAX"]).
            year_start (int): Start year for the data.
            year_end (int): End year for the data.
            area (list): Bounding box as [North, West, South, East].
            seas (list): List of months (e.g., ["11","12","01"] for NDJ).
            force_download (bool): If True, forces download even if file exists.
            max_retries (int): Maximum number of retry attempts for failed downloads (default: 3).
            retry_delay (int): Seconds to wait between retry attempts (default: 5).
        """
        dir_to_save = Path(dir_to_save)
        dir_to_save.mkdir(parents=True, exist_ok=True)

        # Convert season months to integers (e.g., ["11","12","01"] -> [11,12,1])
        season_months = [int(m) for m in seas]
        # Identify the pivot = the first month in your `seas` list
        pivot = season_months[0]

        # Basic mapping
        variable_mapping = {
            "AGRO.PRCP": ("precipitation_flux", None),
            "AGRO.TMAX": ("2m_temperature", "24_hour_maximum"),
            "AGRO.TEMP": ("2m_temperature", "24_hour_mean"),
            "AGRO.TMIN": ("2m_temperature", "24_hour_minimum"),
            "AGRO.DSWR": ("solar_radiation_flux", None),
            "AGRO.ETP": ("reference_evapotranspiration", None),
            "AGRO.WFF": ("10m_wind_speed", "24_hour_mean"),
            "AGRO.HUMAX": ("2m_relative_humidity_derived", "24_hour_maximum"),
            "AGRO.HUMIN": ("2m_relative_humidity_derived", "24_hour_minimum"),
        }

        version = "2_0"
        days = [f"{day:02d}" for day in range(1, 32)]

        # Build a season string for naming (e.g., NDJ)
        season_str = "".join([calendar.month_abbr[m] for m in season_months])

        def month_str(m):
            """Return a zero-padded string month from int."""
            return f"{m:02d}"

        for var in variables:
            if var not in variable_mapping:
                print(f"Unknown variable: {var}. Skipping.")
                continue

            cds_variable, statistic = variable_mapping[var]
            var_short = var.split(".")[1]  # e.g., "PRCP" from "AGRO.PRCP"

            # Output path for the combined dataset across all years
            output_path = dir_to_save / f"Obs_{var_short}_{year_start}_{year_end}_{season_str}.nc"
            if not force_download and output_path.exists():
                print(f"{output_path} already exists. Skipping download.")
                continue

            # Accumulate all partial datasets
            all_years_datasets = []
            all_years_downloaded = True

            # Loop over each year in the requested range
            for year in range(year_start, year_end + 1):
                # Split the months into those belonging to "base" year vs "next" year
                base_months = [m for m in season_months if m >= pivot]
                next_months = [m for m in season_months if m < pivot]

                # 1) Download part A (base-year months), if any
                if base_months:
                    months_base = [month_str(m) for m in base_months]
                    zip_file_path = dir_to_save / f"Obs_{var_short}_{year}_{season_str}_partA.zip"
                    success = False
                    retries = 0

                    while retries < max_retries and not success:
                        try:
                            client = cdsapi.Client()
                            request = {
                                "variable": cds_variable,
                                "year": str(year),
                                "month": months_base,
                                "day": days,
                                "version": version,
                                "area": area,
                            }
                            if statistic:
                                request["statistic"] = [statistic]

                            print(f"Attempt {retries + 1}/{max_retries}: Downloading {cds_variable} for {year} months={months_base}")
                            client.retrieve("sis-agrometeorological-indicators", request).download(str(zip_file_path))
                            success = True
                        except Exception as e:
                            retries += 1
                            print(f"Attempt {retries}/{max_retries} failed for {cds_variable} year={year} Part A: {e}")
                            if retries < max_retries:
                                print(f"Retrying after {retry_delay} seconds...")
                                _time.sleep(retry_delay)
                            if zip_file_path.exists():
                                os.remove(zip_file_path)
                                print(f"Deleted incomplete ZIP file: {zip_file_path}")

                    if not success:
                        print(f"Failed to download {cds_variable} year={year} Part A after {max_retries} attempts.")
                        all_years_downloaded = False
                        continue

                    # Unzip each netCDF and append
                    try:
                        with zipfile.ZipFile(zip_file_path, 'r') as z:
                            for nc_name in z.namelist():
                                with z.open(nc_name) as f:
                                    ds = xr.open_dataset(io.BytesIO(f.read()))
                                    all_years_datasets.append(ds)
                        os.remove(zip_file_path)
                        print(f"Deleted ZIP file: {zip_file_path}")
                    except Exception as e:
                        print(f"Failed to extract/process {zip_file_path}: {e}")
                        all_years_downloaded = False
                        if zip_file_path.exists():
                            os.remove(zip_file_path)
                            print(f"Deleted ZIP file due to processing error: {zip_file_path}")
                        continue

                # 2) Download part B (next-year months), if any and if we have a next year
                if next_months and (year < year_end + 1):
                    year_next = year + 1
                    months_next = [month_str(m) for m in next_months]
                    zip_file_path = dir_to_save / f"Obs_{var_short}_{year}_{season_str}_partB_{year_next}.zip"
                    success = False
                    retries = 0

                    while retries < max_retries and not success:
                        try:
                            client = cdsapi.Client()
                            request = {
                                "variable": cds_variable,
                                "year": str(year_next),
                                "month": months_next,
                                "day": days,
                                "version": version,
                                "area": area,
                            }
                            if statistic:
                                request["statistic"] = [statistic]

                            print(f"Attempt {retries + 1}/{max_retries}: Downloading {cds_variable} for {year_next} months={months_next}")
                            client.retrieve("sis-agrometeorological-indicators", request).download(str(zip_file_path))
                            success = True
                        except Exception as e:
                            retries += 1
                            print(f"Attempt {retries}/{max_retries} failed for {cds_variable} year={year_next} Part B: {e}")
                            if retries < max_retries:
                                print(f"Retrying after {retry_delay} seconds...")
                                _time.sleep(retry_delay)
                            if zip_file_path.exists():
                                os.remove(zip_file_path)
                                print(f"Deleted incomplete ZIP file: {zip_file_path}")

                    if not success:
                        print(f"Failed to download {cds_variable} year={year_next} Part B after {max_retries} attempts.")
                        all_years_downloaded = False
                        continue

                    # Unzip each netCDF and append
                    try:
                        with zipfile.ZipFile(zip_file_path, 'r') as z:
                            for nc_name in z.namelist():
                                with z.open(nc_name) as f:
                                    ds = xr.open_dataset(io.BytesIO(f.read()))
                                    all_years_datasets.append(ds)
                        os.remove(zip_file_path)
                        print(f"Deleted ZIP file: {zip_file_path}")
                    except Exception as e:
                        print(f"Failed to extract/process {zip_file_path}: {e}")
                        all_years_downloaded = False
                        if zip_file_path.exists():
                            os.remove(zip_file_path)
                            print(f"Deleted ZIP file due to processing error: {zip_file_path}")
                        continue

            # Post-process & combine all partial years
            if all_years_datasets and all_years_downloaded:
                try:
                    combined_ds = xr.concat(all_years_datasets, dim="time").drop_vars('crs', errors="ignore")
                    
                    # Unit conversions
                    if var in ["AGRO.TMIN", "AGRO.TEMP", "AGRO.TMAX"]:
                        combined_ds = combined_ds - 273.15  # Kelvin to Celsius
  
                    # Aggregate for cross-year seasons
                    combined_ds = self._aggregate_crossyear(
                        ds=combined_ds,
                        season_months=season_months,
                        var_name=var
                    )

                    ########## Revoir ceci surtout l'emplacement et calcul
                    if var == "AGRO.DSWR":
                        combined_ds = combined_ds / 86400  # J/m^2/day to W/m^2 

                    # Rename dimensions
                    if "lon" in combined_ds.dims:
                        combined_ds = combined_ds.rename({"lon": "X"})
                    if "lat" in combined_ds.dims:
                        combined_ds = combined_ds.rename({"lat": "Y"})
                    combined_ds = combined_ds.isel(Y=slice(None, None, -1))

                    # Adjust time coordinate
                    if len(seas)==1:
                        combined_ds["time"] = [f"{year}-{seas[0]}-01" for year in combined_ds["time"].astype(str).values]
                    elif len(seas) in [2,3]:
                        combined_ds["time"] = [f"{year}-{seas[1]}-01" for year in combined_ds["time"].astype(str).values]
                    elif len(seas) in [4,5]:
                        combined_ds["time"] = [f"{year}-{seas[3]}-01" for year in combined_ds["time"].astype(str).values]
                    else:
                        combined_ds["time"] = [f"{year}-{seas[4]}-01" for year in combined_ds["time"].astype(str).values]
                    combined_ds["time"] = combined_ds["time"].astype("datetime64[ns]")
                    combined_ds = combined_ds.rename({"time": "T"})

                    # Save to NetCDF
                    combined_ds.to_netcdf(output_path)
                    combined_ds.close()
                    print(f"Saved final dataset for {var} to: {output_path}")
                except Exception as e:
                    print(f"Failed to process or save combined dataset for {var}: {e}")
            else:
                print(f"No data downloaded for {var} in {season_str}.")
   
    # -------------------------------------------------------------------------
    # Helper for Reanalysis cross-year post-processing (optional)
    # -------------------------------------------------------------------------
    def _postprocess_reanalysis(self, ds, var_name):
        """
        Drop extra coords, rename dims, flip lat, etc.
        Adjust as needed for ERA5 quirks.
        """
        # Drop some known extraneous coords
        drop_list = []
        for extra in ["number", "expver", "pressure_level"]:
            if extra in ds.coords or extra in ds.variables:
                drop_list.append(extra)

        ds = ds.drop_vars(drop_list, errors="ignore").squeeze()

        # Flip latitude if it exists
        if "latitude" in ds.coords:
            ds = ds.isel(latitude=slice(None, None, -1))
            # rename directly to X, Y
            ds = ds.rename({"latitude": "Y", "longitude": "X"})

        # If "valid_time" is present, rename it to "time"
        if "valid_time" in ds.coords:
            ds = ds.assign_coords(valid_time=pd.to_datetime(ds.valid_time.values))
            ds = ds.rename({"valid_time": "time"})

        return ds

    def _postprocess_reanalysis_ersst(self, ds, var_name):       
        # Drop unnecessary variables
        # ds = ds.drop_vars('zlev').squeeze()
        # ds = ds.drop_vars('lev').squeeze()
        keep_vars = [var_name, 'T', 'X', 'Y']
        drop_vars = [v for v in ds.variables if v not in keep_vars]
        return ds.drop_vars(drop_vars, errors="ignore")

    def _aggregate_crossyear(self, ds, season_months, var_name):
        """
        Group ds by a custom 'season_year' coordinate so that all months
        in 'season_months' belong to one group that may cross Dec→Jan.
    
        Parameters:
            ds (xarray.Dataset or DataArray): The data to aggregate (daily, monthly, etc.).
            season_months (list[int]): e.g. [11,12,1] for NDJ.
            var_name (str): e.g. "AGRO.PRCP", "TEMP", "TMIN", etc. 
                           Used to decide 'mean' vs 'sum'.
    
        Returns:
            ds_out (xarray.Dataset or DataArray): Aggregated by season, 
                          dimension renamed from 'season_year' to 'time'.
        """

        if "time" not in ds.coords:
            raise ValueError("Dataset must have a 'time' dimension for aggregation.")
    
        pivot = season_months[0]
    
        # 1) Tag each time with the "season_year"
        # If month >= pivot => same year's label, else => year - 1
        season_year = ds["time"].dt.year.where(ds["time"].dt.month >= pivot,
                                               ds["time"].dt.year - 1)
    
        ds = ds.assign_coords(season_year=season_year)
        
        # 2) Keep only the months we actually want
        ds = ds.where(ds["time"].dt.month.isin(season_months), drop=True)
    
        # 3) Decide mean or sum based on var_name 

        if any(x in var_name for x in ["TEMP","TMIN","TMAX","SST","SLP","RUNOFF"]):
            ds_out = ds.groupby("season_year").mean("time")

        elif any(x in var_name for x in ["PRCP","DSWR","DLWR","NOLR"]):
            # For precipitation and radiation, we sum over time
            ds_out = ds.groupby("season_year").sum("time")
            # ds_out = ds.groupby("season_year").mean("time")
        else:
            ds_out = ds.groupby("season_year").mean("time")
        # 4) Rename "season_year" to "time", 
        #    so we end up with a time dimension (representing each seasonal year).
        ds_out = ds_out.rename({"season_year": "time"})
    
        return ds_out


    # def WAS_Download_Reanalysis(
    #         self,
    #         dir_to_save,
    #         center_variable,
    #         year_start,
    #         year_end,
    #         area,
    #         seas=["01", "02", "03"],  # e.g. NDJ = ["11","12","01"]
    #         force_download=False,
    #         run_avg=1
    #     ):
        
    #         """
    #         Download reanalysis data for specified center-variable combinations, years, and months,
    #         handling cross-year seasons (e.g., NDJ).
    #         """
    
    #         dir_to_save = Path(dir_to_save)
    #         dir_to_save.mkdir(parents=True, exist_ok=True)
    
    #         # Parse center and variable strings
    #         centers = [cv.split(".")[0] for cv in center_variable]
    #         vars_   = [cv.split(".")[1] for cv in center_variable]
    
    #         # Example reanalysis centers
    #         centre_dict = {"ERA5": "ERA5", "MERRA2": "MERRA2", "NOAA": "NOAA"}
    
    #         # Single-level monthly means
    #         variables_1 = {
    #             "PRCP": "total_precipitation",
    #             "TEMP": "2m_temperature",
    #             "TMAX": "maximum_2m_temperature_in_the_last_24_hours",
    #             "TMIN": "minimum_2m_temperature_in_the_last_24_hours",
    #             "UGRD10": "10m_u_component_of_wind",
    #             "VGRD10": "10m_v_component_of_wind",
    #             "SST": "sea_surface_temperature",
    #             "SLP": "mean_sea_level_pressure",
    #             "DSWR": "surface_solar_radiation_downwards",
    #             "DLWR": "surface_thermal_radiation_downwards",
    #             "NOLR": "top_net_thermal_radiation",
    #         }
            
    #         # Pressure-level monthly means
    #         variables_2 = {
    #             "HUSS_1000": "specific_humidity",
    #             "HUSS_925": "specific_humidity",
    #             "HUSS_850": "specific_humidity",
    #             "UGRD_1000": "u_component_of_wind",
    #             "UGRD_925": "u_component_of_wind",
    #             "UGRD_850": "u_component_of_wind",
    #             "UGRD_700": "u_component_of_wind",
    #             "UGRD_600": "u_component_of_wind",
    #             "UGRD_200": "u_component_of_wind",
    #             "VGRD_1000": "v_component_of_wind",
    #             "VGRD_925": "v_component_of_wind",
    #             "VGRD_850": "v_component_of_wind",
    #             "VGRD_700": "v_component_of_wind",
    #             "VGRD_600": "v_component_of_wind",
    #             "VGRD_200": "v_component_of_wind",
    #         }
            
    #         # Helper for zero-padded month strings
    #         def m2str(m: int):
    #             return f"{m:02d}"
    
    #         # Convert months to integers (e.g. ["11","12","01"] -> [11,12,1])
    #         season_months = [int(m) for m in seas]
    #         pivot = season_months[0]
    #         # For naming
    #         season_str = "".join([calendar.month_abbr[m] for m in season_months])

    #         now = datetime.datetime.now()
    #         curr_yr, curr_mon = now.year, now.month

    #         file_path = []
    #         for c, v in zip(centers, vars_):
                
    #             # =================================================================
    #             # Special Case: NOAA ERSST from NCEI (Direct Download) 
    #             # https://www.ncei.noaa.gov/data/sea-surface-temperature-extended-reconstructed/v6/access/
    #             # =================================================================

    #             if c == "NOAA" and v == "SST":
    #                 out_file = dir_to_save / f"{c}_{v}_{year_start}_{year_end}_{season_str}.nc"
    #                 if not force_download and out_file.exists():
    #                     print(f"{out_file} exists. Skipping.")
    #                     file_path.append(out_file)
    #                     continue
                    
    #                 print(f"Starting download for NOAA ERSST (v6) for {year_start}-{year_end}")
    
    #                 # Create a cache dir for individual monthly files so we don't redownload them unnecessarily
    #                 cache_dir = dir_to_save / "ersst_cache"
    #                 cache_dir.mkdir(exist_ok=True)
                    
    #                 # Identify exact list of year-months required for the seasonal request
    #                 # This ensures we handle cross-year (e.g. NDJ) correctly
    #                 required_files = []
                    
    #                 # Loop through the "Seasonal Years"
    #                 for s_year in range(year_start, year_end + 1):
                        
    #                     for month in season_months:
    #                         cal_year = s_year
    #                         # if month >= pivot:
    #                         #     cal_year = s_year
    #                         # else:
    #                         #     cal_year = s_year + 1

    #                         # if cal_year == year_end + 1 and month >= pivot:
    #                         #     continue
    #                         # if cal_year > year_end + 1:
    #                         #     continue
    #                         if (cal_year > curr_yr) or (cal_year == curr_yr and month > curr_mon):
    #                             continue
                                
    #                         # ERSST v5 filename format: ersst.v5.YYYYMM.nc
    #                         fname = f"ersst.v6.{cal_year}{month:02d}.nc"
    #                         required_files.append((cal_year, month, fname))
    
    #                 # Download 
    #                 # Base URL for ERSST 'v6'.
    #                 base_url = "https://www.ncei.noaa.gov/data/sea-surface-temperature-extended-reconstructed/v6/access/"
                    
    #                 downloaded_paths = []
                    
    #                 try:
    #                     for (yr, mn, fname) in required_files:
    #                         local_path = cache_dir / fname
    #                         if not local_path.exists():
    #                             # Construct full URL
    #                             url = f"{base_url}/{fname}"
    #                             print(f"Downloading {fname}...")
    #                             r = requests.get(url, timeout=30)
    #                             r.raise_for_status()
    #                             with open(local_path, 'wb') as f:
    #                                 f.write(r.content)
    #                         downloaded_paths.append(local_path)
                        
    #                     print("All files downloaded. merging and processing...")
    
    #                     # Open all files
    #                     # use_cftime=True is safer for long historical records
    #                     ds = xr.open_mfdataset(downloaded_paths)#, combine='by_coords', decode_times=True)
    #                     ds = ds[["sst"]].drop_vars("lev").squeeze()
   
    #                     # --- PRE-PROCESSING SPECIFIC TO ERSST ---
                        
    #                     # 1. Standardize Coordinates (Lat/Lon/Time)
    #                     # ERSST uses 'lat', 'lon', 'time' usually.
    #                     # ERSST Longitude is 0 to 360. We likely want -180 to 180 to match ERA5/bounding box.
    #                     if 'lon' in ds.coords:
    #                         ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
    #                         ds = ds.sortby('lon')
                        
    #                     # 2. Rename to internal standard (X, Y)
    #                     rename_dict = {}
    #                     if 'lat' in ds.coords: rename_dict['lat'] = 'Y'
    #                     if 'lon' in ds.coords: rename_dict['lon'] = 'X'
    #                     if 'sst' in ds.variables: rename_dict['sst'] = 'SST' # Rename var to capital SST for consistency
                        
    #                     ds = ds.rename(rename_dict)

    #                     # print(ds)
                        
    #                     # 3. Slice the Area
    #                     # Area format: [N, W, S, E]
    #                     lat_max, lon_min, lat_min, lon_max = area
                        
    #                     # Ensure X and Y are sorted for slicing
    #                     ds = ds.sortby(['X', 'Y'])
    #                     ds = ds.sel(X=slice(lon_min, lon_max), Y=slice(lat_min, lat_max))
    
    #                     # 5. Aggregate Cross-Year
    #                     # ERSST is in Celsius. Do NOT subtract 273.15 later.
    #                     # We pass 'SST' as var_name so aggregator calculates MEAN.
    #                     ds_agg = self._aggregate_crossyear(ds, season_months, "SST")

    
    #                     # 6. Final Format
    #                     # Rename 'time' to 'T' as per your output spec
    #                     ds_agg = ds_agg.rename({"time": "T"})
                        
    #                     # Ensure Variable name is 'sst' (lowercase) for final output if that is preferred
    #                     if 'SST' in ds_agg:
    #                         ds_agg = ds_agg.rename({'SST': 'sst'})
    #                     if len(seas)==1:
    #                         ds_agg["T"] = [f"{year}-{seas[0]}-01" for year in ds_agg["T"].astype(str).values]
    #                     elif len(seas) in [2,3]:
    #                         ds_agg["T"] = [f"{year}-{seas[1]}-01" for year in ds_agg["T"].astype(str).values]
    #                     elif len(seas) in [4,5]:
    #                         ds_agg["T"] = [f"{year}-{seas[2]}-01" for year in ds_agg["T"].astype(str).values]
    #                     else:
    #                         ds_agg["T"] = [f"{year}-{seas[3]}-01" for year in ds_agg["T"].astype(str).values]
                            
    #                     ds_agg["T"] = ds_agg["T"].astype("datetime64[ns]")
    
    #                     # 7. Save
    #                     ds_agg.to_netcdf(out_file)
    #                     file_path.append(out_file)
    #                     print(f"Saved NOAA ERSST data to {out_file}")
    
    #                 except Exception as e:
    #                     print(f"Failed to download or process NOAA/SST: {str(e)}")
    #                     # Optional: clean up partial downloads if needed, or leave for retry
    #                     import traceback
    #                     traceback.print_exc()
                    
    #                 continue 
    
    #             # =================================================================
    #             # ERA5 / Other Reanalysis Logic
    #             # =================================================================
    #             if c not in centre_dict:
    #                 print(f"Unknown center: {c}, skipping.")
    #                 continue
    
    #             rean = centre_dict[c]
    #             out_file = dir_to_save / f"{c}_{v}_{year_start}_{year_end}_{season_str}.nc"
    #             if (not force_download) and out_file.exists():
    #                 file_path.append(out_file)
    #                 print(f"{out_file} already exists. Skipping.")
    #                 continue
    
    #             # List to accumulate partial downloads
    #             combined_datasets = []
    
    #             # Iterate over each year in [year_start..year_end]
    #             for year in range(year_start, year_end + 1):
    #                 # Split months
    #                 base_months = [m for m in season_months if m >= pivot]
    #                 next_months = [m for m in season_months if m < pivot]
    
    #                 # (A) Base-year
    #                 if base_months:
    #                     base_str = [m2str(m) for m in base_months]
    #                     partA = dir_to_save / f"{c}_{v}_{year}_{season_str}_partA.nc"
    
    #                     # Decide dataset + request
    #                     if v in variables_2:
    #                         press_level = v.split("_")[1]  # e.g. 925 from "HUSS_925"
    #                         dataset = "reanalysis-era5-pressure-levels-monthly-means"
    #                         request = {
    #                             "product_type": ["monthly_averaged_reanalysis"],
    #                             "variable": variables_2[v],
    #                             "pressure_level": press_level,
    #                             "year": str(year),
    #                             "month": base_str,
    #                             "time": ["00:00"],
    #                             "area": area,
    #                             "data_format": "netcdf",
    #                         }
    #                     else:
    #                         dataset = "reanalysis-era5-single-levels-monthly-means"
    #                         request = {
    #                             "product_type": ["monthly_averaged_reanalysis"],
    #                             "variable": variables_1.get(v),
    #                             "year": str(year),
    #                             "month": base_str,
    #                             "time": ["00:00"],
    #                             "area": area,
    #                             "data_format": "netcdf",
    #                         }
    
    #                     # Download
    #                     try:
    #                         client = cdsapi.Client()
    #                         print(f"Downloading {c}/{v}: {year}, months={base_str}")
    #                         client.retrieve(dataset, request).download(str(partA))
                            
    #                         with xr.open_dataset(partA) as dsA:
    #                             dsA = dsA.load()
    #                             dsA = self._postprocess_reanalysis(dsA, v)
    #                             combined_datasets.append(dsA)
    #                         os.remove(partA)
    #                     except Exception as e:
    #                         print(f"Download/Process error for {c}/{v}, year={year} partA: {e}")
    #                         if os.path.exists(partA): os.remove(partA)
    #                         continue
    
    #                 # (B) Next-year
    #                 if next_months and (year < year_end+1):
    #                     year_next = year + 1
    #                     next_str = [m2str(m) for m in next_months]
    #                     partB = dir_to_save / f"{c}_{v}_{year}_{season_str}_partB_{year_next}.nc"
    
    #                     if v in variables_2:
    #                         press_level = v.split("_")[1]
    #                         dataset = "reanalysis-era5-pressure-levels-monthly-means"
    #                         request = {
    #                             "product_type": ["monthly_averaged_reanalysis"],
    #                             "variable": variables_2[v],
    #                             "pressure_level": press_level,
    #                             "year": str(year_next),
    #                             "month": next_str,
    #                             "time": ["00:00"],
    #                             "area": area,
    #                             "data_format": "netcdf",
    #                         }
    #                     else:
    #                         dataset = "reanalysis-era5-single-levels-monthly-means"
    #                         request = {
    #                             "product_type": ["monthly_averaged_reanalysis"],
    #                             "variable": variables_1.get(v),
    #                             "year": str(year_next),
    #                             "month": next_str,
    #                             "time": ["00:00"],
    #                             "area": area,
    #                             "data_format": "netcdf",
    #                         }
    
    #                     # Download
    #                     try:
    #                         client = cdsapi.Client()
    #                         print(f"Downloading {c}/{v}: {year_next}, months={next_str}")
    #                         client.retrieve(dataset, request).download(str(partB))
    
    #                         with xr.open_dataset(partB) as dsB:
    #                             dsB = dsB.load()
    #                             dsB = self._postprocess_reanalysis(dsB, v)
    #                             combined_datasets.append(dsB)
    #                         os.remove(partB)
    #                     except Exception as e:
    #                         print(f"Download/Process error for {c}/{v}, year={year_next} partB: {e}")
    #                         if os.path.exists(partB): os.remove(partB)
    #                         continue
    
    #             if combined_datasets:
    #                 dsC = xr.concat(combined_datasets, dim="time")
                    
    #                 # If T variable -> K to °C
    #                 # IMPORTANT: Only for ERA5/MERRA which are usually Kelvin. 
    #                 # NOAA ERSST is skipped above, so this block only runs for ERA5/MERRA.
    #                 if v in ["TMIN","TEMP","TMAX","SST"]:
    #                     dsC = dsC - 273.15
                    
    #                 # For precipitation or others, the aggregator decides sum vs mean
    #                 dsC = self._aggregate_crossyear(
    #                     ds=dsC,
    #                     season_months=season_months,
    #                     var_name=v
    #                 )
                    
    #                 if v == "PRCP":
    #                     # Convert to mm/month if ERA5 is in m/s or m/day
    #                     # Note: ERA5 monthly means are usually "m per day" or "m total". 
    #                     # Assuming ERA5 is meter -> multiply by 1000 for mm. 
    #                     # Your original code: dsC = 1000 * ds * 30. (Assuming ds variable name error in your snippet, likely meant dsC)
    #                     dsC = dsC * 1000 * 30 
                    
    #                 if v in ["DSWR", "DLWR","NOLR"]:
    #                     dsC = dsC/86400  # Convert to W/m2 if needed
    
    #                 if v == "SLP":
    #                     dsC = dsC / 100  # Convert to hPa(mb)
                       
    #                 if len(seas)==1:
    #                     dsC["time"] = [f"{year}-{seas[0]}-01" for year in dsC["time"].astype(str).values]
    #                 elif len(seas) in [2,3]:
    #                     dsC["time"] = [f"{year}-{seas[1]}-01" for year in dsC["time"].astype(str).values]
    #                 elif len(seas) in [4,5]:
    #                     dsC["time"] = [f"{year}-{seas[2]}-01" for year in dsC["time"].astype(str).values]
    #                 else:
    #                     dsC["time"] = [f"{year}-{seas[3]}-01" for year in dsC["time"].astype(str).values]
    #                 dsC["time"] = dsC["time"].astype("datetime64[ns]")
    #                 dsC = dsC.rename({"time": "T"})
                    
    #                 # Save final
    #                 dsC.to_netcdf(out_file)
    #                 file_path.append(out_file)
    #                 print(f"Saved final reanalysis file: {out_file}")
    #             else:
    #                 print(f"No data found for {c}/{v}.")
                
    #         return file_path


    

    def WAS_Download_Reanalysis(
        self,
        dir_to_save,
        center_variable,
        year_start,
        year_end,
        area,
        seas=("01", "02", "03"),
        force_download=False,
        max_retries=3,
        retry_delay=5,
    ):
        """
        Download and aggregate monthly reanalysis predictors/predictands.
    
        Supported inputs
        ----------------
        ERA5 monthly means from CDS:
            "ERA5.PRCP", "ERA5.TEMP", "ERA5.SST", "ERA5.SLP",
            "ERA5.DSWR", "ERA5.DLWR", "ERA5.NOLR",
            "ERA5.UGRD_850", "ERA5.VGRD_200", etc.
    
        NOAA ERSST v6 direct monthly SST:
            "NOAA.SST"
    
        Output convention
        -----------------
        NetCDF with dimensions:
            T, Y, X
    
        Unit conventions
        ----------------
        ERA5 monthly averaged reanalysis:
            PRCP/RUNOFF are m/day, so monthly total is value * days * 1000.
            Radiation accumulations are J m-2/day, so seasonal mean flux is
            day-weighted sum(value) / (total_days * 86400).
            Other monthly means are day-weighted seasonal means.
        """
        from pathlib import Path
        import calendar
        import datetime
        import gc
        import os
        import time as _time
    
        import cdsapi
        import numpy as np
        import pandas as pd
        import requests
        import xarray as xr
    
        if isinstance(center_variable, str):
            raise TypeError(
                "center_variable must be a list, for example "
                "['ERA5.PRCP'] or ['ERA5.PRCP', 'NOAA.SST']."
            )
    
        center_variable = list(center_variable)
        season_months = [int(m) for m in seas]
        season_codes = [f"{m:02d}" for m in season_months]
        season_str = "".join(calendar.month_abbr[m] for m in season_months)
        pivot_month = season_months[0]
    
        dir_to_save = Path(dir_to_save)
        dir_to_save.mkdir(parents=True, exist_ok=True)
    
        era5_single_level = {
            "PRCP": "total_precipitation",
            "RUNOFF": "runoff",
            "TEMP": "2m_temperature",
            "TDEW": "2m_dewpoint_temperature",
            "TMAX": "maximum_2m_temperature_since_previous_post_processing",
            "TMIN": "minimum_2m_temperature_since_previous_post_processing",
            "UGRD10": "10m_u_component_of_wind",
            "VGRD10": "10m_v_component_of_wind",
            "SST": "sea_surface_temperature",
            "SLP": "mean_sea_level_pressure",
            "DSWR": "surface_solar_radiation_downwards",
            "DLWR": "surface_thermal_radiation_downwards",
            "NOLR": "top_net_thermal_radiation",
        }
    
        era5_pressure_level = {
            "HUSS_1000": "specific_humidity",
            "HUSS_925": "specific_humidity",
            "HUSS_850": "specific_humidity",
            "UGRD_1000": "u_component_of_wind",
            "UGRD_925": "u_component_of_wind",
            "UGRD_850": "u_component_of_wind",
            "UGRD_700": "u_component_of_wind",
            "UGRD_600": "u_component_of_wind",
            "UGRD_200": "u_component_of_wind",
            "VGRD_1000": "v_component_of_wind",
            "VGRD_925": "v_component_of_wind",
            "VGRD_850": "v_component_of_wind",
            "VGRD_700": "v_component_of_wind",
            "VGRD_600": "v_component_of_wind",
            "VGRD_200": "v_component_of_wind",
        }
    
        total_vars = {"PRCP", "RUNOFF"}
        flux_vars = {"DSWR", "DLWR", "NOLR"}
        kelvin_vars = {"TEMP", "TDEW", "TMAX", "TMIN", "SST"}
        pressure_vars = {"SLP"}
    
        def _season_calendar_months(season_year):
            months = []
            for month in season_months:
                cal_year = season_year if month >= pivot_month else season_year + 1
                months.append((season_year, cal_year, month))
            return months
    
        def _representative_time(season_year):
            mid_month = season_months[len(season_months) // 2]
            cal_year = season_year if mid_month >= pivot_month else season_year + 1
            return pd.Timestamp(cal_year, mid_month, 1)
    
        def _download_with_retries(client, dataset, request, target_file, label):
            for attempt in range(1, max_retries + 1):
                try:
                    print(f"Attempt {attempt}/{max_retries}: downloading {label}...")
                    client.retrieve(dataset, request).download(str(target_file))
                    print(f"Downloaded: {target_file}")
                    return True
                except Exception as exc:
                    print(f"Attempt {attempt}/{max_retries} failed for {label}: {exc}")
                    if target_file.exists():
                        os.remove(target_file)
                        print(f"Deleted incomplete file: {target_file}")
                    if attempt < max_retries:
                        print(f"Retrying after {retry_delay} seconds...")
                        _time.sleep(retry_delay)
            return False
    
        def _normalize_coords(ds):
            rename_map = {}
            if "valid_time" in ds.dims or "valid_time" in ds.coords:
                rename_map["valid_time"] = "time"
            if "longitude" in ds.dims or "longitude" in ds.coords:
                rename_map["longitude"] = "X"
            if "latitude" in ds.dims or "latitude" in ds.coords:
                rename_map["latitude"] = "Y"
            if "lon" in ds.dims or "lon" in ds.coords:
                rename_map["lon"] = "X"
            if "lat" in ds.dims or "lat" in ds.coords:
                rename_map["lat"] = "Y"
    
            ds = ds.rename({old: new for old, new in rename_map.items() if old in ds})
    
            if "expver" in ds.dims and ds.sizes["expver"] == 1:
                ds = ds.isel(expver=0, drop=True)
    
            for coord in ["pressure_level", "level", "isobaricInhPa"]:
                if coord in ds.dims and ds.sizes[coord] == 1:
                    ds = ds.isel({coord: 0}, drop=True)
    
            if "X" in ds.coords and float(ds["X"].max()) > 180:
                ds = ds.assign_coords(X=(((ds["X"] + 180) % 360) - 180)).sortby("X")
    
            if "Y" in ds.coords:
                ds = ds.sortby("Y")
    
            return ds
    
        def _subset_area(ds):
            if "X" not in ds.coords or "Y" not in ds.coords:
                return ds
            north, west, south, east = area
            ds = ds.sortby(["X", "Y"])
            return ds.sel(X=slice(west, east), Y=slice(south, north))
    
        def _rename_single_data_var(ds, var_code):
            data_vars = list(ds.data_vars)
            if len(data_vars) == 1 and data_vars[0] != var_code:
                ds = ds.rename({data_vars[0]: var_code})
            return ds
    
        def _convert_era5_monthly(ds, var_code, days_in_month):
            ds = _rename_single_data_var(ds, var_code)
    
            if var_code in total_vars:
                ds = ds * (1000.0 * days_in_month)
                units = "mm/month"
    
            elif var_code in flux_vars:
                ds = ds / 86400.0
                units = "W m-2"
    
            elif var_code in kelvin_vars:
                ds = ds - 273.15
                units = "degC"
    
            elif var_code in pressure_vars:
                ds = ds / 100.0
                units = "hPa"
    
            else:
                units = None
    
            for name in ds.data_vars:
                if units is not None:
                    ds[name].attrs["units"] = units
                ds[name].attrs["monthly_days"] = int(days_in_month)
    
            ds.attrs["monthly_days"] = int(days_in_month)
            return ds
    
        def _aggregate_months(monthly_datasets, var_code):
            ds = xr.concat(monthly_datasets, dim="month_index")
            monthly_days = [
                int(item.attrs.get("monthly_days", next(iter(item.data_vars.values())).attrs["monthly_days"]))
                for item in monthly_datasets
            ]
            weights = xr.DataArray(
                monthly_days,
                dims="month_index",
            )
    
            if var_code in total_vars:
                out = ds.sum("month_index", keep_attrs=True)
                units = "mm"
    
            elif var_code in flux_vars:
                out = ds.weighted(weights).mean("month_index", keep_attrs=True)
                units = "W m-2"
    
            else:
                out = ds.weighted(weights).mean("month_index", keep_attrs=True)
                units = None
    
            for name in out.data_vars:
                if units is not None:
                    out[name].attrs["units"] = units
                out[name].attrs.pop("monthly_days", None)
    
            return out
    
        def _download_era5_month(client, c, v, season_year, cal_year, month):
            if v in era5_pressure_level:
                dataset = "reanalysis-era5-pressure-levels-monthly-means"
                pressure_level = v.rsplit("_", 1)[1]
                request = {
                    "product_type": ["monthly_averaged_reanalysis"],
                    "variable": [era5_pressure_level[v]],
                    "pressure_level": [pressure_level],
                    "year": str(cal_year),
                    "month": f"{month:02d}",
                    "time": ["00:00"],
                    "area": area,
                    "data_format": "netcdf",
                }
            elif v in era5_single_level:
                dataset = "reanalysis-era5-single-levels-monthly-means"
                request = {
                    "product_type": ["monthly_averaged_reanalysis"],
                    "variable": [era5_single_level[v]],
                    "year": str(cal_year),
                    "month": f"{month:02d}",
                    "time": ["00:00"],
                    "area": area,
                    "data_format": "netcdf",
                }
            else:
                raise ValueError(f"Unknown ERA5 variable code: {v}")
    
            tmp = dir_to_save / f"tmp_{c}_{v}_{season_year}_{cal_year}{month:02d}.nc"
            label = f"{c}.{v} {cal_year}-{month:02d}"
            ok = _download_with_retries(client, dataset, request, tmp, label)
            if not ok:
                return None
    
            try:
                with xr.open_dataset(tmp) as ds:
                    ds = _normalize_coords(ds).load()
                days = calendar.monthrange(cal_year, month)[1]
                ds = _convert_era5_monthly(ds, v, days)
                return ds
            finally:
                if tmp.exists():
                    os.remove(tmp)
                    print(f"Deleted temporary file: {tmp}")
    
        def _download_noaa_ersst(v, out_file):
            if v != "SST":
                print(f"NOAA direct download currently supports SST only. Skipping NOAA.{v}.")
                return None
    
            cache_dir = dir_to_save / "ersst_cache"
            cache_dir.mkdir(exist_ok=True)
            base_url = "https://www.ncei.noaa.gov/data/sea-surface-temperature-extended-reconstructed/v6/access"
    
            now = datetime.datetime.now()
            required = []
            for season_year in range(year_start, year_end + 1):
                for _, cal_year, month in _season_calendar_months(season_year):
                    if (cal_year > now.year) or (cal_year == now.year and month > now.month):
                        continue
                    required.append((cal_year, month, f"ersst.v6.{cal_year}{month:02d}.nc"))
    
            paths = []
            for cal_year, month, fname in sorted(set(required)):
                local_path = cache_dir / fname
                if not local_path.exists() or force_download:
                    url = f"{base_url}/{fname}"
                    print(f"Downloading NOAA ERSST {cal_year}-{month:02d}...")
                    response = requests.get(url, timeout=60)
                    response.raise_for_status()
                    with open(local_path, "wb") as handle:
                        handle.write(response.content)
                paths.append(local_path)
    
            ds = xr.open_mfdataset(paths, combine="by_coords")
            if "sst" not in ds:
                raise ValueError("NOAA ERSST file does not contain variable 'sst'.")
    
            ds = ds[["sst"]]
            ds = _normalize_coords(ds)
            ds = _subset_area(ds)
            ds = ds.rename({"sst": "SST"})
            ds = ds.drop_vars("lev", errors="ignore")
            if "lev" in ds.dims:
                ds = ds.squeeze("lev", drop=True)

            seasonal, incomplete = [], []
            try:
                for season_year in range(year_start, year_end + 1):
                    required_months = _season_calendar_months(season_year)
                    indices = _noaa_exact_month_indices(
                        ds["time"].values, [(y, m) for _, y, m in required_months])
                    if indices is None:
                        incomplete.append(season_year)
                        print(f"NOAA SST: incomplete season {season_year}; skipped (missing calendar month).")
                        continue
                    monthly = []
                    for index, (_, cal_year, month) in zip(indices, required_months):
                        month_ds = ds.isel(time=int(index), drop=True).load()
                        days = calendar.monthrange(cal_year, month)[1]
                        month_ds["SST"].attrs["monthly_days"] = int(days)
                        month_ds.attrs["monthly_days"] = int(days)
                        monthly.append(month_ds)
                    out = _aggregate_months(monthly, "SST")
                    out = out.expand_dims(time=[_representative_time(season_year)])
                    seasonal.append(out)
            finally:
                ds.close()
            if not seasonal:
                raise ValueError("NOAA SST: no complete requested season; missing months are never substituted.")
            out = xr.concat(seasonal, dim="time")
            out.attrs["incomplete_seasons_skipped"] = ",".join(map(str, incomplete))
            out = out.rename({"time": "T"})
            out = _finalize_output(out, "NOAA", v)
            out.to_netcdf(out_file)
            return out_file
    
        def _finalize_output(ds, center, var_code):
            if "time" in ds.dims or "time" in ds.coords:
                ds = ds.rename({"time": "T"})
    
            preferred = ["T", "Y", "X"]
            for name in list(ds.data_vars):
                dims = [dim for dim in preferred if dim in ds[name].dims]
                dims += [dim for dim in ds[name].dims if dim not in dims]
                ds[name] = ds[name].transpose(*dims)
    
            ds.attrs["center"] = center
            ds.attrs["variable_code"] = var_code
            ds.attrs["season"] = season_str
            ds.attrs["season_months"] = ",".join(season_codes)
            return ds
    
        file_path = []
        client = cdsapi.Client()
    
        for cv in center_variable:
            try:
                c, v = cv.split(".", 1)
                out_file = dir_to_save / f"{c}_{v}_{year_start}_{year_end}_{season_str}.nc"
    
                if out_file.exists() and not force_download:
                    print(f"{out_file} already exists. Skipping.")
                    file_path.append(out_file)
                    continue
    
                if c == "NOAA":
                    result = _download_noaa_ersst(v, out_file)
                    if result is not None:
                        file_path.append(result)
                        print(f"Saved NOAA.{v} seasonal file: {out_file}")
                    continue
    
                if c != "ERA5":
                    print(f"Unsupported center in this corrected function: {c}. Skipping.")
                    continue
    
                seasonal_outputs = []
                all_ok = True
                for season_year in range(year_start, year_end + 1):
                    monthly_datasets = []
                    for _, cal_year, month in _season_calendar_months(season_year):
                        ds_month = _download_era5_month(client, c, v, season_year, cal_year, month)
                        if ds_month is None:
                            all_ok = False
                            continue
                        monthly_datasets.append(ds_month)
    
                    if len(monthly_datasets) != len(season_months):
                        print(f"Incomplete season {season_year} for {c}.{v}.")
                        all_ok = False
                        continue
    
                    ds_season = _aggregate_months(monthly_datasets, v)
                    ds_season = ds_season.expand_dims(time=[_representative_time(season_year)])
                    seasonal_outputs.append(ds_season)
    
                if not seasonal_outputs:
                    print(f"No complete data found for {c}.{v}.")
                    continue
    
                if not all_ok:
                    print(f"Skipping save for {c}.{v} because some months/seasons failed.")
                    continue
    
                ds_out = xr.concat(seasonal_outputs, dim="time").sortby("time")
                ds_out = _finalize_output(ds_out, c, v)
    
                encoding = {name: {"zlib": True, "complevel": 4} for name in ds_out.data_vars}
                ds_out.to_netcdf(out_file, encoding=encoding)
                ds_out.close()
                file_path.append(out_file)
                print(f"Saved final reanalysis file: {out_file}")
    
                del seasonal_outputs, ds_out
                gc.collect()
    
            except Exception as exc:
                print(f"Failed to process {cv}: {exc}")
    
        return file_path



    # def WAS_Download_ERA5Land(
    #     self,
    #     dir_to_save,
    #     center_variable,
    #     year_start,
    #     year_end,
    #     area,
    #     seas=["01", "02", "03"],  # e.g. NDJ = ["11","12","01"]
    #     force_download=False,
    # ):
    #     """
    #     Download ERA5Land reanalysis data for specified variable combinations, years, and months,
    #     handling cross-year seasons (e.g., NDJ).
    #     Parameters:
    #         dir_to_save (str): Directory to save the downloaded files.
    #         center_variable (list): List of center-variable identifiers (e.g., ["ERA5Land.PRCP", "ERA5Land.TEMP"]).
    #         year_start (int): Start year for the data.
    #         year_end (int): End year for the data.
    #         area (list): Bounding box as [North, West, South, East].
    #         seas (list): List of months (e.g., ["11","12","01"] for NDJ).
    #         force_download (bool): If True, forces download even if file exists.
    #     """
    #     dir_to_save = Path(dir_to_save)
    #     dir_to_save.mkdir(parents=True, exist_ok=True)
    #     # Parse center and variable strings
    #     centers = [cv.split(".")[0] for cv in center_variable]
    #     vars_ = [cv.split(".")[1] for cv in center_variable]
    #     # ERA5-Land does not have pressure levels, only single-level variables
    #     variables_1 = {
    #         "PRCP": "total_precipitation",
    #         "TEMP": "2m_temperature",
    #         "TDEW": "2m_dewpoint_temperature",
    #         "UGRD10": "10m_u_component_of_wind",
    #         "VGRD10": "10m_v_component_of_wind",
    #         "DSWR": "surface_solar_radiation_downwards",
    #         "DLWR": "surface_thermal_radiation_downwards",
    #         "NOLR": "surface_net_thermal_radiation_downwards",  # Adjusted for ERA5-Land
    #         "RUNOFF": "surface_runoff",
    #         "SOILWATER1": "volumetric_soil_water_layer_1",
    #         "SOILWATER2": "volumetric_soil_water_layer_2",
    #         "SOILWATER3": "volumetric_soil_water_layer_3",            
    #     }
    #     # Helper for zero-padded month strings
    #     def m2str(m: int):
    #         return f"{m:02d}"
    #     # Convert months to integers
    #     season_months = [int(m) for m in seas]
    #     pivot = season_months[0]
    #     # For naming
    #     season_str = "".join([calendar.month_abbr[m] for m in season_months])
    #     for c, v in zip(centers, vars_):
    #         if c != "ERA5Land":
    #             print(f"This function is for ERA5Land only. Skipping {c}.")
    #             continue
    #         if v not in variables_1:
    #             print(f"Unknown variable for ERA5Land: {v}. Skipping.")
    #             continue
    #         cds_var = variables_1[v]
    #         # if cds_var == "surface_runoff":
    #         #     pivot_int = pivot
    #         #     previous = pivot_int - 1
    #         #     previous = str(previous).zfill(2) if previous > 0 else 12
    #         #     season_months = [previous] + season_months
    #         #     pivot = previous
            
    #         out_file = dir_to_save / f"{c}_{v}_{year_start}_{year_end}_{season_str}.nc"
    #         if not force_download and out_file.exists():
    #             print(f"{out_file} already exists. Skipping.")
    #             continue
    #         combined_datasets = []
    #         for year in range(year_start, year_end + 1):
    #             base_months = [m for m in season_months if m >= pivot]
    #             next_months = [m for m in season_months if m < pivot]
    #             # (A) Base-year
    #             if base_months:
    #                 base_str = [m2str(m) for m in base_months]
    #                 partA = dir_to_save / f"{c}_{v}_{year}_{season_str}_partA.nc"
    #                 dataset = "reanalysis-era5-land-monthly-means"
    #                 request = {
    #                     "product_type": "monthly_averaged_reanalysis",
    #                     "variable": cds_var,
    #                     "year": str(year),
    #                     "month": base_str,
    #                     "time": "00:00",
    #                     "area": area,
    #                     "data_format": "netcdf",
    #                 }
    #                 try:
    #                     client = cdsapi.Client()
    #                     print(f"Downloading {c}/{v}: {year}, months={base_str}")
    #                     client.retrieve(dataset, request).download(str(partA))
    #                 except Exception as e:
    #                     print(f"Download error for {c}/{v}, year={year} partA: {e}")
    #                     continue
    #                 with xr.open_dataset(partA) as dsA:
    #                     dsA = dsA.load()
    #                     dsA = self._postprocess_reanalysis(dsA, v)
    #                     combined_datasets.append(dsA)
    #                 os.remove(partA)
    #             # (B) Next-year
    #             if next_months and (year < year_end + 1):
    #                 year_next = year + 1
    #                 next_str = [m2str(m) for m in next_months]
    #                 partB = dir_to_save / f"{c}_{v}_{year}_{season_str}_partB_{year_next}.nc"
    #                 request["year"] = str(year_next)
    #                 request["month"] = next_str
    #                 try:
    #                     client = cdsapi.Client()
    #                     print(f"Downloading {c}/{v}: {year_next}, months={next_str}")
    #                     client.retrieve(dataset, request).download(str(partB))
    #                 except Exception as e:
    #                     print(f"Download error for {c}/{v}, year={year_next} partB: {e}")
    #                     continue
    #                 with xr.open_dataset(partB) as dsB:
    #                     dsB = dsB.load()
    #                     dsB = self._postprocess_reanalysis(dsB, v)
    #                     combined_datasets.append(dsB)
    #                 os.remove(partB)
    #         if combined_datasets:
    #             dsC = xr.concat(combined_datasets, dim="time")
    #             # Unit conversions
    #             if v in ["TEMP", "TDEW"]:
    #                 dsC = dsC - 273.15
    #             # Aggregate
    #             dsC = self._aggregate_crossyear(
    #                 ds=dsC,
    #                 season_months=season_months,
    #                 var_name=v
    #             )
    #             if v == "PRCP":
    #                 dsC = 1000 * dsC * 30  # Convert to mm 
    #             if v == "RUNOFF":
    #                 #dsC = dsC * len(season_months) * 30 # Convert to mm (approximate, as in existing code)
    #                 lat = dsC.Y
    #                 lon = dsC.X
    #                 dlon = np.deg2rad(0.1)
    #                 dlat = np.deg2rad(0.1)
    #                 r = 6371000 # Earth radius in meters
    #                 # Area for each grid cell
    #                 area_ = (r ** 2) * dlon * np.cos(np.deg2rad(lat)) * dlat
    #                 # Perform the conversion to m^3/s
    #                 dsC = (dsC * area_) / 86400
                    
    #             if v in ["DSWR", "DLWR", "NOLR"]:
    #                 # nbjour = len(season_months) * 30
    #                 dsC = dsC / 86400  # Convert to W/m2 (approximate)
    #             dsC["time"] = [f"{year}-{seas[1]}-01" for year in dsC["time"].astype(str).values]
    #             dsC["time"] = dsC["time"].astype("datetime64[ns]")
    #             dsC = dsC.rename({"time": "T"})
    #             # Save final
    #             dsC.to_netcdf(out_file)
    #             print(f"Saved final ERA5-Land file: {out_file}")
    #         else:
    #             print(f"No data found for {c}/{v}.")

    def WAS_Download_ERA5Land(
        self,
        dir_to_save,
        center_variable,
        year_start,
        year_end,
        area,
        seas=("01", "02", "03"),
        force_download=False,
        runoff_units="mm",
        max_retries=3,
        retry_delay=5,
    ):
        """
        Download ERA5-Land monthly averaged data and aggregate it by season.
    
        Output convention:
            T, Y, X
    
        ERA5-Land monthly averaged accumulation convention:
            PRCP/RUNOFF are in m/day. Monthly totals are:
                value * number_of_days * 1000
    
            Radiation accumulations are daily mean accumulations in J m-2/day.
            Mean fluxes are:
                value / 86400
    
        Cross-year seasons are handled using the first season month as pivot.
        Example:
            seas=("11", "12", "01") for NDJ gives Nov-Dec of year Y and Jan of Y+1.
        """
        from pathlib import Path
        import calendar
        import gc
        import os
        import time as _time
    
        import cdsapi
        import numpy as np
        import pandas as pd
        import xarray as xr
    
        if isinstance(center_variable, str):
            raise TypeError(
                "center_variable must be a list, for example "
                "['ERA5Land.PRCP'] or ['ERA5Land.PRCP', 'ERA5Land.TEMP']."
            )
        center_variable = list(center_variable)
    
        if runoff_units not in ["mm", "m3/s"]:
            raise ValueError("runoff_units must be 'mm' or 'm3/s'.")
    
        dir_to_save = Path(dir_to_save)
        dir_to_save.mkdir(parents=True, exist_ok=True)
    
        variables_1 = {
            "PRCP": "total_precipitation",
            "RUNOFF": "runoff",
            "TEMP": "2m_temperature",
            "TDEW": "2m_dewpoint_temperature",
            "UGRD10": "10m_u_component_of_wind",
            "VGRD10": "10m_v_component_of_wind",
            "DSWR": "surface_solar_radiation_downwards",
            "DLWR": "surface_thermal_radiation_downwards",
            "NOLR": "surface_net_thermal_radiation",
            "SOILWATER1": "volumetric_soil_water_layer_1",
            "SOILWATER2": "volumetric_soil_water_layer_2",
            "SOILWATER3": "volumetric_soil_water_layer_3",
            "SOILWATER4": "volumetric_soil_water_layer_4",
        }
    
        total_vars = {"PRCP", "SRUNOFF"}
        flux_vars = {"DSWR", "DLWR", "NOLR"}
        kelvin_vars = {"TEMP", "TDEW"}
    
        season_months = [int(m) for m in seas]
        season_codes = [f"{m:02d}" for m in season_months]
        pivot_month = season_months[0]
        season_str = "".join(calendar.month_abbr[m] for m in season_months)
    
        dataset = "reanalysis-era5-land-monthly-means"
        client = cdsapi.Client()
    
        def _season_calendar_months(season_year):
            months = []
            for month in season_months:
                cal_year = season_year if month >= pivot_month else season_year + 1
                months.append((cal_year, month))
            return months
    
        def _representative_time(season_year):
            mid_month = season_months[len(season_months) // 2]
            cal_year = season_year if mid_month >= pivot_month else season_year + 1
            return pd.Timestamp(cal_year, mid_month, 1)
    
        def _download_with_retries(request, target_file, label):
            for attempt in range(1, max_retries + 1):
                try:
                    print(f"Attempt {attempt}/{max_retries}: downloading {label}...")
                    client.retrieve(dataset, request).download(str(target_file))
                    print(f"Downloaded: {target_file}")
                    return True
                except Exception as exc:
                    print(f"Attempt {attempt}/{max_retries} failed for {label}: {exc}")
                    if target_file.exists():
                        os.remove(target_file)
                        print(f"Deleted incomplete file: {target_file}")
                    if attempt < max_retries:
                        print(f"Retrying after {retry_delay} seconds...")
                        _time.sleep(retry_delay)
            return False
    
        def _normalize_coords(ds):
            rename_map = {}
            if "valid_time" in ds.dims or "valid_time" in ds.coords:
                rename_map["valid_time"] = "time"
            if "longitude" in ds.dims or "longitude" in ds.coords:
                rename_map["longitude"] = "X"
            if "latitude" in ds.dims or "latitude" in ds.coords:
                rename_map["latitude"] = "Y"
    
            ds = ds.rename({old: new for old, new in rename_map.items() if old in ds})
    
            if "expver" in ds.dims and ds.sizes["expver"] == 1:
                ds = ds.isel(expver=0, drop=True)
    
            if "Y" in ds.coords:
                ds = ds.sortby("Y")
    
            return ds
    
        def _rename_single_data_var(ds, var_code):
            data_vars = list(ds.data_vars)
            if len(data_vars) == 1 and data_vars[0] != var_code:
                ds = ds.rename({data_vars[0]: var_code})
            return ds
    
        def _grid_cell_area_m2(ds):
            if "Y" not in ds.coords or "X" not in ds.coords:
                raise ValueError("SRUNOFF conversion to m3/s needs X and Y coordinates.")
            if ds.sizes.get("Y", 0) < 2 or ds.sizes.get("X", 0) < 2:
                raise ValueError("SRUNOFF conversion to m3/s needs at least two X and Y points.")
    
            radius = 6371000.0
            dlat = np.deg2rad(float(abs(ds["Y"].diff("Y").median())))
            dlon = np.deg2rad(float(abs(ds["X"].diff("X").median())))
            area_y = (radius ** 2) * dlat * dlon * np.cos(np.deg2rad(ds["Y"]))
            area_y.attrs["units"] = "m2"
            return area_y
    
        def _convert_month(ds, var_code, days_in_month):
            ds = _rename_single_data_var(ds, var_code)
    
            if var_code == "PRCP":
                ds = ds * (1000.0 * days_in_month)
                units = "mm/month"
    
            elif var_code in ["SRUNOFF", "RUNOFF"]:
                if runoff_units == "mm":
                    ds = ds * (1000.0 * days_in_month)
                    units = "mm/month"
                else:
                    # m/day * days * area gives monthly volume; seasonal aggregation
                    # divides by total seconds later.
                    ds = ds * days_in_month * _grid_cell_area_m2(ds)
                    units = "m3/month"
    
            elif var_code in flux_vars:
                ds = ds / 86400.0
                units = "W m-2"
    
            elif var_code in kelvin_vars:
                ds = ds - 273.15
                units = "degC"
    
            else:
                units = None
    
            ds.attrs["monthly_days"] = int(days_in_month)
            for name in ds.data_vars:
                if units is not None:
                    ds[name].attrs["units"] = units
                ds[name].attrs["monthly_days"] = int(days_in_month)
    
            return ds
    
        def _aggregate_months(monthly_datasets, var_code):
            ds = xr.concat(monthly_datasets, dim="month_index")
            monthly_days = [
                int(item.attrs.get("monthly_days", next(iter(item.data_vars.values())).attrs["monthly_days"]))
                for item in monthly_datasets
            ]
            weights = xr.DataArray(monthly_days, dims="month_index")
            total_days = float(sum(monthly_days))
    
            if var_code == "PRCP":
                out = ds.sum("month_index", keep_attrs=True)
                units = "mm"
    
            elif var_code == "SRUNOFF":
                if runoff_units == "mm":
                    out = ds.sum("month_index", keep_attrs=True)
                    units = "mm"
                else:
                    out = ds.sum("month_index", keep_attrs=True) / (total_days * 86400.0)
                    units = "m3 s-1"
    
            elif var_code in flux_vars:
                out = ds.weighted(weights).mean("month_index", keep_attrs=True)
                units = "W m-2"
    
            else:
                out = ds.weighted(weights).mean("month_index", keep_attrs=True)
                units = None
    
            for name in out.data_vars:
                if units is not None:
                    out[name].attrs["units"] = units
                out[name].attrs.pop("monthly_days", None)
    
            return out
    
        def _download_month(var_code, cds_var, season_year, cal_year, month):
            tmp = dir_to_save / f"tmp_ERA5Land_{var_code}_{season_year}_{cal_year}{month:02d}.nc"
            request = {
                "product_type": ["monthly_averaged_reanalysis"],
                "variable": [cds_var],
                "year": str(cal_year),
                "month": f"{month:02d}",
                "time": ["00:00"],
                "area": area,
                "data_format": "netcdf",
            }
    
            label = f"ERA5Land.{var_code} {cal_year}-{month:02d}"
            if not _download_with_retries(request, tmp, label):
                return None
    
            try:
                with xr.open_dataset(tmp) as ds:
                    ds = _normalize_coords(ds).load()
                days = calendar.monthrange(cal_year, month)[1]
                return _convert_month(ds, var_code, days)
            finally:
                if tmp.exists():
                    os.remove(tmp)
                    print(f"Deleted temporary file: {tmp}")
    
        def _finalize_output(ds, var_code):
            if "time" in ds.dims or "time" in ds.coords:
                ds = ds.rename({"time": "T"})
    
            preferred = ["T", "Y", "X"]
            for name in list(ds.data_vars):
                dims = [dim for dim in preferred if dim in ds[name].dims]
                dims += [dim for dim in ds[name].dims if dim not in dims]
                ds[name] = ds[name].transpose(*dims)
    
            ds.attrs["center"] = "ERA5Land"
            ds.attrs["variable_code"] = var_code
            ds.attrs["season"] = season_str
            ds.attrs["season_months"] = ",".join(season_codes)
            if var_code == "SRUNOFF":
                ds.attrs["runoff_units"] = runoff_units
            return ds
    
        file_path = []
    
        for cv in center_variable:
            try:
                center, var_code = cv.split(".", 1)
                if center != "ERA5Land":
                    print(f"This function is for ERA5Land only. Skipping {center}.")
                    continue
                if var_code not in variables_1:
                    print(f"Unknown variable for ERA5Land: {var_code}. Skipping.")
                    continue
    
                unit_tag = "_m3s" if var_code == "SRUNOFF" and runoff_units == "m3/s" else ""
                out_file = dir_to_save / f"ERA5Land_{var_code}_{year_start}_{year_end}_{season_str}{unit_tag}.nc"
                if out_file.exists() and not force_download:
                    print(f"{out_file} already exists. Skipping.")
                    file_path.append(out_file)
                    continue
    
                seasonal_outputs = []
                all_ok = True
                cds_var = variables_1[var_code]
    
                for season_year in range(year_start, year_end + 1):
                    monthly_datasets = []
                    for cal_year, month in _season_calendar_months(season_year):
                        ds_month = _download_month(var_code, cds_var, season_year, cal_year, month)
                        if ds_month is None:
                            all_ok = False
                            continue
                        monthly_datasets.append(ds_month)
    
                    if len(monthly_datasets) != len(season_months):
                        print(f"Incomplete season {season_year} for ERA5Land.{var_code}.")
                        all_ok = False
                        continue
    
                    ds_season = _aggregate_months(monthly_datasets, var_code)
                    ds_season = ds_season.expand_dims(time=[_representative_time(season_year)])
                    seasonal_outputs.append(ds_season)
    
                if not seasonal_outputs:
                    print(f"No complete data found for ERA5Land.{var_code}.")
                    continue
    
                if not all_ok:
                    print(f"Skipping save for ERA5Land.{var_code} because some months/seasons failed.")
                    continue
    
                ds_out = xr.concat(seasonal_outputs, dim="time").sortby("time")
                ds_out = _finalize_output(ds_out, var_code)
    
                encoding = {name: {"zlib": True, "complevel": 4} for name in ds_out.data_vars}
                ds_out.to_netcdf(out_file, encoding=encoding)
                ds_out.close()
                file_path.append(out_file)
                print(f"Saved final ERA5-Land file: {out_file}")
    
                del seasonal_outputs, ds_out
                gc.collect()
    
            except Exception as exc:
                print(f"Failed to process {cv}: {exc}")
    
        return file_path


    def WAS_Download_ERA5Land_daily(
        self,
        dir_to_save,
        center_variable,
        year_start,
        year_end,
        area,
        force_download=False,
        max_retries=3,
        retry_delay=5,
        runoff_units="m3/s",
    ):
        """
        Download ERA5-Land daily data from CDS and save WAS-style NetCDF files.
    
        Output convention:
            T, Y, X
    
        ERA5-Land convention for accumulated variables:
            total precipitation, runoff and radiation at 00 UTC represent the
            accumulation over the previous day. Therefore the daily value for
            YYYY-MM-DD is taken from YYYY-MM-DD + 1 day at 00 UTC.
    
        Parameters
        ----------
        dir_to_save : str or Path
            Output directory.
        center_variable : list of str
            Variables to download, for example ["ERA5Land.PRCP", "ERA5Land.TEMP"].
        year_start, year_end : int
            Inclusive year range.
        area : list
            CDS area as [North, West, South, East].
        force_download : bool
            Redownload even when the output file exists.
        max_retries, retry_delay : int
            Download retry controls.
        runoff_units : {"mm", "m3/s"}
            Output units for RUNOFF. Other variables keep their natural daily units:
            PRCP in mm/day, radiation in W m-2, temperature in degC.
        """
        from pathlib import Path
        import calendar
        import gc
        import os
        import time as _time
    
        import cdsapi
        import numpy as np
        import xarray as xr
    
        if isinstance(center_variable, str):
            raise TypeError(
                "center_variable must be a list, for example "
                "['ERA5Land.PRCP'] or ['ERA5Land.PRCP', 'ERA5Land.RUNOFF']."
            )
        center_variable = list(center_variable)
    
        if runoff_units not in ["mm", "m3/s"]:
            raise ValueError("runoff_units must be 'mm' or 'm3/s'.")
    
        dir_to_save = Path(dir_to_save)
        dir_to_save.mkdir(parents=True, exist_ok=True)
    
        variables_1 = {
            "PRCP": "total_precipitation",
            "TEMP": "2m_temperature",
            "TDEW": "2m_dewpoint_temperature",
            "UGRD10": "10m_u_component_of_wind",
            "VGRD10": "10m_v_component_of_wind",
            "DSWR": "surface_solar_radiation_downwards",
            "DLWR": "surface_thermal_radiation_downwards",
            "NOLR": "surface_net_thermal_radiation",
            "SRUNOFF": "surface_runoff",
            "RUNOFF": "runoff",
        }
        accumulated_vars = {"PRCP", "SRUNOFF", "DSWR", "DLWR", "NOLR"}
        temperature_vars = {"TEMP", "TDEW"}
    
        dataset = "reanalysis-era5-land"
        client = cdsapi.Client()
    
        def _days_in_month(year, month):
            last_day = calendar.monthrange(year, month)[1]
            return [f"{day:02d}" for day in range(1, last_day + 1)]
    
        def _next_year_month(year, month):
            if month == 12:
                return year + 1, 1
            return year, month + 1
    
        def _download_request(target_file, variable, year, month, days, hours):
            request = {
                "variable": [variable],
                "year": str(year),
                "month": f"{month:02d}",
                "day": days,
                "time": hours,
                "area": area,
                "data_format": "netcdf",
                "download_format": "unarchived",
            }
    
            success = False
            for attempt in range(1, max_retries + 1):
                try:
                    print(
                        f"Attempt {attempt}/{max_retries}: downloading "
                        f"{variable} for {year}-{month:02d}..."
                    )
                    client.retrieve(dataset, request).download(str(target_file))
                    print(f"Downloaded: {target_file}")
                    success = True
                    break
                except Exception as exc:
                    print(
                        f"Attempt {attempt}/{max_retries} failed for "
                        f"{variable} {year}-{month:02d}: {exc}"
                    )
                    if target_file.exists():
                        os.remove(target_file)
                        print(f"Deleted incomplete file: {target_file}")
                    if attempt < max_retries:
                        print(f"Retrying after {retry_delay} seconds...")
                        _time.sleep(retry_delay)
    
            return success
    
        def _normalize_coords(ds):
            rename_map = {}
            if "valid_time" in ds.dims or "valid_time" in ds.coords:
                rename_map["valid_time"] = "time"
            if "longitude" in ds.dims or "longitude" in ds.coords:
                rename_map["longitude"] = "X"
            if "latitude" in ds.dims or "latitude" in ds.coords:
                rename_map["latitude"] = "Y"
            ds = ds.rename({old: new for old, new in rename_map.items() if old in ds})
    
            if "expver" in ds.dims and ds.sizes["expver"] == 1:
                ds = ds.isel(expver=0, drop=True)
    
            if "Y" in ds.coords:
                ds = ds.sortby("Y")
    
            return ds
    
        def _rename_data_variable(ds, var_code):
            data_vars = list(ds.data_vars)
            if len(data_vars) == 1 and data_vars[0] != var_code:
                ds = ds.rename({data_vars[0]: var_code})
            return ds
    
        def _grid_cell_area_m2(ds):
            if "Y" not in ds.coords or "X" not in ds.coords:
                raise ValueError("SRUNOFF conversion to m3/s needs X and Y coordinates.")
            if ds.sizes.get("Y", 0) < 2 or ds.sizes.get("X", 0) < 2:
                raise ValueError("SRUNOFF conversion to m3/s needs at least two X and Y points.")
    
            radius = 6371000.0
            dlat = np.deg2rad(float(abs(ds["Y"].diff("Y").median())))
            dlon = np.deg2rad(float(abs(ds["X"].diff("X").median())))
            area_y = (radius ** 2) * dlat * dlon * np.cos(np.deg2rad(ds["Y"]))
            area_y.attrs["units"] = "m2"
            return area_y
    
        def _open_many(paths):
            datasets = []
            for path in paths:
                ds = xr.open_dataset(path)
                datasets.append(_normalize_coords(ds))
            combined = xr.concat(datasets, dim="time").sortby("time").load()
            for ds in datasets:
                ds.close()
            return combined
    
        def _process_accumulated_month(v, paths, year, month):
            ds = _open_many(paths)
            ds = _rename_data_variable(ds, v)
    
            # 00 UTC belongs to the previous day for ERA5-Land accumulated fields.
            ds = ds.sel(time=ds["time"].dt.hour == 0)
            ds = ds.assign_coords(time=ds["time"] - np.timedelta64(1, "D"))
    
            start = np.datetime64(f"{year}-{month:02d}-01")
            end_day = calendar.monthrange(year, month)[1]
            end = np.datetime64(f"{year}-{month:02d}-{end_day:02d}")
            ds = ds.sel(time=slice(start, end))
    
            if v == "PRCP":
                ds = (ds * 1000.0).where(ds >= 0, 0)
                for name in ds.data_vars:
                    ds[name].attrs["units"] = "mm/day"
    
            elif v in ["RUNOFF", "SRUNOFF"]:
                ds = ds.where(ds >= 0, 0)
                if runoff_units == "mm":
                    ds = ds * 1000.0
                    for name in ds.data_vars:
                        ds[name].attrs["units"] = "mm/day"
                else:
                    ds = (ds * _grid_cell_area_m2(ds)) / 86400.0
                    for name in ds.data_vars:
                        ds[name].attrs["units"] = "m3 s-1"
    
            elif v in ["DSWR", "DLWR", "NOLR"]:
                ds = ds / 86400.0
                for name in ds.data_vars:
                    ds[name].attrs["units"] = "W m-2"
    
            return ds
    
        def _process_instantaneous_month(v, path):
            ds = xr.open_dataset(path)
            ds = _normalize_coords(ds)
            ds = _rename_data_variable(ds, v)
    
            ds_daily = ds.resample(time="1D").mean(keep_attrs=True).load()
            ds.close()
    
            if v in temperature_vars:
                ds_daily = ds_daily - 273.15
                for name in ds_daily.data_vars:
                    ds_daily[name].attrs["units"] = "degC"
    
            return ds_daily
    
        def _finalize(ds, v):
            if "time" in ds.dims or "time" in ds.coords:
                ds = ds.rename({"time": "T"})
    
            preferred = ["T", "Y", "X"]
            for name in list(ds.data_vars):
                dims = [dim for dim in preferred if dim in ds[name].dims]
                dims += [dim for dim in ds[name].dims if dim not in dims]
                ds[name] = ds[name].transpose(*dims)
    
            ds.attrs["source_dataset"] = dataset
            ds.attrs["center"] = "ERA5Land"
            ds.attrs["variable_code"] = v
            if v in ["SRUNOFF", "RUNOFF"]:
                ds.attrs["runoff_units"] = runoff_units
            return ds
    
        store_file_path = {}
    
        for cv in center_variable:
            try:
                center, v = cv.split(".", 1)
                if center != "ERA5Land":
                    print(f"Invalid center for this function: {center}. Skipping.")
                    continue
                if v not in variables_1:
                    print(f"Unknown variable: {v}. Skipping.")
                    continue
    
                cds_variable = variables_1[v]
                unit_tag = "_m3s" if v in ["SRUNOFF", "RUNOFF"] and runoff_units == "m3/s" else ""
                output_path = dir_to_save / f"Daily_{v}_{year_start}_{year_end}{unit_tag}.nc"
    
                if output_path.exists() and not force_download:
                    print(f"{output_path} already exists. Skipping download.")
                    store_file_path[cv] = output_path
                    continue
    
                combined_datasets = []
                all_downloads_ok = True
    
                for year in range(year_start, year_end + 1):
                    for month in range(1, 13):
                        temp_files = []
                        try:
                            if v in accumulated_vars:
                                last_day = calendar.monthrange(year, month)[1]
                                current_days = [f"{day:02d}" for day in range(2, last_day + 1)]
                                current_file = dir_to_save / f"era5land_00_{v}_{year}_{month:02d}.nc"
    
                                if current_days:
                                    ok = _download_request(
                                        current_file,
                                        cds_variable,
                                        year,
                                        month,
                                        current_days,
                                        ["00:00"],
                                    )
                                    if not ok:
                                        all_downloads_ok = False
                                        continue
                                    temp_files.append(current_file)
    
                                next_year, next_month = _next_year_month(year, month)
                                boundary_file = (
                                    dir_to_save
                                    / f"era5land_00_{v}_{next_year}_{next_month:02d}_day01.nc"
                                )
                                ok = _download_request(
                                    boundary_file,
                                    cds_variable,
                                    next_year,
                                    next_month,
                                    ["01"],
                                    ["00:00"],
                                )
                                if not ok:
                                    all_downloads_ok = False
                                    continue
                                temp_files.append(boundary_file)
    
                                ds_daily = _process_accumulated_month(v, temp_files, year, month)
    
                            else:
                                hourly_file = dir_to_save / f"era5land_hourly_{v}_{year}_{month:02d}.nc"
                                ok = _download_request(
                                    hourly_file,
                                    cds_variable,
                                    year,
                                    month,
                                    _days_in_month(year, month),
                                    [f"{hour:02d}:00" for hour in range(24)],
                                )
                                if not ok:
                                    all_downloads_ok = False
                                    continue
                                temp_files.append(hourly_file)
    
                                ds_daily = _process_instantaneous_month(v, hourly_file)
    
                            combined_datasets.append(ds_daily)
    
                        except Exception as exc:
                            print(f"Failed to process ERA5Land.{v} for {year}-{month:02d}: {exc}")
                            all_downloads_ok = False
    
                        finally:
                            for path in temp_files:
                                if path.exists():
                                    os.remove(path)
                                    print(f"Deleted temporary file: {path}")
                            gc.collect()
    
                if not combined_datasets:
                    print(f"No data processed for ERA5Land.{v}. Skipping save.")
                    continue
    
                if not all_downloads_ok:
                    print(f"Skipping save for ERA5Land.{v} due to incomplete downloads.")
                    continue
    
                combined_ds = xr.concat(combined_datasets, dim="time").sortby("time")
                combined_ds = _finalize(combined_ds, v)
    
                encoding = {name: {"zlib": True, "complevel": 4} for name in combined_ds.data_vars}
                combined_ds.to_netcdf(output_path, encoding=encoding)
                combined_ds.close()
                store_file_path[cv] = output_path
                print(f"Combined dataset for ERA5Land.{v} saved to {output_path}")
    
                del combined_ds, combined_datasets
                gc.collect()
    
            except Exception as exc:
                print(f"Failed ERA5Land download for {cv}: {exc}")
    
        return store_file_path
    
    # def WAS_Download_ERA5Land_daily(
    #         self,
    #         dir_to_save,
    #         center_variable,
    #         year_start,
    #         year_end,
    #         area,
    #         force_download=False,
    #         max_retries=3,
    #         retry_delay=5,
    #     ):
    #     """
    #     Download daily ERA5Land reanalysis data from the Copernicus Data Store (CDS)
    #     for specified variables and years, with retries for failed downloads.

    #     Parameters
    #     ----------
    #     dir_to_save : str or pathlib.Path
    #         Directory path where the downloaded NetCDF files will be saved.
    #         The directory will be created if it does not exist.
    #     center_variable : list of str
    #         List of center-variable identifiers (e.g., ["ERA5Land.PRCP", "ERA5Land.TEMP"]).
    #     year_start : int
    #         Start year for the data to download (inclusive).
    #     year_end : int
    #         End year for the data to download (inclusive).
    #     area : list of float
    #         Bounding box for spatial subsetting in the format [North, West, South, East].
    #     force_download : bool, optional
    #         If True, forces download even if the output file exists. Default is False.
    #     max_retries : int, optional
    #         Maximum number of retry attempts for failed downloads. Default is 3.
    #     retry_delay : int, optional
    #         Seconds to wait between retry attempts. Default is 5.

    #     Returns
    #     -------
    #     None
    #         The function saves NetCDF files to `dir_to_save` but does not return a value.
    #         Output files are named as `Daily_<variable>_<year_start>_<year_end>.nc`.

    #     Notes
    #     -----
    #     - The function downloads hourly data from the CDS dataset "reanalysis-era5-land"
    #       and aggregates it to daily resolution.
    #     - Aggregation: sum for accumulative variables (e.g., PRCP, radiation), mean for others (e.g., TEMP).
    #     - Unit conversions: PRCP to mm/day, TEMP to °C, radiation to W/m² (daily average).
    #     - Coordinates are renamed to "X" (longitude), "Y" (latitude), and "T" (time),
    #       with latitude flipped.
    #     - Downloads are performed month-by-month due to API limitations.
    #     - Requires a valid CDS API key configured in `~/.cdsapirc`.
    #     """
    #     dir_to_save = Path(dir_to_save)
    #     dir_to_save.mkdir(parents=True, exist_ok=True)
    #     days = [f"{day:02}" for day in range(1, 32)]
    #     months = [f"{month:02}" for month in range(1, 13)]
    #     hours = [f"{h:02}:00" for h in range(24)]
    #     variables_1 = {
    #         "PRCP": "total_precipitation",
    #         "TEMP": "temperature_2m",
    #         "TDEW": "dewpoint_temperature_2m",
    #         "UGRD10": "u_component_of_wind_10m",
    #         "VGRD10": "v_component_of_wind_10m",
    #         "DSWR": "surface_solar_radiation_downwards",
    #         "DLWR": "surface_thermal_radiation_downwards",
    #         "NOLR": "surface_net_thermal_radiation_downwards",
    #         "RUNOFF": "surface_runoff",
    #     }
    #     centers = [cv.split(".")[0] for cv in center_variable]
    #     vars_short = [cv.split(".")[1] for cv in center_variable]
    #     for c, v in zip(centers, vars_short):
    #         if c != "ERA5Land":
    #             print(f"Invalid center for this function: {c}. Must be 'ERA5Land'. Skipping.")
    #             continue
    #         if v not in variables_1:
    #             print(f"Unknown variable: {v}. Skipping.")
    #             continue
    #         cds_variable = variables_1[v]
    #         output_path = dir_to_save / f"Daily_{v}_{year_start}_{year_end}.nc"
    #         if not force_download and output_path.exists():
    #             print(f"{output_path} already exists. Skipping download.")
    #             continue
    #         combined_datasets = []
    #         all_years_downloaded = True
    #         for year in range(year_start, year_end + 1):
    #             for month in range(1, 13):
    #                 nc_file_path = dir_to_save / f"hourly_{v}_{year}_{month:02d}.nc"
    #                 success = False
    #                 retries = 0
    #                 while retries < max_retries and not success:
    #                     try:
    #                         client = cdsapi.Client()
    #                         dataset = "reanalysis-era5-land"
    #                         request = {
    #                             "variable": cds_variable,
    #                             "year": str(year),
    #                             "month": f"{month:02}",
    #                             "day": days,
    #                             "time": hours,
    #                             "area": area,
    #                             "data_format": "netcdf",
    #                             "download_format": "unarchived"
    #                         }
    #                         print(f"Attempt {retries + 1}/{max_retries}: Downloading {cds_variable} data for {year}-{month:02d}...")
    #                         client.retrieve(dataset, request).download(str(nc_file_path))
    #                         print(f"Downloaded: {nc_file_path}")
    #                         success = True
    #                     except Exception as e:
    #                         retries += 1
    #                         print(f"Attempt {retries}/{max_retries} failed for {cds_variable} data for {year}-{month:02d}: {e}")
    #                         if retries < max_retries:
    #                             print(f"Retrying after {retry_delay} seconds...")
    #                             _time.sleep(retry_delay)
    #                         if nc_file_path.exists():
    #                             os.remove(nc_file_path)
    #                             print(f"Deleted incomplete file: {nc_file_path}")
    #                 if not success:
    #                     print(f"Failed to download {cds_variable} data for {year}-{month:02d} after {max_retries} attempts.")
    #                     all_years_downloaded = False
    #                     continue
    #                 try:
    #                     ds_month = xr.open_dataset(nc_file_path)
    #                     ds_daily = self._postprocess_reanalysis(ds_month, v)
    #                     if v == "PRCP":
    #                         ds_daily = ds_daily.sel(time=ds_daily.time.dt.hour == 0) * 1000
    #                         ds_daily_0 = ds_daily.isel(time=0)
    #                         ds_daily_0.coords['time'] = pd.Timestamp(f"{year}-{month}-{calendar.monthrange(year, month)[1]}")
    #                         ds_daily = ds_daily.shift(time=-1).dropna(dim="time")
    #                         ds_daily = xr.concat([ds_daily_0, ds_daily], dim="time")
    #                     if v == "RUNOFF":
    #                         ds_daily = ds_daily.sel(time=ds_daily.time.dt.hour == 0)
    #                         ds_daily_0 = ds_daily.isel(time=0)
    #                         ds_daily_0.coords['time'] = pd.Timestamp(f"{year}-{month}-{calendar.monthrange(year, month)[1]}")
    #                         ds_daily = ds_daily.shift(time=-1).dropna(dim="time")
    #                         ds_daily = xr.concat([ds_daily_0, ds_daily], dim="time")
    #                         lat = ds_daily.Y
    #                         lon = ds_daily.X
    #                         dlon = np.deg2rad(0.1)
    #                         dlat = np.deg2rad(0.1)
    #                         r = 6371000 # Earth radius in meters
    #                         # Area for each grid cell
    #                         area_= (r ** 2) * dlon * np.cos(np.deg2rad(lat)) * dlat
    #                         # Perform the conversion to m^3/s
    #                         ds_daily = (ds_daily * area_) / 86400
    #                     if v in ["DSWR", "DLWR", "NOLR"]:
    #                         ds_daily = ds_daily.sel(time=ds_daily.time.dt.hour == 0) / 86400
    #                         ds_daily_0 = ds_daily.isel(time=0)
    #                         ds_daily_0.coords['time'] = pd.Timestamp(f"{year}-{month}-{calendar.monthrange(year, month)[1]}")
    #                         ds_daily = ds_daily.shift(time=-1).dropna(dim="time")
    #                         ds_daily = xr.concat([ds_daily_0, ds_daily], dim="time")
    #                     if v in ["TEMP", "TDEW"]:
    #                         ds_daily = ds_daily - 273.15  # K to °C
    #                     combined_datasets.append(ds_daily)
    #                 except Exception as e:
    #                     print(f"Failed to process {nc_file_path}: {e}")
    #                     all_years_downloaded = False
    #                 if nc_file_path.exists():
    #                     os.remove(nc_file_path)
    #                     print(f"Deleted hourly file: {nc_file_path}")
    #         if combined_datasets and all_years_downloaded:
    #             try:
    #                 combined_ds = xr.concat(combined_datasets, dim="time")
    #                 combined_ds = combined_ds.rename({"time": "T"})
    #                 combined_ds.to_netcdf(output_path)
    #                 combined_ds.close()
    #                 print(f"Combined dataset for ERA5Land.{v} saved to {output_path}")
    #             except Exception as e:
    #                 print(f"Failed to process or save combined dataset for ERA5Land.{v}: {e}")
    #         else:
    #             print(f"Skipping save for ERA5Land.{v} due to incomplete downloads.")

#     def WAS_Download_CHIRPSv3_Daily(
#             self,
#             dir_to_save,
#             year_start,
#             year_end,
#             blend_type="ERA5",
#             area=None,
#             force_download=False,
#             max_retries=3,
#             retry_delay=5,
#         ):
#         """
#         Download daily CHIRPS v3.0 precipitation (blended with ERA5 or IMERGlate-v07) from the Copernicus Data Store (CDS)
#         for specified years, with retries for failed downloads.

#         Parameters
#         ----------
#         dir_to_save : str or pathlib.Path
#             Directory path where the downloaded NetCDF files will be saved.
#             The directory will be created if it does not exist.
#         year_start : int
#             Start year for the data to download (inclusive).
#         year_end : int
#             End year for the data to download (inclusive).
#         blend_type : str
#             Blend type, either "ERA5" or "IMERGlate-v07" (default: "ERA5").
#             Note: IMERGlate-v07 availability starts from ~2000; earlier years may fail.
#         area : list of float
#             Bounding box for spatial subsetting in the format [North, West, South, East].
#         force_download : bool, optional
#             If True, forces download even if the output file exists. Default is False.
#         max_retries : int, optional
#             Maximum number of retry attempts for failed downloads. Default is 3.
#         retry_delay : int, optional
#             Seconds to wait between retry attempts. Default is 5.

#         Returns
#         -------
#         None
#             The function saves NetCDF files to `dir_to_save` but does not return a value.
#             Output files are named as `Daily_PRCP_{blend_type}_{year_start}_{year_end}.nc`.

#         Notes
#         -----
#         - Downloads individual daily TIFF files, processes them with rioxarray, clips if area specified,
#           and combines into a single NetCDF with daily time dimension.
#         - Units: precipitation in mm/day.
#         - Coordinates renamed to "X" (longitude), "Y" (latitude), "T" (time), with Y flipped if needed.
#         - Deletes temporary TIFF files after processing.
#         - Skips invalid dates (e.g., Feb 30) automatically.
#         """
#         dir_to_save = Path(dir_to_save)
#         dir_to_save.mkdir(parents=True, exist_ok=True)

#         output_path = dir_to_save / f"Daily_PRCP_{year_start}_{year_end}.nc"
#         if not force_download and output_path.exists():
#             print(f"{output_path} already exists. Skipping download.")
#             return output_path

#         else:
            
#             combined_datasets = []
#             all_years_downloaded = True
#             for year in range(year_start, year_end + 1):
#                 for month in range(1, 13):
#                     ndays = calendar.monthrange(year, month)[1]
#                     for day in range(1, ndays + 1):
#                         tif_file_path = dir_to_save / f"chirps-v3.0.{year}.{month:02d}.{day:02d}.tif"
#                         success = False
#                         retries = 0
#                         while retries < max_retries and not success:
#                             try:
#                                 da = self._fetch_chirps_daily(
#                                     year=year,
#                                     month=month,
#                                     day=day,
#                                     dir_to_save=dir_to_save,
#                                     blend_type=blend_type,
#                                     force_download=force_download,
#                                     area=area
#                                 )
#                                 if da is not None:
#                                     combined_datasets.append(da)
#                                     success = True
#                             except Exception as e:
#                                 retries += 1
#                                 print(f"Attempt {retries}/{max_retries} failed for {year}-{month:02d}-{day:02d}: {e}")
#                                 if retries < max_retries:
#                                     print(f"Retrying after {retry_delay} seconds...")
#                                     _time.sleep(retry_delay)
#                                 if tif_file_path.exists():
#                                     os.remove(tif_file_path)
#                                     print(f"Deleted incomplete TIFF: {tif_file_path}")
#                         if not success:
#                             print(f"Failed to download/process {year}-{month:02d}-{day:02d} after {max_retries} attempts.")
#                             # Continue to next day; don't set all_years_downloaded=False to allow partial data
#                         if tif_file_path.exists():
#                             os.remove(tif_file_path)
#                             print(f"Deleted TIFF: {tif_file_path}")
#             if combined_datasets:
#                 try:
#                     combined_ds = xr.concat(combined_datasets, dim="time").to_dataset(name="precip")
#                     combined_ds = combined_ds.rename({"x": "X", "y": "Y", "time": "T"}).drop_vars('band')
#                     combined_ds = combined_ds.isel(Y=slice(None, None, -1))
#                     combined_ds.to_netcdf(output_path)
#                     combined_ds.close()
#                     print(f"Combined daily dataset for CHIRPS ({blend_type}) saved to {output_path}")
#                     return output_path
#                 except Exception as e:
#                     print(f"Failed to process or save combined dataset for CHIRPS ({blend_type}): {e}")
#             else:
#                 print(f"No data downloaded for CHIRPS ({blend_type}).")

    
#     def _fetch_chirps_daily(self, year, month, day, dir_to_save, blend_type, force_download, area):
#             """
#             Construct the CHIRPS v3.0 daily TIF URL for (year, month, day),
#             download if needed, open as xarray, and optionally clip to 'area'.
           
#             File format is: chirps-v3.0.YYYY.MM.DD.tif
#             """
#             _type = "rnl" if blend_type == "ERA5" else "sat"
                
#             base_url = f"https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/{_type}/{year}"
#             fname = f"chirps-v3.0.{_type}.{year}.{month:02d}.{day:02d}.tif"
#             url = f"{base_url}/{fname}"
#             local_path = Path(dir_to_save) / fname
#             if not local_path.exists() or force_download:
#                 try:
#                     with requests.get(url, stream=True, timeout=120) as r:
#                         r.raise_for_status()
#                         with open(local_path, "wb") as f:
#                             for chunk in r.iter_content(chunk_size=8192):
#                                 f.write(chunk)
#                 except Exception as e:
#                     print(f"[ERROR] Could not download {url}: {e}")
#                     return None
#             else:
#                 print(f"[SKIP] {fname} is already present. (Use force_download=True to overwrite)")
#             # Open as xarray via rioxarray
#             try:
#                 da = rioxr.open_rasterio(local_path, masked=True).squeeze()
#                 time_coord = pd.to_datetime(f"{year}-{month:02d}-{day:02d}")
#                 da = da.expand_dims(time=[time_coord])
#                 da.name = "precip"
#                 # If area is provided, clip
#                 if area and len(area) == 4:
#                     north, west, south, east = area
#                     da = da.rio.clip_box(
#                         minx=west,
#                         miny=south,
#                         maxx=east,
#                         maxy=north
#                     )
#                 return da
#             except Exception as e:
#                 print(f"[ERROR] Could not open/parse {local_path}: {e}")
#                 return None
                

#     def WAS_Download_CHIRPSv3_Seasonal(
#         self,
#         dir_to_save,
#         variables,
#         year_start,
#         year_end,
#         region="africa",
#         area=None,
#         season_months=["03", "04", "05"],
#         force_download=False        
#     ):
        
#         """
#         Download CHIRPS v3.0 monthly precipitation for a specified cross-year season
#         from year_start to year_end, optionally clipped to 'area',
#         and aggregate them into a single NetCDF file.
#         Parameters:
#             dir_to_save (str): Directory to save the downloaded files.
#             variables (list): List of variables to download (e.g., ["PRCP"]).
#             year_start (int): Start year for the data.
#             year_end (int): End year for the data.
#             region (str): CHIRPS region (default: "africa").
#             area (list): Bounding box as [North, West, South, East] (optional).
#             season_months (list): List of months as strings (e.g., ["03", "04", "05"]).
#             force_download (bool): If True, forces download even if file exists.
#         Returns:
#             None: Saves the aggregated seasonal data to a NetCDF file.  
#         """
#         dir_to_save = Path(dir_to_save)
#         dir_to_save.mkdir(parents=True, exist_ok=True)
#         season_months = [int(m) for m in season_months]
#         variables = variables
#         # Example: "MAM"
#         season_str = "".join([calendar.month_abbr[m] for m in season_months])
#         pivot = season_months[0]

#         out_nc = dir_to_save / f"Obs_PRCP_{year_start}_{year_end}_{season_str}.nc"
#         if out_nc.exists() and not force_download:
#             print(f"[INFO] {out_nc} already exists. Skipping.")
#             return out_nc

#         else:
            
#             # We'll store monthly DataArrays here
#             all_data_arrays = []
    
#             # Loop over years
#             for year in range(year_start, year_end + 1):
#                 # Base-year months (>= pivot)
#                 base_months = [m for m in season_months if m >= pivot]
#                 # Next-year months (< pivot)
#                 next_months = [m for m in season_months if m < pivot]
    
#                 # Part A: Base-year months
#                 for m in base_months:
#                     da = self._fetch_chirps_monthly(
#                         year=year,
#                         month=m,
#                         dir_to_save=dir_to_save,
#                         region=region,
#                         force_download=force_download,
#                         area=area
#                     )
#                     if da is not None:
#                         all_data_arrays.append(da)
    
#                 # Part B: Next-year months
#                 if next_months and (year < year_end + 1):
#                     year_next = year + 1
#                     for m in next_months:
#                         da = self._fetch_chirps_monthly(
#                             year=year_next,
#                             month=m,
#                             dir_to_save=dir_to_save,
#                             region=region,
#                             force_download=force_download,
#                             area=area
#                         )
#                         if da is not None:
#                             all_data_arrays.append(da)
    
#             if len(all_data_arrays) == 0:
#                 print("[WARNING] No CHIRPS data arrays were opened/downloaded.")
#                 return
    
#             # Concatenate along time
#             ds_all = xr.concat(all_data_arrays, dim="time").to_dataset(name="precip")
    
#             # Aggregate across the cross-year season (summing monthly precipitation)
#             ds_season = self._aggregate_chirps(ds_all, season_months)
    
#             # Rename dims if desired
#             if "x" in ds_season.dims:
#                 ds_season = ds_season.rename({"x": "X"})
#             if "y" in ds_season.dims:
#                 ds_season = ds_season.rename({"y": "Y"})
#             if "time" in ds_season.dims:
#                 ds_season = ds_season.rename({"time": "T"})
                

#             if len(season_months)==1:
#                 ds_season["T"] = [f"{year}-{season_months[0]:02d}-01" for year in ds_season["T"].dt.year.astype(str).values]
#             elif len(season_months) in [2,3]:
#                 ds_season["T"] = [f"{year}-{season_months[1]:02d}-01" for year in ds_season["T"].dt.year.astype(str).values]
#             elif len(season_months) in [4,5]:
#                 ds_season["T"] = [f"{year}-{season_months[2]:02d}-01" for year in ds_season["T"].dt.year.astype(str).values]
#             else:
#                 ds_season["T"] = [f"{year}-{season_months[3]:02d}-01" for year in ds_season["T"].dt.year.astype(str).values]
#             ds_season["T"] = ds_season["T"].astype("datetime64[ns]")

            
#             # Write to NetCDF
#             ds_season.drop_vars(['band','spatial_ref']).squeeze().isel(Y=slice(None, None, -1)).to_netcdf(out_nc)
#             print(f"[INFO] Saved seasonal CHIRPS data to {out_nc}")
#             # Delete individual monthly TIF files
#             for tif_file in dir_to_save.glob("chirps-v3.0.*.tif"):
#                 try:
#                     os.remove(tif_file)
#                     print(f"[CLEANUP] Deleted {tif_file}")
#                 except Exception as e:
#                     print(f"[ERROR] Could not delete {tif_file}: {e}")
#             return out_nc


#     def _fetch_chirps_monthly(self, year, month, dir_to_save, region, force_download, area):
#         """
#         Construct the CHIRPS v3.0 monthly TIF URL for (year, month), 
#         download if needed, open as xarray, and optionally clip to 'area'.
        
#         File format is: chirps-v3.0.YYYY.MM.tif
#         """
#         base_url = f"https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/{region}/tifs"
#         fname = f"chirps-v3.0.{year}.{month:02d}.tif"
#         url = f"{base_url}/{fname}"

#         local_path = Path(dir_to_save) / fname
#         download_file(url, local_path, force_download=force_download, chunk_size=8192, timeout=120)
#         try:
#             da = rioxr.open_rasterio(local_path, masked=True).squeeze()
#             time_coord = pd.to_datetime(f"{year}-{month:02d}-01")
#             da = da.expand_dims(time=[time_coord])
#             da.name = "precip"

#             # If area is provided, clip
#             if area and len(area) == 4:
#                 north, west, south, east = area
#                 da = da.rio.clip_box(
#                     minx=west,
#                     miny=south,
#                     maxx=east,
#                     maxy=north
#                 )

#             return da

#         except Exception as e:
#             print(f"[ERROR] Could not open/parse {local_path}: {e}")
#             return None

#     def _aggregate_chirps(self, ds, season_months):
#         """
#         Sum monthly precipitation across the cross-year season.
#         """
#         if "time" not in ds.coords:
#             raise ValueError("Dataset must have a 'time' dimension.")

#         pivot = season_months[0]
#         # Label each time with 'season_year'
#         season_year = ds["time"].dt.year.where(ds["time"].dt.month >= pivot,
#                                                ds["time"].dt.year - 1)
#         ds = ds.assign_coords(season_year=season_year)

#         # Keep only the months we want
#         ds = ds.where(ds["time"].dt.month.isin(season_months), drop=True)

#         # Sum across the months for precipitation
#         ds_out = ds.groupby("season_year").sum("time", skipna=True)

#         # Rename season_year -> time
#         ds_out = ds_out.rename({"season_year": "time"})

#         # Optionally make the new time coordinate more descriptive:
#         new_times = []
#         for sy in ds_out.coords["time"].values:
#             new_times.append(f"{sy}-{pivot:02d}-01")
#         ds_out = ds_out.assign_coords(time=pd.to_datetime(new_times))

#         return ds_out
# ####
#     def WAS_Download_TAMSAT_Seasonal(
#             self,
#             dir_to_save: Union[str, Path],
#             product: Literal["rfe", "soil_moisture"] = "rfe",
#             variables: Optional[Sequence[str]] = None,
#             year_start: int = 1983,
#             year_end: int = 2025,
#             area: Optional[List[float]] = None,
#             season_months: List[str] = ["03", "04", "05"],
#             version: Optional[str] = None,
#             force_download: bool = False,
#             agg: Optional[Literal["sum", "mean"]] = None,
#         ) -> Path:
        
#             """
#             Download and aggregate TAMSAT monthly data (RFE v3.1 or Soil Moisture v2.3.1) for a specified season.

#             Parameters
#             ----------
#             dir_to_save : str | Path
#                 Directory where monthly files and the seasonal output will be saved.
#             product : {"rfe", "soil_moisture"}, default "rfe"
#                 Dataset to download: "rfe" (precipitation) or "soil_moisture".
#             variables : sequence of str, optional
#                 Names of variables to extract from NetCDF. If None, chosen by product.
#                 - rfe: defaults to ("rfe",)
#                 - soil_moisture: defaults to ("sm",)
#             year_start : int, default 1983
#                 First seasonal year (pivot year) to include.
#             year_end : int, default 2025
#                 Last year for which data is included (inclusive). For seasons spanning calendar years,
#                 the last pivot year processed will be year_end - 1 to ensure no data from year_end + 1 is fetched.
#             area : list[float], optional
#                 Bounding box [north, west, south, east] in degrees.
#             season_months : sequence[str], default ["03","04","05"]
#                 Months defining the season, e.g. ["11","12","01"] for NDJ.
#             version : str, optional
#                 Product version. Defaults to:
#                 - rfe: "v3.1"
#                 - soil_moisture: "v2.3.1"
#             force_download : bool, default False
#                 Re-download monthly files even if present locally.
#             agg : {"sum","mean"}, optional
#                 Seasonal aggregation. Defaults to:
#                 - rfe: "sum"
#                 - soil_moisture: "mean"

#             Returns
#             -------
#             Path
#                 Path to the seasonal aggregated NetCDF file.
#             """
#             dir_to_save = Path(dir_to_save)
#             dir_to_save.mkdir(parents=True, exist_ok=True)
#             season_months = tuple(season_months)
#             # ---- sensible defaults by product ----
#             if product == "rfe":
#                 variables = variables or ("rfe")
#                 version = version or "v3.1"
#                 agg = agg or "sum"
#                 std_name = "precip"
#             elif product == "soilmoisture":
#                 variables = variables or ("smc_avail_top",)
#                 version = version or "v2.3.1"
#                 agg = agg or "mean"
#                 std_name = "soil_moisture"
#             else:
#                 raise ValueError("product must be 'rfe' or 'soilmoisture'")
#             # ---- validate inputs ----
#             if year_start > year_end:
#                 raise ValueError("year_start must be <= year_end.")
#             season_months_int: List[int] = [int(m) for m in season_months]
#             if not all(1 <= m <= 12 for m in season_months_int):
#                 raise ValueError("Season months must be valid month numbers (1-12).")
#             area_tuple: Optional[Tuple[float, float, float, float]] = (
#                 tuple(map(float, area)) if area else None
#             )
#             season_str = "".join(calendar.month_abbr[m] for m in season_months_int)
#             pivot = season_months_int[0]
#             # ---- output filename ----
#             out_nc = dir_to_save / f"Obs_{product.upper()}_{year_start}_{year_end}_{season_str}.nc"
#             if out_nc.exists() and not force_download:
#                 print(f"[INFO] {out_nc} already exists – skip download.")
#                 return out_nc
#             else:
#                 # ---- determine if season spans years ----
#                 spanning = any(m < pivot for m in season_months_int)
#                 last_season_year = year_end if not spanning else year_end - 1
#                 if year_start > last_season_year:
#                     raise ValueError("No seasons to process based on year_start and year_end.")
#                 # ---- build seasonal series (aggregate per season_year then stack) ----
#                 seasonal_list: List[xr.DataArray] = []
#                 for season_year in range(year_start, last_season_year + 1):
#                     monthly_das: List[xr.DataArray] = []
#                     # Part A: base-year months (>= pivot)
#                     for m in (m for m in season_months_int if m >= pivot):
#                         da = self._fetch_tamsat_monthly(
#                             product=product,
#                             version=version,
#                             year=season_year,
#                             month=m,
#                             dir_to_save=dir_to_save,
#                             force_download=force_download,
#                             area=area_tuple,
#                             keep_vars=variables,
#                             std_name=std_name,
#                         )
#                         if da is not None:
#                             monthly_das.append(da)
#                     # Part B: next-year months (< pivot)
#                     if spanning:
#                         next_year = season_year + 1
#                         for m in (m for m in season_months_int if m < pivot):
#                             da = self._fetch_tamsat_monthly(
#                                 product=product,
#                                 version=version,
#                                 year=next_year,
#                                 month=m,
#                                 dir_to_save=dir_to_save,
#                                 force_download=force_download,
#                                 area=area_tuple,
#                                 keep_vars=variables,
#                                 std_name=std_name,
#                             )
#                             if da is not None:
#                                 monthly_das.append(da)
#                     if not monthly_das:
#                         # nothing for this season_year → skip
#                         continue
#                     # stack months then aggregate for this season
#                     season_stack = xr.concat(monthly_das, dim="time")
#                     if agg == "sum":
#                         season_da = season_stack.sum(dim="time", keep_attrs=True)
#                     elif agg == "mean":
#                         season_da = season_stack.mean(dim="time", keep_attrs=True)
#                     else:
#                         raise ValueError("agg must be 'sum' or 'mean'.")
#                     # give a representative time stamp (pivot-year)
#                     season_time = pd.to_datetime(f"{season_year}-{pivot:02d}-15")
#                     season_da = season_da.expand_dims(time=[season_time])
#                     seasonal_list.append(season_da)
#                 if not seasonal_list:
#                     raise RuntimeError("No TAMSAT files were downloaded or opened for any season.")
#                 # ---- concat seasons and save ----
#                 da_all = xr.concat(seasonal_list, dim="time")
#                 ds_out = da_all.to_dataset(name=std_name)
    
#                 # harmonize dims if needed
#                 rename_dict = {k: v for k, v in {"lon": "X", "lat": "Y", "time": "T"}.items() if k in ds_out.dims}
#                 ds_out = ds_out.rename(rename_dict)
#                 season_months_ = [int(m) for m in season_months]
#                 if len(season_months_)==1:
#                     ds_out["T"] = [f"{year}-{season_months_[0]:02d}-01" for year in ds_out["T"].dt.year.astype(str).values]
#                 elif len(season_months_) in [2,3]:
#                     ds_out["T"] = [f"{year}-{ season_months_[1]:02d}-01" for year in ds_out["T"].dt.year.astype(str).values]
#                 elif len(season_months_) in [4,5]:
#                     ds_out["T"] = [f"{year}-{season_months_[2]:02d}-01" for year in ds_out["T"].dt.year.astype(str).values]
#                 else:
#                     ds_out["T"] = [f"{year}-{season_months_[3]:02d}-01" for year in ds_out["T"].dt.year.astype(str).values]                    
#                 ds_out["T"] = ds_out["T"].astype("datetime64[ns]")
#                 ds_out.to_netcdf(out_nc)
#                 print(f"[INFO] Saved seasonal {product.upper()} data → {out_nc}")
#                 return out_nc
                
#     def _fetch_tamsat_monthly(
#         self,
#         product: Literal["rfe", "soilmoisture"],
#         version: str,
#         year: int,
#         month: int,
#         dir_to_save: Path,
#         force_download: bool,
#         area: Optional[Tuple[float, float, float, float]],
#         keep_vars: Sequence[str],
#         std_name: str,
#     ) -> Optional[xr.DataArray]:
#         """
#         Download & open a single monthly TAMSAT file (RFE v3.1 or Soil Moisture v2.3.1),
#         clip to bbox if provided, and return a standardized DataArray.
#         Returns
#         -------
#         xr.DataArray | None
#         """
#         if product == "rfe":
#             # e.g. .../tamsat/rfe/data/v3.1/monthly/1983/01/rfe1983_01.v3.1.nc
#             base = (
#                 "https://gws-access.jasmin.ac.uk/public/tamsat/rfe/data/"
#                 f"{version}/monthly/{{year}}/{{month:02d}}/rfe{{year}}_{{month:02d}}.{version}.nc"
#             )
#         else: # soil_moisture
#             # e.g. .../tamsat/soil_moisture/data/v2.3.1/monthly/1983/01/sm1983_01.v2.3.1.nc
#             base = (
#                 "https://gws-access.jasmin.ac.uk/public/tamsat/soil_moisture/data/"
#                 f"{version}/monthly/{{year}}/{{month:02d}}/sm{{year}}_{{month:02d}}.{version}.nc"
#             )
#         url = base.format(year=year, month=month)
#         fname = dir_to_save / url.split("/")[-1]
#         # download if needed
#         if not fname.exists() or force_download:
#             try:
#                 print(f"[DL ] {url}")
#                 with requests.get(url, stream=True, timeout=180) as r:
#                     r.raise_for_status()
#                     with open(fname, "wb") as f:
#                         for chunk in r.iter_content(chunk_size=8192):
#                             if chunk:
#                                 f.write(chunk)
#             except Exception as exc:
#                 print(f"[ERR] Download failed: {exc}")
#                 return None
#         else:
#             print(f"[SKP] {fname.name} already present.")
#         # open & standardize
#         try:
#             ds = xr.open_dataset(fname)
            
#             var = keep_vars # next((v for v in keep_vars if v in ds.data_vars), None)
#             if var is None:
#                 raise KeyError(f"None of {keep_vars} found in {fname.name}; available: {list(ds.data_vars)}")
#             # make a 1-step time axis for this month
#             da = ds[var].assign_coords(time=[pd.to_datetime(f"{year}-{month:02d}-01")]).astype("float32")
#             # spatial clip if requested
#             if area:
#                 n, w, s, e = area
#                 # TAMSAT is lat/lon naming; ensure correct orientation
#                 latn = "lat" if "lat" in da.coords else "latitude"
#                 lonn = "lon" if "lon" in da.coords else "longitude"
#                 da = da.where(
#                     (da[latn] <= n) & (da[latn] >= s) & (da[lonn] >= w) & (da[lonn] <= e),
#                     drop=True,
#                 )
#             da.name = std_name
#             return da
#         except Exception as exc:
#             print(f"[ERR] Failed to open dataset: {exc}")
#             return None


#     def WAS_Download_TAMSAT_Daily(
#             self,
#             dir_to_save: Union[str, Path],
#             product: Literal["rfe", "soilmoisture"] = "rfe",
#             variables: Optional[Sequence[str]] = None,
#             year_start: str = "1983",
#             year_end: str = "2024",
#             area: Optional[List[float]] = None,
#             version: Optional[str] = None,
#             force_download: bool = False,
#         ) -> Path:
#             """
#             Download TAMSAT daily data (RFE v3.1 or Soil Moisture v2.3.1) and combine into a single NetCDF file.

#             Parameters
#             ----------
#             dir_to_save : str | Path
#                 Directory where daily files and the combined output will be saved.
#             product : {"rfe", "soil_moisture"}, default "rfe"
#                 Dataset to download: "rfe" (precipitation) or "soil_moisture".
#             variables : sequence of str, optional
#                 Names of variables to extract from NetCDF. If None, chosen by product.
#                 - rfe: defaults to ("rfe",)
#                 - soil_moisture: defaults to ("sm",)
#             start_date : str, default "1983-01-01"
#                 Start date in "YYYY-MM-DD" format.
#             end_date : str, default "2025-10-20"
#                 End date in "YYYY-MM-DD" format (inclusive).
#             area : list[float], optional
#                 Bounding box [north, west, south, east] in degrees.
#             version : str, optional
#                 Product version. Defaults to:
#                 - rfe: "v3.1"
#                 - soil_moisture: "v2.3.1"
#             force_download : bool, default False
#                 Re-download daily files even if present locally.

#             Returns
#             -------
#             Path
#                 Path to the combined daily NetCDF file.
#             """
#             dir_to_save = Path(dir_to_save)
#             dir_to_save.mkdir(parents=True, exist_ok=True)
#             start_date = f"{year_start}-01-01"
#             end_date = f"{year_end}-12-31" if len(str(year_end)) == 4 else f"{year_end.year:04d}-{year_end.month:02d}-{year_end.day:02d}"
#             yr_end = year_end if len(str(year_end)) == 4 else year_end.year
        
#             # ---- sensible defaults by product ----
#             if product == "rfe":
#                 variables = variables or ("rfe",)
#                 version = version or "v3.1"
#                 prefix = "rfe"
#                 std_name = "precip"
#             elif product == "soilmoisture":
#                 variables = variables or ("smc_avail_top",)
#                 version = version or "v2.3.1"
#                 prefix = "sm"
#                 std_name = "soil_moisture"
#             else:
#                 raise ValueError("product must be 'rfe' or 'soilmoisture'")
#             # ---- validate inputs ----
#             start_dt = pd.to_datetime(start_date)
#             end_dt = pd.to_datetime(end_date)
#             if start_dt > end_dt:
#                 raise ValueError("start_date must be <= end_date.")
#             dates = pd.date_range(start_dt, end_dt, freq="D")
#             area_tuple: Optional[Tuple[float, float, float, float]] = (
#                 tuple(map(float, area)) if area else None
#             )
#             # ---- output filename ----
#             sdate_str = start_date.replace("-", "")
#             edate_str = end_date.replace("-", "")
#             out_nc = dir_to_save /  f"Daily_PRCP_{year_start}_{yr_end}.nc" # f"Obs_{product.upper()}_daily_{sdate_str}_{edate_str}.nc"
#             if out_nc.exists() and not force_download:
#                 print(f"[INFO] {out_nc} already exists – skip download.")
#                 return out_nc
#             else:
#                 # ---- build daily series ----
#                 daily_list: List[xr.DataArray] = []
#                 for date in dates:
#                     y = date.year
#                     m = date.month
#                     d = date.day
#                     da = self._fetch_tamsat_daily(
#                         product=product,
#                         version=version,
#                         year=y,
#                         month=m,
#                         day=d,
#                         dir_to_save=dir_to_save,
#                         force_download=force_download,
#                         area=area_tuple,
#                         keep_vars=variables,
#                         std_name=std_name,
#                         prefix=prefix,
#                     )
#                     if da is not None:
#                         daily_list.append(da)
#                 if not daily_list:
#                     raise RuntimeError("No TAMSAT files were downloaded or opened.")
#                 # ---- concat days and save ----
#                 da_all = xr.concat(daily_list, dim="time")
#                 ds_out = da_all.to_dataset(name=std_name)
    
#                 # harmonize dims if needed
#                 rename_dict = {k: v for k, v in {"lon": "X", "lat": "Y", "time": "T"}.items() if k in ds_out.dims}
#                 ds_out = ds_out.rename(rename_dict)
#                 ds_out.to_netcdf(out_nc)
#                 print(f"[INFO] Saved daily {product.upper()} data → {out_nc}")
#                 return out_nc
#     def _fetch_tamsat_daily(
#         self,
#         product: Literal["rfe", "soilmoisture"],
#         version: str,
#         year: int,
#         month: int,
#         day: int,
#         dir_to_save: Path,
#         force_download: bool,
#         area: Optional[Tuple[float, float, float, float]],
#         keep_vars: Sequence[str],
#         std_name: str,
#         prefix: str,
#     ) -> Optional[xr.DataArray]:
#         """
#         Download & open a single daily TAMSAT file (RFE v3.1 or Soil Moisture v2.3.1),
#         clip to bbox if provided, and return a standardized DataArray.
#         Returns
#         -------
#         xr.DataArray | None
#         """
#         if product == "soilmoisture":
#             product_ = "soil_moisture"
#         else:
#             product_ = product
#         base = (
#             f"https://gws-access.jasmin.ac.uk/public/tamsat/{product_}/data/"
#             f"{version}/daily/{{year}}/{{month:02d}}/{prefix}{{year}}_{{month:02d}}_{{day:02d}}.{version}.nc"
#         )
#         url = base.format(year=year, month=month, day=day)
#         fname = dir_to_save / url.split("/")[-1]
#         # download if needed
#         if not fname.exists() or force_download:
#             try:
#                 print(f"[DL ] {url}")
#                 with requests.get(url, stream=True, timeout=180) as r:
#                     r.raise_for_status()
#                     with open(fname, "wb") as f:
#                         for chunk in r.iter_content(8192):
#                             if chunk:
#                                 f.write(chunk)
#             except Exception as exc:
#                 print(f"[ERR] Download failed: {exc}")
#                 return None
#         else:
#             print(f"[SKP] {fname.name} already present.")
#         # open & standardize
#         try:
#             ds = xr.open_dataset(fname)
#             var = keep_vars #next((v for v in keep_vars if v in ds.data_vars), None)
#             if var is None:
#                 raise KeyError(f"None of {keep_vars} found in {fname.name}; available: {list(ds.data_vars)}")
#             # make- a 1-step time axis for this day
#             da = ds[var].assign_coords(time=[pd.to_datetime(f"{year}-{month:02d}-{day:02d}")]).astype("float32")
#             # spatial clip if requested
#             if area:
#                 n, w, s, e = area
#                 # TAMSAT is lat/lon naming; ensure correct orientation
#                 latn = "lat" if "lat" in da.coords else "latitude"
#                 lonn = "lon" if "lon" in da.coords else "longitude"
#                 da = da.where(
#                     (da[latn] <= n) & (da[latn] >= s) & (da[lonn] >= w) & (da[lonn] <= e),
#                     drop=True,
#                 )
#             da.name = std_name
#             return da
#         except Exception as exc:
#             print(f"[ERR] Failed to open dataset: {exc}")
#             return None
    
    
    # def WAS_Download_CHIRPSv3_Daily(
    #     self,
    #     dir_to_save,
    #     year_start,
    #     year_end,
    #     blend_type="ERA5",
    #     area=None,
    #     force_download=False,
    #     max_retries=3,
    #     retry_delay=5,
    #     cleanup=True,
    #     allow_incomplete=False,
    # ):
    #     """
    #     Download CHIRPS v3 daily precipitation and save a WAS-style NetCDF.
    
    #     allow_incomplete=False (default): the final NetCDF is written only when
    #     every requested year is complete; otherwise a RuntimeError asks you to
    #     re-run (all intermediate files are kept, so the re-run only fetches what
    #     is missing). Set allow_incomplete=True to write the final file from the
    #     complete years only (previous behaviour).
    
    #     Resumable: daily .tif files already on disk are reused; only missing days
    #     are downloaded. Intermediate files are deleted only once the final NetCDF
    #     has been written (if cleanup=True).
    
    #     Output:
    #         variable: PRCP
    #         dims: T, Y, X
    #         units: mm/day
    #     """
    #     from pathlib import Path
    #     import calendar
    #     import gc
    #     import os
    
    #     import pandas as pd
    #     import rioxarray as rioxr
    #     import xarray as xr
    
    #     dir_to_save = Path(dir_to_save)
    #     dir_to_save.mkdir(parents=True, exist_ok=True)
    #     area = tuple(area)
    #     blend_map = {"ERA5": "rnl", "IMERGlate-v07": "sat", "IMERG": "sat", "SAT": "sat"}
    #     if blend_type not in blend_map:
    #         raise ValueError("blend_type must be 'ERA5' or 'IMERGlate-v07'.")
    
    #     product_tag = blend_map[blend_type]
    #     output_path = dir_to_save / f"Daily_PRCP_CHIRPSv3_{product_tag}_{year_start}_{year_end}.nc"
    #     if output_path.exists() and not force_download:
    #         print(f"{output_path} already exists. Skipping download.")
    #         return output_path
    
    #     def _open_chirps_tif(path, date):
    #         da = rioxr.open_rasterio(path, masked=True).squeeze(drop=True)
    #         if area is not None:
    #             north, west, south, east = area
    #             da = da.rio.clip_box(minx=west, miny=south, maxx=east, maxy=north)
    #         da = da.rename({name: {"x": "X", "y": "Y"}.get(name, name) for name in da.dims})
    #         if "spatial_ref" in da.coords:
    #             da = da.drop_vars("spatial_ref", errors="ignore")
    #         da = da.expand_dims(T=[pd.Timestamp(date)])
    #         da.name = "PRCP"
    #         da.attrs["units"] = "mm/day"
    #         da.attrs["source"] = "CHIRPS v3"
    #         if "Y" in da.coords:
    #             da = da.sortby("Y")
    #         if "X" in da.coords:
    #             da = da.sortby("X")
    #         return da
    
    #     yearly_paths = []
    #     incomplete_years = []
    #     for year in range(int(year_start), int(year_end) + 1):
    #         year_path = dir_to_save / f"tmp_CHIRPSv3_{product_tag}_{year}.nc"
    #         if year_path.exists() and not force_download:
    #             print(f"[RESUME] Year checkpoint found, reusing {year_path.name}")
    #             yearly_paths.append(year_path)
    #             continue
    
    #         daily_arrays = []
    #         year_tifs = []
    #         all_ok = True
    #         n_reused = 0
    #         for month in range(1, 13):
    #             ndays = calendar.monthrange(year, month)[1]
    #             for day in range(1, ndays + 1):
    #                 fname = f"chirps-v3.0.{product_tag}.{year}.{month:02d}.{day:02d}.tif"
    #                 url = f"https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/{product_tag}/{year}/{fname}"
    #                 tif_path = dir_to_save / fname
    
    #                 if tif_path.exists() and not force_download:
    #                     n_reused += 1
    #                 else:
    #                     ok = _download_atomic(url, tif_path, max_retries, retry_delay)
    #                     if not ok:
    #                         all_ok = False
    #                         continue
    
    #                 try:
    #                     da = _open_chirps_tif(tif_path, f"{year}-{month:02d}-{day:02d}").load()
    #                     daily_arrays.append(da)
    #                     year_tifs.append(tif_path)
    #                 except Exception as exc:
    #                     # Likely a truncated file from an old non-atomic run:
    #                     # remove it so the next run re-downloads it cleanly.
    #                     print(f"Failed to open/process {tif_path}: {exc} -> removing corrupt file.")
    #                     _remove_quiet(tif_path)
    #                     all_ok = False
    
    #         if n_reused:
    #             print(f"[RESUME] Year {year}: reused {n_reused} daily file(s) already on disk.")
    
    #         if not daily_arrays:
    #             print(f"No CHIRPS daily data for {year}.")
    #             incomplete_years.append(year)
    #             continue
    
    #         if not all_ok:
    #             # Do NOT delete the tifs: they are the resume state.
    #             print(
    #                 f"Year {year} incomplete ({len(daily_arrays)} day(s) on disk). "
    #                 f"Daily files are kept; re-run to download only the missing days."
    #             )
    #             incomplete_years.append(year)
    #             del daily_arrays
    #             gc.collect()
    #             continue
    
    #         ds_year = xr.concat(daily_arrays, dim="T").to_dataset(name="PRCP").sortby("T")
    #         encoding = {"PRCP": {"zlib": True, "complevel": 4}}
    #         # Atomic write of the yearly checkpoint as well.
    #         year_tmp = year_path.parent / (year_path.name + ".part")
    #         ds_year.to_netcdf(year_tmp, encoding=encoding)
    #         ds_year.close()
    #         os.replace(year_tmp, year_path)
    #         yearly_paths.append(year_path)
    
    #         # Year checkpoint safely on disk -> the daily tifs may now be removed.
    #         if cleanup:
    #             for tif_path in year_tifs:
    #                 _remove_quiet(tif_path)
    
    #         del daily_arrays, ds_year
    #         gc.collect()
    
    #     if not yearly_paths:
    #         raise RuntimeError(
    #             "No complete CHIRPS daily year was processed. "
    #             "Downloaded daily files are kept on disk; re-run to resume."
    #         )
    
    #     if incomplete_years and not allow_incomplete:
    #         raise RuntimeError(
    #             f"Incomplete year(s) {incomplete_years}: the final NetCDF was NOT "
    #             f"written so it does not block the resume. Re-run to download only "
    #             f"the missing days, or pass allow_incomplete=True to write the "
    #             f"final file from complete years only."
    #         )
    
    #     datasets = [xr.open_dataset(path) for path in yearly_paths]
    #     ds_out = xr.concat(datasets, dim="T").sortby("T")
    #     ds_out.attrs["source"] = "CHIRPS v3 daily"
    #     ds_out.attrs["blend_type"] = blend_type
    #     out_tmp = output_path.parent / (output_path.name + ".part")
    #     ds_out.to_netcdf(out_tmp, encoding={"PRCP": {"zlib": True, "complevel": 4}})
    #     for ds in datasets:
    #         ds.close()
    #     ds_out.close()
    #     os.replace(out_tmp, output_path)
    
    #     # Final output safely on disk -> yearly checkpoints may now be removed.
    #     if cleanup:
    #         for path in yearly_paths:
    #             _remove_quiet(path)
    
    #     print(f"Saved CHIRPS daily data to {output_path}")
    #     return output_path


    def WAS_Download_CHIRPSv3_Daily(
        self,
        dir_to_save,
        year_start,
        year_end,
        blend_type="ERA5",
        area=None,
        force_download=False,
        max_retries=3,
        retry_delay=5,
        cleanup=True,
        allow_incomplete=False,
    ):
        """
        Download CHIRPS v3 daily precipitation and save a WAS-style NetCDF.
    
        Memory-optimized: monthly checkpoints + float32 + dask streaming write.
        Resumable: existing .tif and tmp monthly .nc files are reused.
    
        Output: PRCP, dims T/Y/X, mm/day.
        """
        from pathlib import Path
        import calendar
        import gc
        import os
    
        import pandas as pd
        import rioxarray as rioxr
        import xarray as xr
    
        dir_to_save = Path(dir_to_save)
        dir_to_save.mkdir(parents=True, exist_ok=True)
        area = tuple(area)
        blend_map = {"ERA5": "rnl", "IMERGlate-v07": "sat", "IMERG": "sat", "SAT": "sat"}
        if blend_type not in blend_map:
            raise ValueError("blend_type must be 'ERA5' or 'IMERGlate-v07'.")
    
        product_tag = blend_map[blend_type]
        output_path = dir_to_save / f"Daily_PRCP_{year_start}_{year_end}.nc"
        if output_path.exists() and not force_download:
            print(f"{output_path} already exists. Skipping download.")
            return output_path
    
        def _open_chirps_tif(path, date):
            da = rioxr.open_rasterio(path, masked=True).squeeze(drop=True)
            if area is not None:
                north, west, south, east = area
                da = da.rio.clip_box(minx=west, miny=south, maxx=east, maxy=north)
            da = da.rename({name: {"x": "X", "y": "Y"}.get(name, name) for name in da.dims})
            if "spatial_ref" in da.coords:
                da = da.drop_vars("spatial_ref", errors="ignore")
            # masked=True promotes to float64 -> cast back to float32 immediately.
            da = da.astype("float32")
            da = da.expand_dims(T=[pd.Timestamp(date)])
            da.name = "PRCP"
            da.attrs["units"] = "mm/day"
            if "Y" in da.coords:
                da = da.sortby("Y")
            if "X" in da.coords:
                da = da.sortby("X")
            return da
    
        monthly_paths = []
        incomplete_months = []
    
        for year in range(int(year_start), int(year_end) + 1):
            for month in range(1, 13):
                month_path = dir_to_save / f"tmp_CHIRPSv3_{product_tag}_{year}{month:02d}.nc"
                if month_path.exists() and not force_download:
                    monthly_paths.append(month_path)
                    continue
    
                ndays = calendar.monthrange(year, month)[1]
                daily_arrays = []
                month_tifs = []
                all_ok = True
                n_reused = 0
    
                for day in range(1, ndays + 1):
                    fname = f"chirps-v3.0.{product_tag}.{year}.{month:02d}.{day:02d}.tif"
                    url = f"https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/{product_tag}/{year}/{fname}"
                    tif_path = dir_to_save / fname
    
                    if tif_path.exists() and not force_download:
                        n_reused += 1
                    else:
                        if not _download_atomic(url, tif_path, max_retries, retry_delay):
                            all_ok = False
                            continue
    
                    try:
                        daily_arrays.append(_open_chirps_tif(tif_path, f"{year}-{month:02d}-{day:02d}").load())
                        month_tifs.append(tif_path)
                    except Exception as exc:
                        print(f"Failed to open {tif_path}: {exc} -> removing corrupt file.")
                        _remove_quiet(tif_path)
                        all_ok = False
    
                if n_reused:
                    print(f"[RESUME] {year}-{month:02d}: reused {n_reused} daily file(s).")
    
                if not daily_arrays or not all_ok:
                    print(f"Month {year}-{month:02d} incomplete; daily files kept for resume.")
                    incomplete_months.append(f"{year}-{month:02d}")
                    del daily_arrays
                    gc.collect()
                    continue
    
                ds_month = xr.concat(daily_arrays, dim="T").to_dataset(name="PRCP").sortby("T")
                month_tmp = month_path.parent / (month_path.name + ".part")
                ds_month.to_netcdf(month_tmp, encoding={"PRCP": {"zlib": True, "complevel": 4}})
                ds_month.close()
                os.replace(month_tmp, month_path)
                monthly_paths.append(month_path)
    
                # Monthly checkpoint safely on disk -> tifs of this month removable.
                if cleanup:
                    for tif_path in month_tifs:
                        _remove_quiet(tif_path)
    
                del daily_arrays, ds_month
                gc.collect()
    
        if not monthly_paths:
            raise RuntimeError("No complete CHIRPS month was processed; re-run to resume.")
    
        if incomplete_months and not allow_incomplete:
            raise RuntimeError(
                f"Incomplete month(s) {incomplete_months}: final NetCDF NOT written. "
                f"Re-run to fetch only missing days, or pass allow_incomplete=True."
            )
    
        # LAZY final assembly: dask streams the write, nothing is fully loaded.
        print(f"Assembling {len(monthly_paths)} monthly checkpoint(s) lazily with dask...")
        ds_out = xr.open_mfdataset(
            sorted(monthly_paths),
            combine="nested",
            concat_dim="T",
            chunks={"T": 31},
            parallel=False,
        ).sortby("T")
        ds_out.attrs["source"] = "CHIRPS v3 daily"
        ds_out.attrs["blend_type"] = blend_type
    
        out_tmp = output_path.parent / (output_path.name + ".part")
        encoding = {"PRCP": {"zlib": True, "complevel": 4, "dtype": "float32"}}
        delayed = ds_out.to_netcdf(out_tmp, encoding=encoding, compute=False)
        try:
            from dask.diagnostics import ProgressBar
            with ProgressBar():
                delayed.compute()
        except ImportError:
            delayed.compute()
        ds_out.close()
        os.replace(out_tmp, output_path)
    
        if cleanup:
            for path in monthly_paths:
                _remove_quiet(path)
    
        print(f"Saved CHIRPS daily data to {output_path}")
        return output_path
    
    def WAS_Download_CHIRPSv3_Seasonal(
        self,
        dir_to_save,
        variables=("PRCP",),
        year_start=1981,
        year_end=2024,
        region="africa",
        area=None,
        season_months=["03", "04", "05"],
        force_download=False,
        cleanup=True,
        max_retries=3,
        retry_delay=5,
        allow_incomplete=False,
    ):
        """
        Download CHIRPS v3 monthly precipitation and aggregate by season.
    
        Resumable: monthly .tif files already on disk are reused; cleanup is
        deferred until the final seasonal NetCDF is written.
    
        Output:
            variable: PRCP
            dims: T, Y, X
            units: mm/season
    
        Cross-year seasons are handled with the first season month as pivot.
        For NDJ, season_year=2020 means Nov-Dec 2020 + Jan 2021.
        """
        from pathlib import Path
        import calendar
        import os
    
        import pandas as pd
        import rioxarray as rioxr
        import xarray as xr
    
        if isinstance(variables, str):
            variables = (variables,)
        if "PRCP" not in variables:
            raise ValueError("CHIRPS seasonal currently supports PRCP only.")
    
        dir_to_save = Path(dir_to_save)
        dir_to_save.mkdir(parents=True, exist_ok=True)
        
        area = tuple(area)
        season_months = tuple(season_months)
        
        season_months_int = [int(m) for m in season_months]
        pivot = season_months_int[0]
        season_str = "".join(calendar.month_abbr[m] for m in season_months_int)
        out_nc = dir_to_save / f"Obs_PRCP_{year_start}_{year_end}_{season_str}.nc"
        if out_nc.exists() and not force_download:
            print(f"[INFO] {out_nc} already exists. Skipping.")
            return out_nc
    
        def _season_calendar_months(season_year):
            return [
                (season_year if month >= pivot else season_year + 1, month)
                for month in season_months_int
            ]
    
        def _representative_time(season_year):
            mid_month = season_months_int[len(season_months_int) // 2]
            cal_year = season_year if mid_month >= pivot else season_year + 1
            return pd.Timestamp(cal_year, mid_month, 1)
    
        used_paths = set()
    
        def _fetch_month(year, month):
            fname = f"chirps-v3.0.{year}.{month:02d}.tif"
            url = f"https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/{region}/tifs/{fname}"
            tif_path = dir_to_save / fname
    
            if not tif_path.exists() or force_download:
                ok = _download_atomic(url, tif_path, max_retries, retry_delay)
                if not ok:
                    return None
    
            try:
                da = rioxr.open_rasterio(tif_path, masked=True).squeeze(drop=True)
                if area is not None:
                    north, west, south, east = area
                    da = da.rio.clip_box(minx=west, miny=south, maxx=east, maxy=north)
                da = da.rename({name: {"x": "X", "y": "Y"}.get(name, name) for name in da.dims})
                if "spatial_ref" in da.coords:
                    da = da.drop_vars("spatial_ref", errors="ignore")
                da = da.expand_dims(month_time=[pd.Timestamp(year, month, 1)])
                da.name = "PRCP"
                da.attrs["units"] = "mm/month"
                if "Y" in da.coords:
                    da = da.sortby("Y")
                if "X" in da.coords:
                    da = da.sortby("X")
                da = da.load()
                used_paths.add(tif_path)  # deletion deferred to the very end
                return da
            except Exception as exc:
                print(f"[ERR] Failed to open/process {tif_path}: {exc} -> removing corrupt file.")
                _remove_quiet(tif_path)
                return None
    
        seasonal_arrays = []
        all_ok = True
        for season_year in range(int(year_start), int(year_end) + 1):
            monthly = []
            for cal_year, month in _season_calendar_months(season_year):
                da = _fetch_month(cal_year, month)
                if da is None:
                    all_ok = False
                    print(f"Incomplete CHIRPS season {season_year}: missing {cal_year}-{month:02d}.")
                    continue
                monthly.append(da)
    
            if len(monthly) != len(season_months_int):
                continue
    
            da_season = xr.concat(monthly, dim="month_time").sum("month_time", keep_attrs=True)
            da_season = da_season.expand_dims(T=[_representative_time(season_year)])
            da_season.name = "PRCP"
            da_season.attrs["units"] = "mm"
            seasonal_arrays.append(da_season)
    
        if not seasonal_arrays:
            raise RuntimeError(
                "No complete CHIRPS seasonal data were processed. "
                "Downloaded monthly files are kept on disk; re-run to resume."
            )
        if not all_ok:
            if not allow_incomplete:
                raise RuntimeError(
                    "Incomplete CHIRPS seasonal download: the final NetCDF was NOT "
                    "written so it does not block the resume. Monthly files already "
                    "downloaded are kept; re-run to fetch only the missing months, "
                    "or pass allow_incomplete=True to write the complete seasons only."
                )
            print("[WARNING] Some seasons were incomplete and skipped (allow_incomplete=True).")
    
        ds_out = xr.concat(seasonal_arrays, dim="T").to_dataset(name="PRCP").sortby("T")
        ds_out.attrs["source"] = "CHIRPS v3 monthly"
        ds_out.attrs["season"] = season_str
        ds_out.attrs["season_months"] = ",".join(f"{m:02d}" for m in season_months_int)
        out_tmp = out_nc.parent / (out_nc.name + ".part")
        ds_out.to_netcdf(out_tmp, encoding={"PRCP": {"zlib": True, "complevel": 4}})
        ds_out.close()
        os.replace(out_tmp, out_nc)
    
        # Final output safely on disk -> monthly tifs may now be removed.
        if cleanup:
            for path in used_paths:
                _remove_quiet(path)
    
        print(f"[INFO] Saved seasonal CHIRPS data to {out_nc}")
        return out_nc
    
    
    def WAS_Download_TAMSAT_Seasonal(
        self,
        dir_to_save,
        product="rfe",
        variables=None,
        year_start=1983,
        year_end=2025,
        area=None,
        season_months=["03", "04", "05"],
        version=None,
        force_download=False,
        agg=None,
        cleanup=False,
        max_retries=3,
        retry_delay=5,
        allow_incomplete=False,
    ):
        """
        Download and aggregate TAMSAT monthly data by season.
    
        Resumable: monthly .nc files already on disk are reused; corrupt files
        are removed and re-downloaded on the next run; cleanup is deferred until
        the final seasonal NetCDF is written.
    
        product:
            "rfe" -> precipitation, default variable "rfe", seasonal sum.
            "soilmoisture" or "soil_moisture" -> default variable "smc_avail_top",
            seasonal mean.
    
        Output dims:
            T, Y, X
        """
        from pathlib import Path
        import calendar
        import os
    
        import pandas as pd
        import xarray as xr
    
        dir_to_save = Path(dir_to_save)
        dir_to_save.mkdir(parents=True, exist_ok=True)
        
        area = tuple(area)
        season_months = tuple(season_months) 
        
        product_key = "soilmoisture" if product in ["soilmoisture", "soil_moisture"] else product
        if product_key == "rfe":
            variables = _as_tuple(variables, default=("rfe",))
            version = version or "v3.1"
            agg = agg or "sum"
            std_name = "PRCP"
            url_product = "rfe"
            prefix = "rfe"
        elif product_key == "soilmoisture":
            variables = _as_tuple(variables, default=("smc_avail_top",))
            version = version or "v2.3.1"
            agg = agg or "mean"
            std_name = "soil_moisture"
            url_product = "soil_moisture"
            prefix = "sm"
        else:
            raise ValueError("product must be 'rfe', 'soilmoisture', or 'soil_moisture'.")
    
        season_months_int = [int(m) for m in season_months]
        pivot = season_months_int[0]
        season_str = "".join(calendar.month_abbr[m] for m in season_months_int)

        if product_key == "soilmoisture":
            out_nc = dir_to_save / f"Obs_SMC_{year_start}_{year_end}_{season_str}.nc"
        else:
            out_nc = dir_to_save / f"Obs_PRCP_{year_start}_{year_end}_{season_str}.nc"
            
        if out_nc.exists() and not force_download:
            print(f"[INFO] {out_nc} already exists. Skipping.")
            return out_nc
    
        def _season_calendar_months(season_year):
            return [
                (season_year if month >= pivot else season_year + 1, month)
                for month in season_months_int
            ]
    
        def _representative_time(season_year):
            mid_month = season_months_int[len(season_months_int) // 2]
            cal_year = season_year if mid_month >= pivot else season_year + 1
            return pd.Timestamp(cal_year, mid_month, 1)
    
        used_paths = set()
    
        def _fetch_month(year, month):
            url = (
                f"https://gws-access.jasmin.ac.uk/public/tamsat/{url_product}/data/"
                f"{version}/monthly/{year}/{month:02d}/{prefix}{year}_{month:02d}.{version}.nc"
            )
            path = dir_to_save / url.split("/")[-1]
            if not path.exists() or force_download:
                ok = _download_atomic(url, path, max_retries, retry_delay)
                if not ok:
                    return None
    
            try:
                with xr.open_dataset(path) as ds:
                    var_name = _select_variable(ds, variables)
                    da = ds[var_name].squeeze(drop=True).astype("float32").load()
                da = _standardize_latlon_da(da, area)
                da = da.expand_dims(month_time=[pd.Timestamp(year, month, 1)])
                da.name = std_name
                used_paths.add(path)  # deletion deferred to the very end
                return da
            except Exception as exc:
                print(f"[ERR] Failed to open/process {path}: {exc} -> removing corrupt file.")
                _remove_quiet(path)
                return None
    
        seasonal = []
        incomplete = False
        for season_year in range(int(year_start), int(year_end) + 1):
            monthly = []
            for cal_year, month in _season_calendar_months(season_year):
                da = _fetch_month(cal_year, month)
                if da is not None:
                    monthly.append(da)
    
            if len(monthly) != len(season_months_int):
                print(f"[WARNING] Skipping incomplete TAMSAT season {season_year}.")
                incomplete = True
                continue
    
            stack = xr.concat(monthly, dim="month_time")
            if agg == "sum":
                da_season = stack.sum("month_time", keep_attrs=True)
            elif agg == "mean":
                da_season = stack.mean("month_time", keep_attrs=True)
            else:
                raise ValueError("agg must be 'sum' or 'mean'.")
            da_season = da_season.expand_dims(T=[_representative_time(season_year)])
            da_season.name = std_name
            seasonal.append(da_season)
    
        if not seasonal:
            raise RuntimeError(
                "No complete TAMSAT seasons were processed. "
                "Downloaded monthly files are kept on disk; re-run to resume."
            )
        if incomplete:
            if not allow_incomplete:
                raise RuntimeError(
                    "Incomplete TAMSAT seasonal download: the final NetCDF was NOT "
                    "written so it does not block the resume. Monthly files already "
                    "downloaded are kept; re-run to fetch only the missing months, "
                    "or pass allow_incomplete=True to write the complete seasons only."
                )
            print("[WARNING] Some TAMSAT seasons were incomplete and skipped (allow_incomplete=True).")
    
        ds_out = xr.concat(seasonal, dim="T").to_dataset(name=std_name).sortby("T")
        ds_out.attrs["source"] = "TAMSAT"
        ds_out.attrs["product"] = product_key
        ds_out.attrs["season"] = season_str
        ds_out.attrs["season_months"] = ",".join(f"{m:02d}" for m in season_months_int)
        out_tmp = out_nc.parent / (out_nc.name + ".part")
        ds_out.to_netcdf(out_tmp, encoding={std_name: {"zlib": True, "complevel": 4}})
        ds_out.close()
        os.replace(out_tmp, out_nc)
    
        if cleanup:
            for path in used_paths:
                _remove_quiet(path)
    
        print(f"[INFO] Saved seasonal TAMSAT data to {out_nc}")
        return out_nc
    
    
    # def WAS_Download_TAMSAT_Daily(
    #     self,
    #     dir_to_save,
    #     product="rfe",
    #     variables=None,
    #     year_start=1983,
    #     year_end=2024,
    #     area=None,
    #     version=None,
    #     force_download=False,
    #     cleanup=False,
    #     max_retries=3,
    #     retry_delay=5,
    #     allow_incomplete=False,
    # ):
    #     """
    #     Download TAMSAT daily data and save a WAS-style NetCDF.
    
    #     Resumable: daily .nc files already on disk are reused; corrupt files are
    #     removed and re-downloaded on the next run; cleanup is deferred until the
    #     final NetCDF is written.
    
    #     Output dims:
    #         T, Y, X
    #     """
    #     from pathlib import Path
    #     import os
    
    #     import pandas as pd
    #     import xarray as xr
    
    #     dir_to_save = Path(dir_to_save)
    #     dir_to_save.mkdir(parents=True, exist_ok=True)
    #     area = tuple(area)
 
        
    #     product_key = "soilmoisture" if product in ["soilmoisture", "soil_moisture"] else product
    #     if product_key == "rfe":
    #         variables = _as_tuple(variables, default=("rfe",))
    #         version = version or "v3.1"
    #         std_name = "PRCP"
    #         url_product = "rfe"
    #         prefix = "rfe"
    #     elif product_key == "soilmoisture":
    #         variables = _as_tuple(variables, default=("smc_avail_top",))
    #         version = version or "v2.3.1"
    #         std_name = "soil_moisture"
    #         url_product = "soil_moisture"
    #         prefix = "sm"
    #     else:
    #         raise ValueError("product must be 'rfe', 'soilmoisture', or 'soil_moisture'.")
    
    #     start = pd.Timestamp(int(year_start), 1, 1) if str(year_start).isdigit() else pd.Timestamp(year_start)
    #     end = pd.Timestamp(int(year_end), 12, 31) if str(year_end).isdigit() else pd.Timestamp(year_end)
    #     if start > end:
    #         raise ValueError("year_start/start date must be <= year_end/end date.")
    
    #     out_nc = dir_to_save / f"Daily_TAMSAT_{product_key}_{start:%Y%m%d}_{end:%Y%m%d}.nc"
    #     if out_nc.exists() and not force_download:
    #         print(f"[INFO] {out_nc} already exists. Skipping.")
    #         return out_nc
    
    #     used_paths = set()
    
    #     def _fetch_day(date):
    #         year, month, day = date.year, date.month, date.day
    #         url = (
    #             f"https://gws-access.jasmin.ac.uk/public/tamsat/{url_product}/data/"
    #             f"{version}/daily/{year}/{month:02d}/{prefix}{year}_{month:02d}_{day:02d}.{version}.nc"
    #         )
    #         path = dir_to_save / url.split("/")[-1]
    #         if not path.exists() or force_download:
    #             ok = _download_atomic(url, path, max_retries, retry_delay)
    #             if not ok:
    #                 return None
    
    #         try:
    #             with xr.open_dataset(path) as ds:
    #                 var_name = _select_variable(ds, variables)
    #                 da = ds[var_name].squeeze(drop=True).astype("float32").load()
    #             da = _standardize_latlon_da(da, area)
    #             da = da.expand_dims(T=[date])
    #             da.name = std_name
    #             used_paths.add(path)  # deletion deferred to the very end
    #             return da
    #         except Exception as exc:
    #             print(f"[ERR] Failed to open/process {path}: {exc} -> removing corrupt file.")
    #             _remove_quiet(path)
    #             return None
    
    #     daily = []
    #     missing = []
    #     for date in pd.date_range(start, end, freq="D"):
    #         da = _fetch_day(date)
    #         if da is not None:
    #             daily.append(da)
    #         else:
    #             missing.append(date)
    
    #     if not daily:
    #         raise RuntimeError(
    #             "No TAMSAT daily files were processed. "
    #             "Downloaded files are kept on disk; re-run to resume."
    #         )
    
    #     if missing:
    #         msg = (
    #             f"{len(missing)} day(s) missing "
    #             f"(first: {missing[0]:%Y-%m-%d}, last: {missing[-1]:%Y-%m-%d})."
    #         )
    #         if not allow_incomplete:
    #             raise RuntimeError(
    #                 f"Incomplete TAMSAT daily download: {msg} The final NetCDF was "
    #                 f"NOT written so it does not block the resume. Files already "
    #                 f"downloaded are kept; re-run to fetch only the missing days, "
    #                 f"or pass allow_incomplete=True to write the available days only."
    #             )
    #         print(f"[WARNING] {msg} Writing available days only (allow_incomplete=True).")
    
    #     ds_out = xr.concat(daily, dim="T").to_dataset(name=std_name).sortby("T")
    #     ds_out.attrs["source"] = "TAMSAT"
    #     ds_out.attrs["product"] = product_key
    #     out_tmp = out_nc.parent / (out_nc.name + ".part")
    #     ds_out.to_netcdf(out_tmp, encoding={std_name: {"zlib": True, "complevel": 4}})
    #     ds_out.close()
    #     os.replace(out_tmp, out_nc)
    
    #     if cleanup:
    #         for path in used_paths:
    #             _remove_quiet(path)
    
    #     print(f"[INFO] Saved daily TAMSAT data to {out_nc}")
    #     return out_nc
    

    def WAS_Download_TAMSAT_Daily(
        self,
        dir_to_save,
        product="rfe",
        variables=None,
        year_start=1983,
        year_end=2024,
        area=None,
        version=None,
        force_download=False,
        cleanup=False,
        max_retries=3,
        retry_delay=5,
        allow_incomplete=False,
    ):
        """
        Download TAMSAT daily data and save a WAS-style NetCDF.
    
        Memory-optimized: monthly checkpoints + dask streaming write. The old
        version accumulated the ENTIRE period in memory before concat, which
        caused out-of-memory failures for long periods / large domains.
    
        Output dims: T, Y, X.
        """
        from pathlib import Path
        import calendar
        import gc
        import os
    
        import pandas as pd
        import xarray as xr
    
        dir_to_save = Path(dir_to_save)
        dir_to_save.mkdir(parents=True, exist_ok=True)
        area = tuple(area)
    
        product_key = "soilmoisture" if product in ["soilmoisture", "soil_moisture"] else product
        if product_key == "rfe":
            variables = _as_tuple(variables, default=("rfe",))
            version = version or "v3.1"
            std_name = "PRCP"
            url_product = "rfe"
            prefix = "rfe"
        elif product_key == "soilmoisture":
            variables = _as_tuple(variables, default=("smc_avail_top",))
            version = version or "v2.3.1"
            std_name = "soil_moisture"
            url_product = "soil_moisture"
            prefix = "sm"
        else:
            raise ValueError("product must be 'rfe', 'soilmoisture', or 'soil_moisture'.")
    
        start = pd.Timestamp(int(year_start), 1, 1) if str(year_start).isdigit() else pd.Timestamp(year_start)
        end = pd.Timestamp(int(year_end), 12, 31) if str(year_end).isdigit() else pd.Timestamp(year_end)
        if start > end:
            raise ValueError("year_start/start date must be <= year_end/end date.")
    
        out_nc = dir_to_save / f"Daily_PRCP_{year_start}_{year_end}.nc"
        if out_nc.exists() and not force_download:
            print(f"[INFO] {out_nc} already exists. Skipping.")
            return out_nc
    
        def _fetch_day(date):
            year, month, day = date.year, date.month, date.day
            url = (
                f"https://gws-access.jasmin.ac.uk/public/tamsat/{url_product}/data/"
                f"{version}/daily/{year}/{month:02d}/{prefix}{year}_{month:02d}_{day:02d}.{version}.nc"
            )
            path = dir_to_save / url.split("/")[-1]
            if not path.exists() or force_download:
                if not _download_atomic(url, path, max_retries, retry_delay):
                    return None, None
    
            try:
                with xr.open_dataset(path) as ds:
                    var_name = _select_variable(ds, variables)
                    da = ds[var_name].squeeze(drop=True).astype("float32").load()
                da = _standardize_latlon_da(da, area)
                da = da.expand_dims(T=[date])
                da.name = std_name
                return da, path
            except Exception as exc:
                print(f"[ERR] Failed to open {path}: {exc} -> removing corrupt file.")
                _remove_quiet(path)
                return None, None
    
        # Build the list of (year, month) chunks within [start, end].
        month_starts = pd.date_range(start.normalize().replace(day=1), end, freq="MS")
        if len(month_starts) == 0:
            month_starts = pd.DatetimeIndex([start.normalize().replace(day=1)])
    
        monthly_paths = []
        incomplete_months = []
    
        for month_start in month_starts:
            year, month = month_start.year, month_start.month
            month_path = dir_to_save / f"tmp_TAMSAT_{product_key}_{year}{month:02d}.nc"
            if month_path.exists() and not force_download:
                monthly_paths.append(month_path)
                continue
    
            last_day = calendar.monthrange(year, month)[1]
            d0 = max(start, pd.Timestamp(year, month, 1))
            d1 = min(end, pd.Timestamp(year, month, last_day))
    
            daily = []
            month_files = []
            missing = []
            for date in pd.date_range(d0, d1, freq="D"):
                da, path = _fetch_day(date)
                if da is not None:
                    daily.append(da)
                    month_files.append(path)
                else:
                    missing.append(date)
    
            if not daily or missing:
                print(
                    f"[WARN] Month {year}-{month:02d} incomplete "
                    f"({len(missing)} day(s) missing); files kept for resume."
                )
                incomplete_months.append(f"{year}-{month:02d}")
                del daily
                gc.collect()
                continue
    
            ds_month = xr.concat(daily, dim="T").to_dataset(name=std_name).sortby("T")
            month_tmp = month_path.parent / (month_path.name + ".part")
            ds_month.to_netcdf(month_tmp, encoding={std_name: {"zlib": True, "complevel": 4}})
            ds_month.close()
            os.replace(month_tmp, month_path)
            monthly_paths.append(month_path)
    
            # Monthly checkpoint on disk -> daily source files removable.
            if cleanup:
                for path in month_files:
                    _remove_quiet(path)
    
            del daily, ds_month
            gc.collect()
    
        if not monthly_paths:
            raise RuntimeError("No TAMSAT month was processed; re-run to resume.")
    
        if incomplete_months and not allow_incomplete:
            raise RuntimeError(
                f"Incomplete month(s) {incomplete_months}: final NetCDF NOT written. "
                f"Re-run to fetch only missing days, or pass allow_incomplete=True."
            )
    
        # LAZY final assembly with dask streaming write.
        print(f"[INFO] Assembling {len(monthly_paths)} monthly checkpoint(s) lazily with dask...")
        ds_out = xr.open_mfdataset(
            sorted(monthly_paths),
            combine="nested",
            concat_dim="T",
            chunks={"T": 31},
            parallel=False,
        ).sortby("T")
        ds_out.attrs["source"] = "TAMSAT"
        ds_out.attrs["product"] = product_key
    
        out_tmp = out_nc.parent / (out_nc.name + ".part")
        encoding = {std_name: {"zlib": True, "complevel": 4, "dtype": "float32"}}
        delayed = ds_out.to_netcdf(out_tmp, encoding=encoding, compute=False)
        try:
            from dask.diagnostics import ProgressBar
            with ProgressBar():
                delayed.compute()
        except ImportError:
            delayed.compute()
        ds_out.close()
        os.replace(out_tmp, out_nc)
    
        if cleanup:
            for path in monthly_paths:
                _remove_quiet(path)
    
        print(f"[INFO] Saved daily TAMSAT data to {out_nc}")
        return out_nc
    
    def WAS_Download_GloFAS_historical(
        self,
        dir_to_save,
        variables,
        year_start,
        year_end,
        area,
        system_version="version_5_0",
        product_type="consolidated",
        force_download=False,
        max_retries=3,
        retry_delay=10,
        keep_temporary=False,
        upstream_area_url=None,
        attach_upstream_area_to_q=True,
        ewds_url="https://ewds.climate.copernicus.eu/api",
        ewds_key=None,
        ewds_rc_file=None,
        allow_incomplete=False,
        end_date=None,
        refresh_incomplete=True,
        *,
        attach_static_maps_to_q=True,
        network_area=None,
        static_map_urls=None,
        static_map_files=None,
        force_static_download=None,
        return_paths=False,
    ):

        """Download daily GloFAS and optional static maps through ONE interface.

        Backward-compatible inputs / outputs
        -----------------------------------
        Existing Q/UPAREA calls, Q filenames, incomplete-year policy, monthly
        checkpoints and EWDS authentication are retained. A Q request returns
        its Path, as in the user's original module. A static-only request returns
        None unless return_paths=True. No files are downloaded at import.

        variables : sequence of str
            "GLOFAS.Q", "GLOFAS.UPAREA" (or "GLOFAS.DRAINAGE_AREA"),
            "GLOFAS.LDD", "GLOFAS.CHAN", "GLOFAS.CELLAREA".
            Q is m3 s-1. UPAREA is m2, UPAREA_KM2 is km2, CELLAREA is m2.
            LDD retains PCRaster codes; CHAN is a 0/1 mask. Static fields are Y,X,
            never broadcast over time. Zero Q and zero CHAN remain valid.

        attach_static_maps_to_q : bool, default True
            Attach requested LDD/CHAN/CELLAREA to Q on its native grid. UPAREA
            remains controlled by attach_upstream_area_to_q. Existing complete
            Q files can be enriched without a new EWDS discharge request.
        network_area : [North, West, South, East] or None
            Optional larger domain for the combined static file. None uses area.
            Q retains the requested area. Attached fields are a native subset.
            A crop is NOT proof that the complete upstream domain is included.
        static_map_urls : mapping or None
            Explicit version-compatible HTTPS URLs, keyed by LDD, UPAREA, CHAN,
            CELLAREA (GLOFAS. prefixes also accepted). There is no cross-version
            fallback. upstream_area_url remains supported; conflicting entries
            are refused. v4 CHAN/CELLAREA require compatible explicit sources.
        static_map_files : mapping or None
            Optional existing NetCDF sources for these maps, for offline use.
            Source version conflicts and invalid direction codes are rejected.
            A source without version metadata is recorded as user-asserted.
        force_static_download : bool or None
            None follows force_download. True refreshes ONLY auxiliary data when
            force_download=False, without forcing a Q re-download.
        return_paths : bool, default False
            True returns {"discharge": Path or None, "static": Path or None,
            "uparea": Path or None, "system_version": str, "area": tuple,
            "network_area": tuple, "requested_variables": tuple}.
            This manifest is also stored in self.last_glofas_download.

        allow_incomplete : bool, default False
            False requires all requested days. True saves available days and
            permits an incomplete year. Missing Q is never filled/interpolated.
            Monthly checkpoints and a .coverage.json report permit later refresh.
        end_date : date-like or None
            Inclusive final request date within year_start..year_end. Requests
            stop at yesterday UTC. Provider valid time T is NEVER shifted;
            REQUEST_DATE is retained when different. An explicit end adds a
            _toYYYYMMDD filename suffix. Static-only requests do not use dates.
        refresh_incomplete : bool, default True
            Refresh incomplete periods only; complete monthly caches are reused.

        Notes
        -----
        Static global NetCDF sources are cached in .glofas_static before native
        cropping. Static_GLOFAS_<version>.nc groups the requested fields;
        Static_UPAREA_<version>.nc remains available for older workflows.
        Native labels may be reconciled for tiny coordinate-rounding noise only;
        no interpolation, flow repair, or snapping to a neighbouring river cell.
        Keep source datasets closed before enriching files on Windows.
        """

        import calendar
        import time as _time
        from pathlib import Path
        from urllib.request import Request, urlopen

        import numpy as np
        import xarray as xr


        import pandas as pd
        _glofas_patch_revision = "GLOFAS_UNIFIED_AUXILIARIES_HYDRO_V52"

        # Private helpers belong INSIDE this method so copying this complete
        # definition into WAS_Download does not need other WASS2S module globals.
        def _was_validate_area(area):
            """Validate [north, west, south, east]; None means no spatial subset."""
            if area is None:
                return None
            if len(area) != 4:
                raise ValueError("area must be [North, West, South, East].")
            vals = tuple(float(v) for v in area)
            if not np.isfinite(vals).all():
                raise ValueError("area must contain finite coordinates.")
            n, w, s, e = vals
            if not (-90 <= s < n <= 90 and -180 <= w <= 180 and -180 <= e <= 180 and w != e):
                raise ValueError("Invalid area: require -90 <= South < North <= 90 and longitudes in [-180, 180].")
            return vals

        def _was_cache_key(**identity):
            """Deterministic configuration key; never pass tokens/passwords here."""
            import hashlib
            import json
            return hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()[:16]

        def _was_write_netcdf(obj, path, *, encoding=None, compute=True, before_replace=None, **kwargs):
            """Atomic replacement; never leave a half-written final NetCDF file.

            netCDF4/h5netcdf compression is used when installed. Small-data/test
            installations can use scipy (no compression); this fallback is announced.
            A failed write leaves any previously valid destination untouched.
            """
            import importlib.util
            import os
            import tempfile
            import warnings
            if not compute:
                from dask import delayed
                return delayed(_was_write_netcdf)(obj, path, encoding=encoding, compute=True, before_replace=before_replace, **kwargs)
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            enc = {k: dict(v) for k, v in (encoding or {}).items()}
            if "engine" not in kwargs:
                if importlib.util.find_spec("netCDF4") is not None:
                    kwargs["engine"] = "netcdf4"
                elif importlib.util.find_spec("h5netcdf") is not None:
                    kwargs["engine"] = "h5netcdf"
                else:
                    kwargs["engine"] = "scipy"
                    dropped = {"zlib", "complevel", "shuffle", "fletcher32", "chunksizes", "contiguous", "compression", "compression_opts"}
                    if any(dropped.intersection(v) for v in enc.values()):
                        warnings.warn("netCDF4/h5netcdf unavailable: writing uncompressed NetCDF with scipy.", RuntimeWarning)
                    enc = {k: {a: b for a, b in v.items() if a not in dropped} for k, v in enc.items()}
            fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=path.parent)
            os.close(fd)
            try:
                obj.to_netcdf(tmp, encoding=enc, **kwargs)
                # Windows refuses replacing a file still open in the reading Dataset.
                # The callback runs only AFTER the temporary file was fully written.
                if before_replace is not None:
                    before_replace()
                os.replace(tmp, path)
            finally:
                Path(tmp).unlink(missing_ok=True)

        def _was_concat_files(files, time_dim="T", max_eager_bytes=512 * 1024**2):
            """Lazy concatenation with Dask; explicit bounded-memory fallback otherwise.

            The caller must close the returned Dataset. No spatial outer join and no
            silent duplicate timestamp removal are allowed.
            """
            import importlib.util
            files = list(files)
            if not files:
                raise ValueError("No files available for concatenation.")
            if importlib.util.find_spec("dask") is not None:
                out = xr.open_mfdataset(files, combine="nested", concat_dim=time_dim,
                                       chunks={time_dim: 31}, parallel=False,
                                       join="exact", data_vars="minimal", coords="minimal", compat="equals")
                close = out.close
                try:
                    out = out.sortby(time_dim)
                    if not out.get_index(time_dim).is_unique:
                        raise ValueError("Duplicate timestamps across cache files; inspect overlapping source periods.")
                    out.set_close(close)
                    return out
                except Exception:
                    close()
                    raise
            total = 0
            for path in files:
                with xr.open_dataset(path) as item:
                    total += item.nbytes
            # Concatenation needs additional copies, hence a conservative small ceiling.
            if total > max_eager_bytes:
                raise ImportError(f"Dask is needed to assemble {total / 1024**2:.1f} MiB of data safely. Install dask[array].")
            arrays = []
            for path in files:
                with xr.open_dataset(path) as item:
                    arrays.append(item.load())
            out = xr.concat(arrays, dim=time_dim, join="exact", data_vars="minimal", coords="minimal", compat="equals").sortby(time_dim)
            if not out.get_index(time_dim).is_unique:
                raise ValueError("Duplicate timestamps across cache files; inspect overlapping source periods.")
            return out

        def _was_ewds_client(url="https://ewds.climate.copernicus.eu/api", key=None, rc_file=None):
            """Resolve a token without modifying CDS configuration or printing secrets."""
            import os
            import yaml
            import cdsapi
            from urllib.parse import urlsplit
            parsed = urlsplit(str(url))
            if parsed.scheme != "https" or parsed.hostname != "ewds.climate.copernicus.eu" or parsed.path.rstrip("/") != "/api":
                raise ValueError("GloFAS must use https://ewds.climate.copernicus.eu/api, not the CDS endpoint.")
            resolved = key or os.getenv("EWDS_API_KEY")
            if not resolved and rc_file is not None:
                config = Path(rc_file).expanduser()
                if not config.is_file():
                    raise FileNotFoundError(f"EWDS configuration not found: {config}")
                with config.open() as stream:
                    resolved = (yaml.safe_load(stream) or {}).get("key")
                if not resolved:
                    raise ValueError("The explicit EWDS configuration contains no key.")
            if not resolved:
                resolved = os.getenv("CDSAPI_KEY")
            candidates = [Path.home() / ".ewdsapirc"]
            if os.getenv("CDSAPI_RC"):
                candidates.append(Path(os.environ["CDSAPI_RC"]).expanduser())
            candidates.append(Path.home() / ".cdsapirc")
            for config in candidates:
                if resolved:
                    break
                if config.is_file():
                    with config.open() as stream:
                        resolved = (yaml.safe_load(stream) or {}).get("key")
            if not resolved:
                raise ValueError("No EWDS token. Set EWDS_API_KEY, ewds_key, or a key in ~/.ewdsapirc / ~/.cdsapirc.")
            print(f"[GloFAS] EWDS endpoint: {str(url).rstrip('/')}")
            return cdsapi.Client(url=str(url).rstrip("/"), key=str(resolved).strip())

        def _was_permanent_http_error(exc):
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is None:
                status = getattr(exc, "code", None)  # urllib HTTPError
            return status in {400, 401, 403, 404, 405, 410, 422}

        DATASET = "cems-glofas-historical"
        REQUIRED_DIMS = {"T", "Y", "X"}

        COORD_ALIASES = {
            "longitude": "X",
            "lon": "X",
            "latitude": "Y",
            "lat": "Y",
            "time": "T",
            "valid_time": "T",
            "date": "T",
        }

        UPAREA_ALIASES = {"GLOFAS.UPAREA", "GLOFAS.DRAINAGE_AREA"}
        AUXILIARY_ALIASES = {
            "GLOFAS.UPAREA": "UPAREA", "GLOFAS.DRAINAGE_AREA": "UPAREA",
            "GLOFAS.LDD": "LDD", "GLOFAS.CHAN": "CHAN",
            "GLOFAS.CELLAREA": "CELLAREA",
        }

        VARIABLE_MAPPING = {
            "GLOFAS.Q": {
                # Current EWDS schema (introduced with the GloFAS v5 rollout).
                # This schema is also used when requesting v4 from the
                # restructured cems-glofas-historical catalogue.
                "cds_variable": "average_river_discharge_in_the_last_24_hours",
                "timespan": "time_mean",
                "output_name": "Q",
                "long_name": "GloFAS average daily river discharge",
                "units": "m3 s-1",
            },
        }

        area = _was_validate_area(area)
        if area is None:
            raise ValueError("GloFAS requires a bounding box.")
        year_start, year_end = int(year_start), int(year_end)
        if year_start > year_end:
            raise ValueError("year_start must be <= year_end.")
        variables = [variables] if isinstance(variables, str) else list(dict.fromkeys(variables))
        if not variables:
            raise ValueError("At least one GloFAS variable is required.")
        want_uparea = any(v in UPAREA_ALIASES for v in variables)
        auxiliary_variables = tuple(dict.fromkeys(
            AUXILIARY_ALIASES[v] for v in variables if v in AUXILIARY_ALIASES))
        dynamic_variables = [v for v in variables if v not in AUXILIARY_ALIASES]

        unknown = [
            v for v in dynamic_variables
            if v not in VARIABLE_MAPPING
        ]
        if unknown:
            raise ValueError(
                f"Unknown GloFAS variable(s): {unknown}. Supported: "
                f"{list(VARIABLE_MAPPING) + list(AUXILIARY_ALIASES)}"
            )

        dir_to_save = Path(dir_to_save)
        dir_to_save.mkdir(parents=True, exist_ok=True)

        tmp_root = dir_to_save / ".glofas_tmp" / _was_cache_key(version=system_version, product=product_type, area=area)
        tmp_root.mkdir(parents=True, exist_ok=True)

        # ================================================================
        # Generic coordinate helpers
        # ================================================================
        def _wrap_sort_xy(da):
            """Standardize horizontal coordinates to ascending WASS2S X/Y."""
            rename = {
                old: new
                for old, new in COORD_ALIASES.items()
                if old in da.dims or old in da.coords
            }
            if rename:
                da = da.rename(rename)

            if "X" not in da.coords or "Y" not in da.coords:
                raise ValueError(
                    f"Could not identify GloFAS X/Y coordinates. Found dims={da.dims}, "
                    f"coords={list(da.coords)}"
                )

            if float(da["X"].max()) > 180.0:
                da = da.assign_coords(
                    X=((da["X"] + 180.0) % 360.0) - 180.0
                )

            return da.sortby("X").sortby("Y")

        def _subset_area_xy(da):
            """Subset an already standardized X/Y DataArray to [N,W,S,E]."""
            north, west, south, east = map(float, area)

            if south > north:
                raise ValueError(
                    f"Invalid area={area}: South ({south}) > North ({north})."
                )

            da = da.sel(Y=slice(south, north))

            # Standard non-dateline-crossing case (West Africa, etc.)
            if west <= east:
                da = da.sel(X=slice(west, east))
            else:
                # Dateline crossing, e.g. [N, 170, S, -170]
                left = da.sel(X=slice(west, 180.0))
                right = da.sel(X=slice(-180.0, east))
                da = xr.concat([left, right], dim="X").sortby("X")

            if da.sizes.get("X", 0) == 0 or da.sizes.get("Y", 0) == 0:
                raise ValueError(
                    f"Area {area} produced an empty subset. "
                    f"Available X=[{float(da.X.min()) if da.sizes.get('X',0) else 'NA'}, "
                    f"{float(da.X.max()) if da.sizes.get('X',0) else 'NA'}], "
                    f"Y=[{float(da.Y.min()) if da.sizes.get('Y',0) else 'NA'}, "
                    f"{float(da.Y.max()) if da.sizes.get('Y',0) else 'NA'}]."
                )

            return da

        # ================================================================
        # Dynamic T/Y/X standardization
        # ================================================================
        def _standardize_dynamic(ds, output_name):
            ds = ds.drop_vars(
                [
                    v
                    for v in ("crs", "spatial_ref", "surface")
                    if v in ds.variables
                ],
                errors="ignore",
            )

            data_vars = list(ds.data_vars)
            if not data_vars:
                raise ValueError(
                    "No data variable found in downloaded GloFAS file."
                )

            preferred = [
                "dis24",
                "q",
                "Q",
                "river_discharge_in_the_last_24_hours",
                "average_river_discharge_in_the_last_24_hours",
            ]
            source = next((v for v in preferred if v in data_vars), None)

            if source is None:
                candidates = [v for v in data_vars if ds[v].ndim >= 2] or data_vars
                source = max(candidates, key=lambda v: ds[v].size)

            da = ds[source]
            # Keep the provider reference date before dropping alternative time
            # coordinates. GloFAS may expose time + step = valid_time.
            if "REQUEST_DATE" not in da.coords:
                valid = next((n for n in ("valid_time", "time", "date")
                              if n in da.coords and da[n].ndim == 1), None)
                if valid:
                    ref = next((n for n in ("time", "forecast_reference_time", "date")
                                if n in da.coords and da[n].dims == da[valid].dims
                                and n != valid), None)
                    if ref is not None:
                        da = da.assign_coords(REQUEST_DATE=(da[valid].dims,
                            pd.DatetimeIndex(da[ref].values).normalize().values))
                    elif valid in ("time", "date"):
                        da = da.assign_coords(REQUEST_DATE=(da[valid].dims,
                            pd.DatetimeIndex(da[valid].values).normalize().values))


            if "T" not in da.dims:
                time_name = next((n for n in ("valid_time", "time", "date")
                                  if n in da.coords and da[n].ndim == 1), None)
                if time_name is None:
                    raise ValueError("No one-dimensional valid-time coordinate found in GloFAS discharge.")
                dim = da[time_name].dims[0]
                if dim != time_name:
                    da = da.swap_dims({dim: time_name}).drop_vars(dim, errors="ignore")
                # Other time aliases can refer to initialization time, not valid time.
                da = da.drop_vars([n for n in ("time", "date", "valid_time")
                                   if n != time_name and n in da.coords], errors="ignore")
                da = da.rename({time_name: "T"})
            rename = {}
            for target, aliases in (("X", ("longitude", "lon")), ("Y", ("latitude", "lat"))):
                if target not in da.dims:
                    old = next((n for n in aliases if n in da.dims), None)
                    if old:
                        rename[old] = target
            if rename:
                da = da.rename(rename)

            # Drop singleton extra dimensions only.
            for dim in [d for d in da.dims if d not in REQUIRED_DIMS]:
                if da.sizes[dim] == 1:
                    da = da.isel({dim: 0}, drop=True)
                else:
                    raise ValueError(
                        f"Unexpected non-singleton dimension '{dim}' "
                        f"with size {da.sizes[dim]}"
                    )

            missing = REQUIRED_DIMS - set(da.dims)
            if missing:
                raise ValueError(
                    "Could not identify all required GloFAS dimensions T/Y/X. "
                    f"Missing: {missing}. Found: {da.dims}"
                )

            if float(da["X"].max()) > 180.0:
                da = da.assign_coords(
                    X=((da["X"] + 180.0) % 360.0) - 180.0
                )

            return (
                da.sortby("X")
                .sortby("Y")
                .sortby("T")
                .transpose("T", "Y", "X")
                .rename(output_name)
            )

        def _dedup_time(da):
            _, idx = np.unique(da["T"].values, return_index=True)
            return da.isel(T=np.sort(idx))

        def _encoding_dynamic(name):
            return {
                name: {
                    "dtype": "float32",
                    "zlib": True,
                    "complevel": 4,
                    "_FillValue": -9999.0,
                }
            }

        def _encoding_static(dataset):
            encoding = {}
            for name in dataset.data_vars:
                dtype, fill = (("int16", -9999) if name == "LDD" else
                               ("int8", -1) if name == "CHAN" else
                               ("float32", -9999.0))
                encoding[name] = dict(dtype=dtype, zlib=True, complevel=4,
                                      _FillValue=fill)
            return encoding

        def _align_static_to_dynamic_grid(ds_up, dynamic_da):
            """
            Align requested static maps to the exact Q Y/X grid without regridding the
            hydrological values. Only coordinate-rounding differences are allowed;
            a shift to another river pixel is rejected.
            """
            x = dynamic_da["X"].values
            y = dynamic_da["Y"].values

            dx = (
                float(np.nanmedian(np.abs(np.diff(x))))
                if x.size > 1
                else 0.05
            )
            dy = (
                float(np.nanmedian(np.abs(np.diff(y))))
                if y.size > 1
                else 0.05
            )
            # Correct coordinate-rounding noise only, never move to an adjacent river pixel.
            tolerance = min(dx, dy) * 1e-3 + 1e-10
            for dim in ("X", "Y"):
                idx = ds_up.get_index(dim).get_indexer(dynamic_da[dim].values, method="nearest", tolerance=tolerance)
                if (idx < 0).any() or len(np.unique(idx)) != len(idx):
                    raise ValueError(f"Auxiliary and Q grids do not match along {dim}; refusing hydrological nearest-neighbour remapping.")

            aligned = ds_up.reindex(
                X=dynamic_da["X"],
                Y=dynamic_da["Y"],
                method="nearest",
                tolerance=tolerance,
            )

            return aligned

        for name, value in (("attach_static_maps_to_q", attach_static_maps_to_q),
                            ("attach_upstream_area_to_q", attach_upstream_area_to_q),
                            ("return_paths", return_paths)):
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must be True or False")
        if force_static_download is not None and not isinstance(force_static_download, (bool, np.bool_)):
            raise TypeError("force_static_download must be True, False or None")
        auxiliary_area = area if network_area is None else _was_validate_area(network_area)
        paths = dict(discharge=None, static=None, uparea=None,
                     system_version=system_version, area=area,
                     network_area=auxiliary_area,
                     requested_variables=tuple(variables))
        self.last_glofas_download = dict(paths)
        ds_aux = None
        if auxiliary_variables:
            if __package__:
                from .glofas_static_download import prepare_glofas_auxiliaries
            else:
                from glofas_static_download import prepare_glofas_auxiliaries
            urls = {str(k).removeprefix("GLOFAS.").upper(): v
                    for k, v in (static_map_urls or {}).items()}
            if upstream_area_url is not None:
                if "UPAREA" in urls and urls["UPAREA"] != upstream_area_url:
                    raise ValueError("Conflicting upstream_area_url and static_map_urls['UPAREA']")
                urls["UPAREA"] = upstream_area_url
            ds_aux, auxiliary_paths = prepare_glofas_auxiliaries(
                dir_to_save, variables=auxiliary_variables, version=system_version,
                area=auxiliary_area,
                force=force_download if force_static_download is None else force_static_download,
                url_overrides=urls, local_files=static_map_files,
                max_retries=max_retries, retry_delay=retry_delay,
            )
            paths.update(auxiliary_paths)
            self.last_glofas_download = dict(paths)

        def _attach_auxiliaries(dataset, q):
            """Attach only requested fields, never expand the dynamic spatial grid."""
            if ds_aux is None:
                return dataset, {}
            names = []
            if want_uparea and attach_upstream_area_to_q:
                names += ["UPAREA", "UPAREA_KM2"]
            if attach_static_maps_to_q:
                names += [name for name in auxiliary_variables if name != "UPAREA"]
            if not names:
                return dataset, {}
            aligned = _align_static_to_dynamic_grid(ds_aux[names], q)
            # Explicit encoding below must not conflict with inherited source packing.
            aligned = aligned.drop_encoding()
            for name in aligned:
                aligned[name].attrs = {k: v for k, v in aligned[name].attrs.items()
                                       if k not in ("_FillValue", "missing_value", "scale_factor", "add_offset")}
            out = dataset.assign({name: aligned[name] for name in names})
            out.attrs = dict(dataset.attrs)
            out.attrs.update(glofas_static_file=str(paths["static"]),
                             glofas_static_network_area_NWSE=str(list(auxiliary_area)),
                             attached_static_variables=",".join(names))
            if "UPAREA" in names:
                out.attrs["upstream_area_static_file"] = str(paths["uparea"])
            return out, _encoding_static(aligned)

        if not dynamic_variables:
            if not keep_temporary and tmp_root.exists() and not any(tmp_root.iterdir()):
                tmp_root.rmdir()
            return dict(paths) if return_paths else None

        # ================================================================
        # Temporal policy: calendar completeness is distinct from file validity.
        # ================================================================
        import json
        import os
        import tempfile
        from datetime import datetime, timezone

        if not isinstance(allow_incomplete, (bool, np.bool_)):
            raise TypeError("allow_incomplete must be True or False.")
        if not isinstance(refresh_incomplete, (bool, np.bool_)):
            raise TypeError("refresh_incomplete must be True or False.")
        if int(max_retries) != max_retries or max_retries < 1 or retry_delay < 0:
            raise ValueError("Require max_retries >= 1 and retry_delay >= 0.")
        max_retries = int(max_retries)
        requested_start = pd.Timestamp(year_start, 1, 1)
        calendar_end = pd.Timestamp(year_end, 12, 31)
        requested_end = calendar_end if end_date is None else pd.Timestamp(end_date)
        if pd.isna(requested_end):
            raise ValueError("end_date must be a valid date.")
        if requested_end.tzinfo is not None:
            requested_end = requested_end.tz_convert("UTC").tz_localize(None)
        requested_end = requested_end.normalize()
        if not requested_start <= requested_end <= calendar_end:
            raise ValueError("end_date must lie within year_start..year_end.")
        today_utc = pd.Timestamp(datetime.now(timezone.utc).date())
        last_request_day = today_utc - pd.Timedelta(days=1)
        effective_end = min(requested_end, last_request_day)
        if dynamic_variables:
            if effective_end < requested_start:
                raise ValueError("The requested period is entirely in the future; no request was sent.")
            if requested_end > last_request_day and not allow_incomplete:
                raise ValueError(
                    "The requested calendar year/period is not finished. "
                    "Use allow_incomplete=True or provide an end_date before today UTC."
                )
            if effective_end < requested_end:
                print(f"[GloFAS] Partial-year mode: API requests stop at {effective_end:%Y-%m-%d}. "
                      "This is a calendar limit, not a guarantee of provider availability.")

        _day_ns = int(pd.Timedelta(days=1).value)
        _known_time_offset = None

        def _dates_for(start, end):
            return pd.date_range(start, end, freq="D")

        def _time_info(da, start, end, *, infer_full_window=False):
            """Validate real daily samples, allowing missing dates but NOT bad dates.

            REQUEST_DATE distinguishes the API date from the unchanged valid time T.
            Old full-month/year checkpoints can infer a 0..24h offset from their
            exact complete time axis. Partial raw responses use source metadata or
            the offset established from earlier complete checkpoints.
            """
            nonlocal _known_time_offset
            if "T" not in da.dims or not np.issubdtype(da["T"].dtype, np.datetime64):
                raise ValueError("Invalid GloFAS time axis: decoded datetime T is required.")
            times = pd.DatetimeIndex(da["T"].values)
            if times.hasnans or not times.is_unique:
                raise ValueError("Invalid GloFAS time axis: NaT or duplicate valid times.")
            if not times.is_monotonic_increasing:
                raise ValueError("Invalid GloFAS time axis: times are not increasing.")
            if len(times) > 1:
                jumps = np.diff(times.values.astype("datetime64[ns]").astype("int64"))
                if np.any(jumps <= 0) or np.any(jumps % _day_ns):
                    raise ValueError("Invalid GloFAS time axis: subdaily or irregular intraday samples.")
            if len(times) == 0:
                return da, pd.DatetimeIndex([])
            basis = da.attrs.get("request_date_basis", "")
            if "REQUEST_DATE" in da.coords:
                days = pd.DatetimeIndex(da["REQUEST_DATE"].values)
                basis = basis or "provider_reference_date"
            else:
                offset = da.attrs.get("request_date_offset_ns")
                if offset is not None:
                    offset = int(offset)
                    basis = "checkpoint_offset"
                elif "step" in da.coords and da["step"].ndim == 0 and np.issubdtype(da["step"].dtype, np.timedelta64):
                    offset = int(pd.Timedelta(da["step"].values).value)
                    basis = "provider_step"
                elif infer_full_window and len(times) == len(_dates_for(start, end)):
                    candidate = int((times[0] - pd.Timestamp(start)).value)
                    offset = candidate if 0 <= candidate <= _day_ns else None
                    basis = "complete_checkpoint_time_axis"
                else:
                    offset = _known_time_offset
                    basis = "previous_complete_month_offset"
                if offset is None:
                    # A file exposing only valid_time, without time/step metadata:
                    # retain its calendar labels, rather than inventing a shift.
                    offset = int((times[0] - times[0].normalize()).value)
                    basis = "valid_time_calendar_labels_no_reference"
                days = (times - pd.Timedelta(offset, unit="ns")).normalize()
                da = da.assign_coords(REQUEST_DATE=("T", days.values))
            if (days.hasnans or not days.is_unique or not days.is_monotonic_increasing
                    or not days.equals(days.normalize()) or len(days) != len(times)):
                raise ValueError("Invalid REQUEST_DATE: require one distinct, ordered calendar date per sample.")
            offsets = times.values.astype("datetime64[ns]").astype("int64") - days.values.astype("datetime64[ns]").astype("int64")
            if len(np.unique(offsets)) != 1 or not 0 <= int(offsets[0]) <= _day_ns:
                raise ValueError("Inconsistent provider valid_time/reference date offset; inspect the raw file.")
            offset = int(offsets[0])
            if _known_time_offset is not None and offset != _known_time_offset:
                raise ValueError("Provider time convention changed across files; refusing a silent date shift.")
            _known_time_offset = offset
            if days.min() < start or days.max() > end:
                raise ValueError(f"Unexpected dates outside {start:%Y-%m-%d}..{end:%Y-%m-%d} in a GloFAS source/cache.")
            da.attrs.update(request_date_offset_ns=str(offset), request_date_basis=basis)
            da["REQUEST_DATE"].attrs.update(
                long_name="Date requested from EWDS; T retains the provider valid time",
                description="Coverage accounting coordinate; no shift or interpolation of discharge values",
            )
            return da, days

        def _check_identity(ds, *, require=False):
            """Old hash-namespaced checkpoints can lack metadata, but never conflict."""
            import ast as _ast
            attrs = ds.attrs
            field = ds["Q"].attrs if "Q" in ds else {}
            for name, expected in (("system_version", system_version), ("product_type", product_type)):
                actual = attrs.get(name, field.get(name))
                if actual is None and require:
                    raise ValueError(f"Existing Q has unknown {name}; use a separate directory.")
                if actual is not None and actual != expected:
                    raise ValueError(f"Existing Q has another {name}; use a separate directory.")
            old_area = attrs.get("spatial_subset_NWSE")
            if old_area is not None:
                vals = _ast.literal_eval(old_area) if isinstance(old_area, str) else old_area
                if tuple(map(float, vals)) != tuple(area):
                    raise ValueError("Existing Q has another area; use a separate directory.")

        def _merge_samples(old, new):
            """New samples replace same-day samples; absent old days are retained.

            Do not combine_first Q: a newly reported NaN must not become an old
            value. Never remap hydrological grids while refreshing a month.
            """
            if old is None or old.sizes.get("T", 0) == 0:
                return new
            if new is None or new.sizes.get("T", 0) == 0:
                return old
            old_days = pd.DatetimeIndex(old["REQUEST_DATE"].values)
            new_days = pd.DatetimeIndex(new["REQUEST_DATE"].values)
            common = old_days.intersection(new_days)
            if len(common):
                if not np.array_equal(old["T"].values[old_days.get_indexer(common)],
                                      new["T"].values[new_days.get_indexer(common)]):
                    raise ValueError("Changed timestamps for the same API dates; refusing ambiguous refresh.")
            keep = ~old_days.isin(new_days)
            return xr.concat([old.isel(T=keep), new], dim="T", join="exact",
                             coords="minimal", compat="equals").sortby("T")

        def _common_attrs():
            return dict(system_version=system_version, product_type=product_type,
                        spatial_subset_NWSE=str(list(area)), ewds_endpoint=ewds_url,
                        glofas_downloader_revision=_glofas_patch_revision)

        def _write_checkpoint(da, path, start, end):
            da.attrs.update(long_name="GloFAS average daily river discharge", units="m3 s-1",
                            system_version=system_version, product_type=product_type)
            ds = da.to_dataset(name="Q")
            ds.attrs.update(_common_attrs())
            days = pd.DatetimeIndex(da["REQUEST_DATE"].values)
            ds.attrs.update(expected_start=str(start.date()), expected_end=str(end.date()),
                            n_available_days=len(days), n_expected_days=len(_dates_for(start, end)),
                            is_complete=int(days.equals(_dates_for(start, end))))
            _was_write_netcdf(ds, path, encoding=_encoding_dynamic("Q"))

        def _write_json_atomic(content, path):
            fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(content, handle, ensure_ascii=False, indent=2)
                    handle.write("\n")
                os.replace(tmp, path)
            finally:
                Path(tmp).unlink(missing_ok=True)

        client = None  # No credentials needed until a real Q request is necessary.
        for var in dynamic_variables:
            mapping = VARIABLE_MAPPING[var]
            output_name = mapping["output_name"]
            short_name = var.split(".")[-1]
            date_tag = "" if end_date is None else f"_to{requested_end:%Y%m%d}"
            output_path = dir_to_save / f"Daily_{short_name}_{year_start}_{year_end}{date_tag}.nc"
            report_path = output_path.with_suffix(".coverage.json")
            paths["discharge"] = output_path
            var_tmp = tmp_root / short_name
            var_tmp.mkdir(parents=True, exist_ok=True)
            final_seed = None

            # A partial final file is a useful seed, NOT proof of a finished archive.
            if output_path.exists() and not force_download:
                with xr.open_dataset(output_path) as cached_q:
                    _check_identity(cached_q, require=True)
                    candidate, days = _time_info(cached_q[output_name], requested_start,
                                                 calendar_end, infer_full_window=True)
                    needed = _dates_for(requested_start, requested_end)
                    covers_request = len(needed.difference(days)) == 0
                    if covers_request:
                        enriched, auxiliary_encoding = _attach_auxiliaries(cached_q, candidate)
                        if auxiliary_encoding:
                            _was_write_netcdf(enriched, output_path,
                                encoding={**_encoding_dynamic(output_name), **auxiliary_encoding},
                                before_replace=cached_q.close)
                        self.last_glofas_download = dict(paths)
                        print(f"{output_path} reused; requested period is complete, no discharge redownload.")
                        continue
                final_seed = output_path
                print(f"[GloFAS] Existing output is incomplete: checking monthly checkpoints ({output_path.name}).")

            month_reports = []
            year_files = []

            def _read_seed(path, scope_start, scope_end, first, last):
                if path is None or not path.exists() or force_download:
                    return None
                with xr.open_dataset(path) as cached:
                    _check_identity(cached)
                    # _time_info touches coordinates, not the full gridded archive.
                    da, _ = _time_info(cached[output_name], scope_start, scope_end,
                                        infer_full_window=True)
                    rd = pd.DatetimeIndex(da["REQUEST_DATE"].values)
                    selected = da.isel(T=(rd >= first) & (rd <= last))
                    return selected.load() if selected.sizes["T"] else None

            for year in range(year_start, min(year_end, effective_end.year) + 1):
                print(f"\n{'=' * 70}\nGloFAS {output_name}: processing year {year}\n{'=' * 70}")
                year_start_date = pd.Timestamp(year, 1, 1)
                year_end_date = pd.Timestamp(year, 12, 31)
                year_file = var_tmp / f"{short_name}_{year}.nc"
                month_paths = []
                month_end_number = effective_end.month if year == effective_end.year else 12

                for month in range(1, month_end_number + 1):
                    first = pd.Timestamp(year, month, 1)
                    full_last = pd.Timestamp(year, month, calendar.monthrange(year, month)[1])
                    last = min(full_last, effective_end)
                    wanted = _dates_for(first, last)
                    month_file = var_tmp / f"processed_{short_name}_{year}{month:02d}.nc"
                    raw_file = var_tmp / f"raw_{short_name}_{year}{month:02d}.nc"
                    old = None
                    normalized_month_checkpoint = False
                    if month_file.exists() and not force_download:
                        with xr.open_dataset(month_file) as cached:
                            _check_identity(cached)
                            normalized_month_checkpoint = "REQUEST_DATE" in cached.coords
                            old, _ = _time_info(cached[output_name], first, full_last, infer_full_window=True)
                            old = old.load()
                    # Preserve work even if a prior cleanup removed monthly caches.
                    if old is None and not force_download:
                        old = _read_seed(year_file, year_start_date, year_end_date, first, full_last)
                    if old is None and final_seed is not None:
                        old = _read_seed(final_seed, requested_start, calendar_end, first, full_last)
                    old_days = pd.DatetimeIndex(old["REQUEST_DATE"].values) if old is not None else pd.DatetimeIndex([])
                    cached_complete = len(wanted.difference(old_days)) == 0
                    # A month completed before today is stable for cache purposes.
                    # The open current month may still change, even if it reaches yesterday.
                    current_month = first.year == today_utc.year and first.month == today_utc.month
                    refresh = (force_download or old is None or
                               (refresh_incomplete and (not cached_complete or current_month)))
                    source = "cache"
                    failure = None
                    new = None

                    if refresh:
                        request = {
                            "system_version": [system_version],
                            "hydrological_model": ["lisflood"],
                            "product_type": [product_type],
                            "variable": [mapping["cds_variable"]],
                            "timespan": [mapping.get("timespan", "time_mean")],
                            "year": [f"{year:04d}"],
                            "month": [f"{month:02d}"],
                            "day": [f"{d.day:02d}" for d in wanted],
                            "data_format": "netcdf",
                            "download_format": "unarchived",
                            "area": list(area),
                        }
                        # Raw files from the old failing run can be salvaged once.
                        # Never reuse an old raw response to freeze a partial cache.
                        use_raw = raw_file.exists() and raw_file.stat().st_size > 0 and old is None and not force_download
                        response_available = use_raw
                        source = "raw_cache" if use_raw else "download"
                        if not use_raw:
                            # Configuration/import errors must never become 'missing dates'.
                            if client is None:
                                client = _was_ewds_client(ewds_url, ewds_key, ewds_rc_file)
                            for attempt in range(1, max_retries + 1):
                                raw_part = raw_file.with_name(raw_file.name + ".part")
                                try:
                                    raw_part.unlink(missing_ok=True)
                                    print(f"Attempt {attempt}/{max_retries}: downloading {mapping['cds_variable']} "
                                          f"{year}-{month:02d}, days 01..{last.day:02d}")
                                    client.retrieve(DATASET, request, str(raw_part))
                                    if not raw_part.exists() or raw_part.stat().st_size == 0:
                                        raise OSError("The download returned no nonempty file.")
                                    os.replace(raw_part, raw_file)
                                    response_available = True
                                    print(f"Downloaded: {raw_file}")
                                    break
                                except (NameError, ImportError, TypeError, AttributeError):
                                    raise
                                except Exception as exc:
                                    status = getattr(getattr(exc, "response", None), "status_code", getattr(exc, "code", None))
                                    text = str(exc)
                                    # Explicit no-data messages only. 401/403/404 and
                                    # malformed requests are never converted to absence.
                                    no_data = (status in (None, 200, 400) and any(
                                        token in text.lower() for token in (
                                            "no data found", "no data available", "no data is available"))
                                        and not any(token in text.lower() for token in (
                                            "invalid", "not found for url", "process not found", "unauthorized", "forbidden")))
                                    if no_data:
                                        failure = {"kind": "provider_no_data", "message": text[:600]}
                                        break
                                    if _was_permanent_http_error(exc):
                                        raise RuntimeError(f"Permanent EWDS request error for {year}-{month:02d}; "
                                                           "check endpoint, request and licence.") from exc
                                    # Unknown/programming exceptions are not evidence
                                    # of a missing period. Retry only transport failures.
                                    mro_names = {base.__name__ for base in type(exc).__mro__}
                                    transport_error = (
                                        isinstance(exc, OSError)
                                        or bool(mro_names.intersection({"RequestException", "TransportError", "URLError"}))
                                        or status in (408, 425, 429)
                                        or (isinstance(status, int) and 500 <= status < 600)
                                    )
                                    if not transport_error:
                                        raise
                                    failure = {"kind": "download_failed", "message": text[:600]}
                                    print(f"[GloFAS] Download failed for {year}-{month:02d}: {exc}")
                                    if attempt < max_retries:
                                        _time.sleep(retry_delay)
                                finally:
                                    raw_part.unlink(missing_ok=True)
                        else:
                            print(f"[GloFAS] Reading existing raw response: {raw_file.name}")

                        if response_available:
                            # Decoding, time, shape and filesystem errors remain fatal:
                            # allow_incomplete is NOT an ignore-all-errors switch.
                            with xr.open_dataset(raw_file) as ds_raw:
                                new = _standardize_dynamic(ds_raw, output_name)
                                if new.sizes.get("T", 0):
                                    new = new.load()
                            new, new_days = _time_info(new, first, full_last, infer_full_window=True)
                            if len(new_days):
                                new = new.isel(T=(new_days <= last))
                                new_days = new_days[new_days <= last]
                            if len(new_days) == 0:
                                failure = {"kind": "provider_no_data", "message": "Decoded response contains no requested day."}
                                new = None
                                # Do not keep reusing an empty raw response on future
                                # runs: the provider may publish this month later.
                                if raw_file.exists():
                                    if keep_temporary:
                                        os.replace(raw_file, raw_file.with_name(raw_file.stem + "_empty.nc"))
                                    else:
                                        raw_file.unlink()
                            else:
                                failure = None

                    da_month = _merge_samples(old, new)
                    if da_month is not None and da_month.sizes["T"]:
                        # Validate merged chronology; retain all cached days in the
                        # monthly checkpoint, but clip assembly to requested dates.
                        da_month, month_days = _time_info(da_month, first, full_last, infer_full_window=True)
                        if new is not None or not normalized_month_checkpoint:
                            _write_checkpoint(da_month, month_file, first, full_last)
                        received = month_days.intersection(wanted)
                        complete = len(wanted.difference(received)) == 0
                        month_paths.append(month_file)
                        if not complete:
                            print(f"[GloFAS] {year}-{month:02d}: {len(received)}/{len(wanted)} requested days retained; "
                                  "partial data are saved in the monthly checkpoint.")
                        elif source == "cache":
                            print(f"[GloFAS] Reusing complete month: {month_file.name}")
                        if not keep_temporary and raw_file.exists():
                            raw_file.unlink()
                    else:
                        received = pd.DatetimeIndex([])
                        complete = False
                        print(f"[GloFAS] {year}-{month:02d}: no usable day; earlier checkpoints are retained.")

                    report = dict(
                        month=f"{year}-{month:02d}", source=source,
                        requested_start=str(first.date()), requested_end=str(last.date()),
                        n_requested_days=len(wanted), n_available_days=len(received),
                        complete_request=bool(complete),
                        complete_calendar_month=bool(complete and last == full_last),
                        missing_request_dates=[str(x.date()) for x in wanted.difference(received)],
                        failure=failure,
                    )
                    month_reports.append(report)
                    if (not complete or failure is not None) and not allow_incomplete:
                        _write_json_atomic(dict(status="strict_incomplete_no_final_write", months=month_reports), report_path)
                        raise RuntimeError(
                            f"Incomplete GloFAS month {year}-{month:02d}: {len(received)}/{len(wanted)} days. "
                            "Valid monthly checkpoints were kept. Set allow_incomplete=True to save available days."
                        )

                if not month_paths:
                    continue
                combined = _was_concat_files(month_paths)
                try:
                    rd = pd.DatetimeIndex(combined["REQUEST_DATE"].values)
                    da_year = combined[output_name].isel(T=(rd >= year_start_date) & (rd <= effective_end))
                    if da_year.sizes["T"]:
                        _write_checkpoint(da_year, year_file, year_start_date, year_end_date)
                        year_files.append(year_file)
                        print(f"[GloFAS] Year checkpoint saved: {year_file.name} ({da_year.sizes['T']} days).")
                finally:
                    combined.close()

            if not year_files:
                raise RuntimeError("No GloFAS day is available. No empty final NetCDF was written; caches retained.")

            combined_ds = _was_concat_files(year_files)
            try:
                da_final = combined_ds[output_name].sortby("T").transpose("T", "Y", "X")
                da_final, received = _time_info(da_final, requested_start, calendar_end, infer_full_window=True)
                selection = (received >= requested_start) & (received <= effective_end)
                da_final = da_final.isel(T=selection)
                received = received[selection]
                needed = _dates_for(requested_start, requested_end)
                full_calendar = _dates_for(requested_start, calendar_end)
                missing = needed.difference(received)
                request_complete = len(missing) == 0
                calendar_complete = len(full_calendar.difference(received)) == 0
                failures = [r for r in month_reports if r["failure"] is not None]
                if (not request_complete or failures) and not allow_incomplete:
                    raise RuntimeError("Incomplete requested period: final NetCDF was not replaced; checkpoints retained.")
                da_final.attrs.update(
                    long_name=mapping["long_name"], units=mapping["units"],
                    source="Copernicus Emergency Management Service - GloFAS", dataset=DATASET,
                    system_version=system_version, product_type=product_type,
                    hydrological_model="LISFLOOD", frequency="daily", WASS2S_dimensions="T,Y,X",
                )
                ds_out = da_final.to_dataset(name=output_name)
                incomplete_years = []
                for y in range(year_start, year_end + 1):
                    expected_y = _dates_for(pd.Timestamp(y, 1, 1), pd.Timestamp(y, 12, 31))
                    if len(expected_y.difference(received)):
                        incomplete_years.append(y)
                ds_out.attrs.update(_common_attrs())
                ds_out.attrs.update(
                    allow_incomplete=int(bool(allow_incomplete)),
                    is_complete=int(calendar_complete),
                    is_complete_request=int(request_complete),
                    requested_time_start=str(requested_start.date()),
                    requested_time_end=str(requested_end.date()),
                    effective_request_end=str(effective_end.date()),
                    actual_time_start=str(pd.Timestamp(da_final["T"].values[0])),
                    actual_time_end=str(pd.Timestamp(da_final["T"].values[-1])),
                    actual_request_start=str(received[0].date()),
                    actual_request_end=str(received[-1].date()),
                    n_available_days=len(received), n_expected_days=len(needed),
                    n_missing_days=len(missing),
                    incomplete_years=",".join(map(str, incomplete_years)),
                    incomplete_months=",".join(r["month"] for r in month_reports if not r["complete_calendar_month"]),
                    missing_days_policy="No interpolation; absent dates are omitted from T, details in coverage report",
                    coverage_scope="Temporal record coverage, not pixelwise nonmissing-data coverage",
                    temporal_coverage_report=report_path.name,
                    n_failed_requests=len(failures),
                )
                final_encoding = _encoding_dynamic(output_name)
                ds_out, auxiliary_encoding = _attach_auxiliaries(ds_out, da_final)
                final_encoding.update(auxiliary_encoding)
                _was_write_netcdf(ds_out, output_path, encoding=final_encoding)
                report = dict(
                    status="complete" if calendar_complete else "partial",
                    output_file=str(output_path), system_version=system_version,
                    product_type=product_type, area=list(area),
                    requested_start=str(requested_start.date()), requested_end=str(requested_end.date()),
                    effective_request_end=str(effective_end.date()),
                    actual_time_start=ds_out.attrs["actual_time_start"], actual_time_end=ds_out.attrs["actual_time_end"],
                    actual_request_start=str(received[0].date()), actual_request_end=str(received[-1].date()),
                    n_expected_days=len(needed), n_available_days=len(received), n_missing_days=len(missing),
                    is_complete_calendar_years=bool(calendar_complete), is_complete_request=bool(request_complete),
                    missing_request_dates=[str(x.date()) for x in missing],
                    not_yet_requested_dates=[str(x.date()) for x in needed if x > effective_end],
                    incomplete_years=incomplete_years, months=month_reports,
                    generated_at_utc=datetime.now(timezone.utc).isoformat(),
                    coverage_scope="Temporal records only; NaN values within an available day are preserved.",
                )
                _write_json_atomic(report, report_path)
                print(f"\nSaved {'complete' if calendar_complete else 'PARTIAL'} GloFAS dataset: {output_path}\n"
                      f"Available days: {len(received)}/{len(needed)}; actual T: "
                      f"{ds_out.attrs['actual_time_start']} -> {ds_out.attrs['actual_time_end']}\n"
                      f"Coverage report: {report_path}")
            finally:
                combined_ds.close()

            # Never remove resume state while a calendar year is incomplete.
            if not keep_temporary and calendar_complete and not failures:
                for cleanup_year in range(year_start, year_end + 1):
                    for cleanup_month in range(1, 13):
                        (var_tmp / f"processed_{short_name}_{cleanup_year}{cleanup_month:02d}.nc").unlink(missing_ok=True)
                for yf in year_files:
                    yf.unlink(missing_ok=True)
                if var_tmp.exists() and not any(var_tmp.iterdir()):
                    var_tmp.rmdir()
            elif not calendar_complete:
                print("[GloFAS] Monthly/yearly checkpoints retained for partial-year refresh.")

        if not keep_temporary and tmp_root.exists() and not any(tmp_root.iterdir()):
            tmp_root.rmdir()
        
        self.last_glofas_download = dict(paths)
        return dict(paths) if return_paths else output_path


    # Complementary downloaders from WAS_Download_REVIEWED.py

    def _WAS_http_download(
        self,
        url,
        target,
        max_retries=3,
        retry_delay=5,
        timeout=180,
        auth=None,
        headers=None,
    ):
        """Internal atomic HTTP(S) downloader used by the extra sources."""
        from pathlib import Path
        import os
        import time as _time
        import requests

        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")

        for attempt in range(1, max_retries + 1):
            try:
                print(f"[DL] {attempt}/{max_retries}: {url}")
                with requests.get(
                    url,
                    stream=True,
                    timeout=timeout,
                    auth=auth,
                    headers=headers,
                ) as r:
                    r.raise_for_status()
                    with open(part, "wb") as fh:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                fh.write(chunk)
                os.replace(part, target)
                return target
            except Exception as exc:
                if _was_permanent_http_error(exc):
                    part.unlink(missing_ok=True)
                    raise RuntimeError(f"Permanent HTTP error; request was not retried: {url}") from exc
                print(f"[WARN] download failed: {exc}")
                try:
                    part.unlink(missing_ok=True)
                except Exception:
                    pass
                if attempt < max_retries:
                    _time.sleep(retry_delay)
        raise RuntimeError(f"Failed to download after {max_retries} attempts: {url}")


    def _WAS_standardize_grid_dataset(self, ds, area=None, time_name="T"):
        """Internal helper: standardize a gridded Dataset to WASS2S T/Y/X."""
        import numpy as np

        rename = {}
        for old in ("longitude", "lon", "LONGITUDE", "x"):
            if old in ds.dims or old in ds.coords:
                rename[old] = "X"
                break
        for old in ("latitude", "lat", "LATITUDE", "y"):
            if old in ds.dims or old in ds.coords:
                rename[old] = "Y"
                break
        for old in ("time", "date", "valid_time", "TIME", "DATE"):
            if old in ds.dims or old in ds.coords:
                rename[old] = time_name
                break
        if rename:
            ds = ds.rename(rename)

        if "X" in ds.coords and ds["X"].size:
            try:
                if float(ds["X"].max()) > 180.0:
                    ds = ds.assign_coords(X=(((ds["X"] + 180.0) % 360.0) - 180.0))
            except Exception:
                pass
            ds = ds.sortby("X")

        if "Y" in ds.coords:
            ds = ds.sortby("Y")
        if time_name in ds.coords:
            ds = ds.sortby(time_name)

        if area is not None and "X" in ds.coords and "Y" in ds.coords:
            north, west, south, east = [float(v) for v in area]
            if west <= east:
                ds = ds.sel(X=slice(west, east), Y=slice(south, north))
            else:
                # Dateline-crossing domain.
                left = ds.sel(X=slice(west, 180.0), Y=slice(south, north))
                right = ds.sel(X=slice(-180.0, east), Y=slice(south, north))
                ds = __import__("xarray").concat([left, right], dim="X")

        if any(ds.sizes.get(d, 1) == 0 for d in ("X", "Y")):
            raise ValueError("The requested area contains no source grid-cell centres.")
        ds = ds.drop_vars(
            [v for v in ("crs", "spatial_ref") if v in ds.variables],
            errors="ignore",
        )
        return ds


    def _WAS_concat_grid_files(self, files, time_dim="T"):
        """Lazy, grid-strict concatenation; caller closes the returned Dataset."""
        return _was_concat_files(files, time_dim)


    def WAS_Download_NASA_POWER_Daily(
        self,
        dir_to_save,
        variables,
        year_start,
        year_end,
        area,
        force_download=False,
        max_retries=3,
        retry_delay=5,
        time_standard="UTC",
        community="AG",
        keep_yearly_cache=True,
    ):
        """
        Download NASA POWER daily regional data and save WASS2S T/Y/X files.

        Parameters
        ----------
        variables : sequence of str
            WASS2S aliases such as POWER.PRCP, POWER.TEMP, POWER.TMAX,
            POWER.TMIN, POWER.TDEW, POWER.RH2M, POWER.WS10M, POWER.DSWR,
            or a native POWER parameter prefixed with POWER.
        area : [North, West, South, East]
        time_standard : {"UTC", "LST"}
            UTC is recommended for comparison with other daily products.

        Returns
        -------
        dict
            Mapping requested variable -> final NetCDF Path.
        """
        from pathlib import Path
        from urllib.parse import urlencode
        import os
        import xarray as xr

        if isinstance(variables, str):
            variables = [variables]
        variables = list(variables)
        time_standard = str(time_standard).upper()
        if time_standard not in {"UTC", "LST"}:
            raise ValueError("time_standard must be 'UTC' or 'LST'.")

        mapping = {
            "POWER.PRCP": ("PRECTOTCORR", "PRCP"),
            "POWER.TEMP": ("T2M", "TEMP"),
            "POWER.TMAX": ("T2M_MAX", "TMAX"),
            "POWER.TMIN": ("T2M_MIN", "TMIN"),
            "POWER.TDEW": ("T2MDEW", "TDEW"),
            "POWER.RH2M": ("RH2M", "RH2M"),
            "POWER.WS10M": ("WS10M", "WS10M"),
            "POWER.DSWR": ("ALLSKY_SFC_SW_DWN", "DSWR"),
            "POWER.PS": ("PS", "PS"),
        }

        base = "https://power.larc.nasa.gov/api/temporal/daily/regional"
        out_dir = Path(dir_to_save)
        out_dir.mkdir(parents=True, exist_ok=True)
        cache = out_dir / ".power_daily_cache" / _was_cache_key(area=area, time_standard=time_standard, community=community)
        cache.mkdir(parents=True, exist_ok=True)
        north, west, south, east = [float(v) for v in area]
        result = {}

        for requested in variables:
            key = str(requested).upper()
            if key in mapping:
                native, code = mapping[key]
            else:
                if not key.startswith("POWER."):
                    raise ValueError(f"Unknown NASA POWER variable: {requested}")
                native = key.split(".", 1)[1]
                code = native

            final = out_dir / f"Daily_POWER_{code}_{year_start}_{year_end}.nc"
            if final.exists() and not force_download:
                print(f"{final} already exists. Skipping.")
                result[requested] = final
                continue

            var_cache = cache / code
            var_cache.mkdir(parents=True, exist_ok=True)
            yearly = []

            for year in range(int(year_start), int(year_end) + 1):
                regional = var_cache / f"POWER_{code}_{year}.nc"
                if regional.exists() and not force_download:
                    yearly.append(regional)
                    continue

                params = {
                    "parameters": native,
                    "community": community,
                    "longitude-min": west,
                    "longitude-max": east,
                    "latitude-min": south,
                    "latitude-max": north,
                    "start": f"{year}0101",
                    "end": f"{year}1231",
                    "format": "NETCDF",
                    "time-standard": time_standard,
                }
                url = base + "?" + urlencode(params)
                raw = var_cache / f"raw_POWER_{native}_{year}.nc"
                self._WAS_http_download(
                    url, raw, max_retries=max_retries, retry_delay=retry_delay
                )

                try:
                    with xr.open_dataset(raw) as ds:
                        ds = self._WAS_standardize_grid_dataset(ds, area=area).load()
                    if native in ds.data_vars:
                        source = native
                    elif native.lower() in ds.data_vars:
                        source = native.lower()
                    else:
                        candidates = [v for v in ds.data_vars if ds[v].ndim >= 2]
                        if not candidates:
                            raise ValueError(
                                f"NASA POWER response has no usable data variable: {list(ds.data_vars)}"
                            )
                        source = candidates[0]
                    if source != code:
                        ds = ds.rename({source: code})
                    ds = ds[[code]]
                    ds.attrs.update({
                        "source": "NASA POWER Daily API",
                        "native_parameter": native,
                        "time_standard": time_standard,
                        "WASS2S_dimensions": "T,Y,X",
                    })
                    ds[code].attrs.setdefault("source_parameter", native)
                    encoding = {code: {"zlib": True, "complevel": 4, "dtype": "float32"}}
                    _was_write_netcdf(ds, regional, encoding=encoding)
                    yearly.append(regional)
                finally:
                    raw.unlink(missing_ok=True)

            ds_out = self._WAS_concat_grid_files(yearly, "T")
            if "T" in ds_out.coords:
                ds_out = ds_out.sel(T=slice(f"{year_start}-01-01", f"{year_end}-12-31"))
            ds_out[code] = ds_out[code].astype("float32")
            _was_write_netcdf(ds_out, final, encoding={code: {'zlib': True, 'complevel': 4, 'dtype': 'float32'}})
            ds_out.close()
            result[requested] = final
            print(f"Saved NASA POWER: {final}")

            if not keep_yearly_cache:
                for p in yearly:
                    p.unlink(missing_ok=True)

        return result


    def WAS_Download_ARC2_Daily(
        self,
        dir_to_save,
        year_start,
        year_end,
        area,
        force_download=False,
        max_retries=3,
        retry_delay=5,
        keep_yearly_cache=True,
        allow_incomplete=True,
    ):
        """Download NOAA/CPC ARC2 daily rainfall (0.1°, Africa) as PRCP(T,Y,X)."""
        from pathlib import Path
        import gzip
        import io
        import time as _time
        import numpy as np
        import pandas as pd
        import requests
        import xarray as xr

        out_dir = Path(dir_to_save)
        out_dir.mkdir(parents=True, exist_ok=True)
        cache = out_dir / ".arc2_daily_cache" / _was_cache_key(area=area)
        cache.mkdir(parents=True, exist_ok=True)
        final = out_dir / f"Daily_ARC2_PRCP_{year_start}_{year_end}.nc"
        if final.exists() and not force_download:
            print(f"{final} already exists. Skipping.")
            return final

        # Native ARC2 grid: lon -20..55 (751), lat -40..40 (801), 0.1 degree.
        X = np.round(-20.0 + np.arange(751) * 0.1, 10)
        Y = np.round(-40.0 + np.arange(801) * 0.1, 10)
        north, west, south, east = [float(v) for v in area]
        ix = np.where((X >= west - 1e-8) & (X <= east + 1e-8))[0]
        iy = np.where((Y >= south - 1e-8) & (Y <= north + 1e-8))[0]
        if not len(ix) or not len(iy):
            raise ValueError("Requested area does not intersect the ARC2 Africa domain.")
        Xs, Ys = X[ix], Y[iy]
        base = "https://ftp.cpc.ncep.noaa.gov/fews/fewsdata/africa/arc2/bin"
        today = pd.Timestamp.utcnow().tz_localize(None).normalize()
        yearly_files = []

        for year in range(int(year_start), int(year_end) + 1):
            yf = cache / f"ARC2_Daily_{year}.nc"
            if yf.exists() and not force_download:
                yearly_files.append(yf)
                continue

            dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
            dates = dates[dates <= today]
            values = np.full((len(dates), len(iy), len(ix)), np.nan, dtype="float32")
            failed = []

            for i, date in enumerate(dates):
                ymd = date.strftime("%Y%m%d")
                url = f"{base}/daily_clim.bin.{ymd}.gz"
                data = None
                for attempt in range(1, max_retries + 1):
                    try:
                        r = requests.get(url, timeout=60)
                        if r.status_code == 404:
                            break
                        r.raise_for_status()
                        raw = gzip.decompress(r.content)
                        arr = np.frombuffer(raw, dtype=">f4")
                        if arr.size != 801 * 751:
                            raise ValueError(f"Unexpected ARC2 grid size: {arr.size}")
                        arr = arr.reshape(801, 751)
                        arr = arr[np.ix_(iy, ix)].astype("float32")
                        arr[arr <= -998.0] = np.nan
                        values[i] = arr
                        data = True
                        break
                    except Exception as exc:
                        if attempt == max_retries:
                            print(f"[ARC2] {ymd} failed: {exc}")
                        else:
                            _time.sleep(retry_delay)
                if data is None:
                    failed.append(ymd)

            if failed:
                print(f"[ARC2] {year}: {len(failed)} missing/unavailable day(s).")
                if not allow_incomplete and year < today.year:
                    raise RuntimeError(f"ARC2 year {year} incomplete: {failed[:10]}")

            da = xr.DataArray(
                values,
                dims=("T", "Y", "X"),
                coords={"T": dates.values, "Y": Ys, "X": Xs},
                name="PRCP",
                attrs={
                    "long_name": "ARC2 24-hour rainfall",
                    "units": "mm/day",
                    "accumulation_window": "06 UTC to 06 UTC next day",
                    "source": "NOAA CPC Africa Rainfall Climatology version 2",
                    "resolution": "0.1 degree",
                },
            )
            ds = da.to_dataset()
            ds.attrs["missing_source_days"] = ",".join(failed)
            ds.attrs["WASS2S_dimensions"] = "T,Y,X"
            _was_write_netcdf(ds, yf, encoding={'PRCP': {'zlib': True, 'complevel': 4, 'dtype': 'float32', '_FillValue': -9999.0}})
            yearly_files.append(yf)
            print(f"Saved ARC2 yearly cache: {yf}")

        ds_out = self._WAS_concat_grid_files(yearly_files, "T")
        ds_out = ds_out.sel(T=slice(f"{year_start}-01-01", f"{year_end}-12-31"))
        # Continuous calendar; genuine missing days remain NaN.
        if ds_out.sizes.get("T", 0):
            full_t = pd.date_range(ds_out["T"].values[0], ds_out["T"].values[-1], freq="D")
            ds_out = ds_out.reindex(T=full_t)
        _was_write_netcdf(ds_out, final, encoding={'PRCP': {'zlib': True, 'complevel': 4, 'dtype': 'float32', '_FillValue': -9999.0}})
        ds_out.close()
        if not keep_yearly_cache:
            for p in yearly_files:
                p.unlink(missing_ok=True)
        print(f"Saved ARC2 daily: {final}")
        return final


    def WAS_Download_ARC2_Monthly(
        self,
        dir_to_save,
        year_start,
        year_end,
        area,
        force_download=False,
        force_daily_download=False,
        min_daily_coverage=1.0,
        complete_months_only=True,
        max_retries=3,
        retry_delay=5,
        daily_file=None,
    ):
        """Build ARC2 monthly totals from the official daily ARC2 archive."""
        from pathlib import Path
        import calendar
        import pandas as pd
        import xarray as xr

        if not 0 < float(min_daily_coverage) <= 1:
            raise ValueError("min_daily_coverage must be in (0, 1].")
        out_dir = Path(dir_to_save)
        out_dir.mkdir(parents=True, exist_ok=True)
        final = out_dir / f"Monthly_ARC2_PRCP_{year_start}_{year_end}.nc"
        if final.exists() and not force_download:
            print(f"{final} already exists. Skipping.")
            return final

        daily = Path(daily_file) if daily_file is not None else out_dir / f"Daily_ARC2_PRCP_{year_start}_{year_end}.nc"
        if daily_file is not None and not daily.is_file():
            raise FileNotFoundError(f"daily_file does not exist: {daily}")
        if daily_file is not None and force_daily_download:
            raise ValueError("force_daily_download cannot replace an explicitly supplied daily_file.")
        if force_daily_download or not daily.exists():
            daily = self.WAS_Download_ARC2_Daily(
                out_dir, year_start, year_end, area,
                force_download=force_daily_download,
                max_retries=max_retries,
                retry_delay=retry_delay,
                keep_yearly_cache=True,
                allow_incomplete=True,
            )

        with xr.open_dataset(daily) as src:
            subset = self._WAS_standardize_grid_dataset(src, area=area)
            subset = subset.sel(T=slice(f"{year_start}-01-01", f"{year_end}-12-31"))
            out = _was_monthly_rainfall(subset, min_daily_coverage, complete_months_only)
            out = _was_complete_monthly_axis(out, year_start, year_end, complete_months_only)
            out.attrs["source"] = "ARC2 daily aggregated by WASS2S"
            out.attrs["daily_source_file"] = str(daily.resolve())
            _was_write_netcdf(out, final, encoding={v:{"zlib":True,"complevel":4} for v in out.data_vars})
            out.close()
        print(f"Saved ARC2 monthly: {final}")
        return final


    def WAS_Download_GPCC_Monthly(
        self,
        dir_to_save,
        year_start,
        year_end,
        area,
        resolution="0.25",
        include_infill=True,
        include_diagnostics=False,
        force_download=False,
        max_retries=3,
        retry_delay=5,
        keep_compressed=True,
        allow_incomplete=True,
    ):
        """Download GPCC Precipitation Analysis Monthly v2025 on its native grid."""
        from pathlib import Path
        from urllib.parse import urljoin
        import gzip
        import re
        import requests
        import xarray as xr

        res = str(resolution)
        dirmap = {"0.25": ("025_degree", "025"), "0.5": ("050_degree", "050"),
                  "1.0": ("100_degree", "100"), "2.5": ("250_degree", "250")}
        if res not in dirmap:
            raise ValueError("resolution must be one of '0.25', '0.5', '1.0', '2.5'.")
        resdir, tag = dirmap[res]
        base = (
            "https://opendata.dwd.de/climate_environment/GPCC/"
            f"GPCC_Precipitation_Analysis_Monthly/Version_2025/{resdir}/"
        )
        out_dir = Path(dir_to_save)
        out_dir.mkdir(parents=True, exist_ok=True)
        cache = out_dir / ".gpcc_monthly_cache" / f"v2025_{tag}"
        source_cache = cache / "source"
        regional_cache = cache / "regional" / _was_cache_key(area=area, start=year_start, end=year_end, infill=include_infill, diagnostics=include_diagnostics)
        source_cache.mkdir(parents=True, exist_ok=True)
        regional_cache.mkdir(parents=True, exist_ok=True)
        final = out_dir / f"Monthly_GPCC_PRCP_{year_start}_{year_end}_{tag}.nc"
        if final.exists() and not force_download:
            print(f"{final} already exists. Skipping.")
            return final

        html = requests.get(base, timeout=60)
        html.raise_for_status()
        hrefs = sorted(set(re.findall(r'href=["\']([^"\']+\.nc\.gz)["\']', html.text, flags=re.I)))
        if not hrefs:
            raise RuntimeError(f"No GPCC monthly NetCDF archives found at {base}")

        def interval(name):
            m = re.search(r"monthly_(\d{4})_(\d{4})_v2025", name)
            if m:
                return (int(m.group(1)), 1, int(m.group(2)), 12)
            m = re.search(r"monthly_(\d{4})_(\d{2})_v2025", name)
            if m:
                y, mo = int(m.group(1)), int(m.group(2))
                return (y, mo, y, mo)
            return None

        selected = []
        for name in hrefs:
            span = interval(name)
            if span is None:
                continue
            y0, _, y1, _ = span
            if y1 >= int(year_start) and y0 <= int(year_end):
                selected.append(name)
        if not selected:
            raise RuntimeError("No GPCC monthly files overlap the requested years.")

        var_map = {"precip": "PRCP"}
        if include_infill:
            var_map["precip_infill"] = "PRCP_INFILL"
        if include_diagnostics:
            var_map.update({
                "gauge": "GAUGE",
                "gauge_int": "GAUGE_INT",
                "dist_int": "DIST_INT",
                "err": "INTERP_ERROR",
                "sys_err": "SYS_ERROR_FACTOR",
                "liquid": "LIQUID_FRACTION",
                "solid": "SOLID_FRACTION",
            })

        regional_files = []
        for name in selected:
            reg = regional_cache / name.replace(".nc.gz", "_regional.nc")
            if reg.exists() and not force_download:
                regional_files.append(reg)
                continue
            gz_path = source_cache / name
            if not gz_path.exists() or force_download:
                self._WAS_http_download(
                    urljoin(base, name), gz_path,
                    max_retries=max_retries, retry_delay=retry_delay,
                    timeout=600,
                )
            tmp_nc = source_cache / name[:-3]
            try:
                with gzip.open(gz_path, "rb") as src, open(tmp_nc, "wb") as dst:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)
                with xr.open_dataset(tmp_nc) as ds:
                    ds = self._WAS_standardize_grid_dataset(ds, area=area).load()
                keep = [v for v in var_map if v in ds.data_vars]
                if "precip" not in keep:
                    raise ValueError(f"GPCC file {name} has no 'precip' variable. Found {list(ds.data_vars)}")
                ds = ds[keep].rename({v: var_map[v] for v in keep})
                if "T" in ds.coords:
                    ds = ds.sel(T=slice(f"{year_start}-01-01", f"{year_end}-12-31"))
                if ds.sizes.get("T", 1) == 0:
                    continue
                ds.attrs.update({
                    "source": "DWD GPCC Precipitation Analysis Monthly Version 2025",
                    "resolution": res,
                    "WASS2S_dimensions": "T,Y,X",
                })
                ds["PRCP"].attrs.setdefault("units", "mm/month")
                _was_write_netcdf(ds, reg)
                regional_files.append(reg)
            finally:
                tmp_nc.unlink(missing_ok=True)
                if not keep_compressed:
                    gz_path.unlink(missing_ok=True)

        if not regional_files:
            raise RuntimeError("No GPCC monthly regional data could be created.")
        ds_out = self._WAS_concat_grid_files(regional_files, "T")
        if "T" in ds_out.coords:
            ds_out = ds_out.sel(T=slice(f"{year_start}-01-01", f"{year_end}-12-31"))
        actual_years = set(ds_out["T"].dt.year.values.tolist()) if "T" in ds_out.coords else set()
        missing_years = [y for y in range(int(year_start), int(year_end)+1) if y not in actual_years]
        if missing_years and not allow_incomplete:
            raise RuntimeError(f"GPCC monthly missing requested years: {missing_years}")
        ds_out.attrs["requested_year_start"] = int(year_start)
        ds_out.attrs["requested_year_end"] = int(year_end)
        _was_write_netcdf(ds_out, final, encoding={v: {'zlib': True, 'complevel': 4} for v in ds_out.data_vars})
        ds_out.close()
        print(f"Saved GPCC monthly: {final}")
        return final


    def WAS_Download_GPCC_Daily(
        self,
        dir_to_save,
        year_start,
        year_end,
        area,
        resolution="0.5",
        include_diagnostics=False,
        keep_compressed=False,
        keep_yearly_cache=True,
        force_download=False,
        max_retries=3,
        retry_delay=5,
    ):
        """Download GPCC Precipitation Analysis Daily v2025 (0.5° or 1.0°)."""
        from pathlib import Path
        from urllib.parse import urljoin
        import gzip
        import re
        import requests
        import xarray as xr

        res = str(resolution)
        dirmap = {"0.5": ("050_degree", "050"), "1.0": ("100_degree", "100")}
        if res not in dirmap:
            raise ValueError("GPCC Daily v2025 resolution must be '0.5' or '1.0'.")
        resdir, tag = dirmap[res]
        base = (
            "https://opendata.dwd.de/climate_environment/GPCC/"
            f"GPCC_Precipitation_Analysis_Daily/Version_2025/{resdir}/"
        )
        out_dir = Path(dir_to_save)
        out_dir.mkdir(parents=True, exist_ok=True)
        cache = out_dir / ".gpcc_daily_cache" / f"v2025_{tag}"
        source_cache = cache / "source"
        regional_cache = cache / "regional_yearly" / _was_cache_key(area=area, diagnostics=include_diagnostics)
        source_cache.mkdir(parents=True, exist_ok=True)
        regional_cache.mkdir(parents=True, exist_ok=True)
        final = out_dir / f"Daily_GPCC_PRCP_{year_start}_{year_end}_{tag}.nc"
        if final.exists() and not force_download:
            print(f"{final} already exists. Skipping.")
            return final

        html = requests.get(base, timeout=60)
        html.raise_for_status()
        hrefs = sorted(set(re.findall(r'href=["\']([^"\']+\.nc\.gz)["\']', html.text, flags=re.I)))
        if not hrefs:
            raise RuntimeError(f"No GPCC Daily archives found at {base}")

        def year_from_name(name):
            patterns = [
                r"daily_(\d{4})_v2025",
                r"_(\d{4})_v2025_",
                r"v2025_[^_]*_(\d{4})\.nc",
                r"_(19\d{2}|20\d{2})\.nc",
            ]
            for pat in patterns:
                m = re.search(pat, name)
                if m:
                    return int(m.group(1))
            # Last resort: select a plausible year other than the version token.
            years = [int(x) for x in re.findall(r"(?:19|20)\d{2}", name)]
            for y in years:
                if y != 2025 or years.count(2025) > 1:
                    return y
            return None

        by_year = {}
        for name in hrefs:
            y = year_from_name(name)
            if y is not None:
                by_year[y] = name

        requested_years = range(int(year_start), int(year_end)+1)
        regional_files = []
        missing = []
        for year in requested_years:
            name = by_year.get(year)
            if name is None:
                missing.append(year)
                continue
            reg = regional_cache / f"GPCC_Daily_{year}_{tag}.nc"
            if reg.exists() and not force_download:
                regional_files.append(reg)
                continue
            gz_path = source_cache / name
            if not gz_path.exists() or force_download:
                self._WAS_http_download(
                    urljoin(base, name), gz_path,
                    max_retries=max_retries, retry_delay=retry_delay,
                    timeout=600,
                )
            tmp_nc = source_cache / name[:-3]
            try:
                with gzip.open(gz_path, "rb") as src, open(tmp_nc, "wb") as dst:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)
                with xr.open_dataset(tmp_nc) as ds:
                    ds = self._WAS_standardize_grid_dataset(ds, area=area).load()
                if "precip" not in ds.data_vars:
                    candidates = [v for v in ds.data_vars if "precip" in v.lower()]
                    if not candidates:
                        raise ValueError(f"No precipitation variable in {name}: {list(ds.data_vars)}")
                    ds = ds.rename({candidates[0]: "precip"})
                keep = ["precip"]
                if include_diagnostics:
                    keep += [v for v in ("gauge", "gauge_int", "dist_int", "err", "sys_err") if v in ds.data_vars]
                ds = ds[keep]
                ren = {"precip": "PRCP", "gauge": "GAUGE", "gauge_int": "GAUGE_INT",
                       "dist_int": "DIST_INT", "err": "INTERP_ERROR", "sys_err": "SYS_ERROR_FACTOR"}
                ds = ds.rename({k: v for k, v in ren.items() if k in ds.data_vars})
                if "T" in ds.coords:
                    ds = ds.sel(T=slice(f"{year}-01-01", f"{year}-12-31"))
                ds["PRCP"].attrs.setdefault("units", "mm/day")
                ds.attrs.update({
                    "source": "DWD GPCC Precipitation Analysis Daily Version 2025",
                    "resolution": res,
                    "WASS2S_dimensions": "T,Y,X",
                })
                _was_write_netcdf(ds, reg)
                regional_files.append(reg)
            finally:
                tmp_nc.unlink(missing_ok=True)
                if not keep_compressed:
                    gz_path.unlink(missing_ok=True)

        if missing:
            print(f"[GPCC Daily] unavailable years in the current v2025 archive: {missing}")
        if not regional_files:
            raise RuntimeError("No GPCC Daily files available for the requested period.")
        ds_out = self._WAS_concat_grid_files(regional_files, "T")
        if "T" in ds_out.coords:
            ds_out = ds_out.sel(T=slice(f"{year_start}-01-01", f"{year_end}-12-31"))
        _was_write_netcdf(ds_out, final, encoding={v: {'zlib': True, 'complevel': 4} for v in ds_out.data_vars})
        ds_out.close()
        if not keep_yearly_cache:
            for p in regional_files:
                p.unlink(missing_ok=True)
        print(f"Saved GPCC daily: {final}")
        return final


    def WAS_Download_CRU_Monthly(
        self,
        dir_to_save,
        variables,
        year_start,
        year_end,
        area,
        version="4.10",
        force_download=False,
        keep_compressed=True,
        max_retries=3,
        retry_delay=5,
        base_url=None,
    ):
        """Download CRU TS monthly data (default v4.10, 0.5°) in WASS2S format."""
        from pathlib import Path
        from urllib.parse import urljoin
        import gzip
        import re
        import requests
        import xarray as xr

        if str(version) != "4.10" and base_url is None:
            raise ValueError("For CRU versions other than 4.10, provide base_url explicitly.")
        if int(year_end) > 2025 and str(version) == "4.10":
            raise ValueError("CRU TS v4.10 covers through 2025.")
        if isinstance(variables, str):
            variables = [variables]
        mapping = {
            "CRU.PRCP": ("pre", "PRCP"), "CRU.PRE": ("pre", "PRCP"),
            "CRU.TEMP": ("tmp", "TEMP"), "CRU.TMP": ("tmp", "TEMP"),
            "CRU.TMAX": ("tmx", "TMAX"), "CRU.TMX": ("tmx", "TMAX"),
            "CRU.TMIN": ("tmn", "TMIN"), "CRU.TMN": ("tmn", "TMIN"),
            "CRU.DTR": ("dtr", "DTR"), "CRU.VAP": ("vap", "VAP"),
            "CRU.CLD": ("cld", "CLD"), "CRU.WET": ("wet", "WET"),
            "CRU.FRS": ("frs", "FRS"), "CRU.PET": ("pet", "PET"),
        }
        root = base_url or "https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_4.10/cruts.2604091129.v4.10"
        out_dir = Path(dir_to_save)
        out_dir.mkdir(parents=True, exist_ok=True)
        cache = out_dir / ".cru_ts_cache" / f"v{version}"
        cache.mkdir(parents=True, exist_ok=True)
        results = {}

        for requested in variables:
            key = str(requested).upper()
            if key not in mapping:
                raise ValueError(f"Unsupported CRU variable: {requested}")
            native, code = mapping[key]
            var_url = f"{root.rstrip('/')}/{native}/"
            final = out_dir / f"Monthly_CRU_{code}_{year_start}_{year_end}_v{version}.nc"
            if final.exists() and not force_download:
                results[requested] = final
                print(f"{final} already exists. Skipping.")
                continue

            r = requests.get(var_url, timeout=60)
            r.raise_for_status()
            hrefs = sorted(set(re.findall(r'href=["\']([^"\']+\.dat\.nc\.gz)["\']', r.text, flags=re.I)))
            if not hrefs:
                raise RuntimeError(f"No CRU NetCDF archives found at {var_url}")

            def span(name):
                m = re.search(r"cru_ts[\d.]+\.(\d{4})\.(\d{4})\.", name)
                return (int(m.group(1)), int(m.group(2))) if m else None

            selected = []
            for name in hrefs:
                s = span(name)
                if s and s[1] >= int(year_start) and s[0] <= int(year_end):
                    # Prefer decade/short packages over the single 1901-2025 monolith.
                    selected.append((s[1]-s[0], name))
            # Eliminate overlapping monolithic files when shorter packages cover the period.
            selected = [name for _, name in sorted(selected)]
            if not selected:
                raise RuntimeError(f"No CRU files overlap {year_start}-{year_end} for {native}")

            # Greedy coverage by shortest ranges first.
            needed = set(range(int(year_start), int(year_end)+1))
            chosen = []
            for name in selected:
                y0, y1 = span(name)
                cover = needed.intersection(range(y0, y1+1))
                if cover:
                    chosen.append(name)
                    needed -= cover
                if not needed:
                    break
            if needed:
                raise RuntimeError(f"CRU archive selection failed to cover years: {sorted(needed)}")

            regional = []
            var_cache = cache / native
            var_cache.mkdir(parents=True, exist_ok=True)
            for name in chosen:
                reg = var_cache / _was_cache_key(area=area, start=year_start, end=year_end) / name.replace(".dat.nc.gz", "_regional.nc")
                if reg.exists() and not force_download:
                    regional.append(reg)
                    continue
                gz_path = var_cache / name
                if not gz_path.exists() or force_download:
                    self._WAS_http_download(urljoin(var_url, name), gz_path,
                                            max_retries=max_retries, retry_delay=retry_delay, timeout=600)
                tmp = var_cache / name[:-3]
                try:
                    with gzip.open(gz_path, "rb") as src, open(tmp, "wb") as dst:
                        while True:
                            chunk = src.read(1024*1024)
                            if not chunk:
                                break
                            dst.write(chunk)
                    with xr.open_dataset(tmp) as ds:
                        ds = self._WAS_standardize_grid_dataset(ds, area=area).load()
                    source = native if native in ds.data_vars else next(
                        (v for v in ds.data_vars if v.lower() == native), None
                    )
                    if source is None:
                        raise ValueError(f"CRU variable {native} not found in {name}: {list(ds.data_vars)}")
                    ds = ds[[source]].rename({source: code})
                    ds = ds.sel(T=slice(f"{year_start}-01-01", f"{year_end}-12-31"))
                    if ds.sizes.get("T", 0) == 0:
                        continue
                    ds.attrs.update({"source": f"CRU TS v{version}", "resolution": "0.5 degree", "WASS2S_dimensions": "T,Y,X"})
                    _was_write_netcdf(ds, reg)
                    regional.append(reg)
                finally:
                    tmp.unlink(missing_ok=True)
                    if not keep_compressed:
                        gz_path.unlink(missing_ok=True)

            ds_out = self._WAS_concat_grid_files(regional, "T")
            ds_out = ds_out.sel(T=slice(f"{year_start}-01-01", f"{year_end}-12-31"))
            _was_write_netcdf(ds_out, final, encoding={code: {'zlib': True, 'complevel': 4, 'dtype': 'float32'}})
            ds_out.close()
            results[requested] = final
            print(f"Saved CRU: {final}")

        return results


    def WAS_Download_GSMaP_Daily(
        self,
        dir_to_save,
        year_start,
        year_end,
        area,
        product="gauge_standard",
        version="v8",
        algorithm_version="v8.0000",
        time_window="00Z-23Z",
        gportal_user=None,
        gportal_password=None,
        sftp_host="ftp.gportal.jaxa.jp",
        sftp_port=2051,
        remote_root="/standard/v8",
        force_download=False,
        max_retries=3,
        retry_delay=5,
        keep_yearly_cache=True,
        known_hosts=None,
    ):
        """
        Download GSMaP v8 daily 0.1° rainfall using JAXA G-Portal SFTP.

        Current default is the gauge-calibrated Standard product. G-Portal account
        credentials may be passed explicitly or through JAXA_GPORTAL_USER and
        JAXA_GPORTAL_PASSWORD.  The host/port/root are parameters because JAXA has
        announced a G-Portal infrastructure migration for October 2026.
        """
        from pathlib import Path
        import calendar
        import gzip
        import os
        import time as _time
        import numpy as np
        import pandas as pd
        import xarray as xr

        try:
            import paramiko
        except ImportError as exc:
            raise ImportError(
                "WAS_Download_GSMaP_Daily requires 'paramiko' for SFTP. "
                "Install it in the WASS2S environment with: pip install paramiko"
            ) from exc

        product = str(product).lower()
        if product not in {"gauge_standard", "standard"}:
            raise ValueError("product must be 'gauge_standard' or 'standard'.")
        if time_window not in {"00Z-23Z", "12Z-11Z"}:
            raise ValueError("time_window must be '00Z-23Z' or '12Z-11Z'.")
        user = gportal_user or os.getenv("JAXA_GPORTAL_USER")
        password = gportal_password or os.getenv("JAXA_GPORTAL_PASSWORD")
        if not user or not password:
            raise ValueError(
                "GSMaP download needs G-Portal credentials. Set JAXA_GPORTAL_USER "
                "and JAXA_GPORTAL_PASSWORD or pass gportal_user/gportal_password."
            )

        out_dir = Path(dir_to_save)
        out_dir.mkdir(parents=True, exist_ok=True)
        cache = out_dir / ".gsmap_daily_cache" / f"{product}_{version}" / _was_cache_key(area=area, window=time_window, algorithm=algorithm_version, root=remote_root)
        cache.mkdir(parents=True, exist_ok=True)
        tag_window = "00Z23Z" if time_window == "00Z-23Z" else "12Z11Z"
        ptag = "Gauge" if product == "gauge_standard" else "Standard"
        final = out_dir / f"Daily_GSMaP_{ptag}_PRCP_{year_start}_{year_end}_{tag_window}.nc"
        if final.exists() and not force_download:
            print(f"{final} already exists. Skipping.")
            return final

        # Native v8 daily binary: 3600 lon x 1200 lat, 0.1°, 60N..60S,
        # first cell center 0.05E, 59.95N, little-endian float32, mm/hr.
        X0 = np.round(0.05 + np.arange(3600) * 0.1, 10)
        Xwrap = ((X0 + 180.0) % 360.0) - 180.0
        Ydesc = np.round(59.95 - np.arange(1200) * 0.1, 10)
        north, west, south, east = [float(v) for v in area]
        # Read on native ordering, subset after wrapping/sorting.

        ssh = paramiko.SSHClient()
        ssh.load_system_host_keys()
        if known_hosts is not None:
            ssh.load_host_keys(str(Path(known_hosts).expanduser()))
        ssh.set_missing_host_key_policy(paramiko.RejectPolicy())
        try:
            ssh.connect(sftp_host, port=int(sftp_port), username=user, password=password,
                        timeout=60, auth_timeout=60, look_for_keys=False, allow_agent=False)
            sftp = ssh.open_sftp()
        except Exception:
            ssh.close()
            raise
        yearly_files = []
        try:
            for year in range(int(year_start), int(year_end)+1):
                yf = cache / f"GSMaP_{product}_{year}_{tag_window}.nc"
                if yf.exists() and not force_download:
                    yearly_files.append(yf)
                    continue
                dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
                now = pd.Timestamp.utcnow().tz_localize(None).normalize()
                dates = dates[dates <= now]
                pieces = []
                missing = []
                for date in dates:
                    ymd = date.strftime("%Y%m%d")
                    ym = date.strftime("%Y%m")
                    subdir = "00Z-23Z" if time_window == "00Z-23Z" else "p12Z-11Z"
                    if product == "gauge_standard":
                        folder = f"{remote_root.rstrip('/')}/daily_G/{subdir}/{ym}"
                        win = "00Z-23Z" if time_window == "00Z-23Z" else "p12Z-11Z"
                        stem = f"gsmap_gauge.{ymd}.0.1d.daily.{win}.{algorithm_version}.dat.gz"
                        fallback = f"gsmap_gauge.{ymd}.0.1d.daily.{win}.dat.gz"
                    else:
                        folder = f"{remote_root.rstrip('/')}/daily/{subdir}/{ym}"
                        win = "00Z-23Z" if time_window == "00Z-23Z" else "p12Z-11Z"
                        stem = f"gsmap.{ymd}.0.1d.daily.{win}.{algorithm_version}.dat.gz"
                        fallback = f"gsmap.{ymd}.0.1d.daily.{win}.dat.gz"

                    local = cache / f"raw_{ymd}.dat.gz"
                    ok = False
                    for candidate in (stem, fallback):
                        remote = f"{folder}/{candidate}"
                        for attempt in range(1, max_retries+1):
                            try:
                                if local.exists() and not force_download:
                                    ok = True
                                else:
                                    part = local.with_name(local.name + ".part")
                                    try:
                                        sftp.get(remote, str(part))
                                        with gzip.open(part, "rb") as check:
                                            payload_check = check.read()
                                        if len(payload_check) != 1200 * 3600 * 4:
                                            raise ValueError("Invalid GSMaP binary length; refusing to cache partial download.")
                                        os.replace(part, local)
                                    finally:
                                        part.unlink(missing_ok=True)
                                    ok = True
                                break
                            except Exception as exc:
                                if attempt < max_retries:
                                    _time.sleep(retry_delay)
                                else:
                                    last_exc = exc
                        if ok:
                            break
                    if not ok:
                        missing.append(ymd)
                        continue
                    try:
                        with gzip.open(local, "rb") as fh:
                            arr = np.frombuffer(fh.read(), dtype="<f4").astype("float32", copy=True)
                        if arr.size != 1200 * 3600:
                            raise ValueError(f"Unexpected GSMaP binary size {arr.size} for {ymd}")
                        arr = arr.reshape(1200, 3600)
                        arr[arr <= -999.0] = np.nan
                        da = xr.DataArray(
                            arr, dims=("Y", "X"),
                            coords={"Y": Ydesc, "X": Xwrap}, name="PRCP"
                        ).sortby("X").sortby("Y")
                        da = da.sel(X=slice(west, east), Y=slice(south, north))
                        # Daily file is a 24-h mean rate [mm/hr]. Convert to total [mm/day].
                        da = (da * 24.0).astype("float32")
                        da = da.expand_dims(T=[date.to_datetime64()])
                        pieces.append(da)
                    finally:
                        local.unlink(missing_ok=True)

                if pieces:
                    da_year = xr.concat(pieces, dim="T").sortby("T")
                else:
                    print(f"[GSMaP] No source day available for {year}; skipping yearly cache.")
                    continue
                da_year.attrs.update({
                    "units": "mm/day",
                    "native_units": "mm/hr (24-hour mean rate)",
                    "source": f"JAXA GSMaP {product} {version}",
                    "time_window": time_window,
                    "resolution": "0.1 degree",
                })
                ds = da_year.to_dataset()
                ds.attrs["missing_source_days"] = ",".join(missing)
                ds.attrs["WASS2S_dimensions"] = "T,Y,X"
                _was_write_netcdf(ds, yf, encoding={'PRCP': {'zlib': True, 'complevel': 4, 'dtype': 'float32', '_FillValue': -9999.0}})
                yearly_files.append(yf)
        finally:
            sftp.close()
            ssh.close()

        ds_out = self._WAS_concat_grid_files(yearly_files, "T")
        ds_out = ds_out.sel(T=slice(f"{year_start}-01-01", f"{year_end}-12-31"))
        if ds_out.sizes.get("T", 0):
            full_t = pd.date_range(f"{int(year_start)}-01-01", min(pd.Timestamp(int(year_end),12,31), pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()), freq="D")
            ds_out = ds_out.reindex(T=full_t)
        _was_write_netcdf(ds_out, final, encoding={'PRCP': {'zlib': True, 'complevel': 4, 'dtype': 'float32', '_FillValue': -9999.0}})
        ds_out.close()
        if not keep_yearly_cache:
            for p in yearly_files:
                p.unlink(missing_ok=True)
        print(f"Saved GSMaP daily: {final}")
        return final


    def WAS_Download_CPCUnified_Daily(
        self,
        dir_to_save,
        year_start,
        year_end,
        area,
        force_download=False,
        keep_source_cache=False,
        keep_yearly_cache=True,
        max_retries=3,
        retry_delay=5,
    ):
        """Download CPC Global Unified Gauge-Based Daily Precipitation (0.5°)."""
        from pathlib import Path
        import pandas as pd
        import xarray as xr

        out_dir = Path(dir_to_save)
        out_dir.mkdir(parents=True, exist_ok=True)
        cache = out_dir / ".cpc_unified_daily_cache"
        source_cache = cache / "source"
        regional_cache = cache / "regional_yearly" / _was_cache_key(area=area)
        source_cache.mkdir(parents=True, exist_ok=True)
        regional_cache.mkdir(parents=True, exist_ok=True)
        final = out_dir / f"Daily_CPCUnified_PRCP_{year_start}_{year_end}.nc"
        if final.exists() and not force_download:
            print(f"{final} already exists. Skipping.")
            return final

        base = "https://psl.noaa.gov/thredds/fileServer/Datasets/cpc_global_precip"
        regional_files = []
        for year in range(int(year_start), int(year_end)+1):
            reg = regional_cache / f"CPCUnified_Daily_{year}.nc"
            if reg.exists() and not force_download:
                regional_files.append(reg)
                continue
            src = source_cache / f"precip.{year}.nc"
            if not src.exists() or force_download:
                self._WAS_http_download(
                    f"{base}/precip.{year}.nc", src,
                    max_retries=max_retries, retry_delay=retry_delay, timeout=600,
                )
            try:
                with xr.open_dataset(src) as ds:
                    ds = self._WAS_standardize_grid_dataset(ds, area=area).load()
                if "precip" not in ds.data_vars:
                    raise ValueError(f"CPC Unified file lacks precip: {list(ds.data_vars)}")
                ds = ds[["precip"]].rename({"precip": "PRCP"})
                ds["PRCP"] = ds["PRCP"].where(ds["PRCP"] > -1e20).astype("float32")
                ds["PRCP"].attrs.update({"units": "mm/day", "long_name": "CPC Global Unified daily precipitation"})
                ds.attrs.update({"source": "NOAA PSL / CPC Global Unified Gauge-Based Analysis", "resolution": "0.5 degree", "WASS2S_dimensions": "T,Y,X"})
                _was_write_netcdf(ds, reg, encoding={'PRCP': {'zlib': True, 'complevel': 4, 'dtype': 'float32', '_FillValue': -9999.0}})
                regional_files.append(reg)
            finally:
                if not keep_source_cache:
                    src.unlink(missing_ok=True)

        ds_out = self._WAS_concat_grid_files(regional_files, "T")
        ds_out = ds_out.sel(T=slice(f"{year_start}-01-01", f"{year_end}-12-31"))
        if ds_out.sizes.get("T", 0):
            full_t = pd.date_range(ds_out["T"].values[0], ds_out["T"].values[-1], freq="D")
            ds_out = ds_out.reindex(T=full_t)
        _was_write_netcdf(ds_out, final, encoding={'PRCP': {'zlib': True, 'complevel': 4, 'dtype': 'float32', '_FillValue': -9999.0}})
        ds_out.close()
        if not keep_yearly_cache:
            for p in regional_files:
                p.unlink(missing_ok=True)
        print(f"Saved CPC Unified daily: {final}")
        return final


    def WAS_Download_CPCUnified_Monthly(
        self,
        dir_to_save,
        year_start,
        year_end,
        area,
        force_download=False,
        force_daily_download=False,
        min_daily_coverage=1.0,
        complete_months_only=True,
        keep_source_cache=False,
        keep_yearly_cache=True,
        max_retries=3,
        retry_delay=5,
        daily_file=None,
    ):
        """Aggregate CPC Global Unified daily precipitation to monthly totals."""
        from pathlib import Path
        import calendar
        import pandas as pd
        import xarray as xr

        if not 0 < float(min_daily_coverage) <= 1:
            raise ValueError("min_daily_coverage must be in (0, 1].")
        out_dir = Path(dir_to_save)
        out_dir.mkdir(parents=True, exist_ok=True)
        final = out_dir / f"Monthly_CPCUnified_PRCP_{year_start}_{year_end}.nc"
        if final.exists() and not force_download:
            print(f"{final} already exists. Skipping.")
            return final
        daily = Path(daily_file) if daily_file is not None else out_dir / f"Daily_CPCUnified_PRCP_{year_start}_{year_end}.nc"
        if daily_file is not None and not daily.is_file():
            raise FileNotFoundError(f"daily_file does not exist: {daily}")
        if daily_file is not None and force_daily_download:
            raise ValueError("force_daily_download cannot replace an explicitly supplied daily_file.")
        if force_daily_download or not daily.exists():
            daily = self.WAS_Download_CPCUnified_Daily(
                out_dir, year_start, year_end, area,
                force_download=force_daily_download,
                keep_source_cache=keep_source_cache,
                keep_yearly_cache=keep_yearly_cache,
                max_retries=max_retries, retry_delay=retry_delay,
            )
        with xr.open_dataset(daily) as src:
            subset = self._WAS_standardize_grid_dataset(src, area=area)
            subset = subset.sel(T=slice(f"{year_start}-01-01", f"{year_end}-12-31"))
            out = _was_monthly_rainfall(subset, min_daily_coverage, complete_months_only)
            out = _was_complete_monthly_axis(out, year_start, year_end, complete_months_only)
            out.attrs["source"] = "CPCUnified daily aggregated by WASS2S"
            out.attrs["daily_source_file"] = str(daily.resolve())
            _was_write_netcdf(out, final, encoding={v:{"zlib":True,"complevel":4} for v in out.data_vars})
            out.close()
        print(f"Saved CPC Unified monthly: {final}")
        return final


    def _WAS_Download_NCEI_Station_API(
        self,
        dataset_id,
        output_prefix,
        dir_to_save,
        year_start,
        year_end,
        area,
        variables,
        variable_mapping,
        frequency="daily",
        include_qc=False,
        include_provenance=False,
        force_download=False,
        max_retries=3,
        retry_delay=5,
    ):
        """Internal NCEI Access Data Service downloader for station products."""
        from pathlib import Path
        import io
        import time as _time
        import numpy as np
        import pandas as pd
        import requests
        import xarray as xr

        if isinstance(variables, str):
            variables = [variables]
        requested = [str(v).upper() for v in variables]
        unknown = [v for v in requested if v not in variable_mapping]
        if unknown:
            raise ValueError(f"Unsupported {output_prefix} variable(s): {unknown}")
        native_types = [variable_mapping[v] for v in requested]
        out_dir = Path(dir_to_save)
        out_dir.mkdir(parents=True, exist_ok=True)
        cache = out_dir / f".{output_prefix.lower()}_api_cache"
        cache.mkdir(parents=True, exist_ok=True)
        final = out_dir / f"{'Hourly' if frequency == 'hourly' else 'Daily'}_{output_prefix}_{year_start}_{year_end}.nc"
        stations_csv = out_dir / f"Stations_{output_prefix}_{year_start}_{year_end}.csv"
        if final.exists() and not force_download:
            print(f"{final} already exists. Skipping.")
            return final

        endpoint = "https://www.ncei.noaa.gov/access/services/data/v1"
        north, west, south, east = [float(v) for v in area]
        # NCEI boundingBox order is N,W,S,E.
        bounding = f"{north},{west},{south},{east}"
        frames = []

        # Daily summaries: one request/year. Hourly: one request/month to avoid
        # service-size limits and to make interruption/restart practical.
        chunks = []
        if frequency == "hourly":
            for year in range(int(year_start), int(year_end)+1):
                for month in range(1, 13):
                    start = pd.Timestamp(year, month, 1)
                    end = start + pd.offsets.MonthEnd(1)
                    chunks.append((start, end, f"{year}{month:02d}"))
        else:
            for year in range(int(year_start), int(year_end)+1):
                chunks.append((pd.Timestamp(year,1,1), pd.Timestamp(year,12,31), str(year)))

        for start, end, label in chunks:
            cache_csv = cache / f"{output_prefix}_{label}.csv"
            if cache_csv.exists() and not force_download:
                try:
                    df = pd.read_csv(cache_csv, low_memory=False)
                    if not df.empty:
                        frames.append(df)
                    continue
                except Exception:
                    cache_csv.unlink(missing_ok=True)

            params = {
                "dataset": dataset_id,
                "dataTypes": ",".join(native_types),
                "startDate": start.strftime("%Y-%m-%d"),
                "endDate": end.strftime("%Y-%m-%d"),
                "boundingBox": bounding,
                "format": "csv",
                "units": "metric",
                "includeStationName": "true",
                "includeStationLocation": "true",
                "includeAttributes": "true" if (include_qc or include_provenance) else "false",
            }
            ok = False
            for attempt in range(1, max_retries+1):
                try:
                    r = requests.get(endpoint, params=params, timeout=180)
                    if r.status_code == 204 or not r.text.strip():
                        ok = True
                        break
                    r.raise_for_status()
                    # NCEI may return JSON error even when content-type is not ideal.
                    if r.text.lstrip().startswith("{") and "error" in r.text[:300].lower():
                        raise RuntimeError(r.text[:1000])
                    cache_csv.write_text(r.text, encoding="utf-8")
                    df = pd.read_csv(io.StringIO(r.text), low_memory=False)
                    if not df.empty:
                        frames.append(df)
                    ok = True
                    break
                except Exception as exc:
                    print(f"[{output_prefix}] {label} attempt {attempt}/{max_retries} failed: {exc}")
                    if attempt < max_retries:
                        _time.sleep(retry_delay)
            if not ok:
                print(f"[{output_prefix}] skipped failed chunk {label}")

        if not frames:
            raise RuntimeError(f"No {output_prefix} station data returned for area={area} and period {year_start}-{year_end}.")
        df = pd.concat(frames, ignore_index=True, sort=False)

        def find_col(candidates):
            lookup = {str(c).upper(): c for c in df.columns}
            for cand in candidates:
                if cand.upper() in lookup:
                    return lookup[cand.upper()]
            return None

        time_col = find_col(["DATE", "DATE_TIME", "TIME", "timestamp"])
        station_col = find_col(["STATION", "STATION_ID", "station"])
        lat_col = find_col(["LATITUDE", "LAT"])
        lon_col = find_col(["LONGITUDE", "LON"])
        elev_col = find_col(["ELEVATION", "ELEV"])
        name_col = find_col(["NAME", "STATION_NAME"])
        if time_col is None or station_col is None:
            raise ValueError(f"Cannot identify DATE/STATION in NCEI columns: {list(df.columns)}")

        df["_T"] = pd.to_datetime(df[time_col], errors="coerce", utc=True).dt.tz_convert(None)
        df["_STATION"] = df[station_col].astype(str)
        df = df.dropna(subset=["_T", "_STATION"])
        if df.empty:
            raise RuntimeError("NCEI response contained no valid station/time records.")

        # Normalize requested data variables to numeric columns.
        renamed = {}
        col_lookup = {str(c).upper(): c for c in df.columns}
        for alias, native in variable_mapping.items():
            if alias not in requested:
                continue
            source = col_lookup.get(str(native).upper())
            if source is not None:
                renamed[source] = alias
        df = df.rename(columns=renamed)
        for alias in requested:
            if alias in df.columns:
                # Attribute-bearing fields can look like "12.3,flags"; keep leading value.
                if df[alias].dtype == object:
                    df[alias] = df[alias].astype(str).str.split(",", n=1).str[0]
                df[alias] = pd.to_numeric(df[alias], errors="coerce")

        # Station metadata table.
        meta_cols = [c for c in (station_col, lat_col, lon_col, elev_col, name_col) if c is not None]
        meta = df[meta_cols].drop_duplicates(subset=[station_col]).copy()
        meta = meta.rename(columns={station_col: "station"})
        if lat_col: meta = meta.rename(columns={lat_col: "LAT"})
        if lon_col: meta = meta.rename(columns={lon_col: "LON"})
        if elev_col: meta = meta.rename(columns={elev_col: "ELEV"})
        if name_col: meta = meta.rename(columns={name_col: "NAME"})
        meta.to_csv(stations_csv, index=False)

        stations = np.array(sorted(df["_STATION"].unique()), dtype=str)
        times = np.array(sorted(df["_T"].unique()), dtype="datetime64[ns]")
        out = xr.Dataset(coords={"T": times, "station": stations})

        # Keep one value per exact station/timestamp. Raw API CSV cache remains the
        # audit trail when multiple source reports exist.
        for alias in requested:
            if alias not in df.columns:
                continue
            tab = df[["_T", "_STATION", alias]].dropna(subset=[alias])
            tab = tab.drop_duplicates(subset=["_T", "_STATION"], keep="first")
            pivot = tab.pivot(index="_T", columns="_STATION", values=alias)
            pivot = pivot.reindex(index=pd.DatetimeIndex(times), columns=stations)
            out[alias] = (("T", "station"), pivot.to_numpy(dtype="float32"))

        # Station metadata as coordinates.
        meta_idx = meta.set_index("station").reindex(stations)
        for name in ("LAT", "LON", "ELEV"):
            if name in meta_idx.columns:
                out = out.assign_coords({name: ("station", pd.to_numeric(meta_idx[name], errors="coerce").to_numpy())})
        if "NAME" in meta_idx.columns:
            out = out.assign_coords(NAME=("station", meta_idx["NAME"].fillna("").astype(str).to_numpy()))

        # Preserve selected textual QC/provenance columns when they are explicitly
        # requested and when NCEI returns them as separate fields.
        if include_qc or include_provenance:
            keys = []
            for c in df.columns:
                uc = str(c).upper()
                if include_qc and any(k in uc for k in ("QUALITY", "_QC", "QC_", "MEASUREMENT_CODE", "MEAS_CODE")):
                    keys.append(c)
                if include_provenance and any(k in uc for k in ("REPORT_TYPE", "SOURCE_CODE", "SOURCE_STATION")):
                    keys.append(c)
            for c in sorted(set(keys))[:40]:
                tab = df[["_T", "_STATION", c]].dropna(subset=[c]).drop_duplicates(["_T", "_STATION"], keep="first")
                if tab.empty:
                    continue
                pivot = tab.pivot(index="_T", columns="_STATION", values=c)
                pivot = pivot.reindex(index=pd.DatetimeIndex(times), columns=stations).fillna("")
                safe_name = str(c).upper().replace(" ", "_").replace("-", "_")[:63]
                out[safe_name] = (("T", "station"), pivot.astype(str).to_numpy(dtype=str))

        out.attrs.update({
            "source": f"NOAA NCEI {dataset_id}",
            "api": "NCEI Access Data Service",
            "units_request": "metric",
            "station_metadata_csv": str(stations_csv),
            "raw_cache_directory": str(cache),
            "WASS2S_dimensions": "T,station",
        })
        enc = {v: {"zlib": True, "complevel": 4} for v in out.data_vars if out[v].dtype.kind not in "OUS"}
        _was_write_netcdf(out, final, encoding=enc)
        out.close()
        print(f"Saved {output_prefix}: {final}")
        return final


    def _WAS_HTTP_download_cached(
        self,
        url,
        target,
        force_download=False,
        max_retries=3,
        retry_delay=5,
        timeout=120,
        quiet_404=True,
    ):
        """Download a remote file atomically, reusing a non-empty cache file."""
        from pathlib import Path
        import os
        import time as _time
        import requests

        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size > 0 and not force_download:
            return True

        part = target.with_name(target.name + '.part')
        for attempt in range(1, max_retries + 1):
            try:
                with requests.get(url, stream=True, timeout=timeout) as response:
                    if response.status_code == 404 and quiet_404:
                        part.unlink(missing_ok=True)
                        return False
                    response.raise_for_status()
                    with open(part, 'wb') as handle:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                handle.write(chunk)
                if part.stat().st_size == 0:
                    raise IOError(f'Empty download: {url}')
                os.replace(part, target)
                return True
            except Exception as exc:
                part.unlink(missing_ok=True)
                if _was_permanent_http_error(exc):
                    raise RuntimeError(f"Permanent HTTP error; request was not retried: {url}") from exc
                if attempt >= max_retries:
                    print(f'[HTTP] {attempt}/{max_retries} failed: {url}\n       {exc}')
                    return False
                print(f'[HTTP] {attempt}/{max_retries} failed: {url}\n       {exc}')
                _time.sleep(retry_delay)
        return False


    def _WAS_GHCNh_station_list(
        self,
        cache_file,
        area=None,
        force_download=False,
        max_retries=3,
        retry_delay=5,
    ):
        """Load the official GHCNh station list and optionally subset a bbox."""
        from pathlib import Path
        import pandas as pd
        import numpy as np

        cache_file = Path(cache_file)
        url = (
            'https://www.ncei.noaa.gov/oa/global-historical-climatology-network/'
            'hourly/doc/ghcnh-station-list.csv'
        )
        ok = self._WAS_HTTP_download_cached(
            url, cache_file, force_download=force_download,
            max_retries=max_retries, retry_delay=retry_delay,
        )
        if not ok:
            raise RuntimeError(f'Could not download GHCNh station list: {url}')

        def _normalize(s):
            return ''.join(ch for ch in str(s).strip().upper() if ch.isalnum())

        # NCEI has published this file both as a conventional CSV and in forms
        # consumed as headerless station metadata. Support both layouts.
        df = pd.read_csv(cache_file, dtype=str)
        norm = {_normalize(c): c for c in df.columns}
        has_named = any(k in norm for k in ('GHCNID', 'STATION', 'STATIONID')) and 'LATITUDE' in norm and 'LONGITUDE' in norm
        if not has_named:
            raw = pd.read_csv(cache_file, header=None, dtype=str)
            names = ['GHCN_ID', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'STATE', 'NAME', 'GSN', 'HCN_CRN', 'WMO_ID', 'ICAO']
            raw.columns = names[:raw.shape[1]] + [f'EXTRA_{i}' for i in range(raw.shape[1] - len(names))]
            df = raw

        norm = {_normalize(c): c for c in df.columns}
        aliases = {
            'station': ('GHCNID', 'STATIONID', 'STATION'),
            'LAT': ('LATITUDE', 'LAT'),
            'LON': ('LONGITUDE', 'LON'),
            'ELEV': ('ELEVATION', 'ELEV', 'ELEVATIONM'),
            'NAME': ('NAME', 'STATIONNAME'),
            'WMO_ID': ('WMOID',),
            'ICAO': ('ICAO',),
        }
        out = pd.DataFrame(index=df.index)
        for dst, cands in aliases.items():
            src_col = next((norm[c] for c in cands if c in norm), None)
            if src_col is not None:
                out[dst] = df[src_col]

        required = {'station', 'LAT', 'LON'}
        if not required.issubset(out.columns):
            raise ValueError(
                f'Unexpected GHCNh station-list structure. Need {sorted(required)}, '
                f'found columns={list(df.columns)}'
            )

        out['station'] = out['station'].astype(str).str.strip()
        out['LAT'] = pd.to_numeric(out['LAT'], errors='coerce')
        out['LON'] = pd.to_numeric(out['LON'], errors='coerce')
        if 'ELEV' in out:
            out['ELEV'] = pd.to_numeric(out['ELEV'], errors='coerce')
        else:
            out['ELEV'] = np.nan
        if 'NAME' not in out:
            out['NAME'] = ''
        out['NAME'] = out['NAME'].fillna('').astype(str).str.strip()
        out = out.dropna(subset=['LAT', 'LON']).drop_duplicates('station')

        if area is not None:
            north, west, south, east = [float(v) for v in area]
            lat_ok = out['LAT'].between(south, north, inclusive='both')
            if west <= east:
                lon_ok = out['LON'].between(west, east, inclusive='both')
            else:  # dateline-crossing bbox
                lon_ok = (out['LON'] >= west) | (out['LON'] <= east)
            out = out.loc[lat_ok & lon_ok].copy()

        return out.sort_values('station').reset_index(drop=True)


    @staticmethod
    def _WAS_find_column(columns, *candidates):
        """Case/punctuation-insensitive dataframe column lookup."""
        def norm(x):
            return ''.join(ch for ch in str(x).strip().lower() if ch.isalnum())
        lookup = {norm(c): c for c in columns}
        for candidate in candidates:
            hit = lookup.get(norm(candidate))
            if hit is not None:
                return hit
        return None


    @staticmethod
    def _WAS_station_dataset_from_long(
        df,
        meta,
        variables,
        native_map,
        start,
        end,
        frequency='daily',
        attrs=None,
        full_calendar=False,
    ):
        """Convert long station observations to a numeric T x station Dataset."""
        import numpy as np
        import pandas as pd
        import xarray as xr

        if df.empty:
            raise RuntimeError('No station observations remain after filtering.')

        station_col = WAS_Download._WAS_find_column(df.columns, 'STATION', 'Station_ID', 'GHCN_ID')
        date_col = WAS_Download._WAS_find_column(df.columns, 'DATE', 'datetime', 'timestamp')
        if station_col is None or date_col is None:
            raise ValueError(f'Cannot identify station/date columns in {list(df.columns)}')

        work = df.dropna(subset=[station_col]).copy()
        work['_STATION'] = work[station_col].astype(str).str.strip()
        work = work[work['_STATION'].ne('')]
        work['_T'] = pd.to_datetime(work[date_col], errors='coerce', utc=True).dt.tz_convert(None)
        # Convert timezone-aware timestamps, if any, to naive UTC-like timestamps.
        try:
            if getattr(work['_T'].dt, 'tz', None) is not None:
                work['_T'] = work['_T'].dt.tz_convert('UTC').dt.tz_localize(None)
        except Exception:
            pass
        work = work.dropna(subset=['_T', '_STATION'])
        work = work[(work['_T'] >= pd.Timestamp(start)) & (work['_T'] <= pd.Timestamp(end))]

        stations = meta['station'].astype(str).tolist()
        work = work[work['_STATION'].isin(stations)]
        if work.empty:
            raise RuntimeError('No observations found for selected stations and time period.')

        if full_calendar and frequency == 'daily':
            times = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq='D')
        else:
            times = pd.DatetimeIndex(sorted(work['_T'].dropna().unique()))

        ds = xr.Dataset(coords={'T': times, 'station': np.asarray(stations, dtype=str)})

        for alias in variables:
            candidates = native_map.get(alias, ())
            if isinstance(candidates, str):
                candidates = (candidates,)
            native = WAS_Download._WAS_find_column(work.columns, *candidates)
            if native is None:
                print(f'[STATION] variable {alias} not present in downloaded files; filling NaN.')
                ds[alias] = (('T', 'station'), np.full((len(times), len(stations)), np.nan, dtype='float32'))
                continue
            tab = work[['_T', '_STATION', native]].copy()
            tab[native] = pd.to_numeric(
                tab[native].astype(str).str.extract(r'([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)', expand=False),
                errors='coerce',
            )
            # GHCNh/SSOD should be harmonized to one record per station-time, but
            # groupby(first non-null) makes the output deterministic if duplicates occur.
            tab = tab.dropna(subset=[native]).groupby(['_T', '_STATION'], as_index=False)[native].first()
            if tab.empty:
                data = np.full((len(times), len(stations)), np.nan, dtype='float32')
            else:
                pivot = tab.pivot(index='_T', columns='_STATION', values=native)
                pivot = pivot.reindex(index=times, columns=stations)
                data = pivot.to_numpy(dtype='float32')
            ds[alias] = (('T', 'station'), data)

        meta_idx = meta.set_index('station').reindex(stations)
        for c in ('LAT', 'LON', 'ELEV'):
            if c in meta_idx:
                ds = ds.assign_coords({c: ('station', pd.to_numeric(meta_idx[c], errors='coerce').to_numpy(dtype=float))})
        if 'NAME' in meta_idx:
            ds = ds.assign_coords(NAME=('station', meta_idx['NAME'].fillna('').astype(str).to_numpy()))
        if attrs:
            ds.attrs.update(attrs)
        return ds


    def WAS_Download_SSOD_Daily(
        self,
        dir_to_save,
        year_start,
        year_end,
        area,
        variables=('PRCP', 'TEMP', 'TMAX', 'TMIN', 'TDEW', 'SLP', 'SP', 'WIND', 'GUST'),
        include_flags=True,
        force_download=False,
        max_retries=3,
        retry_delay=5,
    ):
        """
        Download NOAA/NCEI Synoptic Summary of the Day Version 2 (SSODv2).

        Access strategy
        ---------------
        1. Select GHCNh candidate stations from the official station list.
        2. Try the official SSODv2 station/year file first:
               .../access/by-year/YYYY/ssod_<GHCN_ID>_YYYY.csv
           and the documented upper-case ``SSOD_`` variant.
        3. If the station/year file is absent, fall back to the official
           period-of-record station file:
               .../access/by-station/SSOD_<GHCN_ID>.csv
           then extract only the requested year locally.

        The fallback is important for the current/incomplete year because the
        SSODv2 product is updated daily but the population of the ``by-year``
        tree can lag or can differ from the complete GHCNh candidate list.

        Notes
        -----
        The GHCNh station list is a *candidate* list. Not every GHCNh station
        necessarily has an SSODv2 daily-summary file for every requested year.

        Output
        ------
        Daily_SSOD_<start>_<end>.nc
            WASS2S NetCDF with dimensions ``T, station`` and coordinates
            LAT/LON/ELEV/NAME.

        Stations_SSOD_candidates_<start>_<end>.csv
            All GHCNh stations geographically inside ``area``.

        Stations_SSOD_<start>_<end>.csv
            Only stations for which at least one SSODv2 record was actually
            obtained in the requested period.

        Native SSODv2 units are preserved:
            temperature degC, precipitation mm, pressure hPa, wind m s-1,
            visibility km, snow depth cm.
        """
        from pathlib import Path
        import pandas as pd
        import numpy as np

        variables = [variables] if isinstance(variables, str) else list(variables)

        native_map = {
            'PRCP': ('total_precipitation',),
            'TEMP': ('mean_temperature',),
            'TMAX': ('max_temperature',),
            'TMIN': ('min_temperature',),
            'TDEW': ('mean_dew_point_temperature',),
            'SLP': ('mean_sea_level_pressure',),
            'SP': ('mean_station_level_pressure',),
            'WIND': ('mean_wind_speed',),
            'WINDMAX': ('max_wind_speed',),
            'GUST': ('max_wind_gust',),
            'VISIB': ('mean_visibility',),
            'SNOW': ('Snow Depth', 'snow_depth', 'mean_snow_depth'),
        }

        unknown = [v for v in variables if v not in native_map]
        if unknown:
            raise ValueError(
                f'Unsupported SSOD variable(s): {unknown}. '
                f'Supported: {sorted(native_map)}'
            )

        year_start = int(year_start)
        year_end = int(year_end)
        if year_end < year_start:
            raise ValueError('year_end must be >= year_start.')

        out_dir = Path(dir_to_save)
        out_dir.mkdir(parents=True, exist_ok=True)

        cache = out_dir / '.ssod_v2_cache'
        cache.mkdir(parents=True, exist_ok=True)

        by_year_cache = cache / 'by_year'
        by_station_cache = cache / 'by_station'
        by_year_cache.mkdir(parents=True, exist_ok=True)
        by_station_cache.mkdir(parents=True, exist_ok=True)

        final = out_dir / f'Daily_SSOD_{year_start}_{year_end}.nc'
        stations_candidates_csv = (
            out_dir / f'Stations_SSOD_candidates_{year_start}_{year_end}.csv'
        )
        stations_csv = out_dir / f'Stations_SSOD_{year_start}_{year_end}.csv'
        flags_csv = out_dir / f'SSOD_Flags_{year_start}_{year_end}.csv.gz'

        if final.exists() and not force_download:
            print(f'{final} already exists. Skipping.')
            return final

        # ------------------------------------------------------------------
        # GHCNh candidate station inventory.
        # ------------------------------------------------------------------
        meta_candidates = self._WAS_GHCNh_station_list(
            cache / 'ghcnh-station-list.csv',
            area=area,
            force_download=False,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

        if meta_candidates.empty:
            raise RuntimeError(
                f'No GHCNh station candidates found inside area={area}'
            )

        meta_candidates.to_csv(stations_candidates_csv, index=False)

        print(
            f'[SSOD] selected {len(meta_candidates)} GHCNh candidate station(s) '
            f'inside area.'
        )
        print(
            '[SSOD] note: GHCNh candidates are not guaranteed to have an '
            'SSODv2 file for every year.'
        )

        base_year = (
            'https://www.ncei.noaa.gov/oa/'
            'synoptic-summary-of-the-day/v2/access/by-year'
        )
        base_station = (
            'https://www.ncei.noaa.gov/oa/'
            'synoptic-summary-of-the-day/v2/access/by-station'
        )

        frames = []
        flag_frames = []
        successful_stations = set()

        # Small status registry for a useful final error message.
        status_rows = []

        def _read_csv(path):
            try:
                df = pd.read_csv(path, dtype=str, low_memory=False)
            except Exception as exc:
                print(f'[SSOD] cannot parse {Path(path).name}: {exc}')
                return None
            if df.empty:
                return None
            return df

        def _filter_one_year(df, station, year):
            """
            Keep only records in ``year`` and make the station identifier
            explicit if NCEI omitted the STATION column.
            """
            date_col = self._WAS_find_column(
                df.columns,
                'DATE',
                'date',
            )
            if date_col is None:
                return None

            dt = pd.to_datetime(df[date_col], errors='coerce')
            keep = dt.dt.year.eq(int(year))

            out = df.loc[keep].copy()
            if out.empty:
                return None

            station_col = self._WAS_find_column(
                out.columns,
                'STATION',
                'Station_ID',
                'station',
            )
            if station_col is None:
                out['STATION'] = str(station)
            else:
                # Keep the file's GHCN identifier when present, but fill any
                # missing values deterministically.
                out[station_col] = (
                    out[station_col]
                    .fillna(str(station))
                    .astype(str)
                    .str.strip()
                )

            return out

        def _collect_flags(df):
            if not include_flags or df is None or df.empty:
                return

            wanted = []
            for c in df.columns:
                cl = str(c).lower()
                if (
                    'measurement_code' in cl
                    or cl.endswith('_attributes')
                    or 'quality' in cl
                    or cl.startswith('qc_')
                    or cl.endswith('_qc')
                ):
                    wanted.append(c)

            date_col = self._WAS_find_column(
                df.columns,
                'DATE',
                'date',
            )
            station_col = self._WAS_find_column(
                df.columns,
                'STATION',
                'Station_ID',
                'station',
            )

            base_cols = [
                c for c in (station_col, date_col)
                if c is not None
            ]

            selected = base_cols + [
                c for c in wanted
                if c not in base_cols
            ]

            if selected and date_col is not None:
                flag_frames.append(df[selected].copy())

        for year in range(year_start, year_end + 1):
            year_cache = by_year_cache / str(year)
            year_cache.mkdir(parents=True, exist_ok=True)

            got = 0
            hit_by_year = 0
            hit_by_station = 0

            for station in meta_candidates['station'].astype(str):
                station = station.strip()
                if not station:
                    continue

                target = year_cache / f'ssod_{station}_{year}.csv'

                # ----------------------------------------------------------
                # 1) Reuse extracted year cache when available.
                # ----------------------------------------------------------
                df_year = None
                source_kind = None

                if (
                    target.exists()
                    and target.stat().st_size > 0
                    and not force_download
                ):
                    cached = _read_csv(target)
                    if cached is not None:
                        df_year = _filter_one_year(
                            cached,
                            station,
                            year,
                        )
                        if df_year is not None:
                            source_kind = 'year-cache'

                # ----------------------------------------------------------
                # 2) Official station/year archive.
                # ----------------------------------------------------------
                if df_year is None:
                    year_urls = [
                        f'{base_year}/{year}/ssod_{station}_{year}.csv',
                        f'{base_year}/{year}/SSOD_{station}_{year}.csv',
                    ]

                    ok = False
                    for url in year_urls:
                        ok = self._WAS_HTTP_download_cached(
                            url,
                            target,
                            force_download=force_download,
                            max_retries=max_retries,
                            retry_delay=retry_delay,
                            quiet_404=True,
                        )
                        if ok:
                            break

                    if ok:
                        downloaded = _read_csv(target)
                        if downloaded is not None:
                            df_year = _filter_one_year(
                                downloaded,
                                station,
                                year,
                            )
                            if df_year is not None:
                                source_kind = 'by-year'
                                hit_by_year += 1

                # ----------------------------------------------------------
                # 3) Official period-of-record station fallback.
                # ----------------------------------------------------------
                if df_year is None:
                    por_target = by_station_cache / f'SSOD_{station}.csv'

                    por_urls = [
                        f'{base_station}/SSOD_{station}.csv',
                        f'{base_station}/ssod_{station}.csv',
                    ]

                    por_ok = (
                        por_target.exists()
                        and por_target.stat().st_size > 0
                        and not force_download
                    )

                    if not por_ok:
                        for url in por_urls:
                            por_ok = self._WAS_HTTP_download_cached(
                                url,
                                por_target,
                                force_download=force_download,
                                max_retries=max_retries,
                                retry_delay=retry_delay,
                                quiet_404=True,
                            )
                            if por_ok:
                                break

                    if por_ok:
                        por = _read_csv(por_target)
                        if por is not None:
                            df_year = _filter_one_year(
                                por,
                                station,
                                year,
                            )

                        if df_year is not None:
                            source_kind = 'by-station'
                            hit_by_station += 1

                            # Materialize a small year-specific cache so the
                            # next run does not have to re-read a potentially
                            # long period-of-record station CSV.
                            try:
                                df_year.to_csv(target, index=False)
                            except Exception as exc:
                                print(
                                    f'[SSOD] warning: could not write '
                                    f'{target.name}: {exc}'
                                )

                if df_year is None:
                    status_rows.append({
                        'station': station,
                        'year': int(year),
                        'status': 'no_ssod_file_or_no_records',
                    })
                    continue

                frames.append(df_year)
                successful_stations.add(station)
                got += 1

                status_rows.append({
                    'station': station,
                    'year': int(year),
                    'status': source_kind or 'available',
                })

                _collect_flags(df_year)

            print(
                f'[SSOD] {year}: {got}/{len(meta_candidates)} candidate '
                f'station files/records available '
                f'(by-year={hit_by_year}, by-station fallback={hit_by_station}).'
            )

        # ------------------------------------------------------------------
        # Fail with diagnostics rather than a generic "0/58" message.
        # ------------------------------------------------------------------
        status_csv = (
            out_dir / f'SSOD_Access_Status_{year_start}_{year_end}.csv'
        )
        if status_rows:
            pd.DataFrame(status_rows).to_csv(status_csv, index=False)

        if not frames:
            raise RuntimeError(
                'No SSODv2 records could be obtained for the requested '
                f'area/period. {len(meta_candidates)} GHCNh candidate station(s) '
                'were tested using both official NCEI access modes: '
                'station/year files and period-of-record by-station files. '
                f'Access diagnostics were saved to {status_csv}. '
                'A GHCNh station being inside the bounding box does not imply '
                'that SSODv2 has a daily-summary record for that station/year.'
            )

        # ------------------------------------------------------------------
        # Restrict metadata to stations that actually supplied SSOD data.
        # ------------------------------------------------------------------
        meta = meta_candidates.loc[
            meta_candidates['station'].astype(str).isin(
                sorted(successful_stations)
            )
        ].copy()

        meta = meta.sort_values('station').reset_index(drop=True)
        meta.to_csv(stations_csv, index=False)

        print(
            f'[SSOD] actual SSODv2 stations retained: '
            f'{len(meta)}/{len(meta_candidates)}.'
        )

        long = pd.concat(
            frames,
            ignore_index=True,
            sort=False,
        )

        # De-duplicate deterministic station/day records if a file was seen
        # through both a cache and a fallback across interrupted runs.
        station_col = self._WAS_find_column(
            long.columns,
            'STATION',
            'Station_ID',
            'station',
        )
        date_col = self._WAS_find_column(
            long.columns,
            'DATE',
            'date',
        )
        if station_col is not None and date_col is not None:
            long['_SSOD_DATE_SORT'] = pd.to_datetime(
                long[date_col],
                errors='coerce',
            )
            long = (
                long.sort_values(
                    [station_col, '_SSOD_DATE_SORT']
                )
                .drop_duplicates(
                    [station_col, '_SSOD_DATE_SORT'],
                    keep='last',
                )
                .drop(columns=['_SSOD_DATE_SORT'])
            )

        _now = pd.Timestamp.utcnow()
        if _now.tzinfo is not None:
            _now = _now.tz_convert(None)

        effective_end = min(
            pd.Timestamp(
                year_end,
                12,
                31,
                23,
                59,
                59,
            ),
            _now,
        )

        ds = self._WAS_station_dataset_from_long(
            long,
            meta,
            variables,
            native_map,
            start=f'{year_start}-01-01',
            end=effective_end,
            frequency='daily',
            full_calendar=True,
            attrs={
                'source': (
                    'NOAA NCEI Synoptic Summary of the Day Version 2'
                ),
                'access': (
                    'official NCEI direct WAF: by-year station CSV with '
                    'by-station period-of-record fallback'
                ),
                'summary_day': '00-23 UTC',
                'station_metadata_csv': str(stations_csv),
                'candidate_station_metadata_csv': str(
                    stations_candidates_csv
                ),
                'access_status_csv': str(status_csv),
                'n_candidate_stations': int(len(meta_candidates)),
                'n_available_stations': int(len(meta)),
                'WASS2S_dimensions': 'T,station',
            },
        )

        units = {
            'PRCP': 'mm/day',
            'TEMP': 'degC',
            'TMAX': 'degC',
            'TMIN': 'degC',
            'TDEW': 'degC',
            'SLP': 'hPa',
            'SP': 'hPa',
            'WIND': 'm s-1',
            'WINDMAX': 'm s-1',
            'GUST': 'm s-1',
            'VISIB': 'km',
            'SNOW': 'cm',
        }

        for v in variables:
            ds[v].attrs['units'] = units[v]

        enc = {
            v: {
                'zlib': True,
                'complevel': 4,
                'dtype': 'float32',
                '_FillValue': -9999.0,
            }
            for v in variables
        }

        _was_write_netcdf(ds, final, encoding=enc)
        ds.close()

        if include_flags and flag_frames:
            (
                pd.concat(
                    flag_frames,
                    ignore_index=True,
                    sort=False,
                )
                .to_csv(
                    flags_csv,
                    index=False,
                    compression='gzip',
                )
            )
            print(f'[SSOD] flags saved: {flags_csv}')

        print(f'[SSOD] access diagnostics: {status_csv}')
        print(f'[SSOD] saved: {final}')
        return final


    def WAS_Download_GHCNh_Hourly(
        self,
        dir_to_save,
        year_start,
        year_end,
        area,
        variables=('PRCP', 'TEMP', 'TDEW', 'RH', 'SLP', 'SP', 'WIND', 'WDIR', 'GUST'),
        include_qc=True,
        include_provenance=True,
        force_download=False,
        max_retries=3,
        retry_delay=5,
    ):
        """
        Download NOAA Global Historical Climatology Network hourly (GHCNh).

        Uses official station/year PSV files from the NCEI direct archive.
        Timestamps are kept at their actual reported minute; no unsafe hourly
        resampling or precipitation summation is performed.
        """
        from pathlib import Path
        import pandas as pd
        import numpy as np

        variables = [variables] if isinstance(variables, str) else list(variables)
        native_map = {
            'PRCP': ('precipitation', 'Precipitation'),
            'TEMP': ('temperature', 'Temperature'),
            'TDEW': ('dew_point_temperature', 'Dew Point Temperature'),
            'RH': ('relative_humidity', 'Relative Humidity'),
            'SLP': ('sea_level_pressure', 'Sea Level Pressure'),
            'SP': ('station_level_pressure', 'Station Level Pressure'),
            'WIND': ('wind_speed', 'Wind Speed'),
            'WDIR': ('wind_direction', 'Wind Direction'),
            'GUST': ('wind_gust', 'Wind Gust'),
            'VISIB': ('visibility', 'Visibility'),
            'ALTIMETER': ('altimeter', 'Altimeter'),
            'SNOW': ('snow_depth', 'Snow Depth'),
        }
        unknown = [v for v in variables if v not in native_map]
        if unknown:
            raise ValueError(f'Unsupported GHCNh variable(s): {unknown}. Supported: {sorted(native_map)}')

        out_dir = Path(dir_to_save)
        out_dir.mkdir(parents=True, exist_ok=True)
        cache = out_dir / '.ghcnh_cache'
        cache.mkdir(parents=True, exist_ok=True)
        final = out_dir / f'Hourly_GHCNh_{year_start}_{year_end}.nc'
        stations_csv = out_dir / f'Stations_GHCNh_{year_start}_{year_end}.csv'
        provenance_csv = out_dir / f'GHCNh_QC_Provenance_{year_start}_{year_end}.csv.gz'
        if final.exists() and not force_download:
            print(f'{final} already exists. Skipping.')
            return final

        meta = self._WAS_GHCNh_station_list(
            cache / 'ghcnh-station-list.csv', area=area,
            force_download=False, max_retries=max_retries, retry_delay=retry_delay,
        )
        if meta.empty:
            raise RuntimeError(f'No GHCNh stations found inside area={area}')
        meta.to_csv(stations_csv, index=False)
        print(f'[GHCNh] selected {len(meta)} stations inside area.')

        base = 'https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/access/by-year'
        frames = []
        provenance = []
        for year in range(int(year_start), int(year_end) + 1):
            year_cache = cache / 'by_year' / str(year)
            year_cache.mkdir(parents=True, exist_ok=True)
            got = 0
            for station in meta['station'].astype(str):
                target = year_cache / f'GHCNh_{station}_{year}.psv'
                url = f'{base}/{year}/psv/GHCNh_{station}_{year}.psv'
                ok = self._WAS_HTTP_download_cached(
                    url, target, force_download=force_download,
                    max_retries=max_retries, retry_delay=retry_delay,
                    quiet_404=True,
                )
                if not ok:
                    continue
                try:
                    df = pd.read_csv(target, sep='|', dtype=str, low_memory=False)
                except Exception as exc:
                    print(f'[GHCNh] cannot parse {target.name}: {exc}')
                    continue
                if df.empty:
                    continue

                # GHCNh PSV uses separate Y/M/D/H/M columns. Build DATE explicitly
                # when a ready timestamp column is absent.
                date_col = self._WAS_find_column(df.columns, 'DATE', 'datetime', 'timestamp')
                if date_col is None:
                    ycol = self._WAS_find_column(df.columns, 'Year')
                    mcol = self._WAS_find_column(df.columns, 'Month')
                    dcol = self._WAS_find_column(df.columns, 'Day')
                    hcol = self._WAS_find_column(df.columns, 'Hour')
                    mincol = self._WAS_find_column(df.columns, 'Minute')
                    if all(c is not None for c in (ycol, mcol, dcol)):
                        stamp = pd.DataFrame({
                            'year': pd.to_numeric(df[ycol], errors='coerce'),
                            'month': pd.to_numeric(df[mcol], errors='coerce'),
                            'day': pd.to_numeric(df[dcol], errors='coerce'),
                            'hour': pd.to_numeric(df[hcol], errors='coerce').fillna(0) if hcol else 0,
                            'minute': pd.to_numeric(df[mincol], errors='coerce').fillna(0) if mincol else 0,
                        })
                        df['DATE'] = pd.to_datetime(stamp, errors='coerce')
                    else:
                        print(f'[GHCNh] no timestamp columns in {target.name}; skipped.')
                        continue

                stcol = self._WAS_find_column(df.columns, 'STATION', 'Station_ID', 'GHCN_ID')
                if stcol is None:
                    df['STATION'] = station
                frames.append(df)
                got += 1

                if include_qc or include_provenance:
                    keep = []
                    for c in df.columns:
                        cl = str(c).lower()
                        if include_qc and ('quality_code' in cl or 'measurement_code' in cl):
                            keep.append(c)
                        if include_provenance and any(x in cl for x in ('report_type', 'source_code', 'source_station_id')):
                            keep.append(c)
                    date_col2 = self._WAS_find_column(df.columns, 'DATE')
                    stcol2 = self._WAS_find_column(df.columns, 'STATION', 'Station_ID', 'GHCN_ID')
                    if keep and date_col2 is not None:
                        cols = [c for c in (stcol2, date_col2) if c is not None] + sorted(set(keep))
                        provenance.append(df[cols].copy())
            print(f'[GHCNh] {year}: {got}/{len(meta)} station files available.')

        if not frames:
            raise RuntimeError('No GHCNh station files could be downloaded for the requested area/period.')
        long = pd.concat(frames, ignore_index=True, sort=False)
        _now = pd.Timestamp.utcnow()
        if _now.tzinfo is not None:
            _now = _now.tz_convert(None)
        effective_end = min(pd.Timestamp(int(year_end), 12, 31, 23, 59, 59), _now)
        ds = self._WAS_station_dataset_from_long(
            long, meta, variables, native_map,
            start=f'{year_start}-01-01', end=effective_end,
            frequency='hourly', full_calendar=False,
            attrs={
                'source': 'NOAA NCEI Global Historical Climatology Network hourly (GHCNh)',
                'access': 'direct NCEI WAF station/year PSV',
                'time_axis': 'actual report timestamps; not resampled',
                'precipitation_note': 'GHCNh precipitation may represent different accumulation periods; inspect Measurement_Code before temporal aggregation.',
                'station_metadata_csv': str(stations_csv),
                'WASS2S_dimensions': 'T,station',
            },
        )
        units = {
            'PRCP': 'mm', 'TEMP': 'degC', 'TDEW': 'degC', 'RH': '%',
            'SLP': 'hPa', 'SP': 'hPa', 'WIND': 'm s-1', 'WDIR': 'degree',
            'GUST': 'm s-1', 'VISIB': 'km', 'ALTIMETER': 'hPa', 'SNOW': 'cm',
        }
        for v in variables:
            ds[v].attrs['units'] = units[v]
        enc = {v: {'zlib': True, 'complevel': 4, 'dtype': 'float32', '_FillValue': -9999.0} for v in variables}
        _was_write_netcdf(ds, final, encoding=enc)
        ds.close()
        if provenance:
            pd.concat(provenance, ignore_index=True, sort=False).to_csv(
                provenance_csv, index=False, compression='gzip'
            )
            print(f'[GHCNh] QC/provenance saved: {provenance_csv}')
        print(f'[GHCNh] saved: {final}')
        return final


    def WAS_Download_ISD_Hourly(self, *args, **kwargs):
        """Compatibility alias: new WASS2S workflows use NOAA GHCNh."""
        print('[INFO] WAS_Download_ISD_Hourly uses the current GHCNh backend.')
        return self.WAS_Download_GHCNh_Hourly(*args, **kwargs)


    def WAS_Download_GSOD_Daily(
        self,
        dir_to_save,
        year_start,
        year_end,
        area,
        variables=('PRCP', 'TEMP', 'TMAX', 'TMIN', 'TDEW', 'SLP', 'SP', 'WIND', 'GUST'),
        force_download=False,
        max_retries=3,
        retry_delay=5,
    ):
        """
        Download legacy NOAA Global Summary of the Day (GSOD) from its native
        annual station-file archive. Direct GSOD files use legacy imperial units;
        this method converts them to SI/metric WASS2S variables.
        """
        from pathlib import Path
        import pandas as pd
        import numpy as np

        variables = [variables] if isinstance(variables, str) else list(variables)
        native_map = {
            'TEMP': ('TEMP',), 'TDEW': ('DEWP',), 'SLP': ('SLP',), 'SP': ('STP',),
            'VISIB': ('VISIB',), 'WIND': ('WDSP',), 'WINDMAX': ('MXSPD',),
            'GUST': ('GUST',), 'TMAX': ('MAX',), 'TMIN': ('MIN',), 'PRCP': ('PRCP',),
            'SNOW': ('SNDP',),
        }
        unknown = [v for v in variables if v not in native_map]
        if unknown:
            raise ValueError(f'Unsupported GSOD variable(s): {unknown}. Supported: {sorted(native_map)}')

        out_dir = Path(dir_to_save)
        out_dir.mkdir(parents=True, exist_ok=True)
        cache = out_dir / '.gsod_cache'
        cache.mkdir(parents=True, exist_ok=True)
        final = out_dir / f'Daily_GSOD_{year_start}_{year_end}.nc'
        stations_csv = out_dir / f'Stations_GSOD_{year_start}_{year_end}.csv'
        if final.exists() and not force_download:
            print(f'{final} already exists. Skipping.')
            return final

        history_file = cache / 'isd-history.csv'
        history_url = 'https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv'
        if not self._WAS_HTTP_download_cached(
            history_url, history_file, force_download=False,
            max_retries=max_retries, retry_delay=retry_delay,
        ):
            raise RuntimeError(f'Could not download station history: {history_url}')
        hist = pd.read_csv(history_file, dtype=str, low_memory=False)
        col = {str(c).strip().upper(): c for c in hist.columns}
        usaf = col.get('USAF'); wban = col.get('WBAN')
        latc = col.get('LAT'); lonc = col.get('LON'); elevc = col.get('ELEV(M)') or col.get('ELEV')
        namec = col.get('STATION NAME') or col.get('STATION_NAME')
        beginc = col.get('BEGIN'); endc = col.get('END')
        if None in (usaf, wban, latc, lonc):
            raise ValueError(f'Unexpected isd-history.csv columns: {list(hist.columns)}')

        hist['LAT'] = pd.to_numeric(hist[latc], errors='coerce')
        hist['LON'] = pd.to_numeric(hist[lonc], errors='coerce')
        north, west, south, east = [float(v) for v in area]
        lat_ok = hist['LAT'].between(south, north, inclusive='both')
        lon_ok = hist['LON'].between(west, east, inclusive='both') if west <= east else ((hist['LON'] >= west) | (hist['LON'] <= east))
        hist = hist.loc[lat_ok & lon_ok].copy()
        if beginc and endc:
            hist['_BEGIN'] = pd.to_datetime(hist[beginc], format='%Y%m%d', errors='coerce')
            hist['_END'] = pd.to_datetime(hist[endc], format='%Y%m%d', errors='coerce')
            hist = hist[(hist['_END'].isna() | (hist['_END'] >= pd.Timestamp(year_start,1,1))) &
                        (hist['_BEGIN'].isna() | (hist['_BEGIN'] <= pd.Timestamp(year_end,12,31)))]

        hist['USAF6'] = hist[usaf].fillna('').astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(6)
        hist['WBAN5'] = hist[wban].fillna('').astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(5)
        hist['station'] = hist['USAF6'] + hist['WBAN5']
        hist['ELEV'] = pd.to_numeric(hist[elevc], errors='coerce') if elevc else np.nan
        hist['NAME'] = hist[namec].fillna('').astype(str) if namec else ''
        meta = hist[['station','LAT','LON','ELEV','NAME']].drop_duplicates('station').sort_values('station')
        if meta.empty:
            raise RuntimeError(f'No GSOD/ISD stations found inside area={area}')
        meta.to_csv(stations_csv, index=False)
        print(f'[GSOD] selected {len(meta)} stations inside area.')

        base = 'https://www.ncei.noaa.gov/data/global-summary-of-the-day/access'
        frames = []
        for year in range(int(year_start), int(year_end) + 1):
            year_cache = cache / str(year)
            year_cache.mkdir(parents=True, exist_ok=True)
            got = 0
            for station in meta['station'].astype(str):
                target = year_cache / f'{station}.csv'
                url = f'{base}/{year}/{station}.csv'
                ok = self._WAS_HTTP_download_cached(
                    url, target, force_download=force_download,
                    max_retries=max_retries, retry_delay=retry_delay,
                    quiet_404=True,
                )
                if not ok:
                    continue
                try:
                    df = pd.read_csv(target, dtype=str, low_memory=False)
                except Exception as exc:
                    print(f'[GSOD] cannot parse {target.name}: {exc}')
                    continue
                if df.empty:
                    continue
                frames.append(df)
                got += 1
            print(f'[GSOD] {year}: {got}/{len(meta)} station files available.')

        if not frames:
            raise RuntimeError('No GSOD station files could be downloaded for the requested area/period.')
        long = pd.concat(frames, ignore_index=True, sort=False)

        # Clean/convert the legacy direct-file units BEFORE matrix construction.
        def numeric_column(name):
            c = self._WAS_find_column(long.columns, name)
            if c is None:
                return None
            vals = pd.to_numeric(
                long[c].astype(str).str.extract(r'([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)', expand=False),
                errors='coerce',
            )
            return c, vals

        # Missing sentinels and conversions: F -> C, inches -> mm, miles -> km, knots -> m/s.
        for name in ('TEMP','DEWP','MAX','MIN'):
            hit = numeric_column(name)
            if hit:
                c, v = hit; v = v.where(v < 9999.0); long[c] = (v - 32.0) * (5.0/9.0)
        for name in ('SLP','STP'):
            hit = numeric_column(name)
            if hit:
                c, v = hit; long[c] = v.where(v < 9999.0)
        hit = numeric_column('VISIB')
        if hit:
            c, v = hit; long[c] = v.where(v < 999.0) * 1.609344
        for name in ('WDSP','MXSPD','GUST'):
            hit = numeric_column(name)
            if hit:
                c, v = hit; long[c] = v.where(v < 999.0) * 0.514444444
        hit = numeric_column('PRCP')
        if hit:
            c, v = hit; long[c] = v.where(v < 99.0) * 25.4
        hit = numeric_column('SNDP')
        if hit:
            c, v = hit; long[c] = v.where(v < 999.0) * 25.4

        ds = self._WAS_station_dataset_from_long(
            long, meta, variables, native_map,
            start=f'{year_start}-01-01', end=f'{year_end}-12-31 23:59:59',
            frequency='daily', full_calendar=True,
            attrs={
                'source': 'NOAA NCEI Global Summary of the Day (GSOD) legacy',
                'access': 'direct NCEI GSOD annual station CSV archive',
                'units_conversion': 'legacy direct-file imperial units converted to SI/metric',
                'station_metadata_csv': str(stations_csv),
                'WASS2S_dimensions': 'T,station',
            },
        )
        units = {
            'PRCP':'mm/day','TEMP':'degC','TMAX':'degC','TMIN':'degC','TDEW':'degC',
            'SLP':'hPa','SP':'hPa','WIND':'m s-1','WINDMAX':'m s-1','GUST':'m s-1',
            'VISIB':'km','SNOW':'mm',
        }
        for v in variables:
            ds[v].attrs['units'] = units[v]
        enc = {v: {'zlib': True, 'complevel': 4, 'dtype': 'float32', '_FillValue': -9999.0} for v in variables}
        _was_write_netcdf(ds, final, encoding=enc)
        ds.close()
        print(f'[GSOD] saved: {final}')
        return final


    def WAS_Download_CAMSOPI_Monthly(
        self,
        dir_to_save,
        year_start,
        year_end,
        area,
        force_download=False,
        max_retries=3,
        retry_delay=5,
        include_components=True,
        convert_to_monthly_total=True,
    ):
        """Download CPC CAMS-OPI monthly fields (2.5°) and save WASS2S T/Y/X."""
        from pathlib import Path
        import calendar
        import gzip
        import time as _time
        import numpy as np
        import pandas as pd
        import requests
        import xarray as xr

        out_dir = Path(dir_to_save)
        out_dir.mkdir(parents=True, exist_ok=True)
        cache = out_dir / ".camsopi_cache"
        cache.mkdir(parents=True, exist_ok=True)
        final = out_dir / f"Monthly_CAMSOPI_PRCP_{year_start}_{year_end}.nc"
        if final.exists() and not force_download:
            print(f"{final} already exists. Skipping.")
            return final

        base = "https://ftp.cpc.ncep.noaa.gov/precip/data-req/cams_opi_v0208"
        X = 1.25 + np.arange(144) * 2.5
        X = ((X + 180.0) % 360.0) - 180.0
        Y = -88.75 + np.arange(72) * 2.5
        north, west, south, east = [float(v) for v in area]
        months = pd.period_range(f"{year_start}-01", f"{year_end}-12", freq="M")
        records = []
        times = []
        missing = []

        for per in months:
            ym = per.strftime("%Y%m")
            local = cache / f"cams_opi_merged.{ym}"
            payload = None
            # Older files are gzip-compressed; recent files may be uncompressed.
            candidates = [f"{base}/cams_opi_merged.{ym}.gz", f"{base}/cams_opi_merged.{ym}"]
            if local.exists() and not force_download:
                cached = local.read_bytes()
                if len(cached) == 7 * 72 * 144 * 4:
                    payload = cached
            if payload is None:
                for url in candidates:
                    for attempt in range(1, max_retries+1):
                        try:
                            r = requests.get(url, timeout=90)
                            if r.status_code == 404:
                                break
                            r.raise_for_status()
                            payload = gzip.decompress(r.content) if url.endswith(".gz") else r.content
                            break
                        except Exception as exc:
                            if attempt < max_retries:
                                _time.sleep(retry_delay)
                            else:
                                last_exc = exc
                    if payload is not None:
                        break
            if payload is None:
                missing.append(ym)
                continue
            arr = np.frombuffer(payload, dtype=">f4").astype("float32", copy=True)
            expected = 7 * 72 * 144
            if arr.size != expected:
                raise ValueError(f"Unexpected CAMS-OPI file size for {ym}: {arr.size}, expected {expected}")
            part = local.with_name(local.name + ".part")
            part.write_bytes(payload)
            part.replace(local)
            arr = arr.reshape(7, 72, 144)
            arr[arr <= -998.0] = np.nan
            # Record order from camsopi.ctl: cams, camsn, opi, comb, xxxx, comba, gam.
            ds = xr.Dataset(
                {
                    "PRCP_GAUGE_RATE": (("Y", "X"), arr[0]),
                    "N_GAUGES": (("Y", "X"), arr[1]),
                    "PRCP_OPI_RATE": (("Y", "X"), arr[2]),
                    "PRCP_RATE": (("Y", "X"), arr[3]),
                    "PRCP_ANOM_OLD_RATE": (("Y", "X"), arr[4]),
                    "PRCP_ANOM_RATE": (("Y", "X"), arr[5]),
                    "GAMMA_PERCENT": (("Y", "X"), arr[6]),
                },
                coords={"Y": Y, "X": X},
            ).sortby("X").sortby("Y")
            ds = ds.sel(X=slice(west, east), Y=slice(south, north))
            days = calendar.monthrange(per.year, per.month)[1]
            if convert_to_monthly_total:
                ds["PRCP"] = (ds["PRCP_RATE"] * days).astype("float32")
                ds["PRCP"].attrs["units"] = "mm/month"
                if include_components:
                    ds["PRCP_GAUGE"] = (ds["PRCP_GAUGE_RATE"] * days).astype("float32")
                    ds["PRCP_OPI"] = (ds["PRCP_OPI_RATE"] * days).astype("float32")
                    ds["PRCP_ANOM"] = (ds["PRCP_ANOM_RATE"] * days).astype("float32")
                    ds["PRCP_GAUGE"].attrs["units"] = "mm/month"
                    ds["PRCP_OPI"].attrs["units"] = "mm/month"
                    ds["PRCP_ANOM"].attrs["units"] = "mm/month"
            else:
                ds = ds.rename({"PRCP_RATE": "PRCP"})
                ds["PRCP"].attrs["units"] = "mm/day"
                if include_components:
                    for source, target in (("PRCP_GAUGE_RATE", "PRCP_GAUGE"), ("PRCP_OPI_RATE", "PRCP_OPI"), ("PRCP_ANOM_RATE", "PRCP_ANOM")):
                        ds[target] = ds[source]
                        ds[target].attrs["units"] = "mm/day"
            keep = ["PRCP", "N_GAUGES", "GAMMA_PERCENT"]
            if include_components:
                keep += ["PRCP_GAUGE", "PRCP_OPI", "PRCP_ANOM"]
            ds = ds[[v for v in keep if v in ds.data_vars]]
            ds = ds.expand_dims(T=[pd.Timestamp(per.year, per.month, 1)])
            records.append(ds)
            times.append(per)

        if not records:
            raise RuntimeError("No CAMS-OPI monthly data could be downloaded.")
        out = xr.concat(records, dim="T", join="exact").sortby("T")
        requested_times = months.to_timestamp()
        available_times = out["T"].values
        out = out.reindex(T=requested_times)
        out["SOURCE_AVAILABLE"] = ("T", np.isin(requested_times.values, available_times).astype("int8"))
        out.attrs.update({
            "source": "NOAA CPC CAMS-OPI",
            "native_resolution": "2.5 degree",
            "native_precipitation_units": "mm/day",
            "missing_source_months": ",".join(missing),
            "WASS2S_dimensions": "T,Y,X",
        })
        _was_write_netcdf(out, final, encoding={v: {'zlib': True, 'complevel': 4} for v in out.data_vars})
        out.close()
        print(f"Saved CAMS-OPI monthly: {final}")
        return final


#####

def plot_map(extent, title="Map"): # [west, east, south, north]
    """
    Plots a map with specified geographic extent.

    Parameters:
    - extent: list of float, specifying [west, east, south, north]
    - title: str, title of the map
    """
    # Create figure and axis for the map
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(3, 2))

    # Set the geographic extent
    ax.set_extent(extent) 
    
    # Add map features
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAND, edgecolor="black")
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    
    # Set title
    ax.set_title(title)
    
    # Show plot
    plt.tight_layout()
    plt.show()


def _noaa_exact_month_indices(times, required_months):
    """Find exact calendar months in monthly NOAA SST; never reuse a neighbour."""
    months = np.asarray(times).astype("datetime64[M]")
    indices = []
    for year, month in required_months:
        match = np.flatnonzero(months == np.datetime64(f"{year:04d}-{month:02d}", "M"))
        if match.size == 0:
            return None
        if match.size != 1:
            raise ValueError(f"NOAA SST: duplicate monthly records for {year}-{month:02d}")
        indices.append(int(match[0]))
    return indices
