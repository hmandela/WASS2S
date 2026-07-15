"""Seasonal diagnostic tools for ERA5 and optional AgERA5 precipitation data.

This module provides:

Monthly pipeline (six months preceding a target date):

- :data:`VAR_CONFIG` — per-variable download and plotting configuration.
- :func:`download_data` — ERA5 monthly download, with optional AgERA5 precipitation.
- :func:`process_variable` — compute anomalies/ratios and dispatch to plots.
- :func:`plot_maps` — 6-panel monthly map layout.
- :func:`plot_hovmoller` — time–longitude Hovmöller diagram.
- :func:`main_driver` — end-to-end monthly entry point.

Daily pipeline (last N days up to a target date, aggregated):

- :func:`download_daily_data` — ERA5 daily statistics or optional AgERA5 precipitation.
- :func:`process_daily` — aggregate (mean/sum) and plot one period map.
- :func:`plot_period_map` — single-panel aggregate map.
- :func:`main_daily_driver` — end-to-end daily entry point.

Interactive viewers:

- :class:`C3SViewer` — Copernicus C3S seasonal forecast viewer.
- :class:`BOMViewer` — BOM MJO viewer.
"""

import os
import warnings
import datetime
from datetime import date, timedelta

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from dateutil.relativedelta import relativedelta
import earthkit.data

import ipywidgets as widgets
from IPython.display import display, IFrame

warnings.filterwarnings("ignore")


def _monthly_precip_to_mm(data):
    """Convert ERA5 monthly-mean precipitation from m/day to mm/month."""
    if "time" not in data.coords:
        raise ValueError("Monthly precipitation conversion requires a time coordinate.")
    return data * data["time"].dt.days_in_month * 1000.0


def _save_earthkit_result(result, path):
    """Write an earthkit retrieval using the current target API."""
    if hasattr(result, "to_target"):
        result.to_target("file", path)
    else:  # Compatibility with older earthkit-data releases.
        result.save(path)


def _first_data_array(data):
    """Return a DataArray from a DataArray or a single-variable Dataset."""
    if isinstance(data, xr.DataArray):
        return data
    if not data.data_vars:
        raise ValueError("The dataset contains no data variables.")
    return data[next(iter(data.data_vars))]


PRECIP_SOURCES = {"era5", "agera5"}
AGERA5_DATASET = "sis-agrometeorological-indicators"


def _normalise_precip_source(precip_source):
    """Validate and normalise the precipitation data-source selector."""
    source = str(precip_source).strip().lower()
    if source not in PRECIP_SOURCES:
        allowed = ", ".join(sorted(PRECIP_SOURCES))
        raise ValueError(f"precip_source must be one of: {allowed}.")
    return source


def _earthkit_to_xarray(result):
    """Convert an earthkit retrieval to one combined xarray Dataset."""
    if not hasattr(result, "to_xarray"):
        raise TypeError("The earthkit result does not provide to_xarray().")

    converted = result.to_xarray()
    if isinstance(converted, xr.DataArray):
        return converted.to_dataset(name=converted.name or "variable")
    if isinstance(converted, xr.Dataset):
        return converted

    if isinstance(converted, dict):
        converted = list(converted.values())
    if isinstance(converted, (list, tuple)):
        datasets = []
        for item in converted:
            if isinstance(item, xr.DataArray):
                item = item.to_dataset(name=item.name or "variable")
            if not isinstance(item, xr.Dataset):
                raise TypeError(f"Unsupported xarray result element: {type(item)!r}")
            datasets.append(item)
        if not datasets:
            raise ValueError("The AgERA5 retrieval returned no xarray datasets.")
        try:
            return xr.combine_by_coords(datasets, combine_attrs="override")
        except Exception:
            return xr.concat(datasets, dim="time", combine_attrs="override")

    raise TypeError(f"Unsupported earthkit to_xarray() result: {type(converted)!r}")


def _agera5_precipitation_array(data):
    """Extract AgERA5 precipitation and standardise it to mm/day."""
    ds = _clean_dataset(data)
    candidates = [
        name for name in ds.data_vars
        if "precipitation_flux" in name.lower()
        or name.lower() in {"precipitation", "precip", "tp"}
    ]
    da = ds[candidates[0]] if candidates else _first_data_array(ds)
    # Preserve time and horizontal coordinates even for a one-day or one-point
    # request; only remove unrelated singleton dimensions.
    for dim in tuple(da.dims):
        if dim not in {"time", "latitude", "longitude"} and da.sizes[dim] == 1:
            da = da.squeeze(dim, drop=True)

    units = str(da.attrs.get("units", "")).lower().replace(" ", "")
    # The official AgERA5 unit is mm/day. Retain compatibility with any file
    # encoded in m/day by converting only when the metadata is unambiguous.
    if units and "mm" not in units and (units.startswith("m/") or "mday" in units):
        da = da * 1000.0

    da = da.rename("precipitation")
    da.attrs.update({
        "units": "mm day-1",
        "long_name": "AgERA5 daily precipitation",
        "source": "AgERA5",
    })
    return da


def _retrieve_agera5_precipitation(request, start=None, end=None):
    """Retrieve AgERA5 precipitation and optionally crop an exact date range."""
    result = earthkit.data.from_source("cds", AGERA5_DATASET, request)
    ds = _earthkit_to_xarray(result)
    da = _agera5_precipitation_array(ds)
    if start is not None or end is not None:
        da = da.sel(time=slice(None if start is None else str(start),
                               None if end is None else str(end)))
    if da.sizes.get("time", 0) == 0:
        raise ValueError("The AgERA5 request returned no data in the requested period.")
    return da.sortby("time")


# ---------------------------------------------------------------------------
# Variable configuration
# ---------------------------------------------------------------------------

VAR_CONFIG = {
    "sst": {
        "cds_name": "sea_surface_temperature",
        "level_type": "single",
        "unit_func": lambda x: x - 273.15,
        "unit_label": "°C",
        "plot_type": "anomaly_only",
        "cmap": "RdBu_r",
        "levels": np.arange(-3.0, 3.5, 0.5),
        "extent": [40, -180, -40, 180],
        "compute_seasonal": True,
    },
    "slp": {
        "cds_name": "mean_sea_level_pressure",
        "level_type": "single",
        "unit_func": lambda x: x / 100,
        "unit_label": "hPa",
        "plot_type": "contour_overlay",
        "cmap": "RdBu_r",
        "levels_anom": np.arange(-5, 6, 1),
        "levels_contour": np.arange(990, 1038, 2),
        "highlight_black": [1015],
        "highlight_green": [1012],
        "extent": [50, -60, -45, 130],
    },
    "olr": {
        "cds_name": "top_net_thermal_radiation",
        "level_type": "single",
        "unit_func": lambda x: x / 86400,
        "unit_label": "W/m²",
        "plot_type": "contour_overlay",
        "cmap": "BrBG",
        "levels_anom": np.arange(-50, 55, 5),
        "levels_contour": np.arange(-320, -160, 20),
        "extent": [40, -180, -40, 180],
        "hov_band": [10, -10],
        "hov_cmap": "BrBG",
        "hov_levels": np.arange(-60, 65, 5),
    },
    "precip": {
        "cds_name": "total_precipitation",
        "level_type": "single",
        "unit_func": _monthly_precip_to_mm,
        "unit_label": "mm/mois",
        "plot_type": "ratio",
        "cmap": "BrBG",
        "levels": [0, 25, 50, 75, 90, 110, 125, 150, 200, 300],
        "extent": [35, -25, -15, 55],
    },
    "wind_850": {
        "cds_name": ["u_component_of_wind", "v_component_of_wind", "specific_humidity"],
        "level_type": "pressure",
        "pressure_level": "850",
        "unit_func": lambda x: x,
        "unit_label": "Humidité Spécifique (g/kg)",
        "plot_type": "stream_humidity",
        "cmap": "GnBu",
        "levels": np.arange(0, 18, 2),
        "extent": [40, -60, -20, 60],
    },
    "wind_surface": {
        "cds_name": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
        "level_type": "single",
        "unit_func": lambda x: x,
        "unit_label": "Vitesse du Vent (m/s)",
        "plot_type": "streamlines",
        "cmap": "YlOrRd",
        "levels": np.arange(0, 15, 1),
        "extent": [35, -30, -15, 50],
    },
}
"""dict: Per-variable ERA5 download and plotting configuration.

Each key is a short variable identifier.  The associated dict contains:

- ``cds_name`` — CDS variable name(s).
- ``level_type`` — ``"single"`` or ``"pressure"``.
- ``unit_func`` — callable that converts raw ERA5 *monthly* units.
- ``unit_label`` — axis / colorbar label.
- ``plot_type`` — rendering strategy used by :func:`plot_maps`.
- ``cmap``, ``levels``, ``extent`` — Matplotlib / Cartopy parameters.
- ``hov_band``, ``hov_cmap``, ``hov_levels`` — Hovmöller parameters (OLR only).
"""


# Per-variable daily conversion.  The CDS daily-statistics service returns
# precipitation with ``daily_sum`` and the other fields with ``daily_mean``.
# Accumulated precipitation is therefore converted directly from m/day to
# mm/day, while accumulated hourly radiation averaged over a day is divided by
# 3600 s to obtain W/m².
DAILY_UNIT = {
    "sst":          (lambda x: x - 273.15, "°C"),
    "slp":          (lambda x: x / 100.0, "hPa"),
    "olr":          (lambda x: x / 3600.0, "W/m²"),
    "precip":       (lambda x: x * 1000.0, "mm/jour"),
    "wind_850":     (lambda x: x, "Humidité Spécifique (g/kg)"),
    "wind_surface": (lambda x: x, "Vitesse du Vent (m/s)"),
}
"""dict: Per-variable ``(unit_func, unit_label)`` for the daily pipeline."""

DAILY_STATISTIC = {
    "sst": "daily_mean",
    "slp": "daily_mean",
    "olr": "daily_mean",
    "precip": "daily_sum",
    "wind_850": "daily_mean",
    "wind_surface": "daily_mean",
}
"""dict: CDS daily statistic requested for each configured variable."""


# ---------------------------------------------------------------------------
# Shared dataset cleaning
# ---------------------------------------------------------------------------

def _clean_dataset(ds):
    """Normalise ERA5 coordinate names and remove singleton metadata dimensions."""
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise TypeError("ds must be an xarray Dataset or DataArray.")

    rename = {}
    if "time" not in ds.coords and "time" not in ds.dims:
        for candidate in ("valid_time", "date"):
            if candidate in ds.coords or candidate in ds.dims:
                rename[candidate] = "time"
                break
    if "latitude" not in ds.coords and "latitude" not in ds.dims:
        if "lat" in ds.coords or "lat" in ds.dims:
            rename["lat"] = "latitude"
    if "longitude" not in ds.coords and "longitude" not in ds.dims:
        if "lon" in ds.coords or "lon" in ds.dims:
            rename["lon"] = "longitude"
    if rename:
        ds = ds.rename(rename)

    if "expver" in ds.dims:
        if ds.sizes["expver"] > 1:
            values = set(np.asarray(ds["expver"].values).tolist())
            if {1, 5}.issubset(values):
                ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))
            else:
                ds = ds.isel(expver=0, drop=True)
        else:
            ds = ds.squeeze("expver", drop=True)

    for level_name in ("pressure_level", "level"):
        if level_name in ds.dims and ds.sizes[level_name] == 1:
            ds = ds.squeeze(level_name, drop=True)

    if "time" not in ds.coords and "time" not in ds.dims:
        raise ValueError("No time coordinate was found in the ERA5 dataset.")

    ds = ds.sortby("time")
    if hasattr(ds, "drop_duplicates"):
        ds = ds.drop_duplicates("time")
    return ds


# ===========================================================================
# MONTHLY PIPELINE
# ===========================================================================

def _download_agera5_monthly_precipitation(
    dir_to_save, clim_start, clim_end, extent, target_date, agera5_version="2_0"
):
    """Download daily AgERA5 precipitation and aggregate it to monthly totals.

    AgERA5 has no native monthly product. Downloads are therefore split by
    calendar year, cached as daily mm/day fields, and then summed to mm/month.
    """
    if clim_start < 1979:
        raise ValueError("AgERA5 is available from 1979; clim_start cannot be earlier.")

    target_month = date(target_date.year, target_date.month, 1)
    requested_months = [target_month - relativedelta(months=i) for i in range(1, 7)]
    month_numbers = sorted({d.strftime("%m") for d in requested_months})
    years = sorted(
        set(range(clim_start, clim_end + 1)) | {d.year for d in requested_months}
    )
    if min(years) < 1979:
        raise ValueError("The requested AgERA5 period begins before 1979.")

    ext_str = f"{extent[0]}_{extent[1]}_{extent[2]}_{extent[3]}"
    months_tag = "".join(month_numbers)
    fname = (
        f"agera5_precip_monthly_v{agera5_version}_{clim_start}-{clim_end}_"
        f"target{target_month:%Y%m}_past6_{ext_str}.nc"
    )
    fpath = os.path.join(dir_to_save, fname)
    if os.path.exists(fpath):
        print(f" Data found: {fname}")
        return fpath

    cache_dir = os.path.join(dir_to_save, "agera5_daily_cache")
    os.makedirs(cache_dir, exist_ok=True)
    daily_paths = []
    all_days = [f"{day:02d}" for day in range(1, 32)]

    for year in years:
        cache_name = (
            f"agera5_precip_daily_v{agera5_version}_{year}_m{months_tag}_{ext_str}.nc"
        )
        cache_path = os.path.join(cache_dir, cache_name)
        if not os.path.exists(cache_path):
            request = {
                "variable": "precipitation_flux",
                "year": [str(year)],
                "month": month_numbers,
                "day": all_days,
                "version": agera5_version,
                "area": extent,
            }
            print(f"  AgERA5 precipitation {year}...", end=" ", flush=True)
            try:
                da = _retrieve_agera5_precipitation(request)
                da.to_dataset().to_netcdf(cache_path)
                print("ok")
            except Exception as exc:
                print(f"failed: {exc}")
                continue
        daily_paths.append(cache_path)

    if not daily_paths:
        print(" No AgERA5 precipitation files are available.")
        return None

    parts = []
    for path in daily_paths:
        with xr.open_dataset(path) as opened:
            parts.append(_first_data_array(_clean_dataset(opened)).load())
    daily = xr.concat(parts, dim="time").sortby("time").drop_duplicates("time")
    monthly = daily.resample(time="MS").sum("time", skipna=True, min_count=1)
    monthly = monthly.rename("precipitation")
    monthly.attrs.update({
        "units": "mm month-1",
        "long_name": "Monthly total precipitation derived from AgERA5",
        "source": f"AgERA5 version {agera5_version}",
    })
    output = monthly.to_dataset()
    output.attrs.update({
        "precip_source": "agera5",
        "agera5_version": agera5_version,
        "aggregation": "sum of daily precipitation flux",
    })
    os.makedirs(dir_to_save, exist_ok=True)
    output.to_netcdf(fpath)
    print(" AgERA5 monthly precipitation aggregation complete.")
    return fpath


def download_data(
    dir_to_save, clim_start, clim_end, var_key, extent, target_date,
    precip_source="era5", agera5_version="2_0",
):
    """Download monthly data for the six complete months before target month.

    For ``var_key="precip"``, ``precip_source="agera5"`` downloads daily
    AgERA5 precipitation and aggregates it to monthly totals. All other
    variables continue to use ERA5.
    """
    if var_key not in VAR_CONFIG:
        raise KeyError(f"Unknown variable key: {var_key!r}")
    if clim_start > clim_end:
        raise ValueError("clim_start must be less than or equal to clim_end.")

    source = _normalise_precip_source(precip_source)
    if var_key == "precip" and source == "agera5":
        return _download_agera5_monthly_precipitation(
            dir_to_save, clim_start, clim_end, extent, target_date,
            agera5_version=agera5_version,
        )

    conf = VAR_CONFIG[var_key]
    target_month = date(target_date.year, target_date.month, 1)
    requested_months = [target_month - relativedelta(months=i) for i in range(1, 7)]
    months_idx = sorted({d.strftime("%m") for d in requested_months})
    years_range = sorted(
        {str(y) for y in range(clim_start, clim_end + 1)}
        | {str(d.year) for d in requested_months}
    )

    ext_str = f"{extent[0]}_{extent[1]}_{extent[2]}_{extent[3]}"
    fname = (
        f"era5_{var_key}_{clim_start}-{clim_end}_"
        f"target{target_month:%Y%m}_past6_{ext_str}.nc"
    )
    fpath = os.path.join(dir_to_save, fname)

    if os.path.exists(fpath):
        print(f" Data found: {fname}")
        return fpath

    print(f" Downloading {var_key} from ERA5...")
    request = {
        "product_type": "monthly_averaged_reanalysis",
        "variable": conf["cds_name"],
        "year": years_range,
        "month": months_idx,
        "time": "00:00",
        "area": extent,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    dataset_name = "reanalysis-era5-single-levels-monthly-means"
    if conf["level_type"] == "pressure":
        dataset_name = "reanalysis-era5-pressure-levels-monthly-means"
        request["pressure_level"] = conf.get("pressure_level", "850")

    try:
        os.makedirs(dir_to_save, exist_ok=True)
        result = earthkit.data.from_source("cds", dataset_name, request)
        _save_earthkit_result(result, fpath)
        print(" Download complete.")
        return fpath
    except Exception as exc:
        print(f" Error downloading {var_key}: {exc}")
        return None


def _latitude_band(data, south, north):
    """Select a latitude band regardless of coordinate orientation."""
    lat = data["latitude"]
    lat_slice = slice(north, south) if lat.values[0] > lat.values[-1] else slice(south, north)
    return data.sel(latitude=lat_slice)


def plot_hovmoller(data_anom, data_abs, var_key):
    """Plot a time–longitude Hovmöller diagram for a supported variable.

    The diagram is only produced for variables that define a ``hov_band`` in
    :data:`VAR_CONFIG` (currently OLR).  For vector fields the U-component
    is used.

    Parameters
    ----------
    data_anom : xarray.Dataset or xarray.DataArray
        Anomaly field, already averaged over the configured latitude band.
    data_abs : xarray.Dataset or xarray.DataArray
        Absolute (non-anomaly) field for overlaid contours.
    var_key : str
        Variable key defined in :data:`VAR_CONFIG`.
    """
    conf = VAR_CONFIG[var_key]
    if "hov_band" not in conf:
        return

    print(f"Generating Hovmöller for {var_key}...")
    lat_max, lat_min = max(conf["hov_band"]), min(conf["hov_band"])

    hov_anom = _latitude_band(data_anom, lat_min, lat_max).mean(dim="latitude")
    hov_abs = _latitude_band(data_abs, lat_min, lat_max).mean(dim="latitude")

    if conf["plot_type"] in ["vector", "streamlines", "stream_humidity"]:
        u_var = [v for v in hov_anom.data_vars if "u" in v.lower() or "var131" in v][0]
        to_plot_anom = hov_anom[u_var]
        to_plot_abs = hov_abs[u_var]
        title_suffix = f"U-Component ({lat_min}° to {lat_max}°)"
    else:
        if isinstance(hov_anom, xr.Dataset):
            var_name = list(hov_anom.data_vars)[0]
            to_plot_anom = hov_anom[var_name]
            to_plot_abs = hov_abs[var_name]
        else:
            to_plot_anom = hov_anom
            to_plot_abs = hov_abs
        title_suffix = f"({lat_min}° to {lat_max}°)"

    fig, ax = plt.subplots(figsize=(14, 10))
    times = to_plot_anom.time.values
    lons = to_plot_anom.longitude.values
    vals_anom = to_plot_anom.values.squeeze()
    vals_abs = to_plot_abs.values.squeeze()

    cf = ax.contourf(lons, times, vals_anom, levels=conf["hov_levels"],
                     cmap=conf["hov_cmap"], extend="both")

    if var_key == "olr":
        ax.contour(lons, times, vals_abs, levels=[-240, -220, -200],
                   colors="k", linewidths=0.8, linestyles="--")
    elif "wind" in var_key:
        ax.contour(lons, times, vals_abs, levels=[0], colors="k", linewidths=1.5)

    ax.set_title(f"Hovmöller: {var_key.upper()} - {title_suffix}",
                 fontsize=16, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Time", fontsize=12)
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.colorbar(cf, ax=ax, label=f"Anomaly ({conf['unit_label']})",
                 orientation="horizontal", pad=0.08)
    plt.tight_layout()
    plt.show()


def _find_vector_variables(data):
    """Identify u, v and optional specific-humidity variables robustly."""
    names = list(data.data_vars)
    lower = {name: name.lower() for name in names}

    def pick(candidates, contains):
        for candidate in candidates:
            if candidate in names:
                return candidate
        for name, low in lower.items():
            if all(token in low for token in contains):
                return name
        return None

    u_name = pick(("u", "u10"), ("u_component", "wind"))
    v_name = pick(("v", "v10"), ("v_component", "wind"))
    q_name = pick(("q",), ("specific", "humidity"))
    if u_name is None or v_name is None:
        raise ValueError(f"Could not identify wind components among: {names}")
    return u_name, v_name, q_name


def _draw_vector_field(ax, data, i, conf):
    """Draw streamlines with humidity or wind-speed shading."""
    if not isinstance(data, xr.Dataset):
        raise TypeError("Vector plotting requires an xarray.Dataset.")

    u_name, v_name, q_name = _find_vector_variables(data)

    def select_2d(name):
        arr = data[name] if i is None else data[name].isel(time=i)
        arr = arr.squeeze(drop=True)
        if arr.ndim != 2:
            raise ValueError(f"Variable {name!r} is not two-dimensional after selection.")
        return arr

    u = select_2d(u_name)
    v = select_2d(v_name)
    q = select_2d(q_name) if q_name is not None else None

    # Matplotlib streamplot requires strictly increasing x and y coordinates.
    if u.longitude.values[0] > u.longitude.values[-1]:
        u = u.sortby("longitude")
        v = v.sortby("longitude")
        if q is not None:
            q = q.sortby("longitude")
    if u.latitude.values[0] > u.latitude.values[-1]:
        u = u.sortby("latitude")
        v = v.sortby("latitude")
        if q is not None:
            q = q.sortby("latitude")

    background = q * 1000.0 if q is not None else np.hypot(u, v)
    cf = ax.contourf(
        background.longitude,
        background.latitude,
        background,
        levels=conf["levels"],
        cmap=conf["cmap"],
        extend="max",
        transform=ccrs.PlateCarree(),
    )
    ax.streamplot(
        u.longitude.values,
        u.latitude.values,
        u.values,
        v.values,
        transform=ccrs.PlateCarree(),
        density=1.0,
        color="k",
        linewidth=0.7,
        arrowsize=0.8,
    )
    return cf


def plot_maps(data_main, data_overlay, var_key, title_prefix="Monthly"):
    """Render a 6-row map panel for the six preceding months.

    Each row displays one month.  The rendering strategy is controlled by
    ``VAR_CONFIG[var_key]["plot_type"]``:

    - ``"anomaly_only"`` — filled contours of the anomaly (SST).
    - ``"contour_overlay"`` — filled anomaly + absolute contour lines with
      optional highlighted isopleths (SLP, OLR).
    - ``"ratio"`` — precipitation ratio relative to climatology.
    - ``"vector"`` — quiver plot of wind vectors.
    - ``"streamlines"`` / ``"stream_humidity"`` — streamlines with specific
      humidity (850 hPa) or wind speed (surface) as the colour background.

    Parameters
    ----------
    data_main : xarray.DataArray or xarray.Dataset
        Primary data field (anomaly, ratio, or absolute for vector fields).
    data_overlay : xarray.DataArray or None
        Absolute field used for overlaid contour lines; ``None`` for types
        that do not need it (ratio, vector, streamlines).
    var_key : str
        Variable key defined in :data:`VAR_CONFIG`.
    title_prefix : str, default ``"Monthly"``
        String prepended to the figure super-title.
    """
    conf = VAR_CONFIG[var_key]
    print(f"Generating {title_prefix} Maps for {var_key}...")

    dates = data_main.time.values
    n_plots = len(dates)
    if n_plots == 0:
        raise ValueError("No time steps are available for plotting.")
    if n_plots > 6:
        raise ValueError(f"Expected at most 6 monthly fields, received {n_plots}.")

    fig, axes = plt.subplots(6, 1, figsize=(12, 24),
                             subplot_kw={"projection": ccrs.PlateCarree()})
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05,
                        hspace=0.2, wspace=0.0)

    for i in range(n_plots, 6):
        fig.delaxes(axes_flat[i])

    plt.suptitle(f"{var_key.upper()} - {title_prefix} Analysis (6 Past Months)",
                 fontsize=20, y=0.98, fontweight="bold")

    last_cf = None
    for i, (ax, dt) in enumerate(zip(axes_flat, dates)):
        date_str = np.datetime_as_string(dt, unit="M")
        ax.coastlines(linewidth=1.0)
        ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.6)
        ax.set_title(date_str, fontsize=12, fontweight="bold", loc="left")

        if conf["plot_type"] == "anomaly_only":
            val = data_main.isel(time=i)
            last_cf = ax.contourf(val.longitude, val.latitude, val,
                                  levels=conf["levels"], cmap=conf["cmap"],
                                  extend="both", transform=ccrs.PlateCarree())

        elif conf["plot_type"] == "contour_overlay":
            val_anom = data_main.isel(time=i)
            last_cf = ax.contourf(val_anom.longitude, val_anom.latitude, val_anom,
                                  levels=conf["levels_anom"], cmap=conf["cmap"],
                                  extend="both", transform=ccrs.PlateCarree())
            val_abs = data_overlay.isel(time=i)
            cs = ax.contour(val_abs.longitude, val_abs.latitude, val_abs,
                            levels=conf["levels_contour"], colors="black",
                            linewidths=0.6, transform=ccrs.PlateCarree())
            ax.clabel(cs, inline=True, fontsize=8, fmt="%1.0f")
            if "highlight_black" in conf:
                cs_blk = ax.contour(val_abs.longitude, val_abs.latitude, val_abs,
                                    levels=conf["highlight_black"], colors="black",
                                    linewidths=2.5, transform=ccrs.PlateCarree())
                ax.clabel(cs_blk, inline=True, fontsize=10, fmt="%1.0f")
            if "highlight_green" in conf:
                cs_grn = ax.contour(val_abs.longitude, val_abs.latitude, val_abs,
                                    levels=conf["highlight_green"], colors=["#008000"],
                                    linewidths=2.5, transform=ccrs.PlateCarree())
                ax.clabel(cs_grn, inline=True, fontsize=10, fmt="%1.0f")

        elif conf["plot_type"] == "ratio":
            val = data_main.isel(time=i)
            norm = mcolors.BoundaryNorm(conf["levels"], ncolors=256)
            last_cf = ax.contourf(val.longitude, val.latitude, val,
                                  levels=conf["levels"], norm=norm, cmap=conf["cmap"],
                                  extend="max", transform=ccrs.PlateCarree())

        elif conf["plot_type"] == "vector":
            u_var, v_var, _ = _find_vector_variables(data_main)
            u = data_main[u_var].isel(time=i).squeeze(drop=True)
            v = data_main[v_var].isel(time=i).squeeze(drop=True)
            skip = 5
            ax.quiver(
                u.longitude[::skip],
                u.latitude[::skip],
                u.values[::skip, ::skip],
                v.values[::skip, ::skip],
                transform=ccrs.PlateCarree(),
                scale=250,
                width=0.003,
            )

        elif conf["plot_type"] in ["streamlines", "stream_humidity"]:
            last_cf = _draw_vector_field(ax, data_main, i, conf)

    if last_cf is not None:
        # A ratio is a percentage of normal; an anomaly_only field is a
        # departure from normal — add the wording once, using a plain unit.
        if conf["plot_type"] == "ratio":
            cbar_label_text = "% de la normale"
        elif conf["plot_type"] == "anomaly_only":
            cbar_label_text = f"Anomalie ({conf['unit_label']})"
        else:
            cbar_label_text = conf["unit_label"]
        cbar_ax = fig.add_axes([0.96, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(last_cf, cax=cbar_ax, orientation="vertical")
        cbar.set_label(cbar_label_text, fontsize=14, fontweight="bold")
        cbar.ax.tick_params(labelsize=12)

    plt.show()


def process_variable(fpath, var_key, clim_start, clim_end, target_date,
                     ratio_mode="monthly", precip_source="era5"):
    """Compute and plot monthly anomalies or precipitation ratios."""
    if var_key not in VAR_CONFIG:
        raise KeyError(f"Unknown variable key: {var_key!r}")
    if ratio_mode not in {"monthly", "cumulative"}:
        raise ValueError('ratio_mode must be either "monthly" or "cumulative".')

    conf = VAR_CONFIG[var_key]
    source = _normalise_precip_source(precip_source)
    with xr.open_dataset(fpath) as opened:
        ds = _clean_dataset(opened).load()
        file_source = str(opened.attrs.get("precip_source", "")).lower()

    # AgERA5 monthly files are already aggregated to mm/month. ERA5 monthly
    # precipitation still needs conversion from m/day to mm/month.
    is_agera5_precip = var_key == "precip" and (source == "agera5" or file_source == "agera5")
    if var_key != "wind_850" and not is_agera5_precip:
        ds = conf["unit_func"](ds)

    target_month = date(target_date.year, target_date.month, 1)
    start_month = target_month - relativedelta(months=6)
    end_month = target_month - relativedelta(months=1)

    is_vector = conf["plot_type"] in {"vector", "streamlines", "stream_humidity"}
    data_obj = ds if is_vector else _first_data_array(ds)

    ref = data_obj.sel(time=slice(f"{clim_start}-01-01", f"{clim_end}-12-31"))
    if ref.sizes.get("time", 0) == 0:
        raise ValueError("No data are available inside the climatological period.")
    clim = ref.groupby("time.month").mean("time", skipna=True)

    time_mask = (
        (data_obj.time >= np.datetime64(start_month))
        & (data_obj.time < np.datetime64(target_month))
    )
    recent = data_obj.where(time_mask, drop=True).sortby("time")
    if recent.sizes.get("time", 0) == 0:
        raise ValueError("No monthly data were found for the requested six-month window.")

    anomaly = recent.groupby("time.month") - clim

    if conf["plot_type"] == "ratio" and ratio_mode == "cumulative":
        monthly_normal = clim.sel(month=recent["time.month"])
        normal_cumulative = monthly_normal.sum("time", skipna=True)
        observed_cumulative = recent.sum("time", skipna=True)
        normal_cumulative = normal_cumulative.where(normal_cumulative > 1.0)
        ratio_cumulative = observed_cumulative / normal_cumulative * 100.0
        plot_period_map(
            ratio_cumulative,
            var_key,
            "ratio",
            start_month,
            end_month,
            f"% de la normale {clim_start}-{clim_end} (cumul 6 mois)",
            ratio=True,
        )
        return ratio_cumulative

    if is_vector:
        map_main, map_overlay = recent, None
    elif conf["plot_type"] == "ratio":
        monthly_normal = clim.sel(month=recent["time.month"])
        monthly_normal = monthly_normal.where(monthly_normal > 1.0)
        map_main = recent / monthly_normal * 100.0
        map_overlay = None
    else:
        map_main, map_overlay = anomaly, recent

    plot_maps(map_main, map_overlay, var_key)
    if "hov_band" in conf:
        plot_hovmoller(anomaly, recent, var_key)
    return map_main


def main_driver(
    dir_save, clim_start, clim_end, target_date_str, variables_list=None,
    ratio_mode="monthly", precip_source="era5", agera5_version="2_0",
):
    """Run the monthly diagnostics with optional AgERA5 precipitation.

    Parameters
    ----------
    precip_source : {"era5", "agera5"}, default "era5"
        Source used only for precipitation. Other variables always use ERA5.
    agera5_version : str, default "2_0"
        AgERA5 version requested from the CDS when ``precip_source="agera5"``.
    """
    target_date = date.fromisoformat(target_date_str)
    source = _normalise_precip_source(precip_source)
    if variables_list is None:
        variables_list = list(VAR_CONFIG.keys())

    for var in variables_list:
        if var not in VAR_CONFIG:
            print(f" Skipping unknown variable: {var}")
            continue
        actual_source = source if var == "precip" else "era5"
        print(f"\n Traitement: {var.upper()} (source: {actual_source.upper()})")
        extent = VAR_CONFIG[var]["extent"]
        fpath = download_data(
            dir_save, clim_start, clim_end, var, extent, target_date,
            precip_source=source, agera5_version=agera5_version,
        )
        if fpath:
            process_variable(
                fpath, var, clim_start, clim_end, target_date,
                ratio_mode=ratio_mode, precip_source=source,
            )

    print("\n Finished.")


# ===========================================================================
# DAILY PIPELINE — last N days up to the target date, aggregated (mean/sum)
# ===========================================================================

def download_daily_data(
    dir_to_save, var_key, extent, target_date, n_days=30,
    precip_source="era5", agera5_version="2_0",
):
    """Download daily data, optionally using AgERA5 for precipitation."""
    if var_key not in VAR_CONFIG:
        raise KeyError(f"Unknown variable key: {var_key!r}")
    if n_days < 1:
        raise ValueError("n_days must be at least 1.")

    source = _normalise_precip_source(precip_source)
    conf = VAR_CONFIG[var_key]
    start = target_date - timedelta(days=n_days - 1)
    dates = [start + timedelta(days=i) for i in range(n_days)]
    years = sorted({str(d.year) for d in dates})
    months = sorted({f"{d.month:02d}" for d in dates})
    days = [f"{i:02d}" for i in range(1, 32)]
    ext_str = f"{extent[0]}_{extent[1]}_{extent[2]}_{extent[3]}"

    if var_key == "precip" and source == "agera5":
        if start.year < 1979:
            raise ValueError("AgERA5 is available from 1979.")
        fname = (
            f"agera5_daily_precip_v{agera5_version}_"
            f"{start:%Y%m%d}-{target_date:%Y%m%d}_{ext_str}.nc"
        )
        fpath = os.path.join(dir_to_save, fname)
        if os.path.exists(fpath):
            print(f" Data found: {fname}")
            return fpath

        request = {
            "variable": "precipitation_flux",
            "year": years,
            "month": months,
            "day": days,
            "version": agera5_version,
            "area": extent,
        }
        print(f" Downloading AgERA5 precipitation ({start} → {target_date})...")
        try:
            os.makedirs(dir_to_save, exist_ok=True)
            da = _retrieve_agera5_precipitation(request, start=start, end=target_date)
            output = da.to_dataset()
            output.attrs.update({
                "precip_source": "agera5",
                "agera5_version": agera5_version,
            })
            output.to_netcdf(fpath)
            print(" Download complete.")
            return fpath
        except Exception as exc:
            print(f" Error downloading AgERA5 precipitation: {exc}")
            return None

    fname = (
        f"era5_daily_{var_key}_{DAILY_STATISTIC[var_key]}_"
        f"{start:%Y%m%d}-{target_date:%Y%m%d}_{ext_str}.nc"
    )
    fpath = os.path.join(dir_to_save, fname)
    if os.path.exists(fpath):
        print(f" Data found: {fname}")
        return fpath

    print(f" Downloading daily {var_key} from ERA5 ({start} → {target_date})...")
    request = {
        "product_type": "reanalysis",
        "variable": conf["cds_name"],
        "year": years,
        "month": months,
        "day": days,
        "daily_statistic": DAILY_STATISTIC[var_key],
        "time_zone": "utc+00:00",
        "frequency": "1_hourly",
        "area": extent,
    }
    dataset_name = "derived-era5-single-levels-daily-statistics"
    if conf["level_type"] == "pressure":
        dataset_name = "derived-era5-pressure-levels-daily-statistics"
        request["pressure_level"] = conf.get("pressure_level", "850")

    try:
        os.makedirs(dir_to_save, exist_ok=True)
        result = earthkit.data.from_source("cds", dataset_name, request)
        _save_earthkit_result(result, fpath)
        print(" Download complete.")
        return fpath
    except Exception as exc:
        print(f" Error downloading daily {var_key}: {exc}")
        return None


def _analog_window(target_date, n_days, year):
    """Return the (start, end) dates of the *n_days* window anchored on *year*.

    The window ends on the same calendar day (month/day) as *target_date* but
    in *year*, and spans *n_days* backwards.  ``29 Feb`` is clamped to
    ``28 Feb`` in non-leap years.

    Parameters
    ----------
    target_date : datetime.date
        Reference target date (its month/day define the anchor).
    n_days : int
        Window length in days.
    year : int
        Year to anchor the analog window on.

    Returns
    -------
    tuple of datetime.date
        ``(start, end)`` — both inclusive.
    """
    try:
        anchor = date(year, target_date.month, target_date.day)
    except ValueError:  # 29 Feb in a non-leap year
        anchor = date(year, target_date.month, 28)
    return anchor - timedelta(days=n_days - 1), anchor


def download_daily_climatology(
    dir_to_save, var_key, extent, target_date, n_days, clim_start, clim_end,
    precip_source="era5", agera5_version="2_0",
):
    """Download per-year analog windows for daily climatological diagnostics."""
    if clim_start > clim_end:
        raise ValueError("clim_start must be less than or equal to clim_end.")
    source = _normalise_precip_source(precip_source)
    if var_key == "precip" and source == "agera5" and clim_start < 1979:
        raise ValueError("AgERA5 climatology cannot begin before 1979.")

    conf = VAR_CONFIG[var_key]
    days = [f"{i:02d}" for i in range(1, 32)]
    ext_str = f"{extent[0]}_{extent[1]}_{extent[2]}_{extent[3]}"
    os.makedirs(dir_to_save, exist_ok=True)
    paths = []

    for y in range(clim_start, clim_end + 1):
        s, e = _analog_window(target_date, n_days, y)
        if var_key == "precip" and source == "agera5" and s.year < 1979:
            print(f"  clim {y} skipped: AgERA5 starts in 1979")
            continue

        yms = set()
        current = s
        while current <= e:
            yms.add((current.year, current.month))
            current += timedelta(days=1)
        req_years = sorted({str(yy) for yy, _ in yms})
        req_months = sorted({f"{mm:02d}" for _, mm in yms})

        if var_key == "precip" and source == "agera5":
            fname = (
                f"agera5_dailyclim_precip_v{agera5_version}_y{y}"
                f"_md{target_date:%m%d}_n{n_days}_{ext_str}.nc"
            )
        else:
            fname = (
                f"era5_dailyclim_{var_key}_{DAILY_STATISTIC[var_key]}_y{y}"
                f"_md{target_date:%m%d}_n{n_days}_{ext_str}.nc"
            )
        fpath = os.path.join(dir_to_save, fname)
        if os.path.exists(fpath):
            paths.append(fpath)
            continue

        print(f"  clim {y} ({s} → {e})...", end=" ", flush=True)
        try:
            if var_key == "precip" and source == "agera5":
                request = {
                    "variable": "precipitation_flux",
                    "year": req_years,
                    "month": req_months,
                    "day": days,
                    "version": agera5_version,
                    "area": extent,
                }
                da = _retrieve_agera5_precipitation(request, start=s, end=e)
                output = da.to_dataset()
                output.attrs.update({
                    "precip_source": "agera5",
                    "agera5_version": agera5_version,
                })
                output.to_netcdf(fpath)
            else:
                request = {
                    "product_type": "reanalysis",
                    "variable": conf["cds_name"],
                    "year": req_years,
                    "month": req_months,
                    "day": days,
                    "daily_statistic": DAILY_STATISTIC[var_key],
                    "time_zone": "utc+00:00",
                    "frequency": "1_hourly",
                    "area": extent,
                }
                dataset_name = "derived-era5-single-levels-daily-statistics"
                if conf["level_type"] == "pressure":
                    dataset_name = "derived-era5-pressure-levels-daily-statistics"
                    request["pressure_level"] = conf.get("pressure_level", "850")
                result = earthkit.data.from_source("cds", dataset_name, request)
                _save_earthkit_result(result, fpath)

            paths.append(fpath)
            print("ok")
        except Exception as exc:
            print(f"failed: {exc}")

    print(f" Climatology: {len(paths)}/{clim_end - clim_start + 1} years available.")
    return paths


def _open_clim(paths, var_key, precip_source="era5"):
    """Open + concatenate per-year climatology files into one DataArray.

    Each file is cleaned individually (time/expver/pressure_level) and unit-
    converted, then concatenated along ``time`` and sorted.

    Parameters
    ----------
    paths : list of str
        Per-year NetCDF paths from :func:`download_daily_climatology`.
    var_key : str
        Variable key defined in :data:`VAR_CONFIG`.

    Returns
    -------
    xarray.DataArray
        Daily field in physical units, time-sorted across all reference years.
    """
    source = _normalise_precip_source(precip_source)
    unit_func, _ = DAILY_UNIT[var_key]
    parts = []
    for path in paths:
        with xr.open_dataset(path) as opened:
            ds = _clean_dataset(opened).load()
            file_source = str(opened.attrs.get("precip_source", "")).lower()
        is_agera5_precip = var_key == "precip" and (source == "agera5" or file_source == "agera5")
        if var_key != "wind_850" and not is_agera5_precip:
            ds = unit_func(ds)
        parts.append(_first_data_array(ds))
    if not parts:
        raise ValueError("No climatology files could be opened.")
    return xr.concat(parts, dim="time").sortby("time").drop_duplicates("time")


def _clim_window_stat(da, target_date, n_days, clim_start, clim_end, reducer="sum"):
    """Mean of per-year analog-window sums or means."""
    if reducer not in {"sum", "mean"}:
        raise ValueError('reducer must be either "sum" or "mean".')

    per_year = []
    years = []
    for year in range(clim_start, clim_end + 1):
        start, end = _analog_window(target_date, n_days, year)
        subset = da.sel(time=slice(str(start), str(end)))
        if subset.sizes.get("time", 0) == 0:
            continue
        aggregate = (
            subset.mean("time", skipna=True)
            if reducer == "mean"
            else subset.sum("time", skipna=True)
        )
        per_year.append(aggregate)
        years.append(year)

    if not per_year:
        raise ValueError("No reference-year data found for the analog window.")

    climatology = xr.concat(per_year, dim=xr.IndexVariable("clim_year", years))
    return climatology.mean("clim_year", skipna=True)


def _clim_window_sum(da, target_date, n_days, clim_start, clim_end):
    """Backward-compatible climatological window-sum helper."""
    return _clim_window_stat(
        da, target_date, n_days, clim_start, clim_end, reducer="sum"
    )


def plot_period_map(data, var_key, stat, start, end, unit_label,
                    ratio=False, anomaly=False):
    """Single-panel map of a period aggregate, styled from :data:`VAR_CONFIG`.

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        Already-aggregated 2-D field (or u/v[/humidity] dataset for vectors).
    var_key : str
        Variable key defined in :data:`VAR_CONFIG`.
    stat : str
        ``"mean"``, ``"sum"``, ``"ratio"`` or ``"anomaly"`` — title only.
    start, end : datetime.date
        Window bounds, for the title.
    unit_label : str
        Colorbar label.
    ratio : bool, default False
        If True, render *data* as a percentage-of-normal field using the
        diverging precipitation-ratio styling (BrBG, 100 % centred).
    anomaly : bool, default False
        If True, render *data* as a departure-from-normal field on a diverging
        scale centred on zero, using the variable's anomaly levels/cmap.
    """
    conf = VAR_CONFIG[var_key]

    fig = plt.figure(figsize=(12, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(linewidth=1.0)
    ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.6)
    ax.set_title(
        f"{var_key.upper()} — {stat.upper()} {start:%d %b %Y} → {end:%d %b %Y}",
        fontsize=14, fontweight="bold",
    )

    if ratio:
        da = data if isinstance(data, xr.DataArray) else _first_data_array(data)
        levels = [0, 25, 50, 75, 90, 110, 125, 150, 200, 300]
        norm = mcolors.BoundaryNorm(levels, ncolors=256)
        cf = ax.contourf(da.longitude, da.latitude, da.values.squeeze(),
                         levels=levels, norm=norm, cmap="BrBG", extend="max",
                         transform=ccrs.PlateCarree())
    elif anomaly:
        da = data if isinstance(data, xr.DataArray) else _first_data_array(data)
        # Prefer the variable's configured anomaly levels; else symmetric auto.
        if "levels_anom" in conf:
            levels = conf["levels_anom"]
        elif conf["plot_type"] == "anomaly_only" and "levels" in conf:
            levels = conf["levels"]
        else:
            finite = np.asarray(da.values)[np.isfinite(da.values)]
            vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
            levels = np.linspace(-vmax, vmax, 15)
        cmap = conf["cmap"] if conf["cmap"] in ("RdBu_r", "BrBG") else "RdBu_r"
        cf = ax.contourf(da.longitude, da.latitude, da.values.squeeze(),
                         levels=levels, cmap=cmap, extend="both",
                         transform=ccrs.PlateCarree())
    elif conf["plot_type"] in ("streamlines", "stream_humidity", "vector"):
        cf = _draw_vector_field(ax, data, None, conf)
    else:
        da = data if isinstance(data, xr.DataArray) else _first_data_array(data)
        vals = da.values.squeeze()
        if var_key == "precip":
            levels = [0, 1, 5, 10, 25, 50, 100, 150, 200, 300, 500]
            norm = mcolors.BoundaryNorm(levels, ncolors=256)
            cf = ax.contourf(da.longitude, da.latitude, vals, levels=levels,
                             norm=norm, cmap=conf["cmap"], extend="max",
                             transform=ccrs.PlateCarree())
        else:
            cf = ax.contourf(da.longitude, da.latitude, vals, levels=15,
                             cmap=conf["cmap"], extend="both",
                             transform=ccrs.PlateCarree())

    cbar = plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.06, shrink=0.8)
    cbar.set_label(unit_label, fontsize=12, fontweight="bold")
    plt.show()


def process_daily(dir_save, var_key, target_date_str, n_days=30, stat=None,
                  clim_start=None, clim_end=None, precip_source="era5",
                  agera5_version="2_0"):
    """Download, aggregate, and plot the last *n_days* up to the target date.

    Parameters
    ----------
    dir_save : str
        Root directory for cached NetCDF files.
    var_key : str
        Variable key defined in :data:`VAR_CONFIG`.
    target_date_str : str
        ISO-format target date (``"YYYY-MM-DD"``); last day of the window.
    n_days : int, default 30
        Length of the window in days.
    stat : {"mean", "sum", "ratio", "anomaly"}, optional
        Aggregation over the window.  Defaults to ``"sum"`` for precipitation
        and ``"mean"`` for every other variable.

        - ``"ratio"`` — window total as a percentage of the climatological
          normal for the same calendar window (precip-oriented).
        - ``"anomaly"`` — window mean minus the climatological window-mean
          normal, on a diverging scale (SST/T2m/SLP-oriented; scalar fields
          only, not the wind/streamline variables).

        ``"ratio"`` and ``"anomaly"`` both require *clim_start*/*clim_end*.
    clim_start, clim_end : int, optional
        Inclusive reference-period bounds.  Required for ``"ratio"``/``"anomaly"``.

    Returns
    -------
    xarray.DataArray or xarray.Dataset or None
        The aggregated field (ratio %, anomaly, sum or mean), or ``None``.
    """
    if var_key not in VAR_CONFIG:
        raise KeyError(f"Unknown variable key: {var_key!r}")
    if n_days < 1:
        raise ValueError("n_days must be at least 1.")

    source = _normalise_precip_source(precip_source)
    conf = VAR_CONFIG[var_key]
    target_date = date.fromisoformat(target_date_str)
    start = target_date - timedelta(days=n_days - 1)
    if stat is None:
        stat = "sum" if var_key == "precip" else "mean"
    if stat not in {"mean", "sum", "ratio", "anomaly"}:
        raise ValueError('stat must be "mean", "sum", "ratio", or "anomaly".')
    if stat == "ratio" and var_key != "precip":
        raise ValueError('stat="ratio" is only meaningful for precipitation.')

    unit_func, unit_label = DAILY_UNIT[var_key]

    # Observed window --------------------------------------------------------
    fpath = download_daily_data(
        dir_save, var_key, conf["extent"], target_date, n_days,
        precip_source=source, agera5_version=agera5_version,
    )
    if fpath is None:
        return None
    with xr.open_dataset(fpath) as opened:
        ds = _clean_dataset(opened).sel(
            time=slice(str(start), str(target_date))
        ).load()
    if ds.sizes.get("time", 0) == 0:
        print(" No daily data are available in the requested window.")
        return None
    if ds.sizes.get("time", 0) < n_days:
        print(
            f" Warning: only {ds.sizes['time']} of {n_days} requested days "
            "are available."
        )
    is_agera5_precip = var_key == "precip" and source == "agera5"
    if var_key != "wind_850" and not is_agera5_precip:
        ds = unit_func(ds)

    # Ratio to normal --------------------------------------------------------
    if stat == "ratio":
        if clim_start is None or clim_end is None:
            raise ValueError('stat="ratio" requires clim_start and clim_end.')

        obs = ds if isinstance(ds, xr.DataArray) else _first_data_array(ds)
        obs_sum = obs.sum("time")

        cpaths = download_daily_climatology(
            dir_save, var_key, conf["extent"], target_date, n_days,
            clim_start, clim_end, precip_source=source,
            agera5_version=agera5_version,
        )
        if not cpaths:
            print(" No climatology data available; skipping ratio.")
            return None
        cda = _open_clim(cpaths, var_key, precip_source=source)

        normal = _clim_window_stat(cda, target_date, n_days, clim_start, clim_end, "sum")
        normal = normal.where(normal > 1.0)          # mask ~dry normals (mm)
        ratio = (obs_sum / normal) * 100.0

        plot_period_map(ratio, var_key, "ratio", start, target_date,
                        f"% de la normale {clim_start}-{clim_end}", ratio=True)
        return ratio

    # Anomaly to normal ------------------------------------------------------
    if stat == "anomaly":
        if clim_start is None or clim_end is None:
            raise ValueError('stat="anomaly" requires clim_start and clim_end.')
        if conf["plot_type"] in ("vector", "streamlines", "stream_humidity"):
            print(f" Anomaly not supported for vector field '{var_key}'.")
            return None

        obs = ds if isinstance(ds, xr.DataArray) else _first_data_array(ds)
        obs_mean = obs.mean("time")

        cpaths = download_daily_climatology(
            dir_save, var_key, conf["extent"], target_date, n_days,
            clim_start, clim_end, precip_source=source,
            agera5_version=agera5_version,
        )
        if not cpaths:
            print(" No climatology data available; skipping anomaly.")
            return None
        cda = _open_clim(cpaths, var_key, precip_source=source)

        normal = _clim_window_stat(cda, target_date, n_days, clim_start, clim_end, "mean")
        anom = obs_mean - normal

        plot_period_map(anom, var_key, "anomaly", start, target_date,
                        f"Anomalie {clim_start}-{clim_end} ({unit_label})",
                        anomaly=True)
        return anom

    # Plain sum / mean -------------------------------------------------------
    agg = ds.sum("time") if stat == "sum" else ds.mean("time")
    if var_key == "precip" and stat == "sum":
        unit_label = f"mm ({n_days} jours)"

    plot_period_map(agg, var_key, stat, start, target_date, unit_label)
    return agg


def main_daily_driver(
    dir_save, target_date_str, n_days=30, variables_list=None, stat=None,
    clim_start=None, clim_end=None, precip_source="era5",
    agera5_version="2_0",
):
    """Run daily diagnostics, optionally sourcing precipitation from AgERA5.

    Examples
    --------
    >>> main_daily_driver(
    ...     "./data", "2026-07-14", n_days=30,
    ...     variables_list=["precip"], stat="ratio",
    ...     clim_start=1991, clim_end=2020, precip_source="agera5",
    ... )
    """
    source = _normalise_precip_source(precip_source)
    if variables_list is None:
        variables_list = list(VAR_CONFIG.keys())
    for var in variables_list:
        if var not in VAR_CONFIG:
            print(f" Skipping unknown variable: {var}")
            continue
        actual_source = source if var == "precip" else "era5"
        print(f"\n Traitement journalier: {var.upper()} (source: {actual_source.upper()})")
        process_daily(
            dir_save, var, target_date_str, n_days=n_days, stat=stat,
            clim_start=clim_start, clim_end=clim_end,
            precip_source=source, agera5_version=agera5_version,
        )
    print("\n Finished.")


# ===========================================================================
# INTERACTIVE VIEWERS
# ===========================================================================

class C3SViewer:
    """Interactive Jupyter widget for Copernicus C3S seasonal forecast maps.

    Displays C3S probabilistic seasonal forecast products (SST, MSLP,
    precipitation, temperature, wind, and Niño plumes) inside an IFrame,
    with dropdown controls for product, start year/month, area, and map type.

    Examples
    --------
    >>> viewer = C3SViewer()
    >>> viewer.show()

    Notes
    -----
    The area selector previously became unresponsive because
    ``_update_chart`` observed the Area/Type dropdowns *while* the product
    handler swapped their options — ipywidgets auto-resets a dropdown's value
    during an options swap, firing the observer in an inconsistent state and
    (on some ipywidgets versions) raising inside the observer chain.  The
    options are now swapped with those observers detached, and the user's zone
    is preserved when switching between two spatial map products.
    """

    def __init__(self):
        self.BASE_PACKAGE = (
            "https://climate.copernicus.eu/charts/packages/c3s_seasonal/products/"
        )

        self.products = {
            "Sea Surface Temperature (SST)": {"slug": "c3s_seasonal_spatial_mm_ssto_3m", "type": "map"},
            "Mean Sea Level Pressure (MSLP)": {"slug": "c3s_seasonal_spatial_mm_mslp_3m", "type": "map"},
            "Precipitation": {"slug": "c3s_seasonal_spatial_mm_rain_3m", "type": "map"},
            "2m Temperature (T2m)": {"slug": "c3s_seasonal_spatial_mm_2mtm_3m", "type": "map"},
            "10m Wind Speed": {"slug": "c3s_seasonal_spatial_mm_wspd_3m", "type": "map"},
            "Nino Ensemble Plumes": {"slug": "c3s_seasonal_plume_mm", "type": "plume"},
        }

        self.areas_spatial = [
            ("Global", "area08"), ("Europe", "area01"), ("Africa", "area02"),
            ("North America", "area05"), ("South America", "area04"),
            ("Asia", "area03"), ("Australasia", "area06"),
        ]
        self.areas_nino = [
            ("Nino 3", "nino3"), ("Nino 3.4", "nino34"),
            ("Nino 4", "nino4"), ("Nino 1+2", "nino12"),
        ]
        self.types_spatial = [
            ("Tercile Summary", "tsum"), ("Prob(most likely)", "prob"), ("Ensemble Mean", "em"),
        ]
        self.types_nino = [("Plume", "plume")]

        current_year = datetime.datetime.now().year
        self.years = [str(y) for y in range(current_year - 2, current_year + 3)]
        self.months = [
            ("Jan", "01"), ("Feb", "02"), ("Mar", "03"), ("Apr", "04"),
            ("May", "05"), ("Jun", "06"), ("Jul", "07"), ("Aug", "08"),
            ("Sep", "09"), ("Oct", "10"), ("Nov", "11"), ("Dec", "12"),
        ]

        self.w_product = widgets.Dropdown(
            options=list(self.products.keys()),
            value="Sea Surface Temperature (SST)", description="Product:",
        )
        self.w_year = widgets.Dropdown(
            options=self.years, value=str(current_year), description="Start Year:"
        )
        self.w_month = widgets.Dropdown(options=self.months, value="01", description="Start Month:")
        self.w_area = widgets.Dropdown(options=self.areas_spatial, value="area08", description="Area:")
        self.w_type = widgets.Dropdown(options=self.types_spatial, value="tsum", description="Map Type:")
        self.w_link = widgets.HTML()
        self.out_display = widgets.Output()

        self._current_kind = "map"

        # Only the product observes _on_product_change; the year/month/area/
        # type dropdowns observe _update_chart.  _on_product_change itself
        # calls _update_chart, so w_product must NOT observe it directly.
        self.w_product.observe(self._on_product_change, names="value")
        for w in (self.w_year, self.w_month, self.w_area, self.w_type):
            w.observe(self._update_chart, names="value")

        self.ui = widgets.VBox([
            self.w_product,
            widgets.HBox([self.w_year, self.w_month, self.w_area, self.w_type]),
            self.w_link,
        ])
        self._update_chart()

    def _detach(self):
        """Temporarily stop area/type from triggering a chart refresh."""
        for w in (self.w_area, self.w_type):
            try:
                w.unobserve(self._update_chart, names="value")
            except ValueError:
                pass

    def _attach(self):
        """Re-enable area/type chart-refresh observers."""
        for w in (self.w_area, self.w_type):
            w.observe(self._update_chart, names="value")

    def _on_product_change(self, change):
        """Swap area/type option sets when the product *kind* changes.

        Niño plume products use a different set of areas and a fixed map type.
        The option swap is done with the area/type observers detached so the
        mid-swap value reset does not fire :meth:`_update_chart` in an
        inconsistent state.  For map→map changes the user's zone is preserved.

        Parameters
        ----------
        change : dict
            ipywidgets observe change dictionary; only ``change['new']`` is used.
        """
        new_kind = self.products[change["new"]]["type"]

        if new_kind != self._current_kind:
            self._detach()
            try:
                if new_kind == "plume":
                    self.w_area.options = self.areas_nino
                    self.w_area.value = "nino3"
                    self.w_type.options = self.types_nino
                    self.w_type.value = "plume"
                    self.w_type.disabled = True
                else:
                    self.w_area.options = self.areas_spatial
                    self.w_area.value = "area08"
                    self.w_type.options = self.types_spatial
                    self.w_type.value = "tsum"
                    self.w_type.disabled = False
            finally:
                self._attach()
            self._current_kind = new_kind

        self._update_chart()

    def _update_chart(self, change=None):
        """Rebuild the C3S chart URL and refresh the embedded IFrame.

        Parameters
        ----------
        change : dict or None
            ipywidgets observe change dictionary (unused; present for
            compatibility with the observe callback signature).
        """
        prod_info = self.products[self.w_product.value]
        base_time = f"{self.w_year.value}{self.w_month.value}010000"

        params = [
            f"area={self.w_area.value}",
            f"base_time={base_time}",
            f"type={self.w_type.value}",
        ]
        if prod_info["type"] == "map":
            dt_base = datetime.date(int(self.w_year.value), int(self.w_month.value), 1)
            dt_valid = dt_base + relativedelta(months=1)
            params.append(f"valid_time={dt_valid.year}{dt_valid.month:02d}010000")

        full_url = f"{self.BASE_PACKAGE}{prod_info['slug']}?{'&'.join(params)}"
        self.w_link.value = (
            f'<a href="{full_url}" target="_blank">Open in browser ↗</a> '
            f'<code style="font-size:10px">{full_url}</code>'
        )
        with self.out_display:
            self.out_display.clear_output(wait=True)
            display(IFrame(src=full_url, width="100%", height=850))

    def show(self):
        """Display the widget interface in the current Jupyter cell."""
        display(self.ui, self.out_display)


class BOMViewer:
    """Interactive Jupyter widget for BOM MJO monitoring charts.

    Embeds the Australian Bureau of Meteorology MJO page inside an IFrame
    with a dropdown to navigate between tabs (Outlooks, MJO Phase Diagram,
    Monitoring, Cloudiness, Regional Cloudiness, Time-Longitude, Tropical
    Update).

    Examples
    --------
    >>> viewer = BOMViewer()
    >>> viewer.show()
    """

    def __init__(self):
        self.bom_tabs = [
            ("Outlooks (Forecasts)", "Outlooks"),
            ("MJO Phase Diagram", "MJO%20phase"),
            ("Monitoring", "Monitoring"),
            ("Cloudiness", "Cloudiness"),
            ("Regional Cloudiness", "Regional%20cloudiness"),
            ("Time-Longitude", "Time-longitude"),
            ("Tropical Update", "Tropical%20update"),
        ]

        self.w_bom_tab = widgets.Dropdown(
            options=self.bom_tabs,
            value="Outlooks",
            description="Select Tab:",
            style={"description_width": "initial"},
        )
        self.out_bom = widgets.Output()

        self.w_bom_tab.observe(self._update_bom, names="value")
        self.ui = widgets.VBox([self.w_bom_tab, self.out_bom])
        self._update_bom()

    def _update_bom(self, change=None):
        """Rebuild the BOM URL and refresh the embedded IFrame.

        Parameters
        ----------
        change : dict or None
            ipywidgets observe change dictionary (unused; present for
            compatibility with the observe callback signature).
        """
        tab_slug = self.w_bom_tab.value
        url = f"https://www.bom.gov.au/climate/mjo/#tabs={tab_slug}"
        with self.out_bom:
            self.out_bom.clear_output(wait=True)
            display(IFrame(src=url, width="100%", height=1000))

    def show(self):
        """Display the widget interface in the current Jupyter cell."""
        display(self.ui)



# """Seasonal diagnostic tools for ERA5 data.

# This module provides:

# - :data:`VAR_CONFIG` — per-variable download and plotting configuration.
# - :func:`download_data` — ERA5 monthly download via earthkit.
# - :func:`process_variable` — compute anomalies/ratios and dispatch to plots.
# - :func:`plot_maps` — 6-panel monthly map layout.
# - :func:`plot_hovmoller` — time–longitude Hovmöller diagram.
# - :func:`main_driver` — end-to-end entry point.
# - :class:`C3SViewer` — interactive Copernicus C3S seasonal forecast viewer.
# - :class:`BOMViewer` — interactive BOM MJO viewer.
# """

# import os
# import warnings
# import numpy as np
# import xarray as xr
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# import matplotlib.colors as mcolors
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# from datetime import date
# from dateutil.relativedelta import relativedelta
# import earthkit.data

# warnings.filterwarnings("ignore")

# # ---------------------------------------------------------------------------
# # Variable configuration
# # ---------------------------------------------------------------------------

# VAR_CONFIG = {
#     "sst": {
#         "cds_name": "sea_surface_temperature",
#         "level_type": "single",
#         "unit_func": lambda x: x - 273.15,
#         "unit_label": "Anomalie SST (°C)",
#         "plot_type": "anomaly_only",
#         "cmap": "RdBu_r",
#         "levels": np.arange(-3.0, 3.5, 0.5),
#         "extent": [40, -180, -40, 180],
#         "compute_seasonal": True,
#     },
#     "slp": {
#         "cds_name": "mean_sea_level_pressure",
#         "level_type": "single",
#         "unit_func": lambda x: x / 100,
#         "unit_label": "hPa",
#         "plot_type": "contour_overlay",
#         "cmap": "RdBu_r",
#         "levels_anom": np.arange(-5, 6, 1),
#         "levels_contour": np.arange(990, 1038, 2),
#         "highlight_black": [1015],
#         "highlight_green": [1012],
#         "extent": [50, -60, -45, 130],
#     },
#     "olr": {
#         "cds_name": "top_net_thermal_radiation",
#         "level_type": "single",
#         "unit_func": lambda x: x / 86400,
#         "unit_label": "W/m²",
#         "plot_type": "contour_overlay",
#         "cmap": "BrBG",
#         "levels_anom": np.arange(-50, 55, 5),
#         "levels_contour": np.arange(-320, -160, 20),
#         "extent": [40, -180, -40, 180],
#         "hov_band": [10, -10],
#         "hov_cmap": "BrBG",
#         "hov_levels": np.arange(-60, 65, 5),
#     },
#     "precip": {
#         "cds_name": "total_precipitation",
#         "level_type": "single",
#         "unit_func": lambda x: x * 1000,
#         "unit_label": "mm/mois",
#         "plot_type": "ratio",
#         "cmap": "BrBG",
#         "levels": [0, 25, 50, 75, 90, 110, 125, 150, 200, 300],
#         "extent": [35, -25, -15, 55],
#     },
#     "wind_850": {
#         "cds_name": ["u_component_of_wind", "v_component_of_wind", "specific_humidity"],
#         "level_type": "pressure",
#         "pressure_level": "850",
#         "unit_func": lambda x: x,
#         "unit_label": "Humidité Spécifique (g/kg)",
#         "plot_type": "stream_humidity",
#         "cmap": "GnBu",
#         "levels": np.arange(0, 18, 2),
#         "extent": [40, -60, -20, 60],
#     },
#     "wind_surface": {
#         "cds_name": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
#         "level_type": "single",
#         "unit_func": lambda x: x,
#         "unit_label": "Vitesse du Vent (m/s)",
#         "plot_type": "streamlines",
#         "cmap": "YlOrRd",
#         "levels": np.arange(0, 15, 1),
#         "extent": [35, -30, -15, 50],
#     },
# }
# """dict: Per-variable ERA5 download and plotting configuration.

# Each key is a short variable identifier.  The associated dict contains:

# - ``cds_name`` — CDS variable name(s).
# - ``level_type`` — ``"single"`` or ``"pressure"``.
# - ``unit_func`` — callable that converts raw ERA5 units.
# - ``unit_label`` — axis / colorbar label.
# - ``plot_type`` — rendering strategy used by :func:`plot_maps`.
# - ``cmap``, ``levels``, ``extent`` — Matplotlib / Cartopy parameters.
# - ``hov_band``, ``hov_cmap``, ``hov_levels`` — Hovmöller parameters (OLR only).
# """


# # ---------------------------------------------------------------------------
# # Download
# # ---------------------------------------------------------------------------

# def download_data(dir_to_save, clim_start, clim_end, var_key, extent, target_date):
#     """Download ERA5 monthly data for the six months preceding *target_date*.

#     Files are saved as NetCDF and cached on disk; if the file already exists
#     the download is skipped.

#     Parameters
#     ----------
#     dir_to_save : str
#         Directory where the NetCDF file will be written.
#     clim_start : int
#         First year of the climatological reference period.
#     clim_end : int
#         Last year of the climatological reference period.
#     var_key : str
#         Variable key defined in :data:`VAR_CONFIG`.
#     extent : list of float
#         Bounding box ``[north, west, south, east]`` passed to the CDS API.
#     target_date : datetime.date
#         The target (forecast) date.  Data are downloaded for the six months
#         *preceding* this date (i.e. T−6 to T−1).

#     Returns
#     -------
#     str or None
#         Absolute path to the downloaded NetCDF file, or ``None`` if the
#         download failed.
#     """
#     conf = VAR_CONFIG[var_key]

#     months_idx = []
#     years_needed = set()
#     for i in range(1, 7):
#         d = target_date - relativedelta(months=i)
#         months_idx.append(d.strftime("%m"))
#         years_needed.add(str(d.year))
#     months_idx = sorted(list(set(months_idx)))

#     years_range = [str(y) for y in range(clim_start, clim_end + 1)]
#     for y in years_needed:
#         if y not in years_range:
#             years_range.append(y)

#     ext_str = f"{extent[0]}_{extent[1]}_{extent[2]}_{extent[3]}"
#     fname = f"era5_{var_key}_{clim_start}-{clim_end}_past5_{ext_str}.nc"
#     fpath = os.path.join(dir_to_save, fname)

#     if os.path.exists(fpath):
#         print(f" Data found: {fname}")
#         return fpath

#     print(f" Downloading {var_key}...")
#     request = {
#         "product_type": "monthly_averaged_reanalysis",
#         "variable": conf["cds_name"],
#         "year": years_range,
#         "month": months_idx,
#         "time": "00:00",
#         "area": extent,
#         "format": "netcdf",
#     }
#     dataset_name = "reanalysis-era5-single-levels-monthly-means"
#     if conf["level_type"] == "pressure":
#         dataset_name = "reanalysis-era5-pressure-levels-monthly-means"
#         request["pressure_level"] = conf.get("pressure_level", "850")

#     try:
#         os.makedirs(dir_to_save, exist_ok=True)
#         ds = earthkit.data.from_source("cds", dataset_name, request)
#         ds.save(fpath)
#         print(" Download complete.")
#         return fpath
#     except Exception as e:
#         print(f" Error: {e}")
#         return None


# # ---------------------------------------------------------------------------
# # Hovmöller diagram
# # ---------------------------------------------------------------------------

# def plot_hovmoller(data_anom, data_abs, var_key):
#     """Plot a time–longitude Hovmöller diagram for a supported variable.

#     The diagram is only produced for variables that define a ``hov_band`` in
#     :data:`VAR_CONFIG` (currently OLR).  For vector fields the U-component
#     is used.

#     Parameters
#     ----------
#     data_anom : xarray.Dataset or xarray.DataArray
#         Anomaly field, already averaged over the configured latitude band.
#     data_abs : xarray.Dataset or xarray.DataArray
#         Absolute (non-anomaly) field for overlaid contours.
#     var_key : str
#         Variable key defined in :data:`VAR_CONFIG`.
#     """
#     conf = VAR_CONFIG[var_key]
#     if "hov_band" not in conf:
#         return

#     print(f"Generating Hovmöller for {var_key}...")
#     lat_max, lat_min = max(conf["hov_band"]), min(conf["hov_band"])

#     hov_anom = data_anom.sel(latitude=slice(lat_max, lat_min)).mean(dim="latitude")
#     hov_abs = data_abs.sel(latitude=slice(lat_max, lat_min)).mean(dim="latitude")

#     if conf["plot_type"] in ["vector", "streamlines", "stream_humidity"]:
#         u_var = [v for v in hov_anom.data_vars if "u" in v.lower() or "var131" in v][0]
#         to_plot_anom = hov_anom[u_var]
#         to_plot_abs = hov_abs[u_var]
#         title_suffix = f"U-Component ({lat_min}° to {lat_max}°)"
#     else:
#         if isinstance(hov_anom, xr.Dataset):
#             var_name = list(hov_anom.data_vars)[0]
#             to_plot_anom = hov_anom[var_name]
#             to_plot_abs = hov_abs[var_name]
#         else:
#             to_plot_anom = hov_anom
#             to_plot_abs = hov_abs
#         title_suffix = f"({lat_min}° to {lat_max}°)"

#     fig, ax = plt.subplots(figsize=(14, 10))
#     times = to_plot_anom.time.values
#     lons = to_plot_anom.longitude.values
#     vals_anom = to_plot_anom.values.squeeze()
#     vals_abs = to_plot_abs.values.squeeze()

#     cf = ax.contourf(lons, times, vals_anom, levels=conf["hov_levels"],
#                      cmap=conf["hov_cmap"], extend="both")

#     if var_key == "olr":
#         ax.contour(lons, times, vals_abs, levels=[-240, -220, -200],
#                    colors="k", linewidths=0.8, linestyles="--")
#     elif "wind" in var_key:
#         ax.contour(lons, times, vals_abs, levels=[0], colors="k", linewidths=1.5)

#     ax.set_title(f"Hovmöller: {var_key.upper()} - {title_suffix}", fontsize=16, fontweight="bold")
#     ax.set_xlabel("Longitude", fontsize=12)
#     ax.set_ylabel("Time", fontsize=12)
#     ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
#     plt.colorbar(cf, ax=ax, label=f"Anomaly ({conf['unit_label']})",
#                  orientation="horizontal", pad=0.08)
#     plt.tight_layout()
#     plt.show()


# # ---------------------------------------------------------------------------
# # Map layout
# # ---------------------------------------------------------------------------

# def plot_maps(data_main, data_overlay, var_key, title_prefix="Monthly"):
#     """Render a 6-row map panel for the six preceding months.

#     Each row displays one month.  The rendering strategy is controlled by
#     ``VAR_CONFIG[var_key]["plot_type"]``:

#     - ``"anomaly_only"`` — filled contours of the anomaly (SST).
#     - ``"contour_overlay"`` — filled anomaly + absolute contour lines with
#       optional highlighted isopleths (SLP, OLR).
#     - ``"ratio"`` — precipitation ratio relative to climatology.
#     - ``"vector"`` — quiver plot of wind vectors.
#     - ``"streamlines"`` / ``"stream_humidity"`` — streamlines with specific
#       humidity (850 hPa) or wind speed (surface) as the colour background.

#     Parameters
#     ----------
#     data_main : xarray.DataArray or xarray.Dataset
#         Primary data field (anomaly, ratio, or absolute for vector fields).
#     data_overlay : xarray.DataArray or None
#         Absolute field used for overlaid contour lines; ``None`` for types
#         that do not need it (ratio, vector, streamlines).
#     var_key : str
#         Variable key defined in :data:`VAR_CONFIG`.
#     title_prefix : str, default ``"Monthly"``
#         String prepended to the figure super-title.
#     """
#     conf = VAR_CONFIG[var_key]
#     print(f"Generating {title_prefix} Maps for {var_key}...")

#     dates = data_main.time.values
#     n_plots = len(dates)

#     fig, axes = plt.subplots(6, 1, figsize=(12, 24),
#                              subplot_kw={"projection": ccrs.PlateCarree()})
#     if n_plots == 1:
#         axes = [axes]
#     axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else axes

#     plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05,
#                         hspace=0.2, wspace=0.0)

#     if n_plots < 6:
#         for i in range(n_plots, 6):
#             fig.delaxes(axes_flat[i])

#     plt.suptitle(f"{var_key.upper()} - {title_prefix} Analysis (6 Past Months)",
#                  fontsize=20, y=0.98, fontweight="bold")

#     last_cf = None
#     cbar_label_text = conf["unit_label"]

#     for i, (ax, dt) in enumerate(zip(axes_flat, dates)):
#         date_str = np.datetime_as_string(dt, unit="M")
#         ax.coastlines(linewidth=1.0)
#         ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.6)
#         ax.set_title(date_str, fontsize=12, fontweight="bold", loc="left")

#         if conf["plot_type"] == "anomaly_only":
#             val = data_main.isel(time=i)
#             last_cf = ax.contourf(val.longitude, val.latitude, val,
#                                   levels=conf["levels"], cmap=conf["cmap"],
#                                   extend="both", transform=ccrs.PlateCarree())

#         elif conf["plot_type"] == "contour_overlay":
#             val_anom = data_main.isel(time=i)
#             last_cf = ax.contourf(val_anom.longitude, val_anom.latitude, val_anom,
#                                   levels=conf["levels_anom"], cmap=conf["cmap"],
#                                   extend="both", transform=ccrs.PlateCarree())
#             val_abs = data_overlay.isel(time=i)
#             cs = ax.contour(val_abs.longitude, val_abs.latitude, val_abs,
#                             levels=conf["levels_contour"], colors="black",
#                             linewidths=0.6, transform=ccrs.PlateCarree())
#             ax.clabel(cs, inline=True, fontsize=8, fmt="%1.0f")
#             if "highlight_black" in conf:
#                 cs_blk = ax.contour(val_abs.longitude, val_abs.latitude, val_abs,
#                                     levels=conf["highlight_black"], colors="black",
#                                     linewidths=2.5, transform=ccrs.PlateCarree())
#                 ax.clabel(cs_blk, inline=True, fontsize=10, fmt="%1.0f")
#             if "highlight_green" in conf:
#                 cs_grn = ax.contour(val_abs.longitude, val_abs.latitude, val_abs,
#                                     levels=conf["highlight_green"], colors=["#008000"],
#                                     linewidths=2.5, transform=ccrs.PlateCarree())
#                 ax.clabel(cs_grn, inline=True, fontsize=10, fmt="%1.0f")

#         elif conf["plot_type"] == "ratio":
#             val = data_main.isel(time=i)
#             norm = mcolors.BoundaryNorm(conf["levels"], ncolors=256)
#             last_cf = ax.contourf(val.longitude, val.latitude, val,
#                                   levels=conf["levels"], norm=norm, cmap=conf["cmap"],
#                                   extend="max", transform=ccrs.PlateCarree())

#         elif conf["plot_type"] == "vector":
#             u_var = [v for v in data_main.data_vars if "u" in v.lower()][0]
#             v_var = [v for v in data_main.data_vars if "v" in v.lower()][0]
#             u = data_main[u_var].isel(time=i).values.squeeze()
#             v = data_main[v_var].isel(time=i).values.squeeze()
#             if u.ndim > 2:
#                 u = u[0]
#             if v.ndim > 2:
#                 v = v[0]
#             skip = 5
#             ax.quiver(data_main.longitude[::skip], data_main.latitude[::skip],
#                       u[::skip, ::skip], v[::skip, ::skip],
#                       transform=ccrs.PlateCarree(), scale=250, width=0.003)

#         elif conf["plot_type"] in ["streamlines", "stream_humidity"]:
#             u_name = [v for v in data_main.data_vars if "u" in v.lower()][0]
#             v_name = [v for v in data_main.data_vars if "v" in v.lower()][0]
#             q_name = [v for v in data_main.data_vars
#                       if "humid" in v.lower() or "q" in v.lower()]

#             u_val = data_main[u_name].isel(time=i).values.squeeze()
#             v_val = data_main[v_name].isel(time=i).values.squeeze()
#             if u_val.ndim > 2:
#                 u_val = u_val[0]
#             if v_val.ndim > 2:
#                 v_val = v_val[0]

#             if q_name:
#                 q_val = data_main[q_name[0]].isel(time=i).values.squeeze()
#                 if q_val.ndim > 2:
#                     q_val = q_val[0]
#                 bg_data = q_val * 1000  # convert kg/kg -> g/kg
#             else:
#                 bg_data = np.sqrt(u_val ** 2 + v_val ** 2)

#             last_cf = ax.contourf(data_main.longitude, data_main.latitude, bg_data,
#                                   levels=conf["levels"], cmap=conf["cmap"],
#                                   extend="max", transform=ccrs.PlateCarree())
#             ax.streamplot(data_main.longitude.values, data_main.latitude.values,
#                           u_val, v_val, transform=ccrs.PlateCarree(),
#                           density=1.0, color="k", linewidth=0.7, arrowsize=0.8)

#     if last_cf is not None:
#         cbar_ax = fig.add_axes([0.96, 0.15, 0.02, 0.7])
#         cbar = fig.colorbar(last_cf, cax=cbar_ax, orientation="vertical")
#         cbar.set_label(cbar_label_text, fontsize=14, fontweight="bold")
#         cbar.ax.tick_params(labelsize=12)

#     plt.show()


# # ---------------------------------------------------------------------------
# # Data processor
# # ---------------------------------------------------------------------------

# def process_variable(fpath, var_key, clim_start, clim_end, target_date):
#     """Load, pre-process, and plot one ERA5 variable.

#     Computes climatology from the reference period, selects the six months
#     preceding *target_date*, derives anomalies or ratios, and delegates to
#     :func:`plot_maps` and :func:`plot_hovmoller`.

#     Parameters
#     ----------
#     fpath : str
#         Path to the ERA5 NetCDF file (typically returned by :func:`download_data`).
#     var_key : str
#         Variable key defined in :data:`VAR_CONFIG`.
#     clim_start : int
#         First year of the climatological reference period.
#     clim_end : int
#         Last year of the climatological reference period.
#     target_date : datetime.date
#         Forecast target date.  The six months T−6 to T−1 are processed.
#     """
#     conf = VAR_CONFIG[var_key]
#     ds = xr.open_dataset(fpath)

#     if "valid_time" in ds.coords or "valid_time" in ds.dims:
#         ds = ds.rename({"valid_time": "time"})
#     if "expver" in ds.dims:
#         if ds.sizes["expver"] > 1:
#             try:
#                 ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))
#             except Exception:
#                 ds = ds.isel(expver=0, drop=True)
#         else:
#             ds = ds.squeeze("expver", drop=True)
#     if "pressure_level" in ds.dims:
#         ds = ds.squeeze("pressure_level", drop=True)

#     ds = ds.sortby("time")
#     if var_key != "wind_850":
#         ds = conf["unit_func"](ds)

#     ref_slice = slice(f"{clim_start}-01-01", f"{clim_end}-12-31")
#     start_date = target_date - relativedelta(months=6)
#     end_date = target_date - relativedelta(months=1)

#     is_vector = conf["plot_type"] in ["vector", "streamlines", "stream_humidity"]
#     if is_vector:
#         data_obj = ds
#     else:
#         var_name = list(ds.data_vars)[0]
#         data_obj = ds[var_name]

#     clim = data_obj.sel(time=ref_slice).groupby("time.month").mean("time")

#     recent_slice = slice(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
#     ds_recent = data_obj.sel(time=recent_slice)
#     anom_recent = ds_recent.groupby("time.month") - clim

#     if is_vector:
#         data_map_main = ds_recent
#         data_map_overlay = None
#         data_hov_anom = anom_recent
#         data_hov_abs = ds_recent
#     elif conf["plot_type"] == "ratio":
#         clim_subset = clim.sel(month=ds_recent["time.month"])
#         clim_subset = clim_subset.where(clim_subset > 1, 1.0)
#         data_map_main = (ds_recent / clim_subset) * 100
#         data_map_overlay = None
#         data_hov_anom = None
#         data_hov_abs = None
#     else:
#         data_map_main = anom_recent
#         data_map_overlay = ds_recent
#         data_hov_anom = anom_recent
#         data_hov_abs = ds_recent

#     plot_maps(data_map_main, data_map_overlay, var_key)
#     if data_hov_anom is not None:
#         plot_hovmoller(data_hov_anom, data_hov_abs, var_key)


# # ---------------------------------------------------------------------------
# # Entry point
# # ---------------------------------------------------------------------------

# def main_driver(dir_save, clim_start, clim_end, target_date_str, variables_list=None):
#     """Run the full seasonal diagnostic pipeline for a set of variables.

#     For each variable: download ERA5 data (if not already cached), compute
#     anomalies relative to the climatological period, and produce maps and
#     Hovmöller diagrams for the six months preceding the target date.

#     Parameters
#     ----------
#     dir_save : str
#         Root directory for cached NetCDF files.
#     clim_start : int
#         First year of the climatological reference period.
#     clim_end : int
#         Last year of the climatological reference period.
#     target_date_str : str
#         ISO-format target date string (``"YYYY-MM-DD"``).  Data for the six
#         months *before* this date are processed and visualised.
#     variables_list : list of str, optional
#         Subset of keys from :data:`VAR_CONFIG` to process.  Defaults to all
#         configured variables.
#     """
#     target_date = date.fromisoformat(target_date_str)
#     if variables_list is None:
#         variables_list = list(VAR_CONFIG.keys())

#     for var in variables_list:
#         if var not in VAR_CONFIG:
#             continue
#         print(f"\n Traitement: {var.upper()}")
#         extent = VAR_CONFIG[var]["extent"]
#         fpath = download_data(dir_save, clim_start, clim_end, var, extent, target_date)
#         if fpath:
#             process_variable(fpath, var, clim_start, clim_end, target_date)

#     print("\n Finished.")


# # ---------------------------------------------------------------------------
# # Interactive viewers
# # ---------------------------------------------------------------------------

# import ipywidgets as widgets
# from IPython.display import display, IFrame
# import datetime


# class C3SViewer:
#     """Interactive Jupyter widget for Copernicus C3S seasonal forecast maps.

#     Displays C3S probabilistic seasonal forecast products (SST, MSLP,
#     precipitation, temperature, wind, and Niño plumes) inside an IFrame,
#     with dropdown controls for product, start year/month, area, and map type.

#     Parameters
#     ----------
#     None

#     Examples
#     --------
#     >>> viewer = C3SViewer()
#     >>> viewer.show()
#     """

#     def __init__(self):
#         self.BASE_PACKAGE = (
#             "https://climate.copernicus.eu/charts/packages/c3s_seasonal/products/"
#         )

#         self.products = {
#             "Sea Surface Temperature (SST)": {"slug": "c3s_seasonal_spatial_mm_ssto_3m", "type": "map"},
#             "Mean Sea Level Pressure (MSLP)": {"slug": "c3s_seasonal_spatial_mm_mslp_3m", "type": "map"},
#             "Precipitation": {"slug": "c3s_seasonal_spatial_mm_rain_3m", "type": "map"},
#             "2m Temperature (T2m)": {"slug": "c3s_seasonal_spatial_mm_2mtm_3m", "type": "map"},
#             "10m Wind Speed": {"slug": "c3s_seasonal_spatial_mm_wspd_3m", "type": "map"},
#             "Nino Ensemble Plumes": {"slug": "c3s_seasonal_plume_mm", "type": "plume"},
#         }

#         self.areas_spatial = [
#             ("Global", "area08"), ("Europe", "area01"), ("Africa", "area02"),
#             ("North America", "area05"), ("South America", "area04"),
#             ("Asia", "area03"), ("Australasia", "area06"),
#         ]
#         self.areas_nino = [
#             ("Nino 3", "nino3"), ("Nino 3.4", "nino34"),
#             ("Nino 4", "nino4"), ("Nino 1+2", "nino12"),
#         ]

#         self.types_spatial = [
#             ("Tercile Summary", "tsum"), ("Prob(most likely)", "prob"), ("Ensemble Mean", "em"),
#         ]
#         self.types_nino = [("Plume", "plume")]

#         current_year = datetime.datetime.now().year
#         self.years = [str(y) for y in range(current_year - 2, current_year + 3)]
#         self.months = [
#             ("Jan", "01"), ("Feb", "02"), ("Mar", "03"), ("Apr", "04"),
#             ("May", "05"), ("Jun", "06"), ("Jul", "07"), ("Aug", "08"),
#             ("Sep", "09"), ("Oct", "10"), ("Nov", "11"), ("Dec", "12"),
#         ]

#         self.w_product = widgets.Dropdown(
#             options=self.products.keys(), value="Sea Surface Temperature (SST)", description="Product:"
#         )
#         self.w_year = widgets.Dropdown(
#             options=self.years, value=str(current_year), description="Start Year:"
#         )
#         self.w_month = widgets.Dropdown(options=self.months, value="01", description="Start Month:")
#         self.w_area = widgets.Dropdown(options=self.areas_spatial, value="area08", description="Area:")
#         self.w_type = widgets.Dropdown(options=self.types_spatial, value="tsum", description="Map Type:")
#         self.out_display = widgets.Output()

#         self.w_product.observe(self._on_product_change, names="value")
#         for w in [self.w_product, self.w_year, self.w_month, self.w_area, self.w_type]:
#             w.observe(self._update_chart, names="value")

#         self.ui = widgets.VBox([
#             self.w_product,
#             widgets.HBox([self.w_year, self.w_month, self.w_area, self.w_type]),
#         ])
#         self._update_chart()

#     def _on_product_change(self, change):
#         """Switch area and map-type options when the product changes.

#         Niño plume products use a different set of areas and a fixed map type.

#         Parameters
#         ----------
#         change : dict
#             ipywidgets observe change dictionary; only ``change['new']`` is used.
#         """
#         prod_info = self.products[change["new"]]
#         if prod_info["type"] == "plume":
#             self.w_area.options = self.areas_nino
#             self.w_area.value = "nino3"
#             self.w_type.options = self.types_nino
#             self.w_type.value = "plume"
#             self.w_type.disabled = True
#         else:
#             self.w_area.options = self.areas_spatial
#             self.w_area.value = "area08"
#             self.w_type.options = self.types_spatial
#             self.w_type.disabled = False

#     def _update_chart(self, change=None):
#         """Rebuild the C3S chart URL and refresh the embedded IFrame.

#         Parameters
#         ----------
#         change : dict or None
#             ipywidgets observe change dictionary (unused; present for
#             compatibility with the observe callback signature).
#         """
#         prod_name = self.w_product.value
#         prod_info = self.products[prod_name]
#         base_time = f"{self.w_year.value}{self.w_month.value}010000"

#         params = [
#             f"area={self.w_area.value}",
#             f"base_time={base_time}",
#             f"type={self.w_type.value}",
#         ]

#         if prod_info["type"] == "map":
#             dt_base = datetime.date(int(self.w_year.value), int(self.w_month.value), 1)
#             if dt_base.month == 12:
#                 dt_valid = datetime.date(dt_base.year + 1, 1, 1)
#             else:
#                 dt_valid = datetime.date(dt_base.year, dt_base.month + 1, 1)
#             valid_time = f"{dt_valid.year}{dt_valid.month:02d}010000"
#             params.append(f"valid_time={valid_time}")

#         full_url = f"{self.BASE_PACKAGE}{prod_info['slug']}?{'&'.join(params)}"
#         with self.out_display:
#             self.out_display.clear_output(wait=True)
#             display(IFrame(src=full_url, width="100%", height=850))

#     def show(self):
#         """Display the widget interface in the current Jupyter cell."""
#         display(self.ui, self.out_display)


# class BOMViewer:
#     """Interactive Jupyter widget for BOM MJO monitoring charts.

#     Embeds the Australian Bureau of Meteorology MJO page inside an IFrame
#     with a dropdown to navigate between tabs (Outlooks, MJO Phase Diagram,
#     Monitoring, Cloudiness, Regional Cloudiness, Time-Longitude, Tropical
#     Update).

#     Parameters
#     ----------
#     None

#     Examples
#     --------
#     >>> viewer = BOMViewer()
#     >>> viewer.show()
#     """

#     def __init__(self):
#         self.bom_tabs = [
#             ("Outlooks (Forecasts)", "Outlooks"),
#             ("MJO Phase Diagram", "MJO%20phase"),
#             ("Monitoring", "Monitoring"),
#             ("Cloudiness", "Cloudiness"),
#             ("Regional Cloudiness", "Regional%20cloudiness"),
#             ("Time-Longitude", "Time-longitude"),
#             ("Tropical Update", "Tropical%20update"),
#         ]

#         self.w_bom_tab = widgets.Dropdown(
#             options=self.bom_tabs,
#             value="Outlooks",
#             description="Select Tab:",
#             style={"description_width": "initial"},
#         )
#         self.out_bom = widgets.Output()

#         self.w_bom_tab.observe(self._update_bom, names="value")
#         self.ui = widgets.VBox([self.w_bom_tab, self.out_bom])
#         self._update_bom()

#     def _update_bom(self, change=None):
#         """Rebuild the BOM URL and refresh the embedded IFrame.

#         Parameters
#         ----------
#         change : dict or None
#             ipywidgets observe change dictionary (unused; present for
#             compatibility with the observe callback signature).
#         """
#         tab_slug = self.w_bom_tab.value
#         url = f"https://www.bom.gov.au/climate/mjo/#tabs={tab_slug}"
#         with self.out_bom:
#             self.out_bom.clear_output(wait=True)
#             display(IFrame(src=url, width="100%", height=1000))

#     def show(self):
#         """Display the widget interface in the current Jupyter cell."""
#         display(self.ui)
