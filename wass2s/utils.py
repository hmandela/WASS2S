"""Shared utility functions for the WASS2S pipeline.

Covers data loading, preprocessing, index computation, predictor retrieval,
and forecast visualisation.

Data utilities
--------------
decode_cf
    Decode CF-convention time coordinates.
fix_time_coord
    Re-assign a seasonal ``T`` coordinate to a dataset.
standardize_timeseries / reverse_standardize
    Climatological standardisation / de-standardisation of xarray datasets.
anomalize_timeseries
    Climatological anomaly computation.
detrended_data / apply_detrend_data
    Fold-safe linear detrending helpers.
predictant_mask
    Binary land/ocean mask derived from a predictand DataArray.
extract_leading_eeof_component
    Extract the leading extended-EOF component.

Index and predictor loaders
---------------------------
compute_sst_indices
    Compute standard SST indices (Niño 3.4, AMM, IOD, …) from gridded data.
compute_other_indices
    Compute arbitrary spatial-average indices over user-defined boxes.
retrieve_several_zones_for_PCR
    Build a multivariate predictor list (one field per zone) for PCR.
retrieve_single_zone_for_PCR
    Build a single standardized predictor DataArray for PCR.
load_gridded_predictor
    Load a gridded model or reanalysis predictor field.
prepare_predictand
    Load, merge, and optionally aggregate an observational predictand.

Network utilities
-----------------
download_file
    Stream a file from a URL with progress reporting and retry logic.
build_iridl_url_ersst / to_iridl_lat / to_iridl_lon
    Build IRI Data Library URLs for ERSSTv5.
parse_variable
    Map user-facing variable names to CDS / NMME field names.

Geo utilities
-------------
get_shapefile / get_shapefile_
    Retrieve country or sub-national shapefiles (Natural Earth / GADM /
    geoBoundaries).
plot_map
    Quick map of a geographic extent with optional SST index overlays.
get_best_models
    Select top-N forecast models by a given skill metric.

Visualisation
-------------
plot_prob_forecasts / plot_prob_forecastsAlpha
    Tercile probability forecast map panels.
"""
from __future__ import annotations

# =============================================================================
# Standard library
# =============================================================================
import os
import io
import zipfile
import calendar
from pathlib import Path
from datetime import timedelta
from typing import Optional, Literal, Tuple, List, Dict, Any

# =============================================================================
# Core scientific stack
# =============================================================================
import numpy as np
import pandas as pd
import xarray as xr

# =============================================================================
# Networking / IO
# =============================================================================
import requests
import urllib3
import cdsapi

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# =============================================================================
# Plotting / mapping
# =============================================================================
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib import gridspec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.colors import ListedColormap, BoundaryNorm
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Keep mpimg optional if you want to be robust on minimal installs
try:
    import matplotlib.image as mpimg
except Exception:
    mpimg = None

# =============================================================================
# Spatial / geospatial
# =============================================================================
import rioxarray as rioxr
from scipy.ndimage import gaussian_filter

# Optional geospatial libs
try:
    import geopandas as gpd
    from cartopy.mpl.path import shapely_to_path
except Exception:
    gpd = None
    shapely_to_path = None

try:
    import regionmask
except Exception:
    regionmask = None

# =============================================================================
# ML / stats utilities
# =============================================================================
import xeofs as xe
from fitter import Fitter
from tqdm import tqdm

# =============================================================================
# Project imports
# =============================================================================
from wass2s.was_compute_predictand import *
from wass2s.was_bias_correction import *

from rasterio.features import rasterize
from rasterio.transform import from_bounds


def decode_cf(ds, time_var):
    """
    Decode time dimension to CFTime standards.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing the time variable to decode.
    time_var : str
        Name of the time variable in the dataset.

    Returns
    -------
    xarray.Dataset
        Dataset with decoded time dimension adhering to CFTime standards.

    Notes
    -----
    If the calendar attribute of the time variable is '360', it is converted to '360_day'
    to ensure compatibility with CFTime decoding.
    """
    if ds[time_var].attrs["calendar"] == "360":
        ds[time_var].attrs["calendar"] = "360_day"
    ds = xr.decode_cf(ds, decode_times=True)
    return ds

def to_iridl_lat(lat: float) -> str:
    """
    Convert numeric latitude to IRIDL latitude string format.

    Parameters
    ----------
    lat : float
        Latitude value in degrees (positive for North, negative for South).

    Returns
    -------
    str
        IRIDL-formatted latitude string (e.g., '10N' for +10, '5S' for -5).

    Examples
    --------
    >>> to_iridl_lat(10)
    '10N'
    >>> to_iridl_lat(-5)
    '5S'
    """
    abs_val = abs(lat)
    suffix = "N" if lat >= 0 else "S"
    return f"{abs_val:g}{suffix}"

def to_iridl_lon(lon: float) -> str:
    """
    Convert numeric longitude to IRIDL longitude string format.

    Parameters
    ----------
    lon : float
        Longitude value in degrees (positive for East, negative for West).

    Returns
    -------
    str
        IRIDL-formatted longitude string (e.g., '15E' for +15, '15W' for -15).

    Examples
    --------
    >>> to_iridl_lon(15)
    '15E'
    >>> to_iridl_lon(-15)
    '15W'
    """
    abs_val = abs(lon)
    suffix = "E" if lon >= 0 else "W"
    return f"{abs_val:g}{suffix}"

def build_iridl_url_ersst(
    year_start: int,
    year_end: int,
    bbox: list,
    run_avg: int = 3,
    month_start: str = "Jan",
    month_end: str = "Dec",
):
    """
    Build a parameterized IRIDL URL for NOAA ERSST dataset.

    Parameters
    ----------
    year_start : int
        Start year for the data request.
    year_end : int
        End year for the data request.
    bbox : list
        Bounding box coordinates in the format [North, West, South, East].
    run_avg : int, optional
        Number of time steps for running average (default is 3). If None, no running average is applied.
    month_start : str, optional
        Starting month for the time range (default is 'Jan').
    month_end : str, optional
        Ending month for the time range (default is 'Dec').

    Returns
    -------
    str
        Constructed IRIDL URL for accessing NOAA ERSST data.

    Notes
    -----
    The bounding box is reordered to match IRIDL's expected format: Y/(south)/(north)/, X/(west)/(east)/.
    """
    north, w, south, e = bbox
    south_str = to_iridl_lat(south)
    north_str = to_iridl_lat(north)
    west_str = to_iridl_lon(w)
    east_str = to_iridl_lon(e)
    t_start_str = f"{month_start}%20{year_start}"
    t_end_str = f"{month_end}%20{year_end}"
    time_part = f"T/({t_start_str})/({t_end_str})/RANGEEDGES/"
    latlon_part = (
        f"Y/({south_str})/({north_str})/RANGEEDGES/"
        f"X/({west_str})/({east_str})/RANGEEDGES/"
    )
    runavg_part = f"T/{run_avg}/runningAverage/" if run_avg is not None else ""
    url = (
        "https://iridl.ldeo.columbia.edu/"
        "SOURCES/.NOAA/.NCDC/.ERSST/.version5/.sst/"
        f"{time_part}"
        f"{latlon_part}"
        f"{runavg_part}"
        "dods"
    )
    return url

def fix_time_coord(ds, seas):
    """
    Fix time coordinates by extracting year and assigning a fixed month/day.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset with time dimension 'T'.
    seas : tuple
        Tuple containing season information, where seas[1] is the month (e.g., '11' for November).

    Returns
    -------
    xarray.Dataset
        Dataset with updated time coordinates in the format 'YYYY-MM-01'.

    Notes
    -----
    Assumes the input dataset's time values can be converted to pandas datetime objects.
    """
    years = pd.to_datetime(ds.T.values).year
    new_dates = [np.datetime64(f"{y}-{seas[1]}-01") for y in years]
    ds = ds.assign_coords(T=("T", new_dates))
    ds["T"] = ds["T"].astype("datetime64[ns]")
    return ds

def download_file(url, local_path, force_download=False, chunk_size=8192, timeout=120):
    """
    Download a file from a URL to a local path with progress tracking.

    Parameters
    ----------
    url : str
        URL of the file to download.
    local_path : str or pathlib.Path
        Local path where the file will be saved.
    force_download : bool, optional
        If True, overwrite existing file (default is False).
    chunk_size : int, optional
        Size of chunks for streaming download (default is 8192 bytes).
    timeout : int, optional
        Timeout for the HTTP request in seconds (default is 120).

    Returns
    -------
    pathlib.Path or None
        Path to the downloaded file, or None if download fails.

    Notes
    -----
    Skips download if the file already exists and `force_download` is False.
    """
    local_path = Path(local_path)
    if local_path.exists() and not force_download:
        print(f"[SKIP] {local_path} already exists.")
        return local_path
    print(f"[DOWNLOAD] {url}")
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(local_path, "wb") as f, tqdm(
                total=total_size, unit="B", unit_scale=True, unit_divisor=1024
            ) as progress:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        progress.update(len(chunk))
        print(f"[SUCCESS] Downloaded to {local_path}")
        return local_path
    except Exception as e:
        print(f"[ERROR] Could not download {url}: {e}")
        return None

def parse_variable(variables_list):
    """
    Extract center and variable names from a variable string.

    Parameters
    ----------
    variables_list : str
        Variable string in the format 'center.variable'.

    Returns
    -------
    tuple
        A tuple containing (center, variable).

    Examples
    --------
    >>> parse_variable("NOAA.SST")
    ('NOAA', 'SST')
    """
    center = variables_list.split(".")[0]
    variable = variables_list.split(".")[1]
    return center, variable

def standardize_timeseries(ds, clim_year_start=None, clim_year_end=None):
    """
    Standardize a dataset over a specified climatology period.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Input dataset or data array to standardize.
    clim_year_start : int, optional
        Start year of the climatology period. If None, uses the full time range.
    clim_year_end : int, optional
        End year of the climatology period. If None, uses the full time range.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Standardized dataset or data array (z-scores).

    Notes
    -----
    Standardization is performed as (ds - mean) / std, where mean and std are
    computed over the specified climatology period or the entire time dimension.
    """
    if clim_year_start is not None and clim_year_end is not None:
        clim_slice = slice(str(clim_year_start), str(clim_year_end))
        clim_mean = ds.sel(T=clim_slice).mean(dim='T')
        clim_std = ds.sel(T=clim_slice).std(dim='T')
    else:
        clim_mean = ds.mean(dim='T')
        clim_std = ds.std(dim='T')
    return (ds - clim_mean) / clim_std

def reverse_standardize(ds_st, ds, clim_year_start=None, clim_year_end=None):
    """
    Reverse standardization of a dataset to original units.

    Parameters
    ----------
    ds_st : xarray.Dataset or xarray.DataArray
        Standardized dataset or data array (z-scores).
    ds : xarray.Dataset or xarray.DataArray
        Original dataset used to compute standardization parameters.
    clim_year_start : int, optional
        Start year of the climatology period. If None, uses the full time range.
    clim_year_end : int, optional
        End year of the climatology period. If None, uses the full time range.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Dataset or data array in original units.

    Notes
    -----
    Reverses standardization using ds_st * std + mean, where mean and std are
    computed from the original dataset over the specified period.
    """
    if clim_year_start is not None and clim_year_end is not None:
        clim_slice = slice(str(clim_year_start), str(clim_year_end))
        clim_mean = ds.sel(T=clim_slice).mean(dim='T')
        clim_std = ds.sel(T=clim_slice).std(dim='T')
    else:
        clim_mean = ds.mean(dim='T')
        clim_std = ds.std(dim='T')
    return ds_st * clim_std + clim_mean

def anomalize_timeseries(ds, clim_year_start=None, clim_year_end=None):
    """
    Compute anomalies by subtracting the climatological mean.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Input dataset or data array.
    clim_year_start : int, optional
        Start year of the climatology period. If None, uses the full time range.
    clim_year_end : int, optional
        End year of the climatology period. If None, uses the full time range.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Anomalized dataset or data array (ds - mean).

    Notes
    -----
    Anomalies are computed as ds - mean, where mean is calculated over the
    specified climatology period or the entire time dimension.
    """
    if clim_year_start is not None and clim_year_end is not None:
        clim_slice = slice(str(clim_year_start), str(clim_year_end))
        clim_mean = ds.sel(T=clim_slice).mean(dim='T')
    else:
        clim_mean = ds.mean(dim='T')
    return ds - clim_mean

def predictant_mask(data):
    """
    Create a mask for predictand data based on rainfall thresholds and latitude.

    Parameters
    ----------
    data : xarray.DataArray
        Input data array with a time dimension 'T' and spatial dimensions 'Y', 'X'.

    Returns
    -------
    xarray.DataArray
        Mask with 1 where conditions are met, NaN elsewhere.

    Notes
    -----
    The mask is set to 1 where mean rainfall > 20 mm and latitude is within ±19.5 degrees,
    otherwise NaN.
    """
    mean_rainfall = data.mean(dim="T").squeeze()
    mask = xr.where(mean_rainfall <= 20, np.nan, 1)
    mask = mask.where(abs(mask.Y) <= 19.5, np.nan)
    return mask

def extract_leading_eeof_component(data):
    """
    extract leading eeof component using ExtendedEOF.

    Parameters
    ----------
    data : xarray.DataArray
        Input data array with time dimension 'T'.

    Returns
    -------
    xarray.DataArray
        Trended data array based on the first mode of ExtendedEOF.

    Raises
    ------
    RuntimeError
        If ExtendedEOF fitting fails.

    Notes
    -----
    Missing values are filled with the time mean before fitting.
    Uses ExtendedEOF with specified parameters (n_modes=2, tau=1, embedding=3, n_pca_modes=20).
    """
    try:
        data_filled = data.fillna(data.mean(dim="T", skipna=True))
        eeof = xe.single.ExtendedEOF(n_modes=2, tau=1, embedding=3, n_pca_modes=22)
        eeof.fit(data_filled, dim="T")
        scores_ext = eeof.scores()
        data_trends = eeof.inverse_transform(scores_ext.sel(mode=1))
        return data_trends
    except Exception as e:
        raise RuntimeError(f"Failed to detrend data using ExtendedEOF: {e}")

def detrended_data(da, dim="T", min_valid=10):
    if dim not in da.dims:
        raise ValueError(f"Dimension '{dim}' not found in DataArray.")

    x = da[dim]

    if np.issubdtype(x.dtype, np.datetime64):
        x0 = x.isel({dim: 0}).values
        x_days = (x - x.isel({dim: 0})).astype("timedelta64[D]").astype(np.float64)
        x_type = "datetime"
    else:
        x0 = float(x.isel({dim: 0}).values) if x.size else 0.0
        x_days = x.astype(np.float64)
        x_type = "numeric"

    ok = da.notnull().sum(dim=dim) >= int(min_valid)
    da_fit = da.where(ok)

    try:
        coeffs = da_fit.assign_coords({dim: x_days}).polyfit(dim=dim, deg=1, skipna=True)
    except TypeError:
        coeffs = da_fit.assign_coords({dim: x_days}).polyfit(dim=dim, deg=1)

    if not isinstance(x_days, xr.DataArray):
         x_days = xr.DataArray(x_days, dims=dim, coords={dim: da[dim]})

    trend = xr.polyval(x_days, coeffs.polyfit_coefficients)
    da_detrended = da - trend
    da_detrended.attrs = da.attrs.copy()

    meta = {
        "dim": dim,
        "x0": x0,
        "x_units": "days",
        "type": x_type,
        "min_valid": int(min_valid),
    }
    return da_detrended, coeffs, meta

def apply_detrend_data(da, coeffs, meta):
    dim = meta.get("dim", "T")
    if dim not in da.dims:
        if "T" in da.dims:
            dim = "T"
        elif "time" in da.dims:
            dim = "time"
        else:
            raise ValueError(f"Cannot find time dim. Expected '{meta.get('dim')}'. Found: {da.dims}")

    x = da[dim]

    if meta.get("type") == "datetime":
        x0 = np.datetime64(meta["x0"])
        x_days = (x - x0).astype("timedelta64[D]").astype(np.float64)
    else:
        x_days = x.astype(np.float64)

    if not isinstance(x_days, xr.DataArray):
         x_days = xr.DataArray(x_days, dims=dim, coords={dim: da[dim]})

    trend = xr.polyval(x_days, coeffs.polyfit_coefficients)
    return trend

def prepare_predictand(dir_to_save_Obs, variables_obs, year_start, year_end, season=None, ds=True, daily=False, param="prcp"):
    """
    Prepare the predictand dataset from observational data.
    """
    _, variable = parse_variable(variables_obs[0])
    
    # 1. Determine Filepath
    if daily:
        filepath = f'{dir_to_save_Obs}/Daily_{variable}_{year_start}_{year_end}.nc'
    else:
        season_str = "".join([calendar.month_abbr[int(month)] for month in season])
        filepath = f'{dir_to_save_Obs}/Obs_{variable}_{year_start}_{year_end}_{season_str}.nc'
    
    data = xr.open_dataset(filepath)

    # 2. Conditional Masking: Only apply to Precipitation
    if param == "prcp":
        if daily:
            # Basic daily thresholding
            data = xr.where(data < 0.1, 0, data)
        else:
            # Seasonal masking: Mean rainfall > 20 and Latitude (Y) <= 20
            mean_val = data.mean(dim="T").to_array().squeeze()
            mask = xr.where((mean_val > 20) & (abs(data.Y) <= 20), 1, np.nan)
            data = data.where(mask == 1)
    
    # Otherwise, if param is "temp" or others, it skips the masking above.

    # 3. Format Time Coordinate
    if 'T' in data.coords:
        data['T'] = data['T'].astype('datetime64[ns]')

    # 4. Final Formatting & Return
    data_out = data.squeeze().transpose('T', 'Y', 'X').sortby("T")
    
    if ds:
        return data_out
    else:
        # Renames the variable to your 'param' string (e.g., 'temp')
        return data_out.to_array().drop_vars("variable").squeeze().rename(param)


def load_gridded_predictor(dir_to_data, variables_list, year_start, year_end, season=None, model=False, month_of_initialization=None, lead_time=None, year_forecast=None):
    """
    Load gridded predictor data for reanalysis or model.

    Parameters
    ----------
    dir_to_data : str
        Directory path where predictor data is stored.
    variables_list : str
        Variable string in the format 'center.variable'.
    year_start : int
        Start year of the data.
    year_end : int
        End year of the data.
    season : list, optional
        List of month numbers defining the season.
    model : bool, optional
        If True, load model data (hindcast/forecast); otherwise, load reanalysis data (default is False).
    month_of_initialization : int, optional
        Month of model initialization (required if model=True).
    lead_time : list, optional
        List of lead times in months (required if model=True).
    year_forecast : int, optional
        Forecast year (required for forecast data if model=True).

    Returns
    -------
    xarray.DataArray
        Loaded predictor data array renamed to 'predictor'.

    Notes
    -----
    Converts dataset to data array, drops 'variable' dimension, and ensures datetime64[ns] time coordinates.
    """
    center, variable = parse_variable(variables_list)
    if model:
        abb_month_ini = calendar.month_abbr[int(month_of_initialization)]
        season_str = "".join([calendar.month_abbr[(int(i) + int(month_of_initialization)) % 12 or 12] for i in lead_time])
        center = center.lower().replace("_", "")
        file_prefix = "forecast" if year_forecast else "hindcast"
        filepath = f"{dir_to_data}/{file_prefix}_{center}_{variable}_{abb_month_ini}Ic_{season_str}_{lead_time[0]}.nc"
    else:
        season_str = "".join([calendar.month_abbr[int(month)] for month in season])
        filepath = f'{dir_to_data}/{center}_{variable}_{year_start}_{year_end}_{season_str}.nc'
    predictor = xr.open_dataset(filepath)
    predictor['T'] = predictor['T'].astype('datetime64[ns]')
    return predictor.to_array().drop_vars("variable").squeeze("variable").rename("predictor").transpose('T', 'Y', 'X')

def compute_sst_indices(dir_to_data, indices, variables_list, year_start, year_end, season, clim_year_start=None, clim_year_end=None, others_zone=None, model=False, month_of_initialization=None, lead_time=None, year_forecast=None):
    """
    Compute Sea Surface Temperature (SST) indices for reanalysis or model data.

    Parameters
    ----------
    dir_to_data : str
        Directory path where data is stored.
    indices : list
        List of SST indices to compute (e.g., ['NINO34', 'TNA']).
    variables_list : str
        Variable string in the format 'center.variable'.
    year_start : int
        Start year of the data.
    year_end : int
        End year of the data.
    season : list
        List of month numbers defining the season.
    clim_year_start : int, optional
        Start year for climatology period.
    clim_year_end : int, optional
        End year for climatology period.
    others_zone : dict, optional
        Additional custom zones for SST indices with coordinates.
    model : bool, optional
        If True, load model data; otherwise, load reanalysis data (default is False).
    month_of_initialization : int, optional
        Month of model initialization.
    lead_time : list, optional
        List of lead times in months.
    year_forecast : int, optional
        Forecast year.

    Returns
    -------
    xarray.Dataset
        Dataset containing computed SST indices as variables.

    Notes
    -----
    Derived indices like 'TASI' and 'DMI' are computed as differences of other indices.
    """
    center, variable = parse_variable(variables_list)
    print(center, variable)
    if model:
        abb_month_ini = calendar.month_abbr[int(month_of_initialization)]
        season_str = "".join([calendar.month_abbr[(int(i) + int(month_of_initialization)) % 12 or 12] for i in lead_time])
        center = center.lower().replace("_", "")
        file_prefix = "forecast" if year_forecast else "hindcast"
        filepath = f"{dir_to_data}/{file_prefix}_{center}_{variable}_{abb_month_ini}Ic_{season_str}_{lead_time[0]}.nc"
    else:
        season_str = "".join([calendar.month_abbr[int(month)] for month in season])
        filepath = f'{dir_to_data}/{center}_{variable}_{year_start}_{year_end}_{season_str}.nc'
    sst = xr.open_dataset(filepath)
    sst['T'] = pd.to_datetime(sst['T'].values)
    predictor = {}
    for idx in sst_indices_name.keys():
        if idx in ["TASI", "DMI"]:
            continue
        _, lon_min, lon_max, lat_min, lat_max = sst_indices_name[idx]
        sst_region = sst.sel(X=slice(lon_min, lon_max), Y=slice(lat_min, lat_max)).mean(dim=["X", "Y"], skipna=True)
        sst_region = standardize_timeseries(sst_region, clim_year_start, clim_year_end)
        predictor[idx] = sst_region
    if others_zone is not None:
        indices = indices + list(others_zone.keys())
        for idx, coords in others_zone.items():
            _, lon_min, lon_max, lat_min, lat_max = coords
            sst_region = sst.sel(X=slice(lon_min, lon_max), Y=slice(lat_min, lat_max)).mean(dim=["X", "Y"])
            sst_region = standardize_timeseries(sst_region, clim_year_start, clim_year_end)
            predictor[idx] = sst_region
    predictor["TASI"] = predictor["NAT"] - predictor["SAT"]
    predictor["DMI"] = predictor["WTIO"] - predictor["SETIO"]
    selected_indices = {i: predictor[i] for i in indices}
    data_vars = {key: ds[variable.lower()].rename(key) for key, ds in selected_indices.items()}
    combined_dataset = xr.Dataset(data_vars)
    return combined_dataset

def compute_other_indices(dir_to_data, indices_dict, variables_list, year_start, year_end, season, clim_year_start=None, clim_year_end=None, model=False, month_of_initialization=None, lead_time=None, year_forecast=None):
    """
    Compute indices for non-SST variables.

    Parameters
    ----------
    dir_to_data : str
        Directory path where data is stored.
    indices_dict : dict
        Dictionary mapping index names to coordinates (label, lon_min, lon_max, lat_min, lat_max).
    variables_list : str
        Variable string in the format 'center.variable'.
    year_start : int
        Start year of the data.
    year_end : int
        End year of the data.
    season : list
        List of month numbers defining the season.
    clim_year_start : int, optional
        Start year for climatology period.
    clim_year_end : int, optional
        End year for climatology period.
    model : bool, optional
        If True, load model data; otherwise, load reanalysis data (default is False).
    month_of_initialization : int, optional
        Month of model initialization.
    lead_time : list, optional
        List of lead times in months.
    year_forecast : int, optional
        Forecast year.

    Returns
    -------
    xarray.Dataset
        Dataset containing computed indices as variables.
    """
    center, variable = parse_variable(variables_list)
    if model:
        abb_month_ini = calendar.month_abbr[int(month_of_initialization)]
        season_str = "".join([calendar.month_abbr[(int(i) + int(month_of_initialization)) % 12 or 12] for i in lead_time])
        center = center.lower().replace("_", "")
        file_prefix = "forecast" if year_forecast else "hindcast"
        filepath = f"{dir_to_data}/{file_prefix}_{center}_{variable}_{abb_month_ini}Ic_{season_str}_{lead_time[0]}.nc"
    else:
        season_str = "".join([calendar.month_abbr[int(month)] for month in season])
        filepath = f'{dir_to_data}/{center}_{variable}_{year_start}_{year_end}_{season_str}.nc'
    data = xr.open_dataset(filepath).to_array().drop_vars('variable').squeeze()
    data['T'] = pd.to_datetime(data['T'].values)
    predictor = {}
    for idx, coords in indices_dict.items():
        _, lon_min, lon_max, lat_min, lat_max = coords
        var_region = data.sel(X=slice(lon_min, lon_max), Y=slice(lat_min, lat_max)).mean(dim=["X", "Y"])
        var_region = standardize_timeseries(var_region, clim_year_start, clim_year_end)
        predictor[idx] = var_region
    data_vars = {key: ds.rename(key) for key, ds in predictor.items()}
    combined_dataset = xr.Dataset(data_vars)
    return combined_dataset

def retrieve_several_zones_for_PCR(dir_to_data, indices_dict, variables_list, year_start, year_end, season, clim_year_start=None, clim_year_end=None, model=False, month_of_initialization=None, lead_time=None, year_forecast=None):
    """
    Retrieve data for multiple zones for Principal Component Regression (PCR).

    Parameters
    ----------
    dir_to_data : str
        Directory path where data is stored.
    indices_dict : dict
        Dictionary mapping index names to coordinates (label, lon_min, lon_max, lat_min, lat_max).
    variables_list : str
        Variable string in the format 'center.variable'.
    year_start : int
        Start year of the data.
    year_end : int
        End year of the data.
    season : list
        List of month numbers defining the season.
    clim_year_start : int, optional
        Start year for climatology period.
    clim_year_end : int, optional
        End year for climatology period.
    model : bool, optional
        If True, load model data (hindcast and forecast); otherwise, load reanalysis data (default is False).
    month_of_initialization : int, optional
        Month of model initialization.
    lead_time : list, optional
        List of lead times in months.
    year_forecast : int, optional
        Forecast year.

    Returns
    -------
    list
        List of xarray.DataArray objects, each representing a standardized region.
    """
    center, variable = parse_variable(variables_list)
    if model:
        abb_month_ini = calendar.month_abbr[int(month_of_initialization)]
        season_str = "".join([calendar.month_abbr[(int(i) + int(month_of_initialization)) % 12 or 12] for i in lead_time])
        center = center.lower().replace("_", "")
        filepath_hdcst = f"{dir_to_data}/hindcast_{center}_{variable}_{abb_month_ini}Ic_{season_str}_{lead_time[0]}.nc"
        filepath_fcst = f"{dir_to_data}/forecast_{center}_{variable}_{abb_month_ini}Ic_{season_str}_{lead_time[0]}.nc"
        data_hdcst = xr.open_dataset(filepath_hdcst).to_array().drop_vars('variable').squeeze('variable')
        data_hdcst['T'] = data_hdcst['T'].astype('datetime64[ns]')
        data_fcst = xr.open_dataset(filepath_fcst).to_array().drop_vars('variable').squeeze('variable')
        data_fcst['T'] = data_fcst['T'].astype('datetime64[ns]')
        data = xr.concat([data_hdcst, data_fcst], dim='T')
    else:
        season_str = "".join([calendar.month_abbr[int(month)] for month in season])
        filepath = f'{dir_to_data}/{center}_{variable}_{year_start}_{year_end}_{season_str}.nc'
        data = xr.open_dataset(filepath).to_array().drop_vars('variable').squeeze('variable')
        data['T'] = data['T'].astype('datetime64[ns]')
    predictor = {}
    for idx, coords in indices_dict.items():
        _, lon_min, lon_max, lat_min, lat_max = coords
        var_region = data.sel(X=slice(lon_min, lon_max), Y=slice(lat_min, lat_max))
        var_region = standardize_timeseries(var_region, clim_year_start, clim_year_end)
        predictor[idx] = var_region
    data_vars = [ds.rename(key) for key, ds in predictor.items()]
    return data_vars

def retrieve_single_zone_for_PCR(dir_to_data, indices_dict, variables_list, year_start, year_end, season=None, clim_year_start=None, clim_year_end=None, model=False, month_of_initialization=None, lead_time=None, year_forecast=None, standardize=True):
    """
    Retrieve data for a single zone for Principal Component Regression (PCR) with interpolation.
skill_mask
    Parameters
    ----------
    dir_to_data : str
        Directory path where data is stored.
    indices_dict : dict
        Dictionary mapping a single index name to coordinates (label, lon_min, lon_max, lat_min, lat_max).
    variables_list : str
        Variable string in the format 'center.variable'.
    year_start : int
        Start year of the data.
    year_end : int
        End year of the data.
    season : list, optional
        List of month numbers defining the season.
    clim_year_start : int, optional
        Start year for climatology period.
    clim_year_end : int, optional
        End year for climatology period.
    model : bool, optional
        If True, load model data (hindcast and forecast); otherwise, load reanalysis data (default is False).
    month_of_initialization : int, optional
        Month of model initialization.
    lead_time : list, optional
        List of lead times in months.
    year_forecast : int, optional
        Forecast year.

    Returns
    -------
    xarray.DataArray
        Standardized and interpolated data array for the specified region.
    """
    center, variable = parse_variable(variables_list)
    if model:
        abb_month_ini = calendar.month_abbr[int(month_of_initialization)]
        center = center.lower().replace("_", "")
        
        if lead_time is None:
            filepath_hdcst = f"{dir_to_data}/hindcast_{center}_{variable}_{abb_month_ini}Ic.nc"
            ### A revoir
            filepath_fcst = f"{dir_to_data}/forecast_{center}_{variable}_{abb_month_ini}Ic.nc"
        else:
            season_str = "".join([calendar.month_abbr[(int(i) + int(month_of_initialization)) % 12 or 12] for i in lead_time])
            filepath_hdcst = f"{dir_to_data}/hindcast_{center}_{variable}_{abb_month_ini}Ic_{season_str}_{lead_time[0]}.nc"
            ### A revoir
            filepath_fcst = f"{dir_to_data}/forecast_{center}_{variable}_{abb_month_ini}Ic_{season_str}_{lead_time[0]}.nc"
            
        data_hdcst = xr.open_dataset(filepath_hdcst).to_array().drop_vars('variable').squeeze('variable')
        data_hdcst['T'] = data_hdcst['T'].astype('datetime64[ns]')
        ### A revoir
        data_fcst = xr.open_dataset(filepath_fcst).to_array().drop_vars('variable').squeeze('variable')
        data_fcst['T'] = data_fcst['T'].astype('datetime64[ns]')
        data = xr.concat([data_hdcst, data_fcst], dim='T')
    else:
        season_str = "".join([calendar.month_abbr[int(month)] for month in season])
        filepath = f'{dir_to_data}/{center}_{variable}_{year_start}_{year_end}_{season_str}.nc'
        data = xr.open_dataset(filepath).to_array().drop_vars('variable').squeeze('variable')
        data['T'] = data['T'].astype('datetime64[ns]')

    new_resolution = {'Y': 1, 'X': 1}

    if standardize:
        data = standardize_timeseries(data, clim_year_start, clim_year_end)
    else:
        data = anomalize_timeseries(data, clim_year_start, clim_year_end)

    for idx, coords in indices_dict.items():
        _, lon_min, lon_max, lat_min, lat_max = coords
        var_region = data.sel(X=slice(lon_min, lon_max), Y=slice(lat_min, lat_max))
        Y_new = xr.DataArray(np.arange(lat_min, lat_max + 1, new_resolution['Y']), dims='Y')
        X_new = xr.DataArray(np.arange(lon_min, lon_max + 1, new_resolution['X']), dims='X')
        data_vars = var_region.interp(Y=Y_new, X=X_new, method='linear')
    return data_vars

def plot_map(extent, title="Map", sst_indices=None, fig_size=(10, 8)):
    """
    Plot a map with specified geographic extent and optional SST index boxes.

    Parameters
    ----------
    extent : list
        Geographic extent in the format [west, east, south, north].
    title : str, optional
        Title of the map (default is 'Map').
    sst_indices : dict, optional
        Dictionary containing SST index information with keys as index names and
        values as tuples (label, lon_w, lon_e, lat_s, lat_n).
    fig_size : tuple, optional
        Figure size as (width, height) in inches (default is (10, 8)).

    Notes
    -----
    Uses Cartopy for map projection and Matplotlib for plotting.
    """
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=fig_size)
    ax.set_extent(extent)
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAND, edgecolor="black")
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    if sst_indices:
        for index, (label, lon_w, lon_e, lat_s, lat_n) in sst_indices.items():
            if lon_w is not None:
                ax.add_patch(Rectangle(
                    (lon_w, lat_s), lon_e - lon_w, lat_n - lat_s,
                    linewidth=2, edgecolor='red', facecolor='none', linestyle='--'))
                ax.text(lon_w + 1, lat_s + 1, index, color='red', fontsize=10, ha='left')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def get_best_models(center_variable, scores, metric='MAE', threshold=None, top_n=6, gcm=False, agroparam=False, hydro=False):

    # 1. Provide default thresholds if none given
    if threshold is None:
        if metric.lower() == 'mae':
            threshold = 500
        elif metric.lower() == 'pearson':
            threshold = 0.3
        elif metric.lower() == 'groc':
            threshold = 0.5
        else:
            ### To complete
            threshold = threshold  # or any other default you prefer
    
    # 2. Check if the given metric is in scores
    metric_key = metric  # for direct indexing
    if metric_key not in scores:
        raise ValueError(f"Metric '{metric_key}' not found in scores dictionary.")
    
    metric_data = scores[metric_key]  # e.g., scores["MAE"] or scores["Pearson"]
    
    # 3. Decide the comparison operator based on the metric
    #    (MAE typically: < threshold; Pearson typically: > threshold)
    if metric.lower() == 'mae':
        cmp_operator = 'lt'  # less than
    elif metric.lower() == 'pearson':
        cmp_operator = 'gt'  # greater than
    elif metric.lower() == 'groc':
        cmp_operator = 'gt'  # greater than
    else:
        cmp_operator = 'lt' 
    
    # 4. Compute the counts
    best_models = {}
    for model_name, da in metric_data.items():
        # Compare against threshold
        if cmp_operator == 'lt':
            arr_count = xr.where(da < threshold, 1, 0).sum(dim=["X","Y"], skipna=True).item()
        elif cmp_operator == 'gt':
            arr_count = xr.where(da > threshold, 1, 0).sum(dim=["X","Y"], skipna=True).item()
        else:
            # If needed, add more operators (<=, >=, etc.)
            arr_count = 0
        
        best_models[model_name] = arr_count
    
    # 5. Sort by descending count
    best_models = dict(sorted(best_models.items(), key=lambda item: item[1], reverse=True))

    # 6. Take the top N
    top_n_models = dict(list(best_models.items())[:top_n])
  
    # Normalize a variable name by removing ".suffix", removing underscores, and lowercasing
    def normalize_var(var):
        base = var.split('.')[0]           # "DWD_21" from "DWD_21.PRCP"
        base_no_underscore = base.replace('_', '')  # "DWD21"
        return base_no_underscore.lower()           # "dwd21"
    
    # Collect matches in the order of the dictionary keys
    selected_vars_in_order = []
    if gcm:
        for key in top_n_models:
            # Key looks like "eccc_5_JanIc_"; we take only "dwd21"
            key_prefix = "".join([key.split('_')[0].lower(),key.split('_')[1].lower()])
            
            # Find all matching variables for this key
            matches = [            
                var for var in center_variable
                if normalize_var(var).startswith(key_prefix)
            ]
            
            # Extend the list by all matches (or pick just the first one, depending on your needs)
            selected_vars_in_order.extend(matches)
    elif hydro:
        for key in top_n_models:
            # Key looks like "eccc_5_JanIc_"; we take only "dwd21"
            key_prefix = key.split(".")[0].lower().replace("_","")
            
            # Find all matching variables for this key
            matches = [            
                var for var in center_variable
                if normalize_var(var).startswith(key_prefix)
            ]
            
            # Extend the list by all matches (or pick just the first one, depending on your needs)
            selected_vars_in_order.extend(matches)
    elif agroparam:
        for key in top_n_models:
            # Key looks like "eccc_5_JanIc_"; we take only "dwd21"
            key_prefix = key.split('_')[0][0:5].lower()
            
            # Find all matching variables for this key
            matches = [            
                var for var in center_variable
                if normalize_var(var).startswith(key_prefix)
            ]
            
            # Extend the list by all matches (or pick just the first one, depending on your needs)
            selected_vars_in_order.extend(matches)        
    else:
        for key in top_n_models:
            key_prefix = key.split('.')[0]
            
            # Find all matching variables for this key
            matches = [            
                var for var in center_variable
                if var.startswith(key_prefix)
            ]
            
            # Extend the list by all matches (or pick just the first one, depending on your needs)
            selected_vars_in_order.extend(matches)        
    return selected_vars_in_order # selected_vars


import os
import requests
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import regionmask
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as path_effects
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
from cartopy.feature import ShapelyFeature
from zipfile import ZipFile
from scipy.ndimage import gaussian_filter

def get_shapefile_(country_code, admin_level=0, source="naturalearth"):
    """
    Downloads or loads shapefiles based on country ISO3 code.
    """
    if source == "naturalearth":
        res = '10m'
        category = 'cultural'
        name = 'admin_1_states_provinces' if admin_level > 0 else 'admin_0_countries'
        shp_path = shapereader.natural_earth(resolution=res, category=category, name=name)
        gdf = gpd.read_file(shp_path)
        
        # Natural Earth uses different column names for ISO codes
        iso_col = 'ADM0_A3' if admin_level == 0 else 'adm0_a3'
        gdf = gdf[gdf[iso_col].str.contains(country_code.upper())]
        return gdf

    elif source == "gadm":
        url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_{country_code.upper()}_shp.zip"
        local_zip = f"{country_code}_shp.zip"
        extract_dir = f"shp_{country_code}"
        
        if not os.path.exists(extract_dir):
            print(f"Downloading GADM {country_code}...")
            r = requests.get(url)
            with open(local_zip, 'wb') as f: f.write(r.content)
            with ZipFile(local_zip, 'r') as zip_ref: zip_ref.extractall(extract_dir)
        
        return gpd.read_file(f"{extract_dir}/gadm41_{country_code.upper()}_{admin_level}.shp")

def get_shapefile(country_code, admin_level=0, source="naturalearth"):
    """
    Downloads or loads shapefiles based on country ISO3 code.
    Automatically switches to GADM if admin_level > 1.
    """
    # Force GADM if level 2 or 3 is requested (Natural Earth doesn't have these)
    if admin_level >= 2:
        print(f"Level {admin_level} requested. Switching source to 'gadm'...")
        source = "gadm"

    if source == "naturalearth":
        res = '10m'
        category = 'cultural'
        # Level 0 is Countries, Level 1 is States/Provinces
        name = 'admin_1_states_provinces' if admin_level > 0 else 'admin_0_countries'
        
        shp_path = shapereader.natural_earth(resolution=res, category=category, name=name)
        gdf = gpd.read_file(shp_path)
        
        # Filter logic
        iso_col = 'ADM0_A3' if admin_level == 0 else 'adm0_a3'
        gdf = gdf[gdf[iso_col].str.contains(country_code.upper())]
        
        if gdf.empty:
            print(f"Warning: No data found for {country_code} in Natural Earth level {admin_level}")
        return gdf

    elif source == "gadm":
        # Ensure the directory exists
        extract_dir = f"shp_{country_code}"
        if not os.path.exists(extract_dir):
            os.makedirs(extract_dir, exist_ok=True)
            
        local_zip = f"{extract_dir}/{country_code}_shp.zip"
        # GADM 4.1 URL structure
        url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_{country_code.upper()}_shp.zip"
        
        shp_file = f"{extract_dir}/gadm41_{country_code.upper()}_{admin_level}.shp"
        
        if not os.path.exists(shp_file):
            print(f"Fetching GADM Level {admin_level} for {country_code}...")
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                with open(local_zip, 'wb') as f:
                    f.write(r.content)
                with ZipFile(local_zip, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            except Exception as e:
                print(f"Error downloading GADM data: {e}")
                return None
        
        return gpd.read_file(shp_file)


def plot_prob_forecastsAlpha(dir_to_save, forecast_prob, model_name, 
                        country_code="GHA", admin_level=0, source="naturalearth",
                        labels=["Below-Normal", "Near-Normal", "Above-Normal"], 
                        reverse_cmap=True, hspace=None, logo=None, 
                        logo_size=("10%", "10%"), logo_position="lower left", 
                        sigma=None, res=None, stations_df=None, out="pdf", dynamic_scalebar=False):

    # 3. Spatial Smoothing
    if sigma:
        for p in forecast_prob.probability:
            forecast_prob.loc[{'probability': p}] = gaussian_filter(forecast_prob.sel(probability=p), sigma=sigma)

    if res is not None:
        min_X = forecast_prob['X'].min().values
        max_X = forecast_prob['X'].max().values
        min_Y = forecast_prob['Y'].min().values
        max_Y = forecast_prob['Y'].max().values
        num_X = int((max_X - min_X) / res) + 1
        num_Y = int((max_Y - min_Y) / res) + 1
        new_X = np.linspace(min_X, max_X, num_X)
        new_Y = np.linspace(min_Y, max_Y, num_Y)
        forecast_prob = forecast_prob.interp(X=new_X, Y=new_Y, method='linear',
                                                kwargs={'fill_value': 'extrapolate'}    
                                            )
    
    # 1. Automatic Shapefile Ingestion
    gdf = get_shapefile(country_code, admin_level, source)
    if gdf.crs != "EPSG:4326": gdf = gdf.to_crs("EPSG:4326")

    # 2. Map Cutting (Masking)
    mask = regionmask.mask_3D_geopandas(gdf, forecast_prob.X, forecast_prob.Y)
    forecast_prob = forecast_prob.where(mask.any(dim='region'))

    # 4. Tercile Selection
    max_prob = forecast_prob.max(dim="probability")
    max_cat = forecast_prob.fillna(-1).argmax(dim="probability")
    
    # 5. Global Dynamic Scaling (Calculated ONLY from values inside boundary)
    plotted_floats = max_prob.values.flatten()
    plotted_floats = plotted_floats[~np.isnan(plotted_floats)] * 100
    
    if dynamic_scalebar and len(plotted_floats) > 0:
        g_vmin = max(35, np.floor(plotted_floats.min() / 5) * 5)
        g_vmax = min(100, np.ceil(plotted_floats.max() / 5) * 5)
        # Create ticks based on dynamic g_vmin and g_vmax with 5% intervals
        cbar_ticks_bn = np.arange(g_vmin, g_vmax + 1, 5)
        cbar_ticks_nn = np.arange(g_vmin, g_vmax + 1, 5)
        cbar_ticks_an = np.arange(g_vmin, g_vmax + 1, 5)
    else:
        cbar_ticks_bn = np.arange(35, 85 + 1, 5)
        cbar_ticks_nn = np.arange(35, 65 + 1, 5)
        cbar_ticks_an = np.arange(35, 85 + 1, 5)

    num_bins_bn = len(cbar_ticks_bn) - 1
    num_bins_nn = len(cbar_ticks_nn) - 1
    num_bins_an = len(cbar_ticks_an) - 1
    
    # 6. Plotting Infrastructure
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 3, height_ratios=[10, 0.2], width_ratios=[1.2, 0.6, 1.2],hspace=hspace or -0.5, wspace=0.3)
    ax = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
    
    # Zoom to shapefile
    bounds = gdf.total_bounds
    ax.set_extent([bounds[0]-0.1, bounds[2]+0.1, bounds[1]-0.1, bounds[3]+0.1])

    # ---- DISCRETE COLORMAP AND BOUNDARY NORM SETUP ----
    

    # Base Colors
    an_colors = ['#d9f0a3', '#addd8e', '#78c679', '#41ab5d', '#238443', '#006837', '#004529']
    bn_colors = ['#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#bd0026', '#800026']
    nn_colors = ["#ffeda0", "#f7fcb9"]
    
    if reverse_cmap:
        an_colors, bn_colors = bn_colors, an_colors

    # Build discrete colormaps matching the exact number of required bins
    cmaps = [
        mcolors.LinearSegmentedColormap.from_list('BN', bn_colors, N=num_bins_bn),
        mcolors.LinearSegmentedColormap.from_list('NN', nn_colors, N=num_bins_nn),
        mcolors.LinearSegmentedColormap.from_list('AN', an_colors, N=num_bins_an)
    ]
    
    # Create BoundaryNorms to strictly enforce discrete color blocks
    norms = [
        mcolors.BoundaryNorm(cbar_ticks_bn, cmaps[0].N),
        mcolors.BoundaryNorm(cbar_ticks_nn, cmaps[1].N),
        mcolors.BoundaryNorm(cbar_ticks_an, cmaps[2].N)
    ]

    # Plot terciles using the discrete norms
    data_layers = [
        (max_prob.where(max_cat == 0)*100), 
        (max_prob.where(max_cat == 1)*100), 
        (max_prob.where(max_cat == 2)*100)
    ]
    
    ims = []
    for i, data in enumerate(data_layers):
        if np.any(~np.isnan(data.values)):
            im = ax.pcolormesh(forecast_prob.X, forecast_prob.Y, data, 
                               cmap=cmaps[i], norm=norms[i], 
                               transform=ccrs.PlateCarree(), alpha=0.9, zorder=2)
        else:
            im = cm.ScalarMappable(norm=norms[i], cmap=cmaps[i])
            im.set_array([])
        ims.append(im)

    # 7. Add Boundary & Stations
    ax.add_feature(ShapelyFeature(gdf.geometry, ccrs.PlateCarree(), facecolor='none', edgecolor='black', lw=1.5), zorder=5)
    gl = ax.gridlines(draw_labels=True, linewidth=0.01, color='gray', alpha=0.8)
    gl.top_labels = False
    gl.right_labels = False
    
    if stations_df is not None:
        for col in stations_df.columns[1:]:
            lat, lon = stations_df.loc[0, col], stations_df.loc[1, col]
            if not np.isnan(lat):
                ax.plot(lon, lat, 'ro', markersize=4, markeredgecolor='w', transform=ccrs.PlateCarree(), zorder=10)
                txt = ax.text(lon + 0.02, lat + 0.02, col, fontsize=8, transform=ccrs.PlateCarree(), zorder=11)
                txt.set_path_effects([path_effects.withStroke(linewidth=2, foreground='w')])

    # 8. Add logo if provided
    if logo is not None:
        ax_logo = inset_axes(ax,
                            width=logo_size[0],    
                            height=logo_size[1],        
                            loc=logo_position,
                            borderpad=0)        
        ax_logo.imshow(mpimg.imread(logo))
        ax_logo.axis("off") 


    # 9. Unified Discrete Colorbars with Inward Ticks
    all_ticks = [cbar_ticks_bn, cbar_ticks_nn, cbar_ticks_an]

    for i, label in enumerate(labels):
        current_ticks = all_ticks[i]  # Fetch the specific ticks for this category
        
        cax = fig.add_subplot(gs[1, i])
        
        cb = plt.colorbar(ims[i], cax=cax, orientation='horizontal', 
                          spacing='uniform', ticks=current_ticks)
        
        cb.set_label(f"{label} (%)", fontsize=10)
        cb.set_ticklabels([f"{int(t)}" for t in current_ticks])
        cb.ax.tick_params(axis='x', direction='in', length=6, color='black')
        
        # INVERT THE BN AXIS (First loop iteration)s
        if i == 0:
            cax.invert_xaxis()

    # 10. Final Aesthetics
    ax.set_title(f"Probabilistic Seasonal Forecast\n{model_name}", fontsize=14, pad=20, fontweight='bold')
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='#e0f3ff')
    ax.coastlines(resolution='10m')
    
    plt.savefig(os.path.join(dir_to_save, f"{model_name}.{out}"), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close('all')


def plot_prob_forecasts(dir_to_save, forecast_prob, model_name, title = "Seasonal Forecast for Gulf of Guinea Countries \n Valid for March-April-May 2026, Issued February 27, 2026" ,
                        country_code="GHA", admin_level=0, source="naturalearth",
                        labels=["Below-Normal", "Near-Normal", "Above-Normal"], 
                        reverse_cmap=True, hspace=None, 
                        logo=None, logo_size=("10%", "10%"), logo_position="lower left", 
                        logo_left=None, logo_left_size=("12%", "12%"), 
                        logo_right=None, logo_right_size=("12%", "12%"),
                        sigma=None, res=None, stations_df=None, out="pdf", dynamic_scalebar=False):

    # 3. Spatial Smoothing
    if sigma:
        for p in forecast_prob.probability:
            forecast_prob.loc[{'probability': p}] = gaussian_filter(forecast_prob.sel(probability=p), sigma=sigma)

    if res is not None:
        min_X = forecast_prob['X'].min().values
        max_X = forecast_prob['X'].max().values
        min_Y = forecast_prob['Y'].min().values
        max_Y = forecast_prob['Y'].max().values
        num_X = int((max_X - min_X) / res) + 1
        num_Y = int((max_Y - min_Y) / res) + 1
        new_X = np.linspace(min_X, max_X, num_X)
        new_Y = np.linspace(min_Y, max_Y, num_Y)
        forecast_prob = forecast_prob.interp(X=new_X, Y=new_Y, method='linear',
                                                kwargs={'fill_value': 'extrapolate'}    
                                            )
    
    # 1. Automatic Shapefile Ingestion
    gdf = get_shapefile(country_code, admin_level, source)
    if gdf.crs != "EPSG:4326": gdf = gdf.to_crs("EPSG:4326")

    # 2. Map Cutting (Masking)
    mask = regionmask.mask_3D_geopandas(gdf, forecast_prob.X, forecast_prob.Y)
    forecast_prob = forecast_prob.where(mask.any(dim='region'))

    # 4. Tercile Selection
    max_prob = forecast_prob.max(dim="probability")
    max_cat = forecast_prob.fillna(-1).argmax(dim="probability")
    
    # 5. Global Dynamic Scaling (Calculated ONLY from values inside boundary)
    plotted_floats = max_prob.values.flatten()
    plotted_floats = plotted_floats[~np.isnan(plotted_floats)] * 100
    
    if dynamic_scalebar and len(plotted_floats) > 0:
        g_vmin = max(35, np.floor(plotted_floats.min() / 5) * 5)
        g_vmax = min(100, np.ceil(plotted_floats.max() / 5) * 5)
        # Create ticks based on dynamic g_vmin and g_vmax with 5% intervals
        cbar_ticks_bn = np.arange(g_vmin, g_vmax + 1, 5)
        cbar_ticks_nn = np.arange(g_vmin, g_vmax + 1, 5)
        cbar_ticks_an = np.arange(g_vmin, g_vmax + 1, 5)
    else:
        cbar_ticks_bn = np.arange(35, 85 + 1, 5)
        cbar_ticks_nn = np.arange(35, 65 + 1, 5)
        cbar_ticks_an = np.arange(35, 85 + 1, 5)

    num_bins_bn = len(cbar_ticks_bn) - 1
    num_bins_nn = len(cbar_ticks_nn) - 1
    num_bins_an = len(cbar_ticks_an) - 1
    
    # 6. Plotting Infrastructure
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 3, height_ratios=[10, 0.2], width_ratios=[1.2, 0.6, 1.2],hspace=hspace or -0.5, wspace=0.3)
    ax = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
    
    # Zoom to shapefile
    bounds = gdf.total_bounds
    ax.set_extent([bounds[0]-0.1, bounds[2]+0.1, bounds[1]-0.1, bounds[3]+0.1])

    # ---- DISCRETE COLORMAP AND BOUNDARY NORM SETUP ----
    
    # Base Colors
    an_colors = ['#d9f0a3', '#addd8e', '#78c679', '#41ab5d', '#238443', '#006837', '#004529']
    bn_colors = ['#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#bd0026', '#800026']
    nn_colors = ["#ffeda0", "#f7fcb9"]
    
    if reverse_cmap:
        an_colors, bn_colors, nn_colors  = bn_colors, an_colors, nn_colors[::-1]

    # Build discrete colormaps matching the exact number of required bins
    cmaps = [
        mcolors.LinearSegmentedColormap.from_list('BN', bn_colors, N=num_bins_bn),
        mcolors.LinearSegmentedColormap.from_list('NN', nn_colors, N=num_bins_nn),
        mcolors.LinearSegmentedColormap.from_list('AN', an_colors, N=num_bins_an)
    ]
    
    # Create BoundaryNorms to strictly enforce discrete color blocks
    norms = [
        mcolors.BoundaryNorm(cbar_ticks_bn, cmaps[0].N),
        mcolors.BoundaryNorm(cbar_ticks_nn, cmaps[1].N),
        mcolors.BoundaryNorm(cbar_ticks_an, cmaps[2].N)
    ]

    # Plot terciles using the discrete norms
    data_layers = [
        (max_prob.where(max_cat == 0)*100), 
        (max_prob.where(max_cat == 1)*100), 
        (max_prob.where(max_cat == 2)*100)
    ]
    
    ims = []
    for i, data in enumerate(data_layers):
        if np.any(~np.isnan(data.values)):
            im = ax.pcolormesh(forecast_prob.X, forecast_prob.Y, data, 
                               cmap=cmaps[i], norm=norms[i], 
                               transform=ccrs.PlateCarree(), alpha=0.9, zorder=2)
        else:
            im = cm.ScalarMappable(norm=norms[i], cmap=cmaps[i])
            im.set_array([])
        ims.append(im)

    # 7. Add Boundary & Stations
    ax.add_feature(ShapelyFeature(gdf.geometry, ccrs.PlateCarree(), facecolor='none', edgecolor='black', lw=1.5), zorder=5)
    gl = ax.gridlines(draw_labels=True, linewidth=0.01, color='gray', alpha=0.8)
    gl.top_labels = False
    gl.right_labels = False
    
    if stations_df is not None:
        for col in stations_df.columns[1:]:
            lat, lon = stations_df.loc[0, col], stations_df.loc[1, col]
            if not np.isnan(lat):
                ax.plot(lon, lat, 'ro', markersize=4, markeredgecolor='w', transform=ccrs.PlateCarree(), zorder=10)
                txt = ax.text(lon + 0.02, lat + 0.02, col, fontsize=8, transform=ccrs.PlateCarree(), zorder=11)
                txt.set_path_effects([path_effects.withStroke(linewidth=2, foreground='w')])

    # ==========================================
    # 8. ADD LOGOS (Inside and Outside Options)
    # ==========================================
    
    # Original: Logo inside the map
    if logo is not None:
        ax_logo = inset_axes(ax, width=logo_size[0], height=logo_size[1], 
                             loc=logo_position, borderpad=0)        
        ax_logo.imshow(mpimg.imread(logo))
        ax_logo.axis("off") 

    # New: Outside Left Logo
    if logo_left is not None:
        # bbox_to_anchor creates a bounding box starting slightly above the top-left of the axis (y=1.02)
        ax_logo_l = inset_axes(ax, width=logo_left_size[0], height=logo_left_size[1],
                               loc='lower left', bbox_to_anchor=(0.0, 1.06, 1, 1),
                               bbox_transform=ax.transAxes, borderpad=0)
        ax_logo_l.imshow(mpimg.imread(logo_left))
        ax_logo_l.axis("off")

    # New: Outside Right Logo
    if logo_right is not None:
        # Pinned to the lower right of the bounding box placed above the axis
        ax_logo_r = inset_axes(ax, width=logo_right_size[0], height=logo_right_size[1],
                               loc='lower right', bbox_to_anchor=(0.0, 1.06, 1, 1),
                               bbox_transform=ax.transAxes, borderpad=0)
        ax_logo_r.imshow(mpimg.imread(logo_right))
        ax_logo_r.axis("off")


    # 9. Unified Discrete Colorbars with Inward Ticks
    all_ticks = [cbar_ticks_bn, cbar_ticks_nn, cbar_ticks_an]

    for i, label in enumerate(labels):
        current_ticks = all_ticks[i]  # Fetch the specific ticks for this category
        
        cax = fig.add_subplot(gs[1, i])
        
        cb = plt.colorbar(ims[i], cax=cax, orientation='horizontal', 
                          spacing='uniform', ticks=current_ticks)
        
        cb.set_label(f"{label} (%)", fontsize=10)
        cb.set_ticklabels([f"{int(t)}" for t in current_ticks])
        cb.ax.tick_params(axis='x', direction='in', length=6, color='black')
        
        # INVERT THE BN AXIS (First loop iteration)s
        if i == 0:
            cax.invert_xaxis()

    # 10. Final Aesthetics
    # Dynamically increase title padding if top outside logos are active
    t_pad = 30 if (logo_left is not None or logo_right is not None) else 20
    ax.set_title(title.upper(), fontsize=14, pad=t_pad, fontweight='bold')
    
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='#e0f3ff')
    ax.coastlines(resolution='10m')
    
    # bbox_inches='tight' will automatically ensure the outside logos and new padding are saved cleanly
    plt.savefig(os.path.join(dir_to_save, f"{model_name}.{out}"), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close('all')
def plot_prob_forecasts_(dir_to_save, forecast_prob, model_name, labels=["Below-Normal", "Near-Normal", "Above-Normal"], reverse_cmap=True, hspace=None, logo=None, logo_size=(None,None), logo_position="lower left", sigma=None, res=None):
    """
    Plot probabilistic forecasts with tercile categories using discrete block colorbars.
    """

    if res is not None:
        min_X = forecast_prob['X'].min().values
        max_X = forecast_prob['X'].max().values
        min_Y = forecast_prob['Y'].min().values
        max_Y = forecast_prob['Y'].max().values
        num_X = int((max_X - min_X) / res) + 1
        num_Y = int((max_Y - min_Y) / res) + 1
        new_X = np.linspace(min_X, max_X, num_X)
        new_Y = np.linspace(min_Y, max_Y, num_Y)
        forecast_prob = forecast_prob.interp(X=new_X, Y=new_Y, method='linear',
                                                kwargs={'fill_value': 'extrapolate'}    
                                            )

    if sigma is not None:
        # Create a smoothed copy
        forecast_prob_smoothed = forecast_prob * 0.0  
        
        # Smooth each probability layer spatially
        for p in forecast_prob.probability.values:
            layer = forecast_prob.sel(probability=p)
            layer_smoothed = xr.apply_ufunc(
                gaussian_filter,
                layer,
                input_core_dims=[['Y', 'X']],
                output_core_dims=[['Y', 'X']],
                kwargs={'sigma': sigma}
            )
            forecast_prob_smoothed.loc[{'probability': p}] = layer_smoothed
        
        # Normalize smoothed probabilities to sum to 1 at each grid point
        sum_probs = forecast_prob_smoothed.sum('probability')
        forecast_prob_smoothed = forecast_prob_smoothed / sum_probs.where(sum_probs != 0, 1.0)
        forecast_prob = forecast_prob_smoothed

    # Step 1: Extract maximum probability and category
    max_prob = forecast_prob.max(dim="probability", skipna=True) 
    filled_prob = forecast_prob.fillna(-9999)
    max_category = filled_prob.argmax(dim="probability")
    
    # Step 2: Create masks for each category
    mask_bn = max_category == 0  # Below Normal (BN)
    mask_nn = max_category == 1  # Near Normal (NN)
    mask_an = max_category == 2  # Above Normal (AN)
    
    # Step 3: Define Custom Colors and Ticks for Discrete Bins
    def create_ticks(vn=35, vx=86, step=5):
        return np.arange(vn, vx, step)

    ticks_bn = create_ticks(vn=35, vx=86, step=5) # 11 ticks, 10 bins
    ticks_nn = create_ticks(vn=35, vx=66, step=5) # 7 ticks, 6 bins
    ticks_an = create_ticks(vn=35, vx=86, step=5) # 11 ticks, 10 bins

    # Requested Hex Codes
    colors_bn = ['#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#bd0026', '#800026']
    colors_nn = ["#ffeda0", "#f7fcb9"] #['#ffeda0', '#ffffcc', '#ffeda0', '#fed976', '#f7fcb9']
    colors_an = ['#d9f0a3', '#addd8e', '#78c679', '#41ab5d', '#238443', '#006837', '#004529']

    # Build colormaps to have exactly the number of bins as our ticks
    if reverse_cmap:
        BN_cmap = mcolors.LinearSegmentedColormap.from_list('BN', colors_an, N=len(ticks_bn)-1)
        NN_cmap = mcolors.LinearSegmentedColormap.from_list('NN', colors_nn[::-1], N=len(ticks_nn)-1)
        AN_cmap = mcolors.LinearSegmentedColormap.from_list('AN', colors_bn, N=len(ticks_an)-1)
    else:
        BN_cmap = mcolors.LinearSegmentedColormap.from_list('BN', colors_bn, N=len(ticks_bn)-1)
        NN_cmap = mcolors.LinearSegmentedColormap.from_list('NN', colors_nn, N=len(ticks_nn)-1)
        AN_cmap = mcolors.LinearSegmentedColormap.from_list('AN', colors_an, N=len(ticks_an)-1)
        
    # NN_cmap = mcolors.LinearSegmentedColormap.from_list('NN', colors_nn, N=len(ticks_nn)-1)

    # BoundaryNorm maps the exact tick intervals to the discrete colors
    norm_bn = mcolors.BoundaryNorm(ticks_bn, BN_cmap.N)
    norm_nn = mcolors.BoundaryNorm(ticks_nn, NN_cmap.N)
    norm_an = mcolors.BoundaryNorm(ticks_an, AN_cmap.N)
    
    # Create a figure with GridSpec
    fig = plt.figure(figsize=(10, 8))
    if hspace is None:
        hspace = -0.6  
    if logo is not None and logo_size == (None, None):
        logo_size = ("7%", "21%") 

    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 3, height_ratios=[10, 0.2], width_ratios=[1.2, 0.6, 1.2], hspace=hspace, wspace=0.2)

    # Main map axis
    ax = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linewidth=0.05, color='gray', alpha=0.8)
    gl.top_labels = False
    gl.right_labels = False
    
    bn_data = (max_prob.where(mask_bn) * 100).values
    nn_data = (max_prob.where(mask_nn) * 100).values
    an_data = (max_prob.where(mask_an) * 100).values

    # Add Land colors
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#fde0dd", edgecolor="black", zorder=0) 
    
    # Step 4: Plot Discrete Blocks (vmin/vmax are replaced by our custom norms)
    if np.any(~np.isnan(bn_data)):
        bn_plot = ax.pcolormesh(
            forecast_prob['X'], forecast_prob['Y'], bn_data,
            cmap=BN_cmap, norm=norm_bn, transform=ccrs.PlateCarree(), alpha=0.9
        )
    else:
        bn_plot = cm.ScalarMappable(norm=norm_bn, cmap=BN_cmap)
        bn_plot.set_array([])

    if np.any(~np.isnan(nn_data)):
        nn_plot = ax.pcolormesh(
            forecast_prob['X'], forecast_prob['Y'], nn_data,
            cmap=NN_cmap, norm=norm_nn, transform=ccrs.PlateCarree(), alpha=0.9
        )
    else:
        nn_plot = cm.ScalarMappable(norm=norm_nn, cmap=NN_cmap)
        nn_plot.set_array([])

    if np.any(~np.isnan(an_data)):
        an_plot = ax.pcolormesh(
            forecast_prob['X'], forecast_prob['Y'], an_data,
            cmap=AN_cmap, norm=norm_an, transform=ccrs.PlateCarree(), alpha=0.9
        )
    else:
        an_plot = cm.ScalarMappable(norm=norm_an, cmap=AN_cmap)
        an_plot.set_array([])

    # Add coastlines and borders
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=1.0, linestyle='solid')
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    
    # Step 6: Add individual colorbars with exact tick snapping
    
    # For BN (Below Normal)
    cbar_ax_bn = fig.add_subplot(gs[1, 0])
    cbar_bn = plt.colorbar(bn_plot, cax=cbar_ax_bn, orientation='horizontal', spacing='uniform', ticks=ticks_bn)
    cbar_bn.set_label(f'{labels[0]} (%)')
    cbar_bn.set_ticklabels([f"{tick}" for tick in ticks_bn])
    cbar_ax_bn.invert_xaxis() # Invert scale from 85 down to 35
    cbar_bn.ax.tick_params(axis='x', direction='in', length=6, color='black') # Ticks inward

    # For NN (Near Normal)
    cbar_ax_nn = fig.add_subplot(gs[1, 1])
    cbar_nn = plt.colorbar(nn_plot, cax=cbar_ax_nn, orientation='horizontal', spacing='uniform', ticks=ticks_nn)
    cbar_nn.set_label(f'{labels[1]} (%)')
    cbar_nn.set_ticklabels([f"{tick}" for tick in ticks_nn])
    cbar_nn.ax.tick_params(axis='x', direction='in', length=6, color='black') # Ticks inward

    # For AN (Above Normal)
    cbar_ax_an = fig.add_subplot(gs[1, 2])
    cbar_an = plt.colorbar(an_plot, cax=cbar_ax_an, orientation='horizontal', spacing='uniform', ticks=ticks_an)
    cbar_an.set_label(f'{labels[2]} (%)')
    cbar_an.set_ticklabels([f"{tick}" for tick in ticks_an])
    cbar_an.ax.tick_params(axis='x', direction='in', length=6, color='black') # Ticks inward
    
    # Set the title
    if isinstance(model_name, np.ndarray):
        model_name_str = str(model_name.item())
    else:
        model_name_str = str(model_name)
    ax.set_title(f"{model_name_str}", fontsize=13, pad=20)

    # Add logo if provided
    if logo is not None:
        ax_logo = inset_axes(ax,
                            width=logo_size[0],        
                            height=logo_size[1],       
                            loc=logo_position,
                            borderpad=0.1)        
        ax_logo.imshow(mpimg.imread(logo))
        ax_logo.axis("off") 

    plt.subplots_adjust(top=0.95, bottom=0.08, left=0.06, right=0.94, hspace=-0.6, wspace=0.2)
    plt.savefig(f"{dir_to_save}/{model_name_str.replace(' ', '_')}.pdf", dpi=300, bbox_inches='tight')
    plt.show()
def plot_det_forecasts(da2d: xr.DataArray, title: str, outpng: str = None):
    """Plot a 2D field (Y, X) on a PlateCarree map."""
    if not set(["Y", "X"]).issubset(set(da2d.dims)):
        raise ValueError(f"Expected dims (Y, X). Got {da2d.dims}")

    # Ensure sorted coords for clean pcolormesh
    da2d = da2d.sortby(["Y", "X"])

    lon = da2d["X"].values
    lat = da2d["Y"].values
    data = da2d.values

    # Robust limits (avoids extreme outliers dominating the color range)
    vmin = np.nanpercentile(data, 2)
    vmax = np.nanpercentile(data, 98)

    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Extent: [west, east, south, north]
    ax.set_extent([float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())], crs=ccrs.PlateCarree())

    im = ax.pcolormesh(
        lon, lat, data,
        transform=ccrs.PlateCarree(),
        shading="auto",
        vmin=vmin, vmax=vmax,
    )

    ax.coastlines(linewidth=1.0)
    ax.add_feature(cfeature.BORDERS, linewidth=0.8)
    ax.add_feature(cfeature.LAND, edgecolor="black", linewidth=0.3, alpha=0.2)
    ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

    cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.06, fraction=0.05)
    # cb.set_label(f"{da2d.name or 'variable'} (units unknown)")

    ax.set_title(title)
    plt.tight_layout()

    if outpng:
        plt.savefig(outpng, dpi=200, bbox_inches="tight")
        print(f"Saved figure: {outpng}")

    plt.show()


def plot_tercile(A, save_dir=None, colors=None, year=None):
    """
    Plot a tercile map with categories: Below, Normal, Above.

    Parameters
    ----------
    A : xarray.DataArray
        Data array with tercile categories (0: Below, 1: Normal, 2: Above) and dimensions 'T', 'Y', 'X'.

    Notes
    -----
    Uses a custom colormap and displays a legend for tercile categories.
    """
    fig = plt.figure(figsize=(9, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())
    if colors is None:
        colors = ['#fc8d59', '#bdbdbd', '#99d594']
    else:
        colors
    cmap = ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = BoundaryNorm(bounds, cmap.N)
    lon = A['X']
    lat = A['Y']
    img = ax.pcolormesh(lon, lat, A.isel(T=0), cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.BORDERS, linewidth=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=1)
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
    if year is None:
        title = "Observed terciles"
    else:
        title = f"Observed terciles {year}"
    plt.title(title, fontsize=16, weight='bold')
    legend_elements = [
        mpatches.Patch(color=colors[2], label='ABOVE AVERAGE'),
        mpatches.Patch(color=colors[1], label='NEAR AVERAGE'),
        mpatches.Patch(color=colors[0], label='BELOW AVERAGE')
    ]
    plt.legend(handles=legend_elements, loc='lower left')
    plt.tight_layout()
    if save_dir is None:
        plt.show()
    else:
        plt.savefig(f"{save_dir}.pdf", dpi=300, bbox_inches='tight')
        plt.show()


# Transform consensus#################################################################################"
######################################################################################################


def _parse_forecast_str(s, cats=('PB','PN','PA')):
    """
    Parse strings like '45-35-20' into a dict {PB:0.45, PN:0.35, PA:0.20}.
    """
    parts = [p.strip() for p in str(s).split('-')]
    if len(parts) != len(cats):
        raise ValueError(f"Forecast '{s}' must have {len(cats)} parts")
    vals = [float(p)/100.0 for p in parts]
    return dict(zip(cats, vals))

def polygons_to_prob_ds(
    da_like: xr.DataArray,
    gdf: gpd.GeoDataFrame,
    forecast_col: str = "Forecast",
    categories=('PB','PN','PA'),
    time_dim: str = "T",
    y_dim: str = "Y",
    x_dim: str = "X",
):
    """
    Rasterize polygon forecasts onto the (Y,X) grid of `da_like` and return:
      xr.Dataset with dims (probability, T, Y, X) and coords probability=['PB','PN','PA'].
    `da_like` is only used for its coords (1 time step expected).
    """
    # --- Safety checks & CRS ---
    if gdf.crs is None:
        # assume lon/lat
        gdf = gdf.set_crs(4326)
    else:
        gdf = gdf.to_crs(4326)

    X = da_like.coords[x_dim].values
    Y = da_like.coords[y_dim].values
    T = da_like.coords[time_dim].values  # expected shape (1,)

    nx = X.size
    ny = Y.size
    if T.size != 1:
        raise ValueError("This helper expects a single time step in da_like.")

    # --- Build transform aligned to cell edges (coords are assumed centers) ---
    dx = float(np.mean(np.diff(X)))
    dy = float(np.mean(np.diff(Y)))
    west  = float(X.min() - dx/2)
    east  = float(X.max() + dx/2)
    south = float(Y.min() - dy/2)
    north = float(Y.max() + dy/2)
    transform = from_bounds(west, south, east, north, nx, ny)

    # --- Prepare output arrays per probability category ---
    out = {cat: np.full((ny, nx), np.nan, dtype=float) for cat in categories}

    # --- Rasterize polygons and assign values ---
    # If your polygons do not overlap, later ones won't matter; if they do,
    # later rows in gdf will overwrite earlier ones.
    for row in gdf.itertuples(index=False):
        geom = getattr(row, "geometry")
        fstr = getattr(row, forecast_col)
        probs = _parse_forecast_str(fstr, categories)

        # mask for this geometry (1 inside polygon, 0 outside)
        mask = rasterize(
            [(geom, 1)],
            out_shape=(ny, nx),
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=False,  # set True if you want a slightly “fatter” polygon fill
        ).astype(bool)

        # Align raster rows (north→south) with Y coords (south→north)
        if np.all(np.diff(Y) > 0):   # Y ascending
            mask = np.flipud(mask)

        for cat in categories:
            out_arr = out[cat]
            out_arr[mask] = probs[cat]

    # --- Assemble into xarray with desired dims (probability, T, Y, X) ---
    data_stack = np.stack([out[cat] for cat in categories], axis=0)  # (probability, Y, X)
    data_stack = data_stack[:, np.newaxis, :, :]  # add T dim -> (probability, T, Y, X)

    ds = xr.Dataset(
        data_vars=dict(
            forecast=(("probability", time_dim, y_dim, x_dim), data_stack)
        ),
        coords={
            "probability": np.array(categories, dtype="<U2"),
            time_dim: T,
            y_dim: Y,
            x_dim: X,
        },
        attrs={}
    )

    return ds


def plot_georeferenced_forecast_map(
    shapefile_zip,
    column="Forecast",
    extent_obs=[12, -3.5, 4, 1.5],
    title="Georeferenced Map of 2024 consensual Forecast",
    figsize=(9, 7),
    cmap="tab20",
    outpath="georeferenced_map.pdf",
    dpi=300,
    show=True,
):
    n, w, s, e = extent_obs
    cartopy_extent = [w, e, s, n]

    shapefile_zip = Path(shapefile_zip)
    outpath = Path(outpath)

    if not shapefile_zip.exists():
        raise FileNotFoundError(f"Shapefile zip not found: {shapefile_zip}")

    gdf = gpd.read_file(f"zip://{shapefile_zip}")

    if column not in gdf.columns:
        raise KeyError(f"Column '{column}' not found. Available: {list(gdf.columns)}")

    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    else:
        gdf = gdf.to_crs(4326)

    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": proj})

    ax.set_extent(cartopy_extent, crs=proj)
    
    ax.add_feature(cfeature.OCEAN, zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="whitesmoke", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAKES, alpha=0.3)
    ax.add_feature(cfeature.RIVERS, linewidth=0.5, alpha=0.5)

    gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.4)
    gl.top_labels = False
    gl.right_labels = False

    gdf.plot(
        column=column,
        ax=ax,
        transform=proj,
        edgecolor="0.3",
        linewidth=0.6,
        cmap=cmap,
        legend=True,
        legend_kwds={"title": column},
        categorical=True,
    )

    ax.set_title(title, fontsize=12)
    plt.tight_layout()

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=dpi)

    if show:
        plt.show()

    return fig, ax, gdf
#########################################################################
import inspect

def process_model_for_other_params(
    agmParamModel, 
    dir_to_save, 
    hdcst_file_path, 
    fcst_file_path, 
    obs_hdcst, 
    obs_fcst_year, 
    month_of_initialization, 
    year_start, 
    year_end, 
    year_forecast, 
    nb_cores=2, 
    agrometparam="Onset"
):
    """
    Process model hindcast and forecast data for agrometeorological parameters.

    Parameters
    ----------
    agmParamModel : object
        Model object with a compute method for processing agrometeorological parameters.
    dir_to_save : str
        Directory path to save processed data.
    hdcst_file_path : dict
        Dictionary mapping model names to hindcast file paths.
    fcst_file_path : dict
        Dictionary mapping model names to forecast file paths.
    obs_hdcst : xarray.Dataset
        Observational hindcast dataset.
    obs_fcst_year : xarray.Dataset
        Observational forecast dataset for the specified year.
    month_of_initialization : int
        Month of model initialization.
    year_start : int
        Start year of the hindcast data.
    year_end : int
        End year of the hindcast data.
    year_forecast : int
        Forecast year.
    nb_cores : int, optional
        Number of CPU cores for parallel processing (default is 2).
    agrometparam : str, optional
        Name of the agrometeorological parameter (default is 'Onset').

    Returns
    -------
    tuple
        Tuple of dictionaries (saved_hindcast_paths, saved_forecast_paths) mapping model names to saved file paths.
    """
    
    # =========================================================================
    # DYNAMIC ARGUMENT EXTRACTION
    # Automatically varying arguments based on the specific agmParamModel passed
    # =========================================================================
    sig = inspect.signature(agmParamModel.compute)
    valid_compute_args = sig.parameters.keys()
    model_attributes = vars(agmParamModel)
    
    dynamic_kwargs = {
        key: model_attributes[key] 
        for key in valid_compute_args 
        if key in model_attributes
    }
    
    # Add nb_cores manually if the compute method accepts it
    if "nb_cores" in valid_compute_args:
        dynamic_kwargs["nb_cores"] = nb_cores
    # =========================================================================

    mask = xr.where(~np.isnan(obs_fcst_year.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
    t_coord = pd.date_range(start=f"{year_forecast}-01-01", end=f"{year_forecast}-12-31", freq="D")
    y_coords = obs_fcst_year.Y
    x_coords = obs_fcst_year.X

    if calendar.isleap(year_forecast):
        daily_climatology = obs_hdcst.groupby('T.dayofyear').mean(dim='T')
        data = daily_climatology.to_numpy()
        dummy = xr.DataArray(
            data=data,
            coords={"T": t_coord, "Y": y_coords, "X": x_coords},
            dims=["T", "Y", "X"]
        ) * mask
    else:
        daily_climatology = obs_hdcst.groupby('T.dayofyear').mean(dim='T')
        daily_climatology = daily_climatology.sel(dayofyear=~((daily_climatology['dayofyear'] == 60)))
        data = daily_climatology.to_numpy()
        dummy = xr.DataArray(
            data=data,
            coords={"T": t_coord, "Y": y_coords, "X": x_coords},
            dims=["T", "Y", "X"]
        ) * mask

    abb_mont_ini = calendar.month_abbr[int(month_of_initialization)]
    dir_to_save = Path(f"{dir_to_save}/model_data")
    os.makedirs(dir_to_save, exist_ok=True)
    
    saved_hindcast_paths = {}
    for i in hdcst_file_path.keys():
        save_path = f"{dir_to_save}/hindcast_{i}_{agrometparam}_{abb_mont_ini}Ic.nc"
        if not os.path.exists(save_path):
            hdcst = xr.open_dataset(hdcst_file_path[i])
            if 'number' in hdcst.dims:
                hdcst = hdcst.mean(dim="number")
            hdcst = hdcst.to_array().drop_vars("variable").squeeze()
            obs_hdcst_sel = obs_hdcst.sel(T=slice(str(year_start), str(year_end)))
            obs_hdcst_interp = obs_hdcst_sel.interp(Y=hdcst.Y, X=hdcst.X, method="linear", kwargs={"fill_value": "extrapolate"})
            ds1_aligned, ds2_aligned = xr.align(hdcst, obs_hdcst_interp, join='outer')
            filled_ds = ds1_aligned.fillna(ds2_aligned)
            ds_filled = filled_ds.copy()
            
            # Use dynamic_kwargs here
            agpm_model = agmParamModel.compute(ds_filled.sortby("T"), **dynamic_kwargs)
            
            ds_processed = agpm_model.to_dataset(name=agrometparam)
            ds_processed.to_netcdf(save_path)
        else:
            print(f"[SKIP] {save_path} already exists.")
        saved_hindcast_paths[i] = save_path
        
    saved_forecast_paths = {}
    for i in fcst_file_path.keys():
        save_path = f"{dir_to_save}/forecast_{i}_{agrometparam}_{abb_mont_ini}Ic.nc"
        if not os.path.exists(save_path):
            fcst = xr.open_dataset(fcst_file_path[i])
            if 'number' in fcst.dims:
                fcst = fcst.mean(dim="number")
            fcst = fcst.to_array().drop_vars("variable").squeeze()
            obs_fcst_sel = obs_fcst_year.sortby("T").sel(T=str(year_forecast))
            obs_fcst_interp = obs_fcst_sel.interp(Y=fcst.Y, X=fcst.X, method="linear", kwargs={"fill_value": "extrapolate"})
            ds1_aligned, ds2_aligned = xr.align(fcst, obs_fcst_interp, join='outer')
            filled_fcst = ds1_aligned.fillna(ds2_aligned)
            ds_filled = filled_fcst.copy()
            ds_filled = ds_filled.sortby("T")
            dummy = dummy.interp(Y=fcst.Y, X=fcst.X, method="linear", kwargs={"fill_value": "extrapolate"})
            ds1_aligned, ds2_aligned = xr.align(ds_filled, dummy, join='outer')
            filled_fcst_ = ds1_aligned.fillna(ds2_aligned)
            ds_filled = filled_fcst_.copy()
            ds_filled = ds_filled.sortby("T")
            
            # Use dynamic_kwargs here
            agpm_model = agmParamModel.compute(ds_filled, **dynamic_kwargs)
            
            ds_processed = agpm_model.to_dataset(name=agrometparam)
            ds_processed.to_netcdf(save_path)
        else:
            print(f"[SKIP] {save_path} already exists.")
        saved_forecast_paths[i] = save_path
        
    return saved_hindcast_paths, saved_forecast_paths
def plot_date(A):
    """
    Plot a map of dates, interpreting values as offsets from 2024-01-01.

    Parameters
    ----------
    A : xarray.DataArray
        Data array with values representing days since 2024-01-01, and dimensions 'Y', 'X'.

    Notes
    -----
    Converts colorbar ticks to calendar dates (e.g., '01-Jan') for readability.
    """
    A = xr.where(A > 366, A - 366, A)
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(projection=ccrs.PlateCarree()))
    plt_obj = A.plot(
        ax=ax,
        x="X",
        y="Y",
        transform=ccrs.PlateCarree(),
        cbar_kwargs={
            'label': 'Date',
            'orientation': 'horizontal',
            'pad': 0.01,
            'shrink': 1,
            'aspect': 25
        }
    )
    cbar = plt_obj.colorbar
    ticks = cbar.get_ticks()
    tick_labels = [
        (datetime.datetime(2024, 1, 1) + timedelta(days=int(tick))).strftime('%d-%b')
        for tick in ticks
    ]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.OCEAN)
    plt.tight_layout()
    plt.show()

def verify_station_network(df_filtered, extent, map_name="Rain-gauge network"):
    """
    Plot station locations on a map to verify the station network.

    Parameters
    ----------
    df_filtered : pandas.DataFrame
        DataFrame containing station data with 'STATION' column ('LAT', 'LON') and station names.
    extent : list
        Geographic extent in the format [west, east, south, north].
    map_name : str, optional
        Title of the map (default is 'Rain-gauge network').

    Notes
    -----
    Stations are plotted as red markers with labels, using Cartopy for map features.
    """
    lat_row = df_filtered.loc[df_filtered["STATION"] == "LAT"].squeeze()
    lon_row = df_filtered.loc[df_filtered["STATION"] == "LON"].squeeze()
    station_names = df_filtered.columns[1:]
    lats = lat_row[1:].astype(float).values
    lons = lon_row[1:].astype(float).values
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=proj)
    ax.set_extent([extent[1], extent[3], extent[2], extent[0]], crs=proj)
    ax.add_feature(cfeature.LAND, facecolor="cornsilk")
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    ax.add_feature(cfeature.BORDERS, linewidth=0.6)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.LAKES, alpha=0.4)
    ax.scatter(lons, lats, s=30, marker="o", facecolor="red", edgecolor="black", transform=proj, zorder=5)
    for lon, lat, name in zip(lons, lats, station_names):
        ax.text(lon + 0.3, lat + 0.3, name, fontsize=6, transform=proj)
    ax.set_title(map_name, fontsize=14)
    plt.tight_layout()
    plt.show()

# Indices definition
sst_indices_name = {
    "NINO34": ("Nino3.4", -170, -120, -5, 5),
    "NINO12": ("Niño1+2", -90, -80, -10, 0),
    "NINO3": ("Nino3", -150, -90, -5, 5),
    "NINO4": ("Nino4", -150, 160, -5, 5),
    "NINO_Global": ("ALL NINO Zone", -80, 160, -10, 5),
    "TNA": ("Tropical Northern Atlantic Index", -55, -15, 5, 25),
    "TSA": ("Tropical Southern Atlantic Index", -30, 10, -20, 0),
    "NAT": ("North Atlantic Tropical", -40, -20, 5, 20),
    "SAT": ("South Atlantic Tropical", -15, 5, -20, 5),
    "TASI": ("NAT-SAT", None, None, None, None),
    "WTIO": ("Western Tropical Indian Ocean (WTIO)", 50, 70, -10, 10),
    "SETIO": ("Southeastern Tropical Indian Ocean (SETIO)", 90, 110, -10, 0),
    "DMI": ("WTIO - SETIO", None, None, None, None),
    "MB": ("Mediterranean Basin", 0, 50, 30, 42),
}


################################ agroparameters compute ################

onset_criteria = {
0: {"zone_name": "Sahel100_0mm", "start_search": "06-01", "cumulative": 15, "number_dry_days": 25, "thrd_rain_day": 0.85, "end_search": "09-01"},
1: {"zone_name": "Sahel200_100mm", "start_search": "05-15", "cumulative": 15, "number_dry_days": 25, "thrd_rain_day": 0.85, "end_search": "08-15"},
2: {"zone_name": "Sahel400_200mm", "start_search": "05-01", "cumulative": 15, "number_dry_days": 20, "thrd_rain_day": 0.85, "end_search": "07-31"},
3: {"zone_name": "Sahel600_400mm", "start_search": "03-15", "cumulative": 20, "number_dry_days": 20, "thrd_rain_day": 0.85, "end_search": "07-31"},
4: {"zone_name": "Soudan",         "start_search": "03-15", "cumulative": 20, "number_dry_days": 10, "thrd_rain_day": 0.85, "end_search": "07-31"},
5: {"zone_name": "Golfe_Of_Guinea","start_search": "02-01", "cumulative": 20, "number_dry_days": 10, "thrd_rain_day": 0.85, "end_search": "06-15"},
    }

onset_dryspell_criteria = {
    0: {"zone_name": "Sahel100_0mm", "start_search": "06-01", "cumulative": 15, "number_dry_days": 25, "thrd_rain_day": 0.85, "end_search": "09-01", "nbjour":30},
    1: {"zone_name": "Sahel200_100mm", "start_search": "05-15", "cumulative": 15, "number_dry_days": 25, "thrd_rain_day": 0.85, "end_search": "08-15", "nbjour":40},
    2: {"zone_name": "Sahel400_200mm", "start_search": "05-01", "cumulative": 15, "number_dry_days": 20, "thrd_rain_day": 0.85, "end_search": "07-31", "nbjour":40},
    3: {"zone_name": "Sahel600_400mm", "start_search": "03-15", "cumulative": 20, "number_dry_days": 20, "thrd_rain_day": 0.85, "end_search": "07-31", "nbjour":45},
    4: {"zone_name": "Soudan",         "start_search": "03-15", "cumulative": 20, "number_dry_days": 10, "thrd_rain_day": 0.85, "end_search": "07-31", "nbjour":50},
    5: {"zone_name": "Golfe_Of_Guinea","start_search": "02-01", "cumulative": 20, "number_dry_days": 10, "thrd_rain_day": 0.85, "end_search": "06-15", "nbjour":50},
}

cessation_criteria = {
    0: {"zone_name": "Sahel100_0mm", "date_dry_soil":"01-01", "start_search": "09-15", "ETP": 5.0, "Cap_ret_maxi": 70, "end_search": "10-05"},
    1: {"zone_name": "Sahel200_100mm", "date_dry_soil":"01-01", "start_search": "09-01", "ETP": 5.0, "Cap_ret_maxi": 70, "end_search": "10-05"},
    2: {"zone_name": "Sahel400_200mm", "date_dry_soil":"01-01", "start_search": "09-01", "ETP": 5.0, "Cap_ret_maxi": 70, "end_search": "11-10"},
    3: {"zone_name": "Sahel600_400mm", "date_dry_soil":"01-01", "start_search": "09-15", "ETP": 5.0, "Cap_ret_maxi": 70, "end_search": "11-15"},
    4: {"zone_name": "Soudan", "date_dry_soil":"01-01", "start_search": "10-01", "ETP": 4.5, "Cap_ret_maxi": 70, "end_search": "11-30"},
    5: {"zone_name": "Golfe_Of_Guinea", "date_dry_soil":"01-01", "start_search": "10-15", "ETP": 4.0, "Cap_ret_maxi": 70, "end_search": "12-01"},
}

# Default class-level criteria dictionary
cessation_dryspell_criteria = {
    0: {"zone_name": "Sahel100_0mm", "start_search1": "06-01", "cumulative": 15, "number_dry_days": 25,
        "thrd_rain_day": 0.85,
        "end_search1": "09-01",
        "nbjour": 30,
        "date_dry_soil": "01-01",
        "start_search2": "09-15",
        "ETP": 5.0,
        "Cap_ret_maxi": 70,
        "end_search2": "10-05"
    },
    1: {"zone_name": "Sahel200_100mm", "start_search1": "05-15", "cumulative": 15, "number_dry_days": 25,
        "thrd_rain_day": 0.85,
        "end_search1": "08-15",
        "nbjour": 40,
        "date_dry_soil": "01-01",
        "start_search2": "09-01",
        "ETP": 5.0,
        "Cap_ret_maxi": 70,
        "end_search2": "10-05"
    },
    2: {
        "zone_name": "Sahel400_200mm",
        "start_search1": "05-01",
        "cumulative": 15,
        "number_dry_days": 20,
        "thrd_rain_day": 0.85,
        "end_search1": "07-31",
        "nbjour": 40,
        "date_dry_soil": "01-01",
        "start_search2": "09-01",
        "ETP": 5.0,
        "Cap_ret_maxi": 70,
        "end_search2": "11-10"
    },
    3: {
        "zone_name": "Sahel600_400mm",
        "start_search1": "03-15",
        "cumulative": 20,
        "number_dry_days": 20,
        "thrd_rain_day": 0.85,
        "end_search1": "07-31",
        "nbjour": 45,
        "date_dry_soil": "01-01",
        "start_search2": "09-15",
        "ETP": 5.0,
        "Cap_ret_maxi": 70,
        "end_search2": "11-15"
    },
    4: {
        "zone_name": "Soudan",
        "start_search1": "03-15",
        "cumulative": 20,
        "number_dry_days": 10,
        "thrd_rain_day": 0.85,
        "end_search1": "07-31",
        "nbjour": 50,
        "date_dry_soil": "01-01",
        "start_search2": "10-01",
        "ETP": 4.5,
        "Cap_ret_maxi": 70,
        "end_search2": "11-30"
    },
    5: {
        "zone_name": "Golfe_Of_Guinea",
        "start_search1": "02-01",
        "cumulative": 20,
        "number_dry_days": 10,
        "thrd_rain_day": 0.85,
        "end_search1": "06-15",
        "nbjour": 50,
        "date_dry_soil": "01-01",
        "start_search2": "10-15",
        "ETP": 4.0,
        "Cap_ret_maxi": 70,
        "end_search2": "12-01"
    },
}

########################## Extended seasonal forecast ##########################################
qmap = WAS_Qmap()
qmap_ = WAS_bias_correction()
def proceed_seasonal_daily_bias_correction(dir_to_save_model, observation, hindcast_files, forecast_files, varname="PRCP", method='QUANT', wet_day=True, qstep=0.01, distr=None, transfun=None):
    hindcast_files_={}
    forecast_files_={}
    os.makedirs(f"{dir_to_save_model}/corrected", exist_ok=True)
    for i in hindcast_files.keys():
        hcst = xr.open_dataset(hindcast_files[i])[varname]
        fcst = xr.open_dataset(forecast_files[i])[varname]
        corrected_hcst = []
        corrected_fcst = []
        save_path_hcst = f"{dir_to_save_model}/corrected/{os.path.basename(hindcast_files[i])}"
        save_path_fcst = f"{dir_to_save_model}/corrected/{os.path.basename(forecast_files[i])}"
        if not os.path.exists(save_path_hcst):
            for month in range(1, 13):
                obs_month = observation.sel(T=observation['T'].dt.month == month).interp(Y=hcst.Y, X=hcst.X, method="linear", kwargs={"fill_value": "extrapolate"})
                mask = xr.where(~np.isnan(obs_month.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
                hcst_month = hcst.sel(T=hcst['T'].dt.month == month)
                obs_month['T'] = hcst_month['T']
                # obs_month, hcst_month = xr.align(obs_month, hcst_month, join='inner')
                fcst_month = fcst.sel(T=fcst['T'].dt.month == month)
                if varname == "PRCP":
                    if method in ['QUANT','RQUANT']:
                        fobj_quant = qmap.fitQmap(obs_month, hcst_month, method=method, wet_day=wet_day, qstep=qstep)
                        hcst_month_corr = qmap.doQmap(hcst_month, fobj_quant, type='linear')
                        fcst_month_corr = qmap.doQmap(fcst_month, fobj_quant, type='linear')
                    elif method == 'SSPLIN':
                        fobj_quant = qmap.fitQmap(obs_month, hcst_month, method=method, wet_day=wet_day, qstep=qstep)
                        hcst_month_corr = qmap.doQmap(hcst_month, fobj_quant)
                        fcst_month_corr = qmap.doQmap(fcst_month, fobj_quant)
                    elif method == 'PTF':
                        fobj_quant = qmap.fitQmap(obs_month, hcst_month, method=method, wet_day=wet_day, qstep=qstep, transfun=transfun)
                        hcst_month_corr = qmap.doQmap(hcst_month, fobj_quant)
                        fcst_month_corr = qmap.doQmap(fcst_month, fobj_quant)
                    elif method == 'DIST':                   
                        fobj_quant = qmap.fitQmap(obs_month, hcst_month, method=method, wet_day=wet_day, qstep=qstep, distr=distr)
                        hcst_month_corr = qmap.doQmap(hcst_month, fobj_quant)
                        fcst_month_corr = qmap.doQmap(fcst_month, fobj_quant)  
                    else:
                        print('please choose method between QUANT','RQUANT','SSPLIN','DIST','PTF')
                else:
                    if method=='QUANT':
                        fobj_quant = qmap_.fitBC(obs_month, hcst_month, method=method, qstep=qstep, nboot=20)
                        hcst_month_corr = qmap_.doBC(hcst_month, fobj_quant, type='linear')
                        fcst_month_corr = qmap_.doBC(fcst_month, fobj_quant, type='linear')
                    elif method in ['MEAN','VARSCALE']:
                        fobj_quant = qmap_.fitBC(obs_month, hcst_month, method=method)
                        hcst_month_corr = qmap_.doBC(hcst_month, fobj_quant)
                        fcst_month_corr = qmap_.doBC(fcst_month, fobj_quant)
                    elif method=='DIST':
                        fobj_quant = qmap_.fitBC(obs_month, hcst_month, method=method, qstep=qstep, distr=distr)
                        hcst_month_corr = qmap_.doBC(hcst_month, fobj_quant)
                        fcst_month_corr = qmap_.doBC(fcst_month, fobj_quant)
                    else:
                        print('please choose method between QUANT','MEAN','VARSCALE','DIST')
                corrected_hcst.append(hcst_month_corr)
                corrected_fcst.append(fcst_month_corr)
            corrected_hcst_ = xr.concat(corrected_hcst, dim='T').sortby('T').fillna(0) * mask
            corrected_fcst_ = xr.concat(corrected_fcst, dim='T').sortby('T').fillna(0) * mask
            corrected_hcst_.to_dataset(name='corrected').to_netcdf(save_path_hcst)
            corrected_fcst_.to_dataset(name='corrected').to_netcdf(save_path_fcst)
            hindcast_files_[i] = save_path_hcst
            forecast_files_[i] = save_path_fcst
        else:
            print(f"[SKIP] {save_path_hcst} already exists.")
            print(f"[SKIP] {save_path_fcst} already exists.")
            hindcast_files_[i] = save_path_hcst
            forecast_files_[i] = save_path_fcst 
    return hindcast_files_, forecast_files_


def pre_process_biophysical_model(dir_to_save, hdcst_file_path, 
fcst_file_path, obs_hdcst, obs_fcst_year, month_of_initialization,
 year_start, year_end, year_forecast, param="PRCP"):
    """
    Process model hindcasts and forecasts data for extended seasonal forecasts with biophysical models (Hype, SARRAO).

    Parameters
    ----------
    dir_to_save : str
        Directory path to save processed data.
    hdcst_file_path : dict
        Dictionary mapping model names to hindcast file paths.
    fcst_file_path : dict
        Dictionary mapping model names to forecast file paths.
    obs_hdcst : xarray.Dataset
        Observational hindcast dataset.
    obs_fcst_year : xarray.Dataset
        Observational forecast dataset for the specified year.
    month_of_initialization : int
        Month of model initialization.
    year_start : int
        Start year of the hindcast data.
    year_end : int
        End year of the hindcast data.
    year_forecast : int
        Forecast year.

    Returns
    -------
    tuple
        Tuple of dictionaries (saved_hindcast_paths, saved_forecast_paths) mapping model names to saved file paths.
    """
    mask = xr.where(~np.isnan(obs_fcst_year.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
    t_coord = pd.date_range(start=f"{year_forecast}-01-01", end=f"{year_forecast}-12-31", freq="D")
    y_coords = obs_fcst_year.Y
    x_coords = obs_fcst_year.X

    if calendar.isleap(year_forecast):
        # dayofyear = obs_hdcst['T'].dt.dayofyear
        daily_climatology = obs_hdcst.groupby('T.dayofyear').mean(dim='T')
        data = daily_climatology.to_numpy()
        dummy = xr.DataArray(
            data=data,
            coords={"T": t_coord, "Y": y_coords, "X": x_coords},
            dims=["T", "Y", "X"]
        ) * mask
    else:
        # da_noleap = obs_hdcst.sel(T=~((obs_hdcst['T'].dt.month == 2) & (obs_hdcst['T'].dt.day == 29)))
        daily_climatology = obs_hdcst.groupby('T.dayofyear').mean(dim='T')
        daily_climatology = daily_climatology.sel(dayofyear=~((daily_climatology['dayofyear'] == 60)))
        data = daily_climatology.to_numpy()
        dummy = xr.DataArray(
            data=data,
            coords={"T": t_coord, "Y": y_coords, "X": x_coords},
            dims=["T", "Y", "X"]
        ) * mask

    abb_mont_ini = calendar.month_abbr[int(month_of_initialization)]
    dir_to_save = Path(f"{dir_to_save}/model_data")
    os.makedirs(dir_to_save, exist_ok=True)
    saved_hindcast_paths = {}
    for i in hdcst_file_path.keys():
        save_path = f"{dir_to_save}/hindcast_{i}_{param}_{abb_mont_ini}Ic.nc"
        if not os.path.exists(save_path):
            hdcst = xr.open_dataset(hdcst_file_path[i])
            if 'number' in hdcst.dims:
                hdcst = hdcst.mean(dim="number")
            hdcst = hdcst.to_array().drop_vars("variable").squeeze()
            obs_hdcst_sel = obs_hdcst.sel(T=slice(str(year_start), str(year_end)))
            obs_hdcst_interp = obs_hdcst_sel.interp(Y=hdcst.Y, X=hdcst.X, method="linear", kwargs={"fill_value": "extrapolate"})
            ds1_aligned, ds2_aligned = xr.align(hdcst, obs_hdcst_interp, join='outer')
            filled_ds = ds1_aligned.fillna(ds2_aligned)
            ds_filled = filled_ds.copy()
            ds_filled = ds_filled.sortby("T")
            if param=="PRCP":
                ds_filled.where(ds_filled >= 0, 0)
            ds_processed = ds_filled.to_dataset(name=param)
            ds_processed = ds_processed.transpose("T", "Y", "X")
            ds_processed.to_netcdf(save_path)
        else:
            print(f"[SKIP] {save_path} already exists.")
        saved_hindcast_paths[i] = save_path
    saved_forecast_paths = {}
    for i in fcst_file_path.keys():
        save_path = f"{dir_to_save}/forecast_{i}_{param}_{abb_mont_ini}Ic.nc"
        if not os.path.exists(save_path):
            fcst = xr.open_dataset(fcst_file_path[i])
            if 'number' in fcst.dims:
                fcst = fcst.mean(dim="number")
            fcst = fcst.to_array().drop_vars("variable").squeeze()
            obs_fcst_sel = obs_fcst_year.sortby("T").sel(T=str(year_forecast))
            obs_fcst_interp = obs_fcst_sel.interp(Y=fcst.Y, X=fcst.X, method="linear", kwargs={"fill_value": "extrapolate"})
            ds1_aligned, ds2_aligned = xr.align(fcst, obs_fcst_interp, join='outer')
            filled_fcst = ds1_aligned.fillna(ds2_aligned)
            ds_filled = filled_fcst.copy()
            ds_filled = ds_filled.sortby("T")
            dummy = dummy.interp(Y=fcst.Y, X=fcst.X, method="linear", kwargs={"fill_value": "extrapolate"})
            ds1_aligned, ds2_aligned = xr.align(ds_filled, dummy, join='outer')
            filled_fcst_ = ds1_aligned.fillna(ds2_aligned)
            ds_filled = filled_fcst_.copy()
            ds_filled = ds_filled.sortby("T")
            if param=="PRCP":
                ds_filled.where(ds_filled >= 0, 0)
            ds_processed = ds_filled.to_dataset(name=param)
            ds_processed = ds_processed.transpose("T", "Y", "X")
            ds_processed.to_netcdf(save_path)
        else:
            print(f"[SKIP] {save_path} already exists.")
        saved_forecast_paths[i] = save_path
    return saved_hindcast_paths, saved_forecast_paths


def extraterrestrial_radiation(lat_da, time):
    """
    Calculate daily extraterrestrial radiation (Ra) following FAO-56 equations (21–23).

    Computes the solar radiation at the top of the Earth's atmosphere, accounting
    for latitude, day of the year, and solar geometry. Handles polar day/night
    conditions by setting radiation to zero during polar night.

    Parameters
    ----------
    lat_da : xarray.DataArray
        2D latitude field in degrees, with dimensions ('Y', 'X').
    time : pandas.DatetimeIndex
        1D array of datetime objects representing the time dimension.

    Returns
    -------
    xarray.DataArray
        Daily extraterrestrial radiation (Ra) with dimensions ('T', 'Y', 'X'),
        units MJ m^-2 day^-1, and attributes for units and long name.

    Notes
    -----
    - Based on FAO-56 Reference Evapotranspiration methodology (Allen et al., 1998).
    - Polar day/night is handled by setting Ra to 0 where the sunset hour angle is undefined.
    - The solar constant used is 0.082 MJ m^-2 min^-1, converted to daily values.

    References
    ----------
    Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998). Crop
    evapotranspiration: Guidelines for computing crop water requirements.
    FAO Irrigation and Drainage Paper 56.
    """
    phi = np.deg2rad(lat_da)  # Latitude in radians (Y, X)
    J = xr.DataArray(time.dayofyear.values, dims="T", coords={"T": time})
    θ = 2 * np.pi * (J - 1) / 365.25  # Day angle (rad)
    dr = 1 + 0.033 * np.cos(θ)  # Relative Earth-Sun distance
    δ = 0.409 * np.sin(θ - 1.39)  # Solar declination (rad)
    ws = np.arccos(xr.where(np.abs(np.tan(phi) * np.tan(δ)) >= 1,
                            np.sign(phi) * np.nan,  # Polar day/night
                            -np.tan(phi) * np.tan(δ)))  # Sunset hour angle (rad)

    # (24*60/π) * G_sc = 37.586 MJ m^-2 day^-1
    Ra = 37.586 * dr * (ws * np.sin(phi) * np.sin(δ) +
                        np.cos(phi) * np.cos(δ) * np.sin(ws))
    Ra = Ra.where(ws.notnull(), 0.)  # Set polar night to 0
    Ra.name = "Ra"
    Ra.attrs = {"units": "MJ m^-2 day^-1", "long_name": "Extraterrestrial radiation (daily)"}
    return Ra

def svp(T):
    """
    Calculate saturated vapor pressure (es) using FAO-56 Equation 3.

    Parameters
    ----------
    T : xarray.DataArray or numpy.ndarray
        Air temperature in degrees Celsius.

    Returns
    -------
    xarray.DataArray or numpy.ndarray
        Saturated vapor pressure in kPa, with same shape as input.

    Notes
    -----
    - Based on the Tetens equation as presented in FAO-56.
    - Assumes input temperature is in Celsius.

    References
    ----------
    Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998). Crop
    evapotranspiration: Guidelines for computing crop water requirements.
    FAO Irrigation and Drainage Paper 56.
    """
    return 0.6108 * np.exp(17.27 * T / (T + 237.3))

def et0_fao56_daily(tmax, tmin, rs, mlsp, dem, tdew=None, u10=None, v10=None, wff=None):
    """
    Compute daily reference evapotranspiration (ET₀) using the FAO-56 Penman-Monteith equation.

    Calculates ET₀ for a hypothetical reference crop (grass, 0.12 m height) based on
    daily meteorological data, following the standardized FAO-56 methodology.

    Parameters
    ----------
    tmax : xarray.DataArray
        Daily maximum temperature in °C, with dimensions ('T', 'Y', 'X').
    tmin : xarray.DataArray
        Daily minimum temperature in °C, with dimensions ('T', 'Y', 'X').
    tdew : xarray.DataArray
        Daily mean dew point temperature in °C, with dimensions ('T', 'Y', 'X').
    u10 : xarray.DataArray, optional
        Daily mean zonal wind speed at 10 m height in m s^-1, with dimensions ('T', 'Y', 'X').
    v10 : xarray.DataArray, optional
        Daily mean meridional wind speed at 10 m height in m s^-1, with dimensions ('T', 'Y', 'X').
    wff : xarray.DataArray, optional
        Daily mean wind speed at 10 m height in m s^-1, with dimensions ('T', 'Y', 'X').
    rs : xarray.DataArray
        Daily mean incoming solar radiation in W m^-2 or MJ m^-2 day^-1, with dimensions ('T', 'Y', 'X').
    mlsp : xarray.DataArray
        Daily mean sea-level pressure in hPa, with dimensions ('T', 'Y', 'X').
    dem : xarray.DataArray
        Digital elevation model (terrain height) in meters, with dimensions ('Y', 'X').

    Returns
    -------
    xarray.DataArray
        Daily reference evapotranspiration (ET₀) in mm day^-1, with dimensions ('T', 'Y', 'X'),
        and attributes for units and long name.

    Notes
    -----
    - All input arrays must be on matching grids with dimensions ('T', 'Y', 'X') except for
      `dem`, which is typically ('Y', 'X') and interpolated to match.
    - Solar radiation input (`rs`) is automatically converted from W m^-2 to MJ m^-2 day^-1
      if values exceed 200 (assumed to be in W m^-2).
    - Wind speed at 10 m is converted to 2 m height using a logarithmic profile.
    - Soil heat flux (G) is assumed to be zero for daily calculations.
    - The albedo for the reference crop is fixed at 0.23.
    - Polar night conditions are handled in the extraterrestrial radiation calculation.

    References
    ----------
    Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998). Crop
    evapotranspiration: Guidelines for computing crop water requirements.
    FAO Irrigation and Drainage Paper 56.
    """
    # --- 0 Compute extraterrestrial radiation ---
    dem = dem.interp_like(tmax, method="linear", kwargs={"fill_value": "extrapolate"})
    lat_da, _ = xr.broadcast(dem["Y"], dem["X"])
    lat_da = lat_da.rename("lat")
    ra = extraterrestrial_radiation(lat_da, tmax["T"].to_index())  # MJ m^-2 day^-1

    # --- 1 Broadcast static fields ---
    if {"Y", "X"} <= set(tmax.dims):
        dem = dem.broadcast_like(tmax)

    # --- 2 Temperature terms ---
    tmean = (tmax + tmin) / 2
    Δ = 4098 * svp(tmean) / (tmean + 237.3)**2  # Slope of svp curve (kPa °C^-1)

    # --- 3 Pressures ---
    P0 = mlsp / 10.0  # hPa to kPa
    P = P0 * (1 - 0.0065 * dem / (tmean + 273.15))**5.257  # Hypsometric equation
    PSI = 0.665e-3 * P  # Psychrometric constant (kPa °C^-1)

    # --- 4 Wind ---
    if wff is not None:
        ws10 = wff
        u2 = ws10 * 4.87 / np.log(67.8 * 10.0 - 5.42)  # Wind speed at 2 m
    elif u10 is not None and v10 is not None: # Use u10 and v10 if available
        ws10 = np.hypot(u10, v10)  # Wind speed at 10 m
        u2 = ws10 * 4.87 / np.log(67.8 * 10.0 - 5.42)  # Wind speed at 2 m
    else:
        raise ValueError("Either 'wff' or both 'u10' and 'v10' must be provided for wind speed calculation.")


    # --- 5 Vapour pressures ---
    if tdew is None:
        # If dew point is not provided, use tmin
        ea = svp(tmin)  # Actual vapor pressure (kPa)
    else:
        ea = svp(tdew)  # Actual vapor pressure (kPa)
    
    es = (svp(tmax) + svp(tmin)) / 2  # Saturated vapor pressure (kPa)
    vpd = es - ea  # Vapor pressure deficit (kPa)

    # --- 6 Radiation ---
    if rs.max() > 200:  # Convert W m^-2 to MJ m^-2 day^-1
        Rs = rs * 86400 / 1e6
    else:
        Rs = rs

    alpha = 0.23  # Albedo
    Rns = (1 - alpha) * Rs  # Net shortwave radiation

    # Clear-sky solar radiation at surface (Eq. 37)
    Rs0 = ra * (0.75 + 2e-5 * dem)
    sigma = 4.903e-9  # Stefan-Boltzmann constant
    # Net longwave radiation (Eq. 39)
    Rnl = (sigma *
           ((tmax + 273.16)**4 + (tmin + 273.16)**4) / 2 *
           (0.34 - 0.14 * np.sqrt(ea)) *
           np.clip(1.35 * Rs / Rs0, 0, 1) - 0.35)
    Rn = Rns - Rnl  # Net radiation
    G = 0.0  # Soil heat flux (daily scale)

    # --- 7 Penman-Monteith ---
    num = 0.408 * Δ * (Rn - G) + PSI * (900 / (tmean + 273)) * u2 * vpd
    den = Δ + PSI * (1 + 0.34 * u2)
    et0 = (num / den).clip(min=0)
    et0 = et0.assign_attrs(units="mm day^-1",
                           long_name="Reference ET0 (FAO-56 PM)")
    et0.name = "ET0"
    return et0
