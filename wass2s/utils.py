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

def trend_data(data):
    """
    Compute trends in data using ExtendedEOF.

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
        season_str = "".join([calendar.month_abbr[(int(i) + int(month_of_initialization)) % 12 or 12] for i in lead_time])
        center = center.lower().replace("_", "")
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

def plot_prob_forecasts(dir_to_save, forecast_prob, model_name, 
                        country_code="GHA", admin_level=0, source="naturalearth",
                        labels=["Below-Normal", "Near-Normal", "Above-Normal"], 
                        reverse_cmap=True, hspace=None, logo=None, 
                        logo_size=("10%", "10%"), logo_position="lower left", 
                        sigma=None, res=None, stations_df=None):

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


    # # 3. Spatial Smoothing
    # if sigma:
    #     for p in forecast_prob.probability:
    #         forecast_prob.loc[{'probability': p}] = gaussian_filter(forecast_prob.sel(probability=p), sigma=sigma)

    # if res is not None:
    #     min_X = forecast_prob['X'].min().values
    #     max_X = forecast_prob['X'].max().values
    #     min_Y = forecast_prob['Y'].min().values
    #     max_Y = forecast_prob['Y'].max().values
    #     num_X = int((max_X - min_X) / res) + 1
    #     num_Y = int((max_Y - min_Y) / res) + 1
    #     new_X = np.linspace(min_X, max_X, num_X)
    #     new_Y = np.linspace(min_Y, max_Y, num_Y)
    #     forecast_prob = forecast_prob.interp(X=new_X, Y=new_Y, method='linear',
    #                                             kwargs={'fill_value': 'extrapolate'}    
    #                                         )

    # 4. Tercile Selection
    max_prob = forecast_prob.max(dim="probability")
    max_cat = forecast_prob.fillna(-1).argmax(dim="probability")
    
    # 5. Global Dynamic Scaling (Calculated ONLY from values inside boundary)
    plotted_floats = max_prob.values.flatten()
    plotted_floats = plotted_floats[~np.isnan(plotted_floats)] * 100
    
    if len(plotted_floats) > 0:
        g_vmin = max(35, np.floor(plotted_floats.min() / 5) * 5)
        g_vmax = min(100, np.ceil(plotted_floats.max() / 5) * 5)
    else:
        g_vmin, g_vmax = 35, 85

    # 6. Plotting Infrastructure
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 3, height_ratios=[10, 0.4], hspace=hspace or -0.5, wspace=0.3)
    ax = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
    
    # Zoom to shapefile
    bounds = gdf.total_bounds
    ax.set_extent([bounds[0]-0.1, bounds[2]+0.1, bounds[1]-0.1, bounds[3]+0.1])

    # Colormaps
    # an_colors = ["#fff7bc", "#662506"] if reverse_cmap else ["#e5f5f9", "#00441b"]
    # bn_colors = ["#e5f5f9", "#00441b"] if reverse_cmap else ["#fff7bc", "#662506"]

    an_colors = ["#fff7bc", "#fee391", "#fec44f", "#fe9929", "#ec7014", "#cc4c02", "#993404", "#662506"] if reverse_cmap else ["#e5f5f9", "#ccece6", "#99d8c9", "#66c2a4", "#41ae76", "#238b45", "#006d2c", "#00441b"]
    bn_colors = ["#e5f5f9", "#ccece6", "#99d8c9", "#66c2a4", "#41ae76", "#238b45", "#006d2c", "#00441b"] if reverse_cmap else ["#fff7bc", "#fee391", "#fec44f", "#fe9929", "#ec7014", "#cc4c02", "#993404", "#662506"]
    
    cmaps = [
        mcolors.LinearSegmentedColormap.from_list('BN', bn_colors),
        mcolors.LinearSegmentedColormap.from_list('NN', ["#d9d9d9", "#525252"]),
        mcolors.LinearSegmentedColormap.from_list('AN', an_colors)
    ]

    # Plot terciles with shared Vmin/Vmax
    data_layers = [
        (max_prob.where(max_cat == 0)*100), 
        (max_prob.where(max_cat == 1)*100), 
        (max_prob.where(max_cat == 2)*100)
    ]
    
    ims = []
    for i, data in enumerate(data_layers):
        im = ax.pcolormesh(forecast_prob.X, forecast_prob.Y, data, 
                           cmap=cmaps[i], vmin=g_vmin, vmax=g_vmax, 
                           transform=ccrs.PlateCarree(), alpha=0.9, zorder=2)
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
                            width=logo_size[0],  # Width of the logo    
                            height=logo_size[1],  # Height of the logo        
                            loc=logo_position,
                            borderpad=0)        
        ax_logo.imshow(mpimg.imread(logo))
        ax_logo.axis("off") 
    if logo is not None:
        logo_size=  ("21%","21%")  # Default logo size if not provided
    else:
        logo_size = logo_size  # Use the provided logo size
        

    # 9. Unified Colorbars
    cbar_ticks = np.linspace(g_vmin, g_vmax, 6).astype(int)
    for i, label in enumerate(labels):
        cax = fig.add_subplot(gs[1, i])
        cb = plt.colorbar(ims[i], cax=cax, orientation='horizontal')
        cb.set_label(f"{label} (%)", fontsize=10)
        cb.set_ticks(cbar_ticks)

    # for i, label in enumerate(labels):
    #         cax = fig.add_subplot(gs[1, i])
    #         cb = plt.colorbar(ims[i], cax=cax, orientation='horizontal')
    #         cb.set_label(f"{label} (%)", fontsize=10, fontweight='bold')
    #         cb.set_ticks(cbar_ticks)
    #         cb.ax.tick_params(labelsize=9, length=0) # Hide default tick marks
            
    #         # --- THE FIX: Draw blank fine lines (white separators) ---
    #         # We normalize the tick values to the 0-1 range of the colorbar axis
    #         tick_locs = (cbar_ticks - g_vmin) / (g_vmax - g_vmin)
    #         cb.ax.vlines(tick_locs, 0, 1, colors='white', linewidth=1.5, zorder=10)
            
    #         # Optional: Outline the colorbar for a "boxed" look
    #         for spine in cb.ax.spines.values():
    #             spine.set_visible(True)
    #             spine.set_linewidth(0.5)

    # 10. Final Aesthetics
    ax.set_title(f"Probabilistic Seasonal Forecast\n{model_name}", fontsize=14, pad=20, fontweight='bold')
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='#e0f3ff')
    ax.coastlines(resolution='10m')

    
    
    plt.savefig(os.path.join(dir_to_save, f"{model_name}.png"), dpi=300, bbox_inches='tight')
    plt.show()

######################################################################################################################
####################### To use after deleting your xcast from the tools scipy 1.10 is a limit right now ##############

# """
# WAS probabilistic tercile plotting utilities (global functions)

# Features
# --------
# 1) Probabilistic tercile map plotting (BN/NN/AN) based on max-prob category + intensity (%)
# 2) Boundary ingestion:
#    - user-provided shapefile path
#    - OR auto-download by country + admin level from:
#        * geoBoundaries (ADM0..ADM5)   [preferred]
#        * GADM 3.6      (ADM0..ADM*)   [fallback]
#        * Natural Earth  (ADM0/ADM1)   [fast/coarse]
# 3) True masking outside polygon (requires regionmask) + visual clipping to polygon edge
# 4) Plot stations from your "wide" table format (LAT/LON rows)
# 5) Colorbar scaling:
#    - global dynamic scale (one vmin/vmax shared by BN/NN/AN) computed from plotted values
#    - optional robust percentiles + hard floors/ceilings to keep comparability

# Dependencies (recommended)
# --------------------------
# - numpy, xarray, matplotlib, cartopy, scipy
# - geopandas + shapely (for reading/writing boundaries)
# - regionmask (for masking outside boundary)
# - requests (for downloading boundaries)
# - pycountry (optional but recommended for name -> ISO3)
# """

# # =============================================================================
# # Country -> ISO3
# # =============================================================================
# def country_to_iso3(country: str) -> str:
#     """
#     Convert country name / ISO2 / ISO3 to ISO3.
#     Uses pycountry if available; otherwise minimal manual fallback.
#     """
#     c = (country or "").strip()
#     if not c:
#         raise ValueError("country must be a non-empty string")

#     if len(c) == 3 and c.isalpha():
#         return c.upper()

#     if len(c) == 2 and c.isalpha():
#         try:
#             import pycountry
#             obj = pycountry.countries.get(alpha_2=c.upper())
#             if obj:
#                 return obj.alpha_3.upper()
#         except Exception:
#             pass

#     try:
#         import pycountry
#         obj = pycountry.countries.lookup(c)
#         return obj.alpha_3.upper()
#     except Exception:
#         manual = {
#             # BENIN
#             "BENIN": "BEN",
#             "RÉPUBLIQUE DU BÉNIN": "BEN",
#             "REPUBLIQUE DU BENIN": "BEN",
        
#             # BURKINA FASO
#             "BURKINA FASO": "BFA",
        
#             # CABO VERDE
#             "CABO VERDE": "CPV",
#             "CAPE VERDE": "CPV",
#             "CAP-VERT": "CPV",
        
#             # COTE D'IVOIRE
#             "COTE D IVOIRE": "CIV",
#             "CÔTE D IVOIRE": "CIV",
#             "COTE D'IVOIRE": "CIV",
#             "CÔTE D'IVOIRE": "CIV",
#             "IVORY COAST": "CIV",
        
#             # GAMBIA
#             "GAMBIA": "GMB",
#             "THE GAMBIA": "GMB",
        
#             # GHANA
#             "GHANA": "GHA",
        
#             # GUINEA
#             "GUINEA": "GIN",
#             "GUINÉE": "GIN",
        
#             # GUINEA-BISSAU
#             "GUINEA-BISSAU": "GNB",
#             "GUINÉE-BISSAU": "GNB",
        
#             # LIBERIA
#             "LIBERIA": "LBR",
        
#             # MALI
#             "MALI": "MLI",
        
#             # MAURITANIA
#             "MAURITANIA": "MRT",
#             "MAURITANIE": "MRT",
        
#             # NIGER
#             "NIGER": "NER",
        
#             # NIGERIA
#             "NIGERIA": "NGA",
        
#             # SENEGAL
#             "SENEGAL": "SEN",
#             "SÉNÉGAL": "SEN",
        
#             # SIERRA LEONE
#             "SIERRA LEONE": "SLE",
        
#             # TOGO
#             "TOGO": "TGO",
        
#             # --- (extra non-West-Africa mappings) ---
#             "DRC": "COD",
#             "CONGO (KINSHASA)": "COD",
#             "CONGO (BRAZZAVILLE)": "COG",
#             "SOUTH KOREA": "KOR",
#             "NORTH KOREA": "PRK",
#         }

#         key = c.upper()
#         if key in manual:
#             return manual[key]
#         raise ValueError(
#             f"Could not resolve ISO3 for country='{country}'. "
#             "Install pycountry (pip install pycountry) or pass ISO3 directly."
#         )


# # =============================================================================
# # Boundary downloaders
# # =============================================================================
# def _download_bytes(url: str, timeout: int = 120) -> bytes:
#     r = requests.get(url, timeout=timeout)
#     r.raise_for_status()
#     return r.content


# def _extract_zip_bytes(zip_bytes: bytes, out_dir: Path, overwrite: bool = False) -> None:
#     out_dir.mkdir(parents=True, exist_ok=True)
#     with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
#         for member in zf.infolist():
#             target = out_dir / member.filename
#             if target.exists() and not overwrite:
#                 continue
#             zf.extract(member, path=out_dir)


# def _find_shps(root: Path) -> List[Path]:
#     return sorted([p for p in root.rglob("*.shp") if p.is_file()])


# def _pick_best_shp(shps: List[Path], admin_level: int) -> Path:
#     if not shps:
#         raise FileNotFoundError("No .shp files found after extraction.")
#     tag = f"ADM{admin_level}".lower()
#     exact = [p for p in shps if tag in p.name.lower()]
#     if exact:
#         return sorted(exact, key=lambda p: (len(p.name), len(str(p))))[0]
#     if len(shps) == 1:
#         return shps[0]
#     return shps[0]


# def _naturalearth_dataset_name(admin_level: int, with_lakes: bool = True) -> Tuple[str, str]:
#     if admin_level == 0:
#         return ("cultural", "admin_0_countries")
#     if admin_level == 1:
#         return ("cultural", "admin_1_states_provinces_lakes" if with_lakes else "admin_1_states_provinces")
#     raise ValueError("Natural Earth supports only admin_level 0 or 1.")


# def _download_naturalearth_country_shp(
#     iso3: str,
#     admin_level: int,
#     out_dir: Path,
#     resolution: str = "10m",
#     with_lakes: bool = True,
#     overwrite: bool = False,
# ) -> Path:
#     """
#     Uses cartopy's Natural Earth downloader, then filters to one country and writes a country-only shapefile.
#     """
#     if gpd is None:
#         raise ImportError("geopandas is required for Natural Earth filtering/writing.")
#     try:
#         from cartopy.io import shapereader as shpreader
#     except Exception as e:
#         raise ImportError("cartopy is required for Natural Earth downloads.") from e

#     category, name = _naturalearth_dataset_name(admin_level, with_lakes=with_lakes)
#     ne_path = Path(shpreader.natural_earth(resolution=resolution, category=category, name=name))
#     gdf = gpd.read_file(ne_path)

#     iso3u = iso3.upper()
#     # Try typical NE fields
#     candidates = [("ADM0_A3", iso3u), ("ISO_A3", iso3u), ("adm0_a3", iso3u), ("iso_a3", iso3u)]
#     mask = None
#     for col, val in candidates:
#         if col in gdf.columns:
#             m = gdf[col].astype(str).str.upper() == val
#             if m.any():
#                 mask = m
#                 break
#     if mask is None:
#         raise ValueError(f"Natural Earth: cannot filter by ISO3='{iso3u}'. Available columns: {list(gdf.columns)}")

#     sub = gdf.loc[mask].copy()
#     if sub.empty:
#         raise ValueError(f"Natural Earth: no features found for ISO3='{iso3u}' at ADM{admin_level}.")

#     if sub.crs is not None and str(sub.crs).lower() not in ["epsg:4326", "wgs84"]:
#         sub = sub.to_crs("EPSG:4326")

#     target_dir = out_dir / f"naturalearth_{resolution}_{iso3u}_ADM{admin_level}"
#     target_dir.mkdir(parents=True, exist_ok=True)
#     shp_out = target_dir / f"naturalearth_{resolution}_{iso3u}_ADM{admin_level}.shp"

#     if shp_out.exists() and not overwrite:
#         return shp_out

#     sub.to_file(shp_out, driver="ESRI Shapefile")
#     return shp_out


# def download_admin_shapefile(
#     country: str,
#     admin_level: int = 0,
#     out_dir: str | Path = "boundaries_cache",
#     source: Literal["auto", "geoboundaries", "gadm36", "naturalearth"] = "auto",
#     geoboundaries_release: Literal["gbOpen", "gbHumanitarian", "gbAuthoritative"] = "gbOpen",
#     naturalearth_resolution: Literal["10m", "50m", "110m"] = "10m",
#     naturalearth_with_lakes: bool = True,
#     overwrite: bool = False,
#     timeout: int = 120,
# ) -> Path:
#     """
#     Download and extract a boundary shapefile given country + admin_level.
#     - auto: try geoBoundaries -> GADM -> Natural Earth (ADM0/1 only)
#     - geoboundaries: ADM0..ADM5
#     - gadm36: ADM0..ADM* (zip includes multiple levels; we select the requested)
#     - naturalearth: ADM0/ADM1 only (fast/coarse)
#     """
#     iso3 = country_to_iso3(country)
#     out_dir = Path(out_dir)
#     admin_level = int(admin_level)
#     adm = f"ADM{admin_level}"

#     def _try_geoboundaries() -> Path:
#         api = f"https://www.geoboundaries.org/api/current/{geoboundaries_release}/{iso3}/{adm}/"
#         meta = requests.get(api, timeout=timeout)
#         meta.raise_for_status()
#         js = meta.json()
#         zip_url = js.get("staticDownloadLink") or js.get("downloadURL") or js.get("staticDownloadURL")
#         if not zip_url and isinstance(js, list) and js and isinstance(js[0], dict):
#             zip_url = js[0].get("staticDownloadLink") or js[0].get("downloadURL")
#         if not zip_url:
#             raise RuntimeError(f"geoBoundaries API returned no zip link for {iso3} {adm}")

#         target_dir = out_dir / f"geoboundaries_{geoboundaries_release}_{iso3}_{adm}"
#         if target_dir.exists() and not overwrite:
#             shps = _find_shps(target_dir)
#             if shps:
#                 return _pick_best_shp(shps, admin_level)

#         zbytes = _download_bytes(zip_url, timeout=timeout)
#         _extract_zip_bytes(zbytes, target_dir, overwrite=overwrite)
#         return _pick_best_shp(_find_shps(target_dir), admin_level)

#     def _try_gadm36() -> Path:
#         gadm_zip = f"https://geodata.ucdavis.edu/gadm/gadm3.6/shp/gadm36_{iso3}_shp.zip"
#         target_dir = out_dir / f"gadm36_{iso3}"
#         if target_dir.exists() and not overwrite:
#             shps = _find_shps(target_dir)
#             if shps:
#                 return _pick_best_shp(shps, admin_level)

#         zbytes = _download_bytes(gadm_zip, timeout=timeout)
#         _extract_zip_bytes(zbytes, target_dir, overwrite=overwrite)
#         return _pick_best_shp(_find_shps(target_dir), admin_level)

#     def _try_naturalearth() -> Path:
#         if admin_level not in (0, 1):
#             raise ValueError("naturalearth supports only admin_level 0 or 1.")
#         return _download_naturalearth_country_shp(
#             iso3=iso3,
#             admin_level=admin_level,
#             out_dir=out_dir,
#             resolution=naturalearth_resolution,
#             with_lakes=naturalearth_with_lakes,
#             overwrite=overwrite,
#         )

#     if source == "geoboundaries":
#         return _try_geoboundaries()
#     if source == "gadm36":
#         return _try_gadm36()
#     if source == "naturalearth":
#         return _try_naturalearth()

#     # auto
#     errors = []
#     try:
#         return _try_geoboundaries()
#     except Exception as e:
#         errors.append(f"geoBoundaries failed: {e}")
#     try:
#         return _try_gadm36()
#     except Exception as e:
#         errors.append(f"GADM36 failed: {e}")
#     try:
#         return _try_naturalearth()
#     except Exception as e:
#         errors.append(f"NaturalEarth failed: {e}")

#     raise RuntimeError("All boundary sources failed:\n- " + "\n- ".join(errors))


# # =============================================================================
# # Boundary geometry + masking
# # =============================================================================
# def load_boundary_geometry(boundary_shp: str | Path) -> Any:
#     """
#     Read boundary shapefile, dissolve to a single geometry, reproject to EPSG:4326 if needed.
#     Returns a shapely geometry (MultiPolygon/Polygon).
#     """
#     if gpd is None or shapely_to_path is None:
#         raise ImportError("geopandas + cartopy.mpl.path (shapely_to_path) are required for boundary operations.")

#     gdf = gpd.read_file(boundary_shp)
#     if gdf.empty:
#         raise ValueError(f"Shapefile is empty: {boundary_shp}")

#     if gdf.crs is not None and str(gdf.crs).lower() not in ["epsg:4326", "wgs84"]:
#         gdf = gdf.to_crs("EPSG:4326")

#     return gdf.unary_union


# def mask_forecast_to_geometry(forecast_prob: xr.DataArray, geom: Any) -> xr.DataArray:
#     """
#     Mask forecast_prob outside polygon geometry using regionmask.
#     Requires regionmask. If missing, returns unchanged.
#     """
#     if geom is None or regionmask is None:
#         return forecast_prob

#     lon = forecast_prob["X"].values
#     lat = forecast_prob["Y"].values

#     regs = regionmask.Regions([geom], names=["boundary"], abbrevs=["B"])
#     m = regs.mask(lon, lat)  # (Y,X): inside==0 else NaN
#     return forecast_prob.where(m == 0)


# # =============================================================================
# # Station parsing + plotting
# # =============================================================================
# def parse_station_table(stations_df) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Supports your wide format:
#         columns: ['STATION', 'Abetifi', 'Aburi', ...]
#         rows include STATION == 'LAT' and 'LON'
#     Returns: names, lons, lats (finite only)
#     """
#     import pandas as pd
#     if stations_df is None:
#         return np.array([]), np.array([]), np.array([])
#     if not isinstance(stations_df, pd.DataFrame):
#         raise TypeError("stations_df must be a pandas.DataFrame")

#     cols = list(stations_df.columns)
#     if "STATION" in cols:
#         st = stations_df.copy()
#         st["STATION"] = st["STATION"].astype(str)

#         lat_row = st.loc[st["STATION"].str.upper() == "LAT"]
#         lon_row = st.loc[st["STATION"].str.upper() == "LON"]
#         if lat_row.empty or lon_row.empty:
#             raise ValueError("stations_df has 'STATION' but LAT/LON rows are missing.")

#         lat_row = lat_row.iloc[0]
#         lon_row = lon_row.iloc[0]
#         names = [c for c in cols if c != "STATION"]

#         lats = np.array([float(lat_row[n]) for n in names], dtype=float)
#         lons = np.array([float(lon_row[n]) for n in names], dtype=float)
#         ok = np.isfinite(lats) & np.isfinite(lons)
#         return np.array(names)[ok], lons[ok], lats[ok]

#     # Tidy fallback: require LAT/LON columns
#     if {"LAT", "LON"}.issubset(set(cols)):
#         name_col = "STATION" if "STATION" in cols else ("NAME" if "NAME" in cols else None)
#         names = stations_df[name_col].astype(str).values if name_col else np.array([""] * len(stations_df))
#         lats = stations_df["LAT"].astype(float).values
#         lons = stations_df["LON"].astype(float).values
#         ok = np.isfinite(lats) & np.isfinite(lons)
#         return names[ok], lons[ok], lats[ok]

#     raise ValueError("stations_df format not recognized. Provide wide format or LAT/LON columns.")


# # =============================================================================
# # Colorbar scaling (global)
# # =============================================================================
# def global_cbar_stats(
#     arrays_2d: List[np.ndarray],
#     q: Tuple[float, float] = (10, 95),
#     floor: Optional[float] = 35.0,
#     ceil: Optional[float] = 85.0,
#     nticks: int = 6,
# ) -> Tuple[Optional[float], Optional[float], Optional[np.ndarray]]:
#     """
#     Compute ONE vmin/vmax/ticks from finite values of multiple 2D arrays.
#     """
#     vals = []
#     for a in arrays_2d:
#         v = np.asarray(a).ravel()
#         v = v[np.isfinite(v)]
#         if v.size:
#             vals.append(v)
#     if not vals:
#         return None, None, None

#     v = np.concatenate(vals)
#     vmin = np.nanpercentile(v, q[0])
#     vmax = np.nanpercentile(v, q[1])

#     if floor is not None:
#         vmin = max(vmin, float(floor))
#     if ceil is not None:
#         vmax = min(vmax, float(ceil))

#     if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
#         return None, None, None

#     ticks = np.linspace(vmin, vmax, nticks)
#     return float(vmin), float(vmax), ticks


# # =============================================================================
# # Probabilistic plot
# # =============================================================================
# # def plot_prob_forecasts(
# #     dir_to_save: str | Path,
# #     forecast_prob: xr.DataArray,
# #     model_name: str,
# #     labels: Tuple[str, str, str] = ("Below-Normal", "Near-Normal", "Above-Normal"),
# #     reverse_cmap: bool = True,
# #     hspace: Optional[float] = None,
# #     sigma: Optional[float] = None,
# #     res: Optional[float] = None,
# #     # boundary options
# #     boundary_shp: Optional[str | Path] = None,
# #     boundary_auto: Optional[Dict[str, Any]] = None,
# #     # boundary drawing options
# #     clip_to_boundary: bool = True,
# #     mask_outside_boundary: bool = True,
# #     boundary_edgecolor: str = "black",
# #     boundary_linewidth: float = 1.2,
# #     # stations
# #     stations_df=None,
# #     plot_stations: bool = False,
# #     label_stations: bool = False,
# #     station_marker: str = "^",
# #     station_size: float = 35,
# #     station_facecolor: str = "black",
# #     station_edgecolor: str = "white",
# #     station_linewidth: float = 0.6,
# #     station_zorder: int = 6,
# #     station_label_fontsize: int = 7,
# #     station_label_offset: Tuple[float, float] = (0.03, 0.03),
# #     # global colorbar scaling
# #     dynamic_cbar_global: bool = True,
# #     cbar_q: Tuple[float, float] = (10, 95),
# #     cbar_floor: Optional[float] = 35.0,
# #     cbar_ceil: Optional[float] = 85.0,
# #     cbar_nticks: int = 6,
# #     # map aesthetics
# #     land_facecolor: str = "#fde0dd",
# #     ocean_facecolor: str = "lightblue",
# #     # logo
# #     logo: Optional[str | Path] = None,
# #     logo_size: Tuple[Optional[str], Optional[str]] = (None, None),  # e.g. ("7%","21%")
# #     logo_position: str = "lower left",
# #     output_ext: Literal["pdf", "png"] = "pdf",
# #     dpi: int = 300,
# # ) -> None:
# #     """
# #     Plot probabilistic tercile forecasts with boundary clip/mask, station overlay, and global colorbar scaling.

# #     boundary_auto example:
# #     ----------------------
# #     boundary_auto = dict(
# #         country="Ghana",
# #         admin_level=0,
# #         source="auto",  # "geoboundaries"|"gadm36"|"naturalearth"|"auto"
# #         out_dir="boundaries_cache",
# #         geoboundaries_release="gbOpen",
# #         naturalearth_resolution="10m",
# #         overwrite=False,
# #         timeout=120,
# #     )
# #     """
# #     dir_to_save = Path(dir_to_save)
# #     dir_to_save.mkdir(parents=True, exist_ok=True)

# #     # -----------------------------
# #     # Optional interpolation
# #     # -----------------------------
# #     if res is not None:
# #         min_X = float(forecast_prob["X"].min().values)
# #         max_X = float(forecast_prob["X"].max().values)
# #         min_Y = float(forecast_prob["Y"].min().values)
# #         max_Y = float(forecast_prob["Y"].max().values)
# #         num_X = int((max_X - min_X) / res) + 1
# #         num_Y = int((max_Y - min_Y) / res) + 1
# #         forecast_prob = forecast_prob.interp(
# #             X=np.linspace(min_X, max_X, num_X),
# #             Y=np.linspace(min_Y, max_Y, num_Y),
# #             method="linear",
# #             kwargs={"fill_value": "extrapolate"},
# #         )

# #     # -----------------------------
# #     # Optional smoothing
# #     # -----------------------------
# #     if sigma is not None:
# #         fp_sm = forecast_prob * 0.0
# #         for p in forecast_prob.probability.values:
# #             layer = forecast_prob.sel(probability=p)
# #             layer_sm = xr.apply_ufunc(
# #                 gaussian_filter,
# #                 layer,
# #                 input_core_dims=[["Y", "X"]],
# #                 output_core_dims=[["Y", "X"]],
# #                 kwargs={"sigma": sigma},
# #             )
# #             fp_sm.loc[{"probability": p}] = layer_sm
# #         s = fp_sm.sum("probability")
# #         forecast_prob = fp_sm / s.where(s != 0, 1.0)

# #     # -----------------------------
# #     # Boundary: explicit or auto
# #     # -----------------------------
# #     if boundary_shp is None and boundary_auto:
# #         shp = download_admin_shapefile(**boundary_auto)
# #         boundary_shp = shp

# #     geom = None
# #     if boundary_shp is not None:
# #         geom = load_boundary_geometry(boundary_shp)
# #         if mask_outside_boundary:
# #             forecast_prob = mask_forecast_to_geometry(forecast_prob, geom)

# #     # -----------------------------
# #     # Max category and max prob
# #     # -----------------------------
# #     max_prob = forecast_prob.max(dim="probability", skipna=True)
# #     max_category = forecast_prob.fillna(-9999).argmax(dim="probability")
# #     mask_bn = (max_category == 0)
# #     mask_nn = (max_category == 1)
# #     mask_an = (max_category == 2)

# #     # -----------------------------
# #     # Colormaps
# #     # -----------------------------
# #     if reverse_cmap:
# #         AN_cmap = mcolors.LinearSegmentedColormap.from_list(
# #             "AN", ["#fff7bc", "#fee391", "#fec44f", "#fe9929", "#ec7014", "#cc4c02", "#993404", "#662506"]
# #         )
# #         NN_cmap = mcolors.LinearSegmentedColormap.from_list(
# #             "NN", ["#d9d9d9", "#bdbdbd", "#969696", "#737373", "#525252"]
# #         )
# #         BN_cmap = mcolors.LinearSegmentedColormap.from_list(
# #             "BN", ["#e5f5f9", "#ccece6", "#99d8c9", "#66c2a4", "#41ae76", "#238b45", "#006d2c", "#00441b"]
# #         )
# #     else:
# #         BN_cmap = mcolors.LinearSegmentedColormap.from_list(
# #             "BN", ["#fff7bc", "#fee391", "#fec44f", "#fe9929", "#ec7014", "#cc4c02", "#993404", "#662506"]
# #         )
# #         NN_cmap = mcolors.LinearSegmentedColormap.from_list(
# #             "NN", ["#d9d9d9", "#bdbdbd", "#969696", "#737373", "#525252"]
# #         )
# #         AN_cmap = mcolors.LinearSegmentedColormap.from_list(
# #             "AN", ["#e5f5f9", "#ccece6", "#99d8c9", "#66c2a4", "#41ae76", "#238b45", "#006d2c", "#00441b"]
# #         )

# #     bn_data = (max_prob.where(mask_bn) * 100).values
# #     nn_data = (max_prob.where(mask_nn) * 100).values
# #     an_data = (max_prob.where(mask_an) * 100).values

# #     # -----------------------------
# #     # Global colorbar scaling
# #     # -----------------------------
# #     if dynamic_cbar_global:
# #         vmin, vmax, ticks = global_cbar_stats(
# #             [bn_data, nn_data, an_data],
# #             q=cbar_q,
# #             floor=cbar_floor,
# #             ceil=cbar_ceil,
# #             nticks=cbar_nticks,
# #         )
# #     else:
# #         vmin, vmax, ticks = 35.0, 85.0, np.arange(35, 86, 5)

# #     # -----------------------------
# #     # Figure layout
# #     # -----------------------------
# #     if hspace is None:
# #         hspace = -0.6
# #     if logo is not None and logo_size == (None, None):
# #         logo_size = ("7%", "21%")

# #     import matplotlib.gridspec as gridspec
# #     fig = plt.figure(figsize=(10, 8))
# #     gs = gridspec.GridSpec(
# #         2, 3,
# #         height_ratios=[10, 0.2],
# #         width_ratios=[1.2, 0.6, 1.2],
# #         hspace=hspace,
# #         wspace=0.2
# #     )

# #     ax = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
# #     gl = ax.gridlines(draw_labels=True, linewidth=0.05, color="gray", alpha=0.8)
# #     gl.top_labels = False
# #     gl.right_labels = False

# #     # Extent by boundary
# #     if geom is not None:
# #         minx, miny, maxx, maxy = geom.bounds
# #         ax.set_extent([minx, maxx, miny, maxy], crs=ccrs.PlateCarree())

# #     # Base map
# #     ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=land_facecolor, edgecolor="black", zorder=0)
# #     ax.add_feature(cfeature.OCEAN, facecolor=ocean_facecolor)
# #     ax.coastlines()
# #     ax.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=1.0, linestyle="solid")

# #     # Boundary outline + clip
# #     if geom is not None:
# #         ax.add_geometries([geom], crs=ccrs.PlateCarree(),
# #                           facecolor="none", edgecolor=boundary_edgecolor,
# #                           linewidth=boundary_linewidth, zorder=5)
# #         if clip_to_boundary:
# #             ax.set_boundary(shapely_to_path(geom), transform=ccrs.PlateCarree())

# #     # Helper: return a mappable even if empty
# #     def _plot_layer(data2d, cmap):
# #         if vmin is None or vmax is None or not np.any(np.isfinite(data2d)):
# #             sm = cm.ScalarMappable(norm=plt.Normalize(vmin=0, vmax=1), cmap=cmap)
# #             sm.set_array([])
# #             return sm
# #         return ax.pcolormesh(
# #             forecast_prob["X"], forecast_prob["Y"], data2d,
# #             cmap=cmap, transform=ccrs.PlateCarree(),
# #             alpha=0.9, vmin=vmin, vmax=vmax
# #         )

# #     bn_plot = _plot_layer(bn_data, BN_cmap)
# #     nn_plot = _plot_layer(nn_data, NN_cmap)
# #     an_plot = _plot_layer(an_data, AN_cmap)

# #     # Stations
# #     if plot_stations and stations_df is not None:
# #         names, lons, lats = parse_station_table(stations_df)
# #         ax.scatter(
# #             lons, lats,
# #             transform=ccrs.PlateCarree(),
# #             s=station_size,
# #             marker=station_marker,
# #             facecolor=station_facecolor,
# #             edgecolor=station_edgecolor,
# #             linewidth=station_linewidth,
# #             zorder=station_zorder,
# #         )
# #         if label_stations:
# #             dx, dy = station_label_offset
# #             for nm, x, y in zip(names, lons, lats):
# #                 ax.text(
# #                     x + dx, y + dy, str(nm),
# #                     transform=ccrs.PlateCarree(),
# #                     fontsize=station_label_fontsize,
# #                     zorder=station_zorder + 1,
# #                     bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.6, linewidth=0.0),
# #                 )

# #     # Title
# #     ax.set_title(str(model_name), fontsize=13, pad=20)

# #     # Colorbars: same scale
# #     def _apply_cbar(mappable, cax, label):
# #         cb = plt.colorbar(mappable, cax=cax, orientation="horizontal")
# #         cb.set_label(label)
# #         if ticks is not None:
# #             cb.set_ticks(ticks)
# #             cb.set_ticklabels([f"{t:.0f}" for t in ticks])
# #         return cb

# #     cbar_ax_bn = fig.add_subplot(gs[1, 0])
# #     cbar_ax_nn = fig.add_subplot(gs[1, 1])
# #     cbar_ax_an = fig.add_subplot(gs[1, 2])

# #     _apply_cbar(bn_plot, cbar_ax_bn, f"{labels[0]} (%)")
# #     _apply_cbar(nn_plot, cbar_ax_nn, f"{labels[1]} (%)")
# #     _apply_cbar(an_plot, cbar_ax_an, f"{labels[2]} (%)")

# #     # Logo
# #     if logo is not None:
# #         if mpimg is None:
# #             raise ImportError("matplotlib.image is required for logo plotting.")
# #         ax_logo = inset_axes(ax, width=logo_size[0], height=logo_size[1], loc=logo_position, borderpad=0.1)
# #         ax_logo.imshow(mpimg.imread(str(logo)))
# #         ax_logo.axis("off")

# #     plt.subplots_adjust(top=0.95, bottom=0.08, left=0.06, right=0.94, hspace=-0.6, wspace=0.2)

# #     out = dir_to_save / f"{str(model_name).replace(' ', '_')}.{output_ext}"
# #     plt.savefig(out, dpi=dpi, bbox_inches="tight")
# #     plt.show()

# def plot_prob_forecasts(
#     dir_to_save: str | Path,
#     forecast_prob: xr.DataArray,
#     model_name: str,
#     labels: Tuple[str, str, str] = ("Below-Normal", "Near-Normal", "Above-Normal"),
#     reverse_cmap: bool = True,
#     hspace: Optional[float] = None,
#     sigma: Optional[float] = None,
#     res: Optional[float] = None,
#     # boundary options
#     boundary_shp: Optional[str | Path] = None,
#     boundary_auto: Optional[Dict[str, Any]] = None,
#     # boundary drawing options
#     clip_to_boundary: bool = True,
#     mask_outside_boundary: bool = True,
#     boundary_edgecolor: str = "black",
#     boundary_linewidth: float = 1.2,
#     # stations
#     stations_df=None,
#     plot_stations: bool = False,
#     label_stations: bool = False,
#     station_marker: str = "^",
#     station_size: float = 35,
#     station_facecolor: str = "black",
#     station_edgecolor: str = "white",
#     station_linewidth: float = 0.6,
#     station_zorder: int = 6,
#     station_label_fontsize: int = 7,
#     station_label_offset: Tuple[float, float] = (0.03, 0.03),
#     # global colorbar scaling
#     dynamic_cbar_global: bool = True,
#     cbar_q: Tuple[float, float] = (10, 95),
#     cbar_floor: Optional[float] = 35.0,
#     cbar_ceil: Optional[float] = 85.0,
#     cbar_nticks: int = 6,
#     # map aesthetics
#     land_facecolor: str = "#fde0dd",
#     ocean_facecolor: str = "lightblue",
#     # logo
#     logo: Optional[str | Path] = None,
#     logo_size: Tuple[Optional[str], Optional[str]] = (None, None),
#     logo_position: str = "lower left",
#     output_ext: Literal["pdf", "png"] = "pdf",
#     dpi: int = 300,
# ) -> None:
#     """
#     Plot probabilistic tercile forecasts (Final Dimension Fix).
#     Aggressively slices extra dimensions to ensure strictly 2D spatial data.
#     """
#     dir_to_save = Path(dir_to_save)
#     dir_to_save.mkdir(parents=True, exist_ok=True)

#     # =========================================================================
#     # 1. NUCLEAR DIMENSION SANITIZATION
#     # =========================================================================
#     # Remove singleton dims
#     forecast_prob = forecast_prob.squeeze(drop=True)

#     # If we have lat/lon but no Y/X, rename them first
#     if "lat" in forecast_prob.dims and "Y" not in forecast_prob.dims:
#         forecast_prob = forecast_prob.rename({"lat": "Y"})
#     if "lon" in forecast_prob.dims and "X" not in forecast_prob.dims:
#         forecast_prob = forecast_prob.rename({"lon": "X"})

#     # Now, find ANY dimension that is not 'probability', 'Y', or 'X'
#     # and forcibly slice it to index 0. This kills the 80x50x80x50 monster.
#     keep_dims = {"probability", "Y", "X"}
#     extra_dims = [d for d in forecast_prob.dims if d not in keep_dims]
    
#     if extra_dims:
#         print(f"Sanitizing: Dropping extra dimensions {extra_dims} by selecting index 0.")
#         # We create a selector dict to slice all of them at once
#         selector = {d: 0 for d in extra_dims}
#         forecast_prob = forecast_prob.isel(selector, drop=True)

#     # Final check: Ensure we have exactly the dims we expect
#     if not {"Y", "X"}.issubset(set(forecast_prob.dims)):
#         raise ValueError(f"Could not identify Y/X dimensions. Found: {forecast_prob.dims}")
        
#     # Ensure standard order
#     if "probability" in forecast_prob.dims:
#         forecast_prob = forecast_prob.transpose("probability", "Y", "X", ...)

#     # =========================================================================
#     # 2. Interpolation
#     # =========================================================================
#     if res is not None:
#         min_X = float(forecast_prob["X"].min().values)
#         max_X = float(forecast_prob["X"].max().values)
#         min_Y = float(forecast_prob["Y"].min().values)
#         max_Y = float(forecast_prob["Y"].max().values)
#         num_X = int((max_X - min_X) / res) + 1
#         num_Y = int((max_Y - min_Y) / res) + 1
#         forecast_prob = forecast_prob.interp(
#             X=np.linspace(min_X, max_X, num_X),
#             Y=np.linspace(min_Y, max_Y, num_Y),
#             method="linear",
#             kwargs={"fill_value": "extrapolate"},
#         )

#     # =========================================================================
#     # 3. Smoothing
#     # =========================================================================
#     if sigma is not None:
#         fp_sm = forecast_prob * 0.0
#         for p in forecast_prob.probability.values:
#             layer = forecast_prob.sel(probability=p)
#             layer_sm = xr.apply_ufunc(
#                 gaussian_filter, layer,
#                 input_core_dims=[["Y", "X"]], output_core_dims=[["Y", "X"]],
#                 kwargs={"sigma": sigma},
#             )
#             fp_sm.loc[{"probability": p}] = layer_sm
#         s = fp_sm.sum("probability")
#         forecast_prob = fp_sm / s.where(s != 0, 1.0)

#     # =========================================================================
#     # 4. Boundary Logic
#     # =========================================================================
#     if boundary_shp is None and boundary_auto:
#         shp = download_admin_shapefile(**boundary_auto)
#         boundary_shp = shp

#     geom = None
#     if boundary_shp is not None:
#         geom = load_boundary_geometry(boundary_shp)
#         if mask_outside_boundary:
#             forecast_prob = mask_forecast_to_geometry(forecast_prob, geom)

#     # =========================================================================
#     # 5. Extract Data
#     # =========================================================================
#     max_prob = forecast_prob.max(dim="probability", skipna=True)
#     max_category = forecast_prob.fillna(-9999).argmax(dim="probability")
    
#     mask_bn = (max_category == 0)
#     mask_nn = (max_category == 1)
#     mask_an = (max_category == 2)

#     bn_data = (max_prob.where(mask_bn) * 100).values
#     nn_data = (max_prob.where(mask_nn) * 100).values
#     an_data = (max_prob.where(mask_an) * 100).values

#     # Safety: Ensure strictly 2D (Handle any residual sizing issues)
#     ny, nx = forecast_prob.sizes["Y"], forecast_prob.sizes["X"]
    
#     def _force_2d_strict(arr, name):
#         # If it matches exactly, good
#         if arr.shape == (ny, nx): return arr
#         # If size matches, reshape (e.g. flattened)
#         if arr.size == ny * nx: return arr.reshape(ny, nx)
#         # If size is HUGE (outer product), we failed to sanitize earlier. 
#         # But since we added the "Nuclear Option" in Step 1, this *should* be impossible.
#         # Just in case, we try to squeeze.
#         sq = np.squeeze(arr)
#         if sq.shape == (ny, nx): return sq
        
#         raise ValueError(
#             f"{name} shape {arr.shape} is incompatible with grid ({ny}, {nx}). "
#             f"Size: {arr.size} vs Expected: {ny*nx}."
#         )

#     bn_data = _force_2d_strict(bn_data, "BN")
#     nn_data = _force_2d_strict(nn_data, "NN")
#     an_data = _force_2d_strict(an_data, "AN")

#     # =========================================================================
#     # 6. Colormaps & Plotting
#     # =========================================================================
#     if reverse_cmap:
#         AN_cmap = mcolors.LinearSegmentedColormap.from_list("AN", ["#fff7bc", "#fee391", "#fec44f", "#fe9929", "#ec7014", "#cc4c02", "#993404", "#662506"])
#         NN_cmap = mcolors.LinearSegmentedColormap.from_list("NN", ["#d9d9d9", "#bdbdbd", "#969696", "#737373", "#525252"])
#         BN_cmap = mcolors.LinearSegmentedColormap.from_list("BN", ["#e5f5f9", "#ccece6", "#99d8c9", "#66c2a4", "#41ae76", "#238b45", "#006d2c", "#00441b"])
#     else:
#         BN_cmap = mcolors.LinearSegmentedColormap.from_list("BN", ["#fff7bc", "#fee391", "#fec44f", "#fe9929", "#ec7014", "#cc4c02", "#993404", "#662506"])
#         NN_cmap = mcolors.LinearSegmentedColormap.from_list("NN", ["#d9d9d9", "#bdbdbd", "#969696", "#737373", "#525252"])
#         AN_cmap = mcolors.LinearSegmentedColormap.from_list("AN", ["#e5f5f9", "#ccece6", "#99d8c9", "#66c2a4", "#41ae76", "#238b45", "#006d2c", "#00441b"])

#     if dynamic_cbar_global:
#         vmin, vmax, ticks = global_cbar_stats([bn_data, nn_data, an_data], q=cbar_q, floor=cbar_floor, ceil=cbar_ceil, nticks=cbar_nticks)
#     else:
#         vmin, vmax, ticks = 35.0, 85.0, np.arange(35, 86, 5)

#     if hspace is None: hspace = -0.6
#     if logo_size == (None, None): logo_size = ("7%", "21%")

#     import matplotlib.gridspec as gridspec
#     fig = plt.figure(figsize=(10, 8))
#     gs = gridspec.GridSpec(2, 3, height_ratios=[10, 0.2], width_ratios=[1.2, 0.6, 1.2], hspace=hspace, wspace=0.2)
#     ax = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())

#     # Map features
#     if geom:
#         minx, miny, maxx, maxy = geom.bounds
#         ax.set_extent([minx, maxx, miny, maxy], crs=ccrs.PlateCarree())
#     ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=land_facecolor, edgecolor="black", zorder=0)
#     ax.add_feature(cfeature.OCEAN, facecolor=ocean_facecolor)
#     ax.coastlines()
#     ax.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=1.0)
#     if geom:
#         ax.add_geometries([geom], crs=ccrs.PlateCarree(), facecolor="none", edgecolor=boundary_edgecolor, linewidth=boundary_linewidth, zorder=5)
#         if clip_to_boundary:
#             ax.set_boundary(shapely_to_path(geom), transform=ccrs.PlateCarree())

#     # Plot Helper
#     def _plot_layer(data2d, cmap):
#         if vmin is None or vmax is None or not np.any(np.isfinite(data2d)):
#             return None
#         # Explicitly pass .values to avoid xarray metadata issues
#         return ax.pcolormesh(
#             forecast_prob["X"].values, forecast_prob["Y"].values, data2d,
#             cmap=cmap, transform=ccrs.PlateCarree(),
#             alpha=0.9, vmin=vmin, vmax=vmax
#         )

#     bn_pl = _plot_layer(bn_data, BN_cmap)
#     nn_pl = _plot_layer(nn_data, NN_cmap)
#     an_pl = _plot_layer(an_data, AN_cmap)

#     # Stations
#     if plot_stations and stations_df is not None:
#         names, lons, lats = parse_station_table(stations_df)
#         ax.scatter(lons, lats, transform=ccrs.PlateCarree(), s=station_size, marker=station_marker, facecolor=station_facecolor, edgecolor=station_edgecolor, linewidth=station_linewidth, zorder=station_zorder)
#         if label_stations:
#             dx, dy = station_label_offset
#             for nm, x, y in zip(names, lons, lats):
#                 ax.text(x+dx, y+dy, str(nm), transform=ccrs.PlateCarree(), fontsize=station_label_fontsize, zorder=station_zorder+1, bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.6, lw=0))

#     ax.set_title(str(model_name), fontsize=13, pad=20)

#     # Colorbars
#     def _cbar(mappable, ax_loc, label):
#         if mappable:
#             cb = plt.colorbar(mappable, cax=ax_loc, orientation="horizontal")
#             cb.set_label(label)
#             if ticks is not None:
#                 cb.set_ticks(ticks)
#                 cb.set_ticklabels([f"{t:.0f}" for t in ticks])

#     _cbar(bn_pl, fig.add_subplot(gs[1, 0]), f"{labels[0]} (%)")
#     _cbar(nn_pl, fig.add_subplot(gs[1, 1]), f"{labels[1]} (%)")
#     _cbar(an_pl, fig.add_subplot(gs[1, 2]), f"{labels[2]} (%)")

#     if logo:
#         if mpimg is None: raise ImportError("matplotlib.image is required")
#         ax_logo = inset_axes(ax, width=logo_size[0], height=logo_size[1], loc=logo_position, borderpad=0.1)
#         ax_logo.imshow(mpimg.imread(str(logo)))
#         ax_logo.axis("off")

#     plt.subplots_adjust(top=0.95, bottom=0.08, left=0.06, right=0.94, hspace=-0.6, wspace=0.2)
#     out = dir_to_save / f"{str(model_name).replace(' ', '_')}.{output_ext}"
#     plt.savefig(out, dpi=dpi, bbox_inches="tight")
#     plt.show()

    
# # =============================================================================
# # Convenience wrapper: one-call plotting with auto boundary
# # =============================================================================
# def plot_prob_forecasts_auto_boundary(
#     dir_to_save: str | Path,
#     forecast_prob: xr.DataArray,
#     model_name: str,
#     country: str,
#     admin_level: int = 0,
#     boundary_source: Literal["auto", "geoboundaries", "gadm36", "naturalearth"] = "auto",
#     **plot_kwargs,
# ) -> None:
#     """
#     One-liner: downloads boundary and plots.
#     """
#     boundary_auto = dict(
#         country=country,
#         admin_level=admin_level,
#         source=boundary_source,
#         out_dir=plot_kwargs.pop("boundary_cache_dir", "boundaries_cache"),
#         geoboundaries_release=plot_kwargs.pop("geoboundaries_release", "gbOpen"),
#         naturalearth_resolution=plot_kwargs.pop("naturalearth_resolution", "10m"),
#         naturalearth_with_lakes=plot_kwargs.pop("naturalearth_with_lakes", True),
#         overwrite=plot_kwargs.pop("boundary_overwrite", False),
#         timeout=plot_kwargs.pop("boundary_timeout", 120),
#     )
#     plot_prob_forecasts(
#         dir_to_save=dir_to_save,
#         forecast_prob=forecast_prob,
#         model_name=model_name,
#         boundary_auto=boundary_auto,
#         **plot_kwargs,
#     )

########################################################################################################################################
########################################################################################################################################


# # =============================================================================
# # Cartopy geometry -> Path helper (version-safe)
# # =============================================================================
# try:
#     from cartopy.mpl.path import shapely_to_path as _cartopy_geom_to_path
# except Exception:
#     try:
#         from cartopy.mpl.path import geos_to_path as _cartopy_geom_to_path
#     except Exception:
#         _cartopy_geom_to_path = None


# def geom_to_path(geom):
#     """
#     Convert shapely geometry -> Matplotlib Path via Cartopy helper.
#     Works across cartopy versions (shapely_to_path or geos_to_path).
#     """
#     if _cartopy_geom_to_path is None:
#         raise ImportError(
#             "Your Cartopy version does not provide shapely_to_path/geos_to_path. "
#             "Upgrade cartopy or set clip_to_boundary=False."
#         )
#     return _cartopy_geom_to_path(geom)


# # =============================================================================
# # Country -> ISO3
# # =============================================================================
# def country_to_iso3(country: str) -> str:
#     """
#     Convert country name / ISO2 / ISO3 to ISO3.
#     Uses pycountry if available; otherwise minimal manual fallback.
#     """
#     c = (country or "").strip()
#     if not c:
#         raise ValueError("country must be a non-empty string")

#     # ISO3 already
#     if len(c) == 3 and c.isalpha():
#         return c.upper()

#     # ISO2
#     if len(c) == 2 and c.isalpha():
#         try:
#             import pycountry
#             obj = pycountry.countries.get(alpha_2=c.upper())
#             if obj:
#                 return obj.alpha_3.upper()
#         except Exception:
#             pass

#     # Name lookup
#     try:
#         import pycountry
#         obj = pycountry.countries.lookup(c)
#         return obj.alpha_3.upper()
#     except Exception:
#         manual = {
#             "COTE D IVOIRE": "CIV",
#             "CÔTE D'IVOIRE": "CIV",
#             "IVORY COAST": "CIV",
#             "DRC": "COD",
#             "CONGO (KINSHASA)": "COD",
#             "CONGO (BRAZZAVILLE)": "COG",
#             "SOUTH KOREA": "KOR",
#             "NORTH KOREA": "PRK",
#         }
#         key = c.upper()
#         if key in manual:
#             return manual[key]
#         raise ValueError(
#             f"Could not resolve ISO3 for country='{country}'. "
#             "Install pycountry (pip install pycountry) or pass ISO3 directly."
#         )


# # =============================================================================
# # Download helpers
# # =============================================================================
# def _download_bytes(url: str, timeout: int = 120) -> bytes:
#     r = requests.get(url, timeout=timeout)
#     r.raise_for_status()
#     return r.content


# def _extract_zip_bytes(zip_bytes: bytes, out_dir: Path, overwrite: bool = False) -> None:
#     out_dir.mkdir(parents=True, exist_ok=True)
#     with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
#         for member in zf.infolist():
#             target = out_dir / member.filename
#             if target.exists() and not overwrite:
#                 continue
#             zf.extract(member, path=out_dir)


# def _find_shps(root: Path) -> List[Path]:
#     return sorted([p for p in root.rglob("*.shp") if p.is_file()])


# def _pick_best_shp(shps: List[Path], admin_level: int) -> Path:
#     if not shps:
#         raise FileNotFoundError("No .shp files found after extraction.")
#     tag = f"ADM{admin_level}".lower()
#     exact = [p for p in shps if tag in p.name.lower()]
#     if exact:
#         return sorted(exact, key=lambda p: (len(p.name), len(str(p))))[0]
#     if len(shps) == 1:
#         return shps[0]
#     return shps[0]


# # =============================================================================
# # Natural Earth (optional, only ADM0/ADM1 and needs geopandas to filter one country)
# # =============================================================================
# def _naturalearth_dataset_name(admin_level: int, with_lakes: bool = True) -> Tuple[str, str]:
#     if admin_level == 0:
#         return ("cultural", "admin_0_countries")
#     if admin_level == 1:
#         return ("cultural", "admin_1_states_provinces_lakes" if with_lakes else "admin_1_states_provinces")
#     raise ValueError("NaturalEarth supports only admin_level 0 or 1.")


# def _download_naturalearth_country_shp(
#     iso3: str,
#     admin_level: int,
#     out_dir: Path,
#     resolution: Literal["10m", "50m", "110m"] = "10m",
#     with_lakes: bool = True,
#     overwrite: bool = False,
# ) -> Path:
#     """
#     Download NaturalEarth via cartopy and write a country-only shapefile.
#     Requires geopandas to filter and write.
#     """
#     if gpd is None:
#         raise ImportError("geopandas is required for NaturalEarth per-country extraction.")

#     from cartopy.io import shapereader as shpreader

#     category, name = _naturalearth_dataset_name(admin_level, with_lakes=with_lakes)
#     ne_path = Path(shpreader.natural_earth(resolution=resolution, category=category, name=name))
#     gdf = gpd.read_file(ne_path)

#     iso3u = iso3.upper()
#     candidates = [("ADM0_A3", iso3u), ("ISO_A3", iso3u), ("adm0_a3", iso3u), ("iso_a3", iso3u)]
#     mask = None
#     for col, val in candidates:
#         if col in gdf.columns:
#             m = gdf[col].astype(str).str.upper() == val
#             if m.any():
#                 mask = m
#                 break
#     if mask is None:
#         raise ValueError(f"NaturalEarth: cannot filter by ISO3='{iso3u}'. Columns: {list(gdf.columns)}")

#     sub = gdf.loc[mask].copy()
#     if sub.empty:
#         raise ValueError(f"NaturalEarth: no features for ISO3='{iso3u}' at ADM{admin_level}.")

#     # Reproject to EPSG:4326 if needed
#     if sub.crs is not None and str(sub.crs).lower() not in ["epsg:4326", "wgs84"]:
#         sub = sub.to_crs("EPSG:4326")

#     target_dir = out_dir / f"naturalearth_{resolution}_{iso3u}_ADM{admin_level}"
#     target_dir.mkdir(parents=True, exist_ok=True)
#     shp_out = target_dir / f"naturalearth_{resolution}_{iso3u}_ADM{admin_level}.shp"
#     if shp_out.exists() and not overwrite:
#         return shp_out

#     sub.to_file(shp_out, driver="ESRI Shapefile")
#     return shp_out


# # =============================================================================
# # Unified boundary downloader
# # =============================================================================
# def download_admin_shapefile(
#     country: str,
#     admin_level: int = 0,
#     out_dir: str | Path = "boundaries_cache",
#     source: Literal["auto", "geoboundaries", "gadm36", "naturalearth"] = "auto",
#     geoboundaries_release: Literal["gbOpen", "gbHumanitarian", "gbAuthoritative"] = "gbOpen",
#     naturalearth_resolution: Literal["10m", "50m", "110m"] = "10m",
#     naturalearth_with_lakes: bool = True,
#     overwrite: bool = False,
#     timeout: int = 120,
# ) -> Path:
#     """
#     Download a boundary shapefile for a given country and admin level.

#     Sources:
#       - geoboundaries: ADM0..ADM5  (preferred)
#       - gadm36:       ADM0..ADM*   (zip includes multiple levels; we select ADM{level})
#       - naturalearth: ADM0/ADM1 only (fast/coarse; requires geopandas)
#       - auto: tries geoboundaries -> gadm36 -> naturalearth (only if geopandas available)
#     """
#     iso3 = country_to_iso3(country)
#     out_dir = Path(out_dir)
#     admin_level = int(admin_level)
#     adm = f"ADM{admin_level}"

#     def _try_geoboundaries() -> Path:
#         api = f"https://www.geoboundaries.org/api/current/{geoboundaries_release}/{iso3}/{adm}/"
#         meta = requests.get(api, timeout=timeout)
#         meta.raise_for_status()
#         js = meta.json()
#         zip_url = js.get("staticDownloadLink") or js.get("downloadURL") or js.get("staticDownloadURL")
#         if not zip_url and isinstance(js, list) and js and isinstance(js[0], dict):
#             zip_url = js[0].get("staticDownloadLink") or js[0].get("downloadURL")
#         if not zip_url:
#             raise RuntimeError(f"geoBoundaries API returned no zip link for {iso3} {adm}")

#         target_dir = out_dir / f"geoboundaries_{geoboundaries_release}_{iso3}_{adm}"
#         if target_dir.exists() and not overwrite:
#             shps = _find_shps(target_dir)
#             if shps:
#                 return _pick_best_shp(shps, admin_level)

#         zbytes = _download_bytes(zip_url, timeout=timeout)
#         _extract_zip_bytes(zbytes, target_dir, overwrite=overwrite)
#         return _pick_best_shp(_find_shps(target_dir), admin_level)

#     def _try_gadm36() -> Path:
#         gadm_zip = f"https://geodata.ucdavis.edu/gadm/gadm3.6/shp/gadm36_{iso3}_shp.zip"
#         target_dir = out_dir / f"gadm36_{iso3}"
#         if target_dir.exists() and not overwrite:
#             shps = _find_shps(target_dir)
#             if shps:
#                 return _pick_best_shp(shps, admin_level)

#         zbytes = _download_bytes(gadm_zip, timeout=timeout)
#         _extract_zip_bytes(zbytes, target_dir, overwrite=overwrite)
#         return _pick_best_shp(_find_shps(target_dir), admin_level)

#     def _try_naturalearth() -> Path:
#         if admin_level not in (0, 1):
#             raise ValueError("naturalearth supports only admin_level 0 or 1.")
#         return _download_naturalearth_country_shp(
#             iso3=iso3,
#             admin_level=admin_level,
#             out_dir=out_dir,
#             resolution=naturalearth_resolution,
#             with_lakes=naturalearth_with_lakes,
#             overwrite=overwrite,
#         )

#     if source == "geoboundaries":
#         return _try_geoboundaries()
#     if source == "gadm36":
#         return _try_gadm36()
#     if source == "naturalearth":
#         return _try_naturalearth()

#     # auto
#     errors = []
#     try:
#         return _try_geoboundaries()
#     except Exception as e:
#         errors.append(f"geoBoundaries failed: {e}")

#     try:
#         return _try_gadm36()
#     except Exception as e:
#         errors.append(f"GADM36 failed: {e}")

#     if gpd is not None:
#         try:
#             return _try_naturalearth()
#         except Exception as e:
#             errors.append(f"NaturalEarth failed: {e}")
#     else:
#         errors.append("NaturalEarth skipped: geopandas not installed")

#     raise RuntimeError("All boundary sources failed:\n- " + "\n- ".join(errors))


# # =============================================================================
# # Boundary geometry loading (NO geopandas needed)
# # =============================================================================
# def load_boundary_geometry(boundary_shp: str | Path):
#     """
#     Load shapefile geometries and dissolve to a single shapely geometry.

#     Uses cartopy shapereader (no geopandas required).
#     Assumes boundary is already in lon/lat (EPSG:4326), true for geoBoundaries/GADM/NaturalEarth.
#     """
#     from cartopy.io import shapereader as shpreader
#     from shapely.ops import unary_union

#     boundary_shp = str(boundary_shp)
#     if not os.path.exists(boundary_shp):
#         raise FileNotFoundError(boundary_shp)

#     reader = shpreader.Reader(boundary_shp)
#     geoms = list(reader.geometries())
#     if not geoms:
#         raise ValueError(f"No geometries found in: {boundary_shp}")

#     return unary_union(geoms)


# def mask_forecast_to_geometry(forecast_prob: xr.DataArray, geom) -> xr.DataArray:
#     """
#     Mask forecast_prob outside polygon geometry using regionmask if available.
#     If regionmask is missing, returns forecast_prob unchanged.
#     """
#     if geom is None or regionmask is None:
#         return forecast_prob

#     lon = forecast_prob["X"].values
#     lat = forecast_prob["Y"].values
#     regs = regionmask.Regions([geom], names=["boundary"], abbrevs=["B"])
#     m = regs.mask(lon, lat)  # dims (Y, X): inside==0 else NaN
#     return forecast_prob.where(m == 0)


# # =============================================================================
# # Stations (your wide format)
# # =============================================================================
# def parse_station_table(stations_df):
#     """
#     Parse station dataframe in the format:
#         columns: ['STATION', 'Abetifi', 'Aburi', ...]
#         rows include: STATION == 'LAT' and STATION == 'LON'
#     Returns: (names, lons, lats) as numpy arrays (finite only).
#     """
#     if stations_df is None:
#         return np.array([]), np.array([]), np.array([])

#     import pandas as pd
#     if not isinstance(stations_df, pd.DataFrame):
#         raise TypeError("stations_df must be a pandas.DataFrame")

#     cols = list(stations_df.columns)

#     if "STATION" not in cols:
#         # tidy fallback: require LAT/LON columns
#         if {"LAT", "LON"}.issubset(set(cols)):
#             name_col = "STATION" if "STATION" in cols else ("NAME" if "NAME" in cols else None)
#             names = stations_df[name_col].astype(str).values if name_col else np.array([""] * len(stations_df))
#             lats = stations_df["LAT"].astype(float).values
#             lons = stations_df["LON"].astype(float).values
#             ok = np.isfinite(lats) & np.isfinite(lons)
#             return names[ok], lons[ok], lats[ok]
#         raise ValueError("stations_df format not recognized. Expected 'STATION' wide format or LAT/LON columns.")

#     st = stations_df.copy()
#     st["STATION"] = st["STATION"].astype(str)

#     lat_row = st.loc[st["STATION"].str.upper() == "LAT"]
#     lon_row = st.loc[st["STATION"].str.upper() == "LON"]
#     if lat_row.empty or lon_row.empty:
#         raise ValueError("stations_df has 'STATION' but LAT/LON rows are missing.")

#     lat_row = lat_row.iloc[0]
#     lon_row = lon_row.iloc[0]

#     names = [c for c in cols if c != "STATION"]
#     lats = np.array([float(lat_row[n]) for n in names], dtype=float)
#     lons = np.array([float(lon_row[n]) for n in names], dtype=float)
#     ok = np.isfinite(lats) & np.isfinite(lons)
#     return np.array(names)[ok], lons[ok], lats[ok]


# # =============================================================================
# # Global colorbar scaling (one scale for BN/NN/AN)
# # =============================================================================
# def global_cbar_stats(
#     arrays_2d: List[np.ndarray],
#     q: Tuple[float, float] = (10, 95),
#     floor: Optional[float] = 35.0,
#     ceil: Optional[float] = 85.0,
#     nticks: int = 6,
# ) -> Tuple[Optional[float], Optional[float], Optional[np.ndarray]]:
#     """
#     Compute ONE vmin/vmax/ticks from finite values of multiple 2D arrays.
#     """
#     vals = []
#     for a in arrays_2d:
#         v = np.asarray(a).ravel()
#         v = v[np.isfinite(v)]
#         if v.size:
#             vals.append(v)

#     if not vals:
#         return None, None, None

#     v = np.concatenate(vals)
#     vmin = np.nanpercentile(v, q[0])
#     vmax = np.nanpercentile(v, q[1])

#     if floor is not None:
#         vmin = max(vmin, float(floor))
#     if ceil is not None:
#         vmax = min(vmax, float(ceil))

#     if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
#         return None, None, None

#     ticks = np.linspace(vmin, vmax, nticks)
#     return float(vmin), float(vmax), ticks


# # =============================================================================
# # Coordinate auto-detection (X/Y vs lon/lat)
# # =============================================================================
# def standardize_lonlat_coords(da: xr.DataArray) -> xr.DataArray:
#     """
#     Ensure coords are named X (lon) and Y (lat).
#     Accepts coords named: X/Y, lon/lat, longitude/latitude, LON/LAT.
#     """
#     # coord candidates
#     lon_candidates = ["X", "lon", "longitude", "LON", "LONGITUDE"]
#     lat_candidates = ["Y", "lat", "latitude", "LAT", "LATITUDE"]

#     lon_name = next((c for c in lon_candidates if c in da.coords), None)
#     lat_name = next((c for c in lat_candidates if c in da.coords), None)

#     if lon_name is None or lat_name is None:
#         raise ValueError("forecast_prob must have lon/lat coords (X/Y or lon/lat or longitude/latitude).")

#     if lon_name != "X" or lat_name != "Y":
#         da = da.rename({lon_name: "X", lat_name: "Y"})

#     # ensure dims exist
#     if "X" not in da.dims or "Y" not in da.dims:
#         # Some DataArrays have coords but dims named differently; try rename dims too
#         rename_dims = {}
#         if "X" in da.coords and "X" not in da.dims:
#             # find dim carrying X coordinate
#             for d in da.dims:
#                 if "X" in da[d].coords or (da[d].name == "X"):
#                     rename_dims[d] = "X"
#                     break
#         if "Y" in da.coords and "Y" not in da.dims:
#             for d in da.dims:
#                 if "Y" in da[d].coords or (da[d].name == "Y"):
#                     rename_dims[d] = "Y"
#                     break
#         if rename_dims:
#             da = da.rename_dims(rename_dims)

#     return da


# # =============================================================================
# # Main plot function
# # =============================================================================
# def plot_prob_forecasts(
#     dir_to_save: str | Path,
#     forecast_prob: xr.DataArray,
#     model_name: str,
#     labels: Tuple[str, str, str] = ("Below-Normal", "Near-Normal", "Above-Normal"),
#     reverse_cmap: bool = True,
#     hspace: Optional[float] = None,
#     sigma: Optional[float] = None,
#     res: Optional[float] = None,
#     # Boundary:
#     boundary_shp: Optional[str | Path] = None,
#     boundary_auto: Optional[Dict[str, Any]] = None,  # dict passed to download_admin_shapefile(...)
#     clip_to_boundary: bool = True,
#     mask_outside_boundary: bool = True,
#     boundary_edgecolor: str = "black",
#     boundary_linewidth: float = 1.2,
#     # Stations:
#     stations_df=None,
#     plot_stations: bool = False,
#     label_stations: bool = False,
#     station_marker: str = "^",
#     station_size: float = 35,
#     station_facecolor: str = "black",
#     station_edgecolor: str = "white",
#     station_linewidth: float = 0.6,
#     station_zorder: int = 6,
#     station_label_fontsize: int = 7,
#     station_label_offset: Tuple[float, float] = (0.03, 0.03),
#     # Global colorbar scaling:
#     dynamic_cbar_global: bool = True,
#     cbar_q: Tuple[float, float] = (10, 95),
#     cbar_floor: Optional[float] = 35.0,
#     cbar_ceil: Optional[float] = 85.0,
#     cbar_nticks: int = 6,
#     # Map aesthetics:
#     land_facecolor: str = "#fde0dd",
#     ocean_facecolor: str = "lightblue",
#     # Logo:
#     logo: Optional[str | Path] = None,
#     logo_size: Tuple[Optional[str], Optional[str]] = ("7%", "21%"),  # width, height
#     logo_position: str = "lower left",
#     # Output:
#     output_ext: Literal["pdf", "png"] = "pdf",
#     dpi: int = 300,
# ) -> None:
#     """
#     Plot probabilistic tercile forecasts with optional boundary clip/mask and station overlay.

#     forecast_prob requirements:
#       - DataArray with dim 'probability' (size 3) and spatial dims/coords lon/lat.
#       - coords can be X/Y OR lon/lat OR longitude/latitude (auto-renamed to X/Y).
#     """
#     dir_to_save = Path(dir_to_save)
#     dir_to_save.mkdir(parents=True, exist_ok=True)

#     # --- standardize lon/lat coord names
#     forecast_prob = standardize_lonlat_coords(forecast_prob)

#     # --- ensure probability dim exists
#     if "probability" not in forecast_prob.dims:
#         raise ValueError("forecast_prob must have a 'probability' dimension.")

#     # --- ensure order
#     needed = {"probability", "Y", "X"}
#     if not needed.issubset(set(forecast_prob.dims)):
#         raise ValueError(f"forecast_prob must have dims including {needed}. Found: {forecast_prob.dims}")
#     forecast_prob = forecast_prob.transpose("probability", "Y", "X")

#     # --- optional interpolation
#     if res is not None:
#         min_X = float(forecast_prob["X"].min().values)
#         max_X = float(forecast_prob["X"].max().values)
#         min_Y = float(forecast_prob["Y"].min().values)
#         max_Y = float(forecast_prob["Y"].max().values)
#         num_X = int((max_X - min_X) / res) + 1
#         num_Y = int((max_Y - min_Y) / res) + 1
#         forecast_prob = forecast_prob.interp(
#             X=np.linspace(min_X, max_X, num_X),
#             Y=np.linspace(min_Y, max_Y, num_Y),
#             method="linear",
#             kwargs={"fill_value": "extrapolate"},
#         )

#     # --- optional smoothing (then renormalize across probability)
#     if sigma is not None:
#         fp_sm = forecast_prob * 0.0
#         for p in forecast_prob["probability"].values:
#             layer = forecast_prob.sel(probability=p)
#             layer_sm = xr.apply_ufunc(
#                 gaussian_filter,
#                 layer,
#                 input_core_dims=[["Y", "X"]],
#                 output_core_dims=[["Y", "X"]],
#                 kwargs={"sigma": sigma},
#             )
#             fp_sm.loc[{"probability": p}] = layer_sm
#         s = fp_sm.sum("probability")
#         forecast_prob = fp_sm / s.where(s != 0, 1.0)

#     # --- boundary: explicit or auto-download
#     if boundary_shp is None and boundary_auto:
#         boundary_shp = download_admin_shapefile(**boundary_auto)

#     geom = None
#     if boundary_shp is not None:
#         geom = load_boundary_geometry(boundary_shp)
#         if mask_outside_boundary:
#             forecast_prob = mask_forecast_to_geometry(forecast_prob, geom)

#     # --- max category & probability
#     max_prob = forecast_prob.max(dim="probability", skipna=True)
#     max_category = forecast_prob.fillna(-9999).argmax(dim="probability")

#     mask_bn = (max_category == 0)
#     mask_nn = (max_category == 1)
#     mask_an = (max_category == 2)

#     bn_data = (max_prob.where(mask_bn) * 100).values
#     nn_data = (max_prob.where(mask_nn) * 100).values
#     an_data = (max_prob.where(mask_an) * 100).values

#     # --- colormaps
#     if reverse_cmap:
#         AN_cmap = mcolors.LinearSegmentedColormap.from_list(
#             "AN", ["#fff7bc", "#fee391", "#fec44f", "#fe9929", "#ec7014", "#cc4c02", "#993404", "#662506"]
#         )
#         NN_cmap = mcolors.LinearSegmentedColormap.from_list(
#             "NN", ["#d9d9d9", "#bdbdbd", "#969696", "#737373", "#525252"]
#         )
#         BN_cmap = mcolors.LinearSegmentedColormap.from_list(
#             "BN", ["#e5f5f9", "#ccece6", "#99d8c9", "#66c2a4", "#41ae76", "#238b45", "#006d2c", "#00441b"]
#         )
#     else:
#         BN_cmap = mcolors.LinearSegmentedColormap.from_list(
#             "BN", ["#fff7bc", "#fee391", "#fec44f", "#fe9929", "#ec7014", "#cc4c02", "#993404", "#662506"]
#         )
#         NN_cmap = mcolors.LinearSegmentedColormap.from_list(
#             "NN", ["#d9d9d9", "#bdbdbd", "#969696", "#737373", "#525252"]
#         )
#         AN_cmap = mcolors.LinearSegmentedColormap.from_list(
#             "AN", ["#e5f5f9", "#ccece6", "#99d8c9", "#66c2a4", "#41ae76", "#238b45", "#006d2c", "#00441b"]
#         )

#     # --- global colorbar scaling (shared for BN/NN/AN)
#     if dynamic_cbar_global:
#         vmin, vmax, ticks = global_cbar_stats(
#             [bn_data, nn_data, an_data],
#             q=cbar_q,
#             floor=cbar_floor,
#             ceil=cbar_ceil,
#             nticks=cbar_nticks,
#         )
#     else:
#         vmin, vmax, ticks = 35.0, 85.0, np.arange(35, 86, 5)

#     # --- figure layout
#     if hspace is None:
#         hspace = -0.6

#     import matplotlib.gridspec as gridspec
#     fig = plt.figure(figsize=(10, 8))
#     gs = gridspec.GridSpec(
#         2, 3,
#         height_ratios=[10, 0.2],
#         width_ratios=[1.2, 0.6, 1.2],
#         hspace=hspace,
#         wspace=0.2
#     )
#     ax = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())

#     gl = ax.gridlines(draw_labels=True, linewidth=0.05, color="gray", alpha=0.8)
#     gl.top_labels = False
#     gl.right_labels = False

#     # extent to boundary
#     if geom is not None:
#         minx, miny, maxx, maxy = geom.bounds
#         ax.set_extent([minx, maxx, miny, maxy], crs=ccrs.PlateCarree())

#     # base map
#     ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=land_facecolor, edgecolor="black", zorder=0)
#     ax.add_feature(cfeature.OCEAN, facecolor=ocean_facecolor)
#     ax.coastlines()
#     ax.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=1.0, linestyle="solid")

#     # boundary outline + clip
#     if geom is not None:
#         ax.add_geometries([geom], crs=ccrs.PlateCarree(),
#                           facecolor="none", edgecolor=boundary_edgecolor,
#                           linewidth=boundary_linewidth, zorder=5)
#         if clip_to_boundary:
#             ax.set_boundary(geom_to_path(geom), transform=ccrs.PlateCarree())

#     # plot helper
#     def _plot_layer(data2d, cmap):
#         if vmin is None or vmax is None or not np.any(np.isfinite(data2d)):
#             sm = cm.ScalarMappable(norm=plt.Normalize(vmin=0, vmax=1), cmap=cmap)
#             sm.set_array([])
#             return sm
#         return ax.pcolormesh(
#             forecast_prob["X"], forecast_prob["Y"], data2d,
#             cmap=cmap, transform=ccrs.PlateCarree(),
#             alpha=0.9, vmin=vmin, vmax=vmax
#         )

#     bn_plot = _plot_layer(bn_data, BN_cmap)
#     nn_plot = _plot_layer(nn_data, NN_cmap)
#     an_plot = _plot_layer(an_data, AN_cmap)

#     # stations
#     if plot_stations and stations_df is not None:
#         names, lons, lats = parse_station_table(stations_df)
#         ax.scatter(
#             lons, lats, transform=ccrs.PlateCarree(),
#             s=station_size, marker=station_marker,
#             facecolor=station_facecolor, edgecolor=station_edgecolor,
#             linewidth=station_linewidth, zorder=station_zorder
#         )
#         if label_stations:
#             dx, dy = station_label_offset
#             for nm, x, y in zip(names, lons, lats):
#                 ax.text(
#                     x + dx, y + dy, str(nm),
#                     transform=ccrs.PlateCarree(),
#                     fontsize=station_label_fontsize,
#                     zorder=station_zorder + 1,
#                     bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.6, linewidth=0.0),
#                 )

#     # title
#     ax.set_title(str(model_name), fontsize=13, pad=20)

#     # colorbar helper
#     def _apply_cbar(mappable, cax, label):
#         cb = plt.colorbar(mappable, cax=cax, orientation="horizontal")
#         cb.set_label(label)
#         if ticks is not None:
#             cb.set_ticks(ticks)
#             cb.set_ticklabels([f"{t:.0f}" for t in ticks])
#         return cb

#     _apply_cbar(bn_plot, fig.add_subplot(gs[1, 0]), f"{labels[0]} (%)")
#     _apply_cbar(nn_plot, fig.add_subplot(gs[1, 1]), f"{labels[1]} (%)")
#     _apply_cbar(an_plot, fig.add_subplot(gs[1, 2]), f"{labels[2]} (%)")

#     # logo
#     if logo is not None:
#         if mpimg is None:
#             raise ImportError("matplotlib.image is required to add a logo (mpimg).")
#         ax_logo = inset_axes(ax, width=logo_size[0], height=logo_size[1], loc=logo_position, borderpad=0.1)
#         ax_logo.imshow(mpimg.imread(str(logo)))
#         ax_logo.axis("off")

#     plt.subplots_adjust(top=0.95, bottom=0.08, left=0.06, right=0.94, hspace=-0.6, wspace=0.2)
#     out = dir_to_save / f"{str(model_name).replace(' ', '_')}.{output_ext}"
#     plt.savefig(out, dpi=dpi, bbox_inches="tight")
#     plt.show()


# # =============================================================================
# # Convenience wrapper: auto boundary download + plot
# # =============================================================================
# def plot_prob_forecasts_auto_boundary(
#     dir_to_save: str | Path,
#     forecast_prob: xr.DataArray,
#     model_name: str,
#     country: str,
#     admin_level: int = 0,
#     boundary_source: Literal["auto", "geoboundaries", "gadm36", "naturalearth"] = "auto",
#     boundary_cache_dir: str | Path = "boundaries_cache",
#     geoboundaries_release: Literal["gbOpen", "gbHumanitarian", "gbAuthoritative"] = "gbOpen",
#     naturalearth_resolution: Literal["10m", "50m", "110m"] = "10m",
#     naturalearth_with_lakes: bool = True,
#     boundary_overwrite: bool = False,
#     boundary_timeout: int = 120,
#     **plot_kwargs,
# ) -> None:
#     """
#     One-liner usage:
#       plot_prob_forecasts_auto_boundary(... country="Ghana", admin_level=1, ...)
#     """
#     boundary_auto = dict(
#         country=country,
#         admin_level=admin_level,
#         out_dir=boundary_cache_dir,
#         source=boundary_source,
#         geoboundaries_release=geoboundaries_release,
#         naturalearth_resolution=naturalearth_resolution,
#         naturalearth_with_lakes=naturalearth_with_lakes,
#         overwrite=boundary_overwrite,
#         timeout=boundary_timeout,
#     )

#     plot_prob_forecasts(
#         dir_to_save=dir_to_save,
#         forecast_prob=forecast_prob,
#         model_name=model_name,
#         boundary_auto=boundary_auto,
#         **plot_kwargs,
#     )




def plot_prob_forecasts_(dir_to_save, forecast_prob, model_name, labels=["Below-Normal", "Near-Normal", "Above-Normal"], reverse_cmap=True, hspace=None, logo=None, logo_size=(None,None), logo_position="lower left", sigma=None, res=None):
    """
    Plot probabilistic forecasts with tercile categories.

    Parameters
    ----------
    dir_to_save : str
        Directory path to save the plot.
    forecast_prob : xarray.DataArray
        Data array containing probability forecasts with a 'probability' dimension.
    model_name : str or numpy.ndarray
        Name of the model for the plot title.
    labels : list, optional
        Labels for the tercile categories (default is ["Below-Normal", "Near-Normal", "Above-Normal"]).
    reverse_cmap : bool, optional
        If True, reverse the colormap order for categories (default is True).
    hspace : float, optional
        Height space between the main plot and the colorbar (default is -0.6).
    sigma : float, optional
        Standard deviation for Gaussian smoothing of probabilities (default is None, no smoothing).
    res : float, optional
        Resolution for interpolation of the forecast probabilities (default is None, no interpolation). 
    logo : str, optional
        Path to the logo image to be added to the plot (default is None).
    logo_size : float, optional
        Size of the logo image in the plot (default is 0.5).    
    logo_position : str, optional
        Position of the logo image in the plot (default is "lower left").       
    logo_size : tuple, optional
        Size of the logo image in the plot as (width, height) in inches (default is (None, None)).

    Returns
    -------
    None

    Notes
    -----
    Saves the plot as a PNG file and displays it.
    Uses custom colormaps for each tercile category.
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
        forecast_prob_smoothed = forecast_prob * 0.0  # Initialize with same shape and coords
        
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
        
        # Replace original with smoothed
        forecast_prob = forecast_prob_smoothed

    # Step 1: Extract maximum probability and category
    max_prob = forecast_prob.max(dim="probability", skipna=True)  # Maximum probability at each grid point
    # Fill NaN values with a very low value 
    filled_prob = forecast_prob.fillna(-9999)
    # Compute argmax
    max_category = filled_prob.argmax(dim="probability")
    
    # Step 2: Create masks for each category
    mask_bn = max_category == 0  # Below Normal (BN)
    mask_nn = max_category == 1  # Near Normal (NN)
    mask_an = max_category == 2  # Above Normal (AN)
    
    # Step 3: Define custom colormaps
    if reverse_cmap:
        AN_cmap = mcolors.LinearSegmentedColormap.from_list('AN', ["#fff7bc", "#fee391", "#fec44f", "#fe9929", "#ec7014", "#cc4c02", "#993404", "#662506"]) 
        NN_cmap = mcolors.LinearSegmentedColormap.from_list('NN', ["#d9d9d9", "#bdbdbd", "#969696", "#737373", "#525252"])
        BN_cmap = mcolors.LinearSegmentedColormap.from_list('BN', ["#e5f5f9", "#ccece6", "#99d8c9", "#66c2a4", "#41ae76", "#238b45", "#006d2c", "#00441b"])  
    else:
        BN_cmap = mcolors.LinearSegmentedColormap.from_list('BN', ["#fff7bc", "#fee391", "#fec44f", "#fe9929", "#ec7014", "#cc4c02", "#993404", "#662506"]) 
        NN_cmap = mcolors.LinearSegmentedColormap.from_list('NN', ["#d9d9d9", "#bdbdbd", "#969696", "#737373", "#525252"])
        AN_cmap = mcolors.LinearSegmentedColormap.from_list('AN', ["#e5f5f9", "#ccece6", "#99d8c9", "#66c2a4", "#41ae76", "#238b45", "#006d2c", "#00441b"])          
    
    # Create a figure with GridSpec
    fig = plt.figure(figsize=(10, 8))

    if hspace is None:
        hspace = -0.6  # Default height space if not provided
    else:
        hspace = hspace  # Use the provided height space

    if logo is not None:
        logo_size=  ("7%","21%")  # Default logo size if not provided
    else:
        logo_size = logo_size  # Use the provided logo size

    # Define the GridSpec layout
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

    # Step 4: Plot each category
   # def clip_prob(data, mask):
   #     prob = max_prob.where(mask)
   #     prob = xr.where(prob > 0.6, 0.6, prob) * 100
   #     prob = xr.where(prob < 25, 25, prob)
   #     return prob.values

   # bn_data = clip_prob(max_prob, mask_bn)
   # nn_data = clip_prob(max_prob, mask_nn)
   # an_data = clip_prob(max_prob, mask_an)


    # Step _: Add  Land colors
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#fde0dd", edgecolor="black", zorder=0) # #dfc27d
    
    # # Define the data ranges for color normalization  
    # vmin = 25  # Minimum probability percentage
    # vmax = 65  # Maximum probability percentage

    # Plot BN (Below Normal)
    if np.any(~np.isnan(bn_data)):
        bn_plot = ax.pcolormesh(
            forecast_prob['X'], forecast_prob['Y'], bn_data,
            cmap=BN_cmap, transform=ccrs.PlateCarree(), alpha=0.9, vmin=35, vmax=85
        )
    else:
        bn_plot = cm.ScalarMappable(norm=plt.Normalize(vmin=35, vmax=85), cmap=BN_cmap)
        bn_plot.set_array([])

    # Plot NN (Near Normal)
    if np.any(~np.isnan(nn_data)):
        nn_plot = ax.pcolormesh(
            forecast_prob['X'], forecast_prob['Y'], nn_data,
            cmap=NN_cmap, transform=ccrs.PlateCarree(), alpha=0.9, vmin=35, vmax=65
        )
    else:
        nn_plot = cm.ScalarMappable(norm=plt.Normalize(vmin=35, vmax=65), cmap=NN_cmap)
        nn_plot.set_array([])

    # Plot AN (Above Normal)
    if np.any(~np.isnan(an_data)):
        an_plot = ax.pcolormesh(
            forecast_prob['X'], forecast_prob['Y'], an_data,
            cmap=AN_cmap, transform=ccrs.PlateCarree(), alpha=0.9, vmin=35, vmax=85
        )
    else:
        an_plot = cm.ScalarMappable(norm=plt.Normalize(vmin=35, vmax=85), cmap=AN_cmap)
        an_plot.set_array([])

    # Step 5: Add coastlines and borders and Land
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=1.0, linestyle='solid')
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    
    # Step 6: Add individual colorbars with fixed ticks
    def create_ticks(vn=35, vx=86, step=5):
        ticks = np.arange(vn, vx, step)
        return ticks

    #ticks = create_ticks(vn=35, vx=86, step=5)

    # For BN (Below Normal)
    ticks = create_ticks(vn=35, vx=86, step=5)
    cbar_ax_bn = fig.add_subplot(gs[1, 0])
    cbar_bn = plt.colorbar(bn_plot, cax=cbar_ax_bn, orientation='horizontal')
    cbar_bn.set_label(f'{labels[0]} (%)')
    cbar_bn.set_ticks(ticks)
    cbar_bn.set_ticklabels([f"{tick}" for tick in ticks])

    # For NN (Near Normal)
    ticks = create_ticks(vn=35, vx=66, step=5)
    cbar_ax_nn = fig.add_subplot(gs[1, 1])
    cbar_nn = plt.colorbar(nn_plot, cax=cbar_ax_nn, orientation='horizontal')
    cbar_nn.set_label(f'{labels[1]} (%)')
    cbar_nn.set_ticks(ticks)
    cbar_nn.set_ticklabels([f"{tick}" for tick in ticks])

    # For AN (Above Normal)
    ticks = create_ticks(vn=35, vx=86, step=5)
    cbar_ax_an = fig.add_subplot(gs[1, 2])
    cbar_an = plt.colorbar(an_plot, cax=cbar_ax_an, orientation='horizontal')
    cbar_an.set_label(f'{labels[2]} (%)')
    cbar_an.set_ticks(ticks)
    cbar_an.set_ticklabels([f"{tick}" for tick in ticks])
    
    # Set the title with the formatted model_name
    if isinstance(model_name, np.ndarray):
        model_name_str = str(model_name.item())
    else:
        model_name_str = str(model_name)
    ax.set_title(f"{model_name_str}", fontsize=13, pad=20)


    # Step 7: Add logo if provided
    
    if logo is not None:
        ax_logo = inset_axes(ax,
                            width=logo_size[0],  # Width of the logo    
                            height=logo_size[1],  # Height of the logo        
                            loc=logo_position,
                            borderpad=0.1)        
        ax_logo.imshow(mpimg.imread(logo))
        ax_logo.axis("off") 

    # plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92, hspace=0.03, wspace=0.03)
    plt.subplots_adjust(top=0.95, bottom=0.08, left=0.06, right=0.94, hspace=-0.6, wspace=0.2)
    plt.savefig(f"{dir_to_save}/{model_name_str.replace(' ', '_')}.pdf", dpi=300, bbox_inches='tight')
    plt.show()


# def plot_prob_forecasts1(dir_to_save, forecast_prob, model_name, labels=["Below-Normal", "Near-Normal", "Above-Normal"], reverse_cmap=True, logo=None, logo_size=0.5):
#     """
#     Plot probabilistic forecasts with tercile categories.

#     Parameters
#     ----------
#     dir_to_save : str
#         Directory path to save the plot.
#     forecast_prob : xarray.DataArray
#         Data array containing probability forecasts with a 'probability' dimension.
#     model_name : str or numpy.ndarray
#         Name of the model for the plot title.
#     labels : list, optional
#         Labels for the tercile categories (default is ["Below-Normal", "Near-Normal", "Above-Normal"]).
#     reverse_cmap : bool, optional
#         If True, reverse the colormap order for categories (default is True).
#     logo : str, optional
#         Path to the logo image to be added to the plot (default is None).
#     logo_size : float, optional
#         Size of the logo image in the plot (default is 0.5).    

#     Notes
#     -----
#     Saves the plot as a PNG file and displays it.
#     Uses custom colormaps for each tercile category.
#     """
#     # Step 1: Extract maximum probability and category
#     max_prob = forecast_prob.max(dim="probability", skipna=True)  # Maximum probability at each grid point
#     # Fill NaN values with a very low value 
#     filled_prob = forecast_prob.fillna(-9999)
#     # Compute argmax
#     max_category = filled_prob.argmax(dim="probability")
    
#     # Step 2: Create masks for each category
#     mask_bn = max_category == 0  # Below Normal (BN)
#     mask_nn = max_category == 1  # Near Normal (NN)
#     mask_an = max_category == 2  # Above Normal (AN)
    
#     # Step 3: Define custom colormaps
#     if reverse_cmap:
#         AN_cmap = mcolors.LinearSegmentedColormap.from_list('AN', ['#FDAE61', '#F46D43', '#D73027']) 
#         NN_cmap = mcolors.LinearSegmentedColormap.from_list('NN', ['#FFFFE5', '#FFF7BC', '#FEE391'])
#         BN_cmap = mcolors.LinearSegmentedColormap.from_list('BN', ['#ABDDA4', '#66C2A5', '#3288BD'])  
#     else:
#         BN_cmap = mcolors.LinearSegmentedColormap.from_list('BN', ['#FDAE61', '#F46D43', '#D73027']) 
#         NN_cmap = mcolors.LinearSegmentedColormap.from_list('NN', ['#FFFFE5', '#FFF7BC', '#FEE391'])
#         AN_cmap = mcolors.LinearSegmentedColormap.from_list('AN', ['#ABDDA4', '#66C2A5', '#3288BD'])          
    
#     # Create a figure with GridSpec
#     # fig = plt.figure(figsize=(8, 6.5))  # Increased height to accommodate logo
#     # gs = gridspec.GridSpec(3, 3, height_ratios=[15, 0.8, 0.7], hspace=0.2)

#     fig = plt.figure(figsize=(10, 8))
#     gs = gridspec.GridSpec(3, 3, height_ratios=[8, 0.2, 3], hspace=0.03)

#     # Main map axis
#     ax = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())

#     # Modify by Mandela
    
#     ###################
#     ##################

#     # Step 4: Plot each category
#     # Multiply by 100 to convert probabilities to percentages
    
#     # bn_data = (max_prob.where(mask_bn) * 100).values
#     # nn_data = (max_prob.where(mask_nn) * 100).values
#     # an_data = (max_prob.where(mask_an) * 100).values
    
#     # Apply the condition to ensure minimum value of 45% for each category
#     # and a maximum of 60% for BN, NN, and AN
#     # Ensure that the values are at least 45% and at most 60% for each category
#     # and convert to percentage
#     # bn_data = xr.where((max_prob.where(mask_bn) * 100) < 45, 45,
#     #                    xr.where(max_prob.where(mask_bn) * 100 > 60, 60, max_prob.where(mask_bn) * 100)).values
#     # nn_data = xr.where((max_prob.where(mask_nn) * 100) < 45, 45,
#     #                    xr.where(max_prob.where(mask_nn) * 100 > 60, 60, max_prob.where(mask_nn) * 100)).values
#     # an_data = xr.where((max_prob.where(mask_an) * 100) < 45, 45,
#     #                    xr.where(max_prob.where(mask_an) * 100 > 60, 60, max_prob.where(mask_an) * 100)).values    


#     # Step 4: Plot each category
#     bn_data = xr.where((xr.where(max_prob.where(mask_bn)>0.6,0.6,max_prob.where(mask_bn))* 100)<45, 45,
#                        xr.where(max_prob.where(mask_bn)>0.6,0.6,max_prob.where(mask_bn))* 100).values  
#     nn_data = xr.where((xr.where(max_prob.where(mask_nn)>0.6,0.6,max_prob.where(mask_nn))* 100)<45, 45,
#                    xr.where(max_prob.where(mask_nn)>0.6,0.6,max_prob.where(mask_nn))* 100).values
#     an_data = xr.where((xr.where(max_prob.where(mask_an)>0.6,0.6,max_prob.where(mask_an))* 100)<45, 45,
#                    xr.where(max_prob.where(mask_an)>0.6,0.6,max_prob.where(mask_an))* 100).values
    
#     # Define the data ranges for color normalization  
#     vmin = 35  # Minimum probability percentage
#     vmax = 65  # Maximum probability percentage

#     # Plot BN (Below Normal)
#     if np.any(~np.isnan(bn_data)):
#         bn_plot = ax.pcolormesh(
#             forecast_prob['X'], forecast_prob['Y'], bn_data,
#             cmap=BN_cmap, transform=ccrs.PlateCarree(), alpha=0.9, vmin=vmin, vmax=vmax
#         )
#     else:
#         bn_plot = cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=BN_cmap)
#         bn_plot.set_array([])

#     # Plot NN (Near Normal)
#     if np.any(~np.isnan(nn_data)):
#         nn_plot = ax.pcolormesh(
#             forecast_prob['X'], forecast_prob['Y'], nn_data,
#             cmap=NN_cmap, transform=ccrs.PlateCarree(), alpha=0.9, vmin=vmin, vmax=vmax
#         )
#     else:
#         nn_plot = cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=NN_cmap)
#         nn_plot.set_array([])

#     # Plot AN (Above Normal)
#     if np.any(~np.isnan(an_data)):
#         an_plot = ax.pcolormesh(
#             forecast_prob['X'], forecast_prob['Y'], an_data,
#             cmap=AN_cmap, transform=ccrs.PlateCarree(), alpha=0.9, vmin=vmin, vmax=vmax
#         )
#     else:
#         an_plot = cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=AN_cmap)
#         an_plot.set_array([])

#     # Step 5: Add coastlines and borders
#     ax.coastlines()
#     ax.add_feature(cfeature.BORDERS, linestyle=':')
    
#     # Step 6: Add individual colorbars with fixed ticks
#     def create_ticks():
#         ticks = np.arange(35, 66, 5)
#         return ticks

#     ticks = create_ticks()

#     # For BN (Below Normal)
#     cbar_ax_bn = fig.add_subplot(gs[1, 0])
#     cbar_bn = plt.colorbar(bn_plot, cax=cbar_ax_bn, orientation='horizontal')
#     cbar_bn.set_label(f'{labels[0]} (%)')
#     cbar_bn.set_ticks(ticks)
#     cbar_bn.set_ticklabels([f"{tick}" for tick in ticks])

#     # For NN (Near Normal)
#     cbar_ax_nn = fig.add_subplot(gs[1, 1])
#     cbar_nn = plt.colorbar(nn_plot, cax=cbar_ax_nn, orientation='horizontal')
#     cbar_nn.set_label(f'{labels[1]} (%)')
#     cbar_nn.set_ticks(ticks)
#     cbar_nn.set_ticklabels([f"{tick}" for tick in ticks])

#     # For AN (Above Normal)
#     cbar_ax_an = fig.add_subplot(gs[1, 2])
#     cbar_an = plt.colorbar(an_plot, cax=cbar_ax_an, orientation='horizontal')
#     cbar_an.set_label(f'{labels[2]} (%)')
#     cbar_an.set_ticks(ticks)
#     cbar_an.set_ticklabels([f"{tick}" for tick in ticks])
    
#     # Set the title with the formatted model_name
#     if isinstance(model_name, np.ndarray):
#         model_name_str = str(model_name.item())
#     else:
#         model_name_str = str(model_name)
#     ax.set_title(f"{model_name_str}", fontsize=13, pad=20)

#     # Step 7: Add logo if provided
#     logo_ax = fig.add_subplot(gs[2, 2])
#     logo_ax.axis('off')
#     if logo is not None:
#         im = image.imread(logo)
#         addLogo = OffsetImage(im, zoom=logo_size)
#         ab = AnnotationBbox(addLogo, (0.5, 0.5), frameon=False, xycoords='axes fraction')
#         logo_ax.add_artist(ab)

#     plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92, hspace=-0.6, wspace=0.2)
#     plt.savefig(f"{dir_to_save}/{model_name_str.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
#     plt.show()

# def plot_prob_forecasts2(dir_to_save, forecast_prob, model_name, labels=["Below-Normal", "Near-Normal", "Above-Normal"], reverse_cmap=True, logo=None, logo_size=0.5):
#     """
#     Plot probabilistic forecasts with tercile categories.

#     Parameters
#     ----------
#     dir_to_save : str
#         Directory path to save the plot.
#     forecast_prob : xarray.DataArray
#         Data array containing probability forecasts with a 'probability' dimension.
#     model_name : str or numpy.ndarray
#         Name of the model for the plot title.
#     labels : list, optional
#         Labels for the tercile categories (default is ["Below-Normal", "Near-Normal", "Above-Normal"]).
#     reverse_cmap : bool, optional
#         If True, reverse the colormap order for categories (default is True).
#     logo : str, optional
#         Path to the logo image to be added to the plot (default is None).
#     logo_size : float, optional
#         Size of the logo image in the plot (default is 0.5).    

#     Notes
#     -----
#     Saves the plot as a PNG file and displays it.
#     Uses custom colormaps for each tercile category.
#     """
#     # Step 1: Extract maximum probability and category
#     max_prob = forecast_prob.max(dim="probability", skipna=True)  # Maximum probability at each grid point
#     # Fill NaN values with a very low value 
#     filled_prob = forecast_prob.fillna(-9999)
#     # Compute argmax
#     max_category = filled_prob.argmax(dim="probability")
    
#     # Step 2: Create masks for each category
#     mask_bn = max_category == 0  # Below Normal (BN)
#     mask_nn = max_category == 1  # Near Normal (NN)
#     mask_an = max_category == 2  # Above Normal (AN)
    
#     # Step 3: Define custom colormaps
#     if reverse_cmap:
#         AN_cmap = mcolors.LinearSegmentedColormap.from_list('AN', ['#FDAE61', '#F46D43', '#D73027']) 
#         NN_cmap = mcolors.LinearSegmentedColormap.from_list('NN', ['#FFFFE5', '#FFF7BC', '#FEE391'])
#         BN_cmap = mcolors.LinearSegmentedColormap.from_list('BN', ['#ABDDA4', '#66C2A5', '#3288BD'])  
#     else:
#         BN_cmap = mcolors.LinearSegmentedColormap.from_list('BN', ['#FDAE61', '#F46D43', '#D73027']) 
#         NN_cmap = mcolors.LinearSegmentedColormap.from_list('NN', ['#FFFFE5', '#FFF7BC', '#FEE391'])
#         AN_cmap = mcolors.LinearSegmentedColormap.from_list('AN', ['#ABDDA4', '#66C2A5', '#3288BD'])          
    
#     # Create a figure with GridSpec
#     # fig = plt.figure(figsize=(8, 6.5))  # Increased height to accommodate logo
#     # gs = gridspec.GridSpec(3, 3, height_ratios=[15, 0.8, 0.7], hspace=0.2)

#     fig = plt.figure(figsize=(10, 8))
#     gs = gridspec.GridSpec(3, 3, height_ratios=[8, 0.2, 3], hspace=0.03)

#     # Main map axis
#     ax = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())

#     # Modify by Mandela
    
#     ###################
#     ##################
#     sigma = 2.0  # Smoothing parameter (Gaussian sigma); 
    
#     # Create a smoothed copy
#     forecast_prob_smoothed = forecast_prob * 0.0  # Initialize with same shape and coords
    
#     # Smooth each probability layer spatially
#     for p in forecast_prob.probability.values:
#         layer = forecast_prob.sel(probability=p)
#         layer_smoothed = xr.apply_ufunc(
#             gaussian_filter,
#             layer,
#             input_core_dims=[['Y', 'X']],
#             output_core_dims=[['Y', 'X']],
#             kwargs={'sigma': sigma}
#         )
#         forecast_prob_smoothed.loc[{'probability': p}] = layer_smoothed
    
#     # Normalize smoothed probabilities to sum to 1 at each grid point
#     sum_probs = forecast_prob_smoothed.sum('probability')
#     forecast_prob_smoothed = forecast_prob_smoothed / sum_probs.where(sum_probs != 0, 1.0)
    
#     # Replace original with smoothed
#     forecast_prob = forecast_prob_smoothed

#     # Step 4: Plot each category
#     bn_data = xr.where((xr.where(max_prob.where(mask_bn)>0.6,0.6,max_prob.where(mask_bn))* 100)<45, 45,
#                        xr.where(max_prob.where(mask_bn)>0.6,0.6,max_prob.where(mask_bn))* 100).values  
#     nn_data = xr.where((xr.where(max_prob.where(mask_nn)>0.6,0.6,max_prob.where(mask_nn))* 100)<45, 45,
#                    xr.where(max_prob.where(mask_nn)>0.6,0.6,max_prob.where(mask_nn))* 100).values
#     an_data = xr.where((xr.where(max_prob.where(mask_an)>0.6,0.6,max_prob.where(mask_an))* 100)<45, 45,
#                    xr.where(max_prob.where(mask_an)>0.6,0.6,max_prob.where(mask_an))* 100).values
    
#     # Define the data ranges for color normalization  
#     vmin = 35  # Minimum probability percentage
#     vmax = 65  # Maximum probability percentage

#     # Plot BN (Below Normal)
#     if np.any(~np.isnan(bn_data)):
#         bn_plot = ax.pcolormesh(
#             forecast_prob['X'], forecast_prob['Y'], bn_data,
#             cmap=BN_cmap, transform=ccrs.PlateCarree(), alpha=0.9, vmin=vmin, vmax=vmax
#         )
#     else:
#         bn_plot = cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=BN_cmap)
#         bn_plot.set_array([])

#     # Plot NN (Near Normal)
#     if np.any(~np.isnan(nn_data)):
#         nn_plot = ax.pcolormesh(
#             forecast_prob['X'], forecast_prob['Y'], nn_data,
#             cmap=NN_cmap, transform=ccrs.PlateCarree(), alpha=0.9, vmin=vmin, vmax=vmax
#         )
#     else:
#         nn_plot = cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=NN_cmap)
#         nn_plot.set_array([])

#     # Plot AN (Above Normal)
#     if np.any(~np.isnan(an_data)):
#         an_plot = ax.pcolormesh(
#             forecast_prob['X'], forecast_prob['Y'], an_data,
#             cmap=AN_cmap, transform=ccrs.PlateCarree(), alpha=0.9, vmin=vmin, vmax=vmax
#         )
#     else:
#         an_plot = cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=AN_cmap)
#         an_plot.set_array([])

#     # Step 5: Add coastlines and borders
#     ax.coastlines()
#     ax.add_feature(cfeature.BORDERS, linestyle=':')
    
#     # Step 6: Add individual colorbars with fixed ticks
#     def create_ticks():
#         ticks = np.arange(35, 66, 5)
#         return ticks

#     ticks = create_ticks()

#     # For BN (Below Normal)
#     cbar_ax_bn = fig.add_subplot(gs[1, 0])
#     cbar_bn = plt.colorbar(bn_plot, cax=cbar_ax_bn, orientation='horizontal')
#     cbar_bn.set_label(f'{labels[0]} (%)')
#     cbar_bn.set_ticks(ticks)
#     cbar_bn.set_ticklabels([f"{tick}" for tick in ticks])

#     # For NN (Near Normal)
#     cbar_ax_nn = fig.add_subplot(gs[1, 1])
#     cbar_nn = plt.colorbar(nn_plot, cax=cbar_ax_nn, orientation='horizontal')
#     cbar_nn.set_label(f'{labels[1]} (%)')
#     cbar_nn.set_ticks(ticks)
#     cbar_nn.set_ticklabels([f"{tick}" for tick in ticks])

#     # For AN (Above Normal)
#     cbar_ax_an = fig.add_subplot(gs[1, 2])
#     cbar_an = plt.colorbar(an_plot, cax=cbar_ax_an, orientation='horizontal')
#     cbar_an.set_label(f'{labels[2]} (%)')
#     cbar_an.set_ticks(ticks)
#     cbar_an.set_ticklabels([f"{tick}" for tick in ticks])
    
#     # Set the title with the formatted model_name
#     if isinstance(model_name, np.ndarray):
#         model_name_str = str(model_name.item())
#     else:
#         model_name_str = str(model_name)
#     ax.set_title(f"{model_name_str}", fontsize=13, pad=20)

#     # Step 7: Add logo if provided
#     logo_ax = fig.add_subplot(gs[2, 2])
#     logo_ax.axis('off')
#     if logo is not None:
#         im = image.imread(logo)
#         addLogo = OffsetImage(im, zoom=logo_size)
#         ab = AnnotationBbox(addLogo, (0.5, 0.5), frameon=False, xycoords='axes fraction')
#         logo_ax.add_artist(ab)

#     plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92, hspace=0.03, wspace=0.3)
#     plt.savefig(f"{dir_to_save}/{model_name_str.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
#     plt.show()


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


def process_model_for_other_params(agmParamModel, dir_to_save, hdcst_file_path, fcst_file_path, obs_hdcst, obs_fcst_year, month_of_initialization, year_start, year_end, year_forecast, nb_cores=2, agrometparam="Onset"):
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
            agpm_model = agmParamModel.compute(daily_data=ds_filled.sortby("T"), nb_cores=nb_cores)
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
            agpm_model = agmParamModel.compute(daily_data=ds_filled, nb_cores=nb_cores)
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
def proceed_seasonal_daily_bias_correction(dir_to_save_model, observation, hindcast_files, forecast_files, varname="PRCP", wet_day=0.1):
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
                    fobj_quant = qmap.fitQmap(obs_month, hcst_month, method='QUANT', wet_day=0.1,  qstep=0.0001)
                    hcst_month_corr = qmap.doQmap(hcst_month, fobj_quant, type='linear')
                    fcst_month_corr = qmap.doQmap(fcst_month, fobj_quant, type='linear')
                else:
                    fobj_quant = qmap_.fitBC(obs_month, hcst_month, method='QUANT', qstep=0.0001, nboot=5)
                    hcst_month_corr = qmap_.doBC(hcst_month, fobj_quant, type='linear')
                    fcst_month_corr = qmap_.doBC(fcst_month, fobj_quant, type='linear')         
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

# Other commmented code to use after

# retrieve Zone for PCR

###### Code to use after in several zones for PCR ##############################
# pca= xe.single.EOF(n_modes=6, use_coslat=True, standardize=True)
# pca.fit([i.fillna(i.mean(dim="T", skipna=True)).rename({"X": "lon", "Y": "lat"}) for i in predictor], dim="T")
# components = pca.components()
# scores = pca.scores()
# expl = pca.explained_variance_ratio()
# expl



# def plot_prob_forecats(dir_to_save, forecast_prob, model_name):    
#     # Step 1: Extract maximum probability and category
#     max_prob = forecast_prob.max(dim="probability", skipna=True)  # Maximum probability at each grid point
#     # Fill NaN values with a very low value 
#     filled_prob = forecast_prob.fillna(-9999)
#     # Compute argmax
#     max_category = filled_prob.argmax(dim="probability")
    
#     # Step 2: Create masks for each category
#     mask_bn = max_category == 0  # Below Normal (BN)
#     mask_nn = max_category == 1  # Near Normal (NN)
#     mask_an = max_category == 2  # Above Normal (AN)
    
#     # Step 3: Define custom colormaps
#     BN_cmap = mcolors.LinearSegmentedColormap.from_list('BN', ['#FFF5F0', '#FB6A4A', '#67000D'])
#     NN_cmap = mcolors.LinearSegmentedColormap.from_list('NN', ['#F7FCF5', '#74C476', '#00441B'])
#     AN_cmap = mcolors.LinearSegmentedColormap.from_list('AN', ['#F7FBFF', '#6BAED6', '#08306B'])
    
#     # Create a figure with GridSpec
#     fig = plt.figure(figsize=(8, 6))
#     gs = gridspec.GridSpec(2, 3, height_ratios=[15, 0.5])
    
#     # Main map axis
#     ax = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
    
#     # Step 4: Plot each category
#     # Multiply by 100 to convert probabilities to percentages
#     bn_data = (max_prob.where(mask_bn) * 100).values
#     nn_data = (max_prob.where(mask_nn) * 100).values
#     an_data = (max_prob.where(mask_an) * 100).values
    
#     # Plot BN (Below Normal)
#     bn_plot = ax.pcolormesh(
#         forecast_prob['X'], forecast_prob['Y'], bn_data,
#         cmap=BN_cmap, transform=ccrs.PlateCarree(), alpha=0.9
#     )
    
#     # Plot NN (Near Normal)
#     nn_plot = ax.pcolormesh(
#         forecast_prob['X'], forecast_prob['Y'], nn_data,
#         cmap=NN_cmap, transform=ccrs.PlateCarree(), alpha=0.9
#     )
    
#     # Plot AN (Above Normal)
#     an_plot = ax.pcolormesh(
#         forecast_prob['X'], forecast_prob['Y'], an_data,
#         cmap=AN_cmap, transform=ccrs.PlateCarree(), alpha=0.9
#     )
    
#     # Step 5: Add coastlines and borders
#     ax.coastlines()
#     ax.add_feature(cfeature.BORDERS, linestyle=':')
    
#     # Step 6: Add individual colorbars with ticks at intervals of 5
    
#     # Function to create ticks at intervals of 5
#     def create_ticks(data):
#         data_min = np.nanmin(data)
#         data_max = np.nanmax(data)
#         if data_min == data_max:
#             ticks = [data_min]
#         else:
#             # Round min and max to nearest multiples of 5
#             data_min_rounded = (np.floor(data_min / 5) * 5)+5
#             data_max_rounded = (np.ceil(data_max / 5) * 5)-5
#             ticks = np.arange(data_min_rounded, data_max_rounded + 1, 10)
#         return ticks
    
#     # For BN (Below Normal)
#     bn_ticks = create_ticks(bn_data)
    
#     cbar_ax_bn = fig.add_subplot(gs[1, 0])
#     cbar_bn = plt.colorbar(bn_plot, cax=cbar_ax_bn, orientation='horizontal')
#     cbar_bn.set_label('BN (%)')
#     cbar_bn.set_ticks(bn_ticks)
#     cbar_bn.set_ticklabels([f"{tick:.0f}" for tick in bn_ticks])
    
#     # For NN (Near Normal)
#     nn_ticks = create_ticks(nn_data)
    
#     cbar_ax_nn = fig.add_subplot(gs[1, 1])
#     cbar_nn = plt.colorbar(nn_plot, cax=cbar_ax_nn, orientation='horizontal')
#     cbar_nn.set_label('NN (%)')
#     cbar_nn.set_ticks(nn_ticks)
#     cbar_nn.set_ticklabels([f"{tick:.0f}" for tick in nn_ticks])
    
#     # For AN (Above Normal)
#     an_ticks = create_ticks(an_data)
    
#     cbar_ax_an = fig.add_subplot(gs[1, 2])
#     cbar_an = plt.colorbar(an_plot, cax=cbar_ax_an, orientation='horizontal')
#     cbar_an.set_label('AN (%)')
#     cbar_an.set_ticks(an_ticks)
#     cbar_an.set_ticklabels([f"{tick:.0f}" for tick in an_ticks])
#     ax.set_title(f"Forecast - {model_name}", fontsize=14, pad=20)
#     plt.tight_layout()
#     plt.savefig(f"{dir_to_save}/Forecast_{model_name}_.png", dpi=300, bbox_inches='tight')
#     plt.show()
