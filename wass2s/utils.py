import os
import calendar
import requests
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
from matplotlib import gridspec
from matplotlib.offsetbox import  (OffsetImage, AnnotationBbox)
import matplotlib.image as image
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xeofs as xe
from pathlib import Path
import requests
import rioxarray as rioxr
from tqdm import tqdm
from wass2s.was_compute_predictand import *
from scipy.ndimage import gaussian_filter
from fitter import Fitter
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
from datetime import timedelta
import cdsapi
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


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
        eeof = xe.single.ExtendedEOF(n_modes=2, tau=1, embedding=3, n_pca_modes=20)
        eeof.fit(data_filled, dim="T")
        scores_ext = eeof.scores()
        data_trends = eeof.inverse_transform(scores_ext.sel(mode=1))
        return data_trends
    except Exception as e:
        raise RuntimeError(f"Failed to detrend data using ExtendedEOF: {e}")

def prepare_predictand(dir_to_save_Obs, variables_obs, year_start, year_end, season=None, ds=True, daily=False, param="prcp"):
    """
    Prepare the predictand dataset from observational data.

    Parameters
    ----------
    dir_to_save_Obs : str
        Directory path where observational data is stored.
    variables_obs : list
        List containing the variable string in the format 'center.variable'.
    year_start : int
        Start year of the data.
    year_end : int
        End year of the data.
    season : list, optional
        List of month numbers defining the season (e.g., [6, 7, 8] for JJA).
    ds : bool, optional
        If True, return dataset without converting to array (default is True).
    daily : bool, optional
        If True, load daily data; otherwise, load seasonal data (default is False).
    param : Indicate parameter name

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Prepared predictand dataset or data array.

    Notes
    -----
    Applies a mask for non-daily data based on rainfall thresholds and latitude.
    """
    _, variable = parse_variable(variables_obs[0])
    if daily:
        filepath = f'{dir_to_save_Obs}/Daily_{variable}_{year_start}_{year_end}.nc'
        rainfall = xr.open_dataset(filepath)
        rainfall = xr.where(rainfall < 0.1, 0, rainfall)
        rainfall['T'] = rainfall['T'].astype('datetime64[ns]')
    else:
        season_str = "".join([calendar.month_abbr[int(month)] for month in season])
        filepath = f'{dir_to_save_Obs}/Obs_{variable}_{year_start}_{year_end}_{season_str}.nc'
        rainfall = xr.open_dataset(filepath)
        mean_rainfall = rainfall.mean(dim="T").to_array().squeeze()
        mask = xr.where(mean_rainfall <= 20, np.nan, 1)
        mask = mask.where(abs(mask.Y) <= 20, np.nan)
        rainfall = xr.where(mask == 1, rainfall, np.nan)
        rainfall['T'] = rainfall['T'].astype('datetime64[ns]')
    if ds:
        return rainfall.squeeze().transpose('T', 'Y', 'X').sortby("T")
    else:
        # Check in future how to define the variable name other than 'prcp'
        return rainfall.to_array().drop_vars("variable").squeeze().rename(param).transpose('T', 'Y', 'X').sortby("T")

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

def get_best_models(center_variable, scores, metric='MAE', threshold=None, top_n=6, gcm=False, agroparam=False):

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




def plot_prob_forecasts(dir_to_save, forecast_prob, model_name, labels=["Below-Normal", "Near-Normal", "Above-Normal"], reverse_cmap=True, logo=None, logo_size=0.5):
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
    logo : str, optional
        Path to the logo image to be added to the plot (default is None).
    logo_size : float, optional
        Size of the logo image in the plot (default is 0.5).    

    Notes
    -----
    Saves the plot as a PNG file and displays it.
    Uses custom colormaps for each tercile category.
    """
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
    # BN_cmap = mcolors.LinearSegmentedColormap.from_list('BN', ['#FFF5F0', '#FB6A4A', '#67000D'])
    # NN_cmap = mcolors.LinearSegmentedColormap.from_list('NN', ['#F7FCF5', '#74C476', '#00441B'])
    # AN_cmap = mcolors.LinearSegmentedColormap.from_list('AN', ['#F7FBFF', '#6BAED6', '#08306B'])

    if reverse_cmap:
        
        AN_cmap = mcolors.LinearSegmentedColormap.from_list('AN', ['#FDAE61', '#F46D43', '#D73027']) 
        NN_cmap = mcolors.LinearSegmentedColormap.from_list('NN', ['#FFFFE5', '#FFF7BC', '#FEE391'])
        BN_cmap = mcolors.LinearSegmentedColormap.from_list('BN', ['#ABDDA4', '#66C2A5', '#3288BD'])  
    else:
        BN_cmap = mcolors.LinearSegmentedColormap.from_list('BN', ['#FDAE61', '#F46D43', '#D73027']) 
        NN_cmap = mcolors.LinearSegmentedColormap.from_list('NN', ['#FFFFE5', '#FFF7BC', '#FEE391'])
        AN_cmap = mcolors.LinearSegmentedColormap.from_list('AN', ['#ABDDA4', '#66C2A5', '#3288BD'])          
    
    # Create a figure with GridSpec
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(2, 3, height_ratios=[15, 0.5])
    
    # Main map axis
    ax = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
    
    # Step 4: Plot each category
    # Multiply by 100 to convert probabilities to percentages
    
    # bn_data = (max_prob.where(mask_bn) * 100).values
    # nn_data = (max_prob.where(mask_nn) * 100).values
    # an_data = (max_prob.where(mask_an) * 100).values
    
    # Apply the condition to ensure minimum value of 45% for each category
    # and a maximum of 60% for BN, NN, and AN
    # Ensure that the values are at least 45% and at most 60% for each category
    # and convert to percentage
    # bn_data = xr.where((max_prob.where(mask_bn) * 100) < 45, 45,
    #                    xr.where(max_prob.where(mask_bn) * 100 > 60, 60, max_prob.where(mask_bn) * 100)).values
    # nn_data = xr.where((max_prob.where(mask_nn) * 100) < 45, 45,
    #                    xr.where(max_prob.where(mask_nn) * 100 > 60, 60, max_prob.where(mask_nn) * 100)).values
    # an_data = xr.where((max_prob.where(mask_an) * 100) < 45, 45,
    #                    xr.where(max_prob.where(mask_an) * 100 > 60, 60, max_prob.where(mask_an) * 100)).values    
    
    bn_data = xr.where((xr.where(max_prob.where(mask_bn)>0.6,0.6,max_prob.where(mask_bn))* 100)<45, 45,
                       xr.where(max_prob.where(mask_bn)>0.6,0.6,max_prob.where(mask_bn))* 100).values  
    nn_data = xr.where((xr.where(max_prob.where(mask_nn)>0.6,0.6,max_prob.where(mask_nn))* 100)<45, 45,
                   xr.where(max_prob.where(mask_nn)>0.6,0.6,max_prob.where(mask_nn))* 100).values
    an_data = xr.where((xr.where(max_prob.where(mask_an)>0.6,0.6,max_prob.where(mask_an))* 100)<45, 45,
                   xr.where(max_prob.where(mask_an)>0.6,0.6,max_prob.where(mask_an))* 100).values
     

    
    # Define the data ranges for color normalization  
    # vmin and vmax are set to 35% and 65% respectively for all categories
    # This ensures that the color mapping is consistent across all plots 
     
    vmin = 35  # Minimum probability percentage
    vmax = 65  # Maximum probability percentage

    # Plot BN (Below Normal)
    if np.any(~np.isnan(bn_data)):
        bn_plot = ax.pcolormesh(
            forecast_prob['X'], forecast_prob['Y'], bn_data,
            cmap=BN_cmap, transform=ccrs.PlateCarree(), alpha=0.9, vmin=vmin, vmax=vmax
        )
    else:
        # Create a dummy mappable for BN
        bn_plot = cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=BN_cmap)
        bn_plot.set_array([])

    # Plot NN (Near Normal)
    if np.any(~np.isnan(nn_data)):
        nn_plot = ax.pcolormesh(
            forecast_prob['X'], forecast_prob['Y'], nn_data,
            cmap=NN_cmap, transform=ccrs.PlateCarree(), alpha=0.9, vmin=vmin, vmax=vmax
        )
    else:
        # Create a dummy mappable for NN
        nn_plot = cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=NN_cmap)
        nn_plot.set_array([])

    # Plot AN (Above Normal)
    if np.any(~np.isnan(an_data)):
        an_plot = ax.pcolormesh(
            forecast_prob['X'], forecast_prob['Y'], an_data,
            cmap=AN_cmap, transform=ccrs.PlateCarree(), alpha=0.9, vmin=vmin, vmax=vmax
        )
    else:
        # Create a dummy mappable for AN
        an_plot = cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=AN_cmap)
        an_plot.set_array([])

    # Step 5: Add coastlines and borders
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    
    # Step 6: Add individual colorbars with fixed ticks
    
    # Function to create ticks at intervals of 10 from 0 to 100
    def create_ticks():
        ticks = np.arange(35, 66, 5)
        return ticks

    ticks = create_ticks()

    # For BN (Below Normal)
    cbar_ax_bn = fig.add_subplot(gs[1, 0])
    cbar_bn = plt.colorbar(bn_plot, cax=cbar_ax_bn, orientation='horizontal')
    cbar_bn.set_label(f'{labels[0]} (%)')
    cbar_bn.set_ticks(ticks)
    cbar_bn.set_ticklabels([f"{tick}" for tick in ticks])

    # For NN (Near Normal)
    cbar_ax_nn = fig.add_subplot(gs[1, 1])
    cbar_nn = plt.colorbar(nn_plot, cax=cbar_ax_nn, orientation='horizontal')
    cbar_nn.set_label(f'{labels[1]} (%)')
    cbar_nn.set_ticks(ticks)
    cbar_nn.set_ticklabels([f"{tick}" for tick in ticks])

    # For AN (Above Normal)
    cbar_ax_an = fig.add_subplot(gs[1, 2])
    cbar_an = plt.colorbar(an_plot, cax=cbar_ax_an, orientation='horizontal')
    cbar_an.set_label(f'{labels[2]} (%)')
    cbar_an.set_ticks(ticks)
    cbar_an.set_ticklabels([f"{tick}" for tick in ticks])
    
    # Set the title with the formatted model_name
    # Convert model_name to string if necessary
    if isinstance(model_name, np.ndarray):
        model_name_str = str(model_name.item())
    else:
        model_name_str = str(model_name)
    ax.set_title(f"{model_name_str}", fontsize=13, pad=20)
    # Step 7: Add logo if provided
    if logo_path is not None:
        im = image.imread(logo)
        # add logo
        addLogo = OffsetImage(im, zoom=logo_size)
        # addLogo.image.axes = ax
        addLogo.set_offset('lower right')
        ab = AnnotationBbox(addLogo, (1.1, -0.1), frameon=False, xycoords='axes fraction', boxcoords="axes fraction")
        ax.add_artist(ab)   
    # Step 8: Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(f"{dir_to_save}/{model_name_str.replace(" ", "_")}.png", dpi=300, bbox_inches='tight')
    plt.show()
    

def plot_prob_forecasts_(dir_to_save, forecast_prob, model_name):
    """
    Plot probabilistic forecasts with tercile categories using contourf.

    Parameters
    ----------
    dir_to_save : str
        Directory path to save the plot.
    forecast_prob : xarray.DataArray
        Data array containing probability forecasts with a 'probability' dimension.
    model_name : str or numpy.ndarray
        Name of the model for the plot title.

    Notes
    -----
    Similar to plot_prob_forecasts but uses contourf instead of pcolormesh for plotting.
    Saves the plot as a PNG file and displays it.
    """
    max_prob = forecast_prob.max(dim="probability", skipna=True)
    filled_prob = forecast_prob.fillna(-9999)
    max_category = filled_prob.argmax(dim="probability")
    mask_bn = max_category == 0
    mask_nn = max_category == 1
    mask_an = max_category == 2
    BN_cmap = mcolors.LinearSegmentedColormap.from_list('BN', ['#FDAE61', '#F46D43', '#D73027'])
    NN_cmap = mcolors.LinearSegmentedColormap.from_list('NN', ['#FFFFE5', '#FFF7BC', '#FFFFCC'])
    AN_cmap = mcolors.LinearSegmentedColormap.from_list('AN', ['#ABDDA4', '#66C2A5', '#3288BD'])
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(2, 3, height_ratios=[15, 0.5])
    ax = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
    bn_data = xr.where((xr.where(max_prob.where(mask_bn) > 0.6, 0.6, max_prob.where(mask_bn)) * 100) < 45, 45,
                       xr.where(max_prob.where(mask_bn) > 0.6, 0.6, max_prob.where(mask_bn)) * 100).values
    nn_data = xr.where((xr.where(max_prob.where(mask_nn) > 0.6, 0.6, max_prob.where(mask_nn)) * 100) < 45, 45,
                       xr.where(max_prob.where(mask_nn) > 0.6, 0.6, max_prob.where(mask_nn)) * 100).values
    an_data = xr.where((xr.where(max_prob.where(mask_an) > 0.6, 0.6, max_prob.where(mask_an)) * 100) < 45, 45,
                       xr.where(max_prob.where(mask_an) > 0.6, 0.6, max_prob.where(mask_an)) * 100).values
    vmin, vmax = 35, 65
    if np.any(~np.isnan(bn_data)):
        bn_plot = ax.contourf(
            forecast_prob['X'], forecast_prob['Y'], bn_data,
            cmap=BN_cmap, transform=ccrs.PlateCarree(), alpha=0.9, vmin=vmin, vmax=vmax
        )
    else:
        bn_plot = cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=BN_cmap)
        bn_plot.set_array([])
    if np.any(~np.isnan(nn_data)):
        nn_plot = ax.contourf(
            forecast_prob['X'], forecast_prob['Y'], nn_data,
            cmap=NN_cmap, transform=ccrs.PlateCarree(), alpha=0.9, vmin=vmin, vmax=vmax
        )
    else:
        nn_plot = cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=NN_cmap)
        nn_plot.set_array([])
    if np.any(~np.isnan(an_data)):
        an_plot = ax.contourf(
            forecast_prob['X'], forecast_prob['Y'], an_data,
            cmap=AN_cmap, transform=ccrs.PlateCarree(), alpha=0.9, vmin=vmin, vmax=vmax
        )
    else:
        an_plot = cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=AN_cmap)
        an_plot.set_array([])
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    def create_ticks():
        return np.arange(35, 66, 5)
    ticks = create_ticks()
    cbar_ax_bn = fig.add_subplot(gs[1, 0])
    cbar_bn = plt.colorbar(bn_plot, cax=cbar_ax_bn, orientation='horizontal')
    cbar_bn.set_label('Below-Normal (%)')
    cbar_bn.set_ticks(ticks)
    cbar_bn.set_ticklabels([f"{tick}" for tick in ticks])
    cbar_ax_nn = fig.add_subplot(gs[1, 1])
    cbar_nn = plt.colorbar(nn_plot, cax=cbar_ax_nn, orientation='horizontal')
    cbar_nn.set_label('Near-Normal (%)')
    cbar_nn.set_ticks(ticks)
    cbar_nn.set_ticklabels([f"{tick}" for tick in ticks])
    cbar_ax_an = fig.add_subplot(gs[1, 2])
    cbar_an = plt.colorbar(an_plot, cax=cbar_ax_an, orientation='horizontal')
    cbar_an.set_label('Above-Normal (%)')
    cbar_an.set_ticks(ticks)
    cbar_an.set_ticklabels([f"{tick}" for tick in ticks])
    model_name_str = str(model_name.item()) if isinstance(model_name, np.ndarray) else str(model_name)
    ax.set_title(f"Forecast Probabilities - {model_name_str}", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(f"{dir_to_save}/Forecast_{model_name_str}_.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_tercile(A):
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
    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    colors = ['#fc8d59', '#ffffbf', '#99d594']
    cmap = ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = BoundaryNorm(bounds, cmap.N)
    lon = A['X']
    lat = A['Y']
    img = ax.pcolormesh(lon, lat, A.isel(T=0), cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.BORDERS, linewidth=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=1)
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
    plt.title("Terciles MAP", fontsize=16, weight='bold')
    legend_elements = [
        mpatches.Patch(color='#99d594', label='ABOVE'),
        mpatches.Patch(color='#ffffbf', label='NORMAL'),
        mpatches.Patch(color='#fc8d59', label='BELOW')
    ]
    plt.legend(handles=legend_elements, loc='lower left')
    plt.tight_layout()
    plt.show()

def find_best_distribution_grid(rainfall, distribution_map=None):
    """
    Determine the best-fitting distribution for variables data at each grid cell.

    Parameters
    ----------
    rainfall : xarray.DataArray
        Precipitation data with a time dimension 'T' and spatial dimensions 'Y', 'X'.
    distribution_map : dict, optional
        Mapping of distribution names to numeric codes. Defaults to:
        {'norm': 1, 'lognorm': 2, 'expon': 3, 'gamma': 4, 'weibull_min': 5}

    Returns
    -------
    xarray.DataArray
        Array with numeric codes for the best-fitting distribution at each grid cell.

    Notes
    -----
    Uses the Fitter library to fit distributions and select the best based on sum-of-squared errors.
    """
    if distribution_map is None:
        distribution_map = {
            'norm': 1,
            'lognorm': 2,
            'expon': 3,
            'gamma': 4,
            'weibull_min': 5
        }
    def find_best_distribution(precip_data, distribution_map):
        precip_data = np.asarray(precip_data)
        if np.isnan(precip_data).all():
            return np.nan
        f = Fitter(precip_data, distributions=list(distribution_map.keys()))
        f.fit()
        best_fit = f.get_best(method='sumsquare_error')
        best_dist_name = list(best_fit.keys())[0]
        return distribution_map.get(best_dist_name, np.nan)
    best_fit_da = xr.apply_ufunc(
        find_best_distribution,
        rainfall,
        input_core_dims=[["T"]],
        kwargs={'distribution_map': distribution_map},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float]
    )
    return best_fit_da

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
        dayofyear = obs_hdcst['T'].dt.dayofyear
        daily_climatology = da.groupby(dayofyear).mean(dim='T')
        data = daily_climatology.to_numpy()
        dummy = xr.DataArray(
            data=data,
            coords={"T": t_coord, "Y": y_coords, "X": x_coords},
            dims=["T", "Y", "X"]
        ) * mask
    else:
        da_noleap = obs_hdcst.sel(T=~((obs_hdcst['T'].dt.month == 2) & (obs_hdcst['T'].dt.day == 29)))
        daily_climatology = da_noleap.groupby('T.dayofyear').mean(dim='T')
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

def et0_fao56_daily(tmax, tmin, tdew, u10, v10, rs, mlsp, dem):
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
    u10 : xarray.DataArray
        Daily mean zonal wind speed at 10 m height in m s^-1, with dimensions ('T', 'Y', 'X').
    v10 : xarray.DataArray
        Daily mean meridional wind speed at 10 m height in m s^-1, with dimensions ('T', 'Y', 'X').
    rs : xarray.DataArray
        Daily mean incoming solar radiation in W m^-2 or MJ m^-2 day^-1, with dimensions ('T', 'Y', 'X').
    mlsp : xarray.DataArray
        Daily mean sea-level pressure in Pa, with dimensions ('T', 'Y', 'X').
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
    P0 = mlsp / 1000.0  # Pa to kPa
    P = P0 * (1 - 0.0065 * dem / (tmean + 273.15))**5.257  # Hypsometric equation
    PSI = 0.665e-3 * P  # Psychrometric constant (kPa °C^-1)

    # --- 4 Wind ---
    ws10 = np.hypot(u10, v10)  # Wind speed at 10 m
    u2 = ws10 * 4.87 / np.log(67.8 * 10.0 - 5.42)  # Wind speed at 2 m

    # --- 5 Vapour pressures ---
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
