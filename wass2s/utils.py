import os
import calendar
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
from matplotlib import gridspec
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


def decode_cf(ds, time_var):
    """Decodes time dimension to CFTime standards."""
    if ds[time_var].attrs["calendar"] == "360":
        ds[time_var].attrs["calendar"] = "360_day"
    ds = xr.decode_cf(ds, decode_times=True)
    return ds

# --------------------------------------------------------------------------
# Helpers to convert numeric lat/lon --> IRIDL string, e.g. -15 -> "15W"
# --------------------------------------------------------------------------
def to_iridl_lat(lat: float) -> str:
    """ Convert numeric latitude to IRIDL lat string: +10 -> '10N', -5 -> '5S'. """
    abs_val = abs(lat)
    suffix = "N" if lat >= 0 else "S"
    return f"{abs_val:g}{suffix}"

def to_iridl_lon(lon: float) -> str:
    """ Convert numeric longitude to IRIDL lon string: +15 -> '15E', -15 -> '15W'. """
    abs_val = abs(lon)
    suffix = "E" if lon >= 0 else "W"
    return f"{abs_val:g}{suffix}"

# --------------------------------------------------------------------------
# Builder for IRIDL URL for NOAA ERSST
# --------------------------------------------------------------------------
def build_iridl_url_ersst(
    year_start: int,
    year_end: int,
    bbox: list,         # e.g. [N, W, S, E] = [10, -15, -5, 15]
    run_avg: int = 3,   # e.g. 3 => T/3/runningAverage
    month_start: str = "Jan",
    month_end: str   = "Dec",
):
    """
    Build a parameterized IRIDL URL for NOAA/ERSST, using a numeric bounding box
    of the form [North, West, South, East].
      e.g. area = [10, -15, -5, 15].

    IRIDL wants Y/(south)/(north)/..., X/(west)/(east)/..., so we reorder:
      * south = area[2]
      * north = area[0]
      * west  = area[1]
      * east  = area[3]

    If run_avg is provided, we'll append T/<run_avg>/runningAverage/.
    """
    # 1) Extract numeric values from the bounding box
    #    area = [N, W, S, E]
    north, w, south, e = bbox

    # 2) Convert numeric => IRIDL-friendly strings
    south_str = to_iridl_lat(south)  # e.g. -5  -> '5S'
    north_str = to_iridl_lat(north)  # e.g.  10 -> '10N'
    west_str  = to_iridl_lon(w)      # e.g. -15 -> '15W'
    east_str  = to_iridl_lon(e)      # e.g.  15 -> '15E'

    # 3) Time range, e.g. T/(Jan%201991)/(Dec%202024)/RANGEEDGES/
    t_start_str = f"{month_start}%20{year_start}"  # e.g. 'Jan%201991'
    t_end_str   = f"{month_end}%20{year_end}"      # e.g. 'Dec%202024'
    time_part   = f"T/({t_start_str})/({t_end_str})/RANGEEDGES/"

    # 4) Lat/Lon part, e.g. Y/(5S)/(10N)/RANGEEDGES/ X/(15W)/(15E)/RANGEEDGES/
    latlon_part = (
        f"Y/({south_str})/({north_str})/RANGEEDGES/"
        f"X/({west_str})/({east_str})/RANGEEDGES/"
    )

    # 5) Possibly add run-average
    runavg_part = f"T/{run_avg}/runningAverage/" if run_avg is not None else ""

    # 6) Combine
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
    # We'll parse out just the YEAR from each original date.
    years = pd.to_datetime(ds.T.values).year  # array of integer years

    # seas[1] is presumably a string like "11" for November
    new_dates = []
    for y in years:
        # Build "YYYY-{month}-01", e.g. "1991-11-01"
        new_dates.append(np.datetime64(f"{y}-{seas[1]}-01"))

    ds = ds.assign_coords(T=("T", new_dates))
    ds["T"] = ds["T"].astype("datetime64[ns]")
    return ds

def download_file(url, local_path, force_download=False, chunk_size=8192, timeout=120):
    local_path = Path(local_path)

    # Skip download if file exists and force_download is False
    if local_path.exists() and not force_download:
        print(f"[SKIP] {local_path} already exists.")
        return local_path

    print(f"[DOWNLOAD] {url}")

    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            
            # Get total file size from response headers
            total_size = int(r.headers.get('content-length', 0))

            # Download with progress bar
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
    """Extract center and variable names from the variables list."""
    center = variables_list.split(".")[0]
    variable = variables_list.split(".")[1]
    return center, variable

def standardize_timeseries(ds, clim_year_start=None, clim_year_end=None):
    """Standardize the dataset over a specified climatology period."""
    if clim_year_start is not None and clim_year_end is not None:
        clim_slice = slice(str(clim_year_start), str(clim_year_end))
        clim_mean = ds.sel(T=clim_slice).mean(dim='T')
        clim_std = ds.sel(T=clim_slice).std(dim='T')
    else:
        clim_mean = ds.mean(dim='T')
        clim_std = ds.std(dim='T')
    return (ds - clim_mean) / clim_std

def reverse_standardize(ds_st, ds, clim_year_start=None, clim_year_end=None):

    if clim_year_start is not None and clim_year_end is not None:
        clim_slice = slice(str(clim_year_start), str(clim_year_end))
        clim_mean = ds.sel(T=clim_slice).mean(dim='T')
        clim_std = ds.sel(T=clim_slice).std(dim='T')
    else:
        clim_mean = ds.mean(dim='T')
        clim_std = ds.std(dim='T')
    return ds_st*clim_std + clim_mean

def anomalize_timeseries(ds, clim_year_start=None, clim_year_end=None):

    if clim_year_start is not None and clim_year_end is not None:
        clim_slice = slice(str(clim_year_start), str(clim_year_end))
        clim_mean = ds.sel(T=clim_slice).mean(dim='T')
        # clim_std = ds.sel(T=clim_slice).std(dim='T')
    else:
        clim_mean = ds.mean(dim='T')
        # clim_std = ds.std(dim='T')
    return (ds - clim_mean) #/ clim_std

def predictant_mask(data):
    mean_rainfall = data.mean(dim="T").squeeze()
    mask = xr.where(mean_rainfall <= 20, np.nan, 1)
    mask = mask.where(abs(mask.Y) <= 19.5, np.nan)
    return mask


def trend_data(data):
    """
    trend the data using ExtendedEOF if detrending is enabled.

    Parameters:
    - data: xarray DataArray we want to know trend.

    Returns:
    - data_trended: trended xarray DataArray.
    """
    try:
        # Fill missing values with 0
        # data_filled = data.fillna(0)
        data_filled = data.fillna(data.mean(dim="T", skipna=True))

        # Initialize ExtendedEOF
        eeof = xe.single.ExtendedEOF(n_modes=2, tau=1, embedding=3, n_pca_modes=20)
        eeof.fit(data_filled, dim="T")

        # Extract trends using the first mode
        scores_ext = eeof.scores()
        data_trends = eeof.inverse_transform(scores_ext.sel(mode=1))

        # # Subtract trends to get detrended data
        # data_detrended = data_filled - data_trends
        return data_trends#.fillna(data_trends[-3])
    except Exception as e:
        raise RuntimeError(f"Failed to detrend data using ExtendedEOF: {e}")
        
    
def prepare_predictand(dir_to_save_Obs, variables_obs, year_start, year_end, season=None, ds=True, daily=False):
    """Prepare the predictand dataset."""
    _, variable = parse_variable(variables_obs[0])
    
    if daily:
        filepath = f'{dir_to_save_Obs}/Daily_{variable}_{year_start}_{year_end}.nc'
        rainfall = xr.open_dataset(filepath)
        rainfall = xr.where(rainfall<0.1, 0, rainfall)
        rainfall['T'] = rainfall['T'].astype('datetime64[ns]')
    else:
        season_str = "".join([calendar.month_abbr[int(month)] for month in season])
        filepath = f'{dir_to_save_Obs}/Obs_{variable}_{year_start}_{year_end}_{season_str}.nc'
        rainfall = xr.open_dataset(filepath)
    
        # Create mask
        mean_rainfall = rainfall.mean(dim="T").to_array().squeeze()
        mask = xr.where(mean_rainfall <= 20, np.nan, 1)
        mask = mask.where(abs(mask.Y) <= 20, np.nan)
        rainfall = xr.where(mask == 1, rainfall, np.nan)
        rainfall['T'] = rainfall['T'].astype('datetime64[ns]')
        # rainfall['T'] = pd.to_datetime(rainfall['T'].values)
    if ds :
        return rainfall.drop_vars("variable").squeeze().transpose( 'T', 'Y', 'X').sortby("T")
    else:
        return rainfall.to_array().drop_vars("variable").squeeze().rename("prcp").transpose( 'T', 'Y', 'X').sortby("T")


def load_gridded_predictor(dir_to_data, variables_list, year_start, year_end, season=None, model=False, month_of_initialization=None, lead_time=None, year_forecast=None):
    """Load gridded predictor data for reanalysis or model."""
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

def compute_sst_indices(dir_to_data, indices, variables_list, year_start, year_end, season, clim_year_start=None, clim_year_end=None, others_zone=None, model=False, month_of_initialization=None, lead_time=None, year_forecast=None):
    """Compute SST indices for reanalysis or model data."""
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
            

    # Compute derived indices
    predictor["TASI"] = predictor["NAT"] - predictor["SAT"]
    predictor["DMI"] = predictor["WTIO"] - predictor["SETIO"]

    selected_indices = {i: predictor[i] for i in indices}
    data_vars = {key: ds[variable.lower()].rename(key) for key, ds in selected_indices.items()}
    combined_dataset = xr.Dataset(data_vars)
    return combined_dataset


def compute_other_indices(dir_to_data, indices_dict, variables_list, year_start, year_end, season, clim_year_start=None, clim_year_end=None, model=False, month_of_initialization=None, lead_time=None, year_forecast=None):
    """Compute indices for other variables."""
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

# retrieve Zone for PCR

###### Code to use after in several zones for PCR ##############################
# pca= xe.single.EOF(n_modes=6, use_coslat=True, standardize=True)
# pca.fit([i.fillna(i.mean(dim="T", skipna=True)).rename({"X": "lon", "Y": "lat"}) for i in predictor], dim="T")
# components = pca.components()
# scores = pca.scores()
# expl = pca.explained_variance_ratio()
# expl
################################################################################################

def retrieve_several_zones_for_PCR(dir_to_data, indices_dict, variables_list, year_start, year_end, season, clim_year_start=None, clim_year_end=None, model=False, month_of_initialization=None, lead_time=None, year_forecast=None):
    """Compute indices for other variables."""
    center, variable = parse_variable(variables_list)
    if model:
        abb_month_ini = calendar.month_abbr[int(month_of_initialization)]
        season_str = "".join([calendar.month_abbr[(int(i) + int(month_of_initialization)) % 12 or 12] for i in lead_time])
        center = center.lower().replace("_", "")
        # file_prefix = "forecast" if year_forecast else "hindcast"
        filepath_hdcst = f"{dir_to_data}/hindcast_{center}_{variable}_{abb_month_ini}Ic_{season_str}_{lead_time[0]}.nc"
        filepath_fcst = f"{dir_to_data}/forecast_{center}_{variable}_{abb_month_ini}Ic_{season_str}_{lead_time[0]}.nc"
        data_hdcst = xr.open_dataset(filepath_hdcst).to_array().drop_vars('variable').squeeze('variable')
        data_hdcst['T'] = data_hdcst['T'].astype('datetime64[ns]')    
        data_fcst = xr.open_dataset(filepath_fcst).to_array().drop_vars('variable').squeeze('variable')
        data_fcst['T'] = data_fcst['T'].astype('datetime64[ns]')
        data = xr.concat([data_hdcst,data_fcst], dim='T')
    else:
        season_str = "".join([calendar.month_abbr[int(month)] for month in season])
        filepath = f'{dir_to_data}/{center}_{variable}_{year_start}_{year_end}_{season_str}.nc'
        data = xr.open_dataset(filepath).to_array().drop_vars('variable').squeeze('variable')
        data['T'] = data['T'].astype('datetime64[ns]')
        # data['T'] = pd.to_datetime(data['T'].values)

    predictor = {}
    for idx, coords in indices_dict.items():
        _, lon_min, lon_max, lat_min, lat_max = coords
        var_region = data.sel(X=slice(lon_min, lon_max), Y=slice(lat_min, lat_max))
        var_region = standardize_timeseries(var_region, clim_year_start, clim_year_end)
        predictor[idx] = var_region

    data_vars = [ds.rename(key) for key, ds in predictor.items()]
    return data_vars

def retrieve_single_zone_for_PCR(dir_to_data, indices_dict, variables_list, year_start, year_end, season=None, clim_year_start=None, clim_year_end=None, model=False, month_of_initialization=None, lead_time=None, year_forecast=None):
    """Compute indices for other variables."""
    center, variable = parse_variable(variables_list)
    if model:
        abb_month_ini = calendar.month_abbr[int(month_of_initialization)]
        season_str = "".join([calendar.month_abbr[(int(i) + int(month_of_initialization)) % 12 or 12] for i in lead_time])
        center = center.lower().replace("_", "")
        # file_prefix = "forecast" if year_forecast else "hindcast"
        filepath_hdcst = f"{dir_to_data}/hindcast_{center}_{variable}_{abb_month_ini}Ic_{season_str}_{lead_time[0]}.nc"
        filepath_fcst = f"{dir_to_data}/forecast_{center}_{variable}_{abb_month_ini}Ic_{season_str}_{lead_time[0]}.nc"
        data_hdcst = xr.open_dataset(filepath_hdcst).to_array().drop_vars('variable').squeeze('variable')
        data_hdcst['T'] = data_hdcst['T'].astype('datetime64[ns]')    
        data_fcst = xr.open_dataset(filepath_fcst).to_array().drop_vars('variable').squeeze('variable')
        data_fcst['T'] = data_fcst['T'].astype('datetime64[ns]')
        data = xr.concat([data_hdcst,data_fcst], dim='T')
    else:
        season_str = "".join([calendar.month_abbr[int(month)] for month in season])
        filepath = f'{dir_to_data}/{center}_{variable}_{year_start}_{year_end}_{season_str}.nc'
        data = xr.open_dataset(filepath).to_array().drop_vars('variable').squeeze('variable')
        data['T'] = data['T'].astype('datetime64[ns]')
        # data['T'] = pd.to_datetime(data['T'].values)
    
    new_resolution = {
        'Y': 1,  # For example, set a resolution of 1 degree for latitude
        'X': 1   # For example, set a resolution of 1 degree for longitude
        }
    for idx, coords in indices_dict.items():
        _, lon_min, lon_max, lat_min, lat_max = coords
        var_region = data.sel(X=slice(lon_min, lon_max), Y=slice(lat_min, lat_max))
        var_region = standardize_timeseries(var_region, clim_year_start, clim_year_end)
        # Generate the new coordinate grid for interpolation
        Y_new = xr.DataArray(np.arange(lat_min, lat_max+1, new_resolution['Y']), dims='Y')
        X_new = xr.DataArray(np.arange(lon_min, lon_max+1, new_resolution['X']), dims='X')
        # Interpolate the original data onto the new grid
        data_vars = var_region.interp(Y=Y_new, X=X_new, method='linear')
    return data_vars



def plot_map(extent, title="Map", sst_indices=None, fig_size=(10,8)): 
    """
    Plots a map with specified geographic extent and optionally adds SST index boxes.

    Parameters:
    - extent: list of float, specifying [west, east, south, north]
    - title: str, title of the map
    - sst_indices: dict, optional dictionary containing SST index information
    """
    # Create figure and axis for the map
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=fig_size)

    # Set the geographic extent
    ax.set_extent(extent) 
    
    # Add map features
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAND, edgecolor="black")
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    
    # Add SST index boxes if provided
    if sst_indices:
        for index, (label, lon_w, lon_e, lat_s, lat_n) in sst_indices.items():
            if lon_w is not None:  # Only add box if the coordinates are valid
                ax.add_patch(Rectangle(
                    (lon_w, lat_s), lon_e - lon_w, lat_n - lat_s, 
                    linewidth=2, edgecolor='red', facecolor='none', linestyle='--'))
                ax.text(lon_w + 1, lat_s + 1, index, color='red', fontsize=10, ha='left')
    
    # Set title
    ax.set_title(title)
    
    # Show plot
    plt.tight_layout()
    plt.show()


def save_hindcast_and_forecasts(dir_to_save, data, variable, forecast=None, deterministic=True):
    pass


def save_validation_score(dir_to_save, data, metric, model_name):
    pass
    
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


def plot_prob_forecasts(dir_to_save, forecast_prob, model_name, labels=["Below-Normal", "Near-Normal", "Above-Normal"], reverse_cmap=True):    

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
    
    bn_data = xr.where((xr.where(max_prob.where(mask_bn)>0.6,0.6,max_prob.where(mask_bn))* 100)<45, 45,
                       xr.where(max_prob.where(mask_bn)>0.6,0.6,max_prob.where(mask_bn))* 100).values  
    nn_data = xr.where((xr.where(max_prob.where(mask_nn)>0.6,0.6,max_prob.where(mask_nn))* 100)<45, 45,
                   xr.where(max_prob.where(mask_nn)>0.6,0.6,max_prob.where(mask_nn))* 100).values
    an_data = xr.where((xr.where(max_prob.where(mask_an)>0.6,0.6,max_prob.where(mask_an))* 100)<45, 45,
                   xr.where(max_prob.where(mask_an)>0.6,0.6,max_prob.where(mask_an))* 100).values
     

    
    # Define the data ranges for color normalization
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
    ax.set_title(f"Probabilistic Forecast - {model_name_str}", fontsize=13, pad=20)
    
    plt.tight_layout()
    plt.savefig(f"{dir_to_save}/Forecast_{model_name_str}_.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_prob_forecasts_(dir_to_save, forecast_prob, model_name):    

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
    
    BN_cmap = mcolors.LinearSegmentedColormap.from_list('BN', ['#FDAE61', '#F46D43', '#D73027']) 
    NN_cmap = mcolors.LinearSegmentedColormap.from_list('NN', ['#FFFFE5', '#FFF7BC', '#FFFFCC'])
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
    
    bn_data = xr.where((xr.where(max_prob.where(mask_bn)>0.6,0.6,max_prob.where(mask_bn))* 100)<45, 45,
                       xr.where(max_prob.where(mask_bn)>0.6,0.6,max_prob.where(mask_bn))* 100).values  
    nn_data = xr.where((xr.where(max_prob.where(mask_nn)>0.6,0.6,max_prob.where(mask_nn))* 100)<45, 45,
                   xr.where(max_prob.where(mask_nn)>0.6,0.6,max_prob.where(mask_nn))* 100).values
    an_data = xr.where((xr.where(max_prob.where(mask_an)>0.6,0.6,max_prob.where(mask_an))* 100)<45, 45,
                   xr.where(max_prob.where(mask_an)>0.6,0.6,max_prob.where(mask_an))* 100).values
     

    
    # Define the data ranges for color normalization
    vmin = 35  # Minimum probability percentage
    vmax = 65  # Maximum probability percentage
    
    # Plot BN (Below Normal)
    if np.any(~np.isnan(bn_data)):
        bn_plot = ax.contourf(
            forecast_prob['X'], forecast_prob['Y'], bn_data,
            cmap=BN_cmap, transform=ccrs.PlateCarree(), alpha=0.9, vmin=vmin, vmax=vmax
        )
    else:
        # Create a dummy mappable for BN
        bn_plot = cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=BN_cmap)
        bn_plot.set_array([])

    # Plot NN (Near Normal)
    if np.any(~np.isnan(nn_data)):
        nn_plot = ax.contourf(
            forecast_prob['X'], forecast_prob['Y'], nn_data,
            cmap=NN_cmap, transform=ccrs.PlateCarree(), alpha=0.9, vmin=vmin, vmax=vmax
        )
    else:
        # Create a dummy mappable for NN
        nn_plot = cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=NN_cmap)
        nn_plot.set_array([])

    # Plot AN (Above Normal)
    if np.any(~np.isnan(an_data)):
        an_plot = ax.contourf(
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
    cbar_bn.set_label('Below-Normal (%)')
    cbar_bn.set_ticks(ticks)
    cbar_bn.set_ticklabels([f"{tick}" for tick in ticks])

    # For NN (Near Normal)
    cbar_ax_nn = fig.add_subplot(gs[1, 1])
    cbar_nn = plt.colorbar(nn_plot, cax=cbar_ax_nn, orientation='horizontal')
    cbar_nn.set_label('Near-Normal (%)')
    cbar_nn.set_ticks(ticks)
    cbar_nn.set_ticklabels([f"{tick}" for tick in ticks])

    # For AN (Above Normal)
    cbar_ax_an = fig.add_subplot(gs[1, 2])
    cbar_an = plt.colorbar(an_plot, cax=cbar_ax_an, orientation='horizontal')
    cbar_an.set_label('Above-Normal (%)')
    cbar_an.set_ticks(ticks)
    cbar_an.set_ticklabels([f"{tick}" for tick in ticks])
    
    # Set the title with the formatted model_name
    # Convert model_name to string if necessary
    if isinstance(model_name, np.ndarray):
        model_name_str = str(model_name.item())
    else:
        model_name_str = str(model_name)
    ax.set_title(f"Forecast Probabilities - {model_name_str}", fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(f"{dir_to_save}/Forecast_{model_name_str}_.png", dpi=300, bbox_inches='tight')
    plt.show()
    

def plot_tercile(A):
    # Step 3: Plotting
    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Custom colormap: brown (below), light cyan (normal), green (above)
    colors = ['#fc8d59', '#ffffbf', '#99d594']
    cmap = ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = BoundaryNorm(bounds, cmap.N)
    
    # Plot
    lon = A['X']
    lat = A['Y']
    img = ax.pcolormesh(lon, lat, A.isel(T=0), cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
    
    # Borders and coastlines
    ax.add_feature(cfeature.BORDERS, linewidth=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=1)
    
    # Optional: mask ocean
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
    
    # Title
    plt.title("Terciles MAP", fontsize=16, weight='bold')
    
    # Custom legend
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
    Apply a function across the rainfall DataArray (assumed to have a 'T' dimension)
    to determine the best-fitting distribution, returning a grid of numeric codes.
    
    Parameters
    ----------
    rainfall : xarray.DataArray
        Precipitation data with a time dimension 'T' and additional spatial dimensions.
    distribution_map : dict, optional
        A mapping of distribution names to numeric codes. Defaults to:
            {
                'norm': 1,
                'lognorm': 2,
                'expon': 3,
                'gamma': 4,
                'weibull_min': 5
            }
    
    Returns
    -------
    best_fit_da : xarray.DataArray
        An array of the same spatial dimensions as rainfall with the best-fitting 
        distribution's numeric code at each grid cell.
    """
    # Define default distribution_map if not provided
    if distribution_map is None:
        distribution_map = {
            'norm': 1,
            'lognorm': 2,
            'expon': 3,
            'gamma': 4,
            'weibull_min': 5
        }
    
    def find_best_distribution(precip_data, distribution_map):
        """
        Fits multiple distributions to precipitation data and returns the best-fitting 
        distribution's numeric code.
        """
        # Convert input to a 1D NumPy array
        precip_data = np.asarray(precip_data)
        
        # Skip if all values are NaN (e.g., ocean grid cells)
        if np.isnan(precip_data).all():
            return np.nan
        
        # Fit distributions using the provided distribution_map keys
        f = Fitter(precip_data, distributions=list(distribution_map.keys()))
        f.fit()
    
        # Get the best-fitting distribution using the sum-of-squared errors method
        best_fit = f.get_best(method='sumsquare_error')
        best_dist_name = list(best_fit.keys())[0]  # Get the best-fitting distribution name
    
        # Return the corresponding numeric code
        return distribution_map.get(best_dist_name, np.nan)

    # Apply the function along the 'T' dimension using xarray's apply_ufunc
    best_fit_da = xr.apply_ufunc(
        find_best_distribution, 
        rainfall, 
        input_core_dims=[["T"]],  # Function expects 1D array along T
        kwargs={'distribution_map': distribution_map},
        vectorize=True,           # Broadcast over non-core dimensions
        dask="parallelized",      # If using dask arrays
        output_dtypes=[float]     # Numeric code output, float to accommodate NaN
    )
    
    return best_fit_da
    

################################ agroparameters compute ################

onset_criteria = {
0: {"zone_name": "Sahel100_0mm", "start_search": "06-01", "cumulative": 10, "number_dry_days": 25, "thrd_rain_day": 0.85, "end_search": "08-30"},
1: {"zone_name": "Sahel200_100mm", "start_search": "05-15", "cumulative": 15, "number_dry_days": 25, "thrd_rain_day": 0.85, "end_search": "08-15"},
2: {"zone_name": "Sahel400_200mm", "start_search": "05-01", "cumulative": 15, "number_dry_days": 20, "thrd_rain_day": 0.85, "end_search": "07-31"},
3: {"zone_name": "Sahel600_400mm", "start_search": "03-15", "cumulative": 20, "number_dry_days": 20, "thrd_rain_day": 0.85, "end_search": "07-31"},
4: {"zone_name": "Soudan",         "start_search": "03-15", "cumulative": 20, "number_dry_days": 10, "thrd_rain_day": 0.85, "end_search": "07-31"},
5: {"zone_name": "Golfe_Of_Guinea","start_search": "02-01", "cumulative": 20, "number_dry_days": 10, "thrd_rain_day": 0.85, "end_search": "06-15"},
    }

onset_dryspell_criteria = {
    0: {"zone_name": "Sahel100_0mm", "start_search": "06-01", "cumulative": 10, "number_dry_days": 25, "thrd_rain_day": 0.85, "end_search": "08-15", "nbjour":35},
    1: {"zone_name": "Sahel200_100mm", "start_search": "05-15", "cumulative": 15, "number_dry_days": 25, "thrd_rain_day": 0.85, "end_search": "08-15", "nbjour":40},
    2: {"zone_name": "Sahel400_200mm", "start_search": "05-01", "cumulative": 15, "number_dry_days": 20, "thrd_rain_day": 0.85, "end_search": "07-31", "nbjour":40},
    3: {"zone_name": "Sahel600_400mm", "start_search": "03-15", "cumulative": 20, "number_dry_days": 20, "thrd_rain_day": 0.85, "end_search": "07-31", "nbjour":45},
    4: {"zone_name": "Soudan",         "start_search": "03-15", "cumulative": 20, "number_dry_days": 10, "thrd_rain_day": 0.85, "end_search": "07-31", "nbjour":50},
    5: {"zone_name": "Golfe_Of_Guinea","start_search": "02-01", "cumulative": 20, "number_dry_days": 10, "thrd_rain_day": 0.85, "end_search": "06-15", "nbjour":50},
}

cessation_criteria = {
    0: {"zone_name": "Sahel100_0mm", "date_dry_soil":"01-01", "start_search": "09-01", "ETP": 5.0, "Cap_ret_maxi": 70, "end_search": "09-30"},
    1: {"zone_name": "Sahel200_100mm", "date_dry_soil":"01-01", "start_search": "09-01", "ETP": 5.0, "Cap_ret_maxi": 70, "end_search": "10-05"},
    2: {"zone_name": "Sahel400_200mm", "date_dry_soil":"01-01", "start_search": "09-01", "ETP": 5.0, "Cap_ret_maxi": 70, "end_search": "11-10"},
    3: {"zone_name": "Sahel600_400mm", "date_dry_soil":"01-01", "start_search": "09-15", "ETP": 5.0, "Cap_ret_maxi": 70, "end_search": "11-15"},
    4: {"zone_name": "Soudan", "date_dry_soil":"01-01", "start_search": "10-01", "ETP": 4.5, "Cap_ret_maxi": 70, "end_search": "11-30"},
    5: {"zone_name": "Golfe_Of_Guinea", "date_dry_soil":"01-01", "start_search": "10-15", "ETP": 4.0, "Cap_ret_maxi": 70, "end_search": "12-01"},
}

# Default class-level criteria dictionary
cessation_dryspell_criteria = {
    0: {"zone_name": "Sahel100_0mm", "start_search1": "06-01", "cumulative": 10, "number_dry_days": 25,
        "thrd_rain_day": 0.85,
        "end_search1": "08-15",
        "nbjour": 35,
        "date_dry_soil": "01-01",
        "start_search2": "09-01",
        "ETP": 5.0,
        "Cap_ret_maxi": 70,
        "end_search2": "09-30"
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

def process_model_for_other_params(agmParamModel, dir_to_save, hdcst_file_path, fcst_file_path, obs_hdcst, 
obs_fcst_year, month_of_initialization, year_start, year_end, year_forecast, nb_cores=2, agrometparam="Onset"):
    
    mask = xr.where(~np.isnan(obs_fcst_year.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()

    # create a dummy dataarray
    t_coord = pd.date_range(start=f"{year_forecast}-01-01", end=f"{year_forecast}-12-31", freq="D")
    

    y_coords = obs_fcst_year.Y
    x_coords = obs_fcst_year.X
    
    # Create a zero-filled DataArray with shape (1, Y, X)
    dummy = xr.DataArray(
        data=np.zeros((len(t_coord), len(y_coords), len(x_coords))),
        coords={"T": t_coord, "Y": y_coords, "X": x_coords},
        dims=["T", "Y", "X"]
    )*mask

    abb_mont_ini = calendar.month_abbr[int(month_of_initialization)]
    dir_to_save = Path(f"{dir_to_save}/model_data")
    os.makedirs(dir_to_save, exist_ok=True)

    # process the hindcast datasets
    saved_hindcast_paths = {}

    for i in hdcst_file_path.keys():
        save_path = f"{dir_to_save}/hindcast_{i}_{agrometparam}_{abb_mont_ini}Ic.nc"
        
        if not os.path.exists(save_path):
            hdcst = xr.open_dataset(hdcst_file_path[i])
            if 'number' in hdcst.dims:
                hdcst = hdcst.mean(dim="number")
            hdcst = hdcst.to_array().drop_vars("variable").squeeze()
            obs_hdcst_sel = obs_hdcst.sel(T=slice(str(year_start), str(year_end)))
            obs_hdcst_interp = obs_hdcst_sel.interp(Y=hdcst.Y, X=hdcst.X, 
                                                    method="linear", 
                                                    kwargs={"fill_value": "extrapolate"})
            ds1_aligned, ds2_aligned = xr.align(hdcst, obs_hdcst_interp, join='outer')
            filled_ds = ds1_aligned.fillna(ds2_aligned)
            ds_filled = filled_ds.copy()
            agpm_model = agmParamModel.compute(daily_data=ds_filled.sortby("T"), nb_cores=nb_cores)
            ds_processed = agpm_model.to_dataset(name=agrometparam)
            ds_processed.to_netcdf(save_path)
        else:
            print(f"[SKIP] {save_path} already exists.")
        saved_hindcast_paths[i] = save_path
        
    # process the forecasts datasets
    saved_forecast_paths = {}

    for i in fcst_file_path.keys():
        save_path = f"{dir_to_save}/forecast_{i}_{agrometparam}_{abb_mont_ini}Ic.nc"
        
        if not os.path.exists(save_path):
            fcst = xr.open_dataset(fcst_file_path[i])
            if 'number' in fcst.dims:
                fcst = fcst.mean(dim="number")
            fcst = fcst.to_array().drop_vars("variable").squeeze()
            obs_fcst_sel = obs_fcst_year.sortby("T").sel(T=str(year_forecast))
            obs_fcst_interp = obs_fcst_sel.interp(Y=fcst.Y, X=fcst.X, 
                                                  method="linear", 
                                                  kwargs={"fill_value": "extrapolate"})
            ds1_aligned, ds2_aligned = xr.align(fcst, obs_fcst_interp, join='outer')
            filled_fcst = ds1_aligned.fillna(ds2_aligned)
            ds_filled = filled_fcst.copy()
            ds_filled = ds_filled.sortby("T")

            dummy = dummy.interp(Y=fcst.Y, X=fcst.X, 
                                                    method="linear", 
                                                    kwargs={"fill_value": "extrapolate"})
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


# def plot_date(A):
#     plot = A.plot(cbar_kwargs={'label': 'Date'})
#     cbar = plot.colorbar
#     ticks = cbar.get_ticks()
#     tick_labels = [(datetime.datetime(2024, 1, 1) + timedelta(days=int(tick))).strftime('%d-%b') for tick in ticks]
#     cbar.set_ticks(ticks)
#     cbar.set_ticklabels(tick_labels)
#     # plt.title("Onset Date in Calendar Format")
#     plt.tight_layout()
#     plt.show()


def plot_date(A):
    """
    Plots 'A' on a map, interpreting the data values as
    offsets from 2024-01-01. The colorbar ticks are then
    converted to calendar dates.
    """

    # 1. Create a figure and axis with a map projection
    fig, ax = plt.subplots(
        figsize=(8, 6),
        subplot_kw=dict(projection=ccrs.PlateCarree())
    )

    # 2. Plot the DataArray with a horizontal colorbar
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

    # 3. Extract the colorbar and update tick labels
    cbar = plt_obj.colorbar
    ticks = cbar.get_ticks()
    tick_labels = [
        (datetime.datetime(2024, 1, 1) + timedelta(days=int(tick))).strftime('%d-%b')
        for tick in ticks
    ]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)

    # 4. Add map features
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.OCEAN)

    plt.tight_layout()
    plt.show()

def verify_station_network(df_filtered, extent, map_name="Rain-gauge network"):
    """
    Verify the station network by plotting the locations of the stations
    on a map.
    """

    lat_row = df_filtered.loc[df_filtered["STATION"] == "LAT"].squeeze()
    lon_row = df_filtered.loc[df_filtered["STATION"] == "LON"].squeeze()

    station_names = df_filtered.columns[1:]                      # skip the STATION header
    lats = lat_row[1:].astype(float).values
    lons = lon_row[1:].astype(float).values
    # Create a Basemap instance
    proj = ccrs.PlateCarree() 
    fig = plt.figure(figsize=(10, 8))
    ax  = plt.axes(projection=proj)
 
    # West-Africa extent
    ax.set_extent([extent[1], extent[3], extent[2], extent[0]], crs=proj)

    # Basemap layers
    ax.add_feature(cfeature.LAND,      facecolor="cornsilk")
    ax.add_feature(cfeature.OCEAN,     facecolor="lightblue")
    ax.add_feature(cfeature.BORDERS,   linewidth=0.6)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.LAKES,     alpha=0.4)

    ax.scatter(lons, lats,
            s=30, marker="o", facecolor="red", edgecolor="black",
            transform=proj, zorder=5)

    # label each point
    for lon, lat, name in zip(lons, lats, station_names):
        ax.text(lon + 0.3, lat + 0.3, name,
                fontsize=6, transform=proj)

    ax.set_title(map_name, fontsize=14)
    plt.tight_layout()
    plt.show()

# import matplotlib.pyplot as plt
# import matplotlib.colors as colors

# # On récupère les données qu'on veut tracer
# data = (rainfall.isel(T=0) / rainfall.mean(dim='T', skipna=True)) * 100

# # On fixe nos seuils : 3 "catégories" => <80, [80-120], >120
# bounds = [data.min(), 80, 120, data.max()]

# # On définit la liste de couleurs associées à chaque intervalle 
# # (nombre de couleurs = nombre d’intervalles - 1)
# cmap_list = ['red', 'green', 'blue']

# # On crée un colormap "discret" et la normalisation qui va avec
# cmap = colors.ListedColormap(cmap_list)
# norm = colors.BoundaryNorm(bounds, cmap.N)

# # Enfin on trace
# data.plot(cmap=cmap, norm=norm)
# plt.show()

