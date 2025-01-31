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
    runavg_part = f"T/{run_avg}/runningAverage/" if run_avg else ""

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
    # Suppose ds.T is your time coordinate (datetime64 or cftime).
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
        
    
def prepare_predictand(dir_to_save_Obs, variables_obs, season, year_start, year_end, ds=True):
    """Prepare the predictand dataset."""
    _, variable = parse_variable(variables_obs[0])
    season_str = "".join([calendar.month_abbr[int(month)] for month in season])
    filepath = f'{dir_to_save_Obs}/Obs_{variable}_{year_start}_{year_end}_{season_str}.nc'
    rainfall = xr.open_dataset(filepath)

    # Create mask
    mean_rainfall = rainfall.mean(dim="T").to_array().squeeze()
    mask = xr.where(mean_rainfall <= 20, np.nan, 1)
    mask = mask.where(abs(mask.Y) <= 19.5, np.nan)
    rainfall = xr.where(mask == 1, rainfall, np.nan)
    rainfall['T'] = pd.to_datetime(rainfall['T'].values)
    if ds :
        return rainfall.drop_vars("variable").squeeze().transpose( 'T', 'Y', 'X')
    else:
        return rainfall.to_array().drop_vars("variable").squeeze().rename("prcp").transpose( 'T', 'Y', 'X')


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



def get_best_models(center_variable, scores, metric='MAE', threshold=None, top_n=6):

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
            threshold = 0.0  # or any other default you prefer
    
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
    
    for key in top_n_models:
        # Key looks like "dwd21_JanIc_MarAprMay"; we take only "dwd21"
        key_prefix = key.split('_')[0].lower()
        
        # Find all matching variables for this key
        matches = [
            var for var in center_variable 
            if normalize_var(var) == key_prefix
        ]
        
        # Extend the list by all matches (or pick just the first one, depending on your needs)
        selected_vars_in_order.extend(matches)    
    return selected_vars_in_order # selected_vars


def plot_prob_forecasts(dir_to_save, forecast_prob, model_name):    

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
    BN_cmap = mcolors.LinearSegmentedColormap.from_list('BN', ['#FFF5F0', '#FB6A4A', '#67000D'])
    NN_cmap = mcolors.LinearSegmentedColormap.from_list('NN', ['#F7FCF5', '#74C476', '#00441B'])
    AN_cmap = mcolors.LinearSegmentedColormap.from_list('AN', ['#F7FBFF', '#6BAED6', '#08306B'])
    
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

def build_gcm_for_elm_mlp(center_variable, month_of_initialization, lead_time, dir_model, rainfall, elm=True):
    
    abb_mont_ini = calendar.month_abbr[int(month_of_initialization)]
    season_months = [((int(month_of_initialization) + int(l) - 1) % 12) + 1 for l in lead_time]
    season = "".join([calendar.month_abbr[month] for month in season_months])
    variables = center_variable[0].split(".")[1]
    
    all_model_hdcst = {}
    all_model_fcst = {} 

    for i in center_variable:
        center = i.split(".")[0].lower().replace("_", "")
        model_file = f"{dir_model}/hindcast_{center}_{variables}_{abb_mont_ini}Ic_{season}_{lead_time[0]}.nc"
        model_data_ = xr.open_dataset(model_file)
        all_model_hdcst[center] = model_data_.interp(
                        Y=rainfall.Y,
                        X=rainfall.X,
                        method="nearest",
                        kwargs={"fill_value": "extrapolate"}
                    )
        model_file = f"{dir_model}/forecast_{center}_{variables}_{abb_mont_ini}Ic_{season}_{lead_time[0]}.nc"
        model_data_ = xr.open_dataset(model_file)
        all_model_fcst[center] = model_data_.interp(
                        Y=rainfall.Y,
                        X=rainfall.X,
                        method="nearest",
                        kwargs={"fill_value": "extrapolate"}
                    )  
        
    # Extract the datasets and keys
    hindcast_det_list = list(all_model_hdcst.values()) 
    forecast_det_list = list(all_model_fcst.values())
    predictor_names = list(all_model_hdcst.keys())    
    obs = rainfall.expand_dims({'M':[0]},axis=1)
    mask = xr.where(~np.isnan(rainfall.isel(T=0)), 1, np.nan).drop_vars('T').squeeze()
    mask.name = None
    
    if elm:
        # Concatenate along a new dimension ('M') and assign coordinates
        all_model_hdcst = (
            xr.concat(hindcast_det_list, dim='M')
            .assign_coords({'M': predictor_names})  
            .rename({'T': 'S'})                    
            .transpose('S', 'M', 'Y', 'X')         
        )*mask
        
        all_model_fcst = (
            xr.concat(forecast_det_list, dim='M')
            .assign_coords({'M': predictor_names})  
            .rename({'T': 'S'})                    
            .transpose('S', 'M', 'Y', 'X')         
        )
    else:
        # Concatenate along a new dimension ('M') and assign coordinates
        all_model_hdcst = (
            xr.concat(hindcast_det_list, dim='M')
            .assign_coords({'M': predictor_names})             
            .transpose('T', 'M', 'Y', 'X')         
        )*mask
        
        all_model_fcst = (
            xr.concat(forecast_det_list, dim='M')
            .assign_coords({'M': predictor_names})                     
            .transpose('T', 'M', 'Y', 'X')         
        )*mask
    return all_model_hdcst.to_array().drop('variable').squeeze('variable'), all_model_fcst.to_array().drop('variable').squeeze('variable'), obs


def build_MLP_ELM_mme_datasets(rainfall, hdcsted=None, fcsted=None, gcm=True, ELM=False, dir_to_save_model=None, best_models=None, year_start=None, year_end=None, model=True, month_of_initialization=None, lead_time=None, year_forecast=None):
    all_model_hdcst = {}
    all_model_fcst = {}
    if gcm:
        for i in best_models:
            hdcst = load_gridded_predictor(dir_to_save_model, i, year_start, year_end, model=True, month_of_initialization=month_of_initialization, lead_time=lead_time, year_forecast=None)
            all_model_hdcst[i] = hdcst.interp(
                            Y=rainfall.Y,
                            X=rainfall.X,
                            method="linear",
                            kwargs={"fill_value": "extrapolate"}
                        )
            fcst = load_gridded_predictor(dir_to_save_model, i, year_start, year_end, model=True, month_of_initialization=month_of_initialization, lead_time=lead_time, year_forecast=year_forecast)
            all_model_fcst[i] = fcst.interp(
                            Y=rainfall.Y,
                            X=rainfall.X,
                            method="linear",
                            kwargs={"fill_value": "extrapolate"}
                        )
    else:
        for i in dict_models.keys():
            all_model_hdcst[i] = hdcsted[i].interp(
                            Y=rainfall.Y,
                            X=rainfall.X,
                            method="nearest",
                            kwargs={"fill_value": "extrapolate"}
                        )
            all_model_fcst[i] = fcsted[i].interp(
                            Y=rainfall.Y,
                            X=rainfall.X,
                            method="nearest",
                            kwargs={"fill_value": "extrapolate"}
                        )    
    
    # Extract the datasets and keys
    hindcast_det_list = list(all_model_hdcst.values()) 
    forecast_det_list = list(all_model_fcst.values())
    predictor_names = list(all_model_hdcst.keys())    

    mask = xr.where(~np.isnan(rainfall.isel(T=0)), 1, np.nan).drop_vars('T').squeeze()
    mask.name = None
    
    if ELM:
        # Concatenate along a new dimension ('M') and assign coordinates
        all_model_hdcst = (
            xr.concat(hindcast_det_list, dim='M')
            .assign_coords({'M': predictor_names})  
            .rename({'T': 'S'})                    
            .transpose('S', 'M', 'Y', 'X')         
        )*mask
        
        all_model_fcst = (
            xr.concat(forecast_det_list, dim='M')
            .assign_coords({'M': predictor_names})  
            .rename({'T': 'S'})                    
            .transpose('S', 'M', 'Y', 'X')         
        )
        obs = rainfall.expand_dims({'M':[0]},axis=1)
    else:
        # Concatenate along a new dimension ('M') and assign coordinates
        all_model_hdcst = (
            xr.concat(hindcast_det_list, dim='M')
            .assign_coords({'M': predictor_names})             
            .transpose('T', 'M', 'Y', 'X')         
        )*mask
        
        all_model_fcst = (
            xr.concat(forecast_det_list, dim='M')
            .assign_coords({'M': predictor_names})                     
            .transpose('T', 'M', 'Y', 'X')         
        )*mask
        obs = rainfall.expand_dims({'M':[0]},axis=1)
    return all_model_hdcst, all_model_fcst, obs







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