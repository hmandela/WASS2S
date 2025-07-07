# Core Python Libraries
import os
from pathlib import Path
import calendar
from datetime import datetime
from dateutil.relativedelta import relativedelta
from collections import defaultdict

# Numerical and Data Manipulation Libraries
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da

# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm

# Climate Data API
import cdsapi

# Machine Learning and Statistical Analysis Libraries
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.stats import pearsonr, gamma, lognorm, stats
import scipy.signal as sig

# EOF Analysis and Verification Libraries
import xeofs as xe
import xskillscore as xs
from wass2s.utils import *

# Core Python Libraries
import os
from pathlib import Path
import calendar
from datetime import datetime
from dateutil.relativedelta import relativedelta
from collections import defaultdict

# Numerical and Data Manipulation Libraries
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da

# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm

# Climate Data API
import cdsapi

# Machine Learning and Statistical Analysis Libraries
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.stats import pearsonr, gamma, lognorm, stats
import scipy.signal as sig

# EOF Analysis and Verification Libraries
import xeofs as xe
import xskillscore as xs
from wass2s.utils import *


class WAS_Analog_:
    """Analog-based forecasting toolkit for seasonal climate applications.

    This class orchestrates the end-to-end workflow required to build analog ensembles
    for Sea-Surface Temperature (SST) predictors and to translate them into deterministic
    and probabilistic rainfall forecasts over West Africa or any user-defined domain.
    Supports three analog-selection strategies: Self-Organizing Maps (SOM),
    correlation ranking, and Principal-Component (EOF) similarity.

    Parameters
    ----------
    dir_to_save : str
        Directory path to save downloaded and processed data files.
    year_start : int
        Starting year for historical data.
    year_forecast : int
        Target forecast year.
    reanalysis_name : str
        Name of the reanalysis dataset (e.g., "ERA5.SST" or "NOAA.SST").
    model_name : str
        Name of the forecast model (e.g., "ECMWF_51.SST").
    predictor_vars : list of dict, optional
        List of dictionaries specifying predictor variables, each containing
        'reanalysis_name', 'model_name', 'variable', and 'area'.
        Default is a list with NOAA SST and ERA5 SP, VGRD_850 variables.
    method_analog : str, optional
        Analog method to use ("som", "cor_based", "pca_based"). Default is "som".
        List of best precipitation models to consider.
    month_of_initialization : int, optional
        Month of initialization for forecasts (1-12). If None, uses previous month.
    lead_time : list, optional
        List of lead times in months. If None, defaults to [1, 2, 3, 4, 5].
    ensemble_mean : str, optional
        Method for ensemble mean ("mean" or "median"). Default is "mean".
    clim_year_start : int, optional
        Start year for climatology period.
    clim_year_end : int, optional
        End year for climatology period.
        Bounding box as (lon_min, lon_max, lat_min, lat_max) for regional analysis.
    index_compute : list, optional
        List of climate indices to compute (e.g., ['NINO34', 'DMI']).
    some_grid_size : tuple, optional
        Grid size for SOM (rows, columns). Default is (None, None) for automatic sizing.
    some_learning_rate : float, optional
        Learning rate for SOM training. Default is 0.5.
    radius : float, optional
            Neighborhood radius for analog search. Default is 1.0.
    some_neighborhood_function : str, optional
        Neighborhood function for SOM ("gaussian", "mexican_hat"). Default is "gaussian".
    some_sigma : float, optional
        Initial neighborhood radius for SOM. Default is 1.0.
    some_num_iteration : int, optional
        Number of iterations for SOM training. Default is 2000.
    dist_method : str, optional
        Method for probability calculation ("gamma", "t", "normal", "lognormal", "nonparam").
        Default is "gamma".
    """

    def __init__(self, dir_to_save, year_start, year_forecast,
                 predictor_vars=[{'reanalysis_name': 'NOAA', 'model_name': 'NCEP_2', 'variable': 'SST', 'area': [60, -180, -60, 180]},
                                {'reanalysis_name': 'ERA5', 'model_name': 'NCEP_2', 'variable': 'SP', 'area': [60, -180, -60, 180]},
                                {'reanalysis_name': 'ERA5', 'model_name': 'NCEP_2', 'variable': 'VGRD_850', 'area': [60, -180, -60, 180]}],
                 method_analog="som", month_of_initialization=None,
                 lead_time=None, ensemble_mean="mean", rolling=3, standardize=True, multivariateEOF=False,
                 eof_explained_var=0.95, clim_year_start=None, clim_year_end=None,
                 index_compute=None, some_grid_size=(None, None), some_learning_rate=0.5, radius=1.0,
                 some_neighborhood_function='gaussian', some_sigma=1.0, some_num_iteration=2000, dist_method="gamma"):
        
        self.dir_to_save = dir_to_save
        self.year_start = year_start
        self.year_forecast = year_forecast
        self.predictor_vars = predictor_vars
        self.method_analog = method_analog
        self.month_of_initialization = month_of_initialization
        self.lead_time = lead_time
        self.ensemble_mean = ensemble_mean
        self.eof_explained_var = eof_explained_var
        self.multivariateEOF = multivariateEOF
        self.rolling = rolling
        self.standardize = standardize
        self.clim_year_start = clim_year_start
        self.clim_year_end = clim_year_end
        self.index_compute = index_compute
        self.some_grid_size = some_grid_size
        self.some_learning_rate = some_learning_rate
        self.radius = radius
        self.some_neighborhood_function = some_neighborhood_function
        self.some_sigma = some_sigma
        self.some_num_iteration = some_num_iteration
        self.dist_method = dist_method

    def calc_index(self, indices, sst):
        """Calculate climate indices from SST data.

        Computes specified climate indices (e.g., NINO34, DMI) by averaging over predefined regions
        or computing differences for derived indices.

        Parameters
        ----------
        indices : list
            List of climate indices to compute (e.g., ['NINO34', 'DMI']).
        sst : xarray.DataArray
            SST data with dimensions (T, Y, X).

        Returns
        -------
        indices_dataset : xarray.Dataset
            Dataset containing computed climate indices as variables.
        """
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
            "M1": ("M1", -50, 5, -50, -25),
            "M2": ("M2", -75, -10, 25, 50),
            "M3": ("M3", -175, -125, 25, 50),
            "M4": ("M4", -175, -125, -50, -25),
        }
        
        predictor = {}
        for idx in sst_indices_name:
            if idx in ["TASI", "DMI"]:
                continue
            _, lon_min, lon_max, lat_min, lat_max = sst_indices_name[idx]
            sst_region = sst.sel(X=slice(lon_min, lon_max), Y=slice(lat_min, lat_max)).mean(dim=["X", "Y"], skipna=True)
            predictor[idx] = sst_region

        predictor["TASI"] = predictor["NAT"] - predictor["SAT"]
        predictor["DMI"] = predictor["WTIO"] - predictor["SETIO"]
        
        selected_indices = {i: predictor[i] for i in indices if i in predictor}
        data_vars = {key: ds.rename(key) for key, ds in selected_indices.items()}
        return xr.Dataset(data_vars)

    def _postprocess_ersst(self, ds, var_name):
        """Post-process ERSST dataset to ensure consistent variable names and coordinates.

        Parameters
        ----------
        ds : xarray.Dataset
            Input ERSST dataset.
        var_name : str
            Name of the variable to keep (e.g., 'sst').

        Returns
        -------
        ds : xarray.Dataset
            Processed dataset with specified variable and coordinates.
        """
        ds = ds.drop_vars('zlev', errors='ignore').squeeze()
        keep_vars = [var_name, 'T', 'X', 'Y']
        drop_vars = [v for v in ds.variables if v not in keep_vars]
        return ds.drop_vars(drop_vars, errors="ignore")

    def download_reanalysis(self, force_download=False):
        """Download reanalysis data for specified variables and years.

        Downloads data from NOAA ERSST or ERA5 datasets, processes it, and saves to NetCDF files.

        Parameters
        ----------
        force_download : bool, optional
            If True, forces re-download even if file exists. Default is False.

        Returns
        -------
        store_file_path : dict
            Dictionary mapping variable names to processed xarray.Dataset objects.
        """
        year_end = self.year_forecast
        variables = [item['variable'] for item in self.predictor_vars]
        centers = [item['reanalysis_name'] for item in self.predictor_vars]
        areas = [item['area'] for item in self.predictor_vars]
        
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
            "NOLR": "top_net_thermal_radiation",
        }
        variables_2 = {
            "HUSS_1000": "specific_humidity",
            "HUSS_925": "specific_humidity",
            "HUSS_850": "specific_humidity",
            "UGRD_1000": "u_component_of_wind",
            "UGRD_925": "u_component_of_wind",
            "UGRD_850": "u_component_of_wind",
            "VGRD_1000": "v_component_of_wind",
            "VGRD_925": "v_component_of_wind",
            "VGRD_850": "v_component_of_wind",
        }

        dir_to_save = Path(self.dir_to_save)
        os.makedirs(dir_to_save, exist_ok=True)
    
        months = [f"{m:02d}" for m in range(1, 13)]
        season = "".join([calendar.month_abbr[int(m)] for m in months])
        store_file_path = {}

        for center, var, area in zip(centers, variables, areas):
            combined_output_path = dir_to_save / f"{center}_{var}_{self.month_of_initialization}_{self.year_start}_{year_end}_{season}.nc"
        
            if not force_download and combined_output_path.exists():
                print(f"{combined_output_path} exists. Skipping download.")
                store_file_path[var] = xr.open_dataset(combined_output_path)
                continue

            if f"{center}.{var}" == "NOAA.SST":
                try:
                    url = build_iridl_url_ersst(
                        year_start=self.year_start,
                        year_end=year_end,
                        bbox=area,
                        run_avg=None,
                        month_start="Jan",
                        month_end="Dec"
                    )
                    print(f"Using IRIDL URL: {url}")
        
                    ds = xr.open_dataset(url, decode_times=False)
                    ds = decode_cf(ds, "T").rename({"T": "time"}).convert_calendar("proleptic_gregorian", align_on="year").rename({"time": "T"})
                    ds = ds.assign_coords(T=ds.T - pd.Timedelta(days=15))
                    ds = ds.rename({'sst': 'SST'})
                    ds = self._postprocess_ersst(ds, var)
                    ds['T'] = ds['T'].astype('datetime64[ns]')
                    ds = ds.rename({'SST': 'sst'})
                    store_file_path[var] = ds
                    ds.to_netcdf(combined_output_path)
                    print(f"Saved NOAA ERSST data to {combined_output_path}")
                    return ds
                except Exception as e:
                    print(f"Failed to download NOAA.SST: {e}")
                    continue

            combined_datasets = []
            
        
            for year in range(self.year_start, year_end + 1):
                yearly_file_path = dir_to_save / f"{center}_{var}_{year}.nc"
            
                if not force_download and yearly_file_path.exists():
                    print(f"{yearly_file_path} exists. Loading existing file.")
                    try:
                        ds = xr.open_dataset(yearly_file_path).load()
                        if 'latitude' in ds.coords and 'longitude' in ds.coords:
                            ds = ds.rename({"latitude": "Y", "longitude": "X", "valid_time": "T"})
                        combined_datasets.append(ds)
                        continue
                    except Exception as e:
                        print(f"Failed to load {yearly_file_path}: {e}")
            
                try:
                    client = cdsapi.Client()
                    if var in variables_2:
                        press_level = var.split("_")[1]
                        dataset = "reanalysis-era5-pressure-levels-monthly-means"
                        request = {
                            "product_type": "monthly_averaged_reanalysis",
                            "variable": variables_2[var],
                            "pressure_level": press_level,
                            "year": str(year),
                            "month": months,
                            "time": "00:00",
                            "area": area,
                            "format": "netcdf",
                        }
                    else:
                        dataset = "reanalysis-era5-single-levels-monthly-means"
                        request = {
                            "product_type": "monthly_averaged_reanalysis",
                            "variable": variables_1.get(var, var),
                            "year": str(year),
                            "month": months,
                            "time": "00:00",
                            "area": area,
                            "format": "netcdf",
                        }
            
                    print(f"Downloading {var} for {year} from {center}...")
                    client.retrieve(dataset, request).download(str(yearly_file_path))
                
                    with xr.open_dataset(yearly_file_path) as ds:
                        if 'latitude' in ds.coords and 'longitude' in ds.coords:
                            ds = ds.rename({"latitude": "Y", "longitude": "X", "valid_time": "T"})
                        ds = ds.load()
                        combined_datasets.append(ds)
                except Exception as e:
                    print(f"Failed to download/process {var} for {year}: {e}")
                    continue
        
            if combined_datasets:
                print(f"Concatenating {var} datasets...")
                combined_ds = xr.concat(combined_datasets, dim="T")
                combined_ds = combined_ds.drop_vars(["number", "expver"], errors="ignore").squeeze()
                
                if var in ["TMIN", "TEMP", "TMAX", "SST"]:
                    combined_ds = combined_ds - 273.15
                elif var == "PRCP":
                    combined_ds = combined_ds * 1000
                elif var in ["DSWR", "DLWR", "NOLR"]:
                    combined_ds = combined_ds / 86400
                elif var == "SLP":
                    combined_ds = combined_ds / 100
                
                combined_ds = combined_ds.isel(Y=slice(None, None, -1))
                store_file_path[var] = combined_ds
                combined_ds.to_netcdf(combined_output_path)
                print(f"Saved combined dataset to {combined_output_path}")
                
                for year in range(self.year_start, year_end + 1):
                    single_file_path = dir_to_save / f"{center}_{var}_{year}.nc"
                    if single_file_path.exists():
                        os.remove(single_file_path)
                        print(f"Deleted yearly file: {single_file_path}")
            else:
                print(f"No data combined for {var}. Check download success.")
        
        return store_file_path



    def download_models(self, force_download=False):
        """Download and process seasonal forecast and hindcast data for specified models and variables.

        Parameters
        ----------
        force_download : bool, optional
            If True, forces re-download even if file exists. Default is False.

        Returns
        -------
        store_hdcst_file_path : dict
            Dictionary mapping variable names to processed hindcast xarray.Dataset objects.
        store_file_path : dict
            Dictionary mapping variable names to processed forecast xarray.Dataset objects.
        """
        # Model and variable mappings
        centre_map = {
            "BOM_2": "bom", "ECMWF_51": "ecmwf", "UKMO_604": "ukmo", "UKMO_603": "ukmo",
            "METEOFRANCE_8": "meteo_france", "METEOFRANCE_9": "meteo_france",
            "DWD_21": "dwd", "DWD_22": "dwd", "CMCC_35": "cmcc",
            "NCEP_2": "ncep", "JMA_3": "jma", "ECCC_4": "eccc", "ECCC_5": "eccc"
        }
        system_map = {
            "BOM_2": "2", "ECMWF_51": "51", "UKMO_604": "604", "UKMO_603": "603",
            "METEOFRANCE_8": "8", "METEOFRANCE_9": "9", "DWD_21": "21", "DWD_22": "22",
            "CMCC_35": "35", "NCEP_2": "2", "JMA_3": "3", "ECCC_4": "4", "ECCC_5": "5"
        }
        variables_map = {
            "PRCP": "total_precipitation", "TEMP": "2m_temperature",
            "TMAX": "maximum_2m_temperature_in_the_last_24_hours",
            "TMIN": "minimum_2m_temperature_in_the_last_24_hours",
            "UGRD10": "10m_u_component_of_wind", "VGRD10": "10m_v_component_of_wind",
            "SST": "sea_surface_temperature", "SLP": "mean_sea_level_pressure",
            "DSWR": "surface_solar_radiation_downwards",
            "DLWR": "surface_thermal_radiation_downwards",
            "NOLR": "top_thermal_radiation",
            "HUSS_1000": "specific_humidity", "HUSS_925": "specific_humidity",
            "HUSS_850": "specific_humidity", "UGRD_1000": "u_component_of_wind",
            "UGRD_925": "u_component_of_wind", "UGRD_850": "u_component_of_wind",
            "VGRD_1000": "v_component_of_wind", "VGRD_925": "v_component_of_wind",
            "VGRD_850": "v_component_of_wind",
        }

        # Extract model and variable information
        centers = [item['model_name'] for item in self.predictor_vars]
        variables = [item['variable'] for item in self.predictor_vars]
        areas = [item['area'] for item in self.predictor_vars]
        selected_centres = [centre_map[k] for k in centers]
        selected_systems = [system_map[k] for k in centers]

        # Validate inputs
        month_of_initialization = (datetime.now() - relativedelta(months=1)).month if self.month_of_initialization is None else self.month_of_initialization
        lead_time = [1, 2, 3, 4, 5] if self.lead_time is None else self.lead_time
        if not isinstance(lead_time, list) or any(l < 1 or l > 12 for l in lead_time):
            raise ValueError("lead_time must be a list of integers between 1 and 12.")
        year_forecast = datetime.now().year if self.year_forecast is None else self.year_forecast
        hindcast_years = [str(year) for year in range(self.clim_year_start, self.clim_year_end + 1)]  # Configurable if needed

        # Set up directory and season string
        dir_to_save = Path(self.dir_to_save)
        dir_to_save.mkdir(parents=True, exist_ok=True)
        abb_mont_ini = calendar.month_abbr[int(month_of_initialization)]
        season_months = [((month_of_initialization + l - 1) % 12) + 1 for l in lead_time]
        season_str = "".join(calendar.month_abbr[m] for m in season_months)

        # Initialize output dictionaries
        store_file_path = {}  # Forecasts
        store_hdcst_file_path = {}  # Hindcasts



        for cent, syst, var, area in zip(selected_centres, selected_systems, variables, areas):
            # Define file paths
            forecast_file = dir_to_save / f"forecast_{cent}{syst}_{var}_{abb_mont_ini}Ic_{season_str}_{lead_time[0]}.nc"
            hindcast_file = dir_to_save / f"hindcast_{cent}{syst}_{var}_{abb_mont_ini}Ic_{season_str}_{lead_time[0]}.nc"
               
            # Skip existing files unless force_download is True
            if not force_download and forecast_file.exists():
                print(f"Forecast file {forecast_file} exists. Skipping download.")
                ds = xr.open_dataset(forecast_file)
                store_file_path[var] = ds  # Load into memory
                ds.close()
            else:
                try:
                    # Determine dataset type
                    dataset = "seasonal-monthly-pressure-levels" if var in variables_map and "HUSS" in var or "UGRD" in var or "VGRD" in var else "seasonal-monthly-single-levels"

                    # Forecast request
                    forecast_request = {
                        "originating_centre": cent,
                        "system": syst,
                        "variable": variables_map[var],
                        "product_type": ["monthly_mean"],
                        "year": [str(year_forecast)],
                        "month": [f"{month_of_initialization:02d}"],
                        "leadtime_month": lead_time,
                        "data_format": "netcdf",
                        "area": area,
                    }
                    if var in variables_map and any(v in var for v in ["HUSS", "UGRD", "VGRD"]):
                        forecast_request["pressure_level"] = var.split("_")[1]

                    # Initialize CDS client
                    client = cdsapi.Client()
                    client.retrieve(dataset, forecast_request).download(str(forecast_file))
                    print(f"Downloaded forecast: {forecast_file}")

                    # Process datasets
                    ds_forecast = xr.open_dataset(forecast_file)

                    # Unit conversions
                    if var in ["TMIN", "TEMP", "TMAX", "SST"]:
                        ds_forecast = ds_forecast - 273.15  # Kelvin to Celsius
                    elif var == "PRCP":
                        # Convert kg/m^2/s to mm/month (approximate, using 30 days)
                        ds_forecast = ds_forecast * (1000 * 30 * 24 * 3600)
                    elif var == "SLP":
                        ds_forecast = ds_forecast / 100  # Pa to hPa
                    elif var in ["DSWR", "DLWR", "NOLR"]:
                        ds_forecast = ds_forecast / 86400  # J/m^2 to W/m^2

                    # Compute ensemble mean if specified
                    if self.ensemble_mean and "number" in ds_forecast.dims:
                        ds_forecast = getattr(ds_forecast,self.ensemble_mean)(dim="number", skipna=True)

                    # Flip latitude (if needed)
                    ds_forecast = ds_forecast.isel(latitude=slice(None, None, -1))

                    # Rename coordinates
                    rename_dict = {
                        "latitude": "Y",
                        "longitude": "X",
                        "time": "T" if "time" in ds_forecast.coords else None,
                        "indexing_time": "T" if "indexing_time" in ds_forecast.coords else None,
                        "forecast_reference_time": "T" if "forecast_reference_time" in ds_forecast.coords else None,
                    }
                    rename_dict = {k: v for k, v in rename_dict.items() if v is not None}
                    ds_forecast = ds_forecast.rename(rename_dict)

                    # Drop pressure_level for non-surface variables
                    if var in variables_map and any(v in var for v in ["HUSS", "UGRD", "VGRD"]):
                        ds_forecast = ds_forecast.drop_vars("pressure_level", errors="ignore").squeeze()

                    # Sort by time
                    ds_forecast = ds_forecast.sortby("T")

                    # Save processed datasets
                    ds_forecast.to_netcdf(forecast_file)
                    print(f"Saved processed forecast to {forecast_file}")

                    # Store in dictionaries
                    store_file_path[var] = ds_forecast
                    ds_forecast.close()

                except Exception as e:
                    print(f"Failed to process {var} for {cent}{syst}: {e}")
                    continue

            if not force_download and hindcast_file.exists():
                print(f"Hindcast file {hindcast_file} exists. Skipping download.")
                ds = xr.open_dataset(hindcast_file)
                store_hdcst_file_path[var] = ds  # Load into memory
                ds.close()
            else:
                try:
                    # Determine dataset type
                    dataset = "seasonal-monthly-pressure-levels" if var in variables_map and "HUSS" in var or "UGRD" in var or "VGRD" in var else "seasonal-monthly-single-levels"

                    # Forecast request
                    hindcast_request = {
                        "originating_centre": cent,
                        "system": syst,
                        "variable": variables_map[var],
                        "product_type": ["monthly_mean"],
                        "year": hindcast_years,
                        "month": [f"{month_of_initialization:02d}"],
                        "leadtime_month": lead_time,
                        "data_format": "netcdf",
                        "area": area,
                    }
                    if var in variables_map and any(v in var for v in ["HUSS", "UGRD", "VGRD"]):
                        hindcast_request["pressure_level"] = var.split("_")[1]
        
                    # Initialize CDS client
                    client = cdsapi.Client()
                    client.retrieve(dataset, hindcast_request).download(str(hindcast_file))
                    print(f"Downloaded hindcast: {hindcast_file}")

                # Process datasets
                    ds_hindcast = xr.open_dataset(hindcast_file)

                    # Unit conversions
                    if var in ["TMIN", "TEMP", "TMAX", "SST"]:
                        ds_hindcast = ds_hindcast - 273.15
                    elif var == "PRCP":
                        # Convert kg/m^2/s to mm/month (approximate, using 30 days)
                        ds_hindcast = ds_hindcast * (1000 * 30 * 24 * 3600)
                    elif var == "SLP":
                        ds_hindcast = ds_hindcast / 100
                    elif var in ["DSWR", "DLWR", "NOLR"]:
                        ds_hindcast = ds_hindcast / 86400

                    # Compute ensemble mean if specified
                    if self.ensemble_mean and "number" in ds_hindcast.dims:
                        ds_hindcast = getattr(ds_hindcast, self.ensemble_mean)(dim="number", skipna=True)                        

                    # Flip latitude (if needed)
                    ds_hindcast = ds_hindcast.isel(latitude=slice(None, None, -1))

                    # Rename coordinates
                    rename_dict = {
                        "latitude": "Y",
                        "longitude": "X",
                        "time": "T" if "time" in ds_hindcast.coords else None,
                        "indexing_time": "T" if "indexing_time" in ds_hindcast.coords else None,
                        "forecast_reference_time": "T" if "forecast_reference_time" in ds_hindcast.coords else None,
                    }
                    rename_dict = {k: v for k, v in rename_dict.items() if v is not None}
                    ds_hindcast = ds_hindcast.rename(rename_dict)

                    # Drop pressure_level for non-surface variables
                    if var in variables_map and any(v in var for v in ["HUSS", "UGRD", "VGRD"]):
                        ds_hindcast = ds_hindcast.drop_vars("pressure_level", errors="ignore").squeeze()

                    # Sort by time
                    ds_hindcast = ds_hindcast.sortby("T")

                    # Save processed datasets
                    ds_hindcast.to_netcdf(hindcast_file)
                    print(f"Saved processed hindcast to {hindcast_file}")

                    # Store in dictionaries
                    store_hdcst_file_path[var] = ds_hindcast
                    ds_hindcast.close()

                except Exception as e:
                    print(f"Failed to process {var} for {cent}{syst}: {e}")
                    continue

        return store_hdcst_file_path, store_file_path

    def anomaly_timeseries(self, ds):
        """Compute anomalies by removing the climatological mean.

        Parameters
        ----------
        ds : xarray.Dataset
            Input dataset with time dimension 'T', and spatial dimensions 'Y' and 'X'.

        Returns
        -------
        ds_anomaly : xarray.Dataset
            Dataset with anomalies (climatological mean subtracted) with dimensions 'T', 'Y', 'X'.
        """
        # Validate climatology period
        if self.clim_year_start and self.clim_year_end:
            clim_period = ds.sel(T=slice(f"{self.clim_year_start}-01-01", f"{self.clim_year_end}-12-31"))
        else:
            clim_period = ds

        # Compute monthly climatological mean
        clim_mean = clim_period.groupby("T.month").mean("T", skipna=True)

        # Initialize a list to store anomaly slices for each month
        anomaly_slices = []

        # Loop over each month (1 to 12)
        for month in range(1, 13):
            # Select data for the current month
            month_mask = ds["T"].dt.month == month
            ds_month = ds.where(month_mask, drop=True)
            
            # Get the corresponding climatological mean for this month
            month_mean = clim_mean.sel(month=month)
            
            # Compute anomalies for the month's data
            ds_anomaly_month = ds_month - month_mean
            
            # Append to the list of slices
            anomaly_slices.append(ds_anomaly_month)

        # Concatenate all anomaly slices along the 'T' dimension
        ds_anomaly = xr.concat(anomaly_slices, dim="T")
        
        # Ensure the dataset is sorted by time
        ds_anomaly = ds_anomaly.sortby("T")

        # Drop 'month' coordinate or dimension if it exists
        if "month" in ds_anomaly.coords:
            ds_anomaly = ds_anomaly.drop_vars("month")
        if "month" in ds_anomaly.dims:
            ds_anomaly = ds_anomaly.drop_dims("month")

        return ds_anomaly

    def standardize_timeseries(self, ds):
        """Standardize the dataset by removing climatological mean and scaling by standard deviation.

        Parameters
        ----------
        ds : xarray.Dataset
            Input dataset with time dimension 'T', and spatial dimensions 'Y' and 'X'.

        Returns
        -------
        ds_standardized : xarray.Dataset
            Standardized dataset (mean=0, std=1) with dimensions 'T', 'Y', 'X'.
        """
        # Validate climatology period
        if self.clim_year_start and self.clim_year_end:
            clim_period = ds.sel(T=slice(f"{self.clim_year_start}-01-01", f"{self.clim_year_end}-12-31"))
        else:
            clim_period = ds

        # Compute monthly climatology
        clim_mean = clim_period.groupby("T.month").mean("T", skipna=True)
        clim_std = clim_period.groupby("T.month").std("T", skipna=True)
        clim_std = clim_std.where(clim_std != 0, 1e-10)  # Avoid division by zero

        # Initialize a list to store standardized slices for each month
        standardized_slices = []

        # Loop over each month (1 to 12)
        for month in range(1, 13):
            # Select data for the current month
            month_mask = ds["T"].dt.month == month
            ds_month = ds.where(month_mask, drop=True)
            
            # Get the corresponding climatology for this month
            month_mean = clim_mean.sel(month=month)
            month_std = clim_std.sel(month=month)
            
            # Standardize the month's data
            ds_standardized_month = (ds_month - month_mean) / month_std
            
            # Append to the list of slices
            standardized_slices.append(ds_standardized_month)

        # Concatenate all standardized slices along the 'T' dimension
        ds_standardized = xr.concat(standardized_slices, dim="T")
        
        # Ensure the dataset is sorted by time
        ds_standardized = ds_standardized.sortby("T")

        # Drop 'month' coordinate if it exists
        if "month" in ds_standardized.coords:
            ds_standardized = ds_standardized.drop_vars("month")
        if "month" in ds_standardized.dims:
            ds_standardized = ds_standardized.drop_dims("month")

        return ds_standardized

    def compute_eofs(self, data):
        """Compute EOFs and retain modes explaining at least the specified variance.

        Parameters
        ----------
        data : xarray.DataArray
            Input data with dimensions (T, Y, X).

        Returns
        -------
        pcs : xarray.DataArray
            Principal component scores for retained modes.
        """
        data = data.rename({"X": "lon", "Y": "lat"})
        data = data.fillna(data.mean(dim="T", skipna=True))
        print("Computing EOFs...")
        model = xe.single.EOF(n_modes=50)
        model.fit(data, dim='T')
        explained_var = model.explained_variance_ratio()
        cumulative_var = explained_var.cumsum()
        n_modes = np.where(cumulative_var >= self.eof_explained_var)[0][0] + 1
        print(f"Selected {n_modes} modes explaining {cumulative_var[n_modes-1].values*100:.2f}% variance")
        return model.scores().isel(mode=slice(0, n_modes))

    def download_and_process(self):
        """Download and process reanalysis and forecast data.

        Combines reanalysis and forecast data, applies standardization or anomaly calculation,
        and optionally applies a rolling mean.

        Returns
        -------
        data_var_concatenated : dict
            Dictionary of concatenated, processed datasets.
        data_var_shifted : dict
            Dictionary of time-shifted, processed datasets.
        """
        lead_time = [1, 2, 3, 4, 5] if self.lead_time is None else self.lead_time
        month_of_initialization = (datetime.now() - relativedelta(months=1)).month if self.month_of_initialization is None else self.month_of_initialization
        
        sst_hist = self.download_reanalysis(force_download=False)
        sst_hdcst, sst_for = self.download_models(force_download=False)
        variables = [item['variable'] for item in self.predictor_vars]
        print(f"Processing variables: {variables}")
        data_var_concatenated = {}
        data_var_shifted = {}

        for var in variables:
            if var not in sst_hist or var not in sst_for:
                print(f"Skipping {var}: data not available.")
                continue  

            sst_hist_ = sst_hist[var]
            sst_hdcst_ = sst_hdcst[var]
            sst_for_ = sst_for[var]

            # process hindcast data
            hindcast = []
            for i in sst_hdcst_['T']:
                sst_hdcst_i = sst_hdcst_.sel(T=i)
                base_times = pd.Timestamp(sst_hdcst_i['T'].values)
                new_times = [base_times + pd.DateOffset(months=int(m)) for m in sst_hdcst_i['forecastMonth'].values]
                sst_hdcst_i = sst_hdcst_i.assign_coords(forecastMonth=("forecastMonth", new_times))
                sst_hdcst_i = sst_hdcst_i.drop_vars('T', errors='ignore').squeeze().rename({'forecastMonth': 'T'})
                hindcast.append(sst_hdcst_i)
            sst_hdcst_ = xr.concat(hindcast, dim='T')          
            sst_hdcst_ = sst_hdcst_.interp(Y=sst_hist_.Y, X=sst_hist_.X, method="linear", kwargs={"fill_value": "extrapolate"})

            # process forecast data
            sst_for_ = sst_for_.interp(Y=sst_hist_.Y, X=sst_hist_.X, method="linear", kwargs={"fill_value": "extrapolate"})
            if (isinstance(sst_for_['T'].values, np.ndarray)):
                base_time = pd.Timestamp(sst_for_['T'].values[-1])
            else:
                base_time = pd.Timestamp(sst_for_['T'].values)
            new_times = [base_time + pd.DateOffset(months=int(m)) for m in sst_for_['forecastMonth'].values]
            sst_for_ = sst_for_.assign_coords(forecastMonth=("forecastMonth", new_times))
            sst_for_ = sst_for_.drop_vars('T', errors='ignore').squeeze().rename({'forecastMonth': 'T'})

            # Correct forecast systematic bias
            sst_for_bias_corrected = []
            for m in lead_time:
                sst_hist_mean = sst_hist_.sel(T=slice(str(self.clim_year_start), str(self.clim_year_end))).where(sst_hist_['T'].dt.month.isin(m), drop=True).mean(dim="T",skipna=True)
                sst_hdcst_mean = sst_hdcst_.sel(T=slice(str(self.clim_year_start), str(self.clim_year_end))).where(sst_hdcst_['T'].dt.month.isin(m), drop=True).mean(dim="T",skipna=True)
                if var in ["TMIN", "TEMP", "TMAX", "SST", "HUSS_1000", "HUSS_925", "HUSS_850", "SLP", "UGRD10", "VGRD10", "UGRD_1000", "UGRD_925", "UGRD_850", "VGRD_1000", "VGRD_925", "VGRD_850"]:
                    sst_for_bias_corrected.append(sst_for_.where(sst_for_['T'].dt.month.isin(m), drop=True) - sst_hdcst_mean + sst_hist_mean)
                if var in ["PRCP", "DSWR", "DLWR", "NOLR"]:
                    sst_for_bias_corrected.append(sst_for_.where(sst_for_['T'].dt.month.isin(m), drop=True) * sst_hist_mean / sst_hdcst_mean)
            sst_for_ = xr.concat(sst_for_bias_corrected, dim='T')

            # Concatenate hindcast and forecast data
            sst_hist_ = sst_hist_.sel(T=slice(f"{self.year_start}-01", f"{base_time.year}-{base_time.month:02d}"))
            concatenated_ds = xr.concat([sst_hist_, sst_for_], dim='T')

            if isinstance(self.rolling, int) and self.rolling > 1:
                print(f"Applied rolling mean with window size {self.rolling} for {var}.")
                concatenated_ds = concatenated_ds.rolling(T=self.rolling, center=False, min_periods=self.rolling).mean()

            print(f"Standardized or computed anomalies for {var}.")
            concatenated_ds_st = self.standardize_timeseries(concatenated_ds) if self.standardize else self.anomaly_timeseries(concatenated_ds)


            first_year = concatenated_ds_st.T.dt.year[0].item()
            last_month = concatenated_ds_st.T.dt.month[-1].item() + 1
            start_date = pd.to_datetime(f"{first_year}-{last_month:02d}")
            concatenated_ds_st = concatenated_ds_st.sel(T=slice(start_date, None))
            new_time = pd.DatetimeIndex([pd.to_datetime(f"{first_year+1}-01-01") + pd.DateOffset(months=t) for t in range(len(concatenated_ds_st['T']))])
            # ds_shifted = concatenated_ds_st.assign_coords(T=new_time, month=('T', new_time.month))
            ds_shifted = concatenated_ds_st.assign_coords(T=new_time)
    
            # ds_shifted = ds_shifted.astype({f"{var.lower()}": "float32"})
            # concatenated_ds_st = concatenated_ds_st.astype({f"{var.lower()}": "float32"})
            data_var_shifted[var] = ds_shifted.to_array().drop_vars(['variable'], errors='ignore').squeeze()
            data_var_concatenated[var] = concatenated_ds_st.to_array().drop_vars(['variable'], errors='ignore').squeeze()
        return data_var_concatenated, data_var_shifted

    def arrange_indices_for_som(self, ds):
        """Arrange climate indices for SOM training by pivoting into a year × month table.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset containing climate indices with time dimension 'T'.

        Returns
        -------
        data_array : np.ndarray
            Pivoted data array for SOM training.
        years : pd.Index
            Index of years corresponding to the data array.
        """
        ds = ds.assign(year=('T', ds['T'].dt.year.data), month_name=('T', ds['T'].dt.strftime('%b').data))
        
        def pivot_var(var_name):
            df = ds[[var_name, 'year', 'month_name']].to_dataframe().reset_index()
            df_pivot = df.pivot(index='year', columns='month_name', values=var_name)
            df_pivot = df_pivot.reindex(columns=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            df_pivot.columns = [f"{var_name}_{month}" for month in df_pivot.columns]
            return df_pivot
        
        vars_order = self.index_compute or []
        df_vars = {v: pivot_var(v) for v in vars_order}
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ordered_columns = [f"{v}_{m}" for m in months for v in vars_order]
        
        df_final = pd.concat(df_vars.values(), axis=1)[ordered_columns]
        return np.array(df_final, dtype=np.float64), df_final.index

    def identify_analogs_bmu(self, target_year, bmu_dict):
        """Identify analog years based on SOM Best Matching Units (BMUs).

        Parameters
        ----------
        target_year : int
            The target year to find analogs for.
        bmu_dict : dict
            Dictionary mapping years to BMU coordinates.

        Returns
        -------
        analogs : list
            List of analog years.
        """

        radius = self.radius or 1.0
    
        # Create a dictionary to store neurons and their associated years
        neuron_years = defaultdict(list)
        for year, coords in bmu_dict.items():
            neuron_years[coords].append(year)
        
        target_coords = bmu_dict.get(target_year)
        if not target_coords:
            return []
        
        analogs = [year for year in neuron_years[target_coords] if year != target_year]
        if not analogs:
            for year, coords in bmu_dict.items():
                if year == target_year:
                    continue
                dist = np.sqrt((coords[0] - target_coords[0])**2 + (coords[1] - target_coords[1])**2)
                if dist <= radius:
                    analogs.append(year)
        
        return analogs

    def SOM(self, predictant, itrain, ireference_year):
        """Identify similar years using Self-Organizing Maps (SOM).

        Parameters
        ----------
        predictant : xarray.DataArray
            Observed predictand data with dimensions (T, Y, X).
        itrain : list
            Indices of training years.
        ireference_year : list
            Index of the reference year.

        Returns
        -------
        similar_years : np.ndarray
            Array of years similar to the reference year.
        """
        predictant = predictant.copy()
        predictant['T'] = predictant['T'].astype('datetime64[ns]')
        predictant_ = xr.concat([predictant.isel(T=itrain), predictant.isel(T=ireference_year)], dim="T")
        unique_years = np.unique(predictant_['T'].dt.year)
        reference_year = int(np.unique(predictant.isel(T=ireference_year)['T'].dt.year))

        if self.index_compute:
            _, ddd = self.download_and_process()
            ddd = ddd.get('SST')
            if ddd is None:
                raise ValueError("SST data not found in downloaded datasets.")
            
            indices_dataset = self.calc_index(self.index_compute, ddd)
            indices_dataset_sel_years = indices_dataset.where(indices_dataset['T'].dt.year.isin(unique_years), drop=True)
            data_scaled, index_data = self.arrange_indices_for_som(indices_dataset_sel_years)
        else:
            _, ddd = self.download_and_process()
            if self.multivariateEOF:
                # Combine all variables for multivariate EOF
                data_vars = [ddd[var].rename({"X": "lon", "Y": "lat"}) for var in ddd]
                combined_data = xr.concat([d.to_array().squeeze() for d in data_vars], dim='variable').stack(feature=('variable', 'lon', 'lat'))
                print("Computing EOFs for combined data...")
                model = xe.single.EOF(n_modes=50)
                model.fit(combined_data, dim='T')
                explained_var = model.explained_variance_ratio()
                n_modes = np.where(explained_var.cumsum() >= self.eof_explained_var)[0][0] + 1
                scores = model.scores().isel(mode=slice(0, n_modes))
                scores = scores.where(scores['T'].dt.year.isin(unique_years), drop=True)
                
                df_final = pd.DataFrame()
                scores = scores.assign_coords(year=('T', scores['T'].dt.year.data), month_name=('T', scores['T'].dt.strftime('%b').data))
                df = scores.to_dataframe(name='value').reset_index()
                df['mode_month'] = df['mode'].apply(lambda m: f"mode{m:02d}") + "_" + df['month_name']
                df_pivot = df.pivot(index='year', columns='mode_month', values='value')
                month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                df_pivot = df_pivot.reindex(columns=sorted(df_pivot.columns, key=lambda x: (month_order.index(x.split('_')[1]), x)))
                df_final = pd.concat([df_final, df_pivot], axis=1)
                data_scaled = np.array(df_final, dtype=np.float64)
                index_data = df_final.index
            else:
                df_final = pd.DataFrame()
                for var in ddd:
                    scores = self.compute_eofs(ddd[var])
                    scores = scores.where(scores['T'].dt.year.isin(unique_years), drop=True)
                    scores = scores.assign_coords(year=('T', scores['T'].dt.year.data), month_name=('T', scores['T'].dt.strftime('%b').data))
                    df = scores.to_dataframe(name='value').reset_index()
                    df['mode_month'] = df['mode'].apply(lambda m: f"mode{m:02d}") + "_" + df['month_name'] + f"_{var}"
                    df_pivot = df.pivot(index='year', columns='mode_month', values='value')
                    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    df_pivot = df_pivot.reindex(columns=sorted(df_pivot.columns, key=lambda x: (month_order.index(x.split('_')[1]), x)))
                    df_final = pd.concat([df_final, df_pivot], axis=1)
                data_scaled = np.array(df_final, dtype=np.float64)
                index_data = df_final.index

        som_grid_size = (4, 4) if set(self.some_grid_size) == {None} else self.some_grid_size
        som = MiniSom(
            x=som_grid_size[0],
            y=som_grid_size[1],
            input_len=data_scaled.shape[1],
            sigma=self.some_sigma,
            learning_rate=self.some_learning_rate,
            neighborhood_function=self.some_neighborhood_function,
            random_seed=42
        )
        som.random_weights_init(data_scaled)
        som.train_random(data=data_scaled, num_iteration=self.some_num_iteration)
        
        bmu_coords = [som.winner(x) for x in data_scaled]
        bmu_dict = {year: coords for year, coords in zip(index_data, bmu_coords)}
        similar_years = self.identify_analogs_bmu(reference_year, bmu_dict)
        print(f"Similar years for {reference_year}: {similar_years}")
        return np.array(similar_years)

    def Corr_Based(self, predictant, itrain, ireference_year):
        """Identify similar years using correlation-based analog method.

        Parameters
        ----------
        predictant : xarray.DataArray
            Observed predictand data with dimensions (T, Y, X).
        itrain : list
            Indices of training years.
        ireference_year : list
            Index of the reference year.

        Returns
        -------
        similar_years : np.ndarray
            Array of years similar to the reference year based on spatial correlation.
        """
        _, ddd = self.download_and_process()
        ddd = ddd.get(self.predictor_vars[0]['variable'])
        if ddd is None:
            raise ValueError(f"Variable {self.predictor_vars[0]['variable']} not found in downloaded data.")
        
        predictant = predictant.copy()
        predictant['T'] = predictant['T'].astype('datetime64[ns]')
        reference_year = np.unique(predictant['T'].dt.year)[ireference_year][0]
        sst_reference = ddd.sel(T=str(reference_year))
        
        predictant_ = xr.concat([predictant.isel(T=itrain), predictant.isel(T=ireference_year)], dim="T")
        unique_years = np.unique(predictant_.isel(T=itrain)['T'].dt.year)
        
        correlations = []
        for year in unique_years:
            tmp = ddd.sel(T=str(year))
            tmp['T'] = sst_reference['T']
            correlation = xr.corr(tmp, sst_reference, dim="T").compute()
            correlations.append(correlation)
        
        similar = xr.concat(correlations, dim='T').assign_coords(T=unique_years)
        similar = xr.where(similar > 0.6, 1, 0).sum(dim=["X", "Y"])
        similar = similar.sortby(similar, ascending=False)
        top_3 = similar.isel(T=slice(3))
        similar_years = top_3['T'].to_numpy()
        print(f"Similar years for {reference_year}: {similar_years}")
        return similar_years

    def Pca_Based(self, predictant, itrain, ireference_year):
        """Identify similar years using PCA-based analog method.

        Parameters
        ----------
        predictant : xarray.DataArray
            Observed predictand data with dimensions (T, Y, X).
        itrain : list
            Indices of training years.
        ireference_year : list
            Index of the reference year.

        Returns
        -------
        similar_years : np.ndarray
            Array of years similar to the reference year based on EOF scores.
        """
        _, ddd = self.download_and_process()
        ddd = ddd.get(self.predictor_vars[0]['variable'])
        if ddd is None:
            raise ValueError(f"Variable {self.predictor_vars[0]['variable']} not found in downloaded data.")
        
        predictor_ = ddd.fillna(ddd.groupby("T.month").mean("T", skipna=True))
        predictor_detrend = sig.detrend(predictor_, axis=0)
        ddd = xr.DataArray(predictor_detrend, dims=predictor_.dims, coords=predictor_.coords)
        
        eof = xe.single.EOF(n_modes=50)
        eof.fit(ddd.fillna(ddd.mean(dim="T", skipna=True)), dim="T")
        scores = eof.scores()
        
        predictant = predictant.copy()
        predictant['T'] = predictant['T'].astype('datetime64[ns]')
        reference_year = np.unique(predictant['T'].dt.year)[ireference_year][0]
        sst_ref = scores.sel(T=str(reference_year)).stack(score=('mode', 'T'))
        
        unique_years = np.unique(predictant.isel(T=itrain)['T'].dt.year)
        correlations = []
        for year in unique_years:
            tmp = scores.sel(T=str(year)).stack(score=('mode', 'T'))
            correlation = xr.corr(tmp, sst_ref, dim="score").compute()
            correlations.append(correlation)
        
        similar = xr.concat(correlations, dim='T').assign_coords(T=unique_years)
        similar = similar.sortby(similar, ascending=False)
        top_3 = similar.isel(T=slice(3))
        similar_years = top_3['T'].to_numpy()
        print(f"Similar years for {reference_year}: {similar_years}")
        return similar_years

    @staticmethod
    def calculate_tercile_probabilities(best_guess, error_variance, first_tercile, second_tercile, dof):
        """Calculate tercile probabilities using Student's t-distribution.

        Parameters
        ----------
        best_guess : np.ndarray
            Forecast values with shape (n_time,).
        error_variance : np.ndarray
            Error variance with shape (n_time,).
        first_tercile : np.ndarray
            First tercile threshold.
        second_tercile : np.ndarray
            Second tercile threshold.
        dof : int
            Degrees of freedom for t-distribution.

        Returns
        -------
        pred_prob : np.ndarray
            Probabilities for below, normal, and above terciles with shape (3, n_time).
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time), dtype=np.float64)
        
        if np.all(np.isnan(best_guess)) or np.all(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        
        error_std = np.sqrt(error_variance)
        first_t = (first_tercile - best_guess) / error_std
        second_t = (second_tercile - best_guess) / error_std
        
        pred_prob[0, :] = stats.t.cdf(first_t, df=dof)
        pred_prob[1, :] = stats.t.cdf(second_t, df=dof) - stats.t.cdf(first_t, df=dof)
        pred_prob[2, :] = 1 - stats.t.cdf(second_t, df=dof)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_gamma(best_guess, error_variance, T1, T2):
        """Calculate tercile probabilities using Gamma distribution.

        Parameters
        ----------
        best_guess : np.ndarray
            Forecast values with shape (n_time,).
        error_variance : np.ndarray
            Error variance with shape (n_time,).
        T1 : np.ndarray
            First tercile threshold.
        T2 : np.ndarray
            Second tercile threshold.

        Returns
        -------
        pred_prob : np.ndarray
            Probabilities for below, normal, and above terciles with shape (3, n_time).
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time), dtype=np.float64)
        
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        
        best_guess = np.asarray(best_guess, dtype=np.float64)
        error_variance = np.asarray(error_variance, dtype=np.float64)
        T1 = np.asarray(T1, dtype=np.float64)
        T2 = np.asarray(T2, dtype=np.float64)
        
        alpha = (best_guess**2) / error_variance
        theta = error_variance / best_guess
        cdf_t1 = gamma.cdf(T1, a=alpha, scale=theta)
        cdf_t2 = gamma.cdf(T2, a=alpha, scale=theta)
        
        pred_prob[0, :] = cdf_t1
        pred_prob[1, :] = cdf_t2 - cdf_t1
        pred_prob[2, :] = 1.0 - cdf_t2
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_nonparametric(best_guess, error_samples, first_tercile, second_tercile):
        """Calculate tercile probabilities using a non-parametric method.

        Parameters
        ----------
        best_guess : np.ndarray
            Forecast values with shape (n_time,).
        error_samples : np.ndarray
            Error samples with shape (n_samples, n_time).
        first_tercile : np.ndarray
            First tercile threshold.
        second_tercile : np.ndarray
            Second tercile threshold.

        Returns
        -------
        pred_prob : np.ndarray
            Probabilities for below, normal, and above terciles with shape (3, n_time).
        """
        n_time = len(best_guess)
        pred_prob = np.full((3, n_time), np.nan, dtype=np.float64)
        
        for t in range(n_time):
            if np.isnan(best_guess[t]):
                continue
            dist = best_guess[t] + error_samples
            dist = dist[np.isfinite(dist)]
            if len(dist) == 0:
                continue
            p_below = np.mean(dist < first_tercile)
            p_between = np.mean((dist >= first_tercile) & (dist < second_tercile))
            p_above = 1.0 - (p_below + p_between)
            pred_prob[0, t] = p_below
            pred_prob[1, t] = p_between
            pred_prob[2, t] = p_above
        
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_normal(best_guess, error_variance, first_tercile, second_tercile):
        """Calculate tercile probabilities using Normal distribution.

        Parameters
        ----------
        best_guess : np.ndarray
            Forecast values with shape (n_time,).
        error_variance : np.ndarray
            Error variance with shape (n_time,).
        first_tercile : np.ndarray
            First tercile threshold.
        second_tercile : np.ndarray
            Second tercile threshold.

        Returns
        -------
        pred_prob : np.ndarray
            Probabilities for below, normal, and above terciles with shape (3, n_time).
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time), dtype=np.float64)
        
        if np.all(np.isnan(best_guess)) or np.all(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        
        error_std = np.sqrt(error_variance)
        pred_prob[0, :] = stats.norm.cdf(first_tercile, loc=best_guess, scale=error_std)
        pred_prob[1, :] = stats.norm.cdf(second_tercile, loc=best_guess, scale=error_std) - \
                          stats.norm.cdf(first_tercile, loc=best_guess, scale=error_std)
        pred_prob[2, :] = 1 - stats.norm.cdf(second_tercile, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_lognormal(best_guess, error_variance, first_tercile, second_tercile):
        """Calculate tercile probabilities using Lognormal distribution.

        Parameters
        ----------
        best_guess : np.ndarray
            Forecast values with shape (n_time,).
        error_variance : np.ndarray
            Error variance with shape (n_time,).
        first_tercile : np.ndarray
            First tercile threshold.
        second_tercile : np.ndarray
            Second tercile threshold.

        Returns
        -------
        pred_prob : np.ndarray
            Probabilities for below, normal, and above terciles with shape (3, n_time).
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time), dtype=np.float64)
        
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        
        sigma = np.sqrt(np.log(1 + error_variance / (best_guess**2)))
        mu = np.log(best_guess) - sigma**2 / 2
        pred_prob[0, :] = lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[1, :] = lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu)) - \
                          lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[2, :] = 1 - lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu))
        return pred_prob

    def compute_model(self, predictant, itrain, itest):
        """Compute deterministic hindcast using the specified analog method.

        Parameters
        ----------
        predictant : xarray.DataArray
            Observed predictand data with dimensions (T, Y, X).
        itrain : list
            Indices of training years.
        itest : list
            Index of the test year.

        Returns
        -------
        hindcast_det : xarray.DataArray
            Deterministic hindcast for the test year.
        """

        method_map = {
            "som": self.SOM,
            "cor_based": self.Corr_Based,
            "pca_based": self.Pca_Based
        }
        if self.method_analog not in method_map:
            raise ValueError(f"Invalid analog method: {self.method_analog}. Choose 'som', 'cor_based', or 'pca_based'.")

        similar_years = method_map[self.method_analog](predictant, itrain, itest)
        predictant = predictant.copy()
        predictant['T'] = predictant['T'].astype('datetime64[ns]')
        
        sim_obs = [predictant.sel(T=str(year)) for year in similar_years if year in predictant['T'].dt.year.values]
        if not sim_obs:
            raise ValueError("No valid similar years found for hindcast.")
        
        hindcast_det = xr.concat(sim_obs, dim="T").mean(dim="T").expand_dims({'T': predictant.isel(T=itest)['T'].values})
        return hindcast_det

    def compute_prob(self, predictant, clim_year_start, clim_year_end, hindcast_det):
        """Compute tercile probabilities for the hindcast.

        Parameters
        ----------
        predictant : xarray.DataArray
            Observed data array with dimensions (T, Y, X).
        clim_year_start : int
            Start year for climatology.
        clim_year_end : int
            End year for climatology.
        hindcast_det : xarray.DataArray
            Deterministic hindcast with dimensions (T, Y, X).

        Returns
        -------
        hindcast_prob : xarray.DataArray
            Probabilities for below, normal, and above terciles with dimensions (probability=3, T, Y, X).
        """
        index_start = predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        T1 = terciles.isel(quantile=0).drop_vars('quantile')
        T2 = terciles.isel(quantile=1).drop_vars('quantile')
        
        calc_func_map = {
            "t": (self.calculate_tercile_probabilities, {"dof": len(predictant.get_index("T")) - 2}),
            "gamma": (self.calculate_tercile_probabilities_gamma, {}),
            "normal": (self.calculate_tercile_probabilities_normal, {}),
            "lognormal": (self.calculate_tercile_probabilities_lognormal, {}),
            "nonparam": (self.calculate_tercile_probabilities_nonparametric, {})
        }
        
        if self.dist_method not in calc_func_map:
            raise ValueError(f"Invalid dist_method: {self.dist_method}. Choose 't', 'gamma', 'normal', 'lognormal', or 'nonparam'.")
        
        calc_func, kwargs = calc_func_map[self.dist_method]
        error_input = (predictant - hindcast_det).var(dim='T') if self.dist_method != "nonparam" else (predictant - hindcast_det).rename({'T': 'S'})
        input_core_dims = [('T',), (), (), ()] if self.dist_method != "nonparam" else [('T',), ('S',), (), ()]
        
        hindcast_prob = xr.apply_ufunc(
            calc_func,
            hindcast_det,
            error_input,
            T1,
            T2,
            input_core_dims=input_core_dims,
            vectorize=True,
            dask='parallelized',
            output_core_dims=[('probability', 'T')],
            output_dtypes=[np.float64],
            dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True},
            kwargs=kwargs
        )
        
        return hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA'])).transpose('probability', 'T', 'Y', 'X')

    def forecast(self, predictant, clim_year_start, clim_year_end, hindcast_det):
        """Generate deterministic and probabilistic forecasts for the target year.

        Parameters
        ----------
        predictant : xarray.DataArray
            Observed predictand data with dimensions (T, Y, X).
        clim_year_start : int
            Start year for climatology.
        clim_year_end : int
            End year for climatology.
        hindcast_det : xarray.DataArray
            Deterministic hindcast with dimensions (T, Y, X).

        Returns
        -------
        ddd : dict
            Dictionary of predictor datasets.
        similar_years : np.ndarray
            Array of similar years.
        forecast_det : xarray.DataArray
            Deterministic forecast.
        forecast_prob : xarray.DataArray
            Probabilistic forecast with dimensions (probability=3, T, Y, X).
        """
        predictant = predictant.copy()
        predictant['T'] = predictant['T'].astype('datetime64[ns]')
        reference_year = self.year_forecast
        unique_years = np.append(np.unique(predictant['T'].dt.year), reference_year)
        
        if self.method_analog == "som":
            if self.index_compute:
                _, ddd = self.download_and_process()
                ddd = ddd.get('SST')
                if ddd is None:
                    raise ValueError("SST data not found in downloaded datasets.")
                indices_dataset = self.calc_index(self.index_compute, ddd)
                indices_dataset_sel_years = indices_dataset.where(indices_dataset['T'].dt.year.isin(unique_years), drop=True)
                data_scaled, index_data = self.arrange_indices_for_som(indices_dataset_sel_years)
            else:
                _, ddd = self.download_and_process()
                if self.multivariateEOF:
                    data_vars = [ddd[var].rename({"X": "lon", "Y": "lat"}) for var in ddd]
                    combined_data = xr.concat([d.to_array().squeeze() for d in data_vars], dim='variable').stack(feature=('variable', 'lon', 'lat'))
                    model = xe.single.EOF(n_modes=100)
                    model.fit(combined_data, dim='T')
                    explained_var = model.explained_variance_ratio()
                    n_modes = np.where(explained_var.cumsum() >= self.eof_explained_var)[0][0] + 1
                    scores = model.scores().isel(mode=slice(0, n_modes))
                    scores = scores.where(scores['T'].dt.year.isin(unique_years), drop=True)
                    df_final = pd.DataFrame()
                    scores = scores.assign_coords(year=('T', scores['T'].dt.year.data), month_name=('T', scores['T'].dt.strftime('%b').data))
                    df = scores.to_dataframe(name='value').reset_index()
                    df['mode_month'] = df['mode'].apply(lambda m: f"mode{m:02d}") + "_" + df['month_name']
                    df_pivot = df.pivot(index='year', columns='mode_month', values='value')
                    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    df_pivot = df_pivot.reindex(columns=sorted(df_pivot.columns, key=lambda x: (month_order.index(x.split('_')[1]), x)))
                    df_final = pd.concat([df_final, df_pivot], axis=1)
                    data_scaled = np.array(df_final, dtype=np.float64)
                    index_data = df_final.index
                else:
                    df_final = pd.DataFrame()
                    for var in ddd:
                        scores = self.compute_eofs(ddd[var])
                        scores = scores.where(scores['T'].dt.year.isin(unique_years), drop=True)
                        scores = scores.assign_coords(year=('T', scores['T'].dt.year.data), month_name=('T', scores['T'].dt.strftime('%b').data))
                        df = scores.to_dataframe(name='value').reset_index()
                        df['mode_month'] = df['mode'].apply(lambda m: f"mode{m:02d}") + "_" + df['month_name'] + f"_{var}"
                        df_pivot = df.pivot(index='year', columns='mode_month', values='value')
                        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                        df_pivot = df_pivot.reindex(columns=sorted(df_pivot.columns, key=lambda x: (month_order.index(x.split('_')[1]), x)))
                        df_final = pd.concat([df_final, df_pivot], axis=1)
                    data_scaled = np.array(df_final, dtype=np.float64)
                    index_data = df_final.index
            
            som_grid_size = (2, 4) if set(self.some_grid_size) == {None} else self.some_grid_size
            som = MiniSom(
                x=som_grid_size[0],
                y=som_grid_size[1],
                input_len=data_scaled.shape[1],
                sigma=self.some_sigma,
                learning_rate=self.some_learning_rate,
                neighborhood_function=self.some_neighborhood_function,
                random_seed=42
            )
            som.random_weights_init(data_scaled)
            som.train_random(data=data_scaled, num_iteration=self.some_num_iteration)
            bmu_coords = [som.winner(x) for x in data_scaled]
            bmu_dict = {year: coords for year, coords in zip(index_data, bmu_coords)}
            similar_years = self.identify_analogs_bmu(reference_year, bmu_dict)
        
        elif self.method_analog == "cor_based":
            _, ddd = self.download_and_process()
            ddd = ddd.get(self.predictor_vars[0]['variable'])
            if ddd is None:
                raise ValueError(f"Variable {self.predictor_vars[0]['variable']} not found in downloaded data.")
            sst_reference = ddd.sel(T=str(reference_year))
            unique_years = np.unique(predictant['T'].dt.year)
            correlations = []
            for year in unique_years:
                tmp = ddd.sel(T=str(year))
                tmp['T'] = sst_reference['T']
                correlation = xr.corr(tmp, sst_reference, dim="T").compute()
                correlations.append(correlation)
            similar = xr.concat(correlations, dim='T').assign_coords(T=unique_years)
            similar = xr.where(similar > 0.6, 1, 0).sum(dim=["X", "Y"])
            similar = similar.sortby(similar, ascending=False)
            top_3 = similar.isel(T=slice(3))
            similar_years = top_3['T'].to_numpy()
        
        elif self.method_analog == "pca_based":
            _, ddd = self.download_and_process()
            ddd = ddd.get(self.predictor_vars[0]['variable'])
            if ddd is None:
                raise ValueError(f"Variable {self.predictor_vars[0]['variable']} not found in downloaded data.")
            predictor_ = ddd.fillna(ddd.groupby("T.month").mean("T", skipna=True))
            predictor_detrend = sig.detrend(predictor_, axis=0)
            ddd = xr.DataArray(predictor_detrend, dims=predictor_.dims, coords=predictor_.coords)
            eof = xe.single.EOF(n_modes=50)
            eof.fit(ddd.fillna(ddd.mean(dim="T", skipna=True)), dim="T")
            scores = eof.scores()
            sst_ref = scores.sel(T=str(reference_year)).stack(score=('mode', 'T'))
            unique_years = np.unique(predictant['T'].dt.year)
            correlations = []
            for year in unique_years:
                tmp = scores.sel(T=str(year)).stack(score=('mode', 'T'))
                correlation = xr.corr(tmp, sst_ref, dim="score").compute()
                correlations.append(correlation)
            similar = xr.concat(correlations, dim='T').assign_coords(T=unique_years)
            similar = similar.sortby(similar, ascending=False)
            top_3 = similar.isel(T=slice(3))
            similar_years = top_3['T'].to_numpy()
        else:
            raise ValueError(f"Invalid analog method: {self.method_analog}. Choose 'som', 'cor_based', or 'pca_based'.")
        
        sim_obs = [predictant.sel(T=str(year)) for year in similar_years if year in predictant['T'].dt.year.values]
        if not sim_obs:
            raise ValueError("No valid similar years found for forecast.")
        
        forecast_det = xr.concat(sim_obs, dim="T").mean(dim="T")
        T_value = predictant.isel(T=0).coords['T'].values
        month = T_value.astype('datetime64[M]').astype(int) % 12 + 1
        new_T_value = np.datetime64(f"{reference_year}-{month:02d}-01")
        forecast_det = forecast_det.expand_dims({'T': [new_T_value]})
        forecast_det['T'] = forecast_det['T'].astype('datetime64[ns]')
        
        index_start = predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        T1 = terciles.isel(quantile=0).drop_vars('quantile')
        T2 = terciles.isel(quantile=1).drop_vars('quantile')
        
        calc_func_map = {
            "t": (self.calculate_tercile_probabilities, {"dof": len(predictant.get_index("T")) - 2}),
            "gamma": (self.calculate_tercile_probabilities_gamma, {}),
            "normal": (self.calculate_tercile_probabilities_normal, {}),
            "lognormal": (self.calculate_tercile_probabilities_lognormal, {}),
            "nonparam": (self.calculate_tercile_probabilities_nonparametric, {})
        }
        
        calc_func, kwargs = calc_func_map[self.dist_method]
        error_input = (predictant - hindcast_det).var(dim='T') if self.dist_method != "nonparam" else (predictant - hindcast_det).rename({'T': 'S'})
        input_core_dims = [('T',), (), (), ()] if self.dist_method != "nonparam" else [('T',), ('S',), (), ()]
        
        forecast_prob = xr.apply_ufunc(
            calc_func,
            forecast_det,
            error_input,
            T1,
            T2,
            input_core_dims=input_core_dims,
            vectorize=True,
            dask='parallelized',
            output_core_dims=[('probability', 'T')],
            output_dtypes=[np.float64],
            dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True},
            kwargs=kwargs
        )
        
        forecast_prob = forecast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA'])).transpose('probability', 'T', 'Y', 'X')
        print(f"Similar years for {reference_year}: {similar_years}")
        return similar_years, forecast_det, forecast_prob


    def composite_plot(self, predictant, clim_year_start, clim_year_end, hindcast_det, plot_predictor=True, variable="SST"):
        """Create composite plots of predictors or predictands.

        Parameters
        ----------
        predictant : xarray.DataArray
            Observed predictand data with dimensions (T, Y, X).
        clim_year_start : int
            Start year for climatology period.
        clim_year_end : int
            End year for climatology period.
        hindcast_det : xarray.DataArray
            Deterministic hindcast data with dimensions (T, Y, X).
        plot_predictor : bool, optional
            If True, plot SST predictors; otherwise, plot precipitation ratios. Default is True.

        Returns
        -------
        similar_years : np.ndarray
            Array of similar years used in the composite.
        """
        clim_slice = slice(str(clim_year_start), str(clim_year_end))
        clim_mean = predictant.sel(T=clim_slice).mean(dim='T', skipna=True)
        similar_years, result_, _ = self.forecast(predictant, clim_year_start, clim_year_end, hindcast_det)
        result_ = result_.drop_vars('T', errors='ignore').squeeze()
        reference_year = self.year_forecast
        _, ddd = self.download_and_process()

        ddd = ddd[variable]
        
        sim_all = []
        tmp = ddd.sel(T=str(reference_year))
        months = list(tmp['T'].dt.month.values)
        tmp['T'] = months
        sim_all.append(tmp)
        
        sim_ = []
        pred_rain = []
        for year in np.array([str(i) for i in similar_years]):
            tmp = ddd.sel(T=year)
            months = list(tmp['T'].dt.month.values)
            tmp['T'] = months
            sim_.append(tmp)
            pred_rain.append(100 * predictant.sel(T=year) / clim_mean)
        
        sim__ = xr.concat(sim_, dim="year").assign_coords(year=('year', similar_years)).mean(dim="year", skipna=True)
        sim_all.append(sim__)
        sim_all = xr.concat(sim_all, dim="output").assign_coords(output=('output', ['forecast year', 'composite analog']))
        
        pred_rain = xr.concat(pred_rain, dim='T')
        
        if plot_predictor:
            sim_all.plot(
                x="X", y="Y", row="T", col="output",
                figsize=(12, len(months) * 4),
                cbar_kwargs={"shrink": 0.3, "aspect": 50, "pad": 0.05, "label": f"{variable} anomaly"},
                robust=True,
                subplot_kws={'projection': ccrs.PlateCarree()}
            )
            for ax in plt.gcf().axes:
                if isinstance(ax, plt.Axes) and hasattr(ax, 'coastlines'):
                    ax.coastlines()
                    ax.gridlines(draw_labels=True)
                    ax.add_feature(cfeature.LAND, edgecolor="black")
                    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
            plt.suptitle(f"SST Composites for {reference_year}", fontsize=14)
            plt.tight_layout()
            plt.show()
        else:
            colors_list = ['#d7191c', '#fdae61', '#ffffbf', '#a6d96a', '#1a9641']
            bounds = [0, 50, 90, 110, 150, 200]
            cmap = ListedColormap(colors_list)
            norm = BoundaryNorm(bounds, cmap.N)
            
            data_var = pred_rain.sel(Y=slice(None, 19.5))
            n_times = len(data_var['T'])
            n_cols = 2
            n_rows = (n_times + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(
                n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4),
                subplot_kw={'projection': ccrs.PlateCarree()}
            )
            axes = np.ravel(axes)
            
            for i, t in enumerate(data_var['T'].values):
                ax = axes[i]
                data = data_var.sel(T=t).squeeze()
                im = ax.pcolormesh(
                    data['X'], data['Y'], data, cmap=cmap, norm=norm,
                    transform=ccrs.PlateCarree()
                )
                ax.coastlines()
                ax.gridlines(draw_labels=True)
                ax.add_feature(cfeature.LAND, edgecolor="black")
                ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
                ax.set_title(f"Season: {str(t)[:10]}")
            
            for j in range(n_times, len(axes)):
                fig.delaxes(axes[j])
            
            cbar = fig.colorbar(im, ax=axes[:n_times], orientation="horizontal", fraction=0.05, pad=0.1)
            cbar.set_label('Precipitation Ratio to Normal (%)')
            cbar.set_ticks([0, 50, 90, 110, 150, 200])
            cbar.ax.set_xticklabels(['0', '50', '90', '110', '150', '200'])
            fig.suptitle("Ratio to Normal Precipitation [%]", fontsize=14)
            plt.tight_layout()
            fig.subplots_adjust(top=0.9, bottom=0.15)
            plt.show()        
        return similar_years

    def plot_indices(self, indices_dataset):
        """Plot time series of computed climate indices.

        Parameters
        ----------
        indices_dataset : xarray.Dataset
            Dataset containing climate indices with time dimension 'T'.

        Returns
        -------
        None
        """


class WAS_Analog:
    """Analog-based forecasting toolkit for seasonal climate applications.

    This class orchestrates the end-to-end workflow required to build analog ensembles
    for Sea-Surface Temperature (SST) predictors and to translate them into deterministic
    and probabilistic rainfall forecasts over West Africa or any user-defined domain.
    Supports five analog-selection strategies: Self-Organizing Maps (SOM),
    correlation ranking, Principal-Component (EOF) similarity, K-Means clustering,
    and Agglomerative clustering.

    Parameters
    ----------
    dir_to_save : str
        Directory path to save downloaded and processed data files.
    year_start : int
        Starting year for historical data.
    year_forecast : int
        Target forecast year.
    predictor_vars : list of dict, optional
        List of dictionaries specifying predictor variables, each containing
        'reanalysis_name', 'model_name', 'variable', and 'area'.
        Default is a list with NOAA SST and ERA5 SP, VGRD_850 variables.
    method_analog : str, optional
        Analog method to use ("som", "cor_based", "pca_based", "kmeans_based", "agglomerative_based"). Default is "som".
    month_of_initialization : int, optional
        Month of initialization for forecasts (1-12). If None, uses previous month.
    lead_time : list, optional
        List of lead times in months. If None, defaults to [1, 2, 3, 4, 5].
    ensemble_mean : str, optional
        Method for ensemble mean ("mean" or "median"). Default is "mean".
    rolling : int, optional
        Window size for rolling mean. Default is 3.
    standardize : bool, optional
        If True, standardize data; otherwise, compute anomalies. Default is True.
    multivariateEOF : bool, optional
        If True, use multivariate EOF analysis. Default is False.
    eof_explained_var : float, optional
        Fraction of variance to explain with EOF modes. Default is 0.95.
    clim_year_start : int, optional
        Start year for climatology period.
    clim_year_end : int, optional
        End year for climatology period.
    index_compute : list, optional
        List of climate indices to compute (e.g., ['NINO34', 'DMI']).
    some_grid_size : tuple, optional
        Grid size for SOM (rows, columns). Default is (None, None) for automatic sizing.
    some_learning_rate : float, optional
        Learning rate for SOM training. Default is 0.5.
    radius : float, optional
        Neighborhood radius for SOM analog search. Default is 1.0.
    some_neighborhood_function : str, optional
        Neighborhood function for SOM ("gaussian", "mexican_hat"). Default is "gaussian".
    some_sigma : float, optional
        Initial neighborhood radius for SOM. Default is 1.0.
    some_num_iteration : int, optional
        Number of iterations for SOM training. Default is 2000.
    dist_method : str, optional
        Method for probability calculation ("gamma", "t", "normal", "lognormal", "nonparam").
        Default is "gamma".
    n_clusters : int, optional
        Number of clusters for K-Means and Agglomerative clustering. Default is 4.
    affinity : str, optional
        Distance metric for Agglomerative clustering (e.g., 'euclidean', 'manhattan'). Default is 'euclidean'.
    linkage : str, optional
        Linkage criterion for Agglomerative clustering (e.g., 'ward', 'complete', 'average'). Default is 'ward'.

    Notes
    -----
    - Performance Considerations: For large datasets (e.g., many years or high-dimensional data), the AgglomerativeClustering method (`agglomerative_based`) may be significantly slower than K-Means (`kmeans_based`) or SOM (`som`) due to its computational complexity. Users should consider testing with smaller datasets or reducing the number of clusters when using AgglomerativeClustering.
    - The `plot_clusters` method can be used to visualize clustering results for `kmeans_based` and `agglomerative_based` methods, projecting years onto a 2D PCA space colored by cluster.
    """

    def __init__(self, dir_to_save, year_start, year_forecast,
                 predictor_vars=[{'reanalysis_name': 'NOAA', 'model_name': 'NCEP_2', 'variable': 'SST', 'area': [60, -180, -60, 180]},
                                {'reanalysis_name': 'ERA5', 'model_name': 'NCEP_2', 'variable': 'SP', 'area': [60, -180, -60, 180]},
                                {'reanalysis_name': 'ERA5', 'model_name': 'NCEP_2', 'variable': 'VGRD_850', 'area': [60, -180, -60, 180]}],
                 method_analog="som", month_of_initialization=None,
                 lead_time=None, ensemble_mean="mean", rolling=3, standardize=True, multivariateEOF=False,
                 eof_explained_var=0.95, clim_year_start=None, clim_year_end=None,
                 index_compute=None, some_grid_size=(None, None), some_learning_rate=0.5, radius=1.0,
                 some_neighborhood_function='gaussian', some_sigma=1.0, some_num_iteration=2000, dist_method="gamma",
                 n_clusters=4, affinity='euclidean', linkage='ward'):
        
        self.dir_to_save = dir_to_save
        self.year_start = year_start
        self.year_forecast = year_forecast
        self.predictor_vars = predictor_vars
        self.method_analog = method_analog
        self.month_of_initialization = month_of_initialization
        self.lead_time = lead_time
        self.ensemble_mean = ensemble_mean
        self.eof_explained_var = eof_explained_var
        self.multivariateEOF = multivariateEOF
        self.rolling = rolling
        self.standardize = standardize
        self.clim_year_start = clim_year_start
        self.clim_year_end = clim_year_end
        self.index_compute = index_compute
        self.some_grid_size = some_grid_size
        self.some_learning_rate = some_learning_rate
        self.radius = radius
        self.some_neighborhood_function = some_neighborhood_function
        self.some_sigma = some_sigma
        self.some_num_iteration = some_num_iteration
        self.dist_method = dist_method
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.linkage = linkage

    def calc_index(self, indices, sst):
        """Calculate climate indices from SST data."""
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
            "M1": ("M1", -50, 5, -50, -25),
            "M2": ("M2", -75, -10, 25, 50),
            "M3": ("M3", -175, -125, 25, 50),
            "M4": ("M4", -175, -125, -50, -25),
        }
        
        predictor = {}
        for idx in sst_indices_name:
            if idx in ["TASI", "DMI"]:
                continue
            _, lon_min, lon_max, lat_min, lat_max = sst_indices_name[idx]
            sst_region = sst.sel(X=slice(lon_min, lon_max), Y=slice(lat_min, lat_max)).mean(dim=["X", "Y"], skipna=True)
            predictor[idx] = sst_region

        predictor["TASI"] = predictor["NAT"] - predictor["SAT"]
        predictor["DMI"] = predictor["WTIO"] - predictor["SETIO"]
        
        selected_indices = {i: predictor[i] for i in indices if i in predictor}
        data_vars = {key: ds.rename(key) for key, ds in selected_indices.items()}
        return xr.Dataset(data_vars)

    def _postprocess_ersst(self, ds, var_name):
        """Post-process ERSST dataset."""
        ds = ds.drop_vars('zlev', errors='ignore').squeeze()
        keep_vars = [var_name, 'T', 'X', 'Y']
        drop_vars = [v for v in ds.variables if v not in keep_vars]
        return ds.drop_vars(drop_vars, errors="ignore")

    def download_reanalysis(self, force_download=False):
        """Download reanalysis data."""
        year_end = self.year_forecast
        variables = [item['variable'] for item in self.predictor_vars]
        centers = [item['reanalysis_name'] for item in self.predictor_vars]
        areas = [item['area'] for item in self.predictor_vars]
        
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
            "NOLR": "top_net_thermal_radiation",
        }
        variables_2 = {
            "HUSS_1000": "specific_humidity",
            "HUSS_925": "specific_humidity",
            "HUSS_850": "specific_humidity",
            "UGRD_1000": "u_component_of_wind",
            "UGRD_925": "u_component_of_wind",
            "UGRD_850": "u_component_of_wind",
            "VGRD_1000": "v_component_of_wind",
            "VGRD_925": "v_component_of_wind",
            "VGRD_850": "v_component_of_wind",
        }

        dir_to_save = Path(self.dir_to_save)
        os.makedirs(dir_to_save, exist_ok=True)
    
        months = [f"{m:02d}" for m in range(1, 13)]
        season = "".join([calendar.month_abbr[int(m)] for m in months])
        store_file_path = {}

        for center, var, area in zip(centers, variables, areas):
            combined_output_path = dir_to_save / f"{center}_{var}_{self.month_of_initialization}_{self.year_start}_{year_end}_{season}.nc"
        
            if not force_download and combined_output_path.exists():
                print(f"{combined_output_path} exists. Skipping download.")
                store_file_path[var] = xr.open_dataset(combined_output_path)
                continue

            if f"{center}.{var}" == "NOAA.SST":
                try:
                    url = build_iridl_url_ersst(
                        year_start=self.year_start,
                        year_end=year_end,
                        bbox=area,
                        run_avg=None,
                        month_start="Jan",
                        month_end="Dec"
                    )
                    print(f"Using IRIDL URL: {url}")
        
                    ds = xr.open_dataset(url, decode_times=False)
                    ds = decode_cf(ds, "T").rename({"T": "time"}).convert_calendar("proleptic_gregorian", align_on="year").rename({"time": "T"})
                    ds = ds.assign_coords(T=ds.T - pd.Timedelta(days=15))
                    ds = ds.rename({'sst': 'SST'})
                    ds = self._postprocess_ersst(ds, var)
                    ds['T'] = ds['T'].astype('datetime64[ns]')
                    ds = ds.rename({'SST': 'sst'})
                    store_file_path[var] = ds
                    ds.to_netcdf(combined_output_path)
                    print(f"Saved NOAA ERSST data to {combined_output_path}")
                    return ds
                except Exception as e:
                    print(f"Failed to download NOAA.SST: {e}")
                    continue

            combined_datasets = []
            client = cdsapi.Client()
        
            for year in range(self.year_start, year_end + 1):
                yearly_file_path = dir_to_save / f"{center}_{var}_{year}.nc"
            
                if not force_download and yearly_file_path.exists():
                    print(f"{yearly_file_path} exists. Loading existing file.")
                    try:
                        ds = xr.open_dataset(yearly_file_path).load()
                        if 'latitude' in ds.coords and 'longitude' in ds.coords:
                            ds = ds.rename({"latitude": "Y", "longitude": "X", "valid_time": "T"})
                        combined_datasets.append(ds)
                        continue
                    except Exception as e:
                        print(f"Failed to load {yearly_file_path}: {e}")
            
                try:
                    if var in variables_2:
                        press_level = var.split("_")[1]
                        dataset = "reanalysis-era5-pressure-levels-monthly-means"
                        request = {
                            "product_type": "monthly_averaged_reanalysis",
                            "variable": variables_2[var],
                            "pressure_level": press_level,
                            "year": str(year),
                            "month": months,
                            "time": "00:00",
                            "area": area,
                            "format": "netcdf",
                        }
                    else:
                        dataset = "reanalysis-era5-single-levels-monthly-means"
                        request = {
                            "product_type": "monthly_averaged_reanalysis",
                            "variable": variables_1.get(var, var),
                            "year": str(year),
                            "month": months,
                            "time": "00:00",
                            "area": area,
                            "format": "netcdf",
                        }
            
                    print(f"Downloading {var} for {year} from {center}...")
                    client.retrieve(dataset, request).download(str(yearly_file_path))
                
                    with xr.open_dataset(yearly_file_path) as ds:
                        if 'latitude' in ds.coords and 'longitude' in ds.coords:
                            ds = ds.rename({"latitude": "Y", "longitude": "X", "valid_time": "T"})
                        ds = ds.load()
                        combined_datasets.append(ds)
                except Exception as e:
                    print(f"Failed to download/process {var} for {year}: {e}")
                    continue
        
            if combined_datasets:
                print(f"Concatenating {var} datasets...")
                combined_ds = xr.concat(combined_datasets, dim="T")
                combined_ds = combined_ds.drop_vars(["number", "expver"], errors="ignore").squeeze()
                
                if var in ["TMIN", "TEMP", "TMAX", "SST"]:
                    combined_ds = combined_ds - 273.15
                elif var == "PRCP":
                    combined_ds = combined_ds * 1000
                elif var in ["DSWR", "DLWR", "NOLR"]:
                    combined_ds = combined_ds / 86400
                elif var == "SLP":
                    combined_ds = combined_ds / 100
                
                combined_ds = combined_ds.isel(Y=slice(None, None, -1))
                store_file_path[var] = combined_ds
                combined_ds.to_netcdf(combined_output_path)
                print(f"Saved combined dataset to {combined_output_path}")
                
                for year in range(self.year_start, year_end + 1):
                    single_file_path = dir_to_save / f"{center}_{var}_{year}.nc"
                    if single_file_path.exists():
                        os.remove(single_file_path)
                        print(f"Deleted yearly file: {single_file_path}")
            else:
                print(f"No data combined for {var}. Check download success.")
        
        return store_file_path

    def download_models(self, force_download=False):
        """Download and process seasonal forecast and hindcast data."""
        centre_map = {
            "BOM_2": "bom", "ECMWF_51": "ecmwf", "UKMO_604": "ukmo", "UKMO_603": "ukmo",
            "METEOFRANCE_8": "meteo_france", "METEOFRANCE_9": "meteo_france",
            "DWD_21": "dwd", "DWD_22": "dwd", "CMCC_35": "cmcc",
            "NCEP_2": "ncep", "JMA_3": "jma", "ECCC_4": "eccc", "ECCC_5": "eccc"
        }
        system_map = {
            "BOM_2": "2", "ECMWF_51": "51", "UKMO_604": "604", "UKMO_603": "603",
            "METEOFRANCE_8": "8", "METEOFRANCE_9": "9", "DWD_21": "21", "DWD_22": "22",
            "CMCC_35": "35", "NCEP_2": "2", "JMA_3": "3", "ECCC_4": "4", "ECCC_5": "5"
        }
        variables_map = {
            "PRCP": "total_precipitation", "TEMP": "2m_temperature",
            "TMAX": "maximum_2m_temperature_in_the_last_24_hours",
            "TMIN": "minimum_2m_temperature_in_the_last_24_hours",
            "UGRD10": "10m_u_component_of_wind", "VGRD10": "10m_v_component_of_wind",
            "SST": "sea_surface_temperature", "SLP": "mean_sea_level_pressure",
            "DSWR": "surface_solar_radiation_downwards",
            "DLWR": "surface_thermal_radiation_downwards",
            "NOLR": "top_thermal_radiation",
            "HUSS_1000": "specific_humidity", "HUSS_925": "specific_humidity",
            "HUSS_850": "specific_humidity", "UGRD_1000": "u_component_of_wind",
            "UGRD_925": "u_component_of_wind", "UGRD_850": "u_component_of_wind",
            "VGRD_1000": "v_component_of_wind", "VGRD_925": "v_component_of_wind",
            "VGRD_850": "v_component_of_wind",
        }

        centers = [item['model_name'] for item in self.predictor_vars]
        variables = [item['variable'] for item in self.predictor_vars]
        areas = [item['area'] for item in self.predictor_vars]
        selected_centres = [centre_map[k] for k in centers]
        selected_systems = [system_map[k] for k in centers]

        month_of_initialization = (datetime.now() - relativedelta(months=1)).month if self.month_of_initialization is None else self.month_of_initialization
        lead_time = [1, 2, 3, 4, 5] if self.lead_time is None else self.lead_time
        if not isinstance(lead_time, list) or any(l < 1 or l > 12 for l in lead_time):
            raise ValueError("lead_time must be a list of integers between 1 and 12.")
        year_forecast = datetime.now().year if self.year_forecast is None else self.year_forecast
        hindcast_years = [str(year) for year in range(self.clim_year_start, self.clim_year_end + 1)]

        dir_to_save = Path(self.dir_to_save)
        dir_to_save.mkdir(parents=True, exist_ok=True)
        abb_mont_ini = calendar.month_abbr[int(month_of_initialization)]
        season_months = [((month_of_initialization + l - 1) % 12) + 1 for l in lead_time]
        season_str = "".join(calendar.month_abbr[m] for m in season_months)

        store_file_path = {}
        store_hdcst_file_path = {}

        for cent, syst, var, area in zip(selected_centres, selected_systems, variables, areas):
            forecast_file = dir_to_save / f"forecast_{cent}{syst}_{var}_{abb_mont_ini}Ic_{season_str}_{lead_time[0]}.nc"
            hindcast_file = dir_to_save / f"hindcast_{cent}{syst}_{var}_{abb_mont_ini}Ic_{season_str}_{lead_time[0]}.nc"
               
            if not force_download and forecast_file.exists():
                print(f"Forecast file {forecast_file} exists. Skipping download.")
                ds = xr.open_dataset(forecast_file)
                store_file_path[var] = ds
                ds.close()
            else:
                try:
                    dataset = "seasonal-monthly-pressure-levels" if var in variables_map and any(v in var for v in ["HUSS", "UGRD", "VGRD"]) else "seasonal-monthly-single-levels"
                    forecast_request = {
                        "originating_centre": cent,
                        "system": syst,
                        "variable": variables_map[var],
                        "product_type": ["monthly_mean"],
                        "year": [str(year_forecast)],
                        "month": [f"{month_of_initialization:02d}"],
                        "leadtime_month": lead_time,
                        "data_format": "netcdf",
                        "area": area,
                    }
                    if var in variables_map and any(v in var for v in ["HUSS", "UGRD", "VGRD"]):
                        forecast_request["pressure_level"] = var.split("_")[1]

                    client = cdsapi.Client()
                    client.retrieve(dataset, forecast_request).download(str(forecast_file))
                    print(f"Downloaded forecast: {forecast_file}")

                    ds_forecast = xr.open_dataset(forecast_file)
                    if var in ["TMIN", "TEMP", "TMAX", "SST"]:
                        ds_forecast = ds_forecast - 273.15
                    elif var == "PRCP":
                        ds_forecast = ds_forecast * (1000 * 30 * 24 * 3600)
                    elif var == "SLP":
                        ds_forecast = ds_forecast / 100
                    elif var in ["DSWR", "DLWR", "NOLR"]:
                        ds_forecast = ds_forecast / 86400

                    if self.ensemble_mean and "number" in ds_forecast.dims:
                        ds_forecast = getattr(ds_forecast, self.ensemble_mean)(dim="number", skipna=True)

                    ds_forecast = ds_forecast.isel(latitude=slice(None, None, -1))
                    rename_dict = {
                        "latitude": "Y",
                        "longitude": "X",
                        "time": "T" if "time" in ds_forecast.coords else None,
                        "indexing_time": "T" if "indexing_time" in ds_forecast.coords else None,
                        "forecast_reference_time": "T" if "forecast_reference_time" in ds_forecast.coords else None,
                    }
                    rename_dict = {k: v for k, v in rename_dict.items() if v is not None}
                    ds_forecast = ds_forecast.rename(rename_dict)

                    if var in variables_map and any(v in var for v in ["HUSS", "UGRD", "VGRD"]):
                        ds_forecast = ds_forecast.drop_vars("pressure_level", errors="ignore").squeeze()

                    ds_forecast = ds_forecast.sortby("T")
                    ds_forecast.to_netcdf(forecast_file)
                    print(f"Saved processed forecast to {forecast_file}")
                    store_file_path[var] = ds_forecast
                    ds_forecast.close()

                except Exception as e:
                    print(f"Failed to process {var} for {cent}{syst}: {e}")
                    continue

            if not force_download and hindcast_file.exists():
                print(f"Hindcast file {hindcast_file} exists. Skipping download.")
                ds = xr.open_dataset(hindcast_file)
                store_hdcst_file_path[var] = ds
                ds.close()
            else:
                try:
                    dataset = "seasonal-monthly-pressure-levels" if var in variables_map and any(v in var for v in ["HUSS", "UGRD", "VGRD"]) else "seasonal-monthly-single-levels"
                    hindcast_request = {
                        "originating_centre": cent,
                        "system": syst,
                        "variable": variables_map[var],
                        "product_type": ["monthly_mean"],
                        "year": hindcast_years,
                        "month": [f"{month_of_initialization:02d}"],
                        "leadtime_month": lead_time,
                        "data_format": "netcdf",
                        "area": area,
                    }
                    if var in variables_map and any(v in var for v in ["HUSS", "UGRD", "VGRD"]):
                        hindcast_request["pressure_level"] = var.split("_")[1]
        
                    client = cdsapi.Client()
                    client.retrieve(dataset, hindcast_request).download(str(hindcast_file))
                    print(f"Downloaded hindcast: {hindcast_file}")

                    ds_hindcast = xr.open_dataset(hindcast_file)
                    if var in ["TMIN", "TEMP", "TMAX", "SST"]:
                        ds_hindcast = ds_hindcast - 273.15
                    elif var == "PRCP":
                        ds_hindcast = ds_hindcast * (1000 * 30 * 24 * 3600)
                    elif var == "SLP":
                        ds_hindcast = ds_hindcast / 100
                    elif var in ["DSWR", "DLWR", "NOLR"]:
                        ds_hindcast = ds_hindcast / 86400

                    if self.ensemble_mean and "number" in ds_hindcast.dims:
                        ds_hindcast = getattr(ds_hindcast, self.ensemble_mean)(dim="number", skipna=True)                        

                    ds_hindcast = ds_hindcast.isel(latitude=slice(None, None, -1))
                    rename_dict = {
                        "latitude": "Y",
                        "longitude": "X",
                        "time": "T" if "time" in ds_hindcast.coords else None,
                        "indexing_time": "T" if "indexing_time" in ds_hindcast.coords else None,
                        "forecast_reference_time": "T" if "forecast_reference_time" in ds_hindcast.coords else None,
                    }
                    rename_dict = {k: v for k, v in rename_dict.items() if v is not None}
                    ds_hindcast = ds_hindcast.rename(rename_dict)

                    if var in variables_map and any(v in var for v in ["HUSS", "UGRD", "VGRD"]):
                        ds_hindcast = ds_hindcast.drop_vars("pressure_level", errors="ignore").squeeze()

                    ds_hindcast = ds_hindcast.sortby("T")
                    ds_hindcast.to_netcdf(hindcast_file)
                    print(f"Saved processed hindcast to {hindcast_file}")
                    store_hdcst_file_path[var] = ds_hindcast
                    ds_hindcast.close()

                except Exception as e:
                    print(f"Failed to process {var} for {cent}{syst}: {e}")
                    continue

        return store_hdcst_file_path, store_file_path

    def anomaly_timeseries(self, ds):
        """Compute anomalies by removing the climatological mean."""
        if self.clim_year_start and self.clim_year_end:
            clim_period = ds.sel(T=slice(f"{self.clim_year_start}-01-01", f"{self.clim_year_end}-12-31"))
        else:
            clim_period = ds

        clim_mean = clim_period.groupby("T.month").mean("T", skipna=True)
        anomaly_slices = []

        for month in range(1, 13):
            month_mask = ds["T"].dt.month == month
            ds_month = ds.where(month_mask, drop=True)
            month_mean = clim_mean.sel(month=month)
            ds_anomaly_month = ds_month - month_mean
            anomaly_slices.append(ds_anomaly_month)

        ds_anomaly = xr.concat(anomaly_slices, dim="T")
        ds_anomaly = ds_anomaly.sortby("T")
        if "month" in ds_anomaly.coords:
            ds_anomaly = ds_anomaly.drop_vars("month")
        if "month" in ds_anomaly.dims:
            ds_anomaly = ds_anomaly.drop_dims("month")
        return ds_anomaly

    def standardize_timeseries(self, ds):
        """Standardize the dataset by removing climatological mean and scaling by standard deviation."""
        if self.clim_year_start and self.clim_year_end:
            clim_period = ds.sel(T=slice(f"{self.clim_year_start}-01-01", f"{self.clim_year_end}-12-31"))
        else:
            clim_period = ds

        clim_mean = clim_period.groupby("T.month").mean("T", skipna=True)
        clim_std = clim_period.groupby("T.month").std("T", skipna=True)
        clim_std = clim_std.where(clim_std != 0, 1e-10)

        standardized_slices = []

        for month in range(1, 13):
            month_mask = ds["T"].dt.month == month
            ds_month = ds.where(month_mask, drop=True)
            month_mean = clim_mean.sel(month=month)
            month_std = clim_std.sel(month=month)
            ds_standardized_month = (ds_month - month_mean) / month_std
            standardized_slices.append(ds_standardized_month)

        ds_standardized = xr.concat(standardized_slices, dim="T")
        ds_standardized = ds_standardized.sortby("T")
        if "month" in ds_standardized.coords:
            ds_standardized = ds_standardized.drop_vars("month")
        if "month" in ds_standardized.dims:
            ds_standardized = ds_standardized.drop_dims("month")
        return ds_standardized

    def compute_eofs(self, data):
        """Compute EOFs and retain modes explaining at least the specified variance."""
        data = data.rename({"X": "lon", "Y": "lat"})
        data = data.fillna(data.mean(dim="T", skipna=True))
        print("Computing EOFs...")
        model = xe.single.EOF(n_modes=100)
        model.fit(data, dim='T')
        explained_var = model.explained_variance_ratio()
        cumulative_var = explained_var.cumsum()
        n_modes = np.where(cumulative_var >= self.eof_explained_var)[0][0] + 1
        print(f"Selected {n_modes} modes explaining {cumulative_var[n_modes-1].values*100:.2f}% variance")
        return model.scores().isel(mode=slice(0, n_modes))

    def _prepare_data_for_clustering(self, unique_years, target_year=None):
        """Helper to prepare data (indices or EOFs) for clustering methods."""
        _, ddd = self.download_and_process()
        data_scaled = None
        index_data = None

        if self.index_compute:
            ddd_sst = ddd.get('SST')
            if ddd_sst is None:
                raise ValueError("SST data not found in downloaded datasets for clustering.")
            indices_dataset = self.calc_index(self.index_compute, ddd_sst)
            indices_dataset_sel_years = indices_dataset.where(indices_dataset['T'].dt.year.isin(unique_years), drop=True)
            data_scaled, index_data = self.arrange_indices_for_som(indices_dataset_sel_years)
        else:
            if self.multivariateEOF:
                data_vars = [ddd[var].rename({"X": "lon", "Y": "lat"}) for var in ddd]
                combined_data = xr.concat([d.to_array().squeeze() for d in data_vars], dim='variable').stack(feature=('variable', 'lon', 'lat'))
                model = xe.single.EOF(n_modes=100)
                model.fit(combined_data, dim='T')
                explained_var = model.explained_variance_ratio()
                n_modes = np.where(explained_var.cumsum() >= self.eof_explained_var)[0][0] + 1
                scores = model.scores().isel(mode=slice(0, n_modes))
                scores = scores.where(scores['T'].dt.year.isin(unique_years), drop=True)
                
                df_final = pd.DataFrame()
                scores = scores.assign_coords(year=('T', scores['T'].dt.year.data), month_name=('T', scores['T'].dt.strftime('%b').data))
                df = scores.to_dataframe(name='value').reset_index()
                df['mode_month'] = df['mode'].apply(lambda m: f"mode{m:02d}") + "_" + df['month_name']
                df_pivot = df.pivot(index='year', columns='mode_month', values='value')
                month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                df_pivot = df_pivot.reindex(columns=sorted(df_pivot.columns, key=lambda x: (month_order.index(x.split('_')[1]), x)))
                df_final = df_pivot
                data_scaled = np.array(df_final, dtype=np.float64)
                index_data = df_final.index
            else:
                df_final = pd.DataFrame()
                for var in ddd:
                    scores = self.compute_eofs(ddd[var])
                    scores = scores.where(scores['T'].dt.year.isin(unique_years), drop=True)
                    scores = scores.assign_coords(year=('T', scores['T'].dt.year.data), month_name=('T', scores['T'].dt.strftime('%b').data))
                    df = scores.to_dataframe(name='value').reset_index()
                    df['mode_month'] = df['mode'].apply(lambda m: f"mode{m:02d}") + "_" + df['month_name'] + f"_{var}"
                    df_pivot = df.pivot(index='year', columns='mode_month', values='value')
                    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    df_pivot = df_pivot.reindex(columns=sorted(df_pivot.columns, key=lambda x: (month_order.index(x.split('_')[1]), x)))
                    df_final = pd.concat([df_final, df_pivot], axis=1)
                data_scaled = np.array(df_final, dtype=np.float64)
                index_data = df_final.index
        
        if np.isnan(data_scaled).any():
            print("NaN values found in data for clustering. Imputing with column mean.")
            col_means = np.nanmean(data_scaled, axis=0)
            nan_indices = np.where(np.isnan(data_scaled))
            data_scaled[nan_indices] = col_means[nan_indices[1]]
            
        return data_scaled, index_data

    def arrange_indices_for_som(self, ds):
        """Arrange climate indices for SOM training."""
        ds = ds.assign(year=('T', ds['T'].dt.year.data), month_name=('T', ds['T'].dt.strftime('%b').data))
        
        def pivot_var(var_name):
            df = ds[[var_name, 'year', 'month_name']].to_dataframe().reset_index()
            df_pivot = df.pivot(index='year', columns='month_name', values=var_name)
            df_pivot = df_pivot.reindex(columns=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            df_pivot.columns = [f"{var_name}_{month}" for month in df_pivot.columns]
            return df_pivot
        
        vars_order = self.index_compute or []
        df_vars = {v: pivot_var(v) for v in vars_order}
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ordered_columns = [f"{v}_{m}" for m in months for v in vars_order]
        
        df_final = pd.concat(df_vars.values(), axis=1)[ordered_columns]
        return np.array(df_final, dtype=np.float64), df_final.index

    def identify_analogs_bmu(self, target_year, bmu_dict):
        """Identify analog years based on SOM Best Matching Units (BMUs)."""
        radius = self.radius or 1.0
        neuron_years = defaultdict(list)
        for year, coords in bmu_dict.items():
            neuron_years[coords].append(year)
        
        target_coords = bmu_dict.get(target_year)
        if not target_coords:
            return []
        
        analogs = [year for year in neuron_years[target_coords] if year != target_year]
        if not analogs:
            for year, coords in bmu_dict.items():
                if year == target_year:
                    continue
                dist = np.sqrt((coords[0] - target_coords[0])**2 + (coords[1] - target_coords[1])**2)
                if dist <= radius:
                    analogs.append(year)
        
        return analogs

    def SOM(self, predictant, itrain, ireference_year):
        """Identify similar years using Self-Organizing Maps (SOM)."""
        predictant = predictant.copy()
        predictant['T'] = predictant['T'].astype('datetime64[ns]')
        predictant_ = xr.concat([predictant.isel(T=itrain), predictant.isel(T=ireference_year)], dim="T")
        unique_years = np.unique(predictant_['T'].dt.year)
        reference_year = int(np.unique(predictant.isel(T=ireference_year)['T'].dt.year))

        data_scaled, index_data = self._prepare_data_for_clustering(unique_years, reference_year)

        som_grid_size = (4, 4) if set(self.some_grid_size) == {None} else self.some_grid_size
        som = MiniSom(
            x=som_grid_size[0],
            y=som_grid_size[1],
            input_len=data_scaled.shape[1],
            sigma=self.some_sigma,
            learning_rate=self.some_learning_rate,
            neighborhood_function=self.some_neighborhood_function,
            random_seed=42
        )
        som.random_weights_init(data_scaled)
        som.train_random(data=data_scaled, num_iteration=self.some_num_iteration)
        
        bmu_coords = [som.winner(x) for x in data_scaled]
        bmu_dict = {year: coords for year, coords in zip(index_data, bmu_coords)}
        similar_years = self.identify_analogs_bmu(reference_year, bmu_dict)
        print(f"Similar years for {reference_year}: {similar_years}")
        return np.array(similar_years)

    def Corr_Based(self, predictant, itrain, ireference_year):
        """Identify similar years using correlation-based analog method."""
        _, ddd = self.download_and_process()
        ddd = ddd.get(self.predictor_vars[0]['variable'])
        if ddd is None:
            raise ValueError(f"Variable {self.predictor_vars[0]['variable']} not found in downloaded data.")
        
        predictant = predictant.copy()
        predictant['T'] = predictant['T'].astype('datetime64[ns]')
        reference_year = np.unique(predictant['T'].dt.year)[ireference_year][0]
        sst_reference = ddd.sel(T=str(reference_year))
        
        predictant_ = xr.concat([predictant.isel(T=itrain), predictant.isel(T=ireference_year)], dim="T")
        unique_years = np.unique(predictant_.isel(T=itrain)['T'].dt.year)
        
        correlations = []
        for year in unique_years:
            tmp = ddd.sel(T=str(year))
            tmp['T'] = sst_reference['T']
            correlation = xr.corr(tmp, sst_reference, dim="T").compute()
            correlations.append(correlation)
        
        similar = xr.concat(correlations, dim='T').assign_coords(T=unique_years)
        similar = xr.where(similar > 0.6, 1, 0).sum(dim=["X", "Y"])
        similar = similar.sortby(similar, ascending=False)
        top_3 = similar.isel(T=slice(3))
        similar_years = top_3['T'].to_numpy()
        print(f"Similar years for {reference_year}: {similar_years}")
        return similar_years

    def Pca_Based(self, predictant, itrain, ireference_year):
        """Identify similar years using PCA-based analog method."""
        _, ddd = self.download_and_process()
        ddd = ddd.get(self.predictor_vars[0]['variable'])
        if ddd is None:
            raise ValueError(f"Variable {self.predictor_vars[0]['variable']} not found in downloaded data.")
        
        predictor_ = ddd.fillna(ddd.groupby("T.month").mean("T", skipna=True))
        predictor_detrend = sig.detrend(predictor_, axis=0)
        ddd = xr.DataArray(predictor_detrend, dims=predictor_.dims, coords=predictor_.coords)
        
        eof = xe.single.EOF(n_modes=50)
        eof.fit(ddd.fillna(ddd.mean(dim="T", skipna=True)), dim="T")
        scores = eof.scores()
        
        predictant = predictant.copy()
        predictant['T'] = predictant['T'].astype('datetime64[ns]')
        reference_year = np.unique(predictant['T'].dt.year)[ireference_year][0]
        sst_ref = scores.sel(T=str(reference_year)).stack(score=('mode', 'T'))
        
        unique_years = np.unique(predictant.isel(T=itrain)['T'].dt.year)
        correlations = []
        for year in unique_years:
            tmp = scores.sel(T=str(year)).stack(score=('mode', 'T'))
            correlation = xr.corr(tmp, sst_ref, dim="score").compute()
            correlations.append(correlation)
        
        similar = xr.concat(correlations, dim='T').assign_coords(T=unique_years)
        similar = similar.sortby(similar, ascending=False)
        top_3 = similar.isel(T=slice(3))
        similar_years = top_3['T'].to_numpy()
        print(f"Similar years for {reference_year}: {similar_years}")
        return similar_years

    def KMeans_Based(self, predictant, itrain, ireference_year):
        """Identify similar years using K-Means clustering."""
        predictant = predictant.copy()
        predictant['T'] = predictant['T'].astype('datetime64[ns]')
        predictant_ = xr.concat([predictant.isel(T=itrain), predictant.isel(T=ireference_year)], dim="T")
        unique_years = np.unique(predictant_['T'].dt.year)
        reference_year = int(np.unique(predictant.isel(T=ireference_year)['T'].dt.year))

        data_scaled, index_data = self._prepare_data_for_clustering(unique_years, reference_year)

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(data_scaled)

        target_year_idx = np.where(index_data == reference_year)[0]
        if len(target_year_idx) == 0:
            print(f"Target year {reference_year} not found in index data.")
            return np.array([])
        target_cluster_label = cluster_labels[target_year_idx[0]]

        similar_years_indices = np.where(cluster_labels == target_cluster_label)[0]
        similar_years = [index_data[i] for i in similar_years_indices if index_data[i] != reference_year]
        
        print(f"Similar years for {reference_year} (K-Means): {similar_years}")
        return np.array(similar_years)

    def Agglomerative_Based(self, predictant, itrain, ireference_year):
        """Identify similar years using Agglomerative Clustering."""
        predictant = predictant.copy()
        predictant['T'] = predictant['T'].astype('datetime64[ns]')
        predictant_ = xr.concat([predictant.isel(T=itrain), predictant.isel(T=ireference_year)], dim="T")
        unique_years = np.unique(predictant_['T'].dt.year)
        reference_year = int(np.unique(predictant.isel(T=ireference_year)['T'].dt.year))

        data_scaled, index_data = self._prepare_data_for_clustering(unique_years, reference_year)

        agg_clustering = AgglomerativeClustering(n_clusters=self.n_clusters, metric=self.affinity, linkage=self.linkage)
        cluster_labels = agg_clustering.fit_predict(data_scaled)

        target_year_idx = np.where(index_data == reference_year)[0]
        if len(target_year_idx) == 0:
            print(f"Target year {reference_year} not found in index data.")
            return np.array([])
        target_cluster_label = cluster_labels[target_year_idx[0]]

        similar_years_indices = np.where(cluster_labels == target_cluster_label)[0]
        similar_years = [index_data[i] for i in similar_years_indices if index_data[i] != reference_year]

        print(f"Similar years for {reference_year} (Agglomerative): {similar_years}")
        return np.array(similar_years)

    def download_and_process(self):
        """Download and process reanalysis and forecast data.

        Combines reanalysis and forecast data, applies standardization or anomaly calculation,
        and optionally applies a rolling mean.

        Returns
        -------
        data_var_concatenated : dict
            Dictionary of concatenated, processed datasets.
        data_var_shifted : dict
            Dictionary of time-shifted, processed datasets.
        """
        lead_time = [1, 2, 3, 4, 5] if self.lead_time is None else self.lead_time
        month_of_initialization = (datetime.now() - relativedelta(months=1)).month if self.month_of_initialization is None else self.month_of_initialization
        
        sst_hist = self.download_reanalysis(force_download=False)
        sst_hdcst, sst_for = self.download_models(force_download=False)
        variables = [item['variable'] for item in self.predictor_vars]
        print(f"Processing variables: {variables}")
        data_var_concatenated = {}
        data_var_shifted = {}

        for var in variables:
            if var not in sst_hist or var not in sst_for:
                print(f"Skipping {var}: data not available.")
                continue  

            sst_hist_ = sst_hist[var]
            sst_hdcst_ = sst_hdcst[var]
            sst_for_ = sst_for[var]

            # process hindcast data
            hindcast = []
            for i in sst_hdcst_['T']:
                sst_hdcst_i = sst_hdcst_.sel(T=i)
                base_times = pd.Timestamp(sst_hdcst_i['T'].values)
                new_times = [base_times + pd.DateOffset(months=int(m)) for m in sst_hdcst_i['forecastMonth'].values]
                sst_hdcst_i = sst_hdcst_i.assign_coords(forecastMonth=("forecastMonth", new_times))
                sst_hdcst_i = sst_hdcst_i.drop_vars('T', errors='ignore').squeeze().rename({'forecastMonth': 'T'})
                hindcast.append(sst_hdcst_i)
            sst_hdcst_ = xr.concat(hindcast, dim='T')          
            sst_hdcst_ = sst_hdcst_.interp(Y=sst_hist_.Y, X=sst_hist_.X, method="linear", kwargs={"fill_value": "extrapolate"})

            # process forecast data
            sst_for_ = sst_for_.interp(Y=sst_hist_.Y, X=sst_hist_.X, method="linear", kwargs={"fill_value": "extrapolate"})
            if (isinstance(sst_for_['T'].values, np.ndarray)):
                base_time = pd.Timestamp(sst_for_['T'].values[-1])
            else:
                base_time = pd.Timestamp(sst_for_['T'].values)
            new_times = [base_time + pd.DateOffset(months=int(m)) for m in sst_for_['forecastMonth'].values]
            sst_for_ = sst_for_.assign_coords(forecastMonth=("forecastMonth", new_times))
            sst_for_ = sst_for_.drop_vars('T', errors='ignore').squeeze().rename({'forecastMonth': 'T'})

            # Correct forecast systematic bias
            sst_for_bias_corrected = []
            for m in lead_time:
                sst_hist_mean = sst_hist_.sel(T=slice(str(self.clim_year_start), str(self.clim_year_end))).where(sst_hist_['T'].dt.month.isin(m), drop=True).mean(dim="T",skipna=True)
                sst_hdcst_mean = sst_hdcst_.sel(T=slice(str(self.clim_year_start), str(self.clim_year_end))).where(sst_hdcst_['T'].dt.month.isin(m), drop=True).mean(dim="T",skipna=True)
                if var in ["TMIN", "TEMP", "TMAX", "SST", "HUSS_1000", "HUSS_925", "HUSS_850", "SLP", "UGRD10", "VGRD10", "UGRD_1000", "UGRD_925", "UGRD_850", "VGRD_1000", "VGRD_925", "VGRD_850"]:
                    sst_for_bias_corrected.append(sst_for_.where(sst_for_['T'].dt.month.isin(m), drop=True) - sst_hdcst_mean + sst_hist_mean)
                if var in ["PRCP", "DSWR", "DLWR", "NOLR"]:
                    sst_for_bias_corrected.append(sst_for_.where(sst_for_['T'].dt.month.isin(m), drop=True) * sst_hist_mean / sst_hdcst_mean)
            sst_for_ = xr.concat(sst_for_bias_corrected, dim='T')

            # Concatenate hindcast and forecast data
            sst_hist_ = sst_hist_.sel(T=slice(f"{self.year_start}-01", f"{base_time.year}-{base_time.month:02d}"))
            concatenated_ds = xr.concat([sst_hist_, sst_for_], dim='T')

            if isinstance(self.rolling, int) and self.rolling > 1:
                print(f"Applied rolling mean with window size {self.rolling} for {var}.")
                concatenated_ds = concatenated_ds.rolling(T=self.rolling, center=False, min_periods=self.rolling).mean()

            print(f"Standardized or computed anomalies for {var}.")
            concatenated_ds_st = self.standardize_timeseries(concatenated_ds) if self.standardize else self.anomaly_timeseries(concatenated_ds)


            first_year = concatenated_ds_st.T.dt.year[0].item()
            last_month = concatenated_ds_st.T.dt.month[-1].item() + 1
            start_date = pd.to_datetime(f"{first_year}-{last_month:02d}")
            concatenated_ds_st = concatenated_ds_st.sel(T=slice(start_date, None))
            new_time = pd.DatetimeIndex([pd.to_datetime(f"{first_year+1}-01-01") + pd.DateOffset(months=t) for t in range(len(concatenated_ds_st['T']))])
            # ds_shifted = concatenated_ds_st.assign_coords(T=new_time, month=('T', new_time.month))
            ds_shifted = concatenated_ds_st.assign_coords(T=new_time)
    
            # ds_shifted = ds_shifted.astype({f"{var.lower()}": "float32"})
            # concatenated_ds_st = concatenated_ds_st.astype({f"{var.lower()}": "float32"})
            data_var_shifted[var] = ds_shifted.to_array().drop_vars(['variable'], errors='ignore').squeeze()
            data_var_concatenated[var] = concatenated_ds_st.to_array().drop_vars(['variable'], errors='ignore').squeeze()
        return data_var_concatenated, data_var_shifted


    @staticmethod
    def calculate_tercile_probabilities(best_guess, error_variance, first_tercile, second_tercile, dof):
        """Calculate tercile probabilities using Student's t-distribution."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time), dtype=np.float64)
        
        if np.all(np.isnan(best_guess)) or np.all(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        
        error_std = np.sqrt(error_variance)
        first_t = (first_tercile - best_guess) / error_std
        second_t = (second_tercile - best_guess) / error_std
        
        pred_prob[0, :] = stats.t.cdf(first_t, df=dof)
        pred_prob[1, :] = stats.t.cdf(second_t, df=dof) - stats.t.cdf(first_t, df=dof)
        pred_prob[2, :] = 1 - stats.t.cdf(second_t, df=dof)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_gamma(best_guess, error_variance, T1, T2):
        """Calculate tercile probabilities using Gamma distribution."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time), dtype=np.float64)
        
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        
        best_guess = np.asarray(best_guess, dtype=np.float64)
        error_variance = np.asarray(error_variance, dtype=np.float64)
        T1 = np.asarray(T1, dtype=np.float64)
        T2 = np.asarray(T2, dtype=np.float64)
        
        alpha = (best_guess**2) / error_variance
        theta = error_variance / best_guess
        cdf_t1 = gamma.cdf(T1, a=alpha, scale=theta)
        cdf_t2 = gamma.cdf(T2, a=alpha, scale=theta)
        
        pred_prob[0, :] = cdf_t1
        pred_prob[1, :] = cdf_t2 - cdf_t1
        pred_prob[2, :] = 1.0 - cdf_t2
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_nonparametric(best_guess, error_samples, first_tercile, second_tercile):
        """Calculate tercile probabilities using a non-parametric method."""
        n_time = len(best_guess)
        pred_prob = np.full((3, n_time), np.nan, dtype=np.float64)
        
        for t in range(n_time):
            if np.isnan(best_guess[t]):
                continue
            dist = best_guess[t] + error_samples
            dist = dist[np.isfinite(dist)]
            if len(dist) == 0:
                continue
            p_below = np.mean(dist < first_tercile)
            p_between = np.mean((dist >= first_tercile) & (dist < second_tercile))
            p_above = 1.0 - (p_below + p_between)
            pred_prob[0, t] = p_below
            pred_prob[1, t] = p_between
            pred_prob[2, t] = p_above
        
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_normal(best_guess, error_variance, first_tercile, second_tercile):
        """Calculate tercile probabilities using Normal distribution."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time), dtype=np.float64)
        
        if np.all(np.isnan(best_guess)) or np.all(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        
        error_std = np.sqrt(error_variance)
        pred_prob[0, :] = stats.norm.cdf(first_tercile, loc=best_guess, scale=error_std)
        pred_prob[1, :] = stats.norm.cdf(second_tercile, loc=best_guess, scale=error_std) - \
                          stats.norm.cdf(first_tercile, loc=best_guess, scale=error_std)
        pred_prob[2, :] = 1 - stats.norm.cdf(second_tercile, loc=best_guess, scale=error_std)
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_lognormal(best_guess, error_variance, first_tercile, second_tercile):
        """Calculate tercile probabilities using Lognormal distribution."""
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time), dtype=np.float64)
        
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        
        sigma = np.sqrt(np.log(1 + error_variance / (best_guess**2)))
        mu = np.log(best_guess) - sigma**2 / 2
        pred_prob[0, :] = lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[1, :] = lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu)) - \
                          lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[2, :] = 1 - lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu))
        return pred_prob

    def compute_model(self, predictant, itrain, itest):
        """Compute deterministic hindcast using the specified analog method."""
        method_map = {
            "som": self.SOM,
            "cor_based": self.Corr_Based,
            "pca_based": self.Pca_Based,
            "kmeans_based": self.KMeans_Based,
            "agglomerative_based": self.Agglomerative_Based
        }
        if self.method_analog not in method_map:
            raise ValueError(f"Invalid analog method: {self.method_analog}. Choose 'som', 'cor_based', 'pca_based', 'kmeans_based', or 'agglomerative_based'.")

        similar_years = method_map[self.method_analog](predictant, itrain, itest)
        predictant = predictant.copy()
        predictant['T'] = predictant['T'].astype('datetime64[ns]')
        
        sim_obs = [predictant.sel(T=str(year)) for year in similar_years if year in predictant['T'].dt.year.values]
        if not sim_obs:
            raise ValueError("No valid similar years found for hindcast.")
        
        hindcast_det = xr.concat(sim_obs, dim="T").mean(dim="T").expand_dims({'T': predictant.isel(T=itest)['T'].values})
        return hindcast_det

    def compute_prob(self, predictant, clim_year_start, clim_year_end, hindcast_det):
        """Compute tercile probabilities for the hindcast."""
        index_start = predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        T1 = terciles.isel(quantile=0).drop_vars('quantile')
        T2 = terciles.isel(quantile=1).drop_vars('quantile')
        
        calc_func_map = {
            "t": (self.calculate_tercile_probabilities, {"dof": len(predictant.get_index("T")) - 2}),
            "gamma": (self.calculate_tercile_probabilities_gamma, {}),
            "normal": (self.calculate_tercile_probabilities_normal, {}),
            "lognormal": (self.calculate_tercile_probabilities_lognormal, {}),
            "nonparam": (self.calculate_tercile_probabilities_nonparametric, {})
        }
        
        if self.dist_method not in calc_func_map:
            raise ValueError(f"Invalid dist_method: {self.dist_method}. Choose 't', 'gamma', 'normal', 'lognormal', or 'nonparam'.")
        
        calc_func, kwargs = calc_func_map[self.dist_method]
        error_input = (predictant - hindcast_det).var(dim='T') if self.dist_method != "nonparam" else (predictant - hindcast_det).rename({'T': 'S'})
        input_core_dims = [('T',), (), (), ()] if self.dist_method != "nonparam" else [('T',), ('S',), (), ()]
        
        hindcast_prob = xr.apply_ufunc(
            calc_func,
            hindcast_det,
            error_input,
            T1,
            T2,
            input_core_dims=input_core_dims,
            vectorize=True,
            dask='parallelized',
            output_core_dims=[('probability', 'T')],
            output_dtypes=[np.float64],
            dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True},
            kwargs=kwargs
        )
        
        return hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA'])).transpose('probability', 'T', 'Y', 'X')

    def forecast(self, predictant, clim_year_start, clim_year_end, hindcast_det):
        """Generate deterministic and probabilistic forecasts for the target year."""
        predictant = predictant.copy()
        predictant['T'] = predictant['T'].astype('datetime64[ns]')
        reference_year = self.year_forecast
        unique_years = np.append(np.unique(predictant['T'].dt.year), reference_year)
        
        method_map = {
            "som": self.SOM,
            "cor_based": self.Corr_Based,
            "pca_based": self.Pca_Based,
            "kmeans_based": self.KMeans_Based,
            "agglomerative_based": self.Agglomerative_Based
        }
        if self.method_analog not in method_map:
            raise ValueError(f"Invalid analog method: {self.method_analog}. Choose 'som', 'cor_based', 'pca_based', 'kmeans_based', or 'agglomerative_based'.")

        itrain = [i for i, year in enumerate(predictant['T'].dt.year.values) if year in unique_years and year != reference_year]
        ireference_year = [i for i, year in enumerate(predictant['T'].dt.year.values) if year == reference_year]
        if not ireference_year:
            ireference_year = [len(predictant['T']) - 1]
        
        _, ddd = self.download_and_process()
        similar_years = method_map[self.method_analog](predictant, itrain, ireference_year)
        
        sim_obs = [predictant.sel(T=str(year)) for year in similar_years if year in predictant['T'].dt.year.values]
        if not sim_obs:
            raise ValueError("No valid similar years found for forecast.")
        
        forecast_det = xr.concat(sim_obs, dim="T").mean(dim="T")
        T_value = predictant.isel(T=0).coords['T'].values
        month = T_value.astype('datetime64[M]').astype(int) % 12 + 1
        new_T_value = np.datetime64(f"{reference_year}-{month:02d}-01")
        forecast_det = forecast_det.expand_dims({'T': [new_T_value]})
        forecast_det['T'] = forecast_det['T'].astype('datetime64[ns]')
        
        index_start = predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = predictant.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')
        T1 = terciles.isel(quantile=0).drop_vars('quantile')
        T2 = terciles.isel(quantile=1).drop_vars('quantile')
        
        calc_func_map = {
            "t": (self.calculate_tercile_probabilities, {"dof": len(predictant.get_index("T")) - 2}),
            "gamma": (self.calculate_tercile_probabilities_gamma, {}),
            "normal": (self.calculate_tercile_probabilities_normal, {}),
            "lognormal": (self.calculate_tercile_probabilities_lognormal, {}),
            "nonparam": (self.calculate_tercile_probabilities_nonparametric, {})
        }
        
        calc_func, kwargs = calc_func_map[self.dist_method]
        error_input = (predictant - hindcast_det).var(dim='T') if self.dist_method != "nonparam" else (predictant - hindcast_det).rename({'T': 'S'})
        input_core_dims = [('T',), (), (), ()] if self.dist_method != "nonparam" else [('T',), ('S',), (), ()]
        
        forecast_prob = xr.apply_ufunc(
            calc_func,
            forecast_det,
            error_input,
            T1,
            T2,
            input_core_dims=input_core_dims,
            vectorize=True,
            dask='parallelized',
            output_core_dims=[('probability', 'T')],
            output_dtypes=[np.float64],
            dask_gufunc_kwargs={'output_sizes': {'probability': 3}, 'allow_rechunk': True},
            kwargs=kwargs
        )
        
        forecast_prob = forecast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA'])).transpose('probability', 'T', 'Y', 'X')
        print(f"Similar years for {reference_year}: {similar_years}")
        return ddd, similar_years, forecast_det, forecast_prob

    def composite_plot(self, predictant, clim_year_start, clim_year_end, hindcast_det, plot_predictor=True, variable="SST"):
        """Create composite plots of predictors or predictands."""
        clim_slice = slice(str(clim_year_start), str(clim_year_end))
        clim_mean = predictant.sel(T=clim_slice).mean(dim='T', skipna=True)
        similar_years, result_, _ = self.forecast(predictant, clim_year_start, clim_year_end, hindcast_det)
        result_ = result_.drop_vars('T', errors='ignore').squeeze()
        reference_year = self.year_forecast
        _, ddd = self.download_and_process()

        ddd = ddd[variable]
        
        sim_all = []
        tmp = ddd.sel(T=str(reference_year))
        months = list(tmp['T'].dt.month.values)
        tmp['T'] = months
        sim_all.append(tmp)
        
        sim_ = []
        pred_rain = []
        for year in np.array([str(i) for i in similar_years]):
            tmp = ddd.sel(T=year)
            months = list(tmp['T'].dt.month.values)
            tmp['T'] = months
            sim_.append(tmp)
            pred_rain.append(100 * predictant.sel(T=year) / clim_mean)
        
        sim__ = xr.concat(sim_, dim="year").assign_coords(year=('year', similar_years)).mean(dim="year", skipna=True)
        sim_all.append(sim__)
        sim_all = xr.concat(sim_all, dim="output").assign_coords(output=('output', ['forecast year', 'composite analog']))
        
        pred_rain = xr.concat(pred_rain, dim='T')
        
        if plot_predictor:
            sim_all.plot(
                x="X", y="Y", row="T", col="output",
                figsize=(12, len(months) * 4),
                cbar_kwargs={"shrink": 0.3, "aspect": 50, "pad": 0.05, "label": f"{variable} anomaly", "orientation": "horizontal"},
                robust=True,
                subplot_kws={'projection': ccrs.PlateCarree()}
            )
            for ax in plt.gcf().axes:
                if isinstance(ax, plt.Axes) and hasattr(ax, 'coastlines'):
                    ax.coastlines()
                    ax.gridlines(draw_labels=True)
                    ax.add_feature(cfeature.LAND, edgecolor="black")
                    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
            plt.suptitle(f"SST Composites for {reference_year}", fontsize=14)
            plt.tight_layout()
            plt.show()
        else:
            colors_list = ['#d7191c', '#fdae61', '#ffffbf', '#a6d96a', '#1a9641']
            bounds = [0, 50, 90, 110, 150, 200]
            cmap = ListedColormap(colors_list)
            norm = BoundaryNorm(bounds, cmap.N)
            
            data_var = pred_rain.sel(Y=slice(None, 19.5))
            n_times = len(data_var['T'])
            n_cols = 2
            n_rows = (n_times + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(
                n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4),
                subplot_kw={'projection': ccrs.PlateCarree()}
            )
            axes = np.ravel(axes)
            
            for i, t in enumerate(data_var['T'].values):
                ax = axes[i]
                data = data_var.sel(T=t).squeeze()
                im = ax.pcolormesh(
                    data['X'], data['Y'], data, cmap=cmap, norm=norm,
                    transform=ccrs.PlateCarree()
                )
                ax.coastlines()
                ax.gridlines(draw_labels=True)
                ax.add_feature(cfeature.LAND, edgecolor="black")
                ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
                ax.set_title(f"Season: {str(t)[:10]}")
            
            for j in range(n_times, len(axes)):
                fig.delaxes(axes[j])
            
            cbar = fig.colorbar(im, ax=axes[:n_times], orientation="horizontal", fraction=0.05, pad=0.1)
            cbar.set_label('Precipitation Ratio to Normal (%)')
            cbar.set_ticks([0, 50, 90, 110, 150, 200])
            cbar.ax.set_xticklabels(['0', '50', '90', '110', '150', '200'])
            fig.suptitle("Ratio to Normal Precipitation [%]", fontsize=14)
            plt.tight_layout()
            fig.subplots_adjust(top=0.9, bottom=0.15)
            plt.show()        
        return similar_years