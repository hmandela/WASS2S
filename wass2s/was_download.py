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
import os
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

# Suppress warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.getLogger("cdsapi").setLevel(logging.ERROR)


class WAS_Download:
    def __init__(self):
        """Initialize the WAS_Download class."""
        pass

    def ModelsName(
        self,
        centre={
            "ECMWF_51": "ecmwf",
            "UKMO_604": "ukmo",
            "UKMO_603": "ukmo",
            "METEOFRANCE_8": "meteo_france",
            "DWD_21": "dwd", # month of initialization available for forecast are Jan to Mar
            "DWD_22": "dwd", # month of initialization available for forecast are Apr to __ 
            "CMCC_35": "cmcc",
            # "CMCC_3": "cmcc",
            "NCEP_2": "ncep",
            "JMA_3": "jma",
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
            "NMME_1" : "nmme"
        },
        variables_1={
            "PRCP": "total_precipitation",
            "TEMP": "2m_temperature",
            "UGRD10": "10m_u_component_of_wind",
            "VGRD10": "10m_v_component_of_wind",
            "SST": "sea_surface_temperature",
            "SLP": "mean_sea_level_pressure",
            "DSWR": "surface_solar_radiation_downwards",
            "DLWR": "surface_thermal_radiation_downwards",
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

    def ReanalysisName(
        self,
        centre={"ERA5": "reanalysis ERA5", "NOAA": "NOAA ERSST"},
        variables_1={
            "PRCP": "total_precipitation",
            "TEMP": "2m_temperature",
            "UGRD10": "10m_u_component_of_wind",
            "VGRD10": "10m_v_component_of_wind",
            "SST": "sea_surface_temperature",
            "SLP": "mean_sea_level_pressure",
            "DSWR": "surface_solar_radiation_downwards",
            "DLWR": "surface_thermal_radiation_downwards",
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
        },
    ):
        """
        Generate a dictionary for agro-meteorological observation variables.

        Parameters:
            variables (dict): Mapping of agro variable short names to full names.

        Returns:
            dict: A dictionary mapping agro variables to their corresponding full names.
        """
        return variables

    def WAS_Download_AgroIndicators_(
        self,
        dir_to_save,
        variables,
        year_start,
        year_end,
        area,
        seas=["01", "02", "03"],
        force_download=False,
    ):
        """
        Download agro-meteorological indicators for specified variables, years, and months.

        Parameters:
            dir_to_save (str): Directory to save the downloaded files.
            variables (list): List of shorthand variables to download (e.g., ["AGRO.PRCP", "AGRO.TMAX"]).
            year_start (int): Start year for the data to download.
            year_end (int): End year for the data to download.
            area (list): Bounding box as [North, West, South, East] for clipping.
            seas (list): List of month strings representing the season (e.g., ["01", "02", "03"]- for JanFebMar).
            force_download (bool): If True, forces download even if file exists.
        """
        dir_to_save = Path(dir_to_save)
        os.makedirs(dir_to_save, exist_ok=True)
        days = [f"{day:02}" for day in range(1, 32)]
        version = "1_1"
        variable_mapping = {
            "AGRO.PRCP": ("precipitation_flux", None),
            "AGRO.TMAX": ("2m_temperature", "24_hour_maximum"),
            "AGRO.TEMP": ("2m_temperature", "24_hour_mean"),
            "AGRO.TMIN": ("2m_temperature", "24_hour_minimum"),
        }
        months = [f"{int(m) - 0:02}" for m in seas]
        season = "".join([calendar.month_abbr[int(month)] for month in months])

        for var in variables:
            if var not in variable_mapping:
                print(f"Unknown variable: {var}. Skipping.")
                continue
            combined_datasets = []
            cds_variable, statistic = variable_mapping[var]
            output_path = dir_to_save / f"Obs_{var.split('.')[1]}_{year_start}_{year_end}_{season}.nc"
            if not force_download and os.path.exists(output_path):
                print(f"{output_path} already exists. Skipping download.") 
            else:            
                for year in range(year_start, year_end + 1):
                    zip_file_path = dir_to_save / f"Obs_{var.split('.')[1]}_{year}_{season}.zip"
                
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
                    try:
                        client = cdsapi.Client()
                        print(f"Downloading {cds_variable} ({statistic}) data for {year} and season {season}...")
                        client.retrieve(dataset, request).download(str(zip_file_path))
                        print(f"Downloaded: {zip_file_path}")
                    except Exception as e:
                        print(f"Failed to download {cds_variable} ({statistic}) data for {year}: {e}")
                        continue

                    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                        for netcdf_file_name in zip_ref.namelist():
                            with zip_ref.open(netcdf_file_name) as file:
                                ds = xr.open_dataset(io.BytesIO(file.read()))
                                combined_datasets.append(ds)
                                
                    os.remove(zip_file_path)
                    print(f"Deleted ZIP file: {zip_file_path}")

            if combined_datasets:
                combined_ds = xr.concat(combined_datasets,dim="time")
                if var in ["AGRO.TMIN", "AGRO.TEMP", "AGRO.TMAX"]:
                    combined_ds = combined_ds - 273.15
                    combined_ds = combined_ds.resample(time="YE").mean()
                    combined_ds["time"] = [f"{year}-{months[1]}-01" for year in combined_ds["time"].dt.year.astype(str).values]
                else:
                    combined_ds = combined_ds.resample(time="YE").sum()
                    combined_ds["time"] = [f"{year}-{months[1]}-01" for year in combined_ds["time"].dt.year.astype(str).values]
                combined_ds = combined_ds.rename({"lon": "X", "lat": "Y", "time": "T"})
                combined_ds = combined_ds.isel(Y=slice(None, None, -1))
                # output_path = dir_to_save / f"Obs_{var.split('.')[1]}_{year_start}_{year_end}_{season}.nc"
                combined_ds.to_netcdf(output_path)
                print(f"File downloaded and combined dataset for {var} is saved to {output_path}")

    def WAS_Download_Reanalysis_(
        self,
        dir_to_save,
        center_variable,
        year_start,
        year_end,
        area,
        seas=["01", "02", "03"],
        force_download=False,
    ):
        """
        Download reanalysis data for specified center-variable combinations, years, and months.

        Parameters:
            dir_to_save (str): Directory to save the downloaded files.
            center_variable (list): List of center-variable identifiers (e.g., ["ERA5.PRCP", "MERRA2.TEMP"]).
            year_start (int): Start year for the data to download.
            year_end (int): End year for the data to download.
            area (list): Bounding box as [North, West, South, East] for clipping.
            seas (list): List of month strings representing the season (e.g., ["01", "02", "03"] for JanFebMar).
            force_download (bool): If True, forces download even if file exists.
        """
        center = [item.split(".")[0] for item in center_variable]
        variables = [item.split(".")[1] for item in center_variable]
        centre = {"ERA5": "ERA5", "MERRA2": "MERRA2"}
        variables_1 = {
            "PRCP": "total_precipitation",
            "TEMP": "2m_temperature",
            "TMAX": "maximum_2m_temperature_in_the_last_24_hours",
            "TMIN": "minimum_2m_temperature_in_the_last_24_hours",
            "UGRD10": "10m_u_component_of_wind",
            "VGRD10": "v_component_of_wind",
            "SST": "sea_surface_temperature",
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
        selected_centre = [centre[k] for k in center]
        selected_var = [k for k in variables]
        dir_to_save = Path(dir_to_save)
        os.makedirs(dir_to_save, exist_ok=True)
        months = [f"{int(m) - 0:02}" for m in seas]
        season = "".join([calendar.month_abbr[int(month)] for month in months])

        for cent, k in zip(selected_centre, selected_var):
            output_path = dir_to_save / f"{cent}_{k}_{year_start}_{year_end}_{season}.nc"
            if not force_download and os.path.exists(output_path):
                print(f"{output_path} already exists. Skipping download.") 
            else:
                combined_datasets = []
                for year in range(year_start, year_end + 1):
                    file_path = f"{dir_to_save}/{cent}_{k}_{year}_{season}.nc"
                    try:
                        if k in variables_2:
                            press_level = k.split("_")[1]
                            dataset = "reanalysis-era5-pressure-levels-monthly-means"
                            request = {
                                "product_type": ["monthly_averaged_reanalysis"],
                                "variable": variables_2[k],
                                "pressure_level": press_level,
                                "year": str(year),
                                "month": months,
                                "time": ["00:00"],
                                "data_format": "netcdf",
                                "download_format": "unarchived",
                                "area": area,
                            }
                        else:
                            dataset = "reanalysis-era5-single-levels-monthly-means"
                            request = {
                                "product_type": ["monthly_averaged_reanalysis"],
                                "variable": variables_1[k],
                                "year": str(year),
                                "month": months,
                                "time": ["00:00"],
                                "data_format": "netcdf",
                                "download_format": "unarchived",
                                "area": area,
                            }
                        client = cdsapi.Client()
                        print(f"Downloading {cent} {k} for year {year}...")
                        client.retrieve(dataset, request).download(file_path)
                    except Exception as e:
                        print(f"Failed to download data for {cent} and {k} in year {year}: {e}")
                        continue
                    with xr.open_dataset(file_path) as ds: #ds = xr.open_dataset(file_path)
                        if len(ds.coords) > 4 and k not in ["TMIN", "TEMP", "TMAX", "SST", "PRCP", "UGRD10", "VGRD10"]:
                            ds = ds.drop_vars(["number", "expver", "pressure_level"]).squeeze()
                            ds = ds.isel(latitude=slice(None, None, -1))
                            # ds = ds.assign_coords(valid_time=pd.to_datetime(ds.valid_time.astype(str), format='%Y%m%d'))
                            ds = ds.assign_coords(valid_time=pd.to_datetime(ds.valid_time.astype(str)))
                            ds = ds.rename({"latitude": "lat", "longitude": "lon", "valid_time": "time"})
                        elif len(ds.coords) > 4:
                            ds = ds.drop_vars(["number", "expver"]).squeeze()
                            ds = ds.isel(latitude=slice(None, None, -1))
                            # ds = ds.assign_coords(valid_time=pd.to_datetime(ds.valid_time.astype(str), format='%Y%m%d'))
                            ds = ds.assign_coords(valid_time=pd.to_datetime(ds.valid_time.astype(str)))
                            ds = ds.rename({"latitude": "lat", "longitude": "lon", "valid_time": "time"})
                        ds = ds.load()
                        combined_datasets.append(ds)
                    os.remove(file_path)
                    print(f"Deleted yearly file: {file_path}")
    
                if combined_datasets:
                    combined_ds = xr.concat(combined_datasets, dim="time")
                    if k in ["TMIN", "TEMP", "TMAX", "SST"]:
                        combined_ds = combined_ds - 273.15
                        combined_ds = combined_ds.resample(time="YE").mean()
                        combined_ds["time"] = [f"{year}-{months[1]}-01" for year in combined_ds["time"].dt.year.astype(str).values]
                    elif k == "PRCP":
                        combined_ds = 90 * 1000 * combined_ds.resample(time="YE").sum()
                        combined_ds["time"] = [f"{year}-{months[1]}-01" for year in combined_ds["time"].dt.year.astype(str).values]
                    else:
                        combined_ds = combined_ds.resample(time="YE").mean()
                        combined_ds["time"] = [f"{year}-{months[1]}-01" for year in combined_ds["time"].dt.year.astype(str).values]
                    combined_ds = combined_ds.rename({"lon": "X", "lat": "Y", "time": "T"})
                    combined_ds.to_netcdf(output_path)
                    print(f"Download finished, combined dataset for {cent} {k} to {output_path}")

    # def download_nmme_txt_with_progress(self, url, file_path, chunk_size=1024):
    #     file_path = Path(file_path)
    #     response = requests.get(url, stream=True)
    #     total_size = int(response.headers.get('content-length', 0))
        
    #     with open(file_path, "wb") as f, tqdm(
    #         total=total_size, unit="B", unit_scale=True, desc=file_path.name
    #     ) as progress:
    #         for data in response.iter_content(chunk_size):
    #             progress.update(len(data))
    #             f.write(data)

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
        from datetime import datetime
        
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
                        year, pot_months = t_str.split('-')
                        if '/' in pot_months:
                            start_str, end_str = pot_months.split("/")
                            start_month = int(start_str)
                            end_month = int(end_str[0:2])
                            if start_month <= end_month:
                                months = list(range(start_month, end_month + 1))
                            else:
                                # Wrap around: from start_month to December, then January to end_month
                                months = list(range(start_month, 13)) + list(range(1, end_month + 1))
                            month = months[1]
                            days_in_mon = self.days_in_month(int(year), months[0]) + self.days_in_month(int(year), months[1]) + self.days_in_month(int(year), months[2])
                        else:
                            month = int(pot_months[0:2])
                            days_in_mon = self.days_in_month(int(year), month)
                        
                        days_in_month_values.append(days_in_mon)
                        times.append(datetime(int(year), int(month), 1))
                        
                        #### Retrieve init start
                        start_str = lines[i].split('cpt:S=')[1].split()[0]
                        yearstart, monthstart, daystart = start_str.split('-')
                        times_start.append(datetime(int(yearstart), int(monthstart), 1))
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
            # attrs={
            #     'units': 'mm/day',
            #     'long_name': 'precipitation'
            # }
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
    ):
        """
        Download seasonal forecast model data for specified center-variable combinations, initialization month, lead times, and years.

        Parameters:
            dir_to_save (str): Directory to save the downloaded files.
            center_variable (list): List of center-variable identifiers (e.g., ["ECMWF_51.PRCP", "UKMO_602.TEMP"]).
            month_of_initialization (int): Initialization month as an integer (1-12).
            lead_time (list): List of lead times in months.
            year_start_hindcast (int): Start year for hindcast data.
            year_end_hindcast (int): End year for hindcast data.
            area (list): Bounding box as [North, West, South, East] for clipping.
            year_forecast (int, optional): Forecast year if downloading forecast data. Defaults to None.
            ensemble mean (str,optional): it's can be median, mean or None. Defaults to None. 
            force_download (bool): If True, forces download even if file exists.
        """
        years = (
            [str(year) for year in range(year_start_hindcast, year_end_hindcast + 1)]
            if year_forecast is None
            else [str(year_forecast)]
        )

        center = [item.split(".")[0] for item in center_variable]
        variables = [item.split(".")[1] for item in center_variable]

        centre = {
            "ECMWF_51": "ecmwf",
            "UKMO_604": "ukmo", # month of initialization available for forecast are Apr to __
            "UKMO_603": "ukmo", # month of initialization available for forecast are Jan to Mar
            "METEOFRANCE_8": "meteo_france",
            "DWD_21": "dwd", # month of initialization available for forecast are Jan to Mar
            "DWD_22": "dwd", # month of initialization available for forecast are Apr to __ 
            "CMCC_35": "cmcc",
            # "CMCC_3": "cmcc",
            "NCEP_2": "ncep",
            "JMA_3": "jma",
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
            "NMME_1" : "nmme"
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

        system = {
            "ECMWF_51": "51",
            "UKMO_604": "604",
            "UKMO_603": "603",
            "METEOFRANCE_8": "8",
            "DWD_21": "21",
            "DWD_22": "22",
            # "DWD_2": "2",
            "CMCC_35": "35",
            # "CMCC_3": "3",
            "NCEP_2": "2",
            "JMA_3": "3",
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
            "NMME_1" : "1"

        }
        
        nmme = ["cfsv2", "cmc1", "cmc2", "gfdl",  "nasa", "ncar_ccsm4", "nmme"]
        
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

            if cent in nmme: #### Reconsider an option to download other variable than PRCP, TEMP, and SST from IRIDL "code already available"      

                file_path = f"{dir_to_save}/{file_prefix}_{cent.replace('_', '')}{syst}_{k}_{abb_mont_ini}Ic_{season}_{lead_time[0]}.nc"
                init_str = f"{abb_mont_ini}ic"
                tag = "fcst" if year_forecast else "hcst"
                k = "precip" if k=="PRCP" else k
                k = "tmp2m" if k=="TEMP" else k
                k = "sst" if k=="SST" else k
                if not force_download and os.path.exists(file_path):
                    print(f"{file_path} already exists. Skipping download.")
                    store_file_path[f"{cent}_{syst}"] = file_path                   
                else:
                    # Choose base URL depending on forecast/hindcast and temporal resolution.
                    if len(lead_time) == 3:
                        # Build lead time string using min and max lead time values.
                        lead_str = f"{season_months[0]}-{season_months[-1]}"
                        
                        if year_forecast:
                            base_url = "https://ftp.cpc.ncep.noaa.gov/International/nmme/seasonal_nmme_forecast_in_cpt_format/"
                            year_range = f"{year_forecast}-{year_forecast}"
                        else:
                            base_url = "https://ftp.cpc.ncep.noaa.gov/International/nmme/seasonal_nmme_hindcast_in_cpt_format/"
                            year_range = f"{1991}-{2020}"
                        file_name = f"{cent}_{k}_{tag}_{init_str}_{lead_str}_{year_range}.txt"
                        full_url = base_url + file_name
                        file_txt_path = dir_to_save / file_name
                        if os.path.exists(file_txt_path):
                            da, number_day, times_start = self.parse_cpt_data_optimized(file_txt_path)
                        else:
                            self.download_nmme_txt_with_progress(full_url, file_txt_path)
                            da, number_day, times_start = self.parse_cpt_data_optimized(file_txt_path)

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
                        ds.to_netcdf(file_path)
                        print(f"Download finished for {cent} {syst} {k} to {file_path}")
                        ds.close()
                        store_file_path[f"{cent}_{syst}"] = file_path
                        
                    else:
                        if year_forecast:
                            base_url = "https://ftp.cpc.ncep.noaa.gov/International/nmme/monthly_nmme_forecast_in_cpt_format/"
                            year_range = f"{year_forecast}"
                        else:
                            base_url = "https://ftp.cpc.ncep.noaa.gov/International/nmme/monthly_nmme_hindcast_in_cpt_format/"
                            year_range = f"{1991}"
                        all_da = []
                        for i in season_months:
                            file_name = f"{cent}_{k}_{tag}_{init_str}_{i}_{year_range}.txt"
                            full_url = base_url + file_name
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
                        ds.to_netcdf(file_path)
                        print(f"Download finished for {cent} {syst} {k} to {file_path}")
                        ds.close()
                        store_file_path[f"{cent}_{syst}"] = file_path                          
            else:
                file_path = f"{dir_to_save}/{file_prefix}_{cent.replace('_', '')}{syst}_{k}_{abb_mont_ini}Ic_{season}_{lead_time[0]}.nc"
                if not force_download and os.path.exists(file_path):
                    print(f"{file_path} already exists. Skipping download.")
                    store_file_path[f"{cent}_{syst}"] = file_path
                else:                
                    try:
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
                                "leadtime_month": lead_time,
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
                                "leadtime_month": lead_time,
                                "data_format": "netcdf",
                                "area": area,
                            }
        
                        client = cdsapi.Client()
                        client.retrieve(dataset, request).download(file_path)
                        print(f"Downloaded: {file_path}")
    
        
                        # Load the NetCDF file and apply area selection if specified
                        ds = xr.open_dataset(file_path)
            
                        if k in ["TMIN","TEMP","TMAX","SST"]:
                            ds = ds - 273.15
                            ds = getattr(ds,ensemble_mean)(dim="number") if ensemble_mean != None else ds 
                            ds = ds.mean(dim="forecastMonth").isel(latitude=slice(None, None, -1))
                            if "indexing_time" in ds.coords: 
                                ds = ds.rename({"latitude":"lat","longitude":"lon","indexing_time":"time"})
                            else:
                                ds = ds.rename({"latitude":"lat","longitude":"lon","forecast_reference_time":"time"})
                        if k =="PRCP":
                            ds = getattr(ds,ensemble_mean)(dim="number") if ensemble_mean != None else ds
                            ds = (1000*30*24*60*60*ds).sum(dim="forecastMonth").isel(latitude=slice(None, None, -1))
                            if "indexing_time" in ds.coords: 
                                ds = ds.rename({"latitude":"lat","longitude":"lon","indexing_time":"time"})
                            else:
                                ds = ds.rename({"latitude":"lat","longitude":"lon","forecast_reference_time":"time"})

                        if k == "SLP":
                            ds = ds/100
                            ds = getattr(ds,ensemble_mean)(dim="number") if ensemble_mean != None else ds 
                            ds = ds.mean(dim="forecastMonth").isel(latitude=slice(None, None, -1))
                            if "indexing_time" in ds.coords: 
                                ds = ds.rename({"latitude":"lat","longitude":"lon","indexing_time":"time"})
                            else:
                                ds = ds.rename({"latitude":"lat","longitude":"lon","forecast_reference_time":"time"})

                        if k in ["UGRD10","VGRD10","DSWR","DLWR"]:
                            ds = getattr(ds,ensemble_mean)(dim="number") if ensemble_mean != None else ds
                            ds = ds.mean(dim="forecastMonth").isel(latitude=slice(None, None, -1))
                            if "indexing_time" in ds.coords: 
                                ds = ds.rename({"latitude":"lat","longitude":"lon","indexing_time":"time"})
                            else:
                                ds = ds.rename({"latitude":"lat","longitude":"lon","forecast_reference_time":"time"})

                        if k not in ["TMIN","TEMP","TMAX","SST","UGRD10","VGRD10", "PRCP","SLP","DSWR","DLWR"]:
                            ds = getattr(ds,ensemble_mean)(dim="number") if ensemble_mean != None else ds
                            ds = ds.drop_vars("pressure_level").squeeze().mean(dim="forecastMonth").isel(latitude=slice(None, None, -1))
                            if "indexing_time" in ds.coords: 
                                ds = ds.rename({"latitude":"lat","longitude":"lon","indexing_time":"time"})
                            else:
                                ds = ds.rename({"latitude":"lat","longitude":"lon","forecast_reference_time":"time"})
            
                        os.remove(file_path)
                        print(f"Deleted not process file: {file_path}")
                            
                        ds = ds.rename({"lon":"X","lat":"Y","time":"T"})    
                        output_path = f"{dir_to_save}/{file_prefix}_{cent.replace('_', '')}{syst}_{k}_{abb_mont_ini}Ic_{season}_{lead_time[0]}.nc"
                        
                        # Save the combined dataset for the center-variable combination
                        ds.to_netcdf(output_path)
                        print(f"Download finished, combined dataset for {cent} {syst} {k} to {output_path}")
                        ds.close()
                        store_file_path[f"{cent}_{syst}"] = file_path
                    except Exception as e:
                        print(f"Failed to download data for {k}: {e}")

        return store_file_path


    def WAS_Download_AgroIndicators_daily(
        self,
        dir_to_save,
        variables,
        year_start,
        year_end,
        area,
        force_download=False,
    ):
        """
        Download daily agro-meteorological indicators for specified variables and years.
    
        Parameters:
            dir_to_save (str): Directory to save the downloaded files.
            variables (list): List of shorthand variables to download (e.g., ["AGRO.PRCP", "AGRO.TMAX"]).
            year_start (int): Start year for the data to download.
            year_end (int): End year for the data to download.
            area (list): Bounding box as [North, West, South, East] for clipping.
            force_download (bool): If True, forces download even if file exists.
        """
        dir_to_save = Path(dir_to_save)
        os.makedirs(dir_to_save, exist_ok=True)
        days = [f"{day:02}" for day in range(1, 32)]
        months = [f"{month:02}" for month in range(1, 13)]
        version = "1_1"
    
        # Updated variable mapping with statistic
        variable_mapping = {
            "AGRO.PRCP": ("precipitation_flux", None),
            "AGRO.TMAX": ("2m_temperature", "24_hour_maximum"),
            "AGRO.TEMP": ("2m_temperature", "24_hour_mean"),
            "AGRO.TMIN": ("2m_temperature", "24_hour_minimum"),
        }
    
        for var in variables:
            if var not in variable_mapping:
                print(f"Unknown variable: {var}. Skipping.")
                continue
    
            cds_variable, statistic = variable_mapping[var]
            output_path = dir_to_save / f"Daily_{var.split('.')[1]}_{year_start}_{year_end}.nc"
    
            if not force_download and os.path.exists(output_path):
                print(f"{output_path} already exists. Skipping download.") 
            else:
                combined_datasets = []
                for year in range(year_start, year_end + 1):
                    zip_file_path = dir_to_save / f"Daily_{var.split('.')[1]}_{year}.zip"
                
                    dataset = "sis-agrometeorological-indicators"
                    request = {
                        "variable": cds_variable,
                        "year": str(year),
                        "month": months,
                        "day": days,
                        "version": version,
                        "area": area,
                    }
    
                    # Include the statistic parameter if specified
                    if statistic:
                        request["statistic"] = [statistic]
    
                    try:
                        client = cdsapi.Client()
                        print(f"Downloading {cds_variable} ({statistic}) data for {year}...")
                        client.retrieve(dataset, request).download(str(zip_file_path))
                        print(f"Downloaded: {zip_file_path}")
                    except Exception as e:
                        print(f"Failed to download {cds_variable} ({statistic}) data for {year}: {e}")
                        continue
    
                    # Extract NetCDF files from the ZIP archive
                    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                        for netcdf_file_name in zip_ref.namelist():
                            with zip_ref.open(netcdf_file_name) as file:
                                ds = xr.open_dataset(io.BytesIO(file.read()))
                                combined_datasets.append(ds)
    
                    os.remove(zip_file_path)
                    print(f"Deleted ZIP file: {zip_file_path}")
    
                # Concatenate all daily datasets into a single file
                if combined_datasets:
                    combined_ds = xr.concat(combined_datasets, dim="time")
                    # Convert temperature data from Kelvin to Celsius if needed
                    if var in ["AGRO.TMIN", "AGRO.TEMP", "AGRO.TMAX"]:
                        combined_ds = combined_ds - 273.15  # Convert from Kelvin to Celsius
    
                    # Rename dimensions and save to NetCDF
                    combined_ds = combined_ds.rename({"lon": "X", "lat": "Y", "time": "T"})
                    combined_ds = combined_ds.isel(Y=slice(None, None, -1))
                    combined_ds.to_netcdf(output_path)
                    print(f"File downloaded and combined dataset for {var} is saved to {output_path}")

    def WAS_Download_Models_Daily(
        self,
        dir_to_save,
        center_variable,         # e.g. ["ECMWF_51.PRCP", "UKMO_603.TEMP", ...]
        month_of_initialization, # int: e.g. 2 for February
        day_of_initialization,   # int: e.g. 1 for the 1st day
        leadtime_hour,           # list of strings: e.g. ["24","48",..., "5160"]
        year_start_hindcast,
        year_end_hindcast,
        area,
        year_forecast=None,
        ensemble_mean=None,
        force_download=False,
    ):
        """
        Download daily/sub-daily seasonal forecast model data (original)
        using 'seasonal-original-single-levels' from the CDS.
    
        Parameters:
            dir_to_save (str or Path): Directory to save the downloaded files.
            center_variable (list): Each element e.g. "ECMWF_51.PRCP"
                - left side of '.' is model (ECMWF_51),
                - right side is variable short code (PRCP).
            month_of_initialization (int): Initialization month (1-12).
            day_of_initialization (int): Initialization day (1-31).
            leadtime_hour (list of str): e.g. ["24", "48", ..., "5160"].
            year_start_hindcast (int): Start year for hindcast data.
            year_end_hindcast (int): End year for hindcast data.
            area (list): Bounding box as [North, West, South, East].
            year_forecast (int, optional): If provided, downloads that single
                forecast year. Otherwise downloads hindcast for the specified range.
            ensemble_mean (str, optional): e.g. "mean", "median", or None.
            force_download (bool): Force download if True, even if file exists.
        """
    
        # 1. Determine whether we are downloading hindcast or forecast.
        if year_forecast is None:
            # Hindcast range
            years = [str(y) for y in range(year_start_hindcast, year_end_hindcast + 1)]
            file_prefix = "hindcast"
        else:
            # Single forecast year
            years = [str(year_forecast)]
            file_prefix = "forecast"
    
        # 2. Build standard dictionaries for center/system/variables
        centre = {
            "ECMWF_51": "ecmwf",
            "UKMO_604": "ukmo", # month of initialization available for forecast are Apr to __
            "UKMO_603": "ukmo", # month of initialization available for forecast are Jan to Mar
            "METEOFRANCE_8": "meteo_france",
            "DWD_21": "dwd",
            "DWD_22": "dwd",
            # "DWD_2": "dwd",
            "CMCC_35": "cmcc",
            # "CMCC_3": "cmcc",
            "NCEP_2": "ncep",
            "JMA_3": "jma",
            "ECCC_4": "eccc",
            "ECCC_5": "eccc",
        }
    
        system = {
            "ECMWF_51": "51",
            "UKMO_604": "604",
            "UKMO_603": "603",
            "METEOFRANCE_8": "8",
            "DWD_21": "21",
            "DWD_22": "22",
            # "DWD_2": "2",
            "CMCC_35": "35",
            # "CMCC_3": "3",
            "NCEP_2": "2",
            "JMA_3": "3",
            "ECCC_4": "4",
            "ECCC_5": "5",
        }
    
        variables_1 = {
            "PRCP":  "total_precipitation",
            "TEMP":  "2m_temperature",
            "TMAX":  "maximum_2m_temperature_in_the_last_24_hours",
            "TMIN":  "minimum_2m_temperature_in_the_last_24_hours",
            "UGRD10":"10m_u_component_of_wind",
            "VGRD10":"10m_v_component_of_wind",
            "SST":   "sea_surface_temperature",
            "SLP": "mean_sea_level_pressure",
            "DSWR": "surface_solar_radiation_downwards",
            "DLWR": "surface_thermal_radiation_downwards",
        }
        variables_2 = {
            "HUSS_1000": "specific_humidity",
            "HUSS_925":  "specific_humidity",
            "HUSS_850":  "specific_humidity",
            "UGRD_1000": "u_component_of_wind",
            "UGRD_925":  "u_component_of_wind",
            "UGRD_850":  "u_component_of_wind",
            "VGRD_1000": "v_component_of_wind",
            "VGRD_925":  "v_component_of_wind",
            "VGRD_850":  "v_component_of_wind",
        }

        ### Particularity for day of initialization NCEP and JMA
        init_day_dict_jma = {
            "01":"16", "02":"10", "03":"12", "04":"11", "05":"16", "06":"15",
            "07":"15", "08":"14", "09":"13", "10":"13", "11":"12", "12":"12"
        }

        init_day_dict_ncep = {
            "01":"01", "02":"05", "03":"02", "04":"01", "05":"01", "06":"05",
            "07":"05", "08":"04", "09":"03", "10":"03", "11":"02", "12":"02"
        }
        
    
        # 3. Ensure the output directory exists
        dir_to_save = Path(dir_to_save)
        dir_to_save.mkdir(parents=True, exist_ok=True)
        store_file_path = {}
        # 4. Loop over each center-variable combination
        for cv in center_variable:
            # Example: "ECMWF_51.PRCP"
            c = cv.split(".")[0]  # e.g. "ECMWF_51"
            v = cv.split(".")[1]  # e.g. "PRCP"
    
            # Map to the Copernicus naming
            cent = centre[c]
            syst = system[c]
            if v in variables_1:
                var_cds = variables_1[v]
            elif v in variables_2:
                var_cds = variables_2[v]
            else:
                print(f"Unknown variable code: {v}, skipping.")
                continue
    
            # Build a single output path
            abb_mont_ini = month_abbr[int(month_of_initialization)]
            
            # E.g. "hindcast_ecmwf51_PRCP_Feb01_1981-2016_24-5160.nc"
            years_str = f"{years[0]}_{years[-1]}" if len(years) > 1 else years[0]
            lead_str  = f"{leadtime_hour[0]}-{leadtime_hour[-1]}" if len(leadtime_hour) > 1 else leadtime_hour[0]
    
            output_file = (
                dir_to_save /
                f"{file_prefix}_{cent}{syst}_{v}_{abb_mont_ini}{day_of_initialization}_{years_str}_{lead_str}.nc"
            )
    
            if not force_download and output_file.exists():
                print(f"{output_file} already exists. Skipping download.")
                store_file_path[f"{cent}{syst}"] = output_file
                continue

            if cent == "jma" and year_forecast is None:
                day_of_initialization = init_day_dict_jma[month_of_initialization]
            if cent == "ncep" and year_forecast is None:
                day_of_initialization = init_day_dict_ncep[month_of_initialization]
                    
            # 5. Prepare the request for 'seasonal-original-single-levels'
            dataset = "seasonal-original-single-levels"
            request = {
                "originating_centre": cent,
                "system": syst,
                "variable": [var_cds],
                "year": years,  # list of strings
                "month": [f"{int(month_of_initialization):02}"],
                "day":   [f"{int(day_of_initialization):02}"],
                "leadtime_hour": leadtime_hour,  # e.g. ["24","48",..., "5160"]
                "data_format": "netcdf",
                "area": area,   # e.g. [90, -180, -90, 180]
            }
    
            # Temporary file to download
            temp_file = dir_to_save / f"temp_{cent}{syst}_{v}.nc"
    
            # 6. Download from CDS
            client = cdsapi.Client()
            try:
                print(f"Requesting data from '{dataset}' for {cv}...")
                client.retrieve(dataset, request).download(str(temp_file))
                print(f"Downloaded: {temp_file}")
            except Exception as e:
                print(f"Failed to download data for {cv}: {e}")
                continue
    
            # 7. Post-process with xarray
            try:
                ##########################################################
                # Take in account level pressure for some variables in this part
                ##########################################################
                 
                ds = xr.open_dataset(temp_file)
                time = (ds['forecast_reference_time'] + ds['forecast_period']).data
                ds = ds.assign_coords(time=(('forecast_reference_time', 'forecast_period'), time))
                ds = ds.stack(time=('forecast_reference_time', 'forecast_period'))
                ds = ds.drop_vars(['forecast_reference_time', 'forecast_period'])
                ds = ds.rename({"valid_time":"time"})
    
                # If there's an ensemble dimension, apply ensemble mean/median if requested
                if ensemble_mean in ["mean", "median"] and "number" in ds.dims:
                    ds = getattr(ds, ensemble_mean)(dim="number")
    
                # For example, flip latitude if needed
                if "latitude" in ds.coords:
                    ds = ds.isel(latitude=slice(None, None, -1))

                if v in ["TMIN","TEMP","TMAX","SST"]:
                    ds = ds - 273.15
                if v =="PRCP":
                    ds['time'] = ds['time'].to_index()
                    years = ds['time'].dt.year
                    tampon = []
                    for year in np.unique(years):
                        
                        # Select the data for the specific year
                        yearly_ds = ds.sel(time=ds['time'].dt.year == year)
                        
                        # Calculate differences for the year
                        differences = [yearly_ds.isel(time=i) - yearly_ds.isel(time=i-1) for i in range(1, len(yearly_ds['time']))]
                        differences = xr.concat(differences, dim="time")
                        differences['time'] = yearly_ds['time'].isel(time=slice(1,None))
                        tampon.append(differences)
                    ds = xr.concat(tampon, dim="time")*1000

                    ##########################################################
                    # Include after the processing of SLP, DSWR, DLWR,
                    ##########################################################

                # Finally, rename the coords to X, Y, T to match my style
                if "longitude" in ds.coords:
                    ds = ds.rename({"longitude": "X"})
                if "latitude" in ds.coords:
                    ds = ds.rename({"latitude": "Y"})
                if "time" in ds.coords:
                    ds = ds.rename({"time": "T"})
    
                # 8. Save the processed data
                ds.to_netcdf(output_file)
                ds.close()
                print(f"Saved processed data to: {output_file}")
                store_file_path[f"{cent}{syst}"] = output_file
        
            except Exception as e:
                print(f"Error reading or processing {temp_file}: {e}")
    
            finally:
                # Remove the temporary file
                if temp_file.exists():
                    os.remove(temp_file)
                    print(f"Deleted temp file: {temp_file}")
        return store_file_path
        
    def WAS_Download_AgroIndicators(
        self,
        dir_to_save,
        variables,
        year_start,
        year_end,
        area,
        seas=["01", "02", "03"],  # e.g. NDJ = ["11","12","01"]
        force_download=False,
    ):
        """
        Download agro-meteorological indicators for specified variables, years, and months,
        handling cross-year seasons (e.g., NDJ).
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
        }

        version = "1_1"
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
            if (not force_download) and output_path.exists():
                print(f"{output_path} already exists. Skipping download.")
                continue

            # We'll accumulate all partial datasets here
            all_years_datasets = []

            # Loop over each year in the requested range
            for year in range(year_start, year_end + 1):
                # Split the months into those belonging to "base" year vs "next" year
                base_months = [m for m in season_months if m >= pivot]
                next_months = [m for m in season_months if m < pivot]

                # 1) Download part A (base-year months), if any
                if base_months:
                    months_base = [month_str(m) for m in base_months]
                    zip_file_path = dir_to_save / f"Obs_{var_short}_{year}_{season_str}_partA.zip"

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

                    # Attempt download
                    try:
                        client = cdsapi.Client()
                        print(f"Downloading {cds_variable} for {year} months={months_base}")
                        client.retrieve("sis-agrometeorological-indicators", request).download(str(zip_file_path))
                    except Exception as e:
                        print(f"Failed to download {cds_variable} year={year} Part A: {e}")
                        continue

                    # Unzip each netCDF and append
                    with zipfile.ZipFile(zip_file_path, 'r') as z:
                        for nc_name in z.namelist():
                            with z.open(nc_name) as f:
                                ds = xr.open_dataset(io.BytesIO(f.read()))
                                all_years_datasets.append(ds)
                    os.remove(zip_file_path)

                # 2) Download part B (next-year months), if any and if we have a next year
                if next_months and (year < year_end+1):
                    year_next = year + 1
                    months_next = [month_str(m) for m in next_months]
                    zip_file_path = dir_to_save / f"Obs_{var_short}_{year}_{season_str}_partB_{year_next}.zip"

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

                    # Attempt download
                    try:
                        client = cdsapi.Client()
                        print(f"Downloading {cds_variable} for {year_next} months={months_next}")
                        client.retrieve("sis-agrometeorological-indicators", request).download(str(zip_file_path))
                    except Exception as e:
                        print(f"Failed to download {cds_variable} year={year_next} Part B: {e}")
                        continue

                    # Unzip each netCDF and append
                    with zipfile.ZipFile(zip_file_path, 'r') as z:
                        for nc_name in z.namelist():
                            with z.open(nc_name) as f:
                                ds = xr.open_dataset(io.BytesIO(f.read()))
                                all_years_datasets.append(ds)
                    os.remove(zip_file_path)

            # -----------------------------------------
            # Post-process & combine all partial years
            # -----------------------------------------

            if all_years_datasets:
                combined_ds = xr.concat(all_years_datasets, dim="time")
                # If it's temperature, subtract 273.15 (Kelvin → Celsius)
                if var in ["AGRO.TMIN", "AGRO.TEMP", "AGRO.TMAX"]:
                    combined_ds = combined_ds - 273.15
    
                # Now do custom aggregator for NDJ or any cross-year months
                combined_ds = self._aggregate_crossyear(
                    ds=combined_ds,
                    season_months=season_months,
                    var_name=var
                )
    
                # Finally rename dims
                # (the aggregator function has already replaced 'time' with our 'season_year')
                if "lon" in combined_ds.dims:
                    combined_ds = combined_ds.rename({"lon": "X"})
                if "lat" in combined_ds.dims:
                    combined_ds = combined_ds.rename({"lat": "Y"})
                # Flip lat 
                combined_ds = combined_ds.isel(Y=slice(None, None, -1))
                
                combined_ds["time"] = [f"{year}-{seas[1]}-01" for year in combined_ds["time"].astype(str).values]
                combined_ds["time"] = combined_ds["time"].astype("datetime64[ns]")
                combined_ds = combined_ds.rename({"time": "T"}) 
                
                combined_ds.to_netcdf(output_path)
                print(f"Saved final dataset for {var} to: {output_path}")
            else:
                print(f"No data downloaded for {var} in {season_str}.")
    

    # -------------------------------------------------------------------------
    # Helper for Reanalysis cross-year post-processing (optional)
    # -------------------------------------------------------------------------
    def _postprocess_reanalysis(self, ds, var_name):
        """
        Drop extra coords, rename dims, flip lat, etc.
        Adjust as needed for your MERRA2 or ERA5 quirks.
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
        ds = ds.drop_vars('zlev').squeeze()
        keep_vars = [var_name, 'T', 'X', 'Y']
        drop_vars = [v for v in ds.variables if v not in keep_vars]
        return ds.drop_vars(drop_vars, errors="ignore")

    def _aggregate_crossyear(self, ds, season_months, var_name):
        """s
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

        if any(x in var_name for x in ["TEMP","TMIN","TMAX","SST","SLP"]):
            ds_out = ds.groupby("season_year").mean("time")
        elif any(x in var_name for x in ["PRCP","DSWR","DLWR"]):
            ds_out = ds.groupby("season_year").sum("time")
        else:
            ds_out = ds.groupby("season_year").mean("time")
    
        # 4) Rename "season_year" to "time", 
        #    so we end up with a time dimension (representing each seasonal year).
        ds_out = ds_out.rename({"season_year": "time"})
    
        return ds_out


    def WAS_Download_Reanalysis(
        self,
        dir_to_save,
        center_variable,
        year_start,
        year_end,
        area,
        seas=["01", "02", "03"],  # e.g. NDJ = ["11","12","01"]
        force_download=False,
        run_avg=3
    ):
        """
        Download reanalysis data for specified center-variable combinations, years, and months,
        handling cross-year seasons (e.g., NDJ).
        """
        dir_to_save = Path(dir_to_save)
        dir_to_save.mkdir(parents=True, exist_ok=True)

        # Parse center and variable strings
        centers = [cv.split(".")[0] for cv in center_variable]
        vars_   = [cv.split(".")[1] for cv in center_variable]

        # Example reanalysis centers
        centre_dict = {"ERA5": "ERA5", "MERRA2": "MERRA2"}

        # Single-level monthly means
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

        }
        # Pressure-level monthly means
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
        
        # Helper for zero-padded month strings
        def m2str(m: int):
            return f"{m:02d}"

        # Convert months to integers (e.g. ["11","12","01"] -> [11,12,1])
        season_months = [int(m) for m in seas]
        pivot = season_months[0]
        # For naming
        season_str = "".join([calendar.month_abbr[m] for m in season_months])

        for c, v in zip(centers, vars_):
            # =================================================================
            # Special Case: NOAA ERSST from IRIDL
            # =================================================================
            if c == "NOAA" and v == "SST":
                out_file = dir_to_save / f"{c}_{v}_{year_start}_{year_end}_{season_str}.nc"
                if not force_download and out_file.exists():
                    print(f"{out_file} exists. Skipping.")
                    continue
        
                try:
                    # Build IRIDL URL using bounding box [N, W, S, E]
                    url = build_iridl_url_ersst(
                        year_start=year_start,
                        year_end=year_end,
                        bbox=area,     # e.g. [10, -15, -5, 15]
                        run_avg=run_avg,
                        month_start="Jan",
                        month_end="Dec"
                    )
                    print(f"Using IRIDL URL: {url}")
                    
                    # 2) Open dataset with manual processing
                    ds = xr.open_dataset(url, decode_times=False)
                    ds = decode_cf(ds, "T").rename({"T":"time"}).convert_calendar("proleptic_gregorian", align_on="year").rename({"time":"T"})

                    ds = ds.rename({
                            'sst': 'SST',  # Rename variable to match expected name
                        })
                                                                                                               
                    # 6) Post-process
                    ds = self._postprocess_reanalysis_ersst(ds, v)
                    ds['T'] = ds['T'].astype('datetime64[ns]')                   
                    
                    # 7) Final formatting
                    ds_agg = ds.where(ds.T.dt.month == int(seas[1]), drop=True)
                    ds_agg = fix_time_coord(ds_agg,seas)
                    ds_agg = ds_agg.rename({
                            'SST': 'sst',  # Rename variable to match expected name
                        })
                    # 8) Save
                    ds_agg.to_netcdf(out_file)
                    print(f"Saved NOAA ERSST data to {out_file}")
        
                except Exception as e:
                    print(f"Failed to download {c}/{v}: {str(e)}")
                continue

        # # Loop over each reanalysis center/var
        # for c, v in zip(centers, vars_):
        #     # e.g. ERA5, PRCP
            if c not in centre_dict:
                print(f"Unknown center: {c}, skipping.")
                continue

            rean = centre_dict[c]
            out_file = dir_to_save / f"{c}_{v}_{year_start}_{year_end}_{season_str}.nc"
            if (not force_download) and out_file.exists():
                print(f"{out_file} already exists. Skipping.")
                continue

            # List to accumulate partial downloads
            combined_datasets = []

            # Iterate over each year in [year_start..year_end]
            for year in range(year_start, year_end + 1):
                # Split months
                base_months = [m for m in season_months if m >= pivot]
                next_months = [m for m in season_months if m < pivot]

                # (A) Base-year
                if base_months:
                    base_str = [m2str(m) for m in base_months]
                    partA = dir_to_save / f"{c}_{v}_{year}_{season_str}_partA.nc"

                    # Decide dataset + request
                    if v in variables_2:
                        press_level = v.split("_")[1]  # e.g. 925 from "HUSS_925"
                        dataset = "reanalysis-era5-pressure-levels-monthly-means"
                        request = {
                            "product_type": ["monthly_averaged_reanalysis"],
                            "variable": variables_2[v],
                            "pressure_level": press_level,
                            "year": str(year),
                            "month": base_str,
                            "time": ["00:00"],
                            "area": area,
                            "format": "netcdf",
                        }
                    else:
                        dataset = "reanalysis-era5-single-levels-monthly-means"
                        request = {
                            "product_type": ["monthly_averaged_reanalysis"],
                            "variable": variables_1.get(v),
                            "year": str(year),
                            "month": base_str,
                            "time": ["00:00"],
                            "area": area,
                            "format": "netcdf",
                        }

                    # Download
                    try:
                        client = cdsapi.Client()
                        print(f"Downloading {c}/{v}: {year}, months={base_str}")
                        client.retrieve(dataset, request).download(str(partA))
                    except Exception as e:
                        print(f"Download error for {c}/{v}, year={year} partA: {e}")
                        continue

                    with xr.open_dataset(partA) as dsA:
                        dsA = dsA.load()
                        dsA = self._postprocess_reanalysis(dsA, v)
                        combined_datasets.append(dsA)
                    os.remove(partA)

                # (B) Next-year
                if next_months and (year < year_end+1):
                    year_next = year + 1
                    next_str = [m2str(m) for m in next_months]
                    partB = dir_to_save / f"{c}_{v}_{year}_{season_str}_partB_{year_next}.nc"

                    if v in variables_2:
                        press_level = v.split("_")[1]
                        dataset = "reanalysis-era5-pressure-levels-monthly-means"
                        request = {
                            "product_type": ["monthly_averaged_reanalysis"],
                            "variable": variables_2[v],
                            "pressure_level": press_level,
                            "year": str(year_next),
                            "month": next_str,
                            "time": ["00:00"],
                            "area": area,
                            "format": "netcdf",
                        }
                    else:
                        dataset = "reanalysis-era5-single-levels-monthly-means"
                        request = {
                            "product_type": ["monthly_averaged_reanalysis"],
                            "variable": variables_1.get(v),
                            "year": str(year_next),
                            "month": next_str,
                            "time": ["00:00"],
                            "area": area,
                            "format": "netcdf",
                        }

                    # Download
                    try:
                        client = cdsapi.Client()
                        print(f"Downloading {c}/{v}: {year_next}, months={next_str}")
                        client.retrieve(dataset, request).download(str(partB))
                    except Exception as e:
                        print(f"Download error for {c}/{v}, year={year_next} partB: {e}")
                        continue

                    with xr.open_dataset(partB) as dsB:
                        dsB = dsB.load()
                        dsB = self._postprocess_reanalysis(dsB, v)
                        combined_datasets.append(dsB)
                    os.remove(partB)

            if combined_datasets:
                dsC = xr.concat(combined_datasets, dim="time")
            
                # If T variable -> K to °C
                if v in ["TMIN","TEMP","TMAX","SST"]:
                    dsC = dsC - 273.15
                
                # For precipitation or others, the aggregator decides sum vs mean
                dsC = self._aggregate_crossyear(
                    ds=dsC,
                    season_months=season_months,
                    var_name=v
                )
                
                if v == "PRCP":
                    nbjour = len(season_months)*30
                    dsC = nbjour*1000*dsC  # !!!!! Convert to mm/month
                
                if v in ["DSWR", "DLWR"]:
                    nbjour = len(season_months)*30
                    dsC = (dsC * 30)/(nbjour*86400)  # Convert to W/m2

                if v == "SLP":
                    dsC = dsC / 100  # Convert to hPa(mb)

                dsC["time"] = [f"{year}-{seas[1]}-01" for year in dsC["time"].astype(str).values]
                dsC["time"] = dsC["time"].astype("datetime64[ns]")
                dsC = dsC.rename({"time": "T"})
                
                # Save final
                dsC.to_netcdf(out_file)
                print(f"Saved final reanalysis file: {out_file}")
            else:
                print(f"No data found for {c}/{v}.")


    def WAS_Download_CHIRPSv3(
        self,
        dir_to_save,
        variables,
        year_start,
        year_end,
        area=None,
        season_months=["03", "04", "05"],
        force_download=False        
    ):
        """
        Download CHIRPS v3.0 monthly precipitation for a specified cross-year season
        from year_start to year_end, optionally clipped to 'area',
        and aggregate them into a single NetCDF file.
        """
        dir_to_save = Path(dir_to_save)
        dir_to_save.mkdir(parents=True, exist_ok=True)
        season_months = [int(m) for m in season_months]
        variables = variables
        # Example: "MAM"
        season_str = "".join([calendar.month_abbr[m] for m in season_months])
        pivot = season_months[0]

        out_nc = dir_to_save / f"Obs_PRCP_{year_start}_{year_end}_{season_str}.nc"
        if out_nc.exists() and not force_download:
            print(f"[INFO] {out_nc} already exists. Skipping.")
            return

        # We'll store monthly DataArrays here
        all_data_arrays = []

        # Loop over years
        for year in range(year_start, year_end + 1):
            # Base-year months (>= pivot)
            base_months = [m for m in season_months if m >= pivot]
            # Next-year months (< pivot)
            next_months = [m for m in season_months if m < pivot]

            # Part A: Base-year months
            for m in base_months:
                da = self._fetch_chirps_monthly(
                    year=year,
                    month=m,
                    dir_to_save=dir_to_save,
                    force_download=force_download,
                    area=area
                )
                if da is not None:
                    all_data_arrays.append(da)

            # Part B: Next-year months
            if next_months and (year < year_end + 1):
                year_next = year + 1
                for m in next_months:
                    da = self._fetch_chirps_monthly(
                        year=year_next,
                        month=m,
                        dir_to_save=dir_to_save,
                        force_download=force_download,
                        area=area
                    )
                    if da is not None:
                        all_data_arrays.append(da)

        if len(all_data_arrays) == 0:
            print("[WARNING] No CHIRPS data arrays were opened/downloaded.")
            return

        # Concatenate along time
        ds_all = xr.concat(all_data_arrays, dim="time").to_dataset(name="precip")

        # Aggregate across the cross-year season (summing monthly precipitation)
        ds_season = self._aggregate_chirps(ds_all, season_months)

        # Rename dims if desired
        if "x" in ds_season.dims:
            ds_season = ds_season.rename({"x": "X"})
        if "y" in ds_season.dims:
            ds_season = ds_season.rename({"y": "Y"})
        if "time" in ds_season.dims:
            ds_season = ds_season.rename({"time": "T"})
        # Write to NetCDF
        ds_season.to_netcdf(out_nc)
        print(f"[INFO] Saved seasonal CHIRPS data to {out_nc}")
        # Delete individual monthly TIF files
        for tif_file in dir_to_save.glob("chirps-v3.0.*.tif"):
            try:
                os.remove(tif_file)
                print(f"[CLEANUP] Deleted {tif_file}")
            except Exception as e:
                print(f"[ERROR] Could not delete {tif_file}: {e}")



    def _fetch_chirps_monthly(self, year, month, dir_to_save, force_download, area):
        """
        Construct the CHIRPS v3.0 monthly TIF URL for (year, month), 
        download if needed, open as xarray, and optionally clip to 'area'.
        
        File format is: chirps-v3.0.YYYY.MM.tif
        """
        base_url = "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/africa/tifs"
        fname = f"chirps-v3.0.{year}.{month:02d}.tif"
        url = f"{base_url}/{fname}"

        local_path = Path(dir_to_save) / fname
        download_file(url, local_path, force_download=force_download, chunk_size=8192, timeout=120)
        
        # # Download if needed
        # if (not local_path.exists()) or force_download:
        #     print(f"[DOWNLOAD] {url}")
        #     try:
        #         with requests.get(url, stream=True, timeout=120) as r:
        #             r.raise_for_status()
        #             with open(local_path, "wb") as f:
        #                 for chunk in r.iter_content(chunk_size=8192):
        #                     f.write(chunk)
        #     except Exception as e:
        #         print(f"[ERROR] Could not download {url}: {e}")
        #         return None
        # else:
        #     print(f"[SKIP] {fname} is already present. (Use force_download=True to overwrite)")

        # Open as xarray via rioxarray
        try:
            da = rioxr.open_rasterio(local_path, masked=True).squeeze()
            time_coord = pd.to_datetime(f"{year}-{month:02d}-01")
            da = da.expand_dims(time=[time_coord])
            da.name = "precip"

            # If area is provided, clip
            if area and len(area) == 4:
                north, west, south, east = area
                da = da.rio.clip_box(
                    minx=west,
                    miny=south,
                    maxx=east,
                    maxy=north
                )

            return da

        except Exception as e:
            print(f"[ERROR] Could not open/parse {local_path}: {e}")
            return None

    def _aggregate_chirps(self, ds, season_months):
        """
        Sum monthly precipitation across the cross-year season.
        """
        if "time" not in ds.coords:
            raise ValueError("Dataset must have a 'time' dimension.")

        pivot = season_months[0]
        # Label each time with 'season_year'
        season_year = ds["time"].dt.year.where(ds["time"].dt.month >= pivot,
                                               ds["time"].dt.year - 1)
        ds = ds.assign_coords(season_year=season_year)

        # Keep only the months we want
        ds = ds.where(ds["time"].dt.month.isin(season_months), drop=True)

        # Sum across the months for precipitation
        ds_out = ds.groupby("season_year").sum("time", skipna=True)

        # Rename season_year -> time
        ds_out = ds_out.rename({"season_year": "time"})

        # Optionally make the new time coordinate more descriptive:
        new_times = []
        for sy in ds_out.coords["time"].values:
            new_times.append(f"{sy}-{pivot:02d}-01")
        ds_out = ds_out.assign_coords(time=pd.to_datetime(new_times))

        return ds_out


####

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



