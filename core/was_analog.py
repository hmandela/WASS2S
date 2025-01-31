# Core Python Libraries
import os
from pathlib import Path
import calendar
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Numerical and Data Manipulation Libraries
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da

# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Climate Data API
import cdsapi

# Machine Learning and Statistical Analysis Libraries
from minisom import MiniSom
import somoclu
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from scipy.stats import pearsonr
from scipy import stats

# Deep Learning Libraries
# import tensorflow as tf
# from tensorflow.keras import layers, models, backend as K

# EOF Analysis Library
import xeofs

import matplotlib.pyplot as plt
from matplotlib import colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm

class WAS_Analog:
    
    def __init__(self, dir_to_save, year_start, best_prcp_models=None, month_of_initialization=None,
                  lead_time=None, ensemble_mean = "mean", clim_year_start=None, clim_year_end=None,
                  define_extent = None, index_compute = None, some_grid_size = (None, None)
                ):
        
        self.dir_to_save = dir_to_save
        self.year_start = year_start
        self.month_of_initialization = month_of_initialization
        self.lead_time = lead_time
        self.ensemble_mean = ensemble_mean
        self.clim_year_start = clim_year_start
        self.clim_year_end = clim_year_end
        self.best_prcp_models = best_prcp_models
        self.define_extent = define_extent
        self.index_compute = index_compute
        self.some_grid_size = some_grid_size

    def calc_index(self, indices, sst):
    
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
        
        predictor = {}
        for idx in sst_indices_name.keys():
            if idx in ["TASI", "DMI"]:
                continue
            _, lon_min, lon_max, lat_min, lat_max = sst_indices_name[idx]
            sst_region = sst.sel(X=slice(lon_min, lon_max), Y=slice(lat_min, lat_max)).mean(dim=["X", "Y"], skipna=True)
            predictor[idx] = sst_region
              
        # Compute derived indices
        predictor["TASI"] = predictor["NAT"] - predictor["SAT"]
        predictor["DMI"] = predictor["WTIO"] - predictor["SETIO"]
        
        selected_indices = {i: predictor[i] for i in indices}
        data_vars = {key: ds.rename(key) for key, ds in selected_indices.items()}
        indices_dataset = xr.Dataset(data_vars)
        return indices_dataset
    
    def download_sst_reanalysis(
        self,
        center='ERA5',
        area=[60, -180, -60, 180],
        force_download=False
    ):
        """
        Download Sea Surface Temperature (SST) reanalysis data for specified years and all months.
    
        Parameters:
            dir_to_save (str): Directory to save the downloaded files.
            center (str): Reanalysis center (e.g., "ERA5"). Default is "ERA5".
            year_end (int): End year for the data to download.
            area (list): Bounding box as [North, West, South, East] for clipping.
            force_download (bool): If True, forces download even if file exists.
        """
        year_end = datetime.now().year
        
        # Define variable mapping
        variables_1 = {
            "SST": "sea_surface_temperature",
        }
    
        # Validate center
        if center not in ["ERA5"]:
            raise ValueError(f"Unsupported center '{center}'. Supported centers are 'ERA5'.")
    
        # Define dataset based on center
        if center == "ERA5":
            dataset = "reanalysis-era5-single-levels-monthly-means"
    
        # Define variable
        variable = variables_1["SST"]
    
        # Prepare save directory
        dir_to_save = Path(self.dir_to_save)
        os.makedirs(dir_to_save, exist_ok=True)
    
        # Define all months
        months = [f"{m:02}" for m in range(1, 13)]
        season = "".join([calendar.month_abbr[int(month)] for month in months])
    
        # Define combined output file path
        combined_output_path = dir_to_save / f"{center}_SST_{self.year_start}_{year_end}_{season}.nc"
    
        if not force_download and combined_output_path.exists():
            print(f"{combined_output_path} already exists. Skipping download.")
            combined_ds = xr.open_dataset(combined_output_path)
            return combined_ds
        else:
            combined_datasets = []
            client = cdsapi.Client()  # Initialize once outside the loop for efficiency
    
            for year in range(self.year_start, year_end + 1):
                yearly_file_path = dir_to_save / f"{center}_SST_{year}.nc"
    
                if not force_download and yearly_file_path.exists():
                    print(f"{yearly_file_path} already exists. Loading existing file.")
                    try:
                        ds = xr.open_dataset(yearly_file_path).load()
                        # Rename coordinates if necessary
                        if 'latitude' in ds.coords and 'longitude' in ds.coords:
                            ds = ds.rename({"latitude": "Y", "longitude": "X", "valid_time": "T"})
                        combined_datasets.append(ds)
                        continue
                    except Exception as e:
                        print(f"Failed to load existing file {yearly_file_path}: {e}")
                        # If loading fails, attempt to re-download
    
                try:
                    request = {
                        "product_type": "monthly_averaged_reanalysis",
                        "variable": variable,
                        "year": str(year),
                        "month": months,  # Request all months at once
                        "time": "00:00",
                        "format": "netcdf",
                        "area": area,
                    }
    
                    print(f"Downloading SST for {year} from {center}...")
                    client.retrieve(dataset, request).download(str(yearly_file_path))
                except Exception as e:
                    print(f"Failed to download SST for {year}: {e}")
                    continue
    
                try:
                    with xr.open_dataset(yearly_file_path) as ds:
                        # Rename coordinates for consistency
                        if 'latitude' in ds.coords and 'longitude' in ds.coords:
                            ds = ds.rename({"latitude": "Y", "longitude": "X", "valid_time": "T"})
                        ds = ds.load()
                        combined_datasets.append(ds)
                except Exception as e:
                    print(f"Failed to process {yearly_file_path}: {e}")
                    continue
    
            if combined_datasets:
                print("Concatenating yearly datasets...")
                combined_ds = xr.concat(combined_datasets, dim="T")
                combined_ds = combined_ds.drop_vars(["number", "expver"]).squeeze()
                # combined_ds = combined_ds.rename({"lon": "X", "lat": "Y", "valid_time": "T"})
                combined_ds = combined_ds - 273.15
                combined_ds = combined_ds.isel(Y=slice(None, None, -1))
                combined_ds.to_netcdf(combined_output_path)
                print(f"Download finished. Combined SST dataset saved to {combined_output_path}")
    
                # Optionally, remove individual yearly files
                for year in range(self.year_start, year_end + 1):
                    single_file_path = dir_to_save / f"{center}_SST_{year}.nc"
                    if single_file_path.exists():
                        os.remove(single_file_path)
                        print(f"Deleted yearly file: {single_file_path}")
                return combined_ds
            else:
                print("No datasets were combined. Please check if downloads were successful.")

    def download_models(self,
            center_variable,
            area=[60, -180, -60, 180],
            force_download=False,
            year_start_forecast=None,
            year_end_forecast=None,
        ):
        """
        Download Sea Surface Temperature (SST) seasonal forecast forecast data for specified center-variable combinations,
        initialization month (current month), lead times, and years.
    
        Parameters:
            dir_to_save (str): Directory to save the downloaded files.
            center_variable (list): List of center-variable identifiers (e.g., ["ECMWF_51.SST"]).
            year_start_hindcast (int): Start year for forecast data.
            year_end_hindcast (int): End year for forecast data.
            area (list): Bounding box as [North, West, South, East] for clipping.
            ensemble_mean (str, optional): It can be 'median', 'mean', or None. Defaults to None.
            force_download (bool): If True, forces download even if file exists.
        """
        # 1. Set Current Date and Initialization Month
        if self.month_of_initialization is None:
            current_date = datetime.now() - relativedelta(months=1)
            month_of_initialization = current_date.month 
            print(f"Using current month as initialization month: {calendar.month_abbr[month_of_initialization]} ({month_of_initialization})")
        else:
            month_of_initialization = self.month_of_initialization
            
        # 2. Set Lead Time to Remaining Months if Not Provided
        if self.lead_time is None:
            # lead_time = list(range(1, 13 - month_of_initialization + 1))
            lead_time = [1, 2, 3, 4, 5]
            print(f"Lead time not provided. Using months: {lead_time} with {calendar.month_abbr[month_of_initialization]} as initialization month")
        else:
            # Validate lead_time argument
            if not isinstance(self.lead_time, list):
                raise ValueError("lead_time should be a list of integers representing months ahead.")
            if any(l < 1 or l > 12 for l in self.lead_time):
                raise ValueError("Each lead_time value should be between 1 and 12.")
            print(f"Using provided lead times: {lead_time}")
        
        # 3. Set Forecast Years to Current Year if Not Provided
        if year_start_forecast is None:
            year_start_forecast = datetime.now().year
            print(f"year_start_forecast not provided. Using current year: {year_start_forecast}")
        if year_end_forecast is None:
            year_end_forecast = datetime.now().year
            print(f"year_end_forecast not provided. Using current year: {year_end_forecast}")
    
        # 4. Define Variable Mapping (Only SST)
        variables_1 = {
            "SST": "sea_surface_temperature",
            "PRCP": "total_precipitation",
        }
    
        # 5. Extract Center and Variable
        try:
            center = [item.split(".")[0] for item in center_variable]
            variables = [item.split(".")[1] for item in center_variable]
        except IndexError:
            raise ValueError("center_variable should be in the format 'CENTER.VARIABLE', e.g., 'ECMWF_51.SST'.")
    
        # 6. Define Center Mappings
        centre = {
            "ECMWF_51": "ecmwf",
            "UKMO_602": "ukmo",
            "UKMO_603": "ukmo",
            "METEOFRANCE_8": "meteo_france",
            "DWD_21": "dwd",
            "DWD_2": "dwd",
            "CMCC_35": "cmcc",
            "CMCC_3": "cmcc",
            "NCEP_2": "ncep",
            "JMA_3": "jma",
            "ECCC_2": "eccc",
            "ECCC_3": "eccc",
        }
    
        # 7. Define System Mappings
        system = {
            "ECMWF_51": "51",
            "UKMO_602": "602",
            "UKMO_603": "603",
            "METEOFRANCE_8": "8",
            "DWD_21": "21",
            "DWD_2": "2",
            "CMCC_35": "35",
            "CMCC_3": "3",
            "NCEP_2": "2",
            "JMA_3": "3",
            "ECCC_2": "2",
            "ECCC_3": "3",
        }
    
        # 8. Select Centres, Systems, Variables
        try:
            selected_centre = [centre[k] for k in center]
            selected_system = [system[k] for k in center]
            selected_var = [k for k in variables]
        except KeyError as e:
            raise ValueError(f"Unsupported center identifier: {e}")
    
        # 9. Prepare Save Directory
        dir_to_save = Path(self.dir_to_save)
        os.makedirs(dir_to_save, exist_ok=True)
    
        # 10. Get Abbreviation for Initialization Month
        abb_mont_ini = calendar.month_abbr[int(month_of_initialization)]
    
        # 11. Construct Season String Based on Lead Times
        #    Handles month rollover using modulo arithmetic
        season_months = [((month_of_initialization + l - 1) % 12) + 1 for l in lead_time]
        season_str = "".join([calendar.month_abbr[month] for month in season_months])

        ds_centers = []
        # 13. Iterate Over Centres, Systems, Variables
        for cent, syst, k in zip(selected_centre, selected_system, selected_var):
            # # Only handle SST
            # if k != "SST":
            #     print(f"Skipping variable '{k}' as only SST is supported.")
            #     continue
    
            # 14. Construct File Prefix and Output Filename
            file_prefix = "forecast"  # Changed to 'forecast' as per user request
            output_filename = f"{file_prefix}_{cent}{syst}_{k}_{abb_mont_ini}Ic_{season_str}.nc"
            output_path = dir_to_save / output_filename
            print(output_path)
    
            # 15. Check if File Already Exists
            if not force_download and output_path.exists():
                print(f"File '{output_path}' already exists. Skipping download.")
                with xr.open_dataset(output_path) as ds:
                    ds_centers.append(ds)                
            else:

                # 12. Initialize CDS API Client
                try:
                    client = cdsapi.Client()
                except Exception as e:
                    print(f"Failed to initialize CDS API client: {e}")
                    return
                
                # 16. Define Dataset and Request Parameters
                dataset = "seasonal-monthly-single-levels"
        
                request = {
                    "originating_centre": cent,
                    "system": syst,
                    "variable": variables_1[k],
                    "product_type": ["monthly_mean"],
                    "year": [str(year) for year in range(year_start_forecast - 1, year_end_forecast + 1)],
                    "month": [str(month_of_initialization).zfill(2)],
                    "leadtime_month": lead_time,
                    "format": "netcdf",  # Corrected from 'data_format' to 'format'
                    "area": area,
                        }
        
                # 17. Download the Data to a Temporary File
                temp_file_path = dir_to_save / f"temp_{output_filename}"
                try:
                    print(f"Downloading SST from '{cent}' for initialization month '{calendar.month_abbr[month_of_initialization]}' "
                          f"and lead times {lead_time}...")
                    client.retrieve(dataset, request).download(str(temp_file_path))
                    print(f"Downloaded data to temporary file: '{temp_file_path}'")
                except Exception as e:
                    print(f"Failed to download data for '{cent} {k}': {e}")
                    continue
    
                
                # 18. Load and Process the Dataset
                try:
                    with xr.open_dataset(temp_file_path) as ds:
                        # 19. Convert SST from Kelvin to Celsius
                        if 'sst' in ds.variables:
                            ds['sst'] = ds['sst'] - 273.15
                            # ds = ds.rename({'sst': 'SST'})
                            print("Converted SST from Kelvin to Celsius.")
                        else:
                            ds = 1000*30*24*60*60*ds
                            # print("Variable 'sea_surface_temperature or total_precipitation' not found in the dataset.")
                            # continue  # Skip processing if SST is not found
        
                        # 20. Apply Ensemble Mean if Specified
                        if self.ensemble_mean in ["mean", "median"]:
                            if 'number' in ds.dims:
                                ds = getattr(ds, self.ensemble_mean)(dim="number")
                                print(f"Applied ensemble '{self.ensemble_mean}' across 'number' dimension.")
                            else:
                                print("Ensemble dimension 'number' not found. Skipping ensemble mean.")
        
                        # 21. Rename Coordinates for Consistency
                        if "indexing_time" in ds.coords:
                            ds = ds.rename({"latitude": "lat", "longitude": "lon", "indexing_time": "time"})
                        elif "forecast_reference_time" in ds.coords:
                            ds = ds.rename({"latitude": "lat", "longitude": "lon", "forecast_reference_time": "time"})
                        else:
                            print("Unexpected coordinate names in dataset. Skipping coordinate renaming.")
        
                        # # 22. Reverse Latitude if Necessary
                        # if 'lat' in ds.coords and ds.lat[0] > ds.lat[-1]:
                        #     ds = ds.sortby('lat')
                        #     print("Reversed latitude coordinates for consistency.")
        
                        # 23. Rename to Standard Dimensions
                        ds = ds.rename({"lon": "X", "lat": "Y", "time": "T"})
                        print("Renamed coordinates to standard dimensions: X, Y, T.")
                    
                        ds_centers.append(ds)
                        
                        # 24. Save the Processed Dataset
                        ds.to_netcdf(output_path)
                        print(f"Processed and saved dataset to '{output_path}'.")
    
                except Exception as e:
                    print(f"Failed to process dataset from '{temp_file_path}': {e}")
                    continue
                finally:
                    # 25. Remove the Temporary Download File
                    if temp_file_path.exists():
                        # pass
                        os.remove(temp_file_path)
                        print(f"Deleted temporary file: '{temp_file_path}'")
        ds_centers = xr.concat(ds_centers, dim="models").mean(dim="models").isel(Y=slice(None, None, -1))    
        print("All requested SST data has been processed.")
        return ds_centers

    def standardize_timeseries(self, ds):
        """Standardize the dataset over a specified climatology period."""
        if self.clim_year_start is not None and self.clim_year_end is not None:
            clim_slice = slice(str(self.clim_year_start), str(self.clim_year_end))
            
            clim_mean = ds.sel(T=clim_slice).groupby("T.month").mean(dim='T')
            clim_std = ds.sel(T=clim_slice).groupby("T.month").std(dim='T')
        else:
            clim_mean = ds.groupby("T.month").mean(dim='T')
            clim_std = ds.groupby("T.month").std(dim='T')
        # ds = (((ds.groupby("T.month") - clim_mean) / clim_std).isel(month=0, drop=True).squeeze())
        ds = ((ds.groupby("T.month") - clim_mean))#.isel(month=0, drop=True).squeeze())
        # ds = xr.where((ds > -10) & (ds < 10), ds, np.nan)
        return  ds

    def download_and_process(self, center_variable=None):
        if self.lead_time is None:
            lead_time = [1, 2, 3, 4, 5]
        else:
            lead_time = self.lead_time            
            
        sst_hist = self.download_sst_reanalysis(
            center="ERA5",
            area=[60, -180, -60, 180],
            force_download=False
            # year_start=self.year_start
        )

        sst_for = self.download_models(
            center_variable=["ECMWF_51.SST",
                             # "METEOFRANCE_8.SST",
                             # "DWD_21.SST",
                             # "CMCC_35.SST",
                             # "NCEP_2.SST",
                             # "ECCC_2.SST",
                             # "ECCC_3.SST",
                             # "UKMO_602.SST"
                            ],
            area=[60, -180, -60, 180],
            force_download=False,
            year_start_forecast=None,
            year_end_forecast=None
        )
        sst_for_ = sst_for.interp(
            X=sst_hist.coords['X'],
            Y=sst_hist.coords['Y'],
            method="nearest"
        )
        # Extract the base time
        base_time = pd.Timestamp(sst_for_['T'].values[-1])

        # Create new times by adding forecast months
        new_times = [base_time + pd.DateOffset(months=int(m)) for m in sst_for_['forecastMonth'].values]

        # Assign the new 'time' coordinate
        sst_for_ = sst_for_.assign_coords(forecastMonth=("forecastMonth", new_times))

        sst_for_ = sst_for_.drop_vars('T').squeeze().rename({'forecastMonth':'T'})
        
        concatenated_ds = xr.concat([sst_hist, sst_for_], dim='T')

        concatenated_ds_st = self.standardize_timeseries(concatenated_ds)

        
        concatenated_ds_st = concatenated_ds_st.rolling(T=3, center=False, min_periods=3).mean().isel(T=slice(np.max(lead_time),None))

        new_time = pd.DatetimeIndex([pd.Timestamp(t) + pd.DateOffset(months=12 - np.max(lead_time)) for t in concatenated_ds_st['T'].to_numpy()])

        ds_shifted = concatenated_ds_st.assign_coords(T=new_time)
        return concatenated_ds_st.to_array().drop_vars(['variable']).squeeze(), ds_shifted.to_array().drop_vars(['variable']).squeeze()

    def SOM(self, predictant, ddd, itrain, ireference_year):
        
        predictant['T'] = predictant['T'].astype('datetime64[ns]')

        if self.index_compute is not None:
            indices_dataset = self.calc_index(self.index_compute, ddd)
            
            # Get the unique years from the dataset's 'T' dimension
            predictant_ = xr.concat([predictant.isel(T=itrain), predictant.isel(T=ireference_year)], dim="T")
            
            # Extract years
            unique_years = np.unique(predictant_['T'].dt.year)
            
            # List to hold flattened data for each year
            yearly_data = []
            
            # Loop through each unique year
            for year in unique_years:
                # print("Processing year:", year)
                
                # Temporary list to hold data for all variables in this particular year
                index_data_list = []
                
                # For each data variable, extract and reshape data for this year
                for var_name in indices_dataset.data_vars:
                    # print("  Variable:", var_name)
                    yearly_sst = indices_dataset.sel(T=str(year))[var_name].to_numpy().reshape(12, -1)
                    index_data_list.append(yearly_sst)
                
                # Stack all variables for the year vertically, then flatten
                index_data_array = np.vstack(index_data_list)
                flattened_array = index_data_array.flatten()
                
                # Append the flattened array for this year to yearly_data
                yearly_data.append(flattened_array)
            
            # Convert the list of yearly arrays into a 2D NumPy array: (num_years, data_points_per_year)
            sst_yearly_flat = np.array(yearly_data)
            
            # Adjust the list of years to match the length of data we have
            years_complete = unique_years[:len(sst_yearly_flat)]
            
            # Handle missing values by filtering out rows (years) that contain NaNs
            valid_idx = ~np.isnan(sst_yearly_flat).any(axis=1)
            sst_yearly_flat = sst_yearly_flat[valid_idx]
            years_complete = years_complete[valid_idx]
            
            # Final scaled data (modify this part as needed if you plan on applying a scaler)
            sst_scaled = sst_yearly_flat
            
            # Define SOM parameters
            # n = int(np.sqrt(5*np.sqrt(sst_scaled.shape[1]))) - 2
            if set(self.some_grid_size) == {None}:
                som_grid_size = (2, 4)  # 10x10 grid
            else:
                som_grid_size = self.some_grid_size
            
            input_len = sst_scaled.shape[1]  # Number of features per year (12 * lat * lon)
            
            # Initialize the SOM
            som = MiniSom(
                         x=som_grid_size[0], 
                         y=som_grid_size[1], 
                         input_len=input_len, 
                         sigma=1.0, 
                         learning_rate=0.5, 
                         neighborhood_function='gaussian', 
                         random_seed=42
                         )
            
            # Initialize the weights
            som.random_weights_init(sst_scaled)
            
            # print("Training SOM...")
            # Train the SOM with 1000 iterations (adjust as needed)
            som.train_random(data=sst_scaled, num_iteration=1000)
            # print("SOM training completed.")
            # Map each year to its Best Matching Unit (BMU)
            bmu_indices = np.array([som.winner(x) for x in sst_scaled])
            bmu_x = bmu_indices[:, 0]
            bmu_y = bmu_indices[:, 1]
            
            # Create a DataFrame for easy plotting
            bmu_df = pd.DataFrame({
                'Year': years_complete,
                'BMU_X': bmu_x,
                'BMU_Y': bmu_y
            })
            
            # -------------------- 4. Identifying Similar Years -------------------- #
            # Define the reference year
            reference_year = int(np.unique(predictant.isel(T=ireference_year)['T'].dt.year))
            
            # Check if the reference year is in the dataset
            if reference_year not in years_complete:
                print(f"Reference year {reference_year} is not present in the dataset.")
            else:
                # Find the index of the reference year
                ref_index = np.where(years_complete == reference_year)[0][0]
                
                # Get the BMU of the reference year
                ref_bmu = bmu_indices[ref_index]
            
                
                # Find all years mapped to the same BMU
                similar_years = years_complete[(bmu_x == ref_bmu[0]) & (bmu_y == ref_bmu[1])]
                similar_years = similar_years[similar_years != reference_year]
                print(f"similar years for {reference_year} are {similar_years}")
        
        else:
            
            if self.define_extent is not None:
                lon_min, lon_max, lat_min, lat_max = self.define_extent
                ddd = ddd.sel(X=slice(lon_min, lon_max), Y=slice(lat_min, lat_max))
            
            # -------------------- 6a. Parallel Data Preparation with Dask -------------------- #

            
            dddd = ddd.fillna(0)
            
            # Convert Xarray DataArray to Dask Array
            # Adjust chunk sizes based on HPC resources and data dimensions
            sst_dask = dddd#.chunk({'T': 12, 'Y': 100, 'X': 100})  # Assuming 12 months per year
                      
            predictant_ = xr.concat([predictant.isel(T=itrain), predictant.isel(T=ireference_year)], dim="T")
            
            # Extract years
            unique_years_dask = np.unique(predictant_['T'].dt.year)

            
            # Flatten spatial dimensions and prepare data for SOM
            yearly_data_dask = []
            
            for year in unique_years_dask:
                yearly_sst = sst_dask.sel(T=str(year))
                
                # Check for complete 12 months
                if len(yearly_sst['T'].dt.month) == 12:
                    monthly_flat = yearly_sst.to_numpy().reshape(12, -1)
                    yearly_flat = monthly_flat.flatten()
                    yearly_data_dask.append(yearly_flat)
                else:
                    print(f"Year {year} is incomplete and will be skipped.")
            
            # Convert to NumPy array
            sst_yearly_flat_dask = np.array(yearly_data_dask)
            years_complete_dask = unique_years_dask[:len(sst_yearly_flat_dask)]
            
            
            # Handle missing values by removing any years with NaNs
            valid_idx = ~np.isnan(sst_yearly_flat_dask).any(axis=1)
            sst_yearly_flat_dask = sst_yearly_flat_dask[valid_idx]
            years_complete_dask = years_complete_dask[valid_idx]
            sst_scaled_dask = sst_yearly_flat_dask                

            
            # Initialize SOMoclu

            # n_rows, n_columns = 20, 20
            # my_code = np.float32(np.random.uniform(low=0.0, high=1.0, size=(n_columns * n_rows * 4)))
            
            # som = somoclu.Somoclu(n_columns, n_rows, initialcodebook=my_code, maptype="planar", gridtype="rectangular", neighborhood="gaussian", std_coeff=1.0, initialization=None, verbose=2)
            # som.train(mydata, epochs=10, scalecooling='exponential')
            # som.view_umatrix(bestmatches=True, labels=labels, filename="umatrix_output" )
            # som_state = som.get_surface_state()
            # my_bmus = som.get_bmus(som_state)

            if set(self.some_grid_size) == {None}:
                n_rows, n_columns = (3, 3)  # 10x10 grid
            else:
                n_rows, n_columns = self.some_grid_size

            # somoclu_instance = somoclu.Somoclu(n_columns, n_rows, data=sst_scaled_dask)
            somoclu_instance = somoclu.Somoclu(n_columns, n_rows)

            # Train SOM
            somoclu_instance.train(data=sst_scaled_dask)
            
            # Get BMUs
            bmus = somoclu_instance.bmus  # Shape: (num_samples, 2)
            
            # Map BMUs to years
            bmu_x_dask = bmus[:, 0]
            bmu_y_dask = bmus[:, 1]
            
            # Create DataFrame
            bmu_df_dask = pd.DataFrame({
                'Year': years_complete_dask,
                'BMU_X': bmu_x_dask,
                'BMU_Y': bmu_y_dask
            })
                    
            # Define the reference year
            reference_year = int(np.unique(predictant.isel(T=ireference_year)['T'].dt.year)) #unique_years_dask[ireference_year][0]  
            
            # Identify similar years 
            if reference_year not in bmu_df_dask['Year'].values:
                print(f"Reference year {reference_year} is not present in the dataset.")
            else:
                ref_bmu_dask = bmu_df_dask.loc[bmu_df_dask['Year'] == reference_year, ['BMU_X', 'BMU_Y']].values[0]
                similar_years_dask = bmu_df_dask[
                    (bmu_df_dask['BMU_X'] == ref_bmu_dask[0]) & 
                    (bmu_df_dask['BMU_Y'] == ref_bmu_dask[1])
                ]['Year'].values
            similar_years = similar_years_dask[similar_years_dask != reference_year]  
            print(f"similar years for {reference_year} are {similar_years}")
            
        return similar_years

    @staticmethod
    def calculate_tercile_probabilities(best_guess, error_variance, first_tercile, second_tercile, dof):
        """
        Calculates the tercile probabilities of a prediction based on Student's t-distribution.

        Parameters
        ----------
        best_guess : array-like
            Predicted values.
        error_variance : array-like
            Variance of prediction error.
        first_tercile : float
            First tercile value.
        second_tercile : float
            Second tercile value.
        dof : int
            Degrees of freedom for Student's t-distribution.

        Returns
        -------
        array
            Probabilities for each tercile category.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            first_t = (first_tercile - best_guess) / error_std
            second_t = (second_tercile - best_guess) / error_std
            pred_prob[0, :] = stats.t.cdf(first_t, df=dof)
            pred_prob[1, :] = stats.t.cdf(second_t, df=dof) - stats.t.cdf(first_t, df=dof)
            pred_prob[2, :] = 1 - stats.t.cdf(second_t, df=dof)
        return pred_prob
        
    def compute_model(self, predictant, ddd,  itrain, itest):

        dddd = self.SOM(predictant, ddd, itrain, itest)
        predictant['T'] = predictant['T'].astype('datetime64[ns]')
        sim_obs = []
        
        for i in np.array([str(i) for i in dddd]):
            tmp = predictant.sel(T=i)
            sim_obs.append(tmp)

        hindcast_det = xr.concat(sim_obs,dim="T").mean(dim="T").expand_dims({'T': predictant.isel(T=itest)['T'].values})      
        
        return hindcast_det

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det):

        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        
        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.333, 0.667], dim='T')
        error_variance = (Predictant - hindcast_det).var(dim='T')
        dof = 1
        
        hindcast_prob = xr.apply_ufunc(
            self.calculate_tercile_probabilities,
            hindcast_det,
            error_variance,
            terciles.isel(quantile=0).drop_vars('quantile'),
            terciles.isel(quantile=1).drop_vars('quantile'),
            input_core_dims=[('T',), (), (), ()],
            vectorize=True,
            kwargs={'dof': dof},
            dask='parallelized',
            output_core_dims=[('probability', 'T')],
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'probability': 3}},
        )
        
        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return hindcast_prob.transpose('probability', 'T', 'Y', 'X')

    
    def forecast(self, predictant, clim_year_start, clim_year_end, hindcast_det):
        # Set environment variables for reproducibility
        os.environ['PYTHONHASHSEED'] = '42'
        os.environ['OMP_NUM_THREADS'] = '1'  # Limit OpenMP threads
        
        # Define the reference year
        reference_year = int(np.unique(predictant.isel(T=-1)['T'].dt.year)) + 1
        
        predictant['T'] = predictant['T'].astype('datetime64[ns]')
        
        _, ddd = self.download_and_process()

        if self.index_compute is not None:
            indices_dataset = self.calc_index(self.index_compute, ddd)
                        
            # Extract years
            unique_years = np.append(np.unique(predictant['T'].dt.year), reference_year) 
                                    
            # List to hold flattened data for each year
            yearly_data = []
            
            # Loop through each unique year
            for year in unique_years:
                # print("Processing year:", year)
                
                # Temporary list to hold data for all variables in this particular year
                index_data_list = []
                
                # For each data variable, extract and reshape data for this year
                for var_name in indices_dataset.data_vars:
                    # print("  Variable:", var_name)
                    yearly_sst = indices_dataset.sel(T=str(year))[var_name].to_numpy().reshape(12, -1)
                    index_data_list.append(yearly_sst)
                
                # Stack all variables for the year vertically, then flatten
                index_data_array = np.vstack(index_data_list)
                flattened_array = index_data_array.flatten()
                
                # Append the flattened array for this year to yearly_data
                yearly_data.append(flattened_array)
            
            # Convert the list of yearly arrays into a 2D NumPy array: (num_years, data_points_per_year)
            sst_yearly_flat = np.array(yearly_data)
            
            # Adjust the list of years to match the length of data we have
            years_complete = unique_years[:len(sst_yearly_flat)]
            
            # Handle missing values by filtering out rows (years) that contain NaNs
            valid_idx = ~np.isnan(sst_yearly_flat).any(axis=1)
            sst_yearly_flat = sst_yearly_flat[valid_idx]
            years_complete = years_complete[valid_idx]
            
            # Final scaled data (modify this part as needed if you plan on applying a scaler)
            sst_scaled = sst_yearly_flat
            
            # Define SOM parameters
            if set(self.some_grid_size) == {None}:
                som_grid_size = (2, 4)  # 10x10 grid
            else:
                som_grid_size = self.some_grid_size
                
            input_len = sst_scaled.shape[1]  # Number of features per year (12 * lat * lon)
            
            # Initialize the SOM
            som = MiniSom(
                         x=som_grid_size[0], 
                         y=som_grid_size[1], 
                         input_len=input_len, 
                         sigma=1.0, 
                         learning_rate=0.5, 
                         neighborhood_function='gaussian', 
                         random_seed=42
                         )
            
            # Initialize the weights
            som.random_weights_init(sst_scaled)
            
            # print("Training SOM...")
            # Train the SOM with 1000 iterations (adjust as needed)
            som.train_random(data=sst_scaled, num_iteration=1000)
            # print("SOM training completed.")
            # Map each year to its Best Matching Unit (BMU)
            bmu_indices = np.array([som.winner(x) for x in sst_scaled])
            bmu_x = bmu_indices[:, 0]
            bmu_y = bmu_indices[:, 1]
            
            # Create a DataFrame for easy plotting
            bmu_df = pd.DataFrame({
                'Year': years_complete,
                'BMU_X': bmu_x,
                'BMU_Y': bmu_y
            })
                        
            # Check if the reference year is in the dataset
            if reference_year not in years_complete:
                print(f"Reference year {reference_year} is not present in the dataset.")
            else:
                # Find the index of the reference year
                ref_index = np.where(years_complete == reference_year)[0][0]
                
                # Get the BMU of the reference year
                ref_bmu = bmu_indices[ref_index]
            
                
                # Find all years mapped to the same BMU
                similar_years = years_complete[(bmu_x == ref_bmu[0]) & (bmu_y == ref_bmu[1])]
                similar_years = similar_years[similar_years != reference_year]
        else:
            
            if self.define_extent is not None:
                lon_min, lon_max, lat_min, lat_max = self.define_extent
                ddd = ddd.sel(X=slice(lon_min, lon_max), Y=slice(lat_min, lat_max))
            
            # -------------------- 6a. Parallel Data Preparation with Dask -------------------- #
            dddd = ddd.fillna(0)
            
            # Convert Xarray DataArray to Dask Array
            # Adjust chunk sizes based on HPC resources and data dimensions
            sst_dask = dddd#.chunk({'T': 12, 'Y': 100, 'X': 100})  # Assuming 12 months per year
            
            # Extract years
            unique_years_dask = np.unique(predictant['T'].dt.year)
                        
            # Flatten spatial dimensions and prepare data for SOM
            yearly_data_dask = []
            
            for year in unique_years_dask:
                yearly_sst = sst_dask.sel(T=str(year))
                
                # Check for complete 12 months
                if len(yearly_sst['T'].dt.month) == 12:
                    monthly_flat = yearly_sst.to_numpy().reshape(12, -1)
                    yearly_flat = monthly_flat.flatten()
                    yearly_data_dask.append(yearly_flat)
                else:
                    print(f"Year {year} is incomplete and will be skipped.")
            
            # Convert to NumPy array
            sst_yearly_flat_dask = np.array(yearly_data_dask)
            years_complete_dask = unique_years_dask[:len(sst_yearly_flat_dask)]
            
            
            # Handle missing values by removing any years with NaNs
            valid_idx = ~np.isnan(sst_yearly_flat_dask).any(axis=1)
            sst_yearly_flat_dask = sst_yearly_flat_dask[valid_idx]
            years_complete_dask = years_complete_dask[valid_idx]
            sst_scaled_dask = sst_yearly_flat_dask 
            
            # Initialize SOMoclu
            if set(self.some_grid_size) == {None}:
                n_rows, n_columns = (3, 3)  # 10x10 grid
            else:
                n_rows, n_columns = self.some_grid_size
            
            somoclu_instance = somoclu.Somoclu(n_columns, n_rows, data=sst_scaled_dask)
            
            # Train SOM
            somoclu_instance.train()
            
            # Get BMUs
            bmus = somoclu_instance.bmus  # Shape: (num_samples, 2)
            
            # Map BMUs to years
            bmu_x_dask = bmus[:, 0]
            bmu_y_dask = bmus[:, 1]
            
            # Create DataFrame
            bmu_df_dask = pd.DataFrame({
                'Year': years_complete_dask,
                'BMU_X': bmu_x_dask,
                'BMU_Y': bmu_y_dask
            })
                       
            # Identify similar years 
            if reference_year not in bmu_df_dask['Year'].values:
                print(f"Reference year {reference_year} is not present in the dataset.")
            else:
                ref_bmu_dask = bmu_df_dask.loc[bmu_df_dask['Year'] == reference_year, ['BMU_X', 'BMU_Y']].values[0]
                similar_years_dask = bmu_df_dask[
                    (bmu_df_dask['BMU_X'] == ref_bmu_dask[0]) & 
                    (bmu_df_dask['BMU_Y'] == ref_bmu_dask[1])
                ]['Year'].values
                
            print(similar_years_dask) 
            
            similar_years = similar_years_dask[similar_years_dask != reference_year]            
          
        sim_obs = []
        for i in np.array([str(i) for i in similar_years]):
            tmp = predictant.sel(T=i)
            sim_obs.append(tmp)
        forecast_det = xr.concat(sim_obs,dim="T").mean(dim="T").expand_dims({'T': [predictant.isel(T=-1)['T'].values]})

        index_start = predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = predictant.get_index("T").get_loc(str(clim_year_end)).stop
        
        rainfall_for_tercile = predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.333, 0.667], dim='T')
        error_variance = (predictant - hindcast_det).var(dim='T')
        dof = 1
        
        forecast_prob = xr.apply_ufunc(
            self.calculate_tercile_probabilities,
            forecast_det,
            error_variance,
            terciles.isel(quantile=0).drop_vars('quantile'),
            terciles.isel(quantile=1).drop_vars('quantile'),
            input_core_dims=[('T',), (), (), ()],
            vectorize=True,
            kwargs={'dof': dof},
            dask='parallelized',
            output_core_dims=[('probability', 'T')],
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'probability': 3}},
        )
        
        forecast_prob = forecast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        
        # Unset environment variables
        os.environ.pop('PYTHONHASHSEED', None)
        os.environ.pop('OMP_NUM_THREADS', None)  
        
        return ddd, similar_years, forecast_det, forecast_prob.drop_vars('T').squeeze().transpose('probability', 'Y', 'X')


    def composite_plot(self, predictant, clim_year_start, clim_year_end, hindcast_det, plot_predictor=True):

        clim_slice = slice(str(clim_year_start), str(clim_year_end))
        clim_mean = predictant.sel(T=clim_slice).mean(dim='T')
        
        ddd, similar_years, result_, _ = self.forecast(predictant, clim_year_start, clim_year_end, hindcast_det)
        
        reference_year = int(np.unique(ddd.isel(T=-1)['T'].dt.year))
        
        sim_all = []
        tmp = ddd.sel(T=str(reference_year))
        months = list(tmp['T'].dt.month.values)
        tmp['T'] = months  
        sim_all.append(tmp)
        
        sim_ = []
        pred_rain = []
        
        for i in np.array([str(i) for i in similar_years]):
            tmp = ddd.sel(T=i)
            months = list(tmp['T'].dt.month.values)
            tmp['T'] = months
            sim_.append(tmp)

            tmmp = 100*predictant.sel(T=i)/clim_mean
            pred_rain.append(tmmp)
                 
        sim__ = xr.concat(sim_, dim="year").assign_coords(year=('year', similar_years)).mean(dim="year") 
        sim_all.append(sim__)
        sim_all = xr.concat(sim_all, dim="output").assign_coords(output=('output', ['forecast year','composite analog']))

        pred_rain = xr.concat(pred_rain, dim='T')
        
        if plot_predictor:
            
            sim_all.plot(x="X",
                         y="Y", 
                         row="T", 
                         col="output",
                         figsize=(12, 20),  # Adjust size for better readability
                         cbar_kwargs={
                         "shrink": 0.3,  # Adjust shrink value for a smaller color bar
                         "aspect": 50,  # Change aspect ratio for a wider color bar
                         "pad": 0.05,  # Adjust space between plot and color bar
                         },
                         robust=True,  # Optional: ignores outliers for better contrast
                        )
        else:
            # # Define the custom colormap
            # colors_list = ["red", "grey", "blue"]  # Define color segments
            # bounds = [0, 80, 120, 200]  # Define boundaries for the categories
            # cmap = ListedColormap(colors_list)
            # norm = BoundaryNorm(bounds, cmap.N)
            
            # # Assuming 'dataset' contains your data
            # data_var = pred_rain.isel()#dataset["Precipitation_Flux"]
            
            # # Number of time steps
            # n_times = len(data_var['T'])
            # n_cols = 3  # Number of columns
            # n_rows = (n_times + n_cols - 1) // n_cols  # Calculate rows dynamically
            
            # fig, axes = plt.subplots(
            #     n_rows, n_cols, 
            #     figsize=(n_cols * 6, n_rows * 4), 
            #     subplot_kw={'projection': ccrs.PlateCarree()}
            # )
            
            # axes = axes.flatten()
            
            # # Loop through time steps and plot
            # for i, t in enumerate(data_var['T'].values):
            #     ax = axes[i]
            #     data = data_var.sel(T=t).squeeze()  # Select data for the current time step
                
            #     im = ax.pcolormesh(
            #         data['X'], data['Y'], data, cmap=cmap, norm=norm, 
            #         transform=ccrs.PlateCarree()
            #     )
            
            #     ax.coastlines()
            #     ax.add_feature(cfeature.LAND, edgecolor="black")
            #     ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
            #     ax.set_title(f"Time: {str(t)[:10]}")  # Format time as title
            
            # # Remove unused axes
            # for j in range(n_times, len(axes)):
            #     fig.delaxes(axes[j])
            
            # # Add colorbar outside the plot
            # cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
            # cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical")
            # cbar.set_label('Ratio to Normal [%]')
            # cbar.set_ticks([40, 100, 160])  # Set ticks at middle of each range
            # cbar.ax.set_yticklabels(['0-80 (Red)', '80-120 (Grey)', '>120 (Blue)'])  # Custom labels for each range
            
            # fig.suptitle("Ratio to Normal Precipitation", fontsize=16)
            # plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbar
            # plt.show()

            # Define the custom colormap
            colors_list = ["red", "grey", "blue"]  # Define color segments
            bounds = [0, 80, 120, 200]  # Define boundaries for the categories
            cmap = ListedColormap(colors_list)
            norm = BoundaryNorm(bounds, cmap.N)
            
            # Assuming 'dataset' contains your data
            data_var = pred_rain.sel(Y=slice(None,17))
            
            # Number of time steps
            n_times = len(data_var['T'])
            n_cols = 2  # Number of columns
            n_rows = (n_times + n_cols - 1) // n_cols  # Calculate rows dynamically
            
            fig, axes = plt.subplots(
                n_rows, n_cols, 
                figsize=(n_cols * 6, n_rows * 4), 
                subplot_kw={'projection': ccrs.PlateCarree()}
            )
            
            axes = axes.flatten()
            
            # Loop through time steps and plot
            for i, t in enumerate(data_var['T'].values):
                ax = axes[i]
                data = data_var.sel(T=t).squeeze()  # Select data for the current time step
                
                im = ax.pcolormesh(
                    data['X'], data['Y'], data, cmap=cmap, norm=norm, 
                    transform=ccrs.PlateCarree()
                )
            
                ax.coastlines()
                ax.add_feature(cfeature.LAND, edgecolor="black")
                ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
                ax.set_title(f"Season: {str(t)[:10]}")  # Format time as title
            
            # Remove unused axes
            for j in range(n_times, len(axes)):
                fig.delaxes(axes[j])
            
            # Add colorbar and adjust layout

            cbar = fig.colorbar(im, ax=axes[:n_times], orientation="horizontal", shrink=0.5, pad=0.5, aspect=40)
            cbar.set_label('')
            cbar.set_ticks([40, 100, 160])  # Set ticks at middle of each range
            cbar.ax.set_xticklabels(['0-80 (Dry)', '80-120 (Normal)', '>120 (Excess)'])  # Custom labels for each range
            
            fig.suptitle("Ratio to Normal Precipitation [%]", fontsize=16)

            plt.tight_layout()
            fig.subplots_adjust(top=0.9, bottom=0.15)
            plt.show() 

        return similar_years

    def plot_indices():
        
        pass


# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.patches import Ellipse

# # Sample data (replace with your actual data)
# x = np.array([-300, -250, -200, -100, 0, 100, 200, 300, 400, 500, 600, 700])
# y = np.array([-1.5, -1.2, -1.0, -0.8, 0.0, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
# years = np.array([1981, 1983, 1990, 2000, 2005, 2010, 2015, 2020, 2024, 2023, 2022, 2019])

# # Categories
# dry_spi = (y < -0.5)  # Example condition for Dry EEA SPI
# wet_spi = (y > 0.5)   # Example condition for Wet EEA SPI
# forecast_year = (years == 2024)

# # Marker styles
# colors = np.where(dry_spi, 'lightcoral', np.where(wet_spi, 'cornflowerblue', 'gray'))
# sizes = np.where(forecast_year, 200, 100)  # Make sizes the same length as x and y
# edgecolors = ['green' if year == 2024 else 'black' for year in years]

# # Create the plot
# fig, ax = plt.subplots(figsize=(10, 8))
# scatter = ax.scatter(x, y, c=colors, edgecolors=edgecolors, s=sizes, zorder=3)

# # Add ellipses (example: for highlighting data clusters)
# ellipse = Ellipse((-200, -1), 300, 2, alpha=0.2, color='lightcoral')
# ax.add_patch(ellipse)

# # Annotate data points
# for i, year in enumerate(years):
#     ax.text(x[i], y[i], str(year), fontsize=8, ha='center', va='center')

# # Add vertical lines
# ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
# ax.axvline(x=200, color='black', linestyle='-', linewidth=1)

# # Add labels and title
# ax.set_xlabel("OND ERA5 Indo-Warm Pool Heating Gradient [Wm-2]", fontsize=12)
# ax.set_ylabel("OND Eastern Horn SPI [ZS]", fontsize=12)
# ax.set_title("OND SPI vs Indo-Warm Pool Heating Gradient", fontsize=14)

# # Add legend
# legend_elements = [
#     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', markersize=10, label='Dry EEA SPI'),
#     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='cornflowerblue', markersize=10, label='Wet EEA SPI'),
#     plt.Line2D([0], [0], marker='o', color='black', markersize=10, label='1981-2023'),
#     plt.Line2D([0], [0], marker='o', color='green', markersize=10, label='2024 Forecast')
# ]
# ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

# # Grid and layout adjustments
# ax.grid(True, zorder=0)
# plt.tight_layout()
# plt.show()