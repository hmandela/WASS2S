"""Analog-ensemble seasonal forecasting.

Identifies historically similar years (analogs) by comparing climate
indices or EOF scores for the target period and composites their observed
rainfall to produce a forecast.

Classes
-------
WAS_Analog
    Full analog toolkit: downloads ERA5 reanalysis and NMME model data,
    computes SST/OLR/wind indices, performs dimensionality reduction, and
    applies one of several analog selection strategies.

    Analog methods available via ``compute_model``:

    - ``SOM`` — Self-Organising Maps (minisom).
    - ``Bias_Based`` — mean-bias distance in index space.
    - ``Corr_Based`` — Pearson correlation distance.
    - ``Pca_Based`` — PCA followed by Euclidean distance.
    - ``KMeans_Based`` — K-Means clustering in index space.
    - ``Agglomerative_Based`` — Agglomerative (Ward) clustering.

WAS_Analog__
    Compatibility name for the corrected ``WAS_Analog`` interface.
"""
# Core Python Libraries
import os
from pathlib import Path
import calendar
from datetime import datetime
from dateutil.relativedelta import relativedelta
from collections import defaultdict
import warnings

# Numerical and Data Manipulation Libraries
import numpy as np
import pandas as pd
import xarray as xr
from .was_probability_reference import climatological_terciles
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
from wass2s.was_bias_correction import *

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
        Analog method to use ("som", "cor_based", "bias_based, "pca_based", "kmeans_based", "agglomerative_based"). Default is "som".
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
    """

    def __init__(self, dir_to_save, year_start, year_forecast,
                 predictor_vars=[{'reanalysis_name': 'ERA5', 'model_name': 'ECMWF_51', 'variable': 'SST', 'area': [60, -180, -60, 180]},
                                {'reanalysis_name': 'ERA5', 'model_name': 'ECMWF_51', 'variable': 'SP', 'area': [60, -180, -60, 180]},
                                {'reanalysis_name': 'ERA5', 'model_name': 'ECMWF_51', 'variable': 'VGRD_850', 'area': [60, -180, -60, 180]}],
                 method_analog="som", month_of_initialization=None,
                 lead_time=None, ensemble_mean="mean", rolling=3, standardize=True, detrend=False, multivariateEOF=False,
                 eof_explained_var=0.95, clim_year_start=None, clim_year_end=None,
                 index_compute=None, some_grid_size=(None, None), some_learning_rate=0.5, radius=1.0,
                 some_neighborhood_function='gaussian', some_sigma=1.0, some_num_iteration=2000, dist_method="gamma",
                 n_clusters=4, affinity='euclidean', linkage='ward'):
        
        self.dir_to_save = dir_to_save
        self.year_start = year_start
        self.year_forecast = year_forecast
        self.predictor_vars = [dict(spec) for spec in predictor_vars]
        self.method_analog = str(method_analog).lower()
        self.month_of_initialization = month_of_initialization
        self.lead_time = lead_time
        self.ensemble_mean = ensemble_mean
        self.eof_explained_var = eof_explained_var
        self.multivariateEOF = multivariateEOF
        self.rolling = rolling
        self.standardize = standardize
        self.detrend = detrend
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
        self._processed_analog_cache = None

    def calc_index(self, indices, sst):
        """Area-weighted SST box indices on either longitude convention.

        Values are spatial SST means/differences; training-only temporal
        anomaly scaling belongs to the feature engine, not this helper.
        """
        boxes = {
            "NINO34": (-170, -120, -5, 5), "NINO12": (-90, -80, -10, 0),
            "NINO3": (-150, -90, -5, 5), "NINO4": (160, -150, -5, 5),
            "NINO_Global": (160, -80, -10, 5), "TNA": (-55, -15, 5, 25),
            "TSA": (-30, 10, -20, 0), "NAT": (-40, -20, 5, 20),
            "SAT": (-15, 5, -20, 5), "WTIO": (50, 70, -10, 10),
            "SETIO": (90, 110, -10, 0), "MB": (0, 50, 30, 42),
            "M1": (-50, 5, -50, -25), "M2": (-75, -10, 25, 50),
            "M3": (-175, -125, 25, 50), "M4": (-175, -125, -50, -25),
        }
        if isinstance(sst, xr.Dataset):
            names = list(sst.data_vars)
            if len(names) != 1:
                raise ValueError("SST indices require one physical data variable")
            sst = sst[names[0]]
        if not {"Y", "X"}.issubset(sst.dims):
            raise ValueError("SST indices require Y and X coordinates")
        unknown = set(indices) - (set(boxes) | {"TASI", "DMI"})
        if unknown:
            raise ValueError(f"Unknown SST indices: {sorted(unknown)}")
        lon = ((sst.X + 180.0) % 360.0) - 180.0
        # A source may include both cyclic endpoints; count that meridian once.
        _, unique = np.unique(np.asarray(lon), return_index=True)
        sst = sst.isel(X=np.sort(unique)); lon = lon.isel(X=np.sort(unique))
        cache = {}
        def box(name):
            if name in cache:
                return cache[name]
            west, east, south, north = boxes[name]
            xmask = (lon >= west) & (lon <= east) if west <= east else (lon >= west) | (lon <= east)
            region = sst.where(xmask & (sst.Y >= south) & (sst.Y <= north), drop=True)
            if not region.sizes.get("X", 0) or not region.sizes.get("Y", 0):
                result = xr.full_like(sst.mean(("Y", "X"), skipna=True), np.nan)
            else:
                area = np.cos(np.deg2rad(region.Y)).clip(min=0)
                result = region.weighted(area).mean(("Y", "X"), skipna=True)
            cache[name] = result
            return result
        output = {}
        for name in indices:
            value = box("NAT") - box("SAT") if name == "TASI" else (box("WTIO") - box("SETIO") if name == "DMI" else box(name))
            output[name] = value.rename(name)
        return xr.Dataset(output)

    def _postprocess_ersst(self, ds, var_name):
        """Post-process ERSST dataset."""
        ds = ds.drop_vars('zlev', errors='ignore').squeeze()
        keep_vars = [var_name, 'T', 'X', 'Y']
        drop_vars = [v for v in ds.variables if v not in keep_vars]
        return ds.drop_vars(drop_vars, errors="ignore")
    def download_reanalysis_(self, force_download=False):
            """
            Download reanalysis data (ERA5 and NOAA ERSST).
            Saves raw MONTHLY data (1-12) without aggregation, preserving seasonality.
            """
            year_end = self.year_forecast
            
            # We download ALL months to preserve the full seasonal cycle for the analog search
            months_int = list(range(1, 13))
            # Naming convention for the season string in filename
            season_str = "JanDec" 
            
            variables = [item['variable'] for item in self.predictor_vars]
            centers = [item['reanalysis_name'] for item in self.predictor_vars]
            areas = [item['area'] for item in self.predictor_vars]
            
            # Variable Mappings
            variables_1 = {
                "PRCP": "total_precipitation", "TEMP": "2m_temperature",
                "TMAX": "maximum_2m_temperature_in_the_last_24_hours",
                "TMIN": "minimum_2m_temperature_in_the_last_24_hours",
                "UGRD10": "10m_u_component_of_wind", "VGRD10": "10m_v_component_of_wind",
                "SST": "sea_surface_temperature", "SLP": "mean_sea_level_pressure",
                "DSWR": "surface_solar_radiation_downwards", "DLWR": "surface_thermal_radiation_downwards",
                "NOLR": "top_net_thermal_radiation",
            }
            variables_2 = {
                "HUSS_1000": "specific_humidity", "HUSS_925": "specific_humidity", "HUSS_850": "specific_humidity",
                "UGRD_1000": "u_component_of_wind", "UGRD_925": "u_component_of_wind", "UGRD_850": "u_component_of_wind",
                "VGRD_1000": "v_component_of_wind", "VGRD_925": "v_component_of_wind", "VGRD_850": "v_component_of_wind",
            }
        
            dir_to_save = Path(self.dir_to_save)
            dir_to_save.mkdir(parents=True, exist_ok=True)
            store_file_path = {}
        
            for center, var, area in zip(centers, variables, areas):
                # Filename includes "JanDec" to indicate full monthly data
                combined_output_path = dir_to_save / f"{center}_{var}_{self.year_start}_{year_end}_{season_str}_{area[0]}{area[1]}{area[2]}{area[3]}.nc" 
        
                if not force_download and combined_output_path.exists():
                    print(f"{combined_output_path} exists. Skipping download.")
                    store_file_path[var] = xr.open_dataset(combined_output_path)
                    continue
        
                # =================================================================
                # CASE 1: NOAA ERSST (Direct NCEI Download)
                # =================================================================
                if f"{center}.{var}" == "NOAA.SST":
                    try:
                        print(f"Starting NCEI Direct Download for NOAA ERSST: {self.year_start}-{year_end}")
                        cache_dir = dir_to_save / "ersst_cache"
                        cache_dir.mkdir(exist_ok=True)
                        
                        downloaded_paths = []
                        # Using v5 as standard, change URL to v6 if specifically required
                        base_url = "https://www.ncei.noaa.gov/data/sea-surface-temperature-extended-reconstructed/v6/access/" 
        
                        for s_year in range(self.year_start, year_end + 1):
                            for month in months_int:
                                fname = f"ersst.v6.{s_year}{month:02d}.nc"
                                local_path = cache_dir / fname
                                
                                if not local_path.exists():
                                    url = f"{base_url}/{fname}"
                                    # print(f"Downloading {fname}...") 
                                    r = requests.get(url, timeout=30)
                                    r.raise_for_status()
                                    with open(local_path, 'wb') as f:
                                        f.write(r.content)
                                downloaded_paths.append(local_path)
        
                        # Process ERSST
                        # decode_times=True ensures we interpret the time correctly immediately
                        ds = xr.open_mfdataset(downloaded_paths, combine='by_coords')
                        ds = ds[["sst"]].drop_vars("lev", errors="ignore").squeeze()
        
                        # Standardize Coordinates (0-360 -> -180/180)
                        if 'lon' in ds.coords:
                            ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
                            ds = ds.sortby('lon')
                            
                        # Rename to standard (X, Y, T, SST)
                        rename_dict = {}
                        if 'lat' in ds.coords: rename_dict['lat'] = 'Y'
                        if 'lon' in ds.coords: rename_dict['lon'] = 'X'
                        if 'time' in ds.coords: rename_dict['time'] = 'T'
                        if 'sst' in ds.variables: rename_dict['sst'] = 'SST'
                        ds = ds.rename(rename_dict)
                            
                        # Slice Area [N, W, S, E]
                        # Ensure dimensions are sorted before slicing
                        ds = ds.sortby(['X', 'Y'])
                        # Note: ERSST Lat is usually Ascending (S->N), so slice(S, N) i.e., slice(area[2], area[0])
                        ds = ds.sel(X=slice(area[1], area[3]), Y=slice(area[2], area[0]))
        
                        # NO AGGREGATION: Just ensure time is sorted
                        ds = ds.sortby('T')
                            
                        # Final Cleanup
                        # The class typically expects 'sst' (lowercase) in the final dataset
                        if 'SST' in ds.variables:
                            ds = ds.rename({"SST": "sst"}) 
                            
                        ds.to_netcdf(combined_output_path)
                        store_file_path[var] = ds
                        print(f"Saved NOAA ERSST data to {combined_output_path}")
                        continue
                    except Exception as e:
                        print(f"Failed to download NOAA.SST via NCEI: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
        
                # =================================================================
                # CASE 2: ERA5 / CDSAPI
                # =================================================================
                combined_datasets = []
                client = cdsapi.Client()
                
                is_pressure = var in variables_2
                dataset_name = "reanalysis-era5-pressure-levels-monthly-means" if is_pressure else "reanalysis-era5-single-levels-monthly-means"
                api_var = variables_2[var] if is_pressure else variables_1.get(var, var)
                press_level = var.split("_")[1] if is_pressure else None
        
                for year in range(self.year_start, year_end + 1):
                    yearly_file_path = dir_to_save / f"{center}_{var}_{year}.nc"
                    
                    if not force_download and yearly_file_path.exists():
                        try:
                            ds = xr.open_dataset(yearly_file_path).load()
                            if 'latitude' in ds.coords: ds = ds.rename({"latitude": "Y", "longitude": "X", "valid_time": "T"})
                            combined_datasets.append(ds)
                            continue
                        except Exception as e:
                            print(f"Failed to load {yearly_file_path}: {e}")
                    
                    try:
                        request = {
                            "product_type": "monthly_averaged_reanalysis",
                            "variable": api_var,
                            "year": str(year),
                            "month": [f"{m:02d}" for m in months_int],
                            "time": "00:00",
                            "area": area,
                            "format": "netcdf",
                        }
                        if is_pressure:
                            request["pressure_level"] = press_level
        
                        print(f"Downloading {var} for {year} from {center}...")
                        client.retrieve(dataset_name, request).download(str(yearly_file_path))
                        
                        with xr.open_dataset(yearly_file_path) as ds:
                            if 'latitude' in ds.coords: ds = ds.rename({"latitude": "Y", "longitude": "X", "valid_time": "T"})
                            ds = ds.load()
                            combined_datasets.append(ds)
                    except Exception as e:
                        print(f"Failed to download/process {var} for {year}: {e}")
                        continue
                
                if combined_datasets:
                    print(f"Concatenating {var} datasets...")
                    combined_ds = xr.concat(combined_datasets, dim="T")
                    combined_ds = combined_ds.drop_vars(["number", "expver"], errors="ignore").squeeze()
                    
                    # Unit Conversions
                    if var in ["TMIN", "TEMP", "TMAX", "SST"]:
                        combined_ds = combined_ds - 273.15
                    elif var == "PRCP":
                        combined_ds = combined_ds * 1000 
                    elif var in ["DSWR", "DLWR", "NOLR"]:
                        combined_ds = combined_ds / 86400
                    elif var == "SLP":
                        combined_ds = combined_ds / 100
                    
                    combined_ds = combined_ds.isel(Y=slice(None, None, -1))
                    
                    # NO AGGREGATION HERE either
                    combined_ds = combined_ds.sortby('T')
        
                    store_file_path[var] = combined_ds
                    combined_ds.to_netcdf(combined_output_path)
                    print(f"Saved combined dataset to {combined_output_path}")
                    
                    # Cleanup yearly files
                    for year in range(self.year_start, year_end + 1):
                        single_file_path = dir_to_save / f"{center}_{var}_{year}.nc"
                        if single_file_path.exists():
                            try: os.remove(single_file_path)
                            except: pass
                else:
                    print(f"No data combined for {var}. Check download success.")
        
            return store_file_path

    
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
            "SP": "surface_pressure",
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
        now = pd.Timestamp.now()
        curr_yr = int(year_end) - getattr(self, "target_year_offset", 0)
        curr_mon = int(self.month_of_initialization) if self.month_of_initialization is not None else int((now - pd.DateOffset(months=1)).month)
        store_file_path = {}

        for center, var, area in zip(centers, variables, areas):
            combined_output_path = dir_to_save / f"analograw2_{center}_{var}_{self.year_start}_{year_end}_{season}_{area[0]}{area[1]}{area[2]}{area[3]}.nc" 
            if not force_download and combined_output_path.exists():
                print(f"{combined_output_path} exists. Skipping download.")
                store_file_path[var] = xr.open_dataset(combined_output_path)
                continue

            if f"{center}.{var}" == "NOAA.SST":
                
                print(f"Starting download for NOAA ERSST (v6) for {self.year_start}-{year_end}")
                # Create a cache dir for individual monthly files so we don't redownload them unnecessarily
                cache_dir = dir_to_save / "ersst_cache"
                cache_dir.mkdir(exist_ok=True)
                
                # Identify exact list of year-months required for the seasonal request
                # This ensures we handle cross-year (e.g. NDJ) correctly
                required_files = []
                season_months = [int(m) for m in range(1, 13)]
                pivot = season_months[0]
                
                # Loop through the "Seasonal Years"
                for s_year in range(self.year_start, year_end + 1):
                    for month in season_months:
                        cal_year = s_year

                        # if month >= pivot:
                        #     cal_year = s_year
                        # else:
                        #     cal_year = s_year + 1

                        # if cal_year == year_end and month >= pivot:
                        #     continue
                        # if cal_year > year_end:
                        #     continue
                        if (cal_year > curr_yr) or (cal_year == curr_yr and month > curr_mon):
                            continue
                        
                        # ERSST v6 filename format: ersst.v6.YYYYMM.nc
                        fname = f"ersst.v6.{cal_year}{month:02d}.nc"
                        required_files.append((cal_year, month, fname))

                # Download Loop
                # Base URL for ERSST 'v6'.
                base_url = "https://www.ncei.noaa.gov/data/sea-surface-temperature-extended-reconstructed/v6/access/"
                
                downloaded_paths = []
                
                try:
                    for (yr, mn, fname) in required_files:
                        local_path = cache_dir / fname
                        if not local_path.exists():
                            # Construct full URL
                            url = f"{base_url}/{fname}"
                            print(f"Downloading {fname}...")
                            r = requests.get(url, timeout=30)
                            r.raise_for_status()
                            with open(local_path, 'wb') as f:
                                f.write(r.content)
                        downloaded_paths.append(local_path)
                    
                    print("All files downloaded. merging and processing...")

                    # Open all files
                    # use_cftime=True is safer for long historical records
                    ds = xr.open_mfdataset(downloaded_paths)#, combine='by_coords', decode_times=True)
                    ds = ds[["sst"]].drop_vars("lev", errors="ignore")
                    for dim in ("lev", "zlev"):
                        if dim in ds.dims and ds.sizes[dim] == 1:
                            ds = ds.isel({dim: 0}, drop=True)

                    # --- PRE-PROCESSING SPECIFIC TO ERSST ---
                    
                    # 1. Standardize Coordinates (Lat/Lon/Time)
                    # ERSST uses 'lat', 'lon', 'time' usually.
                    # ERSST Longitude is 0 to 360. We likely want -180 to 180 to match ERA5/bounding box.
                    if 'lon' in ds.coords:
                        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
                        ds = ds.sortby('lon')
                    
                    # 2. Rename to internal standard (X, Y)
                    rename_dict = {}
                    if 'lat' in ds.coords: rename_dict['lat'] = 'Y'
                    if 'lon' in ds.coords: rename_dict['lon'] = 'X'
                    if 'time' in ds.variables: rename_dict['time'] = 'T' # Rename var to capital SST for consistency
                    if 'SST' in ds.variables: rename_dict['SST'] = 'sst' # Rename var to capital SST for consistency
            
                    
                    ds = ds.rename(rename_dict)

                    # print(ds)
                    
                    # 3. Slice the Area
                    # Area format: [N, W, S, E]
                    lat_max, lon_min, lat_min, lon_max = area
                    
                    # Ensure X and Y are sorted for slicing
                    ds = ds.sortby(['X', 'Y'])
                    ds = ds.sel(X=slice(lon_min, lon_max), Y=slice(lat_min, lat_max))

                        
                    ds["T"] = ds["T"].astype("datetime64[ns]")
                    # print(ds)
                    # ds = ds.rename({'SST': 'sst'})
                    store_file_path[var] = ds
                    ds.to_netcdf(combined_output_path)
                    print(f"Saved NOAA ERSST data to {combined_output_path}")
                    continue
                    
                except Exception as e:
                    print(f"Failed to download or process NOAA/SST: {str(e)}")
                    # Optional: clean up partial downloads if needed, or leave for retry
                    import traceback
                    traceback.print_exc()
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
                combined_ds = combined_ds.drop_vars(["number", "expver"], errors="ignore")
                for dim in list(combined_ds.dims):
                    if dim not in ("T", "Y", "X") and combined_ds.sizes[dim] == 1:
                        combined_ds = combined_ds.isel({dim: 0}, drop=True)
                
                if var in ["TMIN", "TEMP", "TMAX", "SST"]:
                    combined_ds = combined_ds - 273.15
                elif var == "PRCP":
                    combined_ds = combined_ds * 1000 * combined_ds["T"].dt.days_in_month
                elif var in ["DSWR", "DLWR", "NOLR"]:
                    combined_ds = combined_ds / 86400
                elif var == "SLP":
                    combined_ds = combined_ds / 100
                
                combined_ds = combined_ds.sortby("Y")
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

    @staticmethod
    def _model_precipitation_totals(dataset, initialization_month):
        """Integrate C3S monthly mean rates for each exact valid calendar month."""
        time_name = next((name for name in ("forecast_reference_time", "indexing_time", "time", "T") if name in dataset.coords), None)
        lead_name = next((name for name in ("forecastMonth", "leadtime_month") if name in dataset.dims), None)
        if time_name is None or lead_name is None:
            raise ValueError("C3S monthly precipitation needs initialization dates and lead coordinates")
        dates = pd.DatetimeIndex(np.atleast_1d(dataset[time_name].values))
        leads = np.asarray(dataset[lead_name].values, int)
        days = np.array([[(date + pd.DateOffset(months=int(lead) - 1)).days_in_month for lead in leads] for date in dates])
        if time_name in dataset.dims:
            factor = xr.DataArray(days, dims=(time_name, lead_name), coords={time_name: dataset[time_name], lead_name: dataset[lead_name]})
        else:
            factor = xr.DataArray(days[0], dims=lead_name, coords={lead_name: dataset[lead_name]})
        output = dataset.copy()
        for name, array in dataset.data_vars.items():
            if lead_name not in array.dims:
                continue
            unit = str(array.attrs.get("units", "")).replace(" ", "").lower()
            if unit in {"ms-1", "ms**-1", "m/s", "ms^-1"}:
                scale = 1000 * 86400
            elif unit in {"kgm-2s-1", "kgm**-2s**-1", "mms-1", "mm/s"}:
                scale = 86400
            elif unit in {"mmday-1", "mm/day", "mmd-1"}:
                scale = 1
            else:
                raise ValueError(f"Unsupported C3S precipitation rate units: {array.attrs.get('units')!r}")
            output[name] = array * factor * scale
            output[name].attrs = dict(array.attrs, units="mm", cell_methods="time: sum")
        return output

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
            "SST": "sea_surface_temperature", "SLP": "mean_sea_level_pressure", "SP": "surface_pressure",
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

        month_of_initialization = (pd.Timestamp.now() - pd.DateOffset(months=1)).month if self.month_of_initialization is None else self.month_of_initialization
        lead_time = [1, 2, 3, 4, 5] if self.lead_time is None else self.lead_time
        if not lead_time or any(int(l) != l or l < 0 or l > 11 for l in lead_time):
            raise ValueError("lead_time must contain zero-based integer month offsets in 0..11.")
        year_forecast = pd.Timestamp.now().year if self.year_forecast is None else self.year_forecast
        hindcast_start = getattr(self, "hindcast_year_start", None) or self.clim_year_start or self.year_start
        hindcast_end = getattr(self, "hindcast_year_end", None) or self.clim_year_end or (year_forecast - 1)
        target_offset = getattr(self, "target_year_offset", 0)
        year_forecast = int(year_forecast) - target_offset
        hindcast_start = int(hindcast_start) - target_offset
        hindcast_end = int(hindcast_end) - target_offset
        hindcast_years = [str(year) for year in range(hindcast_start, hindcast_end + 1)]
        member_tag = "members" if self.ensemble_mean is None else str(self.ensemble_mean)

        dir_to_save = Path(self.dir_to_save)
        dir_to_save.mkdir(parents=True, exist_ok=True)
        abb_mont_ini = calendar.month_abbr[int(month_of_initialization)]
        season_months = [((month_of_initialization + l - 1) % 12) + 1 for l in lead_time]
        season_str = "".join(calendar.month_abbr[m] for m in season_months)

        store_file_path = {}
        store_hdcst_file_path = {}

        for cent, syst, var, area in zip(selected_centres, selected_systems, variables, areas):
            forecast_file = dir_to_save / f"analograw2_{member_tag}_forecast_{cent}{syst}_{var}_{year_forecast}_{abb_mont_ini}Ic_{season_str}_{lead_time[0]}_{area[0]}{area[1]}{area[2]}{area[3]}.nc"
            hindcast_file = dir_to_save / f"analograw2_{member_tag}_hindcast_{cent}{syst}_{var}_{hindcast_start}_{hindcast_end}{abb_mont_ini}Ic_{season_str}_{lead_time[0]}_{area[0]}{area[1]}{area[2]}{area[3]}.nc"
               
            if not force_download and forecast_file.exists():
                print(f"Forecast file {forecast_file} exists. Skipping download.")
                ds = xr.load_dataset(forecast_file)
                store_file_path[var] = ds
                ds.close()
            else:
                try:
                    dataset = "seasonal-monthly-pressure-levels" if var in variables_map and "_" in var and var.split("_")[0] in ("HUSS", "UGRD", "VGRD") else "seasonal-monthly-single-levels"
                    forecast_request = {
                        "originating_centre": cent,
                        "system": syst,
                        "variable": variables_map[var],
                        "product_type": ["monthly_mean"],
                        "year": [str(year_forecast)],
                        "month": [f"{month_of_initialization:02d}"],
                        "leadtime_month": [int(value) + 1 for value in lead_time],
                        "data_format": "netcdf",
                        "area": area,
                    }
                    if var in variables_map and "_" in var and var.split("_")[0] in ("HUSS", "UGRD", "VGRD"):
                        forecast_request["pressure_level"] = var.split("_")[1]

                    client = cdsapi.Client()
                    client.retrieve(dataset, forecast_request).download(str(forecast_file))
                    print(f"Downloaded forecast: {forecast_file}")

                    ds_forecast = xr.open_dataset(forecast_file)

                    if var in ["TMIN", "TEMP", "TMAX", "SST"]:
                        ds_forecast = ds_forecast - 273.15
                    elif var == "PRCP":
                        ds_forecast = self._model_precipitation_totals(ds_forecast, month_of_initialization)
                    elif var == "SLP":
                        ds_forecast = ds_forecast / 100
                    elif var in ["DSWR", "DLWR", "NOLR"]:
                        ds_forecast = ds_forecast / 86400

                    if self.ensemble_mean and "number" in ds_forecast.dims:
                        ds_forecast = getattr(ds_forecast, self.ensemble_mean)(dim="number", skipna=True)

                    ds_forecast = ds_forecast.sortby("latitude")
                    rename_dict = {
                        "latitude": "Y",
                        "longitude": "X",
                        "time": "T" if "time" in ds_forecast.coords else None,
                        "indexing_time": "T" if "indexing_time" in ds_forecast.coords else None,
                        "forecast_reference_time": "T" if "forecast_reference_time" in ds_forecast.coords else None,
                    }
                    rename_dict = {k: v for k, v in rename_dict.items() if v is not None and k in ds_forecast.coords}
                    ds_forecast = ds_forecast.rename(rename_dict)
                    if "forecastMonth" not in ds_forecast.dims and "leadtime_month" in ds_forecast.dims:
                        ds_forecast = ds_forecast.rename({"leadtime_month": "forecastMonth"})
                    ds_forecast = ds_forecast.assign_coords(forecastMonth=ds_forecast.forecastMonth.astype(int) - 1)
                    ds_forecast.attrs["analog_lead_convention"] = "zero_based"
                    ds_forecast.attrs["analog_processing_version"] = "2026-10-01-calendar-members"


                    if var in variables_map and "_" in var and var.split("_")[0] in ("HUSS", "UGRD", "VGRD"):
                        ds_forecast = ds_forecast.drop_vars("pressure_level", errors="ignore")
                        if "pressure_level" in ds_forecast.dims:
                            ds_forecast = ds_forecast.isel(pressure_level=0, drop=True)
                    
    
                    ds_forecast.load()
                    ds_forecast.close()
                    os.remove(str(forecast_file))
                    ds_forecast.to_netcdf(str(forecast_file))
                    print(f"Saved processed hindcast to {forecast_file}")
                    store_file_path[var] = ds_forecast

                except Exception as e:
                    print(f"Failed to process {var} for {cent}{syst}: {e}")
                    continue

            if not force_download and hindcast_file.exists():
                print(f"Hindcast file {hindcast_file} exists. Skipping download.")
                ds = xr.load_dataset(hindcast_file)
                store_hdcst_file_path[var] = ds
                ds.close()
            else:
                try:
                    dataset = "seasonal-monthly-pressure-levels" if var in variables_map and "_" in var and var.split("_")[0] in ("HUSS", "UGRD", "VGRD") else "seasonal-monthly-single-levels"
                    hindcast_request = {
                        "originating_centre": cent,
                        "system": syst,
                        "variable": variables_map[var],
                        "product_type": ["monthly_mean"],
                        "year": hindcast_years,
                        "month": [f"{month_of_initialization:02d}"],
                        "leadtime_month": [int(value) + 1 for value in lead_time],
                        "data_format": "netcdf",
                        "area": area,
                    }
                    if var in variables_map and "_" in var and var.split("_")[0] in ("HUSS", "UGRD", "VGRD"):
                        hindcast_request["pressure_level"] = var.split("_")[1]
        
                    client = cdsapi.Client()
                    client.retrieve(dataset, hindcast_request).download(str(hindcast_file))
                    print(f"Downloaded hindcast: {hindcast_file}")

                    ds_hindcast = xr.open_dataset(hindcast_file)
                    if var in ["TMIN", "TEMP", "TMAX", "SST"]:
                        ds_hindcast = ds_hindcast - 273.15
                    elif var == "PRCP":
                        ds_hindcast = self._model_precipitation_totals(ds_hindcast, month_of_initialization)
                    elif var == "SLP":
                        ds_hindcast = ds_hindcast / 100
                    elif var in ["DSWR", "DLWR", "NOLR"]:
                        ds_hindcast = ds_hindcast / 86400

                    if self.ensemble_mean and "number" in ds_hindcast.dims:
                        ds_hindcast = getattr(ds_hindcast, self.ensemble_mean)(dim="number", skipna=True)                        

                    ds_hindcast = ds_hindcast.sortby("latitude")
                    rename_dict = {
                        "latitude": "Y",
                        "longitude": "X",
                        "time": "T" if "time" in ds_hindcast.coords else None,
                        "indexing_time": "T" if "indexing_time" in ds_hindcast.coords else None,
                        "forecast_reference_time": "T" if "forecast_reference_time" in ds_hindcast.coords else None,
                    }
                    rename_dict = {k: v for k, v in rename_dict.items() if v is not None and k in ds_hindcast.coords}
                    ds_hindcast = ds_hindcast.rename(rename_dict)
                    if "forecastMonth" not in ds_hindcast.dims and "leadtime_month" in ds_hindcast.dims:
                        ds_hindcast = ds_hindcast.rename({"leadtime_month": "forecastMonth"})
                    ds_hindcast = ds_hindcast.assign_coords(forecastMonth=ds_hindcast.forecastMonth.astype(int) - 1)
                    ds_hindcast.attrs["analog_lead_convention"] = "zero_based"
                    ds_hindcast.attrs["analog_processing_version"] = "2026-10-01-calendar-members"

                    if var in variables_map and "_" in var and var.split("_")[0] in ("HUSS", "UGRD", "VGRD"):
                        ds_hindcast = ds_hindcast.drop_vars("pressure_level", errors="ignore")
                        if "pressure_level" in ds_hindcast.dims:
                            ds_hindcast = ds_hindcast.isel(pressure_level=0, drop=True)

                    ds_hindcast = ds_hindcast.sortby("T")
                    ds_hindcast.load()
                    ds_hindcast.close()
                    os.remove(hindcast_file)
                    ds_hindcast.to_netcdf(hindcast_file)
                    print(f"Saved processed hindcast to {hindcast_file}")
                    store_hdcst_file_path[var] = ds_hindcast


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
        model = xe.single.EOF(n_modes=100, use_coslat=True, center=False)
        model.fit(data, dim='T')
        explained_var = model.explained_variance_ratio()
        cumulative_var = explained_var.cumsum()
        n_modes = np.where(cumulative_var >= self.eof_explained_var)[0][0] + 1
        print(f"Selected {n_modes} modes explaining {cumulative_var[n_modes-1].values*100:.2f}% variance")
        return model.scores().isel(mode=slice(0, n_modes))
    def _detrended_da(self, da, dim='T'):
            """
            Detrend a DataArray by removing the linear trend.
            """
            if dim not in da.dims:
                raise ValueError(f"DataArray must have a time dimension '{dim}' for detrending.")
    
            # 1. Calculate coefficients (Slope + Intercept)
            # Returns a Dataset containing 'polyfit_coefficients'
            coeffs = da.polyfit(dim=dim, deg=1)
    
            # 2. Reconstruct the trend line using standard xarray syntax
            trend = xr.polyval(da[dim], coeffs.polyfit_coefficients)
    
            # 3. Return anomalies and the FULL coefficients 
            # (We need the full coefficients to detrend future data for prediction)
            return da - trend, coeffs

    def _prepare_data_for_clustering(self, unique_years, target_year=None):
        """Helper to prepare data (indices or EOFs) for clustering methods."""
        _, ddd = self.download_and_process()
        data_scaled = None
        index_data = None

        if self.index_compute:
            ddd_sst = ddd.get('SST')
            if self.detrend:
                ddd_sst, _ = self._detrended_da(ddd_sst)
            if ddd_sst is None:
                raise ValueError("SST data not found in downloaded datasets for clustering.")
            indices_dataset = self.calc_index(self.index_compute, ddd_sst)
            indices_dataset_sel_years = indices_dataset.where(indices_dataset['T'].dt.year.isin(unique_years), drop=True)
            data_scaled, index_data = self.arrange_indices_for_som(indices_dataset_sel_years)
        else:
            if self.multivariateEOF:
                data_vars = [self._detrended_da(ddd[var].rename({"X": "lon", "Y": "lat"}))[0] if self.detrend else ddd[var].rename({"X": "lon", "Y": "lat"}) for var in ddd]
                model = xe.single.EOF(n_modes=100, use_coslat=True, center=False)
                model.fit(data_vars, dim='T')
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
                    if self.detrend:
                        ddd[var], _ = self._detrended_da(ddd[var])
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
        """Compatibility selector using training-only som preprocessing."""
        engine = self._scientific_engine()
        engine.method_analog = "som"
        forecast = np.asarray(ireference_year).size == 1 and int(np.asarray(ireference_year).ravel()[0]) == 2100
        selection = engine.select_analogs(predictant, itrain, None if forecast else ireference_year,
                                           target_source="forecast" if forecast else "hindcast")
        return selection["analog_years"]


    def Bias_Based(self, predictant, itrain, ireference_year):
        """Compatibility selector using training-only bias_based preprocessing."""
        engine = self._scientific_engine()
        engine.method_analog = "bias_based"
        forecast = np.asarray(ireference_year).size == 1 and int(np.asarray(ireference_year).ravel()[0]) == 2100
        selection = engine.select_analogs(predictant, itrain, None if forecast else ireference_year,
                                           target_source="forecast" if forecast else "hindcast")
        return selection["analog_years"]
    
    def Corr_Based(self, predictant, itrain, ireference_year):
        """Compatibility selector using training-only cor_based preprocessing."""
        engine = self._scientific_engine()
        engine.method_analog = "cor_based"
        forecast = np.asarray(ireference_year).size == 1 and int(np.asarray(ireference_year).ravel()[0]) == 2100
        selection = engine.select_analogs(predictant, itrain, None if forecast else ireference_year,
                                           target_source="forecast" if forecast else "hindcast")
        return selection["analog_years"]
    def Pca_Based(self, predictant, itrain, ireference_year):
        """Compatibility selector using training-only pca_based preprocessing."""
        engine = self._scientific_engine()
        engine.method_analog = "pca_based"
        forecast = np.asarray(ireference_year).size == 1 and int(np.asarray(ireference_year).ravel()[0]) == 2100
        selection = engine.select_analogs(predictant, itrain, None if forecast else ireference_year,
                                           target_source="forecast" if forecast else "hindcast")
        return selection["analog_years"]

    def KMeans_Based(self, predictant, itrain, ireference_year):
        """Compatibility selector using training-only kmeans_based preprocessing."""
        engine = self._scientific_engine()
        engine.method_analog = "kmeans_based"
        forecast = np.asarray(ireference_year).size == 1 and int(np.asarray(ireference_year).ravel()[0]) == 2100
        selection = engine.select_analogs(predictant, itrain, None if forecast else ireference_year,
                                           target_source="forecast" if forecast else "hindcast")
        return selection["analog_years"]

    def Agglomerative_Based(self, predictant, itrain, ireference_year):
        """Compatibility selector using training-only agglomerative_based preprocessing."""
        engine = self._scientific_engine()
        engine.method_analog = "agglomerative_based"
        forecast = np.asarray(ireference_year).size == 1 and int(np.asarray(ireference_year).ravel()[0]) == 2100
        selection = engine.select_analogs(predictant, itrain, None if forecast else ireference_year,
                                           target_source="forecast" if forecast else "hindcast")
        return selection["analog_years"]
    def download_and_process(self):
            """Download and process reanalysis and forecast data (cached per instance)."""
            if getattr(self, "_processed_analog_cache", None) is not None:
                return self._processed_analog_cache
            
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
    
                # --- Process Hindcast Data ---
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
    
                # --- Process Forecast Data ---
                sst_for_ = sst_for_.interp(Y=sst_hist_.Y, X=sst_hist_.X, method="linear", kwargs={"fill_value": "extrapolate"})
                
                if isinstance(sst_for_['T'].values, np.ndarray):
                    base_time = pd.Timestamp(sst_for_['T'].values[-1])
                else:
                    base_time = pd.Timestamp(sst_for_['T'].values)
                    
                new_times = [base_time + pd.DateOffset(months=int(m)) for m in sst_for_['forecastMonth'].values]
                sst_for_ = sst_for_.assign_coords(forecastMonth=("forecastMonth", new_times))
                sst_for_ = sst_for_.drop_vars('T', errors='ignore').squeeze().rename({'forecastMonth': 'T'})
    
                # --- Bias Correction ---
                biasr = WAS_bias_correction()
                sst_for_bias_corrected = []
                
                for m in lead_time:
                    # 1. Calcul correct du mois cible (gestion du cycle 12 -> 1)
                    target_time = base_time + pd.DateOffset(months=m)
                    current_target_month = target_time.month
    
                    # 2. Filtrage
                    sst_hist_mean = sst_hist_.where(sst_hist_['T'].dt.month == current_target_month, drop=True)
                    sst_hist_mean = sst_hist_mean.to_array().drop_vars(['variable'], errors='ignore').squeeze()
    
                    sst_hdcst_mean = sst_hdcst_.where(sst_hdcst_['T'].dt.month == current_target_month, drop=True)
                    sst_hdcst_mean = sst_hdcst_mean.to_array().drop_vars(['variable'], errors='ignore').squeeze()
                    
                    # 3. Alignement
                    # Create a set of unique years from the hindcast data
                    years_to_keep = sst_hdcst_mean['T'].dt.year.to_numpy()
                    
                    # Filter the historical data based on this set of years
                    sst_hist_mean = sst_hist_mean.where(sst_hist_mean['T'].dt.year.isin(years_to_keep), drop=True)
                    sst_hdcst_mean['T'] = sst_hist_mean['T']
                    sst_hist_aligned, sst_hdcst_aligned  = xr.align(sst_hist_mean, sst_hdcst_mean, join="inner")
                    
                    # 4. SÉCURITÉ CRITIQUE : Vérifier si les données sont vides après alignement
                    if sst_hist_aligned.sizes['T'] == 0 or sst_hdcst_aligned.sizes['T'] == 0:
                        print(f"WARNING: No overlapping data found for month {current_target_month} (lead {m}). Skipping bias correction for this step.")
                        continue # Passe à l'itération suivante pour éviter le crash
    
                    sst_for_mean = sst_for_.where(sst_for_['T'].dt.month == current_target_month, drop=True)
                    sst_for_mean = sst_for_mean.to_array().drop_vars(['variable'], errors='ignore').squeeze('variable', drop=True)
                    
                    if var in ["TMIN", "TEMP", "TMAX", "SST", "HUSS_1000", "HUSS_925", "HUSS_850", "SLP", "UGRD10", "VGRD10", "UGRD_1000", "UGRD_925", "UGRD_850", "VGRD_1000", "VGRD_925", "VGRD_850"]:
                        # Utilisation des variables alignées (_aligned)
                        fobj_quant_da = biasr.fitBC(sst_hist_aligned, sst_hdcst_aligned, method='QUANT', qstep=0.01, nboot=10)
                        sst_for_bias_corrected.append(biasr.doBC(sst_for_mean, fobj_quant_da, type='linear'))
                
                # --- Concatenation & Post-Processing ---
                if not sst_for_bias_corrected:
                    print(f"ERROR: Bias correction failed for all lead times for variable {var}.")
                    continue
    
                sst_for_ = xr.concat(sst_for_bias_corrected, dim='T').sortby("T")
    
                sst_hist_ = sst_hist_.sel(T=slice(f"{self.year_start}-01", f"{base_time.year}-{base_time.month:02d}"))
                sst_hist_ = sst_hist_.to_array().drop_vars(['variable'], errors='ignore').squeeze()
                
                concatenated_ds = xr.concat([sst_hist_, sst_for_], dim='T')
                concatenated_ds = concatenated_ds.sortby("T")
                concatenated_ds = concatenated_ds.to_dataset(name=var)
    
                if isinstance(self.rolling, int) and self.rolling > 1:
                    print(f"Applied rolling mean with window size {self.rolling} for {var}.")
                    concatenated_ds = concatenated_ds.rolling(T=self.rolling, center=False, min_periods=self.rolling).mean()
    
                if self.standardize:
                    print(f"Standardizing timeseries for {var}.")
                    concatenated_ds_st = self.standardize_timeseries(concatenated_ds)
                else:
                    print(f"Computing anomalies for {var}.")
                    concatenated_ds_st = self.anomaly_timeseries(concatenated_ds)
    
                select_parts = concatenated_ds_st['T'].size - (self.year_forecast - self.year_start) * 12 
                concatenated_ds_st = concatenated_ds_st.isel(T=slice(select_parts, None))
                
                new_time = pd.DatetimeIndex([pd.to_datetime(f"{self.year_start+1}-01-01") + pd.DateOffset(months=t) for t in range(len(concatenated_ds_st['T']))]) 
    
                ds_shifted = concatenated_ds_st.assign_coords(T=new_time)
        
                data_var_shifted[var] = ds_shifted.to_array().drop_vars(['variable'], errors='ignore').squeeze()
                data_var_concatenated[var] = concatenated_ds_st.to_array().drop_vars(['variable'], errors='ignore').squeeze()
                
            self._processed_analog_cache = (data_var_concatenated, data_var_shifted)
            return self._processed_analog_cache

    def _scientific_engine(self):
        """Compatibility facade: legacy names use the same fold-safe engine."""
        import inspect
        engine = getattr(self, "_legacy_scientific_engine", None)
        if engine is None:
            arguments = {name: getattr(self, name) for name in inspect.signature(WAS_Analog.__init__).parameters
                         if name != "self" and hasattr(self, name)}
            arguments.update(feature_mode="indices" if self.index_compute else "eof",
                             nested_tuning=False, probability_shrinkage=0.25,
                             probability_method=getattr(self, "probability_method", "residual"))
            engine = WAS_Analog_Scientific(**arguments)
            engine.download_reanalysis = self.download_reanalysis
            engine.download_models = self.download_models
            self._legacy_scientific_engine = engine
        return engine

    def compute_model(self, predictant, itrain, itest, **kwargs):
        """Compute fold-safe legacy-method hindcasts through the common engine."""
        return self._scientific_engine().compute_model(predictant, itrain, itest, **kwargs)

    # ------------------ Probability Calculation Methods ------------------

    @staticmethod
    def _ppf_terciles_from_code(dist_code, shape, loc, scale):
        """
        Return tercile thresholds (T1, T2) from best-fit distribution parameters.
    
        dist_code:
            1: norm
            2: lognorm
            3: expon
            4: gamma
            5: weibull_min
            6: t
            7: poisson
            8: nbinom
        """
        if np.isnan(dist_code):
            return np.nan, np.nan
    
        code = int(dist_code)
        try:
            if code == 1:
                return (
                    norm.ppf(1.0 / 3.0, loc=loc, scale=scale),
                    norm.ppf(2.0 / 3.0, loc=loc, scale=scale),
                )
            elif code == 2:
                return (
                    lognorm.ppf(1.0 / 3.0, s=shape, loc=loc, scale=scale),
                    lognorm.ppf(2.0 / 3.0, s=shape, loc=loc, scale=scale),
                )
            elif code == 3:
                return (
                    expon.ppf(1.0 / 3.0, loc=loc, scale=scale),
                    expon.ppf(2.0 / 3.0, loc=loc, scale=scale),
                )
            elif code == 4:
                return (
                    gamma.ppf(1.0 / 3.0, a=shape, loc=loc, scale=scale),
                    gamma.ppf(2.0 / 3.0, a=shape, loc=loc, scale=scale),
                )
            elif code == 5:
                return (
                    weibull_min.ppf(1.0 / 3.0, c=shape, loc=loc, scale=scale),
                    weibull_min.ppf(2.0 / 3.0, c=shape, loc=loc, scale=scale),
                )
            elif code == 6:
                # Note: Renamed 't_dist' to 't' for standard scipy.stats
                return (
                    t.ppf(1.0 / 3.0, df=shape, loc=loc, scale=scale),
                    t.ppf(2.0 / 3.0, df=shape, loc=loc, scale=scale),
                )
            elif code == 7:
                # Poisson: poisson.ppf(q, mu, loc=0)
                # ASSUMPTION: 'mu' (mean) is passed as 'shape'
                #             'loc' is passed as 'loc'
                #             'scale' is unused
                return (
                    poisson.ppf(1.0 / 3.0, mu=shape, loc=loc),
                    poisson.ppf(2.0 / 3.0, mu=shape, loc=loc),
                )
            elif code == 8:
                # Negative Binomial: nbinom.ppf(q, n, p, loc=0)
                # ASSUMPTION: 'n' (successes) is passed as 'shape'
                #             'p' (probability) is passed as 'scale'
                #             'loc' is passed as 'loc'
                return (
                    nbinom.ppf(1.0 / 3.0, n=shape, p=scale, loc=loc),
                    nbinom.ppf(2.0 / 3.0, n=shape, p=scale, loc=loc),
                )
        except Exception:
            return np.nan, np.nan
    
        # Fallback if code is not 1-8
        return np.nan, np.nan
        
    @staticmethod
    def weibull_shape_solver(k, M, V):
        """
        Function to find the root of the Weibull shape parameter 'k'.
        We find 'k' such that the theoretical variance/mean^2 ratio
        matches the observed V/M^2 ratio.
        """
        # Guard against invalid 'k' values during solving
        if k <= 0:
            return -np.inf
        try:
            g1 = gamma_function(1 + 1/k)
            g2 = gamma_function(1 + 2/k)
            
            # This is the V/M^2 ratio *implied by k*
            implied_v_over_m_sq = (g2 / (g1**2)) - 1
            
            # This is the *observed* ratio
            observed_v_over_m_sq = V / (M**2)
            
            # Return the difference (we want this to be 0)
            return observed_v_over_m_sq - implied_v_over_m_sq
        except ValueError:
            return -np.inf # Handle math errors

    @staticmethod
    def calculate_tercile_probabilities_bestfit(best_guess, error_variance, T1, T2,
                                               dist_code, dof=3, nonneg=False):
        from .was_linear_models import _regression_tercile_probabilities
        return _regression_tercile_probabilities(best_guess,error_variance,T1,T2,
                                                dist_code,dof,nonneg=nonneg)


    @staticmethod
    def calculate_tercile_probabilities_nonparametric(best_guess, error_samples,
                                                      first_tercile, second_tercile,
                                                      nonneg=False):
        from .was_linear_models import _residual_probabilities
        return _residual_probabilities(best_guess,error_samples,first_tercile,
                                       second_tercile,nonneg=nonneg)



    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det,
                     best_code_da=None, best_shape_da=None, best_loc_da=None,
                     best_scale_da=None, **kwargs):
        """Explicit outer/inner residual calibration for the historical API."""
        from .was_probability_calibration import direct_hindcast_probabilities
        return direct_hindcast_probabilities(self,Predictant,clim_year_start,clim_year_end,
            hindcast_det,best_code_da,best_shape_da,best_loc_da,best_scale_da,**kwargs)



    def forecast(self, predictant, clim_year_start, clim_year_end, hindcast_det,
                 best_code_da=None, best_shape_da=None, best_loc_da=None,
                 best_scale_da=None, **kwargs):
        """Forecast through the same fold-safe engine used for hindcasts."""
        return self._scientific_engine().forecast(predictant,clim_year_start,clim_year_end,
            hindcast_det,best_code_da,best_shape_da,best_loc_da,best_scale_da,**kwargs)


    def composite_plot(self, predictant, clim_year_start, clim_year_end, hindcast_det, plot_predictor=True, variable="SST",best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None):
        """Create composite plots of predictors or predictands."""
        clim_slice = slice(str(clim_year_start), str(clim_year_end))
        clim_mean = predictant.sel(T=clim_slice).mean(dim='T', skipna=True)
        similar_years, result_, _ = self.forecast(predictant, clim_year_start, clim_year_end, hindcast_det, best_code_da, best_shape_da, best_loc_da, best_scale_da)
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
                ax.gridlines(draw_labels=False)
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
########################################################################################################################################################################################################################################

# Core Python Libraries
import os
import requests  # Added for NCEI download
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
from scipy.stats import pearsonr, gamma, lognorm, stats, norm, expon, weibull_min, t, poisson, nbinom
from scipy.optimize import fsolve
from scipy.special import gamma as gamma_function
import scipy.signal as sig

# EOF Analysis and Verification Libraries
import xeofs as xe
import xskillscore as xs
from wass2s.utils import *
from wass2s.was_bias_correction import *

class WAS_Analog__(WAS_Analog):
    """Compatibility name using the corrected common analogue engine."""
    pass
# =============================================================================
# Scientific analogue engine (2026-08 revision)
# =============================================================================
# The historical WAS_Analog / WAS_Analog__ classes above are intentionally kept
# for backwards compatibility.  The classes below add a strict, operationally
# consistent analogue engine in which the target state is a GCM hindcast during
# cross-validation and a GCM forecast in real time, while candidate analogues
# always come from the historical/reanalysis library.

class WAS_Analog_Scientific(WAS_Analog):
    """Leakage-aware analogue-ensemble seasonal forecasting.

    This class preserves the public ``compute_model / compute_prob / forecast``
    contract used by :class:`WAS_Cross_Validator`, but replaces the old
    threshold/top-2 analogue selection with a common scientific core:

    * candidate library = historical/reanalysis states only;
    * CV target state = corresponding GCM hindcast, bias-corrected on training
      years only (operational forecast uses the GCM forecast);
    * fold-local feature standardisation and optional EOF/PCA reduction;
    * continuous Euclidean, correlation, Mahalanobis or hybrid distances;
    * tunable number of analogues and distance weighting;
    * direct weighted analogue-ensemble tercile probabilities with optional
      shrinkage to climatology;
    * optional constructed analogues and an interpretable learned metric.

    The historical :class:`WAS_Analog` interface delegates its selectors and
    operational forecast to this engine; its residual-probability API remains explicit.
    """

    _SCIENTIFIC_METHODS = {
        "scientific", "knn", "weighted", "eof", "hybrid",
        "constructed", "learned_metric", "learned_spatial_metric",
    }

    def __init__(self, *args, **kwargs):
        # Scientific options are popped before forwarding legacy/base options.
        self.feature_mode = kwargs.pop("feature_mode", "hybrid")
        self.analog_metric = str(kwargs.pop("analog_metric", "hybrid")).lower()
        self.metric_grid = tuple(kwargs.pop(
            "metric_grid", ("euclidean", "correlation", "hybrid", "mahalanobis")
        ))
        self.hybrid_alpha = float(kwargs.pop("hybrid_alpha", 0.5))
        self.hybrid_alpha_grid = tuple(kwargs.pop(
            "hybrid_alpha_grid", (0.25, 0.5, 0.75)
        ))
        self.n_analogs = int(kwargs.pop("n_analogs", 4))
        self.k_grid = tuple(int(v) for v in kwargs.pop(
            "k_grid", (2, 3, 4, 5, 6, 8)
        ))
        self.analog_weighting = str(kwargs.pop("analog_weighting", "softmax")).lower()
        self.weighting_grid = tuple(kwargs.pop(
            "weighting_grid", ("equal", "inverse_distance", "softmax")
        ))
        self.tau = float(kwargs.pop("tau", 1.0))
        self.tau_grid = tuple(float(v) for v in kwargs.pop(
            "tau_grid", (0.25, 0.5, 1.0, 2.0)
        ))
        self.inverse_distance_power = float(kwargs.pop("inverse_distance_power", 1.0))

        self.nested_tuning = bool(kwargs.pop("nested_tuning", False))
        self.inner_nb_omit = int(kwargs.pop("inner_nb_omit", 1))
        self.tune_score = str(kwargs.pop("tune_score", "mse")).lower()
        self.one_se = bool(kwargs.pop("one_se", True))
        # ``strict`` rebuilds target features from the corresponding GCM
        # hindcast for every inner validation year.  ``outer`` is much faster
        # and still keeps the outer test completely unseen.
        self.inner_preprocessing = str(kwargs.pop("inner_preprocessing", "strict")).lower()
        if self.inner_preprocessing not in {"strict", "outer"}:
            raise ValueError("inner_preprocessing must be 'strict' or 'outer'.")

        self.probability_method = str(kwargs.pop(
            "probability_method", "analog_ensemble"
        )).lower()
        self.probability_shrinkage = kwargs.pop("probability_shrinkage", 0.25)
        self.probability_shrinkage_grid = tuple(float(v) for v in kwargs.pop(
            "probability_shrinkage_grid", (0.0, 0.10, 0.25, 0.50, 0.75)
        ))
        self.quality_shrinkage = bool(kwargs.pop("quality_shrinkage", True))
        self.max_quality_shrinkage = float(kwargs.pop("max_quality_shrinkage", 0.50))

        self.max_eof_modes = int(kwargs.pop("max_eof_modes", 8))
        self.eof_max_feature_ratio = float(kwargs.pop("eof_max_feature_ratio", 0.25))
        self.default_indices = tuple(kwargs.pop(
            "default_indices", ("NINO34", "TNA", "TSA", "DMI", "TASI")
        ))

        self.analog_bias_method = str(kwargs.pop("analog_bias_method", "VARSCALE")).upper()
        self.analog_bias_kwargs = dict(kwargs.pop("analog_bias_kwargs", {}))
        self.require_model_hindcast_target = bool(kwargs.pop(
            "require_model_hindcast_target", True
        ))
        self.target_member_mode = str(kwargs.pop("target_member_mode", "mean")).lower()
        self.target_member_aggregation = str(kwargs.pop("target_member_aggregation", "mean")).lower()
        self.max_target_members = kwargs.pop("max_target_members", None)
        self.fallback_to_reanalysis_target = bool(kwargs.pop(
            "fallback_to_reanalysis_target", False
        ))
        self.fallback_to_climatology = bool(kwargs.pop("fallback_to_climatology", True))

        self.constructed_lambda = float(kwargs.pop("constructed_lambda", 0.10))
        self.constructed_lambda_grid = tuple(float(v) for v in kwargs.pop(
            "constructed_lambda_grid", (0.0, 0.01, 0.10, 1.0, 10.0)
        ))
        self.constructed_pool_k = int(kwargs.pop("constructed_pool_k", 8))
        self.metric_weight_l2 = float(kwargs.pop("metric_weight_l2", 0.50))
        self.spatial_box_deg = float(kwargs.pop("spatial_box_deg", 20.0))
        self.random_state = int(kwargs.pop("random_state", 42))
        self.min_analogs = int(kwargs.pop("min_analogs", 2))
        self.verbose_analog = bool(kwargs.pop("verbose_analog", True))
        self.hindcast_year_start = kwargs.pop("hindcast_year_start", None)
        self.hindcast_year_end = kwargs.pop("hindcast_year_end", None)
        self.target_year_offset = int(kwargs.pop("target_year_offset", 0))
        self.tercile_reference = str(kwargs.pop("tercile_reference", "climatology")).lower()
        self.forecast_time = kwargs.pop("forecast_time", None)
        if self.tercile_reference not in {"climatology", "training"}:
            raise ValueError("tercile_reference must be climatology or training")

        kwargs.setdefault("method_analog", "hybrid")
        kwargs.setdefault("dist_method", "nonparam")
        # Rolling a 3-lead seasonal vector would leave very little information;
        # the scientific feature engine therefore defaults to no lead smoothing
        # unless the caller explicitly supplied rolling to the base constructor.
        if "rolling" not in kwargs and len(args) < 9:
            kwargs["rolling"] = 1
        super().__init__(*args, **kwargs)

        valid_methods = self._SCIENTIFIC_METHODS | {"som", "bias_based", "cor_based", "pca_based", "kmeans_based", "agglomerative_based"}
        if self.method_analog not in valid_methods:
            raise ValueError(f"Unknown analogue method: {self.method_analog!r}")
        if self.n_analogs < 1 or self.min_analogs < 1 or self.n_analogs < self.min_analogs:
            raise ValueError("n_analogs must be at least min_analogs >= 1")
        if not self.k_grid or min(self.k_grid) < self.min_analogs:
            raise ValueError("k_grid must contain positive sizes >= min_analogs")
        if self.inner_nb_omit < 0 or self.max_eof_modes < 1 or not 0 < self.eof_max_feature_ratio <= 1:
            raise ValueError("Invalid inner buffer or EOF feature limits")
        if not 1 <= self._init_month() <= 12:
            raise ValueError("month_of_initialization must be in 1..12")
        leads = self._lead_values()
        if not leads or len(set(leads)) != len(leads) or any(v < 0 or v > 11 for v in leads):
            raise ValueError("lead_time must contain unique zero-based offsets in 0..11")
        if self.target_member_mode not in {"mean", "median", "first", "memberwise"}:
            raise ValueError("Unknown target_member_mode")
        if self.target_member_mode == "memberwise" and self.ensemble_mean is not None:
            raise ValueError("memberwise analogues require ensemble_mean=None to preserve raw members")
        if self.target_member_aggregation not in {"mean", "median"}:
            raise ValueError("target_member_aggregation must be mean or median")
        if self.max_target_members is not None and int(self.max_target_members) < 1:
            raise ValueError("max_target_members must be positive")
        if self.tune_score not in {"mse", "rps"}:
            raise ValueError("tune_score must be mse or rps")
        metrics = {"euclidean", "weighted_euclidean", "correlation", "hybrid", "mahalanobis"}
        weightings = {"equal", "inverse_distance", "softmax", "gaussian_kernel"}
        if self.analog_metric not in metrics | {"auto"} or not set(self.metric_grid) <= metrics:
            raise ValueError("Unknown analogue distance metric")
        if self.analog_weighting not in weightings | {"auto"} or not set(self.weighting_grid) <= weightings:
            raise ValueError("Unknown analogue weighting")
        if self.tau <= 0 or any(v <= 0 for v in self.tau_grid) or self.inverse_distance_power <= 0:
            raise ValueError("Distance weighting scales and powers must be positive")
        if not 0 <= self.hybrid_alpha <= 1 or any(not 0 <= v <= 1 for v in self.hybrid_alpha_grid):
            raise ValueError("Hybrid metric weights must be in [0,1]")
        if self.probability_shrinkage != "auto":
            self.probability_shrinkage = float(self.probability_shrinkage)
            if not 0 <= self.probability_shrinkage <= 1:
                raise ValueError("probability_shrinkage must be auto or a value in [0,1]")
        if any(not 0 <= v <= 1 for v in self.probability_shrinkage_grid) or not 0 <= self.max_quality_shrinkage <= 1:
            raise ValueError("Shrinkage weights must be in [0,1]")
        if self.constructed_lambda < 0 or any(v < 0 for v in self.constructed_lambda_grid):
            raise ValueError("Constructed analogue ridge strengths must be nonnegative")
        if self.hindcast_year_start is None:
            self.hindcast_year_start = self.clim_year_start if self.clim_year_start is not None else self.year_start
        if self.hindcast_year_end is None:
            self.hindcast_year_end = self.clim_year_end if self.clim_year_end is not None else int(self.year_forecast) - 1

        self._analog_raw_cache = None
        self._analog_feature_cache = {}
        self._analog_cv_cache = {}
        self._analog_member_cv_cache = {}
        self._analog_last_selection = None
        self._analog_tuning_table = None

    # ------------------------------------------------------------------ public
    def clear_analog_cache(self):
        """Clear downloaded/preprocessed scientific analogue caches."""
        self._analog_raw_cache = None
        self._analog_feature_cache.clear()
        self._analog_cv_cache.clear()
        self._analog_member_cv_cache.clear()
        self._analog_last_selection = None
        self._analog_tuning_table = None
        getattr(self, "_analog_selector_cache", {}).clear()

    @property
    def analog_diagnostics(self):
        """Return a shallow copy of the most recent analogue selection."""
        return None if self._analog_last_selection is None else dict(self._analog_last_selection)

    def tuning_table(self):
        """Return the last inner-CV tuning table as a DataFrame, if available."""
        return None if self._analog_tuning_table is None else self._analog_tuning_table.copy()

    def selection_table(self):
        """Return year/distance/weight for the most recent analogue selection."""
        s = self._analog_last_selection
        if s is None:
            return pd.DataFrame(columns=["year", "distance", "weight"])
        return pd.DataFrame({
            "year": np.asarray(s["analog_years"], int),
            "distance": np.asarray(s["distances"], float),
            "weight": np.asarray(s["weights"], float),
        }).sort_values("weight", ascending=False, ignore_index=True)

    def learned_spatial_weights(self):
        """Return the last learned coarse-box weights as a tidy DataFrame."""
        s = self._analog_last_selection
        gw = None if s is None else s.get("group_weights")
        if not gw:
            return pd.DataFrame(columns=["box", "weight"])
        rows = [(g, w) for g, w in gw.items() if str(g).startswith("BOX:")]
        return pd.DataFrame(rows, columns=["box", "weight"]).sort_values(
            "weight", ascending=False, ignore_index=True
        )

    # --------------------------------------------------------------- raw inputs
    @staticmethod
    def _as_dataarray(obj, var_name=None):
        if isinstance(obj, xr.DataArray):
            return obj
        if isinstance(obj, xr.Dataset):
            if var_name in obj.data_vars:
                out = obj[var_name].copy(deep=False)
                out.attrs = dict(obj.attrs, **obj[var_name].attrs)
                return out
            names = list(obj.data_vars)
            if len(names) == 1:
                out = obj[names[0]].copy(deep=False)
                out.attrs = dict(obj.attrs, **obj[names[0]].attrs)
                return out
            raise ValueError("Dataset contains several variables; specify var_name to avoid mixing physical quantities.")
        raise TypeError(f"Expected xarray Dataset/DataArray, got {type(obj)!r}")

    @staticmethod
    def _normalize_time_axis_for_dt(obj):
        """Normalize Dataset/DataArray dates, rejecting ambiguous numeric epochs."""
        if isinstance(obj, dict):
            return {key: WAS_Analog_Scientific._normalize_time_axis_for_dt(value) for key, value in obj.items()}
        if not isinstance(obj, (xr.DataArray, xr.Dataset)) or "T" not in obj.coords:
            return obj
        values = np.asarray(obj["T"].values)
        if np.issubdtype(values.dtype, np.datetime64):
            return obj
        try:
            obj["T"].dt.year
            return obj
        except (AttributeError, TypeError):
            pass
        bare_years = np.issubdtype(values.dtype, np.number)
        if bare_years:
            if not np.all(np.isfinite(values) & (values == np.floor(values)) & (values >= 1600) & (values <= 2200)):
                raise ValueError("Numeric analogue T must contain integer calendar years; decode numeric CF time first")
            dates = pd.to_datetime([f"{int(y):04d}-01-01" for y in values.ravel()])
        else:
            try:
                dates = pd.to_datetime(values.ravel())
            except Exception as exc:
                raise ValueError("Analogue time coordinates cannot be decoded") from exc
        if pd.isna(dates).any():
            raise ValueError("Analogue time coordinates contain missing dates")
        result = obj.assign_coords(T=dates if values.ndim else dates[0])
        if bare_years:
            result["T"].attrs["analog_bare_years"] = True
        return result

    def _get_raw_sources(self):
        """Return (reanalysis, hindcast, forecast) dictionaries, cached."""
        if self._analog_raw_cache is None:
            hist = self.download_reanalysis(force_download=False)
            hdcst, fcst = self.download_models(force_download=False)
            # Normalize T to datetime64 at the single point every downstream
            # .dt access flows through, instead of guarding each of the dozens
            # of ["T"].dt call sites individually.
            hist = self._normalize_time_axis_for_dt(hist)
            hdcst = self._normalize_time_axis_for_dt(hdcst)
            fcst = self._normalize_time_axis_for_dt(fcst)
            self._analog_raw_cache = (hist, hdcst, fcst)
        return self._analog_raw_cache

    def _lead_values(self):
        return [1, 2, 3, 4, 5] if self.lead_time is None else [int(v) for v in self.lead_time]

    def _init_month(self):
        return int(self.month_of_initialization) if self.month_of_initialization is not None else int((pd.Timestamp.now() - pd.DateOffset(months=1)).month)

    def _valid_dates(self, anchor_year):
        base = pd.Timestamp(int(anchor_year) - self.target_year_offset, self._init_month(), 1)
        return [base + pd.DateOffset(months=int(lead)) for lead in self._lead_values()]

    @staticmethod
    def _finite_template(da):
        if "T" in da.dims:
            return xr.full_like(da.isel(T=0, drop=True), np.nan, dtype=float)
        return xr.full_like(da, np.nan, dtype=float)

    def _reanalysis_state(self, hist_da, anchor_year):
        """Seasonal lead sequence from the reanalysis analogue library."""
        hist_da = self._as_dataarray(hist_da)
        out = []
        template = self._finite_template(hist_da)
        for lead, dt in zip(self._lead_values(), self._valid_dates(anchor_year)):
            sel = hist_da.where(
                (hist_da["T"].dt.year == dt.year) & (hist_da["T"].dt.month == dt.month),
                drop=True,
            )
            if sel.sizes.get("T", 0):
                v = sel.mean("T", skipna=True)
            else:
                v = template.copy()
            out.append(v.expand_dims(L=[lead]))
        return xr.concat(out, dim="L")

    def _model_state(self, model_da, anchor_year=None, forecast=False, member=None):
        """Select the declared initialization and exact zero-based lead labels."""
        da = self._as_dataarray(self._normalize_time_axis_for_dt(model_da))
        requested_year = int(self.year_forecast if forecast and anchor_year is None else anchor_year) - self.target_year_offset
        if "T" in da.coords:
            years = np.atleast_1d(da["T"].dt.year.values)
            months = np.atleast_1d(da["T"].dt.month.values)
            matches = (years == requested_year)
            if not da["T"].attrs.get("analog_bare_years", False):
                matches &= months == self._init_month()
            indices = np.flatnonzero(matches)
            if len(indices) != 1:
                raise KeyError(f"Expected one model initialization for {requested_year}-{self._init_month():02d}; found {len(indices)}")
            if "T" in da.dims:
                da = da.isel(T=int(indices[0]), drop=True)
        elif not forecast:
            raise ValueError("Hindcast state requires a dated T coordinate")
        lead_dim = next((name for name in ("forecastMonth", "leadtime_month", "lead", "L") if name in da.dims), None)
        if lead_dim is None or lead_dim not in da.coords:
            raise ValueError("Model state requires explicit forecastMonth/lead coordinates")
        try:
            labels = np.asarray(da[lead_dim].values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Decode model lead labels to zero-based integer month offsets") from exc
        if not np.all(np.isfinite(labels) & (labels == np.floor(labels))):
            raise ValueError("Model lead labels must be integer month offsets")
        if len(np.unique(labels)) != len(labels):
            raise ValueError("Model lead labels must be unique")
        # Raw CDS fields can explicitly declare their one-based convention.
        if da.attrs.get("analog_lead_convention") == "c3s_one_based":
            labels = labels - 1
        pieces = []
        for lead in self._lead_values():
            found = np.flatnonzero(labels == lead)
            if len(found) != 1:
                raise KeyError(f"Lead offset {lead} unavailable; available offsets are {labels.astype(int).tolist()}")
            value = da.isel({lead_dim: int(found[0])}, drop=True)
            member_dim = next((dim for dim in ("number", "M", "member") if dim in value.dims), None)
            if member_dim:
                if member is not None:
                    value = value.isel({member_dim: int(member)}, drop=True)
                elif self.target_member_mode == "median":
                    value = value.median(member_dim, skipna=True)
                elif self.target_member_mode in ("mean", "memberwise"):
                    value = value.mean(member_dim, skipna=True)
                else:
                    value = value.isel({member_dim: 0}, drop=True)
            for dim in list(value.dims):
                if dim not in ("Y", "X") and value.sizes[dim] == 1:
                    value = value.isel({dim: 0}, drop=True)
            if set(value.dims) != {"Y", "X"}:
                raise ValueError(f"Model state has unexpected dimensions {value.dims}")
            pieces.append(value.expand_dims(L=[lead]))
        return xr.concat(pieces, dim="L")

    @staticmethod
    def _interp_like_state(state, ref_state):
        """Interpolate within the source domain; preserve singleton coordinates."""
        result = state
        if "X" in result.coords:
            result = result.assign_coords(X=((result.X + 180.0) % 360.0) - 180.0)
            _, unique = np.unique(result.X.values, return_index=True)
            result = result.isel(X=np.sort(unique)).sortby("X")
        for dim in ("Y", "X"):
            if dim not in result.coords or dim not in ref_state.coords:
                continue
            target = ref_state[dim]
            if dim == "X":
                target = ((target + 180.0) % 360.0) - 180.0
            if np.array_equal(result[dim].values, target.values):
                continue
            result = result.sortby(dim)
            if result.sizes[dim] == 1:
                result = result.reindex({dim: target.values})
            else:
                result = result.interp({dim: target}, method="linear")
        # Keep the reference labels (including 0..360 input) for exact alignment.
        return result.assign_coords({dim: ref_state[dim] for dim in ("Y", "X") if dim in ref_state.coords})

    def _bias_correct_target(self, var, target_state, hist_da, hdcst_da, train_years, member=None):
        """Fold-safe model->reanalysis correction of a target lead vector."""
        method = self.analog_bias_method
        if method in ("NONE", "RAW", "OFF"):
            return target_state

        corrected = []
        # Build raw aligned training states once, not once per lead per year.
        training_states = []
        for year in sorted({int(value) for value in train_years}):
            observed = self._reanalysis_state(hist_da, year)
            modeled = self._model_state(hdcst_da, year, forecast=False, member=member)
            training_states.append((year, observed, self._interp_like_state(modeled, observed)))
        for il, lead in enumerate(self._lead_values()):
            obs_maps, mod_maps, labels = [], [], []
            for y, observed, modeled in training_states:
                obs_maps.append(observed.sel(L=lead, drop=True))
                mod_maps.append(modeled.sel(L=lead, drop=True))
                labels.append(np.datetime64(f"{y}-01-01"))

            target_map = target_state.sel(L=lead, drop=True)
            if len(obs_maps) < 3:
                corrected.append(target_map.expand_dims(L=[lead]))
                continue

            obs = xr.concat(obs_maps, dim="T").assign_coords(T=labels).transpose("T", "Y", "X")
            mod = xr.concat(mod_maps, dim="T").assign_coords(T=labels).transpose("T", "Y", "X")
            target3 = target_map.expand_dims(T=[np.datetime64("2100-01-01")]).transpose("T", "Y", "X")
            try:
                if var.upper() == "PRCP":
                    qmethod = method if method in ("QUANT", "RQUANT", "SSPLIN", "PTF", "DIST") else "RQUANT"
                    fobj = WAS_Qmap.fitQmap(obs, mod, qmethod, **self.analog_bias_kwargs)
                    corr = WAS_Qmap.doQmap(target3, fobj)
                else:
                    bmethod = method if method in ("MEAN", "VARSCALE", "QUANT", "NORM", "DIST") else "VARSCALE"
                    fobj = WAS_bias_correction.fitBC(obs, mod, bmethod, **self.analog_bias_kwargs)
                    corr = WAS_bias_correction.doBC(target3, fobj)
                corrected.append(corr.isel(T=0, drop=True).expand_dims(L=[lead]))
            except Exception as exc:
                # Stable additive fallback fitted only on training years.
                delta = (obs - mod).mean("T", skipna=True)
                if self.verbose_analog:
                    warnings.warn(
                        f"Analog bias correction {method} failed for {var}/L{lead}: {exc}; "
                        "using fold-local mean adjustment.", RuntimeWarning, stacklevel=2,
                    )
                corrected.append((target_map + delta).expand_dims(L=[lead]))
        return xr.concat(corrected, dim="L")

    # ----------------------------------------------------------- feature engine
    @staticmethod
    def _state_to_time(state, anchor_year, init_month, lead_values):
        dates = [pd.Timestamp(int(anchor_year), int(init_month), 1) + pd.DateOffset(months=int(l))
                 for l in lead_values]
        return state.rename({"L": "T"}).assign_coords(T=pd.DatetimeIndex(dates))

    def _index_features(self, candidate_states, target_state, years):
        names = tuple(self.index_compute) if self.index_compute else self.default_indices
        if not names:
            return None
        rows = []
        feature_names = []
        groups = []
        for i, y in enumerate(years):
            ds = self.calc_index(
                names,
                self._state_to_time(candidate_states.isel(A=i, drop=True), y,
                                    self._init_month(), self._lead_values()),
            )
            vals = []
            if i == 0:
                feature_names = []
                groups = []
            for idx in names:
                if idx not in ds:
                    continue
                v = np.asarray(ds[idx].values, dtype=float).ravel()
                vals.extend(v.tolist())
                if i == 0:
                    for lead in self._lead_values()[:len(v)]:
                        feature_names.append(f"IDX:{idx}:L{lead}")
                        groups.append(f"IDX:{idx}")
            rows.append(vals)

        target_ds = self.calc_index(
            names,
            self._state_to_time(target_state, 2100, self._init_month(), self._lead_values()),
        )
        target_vals = []
        for idx in names:
            if idx in target_ds:
                target_vals.extend(np.asarray(target_ds[idx].values, dtype=float).ravel().tolist())
        if not rows or not target_vals:
            return None
        return np.asarray(rows, float), np.asarray(target_vals, float), feature_names, groups

    @staticmethod
    def _fold_linear_detrend(arr, years, target_arr, target_year):
        """Observed-pair least squares; missing cells are never fabricated."""
        values = np.asarray(arr, float)
        target = np.asarray(target_arr, float)
        time = np.asarray(years, float).reshape((-1,) + (1,) * (values.ndim - 1))
        good = np.isfinite(values)
        count = good.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            tmean = np.where(good, time, 0.).sum(axis=0) / count
            ymean = np.where(good, values, 0.).sum(axis=0) / count
            denominator = np.where(good, (time - tmean) ** 2, 0.).sum(axis=0)
            numerator = np.where(good, (time - tmean) * (values - ymean), 0.).sum(axis=0)
            slope = np.divide(numerator, denominator, out=np.zeros_like(ymean), where=denominator > 0)
        fitted = ymean + slope * (time - tmean)
        return values - fitted, target - (ymean + slope * (float(target_year) - tmean))

    def _eof_features(self, candidate_states, target_state, years, var_name, target_year):
        from sklearn.decomposition import PCA as _SKPCA

        arr = np.array(candidate_states.transpose("A", "L", "Y", "X").values, dtype=float, copy=True)
        tar = np.array(target_state.transpose("L", "Y", "X").values, dtype=float, copy=True)
        n_a, n_l, ny, nx = arr.shape
        arr = arr.reshape(n_a, n_l, ny * nx)
        tar = tar.reshape(n_l, ny * nx)

        # cos(latitude) area weighting, square-root form appropriate for EOF/PCA.
        if "Y" in candidate_states.coords:
            lat = np.asarray(candidate_states["Y"].values, float)
            wlat = np.sqrt(np.clip(np.cos(np.deg2rad(lat)), 0.0, None))
            w = np.repeat(wlat[:, None], nx, axis=1).ravel()
        else:
            w = np.ones(ny * nx, dtype=float)

        # Lead-specific climatology/scale from analogue-library training years.
        for l in range(n_l):
            mu = np.nanmean(arr[:, l, :], axis=0)
            sd = np.nanstd(arr[:, l, :], axis=0)
            sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)
            arr[:, l, :] = (arr[:, l, :] - mu) / sd
            tar[l, :] = (tar[l, :] - mu) / sd

        arr *= w[None, None, :]
        tar *= w[None, :]
        flat = arr.reshape(n_a * n_l, -1)
        valid = ((np.isfinite(flat).sum(axis=0) >= max(3, int(np.ceil(0.8 * flat.shape[0]))))
                 & np.isfinite(tar).all(axis=0))
        if not np.any(valid):
            return None
        flat = flat[:, valid]
        tflat = tar[:, valid]
        col_mean = np.nanmean(flat, axis=0)
        flat = np.where(np.isfinite(flat), flat, col_mean)
        tflat = np.where(np.isfinite(tflat), tflat, col_mean)

        year_cap = max(1, int(np.floor(n_a * self.eof_max_feature_ratio)))
        ncomp = min(self.max_eof_modes, year_cap, flat.shape[0] - 1, flat.shape[1])
        if ncomp < 1:
            return None
        pca = _SKPCA(n_components=ncomp, svd_solver="full")
        score = pca.fit_transform(flat).reshape(n_a, n_l, ncomp)
        tscore = pca.transform(tflat).reshape(n_l, ncomp)
        X = score.reshape(n_a, n_l * ncomp)
        x = tscore.reshape(n_l * ncomp)
        fn, groups = [], []
        for lead in self._lead_values()[:n_l]:
            for m in range(ncomp):
                fn.append(f"EOF:{var_name}:L{lead}:PC{m+1}")
                groups.append(f"EOF:{var_name}")
        return X, x, fn, groups

    def _spatial_box_features(self, candidate_states, target_state, years, var_name):
        """Coarse area-weighted box features for an interpretable spatial metric."""
        if "Y" not in candidate_states.coords or "X" not in candidate_states.coords:
            return None
        lat = np.asarray(candidate_states["Y"].values, float)
        lon = np.asarray(candidate_states["X"].values, float)
        deg = max(float(self.spatial_box_deg), 1.0)
        lat_bin = np.floor((lat + 90.0) / deg).astype(int)
        # normalise longitude to [-180,180) before binning
        lon180 = ((lon + 180.0) % 360.0) - 180.0
        lon_bin = np.floor((lon180 + 180.0) / deg).astype(int)
        arr = np.asarray(candidate_states.transpose("A", "L", "Y", "X").values, float)
        tar = np.asarray(target_state.transpose("L", "Y", "X").values, float)
        wlat = np.clip(np.cos(np.deg2rad(lat)), 0.0, None)
        boxes = [(int(a), int(b)) for a in np.unique(lat_bin) for b in np.unique(lon_bin)]
        cols, tcols, names, groups = [], [], [], []
        for ib, (yb, xb) in enumerate(boxes):
            mask_y = lat_bin == yb
            mask_x = lon_bin == xb
            if not mask_y.any() or not mask_x.any():
                continue
            sub = arr[:, :, mask_y, :][:, :, :, mask_x]
            st = tar[:, mask_y, :][:, :, mask_x]
            ww = wlat[mask_y][:, None] * np.ones((mask_y.sum(), mask_x.sum()))
            den = np.sum(ww)
            if den <= 0:
                continue
            # weighted spatial means with NaN-aware denominator
            valid = np.isfinite(sub)
            d = np.sum(np.where(valid, ww[None, None, :, :], 0.0), axis=(2, 3))
            v = np.sum(np.where(valid, sub * ww[None, None, :, :], 0.0), axis=(2, 3))
            cm = np.divide(v, d, out=np.full_like(v, np.nan), where=d > 0)
            vt = np.isfinite(st)
            dt = np.sum(np.where(vt, ww[None, :, :], 0.0), axis=(1, 2))
            tv = np.sum(np.where(vt, st * ww[None, :, :], 0.0), axis=(1, 2))
            tm = np.divide(tv, dt, out=np.full_like(tv, np.nan), where=dt > 0)
            # identify a readable box centre/range
            lat0 = -90.0 + yb * deg; lat1 = lat0 + deg
            lon0 = -180.0 + xb * deg; lon1 = lon0 + deg
            group = f"BOX:{var_name}:{lat0:g}_{lat1:g}:{lon0:g}_{lon1:g}"
            for il, lead in enumerate(self._lead_values()[:cm.shape[1]]):
                cols.append(cm[:, il])
                tcols.append(tm[il])
                names.append(f"{group}:L{lead}")
                groups.append(group)
        if not cols:
            return None
        return np.column_stack(cols), np.asarray(tcols, float), names, groups

    @staticmethod
    def _detrend_feature_matrix(X, x, years, target_year):
        return WAS_Analog_Scientific._fold_linear_detrend(X, years, x, target_year)

    @staticmethod
    def _standardize_features(X, x, names, groups):
        X = np.asarray(X, float); x = np.asarray(x, float)
        count = np.isfinite(X).sum(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mu = np.nanmean(X, axis=0); sd = np.nanstd(X, axis=0)
        # Remove features unavailable at the target instead of declaring their
        # standardized anomaly zero (which would invent a typical climate state).
        keep = np.isfinite(x) & np.isfinite(mu) & (sd > 1e-8) & (count >= max(2, int(np.ceil(.8 * len(X)))))
        if not np.any(keep):
            raise ValueError("No observed target features with sufficient varying training data")
        result = (X[:, keep] - mu[keep]) / sd[keep]
        if np.any(np.isfinite(result).sum(axis=1) == 0):
            raise ValueError("An analogue candidate has no observed features")
        result = np.where(np.isfinite(result), result, 0.)
        return result, (x[keep] - mu[keep]) / sd[keep], [name for name, use in zip(names, keep) if use], [group for group, use in zip(groups, keep) if use]

    def _build_feature_space(self, train_years, target_year, target_source="hindcast", member=None):
        """Build candidate reanalysis features and one model target feature vector."""
        years = np.asarray(sorted({int(v) for v in train_years}), dtype=int)
        if years.size < self.min_analogs:
            raise ValueError("Too few historical years for analogue selection")
        hist, hdcst, fcst = self._get_raw_sources()
        missing = [p["variable"] for p in self.predictor_vars if p.get("variable") not in hist]
        if missing:
            raise KeyError(f"Requested analogue reanalysis predictors unavailable: {missing}")
        variables = [p["variable"] for p in self.predictor_vars]
        if not variables:
            raise ValueError("No requested analogue predictor is available in reanalysis data")

        feature_blocks, target_blocks, names, groups = [], [], [], []
        sst_index_block = None
        mode = self.feature_mode.lower()
        if self.method_analog in ("eof", "pca_based"):
            mode = "eof"
        elif self.method_analog in ("hybrid", "scientific", "learned_metric"):
            mode = "hybrid"
        elif self.method_analog == "learned_spatial_metric":
            mode = "spatial_boxes"
        elif mode == "auto":
            mode = "indices" if self.index_compute else "eof"

        for var in variables:
            hist_da = self._as_dataarray(hist[var], var)
            candidate = xr.concat(
                [self._reanalysis_state(hist_da, int(y)).expand_dims(A=[int(y)]) for y in years],
                dim="A",
            )
            if bool((~np.isfinite(candidate).any(("Y", "X"))).any()):
                raise ValueError(f"Missing reanalysis month in the candidate library for {var}")
            if target_source == "forecast":
                if var not in fcst:
                    raise KeyError(f"Forecast predictor {var} unavailable")
                target = self._model_state(fcst[var], forecast=True, member=member)
            elif target_source == "hindcast":
                if var not in hdcst:
                    if self.fallback_to_reanalysis_target or not self.require_model_hindcast_target:
                        target = self._reanalysis_state(hist_da, int(target_year))
                    else:
                        raise KeyError(f"Hindcast predictor {var} unavailable")
                else:
                    try:
                        target = self._model_state(hdcst[var], int(target_year), forecast=False, member=member)
                    except Exception:
                        if self.fallback_to_reanalysis_target or not self.require_model_hindcast_target:
                            target = self._reanalysis_state(hist_da, int(target_year))
                        else:
                            raise
            elif target_source == "reanalysis":
                target = self._reanalysis_state(hist_da, int(target_year))
            else:
                raise ValueError(f"Unknown target_source={target_source!r}")

            target = self._interp_like_state(target, candidate.isel(A=0, drop=True))
            if bool((~np.isfinite(target).any(("Y", "X"))).any()):
                raise ValueError(f"Missing model target lead or spatial overlap for {var}")
            if target_source in ("hindcast", "forecast") and var in hdcst:
                target = self._bias_correct_target(var, target, hist_da, hdcst[var], years, member=member)

            if self.detrend:
                shape = candidate.shape
                values, target_values = self._fold_linear_detrend(
                    candidate.transpose("A", "L", "Y", "X").values, years,
                    target.transpose("L", "Y", "X").values, target_year)
                candidate = candidate.transpose("A", "L", "Y", "X").copy(data=values)
                target = target.transpose("L", "Y", "X").copy(data=target_values)

            if mode in ("indices", "hybrid") and var.upper() == "SST":
                blk = self._index_features(candidate, target, years)
                if blk is not None:
                    Xb, xb, nb, gb = blk
                    feature_blocks.append(Xb); target_blocks.append(xb)
                    names.extend(nb); groups.extend(gb)
                    sst_index_block = True

            if mode in ("eof", "hybrid"):
                blk = self._eof_features(candidate, target, years, var, target_year)
                if blk is not None:
                    Xb, xb, nb, gb = blk
                    feature_blocks.append(Xb); target_blocks.append(xb)
                    names.extend(nb); groups.extend(gb)

            if mode in ("spatial_boxes", "hybrid_spatial"):
                blk = self._spatial_box_features(candidate, target, years, var)
                if blk is not None:
                    Xb, xb, nb, gb = blk
                    feature_blocks.append(Xb); target_blocks.append(xb)
                    names.extend(nb); groups.extend(gb)

        if not feature_blocks:
            raise ValueError(f"No usable {mode!r} feature block was built")
        X = np.concatenate(feature_blocks, axis=1)
        x = np.concatenate(target_blocks)
        X, x, names, groups = self._standardize_features(X, x, names, groups)
        return {
            "years": years, "X": X, "x": x,
            "feature_names": names, "groups": groups,
            "target_year": int(target_year), "target_source": target_source,
        }

    # -------------------------------------------------------------- distances
    @staticmethod
    def _correlation_distance(X, x, weights=None):
        X = np.asarray(X, float); x = np.asarray(x, float)
        weights = np.ones(x.size) / max(x.size, 1) if weights is None else np.asarray(weights, float)
        weights = weights / weights.sum()
        xc = x - np.sum(weights * x)
        xs = np.sqrt(np.sum(weights * xc * xc))
        out = np.full(X.shape[0], 1.0, dtype=float)
        if xs <= 1e-12:
            return out
        for i, row in enumerate(X):
            rc = row - np.sum(weights * row)
            den = np.sqrt(np.sum(weights * rc * rc)) * xs
            if den > 1e-12:
                out[i] = 1.0 - np.clip(np.sum(weights * rc * xc) / den, -1.0, 1.0)
        return out

    @staticmethod
    def _scale_distance(d):
        d = np.asarray(d, float)
        pos = d[np.isfinite(d) & (d > 0)]
        scale = np.nanmedian(pos) if pos.size else 1.0
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        return d / scale

    @staticmethod
    def _feature_weights(groups, group_weights):
        groups = list(groups)
        ug = list(dict.fromkeys(groups))
        gw = np.asarray([group_weights.get(g, 0.0) for g in ug], float)
        gw = np.maximum(gw, 0.0)
        if gw.sum() <= 0:
            gw[:] = 1.0
        gw /= gw.sum()
        counts = {g: max(1, groups.count(g)) for g in ug}
        fmap = {g: gw[i] / counts[g] for i, g in enumerate(ug)}
        return np.asarray([fmap[g] for g in groups], float)

    def _distances(self, X, x, metric=None, alpha=None, groups=None, group_weights=None):
        metric = (metric or self.analog_metric).lower()
        alpha = self.hybrid_alpha if alpha is None else float(alpha)
        X = np.asarray(X, float); x = np.asarray(x, float)
        if group_weights is not None and groups is not None:
            fw = self._feature_weights(groups, group_weights)
        else:
            fw = np.full(X.shape[1], 1.0 / max(1, X.shape[1]))
        de = np.sqrt(np.maximum(0.0, np.sum(fw[None, :] * (X - x[None, :]) ** 2, axis=1)))
        if metric in ("euclidean", "weighted_euclidean"):
            return de
        if metric == "correlation":
            return self._correlation_distance(X, x, fw)
        if metric == "mahalanobis":
            try:
                from sklearn.covariance import LedoitWolf
                cov = LedoitWolf().fit(X)
                delta = X - x[None, :]
                return np.sqrt(np.maximum(0.0, np.einsum("ij,jk,ik->i", delta, cov.precision_, delta)))
            except Exception:
                return de
        if metric == "hybrid":
            dc = self._correlation_distance(X, x, fw)
            # Scale with library distances only: target-dependent normalization
            # erases amplitude novelty and defeats the quality/OOD diagnostic.
            from scipy.spatial.distance import pdist
            reference = pdist(X * np.sqrt(fw)[None, :])
            positive = reference[np.isfinite(reference) & (reference > 1e-12)]
            scale = float(np.median(positive)) if positive.size else 1.0
            return alpha * de / scale + (1.0 - alpha) * dc
        raise ValueError(f"Unknown analogue metric {metric!r}")

    def _weights_from_distance(self, d, weighting=None, tau=None):
        d = np.asarray(d, float)
        weighting = (weighting or self.analog_weighting).lower()
        tau = self.tau if tau is None else float(tau)
        if d.size == 0:
            return d
        if weighting == "equal":
            return np.full(d.size, 1.0 / d.size)
        if weighting == "inverse_distance":
            w = 1.0 / np.maximum(d, 1e-8) ** max(self.inverse_distance_power, 1e-8)
        elif weighting in ("softmax", "gaussian_kernel"):
            if weighting == "gaussian_kernel":
                scale = max(tau, 1e-8)
                w = np.exp(-0.5 * (d / scale) ** 2)
            else:
                scale = max(tau, 1e-8)
                z = -(d - np.nanmin(d)) / scale
                z = np.clip(z, -700, 0)
                w = np.exp(z)
        else:
            raise ValueError(f"Unknown analogue weighting {weighting!r}")
        if not np.isfinite(w).all() or w.sum() <= 0:
            return np.full(d.size, 1.0 / d.size)
        return w / w.sum()

    def _method_selection(self, X, x, params, groups, group_weights=None):
        """Fit clustering only on candidates; assign the unseen target afterward."""
        method = self.method_analog
        metric = params["metric"]
        if method == "cor_based":
            metric = "correlation"
        elif method == "pca_based":
            metric = "euclidean"
        d = self._distances(X, x, metric, params["alpha"], groups, group_weights)
        if method == "bias_based":
            d = np.mean(np.abs(X - x), axis=1)
        eligible = np.arange(len(X))
        if method in {"som", "kmeans_based", "agglomerative_based"}:
            # Labels depend only on the training feature matrix. Reuse across
            # k/tau tuning candidates; no held-out target enters a fit.
            import hashlib
            signature = (method, hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest(),
                         self.n_clusters, self.affinity, self.linkage, self.some_grid_size,
                         self.some_sigma, self.some_learning_rate, self.some_num_iteration,
                         self.some_neighborhood_function, self.random_state)
            cache = getattr(self, "_analog_selector_cache", None)
            if cache is None:
                self._analog_selector_cache = cache = {}
            fitted = cache.get(signature)
            if fitted is None:
                if method == "som":
                    grid = (4, 4) if all(value is None for value in self.some_grid_size) else self.some_grid_size
                    model = MiniSom(int(grid[0]), int(grid[1]), X.shape[1], sigma=self.some_sigma,
                                    learning_rate=self.some_learning_rate,
                                    neighborhood_function=self.some_neighborhood_function,
                                    random_seed=self.random_state)
                    model.random_weights_init(X)
                    model.train_random(X, self.some_num_iteration)
                    labels = np.asarray([model.winner(row) for row in X])
                elif method == "kmeans_based":
                    clusters = min(int(self.n_clusters), len(np.unique(X, axis=0)))
                    model = KMeans(n_clusters=clusters, random_state=self.random_state, n_init=10).fit(X)
                    labels = model.labels_
                else:
                    clusters = min(int(self.n_clusters), len(X))
                    model = AgglomerativeClustering(n_clusters=clusters, metric=self.affinity, linkage=self.linkage).fit(X)
                    labels = model.labels_
                fitted = cache[signature] = (model, labels)
            model, labels = fitted
            if method == "som":
                winner = np.asarray(model.winner(x))
                topology_distance = np.linalg.norm(labels - winner[None, :], axis=1)
                eligible = np.flatnonzero(topology_distance == 0)
                if len(eligible) < self.min_analogs:
                    eligible = np.flatnonzero(topology_distance <= max(float(self.radius), 0.))
            elif method == "kmeans_based":
                eligible = np.flatnonzero(labels == model.predict(np.asarray(x)[None, :])[0])
            else:
                # Agglomerative clustering has no predict(): an explicit nearest
                # training-centroid assignment supplies the out-of-sample rule.
                unique = np.unique(labels)
                centres = np.asarray([X[labels == label].mean(axis=0) for label in unique])
                from sklearn.metrics import pairwise_distances
                target_label = unique[np.argmin(pairwise_distances(np.asarray(x)[None, :], centres, metric=self.affinity))]
                eligible = np.flatnonzero(labels == target_label)
            if len(eligible) < self.min_analogs:
                # Deterministic distance fallback; the target never joins training.
                eligible = np.arange(len(X))
        eligible = eligible[np.isfinite(d[eligible])]
        take = eligible[np.argsort(d[eligible], kind="stable")[:min(max(self.min_analogs, int(params["k"])), len(eligible))]]
        if not len(take):
            raise ValueError("No finite analogue distances")
        weights = self._weights_from_distance(d[take], params["weighting"], params["tau"])
        return take, weights, d[take]

    @staticmethod
    def _predictand_numpy(predictant, years):
        """Cache annual predictand maps as (year, pixel) NumPy arrays."""
        mats = []
        for y in years:
            da = WAS_Analog_Scientific._map_for_year(predictant, int(y))
            mats.append(np.asarray(da.transpose("Y", "X").values, float).ravel())
        Ymat = np.asarray(mats, float)
        if "Y" in predictant.coords and "X" in predictant.coords:
            lat = np.asarray(predictant["Y"].values, float)
            nx = int(predictant.sizes["X"])
            sw = np.repeat(np.clip(np.cos(np.deg2rad(lat)), 0.0, None), nx)
        else:
            sw = np.ones(Ymat.shape[1], float)
        return Ymat, sw

    @staticmethod
    def _weighted_np(mats, weights):
        mats = np.asarray(mats, float)
        w = np.asarray(weights, float)[:, None]
        valid = np.isfinite(mats)
        den = np.sum(np.where(valid, w, 0.0), axis=0)
        num = np.sum(np.where(valid, mats * w, 0.0), axis=0)
        return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)

    @staticmethod
    def _mse_np(obs, pred, spatial_weights=None):
        obs = np.asarray(obs, float); pred = np.asarray(pred, float)
        valid = np.isfinite(obs) & np.isfinite(pred)
        if not np.any(valid):
            return np.inf
        if spatial_weights is None:
            return float(np.mean((obs[valid] - pred[valid]) ** 2))
        sw = np.asarray(spatial_weights, float)
        ww = sw[valid]
        if ww.sum() <= 0:
            return float(np.mean((obs[valid] - pred[valid]) ** 2))
        return float(np.sum(ww * (obs[valid] - pred[valid]) ** 2) / np.sum(ww))

    # ------------------------------------------------------------ verification
    @staticmethod
    def _map_for_year(predictant, year):
        sel = predictant.where(predictant["T"].dt.year == int(year), drop=True)
        if sel.sizes.get("T", 0) == 0:
            raise KeyError(f"Predictand year {year} not available")
        return sel.mean("T", skipna=True)

    @staticmethod
    def _weighted_map(maps, weights):
        arr = xr.concat(maps, dim="analog")
        w = xr.DataArray(np.asarray(weights, float), dims=("analog",))
        valid = np.isfinite(arr)
        den = xr.where(valid, w, 0.0).sum("analog")
        num = xr.where(valid, arr * w, 0.0).sum("analog")
        return xr.where(den > 0, num / den, np.nan)

    @staticmethod
    def _spatial_mse(obs, pred):
        diff2 = (obs - pred) ** 2
        if "Y" in diff2.dims:
            w = xr.DataArray(np.clip(np.cos(np.deg2rad(diff2["Y"])), 0, None),
                             coords={"Y": diff2["Y"]}, dims="Y")
            val = diff2.weighted(w).mean(("Y", "X"), skipna=True)
        else:
            val = diff2.mean(skipna=True)
        try:
            return float(val.values)
        except Exception:
            return np.inf

    @staticmethod
    def _rps_np(obs, candidate_maps, weights, threshold_maps, spatial_weights=None, thresholds=None):
        obs = np.asarray(obs, float)
        cm = np.asarray(candidate_maps, float)
        w = np.asarray(weights, float)[:, None]
        th = np.asarray(threshold_maps, float)
        if thresholds is None:
            if th.shape[0] < 3:
                return np.inf
            t1 = np.nanquantile(th, 1.0/3.0, axis=0)
            t2 = np.nanquantile(th, 2.0/3.0, axis=0)
        else:
            t1, t2 = thresholds
        valid = np.isfinite(cm)
        den = np.sum(np.where(valid, w, 0.0), axis=0)
        pb = np.sum(np.where(valid & (cm <= t1), w, 0.0), axis=0)
        pn = np.sum(np.where(valid & (cm > t1) & (cm <= t2), w, 0.0), axis=0)
        pb = np.divide(pb, den, out=np.full_like(pb, np.nan), where=den>0)
        pn = np.divide(pn, den, out=np.full_like(pn, np.nan), where=den>0)
        o1 = (obs <= t1).astype(float)
        o2 = (obs <= t2).astype(float)
        rps = (pb-o1)**2 + (pb+pn-o2)**2
        valid2 = np.isfinite(rps) & np.isfinite(obs) & np.isfinite(t1) & np.isfinite(t2)
        if not np.any(valid2):
            return np.inf
        if spatial_weights is None:
            return float(np.mean(rps[valid2]))
        sw = np.asarray(spatial_weights, float)
        ww = sw[valid2]
        return float(np.sum(ww*rps[valid2])/np.sum(ww)) if ww.sum()>0 else float(np.mean(rps[valid2]))

    @staticmethod
    def _rps_map(obs, p, t1, t2):
        # p shape (3,Y,X); RPS over the two cumulative category boundaries.
        below = obs <= t1
        normal = (obs > t1) & (obs <= t2)
        o1 = below.astype(float)
        o2 = (below | normal).astype(float)
        f1 = p[0]
        f2 = p[0] + p[1]
        valid = np.isfinite(obs) & np.isfinite(t1) & np.isfinite(t2) & np.isfinite(p).all("probability")
        return ((f1 - o1) ** 2 + (f2 - o2) ** 2).where(valid)

    def _inner_library(self, years, val_year):
        """Omit validation year and +/- inner_nb_omit calendar years."""
        yrs = np.asarray(sorted({int(v) for v in years}), int)
        if int(val_year) not in yrs:
            raise ValueError("Validation year is absent from the inner archive")
        return yrs[np.abs(yrs - int(val_year)) > self.inner_nb_omit]

    def _tune_parameters_outer_features(self, package, predictant):
        """Tune k/metric/weighting on outer-training years only.

        Predictand maps are materialised once as NumPy matrices; the inner
        hyperparameter search then performs only small dense-array operations.
        """
        years = package["years"]
        X = package["X"]
        groups = package["groups"]
        y_to_i = {int(y): i for i, y in enumerate(years)}
        Ymat, sw = self._predictand_numpy(predictant, years)
        _, fixed_q = self._fixed_tercile_reference(predictant)
        fixed_thresholds = (None if fixed_q is None else
                            np.asarray(fixed_q.transpose("quantile", "Y", "X")).reshape(2, -1))
        rows = []
        metrics = self.metric_grid if self.analog_metric == "auto" else (self.analog_metric,)
        if self.method_analog in ("learned_metric", "learned_spatial_metric"):
            metrics = ("weighted_euclidean",)
        weightings = self.weighting_grid if self.analog_weighting == "auto" else (self.analog_weighting,)
        alphas = self.hybrid_alpha_grid

        for metric in metrics:
            metric_alphas = alphas if metric == "hybrid" else (self.hybrid_alpha,)
            for alpha in metric_alphas:
                for weighting in weightings:
                    taus = self.tau_grid if weighting in ("softmax", "gaussian_kernel") else (self.tau,)
                    for tau in taus:
                        for k in self.k_grid:
                            fold_scores = []
                            for vy in years:
                                lib_years = self._inner_library(years, int(vy))
                                if lib_years.size < self.min_analogs:
                                    continue
                                idx = np.asarray([y_to_i[int(y)] for y in lib_years], int)
                                vi = y_to_i[int(vy)]
                                take, w, _ = self._method_selection(
                                    X[idx], X[vi], {"metric": metric, "alpha": alpha,
                                    "weighting": weighting, "tau": tau, "k": k}, groups)
                                if len(take) < self.min_analogs:
                                    continue
                                if self.tune_score == "rps":
                                    fold_scores.append(self._rps_np(
                                        Ymat[vi], Ymat[idx[take]], w, Ymat[idx], sw, thresholds=fixed_thresholds
                                    ))
                                else:
                                    pred = self._weighted_np(Ymat[idx[take]], w)
                                    fold_scores.append(self._mse_np(Ymat[vi], pred, sw))
                            if fold_scores:
                                a = np.asarray(fold_scores, float)
                                rows.append({
                                    "metric": metric, "alpha": float(alpha),
                                    "weighting": weighting, "tau": float(tau), "k": int(k),
                                    "score": float(np.nanmean(a)),
                                    "se": float(np.nanstd(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0,
                                    "n_folds": len(a),
                                })
        rows = [row for row in rows if np.isfinite(row["score"]) and np.isfinite(row["se"])]
        if not rows:
            return {
                "metric": self.analog_metric if self.analog_metric != "auto" else "hybrid",
                "alpha": self.hybrid_alpha,
                "weighting": self.analog_weighting if self.analog_weighting != "auto" else "softmax",
                "tau": self.tau,
                "k": self.n_analogs,
            }
        df = pd.DataFrame(rows).sort_values("score", kind="stable").reset_index(drop=True)
        self._analog_tuning_table = df
        best = df.iloc[0]
        if self.one_se:
            cutoff = float(best["score"] + best["se"])
            ok = df[df["score"] <= cutoff].copy()
            w_rank = {"equal": 0, "inverse_distance": 1, "softmax": 2, "gaussian_kernel": 3}
            m_rank = {"euclidean": 0, "correlation": 1, "hybrid": 2, "mahalanobis": 3,
                      "weighted_euclidean": 4}
            ok["_wr"] = ok["weighting"].map(w_rank).fillna(9)
            ok["_mr"] = ok["metric"].map(m_rank).fillna(9)
            best = ok.sort_values(["k", "_wr", "_mr", "score"], ascending=[False, True, True, True], kind="stable").iloc[0]
        return {k: best[k].item() if hasattr(best[k], "item") else best[k]
                for k in ("metric", "alpha", "weighting", "tau", "k")}

    def _strict_inner_packages(self, predictant, outer_years, member=None, *, learn_weights=True):
        """Build truly nested inner-validation feature packages.

        Each inner validation year is represented by its own GCM hindcast target;
        all bias correction, scaling and EOF/PCA preprocessing are refitted using
        only the corresponding inner analogue library.
        """
        import hashlib
        training_targets = predictant.where(predictant["T"].dt.year.isin(outer_years), drop=True)
        target_signature = hashlib.sha256(np.asarray(training_targets.values).tobytes()).hexdigest()
        key = (
            "strict_inner", tuple(int(v) for v in sorted(outer_years)), target_signature,
            self.feature_mode, self.analog_bias_method, bool(self.detrend),
            tuple(self._lead_values()), member, bool(learn_weights),
        )
        if key in self._analog_feature_cache:
            return self._analog_feature_cache[key]
        packs = []
        for vy in np.asarray(sorted({int(v) for v in outer_years}), int):
            lib = self._inner_library(outer_years, int(vy))
            if lib.size < self.min_analogs:
                continue
            try:
                pkg = self._build_feature_space(lib, int(vy), target_source="hindcast", member=member)
                if learn_weights and self.method_analog in ("learned_metric", "learned_spatial_metric"):
                    subpackages = self._strict_inner_packages(predictant, lib, member=member,
                                                              learn_weights=False)
                    pkg["group_weights"] = self._learn_group_weights(
                        pkg, predictant, self.n_analogs, strict_packages=subpackages)
                packs.append((int(vy), pkg))
            except Exception as exc:
                raise ValueError(f"Strict inner analogue fold {vy} failed: {exc}") from exc
        self._analog_feature_cache[key] = packs
        return packs

    def _tune_parameters_strict(self, packages, predictant):
        _, fixed_q = self._fixed_tercile_reference(predictant)
        fixed_thresholds = (None if fixed_q is None else
                            np.asarray(fixed_q.transpose("quantile", "Y", "X")).reshape(2, -1))
        rows = []
        all_years = np.asarray(np.unique(predictant["T"].dt.year.values), int)
        Yall, sw = self._predictand_numpy(predictant, all_years)
        y_lookup = {int(y): i for i, y in enumerate(all_years)}
        metrics = self.metric_grid if self.analog_metric == "auto" else (self.analog_metric,)
        if self.method_analog in ("learned_metric", "learned_spatial_metric"):
            metrics = ("weighted_euclidean",)
        weightings = self.weighting_grid if self.analog_weighting == "auto" else (self.analog_weighting,)
        for metric in metrics:
            alphas = self.hybrid_alpha_grid if metric == "hybrid" else (self.hybrid_alpha,)
            for alpha in alphas:
                for weighting in weightings:
                    taus = self.tau_grid if weighting in ("softmax", "gaussian_kernel") else (self.tau,)
                    for tau in taus:
                        for k in self.k_grid:
                            fs = []
                            for vy, pkg in packages:
                                take, w, _ = self._method_selection(
                                    pkg["X"], pkg["x"], {"metric": metric, "alpha": alpha,
                                    "weighting": weighting, "tau": tau, "k": k}, pkg["groups"],
                                    pkg.get("group_weights"))
                                if len(take) < self.min_analogs:
                                    continue
                                inds = [y_lookup[int(pkg["years"][j])] for j in take]
                                lib_inds = [y_lookup[int(y)] for y in pkg["years"]]
                                if self.tune_score == "rps":
                                    fs.append(self._rps_np(
                                        Yall[y_lookup[int(vy)]], Yall[inds], w, Yall[lib_inds], sw, thresholds=fixed_thresholds
                                    ))
                                else:
                                    pred = self._weighted_np(Yall[inds], w)
                                    fs.append(self._mse_np(Yall[y_lookup[int(vy)]], pred, sw))
                            if fs:
                                a = np.asarray(fs, float)
                                rows.append({
                                    "metric": metric, "alpha": float(alpha),
                                    "weighting": weighting, "tau": float(tau), "k": int(k),
                                    "score": float(np.nanmean(a)),
                                    "se": float(np.nanstd(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0,
                                    "n_folds": len(a),
                                })
        rows = [row for row in rows if np.isfinite(row["score"]) and np.isfinite(row["se"])]
        if not rows:
            return {
                "metric": self.analog_metric if self.analog_metric != "auto" else "hybrid",
                "alpha": self.hybrid_alpha,
                "weighting": self.analog_weighting if self.analog_weighting != "auto" else "softmax",
                "tau": self.tau, "k": self.n_analogs,
            }
        df = pd.DataFrame(rows).sort_values("score", kind="stable").reset_index(drop=True)
        self._analog_tuning_table = df
        best = df.iloc[0]
        if self.one_se:
            cutoff = float(best["score"] + best["se"])
            ok = df[df["score"] <= cutoff].copy()
            w_rank = {"equal": 0, "inverse_distance": 1, "softmax": 2, "gaussian_kernel": 3}
            m_rank = {"euclidean": 0, "correlation": 1, "hybrid": 2, "mahalanobis": 3,
                      "weighted_euclidean": 4}
            ok["_wr"] = ok["weighting"].map(w_rank).fillna(9)
            ok["_mr"] = ok["metric"].map(m_rank).fillna(9)
            best = ok.sort_values(["k", "_wr", "_mr", "score"], ascending=[False, True, True, True], kind="stable").iloc[0]
        return {k: best[k].item() if hasattr(best[k], "item") else best[k]
                for k in ("metric", "alpha", "weighting", "tau", "k")}

    # ------------------------------------------------------ learned metric core
    def _learn_group_weights(self, package, predictant, k=None, strict_packages=None):
        groups = list(package["groups"])
        ug = list(dict.fromkeys(groups))
        if len(ug) <= 1:
            return {ug[0]: 1.0} if ug else {}
        X = package["X"]; years = package["years"]
        y_to_i = {int(y): i for i, y in enumerate(years)}
        Ymat, sw = self._predictand_numpy(predictant, years)
        # Put predictive loss and metric regularisation on comparable scales.
        y_scale = float(np.nanvar(Ymat))
        if not np.isfinite(y_scale) or y_scale <= 1e-12:
            y_scale = 1.0
        k = int(k or self.n_analogs)

        def softmax(z):
            z = z - np.max(z)
            e = np.exp(np.clip(z, -50, 50))
            return e / e.sum()

        def objective(z):
            gwv = softmax(z)
            gwd = {g: gwv[i] for i, g in enumerate(ug)}
            scores = []
            folds = (strict_packages if strict_packages is not None else [
                (int(vy), None) for vy in years])
            for vy, pkg in folds:
                lib = self._inner_library(years, int(vy)) if pkg is None else pkg["years"]
                if len(lib) < self.min_analogs:
                    continue
                idx = np.asarray([y_to_i[int(v)] for v in lib], int)
                vi = y_to_i[int(vy)]
                Xi = X[idx] if pkg is None else pkg["X"]
                xv = X[vi] if pkg is None else pkg["x"]
                gi = groups if pkg is None else pkg["groups"]
                d = self._distances(Xi, xv, metric="weighted_euclidean",
                                    groups=gi, group_weights=gwd)
                # Smooth full-library kernel loss gives a differentiable metric
                # objective, including when final predictions use equal kNN weights.
                # Hard top-k + equal weights has a locally constant objective.
                w = self._weights_from_distance(d, weighting="softmax", tau=self.tau)
                pred = self._weighted_np(Ymat[idx], w)
                scores.append(self._mse_np(Ymat[vi], pred, sw))
            if not scores:
                return 1e12
            eq = 1.0 / len(ug)
            penalty = self.metric_weight_l2 * np.sum((gwv - eq) ** 2)
            return float(np.nanmean(scores) / y_scale + penalty)

        try:
            from scipy.optimize import minimize
            res = minimize(objective, np.zeros(len(ug)), method="L-BFGS-B",
                           options={"maxiter": 80, "ftol": 1e-8})
            w = softmax(res.x if res.success else np.zeros(len(ug)))
        except (ValueError, FloatingPointError) as exc:
            warnings.warn(f"Metric learning failed; using equal group weights: {exc}", RuntimeWarning)
            w = np.full(len(ug), 1.0 / len(ug))
        return {g: float(w[i]) for i, g in enumerate(ug)}

    # -------------------------------------------------------- constructed AnEn
    def _constructed_solution(self, X, x, metric, alpha, groups, group_weights,
                              pool_k, ridge_lambda):
        d = self._distances(X, x, metric=metric, alpha=alpha,
                            groups=groups, group_weights=group_weights)
        pool = np.argsort(d)[:min(int(pool_k), len(d))]
        if len(pool) < self.min_analogs:
            return pool, self._weights_from_distance(d[pool], "equal")
        Xp = X[pool]
        n = len(pool)
        x0 = np.full(n, 1.0 / n)

        def obj(w):
            recon = w @ Xp
            return float(np.mean((recon - x) ** 2) + float(ridge_lambda) * np.mean((w - 1.0/n) ** 2))

        try:
            from scipy.optimize import minimize
            res = minimize(obj, x0, method="SLSQP",
                           bounds=[(0.0, 1.0)] * n,
                           constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                           options={"maxiter": 300, "ftol": 1e-10})
            w = res.x if res.success else x0
        except Exception:
            w = x0
        w = np.maximum(w, 0.0)
        w = w / w.sum() if w.sum() > 0 else x0
        keep = w > 1e-7
        return pool[keep], w[keep] / w[keep].sum()

    def _tune_constructed_lambda(self, package, predictant, params, group_weights=None,
                                    strict_packages=None):
        """Select constructed-analogue ridge strength by inner validation."""
        rows = []
        if strict_packages is not None:
            all_years = np.asarray(np.unique(predictant["T"].dt.year.values), int)
            Yall, sw = self._predictand_numpy(predictant, all_years)
            y_lookup = {int(y): i for i, y in enumerate(all_years)}
            for lam in self.constructed_lambda_grid:
                fs = []
                for vy, pkg in strict_packages:
                    take, w = self._constructed_solution(
                        pkg["X"], pkg["x"], params["metric"], params["alpha"], pkg["groups"],
                        pkg.get("group_weights"), max(self.constructed_pool_k, params["k"]), lam,
                    )
                    if len(take) < self.min_analogs:
                        continue
                    inds = [y_lookup[int(pkg["years"][j])] for j in take]
                    pred = self._weighted_np(Yall[inds], w)
                    fs.append(self._mse_np(Yall[y_lookup[int(vy)]], pred, sw))
                if fs:
                    a = np.asarray(fs, float)
                    rows.append((float(lam), float(np.mean(a)),
                                 float(np.std(a, ddof=1)/np.sqrt(len(a))) if len(a)>1 else 0.0))
        else:
            years = package["years"]; X = package["X"]; groups = package["groups"]
            Ymat, sw = self._predictand_numpy(predictant, years)
            y_to_i = {int(y): i for i, y in enumerate(years)}
            for lam in self.constructed_lambda_grid:
                fs = []
                for vy in years:
                    lib = self._inner_library(years, int(vy))
                    if lib.size < self.min_analogs:
                        continue
                    idx = np.asarray([y_to_i[int(y)] for y in lib], int)
                    take, w = self._constructed_solution(
                        X[idx], X[y_to_i[int(vy)]], params["metric"], params["alpha"], groups,
                        group_weights, max(self.constructed_pool_k, params["k"]), lam,
                    )
                    if len(take) < self.min_analogs:
                        continue
                    pred = self._weighted_np(Ymat[idx[take]], w)
                    fs.append(self._mse_np(Ymat[y_to_i[int(vy)]], pred, sw))
                if fs:
                    a = np.asarray(fs, float)
                    rows.append((float(lam), float(np.mean(a)),
                                 float(np.std(a, ddof=1)/np.sqrt(len(a))) if len(a)>1 else 0.0))
        if not rows:
            return self.constructed_lambda
        rows.sort(key=lambda r: r[1])
        best = rows[0]
        if self.one_se:
            cutoff = best[1] + best[2]
            # Prefer stronger regularisation when statistically equivalent.
            candidates = [r for r in rows if r[1] <= cutoff]
            best = max(candidates, key=lambda r: r[0])
        return float(best[0])

    # -------------------------------------------------------- quality / probs
    def _quality_percentile(self, X, x, metric, alpha, groups, group_weights):
        if len(X) < 3:
            return 1.0
        target_d = self._distances(X, x, metric, alpha, groups, group_weights)
        d1 = float(np.nanmin(target_d))
        nn = []
        for i in range(len(X)):
            mask = np.arange(len(X)) != i
            d = self._distances(X[mask], X[i], metric, alpha, groups, group_weights)
            if np.isfinite(d).any():
                nn.append(float(np.nanmin(d)))
        if not nn:
            return 1.0
        return float(np.mean(np.asarray(nn) <= d1))

    def _fixed_tercile_reference(self, predictant, start=None, end=None):
        from .was_analog_probability import event_reference
        return event_reference(self,predictant,start,end)


    @staticmethod
    def _analog_probability_maps(predictant, analog_years, weights, threshold_sample,
                                 shrinkage=0.0, thresholds=None):
        from .was_analog_probability import probability_maps
        return probability_maps(WAS_Analog_Scientific,predictant,analog_years,weights,
                                threshold_sample,shrinkage,thresholds)


    def _tune_probability_shrinkage(self, package, predictant, params, group_weights=None,
                                    strict_packages=None):
        if self.probability_shrinkage != "auto":
            return float(self.probability_shrinkage)
        fixed_clim, fixed_q = self._fixed_tercile_reference(predictant)
        scores = {lam: [] for lam in self.probability_shrinkage_grid}
        years = package["years"]; X = package["X"]; groups = package["groups"]
        y_to_i = {int(y): i for i, y in enumerate(years)}
        folds = (strict_packages if strict_packages is not None else [
            (int(vy), None) for vy in years])
        for vy, pkg in folds:
            lib = self._inner_library(years, int(vy)) if pkg is None else pkg["years"]
            if len(lib) < self.min_analogs:
                continue
            idx = np.asarray([y_to_i[int(y)] for y in lib], int)
            Xi = X[idx] if pkg is None else pkg["X"]
            xv = X[y_to_i[int(vy)]] if pkg is None else pkg["x"]
            gi = groups if pkg is None else pkg["groups"]
            gw = group_weights if pkg is None else pkg.get("group_weights")
            if self.method_analog == "constructed":
                take, w = self._constructed_solution(
                    Xi, xv, params["metric"], params["alpha"], gi, gw,
                    max(self.constructed_pool_k, params["k"]), params["constructed_lambda"])
            else:
                take, w, _ = self._method_selection(Xi, xv, params, gi, gw)
            if len(take) < self.min_analogs:
                continue
            analog_years = lib[take]
            threshold_years = [int(y) for y in lib if int(self.clim_year_start or y) <= int(y) <= int(self.clim_year_end or y)]
            if len(threshold_years) < 3:
                threshold_years = [int(y) for y in lib]
            th = predictant.where(predictant["T"].dt.year.isin(threshold_years), drop=True)
            if fixed_clim is not None:
                th = fixed_clim
            terc = climatological_terciles(th)[0] if fixed_q is None else fixed_q
            t1 = terc.isel(quantile=0, drop=True); t2 = terc.isel(quantile=1, drop=True)
            obs = self._map_for_year(predictant, int(vy))
            for lam in scores:
                p = self._analog_probability_maps(predictant, analog_years, w, th, lam, thresholds=terc)
                rps = self._rps_map(obs, p, t1, t2)
                scores[lam].append(float(rps.mean(skipna=True).values))
        means = {lam: np.nanmean(v) for lam, v in scores.items() if v}
        return float(min(means, key=means.get)) if means else 0.25

    # ------------------------------------------------------------ selection API
    def select_analogs(self, predictant, itrain, itest=None, *, target_source="hindcast",
                       tune=None, member=None):
        """Select analogue years and return a rich diagnostic dictionary."""
        predictant = predictant.copy()
        predictant["T"] = predictant["T"].astype("datetime64[ns]")
        times = predictant.get_index("T")
        if times.has_duplicates or not times.is_monotonic_increasing:
            raise ValueError("Analogue predictand times must be unique and sorted.")
        years_all = np.asarray(np.unique(predictant["T"].dt.year.values), int)
        if len(years_all) != predictant.sizes["T"]:
            raise ValueError("Scientific analogues require one target season per year.")
        train_years = years_all[np.asarray(itrain, int)]
        if target_source == "forecast":
            target_year = int(self.year_forecast)
        else:
            if itest is None:
                raise ValueError("itest is required for a hindcast target")
            ti = np.asarray(itest, int).ravel()
            if ti.size != 1:
                raise ValueError("Scientific analogue compute_model expects one held-out year")
            target_year = int(years_all[int(ti[0])])

        if target_year in train_years:
            raise ValueError("The analogue target year cannot be part of its training library.")
        if len(train_years) != len(np.unique(train_years)):
            raise ValueError("Analogue training indices must be unique.")

        package = self._build_feature_space(train_years, target_year, target_source=target_source, member=member)
        fixed_selection = getattr(self, "_analog_fixed_selection", None)
        do_tune = (self.nested_tuning if tune is None else bool(tune)) and fixed_selection is None
        native_probabilities = self.probability_method in ("analog", "analog_ensemble", "direct")
        tune_shrinkage = native_probabilities and self.probability_shrinkage == "auto" and fixed_selection is None
        strict_packages = None
        if self.inner_preprocessing == "strict" and (
                do_tune or tune_shrinkage or
                self.method_analog in ("learned_metric", "learned_spatial_metric")):
            strict_packages = self._strict_inner_packages(
                predictant, train_years, member=member,
                learn_weights=(do_tune or tune_shrinkage))
        if do_tune and len(train_years) >= max(6, self.min_analogs + 3):
            if self.inner_preprocessing == "strict":
                params = self._tune_parameters_strict(strict_packages, predictant)
            else:
                params = self._tune_parameters_outer_features(package, predictant)
        else:
            params = {
                "metric": self.analog_metric if self.analog_metric != "auto" else "hybrid",
                "alpha": self.hybrid_alpha,
                "weighting": self.analog_weighting if self.analog_weighting != "auto" else "softmax",
                "tau": self.tau, "k": self.n_analogs,
            }

        if fixed_selection is not None:
            params = dict(fixed_selection["params"])
        group_weights = None
        if self.method_analog in ("learned_metric", "learned_spatial_metric"):
            group_weights = self._learn_group_weights(package, predictant, params["k"],
                                                       strict_packages=strict_packages)
            params["metric"] = "weighted_euclidean"

        if self.method_analog == "constructed" and do_tune:
            params["constructed_lambda"] = self._tune_constructed_lambda(
                package, predictant, params, group_weights, strict_packages=strict_packages
            )
        elif fixed_selection is None:
            params["constructed_lambda"] = self.constructed_lambda

        X, x, years = package["X"], package["x"], package["years"]
        if self.method_analog == "constructed":
            # Constructed analogue ridge strength can be supplied explicitly; a
            # compact inner tuning is deliberately omitted when nested_tuning is
            # false to keep operational use predictable.
            pool, w = self._constructed_solution(
                X, x, params["metric"], params["alpha"], package["groups"],
                group_weights, max(self.constructed_pool_k, params["k"]),
                params["constructed_lambda"],
            )
            analog_years = years[pool]
            distances = self._distances(X, x, params["metric"], params["alpha"],
                                        package["groups"], group_weights)[pool]
        else:
            take, w, distances = self._method_selection(X, x, params, package["groups"], group_weights)
            analog_years = years[take]

        quality_pct = self._quality_percentile(
            X, x, params["metric"], params["alpha"], package["groups"], group_weights
        )
        if fixed_selection is not None:
            base_shrink = float(fixed_selection.get("base_probability_shrinkage", fixed_selection["probability_shrinkage"]))
        elif native_probabilities:
            base_shrink = self._tune_probability_shrinkage(
                package, predictant, params, group_weights, strict_packages=strict_packages)
        else:
            base_shrink = 0.0
        quality_extra = 0.0
        if self.quality_shrinkage:
            # No extra shrinkage for an ordinary (<50th percentile) nearest
            # distance; progressively approach climatology for out-of-library
            # target states.
            quality_extra = max(0.0, (quality_pct - 0.50) / 0.50) * self.max_quality_shrinkage
        prob_shrink = 1.0 - (1.0 - base_shrink) * (1.0 - quality_extra)

        out = {
            "target_year": target_year,
            "target_source": target_source,
            "member": None if member is None else int(member),
            "train_years": np.asarray(train_years, int),
            "analog_years": np.asarray(analog_years, int),
            "distances": np.asarray(distances, float),
            "weights": np.asarray(w, float),
            "quality_percentile": float(quality_pct),
            "probability_shrinkage": float(prob_shrink),
            "base_probability_shrinkage": float(base_shrink),
            "tuning_source": "outer_selected_conditional" if fixed_selection is not None else ("inner_validation" if do_tune else "fixed_parameters"),
            "params": params,
            "group_weights": group_weights,
            "feature_names": package["feature_names"],
            "inner_preprocessing": self.inner_preprocessing,
        }
        self._analog_last_selection = out
        self._analog_cv_cache[int(target_year)] = out
        if self.verbose_analog:
            print(
                f"Scientific analogues for {target_year}: "
                f"{[(int(y), round(float(ww),3)) for y,ww in zip(analog_years,w)]}; "
                f"metric={params['metric']}, k={len(analog_years)}, "
                f"quality_pct={quality_pct:.2f}"
            )
        return out

    def _available_target_members(self, target_source="hindcast", target_year=None):
        if self.target_member_mode != "memberwise":
            return [None]
        _, hdcst, fcst = self._get_raw_sources()
        source = fcst if target_source == "forecast" else hdcst
        for p in self.predictor_vars:
            var = p.get("variable")
            if var not in source:
                continue
            da = self._as_dataarray(source[var], var)
            if "number" in da.dims:
                n = int(da.sizes["number"])
                if self.max_target_members is not None:
                    n = min(n, int(self.max_target_members))
                return list(range(n))
        return [None]

    @staticmethod
    def _mean_probabilities(prob_list):
        if not prob_list:
            return None
        p = xr.concat(prob_list, dim="member_analog").mean("member_analog", skipna=True)
        ps = p.sum("probability", skipna=False)
        p = xr.where(ps > 0, p/ps, np.nan)
        dims = [d for d in ("probability", "T", "Y", "X") if d in p.dims]
        return p.transpose(*dims)

    # ---------------------------------------------------------- WAS interface
    def compute_model(self, predictant, itrain, itest, **kwargs):
        from .was_analog_probability import compute_model
        return compute_model(self,predictant,itrain,itest,**kwargs)


    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det,
                     best_code_da=None, best_shape_da=None, best_loc_da=None,
                     best_scale_da=None, **kwargs):
        from .was_analog_probability import NATIVE, native_compute_prob
        if self.probability_method in NATIVE:
            return native_compute_prob(self,Predictant,clim_year_start,clim_year_end,
                                       hindcast_det,**kwargs)
        return super().compute_prob(Predictant,clim_year_start,clim_year_end,hindcast_det,
            best_code_da,best_shape_da,best_loc_da,best_scale_da,**kwargs)


    def forecast(self, predictant, clim_year_start, clim_year_end, hindcast_det,
                 best_code_da=None, best_shape_da=None, best_loc_da=None,
                 best_scale_da=None, **kwargs):
        from .was_analog_probability import forecast
        return forecast(self,predictant,clim_year_start,clim_year_end,hindcast_det,**kwargs)



class WAS_Analog_KNN(WAS_Analog_Scientific):
    """Natural k-nearest analogue ensemble with continuous distances."""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("method_analog", "knn")
        kwargs.setdefault("analog_weighting", "equal")
        super().__init__(*args, **kwargs)


class WAS_Analog_Weighted(WAS_Analog_Scientific):
    """Distance-weighted natural analogue ensemble."""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("method_analog", "weighted")
        kwargs.setdefault("analog_weighting", "softmax")
        super().__init__(*args, **kwargs)


class WAS_Analog_EOF(WAS_Analog_Scientific):
    """Fold-safe EOF-space analogue ensemble."""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("method_analog", "eof")
        kwargs.setdefault("feature_mode", "eof")
        super().__init__(*args, **kwargs)


class WAS_Analog_Hybrid(WAS_Analog_Scientific):
    """Hybrid climate-index + EOF/circulation analogue ensemble."""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("method_analog", "hybrid")
        kwargs.setdefault("feature_mode", "hybrid")
        super().__init__(*args, **kwargs)


class WAS_Analog_Constructed(WAS_Analog_Scientific):
    """Regularised convex constructed-analogue forecast."""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("method_analog", "constructed")
        kwargs.setdefault("feature_mode", "hybrid")
        super().__init__(*args, **kwargs)


class WAS_Analog_LearnedMetric(WAS_Analog_Scientific):
    """Interpretable learned group-weight distance metric for analogues."""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("method_analog", "learned_metric")
        kwargs.setdefault("feature_mode", "hybrid")
        kwargs.setdefault("analog_metric", "weighted_euclidean")
        super().__init__(*args, **kwargs)


class WAS_Analog_LearnedSpatialMetric(WAS_Analog_Scientific):
    """Learned coarse spatial mask / distance metric for analogue selection."""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("method_analog", "learned_spatial_metric")
        kwargs.setdefault("feature_mode", "spatial_boxes")
        kwargs.setdefault("analog_metric", "weighted_euclidean")
        super().__init__(*args, **kwargs)

class WAS_Analog_Regional(WAS_Analog_Scientific):
    """Region-specific analogue selection with a shared predictor library.

    Parameters
    ----------
    regional_masks : dict[str, xr.DataArray] or xr.DataArray
        Either a mapping of region names to boolean/0-1 masks on (Y, X), or a
        labelled DataArray whose finite unique values define regions.  Each
        region tunes/selects its own analogue set; outputs are mosaicked back on
        the predictand grid.  Masks should normally be mutually exclusive.
    """
    def __init__(self, *args, regional_masks=None, **kwargs):
        kwargs.setdefault("method_analog", "hybrid")
        kwargs.setdefault("feature_mode", "hybrid")
        super().__init__(*args, **kwargs)
        self.regional_masks = regional_masks
        self._regional_cv_cache = {}
        self._regional_last = {}

    def _normalise_region_masks(self, predictant):
        from .was_analog_probability import region_masks
        return region_masks(self,predictant)


    def compute_model(self, predictant, itrain, itest, **kwargs):
        from .was_analog_probability import regional_compute_model
        return regional_compute_model(self,predictant,itrain,itest,**kwargs)


    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det, **kwargs):
        from .was_analog_probability import regional_compute_prob
        return regional_compute_prob(self,Predictant,clim_year_start,clim_year_end,hindcast_det,**kwargs)


    def forecast(self, predictant, clim_year_start, clim_year_end, hindcast_det, **kwargs):
        from .was_analog_probability import regional_forecast
        return regional_forecast(self,predictant,clim_year_start,clim_year_end,hindcast_det,**kwargs)


    def regional_selection_tables(self):
        return {
            name: pd.DataFrame({
                "year": np.asarray(sel["analog_years"], int),
                "distance": np.asarray(sel["distances"], float),
                "weight": np.asarray(sel["weights"], float),
            }).sort_values("weight", ascending=False, ignore_index=True)
            for name, sel in self._regional_last.items()
        }


# Keep constructor inputs for fresh model instances in nested cross-validation.
from .was_crossfit_adapters import capture_constructors as _capture_constructors
_capture_constructors(globals())
