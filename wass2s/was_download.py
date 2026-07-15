"""Data download utilities for ERA5, CHIRPS, TAMSAT, and NMME.

Provides a unified interface for downloading and pre-processing gridded
seasonal hindcast and reanalysis datasets.

Class
-----
WAS_Download
    Central download manager.  Key methods:

    ``WAS_Download_Models``
        NMME and C3S seasonal forecast / hindcast data.
    ``WAS_Download_Reanalysis`` / ``WAS_Download_Reanalysis_``
        ERA5 single-level and pressure-level reanalysis (via CDS).
    ``WAS_Download_ERA5Land`` / ``WAS_Download_ERA5Land_daily``
        ERA5-Land daily and monthly data.
    ``WAS_Download_CHIRPSv3_Seasonal`` / ``WAS_Download_CHIRPSv3_Daily``
        CHIRPS v3 precipitation (seasonal aggregates and daily fields).
    ``WAS_Download_TAMSAT_Seasonal`` / ``WAS_Download_TAMSAT_Daily``
        TAMSAT African rainfall estimates.
    ``WAS_Download_AgroIndicators`` / ``WAS_Download_AgroIndicators_daily``
        NMME-derived agroclimatic indicator downloads.

Standalone function
-------------------
plot_map
    Quick Cartopy map of a geographic extent.
"""
from __future__ import annotations
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
import gc
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
import datetime as dt
import time as _time
import urllib.request
from typing import List, Tuple, Sequence, Optional
# Suppress warnings for urllib3 to avoid SSL certificate verification errors
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.getLogger("cdsapi").setLevel(logging.ERROR)


def _download_atomic(url, path, max_retries=3, retry_delay=5, timeout=180):
    """
    Download `url` to `path` through a temporary `<path>.part` file, renamed
    atomically on success. Consequence: if `path` exists, it is complete.
    Returns True on success, False after exhausting retries.
    """

    tmp = path.parent / (path.name + ".part")
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[DL ] Attempt {attempt}/{max_retries}: {url}")
            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                with open(tmp, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            os.replace(tmp, path)  # atomic on POSIX: path never half-written
            return True
        except Exception as exc:
            print(f"[ERR] Attempt {attempt}/{max_retries} failed: {exc}")
            if tmp.exists():
                os.remove(tmp)
            if attempt < max_retries:
                _time.sleep(retry_delay)
    return False


def _remove_quiet(path):
    import os

    try:
        if path.exists():
            os.remove(path)
    except OSError as exc:
        print(f"[WARN] Could not remove {path}: {exc}")


def _as_tuple(value, default):
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _select_variable(ds, candidates):
    candidates = _as_tuple(candidates, default=())
    for name in candidates:
        if name in ds.data_vars:
            return name
    raise KeyError(f"None of {candidates} found. Available variables: {list(ds.data_vars)}")


def _standardize_latlon_da(da, area=None):
    rename = {}
    if "lon" in da.dims or "lon" in da.coords:
        rename["lon"] = "X"
    if "longitude" in da.dims or "longitude" in da.coords:
        rename["longitude"] = "X"
    if "lat" in da.dims or "lat" in da.coords:
        rename["lat"] = "Y"
    if "latitude" in da.dims or "latitude" in da.coords:
        rename["latitude"] = "Y"
    da = da.rename({old: new for old, new in rename.items() if old in da.dims or old in da.coords})

    if "X" in da.coords and float(da["X"].max()) > 180:
        da = da.assign_coords(X=(((da["X"] + 180) % 360) - 180)).sortby("X")
    if "Y" in da.coords:
        da = da.sortby("Y")
    if "X" in da.coords:
        da = da.sortby("X")

    if area is not None:
        north, west, south, east = area
        da = da.sel(X=slice(west, east), Y=slice(south, north))

    drop = [name for name in ["spatial_ref", "crs"] if name in da.coords]
    if drop:
        da = da.drop_vars(drop, errors="ignore")
    return da

class WAS_Download:
    def __init__(self):
        """Initialize the WAS_Download class."""
        pass


    def ModelsName(
        self,
        centre={
            "BOM_2": "bom",
            "ECMWF_51": "ecmwf",
            "UKMO_604": "ukmo",
            "UKMO_603": "ukmo",
            "UKMO_605": "ukmo",
            "UKMO_610": "ukmo",
            "METEOFRANCE_8": "meteo_france",
            "METEOFRANCE_9": "meteo_france",
            "DWD_21": "dwd", # month of initialization available for forecast are Jan to Mar
            "DWD_22": "dwd", # month of initialization available for forecast are Apr to __ 
            "CMCC_35": "cmcc",
            "CMCC_4": "cmcc",
            "NCEP_2": "ncep",
            "JMA_3": "jma",
            "JMA_4": "jma",
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
            "NCAR_CESM1_1": "ncar_cesm1",
            "NMME_1" : "nmme",
        },
        variables_1={
            "PRCP":  "total_precipitation",
            "TEMP":  "2m_temperature",
            "TDEW": "2m_dewpoint_temperature",
            "TMAX":  "maximum_2m_temperature_in_the_last_24_hours",
            "TMIN":  "minimum_2m_temperature_in_the_last_24_hours",
            "UGRD10":"10m_u_component_of_wind",
            "VGRD10":"10m_v_component_of_wind",
            "SST":   "sea_surface_temperature",
            "SLP": "mean_sea_level_pressure",
            "DSWR": "surface_solar_radiation_downwards",
            "DLWR": "surface_thermal_radiation_downwards",
            "NOLR": "top_net_thermal_radiation",
            "RUNOFF":"mean_surface_runoff_rate"
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

    
    # def ModelsName(
    #     self,
    #     source=None,
    #     variable=None,
    #     product="monthly",
    #     include_pressure=True,
    #     include_wmolc_aliases=False,
    #     return_metadata=False,
    # ):
    #     """
    #     Return valid model-variable combinations supported by the downloader.
    
    #     This method avoids creating invalid combinations. For example, WMOLC models
    #     such as Beijing, CPTEC or Moscow are not paired with RUNOFF because the
    #     current WMOLC direct-download files do not provide runoff.
    
    #     Parameters
    #     ----------
    #     source : str or list of str, optional
    #         Source family to include. Accepted values are:
    #         "c3s", "cds", "nmme", "wmolc", or None for all.
    
    #     variable : str or list of str, optional
    #         Variable short name(s) to keep, for example "PRCP", "RUNOFF",
    #         ["PRCP", "TEMP"], etc. If None, all valid variables are returned.
    
    #     product : {"monthly", "daily"}, default "monthly"
    #         Product context. This matters for a few CDS variables whose native
    #         names differ between monthly and original daily data, especially RUNOFF.
    
    #     include_pressure : bool, default True
    #         If True, include pressure-level variables such as UGRD_850, VGRD_200,
    #         HUSS_925, etc.
    
    #     include_wmolc_aliases : bool, default False
    #         If True, also include keys such as "WMOLC_Beijing.PRCP".
    #         Default is False to avoid duplicate downloads.
    
    #     return_metadata : bool, default False
    #         If False, returns the old style:
    #             {"ECMWF_51.PRCP": ("ecmwf", "total_precipitation")}
    #         If True, returns rich metadata for each key.
    
    #     Returns
    #     -------
    #     dict
    #         Dictionary of valid model-variable combinations.
    #     """

    #     # center_variable = [key for key in downloader.ModelsName().keys() if "RUNOFF" in key]
    #     # center_variable = list(downloader.ModelsName(variable="RUNOFF").keys())
    #     # wmolc_prcp = list(downloader.ModelsName(source="wmolc", variable="PRCP").keys())
    #     # models = downloader.ModelsName(variable="PRCP", return_metadata=True)
        
    #     # for key, meta in models.items():
    #     #     print(key, meta["source"], meta["centre"], meta["native_variable"])

    #     # product="monthly" -> RUNOFF = "mean_surface_runoff_rate"
    #     # product="daily"   -> RUNOFF = "surface_runoff"
        
    #     if product not in {"monthly", "daily"}:
    #         raise ValueError("product must be either 'monthly' or 'daily'.")
    
    #     def _as_set(x):
    #         if x is None:
    #             return None
    #         if isinstance(x, str):
    #             return {x.lower()}
    #         return {str(i).lower() for i in x}
    
    #     def _as_var_set(x):
    #         if x is None:
    #             return None
    #         if isinstance(x, str):
    #             return {x.upper()}
    #         return {str(i).upper() for i in x}
    
    #     source_filter = _as_set(source)
    #     variable_filter = _as_var_set(variable)
    
    #     def _source_allowed(name):
    #         if source_filter is None:
    #             return True
    #         name = name.lower()
    #         if name == "cds":
    #             name = "c3s"
    #         return name in source_filter or ("cds" in source_filter and name == "c3s")
    
    #     def _variable_allowed(v):
    #         return variable_filter is None or v.upper() in variable_filter
    
    #     def _native_name(vmap, product_name):
    #         if isinstance(vmap, dict):
    #             if product_name in vmap:
    #                 return vmap[product_name]
    #             if "monthly" in vmap:
    #                 return vmap["monthly"]
    #             if "daily" in vmap:
    #                 return vmap["daily"]
    #             return next(iter(vmap.values()))
    #         return vmap
    
    #     # ------------------------------------------------------------------
    #     # 1. C3S / CDS seasonal models
    #     # ------------------------------------------------------------------
    #     c3s_models = {
    #         "BOM_2":          {"centre": "bom",          "system": "2"},
    #         "ECMWF_51":       {"centre": "ecmwf",        "system": "51"},
    #         "UKMO_603":       {"centre": "ukmo",         "system": "603"},
    #         "UKMO_604":       {"centre": "ukmo",         "system": "604"},
    #         "UKMO_605":       {"centre": "ukmo",         "system": "605"},
    #         "UKMO_610":       {"centre": "ukmo",         "system": "610"},
    #         "METEOFRANCE_8":  {"centre": "meteo_france", "system": "8"},
    #         "METEOFRANCE_9":  {"centre": "meteo_france", "system": "9"},
    #         "DWD_21":         {"centre": "dwd",          "system": "21"},
    #         "DWD_22":         {"centre": "dwd",          "system": "22"},
    #         "CMCC_4":         {"centre": "cmcc",         "system": "4"},
    #         "CMCC_35":        {"centre": "cmcc",         "system": "35"},
    #         "NCEP_2":         {"centre": "ncep",         "system": "2"},
    #         "JMA_3":          {"centre": "jma",          "system": "3"},
    #         "JMA_4":          {"centre": "jma",          "system": "4"},
    #         "ECCC_4":         {"centre": "eccc",         "system": "4"},
    #         "ECCC_5":         {"centre": "eccc",         "system": "5"},
    #     }
    
    #     c3s_single_variables = {
    #         "PRCP":   "total_precipitation",
    #         "TEMP":   "2m_temperature",
    #         "TDEW":   "2m_dewpoint_temperature",
    #         "TMAX":   "maximum_2m_temperature_in_the_last_24_hours",
    #         "TMIN":   "minimum_2m_temperature_in_the_last_24_hours",
    #         "UGRD10": "10m_u_component_of_wind",
    #         "VGRD10": "10m_v_component_of_wind",
    #         "SST":    "sea_surface_temperature",
    #         "SLP":    "mean_sea_level_pressure",
    #         "DSWR":   "surface_solar_radiation_downwards",
    #         "DLWR":   "surface_thermal_radiation_downwards",
    #         "NOLR":   "top_net_thermal_radiation",
    #         "SRUNOFF": {
    #             "monthly": "mean_surface_runoff_rate",
    #             "daily": "surface_runoff",
    #         },
    #         "RUNOFF": {
    #             "monthly": "mean_runoff_rate",
    #             "daily": "runoff",
    #         },
    #     }
    
    #     c3s_pressure_variables = {
    #         "HUSS_1000": "specific_humidity",
    #         "HUSS_925":  "specific_humidity",
    #         "HUSS_850":  "specific_humidity",
    #         "UGRD_1000": "u_component_of_wind",
    #         "UGRD_925":  "u_component_of_wind",
    #         "UGRD_850":  "u_component_of_wind",
    #         "UGRD_700":  "u_component_of_wind",
    #         "UGRD_600":  "u_component_of_wind",
    #         "UGRD_500":  "u_component_of_wind",
    #         "UGRD_200":  "u_component_of_wind",
    #         "VGRD_1000": "v_component_of_wind",
    #         "VGRD_925":  "v_component_of_wind",
    #         "VGRD_850":  "v_component_of_wind",
    #         "VGRD_700":  "v_component_of_wind",
    #         "VGRD_600":  "v_component_of_wind",
    #         "VGRD_500":  "v_component_of_wind",
    #         "VGRD_200":  "v_component_of_wind",
    #         "HGT_500":   "geopotential",
    #     }
    
    #     # ------------------------------------------------------------------
    #     # 2. NMME models
    #     #    Important: NMME does not provide all C3S variables in your current
    #     #    downloader. Do not create RUNOFF, DSWR, DLWR, etc. here.
    #     # ------------------------------------------------------------------
    #     nmme_models = {
    #         "CFSV2_1":       {"centre": "cfsv2",       "system": "1", "netcdf_model": "CFSv2"},
    #         "CMC1_1":        {"centre": "cmc1",        "system": "1", "netcdf_model": "CanESM5"},
    #         "CMC2_1":        {"centre": "cmc2",        "system": "1", "netcdf_model": "GEM5.2_NEMO"},
    #         "GFDL_1":        {"centre": "gfdl",        "system": "1", "netcdf_model": "GFDL-SPEAR"},
    #         "NASA_1":        {"centre": "nasa",        "system": "1", "netcdf_model": "NASA_GEOS5v2"},
    #         "NCAR_CCSM4_1":  {"centre": "ncar_ccsm4",  "system": "1", "netcdf_model": "NCAR_CCSM4"},
    #         "NCAR_CESM1_1":  {"centre": "ncar_cesm1",  "system": "1", "netcdf_model": "NCAR_CESM1"},
    #         "NMME_1":        {"centre": "nmme",        "system": "1", "netcdf_model": "NMME"},
    #     }
    
    #     nmme_variables = {
    #         "PRCP": {"cpt": "precip", "netcdf": "prate"},
    #         "TEMP": {"cpt": "tmp2m",  "netcdf": "tmp2m"},
    #         "SST":  {"cpt": "sst",    "netcdf": "tmpsfc"},
    #     }
    
    #     # ------------------------------------------------------------------
    #     # 3. WMO Lead Centre direct-download models
    #     #    Only include variables for which your WMOLC downloader has a file
    #     #    naming rule.
    #     # ------------------------------------------------------------------
    #     wmolc_models = {
    #         "Beijing":   {"centre": "Beijing",   "file_prefix": "beijing"},
    #         "CPTEC":     {"centre": "CPTEC",     "file_prefix": "cptec"},
    #         "Offenbach": {"centre": "Offenbach", "file_prefix": "offenbach"},
    #         "Montreal":  {"centre": "Montreal",  "file_prefix": "montreal"},
    #         "Pune":      {"centre": "Pune",      "file_prefix": "pune"},
    #         "Pretoria":  {"centre": "Pretoria",  "file_prefix": "pretoria"},
    #         "Moscow":    {"centre": "Moscow",    "file_prefix": "moscow"},
    #     }
    
    #     wmolc_variables = {
    #         "PRCP": {"file_suffix": "prec", "standard_name": "precipitation"},
    #         "TEMP": {"file_suffix": "t2m",  "standard_name": "2m_temperature"},
    #         "T2M":  {"file_suffix": "t2m",  "standard_name": "2m_temperature"},
    #         "SST":  {"file_suffix": "sst",  "standard_name": "sea_surface_temperature"},
    #         "SLP":  {"file_suffix": "mslp", "standard_name": "mean_sea_level_pressure"},
    #         "MSLP": {"file_suffix": "mslp", "standard_name": "mean_sea_level_pressure"},
    #         "H500": {"file_suffix": "h500", "standard_name": "geopotential_height_500"},
    #         "Z500": {"file_suffix": "h500", "standard_name": "geopotential_height_500"},
    #     }
    
    #     registry = {}
    
    #     def _add(key, metadata):
    #         if return_metadata:
    #             registry[key] = metadata
    #         else:
    #             registry[key] = (
    #                 metadata.get("centre"),
    #                 metadata.get("native_variable"),
    #             )
    
    #     # Add C3S/CDS
    #     if _source_allowed("c3s"):
    #         for model_id, model_meta in c3s_models.items():
    #             for var_code, var_native in c3s_single_variables.items():
    #                 if not _variable_allowed(var_code):
    #                     continue
    
    #                 native = _native_name(var_native, product)
    #                 key = f"{model_id}.{var_code}"
    
    #                 _add(
    #                     key,
    #                     {
    #                         "source": "c3s",
    #                         "model_id": model_id,
    #                         "centre": model_meta["centre"],
    #                         "system": model_meta["system"],
    #                         "variable": var_code,
    #                         "native_variable": native,
    #                         "level_type": "single",
    #                         "product": product,
    #                         "dataset_monthly": "seasonal-monthly-single-levels",
    #                         "dataset_daily": "seasonal-original-single-levels",
    #                     },
    #                 )
    
    #             if include_pressure:
    #                 for var_code, native in c3s_pressure_variables.items():
    #                     if not _variable_allowed(var_code):
    #                         continue
    
    #                     level = var_code.split("_")[-1]
    #                     key = f"{model_id}.{var_code}"
    
    #                     _add(
    #                         key,
    #                         {
    #                             "source": "c3s",
    #                             "model_id": model_id,
    #                             "centre": model_meta["centre"],
    #                             "system": model_meta["system"],
    #                             "variable": var_code,
    #                             "native_variable": native,
    #                             "pressure_level": level,
    #                             "level_type": "pressure",
    #                             "product": product,
    #                             "dataset_monthly": "seasonal-monthly-pressure-levels",
    #                             "dataset_daily": "seasonal-original-pressure-levels",
    #                         },
    #                     )
    
    #     # Add NMME
    #     if _source_allowed("nmme"):
    #         for model_id, model_meta in nmme_models.items():
    #             for var_code, var_meta in nmme_variables.items():
    #                 if not _variable_allowed(var_code):
    #                     continue
    
    #                 key = f"{model_id}.{var_code}"
    
    #                 _add(
    #                     key,
    #                     {
    #                         "source": "nmme",
    #                         "model_id": model_id,
    #                         "centre": model_meta["centre"],
    #                         "system": model_meta["system"],
    #                         "variable": var_code,
    #                         "native_variable": var_meta["cpt"],
    #                         "cpt_variable": var_meta["cpt"],
    #                         "netcdf_variable": var_meta["netcdf"],
    #                         "netcdf_model": model_meta["netcdf_model"],
    #                         "level_type": "single",
    #                         "product": "monthly",
    #                     },
    #                 )
    
    #     # Add WMOLC
    #     if _source_allowed("wmolc"):
    #         for model_id, model_meta in wmolc_models.items():
    #             for var_code, var_meta in wmolc_variables.items():
    #                 if not _variable_allowed(var_code):
    #                     continue
    
    #                 key = f"{model_id}.{var_code}"
    
    #                 _add(
    #                     key,
    #                     {
    #                         "source": "wmolc",
    #                         "model_id": model_id,
    #                         "centre": model_meta["centre"],
    #                         "system": None,
    #                         "variable": var_code,
    #                         "native_variable": var_meta["file_suffix"],
    #                         "file_prefix": model_meta["file_prefix"],
    #                         "file_suffix": var_meta["file_suffix"],
    #                         "standard_name": var_meta["standard_name"],
    #                         "level_type": "single",
    #                         "product": "monthly",
    #                     },
    #                 )
    
    #                 if include_wmolc_aliases:
    #                     alias_key = f"WMOLC_{model_id}.{var_code}"
    #                     _add(
    #                         alias_key,
    #                         {
    #                             "source": "wmolc",
    #                             "model_id": f"WMOLC_{model_id}",
    #                             "centre": model_meta["centre"],
    #                             "system": None,
    #                             "variable": var_code,
    #                             "native_variable": var_meta["file_suffix"],
    #                             "file_prefix": model_meta["file_prefix"],
    #                             "file_suffix": var_meta["file_suffix"],
    #                             "standard_name": var_meta["standard_name"],
    #                             "level_type": "single",
    #                             "product": "monthly",
    #                             "alias_of": key,
    #                         },
    #                     )
    
    #     return dict(sorted(registry.items()))
    

    def ReanalysisName(
        self,
        centre={"ERA5": "reanalysis ERA5", "NOAA": "NOAA ERSST", "ERA5Land": "reanalysis ERA5Land"},
        variables_1={
            "PRCP": "total_precipitation",
            "TEMP": "2m_temperature",
            "UGRD10": "10m_u_component_of_wind",
            "VGRD10": "10m_v_component_of_wind",
            "SST": "sea_surface_temperature",
            "SLP": "mean_sea_level_pressure",
            "DSWR": "surface_solar_radiation_downwards",
            "DLWR": "surface_thermal_radiation_downwards",
            "NOLR": "top_thermal_radiation",
            "RUNOFF": "surface_runoff"
            
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
            "AGRO.TMIN": ("2m_temperature", "24_hour_minimum"),
            "AGRO.DSWR": ("solar_radiation_flux", None),
            "AGRO.ETP": ("reference_evapotranspiration", None),
            "AGRO.WFF": ("10m_wind_speed", "24_hour_mean"),
            "AGRO.HUMAX": ("2m_relative_humidity_derived", "24_hour_maximum"),
            "AGRO.HUMIN": ("2m_relative_humidity_derived", "24_hour_minimum"),
        },
    ):
        # 1 W m-2 = 0.0864 MJ m-2 day-1
        """
        Generate a dictionary for agrometeorological observation variables.

        Parameters:
            variables (dict): Mapping of agro variable short names to full names.

        Returns:
            dict: A dictionary mapping agro variables to their corresponding full names.
        """
        return variables
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
                        t_str = t_str.rstrip(',')
                        year_str, pot_months = t_str.split('-', 1)
                        start_year = int(year_str)
                        if '/' in pot_months:
                            start_str, end_str = pot_months.split("/")
                            start_month = int(start_str)
                            if '-' in end_str:
                                end_year_str, end_month_str = end_str.split('-')
                                end_year = int(end_year_str)
                                end_month = int(end_month_str)
                            else:
                                end_year = start_year
                                end_month = int(end_str)
                            # Generate list of (year, month) pairs
                            months_list = []
                            current_year = start_year
                            current_month = start_month
                            while True:
                                months_list.append((current_year, current_month))
                                if current_year == end_year and current_month == end_month:
                                    break
                                current_month += 1
                                if current_month > 12:
                                    current_month = 1
                                    current_year += 1
                            if len(months_list) != 3:
                                raise ValueError("Expected 3-month season")
                            # Use middle month and its year for time
                            time_year, month = months_list[1]
                            days_in_mon = sum(self.days_in_month(y, m) for y, m in months_list)
                        else:
                            month = int(pot_months)
                            time_year = start_year
                            days_in_mon = self.days_in_month(start_year, month)
                            months_list = [(start_year, month)]

                        days_in_month_values.append(days_in_mon)
                        times.append(datetime.datetime(time_year, month, 1))

                        #### Retrieve init start
                        start_str = lines[i].split('cpt:S=')[1].split()[0]
                        start_str = start_str.rstrip(',')
                        yearstart, monthstart, daystart = start_str.split('-')
                        times_start.append(datetime.datetime(int(yearstart), int(monthstart), 1))
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
        nmme_source="cpt",  # Options: "netcdf" or "cpt"
        year_init_nmmeNC=2025
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
            "BOM_2": "bom",
            "ECMWF_51": "ecmwf",
            "UKMO_604": "ukmo", # month of initialization available for forecast are Apr to __
            "UKMO_603": "ukmo", # month of initialization available for forecast are Jan to Mar
            "UKMO_605": "ukmo",
            "UKMO_610": "ukmo",
            "METEOFRANCE_8": "meteo_france",
            "METEOFRANCE_9": "meteo_france", 
            "DWD_21": "dwd", # month of initialization available for forecast are Jan to Mar
            "DWD_22": "dwd", # month of initialization available for forecast are Apr to __ 
            "CMCC_35": "cmcc",
            "CMCC_4": "cmcc",
            "NCEP_2": "ncep",
            "JMA_3": "jma",
            "JMA_4": "jma",
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
            "NCAR_CESM1_1": "ncar_cesm1",
            "NMME_1" : "nmme"
        }
        
        # Mapping for NMME NetCDF filenames/directories on FTP
        nmme_netcdf_model_map = {
            "CFSV2_1": "CFSv2", "CMC1_1": "CanESM5", "CMC2_1": "GEM5.2_NEMO", 
            "GFDL_1": "GFDL-SPEAR", 
            "NASA_1": "NASA_GEOS5v2", "NCAR_CCSM4_1": "NCAR_CCSM4",
            "NCAR_CESM1_1": "NCAR_CESM1",
            "NMME_1" : "NMME"
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
            "NOLR": "top_thermal_radiation",
            "RUNOFF":"mean_surface_runoff_rate"
        }

        variables_2 = {
            "HUSS_1000": "specific_humidity",
            "HUSS_925": "specific_humidity",
            "HUSS_850": "specific_humidity",
            "UGRD_1000": "u_component_of_wind",
            "UGRD_925": "u_component_of_wind",
            "UGRD_850": "u_component_of_wind",
            "UGRD_700": "u_component_of_wind",
            "UGRD_200": "u_component_of_wind",
            "VGRD_1000": "v_component_of_wind",
            "VGRD_925": "v_component_of_wind",
            "VGRD_850": "v_component_of_wind",
            "VGRD_700": "v_component_of_wind",
            "VGRD_200": "v_component_of_wind",
        }

        system = {
            "BOM_2": "2",
            "ECMWF_51": "51",
            "UKMO_604": "604",
            "UKMO_603": "603",
            "UKMO_605": "605",
            "UKMO_610": "610",
            "METEOFRANCE_8": "8",
            "METEOFRANCE_9": "9",
            "DWD_21": "21",
            "DWD_22": "22",
            # "DWD_2": "2",
            "CMCC_35": "35",
            "CMCC_4": "4",
            "NCEP_2": "2",
            "JMA_3": "3",
            "JMA_4": "4",
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
            "NCAR_CESM1_1": "1",
            "NMME_1" : "1"

        }
        
        nmme = ["cfsv2", "cmc1", "cmc2", "gfdl",  "nasa", "ncar_ccsm4", "ncar_cesm1", "nmme"]
        
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

            if cent in nmme:

                if nmme_source == "netcdf":
                    nmme_nc_vars = {"PRCP": "prate", "TEMP": "tmp2m", "SST": "tmpsfc"}
                    if k not in nmme_nc_vars:
                        print(f"Skipping {k}: Not currently mapped for NetCDF FTP.")
                        continue
    
                    ftp_var = nmme_nc_vars[k]
                    ftp_model = nmme_netcdf_model_map[f"{cent.upper()}_{syst}"]
                    file_path = f"{dir_to_save}/{file_prefix}_{cent.replace('_', '')}{syst}_{k}_{abb_mont_ini}Ic_{season}_{lead_time[0]}.nc"
                    
                    if not force_download and os.path.exists(file_path):
                        print(f"{file_path} already exists. Skipping download.")
                        store_file_path[f"{cent}_{syst}"] = file_path                   
                    else:
                        try:
                            list_ds_years = []
                            # ic_folder = f"{abb_mont_ini.lower()}2025ic" 
    
                            for yr in years:
                                # Files always use 'fcst' on the CPC FTP server regardless of year
                                tag = "fcst" 
                                date_str = f"{yr}{int(month_of_initialization):02d}"
                                fname = f"{ftp_model}.{ftp_var}.{date_str}.ENSMEAN.{tag}.nc"
                                ic_folder = f"{abb_mont_ini.lower()}{year_init_nmmeNC}ic"
                                url = f"https://ftp.cpc.ncep.noaa.gov/International/nmme/netcdf/{ic_folder}/{ftp_model}/{fname}"
                                
                                temp_file = dir_to_save / f"tmp_{fname}"
                                
                                # print(f"Downloading from NMME FTP: {url}")
                                urllib.request.urlretrieve(url, temp_file)
                                
                                # Use decode_times=False to avoid "months since" decoding errors
                                ds_y = xr.open_dataset(temp_file, decode_times=False)
                                
                                # Standardize coords
                                if "lon" in ds_y.coords: ds_y = ds_y.rename({"lon":"X", "lat":"Y"})
                                ds_y = ds_y.assign_coords(X=((ds_y.X + 180) % 360 - 180)).sortby("X").sortby("Y")
                                ds_y = ds_y.sel(X=slice(area[1], area[3]), Y=slice(area[2], area[0]))
                                
                                
                                # Select Lead Time (Target) based on index
                                lead_idx = [int(l) for l in lead_time]
                                ds_y = ds_y.isel(target=lead_idx)
    
                                # Unit Conversions
                                if k == "PRCP":
                                    # mm/s to total mm for the month
                                    start_dt = pd.to_datetime(f"{yr}-{int(month_of_initialization):02d}-01")
                                    days = [(start_dt + pd.DateOffset(months=int(l))).days_in_month for l in lead_time]
                                    # Create alignment DataArray using the undecoded numeric target coordinates
                                    days_da = xr.DataArray(days, coords={"target": ds_y.target}, dims="target")
                                    ds_y = ds_y * 86400 * days_da
                                    ds_y = ds_y.sum(dim="target")
                                    ds_y = ds_y.drop_vars("initial_time")#.to_array().drop_vars("variable")
                                elif k in ["TEMP", "SST"]:
                                    ds_y = ds_y - 273.15 # Kelvin to Celsius
                                    ds_y = ds_y.mean(dim="target")
                                    ds_y = ds_y.drop_vars("initial_time")
                                else:
                                    ds_y = ds_y.mean(dim="target")
                                    ds_y = ds_y.drop_vars("initial_time")
    
                                # Assign the year coordinate T
                                ds_y = ds_y.expand_dims("T").assign_coords(T=[f"{yr}-{int(month_of_initialization):02d}-01"])#.assign_coords(T=[str(yr)])
                                # Rename 'fcst' data variable to k
                                ds_y = ds_y.rename({tag: k})
                                list_ds_years.append(ds_y)
                                
                                ds_y.close()
                                if os.path.exists(temp_file): os.remove(temp_file)
    
                            if list_ds_years:
                                ds_final = xr.concat(list_ds_years, dim="T").sortby("T").transpose('T', 'Y', 'X')
                                ds_final.to_netcdf(file_path)
                                store_file_path[f"{cent}_{syst}"] = file_path
                                print(f"Saved: {file_path}")
    
                        except Exception as e:
                            print(f"NMME NetCDF Error for {cent}: {e}")
                else:
                ## to tab
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
                        try:
                            # Choose base URL depending on forecast/hindcast and temporal resolution.
                            if len(lead_time) == 3:
                                # Build lead time string using min and max lead time values.
                                lead_str = f"{season_months[0]}-{season_months[-1]}"
                                crosses_year = season_months[0] > season_months[-1]
                                
                                if year_forecast:
                                    base_url = "https://ftp.cpc.ncep.noaa.gov/International/nmme/seasonal_nmme_forecast_in_cpt_format/"
                                    if crosses_year:
                                        year_range = f"{year_forecast}-{year_forecast + 1}"
                                    elif int(month_of_initialization) > season_months[0]:
                                        year_range = f"{year_forecast + 1}-{year_forecast + 1}"
                                    else:
                                        year_range = f"{year_forecast}-{year_forecast}"
                                        
                                    # year_range = f"{year_forecast}-{year_forecast + 1}" if crosses_year else f"{year_forecast}-{year_forecast}"
                                    file_name = f"{cent}_{k}_{tag}_{init_str}_{lead_str}_{year_range}.txt"
                                    full_url = base_url + file_name
                                    file_txt_path = dir_to_save / file_name
                                    try:
                                        if os.path.exists(file_txt_path):
                                            da, number_day, times_start = self.parse_cpt_data_optimized(file_txt_path)
                                        else:
                                            self.download_nmme_txt_with_progress(full_url, file_txt_path)
                                            da, number_day, times_start = self.parse_cpt_data_optimized(file_txt_path)
                                    except:
                                        print(f"failed to download {file_name}")
                                else:
                                    base_url = "https://ftp.cpc.ncep.noaa.gov/International/nmme/seasonal_nmme_hindcast_in_cpt_format/"
    
                                    # CPC archive: seasons that cross year (NDJ/DJF) start in 1991; others in 1992
                                    hind_start = 1991 if crosses_year else 1992
                                    hind_end   = 2021
                                    year_range1 = f"{hind_start}-{hind_end}"
                                    year_range2 = f"{1991}-{2020}"
            
                                    file_name_1 = f"{cent}_{k}_{tag}_{init_str}_{lead_str}_{year_range1}.txt"
                                    file_name_2 = f"{cent}_{k}_{tag}_{init_str}_{lead_str}_{year_range2}.txt"
                                    full_url1 = base_url + file_name_1
                                    full_url2 = base_url + file_name_2
                                    # print(full_url2)
                                    file_txt_path_1 = dir_to_save / file_name_1
                                    file_txt_path_2 = dir_to_save / file_name_2
                                    try:
                                        if os.path.exists(file_txt_path_1):
                                            da, number_day, times_start = self.parse_cpt_data_optimized(file_txt_path_1)
                                        else:
                                            self.download_nmme_txt_with_progress(full_url1, file_txt_path_1)
                                            da, number_day, times_start = self.parse_cpt_data_optimized(file_txt_path_1)
                                    except:
                                        print(f"failed to download {file_name_1}")
    
                                    try:
                                        
                                        if os.path.exists(file_txt_path_2):
                                            da, number_day, times_start = self.parse_cpt_data_optimized(file_txt_path_2)
                                        else:
                                            self.download_nmme_txt_with_progress(full_url2, file_txt_path_2)
                                            da, number_day, times_start = self.parse_cpt_data_optimized(file_txt_path_2)
                                    except:
                                        print(f"failed to download {file_name_2}")

        
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
                                    if int(month_of_initialization) > season_months[0]:
                                        year_range = f"{year_forecast + 1}"
                                    else:
                                        year_range = f"{year_forecast}"
                                else:
                                    base_url = "https://ftp.cpc.ncep.noaa.gov/International/nmme/monthly_nmme_hindcast_in_cpt_format/"
                                    year_range = f"{1992}"
                                all_da = []
                                for i in season_months:
                                    file_name = f"{cent}_{k}_{tag}_{init_str}_{i}_{year_range}.txt"
                                    full_url = base_url + file_name
                                    print(full_url)
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
                        except:
                            pass
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
                                "leadtime_month": [i.lstrip('0') for i in lead_time],
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
                                "leadtime_month": [i.lstrip('0') for i in lead_time],
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
                            ds = ds.where(lambda x: x >= 0, other=0)
                            if "indexing_time" in ds.coords: 
                                ds = ds.rename({"latitude":"lat","longitude":"lon","indexing_time":"time"})
                            else:
                                ds = ds.rename({"latitude":"lat","longitude":"lon","forecast_reference_time":"time"})

                        if k =="RUNOFF":
                            ds = getattr(ds,ensemble_mean)(dim="number") if ensemble_mean != None else ds
                            ds = ds.mean(dim="forecastMonth").isel(latitude=slice(None, None, -1))
                            ds = ds.where(lambda x: x >= 0, other=0)
                            if "indexing_time" in ds.coords: 
                                ds = ds.rename({"latitude":"lat","longitude":"lon","indexing_time":"time"})
                            else:
                                ds = ds.rename({"latitude":"lat","longitude":"lon","forecast_reference_time":"time"})
                            lat = ds.lat
                            lon = ds.lon
                            dlon = np.deg2rad(0.1)
                            dlat = np.deg2rad(0.1)
                            r = 6371000 # Earth radius in meters
                            # Area for each grid cell
                            area_ = (r ** 2) * dlon * np.cos(np.deg2rad(lat)) * dlat
                            # Perform the conversion to m^3/s
                            ds = (ds * area_) 
                        

                        if k == "SLP":
                            ds = ds/100
                            ds = getattr(ds,ensemble_mean)(dim="number") if ensemble_mean != None else ds 
                            ds = ds.mean(dim="forecastMonth").isel(latitude=slice(None, None, -1))
                            if "indexing_time" in ds.coords: 
                                ds = ds.rename({"latitude":"lat","longitude":"lon","indexing_time":"time"})
                            else:
                                ds = ds.rename({"latitude":"lat","longitude":"lon","forecast_reference_time":"time"})

                        if k in ["UGRD10","VGRD10"]:
                            ds = getattr(ds,ensemble_mean)(dim="number") if ensemble_mean != None else ds
                            ds = ds.mean(dim="forecastMonth").isel(latitude=slice(None, None, -1))
                            if "indexing_time" in ds.coords: 
                                ds = ds.rename({"latitude":"lat","longitude":"lon","indexing_time":"time"})
                            else:
                                ds = ds.rename({"latitude":"lat","longitude":"lon","forecast_reference_time":"time"})

                        if k in ["DSWR","DLWR", "NOLR"]:
                            ds = getattr(ds,ensemble_mean)(dim="number") if ensemble_mean != None else ds
                            ds = ds.sum(dim="forecastMonth").isel(latitude=slice(None, None, -1))
                            if "indexing_time" in ds.coords: 
                                ds = ds.rename({"latitude":"lat","longitude":"lon","indexing_time":"time"})
                            else:
                                ds = ds.rename({"latitude":"lat","longitude":"lon","forecast_reference_time":"time"})

                        if k not in ["TMIN","TEMP","TMAX","SST","UGRD10","VGRD10", "PRCP","SLP","DSWR","DLWR","NOLR"]:
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
            max_retries=3,
            retry_delay=5,
        ):
        """
        Download daily agro-meteorological indicators from the Copernicus Data Store (CDS)
        for specified variables and years, with retries for failed downloads.

        Parameters
        ----------
        dir_to_save : str or pathlib.Path
            Directory path where the downloaded NetCDF files will be saved.
            The directory will be created if it does not exist.
        variables : list of str
            List of variable shorthand names to download. Valid options are:
            ["AGRO.PRCP", "AGRO.TMAX", "AGRO.TEMP", "AGRO.TMIN", "AGRO.DSWR",
            "AGRO.ETP", "AGRO.WFF", "AGRO.HUMAX", "AGRO.HUMIN"].
            Each variable corresponds to a CDS variable and optional statistic
            (e.g., "AGRO.PRCP" maps to "precipitation_flux").
        year_start : int
            Start year for the data to download (inclusive).
        year_end : int
            End year for the data to download (inclusive).
        area : list of float
            Bounding box for spatial subsetting in the format [North, West, South, East].
            Example: [50, -10, 40, 10] for a region in Europe.
        force_download : bool, optional
            If True, forces download even if the output file exists. Default is False.
        max_retries : int, optional
            Maximum number of retry attempts for failed downloads. Default is 3.
        retry_delay : int, optional
            Seconds to wait between retry attempts. Default is 5.

        Returns
        -------
        None
            The function saves NetCDF files to `dir_to_save` but does not return a value.
            Output files are named as `Daily_<variable>_<year_start>_<year_end>.nc`.

        Notes
        -----
        - The function downloads data from the CDS dataset "sis-agrometeorological-indicators".
        - Data is downloaded year-by-year as ZIP files containing NetCDF files, which are
          extracted, concatenated, and saved as a single NetCDF file per variable.
        - Temperature variables ("AGRO.TMIN", "AGRO.TEMP", "AGRO.TMAX") are converted
          from Kelvin to Celsius.
        - Solar radiation ("AGRO.DSWR") is converted from J/m^2/day to W/m^2.
        - Coordinates are renamed to "X" (longitude), "Y" (latitude), and "T" (time),
          with latitude flipped to ascending order.
        - The function requires a valid CDS API key configured in `~/.cdsapirc`.
        - Downloads are skipped for a variable if any year's data fails to download
          after `max_retries` attempts to ensure data completeness.
        """
        dir_to_save = Path(dir_to_save)
        dir_to_save.mkdir(parents=True, exist_ok=True)
        days = [f"{day:02}" for day in range(1, 32)]
        months = [f"{month:02}" for month in range(1, 13)]
        version = "2_0"

        variable_mapping = {
            "AGRO.PRCP": ("precipitation_flux", None),
            "AGRO.TMAX": ("2m_temperature", "24_hour_maximum"),
            "AGRO.TEMP": ("2m_temperature", "24_hour_mean"),
            "AGRO.TMIN": ("2m_temperature", "24_hour_minimum"),
            "AGRO.DSWR": ("solar_radiation_flux", None),
            "AGRO.ETP": ("reference_evapotranspiration", None),
            "AGRO.WFF": ("10m_wind_speed", "24_hour_mean"),
            "AGRO.HUMAX": ("2m_relative_humidity_derived", "24_hour_maximum"),
            "AGRO.HUMIN": ("2m_relative_humidity_derived", "24_hour_minimum")
        }

        for var in variables:
            if var not in variable_mapping:
                print(f"Unknown variable: {var}. Skipping.")
                continue

            cds_variable, statistic = variable_mapping[var]
            output_path = dir_to_save / f"Daily_{var.split('.')[1]}_{year_start}_{year_end}.nc"

            if not force_download and output_path.exists():
                print(f"{output_path} already exists. Skipping download.")
                continue

            combined_datasets = []
            all_years_downloaded = True

            for year in range(year_start, year_end + 1):
                zip_file_path = dir_to_save / f"Daily_{var.split('.')[1]}_{year}.zip"
                success = False
                retries = 0

                while retries < max_retries and not success:
                    try:
                        client = cdsapi.Client()
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

                        print(f"Attempt {retries + 1}/{max_retries}: Downloading {cds_variable} ({statistic}) data for {year}...")
                        client.retrieve(dataset, request).download(str(zip_file_path))
                        print(f"Downloaded: {zip_file_path}")
                        success = True

                    except Exception as e:
                        retries += 1
                        print(f"Attempt {retries}/{max_retries} failed for {cds_variable} ({statistic}) data for {year}: {e}")
                        if retries < max_retries:
                            print(f"Retrying after {retry_delay} seconds...")
                            _time.sleep(retry_delay)
                        if zip_file_path.exists():
                            os.remove(zip_file_path)
                            print(f"Deleted incomplete ZIP file: {zip_file_path}")

                if not success:
                    print(f"Failed to download {cds_variable} ({statistic}) data for {year} after {max_retries} attempts.")
                    all_years_downloaded = False
                    continue

                try:
                    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                        for netcdf_file_name in zip_ref.namelist():
                            with zip_ref.open(netcdf_file_name) as file:
                                ds = xr.open_dataset(io.BytesIO(file.read()))
                                combined_datasets.append(ds)
                except Exception as e:
                    print(f"Failed to extract/process {zip_file_path}: {e}")
                    all_years_downloaded = False
                    if zip_file_path.exists():
                        os.remove(zip_file_path)
                        print(f"Deleted ZIP file due to processing error: {zip_file_path}")
                    continue

                if zip_file_path.exists():
                    os.remove(zip_file_path)
                    print(f"Deleted ZIP file: {zip_file_path}")

            if combined_datasets and all_years_downloaded:
                try:
                    combined_ds = xr.concat(combined_datasets, dim="time").drop_vars('crs')

                    if var in ["AGRO.TMIN", "AGRO.TEMP", "AGRO.TMAX"]:
                        combined_ds = combined_ds - 273.15
                    if var == "AGRO.DSWR":
                        combined_ds = combined_ds / 86400

                    combined_ds = combined_ds.rename({"lon": "X", "lat": "Y", "time": "T"})
                    combined_ds = combined_ds.isel(Y=slice(None, None, -1))

                    combined_ds.to_netcdf(output_path)
                    combined_ds.close()
                    print(f"Combined dataset for {var} saved to {output_path}")
                except Exception as e:
                    print(f"Failed to process or save combined dataset for {var}: {e}")
            else:
                print(f"Skipping save for {var} due to incomplete year downloads.")

    def WAS_Download_Models_Daily(
        self,
        dir_to_save,
        center_variable,
        month_of_initialization,
        day_of_initialization,
        leadtime_hour,
        year_start_hindcast,
        year_end_hindcast,
        area,
        year_forecast=None,
        ensemble_mean=None,
        force_download=False,
        data_format="netcdf",
        output_layout="valid_time",
        runoff_units="m3/s",
    ):
        """
        Download daily seasonal hindcast/forecast data from CDS seasonal-original
        datasets and save a clean WAS-style NetCDF.
    
        Output conventions:
            output_layout="valid_time":
                T, Y, X when ensemble_mean is "mean" or "median"
                T, M, Y, X when members are kept
    
            output_layout="init_leadtime":
                member, T, leadtime, level, Y, X
    
        Notes:
            - In "valid_time" layout, T is the valid forecast date.
            - In "init_leadtime" layout, T is the initialization date/year.
            - Accumulated variables are deaccumulated along leadtime.
            - RUNOFF can be saved as "mm" or "m3/s".
        """
        from calendar import month_abbr
        from pathlib import Path
        import gc
        import os
        import time as _time
    
        import cdsapi
        import numpy as np
        import xarray as xr
    
        if isinstance(center_variable, str):
            raise TypeError(
                "center_variable must be a list, for example "
                "['ECMWF_51.RUNOFF'] or ['ECMWF_51.RUNOFF', 'UKMO_604.RUNOFF']."
            )
        center_variable = list(center_variable)
    
        if isinstance(leadtime_hour, (str, int, np.integer)):
            leadtime_hour = [str(leadtime_hour)]
        else:
            leadtime_hour = [str(lt) for lt in leadtime_hour]
    
        if ensemble_mean not in [None, "mean", "median"]:
            raise ValueError("ensemble_mean must be None, 'mean', or 'median'.")
    
        if output_layout not in ["valid_time", "init_leadtime"]:
            raise ValueError("output_layout must be 'valid_time' or 'init_leadtime'.")
    
        if runoff_units not in ["mm", "m3/s"]:
            raise ValueError("runoff_units must be 'mm' or 'm3/s'.")
    
        if year_forecast is None:
            years = [str(y) for y in range(year_start_hindcast, year_end_hindcast + 1)]
            file_prefix = "hindcast"
        else:
            years = [str(year_forecast)]
            file_prefix = "forecast"
    
        centre = {
            "BOM_2": "bom",
            "ECMWF_51": "ecmwf",
            "UKMO_603": "ukmo",
            "UKMO_604": "ukmo",
            "UKMO_605": "ukmo",
            "UKMO_610": "ukmo",
            "METEOFRANCE_8": "meteo_france",
            "METEOFRANCE_9": "meteo_france",
            "DWD_21": "dwd",
            "DWD_22": "dwd",
            "CMCC_35": "cmcc",
            "CMCC_4": "cmcc",
            "NCEP_2": "ncep",
            "JMA_3": "jma",
            "JMA_4": "jma",
            "ECCC_4": "eccc",
            "ECCC_5": "eccc",
        }
    
        system = {
            "BOM_2": "2",
            "ECMWF_51": "51",
            "UKMO_603": "603",
            "UKMO_604": "604",
            "UKMO_605": "605",
            "UKMO_610": "610",
            "METEOFRANCE_8": "8",
            "METEOFRANCE_9": "9",
            "DWD_21": "21",
            "DWD_22": "22",
            "CMCC_35": "35",
            "CMCC_4": "4",
            "NCEP_2": "2",
            "JMA_3": "3",
            "JMA_4": "4",
            "ECCC_4": "4",
            "ECCC_5": "5",
        }
    
        variables_1 = {
            "PRCP": "total_precipitation",
            "TEMP": "2m_temperature",
            "TDEW": "2m_dewpoint_temperature",
            "TMAX": "maximum_2m_temperature_in_the_last_24_hours",
            "TMIN": "minimum_2m_temperature_in_the_last_24_hours",
            "UGRD10": "10m_u_component_of_wind",
            "VGRD10": "10m_v_component_of_wind",
            "SST": "sea_surface_temperature",
            "SLP": "mean_sea_level_pressure",
            "DSWR": "surface_solar_radiation_downwards",
            "DLWR": "surface_thermal_radiation_downwards",
            "NOLR": "top_net_thermal_radiation",
            "SRUNOFF": "surface_runoff",
            "RUNOFF": "runoff",
        }
    
        variables_2 = {
            "HUSS_1000": "specific_humidity",
            "HUSS_925": "specific_humidity",
            "HUSS_850": "specific_humidity",
            "UGRD_1000": "u_component_of_wind",
            "UGRD_925": "u_component_of_wind",
            "UGRD_850": "u_component_of_wind",
            "UGRD_700": "u_component_of_wind",
            "UGRD_200": "u_component_of_wind",
            "VGRD_1000": "v_component_of_wind",
            "VGRD_925": "v_component_of_wind",
            "VGRD_850": "v_component_of_wind",
            "VGRD_700": "v_component_of_wind",
            "VGRD_200": "v_component_of_wind",
        }
    
        init_day_dict_jma = {
            "01": 16, "02": 10, "03": 12, "04": 11, "05": 16, "06": 15,
            "07": 15, "08": 14, "09": 13, "10": 13, "11": 12, "12": 12,
        }
        init_day_dict_ncep = {
            "01": 1, "02": 5, "03": 2, "04": 1, "05": 1, "06": 5,
            "07": 5, "08": 4, "09": 3, "10": 3, "11": 2, "12": 2,
        }
    
        def _open_download(path, fmt):
            if fmt == "grib":
                return xr.open_dataset(
                    path,
                    engine="cfgrib",
                    backend_kwargs={"indexpath": ""},
                )
            return xr.open_dataset(path)
    
        def _normalize_cds_dataset(ds):
            rename_map = {}
    
            if "number" in ds.dims or "number" in ds.coords:
                rename_map["number"] = "member"
            if "longitude" in ds.dims or "longitude" in ds.coords:
                rename_map["longitude"] = "X"
            if "latitude" in ds.dims or "latitude" in ds.coords:
                rename_map["latitude"] = "Y"
            if "forecast_period" in ds.dims or "forecast_period" in ds.coords:
                rename_map["forecast_period"] = "leadtime"
            if "step" in ds.dims or "step" in ds.coords:
                rename_map["step"] = "leadtime"
            if "forecast_reference_time" in ds.dims:
                rename_map["forecast_reference_time"] = "T"
            elif "indexing_time" in ds.dims:
                rename_map["indexing_time"] = "T"
            elif "time" in ds.dims and ("forecast_period" in ds.dims or "step" in ds.dims):
                rename_map["time"] = "T"
            if "isobaricInhPa" in ds.dims or "isobaricInhPa" in ds.coords:
                rename_map["isobaricInhPa"] = "level"
            if "pressure_level" in ds.dims or "pressure_level" in ds.coords:
                rename_map["pressure_level"] = "level"
    
            ds = ds.rename({k: v for k, v in rename_map.items() if k in ds})
    
            if "T" not in ds.dims:
                if "forecast_reference_time" in ds.coords:
                    ds = ds.expand_dims(T=np.atleast_1d(ds["forecast_reference_time"].values))
                    ds = ds.drop_vars("forecast_reference_time", errors="ignore")
                elif "indexing_time" in ds.coords:
                    ds = ds.expand_dims(T=np.atleast_1d(ds["indexing_time"].values))
                    ds = ds.drop_vars("indexing_time", errors="ignore")
    
            if "valid_time" not in ds.coords and "T" in ds.coords and "leadtime" in ds.coords:
                ds = ds.assign_coords(valid_time=ds["T"] + ds["leadtime"])
    
            drop_vars = [
                "surface",
                "heightAboveGround",
                "meanSea",
                "entireAtmosphere",
            ]
            ds = ds.drop_vars([name for name in drop_vars if name in ds.variables], errors="ignore")
    
            if "Y" in ds.coords:
                ds = ds.sortby("Y")
    
            preferred = ["member", "T", "leadtime", "level", "Y", "X"]
            for name in list(ds.data_vars):
                dims = [dim for dim in preferred if dim in ds[name].dims]
                dims += [dim for dim in ds[name].dims if dim not in dims]
                ds[name] = ds[name].transpose(*dims)
    
            return ds
    
        def _deaccumulate(ds, dim="leadtime"):
            if dim not in ds.dims:
                return ds
            first = ds.isel({dim: slice(0, 1)})
            diff = ds.diff(dim)
            out = xr.concat([first, diff], dim=dim)
            out = out.assign_coords({dim: ds[dim]})
            return out.where(out >= 0, 0)
    
        def _grid_cell_area_m2(ds):
            if "Y" not in ds.coords or "X" not in ds.coords:
                raise ValueError("RUNOFF conversion to m3/s needs X and Y coordinates.")
            if ds.sizes.get("Y", 0) < 2 or ds.sizes.get("X", 0) < 2:
                raise ValueError("RUNOFF conversion to m3/s needs at least two X and Y points.")
    
            radius = 6371000.0
            dlat = np.deg2rad(float(abs(ds["Y"].diff("Y").median())))
            dlon = np.deg2rad(float(abs(ds["X"].diff("X").median())))
            area_y = (radius ** 2) * dlat * dlon * np.cos(np.deg2rad(ds["Y"]))
            area_y.attrs["units"] = "m2"
            return area_y
    
        def _leadtime_interval_seconds(ds):
            if "leadtime" not in ds.coords:
                raise ValueError("RUNOFF conversion to m3/s needs leadtime coordinates.")
    
            lead = ds["leadtime"]
            if np.issubdtype(lead.dtype, np.timedelta64):
                lead_seconds = (lead / np.timedelta64(1, "s")).astype(float)
            else:
                # CDS leadtime_hour requests are in hours when decoded as numeric values.
                lead_seconds = lead.astype(float) * 3600.0
    
            first = lead_seconds.isel(leadtime=slice(0, 1))
            diff = lead_seconds.diff("leadtime")
            seconds = xr.concat([first, diff], dim="leadtime")
            seconds = seconds.assign_coords(leadtime=lead)
            return seconds.where(seconds > 0)
    
        def _apply_units(ds, var_code):
            if var_code in ["TMIN", "TEMP", "TMAX", "SST", "TDEW"]:
                ds = ds - 273.15
                for name in ds.data_vars:
                    ds[name].attrs["units"] = "degC"
    
            elif var_code == "SLP":
                ds = ds / 100.0
                for name in ds.data_vars:
                    ds[name].attrs["units"] = "hPa"
    
            elif var_code == "PRCP":
                ds = _deaccumulate(ds) * 1000.0
                for name in ds.data_vars:
                    ds[name].attrs["units"] = "mm"
    
            elif var_code in ["SRUNOFF", "RUNOFF"]:
                ds = _deaccumulate(ds)
                if runoff_units == "mm":
                    ds = ds * 1000.0
                    for name in ds.data_vars:
                        ds[name].attrs["units"] = "mm"
                else:
                    cell_area = _grid_cell_area_m2(ds)
                    interval_seconds = _leadtime_interval_seconds(ds)
                    ds = (ds * cell_area) / interval_seconds
                    for name in ds.data_vars:
                        ds[name].attrs["units"] = "m3 s-1"
    
            elif var_code in ["DSWR", "DLWR", "NOLR"]:
                ds = _deaccumulate(ds) / 86400.0
                for name in ds.data_vars:
                    ds[name].attrs["units"] = "W m-2"
    
            return ds
    
        def _rearrange_to_valid_time(ds):
            if "T" in ds.dims and "leadtime" in ds.dims:
                if "valid_time" not in ds.coords:
                    ds = ds.assign_coords(valid_time=ds["T"] + ds["leadtime"])
    
                ds = ds.stack(_sample=("T", "leadtime"))
                ds = ds.reset_index("_sample")
    
                valid_time = ds["valid_time"].values
                if "T" in ds.coords:
                    ds = ds.rename({"T": "init_time"})
    
                ds = ds.assign_coords(T=("_sample", valid_time))
                ds = ds.swap_dims({"_sample": "T"})
                ds = ds.drop_vars("_sample", errors="ignore")
                ds = ds.sortby("T")
    
            ds = ds.drop_vars(["valid_time", "init_time", "leadtime"], errors="ignore")
    
            if "member" in ds.dims:
                ds = ds.rename({"member": "M"})
    
            for name in list(ds.data_vars):
                if "M" in ds[name].dims:
                    preferred = ["M", "T", "level", "Y", "X"]
                else:
                    preferred = ["T", "level", "Y", "X"]
                dims = [dim for dim in preferred if dim in ds[name].dims]
                dims += [dim for dim in ds[name].dims if dim not in dims]
                ds[name] = ds[name].transpose(*dims)
    
            return ds
    
        dir_to_save = Path(dir_to_save)
        dir_to_save.mkdir(parents=True, exist_ok=True)
    
        month_key = f"{int(month_of_initialization):02}"
        abb_month_ini = month_abbr[int(month_of_initialization)]
        years_str = f"{years[0]}_{years[-1]}" if len(years) > 1 else years[0]
        lead_str = f"{leadtime_hour[0]}-{leadtime_hour[-1]}" if len(leadtime_hour) > 1 else leadtime_hour[0]
    
        store_file_path = {}
        client = cdsapi.Client()
    
        for cv in center_variable:
            try:
                c, v = cv.split(".", 1)
                if c not in centre:
                    raise ValueError(f"Unknown centre/system code: {c}")
                if v not in variables_1 and v not in variables_2:
                    raise ValueError(f"Unknown variable code: {v}")
    
                cent = centre[c]
                syst = system[c]
    
                request_day = int(day_of_initialization)
                if year_forecast is None and cent == "jma":
                    request_day = init_day_dict_jma[month_key]
                elif year_forecast is None and cent == "ncep":
                    request_day = init_day_dict_ncep[month_key]
    
                day_key = f"{int(request_day):02}"
                unit_tag = "_m3s" if v == "RUNOFF" and runoff_units == "m3/s" else ""
                output_file = (
                    dir_to_save
                    / f"{file_prefix}_{cent}{syst}_{v}_{abb_month_ini}{day_key}_{years_str}_{lead_str}{unit_tag}.nc"
                )
    
                if output_file.exists() and not force_download:
                    print(f"{output_file} already exists. Skipping download.")
                    store_file_path[cv] = output_file
                    continue
    
                fmt = data_format.lower()
                if fmt not in ["netcdf", "grib"]:
                    raise ValueError("data_format must be 'netcdf' or 'grib'.")
    
                suffix = "grib" if fmt == "grib" else "nc"
                temp_file = dir_to_save / f"temp_{cent}{syst}_{v}.{suffix}"
    
                if v in variables_2:
                    dataset = "seasonal-original-pressure-levels"
                    pressure_level = v.rsplit("_", 1)[1]
                    request = {
                        "originating_centre": cent,
                        "system": syst,
                        "variable": [variables_2[v]],
                        "pressure_level": [pressure_level],
                        "year": years,
                        "month": [month_key],
                        "day": [day_key],
                        "leadtime_hour": [str(lt) for lt in leadtime_hour],
                        "data_format": fmt,
                        "area": area,
                    }
                else:
                    dataset = "seasonal-original-single-levels"
                    request = {
                        "originating_centre": cent,
                        "system": syst,
                        "variable": [variables_1[v]],
                        "year": years,
                        "month": [month_key],
                        "day": [day_key],
                        "leadtime_hour": [str(lt) for lt in leadtime_hour],
                        "data_format": fmt,
                        "area": area,
                    }
    
                print(f"Requesting data from '{dataset}' for {cv}...")
                client.retrieve(dataset, request).download(str(temp_file))
                print(f"Downloaded: {temp_file}")
    
                ds = _open_download(temp_file, fmt)
                ds = _normalize_cds_dataset(ds)
    
                if ensemble_mean in ["mean", "median"] and "member" in ds.dims:
                    ds = getattr(ds, ensemble_mean)(dim="member")
    
                ds = _apply_units(ds, v)
                if output_layout == "valid_time":
                    ds = _rearrange_to_valid_time(ds)
    
                ds.attrs["source_dataset"] = dataset
                ds.attrs["originating_centre"] = cent
                ds.attrs["system"] = syst
                ds.attrs["variable_code"] = v
                ds.attrs["download_format"] = fmt
                ds.attrs["output_layout"] = output_layout
                if v == "RUNOFF":
                    ds.attrs["runoff_units"] = runoff_units
    
                encoding = {name: {"zlib": True, "complevel": 4} for name in ds.data_vars}
                ds.to_netcdf(output_file, encoding=encoding)
                print(f"Saved processed data to: {output_file}")
    
                ds.close()
                store_file_path[cv] = output_file
    
                if temp_file.exists():
                    os.remove(temp_file)
                    print(f"Deleted temp file: {temp_file}")
    
                del ds, request
                gc.collect()
    
            except Exception as exc:
                print(f"Failed to download data for {cv}: {exc}")
    
            _time.sleep(1)
    
        return store_file_path
    

    # def WAS_Download_Models_Daily(
    #     self,
    #     dir_to_save,
    #     center_variable,         # e.g. ["ECMWF_51.PRCP", "UKMO_603.TEMP", ...]
    #     month_of_initialization, # int: e.g. 2 for February
    #     day_of_initialization,   # int: e.g. 1 for the 1st day
    #     leadtime_hour,           # list of strings: e.g. ["24","48",..., "5160"]
    #     year_start_hindcast,
    #     year_end_hindcast,
    #     area,
    #     year_forecast=None,
    #     ensemble_mean=None,
    #     force_download=False,
    # ):
    #     """
    #     Download daily/sub-daily seasonal forecast model data (original)
    #     using 'seasonal-original-single-levels' from the CDS.
    
    #     Parameters:
    #         dir_to_save (str or Path): Directory to save the downloaded files.
    #         center_variable (list): Each element e.g. "ECMWF_51.PRCP"
    #             - left side of '.' is model (ECMWF_51),
    #             - right side is variable short code (PRCP).
    #         month_of_initialization (int): Initialization month (1-12).
    #         day_of_initialization (int): Initialization day (1-31).
    #         leadtime_hour (list of str): e.g. ["24", "48", ..., "5160"].
    #         year_start_hindcast (int): Start year for hindcast data.
    #         year_end_hindcast (int): End year for hindcast data.
    #         area (list): Bounding box as [North, West, South, East].
    #         year_forecast (int, optional): If provided, downloads that single
    #             forecast year. Otherwise downloads hindcast for the specified range.
    #         ensemble_mean (str, optional): e.g. "mean", "median", or None.
    #         force_download (bool): Force download if True, even if file exists.
    #     """
    
    #     # 1. Determine whether we are downloading hindcast or forecast.
    #     if year_forecast is None:
    #         # Hindcast range
    #         years = [str(y) for y in range(year_start_hindcast, year_end_hindcast + 1)]
    #         file_prefix = "hindcast"
    #     else:
    #         # Single forecast year
    #         years = [str(year_forecast)]
    #         file_prefix = "forecast"
    
    #     # 2. Build standard dictionaries for center/system/variables
    #     centre = {
    #         "BOM_2": "bom",
    #         "ECMWF_51": "ecmwf",
    #         "UKMO_604": "ukmo", # month of initialization available for forecast are Apr to __
    #         "UKMO_603": "ukmo", # month of initialization available for forecast are Jan to Mar
    #         "UKMO_605": "ukmo",
    #         "UKMO_610": "ukmo",
    #         "METEOFRANCE_8": "meteo_france",
    #         "METEOFRANCE_9": "meteo_france",
    #         "DWD_21": "dwd",
    #         "DWD_22": "dwd",
    #         # "DWD_2": "dwd",
    #         "CMCC_35": "cmcc",
    #         "CMCC_4": "cmcc",
    #         "NCEP_2": "ncep",
    #         "JMA_3": "jma",
    #         "JMA_4": "jma",
    #         "ECCC_4": "eccc",
    #         "ECCC_5": "eccc",
    #     }
    
    #     system = {
    #         "BOM_2": "2",
    #         "ECMWF_51": "51",
    #         "UKMO_604": "604",
    #         "UKMO_603": "603",
    #         "UKMO_605": "605",
    #         "UKMO_610": "610",
    #         "METEOFRANCE_8": "8",
    #         "METEOFRANCE_9": "9",
    #         "DWD_21": "21",
    #         "DWD_22": "22",
    #         # "DWD_2": "2",
    #         "CMCC_35": "35",
    #         "CMCC_4": "4",
    #         "NCEP_2": "2",
    #         "JMA_3": "3",
    #         "JMA_4": "4",
    #         "ECCC_4": "4",
    #         "ECCC_5": "5",
    #     }
    
    #     variables_1 = {
    #         "PRCP":  "total_precipitation",
    #         "TEMP":  "2m_temperature",
    #         "TDEW": "2m_dewpoint_temperature",
    #         "TMAX":  "maximum_2m_temperature_in_the_last_24_hours",
    #         "TMIN":  "minimum_2m_temperature_in_the_last_24_hours",
    #         "UGRD10":"10m_u_component_of_wind",
    #         "VGRD10":"10m_v_component_of_wind",
    #         "SST":   "sea_surface_temperature",
    #         "SLP": "mean_sea_level_pressure",
    #         "DSWR": "surface_solar_radiation_downwards",
    #         "DLWR": "surface_thermal_radiation_downwards",
    #         "NOLR": "top_net_thermal_radiation",
    #         "RUNOFF": "surface_runoff"
            
    #     }
    #     variables_2 = {
    #         "HUSS_1000": "specific_humidity",
    #         "HUSS_925": "specific_humidity",
    #         "HUSS_850": "specific_humidity",
    #         "UGRD_1000": "u_component_of_wind",
    #         "UGRD_925": "u_component_of_wind",
    #         "UGRD_850": "u_component_of_wind",
    #         "UGRD_700": "u_component_of_wind",
    #         "UGRD_200": "u_component_of_wind",
    #         "VGRD_1000": "v_component_of_wind",
    #         "VGRD_925": "v_component_of_wind",
    #         "VGRD_850": "v_component_of_wind",
    #         "VGRD_700": "v_component_of_wind",
    #         "VGRD_200": "v_component_of_wind",
    #     }

    #     ### Particularity for day of initialization NCEP and JMA
    #     init_day_dict_jma = {
    #         "01":16, "02":10, "03":12, "04":11, "05":16, "06":15,
    #         "07":15, "08":14, "09":13, "10":13, "11":12, "12":12
    #     }

    #     init_day_dict_ncep = {
    #         "01":1, "02":5, "03":2, "04":1, "05":1, "06":5,
    #         "07":5, "08":4, "09":3, "10":3, "11":2, "12":2
    #     }
        
    
    #     # 3. Ensure the output directory exists
    #     dir_to_save = Path(dir_to_save)
    #     dir_to_save.mkdir(parents=True, exist_ok=True)
    #     store_file_path = {}
    #     client = cdsapi.Client()        
    #     # 4. Loop over each center-variable combination
    #     for cv in center_variable:
    #         # Example: "ECMWF_51.PRCP"
    #         c = cv.split(".")[0]  # e.g. "ECMWF_51"
    #         v = cv.split(".")[1]  # e.g. "PRCP"
    
    #         # Map to the Copernicus naming
    #         cent = centre[c]
    #         syst = system[c]

    #         ### Particularity for day of initialization NCEP and JMA
    #         if cent == "jma" and year_forecast is None:
    #             day_of_initialization = init_day_dict_jma[month_of_initialization]
    #         if cent == "ncep" and year_forecast is None:
    #             day_of_initialization = init_day_dict_ncep[month_of_initialization]

    #         # Build a single output path
    #         abb_mont_ini = month_abbr[int(month_of_initialization)]
            
    #         # E.g. "hindcast_ecmwf51_PRCP_Feb01_1981-2016_24-5160.nc"
    #         years_str = f"{years[0]}_{years[-1]}" if len(years) > 1 else years[0]
    #         lead_str  = f"{leadtime_hour[0]}-{leadtime_hour[-1]}" if len(leadtime_hour) > 1 else leadtime_hour[0]
    
    #         output_file = (
    #             dir_to_save /
    #             f"{file_prefix}_{cent}{syst}_{v}_{abb_mont_ini}01_{years_str}_{lead_str}.nc"
    #         )

    #         # output_file = (
    #         #     dir_to_save /
    #         #     f"{file_prefix}_{cent}{syst}_{v}_{abb_mont_ini}{day_of_initialization}_{years_str}_{lead_str}.nc"
    #         # )
            

    #         if not force_download and output_file.exists():
    #             print(f"{output_file} already exists. Skipping download.")
    #             store_file_path[f"{cent}{syst}"] = output_file
            
    #         else:

    #             try:
    #                 # Temporary file to download
    #                 temp_file = dir_to_save / f"temp_{cent}{syst}_{v}.nc"

    #                 if v in variables_2:
    #                     press_level = v.split("_")[1]        
    #                     # 5. Prepare the request for 'seasonal-original-pressure-levels'
    #                     dataset = "seasonal-original-pressure-levels"
    #                     request = {
    #                         "originating_centre": cent,
    #                         "system": syst,
    #                         "variable": [variables_2[v]],
    #                         "pressure_level": press_level,
    #                         "year": years,  # list of strings
    #                         "month": [f"{int(month_of_initialization):02}"],
    #                         "day":   [f"{int(day_of_initialization):02}"],
    #                         "leadtime_hour": leadtime_hour,  # e.g. ["24","48",..., "5160"]
    #                         "data_format": "netcdf",
    #                         "area": area,   # e.g. [90, -180, -90, 180]
    #                     }
    #                 else:
    #                     dataset = "seasonal-original-single-levels"
    #                     request = {
    #                         "originating_centre": cent,
    #                         "system": syst,
    #                         "variable": [variables_1[v]],
    #                         "year": years,  # list of strings
    #                         "month": [f"{int(month_of_initialization):02}"],
    #                         "day":   [f"{int(day_of_initialization):02}"],
    #                         "leadtime_hour": leadtime_hour,  # e.g. ["24","48",..., "5160"]
    #                         "data_format": "netcdf",
    #                         "area": area,   # e.g. [90, -180, -90, 180]                    
    #                     }
    #                 # print(request, temp_file, dataset, cent, syst, [variables_1[v]], v, years, month_of_initialization, day_of_initialization, leadtime_hour, area)    
    #                 # 6. Download from CDS
    #                 print(f"Requesting data from '{dataset}' for {cv}...")
    #                 client.retrieve(dataset, request).download(str(temp_file))
    #                 print(f"Downloaded: {temp_file}")

    #                 # 7. Post-process with xarray
    #                 ##########################################################
    #                 # Take in account level pressure for some variables in this part
    #                 ##########################################################
                    
    #                 ds = xr.open_dataset(temp_file)
    #                 if 'forecast_reference_time' in ds.coords:                     
    #                     time = (ds['forecast_reference_time'][0] + ds['forecast_period']).data
    #                     ds = ds.isel(forecast_reference_time=0)
    #                     ds['forecast_period'] = time
    #                     ds = ds.rename({"forecast_period":"time"})
    #                     ds = ds.drop_vars(['forecast_reference_time', 'valid_time', 'number'])

    #                 # elif 'forecast_reference_time' in ds.coords:
    #                 #     time = (ds['forecast_reference_time'] + ds['forecast_period']).data
    #                 #     ds = ds.assign_coords(time=(('forecast_reference_time', 'forecast_period'), time))
    #                 #     ds = ds.stack(time=('forecast_reference_time', 'forecast_period'))
    #                 #     ds = ds.drop_vars(['forecast_reference_time', 'forecast_period'])
    #                 #     ds = ds.rename({"valid_time":"time"})

    #                 else:
    #                     time = (ds['indexing_time'][0] + ds['forecast_period']).data
    #                     ds = ds.isel(forecast_reference_time=0)
    #                     ds['forecast_period'] = time
    #                     ds = ds.rename({"forecast_period":"time"})
    #                     ds = ds.drop_vars(['indexing_time', 'valid_time', 'number'])
                        
    #                     # time = (ds['indexing_time']  + ds['forecast_period']).data
    #                     # ds = ds.assign_coords(time=(('indexing_time', 'forecast_period'), time)).isel(indexing_time=0)
    #                     # ds = ds.stack(time=('indexing_time', 'forecast_period'))
    #                     # ds = ds.drop_vars(['indexing_time', 'forecast_period'])
    #                     # ds = ds.rename({"valid_time":"time"})                    
        
    #                 # If there's an ensemble dimension, apply ensemble mean/median if requested
    #                 if ensemble_mean in ["mean", "median"] and "number" in ds.dims:
    #                     ds = getattr(ds, ensemble_mean)(dim="number")
        
    #                 # Flip latitude
    #                 if "latitude" in ds.coords:
    #                     ds = ds.isel(latitude=slice(None, None, -1))

    #                 if v in ["TMIN","TEMP","TMAX","SST", "TDEW"]:
    #                     ds = ds - 273.15
    #                 if v =="SLP":
    #                     ds = ds / 100
    #                 if v =="PRCP":
    #                     ds['time'] = ds['time'].to_index()
    #                     years_ = ds['time'].dt.year
    #                     tampon = []
    #                     for year_ in np.unique(years_):
                            
    #                         # Select the data for the specific year
    #                         yearly_ds = ds.sel(time=ds['time'].dt.year == year_)
                            
    #                         # Calculate differences for the year
    #                         differences = [yearly_ds.isel(time=i) - yearly_ds.isel(time=i-1) for i in range(1, len(yearly_ds['time']))]
    #                         differences = xr.concat(differences, dim="time")
    #                         differences['time'] = yearly_ds['time'].isel(time=slice(1,None))
    #                         tampon.append(differences)
    #                     ds = (xr.concat(tampon, dim="time") * 1000).where(lambda x: x >= 0, other=0)

    #                 if v=="RUNOFF":
    #                     ds['time'] = ds['time'].to_index()
    #                     diffs = ds.groupby('time.year').apply(lambda x: x.diff('time'))
    #                     ds = diffs.where(lambda x: x >= 0, other=0)
    #                     lat = ds.latitude
    #                     lon = ds.longitude
    #                     dlon = np.deg2rad(0.1)
    #                     dlat = np.deg2rad(0.1)
    #                     r = 6371000 # Earth radius in meters
    #                     # Area for each grid cell
    #                     area_= (r ** 2) * dlon * np.cos(np.deg2rad(lat)) * dlat
    #                     # Perform the conversion to m^3/s
    #                     ds = (ds * area_) / 86400                        

    #                 if v in ["DSWR","DLWR","OLR"]:
    #                     ds['time'] = ds['time'].to_index()
    #                     years_ = ds['time'].dt.year
    #                     tampon = []
    #                     for year_ in np.unique(years_):
                            
    #                         # Select the data for the specific year
    #                         yearly_ds = ds.sel(time=ds['time'].dt.year == year_)
                            
    #                         # Calculate differences for the year
    #                         differences = [yearly_ds.isel(time=i) - yearly_ds.isel(time=i-1) for i in range(1, len(yearly_ds['time']))]
    #                         differences = xr.concat(differences, dim="time")
    #                         differences['time'] = yearly_ds['time'].isel(time=slice(1,None))
    #                         tampon.append(differences)
    #                     ds = xr.concat(tampon, dim="time")/(24*60*60)

    #                 # Finally, rename the coords to X, Y, T to match my style
    #                 if "longitude" in ds.coords:
    #                     ds = ds.rename({"longitude": "X"})
    #                 if "latitude" in ds.coords:
    #                     ds = ds.rename({"latitude": "Y"})
    #                 if "time" in ds.coords:
    #                     ds = ds.rename({"time": "T"})
        
    #                 # 8. Save the processed data
    #                 ds.to_netcdf(output_file)
    #                 print(f"Saved processed data to: {output_file}")
    #                 ds.close()
    #                 store_file_path[f"{cent}{syst}"] = output_file
    #                 os.remove(temp_file)
    #                 print(f"Deleted temp file: {temp_file}")
    #                 del request, ds        
    #                 gc.collect()  
    #             except Exception as e:
    #                 print(f"Failed to download data for {cv}: {e}")  
    #         _time.sleep(1)  # Sleep to avoid overwhelming the server
    #     return store_file_path

    def WAS_Download_AgroIndicators(
            self,
            dir_to_save,
            variables,
            year_start,
            year_end,
            area,
            seas=["01", "02", "03"],  # e.g. NDJ = ["11","12","01"]
            force_download=False,
            max_retries=3,
            retry_delay=5,
        ):
        """
        Download agro-meteorological indicators for specified variables, years, and months,
        handling cross-year seasons (e.g., NDJ) with retries for failed downloads.

        Parameters:
            dir_to_save (str): Directory to save the downloaded files.
            variables (list): List of shorthand variables (e.g., ["AGRO.PRCP", "AGRO.TMAX"]).
            year_start (int): Start year for the data.
            year_end (int): End year for the data.
            area (list): Bounding box as [North, West, South, East].
            seas (list): List of months (e.g., ["11","12","01"] for NDJ).
            force_download (bool): If True, forces download even if file exists.
            max_retries (int): Maximum number of retry attempts for failed downloads (default: 3).
            retry_delay (int): Seconds to wait between retry attempts (default: 5).
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
            "AGRO.DSWR": ("solar_radiation_flux", None),
            "AGRO.ETP": ("reference_evapotranspiration", None),
            "AGRO.WFF": ("10m_wind_speed", "24_hour_mean"),
            "AGRO.HUMAX": ("2m_relative_humidity_derived", "24_hour_maximum"),
            "AGRO.HUMIN": ("2m_relative_humidity_derived", "24_hour_minimum"),
        }

        version = "2_0"
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
            if not force_download and output_path.exists():
                print(f"{output_path} already exists. Skipping download.")
                continue

            # Accumulate all partial datasets
            all_years_datasets = []
            all_years_downloaded = True

            # Loop over each year in the requested range
            for year in range(year_start, year_end + 1):
                # Split the months into those belonging to "base" year vs "next" year
                base_months = [m for m in season_months if m >= pivot]
                next_months = [m for m in season_months if m < pivot]

                # 1) Download part A (base-year months), if any
                if base_months:
                    months_base = [month_str(m) for m in base_months]
                    zip_file_path = dir_to_save / f"Obs_{var_short}_{year}_{season_str}_partA.zip"
                    success = False
                    retries = 0

                    while retries < max_retries and not success:
                        try:
                            client = cdsapi.Client()
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

                            print(f"Attempt {retries + 1}/{max_retries}: Downloading {cds_variable} for {year} months={months_base}")
                            client.retrieve("sis-agrometeorological-indicators", request).download(str(zip_file_path))
                            success = True
                        except Exception as e:
                            retries += 1
                            print(f"Attempt {retries}/{max_retries} failed for {cds_variable} year={year} Part A: {e}")
                            if retries < max_retries:
                                print(f"Retrying after {retry_delay} seconds...")
                                _time.sleep(retry_delay)
                            if zip_file_path.exists():
                                os.remove(zip_file_path)
                                print(f"Deleted incomplete ZIP file: {zip_file_path}")

                    if not success:
                        print(f"Failed to download {cds_variable} year={year} Part A after {max_retries} attempts.")
                        all_years_downloaded = False
                        continue

                    # Unzip each netCDF and append
                    try:
                        with zipfile.ZipFile(zip_file_path, 'r') as z:
                            for nc_name in z.namelist():
                                with z.open(nc_name) as f:
                                    ds = xr.open_dataset(io.BytesIO(f.read()))
                                    all_years_datasets.append(ds)
                        os.remove(zip_file_path)
                        print(f"Deleted ZIP file: {zip_file_path}")
                    except Exception as e:
                        print(f"Failed to extract/process {zip_file_path}: {e}")
                        all_years_downloaded = False
                        if zip_file_path.exists():
                            os.remove(zip_file_path)
                            print(f"Deleted ZIP file due to processing error: {zip_file_path}")
                        continue

                # 2) Download part B (next-year months), if any and if we have a next year
                if next_months and (year < year_end + 1):
                    year_next = year + 1
                    months_next = [month_str(m) for m in next_months]
                    zip_file_path = dir_to_save / f"Obs_{var_short}_{year}_{season_str}_partB_{year_next}.zip"
                    success = False
                    retries = 0

                    while retries < max_retries and not success:
                        try:
                            client = cdsapi.Client()
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

                            print(f"Attempt {retries + 1}/{max_retries}: Downloading {cds_variable} for {year_next} months={months_next}")
                            client.retrieve("sis-agrometeorological-indicators", request).download(str(zip_file_path))
                            success = True
                        except Exception as e:
                            retries += 1
                            print(f"Attempt {retries}/{max_retries} failed for {cds_variable} year={year_next} Part B: {e}")
                            if retries < max_retries:
                                print(f"Retrying after {retry_delay} seconds...")
                                _time.sleep(retry_delay)
                            if zip_file_path.exists():
                                os.remove(zip_file_path)
                                print(f"Deleted incomplete ZIP file: {zip_file_path}")

                    if not success:
                        print(f"Failed to download {cds_variable} year={year_next} Part B after {max_retries} attempts.")
                        all_years_downloaded = False
                        continue

                    # Unzip each netCDF and append
                    try:
                        with zipfile.ZipFile(zip_file_path, 'r') as z:
                            for nc_name in z.namelist():
                                with z.open(nc_name) as f:
                                    ds = xr.open_dataset(io.BytesIO(f.read()))
                                    all_years_datasets.append(ds)
                        os.remove(zip_file_path)
                        print(f"Deleted ZIP file: {zip_file_path}")
                    except Exception as e:
                        print(f"Failed to extract/process {zip_file_path}: {e}")
                        all_years_downloaded = False
                        if zip_file_path.exists():
                            os.remove(zip_file_path)
                            print(f"Deleted ZIP file due to processing error: {zip_file_path}")
                        continue

            # Post-process & combine all partial years
            if all_years_datasets and all_years_downloaded:
                try:
                    combined_ds = xr.concat(all_years_datasets, dim="time").drop_vars('crs', errors="ignore")
                    
                    # Unit conversions
                    if var in ["AGRO.TMIN", "AGRO.TEMP", "AGRO.TMAX"]:
                        combined_ds = combined_ds - 273.15  # Kelvin to Celsius
  
                    # Aggregate for cross-year seasons
                    combined_ds = self._aggregate_crossyear(
                        ds=combined_ds,
                        season_months=season_months,
                        var_name=var
                    )

                    ########## Revoir ceci surtout l'emplacement et calcul
                    if var == "AGRO.DSWR":
                        combined_ds = combined_ds / 86400  # J/m^2/day to W/m^2 

                    # Rename dimensions
                    if "lon" in combined_ds.dims:
                        combined_ds = combined_ds.rename({"lon": "X"})
                    if "lat" in combined_ds.dims:
                        combined_ds = combined_ds.rename({"lat": "Y"})
                    combined_ds = combined_ds.isel(Y=slice(None, None, -1))

                    # Adjust time coordinate
                    if len(seas)==1:
                        combined_ds["time"] = [f"{year}-{seas[0]}-01" for year in combined_ds["time"].astype(str).values]
                    elif len(seas) in [2,3]:
                        combined_ds["time"] = [f"{year}-{seas[1]}-01" for year in combined_ds["time"].astype(str).values]
                    elif len(seas) in [4,5]:
                        combined_ds["time"] = [f"{year}-{seas[3]}-01" for year in combined_ds["time"].astype(str).values]
                    else:
                        combined_ds["time"] = [f"{year}-{seas[4]}-01" for year in combined_ds["time"].astype(str).values]
                    combined_ds["time"] = combined_ds["time"].astype("datetime64[ns]")
                    combined_ds = combined_ds.rename({"time": "T"})

                    # Save to NetCDF
                    combined_ds.to_netcdf(output_path)
                    combined_ds.close()
                    print(f"Saved final dataset for {var} to: {output_path}")
                except Exception as e:
                    print(f"Failed to process or save combined dataset for {var}: {e}")
            else:
                print(f"No data downloaded for {var} in {season_str}.")
   
    # -------------------------------------------------------------------------
    # Helper for Reanalysis cross-year post-processing (optional)
    # -------------------------------------------------------------------------
    def _postprocess_reanalysis(self, ds, var_name):
        """
        Drop extra coords, rename dims, flip lat, etc.
        Adjust as needed for ERA5 quirks.
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
        # ds = ds.drop_vars('zlev').squeeze()
        # ds = ds.drop_vars('lev').squeeze()
        keep_vars = [var_name, 'T', 'X', 'Y']
        drop_vars = [v for v in ds.variables if v not in keep_vars]
        return ds.drop_vars(drop_vars, errors="ignore")

    def _aggregate_crossyear(self, ds, season_months, var_name):
        """
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

        if any(x in var_name for x in ["TEMP","TMIN","TMAX","SST","SLP","RUNOFF"]):
            ds_out = ds.groupby("season_year").mean("time")

        elif any(x in var_name for x in ["PRCP","DSWR","DLWR","NOLR"]):
            # For precipitation and radiation, we sum over time
            ds_out = ds.groupby("season_year").sum("time")
            # ds_out = ds.groupby("season_year").mean("time")
        else:
            ds_out = ds.groupby("season_year").mean("time")
        # 4) Rename "season_year" to "time", 
        #    so we end up with a time dimension (representing each seasonal year).
        ds_out = ds_out.rename({"season_year": "time"})
    
        return ds_out


    # def WAS_Download_Reanalysis(
    #         self,
    #         dir_to_save,
    #         center_variable,
    #         year_start,
    #         year_end,
    #         area,
    #         seas=["01", "02", "03"],  # e.g. NDJ = ["11","12","01"]
    #         force_download=False,
    #         run_avg=1
    #     ):
        
    #         """
    #         Download reanalysis data for specified center-variable combinations, years, and months,
    #         handling cross-year seasons (e.g., NDJ).
    #         """
    
    #         dir_to_save = Path(dir_to_save)
    #         dir_to_save.mkdir(parents=True, exist_ok=True)
    
    #         # Parse center and variable strings
    #         centers = [cv.split(".")[0] for cv in center_variable]
    #         vars_   = [cv.split(".")[1] for cv in center_variable]
    
    #         # Example reanalysis centers
    #         centre_dict = {"ERA5": "ERA5", "MERRA2": "MERRA2", "NOAA": "NOAA"}
    
    #         # Single-level monthly means
    #         variables_1 = {
    #             "PRCP": "total_precipitation",
    #             "TEMP": "2m_temperature",
    #             "TMAX": "maximum_2m_temperature_in_the_last_24_hours",
    #             "TMIN": "minimum_2m_temperature_in_the_last_24_hours",
    #             "UGRD10": "10m_u_component_of_wind",
    #             "VGRD10": "10m_v_component_of_wind",
    #             "SST": "sea_surface_temperature",
    #             "SLP": "mean_sea_level_pressure",
    #             "DSWR": "surface_solar_radiation_downwards",
    #             "DLWR": "surface_thermal_radiation_downwards",
    #             "NOLR": "top_net_thermal_radiation",
    #         }
            
    #         # Pressure-level monthly means
    #         variables_2 = {
    #             "HUSS_1000": "specific_humidity",
    #             "HUSS_925": "specific_humidity",
    #             "HUSS_850": "specific_humidity",
    #             "UGRD_1000": "u_component_of_wind",
    #             "UGRD_925": "u_component_of_wind",
    #             "UGRD_850": "u_component_of_wind",
    #             "UGRD_700": "u_component_of_wind",
    #             "UGRD_600": "u_component_of_wind",
    #             "UGRD_200": "u_component_of_wind",
    #             "VGRD_1000": "v_component_of_wind",
    #             "VGRD_925": "v_component_of_wind",
    #             "VGRD_850": "v_component_of_wind",
    #             "VGRD_700": "v_component_of_wind",
    #             "VGRD_600": "v_component_of_wind",
    #             "VGRD_200": "v_component_of_wind",
    #         }
            
    #         # Helper for zero-padded month strings
    #         def m2str(m: int):
    #             return f"{m:02d}"
    
    #         # Convert months to integers (e.g. ["11","12","01"] -> [11,12,1])
    #         season_months = [int(m) for m in seas]
    #         pivot = season_months[0]
    #         # For naming
    #         season_str = "".join([calendar.month_abbr[m] for m in season_months])

    #         now = datetime.datetime.now()
    #         curr_yr, curr_mon = now.year, now.month

    #         file_path = []
    #         for c, v in zip(centers, vars_):
                
    #             # =================================================================
    #             # Special Case: NOAA ERSST from NCEI (Direct Download) 
    #             # https://www.ncei.noaa.gov/data/sea-surface-temperature-extended-reconstructed/v6/access/
    #             # =================================================================

    #             if c == "NOAA" and v == "SST":
    #                 out_file = dir_to_save / f"{c}_{v}_{year_start}_{year_end}_{season_str}.nc"
    #                 if not force_download and out_file.exists():
    #                     print(f"{out_file} exists. Skipping.")
    #                     file_path.append(out_file)
    #                     continue
                    
    #                 print(f"Starting download for NOAA ERSST (v6) for {year_start}-{year_end}")
    
    #                 # Create a cache dir for individual monthly files so we don't redownload them unnecessarily
    #                 cache_dir = dir_to_save / "ersst_cache"
    #                 cache_dir.mkdir(exist_ok=True)
                    
    #                 # Identify exact list of year-months required for the seasonal request
    #                 # This ensures we handle cross-year (e.g. NDJ) correctly
    #                 required_files = []
                    
    #                 # Loop through the "Seasonal Years"
    #                 for s_year in range(year_start, year_end + 1):
                        
    #                     for month in season_months:
    #                         cal_year = s_year
    #                         # if month >= pivot:
    #                         #     cal_year = s_year
    #                         # else:
    #                         #     cal_year = s_year + 1

    #                         # if cal_year == year_end + 1 and month >= pivot:
    #                         #     continue
    #                         # if cal_year > year_end + 1:
    #                         #     continue
    #                         if (cal_year > curr_yr) or (cal_year == curr_yr and month > curr_mon):
    #                             continue
                                
    #                         # ERSST v5 filename format: ersst.v5.YYYYMM.nc
    #                         fname = f"ersst.v6.{cal_year}{month:02d}.nc"
    #                         required_files.append((cal_year, month, fname))
    
    #                 # Download 
    #                 # Base URL for ERSST 'v6'.
    #                 base_url = "https://www.ncei.noaa.gov/data/sea-surface-temperature-extended-reconstructed/v6/access/"
                    
    #                 downloaded_paths = []
                    
    #                 try:
    #                     for (yr, mn, fname) in required_files:
    #                         local_path = cache_dir / fname
    #                         if not local_path.exists():
    #                             # Construct full URL
    #                             url = f"{base_url}/{fname}"
    #                             print(f"Downloading {fname}...")
    #                             r = requests.get(url, timeout=30)
    #                             r.raise_for_status()
    #                             with open(local_path, 'wb') as f:
    #                                 f.write(r.content)
    #                         downloaded_paths.append(local_path)
                        
    #                     print("All files downloaded. merging and processing...")
    
    #                     # Open all files
    #                     # use_cftime=True is safer for long historical records
    #                     ds = xr.open_mfdataset(downloaded_paths)#, combine='by_coords', decode_times=True)
    #                     ds = ds[["sst"]].drop_vars("lev").squeeze()
   
    #                     # --- PRE-PROCESSING SPECIFIC TO ERSST ---
                        
    #                     # 1. Standardize Coordinates (Lat/Lon/Time)
    #                     # ERSST uses 'lat', 'lon', 'time' usually.
    #                     # ERSST Longitude is 0 to 360. We likely want -180 to 180 to match ERA5/bounding box.
    #                     if 'lon' in ds.coords:
    #                         ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
    #                         ds = ds.sortby('lon')
                        
    #                     # 2. Rename to internal standard (X, Y)
    #                     rename_dict = {}
    #                     if 'lat' in ds.coords: rename_dict['lat'] = 'Y'
    #                     if 'lon' in ds.coords: rename_dict['lon'] = 'X'
    #                     if 'sst' in ds.variables: rename_dict['sst'] = 'SST' # Rename var to capital SST for consistency
                        
    #                     ds = ds.rename(rename_dict)

    #                     # print(ds)
                        
    #                     # 3. Slice the Area
    #                     # Area format: [N, W, S, E]
    #                     lat_max, lon_min, lat_min, lon_max = area
                        
    #                     # Ensure X and Y are sorted for slicing
    #                     ds = ds.sortby(['X', 'Y'])
    #                     ds = ds.sel(X=slice(lon_min, lon_max), Y=slice(lat_min, lat_max))
    
    #                     # 5. Aggregate Cross-Year
    #                     # ERSST is in Celsius. Do NOT subtract 273.15 later.
    #                     # We pass 'SST' as var_name so aggregator calculates MEAN.
    #                     ds_agg = self._aggregate_crossyear(ds, season_months, "SST")

    
    #                     # 6. Final Format
    #                     # Rename 'time' to 'T' as per your output spec
    #                     ds_agg = ds_agg.rename({"time": "T"})
                        
    #                     # Ensure Variable name is 'sst' (lowercase) for final output if that is preferred
    #                     if 'SST' in ds_agg:
    #                         ds_agg = ds_agg.rename({'SST': 'sst'})
    #                     if len(seas)==1:
    #                         ds_agg["T"] = [f"{year}-{seas[0]}-01" for year in ds_agg["T"].astype(str).values]
    #                     elif len(seas) in [2,3]:
    #                         ds_agg["T"] = [f"{year}-{seas[1]}-01" for year in ds_agg["T"].astype(str).values]
    #                     elif len(seas) in [4,5]:
    #                         ds_agg["T"] = [f"{year}-{seas[2]}-01" for year in ds_agg["T"].astype(str).values]
    #                     else:
    #                         ds_agg["T"] = [f"{year}-{seas[3]}-01" for year in ds_agg["T"].astype(str).values]
                            
    #                     ds_agg["T"] = ds_agg["T"].astype("datetime64[ns]")
    
    #                     # 7. Save
    #                     ds_agg.to_netcdf(out_file)
    #                     file_path.append(out_file)
    #                     print(f"Saved NOAA ERSST data to {out_file}")
    
    #                 except Exception as e:
    #                     print(f"Failed to download or process NOAA/SST: {str(e)}")
    #                     # Optional: clean up partial downloads if needed, or leave for retry
    #                     import traceback
    #                     traceback.print_exc()
                    
    #                 continue 
    
    #             # =================================================================
    #             # ERA5 / Other Reanalysis Logic
    #             # =================================================================
    #             if c not in centre_dict:
    #                 print(f"Unknown center: {c}, skipping.")
    #                 continue
    
    #             rean = centre_dict[c]
    #             out_file = dir_to_save / f"{c}_{v}_{year_start}_{year_end}_{season_str}.nc"
    #             if (not force_download) and out_file.exists():
    #                 file_path.append(out_file)
    #                 print(f"{out_file} already exists. Skipping.")
    #                 continue
    
    #             # List to accumulate partial downloads
    #             combined_datasets = []
    
    #             # Iterate over each year in [year_start..year_end]
    #             for year in range(year_start, year_end + 1):
    #                 # Split months
    #                 base_months = [m for m in season_months if m >= pivot]
    #                 next_months = [m for m in season_months if m < pivot]
    
    #                 # (A) Base-year
    #                 if base_months:
    #                     base_str = [m2str(m) for m in base_months]
    #                     partA = dir_to_save / f"{c}_{v}_{year}_{season_str}_partA.nc"
    
    #                     # Decide dataset + request
    #                     if v in variables_2:
    #                         press_level = v.split("_")[1]  # e.g. 925 from "HUSS_925"
    #                         dataset = "reanalysis-era5-pressure-levels-monthly-means"
    #                         request = {
    #                             "product_type": ["monthly_averaged_reanalysis"],
    #                             "variable": variables_2[v],
    #                             "pressure_level": press_level,
    #                             "year": str(year),
    #                             "month": base_str,
    #                             "time": ["00:00"],
    #                             "area": area,
    #                             "data_format": "netcdf",
    #                         }
    #                     else:
    #                         dataset = "reanalysis-era5-single-levels-monthly-means"
    #                         request = {
    #                             "product_type": ["monthly_averaged_reanalysis"],
    #                             "variable": variables_1.get(v),
    #                             "year": str(year),
    #                             "month": base_str,
    #                             "time": ["00:00"],
    #                             "area": area,
    #                             "data_format": "netcdf",
    #                         }
    
    #                     # Download
    #                     try:
    #                         client = cdsapi.Client()
    #                         print(f"Downloading {c}/{v}: {year}, months={base_str}")
    #                         client.retrieve(dataset, request).download(str(partA))
                            
    #                         with xr.open_dataset(partA) as dsA:
    #                             dsA = dsA.load()
    #                             dsA = self._postprocess_reanalysis(dsA, v)
    #                             combined_datasets.append(dsA)
    #                         os.remove(partA)
    #                     except Exception as e:
    #                         print(f"Download/Process error for {c}/{v}, year={year} partA: {e}")
    #                         if os.path.exists(partA): os.remove(partA)
    #                         continue
    
    #                 # (B) Next-year
    #                 if next_months and (year < year_end+1):
    #                     year_next = year + 1
    #                     next_str = [m2str(m) for m in next_months]
    #                     partB = dir_to_save / f"{c}_{v}_{year}_{season_str}_partB_{year_next}.nc"
    
    #                     if v in variables_2:
    #                         press_level = v.split("_")[1]
    #                         dataset = "reanalysis-era5-pressure-levels-monthly-means"
    #                         request = {
    #                             "product_type": ["monthly_averaged_reanalysis"],
    #                             "variable": variables_2[v],
    #                             "pressure_level": press_level,
    #                             "year": str(year_next),
    #                             "month": next_str,
    #                             "time": ["00:00"],
    #                             "area": area,
    #                             "data_format": "netcdf",
    #                         }
    #                     else:
    #                         dataset = "reanalysis-era5-single-levels-monthly-means"
    #                         request = {
    #                             "product_type": ["monthly_averaged_reanalysis"],
    #                             "variable": variables_1.get(v),
    #                             "year": str(year_next),
    #                             "month": next_str,
    #                             "time": ["00:00"],
    #                             "area": area,
    #                             "data_format": "netcdf",
    #                         }
    
    #                     # Download
    #                     try:
    #                         client = cdsapi.Client()
    #                         print(f"Downloading {c}/{v}: {year_next}, months={next_str}")
    #                         client.retrieve(dataset, request).download(str(partB))
    
    #                         with xr.open_dataset(partB) as dsB:
    #                             dsB = dsB.load()
    #                             dsB = self._postprocess_reanalysis(dsB, v)
    #                             combined_datasets.append(dsB)
    #                         os.remove(partB)
    #                     except Exception as e:
    #                         print(f"Download/Process error for {c}/{v}, year={year_next} partB: {e}")
    #                         if os.path.exists(partB): os.remove(partB)
    #                         continue
    
    #             if combined_datasets:
    #                 dsC = xr.concat(combined_datasets, dim="time")
                    
    #                 # If T variable -> K to °C
    #                 # IMPORTANT: Only for ERA5/MERRA which are usually Kelvin. 
    #                 # NOAA ERSST is skipped above, so this block only runs for ERA5/MERRA.
    #                 if v in ["TMIN","TEMP","TMAX","SST"]:
    #                     dsC = dsC - 273.15
                    
    #                 # For precipitation or others, the aggregator decides sum vs mean
    #                 dsC = self._aggregate_crossyear(
    #                     ds=dsC,
    #                     season_months=season_months,
    #                     var_name=v
    #                 )
                    
    #                 if v == "PRCP":
    #                     # Convert to mm/month if ERA5 is in m/s or m/day
    #                     # Note: ERA5 monthly means are usually "m per day" or "m total". 
    #                     # Assuming ERA5 is meter -> multiply by 1000 for mm. 
    #                     # Your original code: dsC = 1000 * ds * 30. (Assuming ds variable name error in your snippet, likely meant dsC)
    #                     dsC = dsC * 1000 * 30 
                    
    #                 if v in ["DSWR", "DLWR","NOLR"]:
    #                     dsC = dsC/86400  # Convert to W/m2 if needed
    
    #                 if v == "SLP":
    #                     dsC = dsC / 100  # Convert to hPa(mb)
                       
    #                 if len(seas)==1:
    #                     dsC["time"] = [f"{year}-{seas[0]}-01" for year in dsC["time"].astype(str).values]
    #                 elif len(seas) in [2,3]:
    #                     dsC["time"] = [f"{year}-{seas[1]}-01" for year in dsC["time"].astype(str).values]
    #                 elif len(seas) in [4,5]:
    #                     dsC["time"] = [f"{year}-{seas[2]}-01" for year in dsC["time"].astype(str).values]
    #                 else:
    #                     dsC["time"] = [f"{year}-{seas[3]}-01" for year in dsC["time"].astype(str).values]
    #                 dsC["time"] = dsC["time"].astype("datetime64[ns]")
    #                 dsC = dsC.rename({"time": "T"})
                    
    #                 # Save final
    #                 dsC.to_netcdf(out_file)
    #                 file_path.append(out_file)
    #                 print(f"Saved final reanalysis file: {out_file}")
    #             else:
    #                 print(f"No data found for {c}/{v}.")
                
    #         return file_path


    def WAS_Download_Reanalysis(
        self,
        dir_to_save,
        center_variable,
        year_start,
        year_end,
        area,
        seas=("01", "02", "03"),
        force_download=False,
        max_retries=3,
        retry_delay=5,
    ):
        """
        Download and aggregate monthly reanalysis predictors/predictands.
    
        Supported inputs
        ----------------
        ERA5 monthly means from CDS:
            "ERA5.PRCP", "ERA5.TEMP", "ERA5.SST", "ERA5.SLP",
            "ERA5.DSWR", "ERA5.DLWR", "ERA5.NOLR",
            "ERA5.UGRD_850", "ERA5.VGRD_200", etc.
    
        NOAA ERSST v6 direct monthly SST:
            "NOAA.SST"
    
        Output convention
        -----------------
        NetCDF with dimensions:
            T, Y, X
    
        Unit conventions
        ----------------
        ERA5 monthly averaged reanalysis:
            PRCP/RUNOFF are m/day, so monthly total is value * days * 1000.
            Radiation accumulations are J m-2/day, so seasonal mean flux is
            day-weighted sum(value) / (total_days * 86400).
            Other monthly means are day-weighted seasonal means.
        """
        from pathlib import Path
        import calendar
        import datetime
        import gc
        import os
        import time as _time
    
        import cdsapi
        import numpy as np
        import pandas as pd
        import requests
        import xarray as xr
    
        if isinstance(center_variable, str):
            raise TypeError(
                "center_variable must be a list, for example "
                "['ERA5.PRCP'] or ['ERA5.PRCP', 'NOAA.SST']."
            )
    
        center_variable = list(center_variable)
        season_months = [int(m) for m in seas]
        season_codes = [f"{m:02d}" for m in season_months]
        season_str = "".join(calendar.month_abbr[m] for m in season_months)
        pivot_month = season_months[0]
    
        dir_to_save = Path(dir_to_save)
        dir_to_save.mkdir(parents=True, exist_ok=True)
    
        era5_single_level = {
            "PRCP": "total_precipitation",
            "RUNOFF": "runoff",
            "TEMP": "2m_temperature",
            "TDEW": "2m_dewpoint_temperature",
            "TMAX": "maximum_2m_temperature_since_previous_post_processing",
            "TMIN": "minimum_2m_temperature_since_previous_post_processing",
            "UGRD10": "10m_u_component_of_wind",
            "VGRD10": "10m_v_component_of_wind",
            "SST": "sea_surface_temperature",
            "SLP": "mean_sea_level_pressure",
            "DSWR": "surface_solar_radiation_downwards",
            "DLWR": "surface_thermal_radiation_downwards",
            "NOLR": "top_net_thermal_radiation",
        }
    
        era5_pressure_level = {
            "HUSS_1000": "specific_humidity",
            "HUSS_925": "specific_humidity",
            "HUSS_850": "specific_humidity",
            "UGRD_1000": "u_component_of_wind",
            "UGRD_925": "u_component_of_wind",
            "UGRD_850": "u_component_of_wind",
            "UGRD_700": "u_component_of_wind",
            "UGRD_600": "u_component_of_wind",
            "UGRD_200": "u_component_of_wind",
            "VGRD_1000": "v_component_of_wind",
            "VGRD_925": "v_component_of_wind",
            "VGRD_850": "v_component_of_wind",
            "VGRD_700": "v_component_of_wind",
            "VGRD_600": "v_component_of_wind",
            "VGRD_200": "v_component_of_wind",
        }
    
        total_vars = {"PRCP", "RUNOFF"}
        flux_vars = {"DSWR", "DLWR", "NOLR"}
        kelvin_vars = {"TEMP", "TDEW", "TMAX", "TMIN", "SST"}
        pressure_vars = {"SLP"}
    
        def _season_calendar_months(season_year):
            months = []
            for month in season_months:
                cal_year = season_year if month >= pivot_month else season_year + 1
                months.append((season_year, cal_year, month))
            return months
    
        def _representative_time(season_year):
            mid_month = season_months[len(season_months) // 2]
            cal_year = season_year if mid_month >= pivot_month else season_year + 1
            return pd.Timestamp(cal_year, mid_month, 1)
    
        def _download_with_retries(client, dataset, request, target_file, label):
            for attempt in range(1, max_retries + 1):
                try:
                    print(f"Attempt {attempt}/{max_retries}: downloading {label}...")
                    client.retrieve(dataset, request).download(str(target_file))
                    print(f"Downloaded: {target_file}")
                    return True
                except Exception as exc:
                    print(f"Attempt {attempt}/{max_retries} failed for {label}: {exc}")
                    if target_file.exists():
                        os.remove(target_file)
                        print(f"Deleted incomplete file: {target_file}")
                    if attempt < max_retries:
                        print(f"Retrying after {retry_delay} seconds...")
                        _time.sleep(retry_delay)
            return False
    
        def _normalize_coords(ds):
            rename_map = {}
            if "valid_time" in ds.dims or "valid_time" in ds.coords:
                rename_map["valid_time"] = "time"
            if "longitude" in ds.dims or "longitude" in ds.coords:
                rename_map["longitude"] = "X"
            if "latitude" in ds.dims or "latitude" in ds.coords:
                rename_map["latitude"] = "Y"
            if "lon" in ds.dims or "lon" in ds.coords:
                rename_map["lon"] = "X"
            if "lat" in ds.dims or "lat" in ds.coords:
                rename_map["lat"] = "Y"
    
            ds = ds.rename({old: new for old, new in rename_map.items() if old in ds})
    
            if "expver" in ds.dims and ds.sizes["expver"] == 1:
                ds = ds.isel(expver=0, drop=True)
    
            for coord in ["pressure_level", "level", "isobaricInhPa"]:
                if coord in ds.dims and ds.sizes[coord] == 1:
                    ds = ds.isel({coord: 0}, drop=True)
    
            if "X" in ds.coords and float(ds["X"].max()) > 180:
                ds = ds.assign_coords(X=(((ds["X"] + 180) % 360) - 180)).sortby("X")
    
            if "Y" in ds.coords:
                ds = ds.sortby("Y")
    
            return ds
    
        def _subset_area(ds):
            if "X" not in ds.coords or "Y" not in ds.coords:
                return ds
            north, west, south, east = area
            ds = ds.sortby(["X", "Y"])
            return ds.sel(X=slice(west, east), Y=slice(south, north))
    
        def _rename_single_data_var(ds, var_code):
            data_vars = list(ds.data_vars)
            if len(data_vars) == 1 and data_vars[0] != var_code:
                ds = ds.rename({data_vars[0]: var_code})
            return ds
    
        def _convert_era5_monthly(ds, var_code, days_in_month):
            ds = _rename_single_data_var(ds, var_code)
    
            if var_code in total_vars:
                ds = ds * (1000.0 * days_in_month)
                units = "mm/month"
    
            elif var_code in flux_vars:
                ds = ds / 86400.0
                units = "W m-2"
    
            elif var_code in kelvin_vars:
                ds = ds - 273.15
                units = "degC"
    
            elif var_code in pressure_vars:
                ds = ds / 100.0
                units = "hPa"
    
            else:
                units = None
    
            for name in ds.data_vars:
                if units is not None:
                    ds[name].attrs["units"] = units
                ds[name].attrs["monthly_days"] = int(days_in_month)
    
            ds.attrs["monthly_days"] = int(days_in_month)
            return ds
    
        def _aggregate_months(monthly_datasets, var_code):
            ds = xr.concat(monthly_datasets, dim="month_index")
            monthly_days = [
                int(item.attrs.get("monthly_days", next(iter(item.data_vars.values())).attrs["monthly_days"]))
                for item in monthly_datasets
            ]
            weights = xr.DataArray(
                monthly_days,
                dims="month_index",
            )
    
            if var_code in total_vars:
                out = ds.sum("month_index", keep_attrs=True)
                units = "mm"
    
            elif var_code in flux_vars:
                out = ds.weighted(weights).mean("month_index", keep_attrs=True)
                units = "W m-2"
    
            else:
                out = ds.weighted(weights).mean("month_index", keep_attrs=True)
                units = None
    
            for name in out.data_vars:
                if units is not None:
                    out[name].attrs["units"] = units
                out[name].attrs.pop("monthly_days", None)
    
            return out
    
        def _download_era5_month(client, c, v, season_year, cal_year, month):
            if v in era5_pressure_level:
                dataset = "reanalysis-era5-pressure-levels-monthly-means"
                pressure_level = v.rsplit("_", 1)[1]
                request = {
                    "product_type": ["monthly_averaged_reanalysis"],
                    "variable": [era5_pressure_level[v]],
                    "pressure_level": [pressure_level],
                    "year": str(cal_year),
                    "month": f"{month:02d}",
                    "time": ["00:00"],
                    "area": area,
                    "data_format": "netcdf",
                }
            elif v in era5_single_level:
                dataset = "reanalysis-era5-single-levels-monthly-means"
                request = {
                    "product_type": ["monthly_averaged_reanalysis"],
                    "variable": [era5_single_level[v]],
                    "year": str(cal_year),
                    "month": f"{month:02d}",
                    "time": ["00:00"],
                    "area": area,
                    "data_format": "netcdf",
                }
            else:
                raise ValueError(f"Unknown ERA5 variable code: {v}")
    
            tmp = dir_to_save / f"tmp_{c}_{v}_{season_year}_{cal_year}{month:02d}.nc"
            label = f"{c}.{v} {cal_year}-{month:02d}"
            ok = _download_with_retries(client, dataset, request, tmp, label)
            if not ok:
                return None
    
            try:
                with xr.open_dataset(tmp) as ds:
                    ds = _normalize_coords(ds).load()
                days = calendar.monthrange(cal_year, month)[1]
                ds = _convert_era5_monthly(ds, v, days)
                return ds
            finally:
                if tmp.exists():
                    os.remove(tmp)
                    print(f"Deleted temporary file: {tmp}")
    
        def _download_noaa_ersst(v, out_file):
            if v != "SST":
                print(f"NOAA direct download currently supports SST only. Skipping NOAA.{v}.")
                return None
    
            cache_dir = dir_to_save / "ersst_cache"
            cache_dir.mkdir(exist_ok=True)
            base_url = "https://www.ncei.noaa.gov/data/sea-surface-temperature-extended-reconstructed/v6/access"
    
            now = datetime.datetime.now()
            required = []
            for season_year in range(year_start, year_end + 1):
                for _, cal_year, month in _season_calendar_months(season_year):
                    if (cal_year > now.year) or (cal_year == now.year and month > now.month):
                        continue
                    required.append((cal_year, month, f"ersst.v6.{cal_year}{month:02d}.nc"))
    
            paths = []
            for cal_year, month, fname in sorted(set(required)):
                local_path = cache_dir / fname
                if not local_path.exists() or force_download:
                    url = f"{base_url}/{fname}"
                    print(f"Downloading NOAA ERSST {cal_year}-{month:02d}...")
                    response = requests.get(url, timeout=60)
                    response.raise_for_status()
                    with open(local_path, "wb") as handle:
                        handle.write(response.content)
                paths.append(local_path)
    
            ds = xr.open_mfdataset(paths, combine="by_coords")
            if "sst" not in ds:
                raise ValueError("NOAA ERSST file does not contain variable 'sst'.")
    
            ds = ds[["sst"]]
            ds = _normalize_coords(ds)
            ds = _subset_area(ds)
            ds = ds.rename({"sst": "SST"})
            ds = ds.drop_vars("elv", errors="ignore").squeeze()
    
            seasonal = []
            for season_year in range(year_start, year_end + 1):
                monthly = []
                for _, cal_year, month in _season_calendar_months(season_year):
                    t0 = np.datetime64(f"{cal_year}-{month:02d}-01")
                    month_ds = ds.sel(time=t0, method="nearest").load()
                    days = calendar.monthrange(cal_year, month)[1]
                    month_ds["SST"].attrs["monthly_days"] = int(days)
                    month_ds.attrs["monthly_days"] = int(days)
                    monthly.append(month_ds)
    
                out = _aggregate_months(monthly, "SST")
                out = out.expand_dims(time=[_representative_time(season_year)])
                seasonal.append(out)
    
            ds.close()
            out = xr.concat(seasonal, dim="time")
            out = out.rename({"time": "T"})
            out = _finalize_output(out, "NOAA", v)
            out.to_netcdf(out_file)
            return out_file
    
        def _finalize_output(ds, center, var_code):
            if "time" in ds.dims or "time" in ds.coords:
                ds = ds.rename({"time": "T"})
    
            preferred = ["T", "Y", "X"]
            for name in list(ds.data_vars):
                dims = [dim for dim in preferred if dim in ds[name].dims]
                dims += [dim for dim in ds[name].dims if dim not in dims]
                ds[name] = ds[name].transpose(*dims)
    
            ds.attrs["center"] = center
            ds.attrs["variable_code"] = var_code
            ds.attrs["season"] = season_str
            ds.attrs["season_months"] = ",".join(season_codes)
            return ds
    
        file_path = []
        client = cdsapi.Client()
    
        for cv in center_variable:
            try:
                c, v = cv.split(".", 1)
                out_file = dir_to_save / f"{c}_{v}_{year_start}_{year_end}_{season_str}.nc"
    
                if out_file.exists() and not force_download:
                    print(f"{out_file} already exists. Skipping.")
                    file_path.append(out_file)
                    continue
    
                if c == "NOAA":
                    result = _download_noaa_ersst(v, out_file)
                    if result is not None:
                        file_path.append(result)
                        print(f"Saved NOAA.{v} seasonal file: {out_file}")
                    continue
    
                if c != "ERA5":
                    print(f"Unsupported center in this corrected function: {c}. Skipping.")
                    continue
    
                seasonal_outputs = []
                all_ok = True
                for season_year in range(year_start, year_end + 1):
                    monthly_datasets = []
                    for _, cal_year, month in _season_calendar_months(season_year):
                        ds_month = _download_era5_month(client, c, v, season_year, cal_year, month)
                        if ds_month is None:
                            all_ok = False
                            continue
                        monthly_datasets.append(ds_month)
    
                    if len(monthly_datasets) != len(season_months):
                        print(f"Incomplete season {season_year} for {c}.{v}.")
                        all_ok = False
                        continue
    
                    ds_season = _aggregate_months(monthly_datasets, v)
                    ds_season = ds_season.expand_dims(time=[_representative_time(season_year)])
                    seasonal_outputs.append(ds_season)
    
                if not seasonal_outputs:
                    print(f"No complete data found for {c}.{v}.")
                    continue
    
                if not all_ok:
                    print(f"Skipping save for {c}.{v} because some months/seasons failed.")
                    continue
    
                ds_out = xr.concat(seasonal_outputs, dim="time").sortby("time")
                ds_out = _finalize_output(ds_out, c, v)
    
                encoding = {name: {"zlib": True, "complevel": 4} for name in ds_out.data_vars}
                ds_out.to_netcdf(out_file, encoding=encoding)
                ds_out.close()
                file_path.append(out_file)
                print(f"Saved final reanalysis file: {out_file}")
    
                del seasonal_outputs, ds_out
                gc.collect()
    
            except Exception as exc:
                print(f"Failed to process {cv}: {exc}")
    
        return file_path



    # def WAS_Download_ERA5Land(
    #     self,
    #     dir_to_save,
    #     center_variable,
    #     year_start,
    #     year_end,
    #     area,
    #     seas=["01", "02", "03"],  # e.g. NDJ = ["11","12","01"]
    #     force_download=False,
    # ):
    #     """
    #     Download ERA5Land reanalysis data for specified variable combinations, years, and months,
    #     handling cross-year seasons (e.g., NDJ).
    #     Parameters:
    #         dir_to_save (str): Directory to save the downloaded files.
    #         center_variable (list): List of center-variable identifiers (e.g., ["ERA5Land.PRCP", "ERA5Land.TEMP"]).
    #         year_start (int): Start year for the data.
    #         year_end (int): End year for the data.
    #         area (list): Bounding box as [North, West, South, East].
    #         seas (list): List of months (e.g., ["11","12","01"] for NDJ).
    #         force_download (bool): If True, forces download even if file exists.
    #     """
    #     dir_to_save = Path(dir_to_save)
    #     dir_to_save.mkdir(parents=True, exist_ok=True)
    #     # Parse center and variable strings
    #     centers = [cv.split(".")[0] for cv in center_variable]
    #     vars_ = [cv.split(".")[1] for cv in center_variable]
    #     # ERA5-Land does not have pressure levels, only single-level variables
    #     variables_1 = {
    #         "PRCP": "total_precipitation",
    #         "TEMP": "2m_temperature",
    #         "TDEW": "2m_dewpoint_temperature",
    #         "UGRD10": "10m_u_component_of_wind",
    #         "VGRD10": "10m_v_component_of_wind",
    #         "DSWR": "surface_solar_radiation_downwards",
    #         "DLWR": "surface_thermal_radiation_downwards",
    #         "NOLR": "surface_net_thermal_radiation_downwards",  # Adjusted for ERA5-Land
    #         "RUNOFF": "surface_runoff",
    #         "SOILWATER1": "volumetric_soil_water_layer_1",
    #         "SOILWATER2": "volumetric_soil_water_layer_2",
    #         "SOILWATER3": "volumetric_soil_water_layer_3",            
    #     }
    #     # Helper for zero-padded month strings
    #     def m2str(m: int):
    #         return f"{m:02d}"
    #     # Convert months to integers
    #     season_months = [int(m) for m in seas]
    #     pivot = season_months[0]
    #     # For naming
    #     season_str = "".join([calendar.month_abbr[m] for m in season_months])
    #     for c, v in zip(centers, vars_):
    #         if c != "ERA5Land":
    #             print(f"This function is for ERA5Land only. Skipping {c}.")
    #             continue
    #         if v not in variables_1:
    #             print(f"Unknown variable for ERA5Land: {v}. Skipping.")
    #             continue
    #         cds_var = variables_1[v]
    #         # if cds_var == "surface_runoff":
    #         #     pivot_int = pivot
    #         #     previous = pivot_int - 1
    #         #     previous = str(previous).zfill(2) if previous > 0 else 12
    #         #     season_months = [previous] + season_months
    #         #     pivot = previous
            
    #         out_file = dir_to_save / f"{c}_{v}_{year_start}_{year_end}_{season_str}.nc"
    #         if not force_download and out_file.exists():
    #             print(f"{out_file} already exists. Skipping.")
    #             continue
    #         combined_datasets = []
    #         for year in range(year_start, year_end + 1):
    #             base_months = [m for m in season_months if m >= pivot]
    #             next_months = [m for m in season_months if m < pivot]
    #             # (A) Base-year
    #             if base_months:
    #                 base_str = [m2str(m) for m in base_months]
    #                 partA = dir_to_save / f"{c}_{v}_{year}_{season_str}_partA.nc"
    #                 dataset = "reanalysis-era5-land-monthly-means"
    #                 request = {
    #                     "product_type": "monthly_averaged_reanalysis",
    #                     "variable": cds_var,
    #                     "year": str(year),
    #                     "month": base_str,
    #                     "time": "00:00",
    #                     "area": area,
    #                     "data_format": "netcdf",
    #                 }
    #                 try:
    #                     client = cdsapi.Client()
    #                     print(f"Downloading {c}/{v}: {year}, months={base_str}")
    #                     client.retrieve(dataset, request).download(str(partA))
    #                 except Exception as e:
    #                     print(f"Download error for {c}/{v}, year={year} partA: {e}")
    #                     continue
    #                 with xr.open_dataset(partA) as dsA:
    #                     dsA = dsA.load()
    #                     dsA = self._postprocess_reanalysis(dsA, v)
    #                     combined_datasets.append(dsA)
    #                 os.remove(partA)
    #             # (B) Next-year
    #             if next_months and (year < year_end + 1):
    #                 year_next = year + 1
    #                 next_str = [m2str(m) for m in next_months]
    #                 partB = dir_to_save / f"{c}_{v}_{year}_{season_str}_partB_{year_next}.nc"
    #                 request["year"] = str(year_next)
    #                 request["month"] = next_str
    #                 try:
    #                     client = cdsapi.Client()
    #                     print(f"Downloading {c}/{v}: {year_next}, months={next_str}")
    #                     client.retrieve(dataset, request).download(str(partB))
    #                 except Exception as e:
    #                     print(f"Download error for {c}/{v}, year={year_next} partB: {e}")
    #                     continue
    #                 with xr.open_dataset(partB) as dsB:
    #                     dsB = dsB.load()
    #                     dsB = self._postprocess_reanalysis(dsB, v)
    #                     combined_datasets.append(dsB)
    #                 os.remove(partB)
    #         if combined_datasets:
    #             dsC = xr.concat(combined_datasets, dim="time")
    #             # Unit conversions
    #             if v in ["TEMP", "TDEW"]:
    #                 dsC = dsC - 273.15
    #             # Aggregate
    #             dsC = self._aggregate_crossyear(
    #                 ds=dsC,
    #                 season_months=season_months,
    #                 var_name=v
    #             )
    #             if v == "PRCP":
    #                 dsC = 1000 * dsC * 30  # Convert to mm 
    #             if v == "RUNOFF":
    #                 #dsC = dsC * len(season_months) * 30 # Convert to mm (approximate, as in existing code)
    #                 lat = dsC.Y
    #                 lon = dsC.X
    #                 dlon = np.deg2rad(0.1)
    #                 dlat = np.deg2rad(0.1)
    #                 r = 6371000 # Earth radius in meters
    #                 # Area for each grid cell
    #                 area_ = (r ** 2) * dlon * np.cos(np.deg2rad(lat)) * dlat
    #                 # Perform the conversion to m^3/s
    #                 dsC = (dsC * area_) / 86400
                    
    #             if v in ["DSWR", "DLWR", "NOLR"]:
    #                 # nbjour = len(season_months) * 30
    #                 dsC = dsC / 86400  # Convert to W/m2 (approximate)
    #             dsC["time"] = [f"{year}-{seas[1]}-01" for year in dsC["time"].astype(str).values]
    #             dsC["time"] = dsC["time"].astype("datetime64[ns]")
    #             dsC = dsC.rename({"time": "T"})
    #             # Save final
    #             dsC.to_netcdf(out_file)
    #             print(f"Saved final ERA5-Land file: {out_file}")
    #         else:
    #             print(f"No data found for {c}/{v}.")

    def WAS_Download_ERA5Land(
        self,
        dir_to_save,
        center_variable,
        year_start,
        year_end,
        area,
        seas=("01", "02", "03"),
        force_download=False,
        runoff_units="mm",
        max_retries=3,
        retry_delay=5,
    ):
        """
        Download ERA5-Land monthly averaged data and aggregate it by season.
    
        Output convention:
            T, Y, X
    
        ERA5-Land monthly averaged accumulation convention:
            PRCP/RUNOFF are in m/day. Monthly totals are:
                value * number_of_days * 1000
    
            Radiation accumulations are daily mean accumulations in J m-2/day.
            Mean fluxes are:
                value / 86400
    
        Cross-year seasons are handled using the first season month as pivot.
        Example:
            seas=("11", "12", "01") for NDJ gives Nov-Dec of year Y and Jan of Y+1.
        """
        from pathlib import Path
        import calendar
        import gc
        import os
        import time as _time
    
        import cdsapi
        import numpy as np
        import pandas as pd
        import xarray as xr
    
        if isinstance(center_variable, str):
            raise TypeError(
                "center_variable must be a list, for example "
                "['ERA5Land.PRCP'] or ['ERA5Land.PRCP', 'ERA5Land.TEMP']."
            )
        center_variable = list(center_variable)
    
        if runoff_units not in ["mm", "m3/s"]:
            raise ValueError("runoff_units must be 'mm' or 'm3/s'.")
    
        dir_to_save = Path(dir_to_save)
        dir_to_save.mkdir(parents=True, exist_ok=True)
    
        variables_1 = {
            "PRCP": "total_precipitation",
            "RUNOFF": "runoff",
            "TEMP": "2m_temperature",
            "TDEW": "2m_dewpoint_temperature",
            "UGRD10": "10m_u_component_of_wind",
            "VGRD10": "10m_v_component_of_wind",
            "DSWR": "surface_solar_radiation_downwards",
            "DLWR": "surface_thermal_radiation_downwards",
            "NOLR": "surface_net_thermal_radiation",
            "SOILWATER1": "volumetric_soil_water_layer_1",
            "SOILWATER2": "volumetric_soil_water_layer_2",
            "SOILWATER3": "volumetric_soil_water_layer_3",
            "SOILWATER4": "volumetric_soil_water_layer_4",
        }
    
        total_vars = {"PRCP", "SRUNOFF"}
        flux_vars = {"DSWR", "DLWR", "NOLR"}
        kelvin_vars = {"TEMP", "TDEW"}
    
        season_months = [int(m) for m in seas]
        season_codes = [f"{m:02d}" for m in season_months]
        pivot_month = season_months[0]
        season_str = "".join(calendar.month_abbr[m] for m in season_months)
    
        dataset = "reanalysis-era5-land-monthly-means"
        client = cdsapi.Client()
    
        def _season_calendar_months(season_year):
            months = []
            for month in season_months:
                cal_year = season_year if month >= pivot_month else season_year + 1
                months.append((cal_year, month))
            return months
    
        def _representative_time(season_year):
            mid_month = season_months[len(season_months) // 2]
            cal_year = season_year if mid_month >= pivot_month else season_year + 1
            return pd.Timestamp(cal_year, mid_month, 1)
    
        def _download_with_retries(request, target_file, label):
            for attempt in range(1, max_retries + 1):
                try:
                    print(f"Attempt {attempt}/{max_retries}: downloading {label}...")
                    client.retrieve(dataset, request).download(str(target_file))
                    print(f"Downloaded: {target_file}")
                    return True
                except Exception as exc:
                    print(f"Attempt {attempt}/{max_retries} failed for {label}: {exc}")
                    if target_file.exists():
                        os.remove(target_file)
                        print(f"Deleted incomplete file: {target_file}")
                    if attempt < max_retries:
                        print(f"Retrying after {retry_delay} seconds...")
                        _time.sleep(retry_delay)
            return False
    
        def _normalize_coords(ds):
            rename_map = {}
            if "valid_time" in ds.dims or "valid_time" in ds.coords:
                rename_map["valid_time"] = "time"
            if "longitude" in ds.dims or "longitude" in ds.coords:
                rename_map["longitude"] = "X"
            if "latitude" in ds.dims or "latitude" in ds.coords:
                rename_map["latitude"] = "Y"
    
            ds = ds.rename({old: new for old, new in rename_map.items() if old in ds})
    
            if "expver" in ds.dims and ds.sizes["expver"] == 1:
                ds = ds.isel(expver=0, drop=True)
    
            if "Y" in ds.coords:
                ds = ds.sortby("Y")
    
            return ds
    
        def _rename_single_data_var(ds, var_code):
            data_vars = list(ds.data_vars)
            if len(data_vars) == 1 and data_vars[0] != var_code:
                ds = ds.rename({data_vars[0]: var_code})
            return ds
    
        def _grid_cell_area_m2(ds):
            if "Y" not in ds.coords or "X" not in ds.coords:
                raise ValueError("SRUNOFF conversion to m3/s needs X and Y coordinates.")
            if ds.sizes.get("Y", 0) < 2 or ds.sizes.get("X", 0) < 2:
                raise ValueError("SRUNOFF conversion to m3/s needs at least two X and Y points.")
    
            radius = 6371000.0
            dlat = np.deg2rad(float(abs(ds["Y"].diff("Y").median())))
            dlon = np.deg2rad(float(abs(ds["X"].diff("X").median())))
            area_y = (radius ** 2) * dlat * dlon * np.cos(np.deg2rad(ds["Y"]))
            area_y.attrs["units"] = "m2"
            return area_y
    
        def _convert_month(ds, var_code, days_in_month):
            ds = _rename_single_data_var(ds, var_code)
    
            if var_code == "PRCP":
                ds = ds * (1000.0 * days_in_month)
                units = "mm/month"
    
            elif var_code in ["SRUNOFF", "RUNOFF"]:
                if runoff_units == "mm":
                    ds = ds * (1000.0 * days_in_month)
                    units = "mm/month"
                else:
                    # m/day * days * area gives monthly volume; seasonal aggregation
                    # divides by total seconds later.
                    ds = ds * days_in_month * _grid_cell_area_m2(ds)
                    units = "m3/month"
    
            elif var_code in flux_vars:
                ds = ds / 86400.0
                units = "W m-2"
    
            elif var_code in kelvin_vars:
                ds = ds - 273.15
                units = "degC"
    
            else:
                units = None
    
            ds.attrs["monthly_days"] = int(days_in_month)
            for name in ds.data_vars:
                if units is not None:
                    ds[name].attrs["units"] = units
                ds[name].attrs["monthly_days"] = int(days_in_month)
    
            return ds
    
        def _aggregate_months(monthly_datasets, var_code):
            ds = xr.concat(monthly_datasets, dim="month_index")
            monthly_days = [
                int(item.attrs.get("monthly_days", next(iter(item.data_vars.values())).attrs["monthly_days"]))
                for item in monthly_datasets
            ]
            weights = xr.DataArray(monthly_days, dims="month_index")
            total_days = float(sum(monthly_days))
    
            if var_code == "PRCP":
                out = ds.sum("month_index", keep_attrs=True)
                units = "mm"
    
            elif var_code == "SRUNOFF":
                if runoff_units == "mm":
                    out = ds.sum("month_index", keep_attrs=True)
                    units = "mm"
                else:
                    out = ds.sum("month_index", keep_attrs=True) / (total_days * 86400.0)
                    units = "m3 s-1"
    
            elif var_code in flux_vars:
                out = ds.weighted(weights).mean("month_index", keep_attrs=True)
                units = "W m-2"
    
            else:
                out = ds.weighted(weights).mean("month_index", keep_attrs=True)
                units = None
    
            for name in out.data_vars:
                if units is not None:
                    out[name].attrs["units"] = units
                out[name].attrs.pop("monthly_days", None)
    
            return out
    
        def _download_month(var_code, cds_var, season_year, cal_year, month):
            tmp = dir_to_save / f"tmp_ERA5Land_{var_code}_{season_year}_{cal_year}{month:02d}.nc"
            request = {
                "product_type": ["monthly_averaged_reanalysis"],
                "variable": [cds_var],
                "year": str(cal_year),
                "month": f"{month:02d}",
                "time": ["00:00"],
                "area": area,
                "data_format": "netcdf",
            }
    
            label = f"ERA5Land.{var_code} {cal_year}-{month:02d}"
            if not _download_with_retries(request, tmp, label):
                return None
    
            try:
                with xr.open_dataset(tmp) as ds:
                    ds = _normalize_coords(ds).load()
                days = calendar.monthrange(cal_year, month)[1]
                return _convert_month(ds, var_code, days)
            finally:
                if tmp.exists():
                    os.remove(tmp)
                    print(f"Deleted temporary file: {tmp}")
    
        def _finalize_output(ds, var_code):
            if "time" in ds.dims or "time" in ds.coords:
                ds = ds.rename({"time": "T"})
    
            preferred = ["T", "Y", "X"]
            for name in list(ds.data_vars):
                dims = [dim for dim in preferred if dim in ds[name].dims]
                dims += [dim for dim in ds[name].dims if dim not in dims]
                ds[name] = ds[name].transpose(*dims)
    
            ds.attrs["center"] = "ERA5Land"
            ds.attrs["variable_code"] = var_code
            ds.attrs["season"] = season_str
            ds.attrs["season_months"] = ",".join(season_codes)
            if var_code == "SRUNOFF":
                ds.attrs["runoff_units"] = runoff_units
            return ds
    
        file_path = []
    
        for cv in center_variable:
            try:
                center, var_code = cv.split(".", 1)
                if center != "ERA5Land":
                    print(f"This function is for ERA5Land only. Skipping {center}.")
                    continue
                if var_code not in variables_1:
                    print(f"Unknown variable for ERA5Land: {var_code}. Skipping.")
                    continue
    
                unit_tag = "_m3s" if var_code == "SRUNOFF" and runoff_units == "m3/s" else ""
                out_file = dir_to_save / f"ERA5Land_{var_code}_{year_start}_{year_end}_{season_str}{unit_tag}.nc"
                if out_file.exists() and not force_download:
                    print(f"{out_file} already exists. Skipping.")
                    file_path.append(out_file)
                    continue
    
                seasonal_outputs = []
                all_ok = True
                cds_var = variables_1[var_code]
    
                for season_year in range(year_start, year_end + 1):
                    monthly_datasets = []
                    for cal_year, month in _season_calendar_months(season_year):
                        ds_month = _download_month(var_code, cds_var, season_year, cal_year, month)
                        if ds_month is None:
                            all_ok = False
                            continue
                        monthly_datasets.append(ds_month)
    
                    if len(monthly_datasets) != len(season_months):
                        print(f"Incomplete season {season_year} for ERA5Land.{var_code}.")
                        all_ok = False
                        continue
    
                    ds_season = _aggregate_months(monthly_datasets, var_code)
                    ds_season = ds_season.expand_dims(time=[_representative_time(season_year)])
                    seasonal_outputs.append(ds_season)
    
                if not seasonal_outputs:
                    print(f"No complete data found for ERA5Land.{var_code}.")
                    continue
    
                if not all_ok:
                    print(f"Skipping save for ERA5Land.{var_code} because some months/seasons failed.")
                    continue
    
                ds_out = xr.concat(seasonal_outputs, dim="time").sortby("time")
                ds_out = _finalize_output(ds_out, var_code)
    
                encoding = {name: {"zlib": True, "complevel": 4} for name in ds_out.data_vars}
                ds_out.to_netcdf(out_file, encoding=encoding)
                ds_out.close()
                file_path.append(out_file)
                print(f"Saved final ERA5-Land file: {out_file}")
    
                del seasonal_outputs, ds_out
                gc.collect()
    
            except Exception as exc:
                print(f"Failed to process {cv}: {exc}")
    
        return file_path


    def WAS_Download_ERA5Land_daily(
        self,
        dir_to_save,
        center_variable,
        year_start,
        year_end,
        area,
        force_download=False,
        max_retries=3,
        retry_delay=5,
        runoff_units="m3/s",
    ):
        """
        Download ERA5-Land daily data from CDS and save WAS-style NetCDF files.
    
        Output convention:
            T, Y, X
    
        ERA5-Land convention for accumulated variables:
            total precipitation, runoff and radiation at 00 UTC represent the
            accumulation over the previous day. Therefore the daily value for
            YYYY-MM-DD is taken from YYYY-MM-DD + 1 day at 00 UTC.
    
        Parameters
        ----------
        dir_to_save : str or Path
            Output directory.
        center_variable : list of str
            Variables to download, for example ["ERA5Land.PRCP", "ERA5Land.TEMP"].
        year_start, year_end : int
            Inclusive year range.
        area : list
            CDS area as [North, West, South, East].
        force_download : bool
            Redownload even when the output file exists.
        max_retries, retry_delay : int
            Download retry controls.
        runoff_units : {"mm", "m3/s"}
            Output units for RUNOFF. Other variables keep their natural daily units:
            PRCP in mm/day, radiation in W m-2, temperature in degC.
        """
        from pathlib import Path
        import calendar
        import gc
        import os
        import time as _time
    
        import cdsapi
        import numpy as np
        import xarray as xr
    
        if isinstance(center_variable, str):
            raise TypeError(
                "center_variable must be a list, for example "
                "['ERA5Land.PRCP'] or ['ERA5Land.PRCP', 'ERA5Land.RUNOFF']."
            )
        center_variable = list(center_variable)
    
        if runoff_units not in ["mm", "m3/s"]:
            raise ValueError("runoff_units must be 'mm' or 'm3/s'.")
    
        dir_to_save = Path(dir_to_save)
        dir_to_save.mkdir(parents=True, exist_ok=True)
    
        variables_1 = {
            "PRCP": "total_precipitation",
            "TEMP": "2m_temperature",
            "TDEW": "2m_dewpoint_temperature",
            "UGRD10": "10m_u_component_of_wind",
            "VGRD10": "10m_v_component_of_wind",
            "DSWR": "surface_solar_radiation_downwards",
            "DLWR": "surface_thermal_radiation_downwards",
            "NOLR": "surface_net_thermal_radiation",
            "SRUNOFF": "surface_runoff",
            "RUNOFF": "runoff",
        }
        accumulated_vars = {"PRCP", "SRUNOFF", "DSWR", "DLWR", "NOLR"}
        temperature_vars = {"TEMP", "TDEW"}
    
        dataset = "reanalysis-era5-land"
        client = cdsapi.Client()
    
        def _days_in_month(year, month):
            last_day = calendar.monthrange(year, month)[1]
            return [f"{day:02d}" for day in range(1, last_day + 1)]
    
        def _next_year_month(year, month):
            if month == 12:
                return year + 1, 1
            return year, month + 1
    
        def _download_request(target_file, variable, year, month, days, hours):
            request = {
                "variable": [variable],
                "year": str(year),
                "month": f"{month:02d}",
                "day": days,
                "time": hours,
                "area": area,
                "data_format": "netcdf",
                "download_format": "unarchived",
            }
    
            success = False
            for attempt in range(1, max_retries + 1):
                try:
                    print(
                        f"Attempt {attempt}/{max_retries}: downloading "
                        f"{variable} for {year}-{month:02d}..."
                    )
                    client.retrieve(dataset, request).download(str(target_file))
                    print(f"Downloaded: {target_file}")
                    success = True
                    break
                except Exception as exc:
                    print(
                        f"Attempt {attempt}/{max_retries} failed for "
                        f"{variable} {year}-{month:02d}: {exc}"
                    )
                    if target_file.exists():
                        os.remove(target_file)
                        print(f"Deleted incomplete file: {target_file}")
                    if attempt < max_retries:
                        print(f"Retrying after {retry_delay} seconds...")
                        _time.sleep(retry_delay)
    
            return success
    
        def _normalize_coords(ds):
            rename_map = {}
            if "valid_time" in ds.dims or "valid_time" in ds.coords:
                rename_map["valid_time"] = "time"
            if "longitude" in ds.dims or "longitude" in ds.coords:
                rename_map["longitude"] = "X"
            if "latitude" in ds.dims or "latitude" in ds.coords:
                rename_map["latitude"] = "Y"
            ds = ds.rename({old: new for old, new in rename_map.items() if old in ds})
    
            if "expver" in ds.dims and ds.sizes["expver"] == 1:
                ds = ds.isel(expver=0, drop=True)
    
            if "Y" in ds.coords:
                ds = ds.sortby("Y")
    
            return ds
    
        def _rename_data_variable(ds, var_code):
            data_vars = list(ds.data_vars)
            if len(data_vars) == 1 and data_vars[0] != var_code:
                ds = ds.rename({data_vars[0]: var_code})
            return ds
    
        def _grid_cell_area_m2(ds):
            if "Y" not in ds.coords or "X" not in ds.coords:
                raise ValueError("SRUNOFF conversion to m3/s needs X and Y coordinates.")
            if ds.sizes.get("Y", 0) < 2 or ds.sizes.get("X", 0) < 2:
                raise ValueError("SRUNOFF conversion to m3/s needs at least two X and Y points.")
    
            radius = 6371000.0
            dlat = np.deg2rad(float(abs(ds["Y"].diff("Y").median())))
            dlon = np.deg2rad(float(abs(ds["X"].diff("X").median())))
            area_y = (radius ** 2) * dlat * dlon * np.cos(np.deg2rad(ds["Y"]))
            area_y.attrs["units"] = "m2"
            return area_y
    
        def _open_many(paths):
            datasets = []
            for path in paths:
                ds = xr.open_dataset(path)
                datasets.append(_normalize_coords(ds))
            combined = xr.concat(datasets, dim="time").sortby("time").load()
            for ds in datasets:
                ds.close()
            return combined
    
        def _process_accumulated_month(v, paths, year, month):
            ds = _open_many(paths)
            ds = _rename_data_variable(ds, v)
    
            # 00 UTC belongs to the previous day for ERA5-Land accumulated fields.
            ds = ds.sel(time=ds["time"].dt.hour == 0)
            ds = ds.assign_coords(time=ds["time"] - np.timedelta64(1, "D"))
    
            start = np.datetime64(f"{year}-{month:02d}-01")
            end_day = calendar.monthrange(year, month)[1]
            end = np.datetime64(f"{year}-{month:02d}-{end_day:02d}")
            ds = ds.sel(time=slice(start, end))
    
            if v == "PRCP":
                ds = (ds * 1000.0).where(ds >= 0, 0)
                for name in ds.data_vars:
                    ds[name].attrs["units"] = "mm/day"
    
            elif v in ["RUNOFF", "SRUNOFF"]:
                ds = ds.where(ds >= 0, 0)
                if runoff_units == "mm":
                    ds = ds * 1000.0
                    for name in ds.data_vars:
                        ds[name].attrs["units"] = "mm/day"
                else:
                    ds = (ds * _grid_cell_area_m2(ds)) / 86400.0
                    for name in ds.data_vars:
                        ds[name].attrs["units"] = "m3 s-1"
    
            elif v in ["DSWR", "DLWR", "NOLR"]:
                ds = ds / 86400.0
                for name in ds.data_vars:
                    ds[name].attrs["units"] = "W m-2"
    
            return ds
    
        def _process_instantaneous_month(v, path):
            ds = xr.open_dataset(path)
            ds = _normalize_coords(ds)
            ds = _rename_data_variable(ds, v)
    
            ds_daily = ds.resample(time="1D").mean(keep_attrs=True).load()
            ds.close()
    
            if v in temperature_vars:
                ds_daily = ds_daily - 273.15
                for name in ds_daily.data_vars:
                    ds_daily[name].attrs["units"] = "degC"
    
            return ds_daily
    
        def _finalize(ds, v):
            if "time" in ds.dims or "time" in ds.coords:
                ds = ds.rename({"time": "T"})
    
            preferred = ["T", "Y", "X"]
            for name in list(ds.data_vars):
                dims = [dim for dim in preferred if dim in ds[name].dims]
                dims += [dim for dim in ds[name].dims if dim not in dims]
                ds[name] = ds[name].transpose(*dims)
    
            ds.attrs["source_dataset"] = dataset
            ds.attrs["center"] = "ERA5Land"
            ds.attrs["variable_code"] = v
            if v in ["SRUNOFF", "RUNOFF"]:
                ds.attrs["runoff_units"] = runoff_units
            return ds
    
        store_file_path = {}
    
        for cv in center_variable:
            try:
                center, v = cv.split(".", 1)
                if center != "ERA5Land":
                    print(f"Invalid center for this function: {center}. Skipping.")
                    continue
                if v not in variables_1:
                    print(f"Unknown variable: {v}. Skipping.")
                    continue
    
                cds_variable = variables_1[v]
                unit_tag = "_m3s" if v in ["SRUNOFF", "RUNOFF"] and runoff_units == "m3/s" else ""
                output_path = dir_to_save / f"Daily_{v}_{year_start}_{year_end}{unit_tag}.nc"
    
                if output_path.exists() and not force_download:
                    print(f"{output_path} already exists. Skipping download.")
                    store_file_path[cv] = output_path
                    continue
    
                combined_datasets = []
                all_downloads_ok = True
    
                for year in range(year_start, year_end + 1):
                    for month in range(1, 13):
                        temp_files = []
                        try:
                            if v in accumulated_vars:
                                last_day = calendar.monthrange(year, month)[1]
                                current_days = [f"{day:02d}" for day in range(2, last_day + 1)]
                                current_file = dir_to_save / f"era5land_00_{v}_{year}_{month:02d}.nc"
    
                                if current_days:
                                    ok = _download_request(
                                        current_file,
                                        cds_variable,
                                        year,
                                        month,
                                        current_days,
                                        ["00:00"],
                                    )
                                    if not ok:
                                        all_downloads_ok = False
                                        continue
                                    temp_files.append(current_file)
    
                                next_year, next_month = _next_year_month(year, month)
                                boundary_file = (
                                    dir_to_save
                                    / f"era5land_00_{v}_{next_year}_{next_month:02d}_day01.nc"
                                )
                                ok = _download_request(
                                    boundary_file,
                                    cds_variable,
                                    next_year,
                                    next_month,
                                    ["01"],
                                    ["00:00"],
                                )
                                if not ok:
                                    all_downloads_ok = False
                                    continue
                                temp_files.append(boundary_file)
    
                                ds_daily = _process_accumulated_month(v, temp_files, year, month)
    
                            else:
                                hourly_file = dir_to_save / f"era5land_hourly_{v}_{year}_{month:02d}.nc"
                                ok = _download_request(
                                    hourly_file,
                                    cds_variable,
                                    year,
                                    month,
                                    _days_in_month(year, month),
                                    [f"{hour:02d}:00" for hour in range(24)],
                                )
                                if not ok:
                                    all_downloads_ok = False
                                    continue
                                temp_files.append(hourly_file)
    
                                ds_daily = _process_instantaneous_month(v, hourly_file)
    
                            combined_datasets.append(ds_daily)
    
                        except Exception as exc:
                            print(f"Failed to process ERA5Land.{v} for {year}-{month:02d}: {exc}")
                            all_downloads_ok = False
    
                        finally:
                            for path in temp_files:
                                if path.exists():
                                    os.remove(path)
                                    print(f"Deleted temporary file: {path}")
                            gc.collect()
    
                if not combined_datasets:
                    print(f"No data processed for ERA5Land.{v}. Skipping save.")
                    continue
    
                if not all_downloads_ok:
                    print(f"Skipping save for ERA5Land.{v} due to incomplete downloads.")
                    continue
    
                combined_ds = xr.concat(combined_datasets, dim="time").sortby("time")
                combined_ds = _finalize(combined_ds, v)
    
                encoding = {name: {"zlib": True, "complevel": 4} for name in combined_ds.data_vars}
                combined_ds.to_netcdf(output_path, encoding=encoding)
                combined_ds.close()
                store_file_path[cv] = output_path
                print(f"Combined dataset for ERA5Land.{v} saved to {output_path}")
    
                del combined_ds, combined_datasets
                gc.collect()
    
            except Exception as exc:
                print(f"Failed ERA5Land download for {cv}: {exc}")
    
        return store_file_path
    
    # def WAS_Download_ERA5Land_daily(
    #         self,
    #         dir_to_save,
    #         center_variable,
    #         year_start,
    #         year_end,
    #         area,
    #         force_download=False,
    #         max_retries=3,
    #         retry_delay=5,
    #     ):
    #     """
    #     Download daily ERA5Land reanalysis data from the Copernicus Data Store (CDS)
    #     for specified variables and years, with retries for failed downloads.

    #     Parameters
    #     ----------
    #     dir_to_save : str or pathlib.Path
    #         Directory path where the downloaded NetCDF files will be saved.
    #         The directory will be created if it does not exist.
    #     center_variable : list of str
    #         List of center-variable identifiers (e.g., ["ERA5Land.PRCP", "ERA5Land.TEMP"]).
    #     year_start : int
    #         Start year for the data to download (inclusive).
    #     year_end : int
    #         End year for the data to download (inclusive).
    #     area : list of float
    #         Bounding box for spatial subsetting in the format [North, West, South, East].
    #     force_download : bool, optional
    #         If True, forces download even if the output file exists. Default is False.
    #     max_retries : int, optional
    #         Maximum number of retry attempts for failed downloads. Default is 3.
    #     retry_delay : int, optional
    #         Seconds to wait between retry attempts. Default is 5.

    #     Returns
    #     -------
    #     None
    #         The function saves NetCDF files to `dir_to_save` but does not return a value.
    #         Output files are named as `Daily_<variable>_<year_start>_<year_end>.nc`.

    #     Notes
    #     -----
    #     - The function downloads hourly data from the CDS dataset "reanalysis-era5-land"
    #       and aggregates it to daily resolution.
    #     - Aggregation: sum for accumulative variables (e.g., PRCP, radiation), mean for others (e.g., TEMP).
    #     - Unit conversions: PRCP to mm/day, TEMP to °C, radiation to W/m² (daily average).
    #     - Coordinates are renamed to "X" (longitude), "Y" (latitude), and "T" (time),
    #       with latitude flipped.
    #     - Downloads are performed month-by-month due to API limitations.
    #     - Requires a valid CDS API key configured in `~/.cdsapirc`.
    #     """
    #     dir_to_save = Path(dir_to_save)
    #     dir_to_save.mkdir(parents=True, exist_ok=True)
    #     days = [f"{day:02}" for day in range(1, 32)]
    #     months = [f"{month:02}" for month in range(1, 13)]
    #     hours = [f"{h:02}:00" for h in range(24)]
    #     variables_1 = {
    #         "PRCP": "total_precipitation",
    #         "TEMP": "temperature_2m",
    #         "TDEW": "dewpoint_temperature_2m",
    #         "UGRD10": "u_component_of_wind_10m",
    #         "VGRD10": "v_component_of_wind_10m",
    #         "DSWR": "surface_solar_radiation_downwards",
    #         "DLWR": "surface_thermal_radiation_downwards",
    #         "NOLR": "surface_net_thermal_radiation_downwards",
    #         "RUNOFF": "surface_runoff",
    #     }
    #     centers = [cv.split(".")[0] for cv in center_variable]
    #     vars_short = [cv.split(".")[1] for cv in center_variable]
    #     for c, v in zip(centers, vars_short):
    #         if c != "ERA5Land":
    #             print(f"Invalid center for this function: {c}. Must be 'ERA5Land'. Skipping.")
    #             continue
    #         if v not in variables_1:
    #             print(f"Unknown variable: {v}. Skipping.")
    #             continue
    #         cds_variable = variables_1[v]
    #         output_path = dir_to_save / f"Daily_{v}_{year_start}_{year_end}.nc"
    #         if not force_download and output_path.exists():
    #             print(f"{output_path} already exists. Skipping download.")
    #             continue
    #         combined_datasets = []
    #         all_years_downloaded = True
    #         for year in range(year_start, year_end + 1):
    #             for month in range(1, 13):
    #                 nc_file_path = dir_to_save / f"hourly_{v}_{year}_{month:02d}.nc"
    #                 success = False
    #                 retries = 0
    #                 while retries < max_retries and not success:
    #                     try:
    #                         client = cdsapi.Client()
    #                         dataset = "reanalysis-era5-land"
    #                         request = {
    #                             "variable": cds_variable,
    #                             "year": str(year),
    #                             "month": f"{month:02}",
    #                             "day": days,
    #                             "time": hours,
    #                             "area": area,
    #                             "data_format": "netcdf",
    #                             "download_format": "unarchived"
    #                         }
    #                         print(f"Attempt {retries + 1}/{max_retries}: Downloading {cds_variable} data for {year}-{month:02d}...")
    #                         client.retrieve(dataset, request).download(str(nc_file_path))
    #                         print(f"Downloaded: {nc_file_path}")
    #                         success = True
    #                     except Exception as e:
    #                         retries += 1
    #                         print(f"Attempt {retries}/{max_retries} failed for {cds_variable} data for {year}-{month:02d}: {e}")
    #                         if retries < max_retries:
    #                             print(f"Retrying after {retry_delay} seconds...")
    #                             _time.sleep(retry_delay)
    #                         if nc_file_path.exists():
    #                             os.remove(nc_file_path)
    #                             print(f"Deleted incomplete file: {nc_file_path}")
    #                 if not success:
    #                     print(f"Failed to download {cds_variable} data for {year}-{month:02d} after {max_retries} attempts.")
    #                     all_years_downloaded = False
    #                     continue
    #                 try:
    #                     ds_month = xr.open_dataset(nc_file_path)
    #                     ds_daily = self._postprocess_reanalysis(ds_month, v)
    #                     if v == "PRCP":
    #                         ds_daily = ds_daily.sel(time=ds_daily.time.dt.hour == 0) * 1000
    #                         ds_daily_0 = ds_daily.isel(time=0)
    #                         ds_daily_0.coords['time'] = pd.Timestamp(f"{year}-{month}-{calendar.monthrange(year, month)[1]}")
    #                         ds_daily = ds_daily.shift(time=-1).dropna(dim="time")
    #                         ds_daily = xr.concat([ds_daily_0, ds_daily], dim="time")
    #                     if v == "RUNOFF":
    #                         ds_daily = ds_daily.sel(time=ds_daily.time.dt.hour == 0)
    #                         ds_daily_0 = ds_daily.isel(time=0)
    #                         ds_daily_0.coords['time'] = pd.Timestamp(f"{year}-{month}-{calendar.monthrange(year, month)[1]}")
    #                         ds_daily = ds_daily.shift(time=-1).dropna(dim="time")
    #                         ds_daily = xr.concat([ds_daily_0, ds_daily], dim="time")
    #                         lat = ds_daily.Y
    #                         lon = ds_daily.X
    #                         dlon = np.deg2rad(0.1)
    #                         dlat = np.deg2rad(0.1)
    #                         r = 6371000 # Earth radius in meters
    #                         # Area for each grid cell
    #                         area_= (r ** 2) * dlon * np.cos(np.deg2rad(lat)) * dlat
    #                         # Perform the conversion to m^3/s
    #                         ds_daily = (ds_daily * area_) / 86400
    #                     if v in ["DSWR", "DLWR", "NOLR"]:
    #                         ds_daily = ds_daily.sel(time=ds_daily.time.dt.hour == 0) / 86400
    #                         ds_daily_0 = ds_daily.isel(time=0)
    #                         ds_daily_0.coords['time'] = pd.Timestamp(f"{year}-{month}-{calendar.monthrange(year, month)[1]}")
    #                         ds_daily = ds_daily.shift(time=-1).dropna(dim="time")
    #                         ds_daily = xr.concat([ds_daily_0, ds_daily], dim="time")
    #                     if v in ["TEMP", "TDEW"]:
    #                         ds_daily = ds_daily - 273.15  # K to °C
    #                     combined_datasets.append(ds_daily)
    #                 except Exception as e:
    #                     print(f"Failed to process {nc_file_path}: {e}")
    #                     all_years_downloaded = False
    #                 if nc_file_path.exists():
    #                     os.remove(nc_file_path)
    #                     print(f"Deleted hourly file: {nc_file_path}")
    #         if combined_datasets and all_years_downloaded:
    #             try:
    #                 combined_ds = xr.concat(combined_datasets, dim="time")
    #                 combined_ds = combined_ds.rename({"time": "T"})
    #                 combined_ds.to_netcdf(output_path)
    #                 combined_ds.close()
    #                 print(f"Combined dataset for ERA5Land.{v} saved to {output_path}")
    #             except Exception as e:
    #                 print(f"Failed to process or save combined dataset for ERA5Land.{v}: {e}")
    #         else:
    #             print(f"Skipping save for ERA5Land.{v} due to incomplete downloads.")

#     def WAS_Download_CHIRPSv3_Daily(
#             self,
#             dir_to_save,
#             year_start,
#             year_end,
#             blend_type="ERA5",
#             area=None,
#             force_download=False,
#             max_retries=3,
#             retry_delay=5,
#         ):
#         """
#         Download daily CHIRPS v3.0 precipitation (blended with ERA5 or IMERGlate-v07) from the Copernicus Data Store (CDS)
#         for specified years, with retries for failed downloads.

#         Parameters
#         ----------
#         dir_to_save : str or pathlib.Path
#             Directory path where the downloaded NetCDF files will be saved.
#             The directory will be created if it does not exist.
#         year_start : int
#             Start year for the data to download (inclusive).
#         year_end : int
#             End year for the data to download (inclusive).
#         blend_type : str
#             Blend type, either "ERA5" or "IMERGlate-v07" (default: "ERA5").
#             Note: IMERGlate-v07 availability starts from ~2000; earlier years may fail.
#         area : list of float
#             Bounding box for spatial subsetting in the format [North, West, South, East].
#         force_download : bool, optional
#             If True, forces download even if the output file exists. Default is False.
#         max_retries : int, optional
#             Maximum number of retry attempts for failed downloads. Default is 3.
#         retry_delay : int, optional
#             Seconds to wait between retry attempts. Default is 5.

#         Returns
#         -------
#         None
#             The function saves NetCDF files to `dir_to_save` but does not return a value.
#             Output files are named as `Daily_PRCP_{blend_type}_{year_start}_{year_end}.nc`.

#         Notes
#         -----
#         - Downloads individual daily TIFF files, processes them with rioxarray, clips if area specified,
#           and combines into a single NetCDF with daily time dimension.
#         - Units: precipitation in mm/day.
#         - Coordinates renamed to "X" (longitude), "Y" (latitude), "T" (time), with Y flipped if needed.
#         - Deletes temporary TIFF files after processing.
#         - Skips invalid dates (e.g., Feb 30) automatically.
#         """
#         dir_to_save = Path(dir_to_save)
#         dir_to_save.mkdir(parents=True, exist_ok=True)

#         output_path = dir_to_save / f"Daily_PRCP_{year_start}_{year_end}.nc"
#         if not force_download and output_path.exists():
#             print(f"{output_path} already exists. Skipping download.")
#             return output_path

#         else:
            
#             combined_datasets = []
#             all_years_downloaded = True
#             for year in range(year_start, year_end + 1):
#                 for month in range(1, 13):
#                     ndays = calendar.monthrange(year, month)[1]
#                     for day in range(1, ndays + 1):
#                         tif_file_path = dir_to_save / f"chirps-v3.0.{year}.{month:02d}.{day:02d}.tif"
#                         success = False
#                         retries = 0
#                         while retries < max_retries and not success:
#                             try:
#                                 da = self._fetch_chirps_daily(
#                                     year=year,
#                                     month=month,
#                                     day=day,
#                                     dir_to_save=dir_to_save,
#                                     blend_type=blend_type,
#                                     force_download=force_download,
#                                     area=area
#                                 )
#                                 if da is not None:
#                                     combined_datasets.append(da)
#                                     success = True
#                             except Exception as e:
#                                 retries += 1
#                                 print(f"Attempt {retries}/{max_retries} failed for {year}-{month:02d}-{day:02d}: {e}")
#                                 if retries < max_retries:
#                                     print(f"Retrying after {retry_delay} seconds...")
#                                     _time.sleep(retry_delay)
#                                 if tif_file_path.exists():
#                                     os.remove(tif_file_path)
#                                     print(f"Deleted incomplete TIFF: {tif_file_path}")
#                         if not success:
#                             print(f"Failed to download/process {year}-{month:02d}-{day:02d} after {max_retries} attempts.")
#                             # Continue to next day; don't set all_years_downloaded=False to allow partial data
#                         if tif_file_path.exists():
#                             os.remove(tif_file_path)
#                             print(f"Deleted TIFF: {tif_file_path}")
#             if combined_datasets:
#                 try:
#                     combined_ds = xr.concat(combined_datasets, dim="time").to_dataset(name="precip")
#                     combined_ds = combined_ds.rename({"x": "X", "y": "Y", "time": "T"}).drop_vars('band')
#                     combined_ds = combined_ds.isel(Y=slice(None, None, -1))
#                     combined_ds.to_netcdf(output_path)
#                     combined_ds.close()
#                     print(f"Combined daily dataset for CHIRPS ({blend_type}) saved to {output_path}")
#                     return output_path
#                 except Exception as e:
#                     print(f"Failed to process or save combined dataset for CHIRPS ({blend_type}): {e}")
#             else:
#                 print(f"No data downloaded for CHIRPS ({blend_type}).")

    
#     def _fetch_chirps_daily(self, year, month, day, dir_to_save, blend_type, force_download, area):
#             """
#             Construct the CHIRPS v3.0 daily TIF URL for (year, month, day),
#             download if needed, open as xarray, and optionally clip to 'area'.
           
#             File format is: chirps-v3.0.YYYY.MM.DD.tif
#             """
#             _type = "rnl" if blend_type == "ERA5" else "sat"
                
#             base_url = f"https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/{_type}/{year}"
#             fname = f"chirps-v3.0.{_type}.{year}.{month:02d}.{day:02d}.tif"
#             url = f"{base_url}/{fname}"
#             local_path = Path(dir_to_save) / fname
#             if not local_path.exists() or force_download:
#                 try:
#                     with requests.get(url, stream=True, timeout=120) as r:
#                         r.raise_for_status()
#                         with open(local_path, "wb") as f:
#                             for chunk in r.iter_content(chunk_size=8192):
#                                 f.write(chunk)
#                 except Exception as e:
#                     print(f"[ERROR] Could not download {url}: {e}")
#                     return None
#             else:
#                 print(f"[SKIP] {fname} is already present. (Use force_download=True to overwrite)")
#             # Open as xarray via rioxarray
#             try:
#                 da = rioxr.open_rasterio(local_path, masked=True).squeeze()
#                 time_coord = pd.to_datetime(f"{year}-{month:02d}-{day:02d}")
#                 da = da.expand_dims(time=[time_coord])
#                 da.name = "precip"
#                 # If area is provided, clip
#                 if area and len(area) == 4:
#                     north, west, south, east = area
#                     da = da.rio.clip_box(
#                         minx=west,
#                         miny=south,
#                         maxx=east,
#                         maxy=north
#                     )
#                 return da
#             except Exception as e:
#                 print(f"[ERROR] Could not open/parse {local_path}: {e}")
#                 return None
                

#     def WAS_Download_CHIRPSv3_Seasonal(
#         self,
#         dir_to_save,
#         variables,
#         year_start,
#         year_end,
#         region="africa",
#         area=None,
#         season_months=["03", "04", "05"],
#         force_download=False        
#     ):
        
#         """
#         Download CHIRPS v3.0 monthly precipitation for a specified cross-year season
#         from year_start to year_end, optionally clipped to 'area',
#         and aggregate them into a single NetCDF file.
#         Parameters:
#             dir_to_save (str): Directory to save the downloaded files.
#             variables (list): List of variables to download (e.g., ["PRCP"]).
#             year_start (int): Start year for the data.
#             year_end (int): End year for the data.
#             region (str): CHIRPS region (default: "africa").
#             area (list): Bounding box as [North, West, South, East] (optional).
#             season_months (list): List of months as strings (e.g., ["03", "04", "05"]).
#             force_download (bool): If True, forces download even if file exists.
#         Returns:
#             None: Saves the aggregated seasonal data to a NetCDF file.  
#         """
#         dir_to_save = Path(dir_to_save)
#         dir_to_save.mkdir(parents=True, exist_ok=True)
#         season_months = [int(m) for m in season_months]
#         variables = variables
#         # Example: "MAM"
#         season_str = "".join([calendar.month_abbr[m] for m in season_months])
#         pivot = season_months[0]

#         out_nc = dir_to_save / f"Obs_PRCP_{year_start}_{year_end}_{season_str}.nc"
#         if out_nc.exists() and not force_download:
#             print(f"[INFO] {out_nc} already exists. Skipping.")
#             return out_nc

#         else:
            
#             # We'll store monthly DataArrays here
#             all_data_arrays = []
    
#             # Loop over years
#             for year in range(year_start, year_end + 1):
#                 # Base-year months (>= pivot)
#                 base_months = [m for m in season_months if m >= pivot]
#                 # Next-year months (< pivot)
#                 next_months = [m for m in season_months if m < pivot]
    
#                 # Part A: Base-year months
#                 for m in base_months:
#                     da = self._fetch_chirps_monthly(
#                         year=year,
#                         month=m,
#                         dir_to_save=dir_to_save,
#                         region=region,
#                         force_download=force_download,
#                         area=area
#                     )
#                     if da is not None:
#                         all_data_arrays.append(da)
    
#                 # Part B: Next-year months
#                 if next_months and (year < year_end + 1):
#                     year_next = year + 1
#                     for m in next_months:
#                         da = self._fetch_chirps_monthly(
#                             year=year_next,
#                             month=m,
#                             dir_to_save=dir_to_save,
#                             region=region,
#                             force_download=force_download,
#                             area=area
#                         )
#                         if da is not None:
#                             all_data_arrays.append(da)
    
#             if len(all_data_arrays) == 0:
#                 print("[WARNING] No CHIRPS data arrays were opened/downloaded.")
#                 return
    
#             # Concatenate along time
#             ds_all = xr.concat(all_data_arrays, dim="time").to_dataset(name="precip")
    
#             # Aggregate across the cross-year season (summing monthly precipitation)
#             ds_season = self._aggregate_chirps(ds_all, season_months)
    
#             # Rename dims if desired
#             if "x" in ds_season.dims:
#                 ds_season = ds_season.rename({"x": "X"})
#             if "y" in ds_season.dims:
#                 ds_season = ds_season.rename({"y": "Y"})
#             if "time" in ds_season.dims:
#                 ds_season = ds_season.rename({"time": "T"})
                

#             if len(season_months)==1:
#                 ds_season["T"] = [f"{year}-{season_months[0]:02d}-01" for year in ds_season["T"].dt.year.astype(str).values]
#             elif len(season_months) in [2,3]:
#                 ds_season["T"] = [f"{year}-{season_months[1]:02d}-01" for year in ds_season["T"].dt.year.astype(str).values]
#             elif len(season_months) in [4,5]:
#                 ds_season["T"] = [f"{year}-{season_months[2]:02d}-01" for year in ds_season["T"].dt.year.astype(str).values]
#             else:
#                 ds_season["T"] = [f"{year}-{season_months[3]:02d}-01" for year in ds_season["T"].dt.year.astype(str).values]
#             ds_season["T"] = ds_season["T"].astype("datetime64[ns]")

            
#             # Write to NetCDF
#             ds_season.drop_vars(['band','spatial_ref']).squeeze().isel(Y=slice(None, None, -1)).to_netcdf(out_nc)
#             print(f"[INFO] Saved seasonal CHIRPS data to {out_nc}")
#             # Delete individual monthly TIF files
#             for tif_file in dir_to_save.glob("chirps-v3.0.*.tif"):
#                 try:
#                     os.remove(tif_file)
#                     print(f"[CLEANUP] Deleted {tif_file}")
#                 except Exception as e:
#                     print(f"[ERROR] Could not delete {tif_file}: {e}")
#             return out_nc


#     def _fetch_chirps_monthly(self, year, month, dir_to_save, region, force_download, area):
#         """
#         Construct the CHIRPS v3.0 monthly TIF URL for (year, month), 
#         download if needed, open as xarray, and optionally clip to 'area'.
        
#         File format is: chirps-v3.0.YYYY.MM.tif
#         """
#         base_url = f"https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/{region}/tifs"
#         fname = f"chirps-v3.0.{year}.{month:02d}.tif"
#         url = f"{base_url}/{fname}"

#         local_path = Path(dir_to_save) / fname
#         download_file(url, local_path, force_download=force_download, chunk_size=8192, timeout=120)
#         try:
#             da = rioxr.open_rasterio(local_path, masked=True).squeeze()
#             time_coord = pd.to_datetime(f"{year}-{month:02d}-01")
#             da = da.expand_dims(time=[time_coord])
#             da.name = "precip"

#             # If area is provided, clip
#             if area and len(area) == 4:
#                 north, west, south, east = area
#                 da = da.rio.clip_box(
#                     minx=west,
#                     miny=south,
#                     maxx=east,
#                     maxy=north
#                 )

#             return da

#         except Exception as e:
#             print(f"[ERROR] Could not open/parse {local_path}: {e}")
#             return None

#     def _aggregate_chirps(self, ds, season_months):
#         """
#         Sum monthly precipitation across the cross-year season.
#         """
#         if "time" not in ds.coords:
#             raise ValueError("Dataset must have a 'time' dimension.")

#         pivot = season_months[0]
#         # Label each time with 'season_year'
#         season_year = ds["time"].dt.year.where(ds["time"].dt.month >= pivot,
#                                                ds["time"].dt.year - 1)
#         ds = ds.assign_coords(season_year=season_year)

#         # Keep only the months we want
#         ds = ds.where(ds["time"].dt.month.isin(season_months), drop=True)

#         # Sum across the months for precipitation
#         ds_out = ds.groupby("season_year").sum("time", skipna=True)

#         # Rename season_year -> time
#         ds_out = ds_out.rename({"season_year": "time"})

#         # Optionally make the new time coordinate more descriptive:
#         new_times = []
#         for sy in ds_out.coords["time"].values:
#             new_times.append(f"{sy}-{pivot:02d}-01")
#         ds_out = ds_out.assign_coords(time=pd.to_datetime(new_times))

#         return ds_out
# ####
#     def WAS_Download_TAMSAT_Seasonal(
#             self,
#             dir_to_save: Union[str, Path],
#             product: Literal["rfe", "soil_moisture"] = "rfe",
#             variables: Optional[Sequence[str]] = None,
#             year_start: int = 1983,
#             year_end: int = 2025,
#             area: Optional[List[float]] = None,
#             season_months: List[str] = ["03", "04", "05"],
#             version: Optional[str] = None,
#             force_download: bool = False,
#             agg: Optional[Literal["sum", "mean"]] = None,
#         ) -> Path:
        
#             """
#             Download and aggregate TAMSAT monthly data (RFE v3.1 or Soil Moisture v2.3.1) for a specified season.

#             Parameters
#             ----------
#             dir_to_save : str | Path
#                 Directory where monthly files and the seasonal output will be saved.
#             product : {"rfe", "soil_moisture"}, default "rfe"
#                 Dataset to download: "rfe" (precipitation) or "soil_moisture".
#             variables : sequence of str, optional
#                 Names of variables to extract from NetCDF. If None, chosen by product.
#                 - rfe: defaults to ("rfe",)
#                 - soil_moisture: defaults to ("sm",)
#             year_start : int, default 1983
#                 First seasonal year (pivot year) to include.
#             year_end : int, default 2025
#                 Last year for which data is included (inclusive). For seasons spanning calendar years,
#                 the last pivot year processed will be year_end - 1 to ensure no data from year_end + 1 is fetched.
#             area : list[float], optional
#                 Bounding box [north, west, south, east] in degrees.
#             season_months : sequence[str], default ["03","04","05"]
#                 Months defining the season, e.g. ["11","12","01"] for NDJ.
#             version : str, optional
#                 Product version. Defaults to:
#                 - rfe: "v3.1"
#                 - soil_moisture: "v2.3.1"
#             force_download : bool, default False
#                 Re-download monthly files even if present locally.
#             agg : {"sum","mean"}, optional
#                 Seasonal aggregation. Defaults to:
#                 - rfe: "sum"
#                 - soil_moisture: "mean"

#             Returns
#             -------
#             Path
#                 Path to the seasonal aggregated NetCDF file.
#             """
#             dir_to_save = Path(dir_to_save)
#             dir_to_save.mkdir(parents=True, exist_ok=True)
#             season_months = tuple(season_months)
#             # ---- sensible defaults by product ----
#             if product == "rfe":
#                 variables = variables or ("rfe")
#                 version = version or "v3.1"
#                 agg = agg or "sum"
#                 std_name = "precip"
#             elif product == "soilmoisture":
#                 variables = variables or ("smc_avail_top",)
#                 version = version or "v2.3.1"
#                 agg = agg or "mean"
#                 std_name = "soil_moisture"
#             else:
#                 raise ValueError("product must be 'rfe' or 'soilmoisture'")
#             # ---- validate inputs ----
#             if year_start > year_end:
#                 raise ValueError("year_start must be <= year_end.")
#             season_months_int: List[int] = [int(m) for m in season_months]
#             if not all(1 <= m <= 12 for m in season_months_int):
#                 raise ValueError("Season months must be valid month numbers (1-12).")
#             area_tuple: Optional[Tuple[float, float, float, float]] = (
#                 tuple(map(float, area)) if area else None
#             )
#             season_str = "".join(calendar.month_abbr[m] for m in season_months_int)
#             pivot = season_months_int[0]
#             # ---- output filename ----
#             out_nc = dir_to_save / f"Obs_{product.upper()}_{year_start}_{year_end}_{season_str}.nc"
#             if out_nc.exists() and not force_download:
#                 print(f"[INFO] {out_nc} already exists – skip download.")
#                 return out_nc
#             else:
#                 # ---- determine if season spans years ----
#                 spanning = any(m < pivot for m in season_months_int)
#                 last_season_year = year_end if not spanning else year_end - 1
#                 if year_start > last_season_year:
#                     raise ValueError("No seasons to process based on year_start and year_end.")
#                 # ---- build seasonal series (aggregate per season_year then stack) ----
#                 seasonal_list: List[xr.DataArray] = []
#                 for season_year in range(year_start, last_season_year + 1):
#                     monthly_das: List[xr.DataArray] = []
#                     # Part A: base-year months (>= pivot)
#                     for m in (m for m in season_months_int if m >= pivot):
#                         da = self._fetch_tamsat_monthly(
#                             product=product,
#                             version=version,
#                             year=season_year,
#                             month=m,
#                             dir_to_save=dir_to_save,
#                             force_download=force_download,
#                             area=area_tuple,
#                             keep_vars=variables,
#                             std_name=std_name,
#                         )
#                         if da is not None:
#                             monthly_das.append(da)
#                     # Part B: next-year months (< pivot)
#                     if spanning:
#                         next_year = season_year + 1
#                         for m in (m for m in season_months_int if m < pivot):
#                             da = self._fetch_tamsat_monthly(
#                                 product=product,
#                                 version=version,
#                                 year=next_year,
#                                 month=m,
#                                 dir_to_save=dir_to_save,
#                                 force_download=force_download,
#                                 area=area_tuple,
#                                 keep_vars=variables,
#                                 std_name=std_name,
#                             )
#                             if da is not None:
#                                 monthly_das.append(da)
#                     if not monthly_das:
#                         # nothing for this season_year → skip
#                         continue
#                     # stack months then aggregate for this season
#                     season_stack = xr.concat(monthly_das, dim="time")
#                     if agg == "sum":
#                         season_da = season_stack.sum(dim="time", keep_attrs=True)
#                     elif agg == "mean":
#                         season_da = season_stack.mean(dim="time", keep_attrs=True)
#                     else:
#                         raise ValueError("agg must be 'sum' or 'mean'.")
#                     # give a representative time stamp (pivot-year)
#                     season_time = pd.to_datetime(f"{season_year}-{pivot:02d}-15")
#                     season_da = season_da.expand_dims(time=[season_time])
#                     seasonal_list.append(season_da)
#                 if not seasonal_list:
#                     raise RuntimeError("No TAMSAT files were downloaded or opened for any season.")
#                 # ---- concat seasons and save ----
#                 da_all = xr.concat(seasonal_list, dim="time")
#                 ds_out = da_all.to_dataset(name=std_name)
    
#                 # harmonize dims if needed
#                 rename_dict = {k: v for k, v in {"lon": "X", "lat": "Y", "time": "T"}.items() if k in ds_out.dims}
#                 ds_out = ds_out.rename(rename_dict)
#                 season_months_ = [int(m) for m in season_months]
#                 if len(season_months_)==1:
#                     ds_out["T"] = [f"{year}-{season_months_[0]:02d}-01" for year in ds_out["T"].dt.year.astype(str).values]
#                 elif len(season_months_) in [2,3]:
#                     ds_out["T"] = [f"{year}-{ season_months_[1]:02d}-01" for year in ds_out["T"].dt.year.astype(str).values]
#                 elif len(season_months_) in [4,5]:
#                     ds_out["T"] = [f"{year}-{season_months_[2]:02d}-01" for year in ds_out["T"].dt.year.astype(str).values]
#                 else:
#                     ds_out["T"] = [f"{year}-{season_months_[3]:02d}-01" for year in ds_out["T"].dt.year.astype(str).values]                    
#                 ds_out["T"] = ds_out["T"].astype("datetime64[ns]")
#                 ds_out.to_netcdf(out_nc)
#                 print(f"[INFO] Saved seasonal {product.upper()} data → {out_nc}")
#                 return out_nc
                
#     def _fetch_tamsat_monthly(
#         self,
#         product: Literal["rfe", "soilmoisture"],
#         version: str,
#         year: int,
#         month: int,
#         dir_to_save: Path,
#         force_download: bool,
#         area: Optional[Tuple[float, float, float, float]],
#         keep_vars: Sequence[str],
#         std_name: str,
#     ) -> Optional[xr.DataArray]:
#         """
#         Download & open a single monthly TAMSAT file (RFE v3.1 or Soil Moisture v2.3.1),
#         clip to bbox if provided, and return a standardized DataArray.
#         Returns
#         -------
#         xr.DataArray | None
#         """
#         if product == "rfe":
#             # e.g. .../tamsat/rfe/data/v3.1/monthly/1983/01/rfe1983_01.v3.1.nc
#             base = (
#                 "https://gws-access.jasmin.ac.uk/public/tamsat/rfe/data/"
#                 f"{version}/monthly/{{year}}/{{month:02d}}/rfe{{year}}_{{month:02d}}.{version}.nc"
#             )
#         else: # soil_moisture
#             # e.g. .../tamsat/soil_moisture/data/v2.3.1/monthly/1983/01/sm1983_01.v2.3.1.nc
#             base = (
#                 "https://gws-access.jasmin.ac.uk/public/tamsat/soil_moisture/data/"
#                 f"{version}/monthly/{{year}}/{{month:02d}}/sm{{year}}_{{month:02d}}.{version}.nc"
#             )
#         url = base.format(year=year, month=month)
#         fname = dir_to_save / url.split("/")[-1]
#         # download if needed
#         if not fname.exists() or force_download:
#             try:
#                 print(f"[DL ] {url}")
#                 with requests.get(url, stream=True, timeout=180) as r:
#                     r.raise_for_status()
#                     with open(fname, "wb") as f:
#                         for chunk in r.iter_content(chunk_size=8192):
#                             if chunk:
#                                 f.write(chunk)
#             except Exception as exc:
#                 print(f"[ERR] Download failed: {exc}")
#                 return None
#         else:
#             print(f"[SKP] {fname.name} already present.")
#         # open & standardize
#         try:
#             ds = xr.open_dataset(fname)
            
#             var = keep_vars # next((v for v in keep_vars if v in ds.data_vars), None)
#             if var is None:
#                 raise KeyError(f"None of {keep_vars} found in {fname.name}; available: {list(ds.data_vars)}")
#             # make a 1-step time axis for this month
#             da = ds[var].assign_coords(time=[pd.to_datetime(f"{year}-{month:02d}-01")]).astype("float32")
#             # spatial clip if requested
#             if area:
#                 n, w, s, e = area
#                 # TAMSAT is lat/lon naming; ensure correct orientation
#                 latn = "lat" if "lat" in da.coords else "latitude"
#                 lonn = "lon" if "lon" in da.coords else "longitude"
#                 da = da.where(
#                     (da[latn] <= n) & (da[latn] >= s) & (da[lonn] >= w) & (da[lonn] <= e),
#                     drop=True,
#                 )
#             da.name = std_name
#             return da
#         except Exception as exc:
#             print(f"[ERR] Failed to open dataset: {exc}")
#             return None


#     def WAS_Download_TAMSAT_Daily(
#             self,
#             dir_to_save: Union[str, Path],
#             product: Literal["rfe", "soilmoisture"] = "rfe",
#             variables: Optional[Sequence[str]] = None,
#             year_start: str = "1983",
#             year_end: str = "2024",
#             area: Optional[List[float]] = None,
#             version: Optional[str] = None,
#             force_download: bool = False,
#         ) -> Path:
#             """
#             Download TAMSAT daily data (RFE v3.1 or Soil Moisture v2.3.1) and combine into a single NetCDF file.

#             Parameters
#             ----------
#             dir_to_save : str | Path
#                 Directory where daily files and the combined output will be saved.
#             product : {"rfe", "soil_moisture"}, default "rfe"
#                 Dataset to download: "rfe" (precipitation) or "soil_moisture".
#             variables : sequence of str, optional
#                 Names of variables to extract from NetCDF. If None, chosen by product.
#                 - rfe: defaults to ("rfe",)
#                 - soil_moisture: defaults to ("sm",)
#             start_date : str, default "1983-01-01"
#                 Start date in "YYYY-MM-DD" format.
#             end_date : str, default "2025-10-20"
#                 End date in "YYYY-MM-DD" format (inclusive).
#             area : list[float], optional
#                 Bounding box [north, west, south, east] in degrees.
#             version : str, optional
#                 Product version. Defaults to:
#                 - rfe: "v3.1"
#                 - soil_moisture: "v2.3.1"
#             force_download : bool, default False
#                 Re-download daily files even if present locally.

#             Returns
#             -------
#             Path
#                 Path to the combined daily NetCDF file.
#             """
#             dir_to_save = Path(dir_to_save)
#             dir_to_save.mkdir(parents=True, exist_ok=True)
#             start_date = f"{year_start}-01-01"
#             end_date = f"{year_end}-12-31" if len(str(year_end)) == 4 else f"{year_end.year:04d}-{year_end.month:02d}-{year_end.day:02d}"
#             yr_end = year_end if len(str(year_end)) == 4 else year_end.year
        
#             # ---- sensible defaults by product ----
#             if product == "rfe":
#                 variables = variables or ("rfe",)
#                 version = version or "v3.1"
#                 prefix = "rfe"
#                 std_name = "precip"
#             elif product == "soilmoisture":
#                 variables = variables or ("smc_avail_top",)
#                 version = version or "v2.3.1"
#                 prefix = "sm"
#                 std_name = "soil_moisture"
#             else:
#                 raise ValueError("product must be 'rfe' or 'soilmoisture'")
#             # ---- validate inputs ----
#             start_dt = pd.to_datetime(start_date)
#             end_dt = pd.to_datetime(end_date)
#             if start_dt > end_dt:
#                 raise ValueError("start_date must be <= end_date.")
#             dates = pd.date_range(start_dt, end_dt, freq="D")
#             area_tuple: Optional[Tuple[float, float, float, float]] = (
#                 tuple(map(float, area)) if area else None
#             )
#             # ---- output filename ----
#             sdate_str = start_date.replace("-", "")
#             edate_str = end_date.replace("-", "")
#             out_nc = dir_to_save /  f"Daily_PRCP_{year_start}_{yr_end}.nc" # f"Obs_{product.upper()}_daily_{sdate_str}_{edate_str}.nc"
#             if out_nc.exists() and not force_download:
#                 print(f"[INFO] {out_nc} already exists – skip download.")
#                 return out_nc
#             else:
#                 # ---- build daily series ----
#                 daily_list: List[xr.DataArray] = []
#                 for date in dates:
#                     y = date.year
#                     m = date.month
#                     d = date.day
#                     da = self._fetch_tamsat_daily(
#                         product=product,
#                         version=version,
#                         year=y,
#                         month=m,
#                         day=d,
#                         dir_to_save=dir_to_save,
#                         force_download=force_download,
#                         area=area_tuple,
#                         keep_vars=variables,
#                         std_name=std_name,
#                         prefix=prefix,
#                     )
#                     if da is not None:
#                         daily_list.append(da)
#                 if not daily_list:
#                     raise RuntimeError("No TAMSAT files were downloaded or opened.")
#                 # ---- concat days and save ----
#                 da_all = xr.concat(daily_list, dim="time")
#                 ds_out = da_all.to_dataset(name=std_name)
    
#                 # harmonize dims if needed
#                 rename_dict = {k: v for k, v in {"lon": "X", "lat": "Y", "time": "T"}.items() if k in ds_out.dims}
#                 ds_out = ds_out.rename(rename_dict)
#                 ds_out.to_netcdf(out_nc)
#                 print(f"[INFO] Saved daily {product.upper()} data → {out_nc}")
#                 return out_nc
#     def _fetch_tamsat_daily(
#         self,
#         product: Literal["rfe", "soilmoisture"],
#         version: str,
#         year: int,
#         month: int,
#         day: int,
#         dir_to_save: Path,
#         force_download: bool,
#         area: Optional[Tuple[float, float, float, float]],
#         keep_vars: Sequence[str],
#         std_name: str,
#         prefix: str,
#     ) -> Optional[xr.DataArray]:
#         """
#         Download & open a single daily TAMSAT file (RFE v3.1 or Soil Moisture v2.3.1),
#         clip to bbox if provided, and return a standardized DataArray.
#         Returns
#         -------
#         xr.DataArray | None
#         """
#         if product == "soilmoisture":
#             product_ = "soil_moisture"
#         else:
#             product_ = product
#         base = (
#             f"https://gws-access.jasmin.ac.uk/public/tamsat/{product_}/data/"
#             f"{version}/daily/{{year}}/{{month:02d}}/{prefix}{{year}}_{{month:02d}}_{{day:02d}}.{version}.nc"
#         )
#         url = base.format(year=year, month=month, day=day)
#         fname = dir_to_save / url.split("/")[-1]
#         # download if needed
#         if not fname.exists() or force_download:
#             try:
#                 print(f"[DL ] {url}")
#                 with requests.get(url, stream=True, timeout=180) as r:
#                     r.raise_for_status()
#                     with open(fname, "wb") as f:
#                         for chunk in r.iter_content(8192):
#                             if chunk:
#                                 f.write(chunk)
#             except Exception as exc:
#                 print(f"[ERR] Download failed: {exc}")
#                 return None
#         else:
#             print(f"[SKP] {fname.name} already present.")
#         # open & standardize
#         try:
#             ds = xr.open_dataset(fname)
#             var = keep_vars #next((v for v in keep_vars if v in ds.data_vars), None)
#             if var is None:
#                 raise KeyError(f"None of {keep_vars} found in {fname.name}; available: {list(ds.data_vars)}")
#             # make- a 1-step time axis for this day
#             da = ds[var].assign_coords(time=[pd.to_datetime(f"{year}-{month:02d}-{day:02d}")]).astype("float32")
#             # spatial clip if requested
#             if area:
#                 n, w, s, e = area
#                 # TAMSAT is lat/lon naming; ensure correct orientation
#                 latn = "lat" if "lat" in da.coords else "latitude"
#                 lonn = "lon" if "lon" in da.coords else "longitude"
#                 da = da.where(
#                     (da[latn] <= n) & (da[latn] >= s) & (da[lonn] >= w) & (da[lonn] <= e),
#                     drop=True,
#                 )
#             da.name = std_name
#             return da
#         except Exception as exc:
#             print(f"[ERR] Failed to open dataset: {exc}")
#             return None
    
    
    # def WAS_Download_CHIRPSv3_Daily(
    #     self,
    #     dir_to_save,
    #     year_start,
    #     year_end,
    #     blend_type="ERA5",
    #     area=None,
    #     force_download=False,
    #     max_retries=3,
    #     retry_delay=5,
    #     cleanup=True,
    #     allow_incomplete=False,
    # ):
    #     """
    #     Download CHIRPS v3 daily precipitation and save a WAS-style NetCDF.
    
    #     allow_incomplete=False (default): the final NetCDF is written only when
    #     every requested year is complete; otherwise a RuntimeError asks you to
    #     re-run (all intermediate files are kept, so the re-run only fetches what
    #     is missing). Set allow_incomplete=True to write the final file from the
    #     complete years only (previous behaviour).
    
    #     Resumable: daily .tif files already on disk are reused; only missing days
    #     are downloaded. Intermediate files are deleted only once the final NetCDF
    #     has been written (if cleanup=True).
    
    #     Output:
    #         variable: PRCP
    #         dims: T, Y, X
    #         units: mm/day
    #     """
    #     from pathlib import Path
    #     import calendar
    #     import gc
    #     import os
    
    #     import pandas as pd
    #     import rioxarray as rioxr
    #     import xarray as xr
    
    #     dir_to_save = Path(dir_to_save)
    #     dir_to_save.mkdir(parents=True, exist_ok=True)
    #     area = tuple(area)
    #     blend_map = {"ERA5": "rnl", "IMERGlate-v07": "sat", "IMERG": "sat", "SAT": "sat"}
    #     if blend_type not in blend_map:
    #         raise ValueError("blend_type must be 'ERA5' or 'IMERGlate-v07'.")
    
    #     product_tag = blend_map[blend_type]
    #     output_path = dir_to_save / f"Daily_PRCP_CHIRPSv3_{product_tag}_{year_start}_{year_end}.nc"
    #     if output_path.exists() and not force_download:
    #         print(f"{output_path} already exists. Skipping download.")
    #         return output_path
    
    #     def _open_chirps_tif(path, date):
    #         da = rioxr.open_rasterio(path, masked=True).squeeze(drop=True)
    #         if area is not None:
    #             north, west, south, east = area
    #             da = da.rio.clip_box(minx=west, miny=south, maxx=east, maxy=north)
    #         da = da.rename({name: {"x": "X", "y": "Y"}.get(name, name) for name in da.dims})
    #         if "spatial_ref" in da.coords:
    #             da = da.drop_vars("spatial_ref", errors="ignore")
    #         da = da.expand_dims(T=[pd.Timestamp(date)])
    #         da.name = "PRCP"
    #         da.attrs["units"] = "mm/day"
    #         da.attrs["source"] = "CHIRPS v3"
    #         if "Y" in da.coords:
    #             da = da.sortby("Y")
    #         if "X" in da.coords:
    #             da = da.sortby("X")
    #         return da
    
    #     yearly_paths = []
    #     incomplete_years = []
    #     for year in range(int(year_start), int(year_end) + 1):
    #         year_path = dir_to_save / f"tmp_CHIRPSv3_{product_tag}_{year}.nc"
    #         if year_path.exists() and not force_download:
    #             print(f"[RESUME] Year checkpoint found, reusing {year_path.name}")
    #             yearly_paths.append(year_path)
    #             continue
    
    #         daily_arrays = []
    #         year_tifs = []
    #         all_ok = True
    #         n_reused = 0
    #         for month in range(1, 13):
    #             ndays = calendar.monthrange(year, month)[1]
    #             for day in range(1, ndays + 1):
    #                 fname = f"chirps-v3.0.{product_tag}.{year}.{month:02d}.{day:02d}.tif"
    #                 url = f"https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/{product_tag}/{year}/{fname}"
    #                 tif_path = dir_to_save / fname
    
    #                 if tif_path.exists() and not force_download:
    #                     n_reused += 1
    #                 else:
    #                     ok = _download_atomic(url, tif_path, max_retries, retry_delay)
    #                     if not ok:
    #                         all_ok = False
    #                         continue
    
    #                 try:
    #                     da = _open_chirps_tif(tif_path, f"{year}-{month:02d}-{day:02d}").load()
    #                     daily_arrays.append(da)
    #                     year_tifs.append(tif_path)
    #                 except Exception as exc:
    #                     # Likely a truncated file from an old non-atomic run:
    #                     # remove it so the next run re-downloads it cleanly.
    #                     print(f"Failed to open/process {tif_path}: {exc} -> removing corrupt file.")
    #                     _remove_quiet(tif_path)
    #                     all_ok = False
    
    #         if n_reused:
    #             print(f"[RESUME] Year {year}: reused {n_reused} daily file(s) already on disk.")
    
    #         if not daily_arrays:
    #             print(f"No CHIRPS daily data for {year}.")
    #             incomplete_years.append(year)
    #             continue
    
    #         if not all_ok:
    #             # Do NOT delete the tifs: they are the resume state.
    #             print(
    #                 f"Year {year} incomplete ({len(daily_arrays)} day(s) on disk). "
    #                 f"Daily files are kept; re-run to download only the missing days."
    #             )
    #             incomplete_years.append(year)
    #             del daily_arrays
    #             gc.collect()
    #             continue
    
    #         ds_year = xr.concat(daily_arrays, dim="T").to_dataset(name="PRCP").sortby("T")
    #         encoding = {"PRCP": {"zlib": True, "complevel": 4}}
    #         # Atomic write of the yearly checkpoint as well.
    #         year_tmp = year_path.parent / (year_path.name + ".part")
    #         ds_year.to_netcdf(year_tmp, encoding=encoding)
    #         ds_year.close()
    #         os.replace(year_tmp, year_path)
    #         yearly_paths.append(year_path)
    
    #         # Year checkpoint safely on disk -> the daily tifs may now be removed.
    #         if cleanup:
    #             for tif_path in year_tifs:
    #                 _remove_quiet(tif_path)
    
    #         del daily_arrays, ds_year
    #         gc.collect()
    
    #     if not yearly_paths:
    #         raise RuntimeError(
    #             "No complete CHIRPS daily year was processed. "
    #             "Downloaded daily files are kept on disk; re-run to resume."
    #         )
    
    #     if incomplete_years and not allow_incomplete:
    #         raise RuntimeError(
    #             f"Incomplete year(s) {incomplete_years}: the final NetCDF was NOT "
    #             f"written so it does not block the resume. Re-run to download only "
    #             f"the missing days, or pass allow_incomplete=True to write the "
    #             f"final file from complete years only."
    #         )
    
    #     datasets = [xr.open_dataset(path) for path in yearly_paths]
    #     ds_out = xr.concat(datasets, dim="T").sortby("T")
    #     ds_out.attrs["source"] = "CHIRPS v3 daily"
    #     ds_out.attrs["blend_type"] = blend_type
    #     out_tmp = output_path.parent / (output_path.name + ".part")
    #     ds_out.to_netcdf(out_tmp, encoding={"PRCP": {"zlib": True, "complevel": 4}})
    #     for ds in datasets:
    #         ds.close()
    #     ds_out.close()
    #     os.replace(out_tmp, output_path)
    
    #     # Final output safely on disk -> yearly checkpoints may now be removed.
    #     if cleanup:
    #         for path in yearly_paths:
    #             _remove_quiet(path)
    
    #     print(f"Saved CHIRPS daily data to {output_path}")
    #     return output_path


    def WAS_Download_CHIRPSv3_Daily(
        self,
        dir_to_save,
        year_start,
        year_end,
        blend_type="ERA5",
        area=None,
        force_download=False,
        max_retries=3,
        retry_delay=5,
        cleanup=True,
        allow_incomplete=False,
    ):
        """
        Download CHIRPS v3 daily precipitation and save a WAS-style NetCDF.
    
        Memory-optimized: monthly checkpoints + float32 + dask streaming write.
        Resumable: existing .tif and tmp monthly .nc files are reused.
    
        Output: PRCP, dims T/Y/X, mm/day.
        """
        from pathlib import Path
        import calendar
        import gc
        import os
    
        import pandas as pd
        import rioxarray as rioxr
        import xarray as xr
    
        dir_to_save = Path(dir_to_save)
        dir_to_save.mkdir(parents=True, exist_ok=True)
        area = tuple(area)
        blend_map = {"ERA5": "rnl", "IMERGlate-v07": "sat", "IMERG": "sat", "SAT": "sat"}
        if blend_type not in blend_map:
            raise ValueError("blend_type must be 'ERA5' or 'IMERGlate-v07'.")
    
        product_tag = blend_map[blend_type]
        output_path = dir_to_save / f"Daily_PRCP_CHIRPSv3_{product_tag}_{year_start}_{year_end}.nc"
        if output_path.exists() and not force_download:
            print(f"{output_path} already exists. Skipping download.")
            return output_path
    
        def _open_chirps_tif(path, date):
            da = rioxr.open_rasterio(path, masked=True).squeeze(drop=True)
            if area is not None:
                north, west, south, east = area
                da = da.rio.clip_box(minx=west, miny=south, maxx=east, maxy=north)
            da = da.rename({name: {"x": "X", "y": "Y"}.get(name, name) for name in da.dims})
            if "spatial_ref" in da.coords:
                da = da.drop_vars("spatial_ref", errors="ignore")
            # masked=True promotes to float64 -> cast back to float32 immediately.
            da = da.astype("float32")
            da = da.expand_dims(T=[pd.Timestamp(date)])
            da.name = "PRCP"
            da.attrs["units"] = "mm/day"
            if "Y" in da.coords:
                da = da.sortby("Y")
            if "X" in da.coords:
                da = da.sortby("X")
            return da
    
        monthly_paths = []
        incomplete_months = []
    
        for year in range(int(year_start), int(year_end) + 1):
            for month in range(1, 13):
                month_path = dir_to_save / f"tmp_CHIRPSv3_{product_tag}_{year}{month:02d}.nc"
                if month_path.exists() and not force_download:
                    monthly_paths.append(month_path)
                    continue
    
                ndays = calendar.monthrange(year, month)[1]
                daily_arrays = []
                month_tifs = []
                all_ok = True
                n_reused = 0
    
                for day in range(1, ndays + 1):
                    fname = f"chirps-v3.0.{product_tag}.{year}.{month:02d}.{day:02d}.tif"
                    url = f"https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/{product_tag}/{year}/{fname}"
                    tif_path = dir_to_save / fname
    
                    if tif_path.exists() and not force_download:
                        n_reused += 1
                    else:
                        if not _download_atomic(url, tif_path, max_retries, retry_delay):
                            all_ok = False
                            continue
    
                    try:
                        daily_arrays.append(_open_chirps_tif(tif_path, f"{year}-{month:02d}-{day:02d}").load())
                        month_tifs.append(tif_path)
                    except Exception as exc:
                        print(f"Failed to open {tif_path}: {exc} -> removing corrupt file.")
                        _remove_quiet(tif_path)
                        all_ok = False
    
                if n_reused:
                    print(f"[RESUME] {year}-{month:02d}: reused {n_reused} daily file(s).")
    
                if not daily_arrays or not all_ok:
                    print(f"Month {year}-{month:02d} incomplete; daily files kept for resume.")
                    incomplete_months.append(f"{year}-{month:02d}")
                    del daily_arrays
                    gc.collect()
                    continue
    
                ds_month = xr.concat(daily_arrays, dim="T").to_dataset(name="PRCP").sortby("T")
                month_tmp = month_path.parent / (month_path.name + ".part")
                ds_month.to_netcdf(month_tmp, encoding={"PRCP": {"zlib": True, "complevel": 4}})
                ds_month.close()
                os.replace(month_tmp, month_path)
                monthly_paths.append(month_path)
    
                # Monthly checkpoint safely on disk -> tifs of this month removable.
                if cleanup:
                    for tif_path in month_tifs:
                        _remove_quiet(tif_path)
    
                del daily_arrays, ds_month
                gc.collect()
    
        if not monthly_paths:
            raise RuntimeError("No complete CHIRPS month was processed; re-run to resume.")
    
        if incomplete_months and not allow_incomplete:
            raise RuntimeError(
                f"Incomplete month(s) {incomplete_months}: final NetCDF NOT written. "
                f"Re-run to fetch only missing days, or pass allow_incomplete=True."
            )
    
        # LAZY final assembly: dask streams the write, nothing is fully loaded.
        print(f"Assembling {len(monthly_paths)} monthly checkpoint(s) lazily with dask...")
        ds_out = xr.open_mfdataset(
            sorted(monthly_paths),
            combine="nested",
            concat_dim="T",
            chunks={"T": 31},
            parallel=False,
        ).sortby("T")
        ds_out.attrs["source"] = "CHIRPS v3 daily"
        ds_out.attrs["blend_type"] = blend_type
    
        out_tmp = output_path.parent / (output_path.name + ".part")
        encoding = {"PRCP": {"zlib": True, "complevel": 4, "dtype": "float32"}}
        delayed = ds_out.to_netcdf(out_tmp, encoding=encoding, compute=False)
        try:
            from dask.diagnostics import ProgressBar
            with ProgressBar():
                delayed.compute()
        except ImportError:
            delayed.compute()
        ds_out.close()
        os.replace(out_tmp, output_path)
    
        if cleanup:
            for path in monthly_paths:
                _remove_quiet(path)
    
        print(f"Saved CHIRPS daily data to {output_path}")
        return output_path
    
    def WAS_Download_CHIRPSv3_Seasonal(
        self,
        dir_to_save,
        variables=("PRCP",),
        year_start=1981,
        year_end=2024,
        region="africa",
        area=None,
        season_months=["03", "04", "05"],
        force_download=False,
        cleanup=True,
        max_retries=3,
        retry_delay=5,
        allow_incomplete=False,
    ):
        """
        Download CHIRPS v3 monthly precipitation and aggregate by season.
    
        Resumable: monthly .tif files already on disk are reused; cleanup is
        deferred until the final seasonal NetCDF is written.
    
        Output:
            variable: PRCP
            dims: T, Y, X
            units: mm/season
    
        Cross-year seasons are handled with the first season month as pivot.
        For NDJ, season_year=2020 means Nov-Dec 2020 + Jan 2021.
        """
        from pathlib import Path
        import calendar
        import os
    
        import pandas as pd
        import rioxarray as rioxr
        import xarray as xr
    
        if isinstance(variables, str):
            variables = (variables,)
        if "PRCP" not in variables:
            raise ValueError("CHIRPS seasonal currently supports PRCP only.")
    
        dir_to_save = Path(dir_to_save)
        dir_to_save.mkdir(parents=True, exist_ok=True)
        
        area = tuple(area)
        season_months = tuple(season_months)
        
        season_months_int = [int(m) for m in season_months]
        pivot = season_months_int[0]
        season_str = "".join(calendar.month_abbr[m] for m in season_months_int)
        out_nc = dir_to_save / f"Obs_PRCP_CHIRPSv3_{year_start}_{year_end}_{season_str}.nc"
        if out_nc.exists() and not force_download:
            print(f"[INFO] {out_nc} already exists. Skipping.")
            return out_nc
    
        def _season_calendar_months(season_year):
            return [
                (season_year if month >= pivot else season_year + 1, month)
                for month in season_months_int
            ]
    
        def _representative_time(season_year):
            mid_month = season_months_int[len(season_months_int) // 2]
            cal_year = season_year if mid_month >= pivot else season_year + 1
            return pd.Timestamp(cal_year, mid_month, 1)
    
        used_paths = set()
    
        def _fetch_month(year, month):
            fname = f"chirps-v3.0.{year}.{month:02d}.tif"
            url = f"https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/{region}/tifs/{fname}"
            tif_path = dir_to_save / fname
    
            if not tif_path.exists() or force_download:
                ok = _download_atomic(url, tif_path, max_retries, retry_delay)
                if not ok:
                    return None
    
            try:
                da = rioxr.open_rasterio(tif_path, masked=True).squeeze(drop=True)
                if area is not None:
                    north, west, south, east = area
                    da = da.rio.clip_box(minx=west, miny=south, maxx=east, maxy=north)
                da = da.rename({name: {"x": "X", "y": "Y"}.get(name, name) for name in da.dims})
                if "spatial_ref" in da.coords:
                    da = da.drop_vars("spatial_ref", errors="ignore")
                da = da.expand_dims(month_time=[pd.Timestamp(year, month, 1)])
                da.name = "PRCP"
                da.attrs["units"] = "mm/month"
                if "Y" in da.coords:
                    da = da.sortby("Y")
                if "X" in da.coords:
                    da = da.sortby("X")
                da = da.load()
                used_paths.add(tif_path)  # deletion deferred to the very end
                return da
            except Exception as exc:
                print(f"[ERR] Failed to open/process {tif_path}: {exc} -> removing corrupt file.")
                _remove_quiet(tif_path)
                return None
    
        seasonal_arrays = []
        all_ok = True
        for season_year in range(int(year_start), int(year_end) + 1):
            monthly = []
            for cal_year, month in _season_calendar_months(season_year):
                da = _fetch_month(cal_year, month)
                if da is None:
                    all_ok = False
                    print(f"Incomplete CHIRPS season {season_year}: missing {cal_year}-{month:02d}.")
                    continue
                monthly.append(da)
    
            if len(monthly) != len(season_months_int):
                continue
    
            da_season = xr.concat(monthly, dim="month_time").sum("month_time", keep_attrs=True)
            da_season = da_season.expand_dims(T=[_representative_time(season_year)])
            da_season.name = "PRCP"
            da_season.attrs["units"] = "mm"
            seasonal_arrays.append(da_season)
    
        if not seasonal_arrays:
            raise RuntimeError(
                "No complete CHIRPS seasonal data were processed. "
                "Downloaded monthly files are kept on disk; re-run to resume."
            )
        if not all_ok:
            if not allow_incomplete:
                raise RuntimeError(
                    "Incomplete CHIRPS seasonal download: the final NetCDF was NOT "
                    "written so it does not block the resume. Monthly files already "
                    "downloaded are kept; re-run to fetch only the missing months, "
                    "or pass allow_incomplete=True to write the complete seasons only."
                )
            print("[WARNING] Some seasons were incomplete and skipped (allow_incomplete=True).")
    
        ds_out = xr.concat(seasonal_arrays, dim="T").to_dataset(name="PRCP").sortby("T")
        ds_out.attrs["source"] = "CHIRPS v3 monthly"
        ds_out.attrs["season"] = season_str
        ds_out.attrs["season_months"] = ",".join(f"{m:02d}" for m in season_months_int)
        out_tmp = out_nc.parent / (out_nc.name + ".part")
        ds_out.to_netcdf(out_tmp, encoding={"PRCP": {"zlib": True, "complevel": 4}})
        ds_out.close()
        os.replace(out_tmp, out_nc)
    
        # Final output safely on disk -> monthly tifs may now be removed.
        if cleanup:
            for path in used_paths:
                _remove_quiet(path)
    
        print(f"[INFO] Saved seasonal CHIRPS data to {out_nc}")
        return out_nc
    
    
    def WAS_Download_TAMSAT_Seasonal(
        self,
        dir_to_save,
        product="rfe",
        variables=None,
        year_start=1983,
        year_end=2025,
        area=None,
        season_months=["03", "04", "05"],
        version=None,
        force_download=False,
        agg=None,
        cleanup=False,
        max_retries=3,
        retry_delay=5,
        allow_incomplete=False,
    ):
        """
        Download and aggregate TAMSAT monthly data by season.
    
        Resumable: monthly .nc files already on disk are reused; corrupt files
        are removed and re-downloaded on the next run; cleanup is deferred until
        the final seasonal NetCDF is written.
    
        product:
            "rfe" -> precipitation, default variable "rfe", seasonal sum.
            "soilmoisture" or "soil_moisture" -> default variable "smc_avail_top",
            seasonal mean.
    
        Output dims:
            T, Y, X
        """
        from pathlib import Path
        import calendar
        import os
    
        import pandas as pd
        import xarray as xr
    
        dir_to_save = Path(dir_to_save)
        dir_to_save.mkdir(parents=True, exist_ok=True)
        
        area = tuple(area)
        season_months = tuple(season_months) 
        
        product_key = "soilmoisture" if product in ["soilmoisture", "soil_moisture"] else product
        if product_key == "rfe":
            variables = _as_tuple(variables, default=("rfe",))
            version = version or "v3.1"
            agg = agg or "sum"
            std_name = "PRCP"
            url_product = "rfe"
            prefix = "rfe"
        elif product_key == "soilmoisture":
            variables = _as_tuple(variables, default=("smc_avail_top",))
            version = version or "v2.3.1"
            agg = agg or "mean"
            std_name = "soil_moisture"
            url_product = "soil_moisture"
            prefix = "sm"
        else:
            raise ValueError("product must be 'rfe', 'soilmoisture', or 'soil_moisture'.")
    
        season_months_int = [int(m) for m in season_months]
        pivot = season_months_int[0]
        season_str = "".join(calendar.month_abbr[m] for m in season_months_int)
        out_nc = dir_to_save / f"Obs_TAMSAT_{product_key}_{year_start}_{year_end}_{season_str}.nc"
        if out_nc.exists() and not force_download:
            print(f"[INFO] {out_nc} already exists. Skipping.")
            return out_nc
    
        def _season_calendar_months(season_year):
            return [
                (season_year if month >= pivot else season_year + 1, month)
                for month in season_months_int
            ]
    
        def _representative_time(season_year):
            mid_month = season_months_int[len(season_months_int) // 2]
            cal_year = season_year if mid_month >= pivot else season_year + 1
            return pd.Timestamp(cal_year, mid_month, 1)
    
        used_paths = set()
    
        def _fetch_month(year, month):
            url = (
                f"https://gws-access.jasmin.ac.uk/public/tamsat/{url_product}/data/"
                f"{version}/monthly/{year}/{month:02d}/{prefix}{year}_{month:02d}.{version}.nc"
            )
            path = dir_to_save / url.split("/")[-1]
            if not path.exists() or force_download:
                ok = _download_atomic(url, path, max_retries, retry_delay)
                if not ok:
                    return None
    
            try:
                with xr.open_dataset(path) as ds:
                    var_name = _select_variable(ds, variables)
                    da = ds[var_name].squeeze(drop=True).astype("float32").load()
                da = _standardize_latlon_da(da, area)
                da = da.expand_dims(month_time=[pd.Timestamp(year, month, 1)])
                da.name = std_name
                used_paths.add(path)  # deletion deferred to the very end
                return da
            except Exception as exc:
                print(f"[ERR] Failed to open/process {path}: {exc} -> removing corrupt file.")
                _remove_quiet(path)
                return None
    
        seasonal = []
        incomplete = False
        for season_year in range(int(year_start), int(year_end) + 1):
            monthly = []
            for cal_year, month in _season_calendar_months(season_year):
                da = _fetch_month(cal_year, month)
                if da is not None:
                    monthly.append(da)
    
            if len(monthly) != len(season_months_int):
                print(f"[WARNING] Skipping incomplete TAMSAT season {season_year}.")
                incomplete = True
                continue
    
            stack = xr.concat(monthly, dim="month_time")
            if agg == "sum":
                da_season = stack.sum("month_time", keep_attrs=True)
            elif agg == "mean":
                da_season = stack.mean("month_time", keep_attrs=True)
            else:
                raise ValueError("agg must be 'sum' or 'mean'.")
            da_season = da_season.expand_dims(T=[_representative_time(season_year)])
            da_season.name = std_name
            seasonal.append(da_season)
    
        if not seasonal:
            raise RuntimeError(
                "No complete TAMSAT seasons were processed. "
                "Downloaded monthly files are kept on disk; re-run to resume."
            )
        if incomplete:
            if not allow_incomplete:
                raise RuntimeError(
                    "Incomplete TAMSAT seasonal download: the final NetCDF was NOT "
                    "written so it does not block the resume. Monthly files already "
                    "downloaded are kept; re-run to fetch only the missing months, "
                    "or pass allow_incomplete=True to write the complete seasons only."
                )
            print("[WARNING] Some TAMSAT seasons were incomplete and skipped (allow_incomplete=True).")
    
        ds_out = xr.concat(seasonal, dim="T").to_dataset(name=std_name).sortby("T")
        ds_out.attrs["source"] = "TAMSAT"
        ds_out.attrs["product"] = product_key
        ds_out.attrs["season"] = season_str
        ds_out.attrs["season_months"] = ",".join(f"{m:02d}" for m in season_months_int)
        out_tmp = out_nc.parent / (out_nc.name + ".part")
        ds_out.to_netcdf(out_tmp, encoding={std_name: {"zlib": True, "complevel": 4}})
        ds_out.close()
        os.replace(out_tmp, out_nc)
    
        if cleanup:
            for path in used_paths:
                _remove_quiet(path)
    
        print(f"[INFO] Saved seasonal TAMSAT data to {out_nc}")
        return out_nc
    
    
    # def WAS_Download_TAMSAT_Daily(
    #     self,
    #     dir_to_save,
    #     product="rfe",
    #     variables=None,
    #     year_start=1983,
    #     year_end=2024,
    #     area=None,
    #     version=None,
    #     force_download=False,
    #     cleanup=False,
    #     max_retries=3,
    #     retry_delay=5,
    #     allow_incomplete=False,
    # ):
    #     """
    #     Download TAMSAT daily data and save a WAS-style NetCDF.
    
    #     Resumable: daily .nc files already on disk are reused; corrupt files are
    #     removed and re-downloaded on the next run; cleanup is deferred until the
    #     final NetCDF is written.
    
    #     Output dims:
    #         T, Y, X
    #     """
    #     from pathlib import Path
    #     import os
    
    #     import pandas as pd
    #     import xarray as xr
    
    #     dir_to_save = Path(dir_to_save)
    #     dir_to_save.mkdir(parents=True, exist_ok=True)
    #     area = tuple(area)
 
        
    #     product_key = "soilmoisture" if product in ["soilmoisture", "soil_moisture"] else product
    #     if product_key == "rfe":
    #         variables = _as_tuple(variables, default=("rfe",))
    #         version = version or "v3.1"
    #         std_name = "PRCP"
    #         url_product = "rfe"
    #         prefix = "rfe"
    #     elif product_key == "soilmoisture":
    #         variables = _as_tuple(variables, default=("smc_avail_top",))
    #         version = version or "v2.3.1"
    #         std_name = "soil_moisture"
    #         url_product = "soil_moisture"
    #         prefix = "sm"
    #     else:
    #         raise ValueError("product must be 'rfe', 'soilmoisture', or 'soil_moisture'.")
    
    #     start = pd.Timestamp(int(year_start), 1, 1) if str(year_start).isdigit() else pd.Timestamp(year_start)
    #     end = pd.Timestamp(int(year_end), 12, 31) if str(year_end).isdigit() else pd.Timestamp(year_end)
    #     if start > end:
    #         raise ValueError("year_start/start date must be <= year_end/end date.")
    
    #     out_nc = dir_to_save / f"Daily_TAMSAT_{product_key}_{start:%Y%m%d}_{end:%Y%m%d}.nc"
    #     if out_nc.exists() and not force_download:
    #         print(f"[INFO] {out_nc} already exists. Skipping.")
    #         return out_nc
    
    #     used_paths = set()
    
    #     def _fetch_day(date):
    #         year, month, day = date.year, date.month, date.day
    #         url = (
    #             f"https://gws-access.jasmin.ac.uk/public/tamsat/{url_product}/data/"
    #             f"{version}/daily/{year}/{month:02d}/{prefix}{year}_{month:02d}_{day:02d}.{version}.nc"
    #         )
    #         path = dir_to_save / url.split("/")[-1]
    #         if not path.exists() or force_download:
    #             ok = _download_atomic(url, path, max_retries, retry_delay)
    #             if not ok:
    #                 return None
    
    #         try:
    #             with xr.open_dataset(path) as ds:
    #                 var_name = _select_variable(ds, variables)
    #                 da = ds[var_name].squeeze(drop=True).astype("float32").load()
    #             da = _standardize_latlon_da(da, area)
    #             da = da.expand_dims(T=[date])
    #             da.name = std_name
    #             used_paths.add(path)  # deletion deferred to the very end
    #             return da
    #         except Exception as exc:
    #             print(f"[ERR] Failed to open/process {path}: {exc} -> removing corrupt file.")
    #             _remove_quiet(path)
    #             return None
    
    #     daily = []
    #     missing = []
    #     for date in pd.date_range(start, end, freq="D"):
    #         da = _fetch_day(date)
    #         if da is not None:
    #             daily.append(da)
    #         else:
    #             missing.append(date)
    
    #     if not daily:
    #         raise RuntimeError(
    #             "No TAMSAT daily files were processed. "
    #             "Downloaded files are kept on disk; re-run to resume."
    #         )
    
    #     if missing:
    #         msg = (
    #             f"{len(missing)} day(s) missing "
    #             f"(first: {missing[0]:%Y-%m-%d}, last: {missing[-1]:%Y-%m-%d})."
    #         )
    #         if not allow_incomplete:
    #             raise RuntimeError(
    #                 f"Incomplete TAMSAT daily download: {msg} The final NetCDF was "
    #                 f"NOT written so it does not block the resume. Files already "
    #                 f"downloaded are kept; re-run to fetch only the missing days, "
    #                 f"or pass allow_incomplete=True to write the available days only."
    #             )
    #         print(f"[WARNING] {msg} Writing available days only (allow_incomplete=True).")
    
    #     ds_out = xr.concat(daily, dim="T").to_dataset(name=std_name).sortby("T")
    #     ds_out.attrs["source"] = "TAMSAT"
    #     ds_out.attrs["product"] = product_key
    #     out_tmp = out_nc.parent / (out_nc.name + ".part")
    #     ds_out.to_netcdf(out_tmp, encoding={std_name: {"zlib": True, "complevel": 4}})
    #     ds_out.close()
    #     os.replace(out_tmp, out_nc)
    
    #     if cleanup:
    #         for path in used_paths:
    #             _remove_quiet(path)
    
    #     print(f"[INFO] Saved daily TAMSAT data to {out_nc}")
    #     return out_nc
    

    def WAS_Download_TAMSAT_Daily(
        self,
        dir_to_save,
        product="rfe",
        variables=None,
        year_start=1983,
        year_end=2024,
        area=None,
        version=None,
        force_download=False,
        cleanup=False,
        max_retries=3,
        retry_delay=5,
        allow_incomplete=False,
    ):
        """
        Download TAMSAT daily data and save a WAS-style NetCDF.
    
        Memory-optimized: monthly checkpoints + dask streaming write. The old
        version accumulated the ENTIRE period in memory before concat, which
        caused out-of-memory failures for long periods / large domains.
    
        Output dims: T, Y, X.
        """
        from pathlib import Path
        import calendar
        import gc
        import os
    
        import pandas as pd
        import xarray as xr
    
        dir_to_save = Path(dir_to_save)
        dir_to_save.mkdir(parents=True, exist_ok=True)
        area = tuple(area)
    
        product_key = "soilmoisture" if product in ["soilmoisture", "soil_moisture"] else product
        if product_key == "rfe":
            variables = _as_tuple(variables, default=("rfe",))
            version = version or "v3.1"
            std_name = "PRCP"
            url_product = "rfe"
            prefix = "rfe"
        elif product_key == "soilmoisture":
            variables = _as_tuple(variables, default=("smc_avail_top",))
            version = version or "v2.3.1"
            std_name = "soil_moisture"
            url_product = "soil_moisture"
            prefix = "sm"
        else:
            raise ValueError("product must be 'rfe', 'soilmoisture', or 'soil_moisture'.")
    
        start = pd.Timestamp(int(year_start), 1, 1) if str(year_start).isdigit() else pd.Timestamp(year_start)
        end = pd.Timestamp(int(year_end), 12, 31) if str(year_end).isdigit() else pd.Timestamp(year_end)
        if start > end:
            raise ValueError("year_start/start date must be <= year_end/end date.")
    
        out_nc = dir_to_save / f"Daily_TAMSAT_{product_key}_{start:%Y%m%d}_{end:%Y%m%d}.nc"
        if out_nc.exists() and not force_download:
            print(f"[INFO] {out_nc} already exists. Skipping.")
            return out_nc
    
        def _fetch_day(date):
            year, month, day = date.year, date.month, date.day
            url = (
                f"https://gws-access.jasmin.ac.uk/public/tamsat/{url_product}/data/"
                f"{version}/daily/{year}/{month:02d}/{prefix}{year}_{month:02d}_{day:02d}.{version}.nc"
            )
            path = dir_to_save / url.split("/")[-1]
            if not path.exists() or force_download:
                if not _download_atomic(url, path, max_retries, retry_delay):
                    return None, None
    
            try:
                with xr.open_dataset(path) as ds:
                    var_name = _select_variable(ds, variables)
                    da = ds[var_name].squeeze(drop=True).astype("float32").load()
                da = _standardize_latlon_da(da, area)
                da = da.expand_dims(T=[date])
                da.name = std_name
                return da, path
            except Exception as exc:
                print(f"[ERR] Failed to open {path}: {exc} -> removing corrupt file.")
                _remove_quiet(path)
                return None, None
    
        # Build the list of (year, month) chunks within [start, end].
        month_starts = pd.date_range(start.normalize().replace(day=1), end, freq="MS")
        if len(month_starts) == 0:
            month_starts = pd.DatetimeIndex([start.normalize().replace(day=1)])
    
        monthly_paths = []
        incomplete_months = []
    
        for month_start in month_starts:
            year, month = month_start.year, month_start.month
            month_path = dir_to_save / f"tmp_TAMSAT_{product_key}_{year}{month:02d}.nc"
            if month_path.exists() and not force_download:
                monthly_paths.append(month_path)
                continue
    
            last_day = calendar.monthrange(year, month)[1]
            d0 = max(start, pd.Timestamp(year, month, 1))
            d1 = min(end, pd.Timestamp(year, month, last_day))
    
            daily = []
            month_files = []
            missing = []
            for date in pd.date_range(d0, d1, freq="D"):
                da, path = _fetch_day(date)
                if da is not None:
                    daily.append(da)
                    month_files.append(path)
                else:
                    missing.append(date)
    
            if not daily or missing:
                print(
                    f"[WARN] Month {year}-{month:02d} incomplete "
                    f"({len(missing)} day(s) missing); files kept for resume."
                )
                incomplete_months.append(f"{year}-{month:02d}")
                del daily
                gc.collect()
                continue
    
            ds_month = xr.concat(daily, dim="T").to_dataset(name=std_name).sortby("T")
            month_tmp = month_path.parent / (month_path.name + ".part")
            ds_month.to_netcdf(month_tmp, encoding={std_name: {"zlib": True, "complevel": 4}})
            ds_month.close()
            os.replace(month_tmp, month_path)
            monthly_paths.append(month_path)
    
            # Monthly checkpoint on disk -> daily source files removable.
            if cleanup:
                for path in month_files:
                    _remove_quiet(path)
    
            del daily, ds_month
            gc.collect()
    
        if not monthly_paths:
            raise RuntimeError("No TAMSAT month was processed; re-run to resume.")
    
        if incomplete_months and not allow_incomplete:
            raise RuntimeError(
                f"Incomplete month(s) {incomplete_months}: final NetCDF NOT written. "
                f"Re-run to fetch only missing days, or pass allow_incomplete=True."
            )
    
        # LAZY final assembly with dask streaming write.
        print(f"[INFO] Assembling {len(monthly_paths)} monthly checkpoint(s) lazily with dask...")
        ds_out = xr.open_mfdataset(
            sorted(monthly_paths),
            combine="nested",
            concat_dim="T",
            chunks={"T": 31},
            parallel=False,
        ).sortby("T")
        ds_out.attrs["source"] = "TAMSAT"
        ds_out.attrs["product"] = product_key
    
        out_tmp = out_nc.parent / (out_nc.name + ".part")
        encoding = {std_name: {"zlib": True, "complevel": 4, "dtype": "float32"}}
        delayed = ds_out.to_netcdf(out_tmp, encoding=encoding, compute=False)
        try:
            from dask.diagnostics import ProgressBar
            with ProgressBar():
                delayed.compute()
        except ImportError:
            delayed.compute()
        ds_out.close()
        os.replace(out_tmp, out_nc)
    
        if cleanup:
            for path in monthly_paths:
                _remove_quiet(path)
    
        print(f"[INFO] Saved daily TAMSAT data to {out_nc}")
        return out_nc


#####

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


