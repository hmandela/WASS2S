"""Agroclimatic predictand computation from daily rainfall.

Implements West African rainfall season onset, cessation, and dry-spell
detection from daily station or gridded precipitation data.  All classes
share a ``compute(daily_data, nb_cores)`` method that parallelises
computation across grid points with joblib.

Classes
-------
WAS_compute_onset
    Season onset detection based on a user-configurable rainfall
    accumulation and dry-spell criteria.
WAS_compute_onset_dry_spell
    Onset detection with an additional post-onset dry-spell check.
WAS_compute_cessation
    Season cessation based on a water-balance (ETP / soil capacity) model.
WAS_compute_cessation_dry_spell
    Cessation detection with a dry-spell termination criterion.
"""
from dask.distributed import Client
import pandas as pd
import xarray as xr
import pandas as pd
import xarray as xr
import numpy as np
import random
import datetime
from joblib import Parallel, delayed
import dask.array as da
from typing import Optional, Union, List, Dict, Tuple
import warnings
from scipy import stats
from dataclasses import dataclass
from enum import Enum


class WAS_compute_onset:
    """
    Class for computing agricultural seasonal onset dates based on cumulative rainfall criteria.

    This class implements a widely used onset detection method in West African agrometeorology,
    particularly for Sahel and Sudanian zones. It supports multiple agro-climatic zones with
    zone-specific onset criteria (start/end search dates, cumulative rainfall threshold,
    maximum dry days allowed, rainy day threshold).

    Key Features:
    - Handles both station-based (CDT format) and gridded (xarray) daily rainfall data
    - Automatically assigns zones based on mean annual rainfall and latitude (Sahel–Guinea)
    - Parallel computation using Dask for large gridded datasets
    - Produces onset dates in day-of-year format (1–365/366)

    Standard Zones & Criteria (default):
    0. Sahel100_0mm:   start 01-Jun, ≥10 mm, ≤25 dry days, end 30-Aug
    1. Sahel200_100mm: start 15-May, ≥15 mm, ≤25 dry days, end 15-Aug
    2. Sahel400_200mm: start 01-May, ≥15 mm, ≤20 dry days, end 31-Jul
    3. Sahel600_400mm: start 15-Mar, ≥20 mm, ≤20 dry days, end 31-Jul
    4. Soudan:         start 15-Mar, ≥20 mm, ≤10 dry days, end 31-Jul
    5. Golfe Of Guinea: start 01-Feb, ≥20 mm, ≤10 dry days, end 15-Jun

    References:
    - AGRHYMET Regional Centre (various operational guidelines)
    - Sivakumar (1988, 1991): Methodology for onset prediction in West Africa
    - Common practice in West African national meteorological services

    Parameters
    ----------
    user_criteria : dict, optional
        Custom zone-specific onset criteria. If None, uses default dictionary.
        Expected format: {zone_id: {"zone_name": str, "start_search": "MM-DD",
        "cumulative": float, "number_dry_days": int,
        "thrd_rain_day": float, "end_search": "MM-DD"}}
    """

    # Default class-level criteria dictionary
    default_criteria = {
        0: {"zone_name": "Sahel100_0mm", "start_search": "06-01", "cumulative": 10, "number_dry_days": 25, "thrd_rain_day": 0.85, "end_search": "08-30"},       
        1: {"zone_name": "Sahel200_100mm", "start_search": "05-15", "cumulative": 15, "number_dry_days": 25, "thrd_rain_day": 0.85, "end_search": "08-15"},
        2: {"zone_name": "Sahel400_200mm", "start_search": "05-01", "cumulative": 15, "number_dry_days": 20, "thrd_rain_day": 0.85, "end_search": "07-31"},
        3: {"zone_name": "Sahel600_400mm", "start_search": "03-15", "cumulative": 20, "number_dry_days": 20, "thrd_rain_day": 0.85, "end_search": "07-31"},
        4: {"zone_name": "Soudan",         "start_search": "03-15", "cumulative": 20, "number_dry_days": 10, "thrd_rain_day": 0.85, "end_search": "07-31"},
        5: {"zone_name": "Golfe_Of_Guinea","start_search": "02-01", "cumulative": 20, "number_dry_days": 10, "thrd_rain_day": 0.85, "end_search": "06-15"},
    }

    def __init__(self, user_criteria=None):
        """
        Initialize the WAS_compute_onset class with user-defined or default criteria.

        Parameters
        ----------
        user_criteria : dict, optional
            A dictionary containing zone-specific criteria. If not provided,
            the class will use the default criteria.
        """
        if user_criteria:
            self.criteria = user_criteria
        else:
            self.criteria = WAS_compute_onset.default_criteria

    @staticmethod
    def adjust_duplicates(series, increment=0.00001):
        """
        If any values in the Series repeat, nudge them by a tiny increment
        so that all are unique (to avoid indexing collisions).
        """
        counts = series.value_counts()
        for val, count in counts[counts > 1].items():
            duplicates = series[series == val].index
            for i, idx in enumerate(duplicates):
                series.at[idx] += increment * i
        return series

 
    @staticmethod
    def transform_cdt(df):
        """
        Transform a DataFrame with:
          - Row 0 = LON
          - Row 1 = LAT
          - Row 2 = ELEV
          - Rows 3+ = daily data (or any date) with 'ID' column containing dates.

        Returns an xarray DataArray with coords = (T, Y, X), variable = 'Observation'.
        """
        # --- 1) Extract metadata (first 3 rows) ---
        metadata = df.iloc[:3].set_index("ID").T.reset_index()
        metadata.columns = ["STATION", "LON", "LAT", "ELEV"]
        
        # Adjust duplicates
        metadata["LON"] = WAS_compute_onset.adjust_duplicates(metadata["LON"])
        metadata["LAT"] = WAS_compute_onset.adjust_duplicates(metadata["LAT"])
        metadata["ELEV"] = WAS_compute_onset.adjust_duplicates(metadata["ELEV"])

        # --- 2) Extract actual data, rename ID -> DATE ---
        data_part = df.iloc[3:].rename(columns={"ID": "DATE"})

        # Melt to long form
        data_long = data_part.melt(id_vars=["DATE"], var_name="STATION", value_name="VALUE")

        # Merge metadata back into the melted data
        final_df = pd.merge(data_long, metadata, on="STATION")

        # Ensure DATE is treated as a proper datetime
        final_df["DATE"] = pd.to_datetime(final_df["DATE"], format="%Y%m%d")

        # Create a complete date range from the first day of the first year to the last day of the last year in the data
        start_date = final_df["DATE"].min().replace(month=1, day=1)
        end_date = final_df["DATE"].max().replace(month=12, day=31)
        date_range = pd.date_range(start=start_date, end=end_date)

        # Create a DataFrame with all combinations of dates and stations
        all_combinations = pd.MultiIndex.from_product([date_range, metadata["STATION"]], names=["DATE", "STATION"]).to_frame(index=False)

        # Merge the complete date-station combinations with the final_df to ensure all dates are present
        final_df = pd.merge(all_combinations, final_df, on=["DATE", "STATION"], how="left")

        # Fill missing values in the VALUE column with -99.0
        final_df["VALUE"] = final_df["VALUE"].fillna(-99.0)

        # Remove invalid rainfall values before computing the mean
        # Calculate the annual rainfall by summing the values for each year and station
        annual_rainfall = final_df[final_df["VALUE"] >= 0].groupby(["STATION", final_df["DATE"].dt.year])["VALUE"].sum().reset_index()

        # Calculate the mean annual rainfall for each station
        mean_annual_rainfall = annual_rainfall.groupby("STATION")["VALUE"].mean().reset_index()
        mean_annual_rainfall.columns = ["STATION", "MEAN_ANNUAL_RAINFALL"]

        # Merge back into final_df
        final_df = pd.merge(final_df, mean_annual_rainfall, on="STATION", how="left")

        # Generate the zonename column based on the conditions for each station
        def determine_zonename(row):
            if row["LAT"] <= 8:
                return 5
            elif 600 >= row["MEAN_ANNUAL_RAINFALL"] > 400:
                return 3
            elif 400 >= row["MEAN_ANNUAL_RAINFALL"] > 200:
                return 2
            elif 200 >= row["MEAN_ANNUAL_RAINFALL"] > 100:
                return 1
            elif 100 >= row["MEAN_ANNUAL_RAINFALL"] > 0:
                return 0
            else:
                return 4

        final_df["zonename"] = final_df.groupby("STATION", group_keys=False).apply(
            lambda x: x.apply(determine_zonename, axis=1)
        )

        return final_df


    @staticmethod
    def day_of_year(i, dem_rech1):
        """
        Given a year 'i' and a month-day string 'dem_rech1' (e.g., '07-23'),
        return the day of the year (1-based).
        """
        year = int(i)
        full_date_str = f"{year}-{dem_rech1}"
        current_date = datetime.datetime.strptime(full_date_str, "%Y-%m-%d").date()
        origin_date = datetime.date(year, 1, 1)
        day_of_year_value = (current_date - origin_date).days + 1
        return day_of_year_value

    def rainf_zone(self, daily_data):
        annual_rainfall = daily_data.resample(T="YE").sum(skipna=True).mean(dim='T')
        mask_5 = annual_rainfall.where(abs(annual_rainfall.Y) <= 8, np.nan)
        mask_5 = xr.where(np.isnan(mask_5), np.nan, 5) 
        mask_4 = xr.where(
            (abs(annual_rainfall.Y) > 8) 
            &
            ((annual_rainfall >= 600)),  
            4,
            np.nan
            )
        mask_3 = xr.where(
            (annual_rainfall < 600) & (annual_rainfall >= 400),
            3,
            np.nan 
            )
        mask_2 = xr.where(
            (annual_rainfall < 400) & (annual_rainfall >= 200),
            2,
            np.nan 
            )
        mask_1 = xr.where(
            (annual_rainfall < 200) & (annual_rainfall >= 100),
            1,np.nan 
            )
        mask_0 = xr.where(
            (annual_rainfall < 100) & (annual_rainfall >= 75),
            0,np.nan 
            )
        return mask_5.fillna(mask_4).fillna(mask_3).fillna(mask_2).fillna(mask_1).fillna(mask_0)
        
    def onset_function(self, x, idebut, cumul, nbsec, jour_pluvieux, irch_fin):
        """
        Calculate the onset date of a season based on cumulative rainfall criteria.

        Parameters
        ----------
        x : array-like
            Daily rainfall or similar values.
        idebut : int
            Start index to begin searching for the onset.
        cumul : float
            Cumulative rainfall threshold to trigger onset.
        nbsec : int
            Maximum number of dry days allowed in the sequence.
        jour_pluvieux : float
            Minimum rainfall to consider a day as rainy.
        irch_fin : int
            Maximum index limit for the onset.

        Returns
        -------
        int or float
            Index of the onset date or NaN if onset not found.
        """
        mask = (np.any(np.isfinite(x)) and 
                np.isfinite(idebut) and 
                np.isfinite(nbsec) and 
                np.isfinite(irch_fin))

        if mask:
            idebut = int(idebut)
            nbsec = int(nbsec)
            irch_fin = int(irch_fin)

            trouv = 0
            idate = idebut

            while True:
                idate += 1
                ipreced = idate - 1
                isuiv = idate + 1

                # Check for missing data or out-of-bounds
                if (ipreced >= len(x) or 
                    idate >= len(x) or 
                    isuiv >= len(x) or 
                    pd.isna(x[ipreced]) or 
                    pd.isna(x[idate]) or 
                    pd.isna(x[isuiv])):
                    deb_saison = np.nan
                    break

                # Check for end search of date
                if idate > irch_fin:
                    deb_saison = random.randint(irch_fin - 5, irch_fin)
                    break

                # Calculate cumulative rainfall over 1, 2, and 3 days
                cumul3jr = x[ipreced] + x[idate] + x[isuiv]
                cumul2jr = x[ipreced] + x[idate]
                cumul1jr = x[ipreced]

                # Check if any cumulative rainfall meets the threshold
                if (cumul1jr >= cumul or 
                    cumul2jr >= cumul or 
                    cumul3jr >= cumul):
                    troisp = np.array([x[ipreced], x[idate], x[isuiv]])
                    itroisp = np.array([ipreced, idate, isuiv])
                    maxp = np.nanmax(troisp)
                    imaxp = np.where(troisp == maxp)[0][0]
                    ideb = itroisp[imaxp]
                    deb_saison = ideb
                    trouv = 1

                    # Check for sequences of dry days within the next 30 days
                    finp = ideb + 30
                    pluie30jr = x[ideb:finp + 1] if finp < len(x) else x[ideb:]
                    isec = 0

                    while True:
                        isec += 1
                        isecf = isec + nbsec
                        if isecf >= len(pluie30jr):
                            break
                        donneeverif = pluie30jr[isec:isecf + 1]

                        # Count days with rainfall below jour_pluvieux
                        test1 = np.sum(donneeverif < jour_pluvieux)

                        # If a dry sequence is found, reset trouv to 0
                        if test1 == (nbsec + 1):
                            trouv = 0

                        # Break if a dry sequence is found or we've reached the end of the window
                        if test1 == (nbsec + 1) or isec == (30 - nbsec):
                            break

                # Break if onset is found
                if trouv == 1:
                    break
        else:
            deb_saison = np.nan

        return deb_saison

    
    def compute_insitu(self, daily_df,):
        daily_df = self.transform_cdt(daily_df)

        unique_stations = daily_df["STATION"].unique()
        unique_years = daily_df["DATE"].dt.year.unique()
        unique_zonenames = daily_df["zonename"].unique()

        results = []

        for year in unique_years:
            for station in unique_stations:
                # Filter data for the current station and year
                station_data = daily_df[(daily_df["STATION"] == station) & (daily_df["DATE"].dt.year == year)]
                # Replace missing values with NaN
                station_data.loc[:, "VALUE"] = station_data["VALUE"].replace(-99.0, np.nan)
                # Extract unique zonenames
                unique_zonenames = station_data["zonename"].unique()
                # Extract the onset criteria for the current zonename
                idebut = self.day_of_year(year, self.criteria[unique_zonenames[0]]["start_search"])
                irch_fin = self.day_of_year(year, self.criteria[unique_zonenames[0]]["end_search"])
                cumul = self.criteria[unique_zonenames[0]]["cumulative"]
                nbsec = self.criteria[unique_zonenames[0]]["number_dry_days"]
                jour_pluvieux = self.criteria[unique_zonenames[0]]["thrd_rain_day"]
                # Compute the onset date
                onset_date = self.onset_function(station_data["VALUE"].values, idebut, cumul, nbsec, jour_pluvieux, irch_fin)
                
                results.append({
                    "year": year,
                    "station": station,
                    "lon": station_data["LON"].iloc[0],
                    "lat": station_data["LAT"].iloc[0],
                    "onset": onset_date
                })
        # Convert results to a DataFrame
        onset_df = pd.DataFrame(results)
        final_df = onset_df
        final_df["onset"] = final_df["onset"].fillna(-999)

        # transform the onset_df to the CPT format
        # Extract unique stations and their corresponding lat/lon
        station_metadata = onset_df.groupby("station")[["lat", "lon"]].first().reset_index()

        # Pivot df_yyy to match the wide format (years as rows, stations as columns)
        df_pivot = onset_df.pivot(index="year", columns="station", values="onset")

        # Extract latitude and longitude values based on station order in pivoted DataFrame
        lat_row = pd.DataFrame([["LAT"] + station_metadata.set_index("station").loc[df_pivot.columns, "lat"].tolist()], 
                            columns=["STATION"] + df_pivot.columns.tolist())

        lon_row = pd.DataFrame([["LON"] + station_metadata.set_index("station").loc[df_pivot.columns, "lon"].tolist()], 
                            columns=["STATION"] + df_pivot.columns.tolist())

        # Reset index to ensure correct structure
        df_pivot.reset_index(inplace=True)

        # Rename the "year" column to "STATION" to match the required format
        df_pivot.rename(columns={"year": "STATION"}, inplace=True)

        # Concatenate latitude, longitude, and pivoted onset values to form the final structure
        df_final = pd.concat([lat_row, lon_row, df_pivot], ignore_index=True)

        return df_final


    def compute(self, daily_data, nb_cores):
        """
        Compute onset dates for each pixel in a given daily rainfall DataArray
        using different criteria based on isohyet zones.

        Parameters
        ----------
        daily_data : xarray.DataArray
            Daily rainfall data, coords = (T, Y, X).
        nb_cores : int
            Number of parallel processes to use.

        Returns
        -------
        xarray.DataArray
            Array with onset dates computed per pixel.
        """
        # # Load zone file & slice it
        # mask_char = xr.open_dataset('./utilities/Isohyet_zones.nc')
        # mask_char = mask_char.sel(X=slice(extent[1], extent[3]),
        #                           Y=slice(extent[0], extent[2]))
        # mask_char = mask_char.isel(Y=slice(None, None, -1)).to_array().drop_vars('variable').squeeze()

        # daily_data = daily_data.sel(
        #     X=mask_char.coords['X'],
        #     Y=mask_char.coords['Y'])

        # mask_ = xr.where(daily_data.resample(T="YE").sum(skipna=True).mean(dim='T') < 75, np.nan, 1)
        
        mask_char = self.rainf_zone(daily_data)
        # Get unique zone IDs
        unique_zone = np.unique(mask_char.to_numpy())
        unique_zone = unique_zone[~np.isnan(unique_zone)]

        # Compute year range and partial T dimension (start_search)
        years = np.unique(daily_data['T'].dt.year.to_numpy())

        # Choose a date to store results
        if unique_zone.size == 0:
            raise ValueError("No valid zones found in the mask.")
        else:
            # Use zone in low latitude
            zone_id_to_use = int(np.max(unique_zone))
        
        T_from_here = daily_data.sel(T=[f"{str(i)}-{self.criteria[zone_id_to_use]['start_search']}" for i in years])

        # Prepare chunk sizes
        chunksize_x = int(np.round(len(daily_data.get_index("X")) / nb_cores))
        chunksize_y = int(np.round(len(daily_data.get_index("Y")) / nb_cores))

        # Initialize placeholders
        mask_char_start_search = mask_char_cumulative = mask_char_number_dry_days = mask_char_thrd_rain_day = mask_char_end_search = mask_char

        store_onset = []
        for i in years:
            for j in unique_zone:
                # Replace zone values with numeric parameters
                mask_char_start_search = xr.where(
                    mask_char_start_search == j,
                    self.day_of_year(i, self.criteria[j]["start_search"]),
                    mask_char_start_search
                )
                mask_char_cumulative = xr.where(
                    mask_char_cumulative == j,
                    self.criteria[j]["cumulative"],
                    mask_char_cumulative
                )
                mask_char_number_dry_days = xr.where(
                    mask_char_number_dry_days == j,
                    self.criteria[j]["number_dry_days"],
                    mask_char_number_dry_days
                )
                mask_char_thrd_rain_day = xr.where(
                    mask_char_thrd_rain_day == j,
                    self.criteria[j]["thrd_rain_day"],
                    mask_char_thrd_rain_day
                )
                mask_char_end_search = xr.where(
                    mask_char_end_search == j,
                    self.day_of_year(i, self.criteria[j]["end_search"]),
                    mask_char_end_search
                )

            # Select data for this particular year
            year_data = daily_data.sel(T=str(i))

            # Set up parallel processing
            client = Client(n_workers=nb_cores, threads_per_worker=1)
            result = xr.apply_ufunc(
                self.onset_function,  # <-- Now calling via self
                year_data.chunk({'Y': chunksize_y, 'X': chunksize_x}),
                mask_char_start_search.chunk({'Y': chunksize_y, 'X': chunksize_x}),
                mask_char_cumulative.chunk({'Y': chunksize_y, 'X': chunksize_x}),
                mask_char_number_dry_days.chunk({'Y': chunksize_y, 'X': chunksize_x}),
                mask_char_thrd_rain_day.chunk({'Y': chunksize_y, 'X': chunksize_x}),
                mask_char_end_search.chunk({'Y': chunksize_y, 'X': chunksize_x}),
                input_core_dims=[('T',), (), (), (), (), ()],
                vectorize=True,
                output_core_dims=[()],
                dask='parallelized',
                output_dtypes=['float'],
            )
            result_ = result.compute()
            client.close()

            store_onset.append(result_)

        # Concatenate final result
        store_onset = xr.concat(store_onset, dim="T")
        store_onset['T'] = T_from_here['T']
        store_onset.name = "Onset"

        return store_onset#.to_array().drop_vars('variable').squeeze('variable')

class WAS_compute_onset_dry_spell:
    """
    Class for computing the **longest dry spell length after the onset** of the rainy season.

    This class extends standard onset detection by calculating the maximum consecutive dry days
    within a user-defined window (``nbjour``) following the detected onset date.

    The onset detection follows the same cumulative rainfall + dry-day tolerance method used
    in West African agrometeorology (AGRHYMET, national services), with zone-specific criteria.

    Key Features:
    - Computes onset date (same logic as ``WAS_compute_onset``)
    - Then finds the longest dry spell (consecutive days ≤ ``thrd_rain_day`` mm) within the next ``nbjour`` days
    - Supports station (CDT) and gridded (xarray) daily rainfall data
    - Automatically assigns agro-climatic zones based on mean annual rainfall and latitude
    - Parallel computation with Dask for large grids

    Default Zones & Criteria:
    0. Sahel100_0mm     → start 01-Jun, ≥10 mm, ≤25 dry days, check 40 days after onset
    1. Sahel200_100mm   → start 15-May, ≥15 mm, ≤25 dry days, check 40 days
    2. Sahel400_200mm   → start 01-May, ≥15 mm, ≤20 dry days, check 40 days
    3. Sahel600_400mm   → start 15-Mar, ≥20 mm, ≤20 dry days, check 45 days
    4. Soudan           → start 15-Mar, ≥20 mm, ≤10 dry days, check 50 days
    5. Golfe of Guinea  → start 01-Feb, ≥20 mm, ≤10 dry days, check 50 days

    References:
    - AGRHYMET Regional Centre operational methodologies
    - Sivakumar (1988, 1991): Onset and dry spell analysis in West Africa
    - Common practice in Sahelian and Sudanian national meteorological services

    Parameters
    ----------
    user_criteria : dict, optional
        Custom zone-specific criteria. If None, uses default dictionary.
        Expected keys: zone_id → {"zone_name": str, "start_search": "MM-DD",
        "cumulative": float, "number_dry_days": int,
        "thrd_rain_day": float, "end_search": "MM-DD",
        "nbjour": int}
    """

    # Default class-level criteria dictionary
    default_criteria = {
        0: {"zone_name": "Sahel100_0mm", "start_search": "06-01", "cumulative": 10, "number_dry_days": 25, "thrd_rain_day": 0.85, "end_search": "08-30", "nbjour":40},
        1: {"zone_name": "Sahel200_100mm", "start_search": "05-15", "cumulative": 15, "number_dry_days": 25, "thrd_rain_day": 0.85, "end_search": "08-15", "nbjour":40},
        2: {"zone_name": "Sahel400_200mm", "start_search": "05-01", "cumulative": 15, "number_dry_days": 20, "thrd_rain_day": 0.85, "end_search": "07-31", "nbjour":40},
        3: {"zone_name": "Sahel600_400mm", "start_search": "03-15", "cumulative": 20, "number_dry_days": 20, "thrd_rain_day": 0.85, "end_search": "07-31", "nbjour":45},
        4: {"zone_name": "Soudan",         "start_search": "03-15", "cumulative": 20, "number_dry_days": 10, "thrd_rain_day": 0.85, "end_search": "07-31", "nbjour":50},
        5: {"zone_name": "Golfe_Of_Guinea","start_search": "02-01", "cumulative": 20, "number_dry_days": 10, "thrd_rain_day": 0.85, "end_search": "06-15", "nbjour":50},
    }

    
    def __init__(self, user_criteria=None):
        """
        Initialize the WAS_compute_dry_spell class with user-defined or default criteria.

        Parameters
        ----------
        user_criteria : dict, optional
            A dictionary containing zone-specific criteria. If not provided,
            the class will use the default criteria.
        """
        if user_criteria:
            self.criteria = user_criteria
        else:
            self.criteria = WAS_compute_onset_dry_spell.default_criteria


    @staticmethod
    def adjust_duplicates(series, increment=0.00001):
        """
        If any values in the Series repeat, nudge them by a tiny increment
        so that all are unique (to avoid indexing collisions).
        """
        counts = series.value_counts()
        for val, count in counts[counts > 1].items():
            duplicates = series[series == val].index
            for i, idx in enumerate(duplicates):
                series.at[idx] += increment * i
        return series

    @staticmethod
    def transform_cdt(df):
        """
        Transform a DataFrame with:
          - Row 0 = LON
          - Row 1 = LAT
          - Row 2 = ELEV
          - Rows 3+ = daily data (or any date) with 'ID' column containing dates.

        Returns an xarray DataArray with coords = (T, Y, X), variable = 'Observation'.
        """
        # --- 1) Extract metadata (first 3 rows) ---
        metadata = df.iloc[:3].set_index("ID").T.reset_index()
        metadata.columns = ["STATION", "LON", "LAT", "ELEV"]
        
        # Adjust duplicates
        metadata["LON"] = WAS_compute_onset.adjust_duplicates(metadata["LON"])
        metadata["LAT"] = WAS_compute_onset.adjust_duplicates(metadata["LAT"])
        metadata["ELEV"] = WAS_compute_onset.adjust_duplicates(metadata["ELEV"])

        # --- 2) Extract actual data, rename ID -> DATE ---
        data_part = df.iloc[3:].rename(columns={"ID": "DATE"})

        # Melt to long form
        data_long = data_part.melt(id_vars=["DATE"], var_name="STATION", value_name="VALUE")

        # Merge metadata back into the melted data
        final_df = pd.merge(data_long, metadata, on="STATION")

        # Ensure DATE is treated as a proper datetime
        final_df["DATE"] = pd.to_datetime(final_df["DATE"], format="%Y%m%d")

        # Create a complete date range from the first day of the first year to the last day of the last year in the data
        start_date = final_df["DATE"].min().replace(month=1, day=1)
        end_date = final_df["DATE"].max().replace(month=12, day=31)
        date_range = pd.date_range(start=start_date, end=end_date)

        # Create a DataFrame with all combinations of dates and stations
        all_combinations = pd.MultiIndex.from_product([date_range, metadata["STATION"]], names=["DATE", "STATION"]).to_frame(index=False)

        # Merge the complete date-station combinations with the final_df to ensure all dates are present
        final_df = pd.merge(all_combinations, final_df, on=["DATE", "STATION"], how="left")

        # Fill missing values in the VALUE column with -99.0
        final_df["VALUE"] = final_df["VALUE"].fillna(-99.0)

        # Remove invalid rainfall values before computing the mean
        # Calculate the annual rainfall by summing the values for each year and station
        annual_rainfall = final_df[final_df["VALUE"] >= 0].groupby(["STATION", final_df["DATE"].dt.year])["VALUE"].sum().reset_index()

        # Calculate the mean annual rainfall for each station
        mean_annual_rainfall = annual_rainfall.groupby("STATION")["VALUE"].mean().reset_index()
        mean_annual_rainfall.columns = ["STATION", "MEAN_ANNUAL_RAINFALL"]

        # Merge back into final_df
        final_df = pd.merge(final_df, mean_annual_rainfall, on="STATION", how="left")

        # Generate the zonename column based on the conditions for each station
        def determine_zonename(row):
            if row["LAT"] <= 8:
                return 5
            elif 600 >= row["MEAN_ANNUAL_RAINFALL"] > 400:
                return 3
            elif 400 >= row["MEAN_ANNUAL_RAINFALL"] > 200:
                return 2
            elif 200 >= row["MEAN_ANNUAL_RAINFALL"] > 100:
                return 1
            elif 100 >= row["MEAN_ANNUAL_RAINFALL"] > 75:
                return 0
            else:
                return 4

        final_df["zonename"] = final_df.groupby("STATION", group_keys=False).apply(
            lambda x: x.apply(determine_zonename, axis=1)
        )

        return final_df

    def rainf_zone(self, daily_data):
        annual_rainfall = daily_data.resample(T="YE").sum(skipna=True).mean(dim='T')
        mask_5 = annual_rainfall.where(abs(annual_rainfall.Y) <= 8, np.nan)
        mask_5 = xr.where(np.isnan(mask_5), np.nan, 5) 
        mask_4 = xr.where(
            (abs(annual_rainfall.Y) > 8) 
            &
            ((annual_rainfall >= 600)),  
            4,
            np.nan
            )
        mask_3 = xr.where(
            (annual_rainfall < 600) & (annual_rainfall >= 400),
            3,
            np.nan 
            )
        mask_2 = xr.where(
            (annual_rainfall < 400) & (annual_rainfall >= 200),
            2,
            np.nan 
            )
        mask_1 = xr.where(
            (annual_rainfall < 200) & (annual_rainfall >= 100),
            1,np.nan 
            )
        mask_0 = xr.where(
            (annual_rainfall < 100) & (annual_rainfall >= 75),
            0,np.nan 
            )
        # Fill NaN values with the next available value
        return mask_5.fillna(mask_4).fillna(mask_3).fillna(mask_2).fillna(mask_1).fillna(mask_0)

    def dry_spell_onset_function(self, x, idebut, cumul, nbsec, jour_pluvieux, irch_fin, nbjour):
        """
        Calculate the onset date of a season based on cumulative rainfall criteria, and
        determine the longest dry spell sequence within a specified period after the onset.
        """
        seq_max = np.nan  # <-- Always defined
        mask = (np.isfinite(x).any() and 
                np.isfinite(idebut) and 
                np.isfinite(nbsec) and 
                np.isfinite(irch_fin) and
                np.isfinite(nbjour))
    
        if mask:
            idebut = int(idebut)
            nbsec = int(nbsec)
            irch_fin = int(irch_fin)
            nbjour = int(nbjour)
            trouv = 0
            idate = idebut
            deb_saison = np.nan  # <--- Initialize here too
    
            while True:
                idate += 1
                ipreced = idate - 1
                isuiv = idate + 1
    
                if (ipreced >= len(x) or idate >= len(x) or isuiv >= len(x) or
                    pd.isna(x[ipreced]) or pd.isna(x[idate]) or pd.isna(x[isuiv])):
                    break
    
                if idate > irch_fin:
                    # deb_saison = random.randint(max(idebut, irch_fin - 5), irch_fin)
                    deb_saison = random.randint(irch_fin - 5, irch_fin)
                    break
    
                cumul3jr = x[ipreced] + x[idate] + x[isuiv]
                cumul2jr = x[ipreced] + x[idate]
                cumul1jr = x[ipreced]
    
                if (cumul1jr >= cumul or cumul2jr >= cumul or cumul3jr >= cumul):
                    troisp = np.array([x[ipreced], x[idate], x[isuiv]])
                    itroisp = np.array([ipreced, idate, isuiv])
                    maxp = np.nanmax(troisp)
                    imaxp = np.where(troisp == maxp)[0][0]
                    ideb = itroisp[imaxp]
                    deb_saison = ideb
                    trouv = 1
    
                    finp = ideb + 30
                    pluie30jr = x[ideb:finp + 1] if finp < len(x) else x[ideb:]
                    isec = 0
    
                    while True:
                        isec += 1
                        isecf = isec + nbsec
                        if isecf >= len(pluie30jr):
                            break
                        donneeverif = pluie30jr[isec:isecf + 1]
                        test1 = np.sum(donneeverif < jour_pluvieux)
    
                        if test1 == (nbsec + 1):
                            trouv = 0
                            break
    
                        if isec == (30 - nbsec):
                            break
    
                if trouv == 1:
                    break
    
            if not np.isnan(deb_saison):
                pluie_nbjour = x[int(deb_saison):min(int(deb_saison) + nbjour + 1, len(x))]
                rainy_days = np.where(pluie_nbjour > jour_pluvieux)[0]
                d1 = np.array([0] + list(rainy_days))
                d2 = np.array(list(rainy_days) + [len(pluie_nbjour)])
                seq_max = np.max(d2 - d1) - 1
    
        return seq_max
    
    
    def dry_spell_onset_function_(self, x, idebut, cumul, nbsec, jour_pluvieux, irch_fin, nbjour):
        """
        Calculate the onset date of a season based on cumulative rainfall criteria, and
        determine the longest dry spell sequence within a specified period after the onset.

        Parameters
        ----------
        x : array-like
            Daily rainfall or similar values.
        idebut : int
            Start index to begin searching for the onset.
        cumul : float
            Cumulative rainfall threshold to trigger onset.
        nbsec : int
            Maximum number of dry days allowed in the sequence.
        jour_pluvieux : float
            Minimum rainfall to consider a day as rainy.
        irch_fin : int
            Maximum index limit for the onset.
        nbjour : int
            Number of days to check for the longest dry spell after onset.

        Returns
        -------
        float
            Length of the longest dry spell sequence after onset or NaN if onset not found.
        """
        # Ensure all input values are valid
        mask = (np.isfinite(x).any() and 
                np.isfinite(idebut) and 
                np.isfinite(nbsec) and 
                np.isfinite(irch_fin) and
                np.isfinite(nbjour))

        if mask:
            idebut = int(idebut)
            nbsec = int(nbsec)
            irch_fin = int(irch_fin)
            nbjour = int(nbjour)
            trouv = 0
            idate = idebut

            while True:
                idate += 1
                ipreced = idate - 1
                isuiv = idate + 1

                # Check for missing data or out-of-bounds
                if (ipreced >= len(x) or 
                    idate >= len(x) or 
                    isuiv >= len(x) or 
                    pd.isna(x[ipreced]) or 
                    pd.isna(x[idate]) or 
                    pd.isna(x[isuiv])):
                    deb_saison = np.nan
                    break

                # Check for end search of date
                if idate > irch_fin:
                    deb_saison = random.randint(irch_fin - 5, irch_fin)
                    break

                # Calculate cumulative rainfall over 1, 2, and 3 days
                cumul3jr = x[ipreced] + x[idate] + x[isuiv]
                cumul2jr = x[ipreced] + x[idate]
                cumul1jr = x[ipreced]

                # Check if any cumulative rainfall meets the threshold
                if (cumul1jr >= cumul or 
                    cumul2jr >= cumul or 
                    cumul3jr >= cumul):
                    troisp = np.array([x[ipreced], x[idate], x[isuiv]])
                    itroisp = np.array([ipreced, idate, isuiv])
                    maxp = np.nanmax(troisp)
                    imaxp = np.where(troisp == maxp)[0][0]
                    ideb = itroisp[imaxp]
                    deb_saison = ideb
                    trouv = 1

                    # Check for sequences of dry days within the next 30 days
                    finp = ideb + 30
                    pluie30jr = x[ideb:finp + 1] if finp < len(x) else x[ideb:]
                    isec = 0

                    while True:
                        isec += 1
                        isecf = isec + nbsec
                        if isecf >= len(pluie30jr):
                            break
                        donneeverif = pluie30jr[isec:isecf + 1]

                        # Count days with rainfall below jour_pluvieux
                        test1 = np.sum(donneeverif < jour_pluvieux)

                        # If a dry sequence is found, reset trouv to 0
                        if test1 == (nbsec + 1):
                            trouv = 0

                        # Break if a dry sequence is found or we've reached the end of the window
                        if test1 == (nbsec + 1) or isec == (30 - nbsec):
                            break

                # Break if onset is found
                if trouv == 1:
                    break

            # Compute the longest dry spell within ``nbjour`` days after the onset
            if not np.isnan(deb_saison):
                pluie_nbjour = x[int(deb_saison) : min(int(deb_saison) + nbjour + 1, len(x))]
                rainy_days = np.where(pluie_nbjour > jour_pluvieux)[0]
                # Build two arrays to measure intervals between rainy days
                d1 = np.array([0] + list(rainy_days))
                d2 = np.array(list(rainy_days) + [len(pluie_nbjour)])
                seq_max = np.max(d2 - d1) - 1  # -1 so that the difference is the gap
        else:
            seq_max = np.nan
        return seq_max

    @staticmethod
    def day_of_year(i, dem_rech1):
        """
        Given a year 'i' and a month-day string 'dem_rech1' (e.g., '07-23'),
        return the 1-based day of the year.
        """
        year = int(i)
        full_date_str = f"{year}-{dem_rech1}"
        current_date = datetime.datetime.strptime(full_date_str, "%Y-%m-%d").date()
        origin_date = datetime.date(year, 1, 1)
        day_of_year_value = (current_date - origin_date).days + 1
        return day_of_year_value

    def compute_insitu(self, daily_df,):
        daily_df = self.transform_cdt(daily_df)

        unique_stations = daily_df["STATION"].unique()
        unique_years = daily_df["DATE"].dt.year.unique()
        unique_zonenames = daily_df["zonename"].unique()

        results = []

        for year in unique_years:
            for station in unique_stations:
                # Filter data for the current station and year
                station_data = daily_df[(daily_df["STATION"] == station) & (daily_df["DATE"].dt.year == year)]
                # Replace missing values with NaN
                station_data.loc[:, "VALUE"] = station_data["VALUE"].replace(-99.0, np.nan)
                # Extract unique zonenames
                unique_zonenames = station_data["zonename"].unique()
                # x, idebut, cumul, nbsec, jour_pluvieux, irch_fin, nbjour
                # Extract the onset criteria for the current zonename
                idebut = self.day_of_year(year, self.criteria[unique_zonenames[0]]["start_search"])
                irch_fin = self.day_of_year(year, self.criteria[unique_zonenames[0]]["end_search"])
                cumul = self.criteria[unique_zonenames[0]]["cumulative"]
                nbsec = self.criteria[unique_zonenames[0]]["number_dry_days"]
                jour_pluvieux = self.criteria[unique_zonenames[0]]["thrd_rain_day"]
                nbjour = self.criteria[unique_zonenames[0]]["nbjour"]
                # Compute the onset date
                onset_dryspell = self.dry_spell_onset_function(station_data["VALUE"].values, idebut, cumul, nbsec, jour_pluvieux, irch_fin, nbjour)
                
                results.append({
                    "year": year,
                    "station": station,
                    "lon": station_data["LON"].iloc[0],
                    "lat": station_data["LAT"].iloc[0],
                    "onsetdryspell": onset_dryspell
                })
        # Convert results to a DataFrame
        onset_df = pd.DataFrame(results)
        final_df = onset_df
        final_df["onsetdryspell"] = final_df["onsetdryspell"].fillna(-999)
 
        # transform the onset_df to the CPT format
        # Extract unique stations and their corresponding lat/lon
        station_metadata = onset_df.groupby("station")[["lat", "lon"]].first().reset_index()

        # Pivot df_yyy to match the wide format (years as rows, stations as columns)
        df_pivot = onset_df.pivot(index="year", columns="station", values="onsetdryspell")

        # Extract latitude and longitude values based on station order in pivoted DataFrame
        lat_row = pd.DataFrame([["LAT"] + station_metadata.set_index("station").loc[df_pivot.columns, "lat"].tolist()], 
                            columns=["STATION"] + df_pivot.columns.tolist())

        lon_row = pd.DataFrame([["LON"] + station_metadata.set_index("station").loc[df_pivot.columns, "lon"].tolist()], 
                            columns=["STATION"] + df_pivot.columns.tolist())

        # Reset index to ensure correct structure
        df_pivot.reset_index(inplace=True)

        # Rename the "year" column to "STATION" to match the required format
        df_pivot.rename(columns={"year": "STATION"}, inplace=True)

        # Concatenate latitude, longitude, and pivoted onset values to form the final structure
        df_final = pd.concat([lat_row, lon_row, df_pivot], ignore_index=True)

        return df_final

    def compute(self, daily_data, nb_cores):
        """
        Compute the longest dry spell length after the onset for each pixel in a
        given daily rainfall DataArray, using different criteria based on isohyet zones.

        Parameters
        ----------
        daily_data : xarray.DataArray
            Daily rainfall data, coords = (T, Y, X).
        nb_cores : int
            Number of parallel processes to use.

        Returns
        -------
        xarray.DataArray
            Array with the longest dry spell length per pixel.
        """
        # # Load zone file & slice it to the area of interest
        # mask_char = xr.open_dataset('./utilities/Isohyet_zones.nc')
        # mask_char = mask_char.sel(X=slice(extent[1], extent[3]),
        #                           Y=slice(extent[0], extent[2]))
        
        # # Flip Y if needed
        # mask_char = mask_char.isel(Y=slice(None, None, -1)).to_array().drop_vars('variable').squeeze()
        
        # daily_data = daily_data.sel(
        #     X=mask_char.coords['X'],
        #     Y=mask_char.coords['Y'])

        mask_char = self.rainf_zone(daily_data)

        # Get unique zone IDs
        unique_zone = np.unique(mask_char.to_numpy())
        unique_zone = unique_zone[~np.isnan(unique_zone)]

        # Compute year range
        years = np.unique(daily_data['T'].dt.year.to_numpy())

        # Create T dimension for the earliest (or any) zone's start date as reference
        zone_id_to_use = int(np.max(unique_zone))  # or some logic of your choosing
        T_from_here = daily_data.sel(T=[f"{str(i)}-{self.criteria[zone_id_to_use]['start_search']}" for i in years])

        # Prepare chunk sizes
        chunksize_x = int(np.round(len(daily_data.get_index("X")) / nb_cores))
        chunksize_y = int(np.round(len(daily_data.get_index("Y")) / nb_cores))

        # Initialize placeholders
        mask_char_start_search = mask_char_cumulative = mask_char_number_dry_days = mask_char_thrd_rain_day = mask_char_end_search = mask_char_nbjour = mask_char

        store_dry_spell = []
        for i in years:
            for j in unique_zone:
                # Replace zone values with numeric parameters
                mask_char_start_search = xr.where(
                    mask_char_start_search == j,
                    self.day_of_year(i, self.criteria[j]["start_search"]),
                    mask_char_start_search
                )
                mask_char_cumulative = xr.where(
                    mask_char_cumulative == j,
                    self.criteria[j]["cumulative"],
                    mask_char_cumulative
                )
                mask_char_number_dry_days = xr.where(
                    mask_char_number_dry_days == j,
                    self.criteria[j]["number_dry_days"],
                    mask_char_number_dry_days
                )
                mask_char_thrd_rain_day = xr.where(
                    mask_char_thrd_rain_day == j,
                    self.criteria[j]["thrd_rain_day"],
                    mask_char_thrd_rain_day
                )
                mask_char_end_search = xr.where(
                    mask_char_end_search == j,
                    self.day_of_year(i, self.criteria[j]["end_search"]),
                    mask_char_end_search
                )
                mask_char_nbjour = xr.where(
                    mask_char_nbjour == j,
                    self.criteria[j]["nbjour"],
                    mask_char_nbjour
                )
            # Select data for this particular year
            year_data = daily_data.sel(T=str(i))

            # Parallel processing
            client = Client(n_workers=nb_cores, threads_per_worker=1)
            result = xr.apply_ufunc(
                self.dry_spell_onset_function,  # <-- Call our instance method
                year_data.chunk({'Y': chunksize_y, 'X': chunksize_x}),
                mask_char_start_search.chunk({'Y': chunksize_y, 'X': chunksize_x}),
                mask_char_cumulative.chunk({'Y': chunksize_y, 'X': chunksize_x}),
                mask_char_number_dry_days.chunk({'Y': chunksize_y, 'X': chunksize_x}),
                mask_char_thrd_rain_day.chunk({'Y': chunksize_y, 'X': chunksize_x}),
                mask_char_end_search.chunk({'Y': chunksize_y, 'X': chunksize_x}),
                mask_char_nbjour.chunk({'Y': chunksize_y, 'X': chunksize_x}),
                input_core_dims=[('T',), (), (), (), (), (), ()],
                vectorize=True,
                output_core_dims=[()],
                dask='parallelized',
                output_dtypes=['float'],
            )
            result_ = result.compute()
            client.close()

            store_dry_spell.append(result_)

        # Concatenate final result
        store_dry_spell = xr.concat(store_dry_spell, dim="T")
        store_dry_spell['T'] = T_from_here['T']
        store_dry_spell.name = "Onset_dryspell"

        return store_dry_spell#.to_array().drop_vars('variable').squeeze('variable')

class WAS_compute_cessation:
    """
    Class for computing the **cessation date** (end of rainy season) based on **soil moisture balance**.

    This class uses a simple water balance model to determine when soil moisture is depleted,
    marking the end of the rainy season. It is widely used in West African agrometeorology
    (AGRHYMET, national services) for Sahel, Sudanian, and Guinean zones.

    Key Features:
    - Soil water balance: rainfall - ETP (evapotranspiration), capped at soil retention capacity
    - Cessation date: first day when soil moisture reaches zero after the search start
    - Zone-specific criteria (start search, ETP, retention capacity, end search)
    - Supports station (CDT format) and gridded (xarray) daily rainfall data
    - Automatic agro-climatic zone assignment based on mean annual rainfall and latitude
    - Parallel computation with Dask for large grids

    Default Zones & Criteria:
    0. Sahel100_0mm     → start search 01-Sep, ETP 5.0 mm/day, capacity 70 mm, end 30-Sep
    1. Sahel200_100mm   → start search 01-Sep, ETP 5.0 mm/day, capacity 70 mm, end 05-Oct
    2. Sahel400_200mm   → start search 01-Sep, ETP 5.0 mm/day, capacity 70 mm, end 10-Nov
    3. Sahel600_400mm   → start search 15-Sep, ETP 5.0 mm/day, capacity 70 mm, end 15-Nov
    4. Soudan           → start search 01-Oct, ETP 4.5 mm/day, capacity 70 mm, end 30-Nov
    5. Golfe of Guinea  → start search 15-Oct, ETP 4.0 mm/day, capacity 70 mm, end 01-Dec

    References:
    - AGRHYMET Regional Centre operational methodologies
    - Sivakumar (1991): Soil water balance and cessation prediction in West Africa
    - Common practice in Sahelian and Sudanian national meteorological services

    Parameters
    ----------
    user_criteria : dict, optional
        Custom zone-specific cessation criteria. If None, uses default dictionary.
        Expected keys: zone_id → {"zone_name": str, "date_dry_soil": "MM-DD",
        "start_search": "MM-DD", "ETP": float,
        "Cap_ret_maxi": float, "end_search": "MM-DD"}
    """

    # Default class-level criteria dictionary
    default_criteria = {
        0: {"zone_name": "Sahel100_0mm", "date_dry_soil":"01-01", "start_search": "09-01", "ETP": 5.0, "Cap_ret_maxi": 70, "end_search": "09-30"},
        1: {"zone_name": "Sahel200_100mm", "date_dry_soil":"01-01", "start_search": "09-01", "ETP": 5.0, "Cap_ret_maxi": 70, "end_search": "10-05", },
        2: {"zone_name": "Sahel400_200mm", "date_dry_soil":"01-01", "start_search": "09-01", "ETP": 5.0, "Cap_ret_maxi": 70, "end_search": "11-10"},
        3: {"zone_name": "Sahel600_400mm", "date_dry_soil":"01-01", "start_search": "09-15", "ETP": 5.0, "Cap_ret_maxi": 70, "end_search": "11-15"},
        4: {"zone_name": "Soudan", "date_dry_soil":"01-01", "start_search": "10-01", "ETP": 4.5, "Cap_ret_maxi": 70, "end_search": "11-30"},
        5: {"zone_name": "Golfe_Of_Guinea", "date_dry_soil":"01-01", "start_search": "10-15", "ETP": 4.0, "Cap_ret_maxi": 70, "end_search": "12-01"},
    }

    def __init__(self, user_criteria=None):
        """
        Initialize the WAS_compute_cessation class with user-defined or default criteria.

        Parameters
        ----------
        user_criteria : dict, optional
            A dictionary containing zone-specific criteria. If not provided,
            the class will use the default criteria.
        """
        if user_criteria:
            self.criteria = user_criteria
        else:
            self.criteria = WAS_compute_cessation.default_criteria

    @staticmethod
    def adjust_duplicates(series, increment=0.00001):
        """
        If any values in the Series repeat, nudge them by a tiny increment
        so that all are unique (to avoid indexing collisions).
        """
        counts = series.value_counts()
        for val, count in counts[counts > 1].items():
            duplicates = series[series == val].index
            for i, idx in enumerate(duplicates):
                series.at[idx] += increment * i
        return series
    
    @staticmethod
    def day_of_year(i, dem_rech1):
        """
        Given a year 'i' and a month-day string 'dem_rech1' (e.g., '07-23'),
        return the 1-based day of the year.
        """
        year = int(i)
        full_date_str = f"{year}-{dem_rech1}"
        current_date = datetime.datetime.strptime(full_date_str, "%Y-%m-%d").date()
        origin_date = datetime.date(year, 1, 1)
        day_of_year_value = (current_date - origin_date).days + 1
        return day_of_year_value

    @staticmethod
    def transform_cdt(df):
        """
        Transform a DataFrame with:
          - Row 0 = LON
          - Row 1 = LAT
          - Row 2 = ELEV
          - Rows 3+ = daily data (or any date) with 'ID' column containing dates.

        Returns an xarray DataArray with coords = (T, Y, X), variable = 'Observation'.
        """
        # --- 1) Extract metadata (first 3 rows) ---
        metadata = df.iloc[:3].set_index("ID").T.reset_index()
        metadata.columns = ["STATION", "LON", "LAT", "ELEV"]
        
        # Adjust duplicates
        metadata["LON"] = WAS_compute_onset.adjust_duplicates(metadata["LON"])
        metadata["LAT"] = WAS_compute_onset.adjust_duplicates(metadata["LAT"])
        metadata["ELEV"] = WAS_compute_onset.adjust_duplicates(metadata["ELEV"])

        # --- 2) Extract actual data, rename ID -> DATE ---
        data_part = df.iloc[3:].rename(columns={"ID": "DATE"})

        # Melt to long form
        data_long = data_part.melt(id_vars=["DATE"], var_name="STATION", value_name="VALUE")

        # Merge metadata back into the melted data
        final_df = pd.merge(data_long, metadata, on="STATION")

        # Ensure DATE is treated as a proper datetime
        final_df["DATE"] = pd.to_datetime(final_df["DATE"], format="%Y%m%d")

        # Create a complete date range from the first day of the first year to the last day of the last year in the data
        start_date = final_df["DATE"].min().replace(month=1, day=1)
        end_date = final_df["DATE"].max().replace(month=12, day=31)
        date_range = pd.date_range(start=start_date, end=end_date)

        # Create a DataFrame with all combinations of dates and stations
        all_combinations = pd.MultiIndex.from_product([date_range, metadata["STATION"]], names=["DATE", "STATION"]).to_frame(index=False)

        # Merge the complete date-station combinations with the final_df to ensure all dates are present
        final_df = pd.merge(all_combinations, final_df, on=["DATE", "STATION"], how="left")

        # Fill missing values in the VALUE column with -99.0
        final_df["VALUE"] = final_df["VALUE"].fillna(-99.0)

        # Remove invalid rainfall values before computing the mean
        # Calculate the annual rainfall by summing the values for each year and station
        annual_rainfall = final_df[final_df["VALUE"] >= 0].groupby(["STATION", final_df["DATE"].dt.year])["VALUE"].sum().reset_index()

        # Calculate the mean annual rainfall for each station
        mean_annual_rainfall = annual_rainfall.groupby("STATION")["VALUE"].mean().reset_index()
        mean_annual_rainfall.columns = ["STATION", "MEAN_ANNUAL_RAINFALL"]

        # Merge back into final_df
        final_df = pd.merge(final_df, mean_annual_rainfall, on="STATION", how="left")

        # Generate the zonename column based on the conditions for each station
        def determine_zonename(row):
            if row["LAT"] <= 8:
                return 5
            elif 600 >= row["MEAN_ANNUAL_RAINFALL"] > 400:
                return 3
            elif 400 >= row["MEAN_ANNUAL_RAINFALL"] > 200:
                return 2
            elif 200 >= row["MEAN_ANNUAL_RAINFALL"] > 100:
                return 1
            elif 100 >= row["MEAN_ANNUAL_RAINFALL"] > 75:
                return 0
            else:           
                return 4

        final_df["zonename"] = final_df.groupby("STATION", group_keys=False).apply(
            lambda x: x.apply(determine_zonename, axis=1)
        )

        return final_df


    def cessation_function(self, x, ijour_dem_cal, idebut, ETP, Cap_ret_maxi, irch_fin):
        """
        Compute cessation date using soil moisture balance criteria.
        """
        mask = (
            np.isfinite(x).any()
            and np.isfinite(idebut)
            and np.isfinite(ijour_dem_cal)
            and np.isfinite(ETP)
            and np.isfinite(Cap_ret_maxi)
            and np.isfinite(irch_fin)
        )
        if not mask:
            return np.nan

        idebut = int(idebut)
        ijour_dem_cal = int(ijour_dem_cal)
        irch_fin = int(irch_fin)
        ru = 0

        for k in range(ijour_dem_cal, idebut + 1):
            if pd.isna(x[k]):
                continue
            ru += x[k] - ETP
            ru = max(0, min(ru, Cap_ret_maxi))

        ifin_saison = idebut
        while ifin_saison < irch_fin:
            ifin_saison += 1
            if pd.isna(x[ifin_saison]):
                continue
            ru += x[ifin_saison] - ETP
            ru = max(0, min(ru, Cap_ret_maxi))
            if ru <= 0:
                break

        return ifin_saison if ifin_saison <= irch_fin else random.randint(irch_fin - 5, irch_fin)


    def compute_insitu(self, daily_df):
        daily_df = self.transform_cdt(daily_df)

        unique_stations = daily_df["STATION"].unique()
        unique_years = daily_df["DATE"].dt.year.unique()
        unique_zonenames = daily_df["zonename"].unique()

        results = []

        for year in unique_years:
            for station in unique_stations:
                # Filter data for the current station and year
                station_data = daily_df[(daily_df["STATION"] == station) & (daily_df["DATE"].dt.year == year)]
                # Replace missing values with NaN
                station_data.loc[:, "VALUE"] = station_data["VALUE"].replace(-99.0, np.nan)
                # Extract unique zonenames
                unique_zonenames = station_data["zonename"].unique()
                # Extract the onset criteria for the current zonename
                ijour_dem_cal = self.day_of_year(year, self.criteria[unique_zonenames[0]]["date_dry_soil"])
                idebut = self.day_of_year(year, self.criteria[unique_zonenames[0]]["start_search"])
                irch_fin = self.day_of_year(year, self.criteria[unique_zonenames[0]]["end_search"])
                ETP = self.criteria[unique_zonenames[0]]["ETP"]
                Cap_ret_maxi = self.criteria[unique_zonenames[0]]["Cap_ret_maxi"]
                
                # Compute the onset date
                cessation_date = self.cessation_function(station_data["VALUE"].values, ijour_dem_cal, idebut, ETP, Cap_ret_maxi, irch_fin)
                
                results.append({
                    "year": year,
                    "station": station,
                    "lon": station_data["LON"].iloc[0],
                    "lat": station_data["LAT"].iloc[0],
                    "cessation": cessation_date
                })
        # Convert results to a DataFrame
        cessation_df = pd.DataFrame(results)
        final_df = cessation_df
        final_df["cessation"] = final_df["cessation"].fillna(-999)

        # transform the onset_df to the CPT format
        # Extract unique stations and their corresponding lat/lon
        station_metadata = cessation_df.groupby("station")[["lat", "lon"]].first().reset_index()

        # Pivot df_yyy to match the wide format (years as rows, stations as columns)
        df_pivot = cessation_df.pivot(index="year", columns="station", values="cessation")

        # Extract latitude and longitude values based on station order in pivoted DataFrame
        lat_row = pd.DataFrame([["LAT"] + station_metadata.set_index("station").loc[df_pivot.columns, "lat"].tolist()], 
                            columns=["STATION"] + df_pivot.columns.tolist())

        lon_row = pd.DataFrame([["LON"] + station_metadata.set_index("station").loc[df_pivot.columns, "lon"].tolist()], 
                            columns=["STATION"] + df_pivot.columns.tolist())

        # Reset index to ensure correct structure
        df_pivot.reset_index(inplace=True)

        # Rename the "year" column to "STATION" to match the required format
        df_pivot.rename(columns={"year": "STATION"}, inplace=True)

        # Concatenate latitude, longitude, and pivoted onset values to form the final structure
        df_final = pd.concat([lat_row, lon_row, df_pivot], ignore_index=True)

        return df_final

    def rainf_zone(self, daily_data):
        annual_rainfall = daily_data.resample(T="YE").sum(skipna=True).mean(dim='T')
        mask_5 = annual_rainfall.where(abs(annual_rainfall.Y) <= 8, np.nan)
        mask_5 = xr.where(np.isnan(mask_5), np.nan, 5) 
        mask_4 = xr.where(
            (abs(annual_rainfall.Y) > 8) 
            &
            ((annual_rainfall >= 600)),  
            4,
            np.nan
            )
        mask_3 = xr.where(
            (annual_rainfall < 600) & (annual_rainfall >= 400),
            3,
            np.nan 
            )
        mask_2 = xr.where(
            (annual_rainfall < 400) & (annual_rainfall >= 200),
            2,
            np.nan 
            )
        mask_1 = xr.where(
            (annual_rainfall < 200) & (annual_rainfall >= 100),
            1,np.nan 
            )
        mask_0 = xr.where(
            (annual_rainfall < 100) & (annual_rainfall >= 75),
            0,
            np.nan 
            )
        return mask_5.fillna(mask_4).fillna(mask_3).fillna(mask_2).fillna(mask_1).fillna(mask_0)
        
    def compute(self, daily_data, nb_cores):
        """
        Compute cessation dates for each pixel using criteria based on regions.
        """
        # # Load zone file & slice it to the area of interest
        # mask_char = xr.open_dataset('./utilities/Isohyet_zones.nc')
        # mask_char = mask_char.sel(X=slice(extent[1], extent[3]),
        #                           Y=slice(extent[0], extent[2]))
        # # Flip Y if needed (as done in your example)
        # mask_char = mask_char.isel(Y=slice(None, None, -1)).to_array().drop_vars('variable').squeeze()

        # daily_data = daily_data.sel(
        #     X=mask_char.coords['X'],
        #     Y=mask_char.coords['Y'])
        
        mask_char = self.rainf_zone(daily_data)
        
        unique_zone = np.unique(mask_char.to_numpy())
        unique_zone = unique_zone[~np.isnan(unique_zone)]

        years = np.unique(daily_data['T'].dt.year.to_numpy())
        zone_id_to_use = int(np.max(unique_zone))
        T_from_here = daily_data.sel(
            T=[f"{i}-{self.criteria[zone_id_to_use]['start_search']}" for i in years]
        )

        chunksize_x = int(np.round(len(daily_data.get_index("X")) / nb_cores))
        chunksize_y = int(np.round(len(daily_data.get_index("Y")) / nb_cores))

        mask_char_start_search = mask_char_date_dry_soil = mask_char_ETP = mask_char_Cap_ret_maxi = mask_char_end_search = mask_char

        store_cessation = []
        for i in years:
            for j in unique_zone:
                mask_char_date_dry_soil = xr.where(
                    mask_char_date_dry_soil == j,
                    self.day_of_year(i, self.criteria[j]["date_dry_soil"]),
                    mask_char_date_dry_soil,
                )
                mask_char_start_search = xr.where(
                    mask_char_start_search == j,
                    self.day_of_year(i, self.criteria[j]["start_search"]),
                    mask_char_start_search,
                )
                mask_char_ETP = xr.where(mask_char_ETP == j, self.criteria[j]["ETP"], mask_char_ETP)
                mask_char_Cap_ret_maxi = xr.where(
                    mask_char_Cap_ret_maxi == j,
                    self.criteria[j]["Cap_ret_maxi"],
                    mask_char_Cap_ret_maxi,
                )
                mask_char_end_search = xr.where(
                    mask_char_end_search == j,
                    self.day_of_year(i, self.criteria[j]["end_search"]),
                    mask_char_end_search,
                )

            year_data = daily_data.sel(T=str(i))

            client = Client(n_workers=nb_cores, threads_per_worker=1)
            result = xr.apply_ufunc(
                self.cessation_function,
                year_data.chunk({'Y': chunksize_y, 'X': chunksize_x}),
                mask_char_date_dry_soil.chunk({'Y': chunksize_y, 'X': chunksize_x}),
                mask_char_start_search.chunk({'Y': chunksize_y, 'X': chunksize_x}),
                mask_char_ETP.chunk({'Y': chunksize_y, 'X': chunksize_x}),
                mask_char_Cap_ret_maxi.chunk({'Y': chunksize_y, 'X': chunksize_x}),
                mask_char_end_search.chunk({'Y': chunksize_y, 'X': chunksize_x}),
                input_core_dims=[('T',), (), (), (), (), ()],
                vectorize=True,
                output_core_dims=[()],
                dask='parallelized',
                output_dtypes=['float'],
            )
            result_ = result.compute()
            client.close()

            store_cessation.append(result_)

        store_cessation = xr.concat(store_cessation, dim="T")
        store_cessation['T'] = T_from_here['T']
        store_cessation.name = "Cessation"

        return store_cessation #.to_array().drop_vars('variable').squeeze('variable')


class WAS_compute_cessation_dry_spell:
    """
    Class for computing the **longest dry spell length** between the onset and cessation
    of the rainy season, using a two-stage approach:

    1. Detect **onset** (start of rainy season) using cumulative rainfall criteria
    2. Detect **cessation** (end of rainy season) using soil water balance
    3. Calculate the maximum consecutive dry days (≤ thrd_rain_day mm) within that window

    This is a critical agro-meteorological indicator in West Africa, especially for assessing
    drought risk during the growing season after planting.

    Key Features:
    - Uses zone-specific criteria for both onset and cessation
    - Automatic agro-climatic zone assignment based on mean annual rainfall and latitude
    - Supports station (CDT format) and gridded (xarray) daily rainfall data
    - Parallel computation with Dask for large domains
    - Returns the longest dry spell length (in days) between onset and cessation

    Default Zones & Criteria:
    0. Sahel100_0mm     → onset: 01-May → 15-Aug | cessation: 01-Sep → 30-Sep
    1. Sahel200_100mm   → onset: 15-May → 15-Aug | cessation: 01-Sep → 05-Oct
    2. Sahel400_200mm   → onset: 01-May → 31-Jul | cessation: 01-Sep → 10-Nov
    3. Sahel600_400mm   → onset: 15-Mar → 31-Jul | cessation: 15-Sep → 15-Nov
    4. Soudan           → onset: 15-Mar → 31-Jul | cessation: 01-Oct → 30-Nov
    5. Golfe of Guinea  → onset: 01-Feb → 15-Jun | cessation: 15-Oct → 01-Dec

    References:
    - AGRHYMET Regional Centre operational methodologies
    - Sivakumar (1991): Methodology for onset, cessation, and dry spell analysis
    - Common practice in West African national meteorological and agricultural services

    Parameters
    ----------
    user_criteria : dict, optional
        Custom zone-specific criteria. If None, uses default dictionary.
        Expected keys: zone_id → {"zone_name": str,
        "start_search1": "MM-DD", "cumulative": float,
        "number_dry_days": int, "thrd_rain_day": float,
        "end_search1": "MM-DD", "nbjour": int,
        "date_dry_soil": "MM-DD", "start_search2": "MM-DD",
        "ETP": float, "Cap_ret_maxi": float,
        "end_search2": "MM-DD"}
    """

    # Default class-level criteria dictionary
    default_criteria = {
        0: {
            "zone_name": "Sahel100_0mm",
            "start_search1": "05-01",
            "cumulative": 10,
            "number_dry_days": 25,
            "thrd_rain_day": 0.85,
            "end_search1": "08-15",
            "nbjour": 40,
            "date_dry_soil": "01-01",
            "start_search2": "09-01",
            "ETP": 5.0,
            "Cap_ret_maxi": 70,
            "end_search2": "09-30"
        },
        1: {
            "zone_name": "Sahel200_100mm",
            "start_search1": "05-15",
            "cumulative": 15,
            "number_dry_days": 25,
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

    def __init__(self, user_criteria=None):
        """
        Initialize the WAS_compute_cessation_dry_spell class with user-defined or default criteria.

        Parameters
        ----------
        user_criteria : dict, optional
            A dictionary containing zone-specific criteria. If not provided,
            the class will use the default criteria.
        """
        if user_criteria:
            self.criteria = user_criteria
        else:
            self.criteria = WAS_compute_cessation_dry_spell.default_criteria

    @staticmethod
    def adjust_duplicates(series, increment=0.00001):
        """
        If any values in the Series repeat, nudge them by a tiny increment
        so that all are unique (to avoid indexing collisions).
        """
        counts = series.value_counts()
        for val, count in counts[counts > 1].items():
            duplicates = series[series == val].index
            for i, idx in enumerate(duplicates):
                series.at[idx] += increment * i
        return series

    @staticmethod
    def transform_cdt(df):
        """
        Transform a DataFrame with:
          - Row 0 = LON
          - Row 1 = LAT
          - Row 2 = ELEV
          - Rows 3+ = daily data (or any date) with 'ID' column containing dates.

        Returns an xarray DataArray with coords = (T, Y, X), variable = 'Observation'.
        """
        # --- 1) Extract metadata (first 3 rows) ---
        metadata = df.iloc[:3].set_index("ID").T.reset_index()
        metadata.columns = ["STATION", "LON", "LAT", "ELEV"]
        
        # Adjust duplicates
        metadata["LON"] = WAS_compute_cessation_dry_spell.adjust_duplicates(metadata["LON"])
        metadata["LAT"] = WAS_compute_cessation_dry_spell.adjust_duplicates(metadata["LAT"])
        metadata["ELEV"] = WAS_compute_cessation_dry_spell.adjust_duplicates(metadata["ELEV"])

        # --- 2) Extract actual data, rename ID -> DATE ---
        data_part = df.iloc[3:].rename(columns={"ID": "DATE"})

        # Melt to long form
        data_long = data_part.melt(id_vars=["DATE"], var_name="STATION", value_name="VALUE")

        # Merge metadata back into the melted data
        final_df = pd.merge(data_long, metadata, on="STATION")

        # Ensure DATE is treated as a proper datetime
        final_df["DATE"] = pd.to_datetime(final_df["DATE"], format="%Y%m%d")

        # Create a complete date range from the first day of the first year to the last day of the last year in the data
        start_date = final_df["DATE"].min().replace(month=1, day=1)
        end_date = final_df["DATE"].max().replace(month=12, day=31)
        date_range = pd.date_range(start=start_date, end=end_date)

        # Create a DataFrame with all combinations of dates and stations
        all_combinations = pd.MultiIndex.from_product([date_range, metadata["STATION"]], names=["DATE", "STATION"]).to_frame(index=False)

        # Merge the complete date-station combinations with the final_df to ensure all dates are present
        final_df = pd.merge(all_combinations, final_df, on=["DATE", "STATION"], how="left")

        # Fill missing values in the VALUE column with -99.0
        final_df["VALUE"] = final_df["VALUE"].fillna(-99.0)

        # Remove invalid rainfall values before computing the mean
        # Calculate the annual rainfall by summing the values for each year and station
        annual_rainfall = final_df[final_df["VALUE"] >= 0].groupby(["STATION", final_df["DATE"].dt.year])["VALUE"].sum().reset_index()

        # Calculate the mean annual rainfall for each station
        mean_annual_rainfall = annual_rainfall.groupby("STATION")["VALUE"].mean().reset_index()
        mean_annual_rainfall.columns = ["STATION", "MEAN_ANNUAL_RAINFALL"]

        # Merge back into final_df
        final_df = pd.merge(final_df, mean_annual_rainfall, on="STATION", how="left")

        # Generate the zonename column based on the conditions for each station
        def determine_zonename(row):
            if row["LAT"] <= 8:
                return 5
            elif 600 >= row["MEAN_ANNUAL_RAINFALL"] > 400:
                return 3
            elif 400 >= row["MEAN_ANNUAL_RAINFALL"] > 200:
                return 2
            elif 200 >= row["MEAN_ANNUAL_RAINFALL"] > 100:
                return 1
            elif 100 >= row["MEAN_ANNUAL_RAINFALL"] > 75:    
                return 0
            else:
                return 4

        final_df["zonename"] = final_df.groupby("STATION", group_keys=False).apply(
            lambda x: x.apply(determine_zonename, axis=1)
        )

        return final_df

    def rainf_zone(self, daily_data):
        annual_rainfall = daily_data.resample(T="YE").sum(skipna=True).mean(dim='T')
        mask_5 = annual_rainfall.where(abs(annual_rainfall.Y) <= 8, np.nan)
        mask_5 = xr.where(np.isnan(mask_5), np.nan, 5) 
        mask_4 = xr.where(
            (abs(annual_rainfall.Y) > 8) 
            &
            ((annual_rainfall >= 600)),  
            4,
            np.nan
            )
        mask_3 = xr.where(
            (annual_rainfall < 600) & (annual_rainfall >= 400),
            3,
            np.nan 
            )
        mask_2 = xr.where(
            (annual_rainfall < 400) & (annual_rainfall >= 200),
            2,
            np.nan 
            )
        mask_1 = xr.where(
            (annual_rainfall < 200) & (annual_rainfall >= 100),
            1,np.nan 
            )
        mask_0 = xr.where(
            (annual_rainfall < 100) & (annual_rainfall >= 75),
            0,
            np.nan 
            )
        return mask_5.fillna(mask_4).fillna(mask_3).fillna(mask_2).fillna(mask_1).fillna(mask_0)

    
    def dry_spell_cessation_function(self,
                                     x,
                                     idebut1,
                                     cumul,
                                     nbsec,
                                     jour_pluvieux,
                                     irch_fin1,
                                     idebut2,
                                     ijour_dem_cal,
                                     ETP,
                                     Cap_ret_maxi,
                                     irch_fin2,
                                     nbjour):
        """
        Computes the longest dry spell length after the onset and
        determines the cessation date (when soil water returns to 0)
        based on water balance, then checks for a dry spell.

        Parameters
        ----------
        x : array-like
            Daily rainfall or similar values.
        idebut1 : int
            Start index to begin searching for the onset.
        cumul : float
            Cumulative rainfall threshold to trigger onset.
        nbsec : int
            Maximum number of dry days allowed in the sequence.
        jour_pluvieux : float
            Minimum rainfall to consider a day as rainy.
        irch_fin1 : int
            Maximum index limit for the onset search.
        idebut2 : int
            Start index for the cessation search.
        ijour_dem_cal : int
            Start index from which the water balance is calculated.
        ETP : float
            Daily evapotranspiration (mm).
        Cap_ret_maxi : float
            Maximum soil water retention capacity (mm).
        irch_fin2 : int
            Maximum index limit for the cessation search.
        nbjour : int
            Number of days after onset to check for the dry spell.

        Returns
        -------
        float
            Length of the longest dry spell sequence after onset and before soil water
            returns to zero, or NaN if not found.
        """
        mask = (
            np.any(np.isfinite(x)) and
            np.isfinite(idebut1) and 
            np.isfinite(nbsec) and 
            np.isfinite(irch_fin1) and
            np.isfinite(idebut2) and
            np.isfinite(ijour_dem_cal) and
            np.isfinite(ETP) and
            np.isfinite(Cap_ret_maxi) and
            np.isfinite(irch_fin2) and
            np.isfinite(nbjour)
        )

        if not mask:
            return np.nan

        # Convert to int where needed
        idebut1 = int(idebut1)
        nbsec = int(nbsec)
        irch_fin1 = int(irch_fin1)
        idebut2 = int(idebut2)
        ijour_dem_cal = int(ijour_dem_cal)
        irch_fin2 = int(irch_fin2)
        nbjour = int(nbjour)

        ru = 0
        trouv = 0
        idate = idebut1

        # --- 1) Find onset date ---
        while True:
            idate += 1
            ipreced = idate - 1
            isuiv = idate + 1

            # Check for missing data or out-of-bounds
            if (
                ipreced >= len(x) or
                idate >= len(x) or
                isuiv >= len(x) or
                pd.isna(x[ipreced]) or
                pd.isna(x[idate]) or
                pd.isna(x[isuiv])
            ):
                deb_saison = np.nan
                break

            # Check if we've exceeded the search limit
            if idate > irch_fin1:
                deb_saison = random.randint(irch_fin1 - 5, irch_fin1)
                break

            # Calculate cumulative rainfall for 1, 2, 3 days
            cumul3jr = x[ipreced] + x[idate] + x[isuiv]
            cumul2jr = x[ipreced] + x[idate]
            cumul1jr = x[ipreced]

            # Check if threshold is met
            if (cumul1jr >= cumul or cumul2jr >= cumul or cumul3jr >= cumul):
                troisp = np.array([x[ipreced], x[idate], x[isuiv]])
                itroisp = np.array([ipreced, idate, isuiv])
                maxp = np.nanmax(troisp)
                imaxp = np.where(troisp == maxp)[0][0]
                ideb = itroisp[imaxp]
                deb_saison = ideb
                trouv = 1

                # Check for sequences of dry days within the next 30 days
                finp = ideb + 30
                if finp < len(x):
                    pluie30jr = x[ideb: finp + 1]
                else:
                    pluie30jr = x[ideb:]

                isec = 0
                while True:
                    isec += 1
                    isecf = isec + nbsec
                    if isecf >= len(pluie30jr):
                        break
                    donneeverif = pluie30jr[isec : isecf + 1]
                    # Count days with rainfall below 'jour_pluvieux'
                    test1 = np.sum(donneeverif < jour_pluvieux)

                    if test1 == (nbsec + 1):  # found a fully dry subsequence
                        trouv = 0

                    if test1 == (nbsec + 1) or isec == (30 - nbsec):
                        break

            if trouv == 1:
                break

        # If deb_saison not found, no need to calculate further
        if pd.isna(deb_saison):
            return np.nan

        # --- 2) Soil water balance from ijour_dem_cal up to idebut2 ---
        for k in range(ijour_dem_cal, idebut2 + 1):
            if k >= len(x) or pd.isna(x[k]):
                continue
            ru += x[k] - ETP
            # Confine to [0, Cap_ret_maxi]
            ru = max(0, min(ru, Cap_ret_maxi))

        # --- 3) Move forward until soil water returns to 0 or we hit irch_fin2 ---
        ifin_saison = idebut2
        while ifin_saison < irch_fin2:
            ifin_saison += 1
            if ifin_saison >= len(x) or pd.isna(x[ifin_saison]):
                continue
            ru += x[ifin_saison] - ETP
            ru = max(0, min(ru, Cap_ret_maxi))
            if ru <= 0:
                break
        fin_saison = ifin_saison if ifin_saison <= irch_fin2 else random.randint(irch_fin2 - 5, irch_fin2)

        # --- 4) If we found a valid fin_saison beyond (deb_saison + nbjour), 
        #         check the longest dry spell between them.
        if (
            not np.isnan(fin_saison) and 
            (fin_saison - (deb_saison + nbjour)) > 0 and 
            (deb_saison + nbjour) < len(x)
        ):
            pluie_period = x[deb_saison + nbjour : fin_saison]
            if len(pluie_period) == 0:
                return np.nan

            # Find indices of rainy days in that window
            rainy_days = np.where(pluie_period > jour_pluvieux)[0]
            d1 = np.array([0] + list(rainy_days))
            d2 = np.array(list(rainy_days) + [len(pluie_period)])
            seq_max = np.max(d2 - d1) - 1
            return seq_max
        else:
            return np.nan

    @staticmethod
    def day_of_year(i, dem_rech1):
        """
        Convert year i and MM-DD string dem_rech1 (e.g., '07-23') 
        into a 1-based day of the year.
        """
        year = int(i)
        full_date_str = f"{year}-{dem_rech1}"
        current_date = datetime.datetime.strptime(full_date_str, "%Y-%m-%d").date()
        origin_date = datetime.date(year, 1, 1)
        return (current_date - origin_date).days + 1


    def compute_insitu(self, daily_df):
        daily_df = self.transform_cdt(daily_df)
        unique_stations = daily_df["STATION"].unique()
        unique_years = daily_df["DATE"].dt.year.unique()
        unique_zonenames = daily_df["zonename"].unique()

        results = []

        for year in unique_years:
            for station in unique_stations:
                # Filter data for the current station and year
                station_data = daily_df[(daily_df["STATION"] == station) & (daily_df["DATE"].dt.year == year)]
                # Replace missing values with NaN
                station_data.loc[:, "VALUE"] = station_data["VALUE"].replace(-99.0, np.nan)
                # Extract unique zonenames
                unique_zonenames = station_data["zonename"].unique()
                # Extract the onset criteria for the current zonename
                idebut1 = self.day_of_year(year, self.criteria[unique_zonenames[0]]["start_search1"])
                irch_fin1 = self.day_of_year(year, self.criteria[unique_zonenames[0]]["end_search1"])
                cumul = self.criteria[unique_zonenames[0]]["cumulative"]
                nbsec = self.criteria[unique_zonenames[0]]["number_dry_days"]
                jour_pluvieux = self.criteria[unique_zonenames[0]]["thrd_rain_day"]

                ijour_dem_cal = self.day_of_year(year, self.criteria[unique_zonenames[0]]["date_dry_soil"])
                idebut2 = self.day_of_year(year, self.criteria[unique_zonenames[0]]["start_search2"])
                irch_fin2 = self.day_of_year(year, self.criteria[unique_zonenames[0]]["end_search2"])
                ETP = self.criteria[unique_zonenames[0]]["ETP"]
                Cap_ret_maxi = self.criteria[unique_zonenames[0]]["Cap_ret_maxi"]
                nbjour = self.criteria[unique_zonenames[0]]["nbjour"]
                
                # Compute the cessation dryspell
                cessation_dryspell = self.dry_spell_cessation_function(station_data["VALUE"].values,
                                                                   idebut1,
                                                                   cumul,
                                                                   nbsec,
                                                                   jour_pluvieux,
                                                                   irch_fin1,
                                                                   idebut2,
                                                                   ijour_dem_cal,
                                                                   ETP,
                                                                   Cap_ret_maxi,
                                                                   irch_fin2,
                                                                   nbjour)
                
                results.append({
                    "year": year,
                    "station": station,
                    "lon": station_data["LON"].iloc[0],
                    "lat": station_data["LAT"].iloc[0],
                    "cessation_dryspell": cessation_dryspell
                })
        # Convert results to a DataFrame
        cessation_df = pd.DataFrame(results)
        final_df = cessation_df
        final_df["cessation_dryspell"] = final_df["cessation_dryspell"].fillna(-999)

        # transform the onset_df to the CPT format
        # Extract unique stations and their corresponding lat/lon
        station_metadata = cessation_df.groupby("station")[["lat", "lon"]].first().reset_index()

        # Pivot df_yyy to match the wide format (years as rows, stations as columns)
        df_pivot = cessation_df.pivot(index="year", columns="station", values="cessation_dryspell")

        # Extract latitude and longitude values based on station order in pivoted DataFrame
        lat_row = pd.DataFrame([["LAT"] + station_metadata.set_index("station").loc[df_pivot.columns, "lat"].tolist()], 
                            columns=["STATION"] + df_pivot.columns.tolist())

        lon_row = pd.DataFrame([["LON"] + station_metadata.set_index("station").loc[df_pivot.columns, "lon"].tolist()], 
                            columns=["STATION"] + df_pivot.columns.tolist())

        # Reset index to ensure correct structure
        df_pivot.reset_index(inplace=True)

        # Rename the "year" column to "STATION" to match the required format
        df_pivot.rename(columns={"year": "STATION"}, inplace=True)

        # Concatenate latitude, longitude, and pivoted onset values to form the final structure
        df_final = pd.concat([lat_row, lon_row, df_pivot], ignore_index=True)

        return df_final


    def compute(self, daily_data, nb_cores):
        """
        Compute the longest dry spell length after the rainy season onset 
        for each pixel in the given daily rainfall DataArray, using different 
        criteria (both for onset and cessation) based on isohyet zones.

        Parameters
        ----------
        daily_data : xarray.DataArray
            Daily rainfall data, coords = (T, Y, X).
        nb_cores : int
            Number of parallel processes (workers) to use.

        Returns
        -------
        xarray.DataArray
            Array with the longest dry spell length per pixel.
        """
        # # 1) Load zone file & slice it
        # mask_char = xr.open_dataset("./utilities/Isohyet_zones.nc")
        # mask_char = mask_char.sel(X=slice(extent[1], extent[3]),
        #                           Y=slice(extent[0], extent[2]))

        # # 2) Flip Y if needed
        # mask_char = mask_char.isel(Y=slice(None, None, -1)).to_array().drop_vars("variable").squeeze()

        # daily_data = daily_data.sel(
        #     X=mask_char.coords['X'],
        #     Y=mask_char.coords['Y'])
        
        mask_char = self.rainf_zone(daily_data)
        
        # 3) Get unique zone IDs
        unique_zone = np.unique(mask_char.to_numpy())
        unique_zone = unique_zone[~np.isnan(unique_zone)]

        # 4) Determine years from the dataset
        years = np.unique(daily_data["T"].dt.year.to_numpy())

        # 5) For illustration, pick the largest zone to define T dimension
        zone_id_to_use = int(np.max(unique_zone))
        T_from_here = daily_data.sel(
            T=[f"{str(i)}-{self.criteria[zone_id_to_use]['start_search2']}" for i in years]
        )

        # 6) Prepare chunk sizes
        chunksize_x = int(np.round(len(daily_data.get_index("X")) / nb_cores))
        chunksize_y = int(np.round(len(daily_data.get_index("Y")) / nb_cores))

        # 7) Create placeholders for all required masks 
        mask_char_start_search1 = mask_char_cumulative = mask_char_number_dry_days = \
            mask_char_thrd_rain_day = mask_char_end_search1 = mask_char_nbjour = \
            mask_char_start_search2 = mask_char_date_dry_soil = mask_char_ETP = \
            mask_char_Cap_ret_maxi = mask_char_end_search2 = mask_char

        store_dry_spell = []

        for i in years:
            # Update masks for each zone 'j'
            for j in unique_zone:
                mask_char_start_search1 = xr.where(
                    mask_char_start_search1 == j,
                    self.day_of_year(i, self.criteria[j]["start_search1"]),
                    mask_char_start_search1
                )
                mask_char_cumulative = xr.where(
                    mask_char_cumulative == j,
                    self.criteria[j]["cumulative"],
                    mask_char_cumulative
                )
                mask_char_number_dry_days = xr.where(
                    mask_char_number_dry_days == j,
                    self.criteria[j]["number_dry_days"],
                    mask_char_number_dry_days
                )
                mask_char_thrd_rain_day = xr.where(
                    mask_char_thrd_rain_day == j,
                    self.criteria[j]["thrd_rain_day"],
                    mask_char_thrd_rain_day
                )
                mask_char_end_search1 = xr.where(
                    mask_char_end_search1 == j,
                    self.day_of_year(i, self.criteria[j]["end_search1"]),
                    mask_char_end_search1
                )
                mask_char_nbjour = xr.where(
                    mask_char_nbjour == j,
                    self.criteria[j]["nbjour"],
                    mask_char_nbjour
                )
                mask_char_date_dry_soil = xr.where(
                    mask_char_date_dry_soil == j,
                    self.day_of_year(i, self.criteria[j]["date_dry_soil"]),
                    mask_char_date_dry_soil
                )
                mask_char_start_search2 = xr.where(
                    mask_char_start_search2 == j,
                    self.day_of_year(i, self.criteria[j]["start_search2"]),
                    mask_char_start_search2
                )
                mask_char_ETP = xr.where(
                    mask_char_ETP == j,
                    self.criteria[j]["ETP"],
                    mask_char_ETP
                )
                mask_char_Cap_ret_maxi = xr.where(
                    mask_char_Cap_ret_maxi == j,
                    self.criteria[j]["Cap_ret_maxi"],
                    mask_char_Cap_ret_maxi
                )
                mask_char_end_search2 = xr.where(
                    mask_char_end_search2 == j,
                    self.day_of_year(i, self.criteria[j]["end_search2"]),
                    mask_char_end_search2
                )

            # Select the daily data for year i
            year_data = daily_data.sel(T=str(i))

            # 8) Parallel processing with Dask
            client = Client(n_workers=nb_cores, threads_per_worker=1)
            result = xr.apply_ufunc(
                self.dry_spell_cessation_function,
                year_data.chunk({"Y": chunksize_y, "X": chunksize_x}),
                mask_char_start_search1.chunk({"Y": chunksize_y, "X": chunksize_x}),
                mask_char_cumulative.chunk({"Y": chunksize_y, "X": chunksize_x}),
                mask_char_number_dry_days.chunk({"Y": chunksize_y, "X": chunksize_x}),
                mask_char_thrd_rain_day.chunk({"Y": chunksize_y, "X": chunksize_x}),
                mask_char_end_search1.chunk({"Y": chunksize_y, "X": chunksize_x}),
                mask_char_start_search2.chunk({"Y": chunksize_y, "X": chunksize_x}),
                mask_char_date_dry_soil.chunk({"Y": chunksize_y, "X": chunksize_x}),
                mask_char_ETP.chunk({"Y": chunksize_y, "X": chunksize_x}),
                mask_char_Cap_ret_maxi.chunk({"Y": chunksize_y, "X": chunksize_x}),
                mask_char_end_search2.chunk({"Y": chunksize_y, "X": chunksize_x}),
                mask_char_nbjour.chunk({"Y": chunksize_y, "X": chunksize_x}),
                input_core_dims=[("T",), (), (), (), (), (), (), (), (), (), (), ()],
                vectorize=True,
                output_core_dims=[()],
                dask="parallelized",
                output_dtypes=["float"],
            )
            result_ = result.compute()
            client.close()

            store_dry_spell.append(result_)

        # 9) Concatenate final result across years
        store_dry_spell = xr.concat(store_dry_spell, dim="T")
        store_dry_spell["T"] = T_from_here["T"]
        store_dry_spell.name = "Cessation_dryspell"

        return store_dry_spell #.to_array().drop_vars('variable').squeeze('variable')
    

class WAS_count_dry_spells:
    """
    Class for computing the **number of dry spells** of a specified length
    occurring between the onset and cessation dates of the rainy season.

    This class is designed to quantify intra-seasonal dry spell risk — a critical
    indicator for agriculture in West Africa, where consecutive dry days after
    planting can severely impact crop establishment and yield.

    Key Features:
    - Counts dry spells (consecutive days ≤ ``dry_threshold`` mm) between onset and cessation
    - Accepts onset and cessation dates as pre-computed xarray DataArrays or CPT DataFrames
    - Supports both station-based (CDT format) and gridded (xarray) daily rainfall data
    - Efficient parallel computation using Dask for large domains
    - Outputs in CPT-compatible wide format for station data

    Typical Use Case:
    After computing onset (e.g., via ``WAS_compute_onset``) and cessation
    (e.g., via ``WAS_compute_cessation``), use this class to count risky dry spells
    (e.g., 10-day, 15-day) during the growing window.

    References:
    - AGRHYMET Regional Centre operational dry spell monitoring
    - Sivakumar (1991): Methodology for dry spell analysis in Sahelian agriculture
    - Common practice in West African national meteorological and agricultural services

    Parameters
    ----------
    None (no initialization parameters required — all criteria passed at compute time)

    Methods
    -------
    compute_insitu(daily_df, onset_df_cpt, cessation_df_cpt, dry_spell_length, dry_threshold=1.0)
        Compute number of dry spells for station data (CDT input, CPT output).

    compute(daily_data, onset_date, cessation_date, dry_spell_length, dry_threshold, nb_cores)
        Compute number of dry spells for gridded xarray data (parallelized).

    Examples
    --------
    Count number of 10-day dry spells between onset and cessation (gridded):

    >>> count_calc = WAS_count_dry_spells()
    >>> dry_spell_count = count_calc.compute(
    ...     daily_pr,           # xarray.DataArray daily rainfall
    ...     onset_da,           # xarray.DataArray of onset dates
    ...     cessation_da,       # xarray.DataArray of cessation dates
    ...     dry_spell_length=10,
    ...     dry_threshold=1.0,
    ...     nb_cores=8
    ... )

    For station data (CDT rainfall, CPT onset/cessation):

    >>> count_cpt = count_calc.compute_insitu(
    ...     daily_cdt_df,
    ...     onset_cpt_df,
    ...     cessation_cpt_df,
    ...     dry_spell_length=15,
    ...     dry_threshold=0.5
    ... )
    """

    @staticmethod
    def adjust_duplicates(series, increment=0.00001):
        """
        If any values in the Series repeat, nudge them by a tiny increment
        so that all are unique (to avoid indexing collisions).
        """
        counts = series.value_counts()
        for val, count in counts[counts > 1].items():
            duplicates = series[series == val].index
            for i, idx in enumerate(duplicates):
                series.at[idx] += increment * i
        return series

    @staticmethod
    def transform_cdt(df):
        """
        Transform a DataFrame with:
          - Row 0 = LON
          - Row 1 = LAT
          - Row 2 = ELEV
          - Rows 3+ = daily data with 'ID' column containing dates.

        Returns a DataFrame with columns like:
          DATE | STATION | VALUE | LON | LAT | ELEV | MEAN_ANNUAL_RAINFALL | zonename
        """
        # 1) Extract metadata (first 3 rows)
        metadata = df.iloc[:3].set_index("ID").T.reset_index()
        metadata.columns = ["STATION", "LON", "LAT", "ELEV"]
        
        # Adjust duplicates
        metadata["LON"] = WAS_count_dry_spells.adjust_duplicates(metadata["LON"])
        metadata["LAT"] = WAS_count_dry_spells.adjust_duplicates(metadata["LAT"])
        metadata["ELEV"] = WAS_count_dry_spells.adjust_duplicates(metadata["ELEV"])

        # 2) Extract daily data, rename ID -> DATE
        data_part = df.iloc[3:].rename(columns={"ID": "DATE"})

        # Melt to long form
        data_long = data_part.melt(id_vars=["DATE"], var_name="STATION", value_name="VALUE")

        # Merge metadata
        final_df = pd.merge(data_long, metadata, on="STATION")

        # Convert DATE to datetime
        final_df["DATE"] = pd.to_datetime(final_df["DATE"], format="%Y%m%d")

        # Create complete date range
        start_date = final_df["DATE"].min().replace(month=1, day=1)
        end_date = final_df["DATE"].max().replace(month=12, day=31)
        date_range = pd.date_range(start=start_date, end=end_date)

        # All combinations of (date, station)
        all_combinations = pd.MultiIndex.from_product(
            [date_range, metadata["STATION"]],
            names=["DATE", "STATION"]
        ).to_frame(index=False)

        # Merge to ensure every (date, station) is present
        final_df = pd.merge(all_combinations, final_df, on=["DATE", "STATION"], how="left")

        # Fill missing rainfall values with -99.0
        final_df["VALUE"] = final_df["VALUE"].fillna(-99.0)

        # Compute mean annual rainfall per station
        annual_rainfall = (
            final_df[final_df["VALUE"] >= 0]
            .groupby(["STATION", final_df["DATE"].dt.year])["VALUE"]
            .sum()
            .reset_index()
        )
        mean_annual_rainfall = annual_rainfall.groupby("STATION")["VALUE"].mean().reset_index()
        mean_annual_rainfall.columns = ["STATION", "MEAN_ANNUAL_RAINFALL"]

        final_df = pd.merge(final_df, mean_annual_rainfall, on="STATION", how="left")

        # Generate a zonename column (optional example logic)
        def determine_zonename(row):
            if row["LAT"] <= 8:
                return 5
            elif 600 >= row["MEAN_ANNUAL_RAINFALL"] > 400:
                return 3
            elif 400 >= row["MEAN_ANNUAL_RAINFALL"] > 200:
                return 2
            elif 200 >= row["MEAN_ANNUAL_RAINFALL"] > 100:
                return 1
            elif 100 >= row["MEAN_ANNUAL_RAINFALL"] > 75:
                return 0
            else:
                return 4

        final_df["zonename"] = final_df.groupby("STATION", group_keys=False).apply(
            lambda x: x.apply(determine_zonename, axis=1)
        )

        return final_df

    @staticmethod
    def transform_cpt(df, missing_value=None):
        """
        Transform a DataFrame in CPT format with:
         - Row 0 = LAT
         - Row 1 = LON
         - Rows 2+ = numeric year data in wide format (stations in columns).

        Returns a DataFrame with columns like:
          YEAR | STATION | VALUE | LAT | LON
        """
        # 1) Extract metadata (first 2 rows: LAT, LON)
        metadata = (
            df.iloc[:2]
            .set_index("STATION")  # index = ["LAT", "LON"]
            .T
            .reset_index()         # columns: ["index", "LAT", "LON"]
        )
        metadata.columns = ["STATION", "LAT", "LON"]
        
        # Adjust duplicates in LAT / LON
        metadata["LAT"] = WAS_count_dry_spells.adjust_duplicates(metadata["LAT"])
        metadata["LON"] = WAS_count_dry_spells.adjust_duplicates(metadata["LON"])
        
        # 2) Extract the data part from row 2 onward
        data_part = df.iloc[2:].copy()
        data_part = data_part.rename(columns={"STATION": "YEAR"})
        data_part["YEAR"] = data_part["YEAR"].astype(int)
        
        # 3) Wide to long
        long_data = data_part.melt(
            id_vars="YEAR",
            var_name="STATION",
            value_name="VALUE"
        )

        # 4) Merge with metadata
        final_df = pd.merge(long_data, metadata, on="STATION", how="left")          

        return final_df

    @staticmethod
    def _parse_cpt_to_long(df_cpt, value_name="onset_or_cessation"):
        """
        Convert a DataFrame in CPT-like format to a long DataFrame with columns:
            [year, station, value_name, lat, lon]

        Assumes:
         - Row 0 = ["LAT", lat_stn1, lat_stn2, ...]
         - Row 1 = ["LON", lon_stn1, lon_stn2, ...]
         - Rows 2+ = [year, station1_val, station2_val, ...]

        Parameters
        ----------
        df_cpt : pd.DataFrame
            CPT-like DataFrame (as returned by, e.g., compute_insitu).
        value_name : str
            Name to give to the column containing the value (e.g. "onset", "cessation").

        Returns
        -------
        pd.DataFrame
            Columns: [station, year, <value_name>, lat, lon]
        """
        # Row 0 for LAT, row 1 for LON
        lat_row = df_cpt.iloc[0, 1:].values  # all station lat
        lon_row = df_cpt.iloc[1, 1:].values  # all station lon

        # Station names from columns
        station_cols = df_cpt.columns[1:].tolist()

        # Rows from index=2 are year + station values
        df_years = df_cpt.iloc[2:].copy()
        df_years.reset_index(drop=True, inplace=True)
        df_years.rename(columns={"STATION": "year"}, inplace=True)

        # Transform to long
        df_long = df_years.melt(
            id_vars=["year"],
            var_name="station",
            value_name=value_name
        )
        df_long["year"] = pd.to_numeric(df_long["year"], errors="coerce")

        # Map station -> lat/lon
        lat_map = dict(zip(station_cols, lat_row))
        lon_map = dict(zip(station_cols, lon_row))
        df_long["lat"] = df_long["station"].map(lat_map)
        df_long["lon"] = df_long["station"].map(lon_map)

        return df_long

    @staticmethod
    def count_dry_spells(x, onset, cessation, dry_spell_length, dry_threshold):
        """
        Count the number of dry spells of a specific length between onset and cessation dates.

        Parameters
        ----------
        x : array-like
            Daily rainfall values.
        onset : int
            Start index for the calculation (onset date).
        cessation : int
            End index for the calculation (cessation date).
        dry_spell_length : int
            The length of a dry spell to count.
        dry_threshold : float
            Rainfall threshold to classify a day as "dry."

        Returns
        -------
        int or float
            The number of dry spells of the specified length (NaN if invalid).
        """
        mask = (
            np.isfinite(x).any()
            and np.isfinite(onset)
            and np.isfinite(cessation)
        )
        if not mask:
            return np.nan
        
        onset = int(onset)
        cessation = int(cessation)

        # Prevent out-of-bounds
        if onset < 0 or cessation < 0 or onset >= len(x):
            return np.nan
        if cessation >= len(x):
            cessation = len(x) - 1  # truncate

        dry_spells_count = 0
        current_dry_days = 0

        for day in range(onset, cessation + 1):
            if x[day] < dry_threshold:
                current_dry_days += 1
            else:
                if current_dry_days == dry_spell_length:
                    dry_spells_count += 1
                current_dry_days = 0

        # Check if the final run of dry days meets the criterion
        if current_dry_days == dry_spell_length:
            dry_spells_count += 1

        return dry_spells_count

    def compute_insitu(self, daily_df, onset_df_cpt, cessation_df_cpt, dry_spell_length, dry_threshold=1.0):
        """
        Compute the number of dry spells (of length = dry_spell_length) between the
        onset and cessation dates for in-situ stations (CDT format).

        Returns a DataFrame in CPT format:
         - Row 0: ["LAT", lat_stn1, lat_stn2, ...]
         - Row 1: ["LON", lon_stn1, lon_stn2, ...]
         - Subsequent rows: [year, station1_value, station2_value, ...]

        Parameters
        ----------
        daily_df : pd.DataFrame
            CDT rainfall data (ID column = date, station columns).
        onset_df_cpt : pd.DataFrame
            CPT-format DataFrame containing onset dates (as returned by some method).
        cessation_df_cpt : pd.DataFrame
            CPT-format DataFrame containing cessation dates.
        dry_spell_length : int
            The length of the dry spell to look for.
        dry_threshold : float, optional
            Rainfall threshold below which a day is considered "dry." Defaults to 1.0 mm.

        Returns
        -------
        pd.DataFrame
            Final dry-spell counts in CPT pivot format.
        """
        # 1) Transform daily_df from CDT to a standard table
        daily_df = self.transform_cdt(daily_df)

        # 2) Convert onset and cessation DataFrames from CPT to long format
        onset_long = self._parse_cpt_to_long(onset_df_cpt, value_name="onset")
        cess_long = self._parse_cpt_to_long(cessation_df_cpt, value_name="cessation")

        # 3) Merge onset & cessation by [station, year]
        merged_data = pd.merge(onset_long, cess_long, on=["station", "year"], suffixes=("_onset", "_cess"))

        # Consolidate lat/lon columns
        merged_data["lat"] = merged_data["lat_onset"].fillna(merged_data["lat_cess"])
        merged_data["lon"] = merged_data["lon_onset"].fillna(merged_data["lon_cess"])
        merged_data.drop(columns=["lat_onset", "lat_cess", "lon_onset", "lon_cess"], inplace=True)

        # 4) Loop over (station, year) to compute the count of dry spells
        results = []
        for (stn, yr), subdf in merged_data.groupby(["station", "year"]):
            onset_val = subdf["onset"].values[0]
            cess_val = subdf["cessation"].values[0]
            lat_val = subdf["lat"].values[0]
            lon_val = subdf["lon"].values[0]

            # Filter daily data for this station and year
            stn_data_year = daily_df[
                (daily_df["STATION"] == stn) & (daily_df["DATE"].dt.year == yr)
            ].copy()

            # Replace -99.0 with NaN
            stn_data_year.loc[:, "VALUE"] = stn_data_year["VALUE"].replace(-99.0, np.nan)

            # Convert the daily values to a NumPy array
            x = stn_data_year["VALUE"].values

            # Apply count_dry_spells
            nb_dry_spells = self.count_dry_spells(x, onset_val, cess_val, dry_spell_length, dry_threshold)

            results.append({
                "year": yr,
                "station": stn,
                "lat": lat_val,
                "lon": lon_val,
                "dry_spells_count": nb_dry_spells
            })

        df_res = pd.DataFrame(results)

        # 5) Pivot back to CPT format
        df_pivot = df_res.pivot(index="year", columns="station", values="dry_spells_count").reset_index()
        df_pivot.rename(columns={"year": "STATION"}, inplace=True)

        # Build LAT and LON rows using the first occurrence of each station in df_res
        station_metadata = df_res.groupby("station")[["lat", "lon"]].first().reset_index()

        lat_row = pd.DataFrame(
            [["LAT"] + station_metadata.set_index("station").loc[df_pivot.columns[1:], "lat"].tolist()],
            columns=df_pivot.columns
        )
        lon_row = pd.DataFrame(
            [["LON"] + station_metadata.set_index("station").loc[df_pivot.columns[1:], "lon"].tolist()],
            columns=df_pivot.columns
        )

        # Concatenate lat, lon, and pivot
        df_final = pd.concat([lat_row, lon_row, df_pivot], ignore_index=True)

        return df_final

    def compute(
        self,
        daily_data,
        onset_date,
        cessation_date,
        dry_spell_length,
        dry_threshold,
        nb_cores
    ):
        """
        Compute the number of dry spells for each pixel within the onset and cessation period
        in a daily xarray DataArray.

        Parameters
        ----------
        daily_data : xarray.DataArray
            Daily rainfall data, coords = (T, Y, X).
        onset_date : xarray.DataArray
            DataArray containing onset dates for each pixel.
        cessation_date : xarray.DataArray
            DataArray containing cessation dates for each pixel.
        dry_spell_length : int
            The length of a dry spell to count.
        dry_threshold : float
            Rainfall threshold to classify a day as "dry."
        nb_cores : int
            Number of parallel processes to use.

        Returns
        -------
        xarray.DataArray
            An array with the count of dry spells per pixel.
        """
        # Ensure alignment
        cessation_date["T"] = onset_date["T"]
        cessation_date, onset_date = xr.align(cessation_date, onset_date)
        daily_data = daily_data.sel(
            X=onset_date.coords["X"],
            Y=onset_date.coords["Y"]
        )

        years = np.unique(daily_data["T"].dt.year.to_numpy())

        # Prepare chunk sizes for parallelization
        chunksize_x = int(np.round(len(daily_data.get_index("X")) / nb_cores))
        chunksize_y = int(np.round(len(daily_data.get_index("Y")) / nb_cores))

        store_nb_dryspell = []

        for i in years:
            # Select data for the current year
            year_data = daily_data.sel(T=str(i))
            year_cessation_date = cessation_date.sel(T=str(i)).squeeze()
            year_onset_date = onset_date.sel(T=str(i)).squeeze()
            
            # Set up parallel processing
            client = Client(n_workers=nb_cores, threads_per_worker=1)
            result = xr.apply_ufunc(
                self.count_dry_spells,
                year_data.chunk({"Y": chunksize_y, "X": chunksize_x}),
                year_onset_date.chunk({"Y": chunksize_y, "X": chunksize_x}),
                year_cessation_date.chunk({"Y": chunksize_y, "X": chunksize_x}),
                input_core_dims=[("T",), (), ()],
                vectorize=True,
                kwargs={
                    "dry_spell_length": dry_spell_length,
                    "dry_threshold": dry_threshold,
                },
                output_core_dims=[()],
                dask="parallelized",
                output_dtypes=["float"],
            )
            result_ = result.compute()
            client.close()

            store_nb_dryspell.append(result_)

        # Concatenate final result
        store_nb_dryspell = xr.concat(store_nb_dryspell, dim="T")
        store_nb_dryspell["T"] = onset_date["T"]
        store_nb_dryspell.name = "Count_dryspell"

        return store_nb_dryspell


class WAS_count_wet_spells:

    """
    Class for computing the **number of wet spells** of a specified length
    occurring between the onset and cessation dates of the rainy season.

    This class is designed to quantify periods of consecutive rainy days during the
    growing season — a key indicator for assessing moisture availability, crop growth,
    and potential waterlogging risk in West African agriculture.

    Key Features:
    - Counts wet spells (consecutive days ≥ ``wet_threshold`` mm) between onset and cessation
    - Accepts pre-computed onset and cessation dates (xarray DataArrays or CPT DataFrames)
    - Supports both station-based (CDT format) and gridded (xarray) daily rainfall data
    - Efficient parallel computation using Dask for large-scale gridded analysis
    - Outputs in CPT-compatible wide format for station data

    Typical Use Case:
    After calculating onset (e.g., ``WAS_compute_onset``) and cessation
    (e.g., ``WAS_compute_cessation``), use this class to count sequences of wet days
    (e.g., 5-day, 7-day wet spells) that support crop development or indicate flood risk.

    References:
    - AGRHYMET Regional Centre operational wet spell monitoring
    - Sivakumar (1991): Rainfall variability and wet spell analysis in Sahelian agriculture
    - Common practice in West African national meteorological and agricultural services

    Parameters
    ----------
    None (no initialization parameters required — all criteria passed at compute time)

    Methods
    -------
    compute_insitu(daily_df, onset_df_cpt, cessation_df_cpt, wet_spell_length, wet_threshold=1.0)
        Compute number of wet spells for station data (CDT input, CPT output).

    compute(daily_data, onset_date, cessation_date, wet_spell_length, wet_threshold, nb_cores)
        Compute number of wet spells for gridded xarray data (parallelized).

    Examples
    --------
    Count number of 7-day wet spells between onset and cessation (gridded):

    >>> wet_calc = WAS_count_wet_spells()
    >>> wet_spell_count = wet_calc.compute(
    ...     daily_pr,           # xarray.DataArray daily rainfall (mm)
    ...     onset_da,           # xarray.DataArray of onset dates (day-of-year)
    ...     cessation_da,       # xarray.DataArray of cessation dates
    ...     wet_spell_length=7,
    ...     wet_threshold=1.0,  # mm/day
    ...     nb_cores=8
    ... )

    For station data (CDT rainfall, CPT onset/cessation):

    >>> wet_cpt = wet_calc.compute_insitu(
    ...     daily_cdt_df,
    ...     onset_cpt_df,
    ...     cessation_cpt_df,
    ...     wet_spell_length=5,
    ...     wet_threshold=2.0
    ... )
    """

    @staticmethod
    def count_wet_spells(x, onset_date, cessation_date, wet_spell_length, wet_threshold):
        """
        Count the number of wet spells of a specific length between onset and cessation dates.

        Parameters
        ----------
        x : array-like
            Daily rainfall values.
        onset_date : int
            Start index for the calculation (onset date).
        cessation_date : int
            End index for the calculation (cessation date).
        wet_spell_length : int
            The length of a wet spell to count.
        wet_threshold : float
            Rainfall threshold to classify a day as "wet."

        Returns
        -------
        int or float
            The number of wet spells of the specified length (NaN if data is invalid).
        """
        mask = (
            np.isfinite(x).any()
            and np.isfinite(onset_date)
            and np.isfinite(cessation_date)
        )
        if not mask:
            return np.nan

        # Convert to int and prevent out-of-bounds
        onset_date = int(onset_date)
        cessation_date = int(cessation_date)
        if onset_date < 0 or cessation_date < 0 or onset_date >= len(x):
            return np.nan
        if cessation_date >= len(x):
            cessation_date = len(x) - 1

        wet_spells_count = 0
        current_wet_days = 0

        for day in range(onset_date, cessation_date + 1):
            if x[day] >= wet_threshold:
                current_wet_days += 1
            else:
                if current_wet_days == wet_spell_length:
                    wet_spells_count += 1
                current_wet_days = 0

        # Check if the last run of wet days also qualifies
        if current_wet_days == wet_spell_length:
            wet_spells_count += 1

        return wet_spells_count

    @staticmethod
    def _parse_cpt_to_long(df_cpt, value_name="onset_or_cessation"):
        """
        Convert a CPT-format DataFrame into a long DataFrame with columns:
            [year, station, value_name, lat, lon]

        Assumes:
         - Row 0: ["LAT", lat_stn1, lat_stn2, ...]
         - Row 1: ["LON", lon_stn1, lon_stn2, ...]
         - Rows 2+: [year, station1_val, station2_val, ...]

        Parameters
        ----------
        df_cpt : pd.DataFrame
            DataFrame in CPT-wide format (as returned by certain compute_insitu methods).
        value_name : str
            Name for the output column containing the values (e.g. "onset", "cessation").

        Returns
        -------
        pd.DataFrame
            Columns: [station, year, <value_name>, lat, lon]
        """
        # Row 0 = LAT, Row 1 = LON
        lat_row = df_cpt.iloc[0, 1:].values  # all station lat
        lon_row = df_cpt.iloc[1, 1:].values  # all station lon

        # Station names (from columns)
        station_cols = df_cpt.columns[1:].tolist()

        # Rows from index=2 => year + station values
        df_years = df_cpt.iloc[2:].copy()
        df_years.reset_index(drop=True, inplace=True)
        df_years.rename(columns={"STATION": "year"}, inplace=True)

        # Melt (wide -> long)
        df_long = df_years.melt(
            id_vars=["year"],
            var_name="station",
            value_name=value_name
        )
        df_long["year"] = pd.to_numeric(df_long["year"], errors="coerce")

        # Map station -> lat/lon
        lat_map = dict(zip(station_cols, lat_row))
        lon_map = dict(zip(station_cols, lon_row))
        df_long["lat"] = df_long["station"].map(lat_map)
        df_long["lon"] = df_long["station"].map(lon_map)

        return df_long

    @staticmethod
    def transform_cdt(df):
        """
        Transform a CDT-format DataFrame into a standard table.

        CDT format assumptions:
         - Row 0 = LON
         - Row 1 = LAT
         - Row 2 = ELEV
         - Rows 3+ = daily data, 'ID' column has dates in YYYYMMDD.

        Returns a DataFrame with columns:
          [DATE, STATION, VALUE, LON, LAT, ELEV, MEAN_ANNUAL_RAINFALL, zonename]
        """
        # Example reuse from previous classes (adjust for your own logic if needed)

        # 1) Extract metadata (first 3 rows)
        metadata = df.iloc[:3].set_index("ID").T.reset_index()
        metadata.columns = ["STATION", "LON", "LAT", "ELEV"]

        # 2) Extract daily data
        data_part = df.iloc[3:].rename(columns={"ID": "DATE"})
        data_long = data_part.melt(id_vars=["DATE"], var_name="STATION", value_name="VALUE")
        final_df = pd.merge(data_long, metadata, on="STATION")

        # Convert DATE to datetime
        final_df["DATE"] = pd.to_datetime(final_df["DATE"], format="%Y%m%d")

        # Fill missing with -99.0
        final_df["VALUE"] = final_df["VALUE"].fillna(-99.0)

        # Create a complete date range and expand data accordingly
        start_date = final_df["DATE"].min().replace(month=1, day=1)
        end_date = final_df["DATE"].max().replace(month=12, day=31)
        date_range = pd.date_range(start=start_date, end=end_date)
        all_combinations = pd.MultiIndex.from_product(
            [date_range, metadata["STATION"]],
            names=["DATE", "STATION"]
        ).to_frame(index=False)

        final_df = pd.merge(all_combinations, final_df, on=["DATE", "STATION"], how="left")
        final_df["VALUE"] = final_df["VALUE"].fillna(-99.0)

        # Compute mean annual rainfall by station
        annual_rainfall = (
            final_df[final_df["VALUE"] >= 0]
            .groupby(["STATION", final_df["DATE"].dt.year])["VALUE"]
            .sum()
            .reset_index()
        )
        mean_annual_rainfall = annual_rainfall.groupby("STATION")["VALUE"].mean().reset_index()
        mean_annual_rainfall.columns = ["STATION", "MEAN_ANNUAL_RAINFALL"]

        final_df = pd.merge(final_df, mean_annual_rainfall, on="STATION", how="left")

        # Assign a 'zonename'
        def determine_zonename(row):
            if row["LAT"] <= 8:
                return 5
            elif 600 >= row["MEAN_ANNUAL_RAINFALL"] > 400:
                return 3
            elif 400 >= row["MEAN_ANNUAL_RAINFALL"] > 200:
                return 2
            elif 200 >= row["MEAN_ANNUAL_RAINFALL"] > 100:
                return 1
            elif 100 >= row["MEAN_ANNUAL_RAINFALL"] > 75:
                return 0
            else:
                return 4

        final_df["zonename"] = final_df.groupby("STATION", group_keys=False).apply(
            lambda x: x.apply(determine_zonename, axis=1)
        )

        return final_df

    def compute(
        self,
        daily_data,
        onset_date,
        cessation_date,
        wet_spell_length,
        wet_threshold,
        nb_cores
    ):
        """
        Compute the number of wet spells for each pixel within the onset and cessation period
        in a daily xarray DataArray.

        Parameters
        ----------
        daily_data : xarray.DataArray
            Daily rainfall data, coords = (T, Y, X).
        onset_date : xarray.DataArray
            DataArray containing onset dates for each pixel.
        cessation_date : xarray.DataArray
            DataArray containing cessation dates for each pixel.
        wet_spell_length : int
            The length of a wet spell to count.
        wet_threshold : float
            Rainfall threshold to classify a day as "wet."
        nb_cores : int
            Number of parallel processes to use.

        Returns
        -------
        xarray.DataArray
            Array with the count of wet spells per pixel.
        """
        # Align onset and cessation
        cessation_date["T"] = onset_date["T"]
        cessation_date, onset_date = xr.align(cessation_date, onset_date)

        # Determine each year
        years = np.unique(daily_data["T"].dt.year.to_numpy())

        # Chunk sizes for parallel processing
        chunksize_x = int(np.round(len(daily_data.get_index("X")) / nb_cores))
        chunksize_y = int(np.round(len(daily_data.get_index("Y")) / nb_cores))

        store_nb_wetspell = []

        for i in years:
            # Data for the current year
            year_data = daily_data.sel(T=str(i))
            year_cessation_date = cessation_date.sel(T=str(i)).squeeze()
            year_onset_date = onset_date.sel(T=str(i)).squeeze()
            
            # Set up parallel
            client = Client(n_workers=nb_cores, threads_per_worker=1)
            result = xr.apply_ufunc(
                self.count_wet_spells,
                year_data.chunk({"Y": chunksize_y, "X": chunksize_x}),
                year_onset_date.chunk({"Y": chunksize_y, "X": chunksize_x}),
                year_cessation_date.chunk({"Y": chunksize_y, "X": chunksize_x}),
                input_core_dims=[("T",), (), ()],
                vectorize=True,
                kwargs={
                    "wet_spell_length": wet_spell_length,
                    "wet_threshold": wet_threshold,
                },
                output_core_dims=[()],
                dask="parallelized",
                output_dtypes=["float"],
            )
            result_ = result.compute()
            client.close()

            store_nb_wetspell.append(result_)

        # Concatenate across all years
        store_nb_wetspell = xr.concat(store_nb_wetspell, dim="T")
        store_nb_wetspell["T"] = onset_date["T"]
        store_nb_wetspell.name = "Count_wetspell"

        return store_nb_wetspell

    def compute_insitu(
        self,
        daily_df,
        onset_df_cpt,
        cessation_df_cpt,
        wet_spell_length,
        wet_threshold=1.0
    ):
        """
        Compute the number of wet spells (of length = wet_spell_length) between
        onset and cessation for in-situ stations (CDT data).

        Returns a DataFrame in CPT format:
         - Row 0: ["LAT", lat_station1, lat_station2, ...]
         - Row 1: ["LON", lon_station1, lon_station2, ...]
         - Then one row per year: [year, station1_value, station2_value, ...]

        Parameters
        ----------
        daily_df : pd.DataFrame
            CDT rainfall data (ID column = date, station columns).
        onset_df_cpt : pd.DataFrame
            CPT-format DataFrame with onset dates (same station columns).
        cessation_df_cpt : pd.DataFrame
            CPT-format DataFrame with cessation dates (same station columns).
        wet_spell_length : int
            The length of a wet spell to count.
        wet_threshold : float, optional
            Rainfall threshold classifying a day as "wet." Defaults to 1.0 mm.

        Returns
        -------
        pd.DataFrame
            Final wet-spell counts in CPT pivot format.
        """
        # 1) Transform the daily CDT data into a standard DataFrame
        daily_df = self.transform_cdt(daily_df)

        # 2) Parse onset and cessation from CPT -> long format
        onset_long = self._parse_cpt_to_long(onset_df_cpt, value_name="onset")
        cess_long = self._parse_cpt_to_long(cessation_df_cpt, value_name="cessation")

        # 3) Merge on station/year
        merged_data = pd.merge(onset_long, cess_long, on=["station", "year"], suffixes=("_onset", "_cess"))
        
        # Consolidate lat/lon columns
        merged_data["lat"] = merged_data["lat_onset"].fillna(merged_data["lat_cess"])
        merged_data["lon"] = merged_data["lon_onset"].fillna(merged_data["lon_cess"])
        merged_data.drop(columns=["lat_onset", "lat_cess", "lon_onset", "lon_cess"], inplace=True)

        # 4) Loop through station-year pairs and count wet spells
        results = []
        for (stn, yr), subdf in merged_data.groupby(["station", "year"]):
            onset_val = subdf["onset"].values[0]
            cess_val = subdf["cessation"].values[0]
            lat_val = subdf["lat"].values[0]
            lon_val = subdf["lon"].values[0]

            # Filter daily data for (station, year)
            stn_data_year = daily_df[
                (daily_df["STATION"] == stn) & (daily_df["DATE"].dt.year == yr)
            ].copy()

            # Replace -99 with NaN
            stn_data_year.loc[:, "VALUE"] = stn_data_year["VALUE"].replace(-99.0, np.nan)

            # Convert to array
            x_vals = stn_data_year["VALUE"].values

            # Apply count_wet_spells
            nb_wet_spells = self.count_wet_spells(
                x_vals, onset_val, cess_val,
                wet_spell_length, wet_threshold
            )

            results.append({
                "year": yr,
                "station": stn,
                "lat": lat_val,
                "lon": lon_val,
                "wet_spells_count": nb_wet_spells
            })

        df_res = pd.DataFrame(results)

        # 5) Pivot back to CPT format
        df_pivot = df_res.pivot(
            index="year", columns="station", values="wet_spells_count"
        ).reset_index()
        df_pivot.rename(columns={"year": "STATION"}, inplace=True)

        # Build LAT and LON rows
        station_metadata = df_res.groupby("station")[["lat", "lon"]].first().reset_index()

        lat_row = pd.DataFrame(
            [["LAT"] + station_metadata.set_index("station").loc[df_pivot.columns[1:], "lat"].tolist()],
            columns=df_pivot.columns
        )
        lon_row = pd.DataFrame(
            [["LON"] + station_metadata.set_index("station").loc[df_pivot.columns[1:], "lon"].tolist()],
            columns=df_pivot.columns
        )

        # Concatenate LAT, LON, and pivot
        df_final = pd.concat([lat_row, lon_row, df_pivot], ignore_index=True)

        return df_final


class WAS_count_rainy_days:
    """
    Class for computing the **number of rainy days** (days with precipitation ≥ a threshold)
    between the onset and cessation dates of the rainy season.

    This is a key agro-meteorological indicator used in West Africa to assess seasonal
    moisture availability during the growing period — directly relevant for crop water
    requirements, yield estimation, and drought/vulnerability assessments.

    Key Features:
    - Counts days where daily rainfall ≥ ``rain_threshold`` (default 0.85 mm)
    - Uses pre-computed onset and cessation dates (xarray DataArrays or CPT DataFrames)
    - Supports both station-based (CDT format) and gridded (xarray) daily rainfall data
    - Efficient parallel computation with Dask for large-scale gridded analysis
    - Outputs in CPT-compatible wide format for station data

    Typical Use Case:
    After calculating onset (e.g., via ``WAS_compute_onset``) and cessation
    (e.g., via ``WAS_compute_cessation``), use this class to quantify the total number
    of rainy days available to crops during the growing window.

    References:
    - AGRHYMET Regional Centre seasonal rainfall monitoring
    - Sivakumar (1991): Rainfall characteristics and agricultural planning in West Africa
    - Common practice in Sahelian and Sudanian national meteorological services

    Parameters
    ----------
    None (no initialization parameters required — all criteria passed at compute time)

    Methods
    -------
    compute_insitu(daily_df, onset_df_cpt, cessation_df_cpt, rain_threshold=0.85)
        Compute number of rainy days for station data (CDT input, CPT output).

    compute(daily_data, onset_date, cessation_date, rain_threshold, nb_cores)
        Compute number of rainy days for gridded xarray data (parallelized).

    Examples
    --------
    Count rainy days (≥ 0.85 mm) between onset and cessation (gridded):

    >>> rainy_calc = WAS_count_rainy_days()
    >>> rainy_days_count = rainy_calc.compute(
    ...     daily_pr,           # xarray.DataArray daily rainfall (mm)
    ...     onset_da,           # xarray.DataArray of onset dates (day-of-year)
    ...     cessation_da,       # xarray.DataArray of cessation dates
    ...     rain_threshold=0.85,
    ...     nb_cores=8
    ... )

    For station data (CDT rainfall, CPT onset/cessation):

    >>> rainy_cpt = rainy_calc.compute_insitu(
    ...     daily_cdt_df,
    ...     onset_cpt_df,
    ...     cessation_cpt_df,
    ...     rain_threshold=1.0
    ... )
    """

    @staticmethod
    def transform_cdt(df):
        """
        Transform a DataFrame in CDT format into a standardized long DataFrame.

        CDT format assumptions:
         - Row 0 = LON
         - Row 1 = LAT
         - Row 2 = ELEV
         - Rows 3+ = daily data with 'ID' column holding dates in YYYYMMDD format.

        This method returns a DataFrame with columns:
            DATE, STATION, VALUE, LON, LAT, ELEV, (optional) MEAN_ANNUAL_RAINFALL, zonename
        """

        # 1) Extract metadata (first 3 rows)
        #    - 'ID' column in these rows has labels ["LON", "LAT", "ELEV"]
        metadata = df.iloc[:3].set_index("ID").T.reset_index()
        metadata.columns = ["STATION", "LON", "LAT", "ELEV"]

        # 2) Extract the daily data portion (from row 3 onward); rename "ID" -> "DATE"
        data_part = df.iloc[3:].rename(columns={"ID": "DATE"})

        # Melt to long format: columns = ["DATE", "STATION", "VALUE"]
        data_long = data_part.melt(
            id_vars=["DATE"],
            var_name="STATION",
            value_name="VALUE"
        )

        # Merge station metadata
        final_df = pd.merge(data_long, metadata, on="STATION")

        # Convert "DATE" from string YYYYMMDD to datetime
        final_df["DATE"] = pd.to_datetime(final_df["DATE"], format="%Y%m%d")

        # 3) Ensure a complete date range from Jan 1 of earliest year to Dec 31 of latest year
        start_date = final_df["DATE"].min().replace(month=1, day=1)
        end_date = final_df["DATE"].max().replace(month=12, day=31)
        date_range = pd.date_range(start=start_date, end=end_date)

        # Create all (date, station) pairs so we don't miss any station or date
        all_combinations = pd.MultiIndex.from_product(
            [date_range, metadata["STATION"]],
            names=["DATE", "STATION"]
        ).to_frame(index=False)

        # Merge to fill in missing rows
        final_df = pd.merge(all_combinations, final_df, on=["DATE", "STATION"], how="left")

        # Fill missing rainfall values with -99.0
        final_df["VALUE"] = final_df["VALUE"].fillna(-99.0)

        # 4) Compute mean annual rainfall per station for classification
        annual_rainfall = (
            final_df[final_df["VALUE"] >= 0]
            .groupby(["STATION", final_df["DATE"].dt.year])["VALUE"]
            .sum()
            .reset_index()
        )
        mean_annual_rainfall = annual_rainfall.groupby("STATION")["VALUE"].mean().reset_index()
        mean_annual_rainfall.columns = ["STATION", "MEAN_ANNUAL_RAINFALL"]
        final_df = pd.merge(final_df, mean_annual_rainfall, on="STATION", how="left")

        # Assign a 'zonename' 
        def determine_zonename(row):
            if row["LAT"] <= 8:
                return 5
            elif 600 >= row["MEAN_ANNUAL_RAINFALL"] > 400:
                return 3
            elif 400 >= row["MEAN_ANNUAL_RAINFALL"] > 200:
                return 2
            elif 200 >= row["MEAN_ANNUAL_RAINFALL"] > 100:
                return 1
            elif 100 >= row["MEAN_ANNUAL_RAINFALL"] > 75:
                return 0
            else:
                return 4

        final_df["zonename"] = final_df.groupby("STATION", group_keys=False).apply(
            lambda x: x.apply(determine_zonename, axis=1)
        )

        return final_df

    @staticmethod
    def count_rainy_days(x, onset_date, cessation_date, rain_threshold):
        """
        Count the number of rainy days between onset and cessation dates.

        Parameters
        ----------
        x : array-like
            Daily rainfall values.
        onset_date : int
            Start index for the calculation (onset date).
        cessation_date : int
            End index for the calculation (cessation date).
        rain_threshold : float
            Rainfall threshold to classify a day as "rainy."

        Returns
        -------
        int or float
            Number of rainy days (returns NaN if data is invalid).
        """
        mask = (
            np.isfinite(x).any()
            and np.isfinite(onset_date)
            and np.isfinite(cessation_date)
        )
        if not mask:
            return np.nan

        # Convert onset and cessation indices to integers
        onset_date = int(onset_date)
        cessation_date = int(cessation_date)

        # Prevent out-of-bounds indices
        if onset_date < 0 or cessation_date < 0 or onset_date >= len(x):
            return np.nan
        if cessation_date >= len(x):
            cessation_date = len(x) - 1  # Truncate if needed

        rainy_days_count = 0
        for day in range(onset_date, cessation_date + 1):
            if x[day] >= rain_threshold:
                rainy_days_count += 1

        return rainy_days_count

    def compute(
        self,
        daily_data,
        onset_date,
        cessation_date,
        rain_threshold,
        nb_cores
    ):
        """
        Compute the number of rainy days for each pixel between onset and cessation dates.

        Parameters
        ----------
        daily_data : xarray.DataArray
            Daily rainfall data, coords = (T, Y, X).
        onset_date : xarray.DataArray
            DataArray containing onset dates for each pixel.
        cessation_date : xarray.DataArray
            DataArray containing cessation dates for each pixel.
        rain_threshold : float
            Rainfall threshold to classify a day as "rainy."
        nb_cores : int
            Number of parallel processes to use.

        Returns
        -------
        xarray.DataArray
            Array with the count of rainy days per pixel.
        """
        # Align onset and cessation dates
        cessation_date['T'] = onset_date['T']
        cessation_date, onset_date = xr.align(cessation_date, onset_date)

        # Compute year range
        years = np.unique(daily_data['T'].dt.year.to_numpy())

        # Prepare chunk sizes
        chunksize_x = int(np.round(len(daily_data.get_index("X")) / nb_cores))
        chunksize_y = int(np.round(len(daily_data.get_index("Y")) / nb_cores))

        store_nb_rainy_days = []

        for i in years:
            # Select data for the current year
            year_data = daily_data.sel(T=str(i))
            year_cessation_date = cessation_date.sel(T=str(i)).squeeze()
            year_onset_date = onset_date.sel(T=str(i)).squeeze()
            
            # Set up parallel processing
            client = Client(n_workers=nb_cores, threads_per_worker=1)
            result = xr.apply_ufunc(
                self.count_rainy_days,
                year_data.chunk({'Y': chunksize_y, 'X': chunksize_x}),
                year_onset_date.chunk({'Y': chunksize_y, 'X': chunksize_x}),
                year_cessation_date.chunk({'Y': chunksize_y, 'X': chunksize_x}),
                input_core_dims=[('T',), (), ()],
                vectorize=True,
                kwargs={'rain_threshold': rain_threshold},
                output_core_dims=[()],
                dask='parallelized',
                output_dtypes=['float'],
            )
            result_ = result.compute()
            client.close()

            store_nb_rainy_days.append(result_)

        # Concatenate the final result
        store_nb_rainy_days = xr.concat(store_nb_rainy_days, dim="T")
        store_nb_rainy_days['T'] = onset_date['T']
        store_nb_rainy_days.name = "nb_rainy_days"

        return store_nb_rainy_days

    @staticmethod
    def _parse_cpt_to_long(df_cpt, value_name="onset_or_cessation"):
        """
        Convert a DataFrame in CPT format (like the one returned by 'compute_insitu')
        into a long format DataFrame: columns = [year, station, value_name, lat, lon].

        Parameters
        ----------
        df_cpt : pd.DataFrame
            - Row 0: ["LAT", lat_stn1, lat_stn2, ...]
            - Row 1: ["LON", lon_stn1, lon_stn2, ...]
            - Rows 2+: [year, station1_value, station2_value, ...]

        value_name : str
            Name for the column containing the values (e.g., "onset", "cessation").

        Returns
        -------
        df_long : pd.DataFrame
            Columns = [station, year, value_name, lat, lon]
        """
        # 1) Extract row 0 (LAT) and row 1 (LON)
        lat_row = df_cpt.iloc[0, 1:].values
        lon_row = df_cpt.iloc[1, 1:].values

        # 2) Extract station names (the columns) to map lat/lon
        station_names = df_cpt.columns[1:].tolist()

        # 3) Extract years + values
        df_years = df_cpt.iloc[2:].copy()
        df_years.reset_index(drop=True, inplace=True)
        df_years.rename(columns={"STATION": "year"}, inplace=True)

        # 4) Reshape to long format
        df_long = df_years.melt(
            id_vars=["year"],
            var_name="station",
            value_name=value_name
        )
        df_long["year"] = pd.to_numeric(df_long["year"], errors="coerce")

        # 5) Add LAT/LON information
        lat_map = dict(zip(station_names, lat_row))
        lon_map = dict(zip(station_names, lon_row))
        df_long["lat"] = df_long["station"].map(lat_map)
        df_long["lon"] = df_long["station"].map(lon_map)

        return df_long

    def compute_insitu(
        self,
        daily_df,
        onset_df_cpt,
        cessation_df_cpt,
        rain_threshold=0.85
    ):
        """
        Compute, for in-situ stations (CDT data), the number of rainy days between
        onset and cessation, for each station and year.

        Parameters
        ----------
        daily_df : pd.DataFrame
            CDT precipitation data (ID column = date; columns = stations).
            Follows the standard CDT format.
        onset_df_cpt : pd.DataFrame
            Result of ``WAS_compute_onset.compute_insitu(...)`` for onset (CPT format).
        cessation_df_cpt : pd.DataFrame
            Same format for cessation (CPT format).
        rain_threshold : float, optional
            Precipitation threshold for counting a day as "rainy," by default 0.85 mm.

        Returns
        -------
        df_final : pd.DataFrame
            The count of rainy days in CPT pivot format.
        """
        # 1) Transform daily_df (CDT format) into a standard table
        daily_df = self.transform_cdt(daily_df)

        # 2) Convert onset_df_cpt and cessation_df_cpt to long format
        onset_long = self._parse_cpt_to_long(onset_df_cpt, value_name="onset")
        cess_long = self._parse_cpt_to_long(cessation_df_cpt, value_name="cessation")

        # 3) Merge onset & cessation => single DataFrame
        merged_onset_cess = pd.merge(
            onset_long,
            cess_long,
            on=["station", "year"],
            suffixes=("_onset", "_cess")
        )

        # Consolidate lat/lon columns
        merged_onset_cess["lat"] = merged_onset_cess["lat_onset"].fillna(
            merged_onset_cess["lat_cess"]
        )
        merged_onset_cess["lon"] = merged_onset_cess["lon_onset"].fillna(
            merged_onset_cess["lon_cess"]
        )
        merged_onset_cess.drop(
            columns=["lat_onset", "lat_cess", "lon_onset", "lon_cess"],
            inplace=True
        )

        # 4) Loop over (station, year) to compute rainy-day counts
        results = []
        for (stn, yr), subdf in merged_onset_cess.groupby(["station", "year"]):
            onset_val = subdf["onset"].values[0]
            cess_val = subdf["cessation"].values[0]
            lat_val = subdf["lat"].values[0]
            lon_val = subdf["lon"].values[0]

            # Filter daily_df for this station and year
            stn_year_data = daily_df[
                (daily_df["STATION"] == stn) & (daily_df["DATE"].dt.year == yr)
            ].copy()

            # Replace -99 with NaN
            stn_year_data.loc[:, "VALUE"] = stn_year_data["VALUE"].replace(-99.0, np.nan)

            # Convert to array
            x_values = stn_year_data["VALUE"].values

            # Apply count_rainy_days
            nb_rainy = self.count_rainy_days(
                x_values, onset_val, cess_val, rain_threshold
            )

            results.append({
                "year": yr,
                "station": stn,
                "lat": lat_val,
                "lon": lon_val,
                "nb_rainy_days": nb_rainy
            })

        df_res = pd.DataFrame(results)

        # 5) Pivot to CPT format
        df_pivot = df_res.pivot(index="year", columns="station", values="nb_rainy_days")
        df_pivot.reset_index(inplace=True)
        df_pivot.rename(columns={"year": "STATION"}, inplace=True)

        # Build LAT and LON rows
        station_metadata = df_res.groupby("station")[["lat", "lon"]].first().reset_index()

        lat_row = pd.DataFrame(
            [["LAT"] + station_metadata.set_index("station").loc[df_pivot.columns[1:], "lat"].tolist()],
            columns=df_pivot.columns
        )
        lon_row = pd.DataFrame(
            [["LON"] + station_metadata.set_index("station").loc[df_pivot.columns[1:], "lon"].tolist()],
            columns=df_pivot.columns
        )

        # Concatenate LAT, LON, and pivot
        df_final = pd.concat([lat_row, lon_row, df_pivot], ignore_index=True)

        return df_final
class ExtremeType(Enum):
    """Type of temperature extreme."""
    HOT = "hot"      # Days above upper percentile (e.g., TX90p, TN90p)
    COLD = "cold"    # Days below lower percentile (e.g., TX10p, TN10p)

class WAS_TempPercentileIndices:
    """
    Implementation of ETCCDI temperature percentile-based extreme indices.

    Computes the annual percentage of days/nights exceeding (hot) or falling below (cold)
    a specified percentile threshold, using a 5-day centered window for percentile calculation
    (standard ETCCDI methodology).

    Supported ETCCDI Indices:
    - **TX90p** / **TN90p**: Percentage of hot days (TX > 90th) / hot nights (TN > 90th)
    - **TX10p** / **TN10p**: Percentage of cold days (TX < 10th) / cold nights (TN < 10th)
    - Custom percentiles are also supported (e.g., TX95p, TN99p, TX05p)

    Key Features:
    - 5-day centered window for calendar-day percentiles (avoids day-of-year bias)
    - Proper leap-year handling (Feb 29 mapped to Feb 28)
    - Seasonal filtering option
    - Bootstrap confidence intervals (optional)
    - Works with both station (CDT format) and gridded (xarray) data

    References:
    - ETCCDI Climate Change Indices (2009)
    - Zhang et al. (2011): Indices for monitoring changes in extremes
    - Sillmann et al. (2013): Climate extremes indices in the CMIP5 ensemble

    Parameters
    ----------
    base_period : slice
        Climatological base period for percentile calculation, e.g. slice("1961", "1990")
    percentile : float, default=90
        Percentile threshold:
        - For hot extremes: 90, 95, 99 (days above percentile)
        - For cold extremes: 10, 5, 1 (days below percentile)
    season : list of int, optional
        Months to include in analysis (e.g., [6,7,8] for JJA)
    var_type : {'TMAX', 'TMIN'}, default='TMAX'
        Temperature variable: 'TMAX' (TX - daily max) or 'TMIN' (TN - daily min)
    extreme_type : {'hot', 'cold'}, default='hot'
        Type of extreme: 'hot' (above percentile) or 'cold' (below percentile)
    bootstrap_samples : int, default=10
        Number of bootstrap samples for confidence interval calculation
    min_base_years : int, default=15
        Minimum number of years required in base period (warning if fewer)

    Examples
    --------
    Standard TX90p (Warm days percentage):

    >>> calc = WAS_TempPercentileIndices(
    ...     base_period=slice("1961", "1990"),
    ...     percentile=90,
    ...     var_type='TMAX',
    ...     extreme_type='hot'
    ... )
    >>> tx90p = calc.compute(tmax_da)

    Cold nights (TN10p) for JJA season:

    >>> cold_nights = WAS_TempPercentileIndices(
    ...     base_period=slice("1961", "1990"),
    ...     percentile=10,
    ...     var_type='TMIN',
    ...     extreme_type='cold',
    ...     season=[6, 7, 8]
    ... ).compute(tmin_da)
    """
    
    def __init__(
        self,
        base_period: slice,
        percentile: float = 90,
        season: Optional[List[int]] = None,
        var_type: str = 'TMAX',          # 'TMAX' or 'TMIN'
        extreme_type: str = 'hot',         # 'hot' or 'cold'
        bootstrap_samples: int = 10,
        min_base_years: int = 15
    ):
        """
        Parameters
        ----------
        base_period : slice
            Slice for base period years, e.g., slice("1961", "1990")
        percentile : float
            Percentile value:
            - For hot extremes: 90, 95, 99 (days above percentile)
            - For cold extremes: 10, 5, 1 (days below percentile)
        season : list, optional
            Months to consider (e.g., [6, 7, 8] for JJA)
        var_type : str
            Temperature variable type: 'TMAX' (TX) or 'TMIN' (TN)
        extreme_type : str
            Type of extreme: 'hot' or 'cold'
        bootstrap_samples : int
            Number of bootstrap samples for confidence intervals
        min_base_years : int
            Minimum years required in base period
        """
        self.base_period = base_period
        self.percentile = percentile
        self.season = season
        self.var_type = var_type
        self.extreme_type = ExtremeType(extreme_type.lower())
        self.bootstrap_samples = bootstrap_samples
        self.min_base_years = min_base_years
        
        # Validate inputs
        self._validate_inputs()
        
        # Set index name
        self.index_name = self._generate_index_name()
    
    def _validate_inputs(self):
        """Validate all input parameters."""
        # Validate percentile based on extreme type
        if self.extreme_type == ExtremeType.HOT:
            if not (50 <= self.percentile < 100):
                raise ValueError(
                    f"For hot extremes, percentile must be >= 50 and < 100, "
                    f"got {self.percentile}. Common values: 90, 95, 99."
                )
        elif self.extreme_type == ExtremeType.COLD:
            if not (0 < self.percentile <= 50):
                raise ValueError(
                    f"For cold extremes, percentile must be > 0 and <= 50, "
                    f"got {self.percentile}. Common values: 10, 5, 1."
                )
        
        # Validate variable type
        if self.var_type not in ['TMAX', 'TMIN']:
            raise ValueError(f"var_type must be 'TMAX' or 'TMIN', got {self.var_type}")
    
    def _generate_index_name(self) -> str:
        """Generate the proper ETCCDI index name."""
        if self.var_type == 'TMAX':
            prefix = "TX"
        else:
            prefix = "TN"
        
        return f"{prefix}{int(self.percentile)}p"
    
    @staticmethod
    def transform_cdt(df: pd.DataFrame) -> pd.DataFrame:
        """Transform CDT format to long format DataFrame."""
        # Extract metadata
        meta = df.iloc[:3].set_index("ID").T.reset_index()
        meta.columns = ["STATION", "LON", "LAT", "ELEV"]
        
        # Extract data
        data = df.iloc[3:].rename(columns={"ID": "DATE"})
        data = data.melt(
            id_vars=["DATE"],
            var_name="STATION",
            value_name="VALUE"
        )
        
        # Merge and clean
        final = pd.merge(data, meta, on="STATION")
        final["DATE"] = pd.to_datetime(final["DATE"], format="%Y%m%d")
        final["VALUE"] = pd.to_numeric(final["VALUE"], errors='coerce')
        
        # Convert -99.0 to NaN
        final["VALUE"] = final["VALUE"].replace(-99.0, np.nan)
        
        return final
    
    def _validate_base_period(self, years: np.ndarray) -> None:
        """Validate that base period has sufficient data."""
        unique_years = np.unique(years)
        if len(unique_years) < self.min_base_years:
            warnings.warn(
                f"Base period has only {len(unique_years)} years, "
                f"which is less than recommended minimum of {self.min_base_years}."
            )
    
    def _calculate_percentile_thresholds(
        self, 
        temp_data: pd.DataFrame,
        confidence: bool = False
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Calculate percentile thresholds using 5-day centered window.
        
        For hot extremes: Calculate upper percentile (e.g., 90th)
        For cold extremes: Calculate lower percentile (e.g., 10th)
        """
        # Pivot to wide format (stations as columns)
        wide = temp_data.pivot(index="DATE", columns="STATION", values="VALUE")
        
        # Handle leap days: map February 29 to February 28
        doy_series = wide.index.dayofyear.replace(366, 365)
        wide_doy = wide.copy()
        wide_doy.index = doy_series
        
        # For each day of year (1-365), use 5-day centered window
        thresholds = {}
        ci_lower = {}
        ci_upper = {}
        
        for doy in range(1, 366):
            # Create 5-day window (centered on doy, circular for year boundaries)
            window_days = []
            for offset in [-2, -1, 0, 1, 2]:
                window_doy = ((doy + offset - 1) % 365) + 1
                window_days.append(window_doy)
            
            # Get all data for this window across all years
            window_mask = wide_doy.index.isin(window_days)
            window_data = wide_doy[window_mask]
            
            if len(window_data) > 0:
                # Calculate percentile for each station
                threshold_values = np.nanpercentile(
                    window_data.values, 
                    self.percentile, 
                    axis=0
                )
                thresholds[doy] = threshold_values
                
                # Bootstrap confidence intervals if requested
                if confidence:
                    n_bootstrap = min(self.bootstrap_samples, len(window_data))
                    bootstrap_samples = []
                    
                    for _ in range(n_bootstrap):
                        idx = np.random.choice(
                            len(window_data), 
                            size=len(window_data), 
                            replace=True
                        )
                        sample = window_data.iloc[idx]
                        sample_percentile = np.nanpercentile(
                            sample.values, 
                            self.percentile, 
                            axis=0
                        )
                        bootstrap_samples.append(sample_percentile)
                    
                    bootstrap_array = np.array(bootstrap_samples)
                    ci_lower[doy] = np.nanpercentile(bootstrap_array, 2.5, axis=0)
                    ci_upper[doy] = np.nanpercentile(bootstrap_array, 97.5, axis=0)
        
        # Convert to DataFrames
        thresholds_df = pd.DataFrame(thresholds, index=wide.columns).T
        thresholds_df.index.name = "DOY"
        thresholds_df = thresholds_df.reset_index().melt(
            id_vars="DOY",
            var_name="STATION",
            value_name="THRESHOLD"
        )
        
        if confidence:
            ci_lower_df = pd.DataFrame(ci_lower, index=wide.columns).T
            ci_lower_df = ci_lower_df.reset_index().melt(
                id_vars="DOY",
                var_name="STATION",
                value_name="CI_LOWER"
            )
            
            ci_upper_df = pd.DataFrame(ci_upper, index=wide.columns).T
            ci_upper_df = ci_upper_df.reset_index().melt(
                id_vars="DOY",
                var_name="STATION",
                value_name="CI_UPPER"
            )
            
            return thresholds_df, ci_lower_df, ci_upper_df
        
        return thresholds_df
    
    def _calculate_extreme_days(
        self, 
        temp_data: pd.DataFrame, 
        thresholds: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate number of extreme temperature days.
        
        For hot extremes: days when temperature > percentile (e.g., TX90p, TN90p)
        For cold extremes: days when temperature < percentile (e.g., TX10p, TN10p)
        """
        # Merge thresholds with data
        temp_data["DOY"] = temp_data["DATE"].dt.dayofyear.replace(366, 365)
        merged = pd.merge(temp_data, thresholds, on=["STATION", "DOY"], how="left")
        
        # Determine if day is extreme based on extreme type
        if self.extreme_type == ExtremeType.HOT:
            # Hot extremes: temperature > percentile threshold
            merged["IS_EXTREME"] = (merged["VALUE"] > merged["THRESHOLD"]).astype(float)
            merged["EXTREME_TYPE"] = "hot"
        elif self.extreme_type == ExtremeType.COLD:
            # Cold extremes: temperature < percentile threshold
            merged["IS_EXTREME"] = (merged["VALUE"] < merged["THRESHOLD"]).astype(float)
            merged["EXTREME_TYPE"] = "cold"
        
        # Preserve NaN values from original data
        merged.loc[merged["VALUE"].isna(), "IS_EXTREME"] = np.nan
        
        return merged
    
    def compute_insitu(
        self, 
        df_cdt: pd.DataFrame,
        return_confidence: bool = False
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Compute index for in-situ (station) data.
        """
        # Transform to long format
        df = self.transform_cdt(df_cdt)
        
        # Filter base period
        base_start = int(self.base_period.start)
        base_stop = int(self.base_period.stop)
        df_base = df[
            df["DATE"].dt.year.between(base_start, base_stop)
        ].copy()
        
        # Validate base period
        self._validate_base_period(df_base["DATE"].dt.year.values)
        
        # Calculate thresholds
        if return_confidence:
            thresholds, ci_lower, ci_upper = self._calculate_percentile_thresholds(
                df_base, confidence=True
            )
        else:
            thresholds = self._calculate_percentile_thresholds(df_base, confidence=False)
        
        # Calculate extreme days
        df_extreme = self._calculate_extreme_days(df, thresholds)
        
        # Apply seasonal filter if specified
        if self.season:
            df_extreme = df_extreme[df_extreme["DATE"].dt.month.isin(self.season)]
        
        # Group by year and station
        df_extreme["YEAR"] = df_extreme["DATE"].dt.year
        
        # Calculate percentage of extreme days
        result = df_extreme.groupby(["STATION", "YEAR", "LAT", "LON"]).apply(
            lambda x: (x["IS_EXTREME"].sum() / x["IS_EXTREME"].notna().sum()) * 100
            if x["IS_EXTREME"].notna().sum() > 0 else np.nan
        ).reset_index()
        
        result.columns = ["STATION", "YEAR", "LAT", "LON", self.index_name]
        
        # Format to CDT
        result_cdt = self._format_to_cdt(result)
        
        if return_confidence:
            return result_cdt, (ci_lower, ci_upper)
        
        return result_cdt
    
    def compute(
        self, 
        ds: Union[xr.Dataset, xr.DataArray],
        var_name: Optional[str] = None,
        # chunk_size: Optional[Dict[str, int]] = None,
        parallel: bool = True,
        nb_cores: int = 4
    ) -> xr.DataArray:
        """
        Compute index for xarray data (gridded).
        """
        # Extract DataArray
        if isinstance(ds, xr.Dataset):
            if var_name is None:
                var_name = self.var_type
            da = ds[var_name]
        else:
            da = ds
        
        # Standardize dimension names
        da = self._standardize_dims(da)
        
        # Apply seasonal mask if specified
        if self.season:
            da = da.where(da.time.dt.month.isin(self.season), drop=True)
        
        # Handle chunking for Dask
        if parallel: # and hasattr(da.data, 'chunks'):
            chunk_size = {'y': int(np.round(len(da.get_index("y")) / nb_cores)), 'x': int(np.round(len(da.get_index("x")) / nb_cores))}
            da = da.chunk({'time': -1, **chunk_size})
            
        # Select base period
        da_base = da.sel(time=self.base_period)
        
        # Validate base period
        base_years = np.unique(da_base.time.dt.year.values)
        self._validate_base_period(base_years)
        
        # Calculate thresholds using 5-day centered window
        windowed = da_base.rolling(time=5, center=True, min_periods=1).construct("window")
        
        # Calculate percentile
        thresholds = windowed.groupby("time.dayofyear").quantile(
            self.percentile / 100.0,
            dim=["time", "window"],
            method='linear',
            skipna=True
        )
        
        # Handle leap days
        doy = da.time.dt.dayofyear
        doy_fixed = xr.where(doy == 366, 365, doy)
        
        # Map thresholds to all time steps
        full_thresholds = thresholds.sel(dayofyear=doy_fixed)
        full_thresholds = full_thresholds.drop_vars("dayofyear")
        full_thresholds = full_thresholds.assign_coords(time=da.time)
        
        # Identify extreme days based on extreme type
        if self.extreme_type == ExtremeType.HOT:
            # Hot extremes: temperature > percentile
            is_extreme = (da > full_thresholds).astype(float)
        elif self.extreme_type == ExtremeType.COLD:
            # Cold extremes: temperature < percentile
            is_extreme = (da < full_thresholds).astype(float)
        
        # Preserve NaN values
        is_extreme = is_extreme.where(da.notnull())
        
        # Calculate annual percentage of extreme days
        result = is_extreme.resample(time='YS').mean(dim='time', skipna=True) * 100
        
        # Set metadata
        result.name = self.index_name
        result.attrs.update(self._get_metadata())
        
        return result.compute().drop_vars("quantile").rename({"time": "T", "x": "X", "y": "Y"})
    
    def _standardize_dims(self, da: xr.DataArray) -> xr.DataArray:
        """Standardize dimension names."""
        dim_map = {}
        
        # Identify time dimension
        time_candidates = ['time', 'T', 'date', 'Date']
        for tc in time_candidates:
            if tc in da.dims:
                dim_map[tc] = 'time'
                break
        
        # Identify spatial dimensions
        spatial_pairs = [
            (['lat', 'y', 'latitude', 'Y'], 'lat'),
            (['lon', 'x', 'longitude', 'X'], 'lon')
        ]
        
        for candidates, std_name in spatial_pairs:
            for cand in candidates:
                if cand in da.dims:
                    dim_map[cand] = std_name
                    break
        
        # Rename dimensions if needed
        if dim_map:
            da = da.rename(dim_map)
        
        if 'time' not in da.dims:
            raise ValueError(f"DataArray must have 'time' dimension. Found: {list(da.dims)}")
        
        return da
    
    def _get_metadata(self) -> Dict:
        """Get metadata dictionary for the index."""
        if self.extreme_type == ExtremeType.HOT:
            if self.var_type == 'TMAX':
                long_name = f'Percentage of hot days (TX > {self.percentile}th percentile)'
            else:
                long_name = f'Percentage of hot nights (TN > {self.percentile}th percentile)'
        else:
            if self.var_type == 'TMAX':
                long_name = f'Percentage of cold days (TX < {self.percentile}th percentile)'
            else:
                long_name = f'Percentage of cold nights (TN < {self.percentile}th percentile)'
        
        return {
            'long_name': long_name,
            'units': '%',
            'base_period': f'{self.base_period.start}-{self.base_period.stop}',
            'percentile': self.percentile,
            'variable': self.var_type,
            'extreme_type': self.extreme_type.value,
            'season': self.season if self.season else 'all months',
            'method': '5-day centered window percentile',
            'reference': 'ETCCDI Climate Change Indices (2009)',
            'calculation': f'Annual percentage of days when {self.var_type} is {">" if self.extreme_type == ExtremeType.HOT else "<"} {self.percentile}th percentile of base period'
        }
    
    def _format_to_cdt(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert to CDT format."""
        # Pivot to wide format
        pivot = df.pivot(
            index="YEAR",
            columns="STATION",
            values=self.index_name
        ).reset_index()
        
        # Create metadata rows
        meta = df.groupby("STATION")[["LAT", "LON"]].first()
        stations = pivot.columns[1:]  # Exclude YEAR column
        meta = meta.reindex(stations)
        
        # Create metadata rows
        lat_row = pd.DataFrame([["LAT"] + meta["LAT"].tolist()], 
                              columns=pivot.columns)
        lon_row = pd.DataFrame([["LON"] + meta["LON"].tolist()], 
                              columns=pivot.columns)
        
        # Rename YEAR column to ID for CDT format
        pivot = pivot.rename(columns={"YEAR": "ID"})
        lat_row = lat_row.rename(columns={"YEAR": "ID"})
        lon_row = lon_row.rename(columns={"YEAR": "ID"})
        
        # Combine all rows
        result = pd.concat([lat_row, lon_row, pivot], ignore_index=True)
        
        return result
    
    def get_index_definition(self) -> Dict:
        """Return index definition metadata."""
        definition = self._get_metadata()
        definition['index_name'] = self.index_name
        definition['etccdi_id'] = self._get_etccdi_id()
        return definition
    
    def _get_etccdi_id(self) -> str:
        """Get ETCCDI official ID for the index."""
        # Standard ETCCDI indices
        if self.index_name == "TX90p":
            return "Warm days"
        elif self.index_name == "TN90p":
            return "Warm nights"
        elif self.index_name == "TX10p":
            return "Cold days"
        elif self.index_name == "TN10p":
            return "Cold nights"
        else:
            return f"Custom: {self.index_name}"


# Convenience class creators for standard ETCCDI indices
class ETCCDITempIndices:
    """Factory for creating standard ETCCDI temperature indices."""
    
    @staticmethod
    def hot_days(base_period: slice, season: Optional[List[int]] = None, 
                 percentile: float = 90) -> WAS_TempPercentileIndices:
        """Create calculator for hot days (TX90p)."""
        return WAS_TempPercentileIndices(
            base_period=base_period,
            percentile=percentile,
            var_type='TMAX',
            extreme_type='hot',
            season=season
        )
    
    @staticmethod
    def hot_nights(base_period: slice, season: Optional[List[int]] = None,
                   percentile: float = 90) -> WAS_TempPercentileIndices:
        """Create calculator for hot nights (TN90p)."""
        return WAS_TempPercentileIndices(
            base_period=base_period,
            percentile=percentile,
            var_type='TMIN',
            extreme_type='hot',
            season=season
        )
    
    @staticmethod
    def cold_days(base_period: slice, season: Optional[List[int]] = None,
                  percentile: float = 10) -> WAS_TempPercentileIndices:
        """Create calculator for cold days (TX10p)."""
        return WAS_TempPercentileIndices(
            base_period=base_period,
            percentile=percentile,
            var_type='TMAX',
            extreme_type='cold',
            season=season
        )
    
    @staticmethod
    def cold_nights(base_period: slice, season: Optional[List[int]] = None,
                    percentile: float = 10) -> WAS_TempPercentileIndices:
        """Create calculator for cold nights (TN10p)."""
        return WAS_TempPercentileIndices(
            base_period=base_period,
            percentile=percentile,
            var_type='TMIN',
            extreme_type='cold',
            season=season
        )
class WAS_PrecipIndices:
    """
    Implementation of ETCCDI extreme precipitation indices, specifically R95p and R99p.

    These indices measure the **annual total precipitation from very/extremely wet days**:

    - **R95p**: Total precipitation from days when precipitation ≥ 95th percentile of wet-day precipitation in the base period
    - **R99p**: Total precipitation from days when precipitation ≥ 99th percentile of wet-day precipitation in the base period

    A **wet day** is defined as precipitation ≥ ``wet_day_threshold`` (default 1.0 mm).

    Key Features:
    - Uses only **wet days** in the base period for percentile calculation (ETCCDI standard)
    - Supports seasonal filtering (e.g., JJAS season)
    - Handles both station-based (CDT format) and gridded (xarray) data
    - Proper leap-year handling and missing value treatment
    - Outputs in CDT-compatible format for station data

    References:
    - ETCCDI Climate Change Indices (2009)
    - Zhang et al. (2011): Indices for monitoring changes in extremes
    - Sillmann et al. (2013): Climate extremes indices in the CMIP5 multi-model ensemble

    Parameters
    ----------
    base_period : slice
        Base period for percentile calculation, e.g. slice("1991", "2020")
    percentile : float, default=95
        Percentile threshold (95 → R95p, 99 → R99p)
    season : list of int, optional
        Months to include in analysis (e.g., [6,7,8,9] for JJAS)
    wet_day_threshold : float, default=1.0
        Minimum precipitation amount (mm) to consider a wet day
    min_base_years : int, default=15
        Minimum number of years required in base period (warning issued if fewer)

    Examples
    --------
    Compute R95p for full year:

    >>> r95p_calc = WAS_PrecipIndices(
    ...     base_period=slice("1991", "2020"),
    ...     percentile=95
    ... )
    >>> r95p = r95p_calc.compute(pr_da)  # pr_da is precipitation DataArray

    Compute R99p for JJAS season only:

    >>> r99p_jjas = WAS_PrecipIndices(
    ...     base_period=slice("1991", "2020"),
    ...     percentile=99,
    ...     season=[6,7,8,9]
    ... ).compute(pr_da)
    """
    
    def __init__(
        self,
        base_period: slice,
        percentile: float = 95,
        season: Optional[List[int]] = None,
        wet_day_threshold: float = 1.0,
        min_base_years: int = 15
    ):
        self.base_period = base_period
        self.percentile = percentile
        self.season = season
        self.wet_day_threshold = wet_day_threshold
        self.min_base_years = min_base_years
        self.index_name = f"R{int(self.percentile)}p"
        
        # Validate percentile
        if not (0 < percentile < 100):
            raise ValueError(f"Percentile must be between 0 and 100, got {percentile}")
    
    @staticmethod
    def transform_cdt(df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform CDT format to long format DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame in CDT format
        
        Returns
        -------
        pd.DataFrame
            Long format DataFrame with columns: DATE, STATION, VALUE, LAT, LON
        """
        # Extract metadata (first 3 rows)
        meta = df.iloc[:3].set_index("ID").T.reset_index()
        meta.columns = ["STATION", "LON", "LAT", "ELEV"]
        
        # Extract data (from row 3 onwards)
        data = df.iloc[3:].rename(columns={"ID": "DATE"})
        data = data.melt(
            id_vars=["DATE"],
            var_name="STATION",
            value_name="VALUE"
        )
        
        # Merge and clean
        final = pd.merge(data, meta, on="STATION")
        final["DATE"] = pd.to_datetime(final["DATE"], format="%Y%m%d")
        final["VALUE"] = pd.to_numeric(final["VALUE"], errors='coerce')
        
        # Convert -99.0 to NaN
        final["VALUE"] = final["VALUE"].replace(-99.0, np.nan)
        
        return final
    
    def _validate_base_period(self, years: np.ndarray) -> None:
        """Validate that base period has sufficient data."""
        unique_years = np.unique(years)
        if len(unique_years) < self.min_base_years:
            warnings.warn(
                f"Base period has only {len(unique_years)} years, "
                f"which is less than recommended minimum of {self.min_base_years}."
            )
    
    def _compute_percentile_threshold(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute percentile threshold from base period wet days.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with threshold per station
        """
        # Filter base period
        base_start = int(self.base_period.start)
        base_stop = int(self.base_period.stop)
        df_base = data[
            data["DATE"].dt.year.between(base_start, base_stop)
        ].copy()
        
        # Filter wet days
        df_wet = df_base[df_base["VALUE"] >= self.wet_day_threshold].copy()
        
        # Validate base period
        self._validate_base_period(df_base["DATE"].dt.year.values)
        
        # Compute threshold per station
        thresholds = df_wet.groupby("STATION")["VALUE"].quantile(
            self.percentile / 100.0
        ).reset_index()
        thresholds.columns = ["STATION", "THRESHOLD"]
        
        return thresholds
    
    def compute_insitu(self, df_cdt: pd.DataFrame) -> pd.DataFrame:
        """
        Compute index for in-situ (station) data in CDT format.
        
        Parameters
        ----------
        df_cdt : pd.DataFrame
            Input data in CDT format
        
        Returns
        -------
        pd.DataFrame
            Result in CDT format
        """
        # Transform to long format
        df = self.transform_cdt(df_cdt)
        
        # Compute thresholds from base period
        thresholds = self._compute_percentile_threshold(df)
        
        # Merge thresholds with data
        df = pd.merge(df, thresholds, on="STATION", how="left")
        
        # Identify extreme precipitation days
        df["EXTREME"] = np.where(
            (df["VALUE"] >= self.wet_day_threshold) & 
            (df["VALUE"] > df["THRESHOLD"]),
            df["VALUE"],
            0.0
        )
        
        # Apply seasonal filter if specified
        if self.season:
            df = df[df["DATE"].dt.month.isin(self.season)]
        
        # Group by year and station
        df["YEAR"] = df["DATE"].dt.year
        result = df.groupby(["STATION", "YEAR", "LAT", "LON"])["EXTREME"] \
                   .sum(min_count=1) \
                   .reset_index()
        result.rename(columns={"EXTREME": self.index_name}, inplace=True)
        
        # Convert back to CDT format
        return self._format_to_cdt(result)
    
    def compute(
        self, 
        da: xr.DataArray,
        # chunk_size: Optional[Dict[str, int]] = None,
        parallel: bool = True,
        nb_cores: int = 4
    ) -> xr.DataArray:
        """
        Compute index for xarray DataArray (gridded data).
        
        Parameters
        ----------
        da : xr.DataArray
            Precipitation DataArray with dimensions (time, y, x) or (time, lat, lon)
        chunk_size : dict, optional
            Chunk sizes for parallel processing, e.g., {'y': 100, 'x': 100}
        parallel : bool
            Whether to use Dask for parallel processing
        
        Returns
        -------
        xr.DataArray
            Annual index values
        """
        # Rename dimensions to standard names if needed
        da = self._standardize_dims(da)

        
        # Apply seasonal mask if specified
        if self.season:
            da = da.where(da.time.dt.month.isin(self.season), drop=True)
        
        # Handle chunking for Dask
        if parallel:# and hasattr(da.data, 'chunks'):
            chunk_size = {'y': int(np.round(len(da.get_index("y")) / nb_cores)), 'x': int(np.round(len(da.get_index("x")) / nb_cores))}
            da = da.chunk({'time': -1, **chunk_size})
        
        # Select base period
        da_base = da.sel(time=self.base_period)
        
        # Get wet days in base period
        wet_base = da_base.where(da_base >= self.wet_day_threshold)
        
        # Validate base period
        base_years = np.unique(da_base.time.dt.year.values)
        self._validate_base_period(base_years)
        
        # Compute percentile threshold from base period wet days
        # Using method='linear' for consistency with ETCCDI
        threshold = wet_base.quantile(
            self.percentile / 100.0, 
            dim=['time'],
            method='linear',
            skipna=True
        )
        
        # Identify extreme precipitation days
        # Condition: precipitation >= wet_day_threshold AND > threshold
        extreme = xr.where(
            (da >= self.wet_day_threshold) & (da > threshold),
            da,
            0.0
        )
        
        # Handle leap days by using 'YS' (year start) resampling
        # This avoids issues with February 29th
        result = extreme.resample(time='YS').sum(dim='time', min_count=1)
        
        # Rename result
        result.name = self.index_name
        result.attrs.update({
            'long_name': f'Annual total precipitation when daily precipitation > {self.percentile}th percentile',
            'units': da.attrs.get('units', 'mm'),
            'base_period': f'{self.base_period.start}-{self.base_period.stop}',
            'percentile': self.percentile,
            'wet_day_threshold': self.wet_day_threshold,
            'season': self.season if self.season else 'all months'
        })
        
        return result.compute().drop_vars("quantile").rename({"time": "T", "x": "X", "y": "Y"})
    
    def _standardize_dims(self, da: xr.DataArray) -> xr.DataArray:
        """Standardize dimension names."""
        dim_map = {}
        
        # Identify time dimension
        time_candidates = ['time', 'T', 'date', 'Date']
        for tc in time_candidates:
            if tc in da.dims:
                dim_map[tc] = 'time'
                break
        
        # Identify spatial dimensions
        spatial_pairs = [
            (['lat', 'y', 'latitude', 'Y'], 'y'),
            (['lon', 'x', 'longitude', 'X'], 'x')
        ]
        
        for candidates, std_name in spatial_pairs:
            for cand in candidates:
                if cand in da.dims:
                    dim_map[cand] = std_name
                    break
        
        # Rename dimensions if needed
        if dim_map:
            da = da.rename(dim_map)
        
        # Ensure required dimensions exist
        required_dims = ['time', 'y', 'x']
        for dim in required_dims:
            if dim not in da.dims:
                raise ValueError(f"DataArray must have '{dim}' dimension. Found: {list(da.dims)}")
        
        return da
    
    def _format_to_cdt(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert long format DataFrame to CDT format.
        
        Parameters
        ----------
        df : pd.DataFrame
            Long format DataFrame with columns: STATION, YEAR, index_name, LAT, LON
        
        Returns
        -------
        pd.DataFrame
            DataFrame in CDT format
        """
        # Pivot to wide format
        pivot = df.pivot(
            index="YEAR",
            columns="STATION",
            values=self.index_name
        ).reset_index()
        
        # Create metadata rows
        meta = df.groupby("STATION")[["LAT", "LON"]].first()
        
        # Ensure column order matches metadata
        stations = pivot.columns[1:]  # Exclude YEAR column
        meta = meta.reindex(stations)
        
        # Create metadata rows
        lat_row = pd.DataFrame([["LAT"] + meta["LAT"].tolist()], 
                              columns=pivot.columns)
        lon_row = pd.DataFrame([["LON"] + meta["LON"].tolist()], 
                              columns=pivot.columns)
        
        # Rename YEAR column to ID for CDT format
        pivot = pivot.rename(columns={"YEAR": "ID"})
        lat_row = lat_row.rename(columns={"YEAR": "ID"})
        lon_row = lon_row.rename(columns={"YEAR": "ID"})
        
        # Combine all rows
        result = pd.concat([lat_row, lon_row, pivot], ignore_index=True)
        
        return result
    
    def get_index_definition(self) -> Dict:
        """Return index definition metadata."""
        return {
            'index_name': self.index_name,
            'definition': f'Annual total precipitation from days > {self.percentile}th percentile of wet days (>= {self.wet_day_threshold} mm) in base period',
            'base_period': f'{self.base_period.start}-{self.base_period.stop}',
            'wet_day_threshold': self.wet_day_threshold,
            'season': self.season if self.season else 'all months',
            'etccdi_reference': 'ETCCDI Climate Change Indices',
            'reference': 'Zhang et al. (2011), Weather and Climate Extremes'
        }
class HeatWaveMetric(Enum):
    """Heat wave / warm spell metrics (ETCCDI-style)."""
    WSDI = "WSDI"  # warm spell duration index (days)
    HWF  = "HWF"   # heat wave frequency (events)
    HWDI = "HWDI"  # heat wave duration index (max duration in days)
    HWN  = "HWN"   # alias of HWF (often used)


@dataclass
class HeatWaveDefinition:
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    duration: int
    max_temp: float
    mean_temp: float


# =============================================================================
# Main class
# =============================================================================

class WAS_HeatWaveIndices:
    """
    ETCCDI-like warm spell / heat wave indices for daily temperature.

    Metrics:
    - WSDI: total number of days that belong to spells of >= min_consecutive_days
            where TX > TXp (p-th percentile threshold based on base_period)
    - HWF : number of spells/events per year (runs >= min_consecutive_days)
    - HWDI: maximum run length per year (runs >= min_consecutive_days)

    Threshold definition:
    - Standard ETCCDI practice uses a 5-day centered window for each calendar day.
      Implementation: rolling(time=5,center=True).construct("window")
      then groupby(time.dayofyear).quantile(..., dim=["time","window"])

    Dask/xarray robustness:
    - quantile along time requires the 'time' dimension to be a single chunk.
      We enforce .chunk({"time": -1}) immediately before quantile.
    - parallelization is across space via lat/lon chunking.

    Output:
    - annual values on a 'T' dimension (one value per year, time stamp YYYY-01-01)
    - dims renamed to (T, Y, X) i.e. time->T, lat->Y, lon->X
    """

    def __init__(
        self,
        base_period: slice,
        tx_percentile: float = 90,
        tn_percentile: Optional[float] = None,
        min_consecutive_days: int = 6,
        season: Optional[List[int]] = None,
        require_both_tx_tn: bool = False,
    ):
        self.base_period = base_period
        self.tx_percentile = float(tx_percentile)
        self.tn_percentile = None if tn_percentile is None else float(tn_percentile)
        self.min_consecutive_days = int(min_consecutive_days)
        self.season = season
        self.require_both_tx_tn = bool(require_both_tx_tn)

        self._validate_inputs()

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    def _validate_inputs(self) -> None:
        if self.min_consecutive_days < 1:
            raise ValueError("min_consecutive_days must be >= 1")
        if not (0.0 < self.tx_percentile < 100.0):
            raise ValueError("tx_percentile must be between 0 and 100")
        if self.require_both_tx_tn and self.tn_percentile is None:
            raise ValueError("tn_percentile must be provided when require_both_tx_tn=True")
        if self.tn_percentile is not None and not (0.0 < self.tn_percentile < 100.0):
            raise ValueError("tn_percentile must be between 0 and 100")

    # -------------------------------------------------------------------------
    # Extract / standardize
    # -------------------------------------------------------------------------
    @staticmethod
    def _extract_da(obj: Union[xr.DataArray, xr.Dataset], candidates: List[str]) -> xr.DataArray:
        if isinstance(obj, xr.DataArray):
            return obj
        if not isinstance(obj, xr.Dataset):
            raise TypeError(f"Expected xarray Dataset or DataArray, got {type(obj)}")

        for name in candidates:
            if name in obj.data_vars:
                return obj[name]
        return obj[list(obj.data_vars)[0]]

    @staticmethod
    def _standardize_dims(da: xr.DataArray) -> xr.DataArray:
        """
        Rename dims to ('time','lat','lon') from common variants.
        """
        dim_map = {}

        # time
        for c in ["time", "T", "date", "Date", "valid_time"]:
            if c in da.dims:
                dim_map[c] = "time"
                break

        # lat
        for c in ["lat", "latitude", "Y", "y"]:
            if c in da.dims:
                dim_map[c] = "lat"
                break

        # lon
        for c in ["lon", "longitude", "X", "x"]:
            if c in da.dims:
                dim_map[c] = "lon"
                break

        if dim_map:
            da = da.rename(dim_map)

        if "time" not in da.dims:
            raise ValueError(f"Missing time dimension. Found dims={list(da.dims)}")
        if "lat" not in da.dims or "lon" not in da.dims:
            raise ValueError(f"Missing lat/lon dimensions. Found dims={list(da.dims)}")

        return da

    @staticmethod
    def _is_dask(da: xr.DataArray) -> bool:
        return hasattr(da.data, "chunks")

    @staticmethod
    def _choose_spatial_chunks(ny: int, nx: int, nb_cores: int) -> Tuple[int, int]:
        """
        Simple heuristic for chunking across space: sqrt(nb_cores) x sqrt(nb_cores)
        """
        nsplit = max(1, int(np.ceil(np.sqrt(max(1, nb_cores)))))
        cy = max(1, int(np.ceil(ny / nsplit)))
        cx = max(1, int(np.ceil(nx / nsplit)))
        return cy, cx

    # -------------------------------------------------------------------------
    # Threshold computation (day-of-year)
    # -------------------------------------------------------------------------
    def _compute_thresholds_dayofyear(
        self,
        da: xr.DataArray,
        percentile: float,
        parallel: bool,
        nb_cores: int,
    ) -> xr.DataArray:
        """
        Returns thresholds(dayofyear, lat, lon) for da(time, lat, lon).
        Compatible across xarray versions (no dask_gufunc_kwargs).
        """

        # chunk across space (optional)
        if parallel and self._is_dask(da):
            ny, nx = int(da.sizes["lat"]), int(da.sizes["lon"])
            cy, cx = self._choose_spatial_chunks(ny, nx, nb_cores)
            da = da.chunk({"lat": cy, "lon": cx})

        # critical: time must be single chunk for quantile core dim
        if parallel and self._is_dask(da):
            da = da.chunk({"time": -1})

        # 5-day centered window
        w = da.rolling(time=5, center=True, min_periods=1).construct("window")
        if parallel and self._is_dask(w):
            # window is tiny; keep single-chunk
            w = w.chunk({"window": -1})

        q = float(percentile) / 100.0

        thr = w.groupby("time.dayofyear").quantile(
            q,
            dim=["time", "window"],
            method="linear",
            skipna=True,
        )

        # clean possible quantile dim/coord
        if "quantile" in thr.dims:
            thr = thr.isel(quantile=0, drop=True)
        if "quantile" in thr.coords:
            thr = thr.drop_vars("quantile")

        return thr

    # -------------------------------------------------------------------------
    # 1D run logic (correct ETCCDI spell counting)
    # -------------------------------------------------------------------------
    @staticmethod
    def _spell_stats_1d(arr: np.ndarray, minlen: int) -> Tuple[int, int, int]:
        """
        For a 1D hot-indicator array (float with {0,1} and possibly NaN),
        return (wsdi_days, hwf_events, hwdi_maxlen).
        """
        if arr.size == 0:
            return 0, 0, 0

        valid = np.isfinite(arr)
        hot = (arr == 1) & valid
        if not np.any(hot):
            return 0, 0, 0

        a = hot.astype(np.int8)
        padded = np.concatenate(([0], a, [0]))
        d = np.diff(padded)

        starts = np.where(d == 1)[0]
        ends   = np.where(d == -1)[0]
        lens = ends - starts

        good = lens >= minlen
        if not np.any(good):
            return 0, 0, 0

        good_lens = lens[good]
        wsdi = int(good_lens.sum())
        hwf = int(good_lens.size)
        hwdi = int(good_lens.max())
        return wsdi, hwf, hwdi

    # -------------------------------------------------------------------------
    # Public API: compute
    # -------------------------------------------------------------------------
    def compute(
        self,
        ds_tx: Union[xr.Dataset, xr.DataArray],
        ds_tn: Optional[Union[xr.Dataset, xr.DataArray]] = None,
        metric: str = "WSDI",
        parallel: bool = True,
        nb_cores: int = 4,
        compute: bool = True,
    ) -> xr.DataArray:
        """
        Compute metric on gridded data.

        Parameters
        ----------
        ds_tx : Dataset/DataArray
            Daily max temperature (tasmax/TMAX).
        ds_tn : Dataset/DataArray, optional
            Daily min temperature (tasmin/TMIN) for compound events.
        metric : str
            "WSDI", "HWF", "HWDI" (also accepts "HWN" alias for HWF).
        parallel : bool
            Use dask parallelization across space.
        nb_cores : int
            Used to choose spatial chunk sizes.
        compute : bool
            If True, triggers compute() at the end.

        Returns
        -------
        xr.DataArray with dims (T,Y,X)
        """
        m = metric.upper()
        if m == "HWN":
            m = "HWF"
        if m not in {"WSDI", "HWF", "HWDI"}:
            raise ValueError("metric must be one of: WSDI, HWF, HWDI (or HWN as alias of HWF)")

        # --- TX
        da_tx = self._extract_da(ds_tx, ["TMAX", "tasmax", "TX"])
        da_tx = self._standardize_dims(da_tx)

        # seasonal filter
        if self.season:
            da_tx = da_tx.where(da_tx["time"].dt.month.isin(self.season), drop=True)

        # spatial chunking early (time chunk enforced where needed)
        if parallel and self._is_dask(da_tx):
            ny, nx = int(da_tx.sizes["lat"]), int(da_tx.sizes["lon"])
            cy, cx = self._choose_spatial_chunks(ny, nx, nb_cores)
            da_tx = da_tx.chunk({"lat": cy, "lon": cx})

        # base period
        da_tx_base = da_tx.sel(time=self.base_period)

        # thresholds(dayofyear,lat,lon)
        tx_thr_doy = self._compute_thresholds_dayofyear(
            da_tx_base, self.tx_percentile, parallel=parallel, nb_cores=nb_cores
        )

        # map thresholds to each time
        doy = da_tx["time"].dt.dayofyear
        doy_fixed = xr.where(doy == 366, 365, doy)

        tx_thr_full = tx_thr_doy.sel(dayofyear=doy_fixed)
        tx_thr_full = tx_thr_full.drop_vars("dayofyear", errors="ignore")
        tx_thr_full = tx_thr_full.assign_coords(time=da_tx["time"])

        is_hot_tx = (da_tx > tx_thr_full).astype(np.float32)
        is_hot_tx = is_hot_tx.where(da_tx.notnull())

        # --- optional TN compound
        if (ds_tn is not None) and self.require_both_tx_tn:
            da_tn = self._extract_da(ds_tn, ["TMIN", "tasmin", "TN"])
            da_tn = self._standardize_dims(da_tn)

            if self.season:
                da_tn = da_tn.where(da_tn["time"].dt.month.isin(self.season), drop=True)

            if parallel and self._is_dask(da_tn):
                # align spatial chunking with TX if possible
                if self._is_dask(da_tx):
                    # use first chunk sizes
                    cy = da_tx.chunksizes["lat"][0]
                    cx = da_tx.chunksizes["lon"][0]
                    da_tn = da_tn.chunk({"lat": cy, "lon": cx})

            da_tn_base = da_tn.sel(time=self.base_period)

            tn_thr_doy = self._compute_thresholds_dayofyear(
                da_tn_base, self.tn_percentile, parallel=parallel, nb_cores=nb_cores
            )

            doy_tn = da_tn["time"].dt.dayofyear
            doy_tn_fixed = xr.where(doy_tn == 366, 365, doy_tn)

            tn_thr_full = tn_thr_doy.sel(dayofyear=doy_tn_fixed)
            tn_thr_full = tn_thr_full.drop_vars("dayofyear", errors="ignore")
            tn_thr_full = tn_thr_full.assign_coords(time=da_tn["time"])

            is_hot_tn = (da_tn > tn_thr_full).astype(np.float32)
            is_hot_tn = is_hot_tn.where(da_tn.notnull())

            is_hot = ((is_hot_tx == 1) & (is_hot_tn == 1)).astype(np.float32)
            is_hot = is_hot.where(is_hot_tx.notnull() & is_hot_tn.notnull())
        else:
            is_hot = (is_hot_tx == 1).astype(np.float32)
            is_hot = is_hot.where(is_hot_tx.notnull())

        # for apply_ufunc core dim safety: time single chunk
        if parallel and self._is_dask(is_hot):
            is_hot = is_hot.chunk({"time": -1})

        years = np.unique(is_hot["time"].dt.year.values)

        annual_list = []
        for yr in years:
            hot_yr = is_hot.sel(time=is_hot["time"].dt.year == yr)

            def _metric_1d(arr: np.ndarray) -> np.int32:
                wsdi, hwf, hwdi = WAS_HeatWaveIndices._spell_stats_1d(arr, self.min_consecutive_days)
                if m == "WSDI":
                    return np.int32(wsdi)
                if m == "HWF":
                    return np.int32(hwf)
                return np.int32(hwdi)

            out_yr = xr.apply_ufunc(
                _metric_1d,
                hot_yr,
                input_core_dims=[["time"]],
                output_core_dims=[[]],
                vectorize=True,
                dask="parallelized" if (parallel and self._is_dask(hot_yr)) else "allowed",
                output_dtypes=[np.int32],
            )

            out_yr = out_yr.expand_dims(time=[pd.Timestamp(f"{int(yr)}-01-01")])
            annual_list.append(out_yr)

        out = xr.concat(annual_list, dim="time")
        out.name = m
        out.attrs.update(self._get_metadata(m))

        # WAS naming convention
        out = out.rename({"time": "T", "lat": "Y", "lon": "X"})

        return out.compute() if compute else out

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------
    def _get_metadata(self, metric: str) -> Dict:
        units = "days" if metric in {"WSDI", "HWDI"} else "count"
        base = f"{self.base_period.start}-{self.base_period.stop}"
        return {
            "long_name": f"{metric} Heat Wave Index",
            "units": units,
            "base_period": base,
            "tx_percentile": self.tx_percentile,
            "tn_percentile": self.tn_percentile,
            "min_consecutive_days": self.min_consecutive_days,
            "season": self.season if self.season else "all months",
            "require_both_tx_tn": self.require_both_tx_tn,
            "reference": "ETCCDI Climate Change Indices",
            "definition": self._definition(metric),
        }

    def _definition(self, metric: str) -> str:
        cond = f"TX > {self.tx_percentile}th percentile"
        if self.require_both_tx_tn:
            cond = f"TX > {self.tx_percentile}th AND TN > {self.tn_percentile}th percentile"
        k = self.min_consecutive_days
        if metric == "WSDI":
            return f"Annual total number of days in spells of at least {k} consecutive days with {cond}."
        if metric == "HWF":
            return f"Annual number of events (spells) of at least {k} consecutive days with {cond}."
        if metric == "HWDI":
            return f"Annual maximum duration (days) among spells of at least {k} consecutive days with {cond}."
        return "Heat wave index."


# =============================================================================
# Factory
# =============================================================================

class ETCCDIHeatWaveIndices:
    """Factory helpers for standard configurations."""

    @staticmethod
    def wsdi(
        base_period: slice,
        tx_percentile: float = 90,
        min_consecutive_days: int = 6,
        season: Optional[List[int]] = None,
    ) -> WAS_HeatWaveIndices:
        return WAS_HeatWaveIndices(
            base_period=base_period,
            tx_percentile=tx_percentile,
            min_consecutive_days=min_consecutive_days,
            season=season,
        )

    @staticmethod
    def heat_wave_frequency(
        base_period: slice,
        tx_percentile: float = 90,
        min_consecutive_days: int = 3,
        season: Optional[List[int]] = None,
    ) -> WAS_HeatWaveIndices:
        return WAS_HeatWaveIndices(
            base_period=base_period,
            tx_percentile=tx_percentile,
            min_consecutive_days=min_consecutive_days,
            season=season,
        )

    @staticmethod
    def heat_wave_duration_index(
        base_period: slice,
        tx_percentile: float = 90,
        min_consecutive_days: int = 3,
        season: Optional[List[int]] = None,
    ) -> WAS_HeatWaveIndices:
        return WAS_HeatWaveIndices(
            base_period=base_period,
            tx_percentile=tx_percentile,
            min_consecutive_days=min_consecutive_days,
            season=season,
        )

    @staticmethod
    def compound_heat_wave(
        base_period: slice,
        tx_percentile: float = 90,
        tn_percentile: float = 90,
        min_consecutive_days: int = 3,
        season: Optional[List[int]] = None,
    ) -> WAS_HeatWaveIndices:
        return WAS_HeatWaveIndices(
            base_period=base_period,
            tx_percentile=tx_percentile,
            tn_percentile=tn_percentile,
            min_consecutive_days=min_consecutive_days,
            season=season,
            require_both_tx_tn=True,
        )
