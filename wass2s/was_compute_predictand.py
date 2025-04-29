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

class WAS_compute_onset:
    """
    A class that encapsulates methods for transforming precipitation data
    from different formats (CPT, CDT) and computing onset dates based on
    rainfall criteria.
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
            (annual_rainfall < 100) & (annual_rainfall >= 0),
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

        mask_char = self.rainf_zone(daily_data)
        mm = mask_char.plot()
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
    A class for computing the longest dry spell length 
    after the onset of a rainy season, based on user-defined criteria.
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
            elif 100 >= row["MEAN_ANNUAL_RAINFALL"] > 0:
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
            (annual_rainfall < 100) & (annual_rainfall >= 0),
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

            # Compute the longest dry spell within `nbjour` days after the onset
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
    A class to compute cessation dates based on soil moisture balance for different
    regions and criteria, leveraging parallel computation for efficiency.
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
            elif 100 >= row["MEAN_ANNUAL_RAINFALL"] > 0:
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
            (annual_rainfall < 100) & (annual_rainfall >= 0),
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
    A class for computing the longest dry spell length 
    after the onset of a rainy season, based on user-defined criteria.
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
            elif 100 >= row["MEAN_ANNUAL_RAINFALL"] > 0:    
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
            (annual_rainfall < 100) & (annual_rainfall >= 0),
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
    A class to compute the number of dry spells within a specified period
    (onset to cessation) for each pixel or station in a daily rainfall dataset.
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
            elif 100 >= row["MEAN_ANNUAL_RAINFALL"] > 0:
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
    A class to compute the number of wet spells within a specified period
    (onset to cessation) for each pixel or station in a daily rainfall dataset.
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
            elif 100 >= row["MEAN_ANNUAL_RAINFALL"] > 0:
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
    A class to compute the number of rainy days between onset and cessation dates
    for each pixel or station in a daily rainfall dataset.
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
            Result of `WAS_compute_onset.compute_insitu(...)` for onset (CPT format).
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

        
class WAS_r95_99p:
    """
    A class to compute the R95p and R99p climate indices using either:
    - Dask-enabled xarray for large raster/time-series
    - An "insitu" method for station-based (CDT) data.
    """

    def __init__(self, base_period: slice, season: list = None):
        """
        Initialize the R95p/R99p computation class.

        Parameters
        ----------
        base_period : slice
            Base period for computing the percentiles, e.g., slice("1961-01-01", "1990-12-31").
            This should be something like slice("YYYY-MM-DD", "YYYY-MM-DD") or 
            slice("YYYY", "YYYY") if you only have year-level bounds.
        season : list, optional
            List of months to include in the analysis (e.g., [6, 7, 8] for JJA).
        """
        self.base_period = base_period
        self.season = season

    @staticmethod
    def transform_cdt(df):
        """
        Transform a DataFrame in CDT format into a standardized long DataFrame.

        CDT format assumptions:
         - Row 0 = LON
         - Row 1 = LAT
         - Row 2 = ELEV
         - Rows 3+ = daily data with 'ID' column holding dates in YYYYMMDD format.

        Returns a DataFrame with columns:
            [DATE, STATION, VALUE, LON, LAT, ELEV]
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

        # Fill missing rainfall values with -99.0
        final_df["VALUE"] = final_df["VALUE"].fillna(-99.0)

        return final_df

    def _compute_percentile_index_insitu(self, df_full, percentile=95) -> pd.DataFrame:
        """
        Internal method that computes the 'Rxp' index (R95p or R99p) for insitu data.

        Parameters
        ----------
        df_full : pd.DataFrame
            Must have columns [DATE, STATION, VALUE, LAT, LON, ELEV].
        percentile : float
            Percentile to compute (e.g., 95 for R95p or 99 for R99p).

        Returns
        -------
        df_result : pd.DataFrame
            DataFrame with [year, station, lat, lon, rX_p_value],
            where rX_p_value is total precipitation above the threshold for each year.
        """

        # 1) Possibly filter by season
        if self.season:
            # Keep only rows whose month is in self.season
            df_full = df_full[df_full["DATE"].dt.month.isin(self.season)]

        # 2) Separate out the base period to compute thresholds
        #    self.base_period is typically something like slice("1961-01-01", "1990-12-31")
        #    We'll interpret it so we can do: df_base = df_full[(df_full["DATE"] >= start) & (df_full["DATE"] <= end)]
        start_str, end_str = self.base_period.start, self.base_period.stop
        start_date = pd.to_datetime(start_str)
        end_date = pd.to_datetime(end_str)
        df_base = df_full[(df_full["DATE"] >= start_date) & (df_full["DATE"] <= end_date)]

        # 3) Compute day-of-year in both data sets
        df_full["DOY"] = df_full["DATE"].dt.dayofyear
        df_base["DOY"] = df_base["DATE"].dt.dayofyear

        # 4) For each station and day-of-year in the base period, compute the percentile threshold
        #    We'll group by (STATION, DOY) and compute np.nanpercentile
        thresholds = (
            df_base[df_base["VALUE"] >= 0]  # ignore negative placeholder
            .groupby(["STATION", "DOY"])["VALUE"]
            .apply(lambda x: np.nanpercentile(x, percentile))
            .reset_index()
            .rename(columns={"VALUE": "THRESHOLD"})
        )

        # Merge thresholds back into df_full on (STATION, DOY)
        df_merged = pd.merge(
            df_full, 
            thresholds, 
            on=["STATION", "DOY"], 
            how="left"
        )

        # 5) Identify days exceeding that threshold and sum them up (precip total) by station & year
        #    We'll also ignore negative precipitation (i.e. -99.0)
        df_merged["EXCEEDS"] = np.where(
            (df_merged["VALUE"] > df_merged["THRESHOLD"]) & (df_merged["VALUE"] >= 0),
            df_merged["VALUE"],
            0
        )
        df_merged["year"] = df_merged["DATE"].dt.year

        # 6) Sum precipitation on those "extreme" days for each station-year
        #    Then keep lat/lon from the first occurrence (assuming station lat/lon is fixed)
        df_result = (
            df_merged
            .groupby(["station", "year", "LAT", "LON"], as_index=False)["EXCEEDS"]
            .sum()
            .rename(columns={"EXCEEDS": f"R{percentile}p"})
        )

        return df_result

    def compute_insitu_r95p(self, df_cdt: pd.DataFrame) -> pd.DataFrame:
        """
        Compute R95p index (total precipitation on days above the daily 95th percentile)
        for station-based data in CDT format.

        Parameters
        ----------
        df_cdt : pd.DataFrame
            CDT-format DataFrame (rows 0..2 = LON/LAT/ELEV, row 3+ = daily data).

        Returns
        -------
        df_final : pd.DataFrame
            A DataFrame in CPT format with the R95p values pivoted by station vs. year.
        """
        # 1) Transform CDT to standard DataFrame
        df_full = self.transform_cdt(df_cdt)

        # 2) Compute R95p
        df_r95 = self._compute_percentile_index_insitu(df_full, percentile=95)

        # 3) Pivot back to CPT format
        #    a) Station in columns, year in rows
        df_pivot = df_r95.pivot(index="year", columns="station", values="R95p").reset_index()
        df_pivot.rename(columns={"year": "STATION"}, inplace=True)

        # b) Build LAT/LON rows, using first occurrence for each station
        station_metadata = (
            df_r95.groupby("station")[["LAT", "LON"]]
            .first()
            .reindex(df_pivot.columns[1:])  # ensure same station order as pivot
        )
        lat_row = ["LAT"] + station_metadata["LAT"].tolist()
        lon_row = ["LON"] + station_metadata["LON"].tolist()

        # c) Insert them above the pivoted DataFrame
        lat_df = pd.DataFrame([lat_row], columns=df_pivot.columns)
        lon_df = pd.DataFrame([lon_row], columns=df_pivot.columns)

        df_final = pd.concat([lat_df, lon_df, df_pivot], ignore_index=True)

        return df_final

    def compute_insitu_r99p(self, df_cdt: pd.DataFrame) -> pd.DataFrame:
        """
        Compute R99p index (total precipitation on days above the daily 99th percentile)
        for station-based data in CDT format.

        Parameters
        ----------
        df_cdt : pd.DataFrame
            CDT-format DataFrame.

        Returns
        -------
        df_final : pd.DataFrame (CPT format)
        """
        # 1) Transform
        df_full = self.transform_cdt(df_cdt)

        # 2) Compute R99p
        df_r99 = self._compute_percentile_index_insitu(df_full, percentile=99)

        # 3) Pivot to CPT format
        df_pivot = df_r99.pivot(index="year", columns="station", values="R99p").reset_index()
        df_pivot.rename(columns={"year": "STATION"}, inplace=True)

        station_metadata = (
            df_r99.groupby("station")[["LAT", "LON"]]
            .first()
            .reindex(df_pivot.columns[1:])
        )
        lat_row = ["LAT"] + station_metadata["LAT"].tolist()
        lon_row = ["LON"] + station_metadata["LON"].tolist()

        lat_df = pd.DataFrame([lat_row], columns=df_pivot.columns)
        lon_df = pd.DataFrame([lon_row], columns=df_pivot.columns)

        df_final = pd.concat([lat_df, lon_df, df_pivot], ignore_index=True)
        return df_final

    #
    # The existing xarray-based methods for large raster data remain the same:
    #
    def compute_r95p(self, pr: "xr.DataArray") -> "xr.DataArray":
        """
        Existing method for xarray-based data (unchanged).
        """
        return self._compute_percentile_index(pr, percentile=95)

    def compute_r99p(self, pr: "xr.DataArray") -> "xr.DataArray":
        """
        Existing method for xarray-based data (unchanged).
        """
        return self._compute_percentile_index(pr, percentile=99)

    def _compute_percentile_index(self, pr: "xr.DataArray", percentile: float) -> "xr.DataArray":
        """
        Existing private method for xarray-based data (unchanged).
        """
        # Subset to base period
        pr_base = pr.sel(time=self.base_period)

        # Apply seasonal filtering if specified
        if self.season:
            pr = pr.where(pr.time.dt.month.isin(self.season), drop=True)
            pr_base = pr_base.where(pr_base.time.dt.month.isin(self.season), drop=True)

        # Compute the percentile for each day-of-year in the base period
        pr_thresh = pr_base.groupby("time.dayofyear").reduce(
            np.nanpercentile, q=percentile, dim="time"
        )

        # Broadcast threshold to full time dimension
        doy = pr.time.dt.dayofyear
        threshold_broadcast = pr_thresh.sel(dayofyear=doy.values)

        # Identify very wet days exceeding the threshold
        extreme_days = pr.where(pr > threshold_broadcast)

        # Sum precipitation on very wet days for each year
        result = extreme_days.resample(time="Y").sum(dim="time", skipna=True)

        return result
        
        
        

#### look this part again -----
class WAS_compute_HWSDI:
    """
    A class to compute the Heat Wave Severity Duration Index (HWSDI),
    including calculating TXin90 (90th percentile of daily max temperature) 
    and annual counts of heatwave days with at least 6 consecutive hot days.
    """

    @staticmethod
    def calculate_TXin90(temperature_data, base_period_start='1961', base_period_end='1990'):
        """
        Calculate the daily 90th percentile temperature (TXin90) centered on a 5-day window
        for each calendar day based on the base period.

        Parameters
        ----------
        temperature_data : xarray.DataArray
            Daily maximum temperature with time dimension.
        base_period_start : str, optional
            Start year of the base period (default is '1961').
        base_period_end : str, optional
            End year of the base period (default is '1990').

        Returns
        -------
        xarray.DataArray
            TXin90 for each day of the year.
        """
        # Filter the data for the base period
        base_period = temperature_data.sel(T=slice(base_period_start, base_period_end))

        # Group by day of the year (DOY) and calculate the 90th percentile over a centered 5-day window
        TXin90 = base_period.rolling(T=5, center=True).construct("window_dim").groupby("T.dayofyear").reduce(
            np.nanpercentile, q=90, dim="window_dim"
        )

        return TXin90

    @staticmethod
    def _count_consecutive_days(data, min_days=6):
        """
        Count sequences of at least `min_days` consecutive True values in a boolean array.

        Parameters
        ----------
        data : np.ndarray
            Boolean array.
        min_days : int
            Minimum number of consecutive True values to count as a sequence.

        Returns
        -------
        int
            Count of sequences with at least `min_days` consecutive True values.
        """
        count = 0
        current_streak = 0

        for value in data:
            if value:
                current_streak += 1
                if current_streak == min_days:
                    count += 1
            else:
                current_streak = 0

        return count

    def count_hot_days(self, temperature_data, TXin90):
        """
        Count the number of days per year with at least 6 consecutive days
        where daily maximum temperature is above the 90th percentile.

        Parameters
        ----------
        temperature_data : xarray.DataArray
            Daily maximum temperature with time dimension.
        TXin90 : xarray.DataArray
            90th percentile temperature for each day of the year.

        Returns
        -------
        xarray.DataArray
            Annual count of hot days.
        """
        # Ensure TXin90 covers each day of the year by broadcasting
        TXin90_full = TXin90.sel(dayofyear=temperature_data.time.dt.dayofyear)

        # Find days where daily temperature exceeds the 90th percentile
        hot_days = temperature_data > TXin90_full

        # Convert to integer (1 for hot day, 0 otherwise) and group by year
        hot_days_per_year = hot_days.astype(int).groupby("time.year")

        # Count sequences of at least 6 consecutive hot days within each year
        annual_hot_days_count = xr.DataArray(
            np.array([
                self._count_consecutive_days(year_data.values, min_days=6) 
                for year_data in hot_days_per_year
            ]),
            coords={"year": list(hot_days_per_year.groups.keys())},
            dims="year"
        )

        return annual_hot_days_count

    def compute(self, temperature_data, base_period_start='1961', base_period_end='1990', nb_cores=4):
        """
        Compute the Heat Wave Severity Duration Index (HWSDI) for each pixel
        in a given daily temperature DataArray.

        Parameters
        ----------
        temperature_data : xarray.DataArray
            Daily maximum temperature data, coords = (T, Y, X).
        base_period_start : str, optional
            Start year of the base period for TXin90 calculation (default is '1961').
        base_period_end : str, optional
            End year of the base period for TXin90 calculation (default is '1990').
        nb_cores : int, optional
            Number of parallel processes to use (default is 4).
        
        Returns
        -------
        xarray.DataArray
            HWSDI computed for each pixel.
        """
        # Rename 'T' dimension to 'time' so dayofyear and year grouping work as expected
        temperature_data = temperature_data.rename({'T': 'time'})

        # Compute TXin90
        TXin90 = self.calculate_TXin90(temperature_data, base_period_start, base_period_end)

        # Prepare chunk sizes
        chunksize_x = int(np.round(len(temperature_data.get_index("X")) / nb_cores))
        chunksize_y = int(np.round(len(temperature_data.get_index("Y")) / nb_cores))

        # Set up parallel processing
        client = Client(n_workers=nb_cores, threads_per_worker=1)

        # Apply function
        result = xr.apply_ufunc(
            self.count_hot_days,
            temperature_data.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            TXin90,
            input_core_dims=[('T',), ('dayofyear',)],
            vectorize=True,
            output_core_dims=[('year',)],
            dask='parallelized',
            output_dtypes=['float']
        )

        result_ = result.compute()
        client.close()

        return result_



class WAS_compute_HWSDI_monthly:
    """
    A class to compute the Heat Wave Severity Duration Index (HWSDI) **monthly**,
    calculating TXin90 (90th percentile of daily max temperature) and counting heatwave days
    for each month with at least 6 consecutive hot days.
    """

    @staticmethod
    def calculate_TXin90(temperature_data, base_period_start='1961', base_period_end='1990'):
        """
        Calculate the monthly 90th percentile temperature (TXin90) centered on a 5-day window
        for each calendar day based on the base period.

        Parameters
        ----------
        temperature_data : xarray.DataArray
            Daily maximum temperature with time dimension.
        base_period_start : str, optional
            Start year of the base period (default is '1961').
        base_period_end : str, optional
            End year of the base period (default is '1990').

        Returns
        -------
        xarray.DataArray
            TXin90 for each month of the year.
        """
        # Filter data for the base period
        base_period = temperature_data.sel(time=slice(base_period_start, base_period_end))

        # Compute the rolling 90th percentile temperature for each **month**
        TXin90 = base_period.rolling(time=5, center=True).construct("window_dim").groupby("time.month").reduce(
            np.nanpercentile, q=90, dim="window_dim"
        )

        return TXin90

    @staticmethod
    def _count_consecutive_days(data, min_days=6):
        """
        Count sequences of at least `min_days` consecutive True values in a boolean array.

        Parameters
        ----------
        data : np.ndarray
            Boolean array.
        min_days : int
            Minimum number of consecutive True values to count as a sequence.

        Returns
        -------
        int
            Count of sequences with at least `min_days` consecutive True values.
        """
        count = 0
        current_streak = 0

        for value in data:
            if value:
                current_streak += 1
                if current_streak == min_days:
                    count += 1
            else:
                current_streak = 0

        return count

    def count_hot_days(self, temperature_data, TXin90):
        """
        Count the number of days per month with at least 6 consecutive days
        where daily maximum temperature is above the 90th percentile.

        Parameters
        ----------
        temperature_data : xarray.DataArray
            Daily maximum temperature with time dimension.
        TXin90 : xarray.DataArray
            90th percentile temperature for each month.

        Returns
        -------
        xarray.DataArray
            Monthly count of hot days.
        """
        # Ensure TXin90 covers each month by broadcasting
        TXin90_full = TXin90.sel(month=temperature_data.time.dt.month)

        # Find days where daily temperature exceeds the 90th percentile
        hot_days = temperature_data > TXin90_full

        # Convert to integer (1 for hot day, 0 otherwise) and group by month
        hot_days_per_month = hot_days.astype(int).groupby("time.month")

        # Count sequences of at least 6 consecutive hot days within each month
        monthly_hot_days_count = xr.DataArray(
            np.array([
                self._count_consecutive_days(month_data.values, min_days=6) 
                for month_data in hot_days_per_month
            ]),
            coords={"month": list(hot_days_per_month.groups.keys())},
            dims="month"
        )

        return monthly_hot_days_count

    def compute(self, temperature_data, base_period_start='1961', base_period_end='1990', nb_cores=4):
        """
        Compute the Monthly Heat Wave Severity Duration Index (HWSDI)
        for each pixel in a given daily temperature DataArray.

        Parameters
        ----------
        temperature_data : xarray.DataArray
            Daily maximum temperature data, coords = (T, Y, X).
        base_period_start : str, optional
            Start year of the base period for TXin90 calculation (default is '1961').
        base_period_end : str, optional
            End year of the base period for TXin90 calculation (default is '1990').
        nb_cores : int, optional
            Number of parallel processes to use (default is 4).
        
        Returns
        -------
        xarray.DataArray
            HWSDI computed for each pixel per month.
        """
        # Rename 'T' dimension to 'time' so month grouping works as expected
        temperature_data = temperature_data.rename({'T': 'time'})

        # Compute TXin90
        TXin90 = self.calculate_TXin90(temperature_data, base_period_start, base_period_end)

        # Prepare chunk sizes for parallel processing
        chunksize_x = int(np.round(len(temperature_data.get_index("X")) / nb_cores))
        chunksize_y = int(np.round(len(temperature_data.get_index("Y")) / nb_cores))

        # Set up parallel processing
        client = Client(n_workers=nb_cores, threads_per_worker=1)

        # Apply function in parallel
        result = xr.apply_ufunc(
            self.count_hot_days,
            temperature_data.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            TXin90,
            input_core_dims=[('time',), ('month',)],
            vectorize=True,
            output_core_dims=[('month',)],
            dask='parallelized',
            output_dtypes=['float']
        )

        result_ = result.compute()
        client.close()

        return result_


class WAS_compute_HWSDI_Seasonal:
    """
    A class to compute the Heat Wave Severity Duration Index (HWSDI) for a given season.
    """

    @staticmethod
    def calculate_TXin90(temperature_data, base_period_start='1961', base_period_end='1990', season=[6, 7, 8]):
        """
        Calculate the daily 90th percentile temperature (TXin90) for each calendar day 
        based on the base period, but only considering the specified season.

        Parameters
        ----------
        temperature_data : xarray.DataArray
            Daily maximum temperature with time dimension.
        base_period_start : str, optional
            Start year of the base period (default is '1961').
        base_period_end : str, optional
            End year of the base period (default is '1990').
        season : list, optional
            List of months to include in the calculation (default is [6, 7, 8] for JJA).

        Returns
        -------
        xarray.DataArray
            TXin90 for each day of the selected season.
        """
        # Filter the data for the base period and only the selected season
        base_period = temperature_data.sel(time=slice(base_period_start, base_period_end))
        seasonal_data = base_period.where(base_period.time.dt.month.isin(season), drop=True)

        # Group by day of the year (DOY) and calculate the 90th percentile
        TXin90 = seasonal_data.rolling(time=5, center=True).construct("window_dim").groupby("time.dayofyear").reduce(
            np.nanpercentile, q=90, dim="window_dim"
        )

        return TXin90

    @staticmethod
    def _count_consecutive_days(data, min_days=6):
        """
        Count sequences of at least `min_days` consecutive True values in a boolean array.

        Parameters
        ----------
        data : np.ndarray
            Boolean array.
        min_days : int
            Minimum number of consecutive True values to count as a sequence.

        Returns
        -------
        int
            Count of sequences with at least `min_days` consecutive True values.
        """
        count = 0
        current_streak = 0

        for value in data:
            if value:
                current_streak += 1
                if current_streak == min_days:
                    count += 1
            else:
                current_streak = 0

        return count

    def count_hot_days(self, temperature_data, TXin90):
        """
        Count the number of days per season with at least 6 consecutive days
        where daily maximum temperature is above the 90th percentile.

        Parameters
        ----------
        temperature_data : xarray.DataArray
            Daily maximum temperature with time dimension.
        TXin90 : xarray.DataArray
            90th percentile temperature for each day of the year.

        Returns
        -------
        xarray.DataArray
            Seasonal count of hot days.
        """
        # Ensure TXin90 covers each day of the season by broadcasting
        TXin90_full = TXin90.sel(dayofyear=temperature_data.time.dt.dayofyear)

        # Find days where daily temperature exceeds the 90th percentile
        hot_days = temperature_data > TXin90_full

        # Convert to integer (1 for hot day, 0 otherwise) and group by year
        hot_days_per_year = hot_days.astype(int).groupby("time.year")

        # Count sequences of at least 6 consecutive hot days within each season
        seasonal_hot_days_count = xr.DataArray(
            np.array([
                self._count_consecutive_days(year_data.values, min_days=6) 
                for year_data in hot_days_per_year
            ]),
            coords={"year": list(hot_days_per_year.groups.keys())},
            dims="year"
        )

        return seasonal_hot_days_count

    def compute(self, temperature_data, base_period_start='1961', base_period_end='1990', nb_cores=4, season=[6, 7, 8]):
        """
        Compute the HWSDI for each pixel in a given daily temperature DataArray for a specific season.

        Parameters
        ----------
        temperature_data : xarray.DataArray
            Daily maximum temperature data, coords = (T, Y, X).
        base_period_start : str, optional
            Start year of the base period for TXin90 calculation (default is '1961').
        base_period_end : str, optional
            End year of the base period for TXin90 calculation (default is '1990').
        nb_cores : int, optional
            Number of parallel processes to use (default is 4).
        season : list, optional
            List of months to include in the calculation (default is [6, 7, 8] for JJA).

        Returns
        -------
        xarray.DataArray
            HWSDI computed for each pixel for the given season.
        """
        # Rename 'T' dimension to 'time' so dayofyear and year grouping work as expected
        temperature_data = temperature_data.rename({'T': 'time'})

        # Filter data to only include selected season
        seasonal_temperature_data = temperature_data.where(temperature_data.time.dt.month.isin(season), drop=True)

        # Compute TXin90 based on the season
        TXin90 = self.calculate_TXin90(seasonal_temperature_data, base_period_start, base_period_end, season)

        # Prepare chunk sizes
        chunksize_x = int(np.round(len(temperature_data.get_index("X")) / nb_cores))
        chunksize_y = int(np.round(len(temperature_data.get_index("Y")) / nb_cores))

        # Set up parallel processing
        client = Client(n_workers=nb_cores, threads_per_worker=1)

        # Apply function
        result = xr.apply_ufunc(
            self.count_hot_days,
            seasonal_temperature_data.chunk({'Y': chunksize_y, 'X': chunksize_x}),
            TXin90,
            input_core_dims=[('time',), ('dayofyear',)],
            vectorize=True,
            output_core_dims=[('year',)],
            dask='parallelized',
            output_dtypes=['float']
        )

        result_ = result.compute()
        client.close()

        return result_