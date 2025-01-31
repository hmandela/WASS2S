from dask.distributed import Client
import pandas as pd
import xarray as xr
import pandas as pd
import xarray as xr
import numpy as np
import datetime


class WAS_compute_onset:
    """
    A class that encapsulates methods for transforming precipitation data
    from different formats (CPT, CDT) and computing onset dates based on
    rainfall criteria.
    """

    # Default class-level criteria dictionary
    default_criteria = {
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
    def transform_cpt(df, date_month_day="08-01", missing_value=None):
        """
        Transform a DataFrame with:
          - Row 0 = LAT
          - Row 1 = LON
          - Rows 2+ = numeric year data in wide format (stations in columns).

        Returns an xarray DataArray with coords = (T, Y, X), variable = 'Observation'.
        """
        # --- 1) Extract metadata ---
        metadata = (
            df
            .iloc[:2]
            .set_index("STATION")  # -> index = ["LAT", "LON"]
            .T
            .reset_index()         # station names in 'index'
        )
        metadata.columns = ["STATION", "LAT", "LON"]

        # Adjust duplicates in LAT / LON
        metadata["LAT"] = WAS_compute_onset.adjust_duplicates(metadata["LAT"])
        metadata["LON"] = WAS_compute_onset.adjust_duplicates(metadata["LON"])

        # --- 2) Extract data part ---
        data_part = df.iloc[2:].copy()
        data_part = data_part.rename(columns={"STATION": "YEAR"})
        data_part["YEAR"] = data_part["YEAR"].astype(int)

        # --- 3) Convert wide -> long ---
        long_data = data_part.melt(
            id_vars="YEAR",
            var_name="STATION",
            value_name="VALUE"
        )

        # --- 4) YEAR -> date (e.g. YYYY-08-01) ---
        long_data["DATE"] = pd.to_datetime(
            long_data["YEAR"].astype(str) + f"-{date_month_day}",
            format="%Y-%m-%d"
        )

        # --- 5) Merge with metadata ---
        final_df = pd.merge(long_data, metadata, on="STATION", how="left")

        # --- 6) Convert to xarray ---
        rainfall_data_array = (
            final_df[["DATE", "LAT", "LON", "VALUE"]]
            .set_index(["DATE", "LAT", "LON"])
            .to_xarray()
            .rename({"VALUE": "Observation", "LAT": "Y", "LON": "X", "DATE": "T"})
        )
        if missing_value is not None:
            rainfall_data_array = rainfall_data_array.where(rainfall_data_array != missing_value)

        return rainfall_data_array

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

        # --- 3) Merge with metadata ---
        final_df = pd.merge(data_long, metadata, on="STATION", how="left")

        # Ensure 'DATE' is a proper datetime (assuming format "YYYYmmdd")
        final_df["DATE"] = pd.to_datetime(final_df["DATE"], format="%Y%m%d", errors="coerce")

        # --- 4) Convert to xarray, rename coords ---
        rainfall_data_array = (
            final_df[["DATE", "LAT", "LON", "VALUE"]]
            .set_index(["DATE", "LAT", "LON"])
            .to_xarray()
            .rename({"VALUE": "Observation", "LAT": "Y", "LON": "X", "DATE": "T"})
        )

        return rainfall_data_array

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
                    deb_saison = irch_fin
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

    
    def compute_onset(self, daily_data, extent, nb_cores):
        """
        Compute onset dates for each pixel in a given daily rainfall DataArray
        using different criteria based on isohyet zones.

        Parameters
        ----------
        daily_data : xarray.DataArray
            Daily rainfall data, coords = (T, Y, X).
        extent : list or tuple
            [Ymin, Xmin, Ymax, Xmax] defining the geographic slice.
        nb_cores : int
            Number of parallel processes to use.

        Returns
        -------
        xarray.DataArray
            Array with onset dates computed per pixel.
        """
        # Load zone file & slice it
        mask_char = xr.open_dataset('./utilities/Isohyet_zones.nc')
        mask_char = mask_char.sel(X=slice(extent[1], extent[3]),
                                  Y=slice(extent[0], extent[2]))
        mask_char = mask_char.isel(Y=slice(None, None, -1)).to_array().drop_vars('variable').squeeze()

        daily_data = daily_data.sel(
            X=mask_char.coords['X'],
            Y=mask_char.coords['Y'])
        
        # Get unique zone IDs
        unique_zone = np.unique(mask_char.to_numpy())
        unique_zone = unique_zone[~np.isnan(unique_zone)]

        # Compute year range and partial T dimension (start_search)
        years = np.unique(daily_data['T'].dt.year.to_numpy())

        # Choose a date to store results
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

        return store_onset.to_array().drop_vars('variable').squeeze('variable')

class WAS_compute_onset_dry_spell:
    """
    A class for computing the longest dry spell length 
    after the onset of a rainy season, based on user-defined criteria.
    """

    # Default class-level criteria dictionary
    default_criteria = {
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
    def transform_cpt(df, date_month_day="08-01", missing_value=None):
        """
        Transform a DataFrame with:
          - Row 0 = LAT
          - Row 1 = LON
          - Rows 2+ = numeric year data in wide format (stations in columns).

        Returns an xarray DataArray with coords = (T, Y, X), variable = 'Observation'.
        """
        # --- 1) Extract metadata ---
        metadata = (
            df
            .iloc[:2]
            .set_index("STATION")  # -> index = ["LAT", "LON"]
            .T
            .reset_index()         # station names in 'index'
        )
        metadata.columns = ["STATION", "LAT", "LON"]

        # Adjust duplicates in LAT / LON
        metadata["LAT"] = WAS_compute_dry_spell.adjust_duplicates(metadata["LAT"])
        metadata["LON"] = WAS_compute_dry_spell.adjust_duplicates(metadata["LON"])

        # --- 2) Extract data part ---
        data_part = df.iloc[2:].copy()
        data_part = data_part.rename(columns={"STATION": "YEAR"})
        data_part["YEAR"] = data_part["YEAR"].astype(int)

        # --- 3) Convert wide -> long ---
        long_data = data_part.melt(
            id_vars="YEAR",
            var_name="STATION",
            value_name="VALUE"
        )

        # --- 4) YEAR -> date (e.g. YYYY-08-01) ---
        long_data["DATE"] = pd.to_datetime(
            long_data["YEAR"].astype(str) + f"-{date_month_day}",
            format="%Y-%m-%d"
        )

        # --- 5) Merge with metadata ---
        final_df = pd.merge(long_data, metadata, on="STATION", how="left")

        # --- 6) Convert to xarray ---
        rainfall_data_array = (
            final_df[["DATE", "LAT", "LON", "VALUE"]]
            .set_index(["DATE", "LAT", "LON"])
            .to_xarray()
            .rename({"VALUE": "Observation", "LAT": "Y", "LON": "X", "DATE": "T"})
        )
        if missing_value is not None:
            rainfall_data_array = rainfall_data_array.where(rainfall_data_array != missing_value)

        return rainfall_data_array

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
        metadata["LON"] = WAS_compute_dry_spell.adjust_duplicates(metadata["LON"])
        metadata["LAT"] = WAS_compute_dry_spell.adjust_duplicates(metadata["LAT"])
        metadata["ELEV"] = WAS_compute_dry_spell.adjust_duplicates(metadata["ELEV"])

        # --- 2) Extract actual data, rename ID -> DATE ---
        data_part = df.iloc[3:].rename(columns={"ID": "DATE"})

        # Melt to long form
        data_long = data_part.melt(id_vars=["DATE"], var_name="STATION", value_name="VALUE")

        # --- 3) Merge with metadata ---
        final_df = pd.merge(data_long, metadata, on="STATION", how="left")

        # Ensure 'DATE' is a proper datetime (assuming format "YYYYmmdd")
        final_df["DATE"] = pd.to_datetime(final_df["DATE"], format="%Y%m%d", errors="coerce")

        # --- 4) Convert to xarray, rename coords ---
        rainfall_data_array = (
            final_df[["DATE", "LAT", "LON", "VALUE"]]
            .set_index(["DATE", "LAT", "LON"])
            .to_xarray()
            .rename({"VALUE": "Observation", "LAT": "Y", "LON": "X", "DATE": "T"})
        )

        return rainfall_data_array

    def dry_spell_onset_function(self, x, idebut, cumul, nbsec, jour_pluvieux, irch_fin, nbjour):
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
                    deb_saison = irch_fin
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

    def compute_dry_spell(self, daily_data, extent, nb_cores):
        """
        Compute the longest dry spell length after the onset for each pixel in a
        given daily rainfall DataArray, using different criteria based on isohyet zones.

        Parameters
        ----------
        daily_data : xarray.DataArray
            Daily rainfall data, coords = (T, Y, X).
        extent : list or tuple
            [Ymin, Xmin, Ymax, Xmax] defining the geographic slice.
        nb_cores : int
            Number of parallel processes to use.
        nbjour : int
            The number of days after onset to search for the longest dry spell.

        Returns
        -------
        xarray.DataArray
            Array with the longest dry spell length per pixel.
        """
        # Load zone file & slice it to the area of interest
        mask_char = xr.open_dataset('./utilities/Isohyet_zones.nc')
        mask_char = mask_char.sel(X=slice(extent[1], extent[3]),
                                  Y=slice(extent[0], extent[2]))
        
        # Flip Y if needed
        mask_char = mask_char.isel(Y=slice(None, None, -1)).to_array().drop_vars('variable').squeeze()
        
        daily_data = daily_data.sel(
            X=mask_char.coords['X'],
            Y=mask_char.coords['Y'])

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

        return store_dry_spell.to_array().drop_vars('variable').squeeze('variable')

class WAS_compute_cessation:
    """
    A class to compute cessation dates based on soil moisture balance for different
    regions and criteria, leveraging parallel computation for efficiency.
    """

    # Default class-level criteria dictionary
    default_criteria = {
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
    def transform_cpt(df, date_month_day="08-01", missing_value=None):
        """
        Transform a DataFrame with:
          - Row 0 = LAT
          - Row 1 = LON
          - Rows 2+ = numeric year data in wide format (stations in columns).

        Returns an xarray DataArray with coords = (T, Y, X), variable = 'Observation'.
        """
        # --- 1) Extract metadata ---
        metadata = (
            df
            .iloc[:2]
            .set_index("STATION")  # -> index = ["LAT", "LON"]
            .T
            .reset_index()         # station names in 'index'
        )
        metadata.columns = ["STATION", "LAT", "LON"]

        # Adjust duplicates in LAT / LON
        metadata["LAT"] = WAS_compute_cessation.adjust_duplicates(metadata["LAT"])
        metadata["LON"] = WAS_compute_cessation.adjust_duplicates(metadata["LON"])

        # --- 2) Extract data part ---
        data_part = df.iloc[2:].copy()
        data_part = data_part.rename(columns={"STATION": "YEAR"})
        data_part["YEAR"] = data_part["YEAR"].astype(int)

        # --- 3) Convert wide -> long ---
        long_data = data_part.melt(
            id_vars="YEAR",
            var_name="STATION",
            value_name="VALUE"
        )

        # --- 4) YEAR -> date (e.g. YYYY-08-01) ---
        long_data["DATE"] = pd.to_datetime(
            long_data["YEAR"].astype(str) + f"-{date_month_day}",
            format="%Y-%m-%d"
        )

        # --- 5) Merge with metadata ---
        final_df = pd.merge(long_data, metadata, on="STATION", how="left")

        # --- 6) Convert to xarray ---
        rainfall_data_array = (
            final_df[["DATE", "LAT", "LON", "VALUE"]]
            .set_index(["DATE", "LAT", "LON"])
            .to_xarray()
            .rename({"VALUE": "Observation", "LAT": "Y", "LON": "X", "DATE": "T"})
        )
        if missing_value is not None:
            rainfall_data_array = rainfall_data_array.where(rainfall_data_array != missing_value)

        return rainfall_data_array

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
        metadata["LON"] = WAS_compute_cessation.adjust_duplicates(metadata["LON"])
        metadata["LAT"] = WAS_compute_cessation.adjust_duplicates(metadata["LAT"])
        metadata["ELEV"] = WAS_compute_cessation.adjust_duplicates(metadata["ELEV"])

        # --- 2) Extract actual data, rename ID -> DATE ---
        data_part = df.iloc[3:].rename(columns={"ID": "DATE"})

        # Melt to long form
        data_long = data_part.melt(id_vars=["DATE"], var_name="STATION", value_name="VALUE")

        # --- 3) Merge with metadata ---
        final_df = pd.merge(data_long, metadata, on="STATION", how="left")

        # Ensure 'DATE' is a proper datetime (assuming format "YYYYmmdd")
        final_df["DATE"] = pd.to_datetime(final_df["DATE"], format="%Y%m%d", errors="coerce")

        # --- 4) Convert to xarray, rename coords ---
        rainfall_data_array = (
            final_df[["DATE", "LAT", "LON", "VALUE"]]
            .set_index(["DATE", "LAT", "LON"])
            .to_xarray()
            .rename({"VALUE": "Observation", "LAT": "Y", "LON": "X", "DATE": "T"})
        )

        return rainfall_data_array
    
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

        return ifin_saison if ifin_saison <= irch_fin else np.nan

    def compute_cessation(self, daily_data, extent, nb_cores):
        """
        Compute cessation dates for each pixel using criteria based on regions.
        """
        # Load zone file & slice it to the area of interest
        mask_char = xr.open_dataset('./utilities/Isohyet_zones.nc')
        mask_char = mask_char.sel(X=slice(extent[1], extent[3]),
                                  Y=slice(extent[0], extent[2]))
        # Flip Y if needed (as done in your example)
        mask_char = mask_char.isel(Y=slice(None, None, -1)).to_array().drop_vars('variable').squeeze()

        daily_data = daily_data.sel(
            X=mask_char.coords['X'],
            Y=mask_char.coords['Y'])
        
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

        return store_cessation.to_array().drop_vars('variable').squeeze('variable')



# class WAS_compute_cessation_dry_spell:
#     """
#     A class for computing the longest dry spell length 
#     after the onset of a rainy season, based on user-defined criteria.
#     """

#     # Default class-level criteria dictionary
#     default_criteria = {
#         1: {"zone_name": "Sahel200_100mm", "start_search1": "05-15", "cumulative": 15, "number_dry_days": 25, "thrd_rain_day": 0.85, "end_search1": "08-15", "nbjour":40, "date_dry_soil":"01-01", "start_search2": "09-01", "ETP": 5.0, "Cap_ret_maxi": 70, "end_search2": "10-05"},
#         2: {"zone_name": "Sahel400_200mm", "start_search1": "05-01", "cumulative": 15, "number_dry_days": 20, "thrd_rain_day": 0.85, "end_search1": "07-31", "nbjour":40, "date_dry_soil":"01-01", "start_search2": "09-01", "ETP": 5.0, "Cap_ret_maxi": 70, "end_search2": "11-10"},
#         3: {"zone_name": "Sahel600_400mm", "start_search1": "03-15", "cumulative": 20, "number_dry_days": 20, "thrd_rain_day": 0.85, "end_search1": "07-31", "nbjour":45, "date_dry_soil":"01-01", "start_search2": "09-15", "ETP": 5.0, "Cap_ret_maxi": 70, "end_search2": "11-15"},
#         4: {"zone_name": "Soudan", "start_search1": "03-15", "cumulative": 20, "number_dry_days": 10, "thrd_rain_day": 0.85, "end_search1": "07-31", "nbjour":50, "date_dry_soil":"01-01", "start_search2": "10-01", "ETP": 4.5, "Cap_ret_maxi": 70, "end_search2": "11-30"},
#         5: {"zone_name": "Golfe_Of_Guinea","start_search1": "02-01", "cumulative": 20, "number_dry_days": 10, "thrd_rain_day": 0.85, "end_search1": "06-15", "nbjour":50, "date_dry_soil":"01-01", "start_search2": "10-15", "ETP": 4.0, "Cap_ret_maxi": 70, "end_search2": "12-01"},
#     }

    
#     def __init__(self, user_criteria=None):
#         """
#         Initialize the WAS_compute_dry_spell class with user-defined or default criteria.

#         Parameters
#         ----------
#         user_criteria : dict, optional
#             A dictionary containing zone-specific criteria. If not provided,
#             the class will use the default criteria.
#         """
#         if user_criteria:
#             self.criteria = user_criteria
#         else:
#             self.criteria = WAS_compute_cessation_dry_spell.default_criteria


#     @staticmethod
#     def adjust_duplicates(series, increment=0.00001):
#         """
#         If any values in the Series repeat, nudge them by a tiny increment
#         so that all are unique (to avoid indexing collisions).
#         """
#         counts = series.value_counts()
#         for val, count in counts[counts > 1].items():
#             duplicates = series[series == val].index
#             for i, idx in enumerate(duplicates):
#                 series.at[idx] += increment * i
#         return series

#     @staticmethod
#     def transform_cpt(df, date_month_day="08-01", missing_value=None):
#         """
#         Transform a DataFrame with:
#           - Row 0 = LAT
#           - Row 1 = LON
#           - Rows 2+ = numeric year data in wide format (stations in columns).

#         Returns an xarray DataArray with coords = (T, Y, X), variable = 'Observation'.
#         """
#         # --- 1) Extract metadata ---
#         metadata = (
#             df
#             .iloc[:2]
#             .set_index("STATION")  # -> index = ["LAT", "LON"]
#             .T
#             .reset_index()         # station names in 'index'
#         )
#         metadata.columns = ["STATION", "LAT", "LON"]

#         # Adjust duplicates in LAT / LON
#         metadata["LAT"] = WAS_compute_cessation_dry_spell.adjust_duplicates(metadata["LAT"])
#         metadata["LON"] = WAS_compute_cessation_dry_spell.adjust_duplicates(metadata["LON"])

#         # --- 2) Extract data part ---
#         data_part = df.iloc[2:].copy()
#         data_part = data_part.rename(columns={"STATION": "YEAR"})
#         data_part["YEAR"] = data_part["YEAR"].astype(int)

#         # --- 3) Convert wide -> long ---
#         long_data = data_part.melt(
#             id_vars="YEAR",
#             var_name="STATION",
#             value_name="VALUE"
#         )

#         # --- 4) YEAR -> date (e.g. YYYY-08-01) ---
#         long_data["DATE"] = pd.to_datetime(
#             long_data["YEAR"].astype(str) + f"-{date_month_day}",
#             format="%Y-%m-%d"
#         )

#         # --- 5) Merge with metadata ---
#         final_df = pd.merge(long_data, metadata, on="STATION", how="left")

#         # --- 6) Convert to xarray ---
#         rainfall_data_array = (
#             final_df[["DATE", "LAT", "LON", "VALUE"]]
#             .set_index(["DATE", "LAT", "LON"])
#             .to_xarray()
#             .rename({"VALUE": "Observation", "LAT": "Y", "LON": "X", "DATE": "T"})
#         )
#         if missing_value is not None:
#             rainfall_data_array = rainfall_data_array.where(rainfall_data_array != missing_value)

#         return rainfall_data_array

#     @staticmethod
#     def transform_cdt(df):
#         """
#         Transform a DataFrame with:
#           - Row 0 = LON
#           - Row 1 = LAT
#           - Row 2 = ELEV
#           - Rows 3+ = daily data (or any date) with 'ID' column containing dates.

#         Returns an xarray DataArray with coords = (T, Y, X), variable = 'Observation'.
#         """
#         # --- 1) Extract metadata (first 3 rows) ---
#         metadata = df.iloc[:3].set_index("ID").T.reset_index()
#         metadata.columns = ["STATION", "LON", "LAT", "ELEV"]

#         # Adjust duplicates
#         metadata["LON"] = WAS_compute_cessation_dry_spell.adjust_duplicates(metadata["LON"])
#         metadata["LAT"] = WAS_compute_cessation_dry_spell.adjust_duplicates(metadata["LAT"])
#         metadata["ELEV"] = WAS_compute_cessation_dry_spell.adjust_duplicates(metadata["ELEV"])

#         # --- 2) Extract actual data, rename ID -> DATE ---
#         data_part = df.iloc[3:].rename(columns={"ID": "DATE"})

#         # Melt to long form
#         data_long = data_part.melt(id_vars=["DATE"], var_name="STATION", value_name="VALUE")

#         # --- 3) Merge with metadata ---
#         final_df = pd.merge(data_long, metadata, on="STATION", how="left")

#         # Ensure 'DATE' is a proper datetime (assuming format "YYYYmmdd")
#         final_df["DATE"] = pd.to_datetime(final_df["DATE"], format="%Y%m%d", errors="coerce")

#         # --- 4) Convert to xarray, rename coords ---
#         rainfall_data_array = (
#             final_df[["DATE", "LAT", "LON", "VALUE"]]
#             .set_index(["DATE", "LAT", "LON"])
#             .to_xarray()
#             .rename({"VALUE": "Observation", "LAT": "Y", "LON": "X", "DATE": "T"})
#         )

#         return rainfall_data_array

#     def dry_spell_cessation_function(x, idebut1, cumul, nbsec, jour_pluvieux, irch_fin1, idebut2, ijour_dem_cal, ETP, Cap_ret_maxi, irch_fin2, nbjour):
        
#         mask = (np.any(np.isfinite(x)) and 
#                     np.isfinite(idebut1) and 
#                     np.isfinite(nbsec) and 
#                     np.isfinite(irch_fin1) and
#                     np.isfinite(idebut2) and
#                     np.isfinite(ijour_dem_cal) and
#                     np.isfinite(ETP) and
#                     np.isfinite(Cap_ret_maxi) and
#                     np.isfinite(irch_fin2)
#             		)
        
#         if not mask:
#             return np.nan
                
#         idebut1 = int(idebut1)
#         nbsec = int(nbsec)
#         irch_fin1 = int(irch_fin1)
#         idebut2 = int(idebut2)
#         ijour_dem_cal = int(ijour_dem_cal)
#         irch_fin2 = int(irch_fin2)
#         nbjour = int(nbjour)
#         ru = 0
#         trouv = 0
#         idate = idebut1
#         while True:
#             idate += 1
#             ipreced = idate - 1
#             isuiv = idate + 1
    
#             # Check for missing data or out-of-bounds
#             if (ipreced >= len(x) or 
#                 idate >= len(x) or 
#                 isuiv >= len(x) or 
#                 pd.isna(x[ipreced]) or 
#                 pd.isna(x[idate]) or 
#                 pd.isna(x[isuiv])):
#                 deb_saison = np.nan
#                 break
    
#             # Check for end search of date
#             if idate > irch_fin1:
#                 deb_saison = irch_fin1
#                 break
    
#             # Calculate cumulative rainfall over 1, 2, and 3 days
#             cumul3jr = x[ipreced] + x[idate] + x[isuiv]
#             cumul2jr = x[ipreced] + x[idate]
#             cumul1jr = x[ipreced]
    
#             # Check if any cumulative rainfall meets the threshold
#             if (cumul1jr >= cumul or 
#                 cumul2jr >= cumul or 
#                 cumul3jr >= cumul):
#                 troisp = np.array([x[ipreced], x[idate], x[isuiv]])
#                 itroisp = np.array([ipreced, idate, isuiv])
#                 maxp = np.nanmax(troisp)
#                 imaxp = np.where(troisp == maxp)[0][0]
#                 ideb = itroisp[imaxp]
#                 deb_saison = ideb
#                 trouv = 1
    
#                 # Check for sequences of dry days within the next 30 days
#                 finp = ideb + 30
#                 pluie30jr = x[ideb:finp + 1] if finp < len(x) else x[ideb:]
#                 isec = 0
    
#                 while True:
#                     isec += 1
#                     isecf = isec + nbsec
#                     if isecf >= len(pluie30jr):
#                         break
#                     donneeverif = pluie30jr[isec:isecf + 1]
    
#                     # Count days with rainfall below jour_pluvieux
#                     test1 = np.sum(donneeverif < jour_pluvieux)
    
#                     # If a dry sequence is found, reset trouv to 0
#                     if test1 == (nbsec + 1):
#                         trouv = 0
    
#                     # Break if a dry sequence is found or we've reached the end of the window
#                     if test1 == (nbsec + 1) or isec == (30 - nbsec):
#                         break
    
#             # Break if onset is found
#             if trouv == 1:
#                 break
     
    
#         for k in range(ijour_dem_cal, idebut2 + 1):
#             if pd.isna(x[k]):
#                 continue
#             ru += x[k] - ETP
#             ru = max(0, min(ru, Cap_ret_maxi))
    
#         ifin_saison = idebut2
#         while ifin_saison < irch_fin2:
#             ifin_saison += 1
#             if pd.isna(x[ifin_saison]):
#                 continue
#             ru += x[ifin_saison] - ETP
#             ru = max(0, min(ru, Cap_ret_maxi))
#             if ru <= 0:
#                 break
#         fin_saison = ifin_saison        
    
#         if not np.isnan(fin_saison) and (fin_saison - (deb_saison + nbjour)) > 0:
#             pluie50jr_finsaison = x[deb_saison + nbjour:fin_saison]
#             d1 = np.array([0] + list(np.where(pluie50jr_finsaison > jour_pluvieux)[0]))
#             d2 = np.array(list(np.where(pluie50jr_finsaison > jour_pluvieux)[0]) + [len(pluie50jr_finsaison)])
#             seq_max = np.max(d2 - d1) - 1
#             return seq_max
#         else:
#             return np.nan  
            
#     @staticmethod
#     def day_of_year(i, dem_rech1):
#         """
#         Given a year 'i' and a month-day string 'dem_rech1' (e.g., '07-23'),
#         return the 1-based day of the year.
#         """
#         year = int(i)
#         full_date_str = f"{year}-{dem_rech1}"
#         current_date = datetime.datetime.strptime(full_date_str, "%Y-%m-%d").date()
#         origin_date = datetime.date(year, 1, 1)
#         day_of_year_value = (current_date - origin_date).days + 1
#         return day_of_year_value

#     def compute_dry_spell(self, daily_data, extent, nb_cores):
#         """
#         Compute the longest dry spell length after the onset for each pixel in a
#         given daily rainfall DataArray, using different criteria based on isohyet zones.

#         Parameters
#         ----------
#         daily_data : xarray.DataArray
#             Daily rainfall data, coords = (T, Y, X).
#         extent : list or tuple
#             [Ymin, Xmin, Ymax, Xmax] defining the geographic slice.
#         nb_cores : int
#             Number of parallel processes to use.
#         nbjour : int
#             The number of days after onset to search for the longest dry spell.

#         Returns
#         -------
#         xarray.DataArray
#             Array with the longest dry spell length per pixel.
#         """
#         # Load zone file & slice it to the area of interest
#         mask_char = xr.open_dataset('./utilities/Isohyet_zones.nc')
#         mask_char = mask_char.sel(X=slice(extent[1], extent[3]),
#                                   Y=slice(extent[0], extent[2]))
#         # Flip Y if needed (as done in your example)
#         mask_char = mask_char.isel(Y=slice(None, None, -1)).to_array().drop_vars('variable').squeeze()

#         # Get unique zone IDs
#         unique_zone = np.unique(mask_char.to_numpy())
#         unique_zone = unique_zone[~np.isnan(unique_zone)]

#         # Compute year range
#         years = np.unique(daily_data['T'].dt.year.to_numpy())

#         # Create T dimension for the earliest (or any) zone's start date as reference
#         zone_id_to_use = int(np.max(unique_zone))  # or some logic of your choosing
#         T_from_here = daily_data.sel(T=[f"{str(i)}-{self.criteria[zone_id_to_use]['start_search2']}" for i in years])

#         # Prepare chunk sizes
#         chunksize_x = int(np.round(len(daily_data.get_index("X")) / nb_cores))
#         chunksize_y = int(np.round(len(daily_data.get_index("Y")) / nb_cores))

#         # Initialize placeholders
#         mask_char_start_search1 = mask_char_cumulative = mask_char_number_dry_days = mask_char_thrd_rain_day = mask_char_end_search1 = mask_char_nbjour = mask_char_start_search2 = mask_char_date_dry_soil = mask_char_ETP = mask_char_Cap_ret_maxi = mask_char_end_search2 = mask_char

#         store_dry_spell = []
#         for i in years:
#             for j in unique_zone:
#                 # Replace zone values with numeric parameters
#                 mask_char_start_search1 = xr.where(
#                     mask_char_start_search1 == j,
#                     self.day_of_year(i, self.criteria[j]["start_search1"]),
#                     mask_char_start_search1
#                 )
                
#                 mask_char_cumulative = xr.where(
#                     mask_char_cumulative == j,
#                     self.criteria[j]["cumulative"],
#                     mask_char_cumulative
#                 )
#                 mask_char_number_dry_days = xr.where(
#                     mask_char_number_dry_days == j,
#                     self.criteria[j]["number_dry_days"],
#                     mask_char_number_dry_days
#                 )
#                 mask_char_thrd_rain_day = xr.where(
#                     mask_char_thrd_rain_day == j,
#                     self.criteria[j]["thrd_rain_day"],
#                     mask_char_thrd_rain_day
#                 )
#                 mask_char_end_search1 = xr.where(
#                     mask_char_end_search1 == j,
#                     self.day_of_year(i, self.criteria[j]["end_search1"]),
#                     mask_char_end_search1
#                 )
#                 mask_char_nbjour = xr.where(
#                     mask_char_nbjour == j,
#                     self.criteria[j]["nbjour"],
#                     mask_char_nbjour
#                 )
#                 mask_char_date_dry_soil = xr.where(
#                     mask_char_date_dry_soil == j,
#                     self.day_of_year(i, self.criteria[j]["date_dry_soil"]),
#                     mask_char_date_dry_soil,
#                 )
#                 mask_char_start_search2 = xr.where(
#                     mask_char_start_search2 == j,
#                     self.day_of_year(i, self.criteria[j]["start_search2"]),
#                     mask_char_start_search2,
#                 )
#                 mask_char_ETP = xr.where(mask_char_ETP == j, self.criteria[j]["ETP"], mask_char_ETP)
#                 mask_char_Cap_ret_maxi = xr.where(
#                     mask_char_Cap_ret_maxi == j,
#                     self.criteria[j]["Cap_ret_maxi"],
#                     mask_char_Cap_ret_maxi,
#                 )
#                 mask_char_end_search2 = xr.where(
#                     mask_char_end_search2 == j,
#                     self.day_of_year(i, self.criteria[j]["end_search2"]),
#                     mask_char_end_search2,
#                 )
            
#             # Select data for this particular year
#             year_data = daily_data.sel(T=str(i))
         
#             # Parallel processing
#             client = Client(n_workers=nb_cores, threads_per_worker=1)
#             result = xr.apply_ufunc(
#                 self.dry_spell_cessation_function,  # <-- Call our instance method
#                 year_data.chunk({'Y': chunksize_y, 'X': chunksize_x}),
#                 mask_char_start_search1.chunk({'Y': chunksize_y, 'X': chunksize_x}),
#                 mask_char_cumulative.chunk({'Y': chunksize_y, 'X': chunksize_x}),
#                 mask_char_number_dry_days.chunk({'Y': chunksize_y, 'X': chunksize_x}),
#                 mask_char_thrd_rain_day.chunk({'Y': chunksize_y, 'X': chunksize_x}),
#                 mask_char_end_search1.chunk({'Y': chunksize_y, 'X': chunksize_x}),
#                 mask_char_start_search2.chunk({'Y': chunksize_y, 'X': chunksize_x}),  
#                 mask_char_date_dry_soil.chunk({'Y': chunksize_y, 'X': chunksize_x}),
#                 mask_char_ETP.chunk({'Y': chunksize_y, 'X': chunksize_x}),
#                 mask_char_Cap_ret_maxi.chunk({'Y': chunksize_y, 'X': chunksize_x}),
#                 mask_char_end_search2.chunk({'Y': chunksize_y, 'X': chunksize_x}),
#                 mask_char_nbjour.chunk({'Y': chunksize_y, 'X': chunksize_x}),
#                 input_core_dims=[('T',), (), (), (), (), (), (), (), (), (), (), ()],
#                 vectorize=True,
#                 output_core_dims=[()],
#                 dask='parallelized',
#                 output_dtypes=['float'],
#             )
#             result_ = result.compute()
#             client.close()

#             store_dry_spell.append(result_)

#         # Concatenate final result
#         store_dry_spell = xr.concat(store_dry_spell, dim="T")
#         store_dry_spell['T'] = T_from_here['T']

#         return store_dry_spell




class WAS_compute_cessation_dry_spell:
    """
    A class for computing the longest dry spell length 
    after the onset of a rainy season, based on user-defined criteria.
    """

    # Default class-level criteria dictionary
    default_criteria = {
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
    def transform_cpt(df, date_month_day="08-01", missing_value=None):
        """
        Transform a DataFrame with:
          - Row 0 = LAT
          - Row 1 = LON
          - Rows 2+ = numeric year data in wide format (stations in columns).

        Returns
        -------
        xarray.DataArray
            Data with coords (T, Y, X) and data variable 'Observation'.
        """
        # --- 1) Extract metadata ---
        metadata = (
            df
            .iloc[:2]
            .set_index("STATION")  # -> index = ["LAT", "LON"]
            .T
            .reset_index()         # station names in 'index'
        )
        metadata.columns = ["STATION", "LAT", "LON"]

        # Adjust duplicates in LAT / LON
        metadata["LAT"] = WAS_compute_cessation_dry_spell.adjust_duplicates(metadata["LAT"])
        metadata["LON"] = WAS_compute_cessation_dry_spell.adjust_duplicates(metadata["LON"])

        # --- 2) Extract data part ---
        data_part = df.iloc[2:].copy()
        data_part = data_part.rename(columns={"STATION": "YEAR"})
        data_part["YEAR"] = data_part["YEAR"].astype(int)

        # --- 3) Convert wide -> long ---
        long_data = data_part.melt(
            id_vars="YEAR",
            var_name="STATION",
            value_name="VALUE"
        )

        # --- 4) YEAR -> date (e.g. YYYY-08-01) ---
        long_data["DATE"] = pd.to_datetime(
            long_data["YEAR"].astype(str) + f"-{date_month_day}",
            format="%Y-%m-%d"
        )

        # --- 5) Merge with metadata ---
        final_df = pd.merge(long_data, metadata, on="STATION", how="left")

        # --- 6) Convert to xarray ---
        rainfall_data_array = (
            final_df[["DATE", "LAT", "LON", "VALUE"]]
            .set_index(["DATE", "LAT", "LON"])
            .to_xarray()
            .rename({"VALUE": "Observation", "LAT": "Y", "LON": "X", "DATE": "T"})
        )
        if missing_value is not None:
            rainfall_data_array = rainfall_data_array.where(rainfall_data_array != missing_value)

        return rainfall_data_array

    @staticmethod
    def transform_cdt(df):
        """
        Transform a DataFrame with:
          - Row 0 = LON
          - Row 1 = LAT
          - Row 2 = ELEV
          - Rows 3+ = daily data with 'ID' column containing dates.

        Returns
        -------
        xarray.DataArray
            Data with coords (T, Y, X) and data variable 'Observation'.
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

        # --- 3) Merge with metadata ---
        final_df = pd.merge(data_long, metadata, on="STATION", how="left")

        # Ensure 'DATE' is a proper datetime (assuming format "YYYYmmdd")
        final_df["DATE"] = pd.to_datetime(final_df["DATE"], format="%Y%m%d", errors="coerce")

        # --- 4) Convert to xarray ---
        rainfall_data_array = (
            final_df[["DATE", "LAT", "LON", "VALUE"]]
            .set_index(["DATE", "LAT", "LON"])
            .to_xarray()
            .rename({"VALUE": "Observation", "LAT": "Y", "LON": "X", "DATE": "T"})
        )

        return rainfall_data_array

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
                deb_saison = irch_fin1
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

        fin_saison = ifin_saison

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

    def compute_dry_spell(self, daily_data, extent, nb_cores):
        """
        Compute the longest dry spell length after the rainy season onset 
        for each pixel in the given daily rainfall DataArray, using different 
        criteria (both for onset and cessation) based on isohyet zones.

        Parameters
        ----------
        daily_data : xarray.DataArray
            Daily rainfall data, coords = (T, Y, X).
        extent : list or tuple
            [Ymin, Xmin, Ymax, Xmax] defining the geographic slice.
        nb_cores : int
            Number of parallel processes (workers) to use.

        Returns
        -------
        xarray.DataArray
            Array with the longest dry spell length per pixel.
        """
        # 1) Load zone file & slice it
        mask_char = xr.open_dataset("./utilities/Isohyet_zones.nc")
        mask_char = mask_char.sel(X=slice(extent[1], extent[3]),
                                  Y=slice(extent[0], extent[2]))

        # 2) Flip Y if needed
        mask_char = mask_char.isel(Y=slice(None, None, -1)).to_array().drop_vars("variable").squeeze()

        daily_data = daily_data.sel(
            X=mask_char.coords['X'],
            Y=mask_char.coords['Y'])

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

        return store_dry_spell.to_array().drop_vars('variable').squeeze('variable')
        
import numpy as np
import xarray as xr
from dask.distributed import Client

class WAS_count_dry_spells:
    """
    A class to compute the number of dry spells within a specified period
    (onset to cessation) for each pixel in a given daily rainfall dataset.
    """

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
        int
            The number of dry spells of the specified length.
        """
        mask = (
            np.isfinite(x).any()
            and np.isfinite(onset)
            and np.isfinite(cessation)
        )
        if not mask:
            return np.nan
        
        
        dry_spells_count = 0
        current_dry_days = 0

        for day in range(int(onset), int(cessation) + 1):
            if x[day] < dry_threshold:
                current_dry_days += 1
            else:
                if current_dry_days == dry_spell_length:
                    dry_spells_count += 1
                current_dry_days = 0

        # Check if the last dry period counts as a dry spell
        if current_dry_days == dry_spell_length:
            dry_spells_count += 1

        return dry_spells_count

    def compute_count_dry_spells(
        self,
        daily_data,
        extent,
        onset_date,
        cessation_date,
        dry_spell_length,
        dry_threshold,
        nb_cores
    ):
        """
        Compute the number of dry spells for each pixel within the onset and cessation period.

        Parameters
        ----------
        daily_data : xarray.DataArray
            Daily rainfall data, coords = (T, Y, X).
        extent : list or tuple
            [Ymin, Xmin, Ymax, Xmax] defining the geographic slice.
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
            Array with the count of dry spells per pixel.
        """

        # Ensure alignment of onset and cessation dates
        cessation_date['T'] = onset_date['T']
        cessation_date, onset_date = xr.align(cessation_date, onset_date)
        daily_data = daily_data.sel(
            X=onset_date.coords['X'],
            Y=onset_date.coords['Y'])
        # Compute year range
        years = np.unique(daily_data['T'].dt.year.to_numpy())

        # Prepare chunk sizes
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
                year_data.chunk({'Y': chunksize_y, 'X': chunksize_x}),
                year_onset_date.chunk({'Y': chunksize_y, 'X': chunksize_x}),
                year_cessation_date.chunk({'Y': chunksize_y, 'X': chunksize_x}),
                input_core_dims=[('T',), (), ()],
                vectorize=True,
                kwargs={
                    'dry_spell_length': dry_spell_length,
                    'dry_threshold': dry_threshold,
                },
                output_core_dims=[()],
                dask='parallelized',
                output_dtypes=['float'],
            )
            result_ = result.compute()
            client.close()

            store_nb_dryspell.append(result_)

        # Concatenate the final result
        store_nb_dryspell = xr.concat(store_nb_dryspell, dim="T")
        store_nb_dryspell['T'] = onset_date['T']

        return store_nb_dryspell



import numpy as np
import xarray as xr
from dask.distributed import Client

class WAS_count_wet_spells:
    """
    A class to compute the number of wet spells within a specified period
    (onset to cessation) for each pixel in a given daily rainfall dataset.
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
        int
            The number of wet spells of the specified length.
        """
        mask = (
            np.isfinite(x).any()
            and np.isfinite(onset_date)
            and np.isfinite(cessation_date)
        )
        if not mask:
            return np.nan        
        
        wet_spells_count = 0
        current_wet_days = 0

        for day in range(int(onset_date), int(cessation_date) + 1):
            if x[day] >= wet_threshold:
                current_wet_days += 1
            else:
                if current_wet_days == wet_spell_length:
                    wet_spells_count += 1
                current_wet_days = 0

        # Check if the last wet period counts as a wet spell
        if current_wet_days == wet_spell_length:
            wet_spells_count += 1

        return wet_spells_count

    def compute_count_wet_spells(
        self,
        daily_data,
        extent,
        onset_date,
        cessation_date,
        wet_spell_length,
        wet_threshold,
        nb_cores
    ):
        """
        Compute the number of wet spells for each pixel within the onset and cessation period.

        Parameters
        ----------
        daily_data : xarray.DataArray
            Daily rainfall data, coords = (T, Y, X).
        extent : list or tuple
            [Ymin, Xmin, Ymax, Xmax] defining the geographic slice.
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
        # Align onset and cessation dates
        cessation_date['T'] = onset_date['T']
        cessation_date, onset_date = xr.align(cessation_date, onset_date)

        # Compute year range
        years = np.unique(daily_data['T'].dt.year.to_numpy())

        # Prepare chunk sizes
        chunksize_x = int(np.round(len(daily_data.get_index("X")) / nb_cores))
        chunksize_y = int(np.round(len(daily_data.get_index("Y")) / nb_cores))

        store_nb_wetspell = []

        for i in years:
            # Select data for the current year
            year_data = daily_data.sel(T=str(i))
            year_cessation_date = cessation_date.sel(T=str(i)).squeeze()
            year_onset_date = onset_date.sel(T=str(i)).squeeze()
            
            # Set up parallel processing
            client = Client(n_workers=nb_cores, threads_per_worker=1)
            result = xr.apply_ufunc(
                self.count_wet_spells,
                year_data.chunk({'Y': chunksize_y, 'X': chunksize_x}),
                year_onset_date.chunk({'Y': chunksize_y, 'X': chunksize_x}),
                year_cessation_date.chunk({'Y': chunksize_y, 'X': chunksize_x}),
                input_core_dims=[('T',), (), ()],
                vectorize=True,
                kwargs={
                    'wet_spell_length': wet_spell_length,
                    'wet_threshold': wet_threshold,
                },
                output_core_dims=[()],
                dask='parallelized',
                output_dtypes=['float'],
            )
            result_ = result.compute()
            client.close()

            store_nb_wetspell.append(result_)

        # Concatenate the final result
        store_nb_wetspell = xr.concat(store_nb_wetspell, dim="T")
        store_nb_wetspell['T'] = onset_date['T']

        return store_nb_wetspell



class WAS_count_rainy_days:
    """
    A class to compute the number of rainy days between onset and cessation dates
    for each pixel in a daily rainfall dataset.
    """

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
        int
            The number of rainy days.
        """

        mask = (
            np.isfinite(x).any()
            and np.isfinite(onset_date)
            and np.isfinite(cessation_date)
        )
        if not mask:
            return np.nan   
        
        rainy_days_count = 0

        for day in range(int(onset_date), int(cessation_date) + 1):
            if x[day] >= rain_threshold:
                rainy_days_count += 1

        return rainy_days_count

    def compute_count_rainy_days(
        self,
        daily_data,
        extent,
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
        extent : list or tuple
            [Ymin, Xmin, Ymax, Xmax] defining the geographic slice.
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

        return store_nb_rainy_days

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

    def compute_HWSDI(self, temperature_data, base_period_start='1961', base_period_end='1990', nb_cores=4):
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


