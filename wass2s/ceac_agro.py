import pandas as pd
import xarray as xr
import numpy as np
import random
import datetime
from dask.distributed import Client
import warnings

warnings.filterwarnings('ignore')

DEFAULT_CRITERIA = {
    1: {"start_search": "05-01", "start_search2": "09-01", "date_dry_soil": "01-01", "cumulative": 15, "number_dry_days": 15, "thrd_rain_day": 0.85, "end_search": "08-30",  "end_search2": "10-30", "nbjour": 35, "ETP": 5.0, "Cap_ret_maxi": 70},
    2: {"start_search": "03-15", "start_search2": "09-01", "date_dry_soil": "01-01", "cumulative": 20, "number_dry_days": 10, "thrd_rain_day": 0.85, "end_search": "08-01",  "end_search2": "11-01", "nbjour": 40, "ETP": 5.0, "Cap_ret_maxi": 70},
    3: {"start_search": "02-01", "start_search2": "10-01", "date_dry_soil": "01-01", "cumulative": 20, "number_dry_days": 10, "thrd_rain_day": 0.85, "end_search": "05-15", "end_search2": "12-30", "nbjour": 45, "ETP": 5.0, "Cap_ret_maxi": 70},
    4: {"start_search": "01-01", "start_search2": "11-01", "date_dry_soil": "01-01", "cumulative": 20, "number_dry_days": 7, "thrd_rain_day": 0.85, "end_search": "04-01", "end_search2": "12-30", "nbjour": 50, "ETP": 5.0, "Cap_ret_maxi": 80},
    5: {"start_search": "01-01", "start_search2": "06-01", "date_dry_soil": "01-01", "cumulative": 25, "number_dry_days": 7, "thrd_rain_day": 0.85, "end_search": "03-10", "end_search2": "08-10", "nbjour": 50, "ETP": 4.0, "Cap_ret_maxi": 60},
    6: {"start_search": "02-01", "start_search2": "10-15", "date_dry_soil": "01-01", "cumulative": 20, "number_dry_days": 7, "thrd_rain_day": 0.85, "end_search": "03-20",  "end_search2": "12-15", "nbjour": 50, "ETP": 4.0, "Cap_ret_maxi": 70},
    7: {"start_search": "02-01", "start_search2": "09-01", "date_dry_soil": "01-01", "cumulative": 20, "number_dry_days": 10, "thrd_rain_day": 0.85, "end_search": "03-20",  "end_search2": "12-15", "nbjour": 50, "ETP": 4.0, "Cap_ret_maxi": 70},
    8: {"start_search": "03-01", "start_search2": "08-01", "date_dry_soil": "01-01", "cumulative": 20, "number_dry_days": 10, "thrd_rain_day": 0.85, "end_search": "04-20",  "end_search2": "10-15", "nbjour": 40, "ETP": 4.0, "Cap_ret_maxi": 70},
    9: {"start_search": "05-01", "start_search2": "08-15", "date_dry_soil": "01-01", "cumulative": 20, "number_dry_days": 10, "thrd_rain_day": 0.85, "end_search": "06-20",  "end_search2": "10-15", "nbjour": 30, "ETP": 4.0, "Cap_ret_maxi": 70},
}

class CAF_AgroClimateBase:
    SHIFT_OFFSET = 244 # L'offset pour marquer l'extension

    def __init__(self, user_criteria=None):
        self.criteria = user_criteria if user_criteria else DEFAULT_CRITERIA

    def _is_shifted(self, z):
        # zone du shift
        return not pd.isna(z) and int(z) >= 6 

    @staticmethod
    def day_of_year(y, mm_dd):
        dt = datetime.datetime.strptime(f"{int(y)}-{mm_dd}", "%Y-%m-%d").date()
        return (dt - datetime.date(int(y), 1, 1)).days + 1

    def get_index_for_station(self, year, mm_dd, z):
        shifted = self._is_shifted(z)
        base_date = datetime.date(year, 8, 1) if shifted else datetime.date(year, 1, 1)
        target_month, target_day = map(int, mm_dd.split('-'))
        target_year = year + 1 if (shifted and target_month < 8) else year
        target_date = datetime.date(target_year, target_month, target_day)
        return (target_date - base_date).days

    def output_format_value(self, v, z):
        if np.isnan(v): return np.nan
        return v + self.SHIFT_OFFSET if self._is_shifted(z) else v + 1

    def revert_to_index(self, v, z):
        if np.isnan(v): return np.nan
        return int(v - self.SHIFT_OFFSET) if self._is_shifted(z) else int(v - 1)

    def _map_criteria(self, mask, key, year):
        def _safe_get(z):
            if np.isnan(z) or int(z) not in self.criteria: return np.nan
            v = self.criteria[int(z)][key]
            return self.day_of_year(year, v) if 'search' in key or 'date' in key else v
        return xr.DataArray(np.vectorize(_safe_get, otypes=[float])(mask.values), coords=mask.coords)

    def shift_gridded_data(self, daily_data, map_reclassified):
        mask = map_reclassified.reindex_like(daily_data, method='nearest')
        y1, y2 = int(daily_data['T'].dt.year.min()), int(daily_data['T'].dt.year.max())
        
        #  mask <= 5 vs mask >= 6
        le5 = daily_data.sel(T=slice(f"{y1}", f"{y2-1}")).where(mask <= 5)
        gt6 = daily_data.sel(T=slice(f"{y1}-08-01", f"{y2}-07-31")).where(mask >= 6)
        
        m_len = min(len(le5['T']), len(gt6['T']))
        le5, gt6 = le5.isel(T=slice(0, m_len)), gt6.isel(T=slice(0, m_len))
        gt6 = gt6.assign_coords(T=le5['T'].values)
        return le5.combine_first(gt6), mask, np.unique(le5['T'].dt.year.to_numpy())

    def format_grid_output(self, res_xr, mask):
        return xr.where(mask >= 6, res_xr + self.SHIFT_OFFSET, res_xr + 1)

    def revert_grid_index(self, res_xr, mask):
        return xr.where(mask >= 6, res_xr - self.SHIFT_OFFSET, res_xr - 1)

    def transform_and_shift_cdt(self, df_raw, map_reclassified):
        if "ID" in df_raw.columns or str(df_raw.columns[0]).upper() == "ID":
            header_row = pd.DataFrame([df_raw.columns])
            header_row.columns = range(df_raw.shape[1])
            df_raw.columns = range(df_raw.shape[1])
            df_raw = pd.concat([header_row, df_raw], ignore_index=True)

        s_ids = pd.Series(df_raw.iloc[0, 1:].values.astype(str))
        ids = s_ids.where(~s_ids.duplicated(), s_ids + "_" + s_ids.groupby(s_ids).cumcount().astype(str)).values
        df_raw.iloc[0, 1:] = ids 
        
        lons, lats = df_raw.iloc[1, 1:].astype(float).values, df_raw.iloc[2, 1:].astype(float).values
        dates = pd.to_datetime(df_raw.iloc[4:, 0], format='%Y%m%d')
        da = xr.DataArray(df_raw.iloc[4:, 1:].astype(float).values, coords={'T': dates, 'station': ids}, dims=['T', 'station'])
        y1, y2 = dates.min().year, dates.max().year
        
        stn_zones, series_lst = {}, []
        for i, stn in enumerate(ids):
            try: z = map_reclassified.sel(X=lons[i], Y=lats[i], method='nearest').values.item()
            except: z = np.nan
            stn_zones[stn] = z
            s = da.isel(station=i)
            # Construction des séries continues par station
            if self._is_shifted(z): series_lst.append(s.sel(T=slice(f"{y1}-08-01", f"{y2}-07-31")))
            else: series_lst.append(s.sel(T=slice(f"{y1}-01-01", f"{y2-1}-12-31")))
                
        m_len = min(len(s) for s in series_lst)
        std_dt_vals = dates.iloc[:m_len].values  
        
        arr_2d = np.column_stack([s.values[:m_len] for s in series_lst])
        df_shifted = pd.DataFrame(arr_2d, index=std_dt_vals, columns=ids)
        
        df_long = df_shifted.reset_index().melt(id_vars="index", var_name="STATION", value_name="VALUE")
        df_long.rename(columns={"index": "DATE"}, inplace=True)
        
        meta_df = pd.DataFrame({"STATION": ids, "LON": lons, "LAT": lats, "zonename": [stn_zones[s] for s in ids]})
        df_long = df_long.merge(meta_df, on="STATION")
        df_long["VALUE"] = df_long["VALUE"].replace(-99.0, np.nan)
        df_long["year"] = df_long["DATE"].dt.year
        
        return df_long, stn_zones, df_raw.iloc[:4, :]

    def _parse_cpt_to_long(self, df_cpt, val_name):
        lats, lons = df_cpt.iloc[0, 1:].values, df_cpt.iloc[1, 1:].values
        cols = df_cpt.columns[1:].tolist()
        df = df_cpt.iloc[2:].copy().reset_index(drop=True).rename(columns={"STATION": "year"})
        df = df.melt(id_vars=["year"], var_name="station", value_name=val_name)
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["lat"], df["lon"] = df["station"].map(dict(zip(cols, lats))), df["station"].map(dict(zip(cols, lons)))
        return df

    def build_cpt_output(self, res_df, val_col):
        res_df[val_col] = res_df[val_col].fillna(-999.0)
        piv = res_df.pivot(index="year", columns="station", values=val_col)
        meta = res_df.groupby("station")[["lat", "lon"]].first()
        lats, lons = meta.loc[piv.columns, "lat"].tolist(), meta.loc[piv.columns, "lon"].tolist()
        lat_row = pd.DataFrame([lats], columns=piv.columns, index=["LAT"])
        lon_row = pd.DataFrame([lons], columns=piv.columns, index=["LON"])
        final = pd.concat([lat_row, lon_row, piv]).reset_index().rename(columns={"index": "STATION"})
        final.columns.name = None 
        return final


class CEAC_compute_onset(CAF_AgroClimateBase):
    @staticmethod
    def onset_function(x, idebut, cumul, nbsec, jour_pluvieux, irch_fin):
        if not (np.any(np.isfinite(x)) and np.isfinite(idebut) and np.isfinite(nbsec) and np.isfinite(irch_fin)): return np.nan
        idebut, nbsec, irch_fin = int(idebut), int(nbsec), int(irch_fin)
        idate, trouv = idebut, 0
        while True:
            idate += 1
            if idate >= len(x)-1 or pd.isna(x[idate-1]) or pd.isna(x[idate]) or pd.isna(x[idate+1]): return np.nan
            if idate > irch_fin: return random.randint(irch_fin - 5, irch_fin)
            c1, c2, c3 = x[idate-1], x[idate-1]+x[idate], x[idate-1]+x[idate]+x[idate+1]
            if c1 >= cumul or c2 >= cumul or c3 >= cumul:
                arr = np.array([x[idate-1], x[idate], x[idate+1]])
                ideb = [idate-1, idate, idate+1][np.argmax(arr)]
                trouv = 1
                pluie30 = x[ideb:ideb+31] if ideb+30 < len(x) else x[ideb:]
                isec = 0
                while True:
                    isec += 1
                    if isec+nbsec >= len(pluie30): break
                    if np.sum(pluie30[isec:isec+nbsec+1] < jour_pluvieux) == (nbsec + 1): trouv = 0; break
                    if isec == (30 - nbsec): break
            if trouv == 1: return ideb
        return np.nan

    def compute_insitu(self, daily_df_raw, map_rec):
        df_long, zones, _ = self.transform_and_shift_cdt(daily_df_raw, map_rec)
        res = []
        for (stn, y), group in df_long.groupby(["STATION", "year"]):
            z = zones[stn]
            if pd.isna(z) or int(z) not in self.criteria: v = np.nan
            else:
                z = int(z)
                c = self.criteria[z]
                v = self.onset_function(group["VALUE"].values, self.get_index_for_station(y, c["start_search"], z), c["cumulative"], c["number_dry_days"], c["thrd_rain_day"], self.get_index_for_station(y, c["end_search"], z))
                v = self.output_format_value(v, z)
            res.append({"year": y, "station": stn, "lat": group["LAT"].iloc[0], "lon": group["LON"].iloc[0], "onset": v})
        return self.build_cpt_output(pd.DataFrame(res), "onset")

    def compute(self, daily_data, map_rec, nb_cores):
        shifted, mask, years = self.shift_gridded_data(daily_data, map_rec)
        cx, cy = int(np.round(len(shifted.X)/nb_cores)), int(np.round(len(shifted.Y)/nb_cores))
        out = []
        for y in years:
            yd, mk = shifted.sel(T=str(y)), lambda k: self._map_criteria(mask, k, y)
            client = Client(n_workers=nb_cores, threads_per_worker=1)
            res = xr.apply_ufunc(
                self.onset_function, yd.chunk({'Y': cy, 'X': cx}),
                mk("start_search").chunk({'Y': cy, 'X': cx}), mk("cumulative").chunk({'Y': cy, 'X': cx}),
                mk("number_dry_days").chunk({'Y': cy, 'X': cx}), mk("thrd_rain_day").chunk({'Y': cy, 'X': cx}),
                mk("end_search").chunk({'Y': cy, 'X': cx}),
                input_core_dims=[('T',)]+[()]*5, vectorize=True, dask='parallelized', output_dtypes=['float']
            ).compute()
            client.close()
            out.append(res)
            
        unique_zone = np.unique(mask.to_numpy())
        unique_zone = unique_zone[~np.isnan(unique_zone)]
        # Choose a date to store results
        if unique_zone.size == 0:
            raise ValueError("No valid zones found in the mask.")
        else:
            # Use zone in low latitude
            zone_id_to_use = int(np.min(unique_zone))
        
        start_search_str = self.criteria[zone_id_to_use]["start_search"]        
        final = self.format_grid_output(xr.concat(out, dim=pd.Index(years, name="T")), mask)
        final['T'] = pd.to_datetime([f"{y}-{start_search_str}" for y in years])
        final.name = "Onset"
        return final


class CEAC_compute_cessation(CAF_AgroClimateBase):
    @staticmethod
    def cessation_function(x, ijour, idebut, ETP, Cap, irch_fin):
        if not (np.isfinite(x).any() and np.isfinite(idebut) and np.isfinite(ijour) and np.isfinite(ETP) and np.isfinite(Cap) and np.isfinite(irch_fin)): return np.nan
        ru, ifin = 0, int(idebut)
        for k in range(int(ijour), ifin + 1):
            if not pd.isna(x[k]): ru = max(0, min(ru + x[k] - ETP, Cap))
        while ifin < int(irch_fin):
            ifin += 1
            if ifin >= len(x) or pd.isna(x[ifin]): continue
            ru = max(0, min(ru + x[ifin] - ETP, Cap))
            if ru <= 0: break
        return ifin if ifin <= int(irch_fin) else random.randint(int(irch_fin) - 5, int(irch_fin))

    def compute_insitu(self, daily_df_raw, map_rec):
        df_long, zones, _ = self.transform_and_shift_cdt(daily_df_raw, map_rec)
        res = []
        for (stn, y), group in df_long.groupby(["STATION", "year"]):
            z = zones[stn]
            if pd.isna(z) or int(z) not in self.criteria: v = np.nan
            else:
                z = int(z)
                c = self.criteria[z]
                v = self.cessation_function(group["VALUE"].values, self.get_index_for_station(y, c["date_dry_soil"], z), self.get_index_for_station(y, c["start_search2"], z), c["ETP"], c["Cap_ret_maxi"], self.get_index_for_station(y, c["end_search2"], z))
                v = self.output_format_value(v, z)
            res.append({"year": y, "station": stn, "lat": group["LAT"].iloc[0], "lon": group["LON"].iloc[0], "cessation": v})
        return self.build_cpt_output(pd.DataFrame(res), "cessation")

    def compute(self, daily_data, map_rec, nb_cores):
        shifted, mask, years = self.shift_gridded_data(daily_data, map_rec)
        cx, cy = int(np.round(len(shifted.X)/nb_cores)), int(np.round(len(shifted.Y)/nb_cores))
        out = []
        for y in years:
            yd, mk = shifted.sel(T=str(y)), lambda k: self._map_criteria(mask, k, y)
            client = Client(n_workers=nb_cores, threads_per_worker=1)
            res = xr.apply_ufunc(
                self.cessation_function, yd.chunk({'Y': cy, 'X': cx}),
                mk("date_dry_soil").chunk({'Y': cy, 'X': cx}), mk("start_search2").chunk({'Y': cy, 'X': cx}),
                mk("ETP").chunk({'Y': cy, 'X': cx}), mk("Cap_ret_maxi").chunk({'Y': cy, 'X': cx}),
                mk("end_search2").chunk({'Y': cy, 'X': cx}),
                input_core_dims=[('T',)]+[()]*5, vectorize=True, dask='parallelized', output_dtypes=['float']
            ).compute()
            client.close()
            out.append(res)

        unique_zone = np.unique(mask.to_numpy())
        unique_zone = unique_zone[~np.isnan(unique_zone)]
        # Choose a date to store results
        if unique_zone.size == 0:
            raise ValueError("No valid zones found in the mask.")
        else:
            # Use zone in low latitude
            zone_id_to_use = int(np.min(unique_zone))
        
        start_search_str = self.criteria[zone_id_to_use]["start_search"]
        
        final = self.format_grid_output(xr.concat(out, dim=pd.Index(years, name="T")), mask)
        final['T'] = pd.to_datetime([f"{y}-{start_search_str}" for y in years])
        final.name = "Cessation"
        return final


class CEAC_compute_onset_dry_spell(CAF_AgroClimateBase):
    @staticmethod
    def ds_onset_func(x, idebut, cumul, nbsec, jp, irch_fin, nbjour):
        if not (np.any(np.isfinite(x)) and np.isfinite(idebut)): return np.nan
        deb = CEAC_compute_onset.onset_function(x, idebut, cumul, nbsec, jp, irch_fin)
        if not np.isnan(deb):
            p = x[int(deb) : min(int(deb) + int(nbjour) + 1, len(x))]
            r = np.where(p > jp)[0]
            d1, d2 = np.array([0] + list(r)), np.array(list(r) + [len(p)])
            return np.max(d2 - d1) - 1
        return np.nan

    def compute_insitu(self, daily_df_raw, map_rec):
        df_long, zones, _ = self.transform_and_shift_cdt(daily_df_raw, map_rec)
        res = []
        for (stn, y), group in df_long.groupby(["STATION", "year"]):
            z = zones[stn]
            if pd.isna(z) or int(z) not in self.criteria: v = np.nan
            else:
                z = int(z)
                c = self.criteria[z]
                v = self.ds_onset_func(group["VALUE"].values, self.get_index_for_station(y, c["start_search"], z), c["cumulative"], c["number_dry_days"], c["thrd_rain_day"], self.get_index_for_station(y, c["end_search"], z), c["nbjour"])
            res.append({"year": y, "station": stn, "lat": group["LAT"].iloc[0], "lon": group["LON"].iloc[0], "onsetdryspell": v})
        return self.build_cpt_output(pd.DataFrame(res), "onsetdryspell")

    def compute(self, daily_data, map_rec, nb_cores):
        shifted, mask, years = self.shift_gridded_data(daily_data, map_rec)
        cx, cy = int(np.round(len(shifted.X)/nb_cores)), int(np.round(len(shifted.Y)/nb_cores))
        out = []
        for y in years:
            yd, mk = shifted.sel(T=str(y)), lambda k: self._map_criteria(mask, k, y)
            client = Client(n_workers=nb_cores, threads_per_worker=1)
            res = xr.apply_ufunc(
                self.ds_onset_func, yd.chunk({'Y':cy,'X':cx}),
                mk("start_search").chunk({'Y':cy,'X':cx}), mk("cumulative").chunk({'Y':cy,'X':cx}),
                mk("number_dry_days").chunk({'Y':cy,'X':cx}), mk("thrd_rain_day").chunk({'Y':cy,'X':cx}),
                mk("end_search").chunk({'Y':cy,'X':cx}), mk("nbjour").chunk({'Y':cy,'X':cx}),
                input_core_dims=[('T',)]+[()]*6, vectorize=True, dask='parallelized', output_dtypes=['float']
            ).compute()
            client.close()
            out.append(res)

        unique_zone = np.unique(mask.to_numpy())
        unique_zone = unique_zone[~np.isnan(unique_zone)]
        # Choose a date to store results
        if unique_zone.size == 0:
            raise ValueError("No valid zones found in the mask.")
        else:
            # Use zone in low latitude
            zone_id_to_use = int(np.min(unique_zone))
        start_search_str = self.criteria[zone_id_to_use]["start_search"]
        
        final = xr.concat(out, dim=pd.Index(years, name="T"))
        final['T'] = pd.to_datetime([f"{y}-{start_search_str}" for y in years])
        final.name = "Onset_dryspell"
        return final


class CEAC_compute_cessation_dry_spell(CAF_AgroClimateBase):
    @staticmethod
    def ds_cess_func(x, id1, cum, nbs, jp, ir1, id2, ijd, ETP, Cap, ir2, nbj):
        if not (np.any(np.isfinite(x)) and np.isfinite(id1)): return np.nan
        deb = CEAC_compute_onset.onset_function(x, id1, cum, nbs, jp, ir1)
        if pd.isna(deb): return np.nan
        fin = CEAC_compute_cessation.cessation_function(x, ijd, id2, ETP, Cap, ir2)
        if not np.isnan(fin) and (fin - (deb + nbj)) > 0 and (deb + nbj) < len(x):
            p = x[int(deb + nbj):int(fin)]
            r = np.where(p > jp)[0]
            if len(r) == 0: return np.nan
            return np.max(np.array(list(r) + [len(p)]) - np.array([0] + list(r))) - 1
        return np.nan

    def compute_insitu(self, daily_df_raw, map_rec):
        df_long, zones, _ = self.transform_and_shift_cdt(daily_df_raw, map_rec)
        res = []
        for (stn, y), group in df_long.groupby(["STATION", "year"]):
            z = zones[stn]
            if pd.isna(z) or int(z) not in self.criteria: v = np.nan
            else:
                z = int(z)
                c = self.criteria[z]
                v = self.ds_cess_func(group["VALUE"].values, self.get_index_for_station(y, c["start_search"], z), c["cumulative"], c["number_dry_days"], c["thrd_rain_day"], self.get_index_for_station(y, c["end_search"], z), self.get_index_for_station(y, c["start_search2"], z), self.get_index_for_station(y, c["date_dry_soil"], z), c["ETP"], c["Cap_ret_maxi"], self.get_index_for_station(y, c["end_search2"], z), c["nbjour"])
            res.append({"year": y, "station": stn, "lat": group["LAT"].iloc[0], "lon": group["LON"].iloc[0], "cessation_dryspell": v})
        return self.build_cpt_output(pd.DataFrame(res), "cessation_dryspell")

    def compute(self, daily_data, map_rec, nb_cores):
        shifted, mask, years = self.shift_gridded_data(daily_data, map_rec)
        cx, cy = int(np.round(len(shifted.X)/nb_cores)), int(np.round(len(shifted.Y)/nb_cores))
        out = []
        for y in years:
            yd, mk = shifted.sel(T=str(y)), lambda k: self._map_criteria(mask, k, y)
            client = Client(n_workers=nb_cores, threads_per_worker=1)
            res = xr.apply_ufunc(
                self.ds_cess_func, yd.chunk({'Y':cy,'X':cx}),
                mk("start_search").chunk({'Y':cy,'X':cx}), mk("cumulative").chunk({'Y':cy,'X':cx}),
                mk("number_dry_days").chunk({'Y':cy,'X':cx}), mk("thrd_rain_day").chunk({'Y':cy,'X':cx}),
                mk("end_search").chunk({'Y':cy,'X':cx}), mk("start_search2").chunk({'Y':cy,'X':cx}),
                mk("date_dry_soil").chunk({'Y':cy,'X':cx}), mk("ETP").chunk({'Y':cy,'X':cx}),
                mk("Cap_ret_maxi").chunk({'Y':cy,'X':cx}), mk("end_search2").chunk({'Y':cy,'X':cx}),
                mk("nbjour").chunk({'Y':cy,'X':cx}),
                input_core_dims=[('T',)]+[()]*11, vectorize=True, dask='parallelized', output_dtypes=['float']
            ).compute()
            client.close()
            out.append(res)
        unique_zone = np.unique(mask.to_numpy())
        unique_zone = unique_zone[~np.isnan(unique_zone)]
        # Choose a date to store results
        if unique_zone.size == 0:
            raise ValueError("No valid zones found in the mask.")
        else:
            # Use zone in low latitude
            zone_id_to_use = int(np.min(unique_zone))
        start_search_str = self.criteria[zone_id_to_use]["start_search"]
            
        final = xr.concat(out, dim=pd.Index(years, name="T"))
        final['T'] = pd.to_datetime([f"{y}-{start_search_str}" for y in years])
        final.name = "Cessation_dryspell"
        return final


class CEAC_count_dry_spells(CAF_AgroClimateBase):
    @staticmethod
    def count_dry_spells(x, onset, cessation, d_len, thresh):
        if not (np.isfinite(x).any() and np.isfinite(onset) and np.isfinite(cessation)): return np.nan
        o, c = int(onset), int(cessation)
        if o < 0 or c < 0 or o >= len(x): return np.nan
        c = min(c, len(x) - 1)
        count, cur = 0, 0
        for day in range(o, c + 1):
            if x[day] < thresh: cur += 1
            else:
                if cur == d_len: count += 1
                cur = 0
        if cur == d_len: count += 1
        return count

    def compute_insitu(self, daily_raw, on_cpt, cess_cpt, map_rec, d_len, thresh=1.0):
        df_long, zones, _ = self.transform_and_shift_cdt(daily_raw, map_rec)
        m = pd.merge(self._parse_cpt_to_long(on_cpt, "o"), self._parse_cpt_to_long(cess_cpt, "c"), on=["station", "year"], suffixes=('_o','_c'))
        res = []
        for (stn, y), group in df_long.groupby(["STATION", "year"]):
            sub = m[(m["station"] == stn) & (m["year"] == y)]
            z = zones[stn]
            if pd.isna(z) or sub.empty: v = np.nan
            else:
                z = int(z)
                o_idx = self.revert_to_index(sub["o"].values[0], z)
                c_idx = self.revert_to_index(sub["c"].values[0], z)
                v = self.count_dry_spells(group["VALUE"].values, o_idx, c_idx, d_len, thresh)
                
            lat_val = sub["lat_o"].values[0] if not sub.empty else group["LAT"].iloc[0]
            lon_val = sub["lon_o"].values[0] if not sub.empty else group["LON"].iloc[0]
            res.append({"year": y, "station": stn, "lat": lat_val, "lon": lon_val, "dry_spells": v})
        return self.build_cpt_output(pd.DataFrame(res), "dry_spells")

    def compute(self, daily_data, on_da, cess_da, map_rec, d_len, thresh, nb_cores):
        shifted, mask, years = self.shift_gridded_data(daily_data, map_rec)
        on_rel, cess_rel = self.revert_grid_index(on_da.reindex_like(mask, method='nearest'), mask), self.revert_grid_index(cess_da.reindex_like(mask, method='nearest'), mask)
        cx, cy = int(np.round(len(shifted.X)/nb_cores)), int(np.round(len(shifted.Y)/nb_cores))
        out = []
        for y in years:
            yd, o_y, c_y = shifted.sel(T=str(y)), on_rel.sel(T=str(y)).squeeze(), cess_rel.sel(T=str(y)).squeeze()
            client = Client(n_workers=nb_cores, threads_per_worker=1)
            res = xr.apply_ufunc(
                self.count_dry_spells, yd.chunk({'Y':cy,'X':cx}), o_y.chunk({'Y':cy,'X':cx}), c_y.chunk({'Y':cy,'X':cx}),
                input_core_dims=[('T',),(),()], vectorize=True, kwargs={'d_len': d_len, 'thresh': thresh}, dask='parallelized', output_dtypes=['float']
            ).compute()
            client.close()
            out.append(res)
        unique_zone = np.unique(mask.to_numpy())
        unique_zone = unique_zone[~np.isnan(unique_zone)]
        # Choose a date to store results
        if unique_zone.size == 0:
            raise ValueError("No valid zones found in the mask.")
        else:
            # Use zone in low latitude
            zone_id_to_use = int(np.min(unique_zone))
        start_search_str = self.criteria[zone_id_to_use]["start_search"]
        final = xr.concat(out, dim=pd.Index(years, name="T"))
        final['T'] = pd.to_datetime([f"{y}-{start_search_str}" for y in years])
        final.name = "Count_dryspell"
        return final


class CEAC_count_wet_spells(CAF_AgroClimateBase):
    @staticmethod
    def count_wet_spells(x, onset, cessation, w_len, thresh):
        if not (np.isfinite(x).any() and np.isfinite(onset) and np.isfinite(cessation)): return np.nan
        o, c = int(onset), int(cessation)
        if o < 0 or c < 0 or o >= len(x): return np.nan
        c = min(c, len(x) - 1)
        count, cur = 0, 0
        for day in range(o, c + 1):
            if x[day] >= thresh: cur += 1
            else:
                if cur == w_len: count += 1
                cur = 0
        if cur == w_len: count += 1
        return count

    def compute_insitu(self, daily_raw, on_cpt, cess_cpt, map_rec, w_len, thresh=1.0):
        df_long, zones, _ = self.transform_and_shift_cdt(daily_raw, map_rec)
        m = pd.merge(self._parse_cpt_to_long(on_cpt, "o"), self._parse_cpt_to_long(cess_cpt, "c"), on=["station", "year"], suffixes=('_o','_c'))
        res = []
        for (stn, y), group in df_long.groupby(["STATION", "year"]):
            sub = m[(m["station"] == stn) & (m["year"] == y)]
            z = zones[stn]
            if pd.isna(z) or sub.empty: v = np.nan
            else:
                z = int(z)
                o_idx = self.revert_to_index(sub["o"].values[0], z)
                c_idx = self.revert_to_index(sub["c"].values[0], z)
                v = self.count_wet_spells(group["VALUE"].values, o_idx, c_idx, w_len, thresh)
            
            lat_val = sub["lat_o"].values[0] if not sub.empty else group["LAT"].iloc[0]
            lon_val = sub["lon_o"].values[0] if not sub.empty else group["LON"].iloc[0]
            res.append({"year": y, "station": stn, "lat": lat_val, "lon": lon_val, "wet_spells": v})
        return self.build_cpt_output(pd.DataFrame(res), "wet_spells")

    def compute(self, daily_data, on_da, cess_da, map_rec, w_len, thresh, nb_cores):
        shifted, mask, years = self.shift_gridded_data(daily_data, map_rec)
        on_rel, cess_rel = self.revert_grid_index(on_da.reindex_like(mask, method='nearest'), mask), self.revert_grid_index(cess_da.reindex_like(mask, method='nearest'), mask)
        cx, cy = int(np.round(len(shifted.X)/nb_cores)), int(np.round(len(shifted.Y)/nb_cores))
        out = []
        for y in years:
            yd, o_y, c_y = shifted.sel(T=str(y)), on_rel.sel(T=str(y)).squeeze(), cess_rel.sel(T=str(y)).squeeze()
            client = Client(n_workers=nb_cores, threads_per_worker=1)
            res = xr.apply_ufunc(
                self.count_wet_spells, yd.chunk({'Y':cy,'X':cx}), o_y.chunk({'Y':cy,'X':cx}), c_y.chunk({'Y':cy,'X':cx}),
                input_core_dims=[('T',),(),()], vectorize=True, kwargs={'w_len': w_len, 'thresh': thresh}, dask='parallelized', output_dtypes=['float']
            ).compute()
            client.close()
            out.append(res)
        unique_zone = np.unique(mask.to_numpy())
        unique_zone = unique_zone[~np.isnan(unique_zone)]
        # Choose a date to store results
        if unique_zone.size == 0:
            raise ValueError("No valid zones found in the mask.")
        else:
            # Use zone in low latitude
            zone_id_to_use = int(np.min(unique_zone))
        start_search_str = self.criteria[zone_id_to_use]["start_search"]
        final = xr.concat(out, dim=pd.Index(years, name="T"))
        final['T'] = pd.to_datetime([f"{y}-{start_search_str}" for y in years])
        final.name = "Count_wetspell"
        return final







# import pandas as pd
# import xarray as xr
# import numpy as np
# import random
# import datetime
# from dask.distributed import Client
# import warnings

# warnings.filterwarnings('ignore')

# DEFAULT_CRITERIA = {
#     1: {"start_search": "05-01", "start_search2": "09-01", "date_dry_soil": "01-01", "cumulative": 15, "number_dry_days": 15, "thrd_rain_day": 0.85, "end_search": "08-30",  "end_search2": "10-30", "nbjour": 35, "ETP": 5.0, "Cap_ret_maxi": 70},
#     2: {"start_search": "03-15", "start_search2": "09-01", "date_dry_soil": "01-01", "cumulative": 20, "number_dry_days": 10, "thrd_rain_day": 0.85, "end_search": "08-01",  "end_search2": "11-01", "nbjour": 40, "ETP": 5.0, "Cap_ret_maxi": 70},
#     3: {"start_search": "02-01", "start_search2": "10-01", "date_dry_soil": "01-01", "cumulative": 20, "number_dry_days": 10, "thrd_rain_day": 0.85, "end_search": "05-15", "end_search2": "12-30", "nbjour": 45, "ETP": 5.0, "Cap_ret_maxi": 70},
#     4: {"start_search": "01-01", "start_search2": "11-01", "date_dry_soil": "01-01", "cumulative": 20, "number_dry_days": 7, "thrd_rain_day": 0.85, "end_search": "04-01", "end_search2": "12-30", "nbjour": 50, "ETP": 5.0, "Cap_ret_maxi": 80},
#     5: {"start_search": "01-01", "start_search2": "06-01", "date_dry_soil": "01-01", "cumulative": 25, "number_dry_days": 7, "thrd_rain_day": 0.85, "end_search": "03-10", "end_search2": "08-10", "nbjour": 50, "ETP": 4.0, "Cap_ret_maxi": 60},
#     6: {"start_search": "02-01", "start_search2": "10-15", "date_dry_soil": "01-01", "cumulative": 20, "number_dry_days": 7, "thrd_rain_day": 0.85, "end_search": "03-20",  "end_search2": "12-15", "nbjour": 50, "ETP": 4.0, "Cap_ret_maxi": 70},
#     7: {"start_search": "02-01", "start_search2": "09-01", "date_dry_soil": "01-01", "cumulative": 20, "number_dry_days": 10, "thrd_rain_day": 0.85, "end_search": "03-20",  "end_search2": "12-15", "nbjour": 50, "ETP": 4.0, "Cap_ret_maxi": 70},
#     8: {"start_search": "03-01", "start_search2": "08-01", "date_dry_soil": "01-01", "cumulative": 20, "number_dry_days": 10, "thrd_rain_day": 0.85, "end_search": "04-20",  "end_search2": "10-15", "nbjour": 40, "ETP": 4.0, "Cap_ret_maxi": 70},
#     9: {"start_search": "05-01", "start_search2": "08-15", "date_dry_soil": "01-01", "cumulative": 20, "number_dry_days": 10, "thrd_rain_day": 0.85, "end_search": "06-20",  "end_search2": "10-15", "nbjour": 30, "ETP": 4.0, "Cap_ret_maxi": 70},
# }

# class CAF_AgroClimateBase:
#     def __init__(self, user_criteria=None):
#         self.criteria = user_criteria if user_criteria else DEFAULT_CRITERIA

#     @staticmethod
#     def adjust_duplicates(series, inc=0.00001):
#         counts = series.value_counts()
#         for val, count in counts[counts > 1].items():
#             for i, idx in enumerate(series[series == val].index):
#                 series.at[idx] += inc * i
#         return series

#     @staticmethod
#     def day_of_year(y, mm_dd):
#         dt = datetime.datetime.strptime(f"{int(y)}-{mm_dd}", "%Y-%m-%d").date()
#         return (dt - datetime.date(int(y), 1, 1)).days + 1

#     def _map_criteria(self, mask, key, year):
#         def _safe_get(z):
#             if np.isnan(z) or int(z) not in self.criteria: return np.nan
#             v = self.criteria[int(z)][key]
#             return self.day_of_year(year, v) if 'search' in key or 'date' in key else v
#         return xr.DataArray(np.vectorize(_safe_get, otypes=[float])(mask.values), coords=mask.coords)

#     def shift_gridded_data(self, daily_data, map_reclassified):
#         mask = map_reclassified.reindex_like(daily_data, method='nearest')
#         y1, y2 = int(daily_data['T'].dt.year.min()), int(daily_data['T'].dt.year.max())
#         le5 = daily_data.sel(T=slice(f"{y1}", f"{y2-1}")).where(mask <= 5)
#         gt6 = daily_data.sel(T=slice(f"{y1}-08-01", f"{y2}-07-31")).where(mask >= 6)
#         m_len = min(len(le5['T']), len(gt6['T']))
#         le5, gt6 = le5.isel(T=slice(0, m_len)), gt6.isel(T=slice(0, m_len))
#         gt6 = gt6.assign_coords(T=le5['T'].values)
#         return le5.combine_first(gt6), mask, np.unique(le5['T'].dt.year.to_numpy())

#     def restore_julian_grid(self, res_xr, mask):
#         fx = xr.where(mask >= 6, res_xr + 212, res_xr)
#         return fx #xr.where(fx > 365, fx - 365, fx)

#     def to_relative_grid(self, res_xr, mask):
#         fx = xr.where(mask >= 6, res_xr - 212, res_xr)
#         return xr.where(fx <= 0, fx + 365, fx)

#     def transform_and_shift_cdt(self, df_raw, map_reclassified):
#         s_ids = pd.Series(df_raw.iloc[0, 1:].values.astype(str))
#         ids = s_ids.where(~s_ids.duplicated(), s_ids + "_" + s_ids.groupby(s_ids).cumcount().astype(str)).values
#         df_raw.iloc[0, 1:] = ids 
        
#         lons, lats = df_raw.iloc[1, 1:].astype(float).values, df_raw.iloc[2, 1:].astype(float).values
#         dates = pd.to_datetime(df_raw.iloc[4:, 0], format='%Y%m%d')
#         da = xr.DataArray(df_raw.iloc[4:, 1:].astype(float).values, coords={'T': dates, 'station': ids}, dims=['T', 'station'])
#         y1, y2 = dates.min().year, dates.max().year
        
#         stn_zones, series_lst = {}, []
#         for i, stn in enumerate(ids):
#             try: z = map_reclassified.sel(X=lons[i], Y=lats[i], method='nearest').values.item()
#             except: z = np.nan
#             stn_zones[stn] = z
#             s = da.isel(station=i)
#             if pd.isna(z) or z <= 5:
#                 series_lst.append(s.sel(T=slice(f"{y1}-01-01", f"{y2-1}-12-31")))
#             else:
#                 series_lst.append(s.sel(T=slice(f"{y1}-08-01", f"{y2}-07-31")))
                
#         m_len = min(len(s) for s in series_lst)
#         std_dt_vals = dates.iloc[:m_len].values  
        
#         arr_2d = np.column_stack([s.values[:m_len] for s in series_lst])
#         df_shifted = pd.DataFrame(arr_2d, index=std_dt_vals, columns=ids)
        
#         df_long = df_shifted.reset_index().melt(id_vars="index", var_name="STATION", value_name="VALUE")
#         df_long.rename(columns={"index": "DATE"}, inplace=True)
        
#         meta_df = pd.DataFrame({"STATION": ids, "LON": lons, "LAT": lats, "zonename": [stn_zones[s] for s in ids]})
#         df_long = df_long.merge(meta_df, on="STATION")
#         df_long["VALUE"] = df_long["VALUE"].replace(-99.0, np.nan)
#         df_long["year"] = df_long["DATE"].dt.year
        
#         return df_long, stn_zones, df_raw.iloc[:4, :]

#     def _parse_cpt_to_long(self, df_cpt, val_name):
#         lats, lons = df_cpt.iloc[0, 1:].values, df_cpt.iloc[1, 1:].values
#         cols = df_cpt.columns[1:].tolist()
#         df = df_cpt.iloc[2:].copy().reset_index(drop=True).rename(columns={"STATION": "year"})
#         df = df.melt(id_vars=["year"], var_name="station", value_name=val_name)
#         df["year"] = pd.to_numeric(df["year"], errors="coerce")
#         df["lat"], df["lon"] = df["station"].map(dict(zip(cols, lats))), df["station"].map(dict(zip(cols, lons)))
#         return df

#     def build_cpt_output(self, res_df, val_col):
#         res_df[val_col] = res_df[val_col].fillna(-999)
#         meta = res_df.groupby("station")[["lat", "lon"]].first().reset_index()
#         piv = res_df.pivot(index="year", columns="station", values=val_col).reset_index().rename(columns={"year": "STATION"})
#         lat_r = pd.DataFrame([["LAT"] + meta.set_index("station").loc[piv.columns[1:], "lat"].tolist()], columns=piv.columns)
#         lon_r = pd.DataFrame([["LON"] + meta.set_index("station").loc[piv.columns[1:], "lon"].tolist()], columns=piv.columns)
#         return pd.concat([lat_r, lon_r, piv], ignore_index=True)


# class CEAC_compute_onset(CAF_AgroClimateBase):
#     @staticmethod
#     def onset_function(x, idebut, cumul, nbsec, jour_pluvieux, irch_fin):
#         if not (np.any(np.isfinite(x)) and np.isfinite(idebut) and np.isfinite(nbsec) and np.isfinite(irch_fin)): return np.nan
#         idebut, nbsec, irch_fin = int(idebut), int(nbsec), int(irch_fin)
#         idate, trouv = idebut, 0
#         while True:
#             idate += 1
#             if idate >= len(x)-1 or pd.isna(x[idate-1]) or pd.isna(x[idate]) or pd.isna(x[idate+1]): return np.nan
#             if idate > irch_fin: return random.randint(irch_fin - 5, irch_fin)
#             c1, c2, c3 = x[idate-1], x[idate-1]+x[idate], x[idate-1]+x[idate]+x[idate+1]
#             if c1 >= cumul or c2 >= cumul or c3 >= cumul:
#                 arr = np.array([x[idate-1], x[idate], x[idate+1]])
#                 ideb = [idate-1, idate, idate+1][np.argmax(arr)]
#                 trouv = 1
#                 pluie30 = x[ideb:ideb+31] if ideb+30 < len(x) else x[ideb:]
#                 isec = 0
#                 while True:
#                     isec += 1
#                     if isec+nbsec >= len(pluie30): break
#                     if np.sum(pluie30[isec:isec+nbsec+1] < jour_pluvieux) == (nbsec + 1):
#                         trouv = 0; break
#                     if isec == (30 - nbsec): break
#             if trouv == 1: return ideb
#         return np.nan

#     def compute_insitu(self, daily_df_raw, map_rec):
#         df_long, zones, header = self.transform_and_shift_cdt(daily_df_raw, map_rec)
#         res = []
#         for (stn, y), group in df_long.groupby(["STATION", "year"]):
#             z = zones[stn]
#             if pd.isna(z) or int(z) not in self.criteria: 
#                 v = np.nan
#             else:
#                 z = int(z)
#                 c = self.criteria[z]
#                 v = self.onset_function(group["VALUE"].values, self.day_of_year(y, c["start_search"]), c["cumulative"], c["number_dry_days"], c["thrd_rain_day"], self.day_of_year(y, c["end_search"]))
#                 if z >= 6 and not np.isnan(v): v = v + 212
#             res.append({"year": y, "station": stn, "lat": group["LAT"].iloc[0], "lon": group["LON"].iloc[0], "onset": v})
#         return self.build_cpt_output(pd.DataFrame(res), "onset")

#     def compute(self, daily_data, map_rec, nb_cores):
#         shifted, mask, years = self.shift_gridded_data(daily_data, map_rec)
#         cx, cy = int(np.round(len(shifted.X)/nb_cores)), int(np.round(len(shifted.Y)/nb_cores))
#         out = []
#         for y in years:
#             yd, mk = shifted.sel(T=str(y)), lambda k: self._map_criteria(mask, k, y)
#             client = Client(n_workers=nb_cores, threads_per_worker=1)
#             res = xr.apply_ufunc(
#                 self.onset_function, yd.chunk({'Y': cy, 'X': cx}),
#                 mk("start_search").chunk({'Y': cy, 'X': cx}), mk("cumulative").chunk({'Y': cy, 'X': cx}),
#                 mk("number_dry_days").chunk({'Y': cy, 'X': cx}), mk("thrd_rain_day").chunk({'Y': cy, 'X': cx}),
#                 mk("end_search").chunk({'Y': cy, 'X': cx}),
#                 input_core_dims=[('T',)]+[()]*5, vectorize=True, dask='parallelized', output_dtypes=['float']
#             ).compute()
#             client.close()
#             out.append(res)
#         final = self.restore_julian_grid(xr.concat(out, dim=pd.Index(years, name="T")), mask)
#         final.name = "Onset"
#         return final


# class CEAC_compute_cessation(CAF_AgroClimateBase):
#     @staticmethod
#     def cessation_function(x, ijour, idebut, ETP, Cap, irch_fin):
#         if not (np.isfinite(x).any() and np.isfinite(idebut) and np.isfinite(ijour) and np.isfinite(ETP) and np.isfinite(Cap) and np.isfinite(irch_fin)): return np.nan
#         ru, ifin = 0, int(idebut)
#         for k in range(int(ijour), ifin + 1):
#             if not pd.isna(x[k]): ru = max(0, min(ru + x[k] - ETP, Cap))
#         while ifin < int(irch_fin):
#             ifin += 1
#             if ifin >= len(x) or pd.isna(x[ifin]): continue
#             ru = max(0, min(ru + x[ifin] - ETP, Cap))
#             if ru <= 0: break
#         return ifin if ifin <= int(irch_fin) else random.randint(int(irch_fin) - 5, int(irch_fin))

#     def compute_insitu(self, daily_df_raw, map_rec):
#         df_long, zones, header = self.transform_and_shift_cdt(daily_df_raw, map_rec)
#         res = []
#         for (stn, y), group in df_long.groupby(["STATION", "year"]):
#             z = zones[stn]
#             if pd.isna(z) or int(z) not in self.criteria: 
#                 v = np.nan
#             else:
#                 z = int(z)
#                 c = self.criteria[z]
#                 v = self.cessation_function(group["VALUE"].values, self.day_of_year(y, c["date_dry_soil"]), self.day_of_year(y, c["start_search2"]), c["ETP"], c["Cap_ret_maxi"], self.day_of_year(y, c["end_search2"]))
#                 if z >= 6 and not np.isnan(v): v = v + 212
#             res.append({"year": y, "station": stn, "lat": group["LAT"].iloc[0], "lon": group["LON"].iloc[0], "cessation": v})
#         return self.build_cpt_output(pd.DataFrame(res), "cessation")

#     def compute(self, daily_data, map_rec, nb_cores):
#         shifted, mask, years = self.shift_gridded_data(daily_data, map_rec)
#         cx, cy = int(np.round(len(shifted.X)/nb_cores)), int(np.round(len(shifted.Y)/nb_cores))
#         out = []
#         for y in years:
#             yd, mk = shifted.sel(T=str(y)), lambda k: self._map_criteria(mask, k, y)
#             client = Client(n_workers=nb_cores, threads_per_worker=1)
#             res = xr.apply_ufunc(
#                 self.cessation_function, yd.chunk({'Y': cy, 'X': cx}),
#                 mk("date_dry_soil").chunk({'Y': cy, 'X': cx}), mk("start_search2").chunk({'Y': cy, 'X': cx}),
#                 mk("ETP").chunk({'Y': cy, 'X': cx}), mk("Cap_ret_maxi").chunk({'Y': cy, 'X': cx}),
#                 mk("end_search2").chunk({'Y': cy, 'X': cx}),
#                 input_core_dims=[('T',)]+[()]*5, vectorize=True, dask='parallelized', output_dtypes=['float']
#             ).compute()
#             client.close()
#             out.append(res)
#         final = self.restore_julian_grid(xr.concat(out, dim=pd.Index(years, name="T")), mask)
#         final.name = "Cessation"
#         return final


# class CEAC_compute_onset_dry_spell(CAF_AgroClimateBase):
#     @staticmethod
#     def ds_onset_func(x, idebut, cumul, nbsec, jp, irch_fin, nbjour):
#         if not (np.any(np.isfinite(x)) and np.isfinite(idebut)): return np.nan
#         deb = CEAC_compute_onset.onset_function(x, idebut, cumul, nbsec, jp, irch_fin)
#         if not np.isnan(deb):
#             p = x[int(deb) : min(int(deb) + int(nbjour) + 1, len(x))]
#             r = np.where(p > jp)[0]
#             d1, d2 = np.array([0] + list(r)), np.array(list(r) + [len(p)])
#             return np.max(d2 - d1) - 1
#         return np.nan

#     def compute_insitu(self, daily_df_raw, map_rec):
#         df_long, zones, _ = self.transform_and_shift_cdt(daily_df_raw, map_rec)
#         res = []
#         for (stn, y), group in df_long.groupby(["STATION", "year"]):
#             z = zones[stn]
#             if pd.isna(z) or int(z) not in self.criteria: 
#                 v = np.nan
#             else:
#                 z = int(z)
#                 c = self.criteria[z]
#                 v = self.ds_onset_func(group["VALUE"].values, self.day_of_year(y, c["start_search"]), c["cumulative"], c["number_dry_days"], c["thrd_rain_day"], self.day_of_year(y, c["end_search"]), c["nbjour"])
#             res.append({"year": y, "station": stn, "lat": group["LAT"].iloc[0], "lon": group["LON"].iloc[0], "onsetdryspell": v})
#         return self.build_cpt_output(pd.DataFrame(res), "onsetdryspell")

#     def compute(self, daily_data, map_rec, nb_cores):
#         shifted, mask, years = self.shift_gridded_data(daily_data, map_rec)
#         cx, cy = int(np.round(len(shifted.X)/nb_cores)), int(np.round(len(shifted.Y)/nb_cores))
#         out = []
#         for y in years:
#             yd, mk = shifted.sel(T=str(y)), lambda k: self._map_criteria(mask, k, y)
#             client = Client(n_workers=nb_cores, threads_per_worker=1)
#             res = xr.apply_ufunc(
#                 self.ds_onset_func, yd.chunk({'Y':cy,'X':cx}),
#                 mk("start_search").chunk({'Y':cy,'X':cx}), mk("cumulative").chunk({'Y':cy,'X':cx}),
#                 mk("number_dry_days").chunk({'Y':cy,'X':cx}), mk("thrd_rain_day").chunk({'Y':cy,'X':cx}),
#                 mk("end_search").chunk({'Y':cy,'X':cx}), mk("nbjour").chunk({'Y':cy,'X':cx}),
#                 input_core_dims=[('T',)]+[()]*6, vectorize=True, dask='parallelized', output_dtypes=['float']
#             ).compute()
#             client.close()
#             out.append(res)
#         final = xr.concat(out, dim=pd.Index(years, name="T"))
#         final.name = "Onset_dryspell"
#         return final


# class CEAC_compute_cessation_dry_spell(CAF_AgroClimateBase):
#     @staticmethod
#     def ds_cess_func(x, id1, cum, nbs, jp, ir1, id2, ijd, ETP, Cap, ir2, nbj):
#         if not (np.any(np.isfinite(x)) and np.isfinite(id1)): return np.nan
#         deb = CEAC_compute_onset.onset_function(x, id1, cum, nbs, jp, ir1)
#         if pd.isna(deb): return np.nan
#         fin = CEAC_compute_cessation.cessation_function(x, ijd, id2, ETP, Cap, ir2)
#         if not np.isnan(fin) and (fin - (deb + nbj)) > 0 and (deb + nbj) < len(x):
#             p = x[int(deb + nbj):int(fin)]
#             r = np.where(p > jp)[0]
#             if len(r) == 0: return np.nan
#             return np.max(np.array(list(r) + [len(p)]) - np.array([0] + list(r))) - 1
#         return np.nan

#     def compute_insitu(self, daily_df_raw, map_rec):
#         df_long, zones, _ = self.transform_and_shift_cdt(daily_df_raw, map_rec)
#         res = []
#         for (stn, y), group in df_long.groupby(["STATION", "year"]):
#             z = zones[stn]
#             if pd.isna(z) or int(z) not in self.criteria: 
#                 v = np.nan
#             else:
#                 z = int(z)
#                 c = self.criteria[z]
#                 v = self.ds_cess_func(group["VALUE"].values, self.day_of_year(y, c["start_search"]), c["cumulative"], c["number_dry_days"], c["thrd_rain_day"], self.day_of_year(y, c["end_search"]), self.day_of_year(y, c["start_search2"]), self.day_of_year(y, c["date_dry_soil"]), c["ETP"], c["Cap_ret_maxi"], self.day_of_year(y, c["end_search2"]), c["nbjour"])
#             res.append({"year": y, "station": stn, "lat": group["LAT"].iloc[0], "lon": group["LON"].iloc[0], "cessation_dryspell": v})
#         return self.build_cpt_output(pd.DataFrame(res), "cessation_dryspell")

#     def compute(self, daily_data, map_rec, nb_cores):
#         shifted, mask, years = self.shift_gridded_data(daily_data, map_rec)
#         cx, cy = int(np.round(len(shifted.X)/nb_cores)), int(np.round(len(shifted.Y)/nb_cores))
#         out = []
#         for y in years:
#             yd, mk = shifted.sel(T=str(y)), lambda k: self._map_criteria(mask, k, y)
#             client = Client(n_workers=nb_cores, threads_per_worker=1)
#             res = xr.apply_ufunc(
#                 self.ds_cess_func, yd.chunk({'Y':cy,'X':cx}),
#                 mk("start_search").chunk({'Y':cy,'X':cx}), mk("cumulative").chunk({'Y':cy,'X':cx}),
#                 mk("number_dry_days").chunk({'Y':cy,'X':cx}), mk("thrd_rain_day").chunk({'Y':cy,'X':cx}),
#                 mk("end_search").chunk({'Y':cy,'X':cx}), mk("start_search2").chunk({'Y':cy,'X':cx}),
#                 mk("date_dry_soil").chunk({'Y':cy,'X':cx}), mk("ETP").chunk({'Y':cy,'X':cx}),
#                 mk("Cap_ret_maxi").chunk({'Y':cy,'X':cx}), mk("end_search2").chunk({'Y':cy,'X':cx}),
#                 mk("nbjour").chunk({'Y':cy,'X':cx}),
#                 input_core_dims=[('T',)]+[()]*11, vectorize=True, dask='parallelized', output_dtypes=['float']
#             ).compute()
#             client.close()
#             out.append(res)
#         final = xr.concat(out, dim=pd.Index(years, name="T"))
#         final.name = "Cessation_dryspell"
#         return final


# class CEAC_count_dry_spells(CAF_AgroClimateBase):
#     @staticmethod
#     def count_dry_spells(x, onset, cessation, d_len, thresh):
#         if not (np.isfinite(x).any() and np.isfinite(onset) and np.isfinite(cessation)): return np.nan
#         o, c = int(onset), int(cessation)
#         if o < 0 or c < 0 or o >= len(x): return np.nan
#         c = min(c, len(x) - 1)
#         count, cur = 0, 0
#         for day in range(o, c + 1):
#             if x[day] < thresh: cur += 1
#             else:
#                 if cur == d_len: count += 1
#                 cur = 0
#         if cur == d_len: count += 1
#         return count

#     def compute_insitu(self, daily_raw, on_cpt, cess_cpt, map_rec, d_len, thresh=1.0):
#         df_long, zones, _ = self.transform_and_shift_cdt(daily_raw, map_rec)
#         m = pd.merge(self._parse_cpt_to_long(on_cpt, "o"), self._parse_cpt_to_long(cess_cpt, "c"), on=["station", "year"], suffixes=('_o','_c'))
#         res = []
#         for (stn, y), group in df_long.groupby(["STATION", "year"]):
#             sub = m[(m["station"] == stn) & (m["year"] == y)]
#             z = zones[stn]
#             if pd.isna(z) or sub.empty: 
#                 v = np.nan
#             else:
#                 o, c, z = sub["o"].values[0], sub["c"].values[0], int(z)
#                 if z >= 6:
#                     if not np.isnan(o): o = o - 212 + 365 if (o - 212) <= 0 else o - 212
#                     if not np.isnan(c): c = c - 212 + 365 if (c - 212) <= 0 else c - 212
#                 v = self.count_dry_spells(group["VALUE"].values, o, c, d_len, thresh)
                
#             lat_val = group["LAT"].iloc[0]
#             lon_val = group["LON"].iloc[0]
#             res.append({"year": y, "station": stn, "lat": lat_val, "lon": lon_val, "dry_spells": v})
#         return self.build_cpt_output(pd.DataFrame(res), "dry_spells")

#     def compute(self, daily_data, on_da, cess_da, map_rec, d_len, thresh, nb_cores):
#         shifted, mask, years = self.shift_gridded_data(daily_data, map_rec)
#         on_rel, cess_rel = self.to_relative_grid(on_da.reindex_like(mask, method='nearest'), mask), self.to_relative_grid(cess_da.reindex_like(mask, method='nearest'), mask)
#         cx, cy = int(np.round(len(shifted.X)/nb_cores)), int(np.round(len(shifted.Y)/nb_cores))
#         out = []
#         for y in years:
#             yd, o_y, c_y = shifted.sel(T=str(y)), on_rel.sel(T=str(y)).squeeze(), cess_rel.sel(T=str(y)).squeeze()
#             client = Client(n_workers=nb_cores, threads_per_worker=1)
#             res = xr.apply_ufunc(
#                 self.count_dry_spells, yd.chunk({'Y':cy,'X':cx}), o_y.chunk({'Y':cy,'X':cx}), c_y.chunk({'Y':cy,'X':cx}),
#                 input_core_dims=[('T',),(),()], vectorize=True, kwargs={'d_len': d_len, 'thresh': thresh}, dask='parallelized', output_dtypes=['float']
#             ).compute()
#             client.close()
#             out.append(res)
#         final = xr.concat(out, dim=pd.Index(years, name="T"))
#         final.name = "Count_dryspell"
#         return final


# class CEAC_count_wet_spells(CAF_AgroClimateBase):
#     @staticmethod
#     def count_wet_spells(x, onset, cessation, w_len, thresh):
#         if not (np.isfinite(x).any() and np.isfinite(onset) and np.isfinite(cessation)): return np.nan
#         o, c = int(onset), int(cessation)
#         if o < 0 or c < 0 or o >= len(x): return np.nan
#         c = min(c, len(x) - 1)
#         count, cur = 0, 0
#         for day in range(o, c + 1):
#             if x[day] >= thresh: cur += 1
#             else:
#                 if cur == w_len: count += 1
#                 cur = 0
#         if cur == w_len: count += 1
#         return count

#     def compute_insitu(self, daily_raw, on_cpt, cess_cpt, map_rec, w_len, thresh=1.0):
#         df_long, zones, _ = self.transform_and_shift_cdt(daily_raw, map_rec)
#         m = pd.merge(self._parse_cpt_to_long(on_cpt, "o"), self._parse_cpt_to_long(cess_cpt, "c"), on=["station", "year"], suffixes=('_o','_c'))
#         res = []
#         for (stn, y), group in df_long.groupby(["STATION", "year"]):
#             sub = m[(m["station"] == stn) & (m["year"] == y)]
#             z = zones[stn]
#             if pd.isna(z) or sub.empty: 
#                 v = np.nan
#             else:
#                 o, c, z = sub["o"].values[0], sub["c"].values[0], int(z)
#                 if z >= 6:
#                     if not np.isnan(o): o = o - 212 + 365 if (o - 212) <= 0 else o - 212
#                     if not np.isnan(c): c = c - 212 + 365 if (c - 212) <= 0 else c - 212
#                 v = self.count_wet_spells(group["VALUE"].values, o, c, w_len, thresh)
            
#             lat_val = group["LAT"].iloc[0]
#             lon_val = group["LON"].iloc[0]
#             res.append({"year": y, "station": stn, "lat": lat_val, "lon": lon_val, "wet_spells": v})
#         return self.build_cpt_output(pd.DataFrame(res), "wet_spells")

#     def compute(self, daily_data, on_da, cess_da, map_rec, w_len, thresh, nb_cores):
#         shifted, mask, years = self.shift_gridded_data(daily_data, map_rec)
#         on_rel, cess_rel = self.to_relative_grid(on_da.reindex_like(mask, method='nearest'), mask), self.to_relative_grid(cess_da.reindex_like(mask, method='nearest'), mask)
#         cx, cy = int(np.round(len(shifted.X)/nb_cores)), int(np.round(len(shifted.Y)/nb_cores))
#         out = []
#         for y in years:
#             yd, o_y, c_y = shifted.sel(T=str(y)), on_rel.sel(T=str(y)).squeeze(), cess_rel.sel(T=str(y)).squeeze()
#             client = Client(n_workers=nb_cores, threads_per_worker=1)
#             res = xr.apply_ufunc(
#                 self.count_wet_spells, yd.chunk({'Y':cy,'X':cx}), o_y.chunk({'Y':cy,'X':cx}), c_y.chunk({'Y':cy,'X':cx}),
#                 input_core_dims=[('T',),(),()], vectorize=True, kwargs={'w_len': w_len, 'thresh': thresh}, dask='parallelized', output_dtypes=['float']
#             ).compute()
#             client.close()
#             out.append(res)
#         final = xr.concat(out, dim=pd.Index(years, name="T"))
#         final.name = "Count_wetspell"
#         return final





