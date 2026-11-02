from pathlib import Path
import pandas as pd
import logging
import shutil
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import re
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

from datetime import datetime
from pandas.tseries.offsets import Day
from shapely.geometry import box

import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



class WAS_Hydro:
    def adjust_duplicates(self, series: pd.Series, increment: float = 0.00001) -> pd.Series:
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

    def subbasin_data_to_xarray(self, df: pd.DataFrame) -> xr.DataArray:
        # --- 1) Extract metadata (first 3 rows: LON, LAT, ELEV) ---
        metadata = df.iloc[:3].set_index("ID").T.reset_index()
        metadata.columns = ["SUBID", "LONG", "LAT", "ELEV"]
       
        # Adjust duplicates
        metadata["LONG"] = self.adjust_duplicates(metadata["LONG"])
        metadata["LAT"] = self.adjust_duplicates(metadata["LAT"])
       
        # --- 2) Extract actual data from row 3 onward, rename ID -> DATE ---
        data_part = df.iloc[3:].rename(columns={"ID": "DATE"})
       
        # Melt to long form
        data_long = data_part.melt(id_vars=["DATE"], var_name="SUBID", value_name="VALUE")
       
        # --- 3) Merge with metadata to attach (LON, LAT, ELEV) ---
        final_df = pd.merge(data_long, metadata, on="SUBID", how="left")
       
        # Ensure 'DATE' is a proper datetime (assuming "YYYYmmdd" format)
        final_df["DATE"] = pd.to_datetime(final_df["DATE"], format="%Y%m%d", errors="coerce")
       
        # --- 4) Convert to xarray, rename coords ---
        rainfall_data_array = (
            final_df[["DATE", "LAT", "LONG", "SUBID", "VALUE"]]
            .set_index(["DATE", "LAT", "LONG"])
            .to_xarray()
            .rename({"VALUE": "Observation", "LAT": "Y", "LONG": "X", "DATE": "T"})
        )
       
        return rainfall_data_array

    def clip_bbox(self, extent_obs, geo_df):
        
        # Your extent: [North, West, South, East]
        north, west, south, east = extent_obs
        minx, miny, maxx, maxy = west, south, east, north  # reorder
        
        # Ensure CRS is geographic (adjust if your data uses another CRS)
        gdf = geo_df.copy()
        if gdf.crs is None:
            gdf = gdf.set_crs(4326)     # set if you know it's WGS84
        else:
            gdf = gdf.to_crs(4326)      # or to the CRS of the bbox
            
        # Build bbox polygon and clip
        bbox_poly = box(minx, miny, maxx, maxy)
        bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_poly], crs=4326)
        
        geo_df_centroid_in = gdf[gdf.geometry.centroid.within(bbox_poly)]
        return geo_df_centroid_in

    def plot_country_subbasins(self, geo_centroid_zone, name="Country Subbasins"):
        
        gdf = geo_centroid_zone.copy()
        if gdf.crs is None:
            gdf = gdf.set_crs(4326)
        else:
            gdf = gdf.to_crs(4326)
        
        # Build the Matplotlib axis with a PlateCarree (lon/lat) projection
        proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=(9, 7))
        ax = plt.axes(projection=proj)
        
        # Add context features (low-cost)
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f8f8f8")
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#e8f4fa")
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5, edgecolor="#666666")
        
        ax.add_feature(cfeature.RIVERS.with_scale("50m"), linewidth=1.0, edgecolor="#0077b6")
        
        # Plot polygons
        gdf.plot(
            ax=ax,
            transform=proj,         # data are in lon/lat
            facecolor="#eaf6fb",
            edgecolor="#0a6aa6",
            linewidth=0.5,
            alpha=0.9
        )
                
        # Set map extent to the clipped layer bounds (with a small padding)
        xmin, ymin, xmax, ymax = gdf.total_bounds
        pad_x = (xmax - xmin) * 0.05 if xmax > xmin else 0.5
        pad_y = (ymax - ymin) * 0.05 if ymax > ymin else 0.5
        ax.set_extent([xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y], crs=proj)
        
        # Gridlines
        gl = ax.gridlines(
            crs=proj, draw_labels=True, linewidth=0.3, color="gray", alpha=0.5, linestyle=":"
        )
        gl.right_labels = False
        gl.top_labels = False
        
        ax.set_title(f"{name}")
        plt.tight_layout()
        plt.show()

    def wide_to_long(self, df):
        """
        Convert a wide HYPE output (DATE, <SUBID1>, <SUBID2>, ...) to long (DATE, SUBID, QObs).
    
        - Keeps only numeric-looking SUBID columns.
        - Ensures DATE is datetime.
        - SUBID returned as Int64 (nullable integer); QObs as float.
        """
        out = df.copy()
    
        # Ensure DATE is datetime
        if "DATE" not in out.columns:
            raise KeyError("Expected a 'DATE' column in the wide table.")
        if not np.issubdtype(out["DATE"].dtype, np.datetime64):
            out["DATE"] = pd.to_datetime(out["DATE"], errors="coerce")
    
        # Select only columns that look like numeric SUBIDs
        subid_cols = [c for c in out.columns if c != "DATE" and re.fullmatch(r"\d+", str(c))]
    
        # Melt to long
        long_df = out.melt(
            id_vars="DATE",
            value_vars=subid_cols,
            var_name="SUBID",
            value_name="QObs"
        )
    
        # Types
        long_df["SUBID"] = pd.to_numeric(long_df["SUBID"], errors="coerce").astype("Int64")
        long_df["QObs"] = pd.to_numeric(long_df["QObs"], errors="coerce")
    
        # Optional: sort
        long_df = long_df.sort_values(["DATE", "SUBID"]).reset_index(drop=True)
        return long_df

    def long_to_wide(self, df):
        """
        Convert a long HYPE output (DATE, SUBID, QObs) to wide (DATE, <SUBID1>, <SUBID2>, ...).
        
        - Expects columns DATE (datetime or convertible), SUBID (numeric), and QObs (numeric).
        - Pivots SUBID and QObs into columns named by SUBID values.
        - Ensures DATE is datetime in the output.
        """
        out = df.copy()
        
        # Validate required columns
        required_cols = ["DATE", "SUBID", "QObs"]
        if not all(col in out.columns for col in required_cols):
            raise KeyError("Expected 'DATE', 'SUBID', and 'QObs' columns in the long table.")
        
        # Ensure DATE is datetime
        if not np.issubdtype(out["DATE"].dtype, np.datetime64):
            out["DATE"] = pd.to_datetime(out["DATE"], errors="coerce")
        
        # Ensure SUBID is numeric (Int64 to handle nullable integers)
        out["SUBID"] = pd.to_numeric(out["SUBID"], errors="coerce").astype("Int64")
        
        # Ensure QObs is numeric
        out["QObs"] = pd.to_numeric(out["QObs"], errors="coerce")
        
        # Pivot to wide format
        wide_df = out.pivot(
            index="DATE",
            columns="SUBID",
            values="QObs"
        ).reset_index()
        
        # Rename columns to string SUBIDs (removing the 'SUBID' prefix from pivot)
        wide_df.columns = ["DATE"] + [str(col) for col in wide_df.columns[1:]]
        
        # Sort by DATE
        wide_df = wide_df.sort_values("DATE").reset_index(drop=True)
        
        return wide_df

    def write_models_nc(
        self,
        geo_df,
        extent,
        resample_type,
        agg,
        season_obs,
        center_variable,
        dir_to_save_model,
        year_start,
        year_end,
        month_day_start="-08-01",
        month_day_end="-12-01",
        hindcast=True,
        skip_existing=True,
        verbose=True
    ):
        out_files = {}
        dir_to_save_model = Path(dir_to_save_model)
        dir_to_save_model.mkdir(parents=True, exist_ok=True)
        prefix = "hindcast" if hindcast else "forecast"
        for var in center_variable:
            varname = var.replace("_", "").lower()
            csv_path = dir_to_save_model / f"{prefix}_{varname}.csv"
            nc_path  = dir_to_save_model / f"{prefix}_{varname}.nc"
            if not csv_path.exists():
                if nc_path.exists():
                    out_files[varname] = str(nc_path)
                    if verbose:
                        print(f"[WARN] Missing CSV: {csv_path.name}. Using existing NC: {nc_path.name}")
                else:
                    if verbose:
                        print(f"[WARN] Both CSV and NC are missing for {varname} ({csv_path.name}, {nc_path.name})")
                continue
            if skip_existing and nc_path.exists() and nc_path.stat().st_mtime >= csv_path.stat().st_mtime:
                out_files[varname] = str(nc_path)
                if verbose:
                    print(f"[SKIP] {nc_path.name} is already up-to-date (>= {csv_path.name}).")
                continue
            try:
                try:
                    df = pd.read_csv(csv_path, skiprows=[1, 2, 3])
                except Exception:
                    df = pd.read_csv(csv_path)
                if "ID" in df.columns and "DATE" not in df.columns:
                    df = df.rename(columns={"ID": "DATE"})
                if "DATE" not in df.columns:
                    raise ValueError(f"'DATE' not found after normalization ({csv_path.name}).")
                df["DATE"] = pd.to_datetime(df["DATE"].astype(str), errors="coerce")
                if df["DATE"].isna().all():
                    df["DATE"] = pd.to_datetime(df["DATE"].astype(str), format="%Y%m%d", errors="coerce")
                if df["DATE"].isna().all():
                    raise ValueError("Unable to parse dates (neither ISO nor %Y%m%d).")
                df_tr = self.transform_subbasin_data(
                    geo_df=geo_df,
                    obs_df=df,
                    start_date=f"{year_start}{month_day_start}",
                    end_date=f"{year_end}{month_day_end}",
                    extent_obs=extent,
                    resample_to=resample_type,
                    resample_agg=agg,
                    season_months=season_obs
                )
                da = self.subbasin_data_to_xarray(df_tr)
                tmp_path = nc_path.with_suffix(".nc.tmp")
                da.to_netcdf(tmp_path)
                tmp_path.replace(nc_path)
                out_files[varname] = str(nc_path)
                if verbose:
                    print(f"[OK] Wrote: {nc_path.name}")
            except Exception as e:
                if nc_path.exists():
                    out_files[varname] = str(nc_path)
                    if verbose:
                        print(f"[ERROR] {csv_path.name}: {e} — returning existing NC: {nc_path.name}")
                else:
                    if verbose:
                        print(f"[ERROR] {csv_path.name}: {e} — no NC available to return.")
        return out_files

    def process_score_gcm(self, r, mod, model_name):
        
        tmp1 = pd.merge(r.to_dataframe().dropna().reset_index(), mod['SUBID'].to_dataframe().dropna().reset_index().drop("T", axis=1).drop_duplicates(), on=["X","Y"], how="left")
        tmp1['SUBID'] = tmp1['SUBID'].astype('int64')
        tmp1 = tmp1[["SUBID", "Observation"]]
        tmp1 = tmp1.rename(columns={"Observation": f"{model_name}"})
        return tmp1

        
    def plot_models_subplots(
        self,
        gdf,
        model_cols,
        ncols=3,
        cmap="viridis",
        scheme=None,            # e.g., "Quantiles", "FisherJenks" (needs mapclassify)
        k=5,
        common_norm=True,
        suptitle=None,
        as_points=False,
        markersize=8,
        # --- Cartopy bits ---
        projection=ccrs.PlateCarree(),       # any Cartopy CRS, e.g. ccrs.Mercator()
        extent=None,                         # (min_lon, max_lon, min_lat, max_lat) in degrees
        add_features=True,
        coastline_kw=None,
        borders_kw=None,
        gridlines_kw=None
    ):
        coastline_kw = coastline_kw or {"linewidth": 0.6}
        borders_kw   = borders_kw   or {"linewidth": 0.4}
        gridlines_kw = gridlines_kw or {"draw_labels": False, "linewidth": 0.3, "alpha": 0.4}
    
        n = len(model_cols)
        nrows = int(np.ceil(n / ncols))
    
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5*ncols, 4.4*nrows),
            constrained_layout=True,
            subplot_kw={"projection": projection}
        )
        axes = np.atleast_2d(axes)
    
        # Compute common vmin/vmax across all models (continuous only)
        sm = None
        norm = None
        if common_norm and scheme is None:
            vals = []
            for col in model_cols:
                a = gdf[col].to_numpy(dtype=float)
                a = a[np.isfinite(a)]
                if a.size:
                    vals.append(a)
            if vals:
                lo, hi = np.nanpercentile(np.concatenate(vals), [2, 98])
                norm = mpl.colors.Normalize(vmin=float(lo), vmax=float(hi))
                sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    
        # Draw each panel
        for ax, col in zip(axes.flat, model_cols):
            if extent is not None:
                ax.set_extent(extent, crs=ccrs.PlateCarree())
    
            if add_features:
                ax.add_feature(cfeature.LAND, facecolor="0.97")
                ax.add_feature(cfeature.OCEAN, facecolor="white")
                ax.coastlines(**coastline_kw)
                ax.add_feature(cfeature.BORDERS, **borders_kw)
                ax.gridlines(crs=ccrs.PlateCarree(), **gridlines_kw)
    
            plot_kw = dict(ax=ax, column=col, cmap=cmap, legend=False, missing_kwds={"color": "lightgrey"})
            if scheme is not None:
                plot_kw.update(scheme=scheme, k=k)
            elif norm is not None:
                plot_kw.update(vmin=norm.vmin, vmax=norm.vmax)
    
            # Try GeoPandas' Cartopy-aware plotting (supports transform=)
            try:
                if as_points:
                    gdf.plot(**plot_kw, markersize=markersize, transform=ccrs.PlateCarree())
                else:
                    gdf.plot(**plot_kw, edgecolor="none", linewidth=0.1, transform=ccrs.PlateCarree())
            except TypeError:
                # Fallback if older GeoPandas: ensure data is in PlateCarree (EPSG:4326)
                gdf_4326 = gdf.to_crs(4326)
                if as_points:
                    gdf_4326.plot(**plot_kw, markersize=markersize)
                else:
                    gdf_4326.plot(**plot_kw, edgecolor="none", linewidth=0.1)
    
            ax.set_title(str(col), fontsize=12)
    
        # Hide unused axes
        for ax in axes.flat[n:]:
            ax.set_visible(False)
    
        # Shared colorbar (continuous)
        if sm is not None:
            cbar = fig.colorbar(sm, ax=axes, shrink=0.7, location="right")
            cbar.set_label("Value")
    
        if suptitle:
            fig.suptitle(suptitle, y=0.995)
    
        return fig, axes

    def plot_one_score(self, gdf, norms, cmap, model_name, Score, dir_to_save_score):
               
        # Ensure CRS is lon/lat for Cartopy plotting
        if not isinstance(gdf, gpd.GeoDataFrame):
            gdf = gpd.GeoDataFrame(gdf, geometry="geometry")
        if gdf.crs is None:
            gdf = gdf.set_crs(4326)
        else:
            gdf = gdf.to_crs(4326)
        
        proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=proj)
        
        # Optional map context
        ax.add_feature(cfeature.LAND, facecolor="0.97")
        ax.add_feature(cfeature.OCEAN, facecolor="white")
        ax.coastlines(linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4)
        ax.gridlines(crs=proj, linewidth=0.3, alpha=0.4, linestyle=":")
        
        # Color scaling
        vmin, vmax = norms
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap(f"{cmap}")
        
        # Plot polygons
        gdf.plot(
            column=f"{model_name}",
            cmap=cmap,
            norm=norm,
            legend=False,                 # manual colorbar below
            edgecolor="k",
            linewidth=0.2,
            transform=proj,
            ax=ax,
            missing_kwds={"color": "white", "label": "Missing"} # "lightgrey"
        )
        
        # Nice extent with padding
        xmin, ymin, xmax, ymax = gdf.total_bounds
        padx, pady = 0.05 * (xmax - xmin or 1), 0.05 * (ymax - ymin or 1)
        ax.set_extent([xmin - padx, xmax + padx, ymin - pady, ymax + pady], crs=proj)
        
        # Manual colorbar
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02, orientation="horizontal")
        cbar.set_label(f"{Score}")
        
        ax.set_title(f"{model_name}")
        plt.tight_layout()
        plt.savefig(f"{dir_to_save_score}/{Score}_{model_name}.png", dpi=300, bbox_inches="tight")
        plt.show()


    def plot_prob_forecasts_gdf(
        self,
        dir_to_save,                               
        geo_centroid_zone,
        forecast_prob_gcm,
        obs,
        prob_cols=("PB", "PN", "PA"),
        model_name="Model",
        labels=("Below-Normal", "Near-Normal", "Above-Normal"),
        reverse_cmap=True,
        prob_unit="fraction",   # "fraction" (0-1) or "percent" (0-100)
        hspace=-0.6,
        logo=None,
        logo_size=("7%","21%"),
        logo_position="lower left"
    ):
        #######    123333
        tmp = geo_centroid_zone[["SUBID", "geometry"]]
        ff = forecast_prob_gcm.drop_vars('T').squeeze()
        ff.name = "forecast"
        ff = ff.to_dataframe().dropna().reset_index()
        ff = ff.pivot_table(
            index=["Y", "X"], 
            columns="probability", 
            values="forecast"
        ).reset_index()
        
        tmp1 = pd.merge(ff, obs['SUBID'].to_dataframe().dropna().reset_index().drop("T", axis=1).drop_duplicates(), on=["X","Y"], how="left")
        tmp1['SUBID'] = tmp1['SUBID'].astype('int64')
        tmp1 = tmp1[["SUBID", "PA", "PB", "PN"]]
        gdf = pd.merge(tmp1, tmp, on="SUBID", how="outer")
        #######    123333
        
        # --- Validate inputs
        if not isinstance(gdf, gpd.GeoDataFrame):
            gdf = gpd.GeoDataFrame(gdf, geometry="geometry")
            
        for c in prob_cols:
            if c not in gdf.columns:
                raise KeyError(f"Missing probability column: {c}")
        if gdf.crs is None:
            gdf = gdf.set_crs(4326)
        else:
            gdf = gdf.to_crs(4326)
    
        # --- Probabilities → percentages
        bn, nn, an = prob_cols
        gdf = gdf.copy()
        if prob_unit.lower().startswith("perc"):
            gdf["_bn"] = pd.to_numeric(gdf[bn], errors="coerce")
            gdf["_nn"] = pd.to_numeric(gdf[nn], errors="coerce")
            gdf["_an"] = pd.to_numeric(gdf[an], errors="coerce")
        else:
            gdf["_bn"] = pd.to_numeric(gdf[bn], errors="coerce") * 100.0
            gdf["_nn"] = pd.to_numeric(gdf[nn], errors="coerce") * 100.0
            gdf["_an"] = pd.to_numeric(gdf[an], errors="coerce") * 100.0
    
        # --- Argmax category per polygon
        probs_stack = np.vstack([gdf["_bn"].to_numpy(),
                                 gdf["_nn"].to_numpy(),
                                 gdf["_an"].to_numpy()]).T
        cat_idx = np.nanargmax(np.where(np.isfinite(probs_stack), probs_stack, -1e9), axis=1)  # 0=BN,1=NN,2=AN
    
        gdf["_max_prob"] = np.take_along_axis(probs_stack, cat_idx[:, None], axis=1).ravel()
        gdf["_is_bn"] = (cat_idx == 0)
        gdf["_is_nn"] = (cat_idx == 1)
        gdf["_is_an"] = (cat_idx == 2)
    
        # masked values: only show max-category probability per polygon
        gdf["_bn_show"] = np.where(gdf["_is_bn"], gdf["_max_prob"], np.nan)
        gdf["_nn_show"] = np.where(gdf["_is_nn"], gdf["_max_prob"], np.nan)
        gdf["_an_show"] = np.where(gdf["_is_an"], gdf["_max_prob"], np.nan)
    
        # --- Colormaps
        if reverse_cmap:
            AN_cmap = mcolors.LinearSegmentedColormap.from_list('AN', ["#fff7bc", "#fee391", "#fec44f", "#fe9929", "#ec7014", "#cc4c02", "#993404", "#662506"])
            NN_cmap = mcolors.LinearSegmentedColormap.from_list('NN', ["#d9d9d9", "#bdbdbd", "#969696", "#737373", "#525252"])
            BN_cmap = mcolors.LinearSegmentedColormap.from_list('BN', ["#e5f5f9", "#ccece6", "#99d8c9", "#66c2a4", "#41ae76", "#238b45", "#006d2c", "#00441b"])
        else:
            BN_cmap = mcolors.LinearSegmentedColormap.from_list('BN', ["#fff7bc", "#fee391", "#fec44f", "#fe9929", "#ec7014", "#cc4c02", "#993404", "#662506"])
            NN_cmap = mcolors.LinearSegmentedColormap.from_list('NN', ["#d9d9d9", "#bdbdbd", "#969696", "#737373", "#525252"])
            AN_cmap = mcolors.LinearSegmentedColormap.from_list('AN', ["#e5f5f9", "#ccece6", "#99d8c9", "#66c2a4", "#41ae76", "#238b45", "#006d2c", "#00441b"])
    
        # --- Figure + gridspec layout
        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(2, 3, height_ratios=[10, 0.2], width_ratios=[1.2, 0.6, 1.2],
                               hspace=hspace, wspace=0.2)
    
        proj = ccrs.PlateCarree()
        ax = fig.add_subplot(gs[0, :], projection=proj)
    
        # Map context
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#fde0dd", edgecolor="black", zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=1.0, linestyle='solid')
    
        gl = ax.gridlines(draw_labels=True, linewidth=0.05, color='gray', alpha=0.8)
        gl.top_labels = False
        gl.right_labels = False
    
        # Extent
        xmin, ymin, xmax, ymax = gdf.total_bounds
        padx, pady = 0.05*(xmax-xmin or 1), 0.05*(ymax-ymin or 1)
        ax.set_extent([xmin-padx, xmax+padx, ymin-pady, ymax+pady], crs=proj)
    
        # --- Plot polygons (only max-category prob per polygon)
        bn_norm = plt.Normalize(vmin=35, vmax=85)
        nn_norm = plt.Normalize(vmin=35, vmax=65)
        an_norm = plt.Normalize(vmin=35, vmax=85)
    
        # BN
        gdf.plot(column="_bn_show", cmap=BN_cmap, norm=bn_norm, legend=False,
                 edgecolor="k", linewidth=0.2, transform=proj, ax=ax, alpha=0.9,
                 missing_kwds={"color": "none"})
        # NN
        gdf.plot(column="_nn_show", cmap=NN_cmap, norm=nn_norm, legend=False,
                 edgecolor="k", linewidth=0.2, transform=proj, ax=ax, alpha=0.9,
                 missing_kwds={"color": "none"})
        # AN
        gdf.plot(column="_an_show", cmap=AN_cmap, norm=an_norm, legend=False,
                 edgecolor="k", linewidth=0.2, transform=proj, ax=ax, alpha=0.9,
                 missing_kwds={"color": "none"})
    
        # --- Colorbars with fixed ticks
        def create_ticks(vn=35, vx=86, step=5):
            return np.arange(vn, vx, step)
    
        sm_bn = cm.ScalarMappable(norm=bn_norm, cmap=BN_cmap); sm_bn.set_array([])
        sm_nn = cm.ScalarMappable(norm=nn_norm, cmap=NN_cmap); sm_nn.set_array([])
        sm_an = cm.ScalarMappable(norm=an_norm, cmap=AN_cmap); sm_an.set_array([])
    
        cbar_ax_bn = fig.add_subplot(gs[1, 0])
        cbar_bn = plt.colorbar(sm_bn, cax=cbar_ax_bn, orientation='horizontal')
        cbar_bn.set_label(f'{labels[0]} (%)')
        cbar_bn.set_ticks(create_ticks(35, 86, 5))
    
        cbar_ax_nn = fig.add_subplot(gs[1, 1])
        cbar_nn = plt.colorbar(sm_nn, cax=cbar_ax_nn, orientation='horizontal')
        cbar_nn.set_label(f'{labels[1]} (%)')
        cbar_nn.set_ticks(create_ticks(35, 66, 5))
    
        cbar_ax_an = fig.add_subplot(gs[1, 2])
        cbar_an = plt.colorbar(sm_an, cax=cbar_ax_an, orientation='horizontal')
        cbar_an.set_label(f'{labels[2]} (%)')
        cbar_an.set_ticks(create_ticks(35, 86, 5))
    
        # Title
        title_str = str(model_name.item()) if isinstance(model_name, np.ndarray) else str(model_name)
        ax.set_title(title_str, fontsize=13, pad=20)
    
        # Optional logo
        if logo is not None:
            ax_logo = inset_axes(ax, width=logo_size[0], height=logo_size[1],
                                 loc=logo_position, borderpad=0.1)
            ax_logo.imshow(mpimg.imread(logo))
            ax_logo.axis("off")
    
        plt.subplots_adjust(top=0.95, bottom=0.08, left=0.06, right=0.94, hspace=-0.6, wspace=0.2)
        plt.savefig(f"{dir_to_save}", dpi=300, bbox_inches='tight')
        plt.show()

    
    def transform_subbasin_data(self, geo_df, obs_df, start_date, end_date, extent_obs=None, resample_to='daily', resample_agg='mean', season_months=None, highflow_threshold=0.75, highflow_months=None):
        """
        Transform subbasin geospatial and observation data into a wide format, handling both long and wide obs_df formats,
        with optional clipping of geo_df by extent and filtering obs_df by SUBID.
        
        Parameters:
        - geo_df: GeoDataFrame with subbasin data (SUBID, POURX, POURY, geometry)
        - obs_df: DataFrame with observation data, either:
            - Long format: columns SUBID, DATE, QObs
            - Wide format: DATE column, followed by SUBID columns with QObs values
        - start_date: String or datetime, start date for filtering (e.g., '1994-08-01')
        - end_date: String or datetime, end date for filtering (e.g., '2016-12-31')
        - extent_obs: List of floats [north, west, south, east] for clipping geo_df; optional
        - resample_to: String, 'daily', 'monthly', 'seasonal', or 'highflow' to specify output temporal resolution
        - resample_agg: String, aggregation method for monthly/seasonal/highflow resampling ('mean', 'sum', 'max', 'median')
        - season_months: List of strings, months (e.g., ['03', '04']) for seasonal resampling; required if resample_to='seasonal'
        - highflow_threshold: Float (0 to 1) or numeric value, threshold for high-flow periods (default: 0.75 quantile);
                              if float between 0 and 1, interpreted as quantile; otherwise, absolute QObs threshold
        - highflow_months: List of strings, months (e.g., ['03', '04']) for high-flow resampling; optional for resample_to='highflow'
        
        Returns:
        - DataFrame in wide format with ID column (LONG, LAT, DAILY/ELEV, dates as YYYYMMDD, YYYYMM, or YYYY)
          and SUBID columns, without row index column.
        """
        # Validate parameters
        if resample_to not in ['daily', 'monthly', 'seasonal', 'highflow']:
            raise ValueError("resample_to must be 'daily', 'monthly', 'seasonal', or 'highflow'")
        if resample_agg not in ['mean', 'sum', 'max', 'median']:
            raise ValueError("resample_agg must be 'mean', 'sum', 'max', or 'median'")
        if resample_to == 'seasonal' and (season_months is None or not isinstance(season_months, list) or not all(m in [f"{i:02d}" for i in range(1, 13)] for m in season_months)):
            raise ValueError("season_months must be a list of valid month strings (e.g., ['03', '04']) when resample_to='seasonal'")
        if resample_to == 'highflow' and not (isinstance(highflow_threshold, (int, float)) and highflow_threshold >= 0):
            raise ValueError("highflow_threshold must be a non-negative number or quantile (0 to 1) when resample_to='highflow'")
        if resample_to == 'highflow' and highflow_months is not None and (not isinstance(highflow_months, list) or not all(m in [f"{i:02d}" for i in range(1, 13)] for m in highflow_months)):
            raise ValueError("highflow_months must be a list of valid month strings (e.g., ['03', '04']) when provided")
        if extent_obs is not None and (not isinstance(extent_obs, list) or len(extent_obs) != 4 or not all(isinstance(x, (int, float)) for x in extent_obs)):
            raise ValueError("extent_obs must be a list of four numbers [north, west, south, east]")
        
        # Ensure dates are in datetime format
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Clip geo_df if extent_obs is provided
        if extent_obs is not None:
            north, west, south, east = extent_obs
            minx, miny, maxx, maxy = west, south, east, north  # Reorder to [minx, miny, maxx, maxy]
            gdf = geo_df.copy()
            if gdf.crs is None:
                gdf = gdf.set_crs(4326)  # Assume WGS84 if no CRS
            else:
                gdf = gdf.to_crs(4326)  # Convert to WGS84
            bbox_poly = box(minx, miny, maxx, maxy)
            bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_poly], crs=4326)
            geo_df = gdf[gdf.geometry.centroid.within(bbox_poly)]
        
        # Process observation data
        obs_df = obs_df.copy()
        obs_df['DATE'] = pd.to_datetime(obs_df['DATE'], format="%Y-%m-%d")
        obs_df = obs_df[(obs_df['DATE'] >= start_date) & (obs_df['DATE'] <= end_date)]
        
        # Convert geo_df SUBID to string for consistency
        geo_df_clean = geo_df.drop(columns=geo_df.geometry.name).assign(
            LONG=pd.to_numeric(geo_df['POURX']),
            LAT=pd.to_numeric(geo_df['POURY'])
        )[['SUBID', 'LONG', 'LAT']]
        geo_df_clean['SUBID'] = geo_df_clean['SUBID'].astype(str)
        
        # Filter obs_df to include only SUBID values in clipped geo_df
        valid_subids = set(geo_df_clean['SUBID'])
        
        # Check obs_df format and convert to long format
        if 'SUBID' in obs_df.columns and 'QObs' in obs_df.columns:
            # Long format: already has SUBID, DATE, QObs
            obs_df_long = obs_df[['SUBID', 'DATE', 'QObs']].copy()
            obs_df_long['SUBID'] = obs_df_long['SUBID'].astype(str)
            # Filter by valid SUBIDs
            obs_df_long = obs_df_long[obs_df_long['SUBID'].isin(valid_subids)]
        else:
            # Wide format: DATE column, followed by SUBID columns with QObs values
            # Filter columns to only those SUBIDs in geo_df_clean
            value_vars = [col for col in obs_df.columns if col != 'DATE' and col in valid_subids]
            if not value_vars:
                raise ValueError("No valid SUBIDs from geo_df found in obs_df columns")
            obs_df = obs_df[['DATE'] + value_vars]
            obs_df_long = pd.melt(obs_df, id_vars=['DATE'], value_vars=value_vars, 
                                 var_name='SUBID', value_name='QObs')
            obs_df_long['SUBID'] = obs_df_long['SUBID'].astype(str)
        
        # Join datasets
        df_clean = geo_df_clean.merge(obs_df_long, on='SUBID', how='outer')
        
        # Complete the daily timeline for each SUBID
        subids = sorted(df_clean['SUBID'].unique())
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        idx = pd.MultiIndex.from_product([subids, date_range], names=['SUBID', 'DATE'])
        idx_df = pd.DataFrame(index=idx).reset_index()
        
        # Merge with df_clean and fill LONG/LAT
        dt = idx_df.merge(df_clean[['SUBID', 'DATE', 'LONG', 'LAT', 'QObs']], 
                         on=['SUBID', 'DATE'], how='left')
        dt[['LONG', 'LAT']] = dt.groupby('SUBID')[['LONG', 'LAT']].fillna(method='ffill').fillna(method='bfill')
        
        # Extract unique LONG, LAT, and ELEV (if present) per SUBID
        locs = dt.groupby('SUBID').agg({
            'LONG': lambda x: x.dropna().iloc[0] if x.dropna().size > 0 else np.nan,
            'LAT': lambda x: x.dropna().iloc[0] if x.dropna().size > 0 else np.nan,
            'QObs': lambda x: np.nan  # Placeholder for ELEV
        }).reset_index()
        locs['ELEV'] = np.nan  # Set ELEV to NA if not present
        
        # Resample based on resample_to
        if resample_to == 'monthly':
            # Convert DATE to year-month format for grouping
            dt['YEAR_MONTH'] = dt['DATE'].dt.to_period('M')
            # Aggregate QObs based on resample_agg
            agg_func = {
                'mean': 'mean',
                'sum': 'sum',
                'max': 'max',
                'median': 'median'
            }[resample_agg]
            dt = dt.groupby(['SUBID', 'YEAR_MONTH'])['QObs'].agg(agg_func).reset_index()
            # Convert YEAR_MONTH back to datetime (first day of the month)
            dt['DATE'] = dt['YEAR_MONTH'].dt.to_timestamp()
            # Update date range for monthly output
            date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
            idx = pd.MultiIndex.from_product([subids, date_range], names=['SUBID', 'DATE'])
            idx_df = pd.DataFrame(index=idx).reset_index()
            dt = idx_df.merge(dt[['SUBID', 'DATE', 'QObs']], on=['SUBID', 'DATE'], how='left')
        elif resample_to == 'seasonal':
            # Filter for specified months
            dt = dt[dt['DATE'].dt.strftime('%m').isin(season_months)]
            # Convert DATE to year-month format for grouping
            dt['YEAR'] = dt['DATE'].dt.year
            # Aggregate QObs based on resample_agg
            agg_func = {
                'mean': 'mean',
                'sum': 'sum',
                'max': 'max',
                'median': 'median'
            }[resample_agg]
            dt = dt.groupby(['SUBID', 'YEAR'])['QObs'].agg(agg_func).reset_index()
            
            # Create date range for specified months across years
            years = range(start_date.year, end_date.year + 1)
            if len(season_months) % 2 == 1:
                m = season_months[len(season_months) // 2]
            else:
                m = season_months[0]
                
            # Convert YEAR_MONTH back to datetime (first day of the month)
            dt['DATE'] = pd.to_datetime(dt['YEAR'].astype(str) +'-'+ m + '-01')
            date_range = [pd.to_datetime(f"{y}-{m}-01") for y in years] 
    
                          # if pd.to_datetime(f"{y}-{m}-01") <= end_date and pd.to_datetime(f"{y}-{m}-01") >= start_date]
            idx = pd.MultiIndex.from_product([subids, date_range], names=['SUBID', 'DATE'])
            idx_df = pd.DataFrame(index=idx).reset_index()
            dt = idx_df.merge(dt[['SUBID', 'DATE', 'QObs']], on=['SUBID', 'DATE'], how='left')
        elif resample_to == 'highflow':
            # Filter for specified months if provided
            if highflow_months is not None:
                dt = dt[dt['DATE'].dt.strftime('%m').isin(highflow_months)]
            # Identify high-flow periods based on threshold
            if 0 <= highflow_threshold <= 1:
                # Quantile-based threshold per SUBID
                thresholds = dt.groupby('SUBID')['QObs'].quantile(highflow_threshold).reset_index()
                dt = dt.merge(thresholds, on='SUBID', suffixes=('', '_threshold'))
                dt = dt[dt['QObs'] >= dt['QObs_threshold']].drop(columns=['QObs_threshold'])
            else:
                # Absolute threshold
                dt = dt[dt['QObs'] >= highflow_threshold]
            # Convert DATE to year for annual aggregation
            dt['YEAR'] = dt['DATE'].dt.year
            # Aggregate QObs based on resample_agg
            agg_func = {
                'mean': 'mean',
                'sum': 'sum',
                'max': 'max',
                'median': 'median'
            }[resample_agg]
            dt = dt.groupby(['SUBID', 'YEAR'])['QObs'].agg(agg_func).reset_index()
            # Convert YEAR to datetime (first day of the year)
            dt['DATE'] = pd.to_datetime(dt['YEAR'].astype(str) + '-01-01')
            # Update date range for yearly output
            date_range = pd.date_range(start=f"{start_date.year}-01-01", end=f"{end_date.year}-01-01", freq='YS')
            idx = pd.MultiIndex.from_product([subids, date_range], names=['SUBID', 'DATE'])
            idx_df = pd.DataFrame(index=idx).reset_index()
            dt = idx_df.merge(dt[['SUBID', 'DATE', 'QObs']], on=['SUBID', 'DATE'], how='left')
        
        # Pivot to wide format
        wide = dt.pivot(index='DATE', columns='SUBID', values='QObs').reset_index()
        id_cols = [col for col in wide.columns if col != 'DATE']
        
        # Prepare header rows
        row_LONG = pd.Series(['LONG'] + list(locs.set_index('SUBID').loc[id_cols, 'LONG']), 
                            index=['ID'] + id_cols)
        row_LAT = pd.Series(['LAT'] + list(locs.set_index('SUBID').loc[id_cols, 'LAT']), 
                           index=['ID'] + id_cols)
        row_DE = pd.Series(['DAILY/ELEV'] + list(locs.set_index('SUBID').loc[id_cols, 'ELEV']), 
                          index=['ID'] + id_cols)
        
        # Format dates and prepare wide output
        wide_out = wide.copy()
        if resample_to == 'daily':
            wide_out['ID'] = wide_out['DATE'].dt.strftime('%Y%m%d')
        elif resample_to in ['monthly', 'seasonal']:
            wide_out['ID'] = wide_out['DATE'].dt.strftime('%Y%m%d')
        else:  # highflow
            wide_out['ID'] = wide_out['DATE'].dt.strftime('%Y%m%d')
        wide_out = wide_out.drop(columns=['DATE'])[['ID'] + id_cols]
        
        # Combine header rows and resampled data
        final = pd.concat([
            row_LONG.to_frame().T,
            row_LAT.to_frame().T,
            row_DE.to_frame().T,
            wide_out
        ], ignore_index=True)
        
        # Delete the first column (row index), if present
        final = final.drop(columns=[' '], errors='ignore')
        
        return final


    def compute_correction(
        self,
        df: pd.DataFrame,
        id_col: str = "HYBAS_ID",
        date_col: str = "DATE",
        qestim_col: str = "QEstim",
        qcorr_col: str = "QCorr",
        qobs_col: str = "QObs",
        method: str = "median",
        time_scale: str = "month",  # "month" or "dayofyear"
        smooth: bool = True,
        interpolate: bool = True,
        window: int = 3,
        replace_with_obs: bool = True
    ) -> pd.DataFrame:
        """
        Compute and apply a correction factor for discharge estimation by month or by day of year.
    
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing columns for HYBAS_ID, DATE, QEstim, QCorr, and QObs.
        id_col : str, optional
            Column name for the basin or hydrological ID (default: "HYBAS_ID").
        date_col : str, optional
            Column name for the date (default: "DATE").
        qestim_col : str, optional
            Column name for estimated discharge (default: "QEstim").
        qcorr_col : str, optional
            Column name for corrected discharge (default: "QCorr").
        qobs_col : str, optional
            Column name for observed discharge (default: "QObs").
        method : str, optional
            Aggregation method: either 'mean' or 'median' (default: "median").
        time_scale : str, optional
            Temporal scale for correction factor: 'month' or 'dayofyear' (default: "month").
        smooth : bool, optional
            If True, apply a moving-average smoothing to the correction factor.
        interpolate : bool, optional
            If True, interpolate missing correction factors across the temporal scale.
        window : int, optional
            Rolling window (in units of months or days) for smoothing.
        replace_with_obs : bool, optional
            If True, replace QCorr, QEstim_corrected, and QEstim using QObs where available.
    
        Returns
        -------
        df_corrected : pd.DataFrame
            DataFrame with:
            - temporal key (month or dayofyear)
            - 'CorrectionFactor'
            - 'QEstim_corrected'
            - updated QCorr and QEstim if replace_with_obs=True
        """
    
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # --- Select time key ---
        if time_scale == "month":
            df["time_key"] = df[date_col].dt.month
            full_range = range(1, 13)
        elif time_scale == "dayofyear":
            df["time_key"] = df[date_col].dt.dayofyear
            full_range = range(1, 367)
        else:
            raise ValueError("time_scale must be 'month' or 'dayofyear'")
    
        # --- Select aggregation method ---
        if method not in ["mean", "median"]:
            raise ValueError("method must be 'mean' or 'median'")
        agg_func = np.nanmean if method == "mean" else np.nanmedian
    
        # --- Compute correction factor ---
        def compute_factor(group):
            qcorr_val = agg_func(group[qcorr_col])
            qestim_val = agg_func(group[qestim_col])
            return np.nan if qestim_val == 0 else qcorr_val / qestim_val
    
        factors = (
            df.groupby([id_col, "time_key"])
            .apply(compute_factor)
            .reset_index(name="CorrectionFactor")
        )
    
        # --- Smooth temporal variations ---
        if smooth:
            factors = (
                factors.sort_values([id_col, "time_key"])
                .groupby(id_col, group_keys=False)
                .apply(lambda g: g.assign(
                    CorrectionFactor=g["CorrectionFactor"]
                    .rolling(window=window, min_periods=1, center=True)
                    .mean()
                ))
            )
    
        # --- Interpolate missing days or months ---
        if interpolate:
            factors = (
                factors.groupby(id_col, group_keys=False)
                .apply(
                    lambda g: g.set_index("time_key")
                    .reindex(full_range)
                    .interpolate(limit_direction="both")
                    .assign(**{id_col: g[id_col].iloc[0]})
                    .reset_index()
                )
            )
    
        # --- Merge correction factor and apply ---
        df_corrected = df.merge(factors, on=[id_col, "time_key"], how="left")
        df_corrected["QEstim_corrected"] = df_corrected[qestim_col] * df_corrected["CorrectionFactor"]
    
        # --- Replace with observed values if available ---
        if replace_with_obs and qobs_col in df_corrected.columns:
            df_corrected[qcorr_col] = df_corrected[qobs_col].combine_first(df_corrected[qcorr_col])
            df_corrected["QEstim_corrected"] = df_corrected[qcorr_col].combine_first(df_corrected["QEstim_corrected"])
            df_corrected[qestim_col] = df_corrected["QEstim_corrected"].combine_first(df_corrected[qestim_col])
    
        return df_corrected[['SUBID', 'DATE', 'QEstim']].rename(
        columns={
            "QEstim": "QObs"
        }
    )
