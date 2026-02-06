import os
import warnings
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import date
from dateutil.relativedelta import relativedelta
import earthkit.data

# Suppress warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CONFIGURATION DICTIONARY
# ==============================================================================

VAR_CONFIG = {
    # --- SST ---
    "sst": {
        "cds_name": "sea_surface_temperature",
        "level_type": "single",
        "unit_func": lambda x: x - 273.15,
        "unit_label": "Anomalie SST (°C)",
        "plot_type": "anomaly_only",
        "cmap": "RdBu_r",
        "levels": np.arange(-3.0, 3.5, 0.5), 
        "extent": [40, -180, -40, 180],
        "compute_seasonal": True,
    },

    # --- SLP ---
    "slp": {
        "cds_name": "mean_sea_level_pressure",
        "level_type": "single",
        "unit_func": lambda x: x / 100,
        "unit_label": "hPa",
        "plot_type": "contour_overlay",
        "cmap": "RdBu_r",
        "levels_anom": np.arange(-5, 6, 1),
        "levels_contour": np.arange(990, 1038, 2),
        "highlight_black": [1015],
        "highlight_green": [1012],
        "extent": [45, -60, -45, 120]
    },

    # --- OLR ---
    "olr": {
        "cds_name": "top_net_thermal_radiation",
        "level_type": "single",
        "unit_func": lambda x: x / 86400,
        "unit_label": "W/m²",
        "plot_type": "contour_overlay",
        "cmap": "BrBG",
        "levels_anom": np.arange(-50, 55, 5),
        "levels_contour": np.arange(-320, -160, 20),
        "extent": [40, -180, -40, 180],
        "hov_band": [10, -10],
        "hov_cmap": "BrBG",
        "hov_levels": np.arange(-60, 65, 5)
    },

    # --- PRECIP ---
    "precip": {
        "cds_name": "total_precipitation",
        "level_type": "single",
        "unit_func": lambda x: x * 1000,
        "unit_label": "mm/mois",
        "plot_type": "ratio",
        "cmap": "BrBG",
        "levels": [0, 25, 50, 75, 90, 110, 125, 150, 200, 300],
        "extent": [35, -25, -5, 55]
    },

    # --- WIND 850 ---
    "wind_850": {
        "cds_name": ["u_component_of_wind", "v_component_of_wind", "specific_humidity"],
        "level_type": "pressure",
        "pressure_level": "850",
        "unit_func": lambda x: x,
        "unit_label": "Humidité Spécifique (g/kg)",
        "plot_type": "stream_humidity",
        "cmap": "GnBu",
        "levels": np.arange(0, 18, 2),
        "extent": [40, -60, -20, 60]
    },
    
    # --- SURFACE WIND ---
    "wind_surface": {
        "cds_name": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
        "level_type": "single",
        "unit_func": lambda x: x,
        "unit_label": "Vitesse du Vent (m/s)", 
        "plot_type": "streamlines",
        "cmap": "YlOrRd",
        "levels": np.arange(0, 15, 1),
        "extent": [35, -30, -5, 50],
    }
}


# ==============================================================================
# 2. DOWNLOAD MANAGER
# ==============================================================================

def download_data(dir_to_save, clim_start, clim_end, var_key, extent, target_date):
    conf = VAR_CONFIG[var_key]
    
    # Calculate months needed (Target-6 to Target-1)
    months_idx = []
    years_needed = set()
    for i in range(1, 7): 
        d = target_date - relativedelta(months=i)
        months_idx.append(d.strftime("%m"))
        years_needed.add(str(d.year))
    months_idx = sorted(list(set(months_idx)))
    
    years_range = [str(y) for y in range(clim_start, clim_end + 1)]
    for y in years_needed:
        if y not in years_range: years_range.append(y)
    
    ext_str = f"{extent[0]}_{extent[1]}_{extent[2]}_{extent[3]}"
    m_str = "".join(months_idx)
    fname = f"era5_{var_key}_{clim_start}-{clim_end}_past5_{ext_str}.nc"
    fpath = os.path.join(dir_to_save, fname)
    
    if os.path.exists(fpath):
        print(f" Data found: {fname}")
        return fpath

    print(f" Downloading {var_key}...")
    request = {
        "product_type": "monthly_averaged_reanalysis",
        "variable": conf["cds_name"],
        "year": years_range,
        "month": months_idx,
        "time": "00:00",
        "area": extent,
        "format": "netcdf",
    }
    dataset_name = "reanalysis-era5-single-levels-monthly-means"
    if conf["level_type"] == "pressure":
        dataset_name = "reanalysis-era5-pressure-levels-monthly-means"
        request["pressure_level"] = conf.get("pressure_level", "850")

    try:
        os.makedirs(dir_to_save, exist_ok=True)
        ds = earthkit.data.from_source("cds", dataset_name, request)
        ds.save(fpath)
        print(" Download complete.")
        return fpath
    except Exception as e:
        print(f" Error: {e}")
        return None


# ==============================================================================
# 3. HOVMOLLER PLOTTER
# ==============================================================================

def plot_hovmoller(data_anom, data_abs, var_key):
    conf = VAR_CONFIG[var_key]
    if "hov_band" not in conf: return

    print(f"Generating Hovmöller for {var_key}...")
    lat_max, lat_min = max(conf['hov_band']), min(conf['hov_band'])
    
    hov_anom = data_anom.sel(latitude=slice(lat_max, lat_min)).mean(dim='latitude')
    hov_abs = data_abs.sel(latitude=slice(lat_max, lat_min)).mean(dim='latitude')
    
    if conf["plot_type"] in ["vector", "streamlines", "stream_humidity"]:
        u_var = [v for v in hov_anom.data_vars if 'u' in v.lower() or 'var131' in v][0]
        to_plot_anom = hov_anom[u_var]
        to_plot_abs = hov_abs[u_var]
        title_suffix = f"U-Component ({lat_min}° to {lat_max}°)"
    else:
        if isinstance(hov_anom, xr.Dataset):
            var_name = list(hov_anom.data_vars)[0]
            to_plot_anom = hov_anom[var_name]
            to_plot_abs = hov_abs[var_name]
        else:
            to_plot_anom = hov_anom
            to_plot_abs = hov_abs
        title_suffix = f"({lat_min}° to {lat_max}°)"

    fig, ax = plt.subplots(figsize=(14, 10))
    times = to_plot_anom.time.values
    lons = to_plot_anom.longitude.values
    
    vals_anom = to_plot_anom.values.squeeze()
    vals_abs = to_plot_abs.values.squeeze()

    cf = ax.contourf(lons, times, vals_anom, levels=conf["hov_levels"], cmap=conf["hov_cmap"], extend='both')
    
    if var_key == "olr":
        ax.contour(lons, times, vals_abs, levels=[-240, -220, -200], colors='k', linewidths=0.8, linestyles='--')
    elif "wind" in var_key:
        ax.contour(lons, times, vals_abs, levels=[0], colors='k', linewidths=1.5)

    ax.set_title(f"Hovmöller: {var_key.upper()} - {title_suffix}", fontsize=16, fontweight='bold')
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Time", fontsize=12)
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.colorbar(cf, ax=ax, label=f"Anomaly ({conf['unit_label']})", orientation='horizontal', pad=0.08)
    plt.tight_layout()
    plt.show()


# ==============================================================================
# 4. MAP PLOTTER (6x1 LAYOUT)
# ==============================================================================

def plot_maps(data_main, data_overlay, var_key, title_prefix="Monthly"):
    conf = VAR_CONFIG[var_key]
    print(f"Generating {title_prefix} Maps for {var_key}...")
    
    dates = data_main.time.values
    n_plots = len(dates)
    
    # Layout 6x1
    fig, axes = plt.subplots(6, 1, figsize=(12, 24), subplot_kw={'projection': ccrs.PlateCarree()})
    
    if n_plots == 1: axes = [axes]
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else axes
    
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.2, wspace=0.0)
    
    if n_plots < 6:
        for i in range(n_plots, 6): fig.delaxes(axes_flat[i])

    plt.suptitle(f"{var_key.upper()} - {title_prefix} Analysis (6 Past Months)", fontsize=20, y=0.98, fontweight='bold')

    last_cf = None
    cbar_label_text = conf['unit_label']

    for i, (ax, dt) in enumerate(zip(axes_flat, dates)):
        date_str = np.datetime_as_string(dt, unit='M')
        ax.coastlines(linewidth=1.0)
        ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.6)
        ax.set_title(date_str, fontsize=12, fontweight='bold', loc='left')
        
        # === 1. SST ===
        if conf["plot_type"] == "anomaly_only":
            val = data_main.isel(time=i)
            last_cf = ax.contourf(val.longitude, val.latitude, val, 
                                  levels=conf["levels"], cmap=conf["cmap"], 
                                  extend='both', transform=ccrs.PlateCarree())

        # === 2. SLP ===
        elif conf["plot_type"] == "contour_overlay":
            val_anom = data_main.isel(time=i)
            last_cf = ax.contourf(val_anom.longitude, val_anom.latitude, val_anom, 
                                  levels=conf["levels_anom"], cmap=conf["cmap"], 
                                  extend='both', transform=ccrs.PlateCarree())
            val_abs = data_overlay.isel(time=i)
            cs = ax.contour(val_abs.longitude, val_abs.latitude, val_abs, 
                            levels=conf["levels_contour"], colors='black', linewidths=0.6, 
                            transform=ccrs.PlateCarree())
            ax.clabel(cs, inline=True, fontsize=8, fmt='%1.0f')
            
            if "highlight_black" in conf:
                cs_blk = ax.contour(val_abs.longitude, val_abs.latitude, val_abs, 
                                    levels=conf["highlight_black"], colors='black', linewidths=2.5, 
                                    transform=ccrs.PlateCarree())
                ax.clabel(cs_blk, inline=True, fontsize=10, fmt='%1.0f')
            if "highlight_green" in conf:
                cs_grn = ax.contour(val_abs.longitude, val_abs.latitude, val_abs, 
                                    levels=conf["highlight_green"], colors=['#008000'], linewidths=2.5, 
                                    transform=ccrs.PlateCarree())
                ax.clabel(cs_grn, inline=True, fontsize=10, fmt='%1.0f')

        # === 3. PRECIP ===
        elif conf["plot_type"] == "ratio":
            val = data_main.isel(time=i)
            norm = mcolors.BoundaryNorm(conf["levels"], ncolors=256)
            last_cf = ax.contourf(val.longitude, val.latitude, val, 
                                  levels=conf["levels"], norm=norm, cmap=conf["cmap"], 
                                  extend='max', transform=ccrs.PlateCarree())

        # === 4. VECTOR ===
        elif conf["plot_type"] == "vector":
            u_var = [v for v in data_main.data_vars if 'u' in v.lower()][0]
            v_var = [v for v in data_main.data_vars if 'v' in v.lower()][0]
            u = data_main[u_var].isel(time=i).values.squeeze()
            v = data_main[v_var].isel(time=i).values.squeeze()
            if u.ndim > 2: u = u[0]
            if v.ndim > 2: v = v[0]
            skip = 5
            ax.quiver(data_main.longitude[::skip], data_main.latitude[::skip], 
                      u[::skip, ::skip], v[::skip, ::skip], 
                      transform=ccrs.PlateCarree(), scale=250, width=0.003)
            last_cf = None 

        # === 5. STREAMLINES ===
        elif conf["plot_type"] in ["streamlines", "stream_humidity"]:
            u_name = [v for v in data_main.data_vars if 'u' in v.lower()][0]
            v_name = [v for v in data_main.data_vars if 'v' in v.lower()][0]
            q_name = [v for v in data_main.data_vars if 'humid' in v.lower() or 'q' in v.lower()]
            
            # Get Vectors
            u_val = data_main[u_name].isel(time=i).values.squeeze()
            v_val = data_main[v_name].isel(time=i).values.squeeze()
            if u_val.ndim > 2: u_val = u_val[0]
            if v_val.ndim > 2: v_val = v_val[0]

            # Determine Background (Humidity or Speed)
            bg_data = None
            
            if q_name:
                # If Humidity exists (850hPa case)
                q_val = data_main[q_name[0]].isel(time=i).values.squeeze()
                if q_val.ndim > 2: q_val = q_val[0]
                bg_data = q_val * 1000 # g/kg
            else:
                # If no Humidity (Surface Wind case) -> Calculate Speed
                bg_data = np.sqrt(u_val**2 + v_val**2)
            
            # Plot Background
            last_cf = ax.contourf(data_main.longitude, data_main.latitude, bg_data,
                                  levels=conf["levels"], cmap=conf["cmap"],
                                  extend='max', transform=ccrs.PlateCarree())
            
            # Plot Streamlines
            ax.streamplot(data_main.longitude.values, data_main.latitude.values, 
                          u_val, v_val,
                          transform=ccrs.PlateCarree(), density=1.0, 
                          color='k', linewidth=0.7, arrowsize=0.8)

    if last_cf is not None:
        cbar_ax = fig.add_axes([0.96, 0.15, 0.02, 0.7]) 
        cbar = fig.colorbar(last_cf, cax=cbar_ax, orientation='vertical')
        cbar.set_label(cbar_label_text, fontsize=14, fontweight='bold')
        cbar.ax.tick_params(labelsize=12)

    plt.show()


# ==============================================================================
# 5. DATA PROCESSOR
# ==============================================================================

def process_variable(fpath, var_key, clim_start, clim_end, target_date):
    conf = VAR_CONFIG[var_key]
    ds = xr.open_dataset(fpath)

    # Clean dimensions
    if "valid_time" in ds.coords or "valid_time" in ds.dims: ds = ds.rename({"valid_time": "time"})
    if "expver" in ds.dims:
        if ds.sizes["expver"] > 1:
            try: ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))
            except: ds = ds.isel(expver=0, drop=True)
        else: ds = ds.squeeze("expver", drop=True)
    if "pressure_level" in ds.dims: ds = ds.squeeze("pressure_level", drop=True)

    ds = ds.sortby("time")
    if var_key != "wind_850": ds = conf["unit_func"](ds)
        
    ref_slice = slice(f"{clim_start}-01-01", f"{clim_end}-12-31")
    
    # --- CRITICAL FIX: STRICT DATE SLICING ---
    # We want exactly [Target-6 months, Target-1 month].
    # Example: Target Feb 2026. We want Aug 2025 -> Jan 2026.
    start_date = target_date - relativedelta(months=6)
    end_date = target_date - relativedelta(months=1)
    
    # Identify Data Object (Dataset vs DataArray)
    # FIX: Extract DataArray for scalars to prevent TypeError in contourf
    is_vector = conf["plot_type"] in ["vector", "streamlines", "stream_humidity"]
    
    if is_vector:
        data_obj = ds # Dataset (needs U and V)
    else:
        var_name = list(ds.data_vars)[0]
        data_obj = ds[var_name] # DataArray (Single variable)

    # Calculate Climatology
    clim = data_obj.sel(time=ref_slice).groupby("time.month").mean("time")
    
    # Select Recent Data (Strict)
    recent_slice = slice(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    ds_recent = data_obj.sel(time=recent_slice)
    
    # Calculate Anomalies
    anom_recent = ds_recent.groupby("time.month") - clim
    
    # --- PREPARE DATA FOR PLOTTING ---
    if is_vector:
        data_map_main = ds_recent # Absolute values for streamlines
        data_map_overlay = None
        data_hov_anom = anom_recent
        data_hov_abs = ds_recent
    elif conf["plot_type"] == "ratio":
        # Recalc ratio for recent data
        clim_subset = clim.sel(month=ds_recent["time.month"])
        clim_subset = clim_subset.where(clim_subset > 1, 1.0)
        
        ratio = (ds_recent / clim_subset) * 100
        
        data_map_main = ratio
        data_map_overlay = None
        data_hov_anom = None
        data_hov_abs = None
    else:
        # Scalar (SST, SLP, OLR)
        data_map_main = anom_recent # DataArray
        data_map_overlay = ds_recent # Absolute for contours
        data_hov_anom = anom_recent
        data_hov_abs = ds_recent
        
    plot_maps(data_map_main, data_map_overlay, var_key)
    
    if data_hov_anom is not None:
        plot_hovmoller(data_hov_anom, data_hov_abs, var_key)

# ==============================================================================
# 6. MAIN
# ==============================================================================

def main_driver(dir_save, clim_start, clim_end, target_date_str, variables_list=None):
    target_date = date.fromisoformat(target_date_str)
    if variables_list is None: variables_list = list(VAR_CONFIG.keys())
    # print(f" Analyse: Cible {target_date_str} (exclue). Affichage des 6 mois précédents.")
    
    for var in variables_list:
        if var not in VAR_CONFIG: continue
        print(f"\n Traitement: {var.upper()}")
        extent = VAR_CONFIG[var]["extent"]
        fpath = download_data(dir_save, clim_start, clim_end, var, extent, target_date)
        if fpath: process_variable(fpath, var, clim_start, clim_end, target_date)
    print("\n Finished.")


import ipywidgets as widgets
from IPython.display import display, IFrame
import datetime

class C3SViewer:
    def __init__(self):
        # --- CONFIGURATION ---
        self.BASE_PACKAGE = "https://climate.copernicus.eu/charts/packages/c3s_seasonal/products/"
        
        # Product definitions
        self.products = {
            'Sea Surface Temperature (SST)': {'slug': 'c3s_seasonal_spatial_mm_ssto_3m', 'type': 'map'},
            'Mean Sea Level Pressure (MSLP)': {'slug': 'c3s_seasonal_spatial_mm_mslp_3m', 'type': 'map'},
            'Precipitation': {'slug': 'c3s_seasonal_spatial_mm_rain_3m', 'type': 'map'},
            '2m Temperature (T2m)': {'slug': 'c3s_seasonal_spatial_mm_2mtm_3m', 'type': 'map'},
            '10m Wind Speed': {'slug': 'c3s_seasonal_spatial_mm_wspd_3m', 'type': 'map'},
            'Nino Ensemble Plumes': {'slug': 'c3s_seasonal_plume_mm', 'type': 'plume'}
        }

        # Area definitions
        self.areas_spatial = [
            ('Global', 'area08'), ('Europe', 'area01'), ('Africa', 'area02'),
            ('North America', 'area05'), ('South America', 'area04'), 
            ('Asia', 'area03'), ('Australasia', 'area06')
        ]
        self.areas_nino = [
            ('Nino 3', 'nino3'), ('Nino 3.4', 'nino34'), 
            ('Nino 4', 'nino4'), ('Nino 1+2', 'nino12')
        ]

        # Map Type definitions
        self.types_spatial = [
            ('Tercile Summary', 'tsum'), ('Prob(most likely)', 'prob'), ('Ensemble Mean', 'em')
        ]
        self.types_nino = [('Plume', 'plume')]

        # Time setup
        current_year = datetime.datetime.now().year
        self.years = [str(y) for y in range(current_year - 2, current_year + 3)]
        self.months = [('Jan', '01'), ('Feb', '02'), ('Mar', '03'), ('Apr', '04'), ('May', '05'), ('Jun', '06'),
                       ('Jul', '07'), ('Aug', '08'), ('Sep', '09'), ('Oct', '10'), ('Nov', '11'), ('Dec', '12')]

        # --- WIDGET CREATION ---
        self.w_product = widgets.Dropdown(options=self.products.keys(), value='Sea Surface Temperature (SST)', description='Product:')
        self.w_year = widgets.Dropdown(options=self.years, value=str(current_year), description='Start Year:')
        self.w_month = widgets.Dropdown(options=self.months, value='01', description='Start Month:')
        self.w_area = widgets.Dropdown(options=self.areas_spatial, value='area08', description='Area:')
        self.w_type = widgets.Dropdown(options=self.types_spatial, value='tsum', description='Map Type:')
        self.out_display = widgets.Output()

        # Bind events
        self.w_product.observe(self._on_product_change, names='value')
        for w in [self.w_product, self.w_year, self.w_month, self.w_area, self.w_type]:
            w.observe(self._update_chart, names='value')

        # Layout
        self.ui = widgets.VBox([
            self.w_product,
            widgets.HBox([self.w_year, self.w_month, self.w_area, self.w_type])
        ])

        # Initial render
        self._update_chart()

    def _on_product_change(self, change):
        """Handle switching between Maps and Nino Plumes."""
        prod_info = self.products[change['new']]
        if prod_info['type'] == 'plume':
            self.w_area.options = self.areas_nino
            self.w_area.value = 'nino3'
            self.w_type.options = self.types_nino
            self.w_type.value = 'plume'
            self.w_type.disabled = True
        else:
            self.w_area.options = self.areas_spatial
            self.w_area.value = 'area08'
            self.w_type.options = self.types_spatial
            self.w_type.disabled = False

    def _update_chart(self, change=None):
        """Calculate URL and refresh IFrame."""
        prod_name = self.w_product.value
        prod_info = self.products[prod_name]
        
        base_time = f"{self.w_year.value}{self.w_month.value}010000"
        
        params = [
            f"area={self.w_area.value}",
            f"base_time={base_time}",
            f"type={self.w_type.value}"
        ]
        
        # Add valid_time only for spatial maps (current month + 1)
        if prod_info['type'] == 'map':
            dt_base = datetime.date(int(self.w_year.value), int(self.w_month.value), 1)
            if dt_base.month == 12:
                dt_valid = datetime.date(dt_base.year + 1, 1, 1)
            else:
                dt_valid = datetime.date(dt_base.year, dt_base.month + 1, 1)
            valid_time = f"{dt_valid.year}{dt_valid.month:02d}010000"
            params.append(f"valid_time={valid_time}")

        full_url = f"{self.BASE_PACKAGE}{prod_info['slug']}?{'&'.join(params)}"
        
        with self.out_display:
            self.out_display.clear_output(wait=True)
            display(IFrame(src=full_url, width="100%", height=850))

    def show(self):
        """Display the widget interface."""
        display(self.ui, self.out_display)




import ipywidgets as widgets
from IPython.display import display, IFrame

class BOMViewer:
    def __init__(self):
        # 1. Define configuration
        self.bom_tabs = [
            ('Outlooks (Forecasts)', 'Outlooks'),
            ('MJO Phase Diagram', 'MJO%20phase'),
            ('Monitoring', 'Monitoring'),
            ('Cloudiness', 'Cloudiness'),
            ('Regional Cloudiness', 'Regional%20cloudiness'),
            ('Time-Longitude', 'Time-longitude'),
            ('Tropical Update', 'Tropical%20update')
        ]
        
        # 2. Create Widgets
        self.w_bom_tab = widgets.Dropdown(
            options=self.bom_tabs,
            value='Outlooks',
            description='Select Tab:',
            style={'description_width': 'initial'}
        )
        
        self.out_bom = widgets.Output()
        
        # 3. Bind Logic
        self.w_bom_tab.observe(self._update_bom, names='value')
        
        # 4. Layout
        self.ui = widgets.VBox([self.w_bom_tab, self.out_bom])
        
        # Initial load
        self._update_bom()

    def _update_bom(self, change=None):
        """Internal function to update the IFrame based on dropdown selection."""
        tab_slug = self.w_bom_tab.value
        # The BOM URL structure
        url = f"https://www.bom.gov.au/climate/mjo/#tabs={tab_slug}"
        
        with self.out_bom:
            self.out_bom.clear_output(wait=True)
            # Display IFrame
            display(IFrame(src=url, width="100%", height=1000))

    def show(self):
        """Call this method to display the widget in the notebook."""
        display(self.ui)