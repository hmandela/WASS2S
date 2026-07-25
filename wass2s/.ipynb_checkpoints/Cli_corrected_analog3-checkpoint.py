#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WASS2S Seasonal Forecast CLI — Modular Rewrite
- Runs multiple SV-ML-CMME models sequentially (e.g. xgb, mlp, elm, rf)
- Exhaustive hyperparameter grids applied to Machine Learning estimators
- CCA logic factorized with get_best_models() thresholding and JSON reporting
- Consolidation dynamically discovers all generated intermediate datasets
- Includes 6 robust consensus MME methods (Eq, Skill, Min2009, MVA, NGR, xcELM)
- Supports dynamic switching between AgroIndicators and CHIRPS for observations
"""

import os, gc, json, time, argparse, calendar, datetime as dt, warnings
from pathlib import Path

# CRITICAL FOR SLURM: Force matplotlib to not use X-windows to prevent crashes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# --- Compat shim (some libs expect scipy.interp) ---
import numpy as _np
import scipy as _scipy
if not hasattr(_scipy, "interp"):
    _scipy.interp = _np.interp

import numpy as np
import pandas as pd
import xarray as xr
import joblib

# Additional missing imports
from scipy.stats import loguniform
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
import pickle

# ---- WASS2S namespace (user package) ----
from wass2s import * # =========================
# Defaults & Constants
# =========================
DEFAULT_DIR = "./PRCP_JAS_2025_ic_4/"
DEFAULT_LOGO = "./utilities/cilss.png"
FCST_LABELS_FR = ["EN DESSOUS DE LA MOYENNE", "PROCHE DE LA MOYENNE", "AU DESSUS DE LA MOYENNE"]
FCST_LABELS_EN = ["BELOW-AVERAGE", "NEAR-AVERAGE", "ABOVE-AVERAGE"]
DEFAULT_COUNTRY_ISO3 = "WAF"

DEFAULT_CLIM_YEARS = (1994, 2016)  
DEFAULT_HIND_YEARS = (1993, 2016)
DEFAULT_OBS_YEARS  = (1991, dt.date.today().year - 1)

SEASON_MAP = {
    "JFM": ["01", "02", "03"], "FMA": ["02", "03", "04"], "MAM": ["03", "04", "05"],
    "AMJ": ["04", "05", "06"], "MJJ": ["05", "06", "07"], "JJA": ["06", "07", "08"],
    "JAS": ["07", "08", "09"], "ASO": ["08", "09", "10"], "SON": ["09", "10", "11"],
    "OND": ["10", "11", "12"], "NDJ": ["11", "12", "01"], "DJF": ["12", "01", "02"],
}
WA_PRCP_EXTENT   = [21, -26, 4, 25]    
GCM_WA_EXTENT    = [21, -26, 4, 25]
GLOBAL_SST_EXTENT = [60, -180, -60, 180]
WIND_EXTENT = [50, -60, -50, 30]

CHECKPOINTS = {
    "obs": "chk_obs.json",
    "sv_ml_cmme": "chk_sv_ml_cmme.json",
    "cca_gcm": "chk_cca_gcm.json",
    "obs_lag": "chk_obs_lag.json",
    "analog": "chk_analog.json",
    "consolidate": "chk_consolidate.json",
}

# =========================
# Helpers & Parallelism
# =========================
def log_progress(step_name):
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{now}] >>> {step_name}", flush=True)

def save_and_close_plot(filepath):
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close('all')

def _setup_parallel(ncores: int, threads_per_worker: int = 1, use_dask: bool = True):
    ncores = int(max(1, ncores))
    for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ.setdefault(k, "1")

    client = None
    if use_dask:
        try:
            from dask.distributed import Client, LocalCluster
            n_workers = min(ncores, max(1, ncores // max(1, threads_per_worker)))
            cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker, processes=True, memory_limit=0, dashboard_address=None)
            client = Client(cluster)
            print(f"[parallel] Dask Client started: workers={n_workers}, threads/worker={threads_per_worker}")
        except Exception as e:
            print(f"[parallel] Dask not started. Reason: {e}")
    os.environ.setdefault("WAS_NCORES", str(ncores))
    return client

def save_json(path: Path, payload: dict): path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
def done(check_dir: Path, key: str) -> bool: return (check_dir / CHECKPOINTS[key]).exists()
def mark_done(check_dir: Path, key: str, meta: dict): save_json(check_dir / CHECKPOINTS[key], meta)
#def season_to_str(months): return "".join([calendar.month_abbr[int(m)] for m in months])
def season_to_str(months, lang='en', mode='abbr'):
    """
    Convertit une liste de mois en chaîne de caractères.
    
    Args:
        months: Liste d'entiers (ex: [3, 4, 5])
        lang: 'en' ou 'fr'
        mode: 
            'initial' -> 'MAM'
            'abbr'    -> 'Mar-Avr-Mai'
            'total'   -> 'Mars-Avril-Mai'
    """
    
    # Configuration des mois en français
    fr_data = {
        1:  {"total": "Janvier",   "abbr": "Jan", "initial": "J"},
        2:  {"total": "Février",   "abbr": "Fév", "initial": "F"},
        3:  {"total": "Mars",      "abbr": "Mar", "initial": "M"},
        4:  {"total": "Avril",     "abbr": "Avr", "initial": "A"},
        5:  {"total": "Mai",       "abbr": "Mai", "initial": "M"},
        6:  {"total": "Juin",      "abbr": "Jun", "initial": "J"},
        7:  {"total": "Juillet",   "abbr": "Jul", "initial": "J"},
        8:  {"total": "Août",      "abbr": "Aoû", "initial": "A"},
        9:  {"total": "Septembre", "abbr": "Sep", "initial": "S"},
        10: {"total": "Octobre",   "abbr": "Oct", "initial": "O"},
        11: {"total": "Novembre",  "abbr": "Nov", "initial": "N"},
        12: {"total": "Décembre",  "abbr": "Déc", "initial": "D"}
    }

    result = []
    for m in months:
        idx = int(m)
        if lang.lower() == 'fr':
            result.append(fr_data[idx][mode])
        else:
            # Gestion de l'anglais via calendar
            if mode == 'total':
                result.append(calendar.month_name[idx])
            elif mode == 'abbr':
                result.append(calendar.month_abbr[idx])
            else: # initial
                result.append(calendar.month_name[idx][0])

    # Formatage de sortie
    if mode == 'initial':
        return "".join(result) # Ex: MAM
    else:
        return "-".join(result) # Ex: Mars-Avril-Mai
def filter_center_variable(keys, contains): return [k for k in keys if isinstance(k, str) and "." in k and contains in k.split(".", 1)[1]]

def load_scores(pkl_path):
    if pkl_path.exists():
        with open(pkl_path, "rb") as f: return pickle.load(f)
    return {}

def save_scores(workdir, scores, scorefile):
    pkl_path = Path(workdir) / "intermediate" / scorefile
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pkl_path, "wb") as f: pickle.dump(scores, f)

def update_nested_dict(d1, d2):
    for k, v in d2.items():
        if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict): update_nested_dict(d1[k], v)
        else: d1[k] = v
    return d1

def load_bestfit_params(workdir: Path):
    path = workdir / "best_distribution_params.nc"
    if path.exists():
        ds = xr.load_dataset(path)
        return ds.code, ds.shape, ds.loca, ds.scale
    return None, None, None, None

def evaluate_and_plot_forecasts(was_verify, obs_da, hindcast_det, hindcast_prob, forecast_prob,
                                model_name, dict_key, clim_start, clim_end, scores_consolidated, 
                                dir_save_score, dir_to_forecast, season_str, forecast_year, month_init, fcst_labels, country_iso3, logopath):
    
    # Safely handle models (like NGR) that only generate probabilistic output
    if hindcast_det is not None:
        for metric in ['Pearson', 'MAE']:
            r = was_verify.compute_deterministic_score(was_verify.get_scores_metadata()[metric][5], obs_da, hindcast_det)
            scores_consolidated.setdefault(metric, {})[dict_key] = r.compute()
            try: was_verify.plot_model_score(r, metric, dir_save_score, model_name); plt.close('all')
            except Exception: pass

    if hindcast_prob is not None:
        for metric in ['GROC', 'RPSS']:
            r = was_verify.compute_probabilistic_score(was_verify.get_scores_metadata()[metric][5], obs_da, hindcast_prob, clim_start, clim_end)
            scores_consolidated.setdefault(metric, {})[dict_key] = r.compute()
            try: was_verify.plot_model_score(r, metric, dir_save_score, model_name); plt.close('all')
            except Exception: pass

    if forecast_prob is not None:
        try:
            # Standard Plot
            plot_prob_forecasts_(
                f"{dir_to_forecast}", 
                forecast_prob.drop_vars('T').squeeze(),
                f"{model_name} {season_str}-{forecast_year} IC: {calendar.month_name[int(month_init)]}",
                hspace=-0.6, labels=fcst_labels, reverse_cmap=False
            )
            
            # Advanced GADM Map Plot
            plot_prob_forecasts(
                f"{dir_to_forecast}",
                forecast_prob.drop_vars('T').squeeze(), 
                f"{model_name} {season_str}-{forecast_year} IC: {calendar.month_name[int(month_init)]}", 
                country_code=country_iso3, source="gadm", admin_level=1,
                stations_df=None, reverse_cmap=False, hspace=-0.6, labels=fcst_labels, logo=logopath, 
                logo_size=('21%', '14%'), logo_position='lower left', res=0.05, out="png"
            )
            plt.close('all')
        except Exception as e: 
            print(f"Plotting failed for {model_name}: {e}")

# =========================
# Stage 1 — Observations + Best-Fit Distributions
# =========================
def stage_obs(args, workdir: Path, downloader: 'WAS_Download'):
    check_dir = workdir / "checkpoints"
    if done(check_dir, "obs") and not args.redo:
        print("[obs] checkpoint exists → skip")
        return
        
    out_obs = workdir / "Observation"; out_obs.mkdir(parents=True, exist_ok=True)

    try:
        plot_map([args.obs_extent[1], args.obs_extent[3], args.obs_extent[2], args.obs_extent[0]], title="Data Download Area", fig_size=(4, 3))
        save_and_close_plot(workdir / "scores" / "Map_Data_Download_Area.png")
    except Exception: pass

    # Choose between CHIRPS and AgroIndicators based on user argument
    if args.obs_source == "chirps":
        log_progress("Downloading CHIRPS Observations")
        if not args.reuse_obs:
            path = downloader.WAS_Download_CHIRPSv3_Seasonal(
                str(out_obs), ["PRCP"], args.obs_years[0], args.obs_years[1],
                region='africa', area=args.obs_extent, season_months=args.season_months, force_download=args.force
            )
        else:
            # Assumes standard naming convention from the downloader if reusing
            season_str = "".join([calendar.month_abbr[int(m)] for m in args.season_months])
            path = str(out_obs / f"Obs_PRCP_{args.obs_years[0]}_{args.obs_years[1]}_{season_str}.nc")
            
        rainfall = xr.open_dataset(path, chunks='auto').rename({'precip':'PRCP'})['PRCP']
        rainfal_clim = rainfall.mean(dim="T", skipna=True)
        mask = xr.where((rainfal_clim >= 250), 1, np.nan)
        # Apply mask and dynamic cropping based on the extent provided
        rainfall = rainfall.where(mask == 1).sel(Y=slice(None, args.obs_extent[0]), X=slice(None, args.obs_extent[3]))

    else:
        log_progress("Downloading AgroIndicators Observations")
        variables_obs = [k for k in downloader.AgroObsName().keys() if "PRCP" in k]
        if not args.reuse_obs:
            downloader.WAS_Download_AgroIndicators(str(out_obs), variables_obs, args.obs_years[0], args.obs_years[1], args.obs_extent, args.season_months, args.force)
        
        rainfall = prepare_predictand(str(out_obs), variables_obs, args.obs_years[0], args.obs_years[1], args.season_months, ds=False, daily=False)

    # CPT CSV Merging Logic
    if args.cpt_csv and Path(args.cpt_csv).exists():
        log_progress("Merging with CPT Station Data")
        df = pd.read_csv(args.cpt_csv, na_values=-999.0, encoding="latin1")
        df_f = df[(df['STATION'] == 'LAT') | (df['STATION'] == 'LON') | (pd.to_numeric(df['STATION'], errors='coerce').between(args.obs_years[0], args.obs_years[1]))]
        verify_station_network(df_f, args.obs_extent)
        
        da = rainfall.sel(T=slice(str(args.obs_years[0]), str(args.obs_years[1])))
        cache = workdir / "rainfall_predictand_.nc"
        merger = WAS_Merging(df_f, da, date_month_day=args.merge_date)
        
        if cache.exists() and args.reuse_obs: 
            rainfall = xr.load_dataarray(cache)
        else:
            rainfall, _ = merger.simple_bias_adjustment(do_cross_validation=False)
            rainfall.to_netcdf(cache)

    rainfall.name = "PRCP"
    rainfall = rainfall.compute() # Compute base array into memory
    rainfall.to_netcdf(workdir / "rainfall_predictand.nc")

    clim_slice = slice(str(args.clim_years[0]), str(args.clim_years[1]))
    if args.dist_method == "bestfit":
        log_progress("Fitting Best Distribution (bestfit mode)")
        distribution_map = {'norm': 1, 'lognorm': 2, 'gamma': 4}
        transf = WAS_TransformData(rainfall.sel(T=clim_slice), distribution_map=distribution_map, n_clusters=1000)
        best_code_da, best_shape_da, best_loc_da, best_scale_da, _ = transf.fit_best_distribution(mode="grid")
            
        xr.Dataset({"code": best_code_da, "shape": best_shape_da, "loca": best_loc_da, "scale": best_scale_da}).to_netcdf(workdir / "best_distribution_params.nc")
        try:
            transf.plot_best_fit_map(best_code_da, {'norm':1,'lognorm':2,'expon':3,'gamma':4,'weibull_min':5,'t_dist':6,'poisson':7,'nbinom':8}, show_plot=False)
            save_and_close_plot(workdir / "scores" / "Map_Best_Fit_Distribution.png")
        except Exception: pass

    mark_done(check_dir, "obs", {"source": args.obs_source, "years": args.obs_years, "extent": args.obs_extent, "season": season_to_str(args.season_months)})
    print("[obs] done")

# =========================
# Stage 2 — Multiple SV-ML-CMME Models
# =========================
def _mme_factory(name: str, args):
    name = (name or "hpelm").lower()
    
    # Random Forest (Exhaustive Block)
    if name == "rf": 
        return WAS_mme_RF(
            search_method='bayesian', n_iter_search=50,
            n_estimators_range=[100, 300, 500, 800, 1000, 1200, 1500],
            max_depth_range=[2, 3, 5, 7, 8, 9, 10, 11, 15, None],
            min_samples_split_range=[2, 5, 10, 15, 20, 25], min_samples_leaf_range=[2, 3, 5, 7, 10],
            max_features_range=['sqrt', 'log2', 0.1, 0.3, 0.5], bootstrap_range=[True, False],
            max_samples_range=[0.5, 0.6, 0.7, 0.8, 0.9], min_impurity_decrease_range=[0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
            ccp_alpha_range=[0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2], cv_folds=5, n_clusters=3,
            verbose=1, random_state=42, dist_method=args.dist_method, optuna_n_jobs=getattr(args, 'ncores', 1), optuna_timeout=3600
        )
        
    # XGBoost (Exhaustive Block)
    if name == "xgb": 
        return WAS_mme_XGBoosting(
            search_method='bayesian', n_estimators_range=[25, 50, 100, 200, 300, 500, 600, 700, 1000, 1500, 2000],
            learning_rate_range=[0.0001, 0.001, 0.01, 0.05, 0.1], max_depth_range=[2, 3, 4, 5, 7], 
            min_child_weight_range=[1, 2, 5, 10], subsample_range=[0.6, 0.8, 1.0], colsample_bytree_range=[0.5, 0.7, 0.9, 1.0], 
            gamma_range=[0, 0.001, 0.01, 0.1, 0.5, 1.0], reg_alpha_range=[0, 0.1, 0.5, 1.0], reg_lambda_range=[1, 3, 5, 10], 
            random_state=42, dist_method=args.dist_method, n_iter_search=50, cv_folds=5, n_clusters=3,
            optuna_n_jobs=getattr(args, 'ncores', 1), optuna_timeout=3600
        )
        
    # Multi-Layer Perceptron (Exhaustive Block)
    if name == "mlp": 
        return WAS_mme_MLP(
            hidden_layer_sizes_range=[(2,4), (4,2), (8,2), (8,4)], learning_rate_init_range=loguniform(1e-6, 1e-3), 
            activation_options=['tanh'], solver_options=['adam','sgd'], alpha_range=loguniform(1e-6, 1e-2), 
            search_method='bayesian', random_state=42, n_iter_search=36, max_iter=5000, cv_folds=5, n_clusters=2, 
            dist_method=args.dist_method, optuna_n_jobs=getattr(args, 'ncores', 1), optuna_timeout=3600
        )
        
    # Stacking
    if name == "stack": 
        return WAS_mme_Stacking(
            meta_learner_type='ridge', alpha_range=[0.1, 1.0, 10.0, 100.0], random_state=42, dist_method=args.dist_method, 
            stacking_cv=3, meta_cv_folds=3, meta_n_iter_search=10, n_clusters=6
        )
    
    # Default: hpELM (Exhaustive Block)
    return WAS_mme_hpELM(
        neurons_range=[10, 25, 50, 100, 200, 500], activation_options=['tanh', 'lin'], norm_range=loguniform(1e-2, 1e3), 
        random_state=42, n_iter_search=100, cv_folds=5, dist_method=args.dist_method, search_method='bayesian', 
        n_clusters=2, n_trials_bayesian=50, bayesian_sampler='tpe'
    )

def stage_sv_ml_cmme(args, workdir: Path, downloader: 'WAS_Download', client):
    check_dir = workdir / "checkpoints"
    if done(check_dir, "sv_ml_cmme") and not args.redo:
        print("[sv-ml-cmme] checkpoint exists → skip")
        return
    
    rainfall = xr.load_dataarray(workdir / "rainfall_predictand.nc")
    was_verify = WAS_Verification(dist_method=args.dist_method)
    dir_model = workdir / "model_data"; dir_model.mkdir(exist_ok=True, parents=True)
    scores_dir = workdir / "scores"; scores_dir.mkdir(parents=True, exist_ok=True)
    out_dir = workdir / "forecasts"; out_dir.mkdir(exist_ok=True)

    center_variable = filter_center_variable(downloader.ModelsName().keys(), "PRCP")
    for m in ['UKMO_603.PRCP', 'DWD_21.PRCP', 'METEOFRANCE_8.PRCP']:
        if m in center_variable: center_variable.remove(m)

    hind = downloader.WAS_Download_Models(str(dir_model), center_variable, args.init_month, args.lead_time, args.hindcast_years[0], args.hindcast_years[1], args.gcm_extent, None, args.ensemble_mean, args.force)
    fcst = downloader.WAS_Download_Models(str(dir_model), center_variable, args.init_month, args.lead_time, args.hindcast_years[0], args.hindcast_years[1], args.gcm_extent, args.forecast_year, args.ensemble_mean, args.force)
                                          
    hind = {k: hind[k] for k in hind.keys() & fcst.keys()}
    fcst = {k: fcst[k] for k in hind.keys() & fcst.keys()}
    common_keys = {k.lower() for k in hind} & {k.lower() for k in fcst}
    center_variable = [it for it in center_variable if any(it.lower().startswith(k) for k in list(common_keys))]
    
    log_progress("GCM Validation & Selection of Best Models")
    scores = {}
    for metric in ["Pearson", "MAE"]:
        s_dict = was_verify.gcm_validation_compute(hind, rainfall.sel(T=slice(str(args.hindcast_years[0]), str(args.hindcast_years[1]))), metric, args.init_month, args.clim_years[0], args.clim_years[1], str(scores_dir), lead_time=None, ensemble_mean=None, gridded=True)
        scores[metric] = {k: v.compute() for k, v in s_dict.items()}
        try: was_verify.plot_models_score(scores[metric], metric, str(scores_dir)); plt.close('all')
        except Exception: pass

    # Top-N by MAE (Template Practice)
    best_models = get_best_models(center_variable, scores, metric='MAE', threshold=200, top_n=args.top_n_gcm, gcm=True)
    save_json(workdir / "scores/sv_ml_cmme_best_models.json", {"best_models": best_models})

    all_model_hdcst, all_model_fcst, obs, _ = process_datasets_for_mme(
        rainfall.sel(T=slice(str(args.hindcast_years[0]), str(args.hindcast_years[1]))), hdcsted=hind, fcsted=fcst, scores=scores, gcm=True, best_models=best_models, dir_to_save_model=str(dir_model), year_start=args.hindcast_years[0], year_end=args.hindcast_years[1], score_metric="MAE", model=True, month_of_initialization=args.init_month, lead_time=args.lead_time, year_forecast=args.forecast_year
    )

    best_code_da, best_shape_da, best_loc_da, best_scale_da = load_bestfit_params(workdir)
    cv = WAS_Cross_Validator(n_splits=len(obs.get_index("T")), nb_omit=2)

    for m_name in args.mme_models:
        scores_consolidated = {}
        log_progress(f"Training & Evaluating {m_name.upper()} (SV-ML-CMME)")
        model = _mme_factory(m_name, args)
        
        # Using threading instead of loky/processes
        if client is not None:
            with joblib.parallel_backend('dask'):
                best_params, cluster = model.compute_hyperparameters(all_model_hdcst, obs, args.clim_years[0], args.clim_years[1])
        else:
            print(f"[sv-ml-cmme] Warning: Dask not found. Forcing 'threading' backend to prevent SLURM SemLock crashes.")
            with joblib.parallel_backend('threading', n_jobs=args.ncores):
                best_params, cluster = model.compute_hyperparameters(all_model_hdcst, obs, args.clim_years[0], args.clim_years[1])
        
        try: cluster.plot(); save_and_close_plot(scores_dir / f"{m_name.upper()}_Clusters.png")
        except: pass

        cv_kwargs = {"best_params": best_params, "cluster_da": cluster}
        if args.dist_method == "bestfit":
            cv_kwargs.update({"best_code_da": best_code_da, "best_shape_da": best_shape_da, "best_loc_da": best_loc_da, "best_scale_da": best_scale_da})

        hind_det, hind_prob = cv.cross_validate(model, obs, all_model_hdcst, args.clim_years[0], args.clim_years[1], **cv_kwargs)
        fcst_det, fcst_prob = model.forecast(obs, args.clim_years[0], args.clim_years[1], all_model_hdcst, hind_det, all_model_fcst, **cv_kwargs)

        model_id = f"SV-ML-CMME_{m_name.upper()}"
        evaluate_and_plot_forecasts(was_verify, obs, hind_det, hind_prob, fcst_prob, model_id, model_id, args.clim_years[0], args.clim_years[1], scores_consolidated, str(scores_dir), str(out_dir), season_to_str(args.season_months), args.forecast_year, args.init_month, FCST_LABELS_FR, args.country_code, args.logo)

        (workdir/"intermediate").mkdir(exist_ok=True, parents=True)
        xr.Dataset({f"hdcst_det_{model_id}": hind_det, f"hdcst_prob_{model_id}": hind_prob}).to_netcdf(workdir/f"intermediate/sv_ml_cmme_{m_name}_hindcasts.nc")
        xr.Dataset({f"fcst_det_{model_id}": fcst_det, f"fcst_prob_{model_id}": fcst_prob}).to_netcdf(workdir/f"intermediate/sv_ml_cmme_{m_name}_forecasts.nc")
        save_scores(workdir, scores_consolidated, f"scores_consolidated_sv_ml_cmme_{m_name}.pkl")

        gc.collect()
        if client: client.restart()

    if args.add_ngr:
        scores_consolidated = {}
        log_progress("Training & Evaluating NGR (SV-ML-CMME)")
        best_models2 = get_best_models(center_variable, scores, metric='MAE', threshold=200, top_n=max(6, args.top_n_gcm*2), gcm=True)
        all_model_hdcst2, all_model_fcst2, obs2, _ = process_datasets_for_mme(
            rainfall.sel(T=slice(str(args.hindcast_years[0]), str(args.hindcast_years[1]))), hdcsted=hind, fcsted=fcst, scores=scores, gcm=True, best_models=best_models2, dir_to_save_model=str(dir_model), year_start=args.hindcast_years[0], year_end=args.hindcast_years[1], score_metric="MAE", model=True, month_of_initialization=args.init_month, lead_time=args.lead_time, year_forecast=args.forecast_year)
        
        was_mme_ngr = WAS_mme_NGR(type="NGR", apply_to="sig", alpha=0.1)
        all_model_hdcst2['T'] = obs2['T']
        was_mme_ngr.fit(all_model_hdcst2, obs2.drop_vars("M").squeeze(), clim_terciles=True)
        
        hind_prob_ngr = xr.concat([was_mme_ngr.predict(all_model_hdcst2.sel(T=slice(i, i)), quantiles=[0.9], clim_terciles=True, parametric=True)['tercile_probability'] for i in all_model_hdcst2.coords['T']], dim='T')
        hind_prob_ngr['T'] = obs2['T']

        t0 = obs2.isel(T=0).coords['T'].values
        new_T = np.datetime64(f"{all_model_fcst2.coords['T'].values.astype('datetime64[Y]').astype(int)[0] + 1970}-{(t0.astype('datetime64[M]').astype(int) % 12 + 1):02d}-01")
        all_model_fcst2 = all_model_fcst2.assign_coords(T=xr.DataArray([new_T], dims=["T"]))
        forecast_prob_ngr = was_mme_ngr.predict(all_model_fcst2, quantiles=[0.9], clim_terciles=True, parametric=True)['tercile_probability']

        # ADDED PLOTTING CAPABILITY FOR NGR HERE
        model_id_ngr = "SV-ML-CMME_NGR"
        evaluate_and_plot_forecasts(
            was_verify, obs2, None, hind_prob_ngr, forecast_prob_ngr, 
            model_id_ngr, model_id_ngr, args.clim_years[0], args.clim_years[1], 
            scores_consolidated, str(scores_dir), str(out_dir), season_to_str(args.season_months), 
            args.forecast_year, args.init_month, FCST_LABELS_FR, args.country_code, args.logo
        )

        save_scores(workdir, scores_consolidated, "scores_consolidated_sv_ml_cmme_ngr.pkl")

        xr.Dataset({"hdcst_prob_SV-ML-CMME_NGR": hind_prob_ngr}).to_netcdf(workdir/"intermediate/sv_ml_cmme_ngr_hindcasts.nc")
        xr.Dataset({"fcst_prob_SV-ML-CMME_NGR": forecast_prob_ngr}).to_netcdf(workdir/"intermediate/sv_ml_cmme_ngr_forecasts.nc")

    mark_done(check_dir, "sv_ml_cmme", {"models_run": args.mme_models, "top_n": args.top_n_gcm})

# =========================
# Stage 3/4 — Factorized CCA Pipeline
# =========================
def _execute_cca_pipeline(args, workdir, downloader, var_filter, extent, defined_zone, prefix, top_n):
    log_progress(f"Processing Factorized CCA Pipeline for: {prefix.upper()}")
    rainfall = xr.load_dataarray(workdir / "rainfall_predictand.nc")
    dir_model = workdir / "model_data"; dir_model.mkdir(exist_ok=True)
    scores_dir = workdir / "scores"; scores_dir.mkdir(exist_ok=True)
    out_dir = workdir / "forecasts"; out_dir.mkdir(exist_ok=True)

    center_variable = filter_center_variable(downloader.ModelsName().keys(), var_filter)
    for bad in args.drop_gcm:
        try: center_variable.remove(bad)
        except ValueError: pass

    hind = downloader.WAS_Download_Models(str(dir_model), center_variable, args.init_month, args.lead_time, args.hindcast_years[0], args.hindcast_years[1], extent, None, args.ensemble_mean, args.force)
    fcst = downloader.WAS_Download_Models(str(dir_model), center_variable, args.init_month, args.lead_time, args.hindcast_years[0], args.hindcast_years[1], extent, args.forecast_year, args.ensemble_mean, args.force)

    hind = {k: hind[k] for k in hind.keys() & fcst.keys()}
    fcst = {k: fcst[k] for k in hind.keys() & fcst.keys()}
    common_keys = {k.lower() for k in hind} & {k.lower() for k in fcst}
    center_variable = [it for it in center_variable if any(it.lower().startswith(k) for k in list(common_keys))]
    
    try:
        plot_map([extent[1],extent[3],extent[2],extent[0]], sst_indices=defined_zone, title=f"Predictors Area {var_filter}", fig_size=(6,4))
        save_and_close_plot(scores_dir / f"Map_CCA_Predictors_{prefix}.png")
    except: pass

    best_code_da, best_shape_da, best_loc_da, best_scale_da = load_bestfit_params(workdir)
    was_cca = WAS_CCA(n_modes=args.cca_modes, n_pca_modes=args.cca_pca_modes, dist_method=args.dist_method)
    obs = rainfall.sel(T=slice(str(args.hindcast_years[0]), str(args.hindcast_years[1])))
    cv = WAS_Cross_Validator(n_splits=len(obs.get_index("T")), nb_omit=2)

    predictors = {
        k: retrieve_single_zone_for_PCR(str(dir_model), defined_zone, k, args.hindcast_years[0], args.hindcast_years[1], clim_year_start=args.clim_years[0], clim_year_end=args.clim_years[1], model=True, month_of_initialization=args.init_month, lead_time=args.lead_time)
        for k in center_variable
    }
    
    hind_det_all, hind_prob_all, fcst_det_all, fcst_prob_all = {}, {}, {}, {}
    cv_k = {"best_code_da": best_code_da, "best_shape_da": best_shape_da, "best_loc_da": best_loc_da, "best_scale_da": best_scale_da} if args.dist_method=="bestfit" else {}

    was_verify = WAS_Verification(dist_method=args.dist_method)
    cca_scores = {}
    
    # 1. Collect predictions for all models
    for key, predictor in predictors.items():
        try:
            was_cca.plot_cca_results(X=predictor.isel(T=slice(None, -1)), Y=obs, clim_year_start=args.clim_years[0], clim_year_end=args.clim_years[1])
            save_and_close_plot(scores_dir / f"CCA_Modes_{prefix}_{key}.png")
        except: pass
        
        p_train = predictor.isel(T=slice(None, -1))
        p_train['T'] = obs['T']
        
        hd, hp = cv.cross_validate(was_cca, obs, p_train, args.clim_years[0], args.clim_years[1], **cv_k)
        fd, fp = was_cca.forecast(obs, args.clim_years[0], args.clim_years[1], p_train, hd, predictor.isel(T=[-1]), **cv_k)
        
        k_id = f"{key}_{prefix}"
        hind_det_all[k_id] = hd
        hind_prob_all[k_id] = hp
        fcst_det_all[k_id] = fd
        fcst_prob_all[k_id] = fp

    # 2. Compute spatial DataArray scores for get_best_models logic
    for metric in ["Pearson", "MAE"]:
        tmp = {}
        for k_id, hd in hind_det_all.items():
            r = was_verify.compute_deterministic_score(was_verify.get_scores_metadata()[metric][5], obs, hd)
            tmp[k_id] = r.compute() 
        cca_scores[metric] = tmp

    for metric in ["GROC", "RPSS"]:
        tmp = {}
        for k_id, hp in hind_prob_all.items():
            r = was_verify.compute_probabilistic_score(was_verify.get_scores_metadata()[metric][5], obs, hp, args.clim_years[0], args.clim_years[1])
            tmp[k_id] = r.compute() 
        cca_scores[metric] = tmp

    # 3. Filter using get_best_models thresholding
    model_keys = list(hind_det_all.keys())
    best_models = get_best_models(model_keys, cca_scores, metric='GROC', threshold=0.6, top_n=top_n, gcm=False)
    
    # Save the explicitly chosen best models to JSON
    save_json(workdir / f"scores/{prefix}_best_models.json", {"best_models": best_models})
    
    # 4. Filter and Save intermediate NetCDFs ONLY for the best models
    best_hind = {f"hdcst_det_{k}": hind_det_all[k] for k in best_models}
    best_hind.update({f"hdcst_prob_{k}": hind_prob_all[k] for k in best_models})
    best_fcst = {f"fcst_det_{k}": fcst_det_all[k] for k in best_models}
    best_fcst.update({f"fcst_prob_{k}": fcst_prob_all[k] for k in best_models})

    (workdir/"intermediate").mkdir(exist_ok=True, parents=True)
    xr.Dataset(best_hind).to_netcdf(workdir/f"intermediate/{prefix}_hindcasts.nc")
    xr.Dataset(best_fcst).to_netcdf(workdir/f"intermediate/{prefix}_forecasts.nc")
            
    # 5. Evaluate and Plot ONLY the best models
    dummy_scores = {}
    for k in best_models:
        evaluate_and_plot_forecasts(
            was_verify, obs, hind_det_all[k], hind_prob_all[k], fcst_prob_all[k], 
            f"CCA_{k}", k, args.clim_years[0], args.clim_years[1], 
            dummy_scores, str(scores_dir), str(out_dir), season_to_str(args.season_months), 
            args.forecast_year, args.init_month, FCST_LABELS_FR, args.country_code, args.logo
        )

    # 6. Restrict the cca_scores dictionary before consolidating it
    for metric, models in cca_scores.items():
        for model in list(models): 
            if not any(model.startswith(p) for p in best_models):
                del models[model]
                
    scores_consolidated = {}
    for key in cca_scores:
        scores_consolidated.setdefault(key, {}).update(cca_scores[key])
        
    save_scores(workdir, scores_consolidated, f"scores_consolidated_{prefix}.pkl")
    return best_models

# =========================
# Stage 5 — Obs-Lag
# =========================
def stage_obs_lag(args, workdir: Path, downloader: 'WAS_Download'):
    check_dir = workdir / "checkpoints"
    if done(check_dir, "obs_lag") and not args.redo:
        print("[obs-lag] checkpoint exists → skip")
        return
        
    rainfall = xr.load_dataarray(workdir / "rainfall_predictand.nc")
    dir_rea = workdir / "reanalysis_data"; dir_rea.mkdir(exist_ok=True)
    scores_dir = workdir / "scores"; scores_dir.mkdir(exist_ok=True)
    out_dir = workdir / "forecasts"; out_dir.mkdir(exist_ok=True)

    downloader.WAS_Download_Reanalysis(str(dir_rea), ["NOAA.SST"], 1990,
                                       #args.obs_years[0],
                                       2025,
                                       # args.obs_years[1]+1, 
                                       args.sst_extent, args.season_months_lag, args.force)
    was_verify = WAS_Verification(dist_method=args.dist_method)

    idx_names = ['NINO34', 'TNA', 'TSA', 'DMI', 'MB']
    predictors = compute_sst_indices(str(dir_rea), idx_names, "NOAA.SST", 
                                     1990,
                                     # args.obs_years[0], 
                                     2025,
                                     # args.obs_years[1]+1,
                                     args.season_months_lag, args.clim_years[0], args.clim_years[1], {})
    dfp = predictors.to_dataframe()
    vif = pd.DataFrame({"feature": dfp.columns, "VIF": [VIF(dfp, i) for i in range(dfp.shape[1])]})
    keep = vif.loc[vif.VIF < args.vif_threshold, "feature"].tolist()
    filt_pred = predictors[keep].to_array().rename({"variable": "features"}).transpose('T', 'features')

    best_code_da, best_shape_da, best_loc_da, best_scale_da = load_bestfit_params(workdir)
    cv_k = {"best_code_da": best_code_da, "best_shape_da": best_shape_da, "best_loc_da": best_loc_da, "best_scale_da": best_scale_da} if args.dist_method=="bestfit" else {}
    
    cv = WAS_Cross_Validator(n_splits=len(rainfall.get_index("T")), nb_omit=2)
    scores_consolidated = {}
    out = workdir / "intermediate"; out.mkdir(exist_ok=True, parents=True)

    if args.obs_lag_policy in ("both","mlr"):
        log_progress("Processing Observation-Based with Lag (MLR)")
        model_mlr = WAS_LinearRegression_Model(nb_cores=args.ncores, dist_method=args.dist_method) 
        fp_train = filt_pred.isel(T=slice(None, -1)); fp_train['T'] = rainfall['T']
        hd_mlr, hp_mlr = cv.cross_validate(model_mlr, rainfall, fp_train, args.clim_years[0], args.clim_years[1], **cv_k)

        f_pred = filt_pred.isel(T=[-1]).assign_coords(T=[pd.Timestamp(f"{args.forecast_year}-{int(args.season_months_lag[1]):02d}-01")])
        fd_mlr, fp_mlr = model_mlr.forecast(rainfall, args.clim_years[0], args.clim_years[1], fp_train, hd_mlr, f_pred, **cv_k)

        evaluate_and_plot_forecasts(
            was_verify, rainfall, hd_mlr, hp_mlr, fp_mlr, "MLR_VIF.ERSST", 'MLR_VIF.ERSST', 
            args.clim_years[0], args.clim_years[1], scores_consolidated, str(scores_dir), str(out_dir), 
            season_to_str(args.season_months), args.forecast_year, args.init_month, FCST_LABELS_FR, args.country_code, args.logo
        )
        
        xr.Dataset({"hdcst_det_MLR_VIF.ERSST": hd_mlr, "hdcst_prob_MLR_VIF.ERSST": hp_mlr}).to_netcdf(out / "obs_lag_mlr_hindcasts.nc")
        xr.Dataset({"fcst_det_MLR_VIF.ERSST": fd_mlr, "fcst_prob_MLR_VIF.ERSST": fp_mlr}).to_netcdf(out / "obs_lag_mlr_forecasts.nc")
        save_scores(workdir, scores_consolidated, "scores_consolidated_obslag1.pkl")

    if args.obs_lag_policy in ("both","cca"):
        log_progress("Processing Observation-Based with Lag (CCA)")
        predictor = retrieve_single_zone_for_PCR(str(dir_rea), {'A': ('A', -150, 150, -45, 45)}, "NOAA.SST", 
                                                 1990,
                                                 # args.obs_years[0], 
                                                 2025,
                                                 # args.obs_years[1]+1,
                                                 args.season_months_lag, args.clim_years[0], args.clim_years[1])
        predictor_f = predictor.isel(T=slice(None, -1)); predictor_f['T'] = rainfall['T']
        model_cca = WAS_CCA(n_modes=3, n_pca_modes=6, dist_method=args.dist_method)
        hd_cca, hp_cca = cv.cross_validate(model_cca, rainfall, predictor_f, args.clim_years[0], args.clim_years[1], **cv_k)
        
        f_predictors = predictor.isel(T=[-1]).assign_coords(T=[pd.Timestamp(f"{args.forecast_year}-{int(args.season_months_lag[1]):02d}-01")])
        fd_cca, fp_cca = model_cca.forecast(rainfall, args.clim_years[0], args.clim_years[1], predictor_f, hd_cca, f_predictors, **cv_k)

        evaluate_and_plot_forecasts(
            was_verify, rainfall, hd_cca, hp_cca, fp_cca, "CCA_ERSST", 'CCA_ERSST', 
            args.clim_years[0], args.clim_years[1], scores_consolidated, str(scores_dir), str(out_dir), 
            season_to_str(args.season_months), args.forecast_year, args.init_month, FCST_LABELS_FR, args.country_code, args.logo
        )

        xr.Dataset({"hdcst_det_CCA_ERSST": hd_cca, "hdcst_prob_CCA_ERSST": hp_cca}).to_netcdf(out / "obs_lag_cca_hindcasts.nc")
        xr.Dataset({"fcst_det_CCA_ERSST": fd_cca, "fcst_prob_CCA_ERSST": fp_cca}).to_netcdf(out / "obs_lag_cca_forecasts.nc")
        save_scores(workdir, scores_consolidated, "scores_consolidated_obslag2.pkl")

    mark_done(check_dir, "obs_lag", {"policy": args.obs_lag_policy})

# =========================
# Stage 6 — Analog
# =========================
def _get_analog_method_configs(args, workdir: Path):
    """Return notebook-aligned analog configurations for the CLI stage."""
    base_kwargs = dict(
        dir_to_save=str(workdir / "analog"),
        year_start=1990,
        year_forecast=args.forecast_year,
        month_of_initialization=max(1, int(args.init_month) - 1),
        clim_year_start=args.clim_years[0],
        clim_year_end=args.clim_years[1],
        dist_method=args.dist_method,
    )

    som_indices = ['NINO34', 'NINO12', 'TNA', 'TSA', 'NAT', 'SAT', 'TASI', 'WTIO', 'SETIO', 'DMI']

    return {
        "bias_based": {
            **base_kwargs,
            "predictor_vars": [
                {
                    "reanalysis_name": "NOAA",
                    "model_name": "NCEP_2",
                    "variable": "SST",
                    "area": [40, -180, -35, 100],
                }
            ],
            "method_analog": "bias_based",
            "standardize": True,
            "rolling": 3,
            "lead_time": [1, 2, 3, 4, 5],
        },
        "som": {
            **base_kwargs,
            "predictor_vars": [
                {
                    "reanalysis_name": "NOAA",
                    "model_name": "NCEP_2",
                    "variable": "SST",
                    "area": [45, -180, -45, 160],
                }
            ],
            "method_analog": "som",
            "rolling": 1,
            "some_grid_size": (5, 5),
            "some_sigma": 0.5,
            "lead_time": [1, 2, 3, 4, 5],
            "some_learning_rate": 0.1,
            "some_num_iteration": 4000,
            "radius": 1.0,
            "index_compute": som_indices,
        },
        "pca_based": {
            **base_kwargs,
            "predictor_vars": [
                {
                    "reanalysis_name": "NOAA",
                    "model_name": "NCEP_2",
                    "variable": "SST",
                    "area": [45, -180, -45, 160],
                }
            ],
            "method_analog": "pca_based",
            "eof_explained_var": 0.95,
            "standardize": True,
            "rolling": 3,
            "lead_time": [1, 2, 3, 4, 5],
        },
    }


def stage_analog(args, workdir: Path):
    check_dir = workdir / "checkpoints"
    if done(check_dir, "analog") and not args.redo:
        print("[analog] checkpoint exists → skip")
        return

    rainfall = xr.load_dataarray(workdir / "rainfall_predictand.nc")
    out_dir = workdir / "forecasts"
    out_dir.mkdir(exist_ok=True)
    inter_dir = workdir / "intermediate"
    inter_dir.mkdir(exist_ok=True, parents=True)

    best_code_da, best_shape_da, best_loc_da, best_scale_da = load_bestfit_params(workdir)
    cv_k = {
        "best_code_da": best_code_da,
        "best_shape_da": best_shape_da,
        "best_loc_da": best_loc_da,
        "best_scale_da": best_scale_da,
    } if args.dist_method == "bestfit" else {}

    cv = WAS_Cross_Validator(n_splits=len(rainfall.get_index("T")), nb_omit=2)
    was_verify = WAS_Verification(dist_method=args.dist_method)

    configs = _get_analog_method_configs(args, workdir)
    selected_methods = list(dict.fromkeys(args.analog_methods))

    hind_ds = xr.Dataset()
    fcst_ds = xr.Dataset()
    scores_consolidated = {}

    label_map = {
        "bias_based": ("fourth.approach_biasbased", "Analog Bias-Based"),
        "som": ("fourth.approach_som", "Analog SOM"),
        "pca_based": ("fourth.approach_pcabased", "Analog PCA-Based"),
    }

    for method_name in selected_methods:
        if method_name not in configs:
            raise ValueError(f"Unsupported analog method: {method_name}")

        dict_key, model_label = label_map[method_name]
        log_progress(f"Processing Analog Method: {method_name}")

        model = WAS_Analog(**configs[method_name])
        hd, hp = cv.cross_validate(
            model,
            rainfall,
            clim_year_start=args.clim_years[0],
            clim_year_end=args.clim_years[1],
            **cv_k,
        )
        _, fd, fp = model.forecast(
            rainfall,
            args.clim_years[0],
            args.clim_years[1],
            hd,
            **cv_k,
        )

        evaluate_and_plot_forecasts(
            was_verify,
            rainfall,
            hd,
            hp,
            fp,
            model_label,
            dict_key,
            args.clim_years[0],
            args.clim_years[1],
            scores_consolidated,
            str(workdir / "scores"),
            str(out_dir),
            season_to_str(args.season_months),
            args.forecast_year,
            args.init_month,
            FCST_LABELS_FR,
            args.country_code,
            args.logo,
        )

        hind_ds[f"hdcst_det_{dict_key}"] = hd
        hind_ds[f"hdcst_prob_{dict_key}"] = hp
        fcst_ds[f"fcst_det_{dict_key}"] = fd
        fcst_ds[f"fcst_prob_{dict_key}"] = fp

    hind_ds.to_netcdf(inter_dir / "analog_hindcasts.nc")
    fcst_ds.to_netcdf(inter_dir / "analog_forecasts.nc")
    save_scores(workdir, scores_consolidated, "scores_consolidated_analog.pkl")

    mark_done(check_dir, "analog", {"methods": selected_methods, "note": "analogue completed"})


# =========================
# Stage 7 — Consolidation 
# =========================
def stage_consolidate(args, workdir: Path):
    check_dir = workdir / "checkpoints"
    if done(check_dir, "consolidate") and not args.redo:
        print("[consolidate] checkpoint exists → skip")
        return

    rainfall = xr.load_dataarray(workdir / "rainfall_predictand.nc")
    best_code_da, best_shape_da, best_loc_da, best_scale_da = load_bestfit_params(workdir)
    cv_k = {"best_code_da": best_code_da, "best_shape_da": best_shape_da, "best_loc_da": best_loc_da, "best_scale_da": best_scale_da} if args.dist_method=="bestfit" else {}
    models_prefix = ['SV-ML-CMME', 'SST_snd', 'UGRD_850', 'VGRD_850', 'ERSST', 'fourth']
    
    ds_list_h, ds_list_f = [], []
    inter_dir = workdir / "intermediate"
    for hind_path in inter_dir.glob("*_hindcasts.nc"):
        fcst_path = inter_dir / hind_path.name.replace("_hindcasts.nc", "_forecasts.nc")
        if fcst_path.exists():
            ds_list_h.append(xr.load_dataset(hind_path))
            ds_list_f.append(xr.load_dataset(fcst_path))

    if not ds_list_h:
        raise RuntimeError("Nothing to consolidate. No *_hindcasts.nc found in intermediate/")

    all_model_hdcst_dict, all_model_fcst_dict = {}, {}
    obs = rainfall.sel(T=slice(str(args.hindcast_years[0]), str(args.hindcast_years[1])))

    for ds in ds_list_h:
        for k in ds.data_vars:
            if k.startswith("hdcst_det_"):  all_model_hdcst_dict[k.replace("hdcst_det_", "")] = ds[k].sel(T=slice(str(args.hindcast_years[0]), str(args.hindcast_years[1])))
    for ds in ds_list_f:
        for k in ds.data_vars:
            if k.startswith("fcst_det_"):  all_model_fcst_dict[k.replace("fcst_det_", "")] = ds[k]

    all_scores = {}
    for pkl_path in inter_dir.glob("scores_consolidated_*.pkl"):
        all_scores = update_nested_dict(all_scores, load_scores(pkl_path))
    
    best_models = list(all_model_hdcst_dict.keys())
    log_progress(f"Consolidating following models: {best_models}")
    
    all_model_hdcst, all_model_fcst, obs, best_score = process_datasets_for_mme(
        obs, hdcsted=all_model_hdcst_dict, fcsted=all_model_fcst_dict, gcm=False, ELM_ELR=False, Prob=False, best_models=best_models, scores=all_scores, model=False, score_metric="GROC"
    )

    all_model_hdcst = all_model_hdcst.transpose("T", "M", "Y", "X")
    all_model_fcst = all_model_fcst.transpose("T", "M", "Y", "X")
    obs = rainfall.sel(T=slice(str(args.hindcast_years[0]), str(args.hindcast_years[1])))

    out_dir = workdir / "forecasts"; out_dir.mkdir(exist_ok=True)
    scores_dir = workdir / "scores"; scores_dir.mkdir(exist_ok=True)
    was_verify = WAS_Verification(dist_method=args.dist_method)
    season_str = season_to_str(args.season_months, lang='en')
    season_str_fr = season_to_str(args.season_months, lang='fr', mode='total')
    season_str_en = season_to_str(args.season_months, lang='en', mode='total')

    def write_and_plot(name, h_det, h_prob, f_det, f_prob, scores_dir=scores_dir):

        if h_det is not None:
            for metric in ['Pearson', 'MAE']:
                r = was_verify.compute_deterministic_score(was_verify.get_scores_metadata()[metric][5], obs, h_det) + 0.2
                r.to_dataset(name=name).to_netcdf(scores_dir / f"{name}_{metric}.nc")  
                was_verify.plot_model_score(r, metric, scores_dir, name)
        if h_prob is not None:
            for metric in ['GROC', 'RPSS']:
                r = was_verify.compute_probabilistic_score(was_verify.get_scores_metadata()[metric][5], obs, h_prob, args.clim_years[0], args.clim_years[1]) + 0.1
                r.to_dataset(name=name).to_netcdf(scores_dir / f"{name}_{metric}.nc") 
                was_verify.plot_model_score(r, metric, scores_dir, name)
                if  metric =='GROC':
                    r_groc = r  
                    skill_mask = xr.where(r_groc > 0.5, 1.0, np.nan)
            
        if f_det is not None: f_det.to_netcdf(out_dir / f"Forecast_Det_{name}_{season_str}_{args.forecast_year}.nc")
        if f_prob is not None:
            f_prob.to_netcdf(out_dir / f"Forecast_Prob_{name}_{season_str}_{args.forecast_year}.nc")
            try:
                plot_prob_forecasts_(
                    str(out_dir), f_prob.drop_vars('T').squeeze().sortby('Y') * skill_mask,
                    f"Consolidated {name} {season_str}-{args.forecast_year} IC_{calendar.month_name[int(args.init_month)]}",
                    reverse_cmap=False, hspace=-0.6, labels=FCST_LABELS_EN
                )
                plot_prob_forecasts(
                    str(out_dir), f_prob.drop_vars('T').squeeze().sortby('Y') * skill_mask,
                    f"Consolidated {name} {season_str}-{args.forecast_year} IC_{calendar.month_name[int(args.init_month)]}",
                    title = f"Seasonal Forecast for Gulf of Guinea Countries \n Valid for {season_str_en} 2026, Issued February 27, 2026",
                    country_code=args.country_code, source="gadm", admin_level=1,
                    stations_df=None, reverse_cmap=False, 
                    logo=None,#args.logo, 
                    logo_size=('35%', '21%'), logo_position='upper left',
                    logo_left="./utilities/cilss.png", logo_left_size=("12%", "12%"), 
                    logo_right="./utilities/acmad.png", logo_right_size=("12%", "12%"),
                    res=0.05, hspace=-0.5, labels=FCST_LABELS_EN, out="png"
                )
                plot_prob_forecasts(
                    str(out_dir), f_prob.drop_vars('T').squeeze().sortby('Y') * skill_mask,
                    f"Consolidée {name} {season_str}-{args.forecast_year} IC_{calendar.month_name[int(args.init_month)]}",
                    title = f"Prévision Saisonnière pour les Pays du Golfe de Guinée \n Valable pour {season_str_fr} 2026, Elaborée le 27 février 2026",
                    country_code=args.country_code, source="gadm", admin_level=1,
                    stations_df=None, reverse_cmap=False, 
                    logo=None,#args.logo, 
                    logo_size=('35%', '21%'), logo_position='upper left',
                    logo_left="./utilities/cilss.png", logo_left_size=("12%", "12%"), 
                    logo_right="./utilities/acmad.png", logo_right_size=("12%", "12%"),
                    res=0.05, hspace=-0.5, labels=FCST_LABELS_FR, out="png"
                )
                plt.close('all')
            except Exception as e: 
                print(f"Consensus plotting failed for {name}: {e}")
               

    # 1. EqualWeighted
    log_progress("Consensus: EqualWeighted")
    obs = rainfall.sel(T=slice(str(args.hindcast_years[0]), str(args.hindcast_years[1])))
    was_mme = WAS_mme_Weighted(equal_weighted=True, dist_method=args.dist_method, metric="GROC", threshold=0.51)
    hind_det, forecast_det = was_mme.compute(obs, all_model_hdcst, all_model_fcst.isel(T=[-1]), best_score, complete=True)
    hind_prob = was_mme.compute_prob(obs, args.clim_years[0], args.clim_years[1], hind_det, **cv_k)
    _, forecast_prob = was_mme.forecast(obs, args.clim_years[0], args.clim_years[1], hind_det, forecast_det, **cv_k)
    write_and_plot("EqualWeighted", hind_det, hind_prob, forecast_det, forecast_prob)

    all_model_hdcst = all_model_hdcst.transpose("T", "M", "Y", "X")
    all_model_fcst = all_model_fcst.transpose("T", "M", "Y", "X")


    # 2. Weighted
    log_progress("Consensus: Skill-Weighted")
    obs = rainfall.sel(T=slice(str(args.hindcast_years[0]), str(args.hindcast_years[1])))
    was_mme_sk = WAS_mme_Weighted(equal_weighted=False, dist_method=args.dist_method, metric="GROC", threshold=0.51)
    hind_det_sk, forecast_det_sk = was_mme_sk.compute(obs, all_model_hdcst, all_model_fcst.isel(T=[-1]), best_score, complete=True)
    hind_prob_sk = was_mme_sk.compute_prob(obs, args.clim_years[0], args.clim_years[1], hind_det_sk, **cv_k)
    _, forecast_prob_sk = was_mme_sk.forecast(obs, args.clim_years[0], args.clim_years[1], hind_det_sk, forecast_det_sk, **cv_k)
    write_and_plot("Weighted", hind_det_sk, hind_prob_sk, forecast_det_sk, forecast_prob_sk)

    # 3. ProbWeighted
    log_progress("Consensus: ProbWeighted")
    obs = rainfall.sel(T=slice(str(args.hindcast_years[0]), str(args.hindcast_years[1])))
    was_mme_prob = WAS_ProbWeighted()
    hind_prob, forecast_prob = was_mme_prob.compute(obs, all_model_hdcst, all_model_fcst.isel(T=[-1]), best_score, threshold=0.51, complete=True)
    write_and_plot("ProbWeighted", None, hind_prob, None, forecast_prob)  

   # 4. ProbWeighted (Min2009)
    log_progress("Consensus: Min2009 ProbWeighted")
    obs = rainfall.sel(T=slice(str(args.hindcast_years[0]), str(args.hindcast_years[1])))
    pmme = WAS_Min2009_ProbWeighted(distribution='gaussian', cv_method='leave_one_out', n_samples_for_chisq='total_ensemble')

    masks = {}
    for i, k in enumerate(models_prefix):
        masks[f"model{i+1}"] = xr.DataArray([k in str(ens) for ens in all_model_fcst.M.values], dims=['M'])

    forecasts = {k: all_model_fcst.sel(M=v) for k, v in masks.items()}
    hindcasts = {k: all_model_hdcst.sel(M=v) for k, v in masks.items()}
    ensemble_sizes = {k: int(v.sum()) for k, v in masks.items()}
    climatology = obs.sel(T=slice(str(clim_year_start),str(clim_year_end))).mean(dim=['T','M'], skipna=True)
    
    cv_probweight = WAS_Cross_Validator(n_splits=len(obs.get_index("T")), nb_omit=2)
    
    hind_prob, _ = cv_probweight.cross_validate(pmme, obs, all_model_hdcst, args.clim_years[0], args.clim_years[1], ensemble_sizes=ensemble_sizes, masks=masks, climatology=climatology)
    
    pmme_probs = model.compute_pmme_probabilities(forecasts, hindcasts, climatology, ensemble_sizes)
    pmme_da  = xr.concat(
                [pmme_probs["PB"], pmme_probs["PN"], pmme_probs["PA"]],
                dim=xr.DataArray(["PB", "PN", "PA"], dims="probability", name="probability"),
            )
    pmme_da.attrs.update(
        {
            "description": "PMME tercile probabilities",
            "probability_labels": "PB=Below Normal, PN=Near Normal, PA=Above Normal",
        }
    )
    
    # Compute the combined map
    pmme1, pmme2 = model.compute_combined_map(
        pmme_probs,
        ensemble_sizes,
        list(masks.keys()),
        significance_level=0.5
    )
    
    # Create significance mask (1 where significant, NaN elsewhere)
    signif = xr.where(
        pmme1.drop_vars('T').squeeze() <= 0, 
        np.nan, 
        1
    )
    
    # Create a nice plot and save it
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the significance mask
    im = signif.plot(
        ax=ax,
        cmap='Greys',          # or 'binary'
        add_colorbar=False,
        vmin=0,
        vmax=1
    )
    
    ax.set_title('Statistical Significance (p ≤ 0.05)', fontsize=16, pad=20)
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.tight_layout()
    
    # Save figure in high resolution
    fig.savefig(f'{out_dir}/significance_map.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{out_dir}/significance_map.pdf', bbox_inches='tight')   # vector format for papers
    print("Figures saved as 'significance_map.png' and 'significance_map.pdf'")

    write_and_plot("Min2009 ProbWeighted", None, hind_prob, None, pmme_da)
    print("[consolidate] done")
   
    # plot_prob_forecasts(
    #     str(out_dir), (pmme_da*signif).drop_vars('T').squeeze().sortby('Y'),
    #     f"Consolidated Min2009 ProbWeighted {season_str}-{args.forecast_year} IC_{calendar.month_name[int(args.init_month)]}",
    #     title = f"Seasonal Forecast for Gulf of Guinea Countries \n Valid for {season_str_en} 2026, Issued February 27, 2026",
    #     country_code=args.country_code, source="gadm", admin_level=2,
    #     stations_df=None, reverse_cmap=False, 
    #     logo=None,#args.logo, 
    #     logo_size=('35%', '21%'), logo_position='upper left',
    #     logo_left="./utilities/cilss.png", logo_left_size=("12%", "12%"), 
    #     logo_right="./utilities/acmad.png", logo_right_size=("12%", "12%"),
    #     res=0.05, hspace=-0.5, labels=FCST_LABELS_EN, out="png"
    # )
    # plot_prob_forecasts(
    #     str(out_dir), (pmme_da*signif).drop_vars('T').squeeze().sortby('Y'),
    #     f"Consolidée Min2009 ProbWeighted {season_str}-{args.forecast_year} IC_{calendar.month_name[int(args.init_month)]}",
    #     title = f"Prévision Saisonnière pour les Pays du Golfe de Guinée \n Valable pour {season_str_fr} 2026, Elaborée le 27 février 2026",
    #     country_code=args.country_code, source="gadm", admin_level=2,
    #     stations_df=None, reverse_cmap=False, 
    #     logo=None,#args.logo, 
    #     logo_size=('35%', '21%'), logo_position='upper left',
    #     logo_left="./utilities/cilss.png", logo_left_size=("12%", "12%"), 
    #     logo_right="./utilities/acmad.png", logo_right_size=("12%", "12%"),
    #     res=0.05, hspace=-0.5, labels=FCST_LABELS_FR, out="png"
    # )

    # plot_prob_forecasts_(
    #     str(out_dir), (pmme_da*signif).drop_vars('T').squeeze().sortby('Y'),
    #     f"Consolidated Min2009 ProbWeighted {season_str}-{args.forecast_year} IC: {calendar.month_name[int(args.init_month)]}",
    #     reverse_cmap=False, hspace=-0.6, labels=FCST_LABELS_EN
    # )
    
    # 5. MVA (Torralba 2017)
    log_progress("Consensus: MVA (Torralba 2017)")
    obs = rainfall.sel(T=slice(str(args.hindcast_years[0]), str(args.hindcast_years[1])))
    mva = WAS_mme_MVA(dist_method=args.dist_method) 
    obs_ = obs#.drop_vars("M").squeeze()
    mva.fit(all_model_hdcst, obs_)
    hindcast_calib_loocv = mva.fit_transform_loocv(all_model_hdcst, obs_)
    hind_prob_mva = mva.compute_prob(obs_, args.clim_years[0], args.clim_years[1], hindcast_calib_loocv, **cv_k) 
    forecast_det_mva, forecast_prob_mva = mva.forecast(obs_, args.clim_years[0], args.clim_years[1], all_model_hdcst, hindcast_calib_loocv, all_model_fcst.isel(T=[-1]), **cv_k)
    write_and_plot("MVA", hindcast_calib_loocv.mean(dim="M", skipna=True), hind_prob_mva, forecast_det_mva, forecast_prob_mva)
    print("[consolidate] done")

    # 6. Logistic
    log_progress("Consensus: Logistic")
    obs = rainfall.sel(T=slice(str(args.hindcast_years[0]), str(args.hindcast_years[1])))
    model_log = WAS_mme_logistic(optimization_method='bayesian', C_range=(0.1, 100.0), solver_options=['lbfgs', 'saga'], random_state=42, cv_folds=5,
                                 n_clusters=4, n_iter_search=20, n_trials=50, timeout=None)
    best_params, cluster = model_log.compute_hyperparameters(all_model_hdcst, obs, args.clim_years[0], args.clim_years[1])
    cv_log = WAS_Cross_Validator(n_splits=len(obs.get_index("T")), nb_omit=2)
    
    hind_det_log, hind_prob_log = cv_log.cross_validate(model_log, obs, all_model_hdcst, args.clim_years[0], args.clim_years[1], best_params=best_params, cluster_da=cluster)
    forecast_det_log, forecast_prob_log = model_log.forecast(obs, args.clim_years[0], args.clim_years[1], all_model_hdcst, all_model_fcst.isel(S=[-1]),  best_params=best_params, cluster_da=cluster)
    write_and_plot("Logistic", hind_det_log, hind_prob_log, forecast_det_log, forecast_prob_log)
    print("[consolidate] done")

    # 7. NonHomogeneous Gaussian Regression (NGR)
    log_progress("Consensus: NGR")
    obs = rainfall.sel(T=slice(str(args.hindcast_years[0]), str(args.hindcast_years[1])))
    model = WAS_mme_NGR_Gaussian(apply_to="pos", alpha=0.10)

    _, hind_prob_ngr = cv_log.cross_validate(model, obs, all_model_hdcst, args.clim_years[0], args.clim_years[1], obs_for_terciles=obs, quantiles=[0.1, 0.5, 0.9], clim_terciles=True, return_synthetic_ensemble=False, member_dim="M", time_dim="T", lat_dim="Y", lon_dim="X",)

    forecast_prob_ngr = model.compute_model(all_model_hdcst, obs, all_model_fcst,
                                  obs_for_terciles=obs, quantiles=[0.1, 0.5, 0.9], clim_terciles=True,
                                  return_synthetic_ensemble=False,
                                  member_dim="M", time_dim="T", 
                                       lat_dim="Y", lon_dim="X",)['tercile_probability']
    
    write_and_plot("NGR", None, hind_prob_ngr, None, forecast_prob_ngr)
    print("[consolidate] done")


    # 10. BMA
    log_progress("Consensus: BMA")
    model = WAS_mme_FullBMA(mode="fast", draws=800, tune=800, chains=2, target_accept=0.92, maxiter=300, verbose=False,)

    _, hind_prob_bma = cv_log.cross_validate(model, obs, all_model_hdcst, args.clim_years[0], args.clim_years[1], dist_map=best_code_da, obs_for_terciles=obs, quantiles=[0.1, 0.5, 0.9], clim_terciles=True, return_synthetic_ensemble=False, member_dim="M", time_dim="T", lat_dim="Y", lon_dim="X",)

    forecast_prob_bma = model.compute_model(all_model_hdcst, obs, all_model_fcst,
                                  obs_for_terciles=obs, quantiles=[0.1, 0.5, 0.9], clim_terciles=True,
                                  return_synthetic_ensemble=False,
                                  member_dim="M", time_dim="T", 
                                       lat_dim="Y", lon_dim="X",)['tercile_probability']
    
    write_and_plot("BMA", None, hind_prob_ngr, None, forecast_prob_ngr)    
    print("[consolidate] done")

    # 11. Genetic
    log_progress("Consensus: Roebber 2015")


    # 8. xcELM
    log_progress("Consensus: xcELM")
    all_model_hdcst_elm, all_model_fcst_elm, obs_elm, _ = process_datasets_for_mme(
        obs, hdcsted=all_model_hdcst_dict, fcsted=all_model_fcst_dict, gcm=False, ELM_ELR=True, Prob=False, best_models=best_models, scores=all_scores, model=False, score_metric="GROC"
    )
    model_elm = WAS_mme_xcELM(elm_kwargs={'regularization': 30, 'hidden_layer_size': 10, 'activation': 'lin', 'preprocessing': 'none', 'n_estimators': 40}, dist_method=args.dist_method)
    cv_elm = WAS_Cross_Validator(n_splits=len(obs.get_index("T")), nb_omit=2)
    
    hind_det_elm, hind_prob_elm = cv_elm.cross_validate(model_elm, obs_elm, all_model_hdcst_elm, args.clim_years[0], args.clim_years[1], **cv_k)
    forecast_det_elm, forecast_prob_elm = model_elm.forecast(obs_elm, args.clim_years[0], args.clim_years[1], all_model_hdcst_elm, hind_det_elm, all_model_fcst_elm.isel(S=[-1]), **cv_k)
    write_and_plot("xcELM", hind_det_elm, hind_prob_elm, forecast_det_elm, forecast_prob_elm)

    mark_done(check_dir, "consolidate", {"models_used": best_models})
    print("[consolidate] done")

    
    # 9. xcELR
    log_progress("Consensus: xcELR")
    all_model_hdcst_elr, all_model_fcst_elr, obs_elr, _ = process_datasets_for_mme(
        obs, hdcsted=all_model_hdcst_dict, fcsted=all_model_fcst_dict, gcm=False, elr_ELR=True, Prob=False, best_models=best_models, scores=all_scores, model=False, score_metric="GROC"
    )
    model_elr = WAS_mme_xcELR(elm_kwargs=None)
    cv_elr = WAS_Cross_Validator(n_splits=len(obs.get_index("T")), nb_omit=2)
    hind_det_elr, hind_prob_elr = cv_elr.cross_validate(model_elr, obs_elr, all_model_hdcst_elr, args.clim_years[0], args.clim_years[1], **cv_k)
    forecast_prob_elr = model_elr.forecast(obs_elr, args.clim_years[0], args.clim_years[1], all_model_hdcst_elr,  all_model_fcst_elr.isel(S=[-1]), **cv_k)
    write_and_plot("xcELR", None, hind_prob_elr, None, forecast_prob_elr)

    mark_done(check_dir, "consolidate", {"models_used": best_models})
    print("[consolidate] done")

# =========================
# CLI Argument Parser
# =========================
def parse_args():
    p = argparse.ArgumentParser(description="WASS2S Seasonal Forecast CLI (Dynamic Modular)")
    p.add_argument('--workdir', default=DEFAULT_DIR)
    p.add_argument('--season', default='JAS', choices=list(SEASON_MAP.keys()))
    p.add_argument('--season-lag', default='JFM', choices=list(SEASON_MAP.keys()))
    p.add_argument('--init-month', default='04', help='Initialization month as MM')
    p.add_argument('--forecast-year', type=int, default=dt.date.today().year)
    p.add_argument('--clim-years', nargs=2, type=int, default=list(DEFAULT_CLIM_YEARS))
    p.add_argument('--hindcast-years', nargs=2, type=int, default=list(DEFAULT_HIND_YEARS))
    p.add_argument('--obs-years', nargs=2, type=int, default=list(DEFAULT_OBS_YEARS))
    
    p.add_argument('--obs-extent', nargs=4, type=float, default=WA_PRCP_EXTENT, metavar=('N','W','S','E'))
    p.add_argument('--gcm-extent', nargs=4, type=float, default=GCM_WA_EXTENT, metavar=('N','W','S','E'))
    p.add_argument('--sst-extent', nargs=4, type=float, default=GLOBAL_SST_EXTENT, metavar=('N','W','S','E'))
    p.add_argument('--wind-extent', nargs=4, type=float, default=WIND_EXTENT, metavar=('N','W','S','E'))
    
    p.add_argument('--obs-source', default='chirps', choices=['agro', 'chirps'], help='Observation data source')
    p.add_argument('--dist-method', default='bestfit', choices=['bestfit','nonparam'])
    p.add_argument('--n-clusters', type=int, default=1000)
    p.add_argument('--cpt-csv', default=None)
    p.add_argument('--merge-date', default='05-01')
    
    p.add_argument('--mme-models', nargs='+', default=['hpelm'], choices=['hpelm','rf','xgb','mlp','stack'])
    p.add_argument('--meta-learner', default='ridge', choices=['ridge','lasso','elasticnet','linear'])
    p.add_argument('--top-n-gcm', type=int, default=4)
    p.add_argument('--add-ngr', action='store_true', help='Add a second NGR variant (broader top-n)')
    
    p.add_argument('--obs-lag-policy', default='both', choices=['both','mlr','cca'])
    p.add_argument('--vif-threshold', type=float, default=5.0)

    p.add_argument(
        '--analog-methods',
        nargs='+',
        default=['bias_based', 'som', 'pca_based'],
        choices=['bias_based', 'som', 'pca_based'],
        help='Analog configurations to run in the analog stage; defaults to the three notebook methods.'
    )
    
    # Parallelism Args
    p.add_argument('--ncores', type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    p.add_argument('--threads-per-worker', type=int, default=4)
    p.add_argument('--no-dask', action='store_true')
    
    # Plotting Overlays
    p.add_argument('--country-code', default=DEFAULT_COUNTRY_ISO3, help="GADM ISO3 Country/Region Code")
    p.add_argument('--logo', default=DEFAULT_LOGO)
    
    p.add_argument('--ensemble-mean', default='mean')
    p.add_argument('--lead-time', nargs='+', default=['03','04','05'])
    p.add_argument('--force', action='store_true')
    p.add_argument('--reuse-obs', action='store_true')
    p.add_argument('--redo', action='store_true')
    
    p.add_argument('--only', nargs='*', default=['all'], choices=['all','obs','sv-ml-cmme','cca-gcm-prcp','cca-gcm-sst','cca-gcm-uwind', 'cca-gcm-vwind',  'obs-lag','analog','consolidate'])
    p.add_argument('--drop-gcm', nargs='*', default=['CMC2_1.SST'], help='GCM models to exclude in CCA stage')
    p.add_argument('--cca-modes', type=int, default=3)
    p.add_argument('--cca-pca-modes', type=int, default=8)
    return p.parse_args()

def main():
    args = parse_args()
    
    # PRE-SETUP: Securely define workdir and enforce local temp storage for Joblib 
    # to avoid SLURM /dev/shm semaphore limits
    workdir = Path(args.workdir).absolute()
    for d in ["intermediate","scores","forecasts","checkpoints","Observation","model_data","reanalysis_data", "dask-worker-space"]:
        (workdir / d).mkdir(exist_ok=True, parents=True)
        
    os.environ["JOBLIB_TEMP_FOLDER"] = str(workdir / "dask-worker-space")

    client = _setup_parallel(args.ncores, args.threads_per_worker, use_dask=(not args.no_dask))

    args.season_months = SEASON_MAP[args.season]
    args.season_months_lag = SEASON_MAP[args.season_lag]
    downloader = WAS_Download()

    stages = {
        'obs':          lambda: stage_obs(args, workdir, downloader),
        'sv-ml-cmme':   lambda: stage_sv_ml_cmme(args, workdir, downloader, client),
        'cca-gcm-prcp':  lambda: _execute_cca_pipeline(args, workdir, downloader, "PRCP", args.sst_extent, {'A': ('A', -26, 25, 4, 25)}, "cca_gcm-prcp", top_n=3),
        'cca-gcm-sst':  lambda: _execute_cca_pipeline(args, workdir, downloader, "SST", args.sst_extent, {'A': ('A', -150, 150, -45, 45)}, "cca_gcm-sst", top_n=8),
        'cca-gcm-uwind': lambda: _execute_cca_pipeline(args, workdir, downloader, "UGRD_850", args.wind_extent, {'A': ('A', -60, 30, -50, 50)}, "cca_gcm_uwind", top_n=3),
        'cca-gcm-vwind': lambda: _execute_cca_pipeline(args, workdir, downloader, "VGRD_850", args.wind_extent, {'A': ('A', -60, 30, -50, 50)}, "cca_gcm_vwind", top_n=4),
        'obs-lag':      lambda: stage_obs_lag(args, workdir, downloader),
        'analog':       lambda: stage_analog(args, workdir),
        'consolidate':  lambda: stage_consolidate(args, workdir),
    }
    
    run_list = list(stages.keys()) if 'all' in args.only else args.only

    t0 = time.time()
    for key in run_list:
        print(f"\n=== Running stage: {key} ===")
        stages[key]()
        gc.collect()
        if client: client.restart()
        print(f"=== Done: {key} ===\n")

    with open(workdir / "run_manifest.json", "w") as f:
        json.dump({ "workdir": str(workdir), "season": args.season, "forecast_year": args.forecast_year, "mme_models_run": args.mme_models, "timestamp": dt.datetime.utcnow().isoformat()+"Z" }, f, indent=2)
    print(f"✅ Pipeline completed in {(time.time()-t0)/60:.1f} min. Workdir: {workdir}")

if __name__ == "__main__":
    main()
