"""
WASS2S Seasonal Forecast CLI — Modular Rewrite
- Approach 1 (SV-ML-CMME): rf, xgb, mlp, hpelm (stacking removed; ML grids refreshed)
- Approach 2 (CCA-GCM): WAS_CCA_op with get_best_models() thresholding and JSON reporting
- Approach 3 (OBS-LAG): MLR (VIF indices), CCA (SST), and PCR (multivariate SST+Wind+Temp)
  with both plain regression and Ridge back-ends
- Approach 5 (EOF-KCCA): fold-safe multi-field EOF reduction followed by KCCA
- Consolidation dynamically discovers all generated intermediate datasets and explores
  a large family of MME consensus methods:
    WAS_mme_Weighted, WAS_ProbWeighted, WAS_Min2009_ProbWeighted, WAS_mme_logistic,
    WAS_mme_ELR, WAS_mme_gaussian_process, WAS_mme_NGR_Gaussian, WAS_mme_NGR_NonGaussian,
    WAS_mme_MVA, WAS_mme_FastBMA, WAS_mme_FullBMA, WAS_mme_PCR, WAS_mme_ExpertKnowledge
- Supports dynamic switching between AgroIndicators and CHIRPS for observations
- Uses one atomic, non-cumulative score registry per stage and rejects duplicate model IDs
"""

import os, gc, json, time, calendar, datetime as dt, warnings
from pathlib import Path

# Persistent Numba cache.  This is set BEFORE importing WASS2S/numba so the
# compiled verification kernels can be reused across seasonal jobs.
_NUMBA_CACHE_DIR = Path(
    os.environ.setdefault(
        "NUMBA_CACHE_DIR",
        str(Path.home() / ".cache" / "wass2s_numba"),
    )
)
try:
    _NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    # A launcher can still provide another writable NUMBA_CACHE_DIR.
    pass

# -----------------------------------------------------------------------------
# Variable pilotée par cette CLI. Les deux CLI sont issues d'un même gabarit :
# toute logique spécifique passe par cette constante, jamais par un littéral.
# -----------------------------------------------------------------------------
VARIABLE = "PRCP"


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
from datetime import date

# Additional missing imports
from scipy.stats import loguniform
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
import pickle

# ---- WASS2S namespace (user package) ----
from wass2s.was_cross_validate import WAS_Cross_Validator
from wass2s import (
    WAS_Analog,
    WAS_CCA_Regularized,
    WAS_CCA_Regularized_strict,
    WAS_CCA_op,
    WAS_CCA_strict,
    WAS_Download,
    WAS_EOF_KCCA,
    WAS_LinearRegression_Model,
    WAS_Merging,
    WAS_Min2009_ProbWeighted,
    WAS_PCR,
    WAS_ProbWeighted,
    WAS_Ridge_Model,
    WAS_TransformData,
    WAS_Verification,
    WAS_mme_ELR,
    WAS_mme_ExpertKnowledge,
    WAS_mme_FastBMA,
    WAS_mme_FullBMA,
    WAS_mme_MLP,
    WAS_mme_MVA,
    WAS_mme_NGR_Gaussian,
    WAS_mme_NGR_NonGaussian,
    WAS_mme_PCR,
    WAS_mme_RF,
    WAS_mme_Weighted,
    WAS_mme_XGBoosting,
    WAS_mme_gaussian_process,
    WAS_mme_hpELM,
    WAS_mme_logistic,
    compute_sst_indices,
    conditional_nonnegative_clip,
    plot_map,
    plot_prob_forecasts,
    plot_prob_forecasts_,
    prepare_predictand,
    process_datasets_for_mme,
    retrieve_single_zone_for_PCR,
    verify_station_network,
    warmup,
)
from wass2s.utils import get_best_models
from wass2s.was_probability_reference import climatological_terciles
from .entrypoints import DEFAULT_SST_DOMAINS, SEASON_MAP, SST_DOMAINS, _cca_mode_selection
from .native_models import align_lagged_pcr_fields, lagged_pcr_year_bounds
from .selection import (GCM_SELECTION_PROCEDURE, fixed_gcm_roster,
    fixed_gcm_selection_metadata)
from .probability import probability_calibration_mode
from .runtime import (configured_cross_validator, probability_score_kwargs,
    climatology_sample, mean_calibrated_probabilities)
from .ensemble import (ensemble_preparation_kwargs,
    ensure_ensemble_configuration, mean_screening_inputs, physical_ensemble_reduction,
    physical_reader_directory)
from .optional import (requested_centres,
    available_inputs, run_optional, mask_products)

# -----------------------------------------------------------------------------
# Scientific WASS2S extensions (2026-08)
# Direct module imports make the CLI robust even when wass2s.__init__ has not
# yet been updated to re-export every new class.
# -----------------------------------------------------------------------------
import inspect
import importlib

_was_mme_module = importlib.import_module("wass2s.was_mme")
_was_analog_module = importlib.import_module("wass2s.was_analog")
from .analog import (
    get_analog_method_configs,
    analog_stage_signature, analog_checkpoint_matches, analog_training_sample,
    preload_analog_sources,
)
from .telnet import run_telnet_stage
from .pbc import run_pbc_stage

_was_verification_module = importlib.import_module("wass2s.was_verification")

# IMPORTANT: always use the class from the verification module directly.
# A stale wass2s.__init__.py must never silently select an older non-Numba
# WAS_Verification implementation.
WAS_Verification = getattr(_was_verification_module, "WAS_Verification")

try:
    _was_cca2_module = importlib.import_module("wass2s.was_cca_2")
    WAS_EOF_KCCA = getattr(_was_cca2_module, "WAS_EOF_KCCA", None)
except Exception as _eof_kcca_import_error:
    WAS_EOF_KCCA = None


_OPTIONAL_WAS_CLASSES = (
    # revised historical + new consolidation methods; import directly from
    # wass2s.was_mme so a stale package __init__.py cannot select an old class.
    "WAS_mme_Weighted",
    "WAS_ProbWeighted",
    "WAS_Min2009_ProbWeighted",
    "WAS_mme_MVA",
    "WAS_mme_NGR_Gaussian",
    "WAS_mme_NGR_NonGaussian",
    "WAS_mme_PCR",
    "WAS_mme_ExpertKnowledge",
    "WAS_mme_ELR",
    "WAS_mme_logistic",
    "WAS_mme_gaussian_process",
    "WAS_mme_FastBMA",
    "WAS_mme_FullBMA",
    "WAS_mme_WeightedContinuous",
    "WAS_ProbWeightedContinuous",
    "WAS_ProbWeightedRPS",
    "WAS_mme_Ridge",
    "WAS_mme_ElasticNet",
    "WAS_mme_SVR",
    "WAS_mme_ExtraTrees",
    "WAS_mme_HistGradientBoosting",
    "WAS_mme_MARS",
    "WAS_mme_PLS",
    "WAS_mme_CCA",
    "WAS_mme_UNet",
    "WAS_mme_ResCNN",
    # scientific analogue engine
    "WAS_Analog_KNN",
    "WAS_Analog_Weighted",
    "WAS_Analog_EOF",
    "WAS_Analog_Hybrid",
    "WAS_Analog_Constructed",
    "WAS_Analog_LearnedMetric",
    "WAS_Analog_LearnedSpatialMetric",
)
for _class_name in _OPTIONAL_WAS_CLASSES:
    if _class_name.startswith("WAS_Analog_"):
        globals()[_class_name] = getattr(_was_analog_module, _class_name, None)
    else:
        globals()[_class_name] = getattr(_was_mme_module, _class_name, None)


def _require_was_class(name):
    cls = globals().get(name)
    if cls is None:
        raise RuntimeError(
            f"{name} n'est pas disponible dans l'installation WASS2S courante. "
            "Installez le bundle scientifique récent avant d'activer cette méthode."
        )
    return cls



def _configure_numba_runtime(args):
    """Configure Numba globally and REQUIRE the Numba verification backend.

    Verification is intentionally strict: deterministic/probabilistic scores
    must come from ``wass2s.was_verification`` with its @njit kernels.  We do
    not silently fall back to an older pure-Python verification module because
    that would make runtime/performance depend on a stale package export.
    """
    requested = int(
        getattr(args, "numba_threads", 0)
        or min(int(args.ncores), 32)
    )
    requested = max(1, min(requested, int(args.ncores)))

    try:
        import numba
    except Exception as exc:
        raise RuntimeError(
            "Numba est requis pour la verification mais ne peut pas etre "
            f"importe: {exc}"
        ) from exc

    ceiling = int(getattr(numba.config, "NUMBA_NUM_THREADS", requested))
    use = max(1, min(requested, ceiling))
    numba.set_num_threads(use)

    # General Numba + verification-specific switches.
    os.environ["WAS_NUMBA_THREADS"] = str(use)
    os.environ["WAS_NCORES"] = str(int(args.ncores))
    os.environ["WAS_VERIF_NUMBA"] = "1"
    os.environ["WAS_VERIF_PARALLEL"] = "1"
    os.environ["WAS_VERIF_MIN_CELLS"] = str(
        max(0, int(getattr(args, "verification_min_cells", 4000)))
    )

    if not bool(getattr(_was_verification_module, "NUMBA_AVAILABLE", False)):
        raise RuntimeError(
            "Le module wass2s.was_verification importe n'est pas la version "
            "Numba requise (NUMBA_AVAILABLE absent/faux). Installez le "
            "was_verification_numba fourni avec cette CLI."
        )

    use_numba = getattr(_was_verification_module, "_use_numba", None)
    warmup = getattr(_was_verification_module, "warmup", None)
    if not callable(use_numba) or not bool(use_numba()):
        raise RuntimeError(
            "wass2s.was_verification ne confirme pas l'activation Numba. "
            "Verifiez WAS_VERIF_NUMBA et la version du module installe."
        )
    if not callable(warmup):
        raise RuntimeError(
            "wass2s.was_verification ne fournit pas warmup(); "
            "la version Numba attendue n'est pas installee."
        )

    # Compile/load cached kernels once before the first real verification map.
    warmup(verbose=True)

    print(
        "[verification] backend=NUMBA "
        f"module={getattr(_was_verification_module, '__file__', '?')} "
        f"numba_threads={numba.get_num_threads()} "
        f"grid_workers={os.environ['WAS_NCORES']} "
        f"parallel={os.environ['WAS_VERIF_PARALLEL']} "
        f"min_cells={os.environ['WAS_VERIF_MIN_CELLS']} "
        f"cache={os.environ.get('NUMBA_CACHE_DIR', '')}",
        flush=True,
    )
    return True

def _scientific_preflight():
    """Fail early on the known cross-validator scope bug instead of hours later."""
    try:
        src = inspect.getsource(WAS_Cross_Validator.cross_validate)
    except Exception:
        return
    if "inner_inspect.signature" in src:
        raise RuntimeError(
            "was_cross_validate.py contient encore inner_inspect.signature(...). "
            "Installez was_cross_validate_scientific_v8_270826.py."
        )
    # A function-local ``import inspect`` makes inspect a local/free variable and
    # triggered the 2026-08-27 NameError in the SV-ML branch.
    local_imports = [ln for ln in src.splitlines()[1:] if ln.strip() == "import inspect"]
    if local_imports:
        raise RuntimeError(
            "was_cross_validate.py contient encore un import inspect local dans "
            "cross_validate(). Installez la version scientifique v8 corrigée."
        )
    # Recent validators delegate KernelCCA to the fold-local native engine.
    # The old inline variable name is no longer a valid compatibility test.
    cv_module = inspect.getmodule(WAS_Cross_Validator.cross_validate)
    native_cv = getattr(cv_module, "_native_regression_cv", None)
    if "kernel_nested_tuning" not in src and not callable(native_cv):
        raise RuntimeError(
            "was_cross_validate.py ne contient pas encore le nested tuning "
            "KernelCCA requis par la cinquième approche. Installez "
            "was_cross_validate_scientific_v8_270826.py."
        )
 # =========================
# Defaults & Constants
# =========================
FCST_LABELS_FR = ["EN DESSOUS DE LA MOYENNE", "PROCHE DE LA MOYENNE", "AU DESSUS DE LA MOYENNE"]
FCST_LABELS_EN = ["BELOW-AVERAGE", "NEAR-AVERAGE", "ABOVE-AVERAGE"]


def resolve_sst_domains(names):
    """Validate domain names and return an ordered {name: coords} mapping.

    A domain crossing the dateline (lon_min > lon_max, e.g. nino4) cannot be
    expressed as a single sel(X=slice(...)) in the -180..180 convention.
    Reject it explicitly instead of silently returning an empty field.
    """
    unknown = [n for n in names if n not in SST_DOMAINS]
    if unknown:
        raise ValueError(
            f"Domaines SST inconnus: {unknown}. Disponibles: {sorted(SST_DOMAINS)}"
        )
    resolved = {}
    for name in _unique_preserve_order(names, label="--cca-sst-domains"):
        coords = SST_DOMAINS[name]
        _, lon_min, lon_max, _, _ = coords
        if lon_min > lon_max:
            raise ValueError(
                f"Le domaine '{name}' traverse la ligne de changement de date "
                f"({lon_min}..{lon_max}). retrieve_single_zone_for_PCR utilise "
                "sel(X=slice(...)) et renverrait un champ vide. Utilisez "
                "'global_trop', ou re-projetez les longitudes en 0..360 en amont."
            )
        resolved[name] = coords
    return resolved


def assert_predictor_alignment(predictor, predictand, *, label, strict=True):
    """Check that a positionally restamped predictor really matches its target.

    The pipeline restamps the predictor time axis onto the predictand:

        p_train = predictor.isel(T=slice(None, -1))
        p_train['T'] = obs['T']

    That assignment is POSITIONAL. It never verifies that year 1 of the
    predictor is year 1 of the predictand. A one-year offset passes silently,
    and on an ENSO-driven signal a one-year offset is enough to flip the sign
    of the correlation, which is what a coherent negative Pearson map over the
    whole Sahel looks like. This check costs nothing and settles the question.
    """
    py = np.asarray(predictor["T"].dt.year.values)
    oy = np.asarray(predictand["T"].dt.year.values)
    if py.size != oy.size:
        message = (f"{label}: {py.size} pas de temps prédicteur contre "
                   f"{oy.size} prédictande.")
    elif not np.array_equal(py, oy):
        offset = int(np.median(py - oy))
        message = (f"{label}: années désalignées (décalage médian {offset:+d} an). "
                   f"Prédicteur {py[0]}..{py[-1]}, prédictande {oy[0]}..{oy[-1]}.")
    else:
        return True
    if strict:
        raise ValueError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    return False

CHECKPOINTS = {
    "obs": "chk_obs.json",
    "sv_ml_cmme": "chk_sv_ml_cmme.json",
    "cca_gcm": "chk_cca_gcm.json",
    "obs_lag": "chk_obs_lag.json",
    "analog": "chk_analog.json",
    "telnet": "chk_telnet.json",
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

def strip_attrs(da):
    """Remove any DataArray/Dataset attribute that xarray cannot serialise to
    NetCDF (dicts, sets, arbitrary objects). Operates in-place and returns the
    object so it can be used inline.

    NetCDF attributes must be str, Number, ndarray, list, tuple, or bytes.
    Anything else (e.g. {'solver_kwargs': {}}) silently gets dropped here
    rather than crashing to_netcdf.
    """
    import numpy as np
    _VALID = (str, int, float, complex, bytes, list, tuple, np.ndarray, np.number)

    def _clean(attrs: dict) -> dict:
        return {k: v for k, v in attrs.items() if isinstance(v, _VALID)}

    if isinstance(da, xr.Dataset):
        da.attrs = _clean(da.attrs)
        for var in da.data_vars:
            da[var].attrs = _clean(da[var].attrs)
        for coord in da.coords:
            da[coord].attrs = _clean(da[coord].attrs)
    elif isinstance(da, xr.DataArray):
        da.attrs = _clean(da.attrs)
        for coord in da.coords:
            da[coord].attrs = _clean(da[coord].attrs)
    return da


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
def load_json(path: Path) -> dict:
    try: return json.loads(Path(path).read_text())
    except Exception: return {}
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

def _canonical_model_id(model_id):
    """Return a stable, non-empty model identifier without changing semantics."""
    model_id = str(model_id).strip()
    if not model_id:
        raise ValueError("Un identifiant de modèle vide n'est pas autorisé.")
    return model_id


def _unique_preserve_order(values, *, label="valeurs"):
    """Remove repeated values while preserving their first occurrence."""
    unique_values = []
    seen = set()

    for raw_value in values or []:
        value = raw_value.strip() if isinstance(raw_value, str) else raw_value
        if value in seen:
            print(f"[dedup] {label}: doublon ignoré: {value!r}")
            continue
        seen.add(value)
        unique_values.append(value)

    return unique_values



def _configure_svml_parallelism(args):
    """Resolve one bounded CPU budget for all SV-ML-CMME models.

    ``args.ncores`` is treated as the total allocation, normally inherited
    from ``SLURM_CPUS_PER_TASK``. ``--svml-n-jobs`` controls model/search
    parallelism, ``--svml-optuna-n-jobs`` controls concurrent Optuna
    trials, ``--svml-cv-n-jobs`` controls concurrent CV folds, and
    ``--svml-tuning-model-n-jobs`` controls threads used by each RF/XGBoost
    estimator during tuning. Their product is bounded by the allocation.
    """
    allocation = int(getattr(args, "ncores", 1))
    if allocation == -1:
        allocation = int(
            os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)
        )
    if allocation < 1:
        raise ValueError(
            "--ncores doit être un entier positif ou -1."
        )
    args.ncores = allocation

    def _resolve(value, *, fallback, upper_bound, label):
        if value is None:
            resolved = fallback
        else:
            resolved = int(value)
            if resolved == -1:
                resolved = upper_bound
            elif resolved < 1:
                raise ValueError(
                    f"{label} doit être un entier positif ou -1."
                )

        if resolved > upper_bound:
            print(
                f"[parallel] {label}={resolved} dépasse l'allocation "
                f"({upper_bound}); valeur ramenée à {upper_bound}."
            )
            resolved = upper_bound
        return max(1, int(resolved))

    model_jobs = _resolve(
        getattr(args, "svml_n_jobs", None),
        fallback=allocation,
        upper_bound=allocation,
        label="--svml-n-jobs",
    )
    # Automatic server profile: up to 16 Optuna trials × 3 CV folds ×
    # 2 estimator threads. Explicit launcher values remain authoritative.
    default_tuning_jobs = 2
    default_optuna_jobs = min(
        model_jobs,
        16,
        max(1, allocation // (3 * default_tuning_jobs)),
    )
    optuna_jobs = _resolve(
        getattr(args, "svml_optuna_n_jobs", None),
        fallback=default_optuna_jobs,
        upper_bound=model_jobs,
        label="--svml-optuna-n-jobs",
    )

    max_cv_jobs = max(1, allocation // max(1, optuna_jobs * default_tuning_jobs))
    cv_jobs = _resolve(
        getattr(args, "svml_cv_n_jobs", None),
        fallback=min(3, max_cv_jobs),
        upper_bound=max(1, allocation // max(1, optuna_jobs)),
        label="--svml-cv-n-jobs",
    )

    max_tuning_jobs = max(1, allocation // max(1, optuna_jobs * cv_jobs))
    tuning_jobs = _resolve(
        getattr(args, "svml_tuning_model_n_jobs", None),
        fallback=min(default_tuning_jobs, max_tuning_jobs),
        upper_bound=max_tuning_jobs,
        label="--svml-tuning-model-n-jobs",
    )

    estimated_nested_workers = optuna_jobs * cv_jobs * tuning_jobs
    if estimated_nested_workers > allocation:
        raise ValueError(
            "Budget CPU SV-ML-CMME invalide: "
            f"optuna_n_jobs({optuna_jobs}) × cv_n_jobs({cv_jobs}) × "
            f"tuning_model_n_jobs({tuning_jobs}) = {estimated_nested_workers} "
            f"> allocation({allocation})."
        )

    args.svml_n_jobs = model_jobs
    args.svml_optuna_n_jobs = optuna_jobs
    args.svml_cv_n_jobs = cv_jobs
    args.svml_tuning_model_n_jobs = tuning_jobs
    os.environ["WAS_NCORES"] = str(allocation)

    print(
        "[parallel] SV-ML-CMME CPU budget: "
        f"allocation={allocation}, n_jobs={model_jobs}, "
        f"optuna_n_jobs={optuna_jobs}, cv_n_jobs={cv_jobs}, "
        f"tuning_model_n_jobs={tuning_jobs}, "
        f"estimated_nested_workers={estimated_nested_workers}"
    )
    return model_jobs, optuna_jobs, cv_jobs, tuning_jobs


def load_scores(pkl_path):
    """Load one score dictionary and validate its top-level type."""
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        return {}
    with pkl_path.open("rb") as stream:
        scores = pickle.load(stream)
    if not isinstance(scores, dict):
        raise TypeError(
            f"{pkl_path.name} doit contenir un dictionnaire, "
            f"reçu: {type(scores).__name__}."
        )
    return scores


def _nested_values_equal(value1, value2):
    """Compare terminal values, including xarray and NumPy objects."""
    if isinstance(value1, xr.DataArray) and isinstance(value2, xr.DataArray):
        return value1.equals(value2)
    if isinstance(value1, xr.Dataset) and isinstance(value2, xr.Dataset):
        return value1.equals(value2)
    if isinstance(value1, np.ndarray) and isinstance(value2, np.ndarray):
        try:
            return np.array_equal(value1, value2, equal_nan=True)
        except TypeError:
            return np.array_equal(value1, value2)
    try:
        result = value1 == value2
        if isinstance(result, (bool, np.bool_)):
            return bool(result)
        if np.isscalar(result):
            return bool(result)
    except Exception:
        pass
    return False


def _validate_score_tree(scores, *, source_name="scores"):
    """Validate the canonical ``metric -> model -> value`` score structure."""
    if not isinstance(scores, dict):
        raise TypeError(
            f"{source_name} doit être un dictionnaire, "
            f"reçu: {type(scores).__name__}."
        )

    for metric, model_scores in scores.items():
        if not isinstance(model_scores, dict):
            raise TypeError(
                f"{source_name}: la métrique {metric!r} doit contenir "
                "un dictionnaire modèle -> score."
            )
        canonical_ids = [_canonical_model_id(model_id) for model_id in model_scores]
        if len(canonical_ids) != len(set(canonical_ids)):
            raise ValueError(
                f"{source_name}: identifiants dupliqués après normalisation "
                f"pour la métrique {metric!r}."
            )
    return scores


def _register_score_value(
    target,
    metric,
    model_id,
    value,
    *,
    source_name,
    duplicate_policy="raise",
):
    """Register one score without silent replacement."""
    if duplicate_policy not in {"raise", "keep", "overwrite"}:
        raise ValueError("duplicate_policy doit être raise, keep ou overwrite.")

    metric = str(metric).strip()
    model_id = _canonical_model_id(model_id)
    bucket = target.setdefault(metric, {})

    if model_id not in bucket:
        bucket[model_id] = value
        return

    existing = bucket[model_id]
    if _nested_values_equal(existing, value):
        print(
            f"[scores] doublon identique ignoré: {metric} -> {model_id} "
            f"({source_name})"
        )
        return

    message = (
        f"Conflit de score pour {metric!r} -> {model_id!r} "
        f"provenant de {source_name}."
    )
    if duplicate_policy == "raise":
        raise KeyError(message)
    if duplicate_policy == "keep":
        print(f"[scores] WARNING: {message} Ancienne valeur conservée.")
        return
    print(f"[scores] WARNING: {message} Nouvelle valeur conservée.")
    bucket[model_id] = value


def merge_score_block(target, source, *, block_name, duplicate_policy="raise"):
    """Merge one independently produced model block into a stage registry."""
    _validate_score_tree(source, source_name=block_name)
    for metric, model_scores in source.items():
        for model_id, value in model_scores.items():
            _register_score_value(
                target,
                metric,
                model_id,
                value,
                source_name=block_name,
                duplicate_policy=duplicate_policy,
            )
    return target


def update_nested_dict(
    target,
    source_dict,
    *,
    source_name=None,
    path=(),
    on_conflict="raise",
):
    """Backward-compatible recursive merge with explicit conflict handling."""
    if path:
        # Preserve compatibility for any external recursive call.
        for key, incoming_value in source_dict.items():
            current_path = path + (str(key),)
            if key not in target:
                target[key] = incoming_value
            elif isinstance(target[key], dict) and isinstance(incoming_value, dict):
                update_nested_dict(
                    target[key], incoming_value,
                    source_name=source_name,
                    path=current_path,
                    on_conflict=on_conflict,
                )
            elif _nested_values_equal(target[key], incoming_value):
                continue
            elif on_conflict == "overwrite":
                target[key] = incoming_value
            elif on_conflict == "keep":
                continue
            else:
                raise KeyError(
                    f"Conflit au chemin {' -> '.join(current_path)!r} "
                    f"provenant de {source_name or 'source inconnue'}."
                )
        return target

    return merge_score_block(
        target,
        source_dict,
        block_name=source_name or "source_dict",
        duplicate_policy=on_conflict,
    )


def save_scores(workdir, scores, scorefile):
    """Validate and atomically save one non-cumulative stage score file."""
    _validate_score_tree(scores, source_name=scorefile)
    pkl_path = Path(workdir) / "intermediate" / scorefile
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = pkl_path.with_suffix(pkl_path.suffix + ".tmp")
    with tmp_path.open("wb") as stream:
        pickle.dump(scores, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, pkl_path)
    return pkl_path


def _remove_matching_files(directory, patterns, *, label):
    """Delete stale outputs belonging to one stage before a clean rerun."""
    directory = Path(directory)
    if not directory.exists():
        return
    removed = set()
    for pattern in patterns:
        for path in directory.glob(pattern):
            if path in removed or not path.is_file():
                continue
            path.unlink()
            removed.add(path)
            print(f"[{label}] ancien fichier supprimé: {path.name}")


def _discover_score_files(inter_dir):
    """Return canonical score files while safely handling legacy OBS-LAG files."""
    inter_dir = Path(inter_dir)
    files = sorted(inter_dir.glob("scores_consolidated_*.pkl"), key=lambda p: p.name)

    canonical_obs_lag = inter_dir / "scores_consolidated_obslag.pkl"
    legacy_obs_lag = [
        path for path in files
        if path.name in {
            "scores_consolidated_obslag1.pkl",
            "scores_consolidated_obslag2.pkl",
            "scores_consolidated_obslag3.pkl",
        }
    ]

    if canonical_obs_lag.exists():
        files = [path for path in files if path not in legacy_obs_lag]
    elif legacy_obs_lag:
        # Migration path for an old work directory: the latest cumulative
        # OBS-LAG file already contains all preceding blocks.
        latest = max(legacy_obs_lag, key=lambda p: p.stat().st_mtime)
        files = [path for path in files if path not in legacy_obs_lag]
        files.append(latest)
        files.sort(key=lambda p: p.name)
        print(
            "[scores] anciens fichiers OBS-LAG détectés; seul le plus récent "
            f"sera chargé: {latest.name}"
        )

    return files


def load_score_catalog(inter_dir):
    """Load all canonical stage score files with model-level provenance."""
    catalog = {}
    provenance = {}
    score_files = _discover_score_files(inter_dir)
    if not score_files:
        raise RuntimeError(
            f"Aucun fichier scores_consolidated_*.pkl trouvé dans {inter_dir}"
        )

    for pkl_path in score_files:
        loaded = load_scores(pkl_path)
        _validate_score_tree(loaded, source_name=pkl_path.name)
        for metric, model_scores in loaded.items():
            for model_id, value in model_scores.items():
                canonical_id = _canonical_model_id(model_id)
                key = (str(metric), canonical_id)
                if key in provenance:
                    old_source = provenance[key]
                    old_value = catalog[str(metric)][canonical_id]
                    if _nested_values_equal(old_value, value):
                        print(
                            f"[scores] doublon identique ignoré: {metric} -> "
                            f"{canonical_id} ({pkl_path.name}; déjà dans {old_source})"
                        )
                        continue
                    raise KeyError(
                        f"Score conflictuel pour {metric!r} -> {canonical_id!r}: "
                        f"{old_source} et {pkl_path.name}."
                    )
                _register_score_value(
                    catalog,
                    metric,
                    canonical_id,
                    value,
                    source_name=pkl_path.name,
                    duplicate_policy="raise",
                )
                provenance[key] = pkl_path.name

    return catalog, score_files, provenance


def _register_model_array(catalog, provenance, model_id, data, *, source_name):
    """Register one intermediate model field without silent NetCDF overwrite."""
    model_id = _canonical_model_id(model_id)
    if model_id in catalog:
        raise KeyError(
            f"Produit intermédiaire dupliqué pour {model_id!r}: "
            f"{provenance[model_id]} et {source_name}."
        )
    catalog[model_id] = data
    provenance[model_id] = source_name


def _ensure_1d_time(data, reference=None, source=None):
    """Return *data* with a valid one-dimensional indexed ``T`` coordinate.

    WASS2S intermediate products must obey one invariant: if ``T`` is a
    dimension, it must also be a 1-D coordinate with dims ``('T',)``.  Some
    historical MME implementations could lose or broadcast that coordinate.

    A damaged/missing axis is repaired ONLY when an authoritative reference
    with exactly the same ``T`` length is supplied.  No calendar is invented
    from the requested hindcast years.  Without such a reference, fail early
    and name the offending source rather than writing a corrupt NetCDF that
    crashes hours later during consolidation.
    """
    if not isinstance(data, xr.DataArray) or "T" not in data.dims:
        return data

    label = f" ({source})" if source else ""
    ref_ok = (
        isinstance(reference, xr.DataArray)
        and "T" in reference.dims
        and "T" in reference.coords
        and reference["T"].ndim == 1
        and int(data.sizes["T"]) == int(reference.sizes["T"])
    )

    # Missing coordinate: only the authoritative reference can restore it.
    if "T" not in data.coords:
        if not ref_ok:
            raise ValueError(
                f"Dimension T sans coordonnee{label}; fournissez une reference "
                "temporelle de meme longueur."
            )
        return data.assign_coords(
            T=xr.IndexVariable("T", np.asarray(reference["T"].values))
        )

    # Broadcast/multidimensional coordinate: discard it and restore the exact
    # reference when available.  Flattening a corrupted date field is unsafe.
    if data["T"].ndim != 1 or data["T"].dims != ("T",):
        if not ref_ok:
            raise ValueError(
                f"Coordonnee T multidimensionnelle/corrompue{label}: "
                f"dims={data['T'].dims}; aucune reference sure disponible."
            )
        data = data.drop_vars("T").assign_coords(
            T=xr.IndexVariable("T", np.asarray(reference["T"].values))
        )

    # For hindcast/forecast outputs produced positionally, the same-length
    # reference is authoritative even if an upstream routine changed dtype.
    if ref_ok:
        data = data.assign_coords(
            T=xr.IndexVariable("T", np.asarray(reference["T"].values))
        )

    if data["T"].ndim != 1 or data["T"].dims != ("T",):
        raise ValueError(f"Invariant T viole{label}: {data['T'].dims}")
    return data


# Member families, matched against the member id. Order matters: the first
# pattern that matches wins, so the more specific prefixes come first.
MEMBER_FAMILIES = (
    ("EOF-KCCA", ("eof-kcca", "fifth.approach")),
    ("CCA-SST", ("cca_gcm-sst", "cca-sst", "CCA_OBS-LAG")),
    ("CCA-PRCP", ("cca_gcm-prcp", "cca-prcp")),
    ("CCA-TEMP", ("cca_gcm-temp", "cca-temp")),
    ("ML-MME", ("hpelm", "mlp", "_rf", "xgb", "rf_", "SV-ML-CMME")),
    ("PCR-ML", ("pcr", "ridge", "poisson", "nbinom", "mars", "knn", "svr")),
    ("ANALOG", ("analog", "som", "bias_based")),
)


def _member_family(name):
    """Family of a member id, or 'AUTRE' when nothing matches."""
    low = str(name).lower()
    for family, patterns in MEMBER_FAMILIES:
        if any(p.lower() in low for p in patterns):
            return family
    return "AUTRE"


def _weighted_spatial_mean(field):
    """cos(lat)-weighted mean over Y and X, ignoring NaNs."""
    weights = np.cos(np.deg2rad(field["Y"].astype(float))).clip(min=0.0)
    weights = weights.broadcast_like(field)
    weights = weights.where(np.isfinite(field))
    total = weights.sum(dim=("Y", "X"), skipna=True)
    if float(total) <= 0:
        return float("nan")
    return float((field * weights).sum(dim=("Y", "X"), skipna=True) / total)


def _pearson_along_T(a, b):
    """Grid-point Pearson correlation between two (T, Y, X) fields."""
    valid = np.isfinite(a) & np.isfinite(b)
    a = a.where(valid)
    b = b.where(valid)
    aa = a - a.mean(dim="T", skipna=True)
    bb = b - b.mean(dim="T", skipna=True)
    cov = (aa * bb).sum(dim="T", skipna=True)
    den = np.sqrt((aa ** 2).sum(dim="T", skipna=True) * (bb ** 2).sum(dim="T", skipna=True))
    return xr.where(den > 0, cov / den, np.nan)


def error_complementarity(obs, hindcasts, min_common_years=8):
    """Correlation between the hindcast ERRORS of every pair of families.

    Parameters
    ----------
    obs : xr.DataArray
        Observed predictand, dims (T, Y, X).
    hindcasts : dict[str, xr.DataArray]
        Deterministic hindcasts, one per member.

    Returns
    -------
    dict with three entries:
        ``members``   family and skill of each member
        ``families``  family-mean error field correlations, pairwise
        ``verdict``   a short reading of each pair

    Reading the numbers
    -------------------
    An error correlation above ~0.8 means the two families are wrong in the
    same years at the same places: combining them smooths, it does not add
    information. Between ~0.4 and ~0.7 the complementarity is real. Below that
    it is strong, and the pair is worth weighting rather than selecting.

    Note the correlation is computed on ERRORS, not on forecasts. Two members
    can correlate at 0.95 with each other and still have weakly correlated
    errors, which is exactly the situation where an MME pays.
    """
    obs = obs.transpose("T", "Y", "X")

    # --- per member: family, ACC, and the error field -----------------------
    members, errors = {}, {}
    for name, hind in hindcasts.items():
        try:
            hind = hind.transpose("T", "Y", "X")
            common = np.intersect1d(obs["T"].values, hind["T"].values)
            if common.size < min_common_years:
                continue
            o = obs.sel(T=common)
            h = hind.sel(T=common)
            err = o - h
            acc = _weighted_spatial_mean(_pearson_along_T(h, o))
            members[name] = {
                "famille": _member_family(name),
                "ACC": round(acc, 3) if np.isfinite(acc) else None,
                "annees": int(common.size),
            }
            errors[name] = err
        except Exception as exc:  # a malformed member must not stop the stage
            members[name] = {"famille": _member_family(name), "erreur": repr(exc)}

    # --- family mean error, then pairwise correlation -----------------------
    by_family = {}
    for name, err in errors.items():
        by_family.setdefault(members[name]["famille"], []).append(err)

    family_error = {}
    for family, fields in by_family.items():
        try:
            stacked = xr.concat(
                [f.assign_coords(T=fields[0]["T"].values[: f.sizes["T"]]) for f in fields],
                dim="member",
            )
            family_error[family] = stacked.mean(dim="member", skipna=True)
        except Exception:
            family_error[family] = fields[0]

    pairs, verdicts = {}, {}
    names = sorted(family_error)
    for i, fa in enumerate(names):
        for fb in names[i + 1:]:
            try:
                a, b = family_error[fa], family_error[fb]
                common = np.intersect1d(a["T"].values, b["T"].values)
                if common.size < min_common_years:
                    continue
                r = _weighted_spatial_mean(
                    _pearson_along_T(a.sel(T=common), b.sel(T=common))
                )
                key = f"{fa} vs {fb}"
                pairs[key] = round(r, 3) if np.isfinite(r) else None
                if not np.isfinite(r):
                    verdicts[key] = "indéterminé"
                elif r > 0.8:
                    verdicts[key] = ("erreurs redondantes: combiner ces deux familles "
                                     "lisse sans ajouter d'information")
                elif r > 0.4:
                    verdicts[key] = "complémentarité réelle: la pondération MME a du sens"
                else:
                    verdicts[key] = ("complémentarité forte: pondérer plutôt que "
                                     "sélectionner entre ces familles")
            except Exception:
                continue

    return {"members": members, "families": pairs, "verdict": verdicts}


def collect_intermediate_products(inter_dir, hindcast_years):
    """Load paired hindcast/forecast files and reject duplicate model IDs."""
    inter_dir = Path(inter_dir)
    hdet, fdet, hprob, fprob = {}, {}, {}, {}
    provenance = {"hdet": {}, "fdet": {}, "hprob": {}, "fprob": {}}
    pairs = []

    for hind_path in sorted(inter_dir.glob("*_hindcasts.nc"), key=lambda p: p.name):
        fcst_path = inter_dir / hind_path.name.replace("_hindcasts.nc", "_forecasts.nc")
        if not fcst_path.exists():
            print(f"[consolidate] paire incomplète ignorée: {hind_path.name}")
            continue
        pairs.append((hind_path, fcst_path))

        hind_ds = xr.load_dataset(hind_path)
        fcst_ds = xr.load_dataset(fcst_path)
        time_slice = slice(str(hindcast_years[0]), str(hindcast_years[1]))

        for variable in hind_ds.data_vars:
            data = _ensure_1d_time(hind_ds[variable], source=f"{hind_path.name}:{variable}")
            if "T" in data.dims:
                if "T" not in data.indexes:
                    raise ValueError(
                        f"Axe T non indexable dans {hind_path.name}:{variable} "
                        "(corruption connue de WAS_mme_Weighted.compute et "
                        "assimilés, écrite sur disque avant réparation). "
                        "Relancez l'étage qui a produit ce fichier après avoir "
                        "appliqué patch_intermediate_time_fix.py, ou reconstruisez "
                        "l'axe avec repair_intermediate_time.py."
                    )
                data = data.sel(T=time_slice)
            if variable.startswith("hdcst_det_"):
                _register_model_array(
                    hdet, provenance["hdet"],
                    variable.removeprefix("hdcst_det_"), data,
                    source_name=hind_path.name,
                )
            elif variable.startswith("hdcst_prob_"):
                _register_model_array(
                    hprob, provenance["hprob"],
                    variable.removeprefix("hdcst_prob_"), data,
                    source_name=hind_path.name,
                )

        for variable in fcst_ds.data_vars:
            data = _ensure_1d_time(fcst_ds[variable])
            if variable.startswith("fcst_det_"):
                _register_model_array(
                    fdet, provenance["fdet"],
                    variable.removeprefix("fcst_det_"), data,
                    source_name=fcst_path.name,
                )
            elif variable.startswith("fcst_prob_"):
                _register_model_array(
                    fprob, provenance["fprob"],
                    variable.removeprefix("fcst_prob_"), data,
                    source_name=fcst_path.name,
                )

    if not pairs:
        raise RuntimeError(
            "Nothing to consolidate. No paired *_hindcasts.nc / "
            "*_forecasts.nc files found in intermediate/."
        )

    # process_datasets_for_mme (wass2s.was_mme) stacks every model's forecast
    # with `xr.concat(..., join="inner")`. That takes the INTERSECTION of the
    # T coordinate across all members before concatenating on M: if even one
    # model's forecast timestamp differs by a single nanosecond from the
    # others, the intersection is empty and EVERY consolidation method fails
    # uniformly with "index -1 is out of bounds for axis 0 with size 0" —
    # since there is nothing left to index. Snap every single-timestep
    # forecast onto the majority date here, once, with a named diagnostic for
    # whichever model diverged, instead of letting the failure surface as an
    # unexplained empty array 14 methods later.
    def _snap_forecast_dates(catalog, label):
        singles = {
            k: v for k, v in catalog.items()
            if "T" in v.dims and v.sizes.get("T") == 1 and "T" in v.indexes
        }
        if len(singles) < 2:
            return
        from collections import Counter
        keys = {
            k: int(np.asarray(v["T"].values[0]).astype("datetime64[ns]").astype("int64"))
            for k, v in singles.items()
        }
        counts = Counter(keys.values())
        canonical, n_majority = counts.most_common(1)[0]
        if len(counts) == 1:
            return
        outliers = [k for k, key in keys.items() if key != canonical]
        print(
            f"[consolidate] {label}: {len(outliers)} membre(s) avaient une date "
            f"de prévision differente de la majorite ({n_majority} membres) — "
            f"alignes sur {np.datetime64(canonical, 'ns')}: {sorted(outliers)}"
        )
        canonical_T = singles[next(k for k, key in keys.items() if key == canonical)]["T"]
        for k in outliers:
            catalog[k] = catalog[k].assign_coords(T=canonical_T.values)

    _snap_forecast_dates(fdet, "fdet")
    _snap_forecast_dates(fprob, "fprob")

    return hdet, fdet, hprob, fprob, pairs, provenance


def load_bestfit_params(workdir: Path):
    path = workdir / "best_distribution_params.nc"
    if path.exists():
        ds = xr.load_dataset(path)
        return ds.code, ds.shape, ds.loca, ds.scale
    return None, None, None, None

def evaluate_and_plot_forecasts(
    was_verify,
    obs_da,
    hindcast_det,
    hindcast_prob,
    forecast_prob,
    model_name,
    dict_key,
    clim_start,
    clim_end,
    scores_consolidated,
    dir_save_score,
    dir_to_forecast,
    season_str,
    forecast_year,
    month_init,
    fcst_labels,
    country_iso3,
    logopath,
):
    """Evaluate one uniquely identified model and register each score once."""
    dict_key = _canonical_model_id(dict_key)

    if hindcast_det is not None:
        for metric in ["Pearson", "MAE"]:
            result = was_verify.compute_deterministic_score(
                was_verify.get_scores_metadata()[metric][5],
                obs_da,
                hindcast_det,
            )
            score = result.compute()
            _register_score_value(
                scores_consolidated,
                metric,
                dict_key,
                score,
                source_name=f"evaluation:{model_name}",
                duplicate_policy="raise",
            )
            try:
                was_verify.plot_model_score(result, metric, dir_save_score, model_name)
                plt.close("all")
            except Exception:
                pass

    if hindcast_prob is not None:
        for metric in ["GROC", "RPSS"]:
            result = was_verify.compute_probabilistic_score(
                was_verify.get_scores_metadata()[metric][5],
                obs_da,
                hindcast_prob,
                clim_start,
                clim_end,
             **probability_score_kwargs(hindcast_prob))
            score = result.compute()
            _register_score_value(
                scores_consolidated,
                metric,
                dict_key,
                score,
                source_name=f"evaluation:{model_name}",
                duplicate_policy="raise",
            )
            try:
                was_verify.plot_model_score(result, metric, dir_save_score, model_name)
                plt.close("all")
            except Exception:
                pass

    if forecast_prob is not None:
        try:
            forecast_map = forecast_prob
            if "T" in forecast_map.coords:
                forecast_map = forecast_map.drop_vars("T")
            forecast_map = forecast_map.squeeze()
            title = (
                f"{model_name} {season_str}-{forecast_year} "
                f"IC_ {calendar.month_name[int(month_init)]}"
            )
            plot_prob_forecasts_(
                str(dir_to_forecast),
                forecast_map,
                title,
                hspace=-0.3,
                labels=fcst_labels,
                reverse_cmap=False,
            )
            plot_prob_forecasts(
                str(dir_to_forecast),
                forecast_map,
                title,
                title,
                country_code=country_iso3,
                source="gadm",
                admin_level=1,
                stations_df=None,
                reverse_cmap=False,
                hspace=-0.3,
                labels=fcst_labels,
                logo=logopath,
                logo_size=("21%", "14%"),
                logo_position="lower left",
                res=0.05,
                out="png",
            )
            plt.close("all")
        except Exception as exc:
            print(f"Plotting failed for {model_name}: {exc}")

    return scores_consolidated



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
            path = str(out_obs / f"Obs_PRCP_CHIRPSv3_{args.obs_years[0]}_{args.obs_years[1]}_{season_str}.nc")
            
        rainfall = xr.open_dataarray(path, chunks='auto') #.rename({'precip':'PRCP'})['PRCP']
        rainfal_clim = rainfall.mean(dim="T", skipna=True)
        mask = xr.where((rainfal_clim >= args.dry_threshold), 1, np.nan) if args.apply_drymask and not getattr(args, "_kharin_calibration", False) else xr.ones_like(rainfal_clim)
        # Apply mask and dynamic cropping based on the extent provided
        rainfall = rainfall.where(mask == 1).sel(Y=slice(None, args.obs_extent[0]), X=slice(None, args.obs_extent[3]))

    elif args.obs_source == "tamsat":
        log_progress("Downloading TAMSAT Observations")
        
        if not args.reuse_obs:
            path = downloader.WAS_Download_TAMSAT_Seasonal(
                str(out_obs), product="rfe", year_start=args.obs_years[0], year_end=args.obs_years[1],
                area=args.obs_extent, season_months=args.season_months, force_download=args.force, cleanup=False
            )
        else:
            # Assumes standard naming convention from the downloader if reusing
            season_str = "".join([calendar.month_abbr[int(m)] for m in args.season_months])
            path = str(out_obs / f"Obs_TAMSAT_rfe_{args.obs_years[0]}_{args.obs_years[1]}_{season_str}.nc")

        rainfall = xr.open_dataarray(path, chunks='auto')
        rainfal_clim = rainfall.mean(dim="T", skipna=True)
        mask = xr.where((rainfal_clim >= args.dry_threshold), 1, np.nan) if args.apply_drymask and not getattr(args, "_kharin_calibration", False) else xr.ones_like(rainfal_clim)
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

    # Preserve the rainfall support for every observation source and after merging.
    if args.apply_drymask and not getattr(args, "_kharin_calibration", False):
        rainfall = rainfall.where(rainfall.mean("T", skipna=True) >= args.dry_threshold)

    rainfall.name = "PRCP"
    rainfall = rainfall.compute() # Compute base array into memory
    rainfall.to_netcdf(workdir / "rainfall_predictand.nc")

    clim_slice = slice(str(args.clim_years[0]), str(args.clim_years[1]))
    if args.dist_method == "bestfit":
        log_progress("Fitting Best Distribution (bestfit mode)")
        distribution_map = {'norm': 1, 'lognorm': 2, 'gamma': 4}
        transf = WAS_TransformData(rainfall.sel(T=clim_slice), distribution_map=distribution_map, n_clusters=args.n_clusters)
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

def _svml_spatial_kwargs(args):
    """Spatial-variability and selection options shared by the pooled ML MMEs.

    See --svml-spatial-mode. Static fields are read once from a NetCDF file
    whose (Y, X) variables become z-scored covariates in soft/both mode.
    """
    static = None
    path = getattr(args, "svml_static_fields", None)
    if path:
        ds = xr.load_dataset(path)
        static = {}
        for name, field in ds.data_vars.items():
            dims = set(field.dims)
            if dims == {"Y", "X"}:
                static[name] = field
            elif dims <= {"Y", "X", "lat", "lon", "latitude", "longitude"}:
                static[name] = field.rename({d: n for d, n in
                    (("lat", "Y"), ("latitude", "Y"), ("lon", "X"), ("longitude", "X"))
                    if d in field.dims})
            else:
                print(f"[sv-ml-cmme] static field {name!r} ignored (dims {field.dims})", flush=True)
        if not static:
            raise ValueError(f"--svml-static-fields {path}: no (Y, X) variable found.")
    return dict(
        spatial_mode=str(getattr(args, "svml_spatial_mode", "zones")),
        covariate_names=tuple(getattr(args, "svml_covariates", None)
                              or ("clim_mean", "clim_logstd", "clim_asym", "dry_frac")),
        coord_step=(float(getattr(args, "svml_coord_step", 0.) or 0.) or None),
        static_fields=static,
        zones_as_features=bool(getattr(args, "svml_zones_as_features", False)),
        predictor_smoothing=int(getattr(args, "svml_predictor_smoothing", 0) or 0),
        one_se_rule=bool(getattr(args, "svml_one_se", True)),
        amplitude_calibration=(
            str(getattr(args, "svml_amplitude_mode", "global"))
            if getattr(args, "svml_amplitude_calibration", True) else False),
        amplitude_calibration_inner_folds=int(getattr(args, "svml_amplitude_inner_folds", 3)),
    )


def _mme_factory(name: str, args):
    """Build a leakage-aware SV-ML-CMME model with regularised search spaces.

    The search spaces intentionally favour low-variance models because the
    independent sample size is the number of hindcast YEARS, not the number of
    pooled grid cells.  Hyperparameter selection for skill estimation is done
    inside the outer CV by WAS_Cross_Validator; the full-period tuning performed
    after CV is used only for the operational forecast.
    """
    name = (name or "ridge").strip().lower()
    if name in {"unetpp", "rise-unet"}:
        from .unet_variants import build_unet_variant
        return build_unet_variant(args, "PRCP", architecture=name)
    profile = str(getattr(args, "svml_profile", "fast")).strip().lower()
    n_jobs = int(args.svml_n_jobs)
    optuna_n_jobs = int(args.svml_optuna_n_jobs)
    cv_n_jobs = int(args.svml_cv_n_jobs)
    tuning_model_n_jobs = int(args.svml_tuning_model_n_jobs)

    profiles = {
        "fast": dict(rf_trials=18, xgb_trials=20, mlp_trials=20, hpelm_trials=40,
                     cv=3, clusters=3),
        "balanced": dict(rf_trials=30, xgb_trials=35, mlp_trials=35, hpelm_trials=70,
                         cv=4, clusters=3),
        "full": dict(rf_trials=60, xgb_trials=70, mlp_trials=60, hpelm_trials=120,
                     cv=5, clusters=4),
    }
    if profile not in profiles:
        raise ValueError(f"Profil SV-ML-CMME inconnu: {profile!r}")
    cfg = profiles[profile]
    spatial = _svml_spatial_kwargs(args)

    print(
        f"[sv-ml-cmme] model={name} profile={profile} n_jobs={n_jobs} "
        f"optuna_n_jobs={optuna_n_jobs} cv_n_jobs={cv_n_jobs} "
        f"tuning_model_n_jobs={tuning_model_n_jobs}", flush=True,
    )

    if name == "rf":
        return WAS_mme_RF(
            search_method="bayesian",
            n_iter_search=cfg["rf_trials"],
            n_estimators_range=[200, 400, 700],
            max_depth_range=[4, 6, 8, 12],
            min_samples_split_range=[10, 20, 40],
            min_samples_leaf_range=[5, 10, 20],
            max_features_range=[0.4, 0.7, 1.0],
            bootstrap_range=[True],
            max_samples_range=[0.5, 0.7, 1.0],
            min_impurity_decrease_range=[0.0, 1e-5, 1e-4],
            ccp_alpha_range=[0.0, 1e-5, 1e-4],
            cv_folds=cfg["cv"], n_clusters=cfg["clusters"], verbose=1,
            random_state=42, dist_method=args.dist_method, warm_start=False,
            leave_one_year_out=False, n_jobs=n_jobs,
            optuna_n_jobs=optuna_n_jobs, cv_n_jobs=cv_n_jobs,
            tuning_model_n_jobs=tuning_model_n_jobs,
            cluster_n_jobs=(getattr(args, "cluster_n_jobs", 0) or None),
            spatial_subsample=getattr(args, "spatial_subsample", 1),
            **spatial,
        )

    if name == "xgb":
        return WAS_mme_XGBoosting(
            search_method="bayesian", n_estimators_range=[100, 200, 400],
            learning_rate_range=[0.01, 0.03, 0.05], max_depth_range=[2, 3, 4],
            min_child_weight_range=[5, 10, 20], subsample_range=[0.6, 0.8],
            colsample_bytree_range=[0.6, 0.8, 1.0], gamma_range=[0.0, 0.1, 1.0],
            reg_alpha_range=[0.01, 0.1, 1.0], reg_lambda_range=[1.0, 5.0, 10.0],
            # Random pooled train/eval splits can put the same year in both
            # sides; disable internal early stopping by default.
            early_stopping_rounds_range=[None], es_validation_fraction=0.10,
            random_state=42, dist_method=args.dist_method,
            n_iter_search=cfg["xgb_trials"], cv_folds=cfg["cv"],
            n_clusters=cfg["clusters"], optuna_n_jobs=optuna_n_jobs,
            optuna_timeout=10800 if profile != "full" else 21600,
            verbose=1, n_jobs=n_jobs, cv_n_jobs=cv_n_jobs,
            tuning_model_n_jobs=tuning_model_n_jobs,
            cluster_n_jobs=(getattr(args, "cluster_n_jobs", 0) or None),
            spatial_subsample=getattr(args, "spatial_subsample", 1),
            **spatial,
        )

    if name == "mlp":
        return WAS_mme_MLP(
            hidden_layer_sizes_range=[(2,), (4,), (8,), (4, 2)],
            learning_rate_init_range=[1e-4, 3e-4, 1e-3, 3e-3],
            activation_options=["tanh", "relu"], solver_options=["lbfgs", "adam"],
            alpha_range=[0.01, 0.1, 1.0, 10.0], search_method="bayesian",
            random_state=42, n_iter_search=cfg["mlp_trials"], max_iter=500,
            cv_folds=cfg["cv"], n_clusters=cfg["clusters"],
            dist_method=args.dist_method, optuna_n_jobs=optuna_n_jobs,
            optuna_timeout=10800 if profile != "full" else 21600,
            early_stopping_options=[False], n_jobs=n_jobs, cv_n_jobs=cv_n_jobs,
            cluster_n_jobs=(getattr(args, "cluster_n_jobs", 0) or None),
            spatial_subsample=getattr(args, "spatial_subsample", 1),
            **spatial,
        )

    if name == "hpelm":
        # ELM computed by Gram matrices in numpy (wass2s.was_elm; the hpelm
        # package is no longer used). F = [1 | canaux | tanh(XW+B)] : neurons=0
        # is a ridge regression, so the grid carries its own baseline. The
        # ridge is PER ROW (norm_scaling='per_sample'). The whole grid is
        # scored from one Gram pass per fold and zone, so the search is 'grid'.
        return WAS_mme_hpELM(
            neurons_range=[0, 50, 100, 300], activation_options=["tanh"],
            norm_range=[1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0], norm_scaling="per_sample",
            linear_part=True, n_seeds=3, random_state=42,
            n_iter_search=cfg["hpelm_trials"], cv_folds=cfg["cv"],
            dist_method=args.dist_method, search_method="grid",
            n_clusters=cfg["clusters"], n_trials_bayesian=cfg["hpelm_trials"],
            bayesian_sampler="tpe", verbose=1, n_jobs=n_jobs,
            optuna_n_jobs=optuna_n_jobs, cv_n_jobs=cv_n_jobs,
            cluster_n_jobs=(getattr(args, "cluster_n_jobs", 0) or None),
            spatial_subsample=getattr(args, "spatial_subsample", 1),
            **spatial,
        )

    # New low/medium-complexity challengers. Their GridSearchCV splitters are
    # grouped by calendar year in was_mme.py.
    common = dict(
        n_clusters=cfg["clusters"], cv_folds=cfg["cv"], leave_one_year_out=False,
        dist_method=args.dist_method, random_state=42, n_jobs=n_jobs,
        min_train_years=10, spatial_subsample=getattr(args, "spatial_subsample", 1),
        **spatial,
    )
    if name == "ridge":
        return _require_was_class("WAS_mme_Ridge")(**common)
    if name == "elasticnet":
        return _require_was_class("WAS_mme_ElasticNet")(**common)
    if name == "svr":
        return _require_was_class("WAS_mme_SVR")(**common)
    if name == "extratrees":
        return _require_was_class("WAS_mme_ExtraTrees")(**common)
    if name in {"histgb", "histgradientboosting"}:
        return _require_was_class("WAS_mme_HistGradientBoosting")(**common)
    if name == "mars":
        # Piecewise-linear challenger: a modest, interpretable non-linearity
        # between the pure-linear (Ridge/ElasticNet) and high-capacity
        # (RF/XGB/MLP/hpELM) ends of this list — see WAS_mme_MARS's docstring.
        return _require_was_class("WAS_mme_MARS")(**common)
    if name == "pls":
        return _require_was_class("WAS_mme_PLS")(**common)
    # MME estimators that also expose the forecast()/cross_validate contract
    # this harness needs. Their constructors do NOT take the shared **common
    # kwargs (n_clusters/cv_folds/dist_method...), so they are built bare and
    # left on their own defaults.
    if name == "elr":
        return _require_was_class("WAS_mme_ELR")(threshold_transform='cbrt')
    if name == "cca":
        return _cca_procedure(args, _require_was_class("WAS_mme_CCA")())
    if name == "pcr":
        return _require_was_class("WAS_mme_PCR")(
            dist_method=args.dist_method, positive_support=True)
    if name == "logistic":
        return _require_was_class("WAS_mme_logistic")()
    if name == "gp":
        return _require_was_class("WAS_mme_gaussian_process")()
    if name == "unet":
        # Optional torch dependency; WAS_mme_UNet raises a clear ImportError
        # itself when torch is missing.
        return _require_was_class("WAS_mme_UNet")()
    if name == "rescnn":
        return _require_was_class("WAS_mme_ResCNN")()

    raise ValueError(f"Méthode SV-ML-CMME inconnue: {name!r}")

def _residual_probability_controls(args):
    """Residual calibration controls, independent of model hyperparameter tuning."""
    return {
        "probability_calibration": probability_calibration_mode(args),
        "strict_oos_probability": True,
        "probability_inner_folds": int(getattr(args, "probability_inner_folds", None) or 5),
        "probability_inner_retune": bool(getattr(args, "probability_inner_retune", True)),
    }


def _svml_probability_kwargs(args, model_name):
    """Residual calibration belongs only to continuous regression models."""
    residual_models = {"hpelm", "rf", "xgb", "mlp", "ridge", "elasticnet", "svr",
                       "extratrees", "histgb", "mars", "pls", "cca", "pcr"}
    if str(model_name).lower() not in residual_models:
        return {}
    return {
        "probability_calibration": probability_calibration_mode(args),
        "strict_oos_probability": True,
        "probability_inner_folds": int(getattr(args, "probability_inner_folds", None) or args.svml_probability_folds),
        "probability_inner_retune": bool(getattr(args, "probability_inner_retune", True)),
    }


def _probability_procedure_signature(args):
    """Common probability semantics of cached deterministic-regression products."""
    return {
        "probability_schema": 5,
        "probability_calibration": probability_calibration_mode(args),
        "dist_method": str(getattr(args, "dist_method", "")),
        "climatology_years": list(getattr(args, "clim_years", ())),
        "hindcast_years": list(getattr(args, "hindcast_years", ())),
        "tercile_reference": str(getattr(args, "tercile_reference", "climatology")),
        "tercile_quantiles": "1/3,2/3",
        "scientific_models_schema": 1,
        "ngr_prcp_family": str(getattr(args, "ngr_prcp_family", "gamma")),
        "reference_version": 1,
        "outer_nb_omit": 2,
    }


def _svml_procedure_signature(args):
    signature = _probability_procedure_signature(args)
    signature.update({
        "models": list(dict.fromkeys(str(v).strip().lower()
                                     for v in getattr(args, "mme_models", ()))),
        "probability_inner_folds": int(getattr(args, "probability_inner_folds", None) or getattr(args, "svml_probability_folds", 4)),
        "probability_inner_retune": bool(getattr(args, "probability_inner_retune", True)),
        "nested_tuning": bool(getattr(args, "svml_nested_tuning", True)),
        "inner_max_folds": int(getattr(args, "svml_inner_folds", 5)),
        "profile": str(getattr(args, "svml_profile", "fast")),
        "add_ngr": bool(getattr(args, "add_ngr", False)),
        "gcm_selection": GCM_SELECTION_PROCEDURE,
        "gcm_models": list(getattr(args, "gcm_models", None) or ()),
        "drop_gcm": list(getattr(args, "drop_gcm", None) or ()),
        "cnn_gap_years": int(getattr(args, "optional_cv_gap_years", 1)),
    })
    if "kharin" in signature["models"]:
        from .kharin import kharin_procedure_signature
        signature["kharin"] = kharin_procedure_signature(args)
    return signature


def _consolidation_procedure_signature(args):
    signature = _probability_procedure_signature(args)
    signature.update({
        "profile": str(getattr(args, "consolidation_profile", "fast")),
        "methods": (None if getattr(args, "consolidation_methods", None) is None else
                    list(dict.fromkeys(str(v).lower() for v in args.consolidation_methods))),
        "probability_inner_folds": int(getattr(args, "probability_inner_folds", None) or getattr(args, "consolidation_prob_folds", 4)),
        "ml_probability_inner_retune": bool(getattr(args, "probability_inner_retune", True)),
        "pcr_probability_inner_retune": bool(getattr(args, "probability_inner_retune", True)),
        "nested_tuning": bool(getattr(args, "consolidation_nested_tuning", True)),
        "inner_max_folds": int(getattr(args, "consolidation_inner_folds", 4)),
        "pcr_modes": int(getattr(args, "consolidation_pcr_modes", 5)),
        "clusters": int(getattr(args, "consolidation_clusters", 3)),
    })
    return signature


def stage_sv_ml_cmme(args, workdir: Path, downloader: 'WAS_Download', client):
    input_specification = ensure_ensemble_configuration(workdir, args)
    ensemble_kwargs = ensemble_preparation_kwargs(args)
    check_dir = workdir / "checkpoints"
    requested_mme_cca = "cca" in {str(name).strip().lower() for name in args.mme_models}
    mme_cca_procedure = _mme_cca_procedure_signature(args) if requested_mme_cca else None
    procedure = _svml_procedure_signature(args)
    if done(check_dir, "sv_ml_cmme") and not args.redo:
        saved = load_json(check_dir / CHECKPOINTS["sv_ml_cmme"])
        saved_mme_cca = "cca" in {str(name).strip().lower() for name in saved.get("models_run", [])}
        if (saved.get("probability_procedure") == procedure and
                requested_mme_cca == saved_mme_cca and
                saved.get("mme_cca_procedure") == mme_cca_procedure):
            print("[sv-ml-cmme] checkpoint exists → skip")
            return
        log_progress("SV-ML procedure changed → recomputing")
    # Every recomputed member changes downstream inputs, including ML-only runs.
    _cca_invalidate_consolidation(workdir)
    (check_dir / CHECKPOINTS["sv_ml_cmme"]).unlink(missing_ok=True)
    
    inter_dir = workdir / "intermediate"
    inter_dir.mkdir(exist_ok=True, parents=True)
    _remove_matching_files(
        inter_dir,
        [
            "scores_consolidated_sv_ml_cmme_*.pkl",
            "sv_ml_cmme_*_hindcasts.nc",
            "sv_ml_cmme_*_forecasts.nc",
        ],
        label="sv-ml-cmme",
    )

    rainfall = xr.load_dataarray(workdir / "rainfall_predictand.nc")
    was_verify = WAS_Verification(dist_method=args.dist_method)
    dir_model = workdir / "model_data"; dir_model.mkdir(exist_ok=True, parents=True)
    scores_dir = workdir / "scores"; scores_dir.mkdir(parents=True, exist_ok=True)
    out_dir = workdir / "forecasts"; out_dir.mkdir(exist_ok=True)

    center_variable = filter_center_variable(downloader.ModelsName().keys(), "PRCP")
    for m in ['UKMO_603.PRCP', 'DWD_21.PRCP', 'METEOFRANCE_8.PRCP', 'UKMO_605.PRCP', 'JMA_3.PRCP', 'UKMO_604.PRCP', 'CMCC_35.PRCP']:
        if m in center_variable: center_variable.remove(m)

    center_variable = requested_centres(center_variable, args)

    hind = downloader.WAS_Download_Models(str(dir_model), center_variable, args.init_month, args.lead_time, args.hindcast_years[0], args.hindcast_years[1], args.gcm_extent, None, args.ensemble_mean, args.force)
    fcst = downloader.WAS_Download_Models(str(dir_model), center_variable, args.init_month, args.lead_time, args.hindcast_years[0], args.hindcast_years[1], args.gcm_extent, args.forecast_year, args.ensemble_mean, args.force)
                                          
    hind, fcst, center_variable = available_inputs(
        hind, fcst, center_variable, scores_dir / "sv_ml_cmme_available_gcms.json")
    
    # Select GCMs on their ensemble mean; retain raw members for ML features.
    screening_hind = (mean_screening_inputs(hind, scores_dir / "sv_ml_cmme_screening_means")
                      if ensemble_kwargs["ensemble_mean"] is None else hind)
    log_progress("GCM Validation & Selection of Best Models")
    scores = {}
    for metric in ["Pearson", "MAE"]:
        s_dict = was_verify.gcm_validation_compute(screening_hind, rainfall.sel(T=slice(str(args.hindcast_years[0]), str(args.hindcast_years[1]))), metric, args.init_month, args.clim_years[0], args.clim_years[1], str(scores_dir), lead_time=None, ensemble_mean=None, gridded=True)
        scores[metric] = {k: v.compute() for k, v in s_dict.items()}
        try: was_verify.plot_models_score(scores[metric], metric, str(scores_dir)); plt.close('all')
        except Exception: pass

    # A fixed roster keeps target scores out of the predictor-selection rule.
    # MAE/Pearson maps above are diagnostic only.
    best_models = fixed_gcm_roster(center_variable)
    log_progress("GCM roster fixed before scoring; --top-n-gcm is ignored. "
                 "Use --gcm-models/--drop-gcm to set the roster.")
    save_json(workdir / "scores/sv_ml_cmme_best_models.json",
              fixed_gcm_selection_metadata(best_models))

    all_model_hdcst, all_model_fcst, obs, _ = process_datasets_for_mme(
        rainfall.sel(T=slice(str(args.hindcast_years[0]), str(args.hindcast_years[1]))), hdcsted=hind, fcsted=fcst, scores=scores, gcm=True, best_models=best_models, dir_to_save_model=str(dir_model), year_start=args.hindcast_years[0], year_end=args.hindcast_years[1], score_metric="MAE", model=True, month_of_initialization=args.init_month, lead_time=args.lead_time, year_forecast=args.forecast_year, **ensemble_kwargs
    )

    best_code_da, best_shape_da, best_loc_da, best_scale_da = load_bestfit_params(workdir)
    cv = configured_cross_validator(args, obs)

    selected_mme_models = _unique_preserve_order(
        [str(name).strip().lower() for name in args.mme_models],
        label="SV-ML-CMME methods",
    )

    supported_mme_models = {"hpelm", "rf", "xgb", "mlp", "ridge", "elasticnet", "svr", "extratrees", "histgb", "mars", "pls", "elr", "cca", "pcr", "logistic", "gp", "unet", "unetpp", "rise-unet", "rescnn", "kharin"}
    unsupported_mme_models = [
        name for name in selected_mme_models
        if name not in supported_mme_models
    ]
    if unsupported_mme_models:
        raise ValueError(
            "Méthodes SV-ML-CMME non prises en charge: "
            f"{unsupported_mme_models}. Valeurs permises: "
            f"{sorted(supported_mme_models)}"
        )

    for m_name in selected_mme_models:
        scores_consolidated = {}
        log_progress(f"Training & Evaluating {m_name.upper()} (SV-ML-CMME)")
        if m_name in {"unet", "unetpp", "rise-unet", "rescnn", "kharin"}:
            hind_det, hind_prob, fcst_det, fcst_prob = run_optional(
                m_name, args, workdir, downloader, obs, all_model_hdcst,
                all_model_fcst, best_models, globals())
        else:
            model = _mme_factory(m_name, args)
        
            model_jobs = int(getattr(model, "n_jobs", args.svml_n_jobs))
            model_optuna_jobs = int(
                getattr(model, "optuna_n_jobs", args.svml_optuna_n_jobs)
            )
            print(
                f"[sv-ml-cmme/{m_name}] n_jobs={model_jobs}, "
                f"optuna_n_jobs={model_optuna_jobs}, "
                f"backend={'dask' if client is not None else 'threading'}"
            )

            # Skill estimation: every transform / cluster / hyperparameter choice is
            # fitted inside the outer training fold.  Full-period tuning happens only
            # AFTER the OOS hindcast and is used solely for the operational forecast.
            cv_kwargs = {
                "nested_tuning": bool(args.svml_nested_tuning),
                "inner_max_folds": int(args.svml_inner_folds),
                **_svml_probability_kwargs(args, m_name),
            }
            if args.dist_method == "bestfit":
                cv_kwargs["probability_distribution_map"] = {"norm": 1, "lognorm": 2, "gamma": 4}

            backend = "dask" if client is not None else "threading"
            with joblib.parallel_backend(backend, n_jobs=model_jobs):
                hind_det, hind_prob = cv.cross_validate(
                    model, obs, all_model_hdcst, args.clim_years[0], args.clim_years[1],
                    **cv_kwargs)

                best_params = cluster = None
                if m_name == "elr":
                    if args.svml_nested_tuning:
                        model.compute_hyperparameters(
                            all_model_hdcst, obs, *args.clim_years,
                            cv_folds=int(args.svml_inner_folds), score="rps", nb_omit=cv.nb_omit)
                elif m_name == "pcr":
                    model.finalize_truncation(
                        all_model_hdcst, obs, *args.clim_years,
                        nb_omit=cv.nb_omit, inner_max_folds=int(args.svml_inner_folds),
                        enabled=bool(args.svml_nested_tuning))
                elif hasattr(model, "compute_hyperparameters"):
                    if args.svml_nested_tuning:
                        best_params, cluster = model.compute_hyperparameters(
                            all_model_hdcst, obs, args.clim_years[0], args.clim_years[1])
                    elif m_name not in {"cca", "pcr"}:
                        from wass2s.was_mme import _regional_params_from
                        fixed_params = _regional_params_from(model, {}, obs)
                        best_params, cluster = fixed_params["best_params"], fixed_params["cluster_da"]

            if cluster is not None:
                try: cluster.plot(); save_and_close_plot(scores_dir / f"{m_name.upper()}_Clusters.png")
                except Exception: pass

            forecast_kwargs = {}
            if args.dist_method == "bestfit":
                forecast_kwargs.update({"best_code_da": best_code_da, "best_shape_da": best_shape_da,
                                        "best_loc_da": best_loc_da, "best_scale_da": best_scale_da})
            if best_params is not None:
                forecast_kwargs["best_params"] = best_params
            if cluster is not None:
                forecast_kwargs["cluster_da"] = cluster

            if m_name == "elr":
                # Native CDF: probabilities have no deterministic physical units.
                fcst_det = None
                fcst_prob = model.forecast(
                    obs, *args.clim_years, all_model_hdcst,
                    all_model_fcst.isel(T=[-1]))
            else:
                fcst_det, fcst_prob = model.forecast(
                    obs, args.clim_years[0], args.clim_years[1], all_model_hdcst,
                    hind_det, all_model_fcst.isel(T=[-1]), **forecast_kwargs)

        # Certains estimateurs (dont WAS_mme_Weighted et ses dérivés) rendent
        # un axe T diffusé sur (T, Y, X) avec des valeurs corrompues (voir
        # _ensure_1d_time). Réparer AVANT d'écrire sur disque : une fois le
        # fichier corrompu écrit, le crash n'arrive qu'à la relecture par
        # consolidate, des heures plus tard, sans dire quel fichier est en cause.
        hind_det = _ensure_1d_time(hind_det, reference=obs)
        hind_prob = _ensure_1d_time(hind_prob, reference=obs)
        fcst_det = _ensure_1d_time(fcst_det, reference=all_model_fcst.isel(T=[-1]))
        fcst_prob = _ensure_1d_time(fcst_prob, reference=all_model_fcst.isel(T=[-1]))

        hind_det, hind_prob, fcst_det, fcst_prob = mask_products(
            (hind_det, hind_prob, fcst_det, fcst_prob), obs)

        model_id = f"SV-ML-CMME_{m_name.upper()}"
        evaluate_and_plot_forecasts(was_verify, obs, hind_det, hind_prob, fcst_prob, model_id, model_id, args.clim_years[0], args.clim_years[1], scores_consolidated, str(scores_dir), str(out_dir), season_to_str(args.season_months), args.forecast_year, args.init_month, FCST_LABELS_FR, args.country_code, args.logo)

        (workdir/"intermediate").mkdir(exist_ok=True, parents=True)
        xr.Dataset({k: v for k, v in {f"hdcst_det_{model_id}": hind_det, f"hdcst_prob_{model_id}": hind_prob}.items() if v is not None}).to_netcdf(workdir/f"intermediate/sv_ml_cmme_{m_name}_hindcasts.nc")
        xr.Dataset({k: v for k, v in {f"fcst_det_{model_id}": fcst_det, f"fcst_prob_{model_id}": fcst_prob}.items() if v is not None}).to_netcdf(workdir/f"intermediate/sv_ml_cmme_{m_name}_forecasts.nc")
        save_scores(workdir, scores_consolidated, f"scores_consolidated_sv_ml_cmme_{m_name}.pkl")

        gc.collect()
        if client: client.restart()

    if args.add_ngr:
        # Modern EMOS/NGR variant.  Unlike the former ``WAS_mme_NGR.fit``
        # shortcut, the hindcast below is genuinely outer-fold OOS.
        scores_consolidated = {}
        log_progress("Training & Evaluating modern NGR/EMOS (SV-ML-CMME)")
        best_models2 = list(best_models)
        all_model_hdcst2, all_model_fcst2, obs2, _ = process_datasets_for_mme(
            rainfall.sel(T=slice(str(args.hindcast_years[0]), str(args.hindcast_years[1]))),
            hdcsted=hind, fcsted=fcst, scores=scores, gcm=True,
            best_models=best_models2, dir_to_save_model=str(dir_model),
            year_start=args.hindcast_years[0], year_end=args.hindcast_years[1],
            score_metric="MAE", model=True,
            month_of_initialization=args.init_month, lead_time=args.lead_time,
            year_forecast=args.forecast_year,
        )
        all_model_hdcst2 = all_model_hdcst2.assign_coords(T=obs2["T"])
        if "M" in obs2.dims:
            obs2 = obs2.isel(M=0, drop=True)
        obs2 = obs2.squeeze(drop=True).transpose("T", "Y", "X")

        ngr_cls = _require_was_class("WAS_mme_NGR_NonGaussian")
        ngr = ngr_cls(family=args.ngr_prcp_family, occurrence_model="auto", apply_to="all", min_train_years=10)
        cv_ngr = configured_cross_validator(args, obs2)
        hind_det_ngr, hind_prob_ngr = cv_ngr.cross_validate(
            ngr, obs2, all_model_hdcst2,
            args.clim_years[0], args.clim_years[1],
        )

        # Operational fit only after the OOS hindcast has been completed.
        ngr_fcst = ngr.compute_model(
            all_model_hdcst2, obs2, all_model_fcst2, clim_terciles=True,
            obs_for_terciles=climatology_sample(obs2, *args.clim_years))
        forecast_det_ngr = ngr_fcst["calibrated_mean"]
        forecast_prob_ngr = ngr_fcst["tercile_probability"]

        hind_det_ngr = _ensure_1d_time(hind_det_ngr, reference=obs2)
        hind_prob_ngr = _ensure_1d_time(hind_prob_ngr, reference=obs2)
        forecast_det_ngr = _ensure_1d_time(forecast_det_ngr, reference=all_model_fcst2)
        forecast_prob_ngr = _ensure_1d_time(forecast_prob_ngr, reference=forecast_det_ngr)

        model_id_ngr = "SV-ML-CMME_NGR"
        evaluate_and_plot_forecasts(
            was_verify, obs2, hind_det_ngr, hind_prob_ngr, forecast_prob_ngr,
            model_id_ngr, model_id_ngr, args.clim_years[0], args.clim_years[1],
            scores_consolidated, str(scores_dir), str(out_dir),
            season_to_str(args.season_months), args.forecast_year, args.init_month,
            FCST_LABELS_FR, args.country_code, args.logo,
        )
        save_scores(workdir, scores_consolidated, "scores_consolidated_sv_ml_cmme_ngr.pkl")
        xr.Dataset({
            "hdcst_det_SV-ML-CMME_NGR": hind_det_ngr,
            "hdcst_prob_SV-ML-CMME_NGR": hind_prob_ngr,
        }).pipe(strip_attrs).to_netcdf(workdir / "intermediate/sv_ml_cmme_ngr_hindcasts.nc")
        xr.Dataset({
            "fcst_det_SV-ML-CMME_NGR": forecast_det_ngr,
            "fcst_prob_SV-ML-CMME_NGR": forecast_prob_ngr,
        }).pipe(strip_attrs).to_netcdf(workdir / "intermediate/sv_ml_cmme_ngr_forecasts.nc")

    mark_done(
        check_dir,
        "sv_ml_cmme",
        {
            "models_run": selected_mme_models,
            "mme_cca_procedure": mme_cca_procedure,
            "probability_procedure": procedure,
            "ensemble_input": input_specification,
            "gcm_selection": GCM_SELECTION_PROCEDURE,
            "gcm_roster": best_models,
            "add_ngr": bool(args.add_ngr),
            "n_jobs": int(args.svml_n_jobs),
            "optuna_n_jobs": int(args.svml_optuna_n_jobs),
            "allocated_ncores": int(args.ncores),
            "profile": str(args.svml_profile),
        },
    )

    os.environ["WAS_NCORES"] = str(args.ncores)


# =========================
# Stage 3/4 — Factorized CCA Pipeline
# =========================
def _cca_auto_truncation(args):
    """Map --cca-auto-truncation to the kwarg WAS_CCA_op/WAS_CCA_strict expect.

    'auto' -> None (module default, itself overridable via WAS_CCA_AUTOTUNE).
    'on'   -> True (force the selected MDL or CV procedure).
    'off'  -> False (fixed n_modes/n_pca_modes; overrides mode selection).
    """
    return {"auto": None, "on": True, "off": False}[getattr(args, "cca_auto_truncation", "auto")]


def _cca_procedure_signature(args):
    """Effective selector, probability procedure and reference of cached CCA results."""
    signature = _cca_selection_signature(args)
    signature.update({
        "probability_calibration": _cca_probability_calibration(args),
        "probability_schema": 5,
        "probability_pooling": "equal_weight_member_probabilities",
        "obs_lag_nested_tuning": bool(getattr(args, "obs_lag_nested_tuning", True)),
        "obs_lag_inner_folds": int(getattr(args, "obs_lag_inner_folds", 3)),
        "obs_lag_ridge_alpha": float(getattr(args, "obs_lag_ridge_alpha", 1.0)),
        "dist_method": str(getattr(args, "dist_method", "")),
        "probability_inner_folds": int(getattr(args, "probability_inner_folds", None) or 5),
        "probability_inner_retune": bool(getattr(args, "probability_inner_retune", True)),
        "outer_nb_omit": 2,
        "climatology_years": list(getattr(args, "clim_years", ())),
        "hindcast_years": list(getattr(args, "hindcast_years", ())),
        "tercile_reference": str(getattr(args, "tercile_reference", "climatology")),
        "tercile_quantiles": "1/3,2/3",
        "scientific_models_schema": 1,
        "ngr_prcp_family": str(getattr(args, "ngr_prcp_family", "gamma")),
        "reference_version": 1,
    })
    return signature


def _mme_cca_procedure_signature(args):
    """Fingerprint the separate MME-CCA procedure, using its actual defaults."""
    model = _cca_procedure(args, _require_was_class("WAS_mme_CCA")())
    nested = bool(getattr(args, "svml_nested_tuning", True))
    autotune = bool(model.autotune_enabled())
    signature = {name: getattr(model, name) for name in (
        "n_modes", "n_pca_modes", "standardize", "use_coslat", "use_pca",
        "combine", "detrend", "dist_method", "positive_support", "random_state")}
    signature.update({
        "mode_selection": model.truncation_selection,
        "detrend_schema": 2,
        "hindcast_tuning_enabled": nested and autotune,
        "forecast_tuning_enabled": autotune,
        "truncation_grid": model.default_truncation_grid() if autotune else None,
        "engine": os.environ.get("WAS_CCA_ENGINE", "exact"),
        "probability_calibration": model.probability_calibration,
        "probability_schema": 5,
        "probability_inner_folds": int(getattr(args, "probability_inner_folds", None) or getattr(args, "svml_probability_folds", 4)),
        "probability_inner_retune": bool(getattr(args, "probability_inner_retune", True)),
        "outer_nb_omit": 2,
        "climatology_years": list(getattr(args, "clim_years", ())),
        "hindcast_years": list(getattr(args, "hindcast_years", ())),
        "tercile_reference": str(getattr(args, "tercile_reference", "climatology")),
        "reference_version": 1,
    })
    return signature



def _cca_detrend(args):
    """One detrending choice for all CLI CCA paths; retain legacy namespaces."""
    canonical = getattr(args, "cca_detrend", None)
    legacy = getattr(args, "eof_kcca_detrend", None)
    if canonical is not None and legacy is not None and bool(canonical) != bool(legacy):
        raise ValueError("Conflicting CCA and legacy EOF-KCCA detrending options")
    value = canonical if canonical is not None else legacy
    return True if value is None else bool(value)


def _cca_probability_calibration(args):
    """Compatibility wrapper for the common residual calibration setting."""
    return probability_calibration_mode(args)


def _cca_procedure(args, model):
    """Set calibration; retain the independent CV/AIC selector of MME CCA."""
    if hasattr(model, "mode_selection"):
        selection = _cca_mode_selection(args)
        model.truncation_selection = "aic" if selection == "aic" else "cv"
    else:
        model.truncation_selection = getattr(args, "cca_truncation_selection", None) or "cv"
        if not getattr(args, "svml_nested_tuning", True):
            # A fixed hindcast procedure must remain fixed at forecast time.
            model.auto_truncation = False
    model.detrend = _cca_detrend(args)
    model.probability_calibration = _cca_probability_calibration(args)
    return model


def _make_cca_op(args, **kw):
    """CCA-GCM engine factory. --cca-engine=regularized swaps the op-style CCA
    for ridge-regularized CCA (covariance shrinkage instead of a mode-count
    search); alpha comes from --cca-alpha. auto_truncation is only meaningful
    for the standard engine (the regularized one has no truncation search)."""
    kw["detrend"] = _cca_detrend(args)
    if getattr(args, "cca_engine", "standard") == "regularized":
        kw.pop("auto_truncation", None)
        return _cca_procedure(args, WAS_CCA_Regularized(alpha=args.cca_alpha, **kw))
    return _cca_procedure(args, WAS_CCA_op(auto_truncation=_cca_auto_truncation(args),
                      mode_selection=_cca_mode_selection(args), **kw))


def _make_cca_strict(args, **kw):
    """OBS-LAG CCA engine factory (strict spatial gap filling — the land-mask-
    safe variant). Mirrors _make_cca_op for the regularized engine."""
    kw["detrend"] = _cca_detrend(args)
    if getattr(args, "cca_engine", "standard") == "regularized":
        kw.pop("auto_truncation", None)
        return _cca_procedure(args, WAS_CCA_Regularized_strict(alpha=args.cca_alpha, **kw))
    return _cca_procedure(args, WAS_CCA_strict(auto_truncation=_cca_auto_truncation(args),
                          mode_selection=_cca_mode_selection(args), **kw))


def _cca_selection_signature(args):
    """Identify the effective CCA selector, including fixed-mode overrides.

    Old checkpoints without this signature cannot certify MDL provenance.
    Shared downloads and non-CCA checkpoints are not affected.
    """
    engine = getattr(args, "cca_engine", "standard")
    enabled = _cca_auto_truncation(args)
    if enabled is None:
        enabled = os.environ.get("WAS_CCA_AUTOTUNE", "1") != "0"
    selection = (_cca_mode_selection(args) if enabled else "fixed")
    signature = {
        "engine": engine,
        "detrend": _cca_detrend(args),
        "detrend_schema": 2,
        "mode_selection": selection if engine == "standard" else "regularized",
        "n_modes": int(getattr(args, "cca_modes", 3)),
        "n_pca_modes": int(getattr(args, "cca_pca_modes", 8)),
    }
    if engine == "regularized":
        signature["alpha"] = float(getattr(args, "cca_alpha", 0.1))
    return signature


def _cca_invalidate_consolidation(workdir):
    """A recomputed CCA member requires the final ensemble to be rebuilt."""
    marker = workdir / "checkpoints" / CHECKPOINTS["consolidate"]
    if marker.exists():
        marker.unlink()


def _save_cca_mdl_diagnostics(model, args, obs, cv, output):
    """Persist train-only rank choices and the held-out dates of outer folds."""
    signature = _cca_selection_signature(args)
    if signature["mode_selection"] != "mdl":
        if output.exists():
            output.unlink()
        return
    times = obs["T"].values
    splits = list(cv.custom_cv.split(obs["T"], cv.nb_omit))
    choices = getattr(model, "truncation_folds_", None) or []
    diagnostics = getattr(model, "truncation_diagnostics_folds_", None) or []
    outer = []
    for index, selection in enumerate(diagnostics):
        row = {"fold_index": index, "selection": selection}
        if index < len(splits):
            train, test = splits[index]
            row["training_times"] = [str(times[j]) for j in train]
            row["held_out_times"] = [str(times[j]) for j in test]
        outer.append(row)
    save_json(output, {
        "cca_selection": signature,
        "reference": "Song et al. (2016), Detector 3; arXiv:1604.02047",
        "climatology_years": list(args.clim_years),
        "full_record_times": [str(t) for t in times],
        "full_record_selection": getattr(model, "truncation_", None),
        "outer_fold_choices": choices,
        "outer_fold_diagnostics": outer,
        "outer_fold_count": len(splits),
        "note": "Mode selection uses training data only; probability-calibration folds remain unchanged.",
    })


def _plot_selected_cca_modes(model, args, predictor, obs, output):
    """Plot full-record MDL modes after CV without changing constructor limits."""
    selection = getattr(model, "truncation_", None)
    if selection is None:
        log_progress("CCA MDL: no full-record selection available for the mode plot")
        return
    if int(selection["n_modes"]) == 0:
        # A previous run can have left a positive-mode figure in the same workdir.
        if output.exists():
            output.unlink()
        log_progress("CCA MDL: 0 shared modes selected; no canonical mode plot")
        return
    state = model.truncation_state()
    try:
        model.apply_truncation(selection)
        model.plot_cca_results(X=predictor, Y=obs,
                               clim_year_start=args.clim_years[0],
                               clim_year_end=args.clim_years[1])
        save_and_close_plot(output)
    except Exception as exc:
        log_progress(f"CCA MDL mode plot unavailable: {exc}")
    finally:
        model.restore_truncation(state)


def _season_target_time(args):
    """Canonical single valid-date for the season being forecast.

    The season's center month (e.g. October for SON), in ``forecast_year`` —
    or ``forecast_year + 1`` if that center month falls numerically before
    ``init_month`` (the season wraps past December relative to
    initialization, e.g. DJF/NDJ). This mirrors, independently of any raw
    predictor's own timestamp, the same convention every non-CCA approach
    gets for free via month_of_initialization/lead_time/year_forecast in
    ``wass2s.utils.load_gridded_predictor``.
    """
    months = [int(m) for m in args.season_months]
    center_month = months[len(months) // 2]
    year = int(args.forecast_year)
    if center_month < int(args.init_month):
        year += 1
    return np.datetime64(f"{year:04d}-{center_month:02d}-01")


def _execute_cca_pipeline(args, workdir, downloader, var_filter, extent, defined_zone, prefix, top_n, *, group_member_cache=False):
    log_progress(f"Processing Factorized CCA Pipeline for: {prefix.upper()}")

    # ------------------------------------------------------------------ #
    # Per-pipeline checkpoint: one JSON file per CCA stage (prefix).      #
    # If the intermediate NetCDFs already exist and --redo is not set,    #
    # we skip all computation and reload results from disk.               #
    # ------------------------------------------------------------------ #
    check_dir  = workdir / "checkpoints"; check_dir.mkdir(exist_ok=True)
    chk_file   = check_dir / f"chk_cca_{prefix}.json"
    members_dir = workdir / "cca_members"; members_dir.mkdir(exist_ok=True, parents=True)
    if group_member_cache:
        # Multi-domain SST EqWt is formed only after EVERY domain has run.
        # Per-domain products therefore live outside intermediate/ and can
        # never be mistaken for final MME members.
        hind_nc = members_dir / f"{prefix}_members_hindcasts.nc"
        fcst_nc = members_dir / f"{prefix}_members_forecasts.nc"
        scores_pkl = members_dir / f"scores_{prefix}.pkl"
    else:
        hind_nc = workdir / f"intermediate/{prefix}_hindcasts.nc"
        fcst_nc = workdir / f"intermediate/{prefix}_forecasts.nc"
        scores_pkl = workdir / "intermediate" / f"scores_consolidated_{prefix}.pkl"
    best_json  = workdir / f"scores/{prefix}_best_models.json"

    _want_eqwt = False if group_member_cache else getattr(args, "cca_equal_weight", False)
    # Fingerprint of what actually determines the result of this stage. The
    # prefix alone is not enough now that several SST domains share the same
    # pipeline: editing the bounds of SST_DOMAINS['nino34'] or changing
    # --cca-domain-top-n must invalidate the cache, otherwise the run silently
    # reloads results computed over a different box.
    _want_zone = {k: list(v) for k, v in defined_zone.items()}
    _want_selection = _cca_procedure_signature(args)
    _chk_ok = False
    if chk_file.exists() and hind_nc.exists() and fcst_nc.exists() and not args.redo:
        _saved = load_json(chk_file)
        _chk_ok = (
            bool(_saved.get("equal_weight", False)) == bool(_want_eqwt)
            and bool(_saved.get("group_member_cache", False)) == bool(group_member_cache)
        )
        if not _chk_ok:
            log_progress(
                f"CCA {prefix.upper()} checkpoint mode mismatch "
                f"(saved equal_weight={_saved.get('equal_weight')}, "
                f"requested={_want_eqwt}) → recomputing"
            )
        elif _saved.get("cca_selection") != _want_selection:
            _chk_ok = False
            log_progress(f"CCA {prefix.upper()} selection changed → recomputing")
        elif _saved.get("defined_zone") not in (None, _want_zone):
            _chk_ok = False
            log_progress(
                f"CCA {prefix.upper()} checkpoint domain mismatch "
                f"(saved {_saved.get('defined_zone')}, requested {_want_zone}) "
                "→ recomputing"
            )
        elif _saved.get("procedure") != _cca_procedure_signature(args):
            _chk_ok = False
            log_progress(
                f"CCA {prefix.upper()} checkpoint procedure mismatch "
                f"(saved {_saved.get('procedure')}, requested "
                f"{_cca_procedure_signature(args)}) → recomputing"
            )
        elif _saved.get("top_n", top_n) != top_n:
            _chk_ok = False
            log_progress(
                f"CCA {prefix.upper()} checkpoint top_n mismatch "
                f"(saved {_saved.get('top_n')}, requested {top_n}) → recomputing"
            )

    if _chk_ok:
        log_progress(f"CCA {prefix.upper()} already done — loading from cache")
        best_models = (
            load_json(best_json).get("best_models", [])
            if best_json.exists() else []
        )
        best_models = _unique_preserve_order(
            best_models, label=f"CCA {prefix} cached best_models"
        )
        print(f"[cca/{prefix}] cached best_models: {best_models}")
        return best_models

    _cca_invalidate_consolidation(workdir)
    if group_member_cache:
        if args.redo:
            for _p in (hind_nc, fcst_nc, scores_pkl):
                if _p.exists():
                    _p.unlink()
                    print(f"[cca/{prefix}] ancien cache membre supprime: {_p.name}")
    else:
        _remove_matching_files(
            workdir / "intermediate",
            [
                f"scores_consolidated_{prefix}.pkl",
                f"{prefix}_hindcasts.nc",
                f"{prefix}_forecasts.nc",
            ],
            label=f"cca/{prefix}",
        )

    rainfall = xr.load_dataarray(workdir / "rainfall_predictand.nc")
    dir_model = workdir / "model_data"; dir_model.mkdir(exist_ok=True)
    scores_dir = workdir / "scores"; scores_dir.mkdir(exist_ok=True)
    out_dir = workdir / "forecasts"; out_dir.mkdir(exist_ok=True)

    center_variable = filter_center_variable(downloader.ModelsName().keys(), var_filter)
    for bad in args.drop_gcm:
        try: center_variable.remove(bad)
        except ValueError: pass

    center_variable = requested_centres(center_variable, args)

    hind = downloader.WAS_Download_Models(str(dir_model), center_variable, args.init_month, args.lead_time, args.hindcast_years[0], args.hindcast_years[1], extent, None, physical_ensemble_reduction(args), args.force)
    fcst = downloader.WAS_Download_Models(str(dir_model), center_variable, args.init_month, args.lead_time, args.hindcast_years[0], args.hindcast_years[1], extent, args.forecast_year, physical_ensemble_reduction(args), args.force)

    hind, fcst, center_variable = available_inputs(
        hind, fcst, center_variable, scores_dir / f"{prefix}_available_gcms.json")
    
    try:
        plot_map([extent[1],extent[3],extent[2],extent[0]], sst_indices=defined_zone, title=f"Predictors Area {var_filter}", fig_size=(6,4))
        save_and_close_plot(scores_dir / f"Map_CCA_Predictors_{prefix}.png")
    except: pass

    best_code_da, best_shape_da, best_loc_da, best_scale_da = load_bestfit_params(workdir)
    was_cca = _make_cca_op(args, n_modes=args.cca_modes,
                           n_pca_modes=args.cca_pca_modes,
                           dist_method=args.dist_method)
    obs = rainfall.sel(T=slice(str(args.hindcast_years[0]), str(args.hindcast_years[1])))
    cv = configured_cross_validator(args, obs)

    reader_directory = physical_reader_directory(args, dir_model, hind, fcst)
    predictors = {
        k: retrieve_single_zone_for_PCR(str(reader_directory), defined_zone, k, args.hindcast_years[0], args.hindcast_years[1], clim_year_start=args.clim_years[0], clim_year_end=args.clim_years[1], model=True, month_of_initialization=args.init_month, lead_time=args.lead_time)
        for k in center_variable
    }
    
    hind_det_all, hind_prob_all, fcst_det_all, fcst_prob_all = {}, {}, {}, {}
    cv_k = {"best_code_da": best_code_da, "best_shape_da": best_shape_da, "best_loc_da": best_loc_da, "best_scale_da": best_scale_da} if args.dist_method=="bestfit" else {}

    was_verify = WAS_Verification(dist_method=args.dist_method)
    cca_scores = {}
    
    # 1. Collect predictions for all models
    for key, predictor in predictors.items():
        _mdl_plot = _want_selection["mode_selection"] == "mdl"
        if not _mdl_plot:
            try:
                was_cca.plot_cca_results(X=predictor.isel(T=slice(None, -1)), Y=obs, clim_year_start=args.clim_years[0], clim_year_end=args.clim_years[1])
                save_and_close_plot(scores_dir / f"CCA_Modes_{prefix}_{key}.png")
            except: pass
        
        p_train = predictor.isel(T=slice(None, -1))
        # The predictor carries the model initialisation calendar; restamping it
        # onto the predictand is what the CCA classes expect. Verify the years
        # line up BEFORE overwriting them (see assert_predictor_alignment).
        assert_predictor_alignment(
            p_train, obs, label=f"cca/{prefix}/{key}",
            strict=getattr(args, "strict_time_alignment", True),
        )
        p_train['T'] = obs['T']

        hd, hp = cv.cross_validate(
            was_cca, obs, p_train, args.clim_years[0], args.clim_years[1],
            **_residual_probability_controls(args), **cv_k)
        _save_cca_mdl_diagnostics(was_cca, args, obs, cv,
                                  scores_dir / f"CCA_MDL_{prefix}_{key}.json")
        if _mdl_plot:
            _plot_selected_cca_modes(was_cca, args, p_train, obs,
                                     scores_dir / f"CCA_Modes_{prefix}_{key}.png")
        forecast_reference = predictor.isel(T=[-1])
        fd, fp = was_cca.forecast(
            obs, args.clim_years[0], args.clim_years[1],
            p_train, hd, forecast_reference, **cv_k
        )
        # forecast_reference is the SST PREDICTOR's own last available
        # observation (i.e. the initialization date) — not the season being
        # forecast. Every other approach (SV-ML-CMME, obs-lag, analog,
        # EOF-KCCA) tags its forecast with the season's target month via
        # month_of_initialization/lead_time/year_forecast; retag CCA-GCM's
        # output the same way so it can't silently disagree with the other
        # 8 sources on what date this forecast is valid for (see
        # _season_target_time). Only the OUTPUT is retagged — forecast_reference
        # itself is left untouched since was_cca.forecast() above already
        # used it as a genuine scientific input.
        target_T = _season_target_time(args)
        if fd.sizes.get("T") == 1:
            fd = fd.assign_coords(T=("T", np.array([target_T], dtype="datetime64[ns]")))
        if fp.sizes.get("T") == 1:
            fp = fp.assign_coords(T=("T", np.array([target_T], dtype="datetime64[ns]")))
        # Enforce the temporal contract BEFORE any CCA product is cached or
        # written. This makes collect_intermediate_products a reader, not a
        # repair shop.
        hd = _ensure_1d_time(hd, reference=obs, source=f"cca/{prefix}/{key}/hind_det")
        hp = _ensure_1d_time(hp, reference=obs, source=f"cca/{prefix}/{key}/hind_prob")
        fd = _ensure_1d_time(fd, reference=forecast_reference, source=f"cca/{prefix}/{key}/fcst_det")
        fp = _ensure_1d_time(fp, reference=fd, source=f"cca/{prefix}/{key}/fcst_prob")
        
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
            r = was_verify.compute_probabilistic_score(was_verify.get_scores_metadata()[metric][5], obs, hp, args.clim_years[0], args.clim_years[1], **probability_score_kwargs(hp))
            tmp[k_id] = r.compute() 
        cca_scores[metric] = tmp

    # 3. Filter using get_best_models thresholding. Keep exact IDs only.
    model_keys = list(hind_det_all.keys())
    # top_n is None (the multi-domain default) => NO selection at all: every GCM
    # of this domain is kept and averaged equally downstream. Selecting members
    # on the same 24 years that later score them is what inflates the apparent
    # skill; see the note next to SST_DOMAINS for the measured figures.
    if top_n is None:
        best_models = list(model_keys)
        print(f"[cca/{prefix}] no within-domain selection: {len(best_models)} members kept")
    else:
        best_models = get_best_models(
            model_keys, cca_scores, metric="GROC", threshold=0.55,
            top_n=top_n, gcm=False,
        )
    best_models = _unique_preserve_order(
        best_models, label=f"CCA {prefix} best_models"
    )
    available_model_ids = set(model_keys)
    unknown_best_models = [
        model_id for model_id in best_models
        if model_id not in available_model_ids
    ]
    if unknown_best_models:
        raise KeyError(
            f"CCA {prefix}: modèles sélectionnés absents des résultats: "
            f"{unknown_best_models}"
        )
    if not best_models:
        raise RuntimeError(
            f"CCA {prefix}: aucun modèle ne satisfait la sélection GROC."
        )

    # Save the explicitly chosen best models to JSON.
    # Record the size of the pool the members were chosen from. Without it
    # nobody can tell whether a GROC of 0.60 is notable or simply what the
    # maximum of N candidates looks like on 24 years.
    save_json(
        workdir / f"scores/{prefix}_best_models.json",
        {
            "best_models": best_models,
            "candidate_pool": len(model_keys),
            "selection": "none" if top_n is None else "GROC>0.55 top_n",
            "top_n": top_n,
            "sst_domain": (prefix.split("cca_gcm-sst-")[-1]
                           if prefix.startswith("cca_gcm-sst-") else None),
        },
    )

    # ------------------------------------------------------------------ #
    # 4. Two ways to feed this CCA field into consolidation, switched by   #
    #    the --cca-equal-weight flag:                                      #
    #                                                                     #
    #    DEFAULT (flag OFF) — per-member: every selected GCM-driven CCA    #
    #      forecast is written to intermediate/ as its own member, so the  #
    #      final consolidation MME sees them all individually (original    #
    #      behaviour, preserved as the default).                           #
    #                                                                     #
    #    flag ON — equal-weight consensus: the selected members are        #
    #      pre-averaged with equal weights (the *same* WAS_mme_Weighted     #
    #      machinery the consolidation stage itself uses) into ONE balanced #
    #      "CCA-EqWt.<VAR>" product, and only THAT enters consolidation,    #
    #      so a single field can no longer dominate the MME with N near-    #
    #      redundant CCA members. The individual members are still saved    #
    #      to cca_members/ (outside intermediate/) for the bulletin annex.  #
    # ------------------------------------------------------------------ #
    # In multi-domain SST EqWt mode, this function is only a producer of
    # selected RAW CCA members.  The family consensus is built once, outside
    # this function, after all requested SST domains are available.
    if group_member_cache:
        selected = list(best_models)
        member_hind = {}
        member_fcst = {}
        for k in selected:
            hd = _ensure_1d_time(
                hind_det_all[k], reference=obs,
                source=f"cca/{prefix}/{k}/hind_det",
            )
            hp = _ensure_1d_time(
                hind_prob_all[k], reference=obs,
                source=f"cca/{prefix}/{k}/hind_prob",
            )
            fd_ref = fcst_det_all[k]
            fd = _ensure_1d_time(
                fcst_det_all[k], reference=fd_ref,
                source=f"cca/{prefix}/{k}/fcst_det",
            )
            fp = _ensure_1d_time(
                fcst_prob_all[k], reference=fd,
                source=f"cca/{prefix}/{k}/fcst_prob",
            )
            member_hind[f"hdcst_det_{k}"] = hd
            member_hind[f"hdcst_prob_{k}"] = hp
            member_fcst[f"fcst_det_{k}"] = fd
            member_fcst[f"fcst_prob_{k}"] = fp

        xr.Dataset(member_hind).pipe(strip_attrs).to_netcdf(hind_nc)
        xr.Dataset(member_fcst).pipe(strip_attrs).to_netcdf(fcst_nc)
        selected_ids = set(selected)
        selected_scores = {
            metric: {k: v for k, v in values.items() if k in selected_ids}
            for metric, values in cca_scores.items()
        }
        with open(scores_pkl, "wb") as stream:
            pickle.dump(selected_scores, stream, protocol=pickle.HIGHEST_PROTOCOL)

        save_json(chk_file, {
            "prefix": prefix,
            "best_models": selected,
            "equal_weight": False,
            "group_member_cache": True,
            "cca_selection": _want_selection,
            "var_filter": var_filter,
            "defined_zone": {k: list(v) for k, v in defined_zone.items()},
            "top_n": top_n,
            "procedure": _cca_procedure_signature(args),
            "candidate_pool": len(hind_det_all),
        })
        print(
            f"[cca/{prefix}] {len(selected)} raw CCA members cached for "
            "family-level EqWt"
        )
        return selected

    eqwt_id = f"CCA-EqWt.{var_filter}"
    use_eqwt = getattr(args, "cca_equal_weight", False)
    scores_consolidated = {}
    (workdir/"intermediate").mkdir(exist_ok=True, parents=True)

    if use_eqwt:
        # --- Equal-weight CCA consensus -------------------------------- #
        # Stack the selected members along M, exactly like stage_consolidate.
        best_hd_det = {k: hind_det_all[k] for k in best_models}
        best_fd_det = {k: fcst_det_all[k] for k in best_models}

        mme_hdcst, mme_fcst, _, best_score = process_datasets_for_mme(
            obs, hdcsted=best_hd_det, fcsted=best_fd_det, gcm=False,
            Prob=False, best_models=best_models,
            scores=cca_scores, model=False, score_metric="GROC",
        )
        mme_hdcst = mme_hdcst.transpose("T", "M", "Y", "X")
        mme_fcst  = mme_fcst.transpose("T", "M", "Y", "X")

        # Deterministic consensus: arithmetic mean across members.
        # Probabilistic consensus: equal-weight mixture of already calibrated
        # member distributions, preserving their nested/outer procedure.
        eqwt_hd_det = mme_hdcst.mean(dim="M", skipna=True)
        eqwt_fd_det = mme_fcst.isel(T=[-1]).mean(dim="M", skipna=True)
        eqwt_hd_det = _ensure_1d_time(
            eqwt_hd_det, reference=obs, source=f"cca/{prefix}/{eqwt_id}/hind_det"
        )
        eqwt_fd_det = _ensure_1d_time(
            eqwt_fd_det, reference=mme_fcst.isel(T=[-1]),
            source=f"cca/{prefix}/{eqwt_id}/fcst_det"
        )
        eqwt_hd_prob = mean_calibrated_probabilities(
            [hind_prob_all[k] for k in best_models],
            calibration=_cca_probability_calibration(args),
        )
        eqwt_fd_prob = mean_calibrated_probabilities(
            [fcst_prob_all[k] for k in best_models],
            calibration=_cca_probability_calibration(args), forecast=True,
        )
        eqwt_hd_prob = _ensure_1d_time(
            eqwt_hd_prob, reference=obs, source=f"cca/{prefix}/{eqwt_id}/hind_prob"
        )
        eqwt_fd_prob = _ensure_1d_time(
            eqwt_fd_prob, reference=eqwt_fd_det, source=f"cca/{prefix}/{eqwt_id}/fcst_prob"
        )

        # Write ONLY the consensus into intermediate/ → one member per field.
        xr.Dataset({
            f"hdcst_det_{eqwt_id}":  eqwt_hd_det,
            f"hdcst_prob_{eqwt_id}": eqwt_hd_prob,
        }).pipe(strip_attrs).to_netcdf(workdir/f"intermediate/{prefix}_hindcasts.nc")
        xr.Dataset({
            f"fcst_det_{eqwt_id}":  eqwt_fd_det,
            f"fcst_prob_{eqwt_id}": eqwt_fd_prob,
        }).pipe(strip_attrs).to_netcdf(workdir/f"intermediate/{prefix}_forecasts.nc")

        # Keep the individual members for the record, OUTSIDE intermediate/
        # so they are NOT re-globbed into consolidation.
        members_dir = workdir / "cca_members"; members_dir.mkdir(exist_ok=True, parents=True)
        member_hind = {f"hdcst_det_{k}": hind_det_all[k] for k in best_models}
        member_hind.update({f"hdcst_prob_{k}": hind_prob_all[k] for k in best_models})
        member_fcst = {f"fcst_det_{k}": fcst_det_all[k] for k in best_models}
        member_fcst.update({f"fcst_prob_{k}": fcst_prob_all[k] for k in best_models})
        xr.Dataset(member_hind).pipe(strip_attrs).to_netcdf(members_dir/f"{prefix}_members_hindcasts.nc")
        xr.Dataset(member_fcst).pipe(strip_attrs).to_netcdf(members_dir/f"{prefix}_members_forecasts.nc")

        # Score / plot the consensus AND register its GROC/RPSS under eqwt_id.
        # REQUIRED: process_datasets_for_mme in the consolidation stage drops
        # any member that has no matching score entry.
        evaluate_and_plot_forecasts(
            was_verify, obs, eqwt_hd_det, eqwt_hd_prob, eqwt_fd_prob,
            f"CCA_EqWt_{var_filter}", eqwt_id, args.clim_years[0], args.clim_years[1],
            scores_consolidated, str(scores_dir), str(out_dir), season_to_str(args.season_months),
            args.forecast_year, args.init_month, FCST_LABELS_FR, args.country_code, args.logo
        )

        # Diagnostics only: still plot each selected member individually
        # (these do NOT enter the MME under the equal-weight path).
        dummy_scores = {}
        for k in best_models:
            evaluate_and_plot_forecasts(
                was_verify, obs, hind_det_all[k], hind_prob_all[k], fcst_prob_all[k],
                f"CCA_{k}", k, args.clim_years[0], args.clim_years[1],
                dummy_scores, str(scores_dir), str(out_dir), season_to_str(args.season_months),
                args.forecast_year, args.init_month, FCST_LABELS_FR, args.country_code, args.logo
            )

    else:
        # --- Per-member (original default) ----------------------------- #
        best_hind = {f"hdcst_det_{k}": hind_det_all[k] for k in best_models}
        best_hind.update({f"hdcst_prob_{k}": hind_prob_all[k] for k in best_models})
        best_fcst = {f"fcst_det_{k}": fcst_det_all[k] for k in best_models}
        best_fcst.update({f"fcst_prob_{k}": fcst_prob_all[k] for k in best_models})

        xr.Dataset(best_hind).to_netcdf(workdir/f"intermediate/{prefix}_hindcasts.nc")
        xr.Dataset(best_fcst).to_netcdf(workdir/f"intermediate/{prefix}_forecasts.nc")

        # Evaluate and Plot ONLY the best models
        dummy_scores = {}
        for k in best_models:
            evaluate_and_plot_forecasts(
                was_verify, obs, hind_det_all[k], hind_prob_all[k], fcst_prob_all[k],
                f"CCA_{k}", k, args.clim_years[0], args.clim_years[1],
                dummy_scores, str(scores_dir), str(out_dir), season_to_str(args.season_months),
                args.forecast_year, args.init_month, FCST_LABELS_FR, args.country_code, args.logo
            )

        # Register only exact selected IDs. Prefix matching could retain a
        # different model whose name merely starts with a selected ID.
        selected_model_ids = set(best_models)
        scores_consolidated = {
            metric: {
                model_id: score
                for model_id, score in model_scores.items()
                if model_id in selected_model_ids
            }
            for metric, model_scores in cca_scores.items()
        }

        missing_score_ids = {
            metric: sorted(selected_model_ids - set(model_scores))
            for metric, model_scores in scores_consolidated.items()
            if selected_model_ids - set(model_scores)
        }
        if missing_score_ids:
            raise KeyError(
                f"CCA {prefix}: scores manquants pour les modèles sélectionnés: "
                f"{missing_score_ids}"
            )

    save_scores(workdir, scores_consolidated, f"scores_consolidated_{prefix}.pkl")

    # Write the per-pipeline checkpoint so subsequent runs skip this stage.
    save_json(chk_file, {"prefix": prefix, "best_models": best_models,
                         "equal_weight": bool(use_eqwt),
                         "group_member_cache": False,
                         "cca_selection": _want_selection,
                         "eqwt_member": eqwt_id if use_eqwt else None,
                         "var_filter": var_filter,
                         "defined_zone": {k: list(v) for k, v in defined_zone.items()},
                         "procedure": _cca_procedure_signature(args),
                         "top_n": top_n,
                         "candidate_pool": len(hind_det_all),
                         "policy": args.obs_lag_policy if hasattr(args, "obs_lag_policy") else ""})
    return best_models



def _combine_cca_sst_equal_weight(args, workdir: Path, domain_prefixes):
    """Build ONE ``CCA-EqWt.SST`` from every cached CCA-SST member.

    This is the multi-domain extension of the historical EqWt behaviour:
    all selected GCM-driven CCA-SST members, regardless of SST domain, receive
    exactly the same weight.  Domain-level consensuses are never fed to the
    final MME, so adding a domain cannot give the CCA family an extra vote.
    """
    rainfall = xr.load_dataarray(workdir / "rainfall_predictand.nc")
    obs = rainfall.sel(
        T=slice(str(args.hindcast_years[0]), str(args.hindcast_years[1]))
    ).transpose("T", "Y", "X")
    inter_dir = workdir / "intermediate"
    members_dir = workdir / "cca_members"
    scores_dir = workdir / "scores"
    out_dir = workdir / "forecasts"

    # Remove stale v8 per-domain products so collect_intermediate_products()
    # can see only the one family member created below.
    _remove_matching_files(
        inter_dir,
        [
            "cca_gcm-sst-*_hindcasts.nc",
            "cca_gcm-sst-*_forecasts.nc",
            "scores_consolidated_cca_gcm-sst-*.pkl",
            "cca_gcm-sst_hindcasts.nc",
            "cca_gcm-sst_forecasts.nc",
            "scores_consolidated_cca_gcm-sst.pkl",
        ],
        label="cca/sst-family-eqwt",
    )

    hdet, fdet, hprob, fprob = {}, {}, {}, {}
    diagnostic_scores = {}
    provenance = {}
    for prefix in domain_prefixes:
        hpath = members_dir / f"{prefix}_members_hindcasts.nc"
        fpath = members_dir / f"{prefix}_members_forecasts.nc"
        spath = members_dir / f"scores_{prefix}.pkl"
        if not (hpath.exists() and fpath.exists() and spath.exists()):
            raise FileNotFoundError(
                f"CCA-SST family cache incomplete for {prefix}: "
                f"{hpath.name}, {fpath.name}, {spath.name}"
            )
        hds = xr.load_dataset(hpath)
        fds = xr.load_dataset(fpath)
        with open(spath, "rb") as stream:
            block_scores = pickle.load(stream)
        merge_score_block(
            diagnostic_scores, block_scores,
            block_name=f"cca-members/{prefix}", duplicate_policy="raise",
        )

        for name, da in hds.data_vars.items():
            if not name.startswith("hdcst_det_"):
                continue
            member_id = name.removeprefix("hdcst_det_")
            if member_id in hdet:
                raise KeyError(
                    f"CCA-SST member duplique: {member_id} dans "
                    f"{provenance[member_id]} et {hpath.name}"
                )
            forecast_name = f"fcst_det_{member_id}"
            if forecast_name not in fds:
                raise KeyError(f"{fpath.name}: {forecast_name} absent")
            hdet[member_id] = _ensure_1d_time(
                da, reference=obs, source=f"{hpath.name}:{name}"
            ).transpose("T", "Y", "X")
            fdet[member_id] = _ensure_1d_time(
                fds[forecast_name], reference=fds[forecast_name],
                source=f"{fpath.name}:{forecast_name}",
            ).transpose("T", "Y", "X")
            hind_prob_name = f"hdcst_prob_{member_id}"
            forecast_prob_name = f"fcst_prob_{member_id}"
            if hind_prob_name not in hds or forecast_prob_name not in fds:
                raise KeyError(f"CCA-SST calibrated probabilities absent for {member_id}")
            hprob[member_id] = _ensure_1d_time(
                hds[hind_prob_name], reference=obs,
                source=f"{hpath.name}:{hind_prob_name}",
            )
            fprob[member_id] = _ensure_1d_time(
                fds[forecast_prob_name], reference=fdet[member_id],
                source=f"{fpath.name}:{forecast_prob_name}",
            )
            provenance[member_id] = hpath.name

    member_ids = list(hdet)
    if not member_ids:
        raise RuntimeError("Aucun membre CCA-SST disponible pour CCA-EqWt.SST.")

    # Every CCA is already predicted on the predictand grid. xr.concat uses
    # the exact coordinates and exposes any accidental incompatibility now.
    hd_stack = xr.concat(
        [hdet[k] for k in member_ids],
        dim=xr.IndexVariable("M", member_ids),
        join="exact",
    ).transpose("T", "M", "Y", "X")

    # All operational members must refer to the same target date. Normalise
    # them to the first forecast member after checking the T length.
    fcst_ref = fdet[member_ids[0]]
    fd_fields = []
    for k in member_ids:
        da = fdet[k]
        if int(da.sizes.get("T", 0)) != int(fcst_ref.sizes.get("T", 0)):
            raise ValueError(
                f"CCA-SST forecast T mismatch: {k}={da.sizes.get('T')} vs "
                f"{member_ids[0]}={fcst_ref.sizes.get('T')}"
            )
        da = da.assign_coords(
            T=xr.IndexVariable("T", np.asarray(fcst_ref["T"].values))
        )
        fd_fields.append(da)
    fd_stack = xr.concat(
        fd_fields, dim=xr.IndexVariable("M", member_ids), join="exact"
    ).transpose("T", "M", "Y", "X")

    # Every CCA has weight 1/N, independent of domain and centre.
    # Keep the deterministic mean and pool calibrated member probabilities;
    # this is a mixture, not a recalibrated distribution of that mean.
    eqwt_hd_det = hd_stack.mean("M", skipna=True).assign_coords(T=obs["T"])
    eqwt_fd_det = fd_stack.mean("M", skipna=True).assign_coords(T=fcst_ref["T"])

    eqwt_hd_prob = mean_calibrated_probabilities(
        [hprob[k] for k in member_ids], calibration=_cca_probability_calibration(args),
    )
    eqwt_fd_prob = mean_calibrated_probabilities(
        [fprob[k].assign_coords(T=fcst_ref["T"]) for k in member_ids],
        calibration=_cca_probability_calibration(args), forecast=True,
    )
    eqwt_hd_prob = _ensure_1d_time(
        eqwt_hd_prob, reference=obs, source="CCA-EqWt.SST/hind_prob"
    )
    eqwt_fd_prob = _ensure_1d_time(
        eqwt_fd_prob, reference=eqwt_fd_det, source="CCA-EqWt.SST/fcst_prob"
    )

    eqwt_id = "CCA-EqWt.SST"
    stage_scores = {}
    was_verify = WAS_Verification(dist_method=args.dist_method)
    evaluate_and_plot_forecasts(
        was_verify, obs, eqwt_hd_det, eqwt_hd_prob, eqwt_fd_prob,
        "CCA_EqWt_SST", eqwt_id,
        args.clim_years[0], args.clim_years[1], stage_scores,
        str(scores_dir), str(out_dir), season_to_str(args.season_months),
        args.forecast_year, args.init_month, FCST_LABELS_FR,
        args.country_code, args.logo,
    )

    xr.Dataset({
        f"hdcst_det_{eqwt_id}": eqwt_hd_det,
        f"hdcst_prob_{eqwt_id}": eqwt_hd_prob,
    }).pipe(strip_attrs).to_netcdf(inter_dir / "cca_gcm-sst_hindcasts.nc")
    xr.Dataset({
        f"fcst_det_{eqwt_id}": eqwt_fd_det,
        f"fcst_prob_{eqwt_id}": eqwt_fd_prob,
    }).pipe(strip_attrs).to_netcdf(inter_dir / "cca_gcm-sst_forecasts.nc")
    save_scores(workdir, stage_scores, "scores_consolidated_cca_gcm-sst.pkl")

    manifest = {
        "member": eqwt_id,
        "weighting": "equal",
        "n_members": len(member_ids),
        "members": member_ids,
        "domains": list(args.sst_domain_map),
        "per_member_weight": 1.0 / len(member_ids),
        "probability_policy": "equal_weight_member_probabilities",
        "probability_calibration": _cca_probability_calibration(args),
        "probability_schema": 5,
    }
    save_json(scores_dir / "cca_gcm-sst_eqwt_manifest.json", manifest)
    save_json(
        workdir / "checkpoints" / "chk_cca_gcm-sst_group.json",
        {**manifest, "domain_prefixes": list(domain_prefixes)},
    )
    print(
        f"[cca/sst-family-eqwt] {eqwt_id}: {len(member_ids)} CCA members "
        f"across {len(args.sst_domain_map)} SST domains; each weight="
        f"{1.0/len(member_ids):.6f}"
    )
    return eqwt_id


def stage_cca_gcm_sst(args, workdir: Path, downloader: "WAS_Download"):
    """Run all requested GCM-SST CCA domains with historical EqWt semantics."""
    prefixes = []
    if getattr(args, "cca_equal_weight", False):
        # Remove stale standard/per-domain members immediately. If a domain run
        # later fails, a user cannot accidentally consolidate an old mixed mode.
        _remove_matching_files(
            workdir / "intermediate",
            [
                "cca_gcm-sst-*_hindcasts.nc",
                "cca_gcm-sst-*_forecasts.nc",
                "scores_consolidated_cca_gcm-sst-*.pkl",
                "cca_gcm-sst_hindcasts.nc",
                "cca_gcm-sst_forecasts.nc",
                "scores_consolidated_cca_gcm-sst.pkl",
            ],
            label="cca/sst-family-eqwt-preclean",
        )
        # EqWt ON: domains are only caches/diagnostics. ONE family consensus is
        # built after every requested domain is available.
        for name, coords in args.sst_domain_map.items():
            prefix = f"cca_gcm-sst-{name}"
            prefixes.append(prefix)
            _execute_cca_pipeline(
                args, workdir, downloader, "SST", args.sst_extent,
                {name: coords}, prefix,
                top_n=(args.cca_domain_top_n or None),
                group_member_cache=True,
            )
        return _combine_cca_sst_equal_weight(args, workdir, prefixes)

    # EqWt OFF: preserve the v8 standard experiment. Every retained CCA member
    # enters final consolidation individually. Remove a stale grouped consensus
    # if this workdir was previously run in EqWt mode.
    _remove_matching_files(
        workdir / "intermediate",
        [
            "cca_gcm-sst_hindcasts.nc",
            "cca_gcm-sst_forecasts.nc",
            "scores_consolidated_cca_gcm-sst.pkl",
        ],
        label="cca/sst-standard",
    )
    results = []
    for name, coords in args.sst_domain_map.items():
        prefix = f"cca_gcm-sst-{name}"
        results.append(_execute_cca_pipeline(
            args, workdir, downloader, "SST", args.sst_extent,
            {name: coords}, prefix,
            top_n=(args.cca_domain_top_n or None),
            group_member_cache=False,
        ))
    return results


# =========================
# Stage 5 — Obs-Lag
# =========================
def stage_obs_lag(args, workdir: Path, downloader: "WAS_Download"):
    """Run OBS-LAG methods and save one non-cumulative score registry."""
    check_dir = workdir / "checkpoints"
    _want_selection = _cca_procedure_signature(args)
    _uses_cca = args.obs_lag_policy in ("both", "cca")
    if done(check_dir, "obs_lag") and not args.redo:
        saved = load_json(check_dir / CHECKPOINTS["obs_lag"])
        saved_procedure = saved.get("cca_procedure") or saved.get("cca_selection")
        if (saved_procedure == _want_selection and
                saved.get("policy", args.obs_lag_policy) == args.obs_lag_policy):
            print("[obs-lag] checkpoint exists → skip")
            return
        log_progress("OBS-LAG probability or model procedure changed → recomputing")
    _cca_invalidate_consolidation(workdir)
    (check_dir / CHECKPOINTS["obs_lag"]).unlink(missing_ok=True)

    rainfall = xr.load_dataarray(workdir / "rainfall_predictand.nc")
    dir_rea = workdir / "reanalysis_data"
    scores_dir = workdir / "scores"
    out_dir = workdir / "forecasts"
    inter_dir = workdir / "intermediate"
    for directory in (dir_rea, scores_dir, out_dir, inter_dir):
        directory.mkdir(exist_ok=True, parents=True)

    # A failed or changed previous OBS-LAG run must not leak stale models into
    # the next consolidation. All legacy cumulative files are removed here.
    _remove_matching_files(
        inter_dir,
        [
            "scores_consolidated_obslag*.pkl",
            "obs_lag_*_hindcasts.nc",
            "obs_lag_*_forecasts.nc",
        ],
        label="obs-lag",
    )

    downloader.WAS_Download_Reanalysis(
        str(dir_rea),
        ["NOAA.SST"],
        args.obs_years[0],
        args.obs_years[1] + 1,
        args.sst_extent,
        args.season_months_lag,
        args.force,
    )
    was_verify = WAS_Verification(dist_method=args.dist_method)

    idx_names = ["NINO34", "TNA", "TSA", "DMI", "MB"]
    predictors = compute_sst_indices(
        str(dir_rea),
        idx_names,
        "NOAA.SST",
        args.obs_years[0],
        args.obs_years[1] + 1,
        args.season_months_lag,
        args.clim_years[0],
        args.clim_years[1],
        {},
    )
    dfp = predictors.to_dataframe()
    vif = pd.DataFrame(
        {
            "feature": dfp.columns,
            "VIF": [VIF(dfp, index) for index in range(dfp.shape[1])],
        }
    )
    keep = vif.loc[vif.VIF < args.vif_threshold, "feature"].tolist()
    if not keep:
        raise RuntimeError(
            "Aucun indice SST n'a été retenu après le filtrage VIF. "
            "Augmentez --vif-threshold ou vérifiez les séries d'indices."
        )
    filt_pred = (
        predictors[keep]
        .to_array()
        .rename({"variable": "features"})
        .transpose("T", "features")
    )

    best_code_da, best_shape_da, best_loc_da, best_scale_da = load_bestfit_params(workdir)
    cv_k = (
        {
            "best_code_da": best_code_da,
            "best_shape_da": best_shape_da,
            "best_loc_da": best_loc_da,
            "best_scale_da": best_scale_da,
        }
        if args.dist_method == "bestfit"
        else {}
    )

    cv = configured_cross_validator(args, rainfall)
    obs_lag_scores = {}
    produced_models = []

    if args.obs_lag_policy in ("both", "mlr"):
        log_progress("Processing Observation-Based with Lag (MLR)")
        model_id = "MLR_VIF.OBS-LAG"
        model_scores = {}
        model_mlr = WAS_LinearRegression_Model(
            nb_cores=args.ncores,
            dist_method=args.dist_method,
        )
        fp_train = filt_pred.isel(T=slice(None, -1))
        fp_train["T"] = rainfall["T"]
        hd_mlr, hp_mlr = cv.cross_validate(
            model_mlr,
            rainfall,
            fp_train,
            args.clim_years[0],
            args.clim_years[1],
            **_residual_probability_controls(args),
            **cv_k,
        )
        f_pred = filt_pred.isel(T=[-1]).assign_coords(
            T=[
                pd.Timestamp(_season_target_time(args))
            ]
        )
        fd_mlr, fp_mlr = model_mlr.forecast(
            rainfall,
            args.clim_years[0],
            args.clim_years[1],
            fp_train,
            hd_mlr,
            f_pred,
            **cv_k,
        )
        evaluate_and_plot_forecasts(
            was_verify,
            rainfall,
            hd_mlr,
            hp_mlr,
            fp_mlr,
            model_id,
            model_id,
            args.clim_years[0],
            args.clim_years[1],
            model_scores,
            str(scores_dir),
            str(out_dir),
            season_to_str(args.season_months),
            args.forecast_year,
            args.init_month,
            FCST_LABELS_FR,
            args.country_code,
            args.logo,
        )
        merge_score_block(
            obs_lag_scores,
            model_scores,
            block_name=model_id,
        )
        hd_mlr = _ensure_1d_time(hd_mlr, reference=rainfall)
        hp_mlr = _ensure_1d_time(hp_mlr, reference=rainfall)
        fd_mlr = _ensure_1d_time(fd_mlr, reference=f_pred)
        fp_mlr = _ensure_1d_time(fp_mlr, reference=fd_mlr)
        xr.Dataset(
            {
                f"hdcst_det_{model_id}": hd_mlr,
                f"hdcst_prob_{model_id}": hp_mlr,
            }
        ).pipe(strip_attrs).to_netcdf(inter_dir / "obs_lag_mlr_hindcasts.nc")
        xr.Dataset(
            {
                f"fcst_det_{model_id}": fd_mlr,
                f"fcst_prob_{model_id}": fp_mlr,
            }
        ).pipe(strip_attrs).to_netcdf(inter_dir / "obs_lag_mlr_forecasts.nc")
        produced_models.append(model_id)

    if args.obs_lag_policy in ("both", "cca"):
        log_progress("Processing Observation-Based with Lag (CCA — SST domains)")

        # Approach 3 differs from approach 2: there is no get_best_models filter
        # here, so K domains give K members directly and there is no selection
        # bias. The risk is redundancy instead -- K near-identical OBS-LAG CCA
        # members would let this single family out-vote the rest of the MME by
        # sheer count. Hence one equal-weight consensus by default.
        domain_map = getattr(args, "sst_domain_map", None) or resolve_sst_domains(
            DEFAULT_SST_DOMAINS
        )
        per_domain_hd, per_domain_hp = {}, {}
        per_domain_fd, per_domain_fp = {}, {}

        # Per-domain cache. The stage-level checkpoint (chk_obs_lag.json) is
        # all-or-nothing: without this, a crash on the fourth domain forces the
        # first three to be recomputed on the next run. Each domain writes its
        # own pair under intermediate/obs_lag_cca_domains/, which is NOT globbed
        # by collect_intermediate_products (it only scans intermediate/ itself).
        dom_cache = inter_dir / "obs_lag_cca_domains"
        dom_cache.mkdir(exist_ok=True, parents=True)

        for dom_name, dom_coords in domain_map.items():
            member_id = f"CCA_OBS-LAG_{dom_name}"
            dom_h = dom_cache / f"{dom_name}_hindcasts.nc"
            dom_f = dom_cache / f"{dom_name}_forecasts.nc"
            dom_meta = dom_cache / f"{dom_name}_meta.json"

            if dom_h.exists() and dom_f.exists() and dom_meta.exists() and not args.redo:
                meta = load_json(dom_meta)
                if (meta.get("coords") == list(dom_coords)
                        and meta.get("cca_selection") == _want_selection):
                    log_progress(f"  OBS-LAG CCA domain {dom_name}: cache hit")
                    dh = xr.load_dataset(dom_h)
                    df = xr.load_dataset(dom_f)
                    per_domain_hd[member_id] = _ensure_1d_time(
                        dh[f"hdcst_det_{member_id}"], reference=rainfall)
                    per_domain_hp[member_id] = _ensure_1d_time(
                        dh[f"hdcst_prob_{member_id}"], reference=rainfall)
                    per_domain_fd[member_id] = _ensure_1d_time(df[f"fcst_det_{member_id}"])
                    per_domain_fp[member_id] = _ensure_1d_time(df[f"fcst_prob_{member_id}"])
                    continue
                log_progress(
                    f"  OBS-LAG CCA domain {dom_name}: bounds or selection changed "
                    f"({meta.get('coords')} -> {list(dom_coords)}), recomputing")

            log_progress(f"  OBS-LAG CCA domain: {dom_name}")
            model_scores = {}

            predictor = retrieve_single_zone_for_PCR(
                str(dir_rea),
                # ONE key on purpose: retrieve_single_zone_for_PCR overwrites its
                # accumulator on every iteration and returns only the last zone,
                # so a multi-key dict would silently drop all but one domain.
                {dom_name: dom_coords},
                "NOAA.SST",
                args.obs_years[0],
                args.obs_years[1] + 1,
                args.season_months_lag,
                args.clim_years[0],
                args.clim_years[1],
            )

            predictor_f = predictor.isel(T=slice(None, -1))
            assert_predictor_alignment(
                predictor_f, rainfall, label=f"obs-lag/cca/{dom_name}",
                strict=getattr(args, "strict_time_alignment", True),
            )
            predictor_f["T"] = rainfall["T"]

            model_cca = _make_cca_strict(args, n_modes=args.cca_modes,
                                         n_pca_modes=args.cca_pca_modes,
                                         dist_method=args.dist_method)
            hd_cca, hp_cca = cv.cross_validate(
                model_cca,
                rainfall,
                predictor_f,
                args.clim_years[0],
                args.clim_years[1],
                **_residual_probability_controls(args),
                **cv_k,
            )
            _save_cca_mdl_diagnostics(model_cca, args, rainfall, cv,
                                      dom_cache / f"{dom_name}_mdl.json")
            f_predictors = predictor.isel(T=[-1]).assign_coords(
                T=[
                    pd.Timestamp(_season_target_time(args))
                ]
            )
            fd_cca, fp_cca = model_cca.forecast(
                rainfall,
                args.clim_years[0],
                args.clim_years[1],
                predictor_f,
                hd_cca,
                f_predictors,
                **cv_k,
            )

            evaluate_and_plot_forecasts(
                was_verify,
                rainfall,
                hd_cca,
                hp_cca,
                fp_cca,
                member_id,
                member_id,
                args.clim_years[0],
                args.clim_years[1],
                model_scores,
                str(scores_dir),
                str(out_dir),
                season_to_str(args.season_months),
                args.forecast_year,
                args.init_month,
                FCST_LABELS_FR,
                args.country_code,
                args.logo,
            )
            merge_score_block(obs_lag_scores, model_scores, block_name=member_id)

            per_domain_hd[member_id] = hd_cca
            per_domain_hp[member_id] = hp_cca
            per_domain_fd[member_id] = fd_cca
            per_domain_fp[member_id] = fp_cca

            # Cache this domain before moving to the next one.
            xr.Dataset({
                f"hdcst_det_{member_id}": hd_cca,
                f"hdcst_prob_{member_id}": hp_cca,
            }).pipe(strip_attrs).to_netcdf(dom_h)
            xr.Dataset({
                f"fcst_det_{member_id}": fd_cca,
                f"fcst_prob_{member_id}": fp_cca,
            }).pipe(strip_attrs).to_netcdf(dom_f)
            save_json(dom_meta, {"domain": dom_name, "coords": list(dom_coords),
                                 "cca_selection": _want_selection})

        # ------------------------------------------------------------------ #
        # What actually reaches consolidation.                                #
        # Default: ONE equal-weight consensus over the domains, so the OBS-LAG #
        # CCA family contributes a single member as it did before.            #
        # --cca-per-domain-members keeps them separate, for diagnosis only.   #
        # ------------------------------------------------------------------ #
        if len(per_domain_hd) == 1 or getattr(args, "cca_per_domain_members", False):
            hd_out, hp_out = dict(per_domain_hd), dict(per_domain_hp)
            fd_out, fp_out = dict(per_domain_fd), dict(per_domain_fp)
        else:
            consensus_id = "CCA_OBS-LAG"
            members = list(per_domain_hd)
            # Equal weights on purpose. Weighting the domains by their own
            # cross-validated score would be a second selection on the same 24
            # years, which is exactly what inflates the apparent skill.
            mme_hdcst, mme_fcst, _, _ = process_datasets_for_mme(
                rainfall,
                hdcsted=per_domain_hd,
                fcsted=per_domain_fd,
                gcm=False,
                Prob=False,
                best_models=members,
                scores=None,
                model=False,
            )
            hd_cons = mme_hdcst.transpose("T", "M", "Y", "X").mean(dim="M")
            fd_cons = mme_fcst.transpose("T", "M", "Y", "X").mean(dim="M")
            hd_cons = conditional_nonnegative_clip(hd_cons, rainfall)
            fd_cons = conditional_nonnegative_clip(fd_cons, rainfall)

            # Preserve the chosen calibration of each domain by pooling its
            # probabilities. Recalibrating pooled hindcast residuals here would
            # bypass nested calibration and use verification outcomes again.
            hp_cons = mean_calibrated_probabilities(
                [per_domain_hp[k] for k in members],
                calibration=_cca_probability_calibration(args),
            )
            fp_cons = mean_calibrated_probabilities(
                [per_domain_fp[k] for k in members],
                calibration=_cca_probability_calibration(args), forecast=True,
            )

            consensus_scores = {}
            evaluate_and_plot_forecasts(
                was_verify,
                rainfall,
                hd_cons,
                hp_cons,
                fp_cons,
                consensus_id,
                consensus_id,
                args.clim_years[0],
                args.clim_years[1],
                consensus_scores,
                str(scores_dir),
                str(out_dir),
                season_to_str(args.season_months),
                args.forecast_year,
                args.init_month,
                FCST_LABELS_FR,
                args.country_code,
                args.logo,
            )
            merge_score_block(
                obs_lag_scores, consensus_scores, block_name=consensus_id
            )

            hd_out = {consensus_id: _ensure_1d_time(hd_cons, reference=rainfall)}
            hp_out = {consensus_id: _ensure_1d_time(hp_cons, reference=rainfall)}
            fd_out = {consensus_id: _ensure_1d_time(fd_cons)}
            fp_out = {consensus_id: _ensure_1d_time(fp_cons)}

        hind_vars = {f"hdcst_det_{mid}": da for mid, da in hd_out.items()}
        hind_vars.update({f"hdcst_prob_{mid}": da for mid, da in hp_out.items()})
        xr.Dataset(hind_vars).pipe(strip_attrs).to_netcdf(
            inter_dir / "obs_lag_cca_hindcasts.nc"
        )
        fcst_vars = {f"fcst_det_{mid}": da for mid, da in fd_out.items()}
        fcst_vars.update({f"fcst_prob_{mid}": da for mid, da in fp_out.items()})
        xr.Dataset(fcst_vars).pipe(strip_attrs).to_netcdf(
            inter_dir / "obs_lag_cca_forecasts.nc"
        )

        save_json(
            scores_dir / "obs_lag_cca_domains.json",
            {
                "domains_tested": list(domain_map),
                "candidate_pool": len(domain_map),
                "members_written": list(hd_out),
                "consensus": len(hd_out) == 1 and len(domain_map) > 1,
            },
        )
        produced_models.extend(hd_out)

    if args.obs_lag_policy in ("both", "pcr"):
        log_progress(
            f"Processing Observation-Based with Lag (PCR — fields: {args.pcr_fields})"
        )
        pcr_zone_global = {"A": ("A", -150, 150, -45, 45)}
        pcr_zone_wind = {
            "A": (
                "A",
                args.wind_extent[1],
                args.wind_extent[3],
                args.wind_extent[2],
                args.wind_extent[0],
            )
        }
        pcr_field_catalog = {
            "SST": ("NOAA.SST", pcr_zone_global, args.sst_extent),
            "UGRD_850": ("ERA5.UGRD_850", pcr_zone_wind, args.wind_extent),
            "VGRD_850": ("ERA5.VGRD_850", pcr_zone_wind, args.wind_extent),
            "TEMP": ("ERA5.TEMP", pcr_zone_global, args.sst_extent),
        }

        requested = list(dict.fromkeys(
            field for field in args.pcr_fields if field in pcr_field_catalog
        ))
        unknown = [field for field in args.pcr_fields if field not in pcr_field_catalog]
        if unknown:
            print(f"[obs-lag/pcr] ignoring unknown field(s): {unknown}")

        pcr_year_start, pcr_year_end = lagged_pcr_year_bounds(
            rainfall, args.season_months_lag, _season_target_time(args), args.init_month)

        for field_name in requested:
            cvar, _, extent = pcr_field_catalog[field_name]
            try:
                downloader.WAS_Download_Reanalysis(
                    str(dir_rea),
                    [cvar],
                    pcr_year_start,
                    pcr_year_end,
                    extent,
                    args.season_months_lag,
                    args.force,
                )
            except Exception as exc:
                print(f"[obs-lag/pcr] download of {cvar} failed (skipped): {exc}")

        predictor_list = []
        predictor_list_f = []
        used_fields = []
        for field_name in requested:
            cvar, zone, _ = pcr_field_catalog[field_name]
            try:
                field = retrieve_single_zone_for_PCR(
                    str(dir_rea),
                    zone,
                    cvar,
                    pcr_year_start,
                    pcr_year_end,
                    args.season_months_lag,
                    args.clim_years[0],
                    args.clim_years[1],
                    standardize=False,
                )
                field_train, field_year = align_lagged_pcr_fields(
                    field, rainfall, args.season_months_lag,
                    _season_target_time(args), args.init_month)
                predictor_list.append(field_train)
                predictor_list_f.append(field_year)
                used_fields.append(field_name)
            except Exception as exc:
                print(f"[obs-lag/pcr] field {field_name} unavailable (skipped): {exc}")

        if not predictor_list:
            print("[obs-lag/pcr] no predictor field available → skipping PCR block")
        else:
            print(
                "[obs-lag/pcr] PCR predictor field(s) actually used: "
                f"{used_fields}"
            )
            multivariate = len(predictor_list) > 1
            predictor_train = predictor_list if multivariate else predictor_list[0]
            predictor_year = predictor_list_f if multivariate else predictor_list_f[0]
            pcr_specs = {
                "PCR-MLR.OBS-LAG": WAS_PCR(
                    regression_model=WAS_LinearRegression_Model(
                        nb_cores=args.ncores,
                        dist_method=args.dist_method,
                    ),
                    n_modes=args.cca_pca_modes,
                    use_coslat=True,
                    standardize=True,
                    detrend=True,
                    normalize=multivariate,
                ),
                "PCR-Ridge.OBS-LAG": WAS_PCR(
                    regression_model=WAS_Ridge_Model(
                        n_clusters=6,
                        alpha_range=np.logspace(-4, 0.1, 20),
                        nb_cores=args.ncores,
                        dist_method=args.dist_method,
                    ),
                    n_modes=args.cca_pca_modes,
                    use_coslat=True,
                    standardize=True,
                    detrend=True,
                    normalize=multivariate,
                ),
            }

            for model_id, pcr_model in pcr_specs.items():
                log_progress(f"Processing {model_id}")
                model_scores = {}
                # No full-period Ridge alpha is passed into the outer CV.
                # was_cross_validate v7 tunes alpha on each outer TRAIN fold.
                cv_kwargs_pcr = {**cv_k, **_residual_probability_controls(args),
                                 "nested_tuning": bool(getattr(args, "obs_lag_nested_tuning", True)),
                                 "pcr_inner_folds": int(getattr(args, "obs_lag_inner_folds", 3))}
                fixed_pcr_parameters = {}
                if (isinstance(pcr_model.reg_model, WAS_Ridge_Model)
                        and not bool(getattr(args, "obs_lag_nested_tuning", True))):
                    fixed_pcr_parameters["alpha"] = xr.full_like(
                        rainfall.isel(T=0, drop=True), float(getattr(args, "obs_lag_ridge_alpha", 1.0)))
                    cv_kwargs_pcr.update(fixed_pcr_parameters)
                hd_pcr, hp_pcr = cv.cross_validate(
                    pcr_model,
                    rainfall,
                    predictor_train,
                    args.clim_years[0],
                    args.clim_years[1],
                    **cv_kwargs_pcr,
                )

                # Full-period tuning is legitimate only for the operational forecast.
                forecast_kwargs_pcr = {**cv_k, **fixed_pcr_parameters}
                if (isinstance(pcr_model.reg_model, WAS_Ridge_Model)
                        and bool(getattr(args, "obs_lag_nested_tuning", True))):
                    from wass2s.was_cross_validate import _tune_pcr_raw
                    # Select on raw fields: refit EOFs inside every selection fold,
                    # then forecast fits its final EOF on all historical years.
                    forecast_kwargs_pcr.update(_tune_pcr_raw(
                        pcr_model, predictor_train, rainfall,
                        args.clim_years[0], args.clim_years[1], cv.nb_omit,
                        {"pcr_inner_folds": int(getattr(args, "obs_lag_inner_folds", 3))},
                    ))
                fd_pcr, fp_pcr = pcr_model.forecast(
                    rainfall,
                    args.clim_years[0],
                    args.clim_years[1],
                    predictor_train,
                    hd_pcr,
                    predictor_year,
                    **forecast_kwargs_pcr,
                )
                evaluate_and_plot_forecasts(
                    was_verify,
                    rainfall,
                    hd_pcr,
                    hp_pcr,
                    fp_pcr,
                    model_id,
                    model_id,
                    args.clim_years[0],
                    args.clim_years[1],
                    model_scores,
                    str(scores_dir),
                    str(out_dir),
                    season_to_str(args.season_months),
                    args.forecast_year,
                    args.init_month,
                    FCST_LABELS_FR,
                    args.country_code,
                    args.logo,
                )
                merge_score_block(
                    obs_lag_scores,
                    model_scores,
                    block_name=model_id,
                )

                tag = model_id.split(".")[0].lower().replace("-", "_")
                xr.Dataset(
                    {
                        f"hdcst_det_{model_id}": hd_pcr,
                        f"hdcst_prob_{model_id}": hp_pcr,
                    }
                ).pipe(strip_attrs).to_netcdf(
                    inter_dir / f"obs_lag_{tag}_hindcasts.nc"
                )
                xr.Dataset(
                    {
                        f"fcst_det_{model_id}": fd_pcr,
                        f"fcst_prob_{model_id}": fp_pcr,
                    }
                ).pipe(strip_attrs).to_netcdf(
                    inter_dir / f"obs_lag_{tag}_forecasts.nc"
                )
                produced_models.append(model_id)
                gc.collect()

    if not obs_lag_scores:
        raise RuntimeError(
            "Aucun score OBS-LAG n'a été produit. Vérifiez --obs-lag-policy."
        )

    score_path = save_scores(
        workdir,
        obs_lag_scores,
        "scores_consolidated_obslag.pkl",
    )
    print(
        f"[obs-lag] fichier de scores unique: {score_path.name}; "
        f"modèles={produced_models}"
    )
    mark_done(
        check_dir,
        "obs_lag",
        {
            "policy": args.obs_lag_policy,
            "cca_selection": _want_selection if _uses_cca else None,
            "score_file": score_path.name,
            "models": produced_models,
            "cca_procedure": _cca_procedure_signature(args),
        },
    )


# =========================
# Stage 6 — Analog (legacy + scientific engine)
# =========================
def _get_analog_method_configs(args, workdir: Path):
    return get_analog_method_configs(args, workdir, _was_analog_module)


def stage_pbc(args, workdir: Path):
    """Optional probability-only correction from existing seasonal products."""
    return run_pbc_stage(args, workdir, VARIABLE)


def stage_telnet(args, workdir: Path):
    """Optional native probabilistic sequence model from monthly observations."""
    return run_telnet_stage(args, workdir, VARIABLE, globals())


def stage_analog(args, workdir: Path):
    check_dir = workdir / "checkpoints"
    signature = analog_stage_signature(args, workdir, VARIABLE)
    if analog_checkpoint_matches(workdir, signature) and not args.redo and not args.force:
        print("[analog] checkpoint exists → skip")
        return
    check_dir.mkdir(exist_ok=True, parents=True)
    _cca_invalidate_consolidation(workdir)
    inter_dir = workdir / "intermediate"; inter_dir.mkdir(exist_ok=True, parents=True)
    _remove_matching_files(inter_dir, ["scores_consolidated_analog.pkl",
        "analog_hindcasts.nc", "analog_forecasts.nc"], label="analog")
    observed_reference = xr.load_dataarray(workdir / "rainfall_predictand.nc")
    rainfall = analog_training_sample(observed_reference, args)
    out_dir = workdir / "forecasts"; out_dir.mkdir(exist_ok=True)
    # Native analogue probabilities do not use global best-fit family maps.
    cv_k = _residual_probability_controls(args)
    cv_k["tercile_reference"] = args.tercile_reference
    cv_k["obs_for_terciles"] = observed_reference
    cv = configured_cross_validator(args, rainfall)
    was_verify = WAS_Verification(dist_method=args.dist_method)
    configs = _get_analog_method_configs(args, workdir)
    selected = _unique_preserve_order([str(m).strip().lower() for m in args.analog_methods],
                                      label="analog methods")
    hind_ds, fcst_ds, scores_consolidated = xr.Dataset(), xr.Dataset(), {}
    raw_sources = None
    for method_name in selected:
        if method_name not in configs:
            raise ValueError(f"Unsupported analog method: {method_name}")
        cls, kwargs, dict_key, model_label = configs[method_name]
        log_progress(f"Processing Analog Method: {method_name}")
        model = cls(**kwargs)
        if raw_sources is None:
            raw_sources = preload_analog_sources(model, force=args.force)
        else:
            model._analog_raw_cache = raw_sources
        hd, hp = cv.cross_validate(model, rainfall, clim_year_start=args.clim_years[0],
                                   clim_year_end=args.clim_years[1], **cv_k)
        reference_date = pd.Timestamp(rainfall["T"].isel(T=0).values)
        target_year = int(getattr(model, "year_forecast", args.forecast_year))
        forecast_time = reference_date + pd.DateOffset(years=target_year - reference_date.year)
        _, fd, fp = model.forecast(rainfall, args.clim_years[0], args.clim_years[1], hd,
                                   forecast_time=forecast_time, **cv_k)
        evaluate_and_plot_forecasts(
            was_verify, rainfall, hd, hp, fp, model_label, dict_key,
            args.clim_years[0], args.clim_years[1], scores_consolidated,
            str(workdir / "scores"), str(out_dir), season_to_str(args.season_months),
            int(getattr(model, "year_forecast", args.forecast_year)), args.init_month, FCST_LABELS_FR,
            args.country_code, args.logo)
        hind_ds[f"hdcst_det_{dict_key}"] = _ensure_1d_time(hd, reference=rainfall)
        hind_ds[f"hdcst_prob_{dict_key}"] = _ensure_1d_time(hp, reference=rainfall)
        fcst_ds[f"fcst_det_{dict_key}"] = fd
        fcst_ds[f"fcst_prob_{dict_key}"] = fp
        gc.collect()
    hind_ds.pipe(strip_attrs).to_netcdf(inter_dir / "analog_hindcasts.nc")
    fcst_ds.pipe(strip_attrs).to_netcdf(inter_dir / "analog_forecasts.nc")
    save_scores(workdir, scores_consolidated, "scores_consolidated_analog.pkl")
    mark_done(check_dir, "analog", {"methods": selected, "engine": "fold-safe-scientific",
                                      "signature": signature})



# =========================
# Stage 5 — EOF -> KCCA (fifth independent forecasting approach)
# =========================
def _eof_kcca_model_keys(downloader, field_name):
    """Exact model keys for one physical field (no substring ambiguity)."""
    wanted = str(field_name).upper()
    keys = []
    for key in downloader.ModelsName().keys():
        if not isinstance(key, str) or "." not in key:
            continue
        suffix = key.split(".", 1)[1].upper()
        if suffix == wanted:
            keys.append(key)
    return keys


def _eof_kcca_extent(args, field_name):
    field = str(field_name).upper()
    if field == "SST":
        return args.sst_extent
    if field.startswith("UGRD") or field.startswith("VGRD"):
        return args.wind_extent
    return args.gcm_extent


def _eof_kcca_center(key):
    return str(key).split(".", 1)[0]


def _load_kcca_field(source, field_name, label):
    """Materialise one WAS_Download_Models result as an xr.DataArray.

    The downloader returns paths in current WASS2S releases, while older
    revisions/tests may return DataArray/Dataset objects.  Accept all three
    contracts explicitly and fail with the source name when the NetCDF is
    ambiguous.
    """
    if isinstance(source, xr.DataArray):
        da = source.load()
    elif isinstance(source, xr.Dataset):
        ds = source.load()
        candidates = [v for v in ds.data_vars if str(v).upper() == str(field_name).upper()]
        if len(candidates) == 1:
            da = ds[candidates[0]]
        elif len(ds.data_vars) == 1:
            da = ds[next(iter(ds.data_vars))]
        else:
            raise ValueError(f"{label}: Dataset ambigu: variables={list(ds.data_vars)}")
    elif isinstance(source, (str, os.PathLike, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"{label}: NetCDF introuvable: {path}")
        try:
            ds = xr.load_dataset(path)
            candidates = [v for v in ds.data_vars if str(v).upper() == str(field_name).upper()]
            if len(candidates) == 1:
                da = ds[candidates[0]]
            elif len(ds.data_vars) == 1:
                da = ds[next(iter(ds.data_vars))]
            else:
                # Some files were written as a single unnamed DataArray and
                # are better handled by open_dataarray.
                try:
                    da = xr.load_dataarray(path)
                except Exception as exc:
                    raise ValueError(
                        f"{label}: impossible d'identifier {field_name}; "
                        f"variables={list(ds.data_vars)}"
                    ) from exc
        except Exception:
            da = xr.load_dataarray(path)
    else:
        raise TypeError(f"{label}: source KCCA non supportee: {type(source).__name__}")

    if "T" not in da.dims:
        raise ValueError(f"{label}: champ sans dimension T; dims={da.dims}")
    da.name = da.name or str(field_name)
    return da


def _align_kcca_history(field, obs, label, strict=True):
    """Align one GCM hindcast field to the predictand years without silent restamping."""
    field = _ensure_1d_time(field)
    # Robust year selection without assuming a PandasIndex.  The downloader
    # normally returns datetime64, but numeric/cftime coordinates are accepted.
    y0 = int(obs["T"].dt.year.min().item())
    y1 = int(obs["T"].dt.year.max().item())
    if "T" in field.coords:
        try:
            years = np.asarray(field["T"].dt.year.values, dtype=int)
        except Exception:
            raw = np.asarray(field["T"].values)
            if raw.dtype.kind in "iuf":
                years = raw.astype(int)
            else:
                years = pd.to_datetime([str(v) for v in raw]).year.to_numpy()
        keep = np.flatnonzero((years >= y0) & (years <= y1))
        field = field.isel(T=keep)
        years = years[keep]
    else:
        years = np.arange(y0, y0 + field.sizes.get("T", 0), dtype=int)
    if "M" in field.dims:
        # WAS_EOF_KCCA accepts a member dimension but its scientific path is
        # defined for one field state.  Retain the ensemble mean here even
        # if the downloader was called with ensemble_mean=None.
        field = field.mean("M", skipna=True)
    if field.sizes.get("T", 0) != obs.sizes["T"]:
        raise ValueError(
            f"{label}: {field.sizes.get('T', 0)} hindcast years for "
            f"{obs.sizes['T']} predictand years."
        )
    obs_years = np.asarray(obs["T"].dt.year.values, dtype=int)
    if not np.array_equal(np.asarray(years, dtype=int), obs_years):
        message = (
            f"{label}: KCCA predictor years {list(np.asarray(years, dtype=int))} "
            f"do not match predictand years {list(obs_years)}."
        )
        if strict:
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return field.assign_coords(T=obs["T"].values)


def stage_eof_kcca(args, workdir: Path, downloader: "WAS_Download"):
    """Fifth approach: multi-field EOF reduction followed by fold-safe KCCA.

    One KCCA model is built per GCM centre from the fields available for that
    centre.  No centre is selected by skill on the verification period.  By
    default the centre predictions are collapsed into ONE equal-weight
    fifth-approach member before final consolidation, preventing this family
    from out-voting CCA/Analog/SV-ML merely because it has more centres.
    """
    if not getattr(args, "eof_kcca_enabled", True):
        print("[eof-kcca] disabled -> skip")
        return

    requested_fields = _unique_preserve_order(
        [str(v).upper() for v in args.eof_kcca_fields], label="--eof-kcca-fields"
    )
    requested_signature = {
        "fields": requested_fields,
        "kernel": str(args.eof_kcca_kernel),
        "eof_modes": int(args.eof_kcca_eof_modes),
        "kcca_modes": int(args.eof_kcca_modes),
        "pca_modes": int(args.eof_kcca_pca_modes),
        "joint": bool(args.eof_kcca_joint),
        "detrend": _cca_detrend(args),
        "detrend_schema": 2,
        "eof_standardize": bool(args.eof_kcca_eof_standardize),
        "nested_tuning": bool(args.eof_kcca_nested_tuning),
        "per_center_members": bool(args.eof_kcca_per_center_members),
        "max_centers": int(args.eof_kcca_max_centers),
        "probability_calibration": _cca_probability_calibration(args),
        "probability_inner_folds": int(getattr(args, "probability_inner_folds", None) or 5),
        "probability_inner_retune": bool(getattr(args, "probability_inner_retune", True)),
        "probability_schema": 5,
        "probability_pooling": "equal_weight_member_probabilities",
        "obs_lag_nested_tuning": bool(getattr(args, "obs_lag_nested_tuning", True)),
        "obs_lag_inner_folds": int(getattr(args, "obs_lag_inner_folds", 3)),
        "obs_lag_ridge_alpha": float(getattr(args, "obs_lag_ridge_alpha", 1.0)),
        "dist_method": str(getattr(args, "dist_method", "")),
        "climatology_years": list(getattr(args, "clim_years", ())),
        "hindcast_years": list(getattr(args, "hindcast_years", ())),
        "tercile_reference": str(getattr(args, "tercile_reference", "climatology")),
        "tercile_quantiles": "1/3,2/3",
        "scientific_models_schema": 1,
        "ngr_prcp_family": str(getattr(args, "ngr_prcp_family", "gamma")),
        "reference_version": 1,
        "outer_nb_omit": 2,
    }
    check_dir = workdir / "checkpoints"
    chk = check_dir / "chk_eof_kcca.json"
    hind_cache = workdir / "intermediate" / "eof_kcca_hindcasts.nc"
    fcst_cache = workdir / "intermediate" / "eof_kcca_forecasts.nc"
    if chk.exists() and hind_cache.exists() and fcst_cache.exists() and not args.redo:
        saved = load_json(chk)
        if all(saved.get(k) == v for k, v in requested_signature.items()):
            print("[eof-kcca] matching checkpoint exists -> skip")
            return
        print("[eof-kcca] checkpoint configuration changed -> recomputing")

    _cca_invalidate_consolidation(workdir)
    if WAS_EOF_KCCA is None:
        raise ImportError(
            "WAS_EOF_KCCA unavailable. Install the scientific was_cca_2.py "
            "and cca-zoo in the WASS2S environment."
        )

    inter_dir = workdir / "intermediate"; inter_dir.mkdir(exist_ok=True, parents=True)
    scores_dir = workdir / "scores"; scores_dir.mkdir(exist_ok=True, parents=True)
    out_dir = workdir / "forecasts"; out_dir.mkdir(exist_ok=True, parents=True)
    members_dir = workdir / "eof_kcca_members"; members_dir.mkdir(exist_ok=True, parents=True)
    _remove_matching_files(
        inter_dir,
        ["scores_consolidated_eof_kcca.pkl", "eof_kcca_hindcasts.nc", "eof_kcca_forecasts.nc"],
        label="eof-kcca",
    )

    predictand = xr.load_dataarray(workdir / "rainfall_predictand.nc")
    obs = predictand.sel(T=slice(str(args.hindcast_years[0]), str(args.hindcast_years[1]))).transpose("T", "Y", "X")
    if len(requested_fields) < 2:
        raise ValueError("EOF-KCCA requires at least two predictor fields.")

    # Download/cache each physical field independently.  Different fields
    # may have different spatial grids; WAS_EOF_KCCA fits one EOF per field.
    field_catalog = {}
    dropped = set(getattr(args, "drop_gcm", []) or [])
    for field_name in requested_fields:
        model_keys = [k for k in _eof_kcca_model_keys(downloader, field_name) if k not in dropped]
        if not model_keys:
            print(f"[eof-kcca] field {field_name} unavailable in ModelsName -> skipped")
            continue
        model_keys = requested_centres(model_keys, args)
        extent = _eof_kcca_extent(args, field_name)
        log_progress(f"EOF-KCCA downloading/reusing {field_name} ({len(model_keys)} model keys)")
        hind = downloader.WAS_Download_Models(
            str(workdir / "model_data"), model_keys, args.init_month, args.lead_time,
            args.hindcast_years[0], args.hindcast_years[1], extent, None,
            physical_ensemble_reduction(args), args.force,
        )
        fcst = downloader.WAS_Download_Models(
            str(workdir / "model_data"), model_keys, args.init_month, args.lead_time,
            args.hindcast_years[0], args.hindcast_years[1], extent, args.forecast_year,
            physical_ensemble_reduction(args), args.force,
        )
        common_keys = sorted(set(hind) & set(fcst))
        per_center = {}
        for key in common_keys:
            per_center[_eof_kcca_center(key)] = (hind[key], fcst[key])
        if per_center:
            field_catalog[field_name] = per_center

    if len(field_catalog) < int(args.eof_kcca_min_fields):
        raise RuntimeError(
            f"EOF-KCCA: only {len(field_catalog)} usable fields; "
            f"minimum={args.eof_kcca_min_fields}. Available={list(field_catalog)}"
        )

    center_to_fields = {}
    for field_name, per_center in field_catalog.items():
        for center in per_center:
            center_to_fields.setdefault(center, []).append(field_name)
    centers = sorted(
        c for c, fields in center_to_fields.items()
        if len(fields) >= int(args.eof_kcca_min_fields)
    )
    if int(args.eof_kcca_max_centers) > 0:
        # Deterministic alphabetical cap: NEVER skill-based selection.
        centers = centers[: int(args.eof_kcca_max_centers)]
    if not centers:
        raise RuntimeError("EOF-KCCA: no GCM centre has enough common predictor fields.")

    print(f"[eof-kcca] centres={centers}")
    print(f"[eof-kcca] requested_fields={requested_fields} usable_fields={list(field_catalog)}")

    best_code_da, best_shape_da, best_loc_da, best_scale_da = load_bestfit_params(workdir)
    prob_kwargs = ({"best_code_da": best_code_da, "best_shape_da": best_shape_da,
                    "best_loc_da": best_loc_da, "best_scale_da": best_scale_da}
                   if args.dist_method == "bestfit" else {})
    cv = configured_cross_validator(args, obs)
    was_verify = WAS_Verification(dist_method=args.dist_method)

    hind_det_center, hind_prob_center = {}, {}
    fcst_det_center, fcst_prob_center = {}, {}
    center_rows = []

    for center in centers:
        fields = [f for f in requested_fields if f in field_catalog and center in field_catalog[f]]
        if len(fields) < int(args.eof_kcca_min_fields):
            continue
        hist_fields, year_fields = [], []
        for field_name in fields:
            h_source, f_source = field_catalog[field_name][center]
            h = _load_kcca_field(
                h_source, field_name, f"eof-kcca/{center}/{field_name}/hindcast"
            )
            f = _load_kcca_field(
                f_source, field_name, f"eof-kcca/{center}/{field_name}/forecast"
            )
            h = _align_kcca_history(
                h, obs, f"eof-kcca/{center}/{field_name}",
                strict=getattr(args, "strict_time_alignment", True),
            )
            if "M" in f.dims:
                f = f.mean("M", skipna=True)
            f = _ensure_1d_time(f)
            if "T" not in f.dims:
                raise ValueError(f"eof-kcca/{center}/{field_name} forecast has no T dimension")
            f = f.isel(T=[-1])
            # The forecast field comes from a separately downloaded GCM file and
            # is NOT guaranteed to share the historical field's spatial grid
            # (h was regridded to the predictand via _align_kcca_history; f was
            # not). The EOF is fitted on h, and xeofs.transform() rejects any
            # field whose coordinates or valid-cell mask differ from the fit —
            # this is the "Data to be transformed has different coordinates"
            # crash. Snap f onto h's grid before it reaches the model; nearest
            # matching absorbs float-level noise, and points with no counterpart
            # become NaN, handled downstream exactly like any masked cell.
            for _ax in ("Y", "X"):
                if _ax in f.coords and _ax in h.coords:
                    f = f.reindex({_ax: h[_ax]}, method="nearest")
            hist_fields.append(h)
            year_fields.append(f)

        predictor_width = (
            int(args.eof_kcca_eof_modes)
            if bool(args.eof_kcca_joint)
            else len(fields) * int(args.eof_kcca_eof_modes)
        )
        if int(args.eof_kcca_modes) > predictor_width:
            raise ValueError(
                f"EOF-KCCA/{center}: n_modes={args.eof_kcca_modes} exceeds "
                f"reduced predictor width={predictor_width}."
            )
        kernel_pca_modes = max(
            int(args.eof_kcca_modes),
            min(int(args.eof_kcca_pca_modes), predictor_width),
        )

        model = WAS_EOF_KCCA(
            eof_n_modes=int(args.eof_kcca_eof_modes),
            joint=bool(args.eof_kcca_joint),
            eof_detrend=_cca_detrend(args),
            eof_standardize=bool(args.eof_kcca_eof_standardize),
            eof_use_coslat=True,
            n_modes=int(args.eof_kcca_modes),
            kernel=str(args.eof_kcca_kernel),
            dist_method=args.dist_method,
            standardize=True,
            cv_folds=max(2, int(args.eof_kcca_inner_folds)),
            random_state=42,
            search_method=str(args.eof_kcca_search_method),
            reg=float(args.eof_kcca_reg),
            # A second PCA layer keeps BOTH KCCA views (including the
            # predictand) below the sample-size danger zone.
            pca_explained_variance=kernel_pca_modes,
            pca_max_feature_ratio=float(args.eof_kcca_max_feature_ratio),
            tuning_objective=str(args.eof_kcca_tuning_objective),
            warn_memorisation=True,
        )

        model.probability_calibration = _cca_probability_calibration(args)
        nonlinear = str(args.eof_kcca_kernel).lower() != "linear"
        cv_control = {
            **_residual_probability_controls(args),
            "nested_tuning": bool(args.eof_kcca_nested_tuning and nonlinear),
            "inner_max_folds": int(args.eof_kcca_inner_folds),
            **prob_kwargs,
        }
        log_progress(
            f"Fifth approach EOF-KCCA: {center} fields={fields} "
            f"kernel={args.eof_kcca_kernel}"
        )
        hd, hp = cv.cross_validate(
            model, obs, hist_fields, args.clim_years[0], args.clim_years[1], **cv_control
        )

        final_best = None
        if nonlinear and args.eof_kcca_nested_tuning:
            # Final operational tuning uses the same procedure as the hindcast;
            # outer-CV skill above used nested tuning and never saw its test year.
            final_best = model.compute_hyperparameters(
                hist_fields, obs, args.clim_years[0], args.clim_years[1]
            )
        fd, fp = model.forecast(
            obs, args.clim_years[0], args.clim_years[1], hist_fields, hd,
            year_fields, best_params=final_best, **prob_kwargs,
        )
        # year_fields[0] carries the raw predictor field's own last available
        # timestep (its observation date, i.e. the initialization month) —
        # not the season being forecast. Same class of bug as CCA-GCM's
        # forecast_reference (see _execute_cca_pipeline / _season_target_time):
        # retag the OUTPUT with the season's target month so EOF-KCCA can't
        # silently disagree with the other approaches on the forecast's valid
        # date. year_fields itself is untouched — it was already consumed as
        # a genuine scientific input by model.forecast() above.
        target_T = _season_target_time(args)
        if fd.sizes.get("T") == 1:
            fd = fd.assign_coords(T=("T", np.array([target_T], dtype="datetime64[ns]")))
        if fp.sizes.get("T") == 1:
            fp = fp.assign_coords(T=("T", np.array([target_T], dtype="datetime64[ns]")))
        hd = _ensure_1d_time(hd, reference=obs, source=f"eof-kcca/{center}/hind_det")
        hp = _ensure_1d_time(hp, reference=obs, source=f"eof-kcca/{center}/hind_prob")
        fd = _ensure_1d_time(fd, source=f"eof-kcca/{center}/fcst_det")
        fp = _ensure_1d_time(fp, reference=fd, source=f"eof-kcca/{center}/fcst_prob")

        member_id = f"fifth.approach_EOF-KCCA_{center}"
        hind_det_center[member_id] = hd
        hind_prob_center[member_id] = hp
        fcst_det_center[member_id] = fd
        fcst_prob_center[member_id] = fp

        # Cheap centre diagnostics; do not use these scores to select centres.
        try:
            pear = was_verify.compute_deterministic_score(
                was_verify.get_scores_metadata()["Pearson"][5], obs, hd).compute()
            groc = was_verify.compute_probabilistic_score(
                was_verify.get_scores_metadata()["GROC"][5], obs, hp,
                args.clim_years[0], args.clim_years[1], **probability_score_kwargs(hp)).compute()
            center_rows.append({
                "center": center, "fields": ",".join(fields),
                "Pearson_spatial_mean": _weighted_spatial_mean(pear),
                "GROC_spatial_mean": _weighted_spatial_mean(groc),
            })
        except Exception as exc:
            center_rows.append({"center": center, "fields": ",".join(fields), "error": repr(exc)})

    if not hind_det_center:
        raise RuntimeError("EOF-KCCA produced no centre prediction.")
    pd.DataFrame(center_rows).to_csv(scores_dir / "eof_kcca_center_diagnostics.csv", index=False)

    # Archive individual centre products outside intermediate/ by default.
    xr.Dataset({f"hdcst_det_{k}": v for k, v in hind_det_center.items()} |
               {f"hdcst_prob_{k}": v for k, v in hind_prob_center.items()}).pipe(strip_attrs).to_netcdf(
        members_dir / "eof_kcca_members_hindcasts.nc")
    xr.Dataset({f"fcst_det_{k}": v for k, v in fcst_det_center.items()} |
               {f"fcst_prob_{k}": v for k, v in fcst_prob_center.items()}).pipe(strip_attrs).to_netcdf(
        members_dir / "eof_kcca_members_forecasts.nc")

    stage_scores = {}
    if getattr(args, "eof_kcca_per_center_members", False):
        # Diagnostic mode: every centre becomes an MME member.
        hind_vars, fcst_vars = {}, {}
        for member_id in hind_det_center:
            evaluate_and_plot_forecasts(
                was_verify, obs, hind_det_center[member_id], hind_prob_center[member_id],
                fcst_prob_center[member_id], member_id, member_id,
                args.clim_years[0], args.clim_years[1], stage_scores,
                str(scores_dir), str(out_dir), season_to_str(args.season_months),
                args.forecast_year, args.init_month, FCST_LABELS_FR,
                args.country_code, args.logo,
            )
            hind_vars[f"hdcst_det_{member_id}"] = hind_det_center[member_id]
            hind_vars[f"hdcst_prob_{member_id}"] = hind_prob_center[member_id]
            fcst_vars[f"fcst_det_{member_id}"] = fcst_det_center[member_id]
            fcst_vars[f"fcst_prob_{member_id}"] = fcst_prob_center[member_id]
    else:
        # Default: ONE equal-weight family member. Its deterministic product
        # is the arithmetic mean; its probabilities are a mixture of calibrated
        # centre distributions, preserving the chosen nested/outer procedure.
        center_ids = list(hind_det_center)
        hd_cons = xr.concat([hind_det_center[k] for k in center_ids], dim="M").mean("M", skipna=True)
        fd_cons = xr.concat([fcst_det_center[k] for k in center_ids], dim="M").mean("M", skipna=True)
        hd_cons = _ensure_1d_time(hd_cons, reference=obs)
        hp_cons = mean_calibrated_probabilities(
            [hind_prob_center[k] for k in center_ids],
            calibration=_cca_probability_calibration(args),
        )
        fp_cons = mean_calibrated_probabilities(
            [fcst_prob_center[k] for k in center_ids],
            calibration=_cca_probability_calibration(args), forecast=True,
        )
        hp_cons = _ensure_1d_time(hp_cons, reference=obs)
        consensus_id = "fifth.approach_EOF-KCCA"
        evaluate_and_plot_forecasts(
            was_verify, obs, hd_cons, hp_cons, fp_cons,
            "EOF-KCCA Fifth Approach", consensus_id,
            args.clim_years[0], args.clim_years[1], stage_scores,
            str(scores_dir), str(out_dir), season_to_str(args.season_months),
            args.forecast_year, args.init_month, FCST_LABELS_FR,
            args.country_code, args.logo,
        )
        hind_vars = {f"hdcst_det_{consensus_id}": hd_cons,
                     f"hdcst_prob_{consensus_id}": hp_cons}
        fcst_vars = {f"fcst_det_{consensus_id}": fd_cons,
                     f"fcst_prob_{consensus_id}": fp_cons}

    xr.Dataset(hind_vars).pipe(strip_attrs).to_netcdf(inter_dir / "eof_kcca_hindcasts.nc")
    xr.Dataset(fcst_vars).pipe(strip_attrs).to_netcdf(inter_dir / "eof_kcca_forecasts.nc")
    save_scores(workdir, stage_scores, "scores_consolidated_eof_kcca.pkl")
    save_json(chk, {**requested_signature, "enabled": True, "centers": centers})
    print("[eof-kcca] done")


# =========================
# Stage 7 — Consolidation 
# =========================
def stage_consolidate(args, workdir: Path):
    """Evaluate fast and optional advanced MME consolidators on one common OOS basis."""
    check_dir = workdir / "checkpoints"
    procedure = _consolidation_procedure_signature(args)
    if done(check_dir, "consolidate") and not args.redo:
        saved = load_json(check_dir / CHECKPOINTS["consolidate"])
        if saved.get("probability_procedure") == procedure:
            print("[consolidate] checkpoint exists → skip")
            return
        log_progress("Consolidation probability procedure changed → recomputing")
    (check_dir / CHECKPOINTS["consolidate"]).unlink(missing_ok=True)

    rainfall = xr.load_dataarray(workdir / "rainfall_predictand.nc")
    best_code_da, best_shape_da, best_loc_da, best_scale_da = load_bestfit_params(workdir)
    cv_k = ({"best_code_da": best_code_da, "best_shape_da": best_shape_da,
             "best_loc_da": best_loc_da, "best_scale_da": best_scale_da}
            if args.dist_method == "bestfit" else {})
    models_prefix = ["SV-ML-CMME", "PRCP", "TEMP", "SST", "OBS-LAG", "fourth", "ANALOG", "fifth", "EOF-KCCA"]
    inter_dir = workdir / "intermediate"
    (all_model_hdcst_dict, all_model_fcst_dict, all_model_hdcst_prob_dict,
     all_model_fcst_prob_dict, product_pairs, product_provenance) = \
        collect_intermediate_products(inter_dir, args.hindcast_years)
    all_scores, score_files, score_provenance = load_score_catalog(inter_dir)
    groc_models = set(all_scores.get("GROC", {}))
    deterministic_models = sorted(set(all_model_hdcst_dict) & set(all_model_fcst_dict) & groc_models)
    probabilistic_models = sorted(set(all_model_hdcst_prob_dict) & set(all_model_fcst_prob_dict) & groc_models)
    if not deterministic_models:
        raise RuntimeError("Aucun modèle déterministe complet avec score GROC.")
    if not probabilistic_models:
        raise RuntimeError("Aucun modèle probabiliste complet avec score GROC.")

    print(f"[consolidate] deterministic={len(deterministic_models)} probabilistic={len(probabilistic_models)}")
    print(f"[consolidate] score files={[p.name for p in score_files]}")
    try:
        comp = error_complementarity(rainfall, {m: all_model_hdcst_dict[m] for m in deterministic_models})
        save_json(workdir / "scores/error_complementarity.json", comp)
    except Exception as exc:
        print(f"[complementarity] skipped: {exc}")

    obs = rainfall.sel(T=slice(str(args.hindcast_years[0]), str(args.hindcast_years[1]))).transpose("T", "Y", "X")
    all_model_hdcst, all_model_fcst, obs, best_score_det = process_datasets_for_mme(
        obs, hdcsted={k: all_model_hdcst_dict[k] for k in deterministic_models},
        fcsted={k: all_model_fcst_dict[k] for k in deterministic_models},
        gcm=False, Prob=False, best_models=deterministic_models,
        scores=all_scores, model=False, score_metric="GROC")
    all_model_hdcst_prob, all_model_fcst_prob, _, best_score_prob = process_datasets_for_mme(
        obs, hdcsted={k: all_model_hdcst_prob_dict[k] for k in probabilistic_models},
        fcsted={k: all_model_fcst_prob_dict[k] for k in probabilistic_models},
        gcm=False, Prob=True, best_models=probabilistic_models,
        scores=all_scores, model=False, score_metric="GROC")
    all_model_hdcst = all_model_hdcst.transpose("T", "M", "Y", "X").assign_coords(T=obs["T"])
    all_model_fcst = all_model_fcst.transpose("T", "M", "Y", "X")
    all_model_hdcst_prob = all_model_hdcst_prob.transpose("probability", "T", "M", "Y", "X").assign_coords(T=obs["T"])
    all_model_fcst_prob = all_model_fcst_prob.transpose("probability", "T", "M", "Y", "X")

    # ------------------------------------------------------------------
    # Central PRCP dry mask: ONE definition applied to every consolidator.
    # It is derived only from observed climatology, never from a forecast.
    # The same mask is applied to deterministic/probabilistic inputs, outputs,
    # verification scores and maps so dry cells cannot reappear downstream.
    # ------------------------------------------------------------------
    if args.apply_drymask:
        clim_obs = obs.sel(T=slice(str(args.clim_years[0]), str(args.clim_years[1])))
        valid_count = clim_obs.count("T")
        clim_mean = clim_obs.mean("T", skipna=True)
        drymask = xr.where(
            (valid_count >= int(args.dry_min_valid_years))
            & np.isfinite(clim_mean)
            & (clim_mean >= float(args.dry_threshold)),
            1.0, np.nan,
        ).rename("drymask")
    else:
        drymask = xr.where(obs.count("T") >= int(args.dry_min_valid_years), 1.0, np.nan).rename("drymask")
    drymask.attrs.update({
        "description": "PRCP consolidation mask from observed climatology",
        "dry_threshold_mm": float(args.dry_threshold),
        "climatology": f"{args.clim_years[0]}-{args.clim_years[1]}",
        "apply_drymask": int(bool(args.apply_drymask)),
    })
    drymask.to_netcdf(workdir / "scores" / "consolidation_drymask.nc")

    # process_datasets_for_mme returns obs carrying a dummy M=[0] dimension.
    # drymask is derived from obs, so it inherits that M — and multiplying the
    # 82-member cube by a mask carrying M=[0] aligns on M by intersection,
    # collapsing the cube to an empty member axis (the "M=0" that made every
    # member-wise consolidator fail while equal/pcr survived). Strip any M
    # from the mask so it broadcasts purely over (Y, X). This is PRCP-only,
    # which is exactly why TEMP was unaffected.
    for _d in ("M", "probability"):
        if _d in drymask.dims:
            drymask = drymask.isel({_d: 0}, drop=True)
        elif _d in drymask.coords:
            drymask = drymask.drop_vars(_d)

    obs = obs * drymask
    all_model_hdcst = all_model_hdcst * drymask
    all_model_fcst = all_model_fcst * drymask
    all_model_hdcst_prob = all_model_hdcst_prob * drymask
    all_model_fcst_prob = all_model_fcst_prob * drymask

    # Diagnostic: several consolidators (weighted, probweighted, mva, min2009)
    # re-align their inputs with join="inner" internally. If the M-stacked
    # hindcast cube and obs disagree on the T axis, that intersection is empty
    # and each method fails with a different opaque symptom ("must supply at
    # least one object to concatenate", "size 0", "index 0 out of bounds",
    # "must be a permuted list of ('T','M','Y','X')"). Surface the axes ONCE
    # here so the cause is visible instead of inferred method by method.
    def _axis_summary(da, label):
        try:
            ny = int(da.sizes.get("T", 0)); nm = int(da.sizes.get("M", 0))
            yrs = np.asarray(da["T"].dt.year.values) if "T" in da.coords else np.array([])
            span = f"{yrs.min()}-{yrs.max()}" if yrs.size else "—"
            return f"{label}: T={ny} ({span}) M={nm}"
        except Exception as exc:
            return f"{label}: <axis read failed: {exc}>"
    print("[consolidate] " + " | ".join([
        _axis_summary(obs, "obs"),
        _axis_summary(all_model_hdcst, "hdcst_det"),
        _axis_summary(all_model_hdcst_prob, "hdcst_prob"),
    ]))
    _obs_years = set(np.asarray(obs["T"].dt.year.values)) if "T" in obs.coords else set()
    _hd_years = set(np.asarray(all_model_hdcst["T"].dt.year.values)) if "T" in all_model_hdcst.coords else set()
    if _obs_years and _hd_years and not (_obs_years & _hd_years):
        print("[consolidate] ATTENTION: obs et hdcst_det n'ont AUCUNE année T en "
              "commun — les consolidateurs à realignement interne (weighted, "
              "probweighted, mva, min2009) vont tous échouer sur un axe vide. "
              "C'est un désalignement de dates en amont, pas un défaut par méthode.")

    profiles = {
        "fast": ["equal", "weighted", "weighted_continuous",
                 "probweighted", "probweighted_continuous", "probweighted_rps",
                 "min2009", "mva", "ngr_nongaussian", "pcr", "ridge",
                 "elasticnet", "expert", "elr"],
        "standard": ["equal", "weighted", "weighted_continuous",
                     "probweighted", "probweighted_continuous", "probweighted_rps",
                     "min2009", "mva", "ngr_nongaussian", "pcr", "ridge",
                     "elasticnet", "expert", "elr", "histgb", "extratrees", "logistic"],
        # Explicitly expensive methods are never part of fast/standard.
        "extended": ["equal", "weighted", "weighted_continuous",
                     "probweighted", "probweighted_continuous", "probweighted_rps",
                     "min2009", "mva", "ngr_gaussian", "ngr_nongaussian", "pcr",
                     "ridge", "elasticnet", "histgb", "extratrees", "svr",
                     "expert", "elr", "logistic", "fastbma", "gaussian_process", "fullbma"],
    }
    selected = (_unique_preserve_order([m.lower() for m in args.consolidation_methods],
                                       label="--consolidation-methods")
                if args.consolidation_methods else profiles[args.consolidation_profile])
    print(f"[consolidate] profile={args.consolidation_profile} methods={selected}")

    out_dir = workdir / "forecasts"; out_dir.mkdir(exist_ok=True)
    scores_dir = workdir / "scores"; scores_dir.mkdir(exist_ok=True)
    was_verify = WAS_Verification(dist_method=args.dist_method)
    cv_cons = configured_cross_validator(args, obs)
    consensus_scores, summary_rows, timings = {}, [], {}
    season_str = season_to_str(args.season_months, lang="en")
    season_str_en = season_to_str(args.season_months, lang="en", mode="total")
    season_str_fr = season_to_str(args.season_months, lang="fr", mode="total")
    mois_fr = {
        1: "janvier", 2: "février", 3: "mars", 4: "avril",
        5: "mai", 6: "juin", 7: "juillet", 8: "août",
        9: "septembre", 10: "octobre", 11: "novembre", 12: "décembre",
    }

    def _thin(da, stride):
        stride = max(1, int(stride))
        if stride == 1: return da
        indexers = {d: slice(None, None, stride) for d in ("Y", "X") if d in da.dims}
        return da.isel(**indexers)

    def _forecast_map(da):
        out = da
        if "T" in out.dims and out.sizes["T"] == 1:
            out = out.isel(T=0, drop=True)
        return out.squeeze(drop=True).sortby("Y") if "Y" in out.dims else out.squeeze(drop=True)

    def write_and_plot(name, h_det=None, h_prob=None, f_det=None, f_prob=None):
        # Enforce the SAME observed-climatology dry mask for every method.
        # Some estimators can numerically fill NaNs, so masking only the inputs
        # is not enough; re-apply it to every output before scoring/saving.
        if h_det is not None: h_det = h_det * drymask
        if h_prob is not None: h_prob = h_prob * drymask
        if f_det is not None: f_det = f_det * drymask
        if f_prob is not None: f_prob = f_prob * drymask
        skill_mask = None
        if h_det is not None:
            h_det = _ensure_1d_time(h_det, reference=obs)
            for metric in ("Pearson", "MAE"):
                r = was_verify.compute_deterministic_score(
                    was_verify.get_scores_metadata()[metric][5], obs, h_det).compute()
                r = r * drymask
                r.to_dataset(name=name).to_netcdf(scores_dir / f"{name}_{metric}.nc")
                _register_score_value(consensus_scores, metric, name, r,
                                      source_name="consolidation", duplicate_policy="raise")
                summary_rows.append({"method": name, "metric": metric,
                    "spatial_mean": _weighted_spatial_mean(r)})
                if args.consolidation_plot_level in ("scores", "all"):
                    try: was_verify.plot_model_score(r, metric, scores_dir, name); plt.close("all")
                    except Exception: pass
        if h_prob is not None:
            h_prob = _ensure_1d_time(h_prob, reference=obs)
            # Retain actual historical probabilities for optional PBC. PBC
            # outputs live outside intermediate/ and cannot become duplicate
            # consolidation members on a subsequent run.
            pbc_hind = strip_attrs(h_prob.copy())
            pbc_hind.attrs["pbc_source_model"] = name
            pbc_hind.to_netcdf(scores_dir / f"Hindcast_Prob_{name}_{season_str}_{args.forecast_year}.nc")
            for metric in ("GROC", "RPSS"):
                r = was_verify.compute_probabilistic_score(
                    was_verify.get_scores_metadata()[metric][5], obs, h_prob,
                    args.clim_years[0], args.clim_years[1], **probability_score_kwargs(h_prob)).compute()
                r = r * drymask
                r.to_dataset(name=name).to_netcdf(scores_dir / f"{name}_{metric}.nc")
                _register_score_value(consensus_scores, metric, name, r,
                                      source_name="consolidation", duplicate_policy="raise")
                summary_rows.append({"method": name, "metric": metric,
                    "spatial_mean": _weighted_spatial_mean(r)})
                if metric == "GROC":
                    skill_mask = xr.where(r > 0.5, 1.0, np.nan)
                if args.consolidation_plot_level in ("scores", "all"):
                    try: was_verify.plot_model_score(r, metric, scores_dir, name); plt.close("all")
                    except Exception: pass
        if f_det is not None:
            strip_attrs(f_det).to_netcdf(out_dir / f"Forecast_Det_{name}_{season_str}_{args.forecast_year}.nc")
        if f_prob is not None:
            strip_attrs(f_prob).to_netcdf(out_dir / f"Forecast_Prob_{name}_{season_str}_{args.forecast_year}.nc")
            if args.consolidation_plot_level in ("forecast", "all"):
                try:
                    fmap = _forecast_map(f_prob)
                    plot_prob_forecasts_(str(out_dir), fmap,
                        f"Consolidated {name} {season_str}-{args.forecast_year} IC_{calendar.month_name[int(args.init_month)]}",
                        reverse_cmap=False, hspace=-0.2, labels=FCST_LABELS_EN,
                        logo=args.logo, logo_size=("12%", "12%"), logo_position="upper left")
                    # Every consolidated method gets the full bulletin map — this
                    # allowlist used to skip Weighted/ProbWeighted/ProbWeightedContinuous,
                    # unlike every prior template, which plots all methods unconditionally.
                    plot_prob_forecasts(str(out_dir), fmap,
                            f"Consolidated {name} {season_str}-{args.forecast_year} IC_{calendar.month_name[int(args.init_month)]}",
                            title=f"Seasonal Forecast for Cumulative Rainfall\nValid for {season_str_en} {args.forecast_year}, Issued {dt.datetime.now().strftime('%B %d, %Y')}",
                            country_code=args.country_code, source="gadm", admin_level=1,
                            stations_df=None, reverse_cmap=False, logo=args.logo,
                            logo_size=("27%", "13%"), logo_position="lower left",
                            res=0.01, hspace=-0.2, labels=FCST_LABELS_EN, out="png")
                    # French bulletin map, dropped in an earlier refactor — restored
                    # here on the same fmap/skill mask, matching every prior template
                    # (English and French were always produced side by side).
                    plot_prob_forecasts(str(out_dir), fmap,
                            f"Consolidée {name} {season_str}-{args.forecast_year} IC_{calendar.month_name[int(args.init_month)]}",
                            title=f"Prévision Saisonnière des Cumuls de pluie\nValable pour {season_str_fr} {args.forecast_year}, Élaborée le {dt.datetime.now().day:02d} {mois_fr[dt.datetime.now().month]} {dt.datetime.now().year}",
                            country_code=args.country_code, source="gadm", admin_level=1,
                            stations_df=None, reverse_cmap=False, logo=args.logo,
                            logo_size=("27%", "13%"), logo_position="lower left",
                            res=0.01, hspace=-0.2, labels=FCST_LABELS_FR, out="png")
                    plt.close("all")
                except Exception as exc:
                    print(f"[consolidate/{name}] plotting skipped: {exc}")

    def _prob_dict(da, time_index=None, stride=1):
        x = da if time_index is None else da.isel(T=time_index)
        x = _thin(x, stride)
        return {str(m): x.sel(M=m, drop=True) for m in x["M"].values}

    def _run_ml(cls_name, label):
        cls = _require_was_class(cls_name)
        model = cls(n_clusters=args.consolidation_clusters,
                    cv_folds=min(4, args.consolidation_inner_folds),
                    leave_one_year_out=False, dist_method=args.dist_method,
                    random_state=42, n_jobs=args.consolidation_n_jobs,
                    min_train_years=10)
        hd, hp = cv_cons.cross_validate(
            model, obs, all_model_hdcst, args.clim_years[0], args.clim_years[1],
            nested_tuning=args.consolidation_nested_tuning,
            strict_oos_probability=True, inner_max_folds=args.consolidation_inner_folds,
            probability_inner_folds=int(getattr(args, "probability_inner_folds", None) or args.consolidation_prob_folds),
            probability_calibration=probability_calibration_mode(args),
            probability_inner_retune=bool(getattr(args, "probability_inner_retune", True)), **cv_k)
        if args.consolidation_nested_tuning:
            bp, cl = model.compute_hyperparameters(all_model_hdcst, obs,
                                                   args.clim_years[0], args.clim_years[1])
        else:
            from wass2s.was_mme import _regional_params_from
            fixed_params = _regional_params_from(model, {}, obs)
            bp, cl = fixed_params["best_params"], fixed_params["cluster_da"]
        fd, fp = model.forecast(obs, args.clim_years[0], args.clim_years[1],
                                all_model_hdcst, hd, all_model_fcst.isel(T=[-1]),
                                best_params=bp, cluster_da=cl, **cv_k)
        write_and_plot(label, hd, hp, fd, fp)

    def _fold_local_family_map(reference):
        """Best-fit family codes {norm:1, lognorm:2, gamma:4} fitted on ONE outer
        training climatology (T,Y,X), for FullBMA strict CV; shares the
        content-keyed best-fit cache of the other bestfit consolidators."""
        families = {"norm": 1, "lognorm": 2, "gamma": 4}
        try:
            from wass2s.was_cross_validate import _cached_best_distribution
            return _cached_best_distribution(reference, families, "grid")[0]
        except ImportError:
            return WAS_TransformData(reference, distribution_map=families,
                                     n_clusters=args.n_clusters).fit_best_distribution(mode="grid")[0]

    def _execute(method):
        if method == "equal":
            m = WAS_mme_Weighted(equal_weighted=True, dist_method=args.dist_method, metric="GROC", threshold=0.51)
            hd, fd = m.compute(obs, all_model_hdcst, all_model_fcst.isel(T=[-1]), best_score_det, complete=True)
            hd = _ensure_1d_time(hd, reference=obs)
            hp = m.compute_prob(obs, args.clim_years[0], args.clim_years[1], hd, **cv_k)
            _, fp = m.forecast(obs, args.clim_years[0], args.clim_years[1], hd, fd, **cv_k)
            return write_and_plot("EqualWeighted", hd, hp, fd, fp)

        if method == "weighted":
            m = WAS_mme_Weighted(equal_weighted=False, dist_method=args.dist_method, metric="GROC", threshold=0.51)
            hd, fd = m.compute(obs, all_model_hdcst, all_model_fcst.isel(T=[-1]), best_score_det, complete=True)
            hd = _ensure_1d_time(hd, reference=obs)
            hp = m.compute_prob(obs, args.clim_years[0], args.clim_years[1], hd, **cv_k)
            _, fp = m.forecast(obs, args.clim_years[0], args.clim_years[1], hd, fd, **cv_k)
            return write_and_plot("WeightedLegacy", hd, hp, fd, fp)

        if method == "weighted_continuous":
            cls = _require_was_class("WAS_mme_WeightedContinuous")
            m = cls(dist_method=args.dist_method, metric="GROC", threshold=0.5,
                    shrinkage=args.consolidation_weight_shrinkage)
            hd, fd = m.compute(obs, all_model_hdcst, all_model_fcst.isel(T=[-1]), best_score_det, complete=True)
            hd = _ensure_1d_time(hd, reference=obs)
            hp = m.compute_prob(obs, args.clim_years[0], args.clim_years[1], hd, **cv_k)
            _, fp = m.forecast(obs, args.clim_years[0], args.clim_years[1], hd, fd, **cv_k)
            return write_and_plot("WeightedContinuous", hd, hp, fd, fp)

        if method in {"probweighted", "probweighted_continuous"}:
            if method == "probweighted":
                m = WAS_ProbWeighted(metric="GROC", threshold=0.5)
                label = "ProbWeightedLegacy"
            else:
                cls = _require_was_class("WAS_ProbWeightedContinuous")
                m = cls(metric="GROC", threshold=0.5, shrinkage=args.consolidation_weight_shrinkage)
                label = "ProbWeightedContinuous"
            hp, fp = m.compute(obs, all_model_hdcst_prob, all_model_fcst_prob.isel(T=[-1]),
                               best_score_prob, threshold=0.51, complete=True)
            return write_and_plot(label, None, hp, None, fp)

        if method == "probweighted_rps":
            cls = _require_was_class("WAS_ProbWeightedRPS")
            stride = args.consolidation_rps_stride
            fold_probs = []
            fixed_classes = (was_verify.compute_class(obs, *args.clim_years)
                             if args.tercile_reference == "climatology" else None)
            for tr, te in cv_cons.custom_cv.split(obs["T"], cv_cons.nb_omit):
                mod = cls(spatial_pooling="global", min_train_years=8)
                mod.fit(_thin(obs.isel(T=tr), stride),
                        _thin(all_model_hdcst_prob.isel(T=tr), stride),
                        args.clim_years[0], args.clim_years[1],
                        obs_class=(None if fixed_classes is None else
                                   _thin(fixed_classes.isel(T=tr), stride)))
                pp = mod.transform(all_model_hdcst_prob.isel(T=te))
                fold_probs.append(pp.assign_coords(T=obs["T"].isel(T=te)))
            hp = xr.concat(fold_probs, dim="T").sortby("T").reindex(T=obs["T"])
            final = cls(spatial_pooling="global", min_train_years=8)
            final.fit(_thin(obs, stride), _thin(all_model_hdcst_prob, stride),
                      args.clim_years[0], args.clim_years[1])
            fp = final.transform(all_model_fcst_prob.isel(T=[-1]))
            try:
                xr.Dataset({"weight": final.weights_, "lambda": xr.DataArray(final.lambda_)}).to_netcdf(
                    scores_dir / "ProbWeightedRPS_weights.nc")
            except Exception: pass
            return write_and_plot("ProbWeightedRPS", None, hp, None, fp)

        if method == "min2009":
            distribution = "lognormal" if VARIABLE == "PRCP" else "gaussian"
            pmme = WAS_Min2009_ProbWeighted(distribution=distribution, cv_method="leave_one_out",
                                            n_samples_for_chisq="effective_sample_size")
            masks, forecasts, hindcasts, ensemble_sizes = {}, {}, {}, {}
            member_names = [str(v) for v in all_model_fcst["M"].values]
            for i, prefix in enumerate(models_prefix, start=1):
                ids = np.flatnonzero([prefix.lower() in n.lower() for n in member_names])
                if ids.size == 0: continue
                gid = f"model{i}"
                mask = np.zeros(len(member_names), dtype=bool); mask[ids] = True
                masks[gid] = xr.DataArray(mask, dims="M", coords={"M": all_model_fcst["M"]})
                hindcasts[gid] = all_model_hdcst.isel(M=ids).assign_coords(T=obs["T"])
                forecasts[gid] = all_model_fcst.isel(M=ids, T=[-1])
                ensemble_sizes[gid] = int(ids.size)
            if not masks:
                raise RuntimeError("Min2009: aucun groupe de membres non vide.")
            climatology = obs.sel(T=slice(str(args.clim_years[0]), str(args.clim_years[1]))).mean("T", skipna=True)
            hp, _ = cv_cons.cross_validate(pmme, obs, all_model_hdcst,
                args.clim_years[0], args.clim_years[1], ensemble_sizes=ensemble_sizes,
                masks=masks, climatology=climatology)
            forecast_q, _ = climatological_terciles(obs, *args.clim_years)
            probs = pmme.compute_pmme_probabilities(
                forecasts, hindcasts, climatology, ensemble_sizes,
                clim_terciles=(forecast_q.isel(quantile=0, drop=True),
                               forecast_q.isel(quantile=1, drop=True)))
            fp = xr.concat([probs["PB"], probs["PN"], probs["PA"]],
                dim=xr.DataArray(["PB", "PN", "PA"], dims="probability", name="probability"))
            fp = fp * drymask
            return write_and_plot("Min2009", None, hp, None, fp)

        if method == "mva":
            m = WAS_mme_MVA(dist_method=args.dist_method)
            hc = m.fit_transform_loocv(all_model_hdcst, obs)
            hp = m.compute_prob(obs, args.clim_years[0], args.clim_years[1], hc, **cv_k)
            fd, fp = m.forecast(obs, args.clim_years[0], args.clim_years[1],
                                all_model_hdcst, hc, all_model_fcst.isel(T=[-1]), **cv_k)
            return write_and_plot("MVA", hc.mean("M", skipna=True), hp, fd, fp)

        if method in {"ngr_gaussian", "ngr_nongaussian"}:
            if method == "ngr_gaussian":
                m = WAS_mme_NGR_Gaussian(apply_to="pos" if VARIABLE == "PRCP" else "all",
                                         alpha=0.10, min_train_years=10)
                label = "NGR_Gaussian"
            else:
                m = WAS_mme_NGR_NonGaussian(
                    family=(args.ngr_prcp_family if VARIABLE == "PRCP" else "transformed_gaussian"),
                    apply_to="pos" if VARIABLE == "PRCP" else "all",
                    alpha=0.10, min_train_years=10)
                label = "NGR_NonGaussian"
            # Event thresholds follow --tercile-reference; parameter fitting remains within each fold.
            hd, hp = cv_cons.cross_validate(m, obs, all_model_hdcst,
                args.clim_years[0], args.clim_years[1], quantiles=[0.1, 0.5, 0.9],
                return_synthetic_ensemble=False,
                member_dim="M", time_dim="T", lat_dim="Y", lon_dim="X")
            ds = m.compute_model(all_model_hdcst, obs, all_model_fcst.isel(T=[-1]),
                obs_for_terciles=climatology_sample(obs, *args.clim_years), quantiles=[0.1, 0.5, 0.9], clim_terciles=True,
                return_synthetic_ensemble=False, member_dim="M", time_dim="T",
                lat_dim="Y", lon_dim="X")
            return write_and_plot(label, hd, hp, ds.get("calibrated_mean"), ds["tercile_probability"])

        if method == "pcr":
            m = WAS_mme_PCR(n_modes=args.consolidation_pcr_modes, use_coslat=True,
                            combine="mean", detrend=False, dist_method=args.dist_method,
                            positive_support=True)
            hd, hp = cv_cons.cross_validate(m, obs, all_model_hdcst,
                args.clim_years[0], args.clim_years[1], nested_tuning=args.consolidation_nested_tuning,
                strict_oos_probability=True, inner_max_folds=args.consolidation_inner_folds,
                probability_inner_folds=int(getattr(args, "probability_inner_folds", None) or args.consolidation_prob_folds),
                probability_calibration=probability_calibration_mode(args),
                probability_inner_retune=bool(getattr(args, "probability_inner_retune", True)), **cv_k)
            m.finalize_truncation(
                all_model_hdcst, obs, *args.clim_years,
                n_modes_grid=range(1, args.consolidation_pcr_modes + 1),
                nb_omit=cv_cons.nb_omit,
                inner_max_folds=int(args.consolidation_inner_folds),
                enabled=bool(args.consolidation_nested_tuning))
            fd, fp = m.forecast(obs, args.clim_years[0], args.clim_years[1],
                                all_model_hdcst, hd, all_model_fcst.isel(T=[-1]), **cv_k)
            return write_and_plot("PCR", hd, hp, fd, fp)

        if method == "ridge": return _run_ml("WAS_mme_Ridge", "RidgeMME")
        if method == "elasticnet": return _run_ml("WAS_mme_ElasticNet", "ElasticNetMME")
        if method == "svr": return _run_ml("WAS_mme_SVR", "SVR_MME")
        if method == "extratrees": return _run_ml("WAS_mme_ExtraTrees", "ExtraTreesMME")
        if method == "histgb": return _run_ml("WAS_mme_HistGradientBoosting", "HistGB_MME")

        if method == "expert":
            cls = WAS_mme_ExpertKnowledge
            stride = args.consolidation_rps_stride
            folds = []
            for tr, te in cv_cons.custom_cv.split(obs["T"], cv_cons.nb_omit):
                obs_train = obs.isel(T=tr)
                obs_class = (was_verify.compute_class(obs, *args.clim_years).isel(T=tr)
                             if args.tercile_reference == "climatology" else
                             was_verify.compute_class(obs_train, *args.clim_years))
                mod = cls(spatial_pooling="global", shrinkage="cv", min_train_years=8)
                mod.fit(_prob_dict(all_model_hdcst_prob, tr, stride), _thin(obs_class, stride))
                pp = mod.predict(_prob_dict(all_model_hdcst_prob, te, 1))
                folds.append(pp.assign_coords(T=obs["T"].isel(T=te)))
            hp = xr.concat(folds, dim="T").sortby("T").reindex(T=obs["T"])
            obs_class = was_verify.compute_class(obs, args.clim_years[0], args.clim_years[1])
            final = cls(spatial_pooling="global", shrinkage="cv", min_train_years=8)
            final.fit(_prob_dict(all_model_hdcst_prob, None, stride), _thin(obs_class, stride))
            fp = final.predict(_prob_dict(all_model_fcst_prob.isel(T=[-1]), None, 1))
            try: final.weights_.to_netcdf(scores_dir / "ExpertKnowledge_weights.nc")
            except Exception: pass
            return write_and_plot("ExpertKnowledge", None, hp, None, fp)

        if method == "elr":
            transform = "cbrt" if VARIABLE == "PRCP" else "identity"
            m = WAS_mme_ELR(predictors="mean", threshold_transform=transform, l2=1e-3, min_obs=8)
            _, hp = cv_cons.cross_validate(m, obs, all_model_hdcst,
                args.clim_years[0], args.clim_years[1],
                nested_tuning=bool(args.consolidation_nested_tuning), tuning_score="rps",
                inner_max_folds=int(args.consolidation_inner_folds))
            if args.consolidation_nested_tuning:
                m.compute_hyperparameters(all_model_hdcst, obs, *args.clim_years,
                                          score="rps",
                                          cv_folds=int(args.consolidation_inner_folds), nb_omit=cv_cons.nb_omit)
            fp = m.forecast(obs, args.clim_years[0], args.clim_years[1],
                            all_model_hdcst, all_model_fcst.isel(T=[-1]), clim_terciles=None)
            return write_and_plot("ELR", None, hp, None, fp)

        if method == "logistic":
            m = WAS_mme_logistic(optimization_method="bayesian", C_range=(0.1, 10.0),
                solver_options=["lbfgs"], random_state=42, cv_folds=3, n_clusters=3,
                n_iter_search=12, n_trials=20, timeout=7200,
                n_jobs=args.consolidation_n_jobs, optuna_n_jobs=min(4, args.consolidation_n_jobs),
                cv_n_jobs=1)
            hd, hp = cv_cons.cross_validate(m, obs, all_model_hdcst,
                args.clim_years[0], args.clim_years[1], nested_tuning=True)
            bp, cl = m.compute_hyperparameters(all_model_hdcst, obs, args.clim_years[0], args.clim_years[1])
            fd, fp = m.forecast(obs, args.clim_years[0], args.clim_years[1],
                                all_model_hdcst, all_model_fcst.isel(T=[-1]),
                                best_params=bp, cluster_da=cl)
            return write_and_plot("LogisticMME", hd, hp, fd, fp)

        if method == "fastbma":
            model_type = "gamma" if VARIABLE == "PRCP" else "normal"
            fold_det, fold_prob = [], []
            for tr, te in cv_cons.custom_cv.split(obs["T"], cv_cons.nb_omit):
                m = WAS_mme_FastBMA(model_type=model_type, bias_correction=True,
                                    bias_shrinkage=0.25, min_bias_years=8)
                m.fit(all_model_hdcst.isel(T=tr), obs.isel(T=tr))
                ds = m.predict_probabilistic(all_model_hdcst.isel(T=te),
                                             obs_for_terciles=climatology_sample(
                                                 obs if args.tercile_reference == "climatology" else obs.isel(T=tr),
                                                 *args.clim_years))
                fold_det.append(ds["predictive_mean"].assign_coords(T=obs["T"].isel(T=te)))
                fold_prob.append(ds["tercile_probability"].assign_coords(T=obs["T"].isel(T=te)))
            hd = xr.concat(fold_det, "T").sortby("T").reindex(T=obs["T"])
            hp = xr.concat(fold_prob, "T").sortby("T").reindex(T=obs["T"])
            m = WAS_mme_FastBMA(model_type=model_type, bias_correction=True,
                                bias_shrinkage=0.25, min_bias_years=8)
            m.fit(all_model_hdcst, obs)
            ds = m.predict_probabilistic(all_model_fcst.isel(T=[-1]), obs_for_terciles=climatology_sample(obs, *args.clim_years))
            return write_and_plot("FastBMA", hd, hp, ds["predictive_mean"], ds["tercile_probability"])

        if method == "gaussian_process":
            m = WAS_mme_gaussian_process(random_state=42, cv_folds=3, n_clusters=3,
                n_jobs=args.consolidation_n_jobs, optuna_n_jobs=min(2, args.consolidation_n_jobs),
                cv_n_jobs=1, n_trials=12, timeout=7200)
            hd, hp = cv_cons.cross_validate(m, obs, all_model_hdcst,
                args.clim_years[0], args.clim_years[1], nested_tuning=True)
            bp, cl = m.compute_hyperparameters(all_model_hdcst, obs, args.clim_years[0], args.clim_years[1])
            fd, fp = m.forecast(obs, args.clim_years[0], args.clim_years[1],
                                all_model_hdcst, all_model_fcst.isel(T=[-1]),
                                best_params=bp, cluster_da=cl)
            return write_and_plot("GaussianProcessMME", hd, hp, fd, fp)

        if method == "fullbma":
            if best_code_da is None:
                raise RuntimeError("FullBMA requires --dist-method bestfit.")
            m = WAS_mme_FullBMA(mode="fast", maxiter=250, verbose=False)
            # Strict CV: the full-record best_code_da contains the held-out years,
            # so the family map {norm, lognorm, gamma} (same candidates/mode as the
            # obs stage) is refitted on each outer-training climatology instead;
            # Event thresholds use the declared climatology; family fitting stays fold-local.
            hd, hp = cv_cons.cross_validate(m, obs, all_model_hdcst,
                args.clim_years[0], args.clim_years[1],
                distribution_fit=_fold_local_family_map, quantiles=[0.1, 0.5, 0.9],
                member_dim="M", time_dim="T", lat_dim="Y", lon_dim="X")
            # Operational fit on the whole record: full-record family map and
            # terciles are legitimate here (clim_terciles expects a threshold
            # DataArray in FullBMA, not a flag; obs_for_terciles provides them).
            ds = m.compute_model(all_model_hdcst, obs, all_model_fcst.isel(T=[-1]),
                dist_map=best_code_da, obs_for_terciles=climatology_sample(obs, *args.clim_years), quantiles=[0.1, 0.5, 0.9],
                member_dim="M", time_dim="T", lat_dim="Y", lon_dim="X")
            return write_and_plot("FullBMA", hd, hp, ds.get("predictive_mean"), ds.get("tercile_probability"))

        raise ValueError(f"Unknown consolidation method: {method}")

    failures = {}
    for method in selected:
        log_progress(f"Consensus: {method}")
        t0 = time.perf_counter()
        try:
            _execute(method)
            timings[method] = {"status": "ok", "seconds": round(time.perf_counter() - t0, 3)}
        except Exception as exc:
            timings[method] = {"status": "failed", "seconds": round(time.perf_counter() - t0, 3),
                               "error": repr(exc)}
            failures[method] = repr(exc)
            print(f"[consolidate/{method}] skipped: {exc}")
        finally:
            gc.collect()

    save_scores(workdir, consensus_scores, "scores_consolidated_consensus.pkl")
    pd.DataFrame(summary_rows).to_csv(scores_dir / "consolidation_summary.csv", index=False)
    save_json(scores_dir / "consolidation_timings.json", timings)
    mark_done(check_dir, "consolidate", {
        "probability_procedure": procedure,
        "profile": args.consolidation_profile, "methods": selected,
        "successful": [k for k,v in timings.items() if v["status"] == "ok"],
        "failed": failures, "numba_threads": int(getattr(args, "numba_threads", 0) or 0),
    })
    print(f"[consolidate] complete: {sum(v['status']=='ok' for v in timings.values())}/{len(selected)} methods succeeded")

def run(args):
    """Run PRCP stages using a parsed forecast configuration."""

    _scientific_preflight()
    if not getattr(args, "consolidation_n_jobs", 0):
        args.consolidation_n_jobs = max(1, min(int(args.ncores), 8))
    if args.consolidation_methods:
        args.consolidation_methods = _unique_preserve_order(
            [str(v).strip().lower() for v in args.consolidation_methods],
            label="--consolidation-methods")
    _configure_numba_runtime(args)

    # Normalize all repeated list arguments once, before any stage runs.
    args.mme_models = _unique_preserve_order(
        args.mme_models, label="--mme-models"
    )
    args.analog_methods = _unique_preserve_order(
        args.analog_methods, label="--analog-methods"
    )
    args.pcr_fields = _unique_preserve_order(
        args.pcr_fields, label="--pcr-fields"
    )
    args.cca_sst_domains = _unique_preserve_order(
        args.cca_sst_domains, label="--cca-sst-domains"
    )
    args.sst_domain_map = resolve_sst_domains(args.cca_sst_domains)
    args.drop_gcm = _unique_preserve_order(
        args.drop_gcm, label="--drop-gcm"
    )
    args.lead_time = _unique_preserve_order(
        args.lead_time, label="--lead-time"
    )
    args.only = _unique_preserve_order(
        args.only, label="--only"
    )

    _configure_svml_parallelism(args)

    #os.environ["WAS_NCORES"] = str(args.ncores)

    
    # PRE-SETUP: Securely define workdir and enforce local temp storage for Joblib 
    # to avoid SLURM /dev/shm semaphore limits
    workdir = Path(args.workdir).absolute()
    for d in ["intermediate","scores","forecasts","checkpoints","Observation","model_data","reanalysis_data", "dask-worker-space"]:
        (workdir / d).mkdir(exist_ok=True, parents=True)
        
    os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(workdir / "dask-worker-space"))

    client = _setup_parallel(args.ncores, args.threads_per_worker, use_dask=(not args.no_dask))

    args.season_months = SEASON_MAP[args.season]
    args.season_months_lag = SEASON_MAP[args.season_lag]
    downloader = WAS_Download()

    stages = {
        'obs':          lambda: stage_obs(args, workdir, downloader),
        'sv-ml-cmme':   lambda: stage_sv_ml_cmme(args, workdir, downloader, client),
        'cca-gcm-prcp':  lambda: _execute_cca_pipeline(args, workdir, downloader, "PRCP", args.gcm_extent, {'A': ('A', -26, 25, 4, 25)}, "cca_gcm-prcp", top_n=4),
        # SST domains are orchestrated as ONE CCA family. With EqWt ON, raw
        # members from every requested domain are pooled and averaged once into
        # CCA-EqWt.SST; with EqWt OFF they enter consolidation individually.
        'cca-gcm-sst':  lambda: stage_cca_gcm_sst(args, workdir, downloader),
       # 'cca-gcm-uwind': lambda: _execute_cca_pipeline(args, workdir, downloader, "UGRD_850", args.wind_extent, {'A': ('A', -60, 30, -50, 50)}, "cca_gcm_uwind", top_n=3),
       # 'cca-gcm-vwind': lambda: _execute_cca_pipeline(args, workdir, downloader, "VGRD_925", args.wind_extent, {'A': ('A', -60, 30, -15, 40)}, "cca_gcm_vwind", top_n=4),
        'obs-lag':      lambda: stage_obs_lag(args, workdir, downloader),
        'analog':       lambda: stage_analog(args, workdir),
        'telnet':       lambda: stage_telnet(args, workdir),
        'eof-kcca':     lambda: stage_eof_kcca(args, workdir, downloader),
        'consolidate':  lambda: stage_consolidate(args, workdir),
        'pbc':          lambda: stage_pbc(args, workdir),
    }
    

    run_list = ([key for key in stages if (key != 'telnet' or args.include_telnet)
                and (key != 'pbc' or args.include_pbc)]
                if 'all' in args.only else args.only)

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
