"""Argument parsing and dispatch for the two seasonal forecast launchers."""

import argparse
import datetime as dt
import math
import os
import sys

from .analog import add_analog_options, validate_analog_options
from .ensemble import add_ensemble_arguments
from .optional import add_optional_arguments
from .pbc import add_pbc_options, validate_pbc_options
from .probability import ProbabilityCalibrationAction, resolve_probability_calibration
from .telnet import add_telnet_options, validate_telnet_options

DEFAULT_LOGO = "./utilities/cilss.png"
DEFAULT_COUNTRY_ISO3 = "WAF"
DEFAULT_CLIM_YEARS = (1994, 2016)
DEFAULT_HIND_YEARS = (1993, 2016)
DEFAULT_OBS_YEARS = (1991, dt.date.today().year - 1)

SEASON_MAP = {
    "JFM": ["01", "02", "03"], "FMA": ["02", "03", "04"], "MAM": ["03", "04", "05"],
    "AMJ": ["04", "05", "06"], "MJJ": ["05", "06", "07"], "JJA": ["06", "07", "08"],
    "JAS": ["07", "08", "09"], "ASO": ["08", "09", "10"], "SON": ["09", "10", "11"],
    "OND": ["10", "11", "12"], "NDJ": ["11", "12", "01"], "DJF": ["12", "01", "02"],
}
GLOBAL_SST_EXTENT = [60, -180, -60, 180]

# =============================================================================
# SST predictor domains for the CCA of approach 2 (GCM-SST) and approach 3
# (OBS-LAG).
#
# Why a short, physically motivated list rather than a lat/lon sweep, and why
# NO selection inside a domain: measured on pure-noise members over 24 years
# (400 replicates), the apparent ACC of the final consensus is
#
#     4 domains, no within-domain selection, equal weight      median  0.01
#     7 domains, no within-domain selection, equal weight      median  0.01
#     4 domains, members kept above a GROC-like threshold      median  0.55
#     4 domains, top_n = 3 members kept per domain             median  0.57
#     4 domains x 8 GCMs, top_n = 8 selected globally          median  0.61
#     7 domains x 8 GCMs, top_n = 8 selected globally          median  0.69
#
# Looping over domains is essentially free. What manufactures skill is
# selecting members on the same 24 years that later score them, and a fixed
# threshold does it almost as badly as a top_n. So: keep every domain, keep
# every GCM inside a domain, average equally, and let the MME weighting
# downstream be the single declared selection step.
#
# Coordinates are (label, lon_min, lon_max, lat_min, lat_max) in the -180..180
# convention used by retrieve_single_zone_for_PCR.
#
# On the historical default ('A', -150, 150, -45, 45): the excluded band is
# 150E..150W, which removes Nino 4 entirely and 40% of Nino 3.4 while keeping
# Nino 3 and Nino 1+2. 'global_trop' restores the full tropical belt;
# 'legacy' reproduces the old domain for continuity checks.
# =============================================================================
SST_DOMAINS = {
    "nino34":      ("nino34",      -170, -120,   -5,    5),
    "nino4":       ("nino4",        160, -150,   -5,    5),   # crosses the dateline
    "atl_nord":    ("atl_nord",     -60,  -20,    5,   25),
    "atl_eq":      ("atl_eq",       -20,   10,   -5,    5),
    "dipole_atl":  ("dipole_atl",   -60,   15,  -20,   25),
    "indien_trop": ("indien_trop",   40,  110,  -20,   20),
    "medit":       ("medit",         -5,   36,   30,   45),
    "global_trop": ("global_trop", -180,  180,  -30,   30),
    "legacy":      ("legacy",      -150,  150,  -45,   45),
}

DEFAULT_SST_DOMAINS = ["global_trop", "nino34", "atl_nord", "dipole_atl"]


def _cca_mode_selection(args):
    """Resolve the canonical selector and its explicit legacy alias."""
    canonical = getattr(args, "cca_mode_selection", None)
    legacy = getattr(args, "cca_truncation_selection", None)
    explicit = getattr(args, "_cca_mode_selection_explicit", canonical is not None)
    if legacy is not None:
        if explicit and canonical is not None and canonical != legacy:
            raise ValueError("Conflicting --cca-mode-selection and --cca-truncation-selection")
        canonical = legacy
    selection = canonical or "mdl"
    if selection not in ("mdl", "cv", "aic"):
        raise ValueError("CCA mode selection must be mdl, cv or aic")
    return selection


def parse_args(variable, argv=None):
    """Parse the historical command line for PRCP or TEMP."""
    if variable not in ("PRCP", "TEMP"):
        raise ValueError(f"Unsupported forecast variable: {variable!r}")
    rain = variable == "PRCP"
    argv = sys.argv[1:] if argv is None else list(argv)
    wa_extent = [21 if rain else 30, -26, 4, 25]
    wind_extent = [50, -60, -15, 30] if rain else [30, -30, -10, 30]

    p = argparse.ArgumentParser(description="WASS2S Seasonal Forecast CLI (Dynamic Modular)")
    add_optional_arguments(p)
    p.add_argument('--workdir', default=f"./{variable}_JAS_2025_ic_4/")
    p.add_argument('--season', default='JAS', choices=list(SEASON_MAP.keys()))
    p.add_argument('--season-lag', default='JFM', choices=list(SEASON_MAP.keys()))
    p.add_argument('--init-month', default='04', help='Initialization month as MM')
    p.add_argument('--forecast-year', type=int, default=dt.date.today().year)
    p.add_argument('--clim-years', nargs=2, type=int, default=list(DEFAULT_CLIM_YEARS))
    p.add_argument('--hindcast-years', nargs=2, type=int, default=list(DEFAULT_HIND_YEARS))
    p.add_argument('--obs-years', nargs=2, type=int, default=list(DEFAULT_OBS_YEARS))
    
    p.add_argument('--obs-extent', nargs=4, type=float, default=wa_extent.copy(), metavar=('N','W','S','E'))
    p.add_argument('--gcm-extent', nargs=4, type=float, default=wa_extent.copy(), metavar=('N','W','S','E'))
    p.add_argument('--sst-extent', nargs=4, type=float, default=GLOBAL_SST_EXTENT, metavar=('N','W','S','E'))
    p.add_argument('--wind-extent', nargs=4, type=float, default=wind_extent, metavar=('N','W','S','E'))
    
    p.add_argument('--obs-source', 
                   default='chirps' if rain else 'agro',
                   choices=['agro', 'chirps', 'tamsat'] if rain else ['agro', 'chirps'], help='Observation data source')
    p.add_argument('--dist-method', default='bestfit', choices=['bestfit','nonparam'])
    p.add_argument('--n-clusters', type=int, default=200, help='Nombre de zones pour l’ajustement des distributions; 200 est le profil WA opérationnel.')
    p.add_argument('--cpt-csv', default=None)
    p.add_argument('--merge-date', default='05-01')
    
    p.add_argument('--mme-models', nargs='+', default=['ridge'], choices=['hpelm','rf','xgb','mlp','ridge','elasticnet','svr','extratrees','histgb','mars','pls','elr','cca','pcr','logistic','gp','unet','unetpp','rise-unet','rescnn','kharin'])
    p.add_argument(
        '--svml-profile',
        choices=['fast', 'balanced', 'full'],
        default='fast' if rain else 'balanced',
        help=(
            'SV-ML-CMME search budget. fast is recommended for operational '
            'West-Africa runs; balanced is the default; full restores the '
            'original exhaustive searches.'
        ),
    )
    p.add_argument('--top-n-gcm', type=int, default=4, help="Deprecated for SV-ML: use --gcm-models/--drop-gcm.")
    p.add_argument('--add-ngr', action='store_true', help='Add a second NGR variant (broader top-n)')
    
    p.add_argument('--obs-lag-policy', default='both', choices=['both','mlr','cca','pcr'])
    p.add_argument(
        '--cluster-n-jobs', type=int, default=0,
        help='Threads used to fit the ML clusters concurrently (0 = auto: '
             'min(n_clusters, svml-n-jobs)). Clusters are independent, and '
             'measurements show n_jobs inside a single RandomForest fit buys '
             'nothing at 10 features, so this is the parallel axis that pays.',
    )
    p.add_argument(
        '--spatial-subsample', type=int, default=1,
        help='Keep every Nth grid cell when FITTING the ML models (1 = off). '
             'Prediction still covers the full grid. On a 350x171 grid a '
             'cluster holds ~287,000 samples for 10 features and neighbouring '
             'cells are near-duplicates; the fit cost grows as n log n, so a '
             'stride of 2 cuts it by about 4.5x. THIS CHANGES THE FORECASTS: '
             'validate it against a stride-1 run before adopting it.',
    )
    p.add_argument(
        '--svml-spatial-mode', choices=['zones', 'soft', 'both'], default='zones',
        help='Spatial variability of the pooled ML MMEs (rf/xgb/mlp/hpelm and the '
             'sklearn challengers). zones = one model per KMeans climate zone '
             '(historical). soft = ONE model on the whole domain with fold-local '
             'spatial covariates appended to the GCM channels (climatological '
             'mean/log-std/asymmetry/dry fraction of the training predictand, '
             'z-scored across pixels; Dabernig et al. 2017 "SAMOS"): no seams, '
             'no unstable zone boundaries, same cost. both = zone models AND the '
             'covariates. Compare in the same cross-validation before adopting.',
    )
    p.add_argument(
        '--svml-covariates', nargs='+', default=['clim_mean', 'clim_logstd', 'clim_asym', 'dry_frac'],
        choices=['clim_mean', 'clim_logstd', 'clim_asym', 'dry_frac'],
        help='Climatological covariates used by --svml-spatial-mode soft/both.',
    )
    p.add_argument(
        '--svml-coord-step', type=float, default=0.,
        help='Add latitude/longitude coarsened to this step (degrees) as '
             'covariates (0 = off; keep off for trees unless the step is >= 1-2 '
             'degrees: raw coordinates let trees carve latitude bands).',
    )
    p.add_argument(
        '--svml-static-fields', default=None,
        help='NetCDF file whose (Y, X) variables (elevation, distance to coast, '
             '...) are appended as static covariates in soft/both mode; they are '
             'interpolated (nearest) to the predictand grid.',
    )
    p.add_argument(
        '--svml-zones-as-features', action='store_true', default=False,
        help='In soft mode, also append the one-hot KMeans zone membership so '
             'the learner can reproduce the zone partition when it helps.',
    )
    p.add_argument(
        '--svml-predictor-smoothing', type=int, default=0,
        help='Append regional means (N x N cells) of every GCM channel as extra '
             'predictors (0 = off). The predictable seasonal signal is large '
             'scale; the grid-point GCM value is noisy.',
    )
    p.add_argument(
        '--svml-one-se', dest='svml_one_se', action='store_true', default=True,
        help='Strict regional tuning keeps the most regularized candidate within '
             'one standard error (over validation years) of the best score '
             '(default on; --no-svml-one-se restores the plain maximum).',
    )
    p.add_argument('--no-svml-one-se', dest='svml_one_se', action='store_false')
    p.add_argument(
        '--svml-amplitude-calibration', dest='svml_amplitude_calibration',
        action='store_true', default=True,
        help='Rescale each pooled ML prediction with a single domain-wide slope '
             'fitted on the training fold. Validated on 17 GCM hindcasts '
             '(MJJ West Africa, 0.1 deg): MLP "zones" went from MSSS -0.522 to '
             '+0.011 while keeping r = 0.187 against 0.198. It also collapses a '
             'degenerate fit to climatology (default on; '
             '--no-svml-amplitude-calibration restores the raw output).',
    )
    p.add_argument('--no-svml-amplitude-calibration',
                   dest='svml_amplitude_calibration', action='store_false')
    p.add_argument(
        '--svml-amplitude-mode', choices=['global', 'spread'], default='global',
        help='global: one slope per fold. spread: one slope per forecast year, '
             'driven by the ensemble signal-to-spread ratio (member statistics '
             'channels model__mean/model__std when ENSEMBLE_MEAN=None, inter-model '
             'spread of the means otherwise). The sensitivity is shrunk by its '
             't statistic and never negative, so it degrades to global.',
    )
    p.add_argument(
        '--svml-amplitude-inner-folds', type=int, default=3,
        help='Estimate the slope on OUT-OF-SAMPLE predictions of the training '
             'years (blocked inner CV, cost x(1+k)). 0 = in-sample slope, biased '
             'towards 1 for any learner that fits its data (measured: MSSS -0.102 '
             'against -0.014 for XGBoost).',
    )
    p.add_argument(
        '--cca-sst-domains', nargs='+', default=DEFAULT_SST_DOMAINS,
        choices=sorted(SST_DOMAINS),
        help='SST predictor domains explored by the CCA of approaches 2 and 3. '
             'For GCM-SST with --cca-equal-weight, all retained CCA members '
             'from all requested domains are pooled into ONE CCA-EqWt.SST. '
             'Selecting members inside a domain is discouraged (see SST_DOMAINS).',
    )
    p.add_argument(
        '--cca-domain-top-n', type=int, default=0,
        help='0 (default) keeps every GCM inside a domain and averages them '
             'equally: no selection, hence no selection bias. A positive value '
             'restores a top_n selection per domain and is DIAGNOSTIC ONLY: on '
             '24 years, top_n=3 of 8 yields a median apparent ACC of 0.57 on '
             'pure noise.',
    )
    p.add_argument(
        '--cca-per-domain-members', action='store_true',
        help='OBS-LAG CCA diagnostic only: keep one member per SST domain '
             'instead of the default OBS-LAG domain consensus. This option '
             'does not change GCM-SST --cca-equal-weight semantics.',
    )
    p.add_argument(
        '--strict-time-alignment', dest='strict_time_alignment',
        action='store_true', default=True,
        help='Abort when a restamped predictor does not span the same years as '
             'the predictand (default).',
    )
    p.add_argument(
        '--no-strict-time-alignment', dest='strict_time_alignment',
        action='store_false',
        help='Warn instead of aborting on a predictor/predictand year mismatch.',
    )
    p.add_argument('--pcr-fields', nargs='+', default=['SST'],
                    choices=['SST','UGRD_850','VGRD_850','TEMP'],
                    help='Predictor fields combined into the obs-lag PCR EOF. '
                         'Default is SST only; add UGRD_850/VGRD_850/TEMP for a '
                         'multivariate PCR (extra fields are downloaded only if requested).')
    p.add_argument('--vif-threshold', type=float, default=5.0)

    add_analog_options(p)
    add_telnet_options(p)
    add_pbc_options(p)
    
    # Fifth approach: EOF -> KCCA
    p.add_argument('--eof-kcca', dest='eof_kcca_enabled', action='store_true', default=True,
                   help='Enable the independent fifth EOF-KCCA approach (default).')
    p.add_argument('--no-eof-kcca', dest='eof_kcca_enabled', action='store_false')
    p.add_argument('--eof-kcca-fields', nargs='+', default=[variable, 'SST'],
                   choices=['PRCP','TEMP','SST','SLP','UGRD_850','VGRD_850','UGRD_925','VGRD_925','DLWR','HUSS_1000'],
                   help='Physical GCM fields reduced independently by EOF before KCCA.')
    p.add_argument('--eof-kcca-min-fields', type=int, default=2)
    p.add_argument('--eof-kcca-eof-modes', type=int, default=3,
                   help='EOF modes retained PER field (joint=False). Keep small on ~24 years.')
    p.add_argument('--eof-kcca-modes', type=int, default=2, help='KCCA latent dimensions.')
    p.add_argument('--eof-kcca-pca-modes', type=int, default=4,
                   help='Final PCA cap applied to both KCCA views after the per-field EOF reduction.')
    p.add_argument('--eof-kcca-kernel', choices=['linear','rbf','poly','sigmoid'], default='linear')
    p.add_argument('--eof-kcca-search-method', choices=['grid','bayesian'], default='grid')
    p.add_argument('--eof-kcca-tuning-objective', choices=['canonical','reconstruction'], default='reconstruction')
    p.add_argument('--eof-kcca-inner-folds', type=int, default=4)
    p.add_argument('--eof-kcca-nested-tuning', dest='eof_kcca_nested_tuning', action='store_true', default=True)
    p.add_argument('--no-eof-kcca-nested-tuning', dest='eof_kcca_nested_tuning', action='store_false')
    p.add_argument('--eof-kcca-joint', action='store_true',
                   help='Use one multivariate EOF instead of one EOF per field.')
    cca_detrending = p.add_mutually_exclusive_group()
    cca_detrending.add_argument(
        '--cca-detrend', '--eof-kcca-detrend', dest='cca_detrend',
        action='store_true', default=True,
        help='Detrend training fields in every CCA path (default): standard, '
             'regularized, MME and EOF-KCCA. The EOF-KCCA flag is a legacy alias '
             'with the same global scope.')
    cca_detrending.add_argument(
        '--no-cca-detrend', '--no-eof-kcca-detrend', dest='cca_detrend',
        action='store_false',
        help='Disable trend fitting/removal/restoration in every CCA path; '
             'cross-validation, calibration and mode selection remain active.')
    p.add_argument('--eof-kcca-eof-standardize', action='store_true', default=False)
    p.add_argument('--eof-kcca-reg', type=float, default=1e-6)
    p.add_argument('--eof-kcca-max-feature-ratio', type=float, default=0.5)
    p.add_argument('--eof-kcca-max-centers', type=int, default=0,
                   help='0=all common centres; positive cap is alphabetical, never skill-selected.')
    p.add_argument('--eof-kcca-per-center-members', action='store_true',
                   help='Diagnostic only: expose every centre to final MME instead of one equal-weight fifth member.')

    # Scientific CV / analog / consolidation controls
    p.add_argument('--numba-threads', type=int, default=0,
                   help='Numba threads (0 = auto, capped by --ncores).')
    p.add_argument('--verification-min-cells', type=int, default=4000,
                   help='Minimum Y*X grid cells before threaded Numba verification '
                        '(default 4000; Numba kernels remain active below this threshold).')
    p.add_argument('--svml-nested-tuning', dest='svml_nested_tuning', action='store_true', default=True)
    p.add_argument('--no-svml-nested-tuning', dest='svml_nested_tuning', action='store_false')
    p.add_argument('--svml-inner-folds', type=int, default=5)
    p.add_argument('--svml-probability-folds', type=int, default=4)
    p.add_argument('--probability-inner-folds', type=int, default=None,
                   help='Common calibration subfold count (>=2); overrides legacy stage counts. Ignored by outer.')
    p.add_argument('--probability-inner-retune', dest='probability_inner_retune',
                   action='store_true', default=True,
                   help='Retune inside each inner calibration fit (research default); independent of outer evaluation.')
    p.add_argument('--no-probability-inner-retune', dest='probability_inner_retune',
                   action='store_false',
                   help='Reuse outer-training hyperparameters in calibration fits; cheaper conditional approximation.')
    p.add_argument('--obs-lag-nested-tuning', dest='obs_lag_nested_tuning', action='store_true', default=True,
                   help='Select PCR parameters with complete EOF refits inside each outer training fold.')
    p.add_argument('--no-obs-lag-nested-tuning', dest='obs_lag_nested_tuning', action='store_false',
                   help='Use fixed PCR backend parameters; probability calibration remains independently controlled.')
    p.add_argument('--obs-lag-inner-folds', type=int, default=3,
                   help='PCR hyperparameter selection folds, distinct from probability calibration folds.')
    p.add_argument('--obs-lag-ridge-alpha', type=float, default=1.0,
                   help='Prespecified Ridge alpha used when OBS-LAG parameter tuning is disabled.')
    p.add_argument('--tercile-reference', choices=['climatology', 'training'],
                   default='climatology',
                   help='Fixed observed climatological terciles (default); training restores fold-specific events.')



    if rain:
        p.add_argument('--dry-threshold', type=float, default=100.0,
                       help='PRCP seasonal mean threshold (mm), shared by observations and outputs; 100 preserves the original observation mask.')
        p.add_argument('--dry-min-valid-years', type=int, default=8)
        p.add_argument('--apply-drymask', dest='apply_drymask', action='store_true', default=True)
        p.add_argument('--no-drymask', dest='apply_drymask', action='store_false',
                       help='Disable the rainfall dry-threshold mask; valid-observation mask is still applied.')

    p.add_argument('--consolidation-profile', choices=['fast','standard','extended'], default='fast',
                   help='fast: cheap/robust methods; standard adds moderate ML/ELR; extended includes expensive BMA/GP.')
    p.add_argument('--consolidation-methods', nargs='*', default=None,
                   choices=['equal','weighted','weighted_continuous','probweighted',
                            'probweighted_continuous','probweighted_rps','min2009','mva',
                            'ngr_gaussian','ngr_nongaussian','pcr','ridge','elasticnet','svr',
                            'extratrees','histgb','expert','elr','logistic','fastbma',
                            'gaussian_process','fullbma'],
                   help='Explicit methods override --consolidation-profile.')
    p.add_argument('--consolidation-plot-level', choices=['none','scores','forecast','all'], default='forecast')
    p.add_argument('--consolidation-n-jobs', type=int, default=0,
                   help='Workers for consolidation ML (0 = min(ncores,8)).')
    p.add_argument('--consolidation-clusters', type=int, default=3)
    p.add_argument('--consolidation-inner-folds', type=int, default=4)
    p.add_argument('--consolidation-prob-folds', type=int, default=4)
    p.add_argument('--consolidation-pcr-modes', type=int, default=5)
    p.add_argument('--consolidation-rps-stride', type=int, default=4,
                   help='Spatial stride used only to FIT global RPS/expert weights; prediction remains full-grid.')
    p.add_argument('--consolidation-weight-shrinkage', type=float, default=0.20)
    p.add_argument('--consolidation-nested-tuning', dest='consolidation_nested_tuning', action='store_true', default=True)
    p.add_argument('--no-consolidation-nested-tuning', dest='consolidation_nested_tuning', action='store_false')
    p.add_argument('--ngr-prcp-family', choices=['gamma','censored_normal','truncated_normal','transformed_gaussian'], default='gamma')

    # Parallelism Args
    p.add_argument('--ncores', type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    p.add_argument(
        '--svml-n-jobs',
        type=int,
        default=None,
        help=(
            'Parallel workers used by SV-ML-CMME model searches and final '
            'RF/XGBoost fits. Default: --ncores. Use -1 for the full allocated '
            'CPU budget; values above --ncores are clamped.'
        ),
    )
    p.add_argument(
        '--svml-optuna-n-jobs',
        type=int,
        default=None,
        help=(
            'Concurrent Optuna trials for SV-ML-CMME XGBoost/MLP. '
            'Default: --svml-n-jobs. Use -1 for the full SV-ML budget.'
        ),
    )
    p.add_argument(
        '--svml-cv-n-jobs',
        type=int,
        default=None,
        help=(
            'Workers utilisés pour paralléliser les plis CV à l’intérieur '
            'de chaque essai Optuna. Valeur automatique: min(3, '
            'ncores // svml-optuna-n-jobs). Les estimateurs RF/XGB restent '
            'à un thread pendant le tuning pour éviter la sur-souscription.'
        ),
    )
    p.add_argument(
        '--svml-tuning-model-n-jobs',
        type=int,
        default=None,
        help=(
            'Threads internes utilisés par chaque estimateur RF/XGBoost pendant '
            'le tuning. Valeur automatique: 2 lorsque le budget le permet. '
            'Le CLI impose optuna_n_jobs × cv_n_jobs × cette valeur <= ncores.'
        ),
    )
    p.add_argument('--threads-per-worker', type=int, default=1)
    p.add_argument('--no-dask', action='store_true')
    
    # Plotting Overlays
    p.add_argument('--country-code', default=DEFAULT_COUNTRY_ISO3, help="GADM ISO3 Country/Region Code")
    p.add_argument('--logo', default=DEFAULT_LOGO)
    
    add_ensemble_arguments(p)
    p.add_argument('--lead-time', nargs='+', default=['03','04','05'])
    p.add_argument('--force', action='store_true')
    p.add_argument('--reuse-obs', action='store_true')
    p.add_argument('--redo', action='store_true')
    
    stages = ['all', 'obs', 'sv-ml-cmme']
    if not rain:
        stages.append('cca-gcm-temp')
    stages.extend(['cca-gcm-prcp', 'cca-gcm-sst', 'obs-lag', 'analog',
                   'telnet', 'eof-kcca', 'consolidate', 'pbc'])
    p.add_argument('--only', nargs='*', default=['all'], choices=stages)
    drop_gcm = (
        ['UKMO_603.PRCP', 'DWD_21.PRCP', 'METEOFRANCE_8.PRCP', 'UKMO_605.PRCP', 'JMA_3.PRCP',
         'UKMO_604.PRCP', 'CMCC_35.PRCP', 'UKMO_603.SST', 'UKMO_605.SST', 'DWD_21.SST',
         'METEOFRANCE_8.SST', 'UKMO_605.SST', 'JMA_3.SST', 'UKMO_604.SST', 'CMCC_35.SST',
         'UKMO_603.VGRD_925', 'DWD_21.VGRD_925', 'METEOFRANCE_8.VGRD_925', 'UKMO_605.VGRD_925',
         'JMA_3.VGRD_925', 'UKMO_604.VGRD_925', 'CMCC_35.VGRD_925'] if rain else ['CMC2_1.SST']
    )
    p.add_argument('--drop-gcm', nargs='*', default=drop_gcm,
                   help='GCM models to exclude in CCA stage')
    p.add_argument('--cca-modes', type=int, default=3)
    p.add_argument('--cca-pca-modes', type=int, default=8)
    p.add_argument(
        '--cca-truncation-selection', choices=['cv', 'aic'], default=None,
        help=("Legacy explicit alias for --cca-mode-selection (cv/aic); also "
              "sets the separate MME-CCA selector. MME-CCA otherwise retains CV. "
              "Conflicting explicit selectors are rejected."))
    p.add_argument(
        '--probability-calibration', choices=['inner', 'nested', 'outer'], default=None,
        action=ProbabilityCalibrationAction,
        help=("Residual calibration shared by CCA, continuous MME regressors, PCR and CNN. "
              "outer (operational default) reuses "
              "other outer-fold errors without calibration subfolds. inner (alias nested) refits within each outer training fold for research; outer historical "
              "probability scores are approximate. Native probability models retain "
              "their own distribution/classification procedure."))
    p.add_argument(
        '--cca-probability-calibration', choices=['inner', 'nested', 'outer'], default=None,
        action=ProbabilityCalibrationAction,
        help="Legacy global alias for --probability-calibration; conflicting aliases are rejected.")
    p.add_argument(
        '--cca-auto-truncation', choices=['auto', 'on', 'off'], default='auto',
        help=(
            "Enable automatic mode selection for the standard CCA engine. "
            "'auto' (default) follows WAS_CCA_AUTOTUNE; 'on' forces the "
            "procedure selected by --cca-mode-selection. 'off' takes priority "
            "and uses fixed --cca-modes/--cca-pca-modes. Ignored by the "
            "regularized engine."
        ),
    )
    p.add_argument(
        '--cca-mode-selection', choices=['mdl', 'cv', 'aic'], default='mdl',
        help=(
            "Mode selection for standard CCA (GCM and OBS-LAG): 'mdl' "
            "(default) uses Song et al. (2016), Detector 3, on each training "
            "sample without inner mode-selection CV; 'cv' restores the "
            "cross-validated one-SE search; aic is an experimental Bartlett-AIC selector. "
            "MDL and AIC may select zero shared modes. MME-CCA retains CV/AIC. "
            "--cca-auto-truncation off overrides this choice. Not used by "
            "the regularized engine; probability-calibration folds are unchanged."
        ),
    )
    p.add_argument(
        '--cca-engine', choices=['standard', 'regularized'], default='standard',
        help=(
            "CCA engine for both CCA-GCM and OBS-LAG. 'standard' (default) is "
            "WAS_CCA_op/WAS_CCA_strict with the mode-count truncation search. "
            "'regularized' uses ridge-regularized CCA (WAS_CCA_Regularized / "
            "WAS_CCA_Regularized_strict): covariance shrinkage instead of a "
            "mode-count search — steadier on a short, correlated-predictor "
            "hindcast. --cca-auto-truncation and --cca-mode-selection are "
            "ignored under 'regularized'."
        ),
    )
    p.add_argument(
        '--cca-alpha', type=float, default=0.1,
        help=(
            "Ridge shrinkage strength for --cca-engine=regularized, in [0, 1]. "
            "0 recovers unregularized CCA; 1 shrinks the covariance fully to a "
            "scaled identity. Tune by out-of-sample CV like any ridge penalty. "
            "Ignored under --cca-engine=standard."
        ),
    )
    eqwt_help = (
        'Historical EqWt semantics: all retained CCA-PRCP members are '
        'collapsed into ONE CCA-EqWt.PRCP and all retained CCA-SST '
        'members across ALL requested SST domains are collapsed into '
        'ONE CCA-EqWt.SST. Individual CCA members remain diagnostics '
        'under cca_members/ and do not enter final consolidation.'
        if rain else
        'Historical EqWt semantics: all retained CCA-TEMP members are '
        'collapsed into ONE CCA-EqWt.TEMP; all retained CCA-SST members '
        'across ALL requested SST domains into ONE CCA-EqWt.SST; and the '
        'existing CCA-PRCP family into ONE CCA-EqWt.PRCP. Individual CCA '
        'members remain diagnostics under cca_members/.'
    )
    p.add_argument('--cca-equal-weight', action='store_true', help=eqwt_help)
    args = p.parse_args(argv)
    if not rain and "ngr_nongaussian" in (args.consolidation_methods or []):
        p.error("ngr_nongaussian is precipitation-only; use ngr_gaussian for TEMP.")
    args.eof_kcca_detrend = args.cca_detrend  # synchronized legacy API attribute
    args._cca_mode_selection_explicit = any(
        item == "--cca-mode-selection" or item.startswith("--cca-mode-selection=")
        for item in argv)
    try:
        args.cca_mode_selection = _cca_mode_selection(args)
        resolve_probability_calibration(args)
        validate_analog_options(args)
        validate_telnet_options(args)
        validate_pbc_options(args)
        if args.obs_lag_inner_folds < 2:
            raise ValueError("--obs-lag-inner-folds must be at least 2")
        if not math.isfinite(args.obs_lag_ridge_alpha) or args.obs_lag_ridge_alpha < 0:
            raise ValueError("--obs-lag-ridge-alpha must be finite and nonnegative")
    except ValueError as exc:
        p.error(str(exc))
    return args



def main(variable, argv=None):
    """Run a seasonal forecast with the same arguments accepted by the scripts."""
    args = parse_args(variable, argv)
    if variable == "PRCP":
        from .prcp import run
    else:
        from .temp import run
    return run(args)
