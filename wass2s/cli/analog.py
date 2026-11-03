"""Shared analogue configuration and checkpoint contract for PRCP and TEMP."""
from pathlib import Path
import hashlib
import json
import math


ANALOG_METHODS = (
    'bias_based', 'som', 'cor_based', 'pca_based', 'kmeans_based',
    'agglomerative_based', 'knn', 'weighted', 'eof', 'hybrid', 'constructed',
    'learned_metric', 'learned_spatial_metric',
)
ANALOG_SCHEMA = '2026-10-01-fold-safe-analogs-1'


def add_analog_options(parser):
    parser.add_argument('--analog-methods', nargs='+', default=['bias_based', 'som'],
                        choices=ANALOG_METHODS,
                        help='Analogue algorithms; all use training-only transformations and native ensemble probabilities.')
    parser.add_argument('--analog-nested-tuning', action='store_true', default=False,
                        help='Select analogue hyperparameters inside each outer training set (research).')
    parser.add_argument('--no-analog-nested-tuning', dest='analog_nested_tuning', action='store_false')
    parser.add_argument('--analog-detrend', action='store_true', default=False,
                        help='Fit predictor trends on the training years only.')
    parser.add_argument('--no-analog-detrend', dest='analog_detrend', action='store_false')
    parser.add_argument('--analog-inner-preprocessing', choices=['outer', 'inner', 'strict'], default='strict',
                        help='strict/inner refits transforms in selection subfolds; outer reuses outer-training transforms (conditional approximation). Inactive with fixed parameters.')
    parser.add_argument('--analog-target-member-mode', choices=['mean', 'median', 'first', 'memberwise'], default='mean')
    parser.add_argument('--analog-max-members', type=int, default=0)
    parser.add_argument('--analog-eof-modes', type=int, default=6)
    parser.add_argument('--analog-k', type=int, default=4,
                        help='Prespecified analogue count when parameter selection is disabled.')
    parser.add_argument('--analog-k-grid', nargs='+', type=int, default=[2, 3, 4, 5, 6])
    parser.add_argument('--analog-probability-shrinkage', default='0.25',
                        help='Fixed shrinkage toward climatology in [0,1] (default .25); auto explicitly selects it using inner validation.')


def validate_analog_options(args):
    if args.analog_inner_preprocessing == 'inner':
        args.analog_inner_preprocessing = 'strict'
    if args.analog_k < 2 or args.analog_eof_modes < 1 or args.analog_max_members < 0:
        raise ValueError('Analogue k must be at least 2, EOF modes positive and max-members nonnegative')
    if not args.analog_k_grid or any(k < 2 for k in args.analog_k_grid):
        raise ValueError('--analog-k-grid must contain integers of at least 2')
    shrinkage = str(args.analog_probability_shrinkage).strip().lower()
    if shrinkage != 'auto':
        try:
            shrinkage = float(shrinkage)
        except ValueError as exc:
            raise ValueError('--analog-probability-shrinkage must be auto or a number in [0,1]') from exc
        if not math.isfinite(shrinkage) or not 0 <= shrinkage <= 1:
            raise ValueError('--analog-probability-shrinkage must be auto or a number in [0,1]')
    args.analog_probability_shrinkage = shrinkage
    return args


def get_analog_method_configs(args, workdir, module):
    """Keep climate events, hindcast initialisations and target years separate."""
    from .runtime import source_hindcast_years
    init = int(args.init_month)
    leads = [int(v) for v in args.lead_time]
    if not leads or not 1 <= init <= 12 or any(v < 0 or v > 11 for v in leads):
        raise ValueError('Analogue lead offsets must be in 0..11 with an initialization month in 1..12')
    offset = (init - 1 + leads[len(leads) // 2]) // 12
    archive = source_hindcast_years(args)
    common = dict(
        dir_to_save=str(Path(workdir) / 'analog'),
        year_start=min(int(args.obs_years[0]), int(archive[0])) - 1,
        year_forecast=int(args.forecast_year) + offset,
        hindcast_year_start=int(archive[0]), hindcast_year_end=int(archive[1]),
        target_year_offset=offset, month_of_initialization=init, lead_time=leads,
        clim_year_start=args.clim_years[0], clim_year_end=args.clim_years[1],
        dist_method=args.dist_method,
        predictor_vars=[dict(reanalysis_name='NOAA', model_name='NCEP_2',
                             variable='SST', area=[45, -180, -45, 160])],
        standardize=True, detrend=bool(args.analog_detrend), rolling=1,
        index_compute=['NINO34', 'NINO12', 'TNA', 'TSA', 'NAT', 'SAT', 'TASI', 'WTIO', 'SETIO', 'DMI'],
        # Cache raw members, then apply the analogue-specific reduction.
        ensemble_mean=None, target_member_mode=args.analog_target_member_mode,
        max_target_members=(args.analog_max_members or None),
        n_analogs=args.analog_k, k_grid=tuple(args.analog_k_grid), tune_score='rps',
        nested_tuning=bool(args.analog_nested_tuning),
        inner_preprocessing=args.analog_inner_preprocessing,
        probability_method='analog_ensemble',
        probability_shrinkage=args.analog_probability_shrinkage,
        quality_shrinkage=True, analog_bias_method='VARSCALE',
        max_eof_modes=args.analog_eof_modes, random_state=42,
        tercile_reference=getattr(args, 'tercile_reference', 'climatology'),
    )
    configs = {}
    legacy = {'bias_based': ('biasbased', 'Bias-Based'), 'som': ('som', 'SOM'),
              'cor_based': ('corbased', 'Correlation'), 'pca_based': ('pcabased', 'PCA-Based'),
              'kmeans_based': ('kmeans', 'KMeans'), 'agglomerative_based': ('agglomerative', 'Agglomerative')}
    for name, (tag, title) in legacy.items():
        kwargs = dict(common, method_analog=name)
        if name == 'som':
            kwargs.update(some_grid_size=(5, 5), some_sigma=0.5,
                          some_learning_rate=0.1, some_num_iteration=4000, radius=1.0)
        configs[name] = (module.WAS_Analog_Scientific, kwargs, 'fourth.approach_' + tag, 'Analog ' + title)
    for name, suffix in [('knn', 'KNN'), ('weighted', 'Weighted'), ('eof', 'EOF'),
                         ('hybrid', 'Hybrid'), ('constructed', 'Constructed'),
                         ('learned_metric', 'LearnedMetric'), ('learned_spatial_metric', 'LearnedSpatialMetric')]:
        configs[name] = (getattr(module, 'WAS_Analog_' + suffix), dict(common),
                         'ANALOG.' + suffix, 'Analog ' + suffix)
    return configs


def analog_training_sample(observations, args):
    """Fit only years for which model initialisations are declared available.

    The caller retains the full observed reference separately for event terciles.
    """
    from .runtime import source_hindcast_years
    leads = [int(value) for value in args.lead_time]
    offset = (int(args.init_month) - 1 + leads[len(leads) // 2]) // 12
    lo, hi = (int(year) + offset for year in source_hindcast_years(args))
    sample = observations.sel(T=slice(str(lo), str(hi)))
    if sample.sizes.get('T', 0) < 4:
        raise ValueError('Analogue validation needs at least four observed target years within the model hindcast archive')
    if int(sample['T'].dt.year.max()) >= int(args.forecast_year) + offset:
        raise ValueError('The analogue forecast target year must follow its training archive')
    return sample


def preload_analog_sources(model, force=False):
    """Load common raw fields once; never share fitted features across methods."""
    if not force:
        return model._get_raw_sources()
    reanalysis = model.download_reanalysis(force_download=True)
    hindcast, forecast = model.download_models(force_download=True)
    raw = tuple(model._normalize_time_axis_for_dt(value)
                for value in (reanalysis, hindcast, forecast))
    model._analog_raw_cache = raw
    return raw


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def analog_stage_signature(args, workdir, variable="PRCP"):
    root = Path(__file__).resolve().parents[1]
    keys = ('init_month', 'lead_time', 'forecast_year', 'season', 'obs_years', 'hindcast_years',
            'hindcast_initialization_years', 'clim_years', 'tercile_reference', 'probability_calibration',
            'probability_inner_folds', 'probability_inner_retune', 'dist_method')
    settings = {key: value for key, value in vars(args).items() if key.startswith('analog_') or key in keys}
    sources = {str(path.relative_to(root)): _sha(path) for path in [
        Path(__file__).resolve(), root / 'was_analog.py', root / 'was_cross_validate.py',
        root / 'was_analog_probability.py', root / 'was_probability_reference.py',
        root / 'was_crossfit.py', root / 'cli/runtime.py']}
    sources['predictand'] = _sha(Path(workdir) / ('rainfall_predictand.nc' if variable == 'PRCP' else 'temp_predictand.nc'))
    payload = dict(schema=ANALOG_SCHEMA, variable=variable, settings=settings, sources=sources)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def analog_checkpoint_matches(workdir, signature):
    root = Path(workdir)
    try:
        record = json.loads((root / 'checkpoints/chk_analog.json').read_text())
    except (OSError, ValueError):
        return False
    files = ['analog_hindcasts.nc', 'analog_forecasts.nc', 'scores_consolidated_analog.pkl']
    return (record.get('signature') == signature and
            all((root / 'intermediate' / name).is_file() for name in files))
