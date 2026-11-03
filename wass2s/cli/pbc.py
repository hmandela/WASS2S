"""Optional seasonal probabilistic bias correction shared by PRCP/TEMP CLIs.

This postprocessing stage consumes existing seasonal probability forecasts. It
never fits the originating prediction model and never republishes corrected
members in intermediate/, where they could be counted twice by consolidation.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import re

import numpy as np

SCHEMA = 'update14-seasonal-pbc-sliding-2026-10-29-2'


def add_pbc_options(parser):
    parser.add_argument('--include-pbc', action='store_true', default=False,
        help='Run optional PBC after the other stages with --only all; --only pbc uses existing products.')
    parser.add_argument('--pbc-models', nargs='+', default=None,
        help='Exact prespecified model IDs to correct; default: every available paired probability product, without skill ranking.')
    parser.add_argument('--pbc-monthly-observations', default=None,
        help='Monthly PRCP totals (mm) or TEMP means on the seasonal predictand grid, needed for the persistence branch.')
    parser.add_argument('--pbc-observation-variable', default=None)
    parser.add_argument('--pbc-temperature-aggregation', choices=['day_weighted_mean', 'mean'],
        default='day_weighted_mean', help='Monthly TEMP to season: day weighting matches the standard WASS2S predictand; mean supports explicitly arithmetic products.')
    parser.add_argument('--pbc-alpha', type=float, default=1.0,
        help='Fixed regularization of the seasonal Persistence++ regression; 0 enables OLS. No hyperparameter subfolds by default.')
    parser.add_argument('--pbc-min-samples', type=int, default=8,
        help='Minimum available historical seasons per cell before calibration/scoring.')
    parser.add_argument('--pbc-years', type=int, default=20,
        help='Maximum historical calendar years for each chronological correction.')
    parser.add_argument('--pbc-observation-lag-months', type=int, default=0,
        help='Additional full months of publication latency; applied at every historical issuance.')
    parser.add_argument('--pbc-persistence-lags', type=int, default=4,
        help='Number of preceding observed seasonal windows for Persistence++; default 4. These are observed fields, not future GCM forecasts.')
    parser.add_argument('--pbc-persistence-step-months', type=int, default=1,
        help='Months between consecutive observed persistence windows; 4 lags and step 1 give FMA,JFM,DJF,NDJ for May issuance. Legacy two disjoint trimesters: 2 lags, step 3.')
    parser.add_argument('--pbc-no-persistence', dest='pbc_persistence', action='store_false', default=True,
        help='Explicit Debias++-only variant; requires no monthly observations. This is not the full two-branch PBC.')
    parser.add_argument('--pbc-base-provenance', choices=['retrospective_oof', 'asof_archive'],
        default='retrospective_oof',
        help='retrospective_oof gives conditional postprocessor validation; asof_archive requires verified base_training_end dates.')


def requested(args):
    stages = set(getattr(args, 'only', ['all']))
    return 'pbc' in stages or ('all' in stages and bool(getattr(args, 'include_pbc', False)))


def validate_pbc_options(args):
    if not np.isfinite(args.pbc_alpha) or args.pbc_alpha < 0:
        raise ValueError('--pbc-alpha must be finite and nonnegative')
    if args.pbc_min_samples < 3:
        raise ValueError('--pbc-min-samples must be at least 3')
    if args.pbc_years < args.pbc_min_samples:
        raise ValueError('--pbc-years must be at least --pbc-min-samples')
    if args.pbc_observation_lag_months < 0:
        raise ValueError('--pbc-observation-lag-months must be nonnegative')
    for name in ('persistence_lags', 'persistence_step_months'):
        value = getattr(args, 'pbc_' + name)
        if isinstance(value, (bool, np.bool_)) or int(value) != value or value < 1:
            raise ValueError('--pbc-' + name.replace('_', '-') + ' must be a positive integer')
    if args.pbc_models and len(set(args.pbc_models)) != len(args.pbc_models):
        raise ValueError('--pbc-models contains duplicate model IDs')
    if requested(args) and args.pbc_persistence:
        if not args.pbc_monthly_observations:
            raise ValueError('--pbc-monthly-observations is required for PBC persistence; --pbc-no-persistence explicitly selects Debias++ only')
        if not Path(args.pbc_monthly_observations).is_file():
            raise ValueError('PBC monthly observations file does not exist: ' + str(args.pbc_monthly_observations))
    return args


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _read_monthly(path, variable, name=None):
    # Shared strict monthly-total/rate validation; this module imports no torch.
    from .telnet import _read_observations
    return _read_observations(path, name, variable)


def read_probability_pairs(workdir, hindcast_years):
    """Load paired probabilities with their actual dates, never snap timestamps.

    Base members are read from intermediate/. Consolidators produced by this
    update also retain a paired historical probability product in scores/.
    A duplicated model ID is an error, rather than an implicit overwrite.
    """
    import xarray as xr

    root = Path(workdir)
    records, source_paths = {}, set()

    def register(model, hind, fore, paths):
        if model in records:
            raise ValueError(f'Duplicate probability model ID for PBC: {model!r}')
        if 'T' not in hind.dims or 'T' not in fore.dims:
            raise ValueError(f'PBC probability pair {model!r} must carry its actual T coordinate')
        hind = hind.sel(T=slice(str(hindcast_years[0]), str(hindcast_years[1])))
        if hind.sizes['T'] == 0:
            raise ValueError(f'No hindcast years for PBC model {model!r}')
        records[model] = (hind, fore)
        source_paths.update(paths)

    for hp in sorted((root / 'intermediate').glob('*_hindcasts.nc')):
        fp = hp.with_name(hp.name.replace('_hindcasts.nc', '_forecasts.nc'))
        if not fp.is_file():
            continue
        hd, fd = xr.load_dataset(hp), xr.load_dataset(fp)
        for name in hd.data_vars:
            if not name.startswith('hdcst_prob_'):
                continue
            model = name.removeprefix('hdcst_prob_')
            fname = 'fcst_prob_' + model
            if fname in fd:
                register(model, hd[name], fd[fname], (hp, fp))
    for hp in sorted((root / 'scores').glob('Hindcast_Prob_*.nc')):
        fp = root / 'forecasts' / hp.name.replace('Hindcast_Prob_', 'Forecast_Prob_', 1)
        if not fp.is_file():
            continue
        hd, fd = xr.load_dataarray(hp), xr.load_dataarray(fp)
        model = str(hd.attrs.get('pbc_source_model', hd.name or ''))
        if not model:
            raise ValueError(f'Missing consolidator model ID in {hp.name}')
        register('consolidated.' + model, hd, fd, (hp, fp))
    if not records:
        raise RuntimeError('PBC requires existing paired seasonal probability hindcasts and forecasts; run the selected base stages first')
    return records, sorted(source_paths)


def signature(args, variable, sources, reference):
    root = Path(__file__).resolve().parents[1]
    names = ('alpha', 'min_samples', 'years', 'observation_lag_months', 'persistence',
             'persistence_lags', 'persistence_step_months',
             'models', 'base_provenance', 'observation_variable', 'temperature_aggregation')
    payload = {'schema': SCHEMA, 'variable': variable,
        'settings': {name: getattr(args, 'pbc_' + name) for name in names},
        'calendar': {name: getattr(args, name, None) for name in
                     ('forecast_year', 'init_month', 'season', 'season_months', 'hindcast_years', 'clim_years')},
        'inputs': {str(p): _hash(p) for p in [reference, *sources]},
        'code': {name: _hash(root/name) for name in (
            'cli/pbc.py', 'was_pbc.py', 'was_pbc_data.py', 'was_pbc_cv.py',
            'was_probability_reference.py')}}
    if args.pbc_persistence:
        payload['monthly'] = _hash(args.pbc_monthly_observations)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _weighted_mean(field):
    import xarray as xr

    finite = np.isfinite(field)
    weights = xr.ones_like(field, dtype=float)
    if 'Y' in field.dims:
        weights = weights * np.cos(np.deg2rad(field['Y']))
    denominator = weights.where(finite).sum(skipna=True)
    if float(denominator) <= 0:
        return float('nan')
    return float((field * weights).sum(skipna=True) / denominator)


def score_pair(result, raw, observations):
    """RPS and category Brier scores on exactly the same eligible cases.

    RPS is the mean squared CDF error at the two finite tercile thresholds;
    Brier uses the same PB<=q1 / PN<=q2 / PA>q2 convention as calibration.
    """
    import xarray as xr

    corrected = result['hindcast_probability'].transpose('probability', 'T', 'Y', 'X')
    raw = raw.sel(probability=corrected['probability']).transpose('probability', 'T', 'Y', 'X')
    y = observations.sel(T=corrected['T']).transpose('T', 'Y', 'X')
    xr.align(y, raw, corrected, join='exact')
    thresholds = result['thresholds'].transpose('cdf_threshold', 'Y', 'X')
    event = (y <= thresholds).astype(float).where(np.isfinite(y) & np.isfinite(thresholds))
    event = event.transpose('cdf_threshold', 'T', 'Y', 'X')
    labels = corrected['probability']
    indicator = xr.concat([event.isel(cdf_threshold=0, drop=True),
        event.isel(cdf_threshold=1, drop=True)-event.isel(cdf_threshold=0, drop=True),
        1-event.isel(cdf_threshold=1, drop=True)], dim=labels)
    common = (np.isfinite(raw).all('probability') & np.isfinite(corrected).all('probability')
              & np.isfinite(event).all('cdf_threshold')
              & result.get('score_eligible_cell', result['score_eligible']))
    metrics = {}
    for kind, probabilities in (('raw', raw), ('pbc', corrected)):
        cdf = xr.concat([probabilities.isel(probability=0, drop=True),
            probabilities.isel(probability=slice(0, 2)).sum('probability', skipna=False)],
            dim=thresholds['cdf_threshold'])
        metrics[kind+'_RPS'] = ((cdf-event)**2).mean('cdf_threshold', skipna=False).where(common).mean('T')
        metrics[kind+'_Brier'] = ((probabilities-indicator)**2).where(common).mean('T')
    metrics['paired_count'] = common.sum('T')
    ds = xr.Dataset(metrics)
    ds.attrs.update(RPS_definition='mean squared error of the two finite cumulative tercile probabilities',
        comparison='same common eligible years and cells for raw and corrected',
        validation_scope=str(result.attrs.get('pbc_validation_scope', result.attrs.get('validation_scope', 'conditional_postprocessor_validation'))))
    return ds


def _safe_model_name(model):
    stem = re.sub(r'[^A-Za-z0-9_.-]+', '_', model).strip('._')[:100] or 'model'
    return stem + '_' + hashlib.sha256(model.encode()).hexdigest()[:8]


def run_pbc_stage(args, workdir, variable='PRCP'):
    """Correct existing probability products, then save paired diagnostics."""
    import xarray as xr

    from wass2s.was_pbc_cv import run_pbc_seasonal
    validate_pbc_options(args)
    root = Path(workdir)
    reference = root / ('rainfall_predictand.nc' if variable == 'PRCP' else 'temp_predictand.nc')
    if not reference.is_file():
        raise FileNotFoundError('PBC requires the existing seasonal observation reference: ' + str(reference))
    records, sources = read_probability_pairs(root, args.hindcast_years)
    selected = args.pbc_models or sorted(records)
    unknown = sorted(set(selected)-set(records))
    if unknown:
        raise ValueError('Requested PBC model IDs are absent: ' + ', '.join(unknown))
    token = signature(args, variable, sources, reference)
    out = root / 'pbc'; out.mkdir(parents=True, exist_ok=True)
    checkpoint = root / 'checkpoints' / 'chk_pbc.json'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    try:
        saved = json.loads(checkpoint.read_text())
    except (OSError, ValueError):
        saved = {}
    if (saved.get('signature') == token and saved.get('outputs')
            and all((root/p).is_file() for p in saved['outputs'])
            and not getattr(args, 'redo', False) and not getattr(args, 'force', False)):
        print('[pbc] matching checkpoint exists → skip')
        return
    checkpoint.unlink(missing_ok=True)
    observations = xr.load_dataarray(reference)
    monthly = (_read_monthly(args.pbc_monthly_observations, variable,
               args.pbc_observation_variable) if args.pbc_persistence else None)
    outputs, rows = [], []
    for model in selected:
        hind, fore = records[model]
        result = run_pbc_seasonal(hind, fore, observations,
            monthly_observations=monthly, initialization_month=int(args.init_month),
            season_months=[int(month) for month in args.season_months],
            clim_year_start=int(args.clim_years[0]),
            clim_year_end=int(args.clim_years[1]), forecast_year=int(args.forecast_year),
            aggregation='sum' if variable == 'PRCP' else args.pbc_temperature_aggregation, alpha=float(args.pbc_alpha),
            min_samples=int(args.pbc_min_samples), training_years=int(args.pbc_years),
            observation_lag_months=int(args.pbc_observation_lag_months),
            persistence_lags=int(args.pbc_persistence_lags),
            persistence_step_months=int(args.pbc_persistence_step_months),
            persistence=bool(args.pbc_persistence), base_provenance=args.pbc_base_provenance)
        result.attrs.update(pbc_source_model=model, source_variable=variable,
                            source_signature=token, schema=SCHEMA)
        scores = score_pair(result, hind, observations)
        name = _safe_model_name(model)
        product = out / f'PBC_{name}.nc'; metric_path = out / f'PBC_scores_{name}.nc'
        result.to_netcdf(product); scores.to_netcdf(metric_path)
        outputs += [product, metric_path]
        row = {'model': model, 'raw_RPS': _weighted_mean(scores['raw_RPS']),
               'pbc_RPS': _weighted_mean(scores['pbc_RPS']),
               'paired_years_max': int(scores['paired_count'].max()),
               'validation_scope': str(result.attrs.get('pbc_validation_scope', result.attrs.get('validation_scope', 'conditional_postprocessor_validation'))),
               'base_provenance': args.pbc_base_provenance,
               'persistence_requested': int(args.pbc_persistence)}
        for i, category in enumerate(('PB', 'PN', 'PA')):
            row['raw_Brier_'+category] = _weighted_mean(scores['raw_Brier'].isel(probability=i))
            row['pbc_Brier_'+category] = _weighted_mean(scores['pbc_Brier'].isel(probability=i))
        rows.append(row)
        print(f'[pbc/{model}] paired RPS raw={row["raw_RPS"]:.6g}, corrected={row["pbc_RPS"]:.6g}; {row["validation_scope"]}')
    import csv
    summary = out/'pbc_summary.csv'
    with summary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    outputs.append(summary)
    checkpoint.write_text(json.dumps({'schema': SCHEMA, 'signature': token, 'models': selected,
        'outputs': [str(p.relative_to(root)) for p in outputs],
        'base_provenance': args.pbc_base_provenance,
        'probability_only': True, 'base_model_refits': 0}, indent=2))
    print('[pbc] complete; corrected probabilities and raw comparisons are in pbc/')
