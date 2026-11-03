"""Optional TelNet stage shared by the precipitation and temperature CLIs.

TelNet consumes monthly observations and monthly scalar covariates. A file
containing only annual seasonal observations cannot reconstruct these inputs.
The stage is opt-in and retains the six-lead predictive distribution separately
from the selected season that enters WASS2S verification/consolidation.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

SCHEMA = 'update13-telnet-2026-10-26-1'


def add_telnet_options(parser):
    parser.add_argument('--include-telnet', action='store_true', default=False,
                        help='Include optional TelNet when --only all is requested; --only telnet selects it directly.')
    parser.add_argument('--telnet-monthly-observations', default=None,
                        help='NetCDF monthly totals (PRCP) or monthly means (TEMP), dimensions T,Y,X. Seasonal-only data are insufficient.')
    parser.add_argument('--telnet-covariates', default=None,
                        help='NetCDF monthly climate indices: Dataset of scalar T variables, or DataArray T,feature.')
    parser.add_argument('--telnet-observation-variable', default=None,
                        help='Variable name if the monthly observations Dataset contains several fields.')
    parser.add_argument('--telnet-covariate-names', nargs='+', default=None,
                        help='Prespecified climate indices; no full-period correlation/VIF feature selection.')
    parser.add_argument('--telnet-history-length', type=int, default=4)
    parser.add_argument('--telnet-forecast-length', type=int, default=6)
    parser.add_argument('--telnet-hidden-dim', type=int, default=32)
    parser.add_argument('--telnet-members', type=int, default=20)
    parser.add_argument('--telnet-epochs', type=int, default=40)
    parser.add_argument('--telnet-batch-size', type=int, default=8)
    parser.add_argument('--telnet-patience', type=int, default=6)
    parser.add_argument('--telnet-learning-rate', type=float, default=1e-3)
    parser.add_argument('--telnet-weight-decay', type=float, default=1e-4)
    parser.add_argument('--telnet-dropout', type=float, default=0.1)
    parser.add_argument('--telnet-seed', type=int, default=42)
    parser.add_argument('--telnet-device', choices=['cpu', 'cuda', 'auto'], default='cpu')
    parser.add_argument('--telnet-threads', type=int, default=1)
    parser.add_argument('--telnet-observation-lag-months', type=int, default=0,
                        help='Additional full months of observation latency beyond the previous completed month.')
    parser.add_argument('--telnet-min-samples', type=int, default=6)
    parser.add_argument('--telnet-max-grid-cells', type=int, default=20000,
                        help='Memory guard for the dense empirical-distribution head; use a coarser common observation grid for large domains.')


def requested(args):
    stages = set(getattr(args, 'only', ['all']))
    return 'telnet' in stages or ('all' in stages and bool(getattr(args, 'include_telnet', False)))


def validate_telnet_options(args):
    for name in ('history_length', 'forecast_length', 'hidden_dim', 'members', 'epochs',
                 'batch_size', 'patience', 'threads', 'min_samples', 'max_grid_cells'):
        if int(getattr(args, 'telnet_' + name)) < 1:
            raise ValueError('--telnet-' + name.replace('_', '-') + ' must be positive')
    if args.telnet_members < 2 or args.telnet_min_samples < 3:
        raise ValueError('TelNet requires at least two members and three training samples')
    if args.telnet_observation_lag_months < 0:
        raise ValueError('--telnet-observation-lag-months must be nonnegative')
    if not 0 <= args.telnet_dropout < 1:
        raise ValueError('--telnet-dropout must lie in [0,1)')
    if not np.isfinite(args.telnet_learning_rate) or args.telnet_learning_rate <= 0:
        raise ValueError('--telnet-learning-rate must be finite and positive')
    if not np.isfinite(args.telnet_weight_decay) or args.telnet_weight_decay < 0:
        raise ValueError('--telnet-weight-decay must be finite and nonnegative')
    if requested(args):
        for name in ('monthly_observations', 'covariates'):
            value = getattr(args, 'telnet_' + name)
            if not value:
                raise ValueError('--telnet-' + name.replace('_', '-') + ' is required for the optional TelNet stage')
            if not Path(value).is_file():
                raise ValueError('TelNet input does not exist: ' + str(value))
    return args


def settings(args, variable):
    result = {name: getattr(args, 'telnet_' + name) for name in (
        'history_length', 'forecast_length', 'hidden_dim', 'epochs', 'batch_size',
        'patience', 'learning_rate', 'weight_decay', 'dropout', 'seed', 'device',
        'observation_lag_months', 'min_samples', 'max_grid_cells')}
    result.update(n_members=args.telnet_members, num_threads=args.telnet_threads,
                  variable=variable, aggregation='sum' if variable == 'PRCP' else 'mean')
    return result


def _read_observations(path, name, variable):
    import xarray as xr

    ds = xr.load_dataset(path)
    if name is not None:
        if name not in ds:
            raise ValueError(f'TelNet observation variable {name!r} is absent from {path}')
        da = ds[name]
    elif variable in ds:
        da = ds[variable]
    elif len(ds.data_vars) == 1:
        da = ds[next(iter(ds.data_vars))]
    else:
        raise ValueError('Specify --telnet-observation-variable for a multi-variable observations file')
    rename = {old: new for old, new in (('time', 'T'), ('latitude', 'Y'), ('lat', 'Y'),
                                       ('longitude', 'X'), ('lon', 'X')) if old in da.dims}
    da = da.rename(rename)
    if set(da.dims) != {'T', 'Y', 'X'}:
        raise ValueError('TelNet monthly observations must have exactly T,Y,X dimensions')
    units = str(da.attrs.get('units', '')).lower().replace(' ', '')
    if variable == 'PRCP' and any(token in units for token in ('/day', 'd-1', '/s', 's-1', '/hour', 'h-1')):
        raise ValueError('TelNet requires monthly precipitation totals in mm; convert the declared rate to monthly amounts first')
    return da.transpose('T', 'Y', 'X')


def load_inputs(args, variable):
    import xarray as xr

    obs = _read_observations(args.telnet_monthly_observations,
                             args.telnet_observation_variable, variable)
    ds = xr.load_dataset(args.telnet_covariates)
    if 'time' in ds.dims:
        ds = ds.rename(time='T')
    if len(ds.data_vars) == 1:
        da = ds[next(iter(ds.data_vars))]
        feature = next((d for d in ('feature', 'features', 'index', 'indices') if d in da.dims), None)
        if feature is not None and set(da.dims) == {'T', feature}:
            cov = da.rename({feature: 'feature'}) if feature != 'feature' else da
            if args.telnet_covariate_names:
                cov = cov.sel(feature=args.telnet_covariate_names)
            return obs, cov.transpose('T', 'feature')
    names = args.telnet_covariate_names or list(ds.data_vars)
    if not names or any(name not in ds for name in names):
        raise ValueError('Requested TelNet climate indices are absent or empty')
    if any(ds[name].dims != ('T',) for name in names):
        raise ValueError('TelNet covariates must be monthly scalar indices, not gridded future observations')
    cov = ds[names].to_array('feature').transpose('T', 'feature')
    return obs, cov


def _hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def signature(args, variable):
    root = Path(__file__).resolve().parents[1]
    files = [Path(__file__), root / 'was_telnet.py', root / 'was_telnet_network.py',
             root / 'was_telnet_data.py', root / 'was_probability_reference.py']
    payload = {'schema': SCHEMA, 'variable': variable, 'settings': settings(args, variable),
        'calendar': {key: getattr(args, key, None) for key in ('init_month', 'forecast_year',
            'season', 'season_months', 'hindcast_years', 'clim_years', 'probability_calibration', 'cv_gap_years')},
        'observation_variable': args.telnet_observation_variable,
        'covariate_names': args.telnet_covariate_names,
        'inputs': [_hash(args.telnet_monthly_observations), _hash(args.telnet_covariates)],
        'code': {p.name: _hash(p) for p in files}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def build_model(args, variable):
    from wass2s.was_telnet import WAS_TelNet
    return WAS_TelNet(**settings(args, variable))


def _seasonal_observations(monthly, labels, target_month, variable):
    """Seasonal verification from the same raw source as TelNet's training."""
    import xarray as xr

    dates = pd.DatetimeIndex(monthly['T'].values)
    if dates.has_duplicates:
        raise ValueError('TelNet monthly observation dates must be unique')
    periods = dates.to_period('M')
    if periods.has_duplicates:
        raise ValueError('TelNet expects one observation per calendar month')
    values=[]
    for label in pd.DatetimeIndex(labels):
        offset = (int(target_month) - label.month) % 12
        if offset > 2:
            raise ValueError('Seasonal labels must identify the first or central month of the target season')
        center = (label + pd.DateOffset(months=offset)).to_period('M')
        months = pd.period_range(center-1, center+1, freq='M')
        positions = periods.get_indexer(months)
        if np.any(positions < 0):
            value = xr.full_like(monthly.isel(T=0, drop=True), np.nan)
        else:
            block = monthly.isel(T=positions)
            valid = block.notnull().all('T')
            value = (block.sum('T', skipna=False) if variable == 'PRCP'
                     else block.mean('T', skipna=False)).where(valid)
        values.append(value.expand_dims(T=[label]))
    return xr.concat(values, dim='T').transpose('T', 'Y', 'X')


def run_telnet_stage(args, workdir, variable, api):
    """Train/evaluate native TelNet and publish the requested seasonal member."""
    import xarray as xr

    validate_telnet_options(args)
    root=Path(workdir)
    for name in ('intermediate', 'scores', 'forecasts', 'checkpoints', 'models'):
        (root/name).mkdir(parents=True, exist_ok=True)
    token=signature(args,variable)
    checkpoint=root/'checkpoints/chk_telnet.json'
    products=[root/'intermediate/telnet_hindcasts.nc',root/'intermediate/telnet_forecasts.nc',
              root/'intermediate/scores_consolidated_telnet.pkl',root/'forecasts/TelNet_six_seasons.nc',
              root/'models/telnet_model.pt']
    try: saved=json.loads(checkpoint.read_text())
    except (OSError, ValueError): saved={}
    if saved.get('signature')==token and all(p.is_file() for p in products) and not args.redo and not args.force:
        print('[telnet] matching checkpoint exists → skip'); return
    checkpoint.unlink(missing_ok=True)
    monthly,cov=load_inputs(args,variable)
    months=[int(m) for m in args.season_months]
    if len(months)!=3 or months!=[months[0],months[0]%12+1,(months[0]+1)%12+1]:
        raise ValueError('TelNet currently requires a three-month consecutive target season')
    center_month=months[1]
    obs_name='rainfall_predictand.nc' if variable=='PRCP' else 'temp_predictand.nc'
    existing=root/obs_name
    if existing.exists():
        reference=xr.load_dataarray(existing)
        xr.align(monthly.isel(T=0,drop=True),reference.isel(T=0,drop=True),join='exact')
        labels=reference.sel(T=slice(str(args.hindcast_years[0]),str(args.hindcast_years[1])))['T'].values
    else:
        labels=pd.to_datetime([f'{year}-{center_month:02d}-01' for year in range(
            int(args.hindcast_years[0]),int(args.hindcast_years[1])+1)])
        all_labels=pd.to_datetime([f'{year}-{center_month:02d}-01' for year in range(
            int(monthly['T'].dt.year.min()),int(monthly['T'].dt.year.max())+1)])
        reference=_seasonal_observations(monthly,all_labels,center_month,variable)
        reference.name=variable
    verification=_seasonal_observations(monthly,labels,center_month,variable)
    if verification.sizes['T']==0:
        raise ValueError('No TelNet target years within --hindcast-years')
    # Combining probability forecasts defined by different observed sources
    # would make their events incompatible. Fail rather than silently mix.
    original=reference.reindex(T=verification['T'])
    compared=np.isfinite(original)&np.isfinite(verification)
    if bool(compared.any()) and not np.allclose(original.where(compared).values,
            verification.where(compared).values,rtol=1e-5,atol=1e-4,equal_nan=True):
        raise ValueError('TelNet monthly observations do not reproduce the current seasonal predictand. '
                         'Use the same observation source/grid/units or a separate workdir.')
    domain=reference.notnull().sum('T')>=3
    monthly=monthly.where(domain); verification=verification.where(domain)
    if not existing.exists(): reference.to_netcdf(existing)
    verification.to_netcdf(root/'scores/telnet_verification_observations.nc')
    model=build_model(args,variable)
    mode=getattr(args,'probability_calibration','outer')
    hd,hp=model.cross_validate(monthly,cov,target_times=verification['T'],
        initialization_month=int(args.init_month),target_month=center_month,
        clim_year_start=args.clim_years[0],clim_year_end=args.clim_years[1],
        obs_for_terciles=reference,gap_years=int(getattr(args,'cv_gap_years',1)),
        probability_calibration=mode)
    if getattr(model,'hindcast_dataset_',None) is not None and 'observation' in model.hindcast_dataset_:
        verification=model.hindcast_dataset_['observation'].transpose('T','Y','X')
    forecast_year=int(args.forecast_year)
    issue=pd.Timestamp(forecast_year,int(args.init_month),1)
    offset=(int(args.init_month)-1+int(args.lead_time[1]))//12
    # Existing forecast products label the same central-season year; retain
    # the observation month/day convention when it belongs to that year.
    first=pd.Timestamp(verification['T'].values[0])
    center=pd.Timestamp(forecast_year+offset,center_month,1)
    label_offset=(center_month-first.month)%12
    target=(center-pd.DateOffset(months=label_offset)).replace(day=first.day)
    fd,fp=model.forecast(monthly,cov,issue_time=issue,target_time=target,target_month=center_month,
        clim_year_start=args.clim_years[0],clim_year_end=args.clim_years[1],
        obs_for_terciles=reference,probability_calibration=mode)
    model.save(products[4])
    api['_cca_invalidate_consolidation'](root)
    key=f'TelNet.{variable}'; scores={}
    verifier=api['WAS_Verification'](dist_method=args.dist_method)
    api['evaluate_and_plot_forecasts'](verifier,verification,hd,hp,fp,'TelNet',key,
        args.clim_years[0],args.clim_years[1],scores,str(root/'scores'),str(root/'forecasts'),
        api['season_to_str'](args.season_months),target.year,args.init_month,api['FCST_LABELS_FR'],
        args.country_code,args.logo)
    xr.Dataset({f'hdcst_det_{key}':hd,f'hdcst_prob_{key}':hp}).pipe(api['strip_attrs']).to_netcdf(products[0])
    xr.Dataset({f'fcst_det_{key}':fd,f'fcst_prob_{key}':fp}).pipe(api['strip_attrs']).to_netcdf(products[1])
    api['save_scores'](root,scores,'scores_consolidated_telnet.pkl')
    if getattr(model,'forecast_dataset_',None) is not None:
        model.forecast_dataset_.pipe(api['strip_attrs']).to_netcdf(products[3])
    else:
        raise RuntimeError('TelNet did not retain its six-lead forecast dataset')
    if getattr(model,'hindcast_dataset_',None) is not None:
        model.hindcast_dataset_.pipe(api['strip_attrs']).to_netcdf(root/'scores/telnet_six_season_hindcasts.nc')
    checkpoint.write_text(json.dumps({'schema':SCHEMA,'signature':token,'model':key,
        'probability_method':'native empirical ensemble','outputs':[str(p.relative_to(root)) for p in products]},indent=2))
    print('[telnet] complete; native ensemble probabilities, no residual calibration subfolds')
