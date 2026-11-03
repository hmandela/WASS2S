"""Small compatibility layer for the original PRCP/TEMP CLI stage layout.

No stage dispatch, default model selection or consolidation is replaced here.
"""
from argparse import BooleanOptionalAction
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .probability import ProbabilityCalibrationAction


def add_optional_arguments(parser):
    group = parser.add_argument_group("U-Net / UNet++ / RISE-UNet / ResCNN / Kharin (optional)")
    group.add_argument('--gcm-models', nargs='+', default=None,
                       help='Optional explicit centres; otherwise retain original catalog/drop list.')
    group.add_argument('--unet-device', choices=['auto', 'cpu', 'cuda'], default='auto')
    group.add_argument('--unet-epochs', type=int, default=40)
    group.add_argument('--unet-base-channels', type=int, default=8)
    group.add_argument('--unet-depth', type=int, default=3)
    group.add_argument('--unet-batch-size', type=int, default=2)
    group.add_argument('--unet-threads', type=int, default=1)
    group.add_argument('--unet-patience', type=int, default=6)
    group.add_argument('--unet-learning-rate', type=float, default=.001)
    group.add_argument('--unet-dropout', type=float, default=.2)
    group.add_argument('--unet-weight-decay', type=float, default=.0001)
    group.add_argument('--unet-deep-supervision', action=BooleanOptionalAction, default=True,
                       help='Auxiliary decoder supervision for UNet++ and RISE-UNet only.')
    group.add_argument('--rise-se-reduction', type=int, default=8,
                       help='Squeeze-excitation reduction ratio for RISE-UNet.')
    group.add_argument('--unet-tuning', choices=['fixed', 'nested_tuned'], default='fixed')
    group.add_argument('--unet-tuning-candidates', type=int, default=8)
    group.add_argument('--unet-tuning-folds', type=int, default=3)
    group.add_argument('--unet-tuning-epochs', type=int, default=20)
    group.add_argument('--unet-tuning-patience', type=int, default=4)
    # Corrections of 15 September 2026 (see CORRECTIFS_2026-09-15.md); shared by
    # every spatial network (unet, unetpp, rise-unet, rescnn).
    group.add_argument('--unet-norm', choices=['none', 'group'], default='none',
                       help='none (default): no spatial normalisation; group reproduces the former networks.')
    group.add_argument('--no-unet-linear-skip', dest='unet_linear_skip', action='store_false', default=True)
    group.add_argument('--no-unet-static-channels', dest='unet_static_channels', action='store_false', default=True)
    group.add_argument('--unet-seeds', type=int, default=3, help='Independently initialised networks averaged.')
    group.add_argument('--unet-no-refit', dest='unet_refit', action='store_false', default=True,
                       help='Keep early-stopped networks instead of refitting on every year (half cost).')
    group.add_argument('--unet-fast-gpu', action='store_true',
                       help='bfloat16/float16 autocast and cudnn.benchmark (not bit-reproducible).')
    group.add_argument('--unet-train-resolution', type=float, default=0.25,
                       help='Training grid in degrees (block means); 0 keeps the observation grid.')
    group.add_argument('--unet-fold-workers', type=int, default=1,
                       help='Outer folds trained in parallel processes (they share the GPU).')
    group.add_argument('--unet-probability-residuals', choices=['inner', 'nested', 'outer_oof'], default=None,
                       action=ProbabilityCalibrationAction,
                       help='Legacy alias of --probability-calibration (outer_oof = outer); '
                            'the chosen policy applies to CCA, residual ML and all CNNs.')
    group.add_argument('--no-unet-checkpoint', dest='unet_checkpoint', action='store_false', default=True)
    group.add_argument('--rescnn-device', choices=['auto', 'cpu', 'cuda'], default='auto')
    group.add_argument('--rescnn-channels', type=int, default=16)
    group.add_argument('--rescnn-blocks', type=int, default=3)
    group.add_argument('--rescnn-dilations', type=int, nargs='+', default=[1, 2, 4],
                       help='One positive dilation per residual block; default: 1 2 4.')
    group.add_argument('--rescnn-epochs', type=int, default=40)
    group.add_argument('--rescnn-batch-size', type=int, default=1)
    group.add_argument('--rescnn-threads', type=int, default=1)
    group.add_argument('--rescnn-patience', type=int, default=6)
    group.add_argument('--rescnn-learning-rate', type=float, default=.001)
    group.add_argument('--rescnn-dropout', type=float, default=.2)
    group.add_argument('--rescnn-weight-decay', type=float, default=.0001)
    group.add_argument('--rescnn-tuning', choices=['fixed', 'nested_tuned'], default='fixed')
    group.add_argument('--rescnn-tuning-candidates', type=int, default=8)
    group.add_argument('--rescnn-tuning-folds', type=int, default=3)
    group.add_argument('--rescnn-tuning-epochs', type=int, default=20)
    group.add_argument('--rescnn-tuning-patience', type=int, default=4)
    group.add_argument('--optional-cv-gap-years', type=int, default=1,
                       help='Calendar gap for U-Net/ResCNN/Kharin; 1 matches the original nb_omit=2.')
    group.add_argument('--kharin-target-seasons', nargs=7, default=None)
    group.add_argument('--kharin-observations-dir', default=None)
    group.add_argument('--kharin-temporal-smoothing', default='1',
                       help="Gaussian sigma in months, 'U' (none) or 'C' (pooled constant).")
    group.add_argument('--kharin-distribution', choices=['auto', 'gamma', 'normal'], default='auto')
    group.add_argument('--kharin-min-samples', type=int, default=8)
    group.add_argument('--kharin-prob-grid-size', type=int, default=121)
    group.add_argument('--kharin-spatial-chunk', type=int, default=32)
    group.add_argument('--kharin-spatial-sigma-deg', type=float, default=0.5,
                       help='Gaussian smoothing of Kharin coefficients in degrees; 0 disables it.')
    group.add_argument('--kharin-combination', choices=['pooled', 'mixture'], default='pooled',
                       help='pooled: one calibration of the equal-weight multi-model distribution; '
                            'mixture: former average of separately calibrated GCMs.')
    group.add_argument('--kharin-workers', type=int, default=0,
                       help='Parallel tile processes; 0 uses the Slurm/affinity CPU count.')
    group.add_argument('--kharin-nmme-source', choices=['cpt', 'ccsr', 'netcdf'], default='cpt',
                       help="NMME route used ONLY for Kharin's own member acquisition of "
                            'CFSV2_1/CMC1_1/CMC2_1/NASA_1/GFDL_1/NCAR_CCSM4_1/NCAR_CESM1_1 '
                            '(other stages keep --gcm-models/the CPT archives). "cpt" (default) '
                            'preserves the previous behaviour: WAS_Download_Models exports T,Y,X '
                            'ensemble means only for these centres, so Kharin excludes them unless '
                            'they are replaced by a C3S roster. "ccsr" fetches true per-member '
                            'T,M,Y,X archives for the same seven centres from '
                            'forecast.ccsr.columbia.edu (LIRE_CCSR.md), so they become usable by '
                            'Kharin; NMME_1 stays mean-only even under ccsr (Columbia does not '
                            'expose it as a raw per-member ensemble). "netcdf" is the other legacy '
                            'WAS_Download_Models route and is mean-only for these centres like cpt.')
    group.add_argument('--kharin-ccsr-engine', choices=['netcdf4', 'pydap'], default='netcdf4',
                       help='OPeNDAP client for --kharin-nmme-source ccsr; pydap needs '
                            "'pip install pydap' in the WASS2S environment.")
    group.add_argument('--kharin-ccsr-cfsv2-sample', choices=['pentad_samples', 'early_month_samples'],
                       default='pentad_samples',
                       help='CFSv2 sampling for --kharin-nmme-source ccsr; keep pentad_samples '
                            '(early_month_samples hindcast was not exposed in the inspected catalogue).')
    group.add_argument('--no-kharin-checkpoint', dest='kharin_checkpoint', action='store_false', default=True)


def requested_centres(centres, args):
    from .roster import canonical_center
    excluded = getattr(args, 'drop_gcm', []) or []
    # A qualified exclusion is variable-specific; an unqualified ID is global.
    centres = [c for c in centres if not any(
        (str(e).upper() == str(c).upper()) if '.' in str(e)
        else canonical_center(e) == canonical_center(c) for e in excluded)]
    requested = getattr(args, 'gcm_models', None)
    if requested:
        wanted = {canonical_center(c) for c in requested}
        centres = [c for c in centres if canonical_center(c) in wanted]
    if not centres:
        raise ValueError('Aucun GCM après --gcm-models/--drop-gcm pour ce champ.')
    return centres


def available_inputs(hind, future, centres, report_path):
    from .roster import reconcile_download_roster, canonical_center
    selected = reconcile_download_roster(hind, future, centres, 'GCM', report_path=report_path)
    # Preserve one stable order and common spelling for downstream original code.
    hi = {canonical_center(k): (k, v) for k, v in hind.items()}
    fi = {canonical_center(k): v for k, v in future.items()}
    hind = {hi[canonical_center(c)][0]: hi[canonical_center(c)][1] for c in selected}
    future = {hi[canonical_center(c)][0]: fi[canonical_center(c)] for c in selected}
    return hind, future, selected


def mask_products(products, obs):
    """Reapply the original observed support to outputs, including U-Net oceans."""
    if 'M' in obs.dims:
        if obs.sizes['M'] != 1:
            raise ValueError('Expected one observed field')
        obs = obs.isel(M=0, drop=True)
    support = np.isfinite(obs).any('T')
    return tuple(p.where(support) if p is not None else None for p in products)


def _season_observations(data, season):
    """Map start/centre/end season labels by calendar, never by row position."""
    from .kharin import CENTERS
    if 'M' in data.dims:
        if data.sizes['M'] != 1:
            raise ValueError('Expected one observed field')
        data = data.isel(M=0, drop=True)
    centre = CENTERS[season]
    dates = pd.DatetimeIndex(data['T'].values)
    allowed = {(centre - 2) % 12 + 1, centre, centre % 12 + 1}
    if not set(dates.month).issubset(allowed):
        raise ValueError(f'{season}: observation timestamps must identify a month of the actual season')
    mapped = []
    for stamp in dates:
        offset = (centre - stamp.month + 6) % 12 - 6
        mapped.append((stamp + pd.DateOffset(months=offset)).to_period('M').to_timestamp())
    if pd.DatetimeIndex(mapped).has_duplicates:
        raise ValueError('Duplicate seasonal observation years')
    return data.assign_coords(T=mapped).transpose('T', 'Y', 'X')


def run_optional(name, args, workdir, downloader, obs, hind, future, best_models, api):
    import xarray as xr

    local = deepcopy(args)
    local.cv_gap_years = int(args.optional_cv_gap_years)
    if local.cv_gap_years < 0:
        raise ValueError('--optional-cv-gap-years must be nonnegative')
    variable = api.get('VARIABLE')
    if variable is None:
        # Backward compatibility for callers supplying only a historical CLI path.
        variable = 'PRCP' if 'PRCP' in Path(api['__file__']).stem.upper() else 'TEMP'
    if variable not in {'PRCP', 'TEMP'}:
        raise ValueError(f'Unsupported forecast variable: {variable!r}')
    if name in {'unet', 'unetpp', 'rise-unet', 'rescnn'}:
        local.cnn_upstream_selection = {
            'method': 'fixed configured/available GCM roster (independent of target scores) in original CLI',
            'models': [str(model) for model in best_models],
            'validation_scope': ('CNN calibration conditional on supplied channels; '
                                 'upstream selection is not cross-fitted'),
        }
    if name == 'unet':
        from .unet import run_unet
        products = run_unet(local, obs, hind, future.isel(T=[-1]), variable, workdir=workdir)
    elif name in {'unetpp', 'rise-unet'}:
        from .unet_variants import run_unet_variant
        products = run_unet_variant(local, obs, hind, future.isel(T=[-1]), variable,
                                    workdir=workdir, architecture=name)
    elif name == 'rescnn':
        from .rescnn import run_rescnn
        products = run_rescnn(local, obs, hind, future.isel(T=[-1]), variable, workdir=workdir)
    elif name == 'kharin':
        from .kharin import assemble_kharin_data, evaluate_kharin, member_capable_roster, _core
        kharin_nmme_source = str(getattr(args, 'kharin_nmme_source', 'cpt')).strip().lower()
        members = member_capable_roster(best_models, explicit=False, nmme_source=kharin_nmme_source)
        print(f'[kharin] selected member-capable GCMs: {members} (nmme_source={kharin_nmme_source})', flush=True)
        # Kharin supplies its native predictive CDF; the original bestfit setting
        # remains unchanged for every other method.
        local.dist_method = 'nonparam'
        target = _season_observations(obs, args.season)
        # The MME response is restricted to hindcast years. Keep the full
        # original predictand separately for the declared climatological events.
        filename = 'rainfall_predictand.nc' if variable == 'PRCP' else 'temp_predictand.nc'
        target_path = Path(workdir) / filename
        full_target = (_season_observations(xr.load_dataarray(target_path), args.season)
                       if target_path.is_file() else target)
        xr.testing.assert_allclose(_core(full_target.sel(T=target['T'])), _core(target))
        local.hindcast_years = [int(target['T'].dt.year.min()), int(target['T'].dt.year.max())]
        support = np.isfinite(target).any('T')
        full_target = full_target.where(support)

        def load_observations(season_args, directory):
            if season_args.season == args.season:
                return full_target
            directory = Path(directory)
            for folder in ('intermediate', 'scores', 'forecasts', 'checkpoints', 'Observation'):
                (directory / folder).mkdir(parents=True, exist_ok=True)
            season_args.merge_date = f'{int(season_args.season_months[1]):02d}-01'
            # All calibration seasons use the TARGET support, not seven separate
            # dry masks that would discard useful dry-season calibration values.
            season_args._kharin_calibration = True
            api['stage_obs'](season_args, directory, downloader)
            filename = 'rainfall_predictand.nc' if variable == 'PRCP' else 'temp_predictand.nc'
            data = xr.load_dataarray(directory / filename)
            return _season_observations(data, season_args.season).where(support)

        observations, historical, operational, event_reference = assemble_kharin_data(
            local, workdir, downloader, variable, members, load_observations,
            return_event_reference=True)
        # Explicit observation directories must share the original target mask.
        observations = {key: value.where(support) for key, value in observations.items()}
        xr.testing.assert_allclose(observations[args.season], target)
        if (int(full_target['T'].dt.year.min()) <= int(local.clim_years[0]) and
                int(full_target['T'].dt.year.max()) >= int(local.clim_years[1])):
            event_reference = full_target
        event_reference = event_reference.where(support)
        products = evaluate_kharin(local, observations, historical, operational, variable,
                                   checkpoint_dir=Path(workdir) / 'kharin' / 'tiles',
                                   event_reference=event_reference)
        # Keep the original output labels expected by the original consolidation.
        if not np.array_equal(products[0]['T'].values, target['T'].values):
            raise ValueError('Kharin hindcast target calendar mismatch')
        hd, hp, fd, fp = products
        hd, hp = hd.assign_coords(T=obs['T']), hp.assign_coords(T=obs['T'])
        if fd.sizes['T'] != 1 or future.sizes['T'] != 1:
            raise ValueError('Expected one operational season')
        actual = str(fd['T'].values[0])
        fd, fp = fd.assign_coords(T=future['T']), fp.assign_coords(T=future['T'])
        products = hd, hp, fd, fp
        for p in products:
            p.attrs.update(selected_gcms=','.join(historical), kharin_actual_target=actual,
                           probability_source='native Kharin CDF')
    else:
        raise ValueError(name)
    return mask_products(products, obs)
