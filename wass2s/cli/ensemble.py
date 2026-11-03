"""Shared ensemble-input options for the PRCP/TEMP MME stage.

The statistical channels are calculated by ``process_datasets_for_mme``. This
module only routes CLI settings, prepares deterministic GCM screening fields,
and prevents reuse of predictions produced with different input channels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


from wass2s.was_ensemble_features import (
    DEFAULT_ENSEMBLE_FEATURES, ensemble_summary_fields, member_dimension,
    normalise_ensemble_features, normalise_ensemble_mean,
)


def _ensemble_mode(value):
    try:
        return normalise_ensemble_mean(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _nonnegative(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("Expected a non-negative integer.")
    return value


def _positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return value


def add_ensemble_arguments(parser):
    parser.add_argument(
        '--ensemble-mean', type=_ensemble_mode, default='mean', metavar='mean|None',
        help="SV-ML-CMME: mean = un canal par GCM; None = statistiques des membres par GCM. "
             "L'ancienne option median reste disponible. Les autres étapes gardent leur agrégation physique.")
    parser.add_argument(
        '--ensemble-features', nargs='+', default=list(DEFAULT_ENSEMBLE_FEATURES),
        help="Canaux par GCM lorsque --ensemble-mean None: mean std q10 q50 q90 par défaut; "
             "aussi median, min, max, iqr, count et q0..q100.")
    parser.add_argument('--ensemble-std-ddof', type=_nonnegative, default=0,
                        help="Diviseur de la variance: nombre de membres valides moins ddof (défaut 0).")
    parser.add_argument('--ensemble-min-members', type=_positive, default=2,
                        help="Nombre minimal de membres valides par pixel pour les statistiques (défaut 2).")


def ensemble_preparation_kwargs(args):
    """Return exactly the shared preparation API, validated before downloads."""
    mode = normalise_ensemble_mean(getattr(args, 'ensemble_mean', 'mean'))
    features = normalise_ensemble_features(
        getattr(args, 'ensemble_features', DEFAULT_ENSEMBLE_FEATURES))
    ddof = int(getattr(args, 'ensemble_std_ddof', 0))
    minimum = int(getattr(args, 'ensemble_min_members', 2))
    if ddof < 0 or minimum < 1:
        raise ValueError('ensemble_std_ddof must be >= 0 and ensemble_min_members >= 1.')
    return dict(ensemble_mean=mode, ensemble_features=features,
                ensemble_std_ddof=ddof, ensemble_min_members=minimum)


def physical_ensemble_reduction(args):
    """Keep non-ML stages deterministic when MME distribution features are used."""
    mode = normalise_ensemble_mean(getattr(args, 'ensemble_mean', 'mean'))
    return 'mean' if mode is None else mode


def physical_reader_directory(args, directory, hindcast_files, forecast_files):
    """Directory in which the legacy CCA reader finds the downloaded archives.

    CCA reconstructs filenames from a directory (``retrieve_single_zone_for_PCR``),
    whereas the downloads may live elsewhere: under ``ensemble_median`` for the
    median reduction, or in the campaign's shared model cache when the run goes
    through ``execution_simple.py`` / ``lancer.sh`` (``cached_download`` stores
    every request under ``<cache-root>/<hash>/`` and returns those paths). When
    every returned file already sits in ``directory`` it is used as is;
    otherwise a private symlink view ``cca_reader_<mode>`` reunites the actual
    paths under their own basenames. Nothing in ``directory`` is replaced.
    """
    directory = Path(directory)
    mode = physical_ensemble_reduction(args)
    sources = {}
    for files in (hindcast_files, forecast_files):
        for source in (files or {}).values():
            if not source:
                continue
            path = Path(source).resolve(strict=True)
            previous = sources.setdefault(path.name, path)
            if previous != path:
                raise ValueError(f'Conflicting {mode} archive filenames: {path.name}.')
    if mode != 'median' and all(p.parent == directory.resolve() for p in sources.values()):
        return directory
    view = directory / f'cca_reader_{mode}'
    view.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        destination = view / name
        if destination.is_symlink():
            if destination.resolve() == source:
                continue
            destination.unlink()
        elif destination.exists():
            raise ValueError(f'Expected a managed symlink in the CCA reader directory: {destination}')
        destination.symlink_to(source)
    return view


def ensemble_input_specification(args):
    values = ensemble_preparation_kwargs(args)
    mode = values['ensemble_mean']
    if mode is not None:
        return {'ensemble_mean': mode, 'ensemble_features': [mode]}
    values['ensemble_features'] = list(values['ensemble_features'])
    return values


def ensure_ensemble_configuration(workdir, args):
    """Do not overwrite or skip old outputs when the caller reuses a workdir.

    The simple launcher gives each configuration its own directory. Direct CLI
    callers get a clear error on incompatible reuse, including old mean-only
    checkpoints without this manifest.
    """
    workdir = Path(workdir)
    path = workdir / 'scores' / 'sv_ml_cmme_input_configuration.json'
    spec = ensemble_input_specification(args)
    if path.exists():
        previous = json.loads(path.read_text(encoding='utf-8'))
    else:
        old_results = list((workdir / 'intermediate').glob('sv_ml_cmme_*_hindcasts.nc'))
        old_checkpoints = list((workdir / 'checkpoints').glob('*sv_ml_cmme*'))
        previous = {'ensemble_mean': 'mean', 'ensemble_features': ['mean']} if old_results or old_checkpoints else spec
    if previous != spec:
        raise ValueError(
            'Ce workdir contient une autre configuration des entrées MME. '
            'Utiliser un --workdir distinct pour comparer mean et None ou modifier les caractéristiques; '
            'lancer.sh crée automatiquement ces dossiers distincts.')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return spec


def mean_screening_inputs(files, directory):
    """Return NetCDF paths with one ensemble-mean field for each raw GCM.

    The verification API accepts paths and historically recognizes only
    ``number``. Normalizing here also covers M/member/realization archives.
    Raw paths remain untouched and continue to feed the feature preparation.
    """
    import xarray as xr

    result = {}
    directory = Path(directory)
    for model, source in files.items():
        with xr.open_dataset(source) as dataset:
            candidates = [da for da in dataset.data_vars.values()
                          if {'T', 'Y', 'X'}.issubset(da.dims)]
            if len(candidates) != 1:
                raise ValueError(f'{model}: expected one GCM variable with T,Y,X; got {len(candidates)}.')
            field = candidates[0]
            if member_dimension(field) is None:
                result[model] = source
                continue
            mean = ensemble_summary_fields(field, ensemble_mean='mean')['mean'].load()
        directory.mkdir(parents=True, exist_ok=True)
        token = hashlib.sha256(str(model).encode('utf-8')).hexdigest()[:16]
        destination = directory / f'{token}_ensemble_mean.nc'
        mean.to_netcdf(destination)
        result[model] = str(destination)
    return result
