"""Seven-season Kharin option for the operational PRCP/TEMP CLIs.

Corrections of 15 September 2026: zero-inflated gamma transform, temporal
smoothing sigma=1 month by default, Gaussian spatial smoothing of the
coefficients (exact across tiles thanks to a halo), one calibration of the
equal-weight multi-model distribution (``pooled``, default), an alternative
mixture of separately calibrated GCMs, parallel tiles and tile checkpoints.
The two combination modes require empirical skill comparisons; neither is
assumed universally superior.

The target window moves the initialization while keeping the three WASS lead
*offsets* fixed. True ensemble members are retained. Each GCM is calibrated
as configured and the declared GCMs receive equal weights. Probabilities evaluate
the calibrated Kharin CDF at the same fixed climatological events as the other
CLI methods; they are never re-dressed using complete-archive residuals.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import ndtri

from .runtime import _attach, atomic_json, deterministic_mixture, probability_mixture
from .probability import probability_calibration_mode
from wass2s.was_mme_kharin2017_v3_7seasons import _normal_event_cdf

CENTERS = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
           "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}


# WAS_Download_Models defaults to the NMME CPT route (nmme_source="cpt") for
# these source IDs; that route exports T,Y,X ensemble means even with
# ensemble_mean=None. The CCSR route (nmme_source="ccsr", LIRE_CCSR.md)
# re-fetches the same seven centres with true per-member T,M,Y,X archives;
# only NMME_1 stays mean-only there, since Columbia does not publish it as a
# raw per-member ensemble (composite product).
MEAN_ONLY_NMME = frozenset({"CFSV2_1", "CMC1_1", "CMC2_1", "GFDL_1", "NASA_1",
                           "NCAR_CCSM4_1", "NCAR_CESM1_1", "NMME_1"})
CCSR_MEAN_ONLY_NMME = frozenset({"NMME_1"})


def member_capable_roster(roster, *, explicit=True, nmme_source="cpt"):
    """Select known member-capable sources without silently changing choices."""
    nmme_source = str(nmme_source).strip().lower()
    mean_only = CCSR_MEAN_ONLY_NMME if nmme_source == "ccsr" else MEAN_ONLY_NMME
    excluded = [center for center in roster if str(center).upper().split(".")[0] in mean_only]
    if excluded and explicit:
        if nmme_source == "ccsr":
            hint = "Columbia does not publish per-member archives for it (composite product)."
        else:
            hint = ("Pass --kharin-nmme-source ccsr to fetch true per-member archives for "
                    "CFSV2_1/CMC1_1/CMC2_1/NASA_1/GFDL_1/NCAR_CCSM4_1/NCAR_CESM1_1 from CCSR "
                    "(see LIRE_CCSR.md), or select a C3S GCM roster instead.")
        raise ValueError(f"Kharin needs actual ensemble members; the configured NMME source "
                         f"({nmme_source!r}) provides ensemble means only for {excluded}. {hint}")
    selected = [center for center in roster if center not in excluded]
    if not selected:
        raise ValueError("Kharin requires at least one member-capable GCM")
    return selected


def _core(array):
    return array.drop_vars([name for name in array.coords if name not in array.dims])


def kharin_design(args):
    """Return seven fixed-lead requests, with years anchored to the target season.

    For a DJF target, SON/OND/NDJ belong to the preceding calendar year but to
    the *same* validation group as DJF/JFM/FMA/MAM. Actual timestamps remain
    their true calendar dates, independently of this validation group.
    """
    leads = tuple(int(v) for v in args.lead_time)
    if len(leads) != 3 or leads != tuple(range(leads[0], leads[0] + 3)) or leads[0] < 0:
        raise ValueError("Kharin requires three consecutive nonnegative WASS lead offsets")
    selected = str(args.season).upper()
    if selected not in CENTERS:
        raise ValueError("Kharin requires a rolling three-month target season")
    names = getattr(args, "kharin_target_seasons", None)
    if not names:
        inverse = {v: k for k, v in CENTERS.items()}
        names = [inverse[(CENTERS[selected] - 1 + delta) % 12 + 1] for delta in range(-3, 4)]
    names = [str(name).upper() for name in names]
    if len(names) != 7 or len(set(names)) != 7 or selected not in names or any(s not in CENTERS for s in names):
        raise ValueError("Kharin needs seven unique consecutive seasons including the requested target")
    if any(CENTERS[b] != CENTERS[a] % 12 + 1 for a, b in zip(names, names[1:])):
        raise ValueError("Kharin target seasons must be consecutive in calendar order")
    selected_index = names.index(selected)
    records = []
    for index, season in enumerate(names):
        relative = index - selected_index
        center_month = CENTERS[season]
        target_start = pd.Timestamp(int(args.hindcast_years[0]), CENTERS[selected], 1) + pd.DateOffset(months=relative)
        target_end = pd.Timestamp(int(args.hindcast_years[1]), CENTERS[selected], 1) + pd.DateOffset(months=relative)
        initial_start = target_start - pd.DateOffset(months=leads[1])
        initial_end = target_end - pd.DateOffset(months=leads[1])
        records.append({"season": season, "center_month": center_month, "relative_month": relative,
                        "init_month": initial_start.month,
                        "initialization_years": [initial_start.year, initial_end.year],
                        "target_years": [target_start.year, target_end.year],
                        "season_months": [(center_month - 2) % 12 + 1, center_month, center_month % 12 + 1],
                        "lead_time": list(leads)})
    return records


def _target_dates(args, item):
    return pd.DatetimeIndex([pd.Timestamp(year, CENTERS[str(args.season).upper()], 1) +
                             pd.DateOffset(months=item["relative_month"])
                             for year in range(int(args.hindcast_years[0]), int(args.hindcast_years[1]) + 1)])


def _season_args(args, item):
    result = deepcopy(args)
    result.season = item["season"]
    result.season_months = [f"{month:02d}" for month in item["season_months"]]
    result.init_month = item["init_month"]
    result.hindcast_initialization_years = list(item["initialization_years"])
    result.hindcast_years = list(item["target_years"])
    result.target_year_offset = result.hindcast_years[0] - result.hindcast_initialization_years[0]
    # Observation archives may use the year of the first season month (DJF).
    start_year = item["target_years"][0] - int(item["season_months"][0] > item["center_month"])
    result.obs_years = [min(int(args.obs_years[0]), start_year),
                        max(int(args.obs_years[1]), item["target_years"][1])]
    calendar_shift = item["target_years"][0] - int(args.hindcast_years[0])
    result.clim_years = [int(value) + calendar_shift for value in args.clim_years]
    result.dist_method = "nonparam"
    result.cpt_csv = None  # Explicit seven-season prepared observations are loaded below.
    return result


def _observations(args, item, directory, load_observations, *, return_reference=False):
    supplied = getattr(args, "kharin_observations_dir", None)
    if supplied:
        path = Path(supplied) / f"observations_{item['season']}.nc"
        with xr.open_dataarray(path) as opened:
            obs = opened.load()
    else:
        obs = load_observations(_season_args(args, item), directory)
    if "M" in obs.dims:
        if obs.sizes["M"] != 1:
            raise ValueError("Kharin observations must have one field")
        obs = obs.isel(M=0, drop=True)
    obs = _core(obs).transpose("T", "Y", "X").sortby("T")
    if obs.get_index("T").has_duplicates:
        raise ValueError(f"{item['season']}: duplicate observed seasons")
    if np.any(np.asarray(obs['T'].dt.month) != item['center_month']):
        raise ValueError(f"{item['season']}: observations must retain true centre-month dates")
    dates = _target_dates(args, item)
    missing = dates.difference(pd.DatetimeIndex(obs["T"].values))
    if len(missing):
        raise ValueError(f"{item['season']}: observations lack true centre-month dates {list(missing)}")
    training = obs.sel(T=dates)
    return (training, obs) if return_reference else training


def _canonical(value):
    return "".join(c for c in str(value).split(".")[0].lower() if c.isalnum())


def _read_member_archive(paths, center, variable, item, dates, reference):
    matching = [Path(path) for key, path in paths.items() if _canonical(key) == _canonical(center)]
    if len(matching) != 1:
        raise ValueError(f"Kharin {item['season']}: missing or ambiguous declared GCM {center}")
    with xr.open_dataset(matching[0]) as opened:
        fields = [name for name, value in opened.data_vars.items() if "T" in value.dims]
        field = opened[variable if variable in opened.data_vars else fields[0]].load() if (variable in opened or len(fields) == 1) else None
    if field is None:
        raise ValueError(f"{matching[0]}: ambiguous ensemble physical field")
    aliases = [name for name in ("number", "member", "realization") if name in field.dims]
    if "M" not in field.dims and len(aliases) == 1:
        field = field.rename({aliases[0]: "M"})
    if "M" not in field.dims or field.sizes["M"] < 2:
        raise ValueError(f"Kharin {center}: at least two actual ensemble members are required; ensemble means cannot calibrate spread")
    for dim in list(field.dims):
        if dim not in {"T", "M", "Y", "X"}:
            if field.sizes[dim] != 1:
                raise ValueError(f"Unsupported ensemble dimension {dim}")
            field = field.isel({dim: 0}, drop=True)
    field = _core(field).transpose("T", "M", "Y", "X").sortby("Y").sortby("X")
    if field.get_index("M").has_duplicates:
        raise ValueError("Ensemble member labels must be unique")
    source = pd.DatetimeIndex(field["T"].values)
    mapped = []
    for date in source:
        if date.month == item["init_month"]:
            date = date + pd.DateOffset(months=item["lead_time"][1])
        elif date.month != item["center_month"]:
            raise ValueError(f"{center}: time stamp identifies neither initialization nor target month")
        mapped.append(date.to_period("M").to_timestamp())
    mapped = pd.DatetimeIndex(mapped)
    if mapped.has_duplicates or len(dates.difference(mapped)):
        raise ValueError(f"{center}: duplicate or missing actual target dates")
    field = field.assign_coords(T=mapped).sel(T=dates)
    # Preserve the native member grid here. Calibration interpolates spatial
    # tiles only, avoiding seven large ensemble cubes on a 0.05-degree grid.
    return field


def assemble_kharin_data(args, workdir, downloader, variable, roster, load_observations,
                         *, return_event_reference=False):
    """Acquire seven historical seasons and only the requested future season.

    A separate raw-member cache prevents accidental reuse of ensemble-mean
    archives previously produced by the ordinary regional MME stage.
    """
    if not roster or len(set(roster)) != len(roster):
        raise ValueError("Kharin requires a nonempty, unique predeclared GCM roster")
    nmme_source = str(getattr(args, "kharin_nmme_source", "cpt")).strip().lower()
    ccsr_kwargs = {}
    if nmme_source == "ccsr":
        ccsr_kwargs = dict(
            ccsr_engine=getattr(args, "kharin_ccsr_engine", "netcdf4"),
            ccsr_cfsv2_sample=getattr(args, "kharin_ccsr_cfsv2_sample", "pentad_samples"),
        )
    roster = member_capable_roster(roster, explicit=True, nmme_source=nmme_source)
    if getattr(args, "cpt_csv", None) and not getattr(args, "kharin_observations_dir", None):
        raise ValueError("A single seasonal CPT file cannot supply seven Kharin seasons; provide --kharin-observations-dir")
    items = kharin_design(args)
    _issuance_target(args)
    observations, historical = {}, {}
    archive_maps = {}
    selected = str(args.season).upper()
    root = Path(workdir) / "kharin"
    root.mkdir(parents=True, exist_ok=True)
    selected_obs = None
    event_reference = None
    for item in items:
        directory = root / item["season"]
        directory.mkdir(parents=True, exist_ok=True)
        for folder in ("Observation", "scores", "forecasts", "checkpoints"):
            (directory / folder).mkdir(exist_ok=True)
        obs, full_obs = _observations(args, item, directory, load_observations,
                                      return_reference=True)
        from .masks import apply_saved_target_mask
        obs = apply_saved_target_mask(obs, workdir)
        observations[item["season"]] = obs
        if item["season"] == selected:
            selected_obs = obs
            event_reference = apply_saved_target_mask(full_obs, workdir)
    for item in items:
        obs = observations[item["season"]]
        # All seven response fields must have exactly the same spatial support.
        xr.align(obs.isel(T=0, drop=True), selected_obs.isel(T=0, drop=True), join="exact")
        directory = root / item["season"] / "member_model_data"
        directory.mkdir(parents=True, exist_ok=True)
        lo, hi = item["initialization_years"]
        files = downloader.WAS_Download_Models(str(directory), list(roster), item["init_month"],
            list(args.lead_time), lo, hi, args.gcm_extent, None, None, args.force,
            nmme_source=nmme_source, **ccsr_kwargs)
        archive_maps[f"hindcast/{item['season']}"] = files
    item = next(item for item in items if item["season"] == selected)
    directory = root / selected / "member_model_data"
    lo, hi = item["initialization_years"]
    files = downloader.WAS_Download_Models(str(directory), list(roster), int(args.init_month),
        list(args.lead_time), lo, hi, args.gcm_extent, int(args.forecast_year), None, args.force,
        nmme_source=nmme_source, **ccsr_kwargs)
    date = pd.Timestamp(int(args.forecast_year), int(args.init_month), 1) + pd.DateOffset(months=int(args.lead_time[1]))
    dates = pd.DatetimeIndex([date])
    if date.month != CENTERS[selected] or date.year <= int(args.hindcast_years[1]):
        raise ValueError("Kharin operational target must follow the historical target years")
    archive_maps[f"forecast/{selected}"] = files
    from .roster import resolve_archives
    roster = resolve_archives(
        archive_maps, roster, "Kharin/seven-seasons",
        policy=getattr(args, "gcm_availability", "available"),
        report_path=root / "roster_effective.json")
    # Use one common roster across all seven seasons and the target forecast.
    historical = {center: {} for center in roster}
    for historical_item in items:
        season = historical_item["season"]
        for center in roster:
            historical[center][season] = _read_member_archive(
                archive_maps[f"hindcast/{season}"], center, variable, historical_item,
                _target_dates(args, historical_item), observations[season])
    future = {center: _read_member_archive(files, center, variable, item, dates, selected_obs)
              for center in roster}
    combination = str(getattr(args, 'kharin_combination', 'pooled'))
    atomic_json(root / "design.json", {"target": selected, "seasons": items,
        "validation_group": "year of requested target season; all seven seasons withheld together",
        "forecast_requests": [{"season": selected, "initialization_year": int(args.forecast_year),
                               "initialization_month": int(args.init_month)}],
        "gcm_roster": list(roster), "gcm_combination": combination,
        "gcm_weights": ("equal GCM moments calibrated once" if combination == 'pooled'
                        else "equal after per-GCM calibration"), "ensemble_mean": None,
        "nmme_source": nmme_source, **({"ccsr_options": ccsr_kwargs} if ccsr_kwargs else {})})
    result = observations, historical, future
    return result + (event_reference,) if return_event_reference else result


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _kharin_code_hashes():
    root = Path(__file__).resolve().parents[1]
    names = ('cli/kharin.py', 'cli/optional.py',
             'cli/runtime.py', 'was_mme_kharin2017_v3_7seasons.py',
             'was_probability_reference.py')
    return {name: _hash_file(root / name) for name in names}


def kharin_procedure_signature(args, workdir=None):
    """Content/configuration identity shared by the stage and tile caches.

    Worker counts are execution controls; fitted distributions, the source
    archives and their observation references belong to the scientific key.
    """
    names = ('season', 'init_month', 'lead_time', 'hindcast_years', 'clim_years',
             'forecast_year', 'cv_gap_years', 'optional_cv_gap_years',
             'probability_calibration', 'kharin_target_seasons',
             'kharin_temporal_smoothing', 'kharin_distribution',
             'kharin_min_samples', 'kharin_prob_grid_size', 'kharin_use_numba',
             'kharin_combination', 'kharin_spatial_sigma_deg',
             'kharin_nmme_source', 'kharin_ccsr_engine', 'kharin_ccsr_cfsv2_sample')
    prepared = getattr(args, 'kharin_observations_dir', None)
    references = {}
    if prepared:
        directory = Path(prepared)
        references.update({str(path.resolve()): _hash_file(path)
                           for path in sorted(directory.glob('observations_*.nc'))})
    directory = workdir if workdir is not None else getattr(args, 'workdir', None)
    if directory is not None:
        for name in ('rainfall_predictand.nc', 'temp_predictand.nc', 'scores/target_mask.nc'):
            path = Path(directory) / name
            if path.is_file():
                references[str(path.resolve())] = _hash_file(path)
    return {'scientific_revision': 'kharin-calendar-reference-cache-v3',
            'settings': {name: getattr(args, name, None) for name in names},
            'prepared_observations_dir': str(Path(prepared).resolve()) if prepared else None,
            'observations': references, 'code': _kharin_code_hashes()}


def _cube(args, observations, historical):
    """Attach fixed-lead provenance while retaining real dates across winter."""
    from wass2s.was_mme_kharin2017_v3_7seasons import WAS_mme_Kharin2017
    items = kharin_design(args)
    anchor = np.arange(int(args.hindcast_years[0]), int(args.hindcast_years[1]) + 1)
    obs, hind = {}, {}
    for item in items:
        name = item["season"]
        coords = {"season_year": ("T", anchor),
                  "season_month": ("T", np.full(anchor.size, item["center_month"]))}
        obs[name] = observations[name].assign_coords(**coords)
        hind[name] = historical[name].assign_coords(**coords)
    obs_all, hind_all = WAS_mme_Kharin2017.assemble_fixed_lead_inputs(obs, hind,
        target_seasons=[item["season"] for item in items],
        initialization_months={item["season"]: item["init_month"] for item in items},
        lead_window=tuple(int(v) for v in args.lead_time))
    relative = {item["season"]: item["relative_month"] for item in items}
    dates = [pd.Timestamp(int(year), CENTERS[str(args.season).upper()], 1) + pd.DateOffset(months=relative[str(name)])
             for year, name in zip(obs_all.season_year.values, obs_all.season_name.values)]
    return obs_all.assign_coords(T=dates).sortby("T"), hind_all.assign_coords(T=dates).sortby("T")


def _cdf_at_events(model, fit, ensemble, q):
    """Evaluate the fitted Kharin law at shared CLI events in physical units."""
    months, _ = model._months_years(ensemble)
    _, mu, sigma = model._apply_probability(fit, ensemble, months)
    mu, sigma = _core(mu), _core(sigma)
    if model._resolved_distribution() == "gamma":
        shape = _core(model._select_season(fit.gamma_obs_shape, months, ensemble["T"]))
        scale = _core(model._select_season(fit.gamma_obs_scale, months, ensemble["T"]))
        p0 = (None if fit.gamma_obs_p0 is None
              else _core(model._select_season(fit.gamma_obs_p0, months, ensemble["T"])))
        transformed = _normal_level(model._gamma_cdf(q, shape, scale, p0))
    else:
        transformed = q
    return _probabilities_from_normal(transformed, mu, sigma)


def _normal_level(u):
    return xr.apply_ufunc(ndtri, u.clip(min=0., max=1.),
                          dask="parallelized", output_dtypes=[float]).where(u.notnull())


def _probabilities_from_normal(transformed, mu, sigma):
    cdf = _normal_event_cdf(transformed, mu, sigma).clip(min=0., max=1.)
    low, high = cdf.isel(quantile=0, drop=True), cdf.isel(quantile=1, drop=True)
    prob = xr.concat([low, (high-low).clip(min=0), 1-high],
                     dim=xr.IndexVariable("probability", ["PB", "PN", "PA"]))
    return prob.transpose("probability", "T", "Y", "X")


# ---------------------------------------------------------------------------
# Pooled multi-model calibration
# ---------------------------------------------------------------------------
def _fit_scores(model, data, months, dims, gamma_space):
    """Per-source, per-season normal-score transform fitted on training data."""
    if gamma_space:
        return ("gamma",) + tuple(model._gamma_moments(data, months, dims))
    means, sds = [], []
    for season in model.seasons:
        x = data.isel(T=np.where(months == season)[0])
        means.append(x.mean(dims, skipna=True))
        sds.append(x.std(dims, skipna=True, ddof=1))
    stack = lambda parts: xr.concat(parts, dim=xr.IndexVariable("season", model.seasons)).transpose("season", "Y", "X")
    sd = stack(sds)
    # Constant observed/source marginals are valid. Unit scale maps their
    # training values to zero without inventing predictive uncertainty.
    return ("normal", stack(means), xr.where(sd == 0., 1., sd))


def _apply_scores(model, data, months, params):
    if params[0] == "gamma":
        return model._gamma_to_normal(data, months, *params[1:])
    return (data - model._select_season(params[1], months, data["T"])) / model._select_season(params[2], months, data["T"])


def _event_scores(model, q, months, T, params):
    if params[0] == "gamma":
        shape, scale, p0 = (_core(model._select_season(v, months, T)) for v in params[1:4])
        return _normal_level(model._gamma_cdf(q, shape, scale, p0))
    mean = _core(model._select_season(params[1], months, T))
    sd = _core(model._select_season(params[2], months, T))
    return (q - mean) / sd


def pooled_pseudo_ensemble(fields):
    """Two members with the equal-weight multi-model mean and total variance.

    total variance = mean within-GCM member variance + between-GCM variance of
    the GCM means, i.e. the moments of the equal-weight mixture. With members
    mu +/- sqrt(v/2), ``var(M, ddof=1)`` equals v exactly.
    """
    means = xr.concat([f.mean("M", skipna=True) for f in fields], dim="gcm", coords="minimal", compat="override")
    within = xr.concat([f.var("M", skipna=True, ddof=1) for f in fields], dim="gcm",
                       coords="minimal", compat="override").mean("gcm", skipna=False)
    centre = means.mean("gcm", skipna=False)
    half = np.sqrt((within + means.var("gcm", ddof=0)) / 2.0)
    pseudo = xr.concat([centre + half, centre - half], dim=xr.IndexVariable("M", [0, 1]))
    return pseudo.transpose("T", "M", "Y", "X")


def _raw_pseudo(fields):
    centre = xr.concat([f.mean("M", skipna=True) for f in fields], dim="gcm",
                       coords="minimal", compat="override").mean("gcm", skipna=False)
    return xr.concat([centre, centre], dim=xr.IndexVariable("M", [0, 1])).transpose("T", "M", "Y", "X")


def _pooled_predict(transformer, calibrator, obs_train, train_members, test_members, q, gamma_space):
    """Fit on one training sample and predict the test members (dict per GCM)."""
    months_tr, _ = transformer._months_years(obs_train)
    obs_params = _fit_scores(transformer, obs_train, months_tr, "T", gamma_space)
    source_params = {g: _fit_scores(transformer, h, months_tr, ("T", "M"), gamma_space)
                     for g, h in train_members.items()}
    z_obs = _apply_scores(transformer, obs_train, months_tr, obs_params)
    z_train = pooled_pseudo_ensemble([_apply_scores(transformer, h, months_tr, source_params[g])
                                      for g, h in train_members.items()])
    fit = calibrator._fit(z_train, z_obs, months_tr)
    any_test = next(iter(test_members.values()))
    months_te, _ = transformer._months_years(any_test)
    z_test = pooled_pseudo_ensemble([_apply_scores(transformer, h, months_te, source_params[g])
                                     for g, h in test_members.items()])
    _, mu, sigma = calibrator._apply_probability(fit, z_test, months_te)
    events = _event_scores(transformer, q, months_te, any_test["T"], obs_params)
    prob = _probabilities_from_normal(events, _core(mu), _core(sigma))
    det_fit = transformer.fit_deterministic(_raw_pseudo(list(train_members.values())), obs_train, months_tr)
    det = _core(transformer._apply_deterministic(det_fit, _raw_pseudo(list(test_members.values())), months_te))
    return det, prob


def _event_reference(args, training_reference, event_reference=None):
    """Keep event climatology independent of the shorter calibration panel."""
    reference = training_reference if event_reference is None else event_reference
    if not isinstance(reference, xr.DataArray):
        raise TypeError('Kharin event_reference must be an observed DataArray')
    if 'M' in reference.dims and reference.sizes['M'] == 1:
        reference = reference.isel(M=0, drop=True)
    if set(reference.dims) != {'T', 'Y', 'X'} or not reference.sizes['T']:
        raise ValueError('Kharin event_reference requires nonempty T,Y,X observations')
    reference = _core(reference).transpose('T', 'Y', 'X').sortby('T')
    dates = pd.DatetimeIndex(reference['T'].values)
    centre = CENTERS[str(args.season).upper()]
    if dates.hasnans or dates.has_duplicates or np.any(dates.month != centre) or dates.year.has_duplicates:
        raise ValueError('Kharin event_reference must contain unique actual target-season centre dates')
    for dim in ('Y', 'X'):
        if not reference.get_index(dim).equals(training_reference.get_index(dim)):
            raise ValueError('Kharin event_reference grid differs from calibration observations')
    start, end = map(int, args.clim_years)
    if dates.year.min() > start or dates.year.max() < end:
        raise ValueError('Kharin climatology extends beyond available event_reference; '
                         'provide complete target-season reference observations')
    common = dates.intersection(pd.DatetimeIndex(training_reference['T'].values))
    if len(common):
        try:
            xr.testing.assert_allclose(reference.sel(T=common), _core(training_reference.sel(T=common)))
        except AssertionError as exc:
            raise ValueError('Kharin event_reference disagrees with calibration observations on shared target dates') from exc
    return reference


def _issuance_target(args):
    """Check the selected fixed-lead request before acquiring any archives."""
    item = next(item for item in kharin_design(args) if item['season'] == str(args.season).upper())
    if int(args.init_month) != item['init_month']:
        raise ValueError('Kharin initialization month does not match selected season and fixed lead')
    issue = pd.Timestamp(int(args.forecast_year), int(args.init_month), 1)
    expected = issue + pd.DateOffset(months=int(args.lead_time[1]))
    if expected.month != CENTERS[str(args.season).upper()]:
        raise ValueError('Kharin operational target month disagrees with selected season')
    return issue, expected


def _operational_training(args, obs, future, min_samples):
    """Retain complete calibration groups observable at the actual issuance."""
    issue, expected = _issuance_target(args)
    for centre, field in future.items():
        dates = pd.DatetimeIndex(field['T'].values)
        if len(dates) != 1 or dates.hasnans or dates[0] != expected:
            raise ValueError(f'Kharin {centre}: operational target must equal {expected.date()} '
                             f'for initialization {issue.date()} and the declared lead')
    groups = np.asarray(obs['season_year'].values, dtype=int)
    # T denotes the central month. The three-month season ends one month
    # later and becomes fully observable at the following month start.
    available = (pd.DatetimeIndex(obs['T'].values).to_period('M') + 2).to_timestamp()
    all_groups = np.unique(groups)
    retained = np.asarray([group for group in all_groups
                           if np.all(available[groups == group] <= issue)], dtype=int)
    excluded = all_groups[~np.isin(all_groups, retained)]
    if len(retained) < int(min_samples):
        raise ValueError(f'Kharin operational calibration has {len(retained)} fully observable '
                         f'independent groups before {issue.date()}, fewer than {min_samples}')
    metadata = {
        'kharin_operational_issue': issue.isoformat(),
        'kharin_operational_training_group_count': int(len(retained)),
        'kharin_operational_training_groups': ','.join(map(str, retained)),
        'kharin_operational_excluded_groups': ','.join(map(str, excluded)),
        'kharin_operational_availability': 'whole calibration groups; every three-month target complete before issuance',
    }
    return np.flatnonzero(np.isin(groups, retained)), metadata


def _evaluate_kharin_block(args, observations, historical, future, variable, spatial_sigma=None,
                           *, event_reference=None):
    """Fit every grouped fold independently; return four standard CLI arrays."""
    from wass2s.was_crossfit import CrossFitConfig, _thresholds
    from wass2s.was_mme_kharin2017_v3_7seasons import WAS_mme_Kharin2017
    if str(args.dist_method).lower() != "nonparam":
        raise ValueError("Kharin CLI pooling currently requires --dist-method nonparam for shared empirical events")
    selected = str(args.season).upper()
    reference = _core(observations[selected])
    cfg = CrossFitConfig(climatology_start=int(args.clim_years[0]), climatology_end=int(args.clim_years[1]))
    events = _event_reference(args, reference, event_reference)
    fixed_q, fixed_ref = _thresholds(events, cfg)
    calibration = probability_calibration_mode(args)
    smoothing = getattr(args, "kharin_temporal_smoothing", "1")
    try:
        smoothing = float(smoothing)
    except (TypeError, ValueError):
        pass
    options = dict(target_seasons=[item["season"] for item in kharin_design(args)],
        fixed_lead_window=tuple(int(v) for v in args.lead_time),
        temporal_smoothing=smoothing,
        distribution=getattr(args, "kharin_distribution", "auto"), positive_support=variable == "PRCP",
        min_samples=int(getattr(args, "kharin_min_samples", 8)),
        prob_grid_size=int(getattr(args, "kharin_prob_grid_size", 121)),
        use_numba=bool(getattr(args, "kharin_use_numba", True)),
        probability_calibration=calibration)
    if spatial_sigma is not None:
        options.update(spatial_method="gaussian", spatial_sigma=spatial_sigma)
    combination = str(getattr(args, "kharin_combination", "pooled"))
    if combination not in {"pooled", "mixture"}:
        raise ValueError("--kharin-combination must be pooled or mixture")
    cubes = {center: _cube(args, observations, seasonal) for center, seasonal in historical.items()}
    obs = next(iter(cubes.values()))[0]
    groups = np.asarray(obs.season_year.values, dtype=int)
    season_month = np.asarray(obs.season_month.values)
    operational_train, operational_metadata = _operational_training(args, obs, future, options['min_samples'])
    operational = {}
    for center in historical:
        field = _core(future[center])
        if not field.Y.identical(reference.Y) or not field.X.identical(reference.X):
            field = field.interp(Y=reference.Y, X=reference.X, method="nearest")
        if int(field["T"].dt.year.min()) <= int(reference["T"].dt.year.max()):
            raise ValueError("Operational target must follow all hindcast training years")
        operational[center] = field

    def folds():
        for year in reference["T"].dt.year.values:
            train = np.flatnonzero(abs(groups - int(year)) > int(args.cv_gap_years))
            test = np.flatnonzero((groups == int(year)) & (season_month == CENTERS[selected]))
            if np.unique(groups[train]).size < options["min_samples"]:
                raise ValueError(f"Kharin {year}: fewer than {options['min_samples']} independent training years after withholding buffer")
            response = reference.isel(T=np.flatnonzero(abs(reference["T"].dt.year.values-int(year)) > int(args.cv_gap_years)))
            yield train, test, response

    if combination == "pooled":
        transformer = WAS_mme_Kharin2017(**options)
        gamma_space = transformer._resolved_distribution() == "gamma"
        calibrator = WAS_mme_Kharin2017(**dict(options, distribution="normal", positive_support=False))
        parts = []
        for train, test, response in folds():
            q, ref = fixed_q, fixed_ref
            det, prob = _pooled_predict(transformer, calibrator, obs.isel(T=train),
                                        {c: h.isel(T=train) for c, (_, h) in cubes.items()},
                                        {c: h.isel(T=test) for c, (_, h) in cubes.items()}, q, gamma_space)
            parts.append((det, _attach(prob, q, ref, "hindcast")))
        hdet = xr.concat([part[0] for part in parts], dim="T").sortby("T")
        hprob = xr.concat([part[1] for part in parts], dim="T").sortby("T")
        q, ref = fixed_q, fixed_ref
        fdet, fprob = _pooled_predict(transformer, calibrator, obs.isel(T=operational_train),
                                      {c: h.isel(T=operational_train) for c, (_, h) in cubes.items()},
                                      operational, q, gamma_space)
        output = (hdet, hprob, fdet, _attach(fprob, q, ref, "operational"))
        weighting = "equal-weight multi-model distribution calibrated once (pooled)"
    else:
        collected = []
        for center, (obs_c, hind) in cubes.items():
            parts = []
            for train, test, response in folds():
                model = WAS_mme_Kharin2017(**options)
                train_obs, train_hind = obs_c.isel(T=train), hind.isel(T=train)
                months, _ = model._months_years(train_obs)
                fit = model._fit(train_hind, train_obs, months)
                held = hind.isel(T=test)
                months, _ = model._months_years(held)
                det = _core(model._apply_deterministic(fit, held, months))
                q, ref = fixed_q, fixed_ref
                prob = _attach(_cdf_at_events(model, fit, held, q), q, ref, "hindcast")
                parts.append((det, prob))
            hdet = xr.concat([part[0] for part in parts], dim="T").sortby("T")
            hprob = xr.concat([part[1] for part in parts], dim="T").sortby("T")
            model = WAS_mme_Kharin2017(**options)
            operational_obs = obs_c.isel(T=operational_train)
            months, _ = model._months_years(operational_obs)
            fit = model._fit(hind.isel(T=operational_train), operational_obs, months)
            months, _ = model._months_years(operational[center])
            fdet = _core(model._apply_deterministic(fit, operational[center], months))
            q, ref = fixed_q, fixed_ref
            fprob = _attach(_cdf_at_events(model, fit, operational[center], q), q, ref, "operational")
            collected.append((hdet, hprob, fdet, fprob))
        output = (deterministic_mixture([part[0] for part in collected]),
                  probability_mixture([part[1] for part in collected], "Kharin equal GCM pool"),
                  deterministic_mixture([part[2] for part in collected]),
                  probability_mixture([part[3] for part in collected], "Kharin equal GCM pool"))
        weighting = "equal after per-GCM calibration (mixture)"
    for array in output:
        array.attrs.update(kharin_target_seasons=",".join(options["target_seasons"]),
            kharin_fixed_lead_window=",".join(str(v) for v in options["fixed_lead_window"]),
            kharin_gcm_weighting=weighting, kharin_temporal_smoothing=str(options["temporal_smoothing"]),
            kharin_spatial_sigma_cells=str(spatial_sigma),
            calibration="Kharin CRPS calibrated predictive CDF; fixed empirical climatological events",
            **WAS_mme_Kharin2017._calibration_metadata(calibration))
        array.attrs.update(fixed_q.attrs)
        array.attrs.update(operational_metadata)
    return output


def _spatial_sigma_cells(reference, sigma_deg):
    if sigma_deg is None or float(sigma_deg) <= 0:
        return None
    steps = []
    for dim in ("Y", "X"):
        values = np.asarray(reference[dim].values, dtype=float)
        steps.append(abs(float(values[1] - values[0])) if values.size > 1 else None)
    known = [step for step in steps if step]
    if not known:
        return None
    steps = [step or known[0] for step in steps]
    return tuple(float(sigma_deg) / step for step in steps)


def _workers(args):
    requested = int(getattr(args, "kharin_workers", 0) or 0)
    if requested > 0:
        return requested
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    return max(1, min(available, int(os.environ.get("SLURM_CPUS_PER_TASK", available))))


def _digest_arrays(arrays):
    """Hash exact data, dimensions and all coordinates without precision loss."""
    digest = hashlib.blake2b(digest_size=20)
    def add_values(values):
        values = np.asarray(values)
        digest.update(json.dumps({'dtype': values.dtype.str, 'shape': values.shape},
                                 sort_keys=True).encode())
        if values.dtype.hasobject:
            digest.update(json.dumps(values.tolist(), sort_keys=True, default=str).encode())
        else:
            digest.update(np.ascontiguousarray(values).tobytes())
    for array in arrays:
        digest.update(json.dumps({'dims': array.dims, 'name': array.name,
                                  'attrs': array.attrs}, sort_keys=True, default=str).encode())
        add_values(array.values)
        for name in sorted(array.coords):
            coordinate = array.coords[name]
            digest.update(json.dumps({'coordinate': name, 'dims': coordinate.dims},
                                     sort_keys=True).encode())
            add_values(coordinate.values)
    return digest.hexdigest()


def _tile_task(args, block_obs, historical, future, variable, sigma, core, path, fingerprint,
               event_reference=None):
    if path is not None and Path(path).is_file():
        try:
            with Path(path).open("rb") as source:
                record = pickle.load(source)
            if record.get("fingerprint") == fingerprint:
                return record["result"]
        except Exception:
            pass
    if int(getattr(args, "_kharin_parallel", 0)) > 1:
        try:
            import numba
            numba.set_num_threads(1)
        except Exception:
            pass
    products = _evaluate_kharin_block(args, block_obs, historical, future, variable,
                                     spatial_sigma=sigma, event_reference=event_reference)
    result = tuple(product.isel(Y=core[0], X=core[1]) for product in products)
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        with temporary.open("wb") as output:
            pickle.dump({"fingerprint": fingerprint, "result": result}, output, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    return result


def evaluate_kharin(args, observations, historical, future, variable, checkpoint_dir=None,
                     *, event_reference=None):
    """Fit spatial tiles without expanding every raw ensemble to the fine grid.

    Each tile is widened by a halo of 4 sigma + 1 cells before the (optional)
    Gaussian smoothing of the coefficients, then cropped: the tiled estimates
    are therefore identical to an untiled fit.
    """
    size = int(getattr(args, "kharin_spatial_chunk", 32))
    if size < 1:
        raise ValueError("Kharin spatial chunk size must be positive")
    reference = observations[str(args.season).upper()]
    event_reference = _event_reference(args, reference, event_reference)
    if variable == "PRCP" or str(getattr(args, "kharin_distribution", "auto")).lower() == "gamma":
        inputs = list(observations.values()) + list(future.values()) + [event_reference]
        inputs += [array for seasonal in historical.values() for array in seasonal.values()]
        if any(bool((array < 0).any()) for array in inputs):
            raise ValueError("Kharin precipitation/gamma inputs cannot contain negative physical values; correct units or missing-value codes")
    sigma = _spatial_sigma_cells(reference, getattr(args, "kharin_spatial_sigma_deg", 0.5))
    halo = (0, 0) if sigma is None else tuple(int(4.0 * value + 0.5) + 1 for value in sigma)
    ny, nx = reference.sizes["Y"], reference.sizes["X"]
    fingerprint = None
    if checkpoint_dir is not None and bool(getattr(args, "kharin_checkpoint", True)):
        described = kharin_procedure_signature(args)
        described['spatial_chunk'] = size
        arrays = list(observations.values()) + list(future.values()) + [event_reference]
        arrays += [array for seasonal in historical.values() for array in seasonal.values()]
        text = json.dumps({"scientific_revision": "calendar-reference-exact-cache-v3",
                           "args": described, "variable": variable, "gcms": list(historical),
                           "data": _digest_arrays(arrays)}, sort_keys=True, default=str)
        fingerprint = hashlib.blake2b(text.encode(), digest_size=20).hexdigest()
        checkpoint_dir = Path(checkpoint_dir) / fingerprint[:16]
    tasks, positions = [], []
    for row, start_y in enumerate(range(0, ny, size)):
        for column, start_x in enumerate(range(0, nx, size)):
            y0, y1 = max(0, start_y - halo[0]), min(ny, start_y + size + halo[0])
            x0, x1 = max(0, start_x - halo[1]), min(nx, start_x + size + halo[1])
            core = (slice(start_y - y0, min(start_y + size, ny) - y0),
                    slice(start_x - x0, min(start_x + size, nx) - x0))
            index = {"Y": slice(y0, y1), "X": slice(x0, x1)}
            block_obs = {name: array.isel(index) for name, array in observations.items()}
            block_reference = event_reference.isel(index)
            path = None if fingerprint is None else checkpoint_dir / f"tile_{start_y:05d}_{start_x:05d}.pkl"
            tasks.append((args, block_obs, historical, future, variable, sigma, core, path,
                          None if fingerprint is None else f"{fingerprint}:{start_y}:{start_x}",
                          block_reference))
            positions.append((row, column))
    workers = min(_workers(args), len(tasks))
    if workers > 1:
        from joblib import Parallel, delayed
        args = deepcopy(args)
        args._kharin_parallel = workers
        tasks = [(args,) + task[1:] for task in tasks]
        results = Parallel(n_jobs=workers, backend="loky")(delayed(_tile_task)(*task) for task in tasks)
    else:
        results = [_tile_task(*task) for task in tasks]
    rows = {}
    for (row, column), result in zip(positions, results):
        rows.setdefault(row, []).append(result)
    stitched = [tuple(xr.concat([block[i] for block in rows[row]], dim="X", join="exact") for i in range(4))
                for row in sorted(rows)]
    return tuple(xr.concat([row[i] for row in stitched], dim="Y", join="exact") for i in range(4))


def run_kharin_stage(args, workdir, downloader, variable, roster, load_observations):
    observations, historical, future, event_reference = assemble_kharin_data(
        args, workdir, downloader, variable, roster, load_observations,
        return_event_reference=True)
    # A prepared seven-season archive must reproduce the selected response used
    # by the remaining CLI methods, including any station merging.
    target_path = Path(workdir) / ("rainfall_predictand.nc" if variable == "PRCP" else "temp_predictand.nc")
    if target_path.exists():
        with xr.open_dataarray(target_path) as opened:
            target = opened.load()
        if "M" in target.dims and target.sizes["M"] == 1:
            target = target.isel(M=0, drop=True)
        from .optional import _season_observations
        target = _season_observations(target, str(args.season).upper())
        reference = observations[str(args.season).upper()]
        xr.testing.assert_allclose(_core(target.sel(T=reference["T"])).transpose("T", "Y", "X"), reference)
        start, end = map(int, args.clim_years)
        if int(target['T'].dt.year.min()) <= start and int(target['T'].dt.year.max()) >= end:
            from .masks import apply_saved_target_mask
            event_reference = apply_saved_target_mask(target, workdir)
    products = evaluate_kharin(args, observations, historical, future, variable,
                               event_reference=event_reference)
    for product in products:
        product.attrs["selected_gcms"] = ",".join(historical)
    return products
