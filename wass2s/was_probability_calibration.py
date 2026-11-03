"""Shared residual-calibration modes and direct hindcast probability contract.

Outer reuse is the operational default. Inner/nested requires predictions
cross-fitted entirely inside each outer training set. Fixed empirical climatic
categories are independent of these model-fitting choices.
"""
from collections.abc import Mapping
import numpy as np
import xarray as xr
from .was_probability_reference import climatological_terciles


def normalize_probability_calibration(value=None, default="outer"):
    """Canonicalize ``inner`` to ``nested``; reject misspelled procedures."""
    name = str(default if value is None else value).strip().lower()
    if name == "inner":
        name = "nested"
    if name not in {"outer", "nested"}:
        raise ValueError("probability_calibration must be 'outer', 'inner' or 'nested'")
    return name


def _field(value, name):
    if not isinstance(value, xr.DataArray):
        raise TypeError(f"{name} must be an xarray.DataArray")
    if "M" in value.dims:
        if value.sizes["M"] != 1:
            raise ValueError(f"{name}: observation/member dimension M must be singleton")
        value = value.isel(M=0, drop=True)
    if set(value.dims) != {"T", "Y", "X"}:
        raise ValueError(f"{name} must have dimensions T,Y,X")
    value = value.transpose("T", "Y", "X")
    if value.get_index("T").has_duplicates or not value.get_index("T").is_monotonic_increasing:
        raise ValueError(f"{name} dates must be unique and sorted")
    return value.where(np.isfinite(value))


def _partitions(obs, calibration_splits, nb_omit):
    n = obs.sizes["T"]
    if int(nb_omit) != nb_omit or nb_omit < 0:
        raise ValueError("nb_omit must be a nonnegative integer")
    years = obs["T"].dt.year.values
    if len(set(years)) != n:
        raise ValueError("Residual calibration requires one target per year")
    gap = int(np.ceil(nb_omit / 2))
    if calibration_splits is None:
        idx = np.arange(n)
        folds = [(idx[np.abs(years - years[i]) > gap], np.array([i])) for i in idx]
    else:
        folds = list(calibration_splits)
    used = set()
    checked = []
    for tr, te in folds:
        tr, te = np.asarray(tr), np.asarray(te)
        if any(a.ndim != 1 or not np.issubdtype(a.dtype, np.integer) for a in (tr, te)):
            raise ValueError("calibration_splits require one-dimensional integer positions")
        if not len(tr) or not len(te) or any(np.any((a < 0) | (a >= n)) for a in (tr, te)):
            raise ValueError("Invalid or empty calibration split")
        if len(np.unique(tr)) != len(tr) or len(np.unique(te)) != len(te) or set(tr) & set(te) or used & set(te):
            raise ValueError("Calibration train/test partitions must be disjoint and test dates unique")
        if np.any(np.abs(years[tr, None] - years[te][None, :]) <= gap):
            raise ValueError("Calibration training dates intersect the test-year exclusion buffer")
        used.update(te.tolist())
        checked.append((tr, te))
    if not checked:
        raise ValueError("At least one calibration split is required")
    return checked


def direct_hindcast_probabilities(model, observations, clim_start, clim_end, hindcast,
        best_code_da=None, best_shape_da=None, best_loc_da=None, best_scale_da=None,
        *, probability_calibration="outer", calibration_splits=None,
        calibration_residuals=None, nb_omit=0, min_oof=3, obs_for_terciles=None,
        tercile_reference="climatology", probability_fit=None,
        probability_distribution_map=None, probability_fit_mode="grid",
        diagnostic=False):
    """Dress hindcasts using explicit, independently defined climate events.

    Default outer pools omit the current year (and ``nb_omit`` buffer). When
    supplied, ``calibration_splits`` is an iterable of (train, test) integer
    positions in observations. Other outer models may have seen the test year;
    this approximation is never labelled strictly nested.

    Inner/nested accepts only a mapping ``calibration_residuals[fold_index]``
    of genuine inner-OOF errors (T,Y,X); their dates must belong to that fold's
    train. The caller is responsible for their fitted-model provenance. A
    single hindcast cannot reconstruct independent nested residuals.

    ``diagnostic=True`` explicitly opts into the historical full residual pool.
    Full-history best-fit maps, including selected family codes, are discarded.
    A separately declared candidate catalog (or support/type-specific default)
    is refitted on each train, just as in WAS_Cross_Validator.
    """
    from .was_crossfit import CrossFitConfig
    from .was_cross_validate import _training_probabilities
    mode = normalize_probability_calibration(probability_calibration)
    if int(min_oof) != min_oof or min_oof < 3:
        raise ValueError("min_oof must be an integer >= 3")
    if tercile_reference not in {"climatology", "training"}:
        raise ValueError("tercile_reference must be climatology or training")
    obs, pred = xr.align(_field(observations, "observations"), _field(hindcast, "hindcast"), join="exact")
    if mode == "nested" and not diagnostic and not isinstance(calibration_residuals, Mapping):
        raise ValueError("inner/nested calibration requires calibration_residuals indexed by fold; use WAS_Cross_Validator to compute them")
    if diagnostic and (calibration_splits is not None or calibration_residuals is not None):
        raise ValueError("diagnostic full-pool calibration cannot accept fold calibration inputs")
    options = {"tercile_reference": tercile_reference,
        "probability_fit": probability_fit, "probability_distribution_map": probability_distribution_map,
        "probability_fit_mode": probability_fit_mode}
    options.update({k:v for k,v in zip(("best_code_da","best_shape_da","best_loc_da","best_scale_da"),
        (best_code_da,best_shape_da,best_loc_da,best_scale_da)) if v is not None})
    if tercile_reference == "climatology":
        reference = obs if obs_for_terciles is None else _field(obs_for_terciles, "obs_for_terciles")
        q, ref = climatological_terciles(reference, clim_start, clim_end, min_count=int(min_oof))
        xr.align(q, obs.isel(T=0,drop=True), join="exact")
        options.update(_probability_thresholds=q, _probability_reference=ref)
    cfg = CrossFitConfig(climatology_start=clim_start, climatology_end=clim_end, min_oof=int(min_oof))
    errors = obs - pred
    if diagnostic:
        prob, _, _, _ = _training_probabilities(model, pred, errors, obs, cfg, options)
        prob.attrs.update(calibration="diagnostic full-hindcast residual pool including verification errors",
            probability_calibration="diagnostic", calibration_strictly_nested=0)
        return prob
    parts = []
    counts = []
    for index, (tr, te) in enumerate(_partitions(obs, calibration_splits, nb_omit)):
        train = obs.isel(T=tr)
        if mode == "nested":
            if index not in calibration_residuals:
                raise ValueError(f"Missing inner calibration residuals for fold {index}")
            residuals = _field(calibration_residuals[index], "calibration_residuals")
            if not residuals.get_index("T").isin(train.get_index("T")).all():
                raise ValueError("Inner residual dates must be within the outer training partition")
            xr.align(residuals.isel(T=0,drop=True),train.isel(T=0,drop=True),join="exact")
        else:
            if calibration_residuals is not None:
                raise ValueError("Explicit calibration_residuals are only accepted with inner/nested")
            residuals = errors.isel(T=tr)
        probability, _, _, _ = _training_probabilities(model, pred.isel(T=te), residuals, train, cfg, options)
        parts.append(probability)
        counts.append(residuals.notnull().sum("T").expand_dims(T=pred["T"].isel(T=te)))
    result = xr.concat(parts,dim="T").sortby("T").reindex(T=obs["T"]).transpose("probability","T","Y","X")
    result.attrs.update(probability_calibration=mode, tercile_reference=tercile_reference,
        calibration=("approximate reuse of other outer-fold residuals; their models may include the verification year"
            if mode == "outer" else "caller-supplied inner OOF residuals; caller guarantees fitting provenance"),
        calibration_strictly_nested=0,
        calibration_provenance=("outer_hindcast_reuse" if mode == "outer" else "caller_supplied_not_independently_verified"),
        minimum_calibration_count=int(min_oof), calibration_buffer_years=int(np.ceil(nb_omit/2)))
    result = result.assign_coords(calibration_count=xr.concat(counts,dim="T").sortby("T").reindex(T=obs["T"]))
    return result
