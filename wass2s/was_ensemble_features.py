"""Exchangeable-member summaries shared by every MME regression estimator.

Reductions use only the members of one GCM, initialization/target and grid cell.
No statistics are estimated over years or observations. Compute these fields on
the native GCM grid, then interpolate them to the predictand grid.
"""
from __future__ import annotations

import re
import numbers

import numpy as np

DEFAULT_ENSEMBLE_FEATURES = ("mean", "std", "q10", "q50", "q90")
MEMBER_DIMENSIONS = ("number", "M", "member", "members", "realization", "ensemble", "ensmem")


def normalise_ensemble_mean(value):
    """Return a deterministic mean/median mode or None for member statistics."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"mean", "median"}:
            return value
        if value == "none":
            return None
    raise ValueError("ensemble_mean must be 'mean', legacy 'median', or None ('none' in a CLI).")


def normalise_ensemble_features(features=DEFAULT_ENSEMBLE_FEATURES):
    """Validate ordered summaries; ``qP`` is percentile P, between 0 and 100."""
    if isinstance(features, str):
        features = features.replace(",", " ").split()
    result = []
    identities = set()
    for feature in features:
        name = str(feature).strip().lower()
        identity = name
        if name in {"mean", "std", "median", "min", "max", "iqr", "count"}:
            if name == "median":
                identity = "q50"
        elif re.fullmatch(r"q(?:\d+(?:\.\d*)?|\.\d+)", name):
            percentile = float(name[1:])
            if not 0 <= percentile <= 100:
                raise ValueError(f"Invalid ensemble percentile {name!r}; expected q0 to q100.")
            name = "q" + format(percentile, "g")
            identity = name
        else:
            raise ValueError(f"Unsupported ensemble feature {name!r}; use mean, std, median, min, max, iqr, count or q0..q100.")
        if identity in identities:
            raise ValueError(f"Duplicate ensemble feature {name!r}.")
        identities.add(identity)
        result.append(name)
    if not result:
        raise ValueError("At least one ensemble feature is required.")
    return tuple(result)


def member_dimension(data, required=False):
    """Find exactly one member dimension; never guess between GCM/member axes."""
    if data.attrs.get("wass2s_ensemble_mode") == "statistics" or "ensemble_statistic" in data.coords:
        raise ValueError("Input already contains ensemble feature channels, not raw members of one GCM.")
    found = [dim for dim in MEMBER_DIMENSIONS if dim in data.dims]
    if len(found) > 1:
        raise ValueError(f"Ambiguous member dimensions {found}; provide one GCM with one member dimension.")
    if not found:
        if required:
            raise ValueError("ensemble_mean=None requires raw members for every GCM; this field has no member dimension and may already be an ensemble mean.")
        return None
    dim = found[0]
    if data.sizes[dim] == 0:
        raise ValueError("The ensemble member dimension is empty.")
    return dim


def ensemble_summary_fields(data, ensemble_mean="mean", features=DEFAULT_ENSEMBLE_FEATURES,
                            std_ddof=0, min_members=2):
    """Return ordered ``{statistic: DataArray(T,Y,X)}`` on the native grid.

    Statistics mode requires raw members. Non-finite values are missing. A
    summary is masked when fewer than ``min_members`` finite members remain;
    std also requires more members than ``std_ddof``. Population std (ddof=0)
    describes the supplied ensemble; ddof=1 is optionally available. ``count``
    reports finite-member counts, including low counts, but remains NaN where
    all members are missing. No zero-valued artificial spread is fabricated.
    Legacy mean mode permits an already deterministic field and requires only
    one valid member when reducing raw member data.
    """
    mode = normalise_ensemble_mean(ensemble_mean)
    if not isinstance(std_ddof, numbers.Integral) or isinstance(std_ddof, bool) or std_ddof < 0:
        raise ValueError("ensemble_std_ddof must be a non-negative integer.")
    if not isinstance(min_members, numbers.Integral) or isinstance(min_members, bool) or min_members < 1:
        raise ValueError("ensemble_min_members must be a positive integer.")
    dim = member_dimension(data, required=mode is None)
    data = data.astype(float).where(np.isfinite(data))
    if mode is not None:
        if dim is None:
            value = data
        elif mode == "median":
            value = data.median(dim, skipna=True, keep_attrs=True)
        else:
            value = data.mean(dim, skipna=True, keep_attrs=True)
        return {mode: value}
    names = normalise_ensemble_features(features)
    count = data.count(dim)
    valid = count >= min_members
    # Quantile needs the whole ensemble, but independent T/Y/X chunks remain.
    quantile_data = data.chunk({dim: -1}) if data.chunks is not None else data

    def quantile(q):
        value = quantile_data.quantile(q, dim=dim, skipna=True, keep_attrs=True)
        return value.drop_vars("quantile") if "quantile" in value.coords else value

    result = {}
    for name in names:
        if name == "mean":
            value = data.mean(dim, skipna=True, keep_attrs=True)
        elif name == "std":
            value = data.std(dim, skipna=True, ddof=std_ddof, keep_attrs=True).where(count > std_ddof)
        elif name == "median":
            value = quantile(0.5)
        elif name == "min":
            value = data.min(dim, skipna=True, keep_attrs=True)
        elif name == "max":
            value = data.max(dim, skipna=True, keep_attrs=True)
        elif name == "iqr":
            value = quantile(0.75) - quantile(0.25)
        elif name == "count":
            value = count.astype(float).where(count > 0)
        else:
            value = quantile(float(name[1:]) / 100.0)
        if name != "count":
            value = value.where(valid)
        value.attrs = dict(data.attrs)
        if name == "count":
            value.attrs["units"] = "1"
        value.attrs.update(ensemble_statistic=name, ensemble_std_ddof=int(std_ddof),
                           ensemble_min_members=int(min_members))
        result[name] = value
    return result
