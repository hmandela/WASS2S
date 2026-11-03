"""Input-channel checks shared by MME regressors and their operational forecasts."""
import numpy as np
import xarray as xr


def validate_mme_predictor_schema(train, query):
    """Require identical ordered predictor channels and spatial grids.

    T is deliberately not compared: a forecast has different target years.
    Member statistics remain predictor columns, never additional training rows.
    """
    for label, field in (("training", train), ("prediction", query)):
        if not isinstance(field, xr.DataArray):
            raise TypeError(f"MME {label} predictors must be an xarray DataArray.")
        if set(field.dims) != {"T", "M", "Y", "X"}:
            raise ValueError(f"MME {label} predictors require T,M,Y,X; got {field.dims}.")
        for dim in ("M", "Y", "X"):
            if field.get_index(dim).has_duplicates:
                raise ValueError(f"MME {label} predictor {dim} coordinates contain duplicates.")
    for dim in ("M", "Y", "X"):
        if not train.get_index(dim).equals(query.get_index(dim)):
            raise ValueError(f"MME prediction {dim} channels/grid must match training in the same order.")
    for name in ("source_model", "ensemble_statistic"):
        if name in train.coords or name in query.coords:
            if name not in train.coords or name not in query.coords or not np.array_equal(
                    train[name].values, query[name].values):
                raise ValueError(f"MME prediction {name} schema differs from training.")


def standardize_mme_predictors(train, query, clim_start=None, clim_end=None):
    """Fit per-channel scales on training years; constant columns use scale one."""
    validate_mme_predictor_schema(train, query)
    ref = train.sel(T=slice(str(clim_start), str(clim_end))) if (
        clim_start is not None and clim_end is not None) else train
    if ref.sizes["T"] < 2:
        ref = train
    mean = ref.mean("T", skipna=True)
    scale = ref.std("T", skipna=True)
    scale = scale.where(np.isfinite(scale) & (scale > 1e-12), 1.)
    return (train - mean) / scale, (query - mean) / scale


def require_physical_ensemble(field):
    """Prevent physical averaging of predictor statistics such as std and q10."""
    if isinstance(field, xr.DataArray) and (
            field.attrs.get("wass2s_ensemble_mode") == "statistics"
            or ("ensemble_statistic" in field.coords and not np.isin(
                field["ensemble_statistic"].values, ["mean", "median"]).all())):
        raise ValueError("Distribution-statistic channels are ML predictors, not physical "
                         "ensemble forecasts. Use ensemble_mean='mean' for weighted MME.")
