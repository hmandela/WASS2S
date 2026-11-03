"""One recorded target domain for observation preparation, fitting and maps.

The historical rainfall-mean mask defines an observation-based analysis domain.
Skill is conditional on that domain; it is not a fold-estimated preprocessing
parameter. Land geometry is independent of both observations and predictions.
"""
from __future__ import annotations

from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr

LAND_ASSET = Path(__file__).resolve().parents[1] / "data" / "ne_50m_land.geojson.gz"


def crop_target(data, extent):
    """Crop all four geographic bounds, regardless of latitude order."""
    if not {"Y", "X"}.issubset(data.dims):
        raise ValueError("Target mask requires one-dimensional Y/X coordinates")
    if data.Y.ndim != 1 or data.X.ndim != 1:
        raise ValueError("Curvilinear grids require a mask on a regular target grid")
    north, west, south, east = map(float, extent)
    if np.any(data.X.values > 180):
        data = data.assign_coords(X=((data.X + 180) % 360) - 180)
    data = data.sortby("Y").sortby("X").sel(Y=slice(south, north), X=slice(west, east))
    if not data.sizes["Y"] or not data.sizes["X"]:
        raise ValueError("Observation extent has no target grid cells")
    return data


@lru_cache(maxsize=1)
def _land_geometries():
    from shapely.geometry import shape
    with gzip.open(LAND_ASSET, "rt", encoding="utf-8") as handle:
        return tuple(shape(item) for item in json.load(handle)["geometries"])


def geographic_land_mask(grid):
    """Rasterize bundled Natural Earth 1:50m land at target cell centres."""
    from shapely import intersects_xy, union_all
    from shapely.geometry import box
    lon, lat = np.meshgrid(((grid.X.values + 180) % 360) - 180, grid.Y.values)
    bounds = box(float(lon.min()) - 1e-8, float(lat.min()) - 1e-8,
                 float(lon.max()) + 1e-8, float(lat.max()) + 1e-8)
    shapes = [g for g in _land_geometries() if g.intersects(bounds)]
    values = intersects_xy(union_all(shapes), lon, lat)
    return xr.DataArray(values, dims=("Y", "X"), coords={"Y": grid.Y, "X": grid.X})


def _on_target_grid(mask, data):
    """Require identical cell coordinates (order and subsets may differ)."""
    if set(mask.dims) != {"Y", "X"}:
        raise ValueError("Land/target mask must be a two-dimensional Y/X field")
    for dim in ("Y", "X"):
        if not mask.get_index(dim).is_unique or not data.get_index(dim).isin(mask.get_index(dim)).all():
            raise ValueError(f"Target mask does not cover the exact {dim} coordinates; regrid the mask explicitly")
    return mask.sel(Y=data.Y, X=data.X)


def _custom_land_mask(path, data):
    with xr.open_dataset(path) as dataset:
        if "land_mask" in dataset:
            field = dataset["land_mask"].load()
        elif len(dataset.data_vars) == 1:
            field = dataset[next(iter(dataset.data_vars))].load()
        else:
            raise ValueError("Custom mask must contain land_mask or one data variable")
    field = field.rename({d: {"latitude": "Y", "lat": "Y", "longitude": "X", "lon": "X"}[d]
                          for d in field.dims if d in {"latitude", "lat", "longitude", "lon"}})
    field = _on_target_grid(field, data)
    finite = field.values[np.isfinite(field.values)]
    if not np.isin(finite, [0, 1]).all():
        raise ValueError("Custom land mask must use 1 for land and 0 or NaN elsewhere")
    return field == 1


def build_target_mask(data, args, variable):
    """Return component masks and their reference fields on the target grid."""
    if "T" not in data.dims or data.sizes["T"] == 0:
        raise ValueError("Target observations must contain a nonempty T dimension")
    reference = getattr(args, "dry_mask_reference", "observations")
    base = data if reference == "observations" else data.sel(
        T=slice(str(args.clim_years[0]), str(args.clim_years[1])))
    if base.sizes["T"] == 0:
        raise ValueError("Dry-mask reference interval contains no observations")
    mean = base.mean("T", skipna=True)
    count = np.isfinite(base).sum("T")
    valid = np.isfinite(data).any("T")
    land = xr.ones_like(valid, dtype=bool)
    land_file = getattr(args, "land_mask_file", None)
    land_enabled = getattr(args, "land_mask", True)
    if land_enabled:
        land = _custom_land_mask(land_file, data) if land_file else geographic_land_mask(data)
    dry = xr.ones_like(valid, dtype=bool)
    dry_enabled = variable == "PRCP" and getattr(args, "apply_drymask", True)
    threshold = float(getattr(args, "dry_threshold", 100.0))
    min_years = int(getattr(args, "dry_min_valid_years", 1))
    if dry_enabled:
        if not np.isfinite(threshold) or threshold < 0 or min_years < 1:
            raise ValueError("Dry threshold must be finite/nonnegative and minimum valid years positive")
        dry = (mean >= threshold) & (count >= min_years)
    keep = valid & land & dry
    result = xr.Dataset({name: xr.where(value, 1.0, np.nan) for name, value in
                         {"land_mask": land, "valid_mask": valid,
                          "dry_mask": dry, "target_mask": keep}.items()})
    result["reference_mean"] = mean
    result["reference_valid_count"] = count
    result.attrs.update(
        target_domain="fixed_observation_defined_domain" if dry_enabled else "land_and_observation_support",
        evaluation_domain_scope="skill conditional on this recorded domain; mask is not estimated inside CV folds",
        variable=variable, dry_mask_enabled=int(dry_enabled),
        dry_threshold=threshold if dry_enabled else "not_applicable",
        dry_mask_reference=reference, dry_min_valid_years=min_years,
        reference_start=str(base["T"].values[0]), reference_end=str(base["T"].values[-1]),
        land_mask_enabled=int(land_enabled),
        land_source=str(land_file) if land_file else ("Natural Earth 1:50m land; grid-cell centres" if land_enabled else "disabled"))
    if land_enabled:
        result.attrs["land_source_sha256"] = hashlib.sha256(Path(land_file or LAND_ASSET).read_bytes()).hexdigest()
    return result


def apply_target_mask(data, mask):
    if data is None or not {"Y", "X"}.issubset(data.dims):
        return data
    support = _on_target_grid(mask["target_mask"], data) == 1
    result = data.where(support)
    metadata = {k: mask.attrs[k] for k in ("target_domain", "evaluation_domain_scope") if k in mask.attrs}
    result.attrs.update(metadata)
    if isinstance(result, xr.Dataset):
        for name in result.data_vars:
            result[name].attrs.update(metadata)
    return result


def prepare_target_observations(data, args, workdir, variable):
    """Preserve the unmasked field; save the mask; return the masked predictand."""
    workdir = Path(workdir)
    data = crop_target(data, args.obs_extent).compute()
    mask = build_target_mask(data, args, variable)
    if not bool(mask.target_mask.notnull().any()):
        raise ValueError("The land/dry/validity mask excludes every target cell; check extent, units and threshold")
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "scores").mkdir(exist_ok=True)
    stem = "rainfall" if variable == "PRCP" else "temp"
    data.to_netcdf(workdir / f"{stem}_predictand_unmasked.nc")
    mask.to_netcdf(workdir / "scores" / "target_mask.nc")
    count = int(mask.target_mask.notnull().sum())
    print(f"[target mask] {variable}: {count}/{mask.target_mask.size} cells retained; "
          f"land={mask.attrs['land_mask_enabled']}, dry={mask.attrs['dry_threshold']}")
    return apply_target_mask(data, mask)


def apply_saved_target_mask(data, workdir):
    """Apply the same recorded support to every exported or plotted product."""
    path = Path(workdir) / "scores" / "target_mask.nc"
    if data is None or not path.exists():
        return data
    with xr.open_dataset(path) as source:
        mask = source[["target_mask"]].load()
    return apply_target_mask(data, mask)
