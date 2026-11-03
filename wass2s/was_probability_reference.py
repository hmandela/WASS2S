"""One explicit climatological definition for probabilistic categories.

The configured reference observations define events, independently of model
fitting.  Call once on the complete reference period and reuse the returned
thresholds for every fold, operational forecast and verification calculation.
"""

import numpy as np
import xarray as xr


def climatological_terciles(observations, clim_start=None, clim_end=None, min_count=3):
    """Return exact tercile thresholds and empirical PB/PN/PA reference rates.

    ``clim_start`` and ``clim_end`` are inclusive calendar years.  If omitted,
    all supplied observations form the reference.  An unavailable/too short
    requested period raises instead of falling back to a different period.
    Cells with fewer than ``min_count`` finite values remain missing.

    The ``quantile`` labels .33/.67 are retained for WASS2S compatibility;
    calculations use exactly 1/3 and 2/3, with events ``y <= low``,
    ``low < y <= high`` and ``y > high``.  Empirical reference probabilities
    preserve ties (for example in dry regions) rather than assuming thirds.
    """
    if not isinstance(observations, xr.DataArray) or "T" not in observations.dims:
        raise TypeError("Reference observations must be a DataArray with dimension T")
    if (clim_start is None) != (clim_end is None):
        raise ValueError("Provide both climatology bounds or neither")
    if int(min_count) != min_count or min_count < 1:
        raise ValueError("min_count must be a positive integer")
    if clim_start is not None and int(clim_start) > int(clim_end):
        raise ValueError("Reversed climatology bounds")
    years = observations["T"].dt.year
    base = observations
    if clim_start is not None:
        base = observations.where((years >= int(clim_start)) & (years <= int(clim_end)), drop=True)
    if base.sizes["T"] < min_count:
        raise ValueError("Insufficient observation dates inside the requested climatology")
    base = base.where(np.isfinite(base))
    if base.chunks is not None:
        base = base.chunk({"T": -1})
    count = base.notnull().sum("T")
    q = base.quantile([1.0 / 3.0, 2.0 / 3.0], dim="T", skipna=True).where(count >= min_count)
    q = q.assign_coords(quantile=[0.33, 0.67]).rename("climatological_terciles")
    low, high = (q.isel(quantile=i, drop=True) for i in range(2))
    categories = xr.where(base <= low, 0, xr.where(base <= high, 1, 2)).where(
        base.notnull() & low.notnull() & high.notnull()
    )
    reference = xr.concat(
        [(categories == k).where(categories.notnull()).mean("T") for k in range(3)],
        dim=xr.IndexVariable("probability", ["PB", "PN", "PA"]),
    ).rename("reference_probabilities")
    first, last = int(base["T"].dt.year.min()), int(base["T"].dt.year.max())
    metadata = {
        "tercile_reference": "climatology",
        "climatology_start": first if clim_start is None else int(clim_start),
        "climatology_end": last if clim_end is None else int(clim_end),
        "climatology_first_year_used": first,
        "climatology_last_year_used": last,
        "climatology_date_count": int(base.sizes["T"]),
        "tercile_quantile_levels": "1/3,2/3",
        "category_definition": "PB: y <= low; PN: low < y <= high; PA: y > high",
        "minimum_reference_count": int(min_count),
    }
    q.attrs.update(metadata)
    reference.attrs.update(metadata, reference_estimator="empirical_category_frequencies")
    if "units" in observations.attrs:
        q.attrs["units"] = observations.attrs["units"]
    return q, reference
