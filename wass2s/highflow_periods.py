"""Explicit control of the periods used at each stage of the chain.

WHY THIS EXISTS
---------------
Three logically distinct periods are involved in building high-flow
predictands, and conflating them is a silent source of error:

``climatology``
    The period over which the CLIMATOLOGICAL REGIME is estimated: daily
    climatology, hydrological anchor, modes, segmentation, and therefore the
    high-water window bounds.  This is the period that should normally be a
    recognised normal (e.g. 1991-2020) so that the regime definition is
    stable, documented and comparable with other studies.

``calibration``
    The period over which a MODEL BIAS is fitted (the multiplicative factor,
    the quantile mapping, or the timing shift).  It need not equal the
    climatological period: bias calibration uses only independent training pairs; regime definition
    uses a fixed or fold-specific reference appropriate to the experiment.

``application``
    The period over which PREDICTANDS are finally computed.  This is normally
    the full available record - deliberately longer than the other two - so
    that a forecast model can be trained on every year for which a target can
    be formed.

Using one period for all three is a defensible choice, but it must be a
CHOICE, not an accident of implementation.  ``ClimatologyPeriod`` makes the
three explicit, lets the user set them independently, and records what was
actually used so it can be reported.

A WARNING THE HELPERS ENFORCE
-----------------------------
Application can be shorter than the reference by experimental design.
``describe`` reports this choice without labeling it scientifically wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

__all__ = [
    "ClimatologyPeriod",
    "slice_series",
    "slice_dataarray",
    "overlap_years",
]

TimeLike = Union[str, pd.Timestamp, None]


@dataclass
class ClimatologyPeriod:
    """The three periods of the high-flow chain, set independently.

    Parameters
    ----------
    climatology_start, climatology_end
        Regime definition period.  ``None`` means "use everything available",
        which is convenient but not reproducible across data updates; setting
        an explicit normal is strongly preferred for published work.
    calibration_start, calibration_end
        Model-bias fitting period.  ``None`` falls back to the climatological
        period when ``calibration_follows_climatology`` is True (the default),
        otherwise to the full record.
    application_start, application_end
        Predictand computation period.  ``None`` means the full record, which
        is the recommended setting.
    """

    climatology_start: TimeLike = None
    climatology_end: TimeLike = None

    calibration_start: TimeLike = None
    calibration_end: TimeLike = None

    application_start: TimeLike = None
    application_end: TimeLike = None

    calibration_follows_climatology: bool = True
    """When no calibration period is given, reuse the climatological one.

    This default keeps the bias factor consistent with the window it was
    fitted on.  Set to False to calibrate on the full record instead, which
    maximises the number of concurrent pairs at the cost of mixing periods.
    """

    require_minimum_years: int = 10
    """Below this many years in the climatological period, ``validate``
    raises.  This is a configurable support threshold, not a universal definition of climatology."""

    # -- resolved accessors -------------------------------------------------

    @property
    def climatology(self) -> tuple[TimeLike, TimeLike]:
        return self.climatology_start, self.climatology_end

    @property
    def calibration(self) -> tuple[TimeLike, TimeLike]:
        if self.calibration_start is None and self.calibration_end is None:
            if self.calibration_follows_climatology:
                return self.climatology_start, self.climatology_end
            return None, None
        return self.calibration_start, self.calibration_end

    @property
    def application(self) -> tuple[TimeLike, TimeLike]:
        return self.application_start, self.application_end

    # -- checks -------------------------------------------------------------

    def validate(self, index: Optional[pd.DatetimeIndex] = None) -> None:
        """Check internal consistency, and coverage against a real index."""
        for label, (a, b) in (
            ("climatology", self.climatology),
            ("calibration", self.calibration),
            ("application", self.application),
        ):
            if a is not None and b is not None:
                if pd.Timestamp(a) > pd.Timestamp(b):
                    raise ValueError(
                        f"{label} period starts after it ends: {a} > {b}"
                    )

        if index is None:
            return

        idx = pd.DatetimeIndex(index)
        a, b = self.climatology
        sub = idx
        if a is not None:
            sub = sub[sub >= pd.Timestamp(a)]
        if b is not None:
            sub = sub[sub <= pd.Timestamp(b)]
        if len(sub) == 0:
            raise ValueError(
                "The climatological period does not overlap the data at all "
                f"({a} to {b} against {idx.min().date()} to {idx.max().date()})."
            )
        n_years = len(np.unique(sub.year))
        if n_years < int(self.require_minimum_years):
            raise ValueError(
                f"Climatological period covers only {n_years} calendar years, "
                f"below require_minimum_years={self.require_minimum_years}. "
                "Widen the period or lower the requirement deliberately."
            )

    def describe(self, index: Optional[pd.DatetimeIndex] = None) -> dict[str, Any]:
        """Human-readable record of what was actually used, for reporting."""
        out: dict[str, Any] = {
            "climatology_start": _fmt(self.climatology[0]),
            "climatology_end": _fmt(self.climatology[1]),
            "calibration_start": _fmt(self.calibration[0]),
            "calibration_end": _fmt(self.calibration[1]),
            "application_start": _fmt(self.application[0]),
            "application_end": _fmt(self.application[1]),
            "calibration_follows_climatology": bool(
                self.calibration_follows_climatology
            ),
        }
        if index is not None:
            idx = pd.DatetimeIndex(index)
            out["record_start"] = _fmt(idx.min())
            out["record_end"] = _fmt(idx.max())
            for label in ("climatology", "calibration", "application"):
                a, b = getattr(self, label)
                sub = _restrict(idx, a, b)
                out[f"{label}_n_years"] = int(len(np.unique(sub.year))) if len(sub) else 0

        app_a, app_b = self.application
        clim_a, clim_b = self.climatology
        out["application_narrower_than_climatology"] = bool(
            app_a is not None and clim_a is not None
            and pd.Timestamp(app_a) > pd.Timestamp(clim_a)
        ) or bool(
            app_b is not None and clim_b is not None
            and pd.Timestamp(app_b) < pd.Timestamp(clim_b)
        )
        if out["application_narrower_than_climatology"]:
            out["warning"] = (
                "The application period is narrower than the climatological "
                "period: predictand years are being discarded that the regime "
                "definition already covers. This can be intentional; document the experiment."
            )
        return out

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _fmt(t: TimeLike) -> Optional[str]:
    return None if t is None else str(pd.Timestamp(t).date())


def _restrict(idx: pd.DatetimeIndex, a: TimeLike, b: TimeLike) -> pd.DatetimeIndex:
    sub = idx
    if a is not None:
        sub = sub[sub >= pd.Timestamp(a)]
    if b is not None:
        sub = sub[sub <= pd.Timestamp(b)]
    return sub


def slice_series(series: pd.Series, start: TimeLike, end: TimeLike) -> pd.Series:
    """Restrict a time-indexed series, tolerating open bounds."""
    out = series
    if start is not None:
        out = out.loc[out.index >= pd.Timestamp(start)]
    if end is not None:
        out = out.loc[out.index <= pd.Timestamp(end)]
    return out


def slice_dataarray(da, start: TimeLike, end: TimeLike, time_dim: str = "T"):
    """Restrict a DataArray along its time dimension, tolerating open bounds."""
    if time_dim not in da.dims:
        for cand in ("T", "time", "Time"):
            if cand in da.dims:
                time_dim = cand
                break
    sel: dict[str, Any] = {}
    if start is not None or end is not None:
        sel[time_dim] = slice(
            None if start is None else pd.Timestamp(start),
            None if end is None else pd.Timestamp(end),
        )
    return da.sel(**sel) if sel else da


def overlap_years(a: pd.Series, b: pd.Series) -> int:
    """Number of calendar years where both series have at least one value."""
    ya = set(pd.DatetimeIndex(a.dropna().index).year)
    yb = set(pd.DatetimeIndex(b.dropna().index).year)
    return len(ya & yb)
