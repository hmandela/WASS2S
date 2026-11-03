"""Shared, explicit information boundary for matching, regimes and calibration.

No training period is inferred from forecast verification data. Dates are daily
Gregorian dates; excluded intervals are inclusive and information_cutoff is
exclusive. A frozen context identifies one external experiment/fold, NOT one
month-by-month moving pixel location.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import json
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MatchingTrainingContext:
    training_start: str | None = None
    training_end: str | None = None
    information_cutoff: str | None = None
    excluded_periods: tuple[tuple[str, str], ...] = ()
    excluded_stations: tuple[str, ...] = ()
    gap_days: int = 0
    label: str = 'fixed_reference'

    def __post_init__(self):
        if int(self.gap_days) != self.gap_days or self.gap_days < 0:
            raise ValueError('gap_days must be a nonnegative integer')
        for val in (self.training_start, self.training_end, self.information_cutoff):
            if val is not None:
                t = pd.Timestamp(val)
                if pd.isna(t) or t.tzinfo is not None or t != t.normalize():
                    raise ValueError('Context dates must be finite timezone-naive day labels')
        if self.training_start and self.training_end:
            if pd.Timestamp(self.training_start) > pd.Timestamp(self.training_end):
                raise ValueError('training_start is after training_end')
        for a, b in self.excluded_periods:
            a, b = pd.Timestamp(a), pd.Timestamp(b)
            if pd.isna(a) or pd.isna(b) or a.tzinfo or b.tzinfo or a > b:
                raise ValueError('Invalid excluded period')
            if a != a.normalize() or b != b.normalize():
                raise ValueError('Excluded periods use daily date labels')
        if not self.label:
            raise ValueError('A context label is required')

    def mask(self, index) -> np.ndarray:
        i = pd.DatetimeIndex(index)
        if i.tz is not None:
            raise ValueError('Use timezone-naive hydrological dates')
        m = np.asarray(~i.isna(), dtype=bool)
        if self.training_start is not None:
            m &= i >= pd.Timestamp(self.training_start)
        if self.training_end is not None:
            m &= i <= pd.Timestamp(self.training_end)
        if self.information_cutoff is not None:
            m &= i < pd.Timestamp(self.information_cutoff)
        delta = pd.Timedelta(days=self.gap_days)
        for a, b in self.excluded_periods:
            m &= ~((i >= pd.Timestamp(a)-delta) & (i <= pd.Timestamp(b)+delta))
        return m

    def filter_series(self, series, station=None):
        if station is not None and str(station) in set(map(str, self.excluded_stations)):
            return series.where(False)
        return series.where(self.mask(series.index))

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, default=str).encode()).hexdigest()

    def as_dict(self):
        return dict(asdict(self), fingerprint=self.fingerprint)
