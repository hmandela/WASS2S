"""Agroclimatic indicators; historical module paths remain valid."""
from .._lazy_api import resolve as _resolve, names as _names

_EXPORTS = {
    "WAS_compute_onset": "was_compute_predictand",
    "WAS_compute_cessation": "was_compute_predictand",
    "WAS_compute_onset_dry_spell": "was_compute_predictand",
    "WAS_compute_cessation_dry_spell": "was_compute_predictand",
    "WAS_count_dry_spells": "was_compute_predictand",
    "WAS_count_wet_spells": "was_compute_predictand",
    "WAS_PrecipIndices": "was_compute_predictand",
    "WAS_HeatWaveIndices": "was_compute_predictand",
    "CAF_AgroClimateBase": "ceac_agro",
    "CEAC_compute_onset": "ceac_agro",
    "CEAC_compute_cessation": "ceac_agro",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name):
    return _resolve(globals(), _EXPORTS, name)


def __dir__():
    return _names(globals(), _EXPORTS)
