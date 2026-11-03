"""Bias and probability calibration, imported only when explicitly requested."""
from .._lazy_api import resolve as _resolve, names as _names

_EXPORTS = {
    "WAS_PBC": "was_pbc",
    "run_pbc_seasonal": "was_pbc_cv",
    "build_pbc_seasonal_features": "was_pbc_data",
    "climatological_terciles": "was_probability_reference",
    "WAS_Qmap": "was_bias_correction",
    "WAS_Qmap_CV": "was_bias_correction",
    "WAS_mme_ELR": "was_mme",
    "WAS_mme_NGR_Gaussian": "was_mme",
    "WAS_mme_NGR_NonGaussian": "was_mme",
    "WAS_mme_FastBMA": "was_mme",
    "WAS_mme_FullBMA": "was_mme",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name):
    return _resolve(globals(), _EXPORTS, name)


def __dir__():
    return _names(globals(), _EXPORTS)
