"""Seasonal prediction: curated lazy aliases of the historical WASS2S API."""
from .._lazy_api import resolve as _resolve, names as _names

_EXPORTS = {
    "WAS_CCA_op": "was_cca",
    "WAS_CCA_Regularized": "was_cca",
    "WAS_KCCA": "was_cca_2",
    "WAS_KGCCA": "was_cca_2",
    "WAS_EOF_KCCA": "was_cca_2",
    "WAS_EOF": "was_eof",
    "WAS_PCR": "was_pcr",
    "WAS_LinearRegression_Model": "was_linear_models",
    "WAS_Ridge_Model": "was_linear_models",
    "WAS_RandomForest_Model": "was_machine_learning",
    "WAS_XGBoost_Model": "was_machine_learning",
    "WAS_Analog": "was_analog",
    "WAS_mme_Ridge": "was_mme",
    "WAS_mme_RF": "was_mme",
    "WAS_mme_Kharin2017": "was_mme",
    "WAS_TelNet": "was_telnet",
    "WAS_Cross_Validator": "was_cross_validate",
    "WAS_CrossFit": "was_crossfit",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name):
    return _resolve(globals(), _EXPORTS, name)


def __dir__():
    return _names(globals(), _EXPORTS)
