"""wass2s — West Africa Seasonal Forecasting System.

This package exposes the full WASS2S modelling pipeline under a single
namespace.  Every public symbol from the sub-modules listed below is
re-exported here so that users only need::

    from wass2s import *

or::

    import wass2s as w2s

Sub-modules
-----------
was_verification
    Deterministic and probabilistic verification scores (Pearson, Spearman,
    GROC, RPSS, Brier, NSE, KGE, Taylor diagrams, …).
was_analog
    Analog-ensemble seasonal forecasting (SOM, correlation-, PCA-, KMeans-
    and agglomerative-clustering methods).
was_cca
    Canonical Correlation Analysis (CCA) with xeofs — legacy ``WAS_CCA``
    and leakage-free ``WAS_CCA_base``.
was_cca_2
    Kernel CCA variants (``WAS_KCCA``, ``WAS_KGCCA``, ``WAS_EOF_KCCA``)
    built on cca-zoo.
was_compute_predictand
    Agroclimatic predictand computation: onset, cessation, and dry-spell
    variants (``WAS_compute_onset``, ``WAS_compute_cessation``, …).
was_download
    ERA5, CHIRPS, TAMSAT, NMME, and agro-indicator download utilities
    (``WAS_Download``).
was_eof
    Leakage-free EOF analysis with detrending, NaN-filling, and optional
    per-field normalization (``WAS_EOF``).
was_linear_models
    Linear regression wrappers: OLS, Ridge, Lasso, LassoLars, ElasticNet,
    Polynomial, Logistic, and Poisson models.
was_machine_learning
    Machine-learning wrappers: SVR, Random Forest, XGBoost, MLP, stacking
    ensembles, and optimizer utilities.
was_merge_predictand
    Station–gridded merging methods: kriging, regression-kriging, conditional
    merging, optimal interpolation, Barnes interpolation, Random Forest and
    XGBoost merging (``WAS_Merging``, ``WAS_Merging_``).
was_mme
    Multi-model ensemble methods: weighted averaging, probabilistic weighting,
    BMA variants, ELM, ELR, logistic, Gaussian process, and xcast wrappers.
was_pcr
    Principal Component Regression wrapping any WAS regression model
    (``WAS_PCR``).
utils
    Shared utilities: standardization, anomalization, detrending, index
    computation, predictor loading, shapefile retrieval, and forecast plotting.
was_cross_validate
    Leakage-free leave-one-out cross-validator for all model types
    (``WAS_Cross_Validator``, ``CustomTimeSeriesSplit``).
was_transformdata
    Data transformation and skewness handling: Box-Cox, Yeo-Johnson,
    distribution fitting, and quantile mapping (``WAS_TransformData``).
was_bias_correction
    Quantile-mapping bias correction suite (``WAS_Qmap``, ``WAS_bias_correction``).
was_seasonal_analysis
    ERA5-based seasonal diagnostic tools, Hovmöller plots, and interactive
    C3S / BOM viewers (``C3SViewer``, ``BOMViewer``).
ceac_agro
    CEAC agroclimatic indicators: onset, cessation, dry/wet spells for both
    station and gridded data (``CEAC_compute_onset``, ``CEAC_compute_cessation``, …).
"""

__version__ = "0.4.8.5"

# Backward-compatibility shim: scipy removed scipy.interp in 1.14.
import numpy as _np
import scipy as _scipy
if not hasattr(_scipy, "interp"):
    _scipy.interp = _np.interp

from wass2s.was_verification import *
from wass2s.was_analog import *
from wass2s.was_cca import *
from wass2s.was_compute_predictand import *
from wass2s.was_download import *
from wass2s.was_eof import *
from wass2s.was_linear_models import *
from wass2s.was_machine_learning import *
from wass2s.was_merge_predictand import *
from wass2s.was_mme import *
from wass2s.was_pcr import *
from wass2s.utils import *
from wass2s.was_cross_validate import *
from wass2s.was_transformdata import *
from wass2s.was_bias_correction import *
from wass2s.was_seasonal_analysis import *
from wass2s.was_cca_2 import *
from wass2s.ceac_agro import *
