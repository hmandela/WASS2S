__version__ = "0.2.19"
# --- SciPy interp patch for legacy dependencies ---
import numpy as _np
import scipy as _scipy
if not hasattr(_scipy, "interp"):
    _scipy.interp = _np.interp
# -------------------------------------------------
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
# from wass2s.was_hydro_processing import *

