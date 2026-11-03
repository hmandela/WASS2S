"""Hydrological facade: high-flow targets, GloFAS matching and corrections.

These are lazy aliases of the historical flat modules, not new algorithms.
"""
from .._lazy_api import resolve as _resolve, names as _names

_EXPORTS = {
    "WAS_Download": "was_download",
    "WAS_Hydro": "WAS_Hydro",
    "WAS_HYDRO_VERSION": "WAS_Hydro",
    "HighWaterConfig": "highwater_season_predictands_modeaware",
    "ClimatologyPeriod": "highflow_periods",
    "PredictandConfig": "highflow_predictand",
    "build_highflow_predictand": "highflow_predictand",
    "aggregate_with_reference": "highflow_predictand",
    "AggregationConfig": "highflow_outlet_aggregation",
    "OutletAggregation": "highflow_outlet_aggregation",
    "GridWindowConfig": "highflow_window_regionalization",
    "regionalize_highflow_windows": "highflow_window_regionalization",
    "TopKrigingConfig": "highflow_topkriging",
    "CatchmentSupport": "highflow_topkriging",
    "MonthlyBiasConfig": "highflow_monthly_bias",
    "MonthlyFactorEstimator": "highflow_monthly_bias",
    "SeasonalEvaluationConfig": "highflow_seasonal_evaluation",
    "RegimeMatchingConfig": "glofas_matching_pixel",
    "GloFASMatchingResult": "glofas_matching_pixel",
    "LDDNetworkConfig": "glofas_static_network",
    "GloFASStaticNetwork": "glofas_static_network",
    "MatchingTrainingContext": "glofas_training_context",
    "plot_matching_map": "glofas_matching_review",
    "plot_matching_regimes": "glofas_matching_review",
    "export_matching_review": "glofas_matching_review",
    "review_template": "glofas_matching_review",
    "save_glofas_matching": "glofas_matching_storage",
    "load_glofas_matching": "glofas_matching_storage",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name):
    return _resolve(globals(), _EXPORTS, name)


def __dir__():
    return _names(globals(), _EXPORTS)
