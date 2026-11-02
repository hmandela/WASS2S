"""Out-of-sample evaluation at the SEASONAL AGGREGATE scale.

Calendars are frozen here. Results are explicitly conditional on this catalogue,
not a claimed end-to-end test of calendar estimation. To test the whole workflow,
fit the catalogue in an independent reference period or refit build_highflow_
predictand inside outer space/time folds. No observed validation is invented in
GloFAS-only mode. The legacy DAILY CV is not called by these functions.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import numpy as np
import pandas as pd
if __package__:
    from .highflow_predictand import PredictandConfig,_fit_model,_apply_model
    from .glofas_highflow_bias_correction import _hydro_metrics
else:
    from highflow_predictand import PredictandConfig,_fit_model,_apply_model
    from glofas_highflow_bias_correction import _hydro_metrics


@dataclass(frozen=True)
class SeasonalEvaluationConfig:
    methods: tuple[str,...] = ('mean_ratio','log_ratio','eqm')
    strategy: str = 'expanding'             # expanding, leave_one_year_out
    min_training_years: int = 10
    test_years: tuple[int,...] | None = None
    purge_days: int = 0


def evaluate_seasonal_correction(reference, config=None):
    """Refit correction per held-out year using one mean/max pair per year.

    The result's calibration period limits eligible TRAINING years. Held-out
    years can also lie outside that period. Absolute start/end dates protect
    cross-calendar-year windows. Expanding uses only past complete windows.

    At least two predictions are needed for temporal correlation, and KGE/NSE
    retain their degeneracy checks. Scores use exactly paired finite OBS/RAW/
    CORRECTED values; the baseline is not evaluated on an easier sample.
    """
    if reference.get('correction_scale') == 'monthly_daily':
        raise ValueError('This evaluator refits SEASONAL factors, not monthly daily factors. '
            'Use explicit outer folds of build_highflow_predictand with correction=monthly_ratio, '
            'MonthlyBiasConfig(temporal_mode=blocked, excluded_periods=..., '
            'window_protocol=refit_excluding), then compare held-out seasonal values on common days. '
            'Do not label seasonal-ratio CV as monthly correction CV.')
    ec=config or SeasonalEvaluationConfig()
    if ec.strategy not in {'expanding','leave_one_year_out'}:raise ValueError('Unknown seasonal CV strategy')
    if ec.min_training_years<2 or ec.purge_days<0:raise ValueError('Invalid seasonal CV controls')
    if any(m not in {'mean_ratio','log_ratio','eqm'} for m in ec.methods):raise ValueError('Unknown correction method')
    info=dict(evaluation_scale='seasonal_aggregate',calendar_validation='fixed_conditional',
              strategy=ec.strategy,calendar_source=reference.get('resolved_reference_source'))
    pairs=reference.get('calibration_pairs',pd.DataFrame())
    if pairs.empty:
        return dict(info,status='UNAVAILABLE_WITHOUT_OBSERVATIONAL_PAIRS',scores=pd.DataFrame(),
                    predictions=pd.DataFrame(),folds=pd.DataFrame())
    cfg=reference.get('config',PredictandConfig())
    cfg=replace(cfg,min_training_years=ec.min_training_years)
    records=[];folds=[]
    for (station,key),part in pairs.groupby(['station','window_key'],sort=False):
        part=part.sort_values('year').copy()
        if part.year.duplicated().any():raise ValueError('Repeated seasonal calibration pair')
        part['observed']=part.observed_all_years
        part['glofas']=part.glofas_all_years
        tests=part[np.isfinite(part.observed)&np.isfinite(part.glofas)]
        if ec.test_years is not None:tests=tests[tests.year.isin(ec.test_years)]
        for _,test in tests.iterrows():
            lo=pd.Timestamp(test.start_date)-pd.Timedelta(days=ec.purge_days)
            hi=pd.Timestamp(test.end_date)+pd.Timedelta(days=ec.purge_days)
            eligible=part.in_calibration_period.fillna(False).astype(bool)
            eligible &= (part.end_date<lo)|(part.start_date>hi)
            if ec.strategy=='expanding':eligible &= part.end_date<lo
            train=part[eligible]
            for method in ec.methods:
                model=_fit_model(train,cfg,method)
                audit=dict(station=station,window_key=key,year=int(test.year),method=method,
                           status=model['status'],n_training_years=model['n_years'],
                           test_start=test.start_date,test_end=test.end_date,
                           training_years=','.join(map(str,model.get('training_years',[]))))
                folds.append(audit)
                if model['status']!='OK':continue
                pred=float(_apply_model(np.asarray([test.glofas],float),model)[0])
                records.append(dict(station=station,window_key=key,year=int(test.year),method=method,
                                    observed=float(test.observed),raw=float(test.glofas),corrected=pred,
                                    coverage=float(test.coverage),start_date=test.start_date,end_date=test.end_date))
    predictions=pd.DataFrame(records);scores=[]
    if len(predictions):
        for (station,key,method),p in predictions.groupby(['station','window_key','method']):
            raw=_hydro_metrics(p.observed.values,p.raw.values)
            cor=_hydro_metrics(p.observed.values,p.corrected.values)
            row=dict(station=station,window_key=key,method=method,n_test_years=len(p),
                     calendar_validation='fixed_conditional',strategy=ec.strategy)
            row.update({'raw_'+k:v for k,v in raw.items()});row.update({'corrected_'+k:v for k,v in cor.items()})
            # Legacy metrics use lowercase names. Do not divide by zero when
            # RAW already reproduces OBS exactly.
            for metric in ('rmse','mae'):
                denom=raw.get(metric,np.nan)
                row[metric+'_skill']=1-cor[metric]/denom if np.isfinite(denom) and denom>0 else np.nan
            row['kge_gain']=cor.get('kge',np.nan)-raw.get('kge',np.nan)
            scores.append(row)
    return dict(info,status='OK' if len(predictions) else 'NO_ELIGIBLE_TEST_FOLD',
                scores=pd.DataFrame(scores),predictions=predictions,folds=pd.DataFrame(folds))
