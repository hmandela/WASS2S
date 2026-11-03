"""TelNet seasonal empirical-distribution adapter for WASS2S.

Architecture: Pinheiro & Ouarda (2025), doi:10.1038/s43247-025-02207-2.
This adapter preserves partial-convolution/variable-selection/LSTM decoding,
with compact fixed operational defaults. It changes the article's experiment:
all transformations, early stopping and refits are training-only; climatological
terciles are a separately declared, fixed event definition. No claimed score
reproduction or automatic hyperparameter search is made.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from .was_telnet_data import build_telnet_samples, purged_train_indices, _observed_monthly
from .was_probability_reference import climatological_terciles

__all__ = ['WAS_TelNet', 'observed_targets']
_SCHEMA = 'WASS2S-TelNet-1'


def _month_date(value):
    return pd.Timestamp(value).to_period('M').to_timestamp()


def _center_date(label, target_month=None):
    date = _month_date(label)
    if target_month is None:
        return date
    month = int(target_month)
    if not 1 <= month <= 12:
        raise ValueError('target_month must be the season centre month in 1..12')
    offset = (month - date.month) % 12
    if offset > 2:
        raise ValueError('Target labels must identify the season first or centre month; specify target_month accordingly')
    return date + pd.DateOffset(months=offset)


def _seasonal_observations(monthly, aggregation):
    obs, periods = _observed_monthly(monthly)
    full = pd.period_range(periods.min(), periods.max(), freq='M')
    data = obs.assign_coords(T=periods.to_timestamp()).reindex(T=full.to_timestamp())
    rolling = data.rolling(T=3, min_periods=3)
    seasonal = rolling.sum() if aggregation == 'sum' else rolling.mean()
    seasonal = seasonal.assign_coords(T=(full - 1).to_timestamp())
    seasonal.attrs.update(obs.attrs)
    seasonal.attrs['seasonal_aggregation'] = aggregation
    return seasonal


def observed_targets(monthly, target_times, target_month=None, aggregation='sum'):
    """Aggregate exact physical seasons while retaining WASS2S product labels."""
    labels = pd.DatetimeIndex(np.asarray(target_times))
    centers = [_center_date(t, target_month) for t in labels]
    seasonal = _seasonal_observations(monthly, aggregation)
    result = seasonal.reindex(T=pd.DatetimeIndex(centers)).assign_coords(T=labels)
    return result.transpose('T','Y','X')


def _safe_stats(values):
    valid = np.isfinite(values)
    count = valid.sum(axis=0)
    total = np.where(valid, values, 0.).sum(axis=0)
    mean = np.divide(total, count, out=np.zeros_like(total, dtype=float), where=count > 0)
    variance = np.divide(np.where(valid, (values - mean)**2, 0.).sum(axis=0),
                         count, out=np.zeros_like(total, dtype=float), where=count > 0)
    std = np.sqrt(variance)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.)
    return mean, std, count


def _unique_seasons(samples, positions):
    records = {}
    for variable, dates in [('history','history_start'), ('targets','target_start')]:
        values = np.asarray(samples[variable].isel(sample=positions).values, float)
        starts = np.asarray(samples[dates].isel(sample=positions).values)
        for n in range(len(positions)):
            for j in range(values.shape[1]):
                center = _month_date(starts[n,j]) + pd.DateOffset(months=1)
                records.setdefault(center, values[n,j])
    ordered = sorted(records)
    return np.stack([records[t] for t in ordered]), np.asarray([t.month-1 for t in ordered])


class WAS_TelNet:
    """Seasonal sequence forecaster with native PB/PN/PA probabilities.

    ``monthly_observations`` are physical monthly totals for PRCP, monthly
    means for TEMP. ``covariates`` are observed monthly indices (T,feature).
    Target product labels can use the first seasonal month; ``target_month``
    explicitly identifies the physical centre (JAS -> 8, DJF -> 1).

    ``outer``/``inner`` only record the requested residual-calibration policy:
    TelNet learns a native predictive ensemble and has no residual calibration
    subfolds. Fixed model parameters and purged training-only early stopping
    are independent of that setting.
    """
    _was_telnet = True

    def __init__(self, *, history_length=4, forecast_length=6, hidden_dim=32,
                 n_members=20, epochs=40, batch_size=8, patience=6,
                 learning_rate=1e-3, weight_decay=1e-4, dropout=.1, seed=42,
                 device='cpu', num_threads=1, variable='PRCP', aggregation=None,
                 observation_lag_months=0, min_samples=6, max_grid_cells=20000,
                 validation_fraction=.2, fair_crps=False, positive_support=None,
                 **kwargs):
        if kwargs:
            raise TypeError('Unknown TelNet options: '+', '.join(sorted(kwargs)))
        variable = str(variable).upper()
        if variable not in ('PRCP','TEMP'):
            raise ValueError('TelNet variable must be PRCP or TEMP')
        aggregation = ('sum' if variable == 'PRCP' else 'mean') if aggregation is None else aggregation
        if aggregation not in ('sum','mean'):
            raise ValueError('aggregation must be sum or mean')
        integers = dict(history_length=history_length, forecast_length=forecast_length,
                        hidden_dim=hidden_dim, n_members=n_members, epochs=epochs,
                        batch_size=batch_size, patience=patience, num_threads=num_threads,
                        min_samples=min_samples, max_grid_cells=max_grid_cells)
        for name, value in integers.items():
            if isinstance(value, (bool,np.bool_)) or int(value) != value or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if n_members < 2 or min_samples < 3:
            raise ValueError('TelNet needs at least two ensemble members and three training samples')
        if int(observation_lag_months) != observation_lag_months or observation_lag_months < 0:
            raise ValueError('observation_lag_months must be nonnegative')
        if not 0 <= dropout < 1 or learning_rate <= 0 or weight_decay < 0 or not 0 <= validation_fraction < .5:
            raise ValueError('Invalid dropout, learning rate, regularization or validation fraction')
        if device not in ('cpu','cuda','auto'):
            raise ValueError('device must be cpu, cuda or auto')
        config = dict(integers, learning_rate=float(learning_rate), weight_decay=float(weight_decay),
                      dropout=float(dropout), seed=int(seed), device=device, variable=variable,
                      aggregation=aggregation, observation_lag_months=int(observation_lag_months),
                      validation_fraction=float(validation_fraction), fair_crps=bool(fair_crps),
                      positive_support=variable=='PRCP' if positive_support is None else bool(positive_support))
        self.config = config
        for name,value in config.items(): setattr(self,name,value)
        self.network_ = None
        self.fit_diagnostics_ = {}
        self.cross_validation_diagnostics_ = None
        self.probability_calibration = 'outer'
        self.native_probability_model = True
        self._was_telnet = True

    def _build(self, monthly, covariates, **kwargs):
        obs, _ = _observed_monthly(monthly)
        if obs.sizes['Y']*obs.sizes['X'] > self.max_grid_cells:
            raise ValueError(f'TelNet dense empirical head exceeds max_grid_cells={self.max_grid_cells}; '
                             'regrid the monthly inputs to a declared coarser grid, or explicitly raise the limit')
        return build_telnet_samples(obs,covariates,history_length=self.history_length,
            forecast_length=self.forecast_length,aggregation=self.aggregation,
            observation_lag_months=self.observation_lag_months,**kwargs)

    @staticmethod
    def _complete(samples):
        # A date may contain a static land/ocean mask. Every lead must have
        # at least one observed cell and usable history/covariate information.
        targets=np.asarray(samples['targets'].values)
        history=np.asarray(samples['history'].values)
        cov=np.asarray(samples['covariates'].values)
        return (np.isfinite(targets).any(axis=(-1,-2)).all(axis=1)
                & np.isfinite(history).any(axis=(-1,-2)).all(axis=1)
                & np.isfinite(cov).any(axis=-1).all(axis=1))

    def _fit_scalers(self,samples,positions):
        raw, months = _unique_seasons(samples,positions)
        center,scale,count = _safe_stats(raw)
        means=[];scales=[]
        for month in range(12):
            part=raw[months==month]
            if len(part):
                m,s,n=_safe_stats(part)
                m=np.where(n>0,m,center);s=np.where(n>1,s,scale)
            else:m,s=center,scale
            means.append(m);scales.append(s)
        cov=np.asarray(samples['covariates'].isel(sample=positions).values,float)
        covmonth=np.asarray(samples['history_month'].isel(sample=positions).values,int)
        covcenter,covscale,covcount=_safe_stats(cov.reshape(-1,cov.shape[-1]))
        selected=np.where(covcount>=2)[0]
        if not len(selected):raise ValueError('No covariate has enough valid training values')
        cm=[];cs=[]
        for month in range(12):
            rows=cov[covmonth==month]
            if len(rows):
                m,s,n=_safe_stats(rows)
                m=np.where(n>0,m,covcenter);s=np.where(n>1,s,covscale)
            else:m,s=covcenter,covscale
            cm.append(m[selected]);cs.append(s[selected])
        mask=count>=3
        if not mask.any():raise ValueError('No target grid cell has at least three valid training seasons')
        return {'mean':np.asarray(means),'scale':np.asarray(scales),'mask':mask,
                'cov_mean':np.asarray(cm),'cov_scale':np.asarray(cs),'features':selected}

    @staticmethod
    def _tensors(samples,positions,scalers):
        lag=np.asarray(samples['history'].isel(sample=positions).values,float)
        cov=np.asarray(samples['covariates'].isel(sample=positions).values,float)[...,scalers['features']]
        hm=np.asarray(samples['history_month'].isel(sample=positions).values,int)
        fm=np.asarray(samples['target_month'].isel(sample=positions).values,int)
        history=(lag-scalers['mean'][hm])/scalers['scale'][hm]
        xcov=(cov-scalers['cov_mean'][hm])/scalers['cov_scale'][hm]
        mask=np.broadcast_to(scalers['mask'],(len(positions),)+scalers['mask'].shape).copy()
        mask &= np.isfinite(lag).all(axis=1)
        inputs=(np.nan_to_num(history,nan=0.,posinf=0.,neginf=0.).astype('float32'),
                np.nan_to_num(xcov,nan=0.,posinf=0.,neginf=0.).astype('float32'),hm,fm,mask)
        target=np.asarray(samples['targets'].isel(sample=positions).values,float)
        target=(target-scalers['mean'][fm])/scalers['scale'][fm]
        target=np.where(scalers['mask'][None,None],target,np.nan).astype('float32')
        return inputs,target

    def _new_network(self,samples,scalers):
        try:import torch
        except ImportError as exc:
            raise ImportError('TelNet requires optional PyTorch; install torch in the WASS2S environment') from exc
        from .was_telnet_network import TelNetNetwork
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            return TelNetNetwork(n_covariates=len(scalers['features']),height=samples.sizes['Y'],
                width=samples.sizes['X'],history_length=self.history_length,
                forecast_length=self.forecast_length,hidden_dim=self.hidden_dim,
                n_members=self.n_members,dropout=self.dropout)

    def _fit_samples(self,samples,positions):
        from .was_telnet_network import fit_network
        positions=np.asarray(positions,int)
        if len(positions)<self.min_samples:
            raise ValueError(f'TelNet has {len(positions)} eligible samples; at least {self.min_samples} are required')
        years=np.asarray(samples['sample'].dt.year)
        unique=np.unique(years[positions])
        pretrain=positions;validation=np.array([],int)
        if self.validation_fraction and len(unique)>=3:
            nval=max(1,int(np.ceil(len(unique)*self.validation_fraction)))
            validation=positions[np.isin(years[positions],unique[-nval:])]
            candidate=positions[~np.isin(years[positions],unique[-nval:])]
            pretrain=purged_train_indices(samples,validation,candidate,purge_history=True)
            if len(pretrain)<self.min_samples:
                pretrain=positions;validation=np.array([],int)
        scalers=self._fit_scalers(samples,pretrain)
        inputs,target=self._tensors(samples,pretrain,scalers)
        vi,vt=(None,None) if not len(validation) else self._tensors(samples,validation,scalers)
        if vt is not None:
            # Score stopping decisions on wholly future seasonal windows;
            # partially observed leading windows remain training targets.
            fully_future=np.asarray(samples['target_fully_future'].isel(sample=validation).values,bool)
            vt=np.where(fully_future[:,:,None,None],vt,np.nan)
        network=self._new_network(samples,scalers)
        options=dict(epochs=self.epochs,batch_size=self.batch_size,learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,patience=self.patience,seed=self.seed,device=self.device,
            num_threads=self.num_threads,fair_crps=self.fair_crps,
            latitude_weights=np.cos(np.deg2rad(np.asarray(samples['Y'].values,float))))
        network,history=fit_network(network,inputs,target,validation_inputs=vi,validation_targets=vt,**options)
        selected=max(1,int(history['best_epoch']))
        if len(validation):
            # Validation never supplies scaling or early-stopping training
            # targets. The selected epoch count is then refitted on all train.
            scalers=self._fit_scalers(samples,positions)
            inputs,target=self._tensors(samples,positions,scalers)
            network=self._new_network(samples,scalers)
            options.update(epochs=selected,patience=max(selected+1,self.patience))
            network,refit=fit_network(network,inputs,target,**options)
        else:refit=history
        self.network_=network;self.scalers_=scalers
        self.feature_names_=np.asarray(samples['feature'].values)[scalers['features']]
        self.grid_={'Y':np.asarray(samples['Y'].values),'X':np.asarray(samples['X'].values)}
        self.fit_diagnostics_={'training_sample_count':len(positions),
            'training_years':np.unique(years[positions]).astype(int).tolist(),
            'early_stopping_sample_count':len(validation),'early_stopping_training_count':len(pretrain),
            'selected_epochs':selected,'refit_on_all_training':bool(len(validation)),
            'history':history,'refit_history':refit,
            'training_issuances':[str(pd.Timestamp(t)) for t in samples['sample'].values[positions]],
            'training_target_end':str(pd.Timestamp(samples['target_end'].isel(sample=positions).max().item()))}
        self.fit_target_end_=pd.Timestamp(np.max(samples['target_end'].isel(sample=positions).values))
        return self

    def fit(self,monthly_observations,covariates,*,sample_times=None,availability_end=None):
        """Fit only complete targets, optionally restricted to issuance dates.

        ``availability_end`` is the last observed month permitted for labels.
        It is essential for an operational forecast with an archive containing
        later observations. With no cutoff, all completed archive targets fit.
        """
        samples=self._build(monthly_observations,covariates)
        eligible=self._complete(samples)
        if sample_times is not None:
            eligible &= samples.get_index('sample').isin(pd.DatetimeIndex(sample_times).to_period('M').to_timestamp())
        if availability_end is not None:
            eligible &= np.asarray(samples['target_end'].max('lead')<=np.datetime64(_month_date(availability_end)))
        self._fit_samples(samples,np.where(eligible)[0])
        return self

    def _predict_samples(self,samples,positions):
        if self.network_ is None:raise ValueError('Fit TelNet before predicting')
        for name in ('Y','X'):
            if not np.array_equal(self.grid_[name],samples[name].values):
                raise ValueError('Prediction grid differs from fitted TelNet grid')
        names=np.asarray(samples['feature'].values)[self.scalers_['features']]
        if not np.array_equal(names,self.feature_names_):raise ValueError('Covariate order/labels differ from fitted TelNet')
        raw_cov=np.asarray(samples['covariates'].isel(sample=positions).values,float)[...,self.scalers_['features']]
        if np.any(~np.isfinite(raw_cov).any(axis=-1)):
            raise ValueError('No observed covariates are available at a requested history step')
        inputs,_=self._tensors(samples,positions,self.scalers_)
        if np.any(~np.any(inputs[4],axis=(-1,-2))):
            raise ValueError('No usable lagged observation cells at a requested issuance')
        import torch
        network=self.network_;network.eval()
        device=next(network.parameters()).device
        members=[];weights=[]
        previous=torch.get_num_threads()
        try:
            torch.set_num_threads(self.num_threads)
            with torch.inference_mode():
                for start in range(0,len(positions),self.batch_size):
                    batch=[torch.as_tensor(a[start:start+self.batch_size],device=device) for a in inputs]
                    predicted,selection=network(*batch)
                    members.append(predicted.detach().cpu().numpy());weights.append(selection.detach().cpu().numpy())
        finally:torch.set_num_threads(previous)
        members=np.concatenate(members);weights=np.concatenate(weights)
        months=np.asarray(samples['target_month'].isel(sample=positions),int)
        members=(members*self.scalers_['scale'][months][:,:,None]+self.scalers_['mean'][months][:,:,None])
        if self.positive_support:members=np.maximum(members,0.)
        members=np.where(inputs[4][:,None,None],members,np.nan)
        return members,weights

    def _thresholds(self,monthly,month,start,end,obs_for_terciles=None):
        if obs_for_terciles is None:
            reference=_seasonal_observations(monthly,self.aggregation)
            reference=reference.where(reference['T'].dt.month==int(month),drop=True)
        else:
            reference=obs_for_terciles
            if 'M' in reference.dims:
                if reference.sizes['M']!=1:raise ValueError('Climatological reference M must be singleton')
                reference=reference.isel(M=0,drop=True)
            reference=reference.transpose('T','Y','X')
        q,prior=climatological_terciles(reference,start,end)
        return q,prior

    @staticmethod
    def _probabilities(members,q):
        low,high=q.isel(quantile=0,drop=True),q.isel(quantile=1,drop=True)
        members,low,high=xr.align(members,low,high,join='exact')
        valid=np.isfinite(members)
        den=valid.sum('member')
        p=xr.concat([((members<=low)&valid).sum('member')/den,
            ((members>low)&(members<=high)&valid).sum('member')/den,
            ((members>high)&valid).sum('member')/den],
            dim=xr.IndexVariable('probability',['PB','PN','PA']))
        return p.where((den>0)&np.isfinite(low)&np.isfinite(high)).transpose('probability','T','Y','X')

    def _metadata(self,prob,mode):
        from .was_probability_calibration import normalize_probability_calibration
        requested=normalize_probability_calibration(mode)
        prob.attrs.update(probability_calibration='native',probability_calibration_requested=requested,
            probability_calibration_effective='native_telnet_empirical_distribution',
            residual_calibration_applicable=0,calibration_strictly_nested=1,
            calibration='training-only TelNet empirical distribution; declared climatological events',
            tercile_reference='climatology',category_definition='PB: y <= low; PN: low < y <= high; PA: y > high',
            hyperparameter_selection='fixed configuration; purged training-only early stopping',
            model='TelNet WASS2S adaptation',telnet_schema=_SCHEMA)
        return prob

    def _full_dataset(self,samples,positions,members,weights,labels=None):
        labels=samples['sample'].values[positions] if labels is None else pd.DatetimeIndex(labels).values
        coords={'T':labels,'lead':samples['lead'].values,'member':np.arange(self.n_members),
                'Y':samples['Y'],'X':samples['X']}
        ds=xr.Dataset({'all_lead_members':xr.DataArray(members,dims=('T','lead','member','Y','X'),coords=coords),
            'selection_weights':xr.DataArray(weights,dims=('T','step','feature_selector'),
              coords={'T':labels,'step':np.arange(weights.shape[1]),
                      'feature_selector':['lagged_target']+list(map(str,self.feature_names_))})})
        ds['all_lead_members'].attrs.update(samples['targets'].attrs)
        for name in ('target_start','target_end','target_month_date','target_fully_future'):
            ds[name]=samples[name].isel(sample=positions).rename(sample='T').assign_coords(T=labels)
        ds['issuance_time']=xr.DataArray(samples['sample'].values[positions],dims='T',coords={'T':labels})
        ds.attrs.update(telnet_schema=_SCHEMA,reference='Pinheiro & Ouarda (2025), doi:10.1038/s43247-025-02207-2',
            implementation='WASS2S adaptation with compact fixed model, purged CV and empirical climatological terciles',
            empirical_members=self.n_members,aggregation=self.aggregation,variable=self.variable)
        return ds

    @staticmethod
    def _requested(samples,positions,centers):
        lead_dates=np.asarray(samples['target_month_date'].isel(sample=positions).values)
        selected=[]
        for i,date in enumerate(centers):
            hits=np.where(lead_dates[i]==np.datetime64(date,'ns'))[0]
            if len(hits)!=1:raise ValueError(f'Target season {date} is outside the {lead_dates.shape[1]} TelNet leads')
            if not bool(samples['target_fully_future'].isel(sample=int(positions[i]),lead=int(hits[0]))):
                raise ValueError('The scored target season is partly observed at issuance; choose an earlier initialization')
            selected.append(int(hits[0]))
        return np.asarray(selected,int)

    def cross_validate(self,monthly_observations,covariates,*,target_times,
                       initialization_month,target_month=None,clim_year_start=None,
                       clim_year_end=None,obs_for_terciles=None,gap_years=0,
                       probability_calibration='outer',splits=None,**kwargs):
        """Annual outer folds; all issuance months of held-out years are excluded.

        Training windows are additionally purged when their history or target
        months intersect any held-out target window. No residual subfolds are
        introduced for this native-distribution model.
        """
        # Generic CV controls may be routed by WASS2S; none trigger silent HPO.
        accepted={'probability_inner_folds','probability_inner_retune','nested_tuning','tercile_reference'}
        unknown=set(kwargs)-accepted
        if unknown:raise TypeError('Unknown TelNet CV options: '+', '.join(sorted(unknown)))
        if kwargs.get('nested_tuning',False):
            raise ValueError('TelNet hyperparameters are fixed; use explicit research configurations rather than requesting unsupported nested_tuning')
        if kwargs.get('tercile_reference','climatology')!='climatology':
            raise ValueError('TelNet uses fixed climatological terciles; training-fold events are not supported')
        labels=pd.DatetimeIndex(np.asarray(target_times));centers=[_center_date(t,target_month) for t in labels]
        if labels.has_duplicates or not labels.is_monotonic_increasing:
            raise ValueError('target_times must be unique and sorted')
        if not 1<=int(initialization_month)<=12 or int(gap_years)!=gap_years or gap_years<0:
            raise ValueError('Invalid initialization month or calendar-year gap')
        issues=[]
        for center in centers:
            issue=pd.Timestamp(center.year,int(initialization_month),1)
            if issue>center:issue=issue-pd.DateOffset(years=1)
            issues.append(issue)
        if len(set(issues))!=len(issues):raise ValueError('Use one target season per issuance in cross_validate')
        samples=self._build(monthly_observations,covariates)
        existing=set(samples['sample'].values)
        if any(np.datetime64(t,'ns') not in existing for t in issues):
            samples=self._build(monthly_observations,covariates,issuance_dates=sorted(
                existing.union(np.datetime64(t,'ns') for t in issues)))
        lookup={pd.Timestamp(t):i for i,t in enumerate(samples['sample'].values)}
        tests=np.asarray([lookup[t] for t in issues],int)
        leads=self._requested(samples,tests,centers)
        complete=self._complete(samples);years=np.asarray(samples['sample'].dt.year,int)
        folds=([(np.delete(np.arange(len(labels)),i),np.array([i])) for i in range(len(labels))]
               if splits is None else list(splits))
        results={};fullrecords=[];diagnostics=[]

        for outer_train,outer_test in folds:
            checked=[]
            for part in (outer_train,outer_test):
                array=np.asarray(part)
                if array.ndim!=1 or not np.issubdtype(array.dtype,np.integer) or len(np.unique(array))!=len(array) or np.any((array<0)|(array>=len(labels))):
                    raise ValueError('Outer splits require unique, in-range integer target positions')
                checked.append(array.astype(int))
            outer_train,outer_test=checked
            if not len(outer_test):continue
            if np.any(outer_test<0) or np.any(outer_test>=len(labels)) or np.intersect1d(outer_train,outer_test).size:
                raise ValueError('Invalid TelNet outer split positions')
            testyears=np.asarray([issues[p].year for p in outer_test],int)
            trainyears=np.asarray([issues[p].year for p in outer_train],int)
            distant=(np.abs(years[:,None]-testyears[None,:])>gap_years).all(axis=1)
            candidate=np.where(complete&np.isin(years,trainyears)&distant)[0]
            test_positions=tests[outer_test]
            train=purged_train_indices(samples,test_positions,candidate,purge_history=True)
            if len(train)<self.min_samples:
                raise ValueError(f'Insufficient TelNet training windows for outer test {list(labels[outer_test])}')
            fitted=type(self)(**self.config)
            fitted._fit_samples(samples,train)
            members,weights=fitted._predict_samples(samples,test_positions)
            full=fitted._full_dataset(samples,test_positions,members,weights,labels[outer_test])
            fullrecords.append(full)
            for j,pos in enumerate(outer_test):
                if int(pos) in results:raise ValueError('Outer verification dates must not be repeated')
                chosen=members[j,leads[pos]]
                ma=xr.DataArray(chosen[None],dims=('T','member','Y','X'),coords={
                    'T':[labels[pos]],'member':np.arange(self.n_members),'Y':samples['Y'],'X':samples['X']})
                ma.attrs.update(samples['targets'].attrs)
                q,_=self._thresholds(monthly_observations,centers[pos].month,clim_year_start,clim_year_end,obs_for_terciles)
                prob=self._probabilities(ma,q);det=ma.mean('member',skipna=True)
                truth=samples['targets'].isel(sample=int(tests[pos]),lead=int(leads[pos])).expand_dims(T=[labels[pos]])
                results[int(pos)]=(det,prob,ma,truth,q)
            diagnostics.append(dict(fitted.fit_diagnostics_,test_issuances=[str(issues[p]) for p in outer_test],
                                    training_positions=train.tolist()))
        template=observed_targets(monthly_observations,labels,target_month,self.aggregation)
        if not results:raise ValueError('No TelNet validation folds were produced')
        hd=xr.concat([results[p][0] for p in sorted(results)],dim='T').sortby('T').reindex(T=labels)
        hp=xr.concat([results[p][1] for p in sorted(results)],dim='T').sortby('T').reindex(T=labels)
        hp=self._metadata(hp,probability_calibration)
        full=xr.concat(fullrecords,dim='T').sortby('T').reindex(T=labels)
        full['members']=xr.concat([results[p][2] for p in sorted(results)],dim='T').sortby('T').reindex(T=labels)
        full['observation']=template
        full['deterministic']=hd;full['probabilities']=hp
        full['selected_lead']=xr.DataArray(leads+1,dims='T',coords={'T':labels})
        full.attrs.update(hp.attrs)
        self.hindcast_dataset_=full
        self.cross_validation_diagnostics_=diagnostics
        if self.network_ is None:
            self.fit_diagnostics_={'outer_folds':diagnostics,'operational_fit_required':True}
        return hd.transpose('T','Y','X'),hp.transpose('probability','T','Y','X')

    def forecast(self,monthly_observations,covariates,*,issue_time,target_time,
                 target_month=None,clim_year_start=None,clim_year_end=None,
                 obs_for_terciles=None,probability_calibration='outer',refit=True,**kwargs):
        """Forecast one actual issuance; no later observations enter fitting."""
        if kwargs:raise TypeError('Unknown TelNet forecast options: '+', '.join(sorted(kwargs)))
        issue=_month_date(issue_time);label=pd.Timestamp(target_time);center=_center_date(label,target_month)
        known=issue-pd.DateOffset(months=1+self.observation_lag_months)
        samples=self._build(monthly_observations,covariates,issuance_dates=[issue],availability_end=known)
        leads=self._requested(samples,np.array([0]),[center])
        if refit or self.network_ is None:
            self.fit(monthly_observations,covariates,availability_end=known)
        elif self.fit_target_end_>known:
            raise ValueError('Fitted TelNet has seen observations unavailable at this issuance; refit with the proper cutoff')
        members,weights=self._predict_samples(samples,np.array([0]))
        full=self._full_dataset(samples,np.array([0]),members,weights,[label])
        ma=full['all_lead_members'].isel(lead=int(leads[0]),drop=True)
        q,_=self._thresholds(monthly_observations,center.month,clim_year_start,clim_year_end,obs_for_terciles)
        det=ma.mean('member',skipna=True);prob=self._metadata(self._probabilities(ma,q),probability_calibration)
        full['members']=ma;full['deterministic']=det;full['probabilities']=prob
        full['selected_lead']=xr.DataArray(leads+1,dims='T',coords={'T':[label]})
        full.attrs.update(prob.attrs)
        self.forecast_dataset_=full
        return det.transpose('T','Y','X'),prob.transpose('probability','T','Y','X')

    @staticmethod
    def observed_targets(monthly,target_times,target_month=None,aggregation='sum'):
        return observed_targets(monthly,target_times,target_month,aggregation)

    def save(self,path):
        """Save a tensor-only checkpoint and explicit config/scaler metadata."""
        if self.network_ is None:raise ValueError('No fitted TelNet model to save')
        import torch
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        payload={'schema':_SCHEMA,'config_json':json.dumps(self.config),
            'state_dict':{k:v.detach().cpu() for k,v in self.network_.state_dict().items()},
            'scalers':{k:torch.as_tensor(v) for k,v in self.scalers_.items()},
            'feature_names':[str(v) for v in self.feature_names_],
            'Y':torch.as_tensor(self.grid_['Y'].copy()),'X':torch.as_tensor(self.grid_['X'].copy()),
            'fit_target_end':str(self.fit_target_end_),
            'fit_diagnostics_json':json.dumps(self.fit_diagnostics_,default=float)}
        temp=path.with_suffix(path.suffix+'.tmp');torch.save(payload,temp);temp.replace(path)
        return path

    @classmethod
    def load(cls,path,*,device=None):
        import torch
        payload=torch.load(Path(path),map_location='cpu',weights_only=True)
        if payload.get('schema')!=_SCHEMA:raise ValueError('Unsupported TelNet checkpoint schema')
        config=json.loads(payload['config_json'])
        if device is not None:config['device']=device
        model=cls(**config)
        model.scalers_={k:v.cpu().numpy() for k,v in payload['scalers'].items()}
        model.feature_names_=np.asarray(payload['feature_names'])
        model.grid_={k:payload[k].cpu().numpy() for k in ('Y','X')}
        stub=xr.Dataset(coords={'Y':model.grid_['Y'],'X':model.grid_['X']})
        model.network_=model._new_network(stub,model.scalers_)
        model.network_.load_state_dict(payload['state_dict'])
        requested=model.device
        resolved=('cuda' if torch.cuda.is_available() else 'cpu') if requested=='auto' else requested
        if resolved=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
        model.network_.to(resolved).eval()
        model.fit_target_end_=pd.Timestamp(payload['fit_target_end'])
        model.fit_diagnostics_=json.loads(payload['fit_diagnostics_json'])
        return model
