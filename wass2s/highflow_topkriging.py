"""Support-aware ordinary kriging for high-flow attributes.

References: Skøien, Merz & Blöschl (2006), doi:10.5194/hess-10-277-2006;
Skøien & Blöschl (2007), doi:10.1029/2006WR005760.

For a point semivariogram gamma_p, J_AB = sum_i sum_j w_Ai w_Bj gamma_p(d_ij)
is a *double integral*, not the semivariogram of areal means. The latter is
Gamma_AB = J_AB - (J_AA+J_BB)/2. Deconvolution fits Gamma to empirical
semivariances. The ordinary-kriging system can equivalently use J, including
its diagonal, provided the prediction variance subtracts the target J_00.

HydroBASINS supports retain ALL incremental sub-basins upstream of an outlet
and share the same quadrature for shared areas. No per-catchment thinning is
allowed. Equivalent-area discs are an explicitly labelled block approximation,
not a claim of true nested top-kriging. Date and log-factor applications are
empirical statistical extensions: neither is assumed to be an areal average,
and independent validation is required. No routing/mass balance is enforced.

Pure module: importing it performs no computation, command-line parsing or I/O.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence
import numpy as np
import pandas as pd

__all__=['TopKrigingConfig','CatchmentSupport','build_supports_from_points',
         'build_supports_from_hybas','point_variogram','regularized_variogram_matrix',
         'areal_semivariogram_matrix','fit_point_variogram','topkriging_weights',
         'topkriging_predict','topkriging_cross_validate','propagate_attribute_topkriging']

@dataclass
class TopKrigingConfig:
    model: str='exponential'
    n_discretisation: int=12  # disc quadrature only; NEVER truncate nested supports
    max_donors: int=10
    points_per_subbasin: int=1  # increase to 3/8/16 and assess quadrature convergence
    fit_nugget: bool=True  # point microscale nugget, NOT gauge measurement error
    min_pairs_for_fit: int=6
    allow_negative_weights: bool=True
    max_range_factor: float=2.
    condition_jitter: float=1e-9
    power_exponent: float=1.0  # dimensionless, separate from range_km
    fit_power_exponent: bool=True
    # Fixed default reference, shared even across separate constructor calls.
    # A different frame must be supplied consistently to donors AND targets.
    frame_lon0: float=0.
    frame_lat0: float=0.

    def __post_init__(self):
        if self.model not in {'exponential','spherical','power'}:raise ValueError('Unknown variogram model.')
        if not 0<self.power_exponent<2:raise ValueError('power_exponent must be in (0,2).')
        if self.n_discretisation<1 or self.max_donors<1 or self.points_per_subbasin<1:raise ValueError('Quadrature/donor counts must be positive.')
        if not -90<=self.frame_lat0<=90:raise ValueError('Invalid projection latitude.')
        if self.condition_jitter<0 or self.min_pairs_for_fit<1:raise ValueError('Invalid numerical controls.')

@dataclass
class CatchmentSupport:
    name: str
    x: np.ndarray
    y: np.ndarray
    w: np.ndarray
    area_km2: float
    frame_id: str='user_supplied_common_km'
    support_kind: str='user_supplied'
    measurement_variance: float=0.  # independent gauge error, attribute squared
    quadrature_ids: tuple=()

    def __post_init__(self):
        self.x=np.asarray(self.x,dtype=float).ravel();self.y=np.asarray(self.y,dtype=float).ravel()
        self.w=np.asarray(self.w,dtype=float).ravel()
        if not len(self.x) or len(self.x)!=len(self.y) or len(self.x)!=len(self.w):
            raise ValueError('Support x,y,w must have equal, nonzero lengths.')
        if not (np.isfinite(self.x).all() and np.isfinite(self.y).all() and np.isfinite(self.w).all()):
            raise ValueError('Nonfinite support coordinate or weight.')
        if (self.w<0).any() or self.w.sum()<=0 or not np.isfinite(self.area_km2) or self.area_km2<=0:
            raise ValueError('Positive total support area/weight is required.')
        if self.measurement_variance<0 or not np.isfinite(self.measurement_variance):
            raise ValueError('Measurement variance must be finite and nonnegative.')

    def normalised_weights(self):return self.w/self.w.sum()

def _frame(cfg):return f'AEQD_WGS84_lon{cfg.frame_lon0:.12g}_lat{cfg.frame_lat0:.12g}'

def _to_local_km(lon,lat,lon0,lat0):
    """Metric azimuthal-equidistant frame shared by all supports.

    Distances away from the projection centre are approximate. For large/global
    domains, use regional fits with appropriate frames and report that choice.
    """
    from pyproj import CRS,Transformer
    crs=CRS.from_proj4(f'+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m')
    tx=Transformer.from_crs(4326,crs,always_xy=True)
    x,y=tx.transform(np.asarray(lon,dtype=float).tolist(),np.asarray(lat,dtype=float).tolist())
    return np.asarray(x)/1000.,np.asarray(y)/1000.

def _common_frame(supports):
    if len({s.frame_id for s in supports})>1:
        raise ValueError('Supports use different coordinate frames. Build all with the same TopKrigingConfig.')

def build_supports_from_points(names,lon,lat,area_km2,config=None):
    """Explicit equivalent-area disc approximation; does not preserve topology."""
    cfg=config or TopKrigingConfig();lon=np.asarray(lon,float);lat=np.asarray(lat,float);area=np.asarray(area_km2,float)
    if not len(names)==len(lon)==len(lat)==len(area):raise ValueError('Unequal metadata lengths.')
    if len(set(map(str,names)))!=len(names):raise ValueError('Support names must be unique.')
    if not (np.isfinite(lon).all() and np.isfinite(lat).all() and np.isfinite(area).all() and (area>0).all()):
        raise ValueError('Explicit finite coordinates and positive upstream areas are required; no invented area.')
    cx,cy=_to_local_km(lon,lat,cfg.frame_lon0,cfg.frame_lat0);out={};n=cfg.n_discretisation
    i=np.arange(n)+.5;th=i*math.pi*(3-math.sqrt(5))
    for k,name in enumerate(names):
        if n==1:px,py=np.array([cx[k]]),np.array([cy[k]])
        else:
            r=math.sqrt(area[k]/math.pi)*np.sqrt(i/n)
            px,py=cx[k]+r*np.cos(th),cy[k]+r*np.sin(th)
        out[str(name)]=CatchmentSupport(str(name),px,py,np.full(n,area[k]/n),float(area[k]),
                                       frame_id=_frame(cfg),support_kind='equivalent_area_disc')
    return out

def build_supports_from_hybas(targets,hybas,*,id_col='HYBAS_ID',next_down_col='NEXT_DOWN',
                              names=None,config=None,points_per_subbasin=None):
    """Quadrature of upstream unions, sharing incremental supports exactly.

    ``n_discretisation`` does not cap this function. Increase
    ``points_per_subbasin`` and check numerical convergence. Geometry and full
    upstream coverage are required for physical supports. A table with lon/lat
    and SUB_AREA is accepted as a labelled representative-point approximation.
    """
    cfg=config or TopKrigingConfig();g=hybas.copy()
    if points_per_subbasin is None:points_per_subbasin=cfg.points_per_subbasin
    if id_col not in g or next_down_col not in g:raise ValueError('HYBAS_ID and NEXT_DOWN are required.')
    if g[id_col].duplicated().any():raise ValueError('Duplicate HYBAS_ID.')
    if int(points_per_subbasin)<1:raise ValueError('points_per_subbasin must be positive.')
    import networkx as nx
    ids=list(g[id_col]);idset=set(ids);G=nx.DiGraph();G.add_nodes_from(ids)
    for hid,nd in zip(ids,g[next_down_col]):
        if pd.notna(nd) and nd in idset:G.add_edge(hid,nd)
    if not nx.is_directed_acyclic_graph(G):raise ValueError('HydroBASINS topology contains a cycle.')
    labels=list(names) if names is not None else list(map(str,targets))
    if len(labels)!=len(targets) or len(set(labels))!=len(labels):raise ValueError('Invalid target names.')
    area=pd.to_numeric(g.get('SUB_AREA',pd.Series(np.nan,index=g.index)),errors='coerce').to_numpy(float)
    geometry='geometry' in g.columns
    if geometry:
        if getattr(g,'crs',None) is None:raise ValueError('Basin CRS is required; do not silently assume WGS84.')
        geom=g.to_crs(4326);projected=g.to_crs(6933)
        measured=projected.geometry.area.to_numpy()/1e6
        area=np.where(np.isfinite(area)&(area>0),area,measured)
        reps=geom.geometry.representative_point();lon=reps.x.to_numpy();lat=reps.y.to_numpy()
    else:
        if not {'lon','lat'}.issubset(g):raise ValueError('Provide polygons or explicit lon/lat representative points.')
        lon=g['lon'].to_numpy(float);lat=g['lat'].to_numpy(float)
    if not (np.isfinite(area).all() and (area>0).all()):raise ValueError('Missing incremental sub-basin area.')
    # Each incremental polygon gets its own deterministic quadrature ONCE.
    # All downstream supports reuse it. No independent thinning can break nesting.
    incremental={}
    for j,hid in enumerate(ids):
        lons,lats=[lon[j]],[lat[j]]
        if geometry and points_per_subbasin>1:
            from shapely.geometry import Point
            from pyproj import Transformer
            poly=projected.geometry.iloc[j];xmin,ymin,xmax,ymax=poly.bounds
            # Quasi-uniform deterministic rejection sampling in equal-area space.
            from scipy.stats import qmc
            sampler=qmc.Halton(2,scramble=False)
            pts=[];attempt=0
            while len(pts)<points_per_subbasin and attempt<10000:
                xy=sampler.random(128);attempt+=128
                for xx,yy in xy:
                    xx=xmin+xx*(xmax-xmin);yy=ymin+yy*(ymax-ymin)
                    if poly.covers(Point(xx,yy)):pts.append((xx,yy))
                    if len(pts)>=points_per_subbasin:break
            if len(pts)<points_per_subbasin:
                raise ValueError(f'Cannot discretise polygon {hid}; reduce points_per_subbasin or repair geometry.')
            tx=Transformer.from_crs(6933,4326,always_xy=True)
            lons,lats=tx.transform(*np.asarray(pts).T)
        px,py=_to_local_km(lons,lats,cfg.frame_lon0,cfg.frame_lat0)
        incremental[hid]=(np.ravel(px),np.ravel(py),np.full(len(np.ravel(px)),area[j]/len(np.ravel(px))))
    out={}
    for label,root in zip(labels,targets):
        if root not in idset:raise KeyError(f'Unknown HYBAS_ID {root}.')
        members=nx.ancestors(G,root)|{root};members=[hid for hid in ids if hid in members]
        px=np.concatenate([incremental[m][0] for m in members]);py=np.concatenate([incremental[m][1] for m in members])
        weights=np.concatenate([incremental[m][2] for m in members]);total=float(weights.sum())
        if 'UP_AREA' in g:
            reported=float(g.loc[g[id_col]==root,'UP_AREA'].iloc[0])
            if np.isfinite(reported) and reported>0 and total<.9*reported:
                raise ValueError(f'Incomplete upstream support for {root}: {total:g} versus UP_AREA={reported:g} km2. Supply upstream closure.')
        out[str(label)]=CatchmentSupport(str(label),px,py,weights,total,frame_id=_frame(cfg),
                     support_kind='hydrobasins_upstream_quadrature' if geometry else 'incremental_representative_points',
                     quadrature_ids=tuple((m,k) for m in members for k in range(len(incremental[m][0]))))
    return out

def point_variogram(h,model,nugget,sill,rng,power_exponent=1.):
    """Point semivariogram: sill is partial sill; power has a separate exponent.

    For power: gamma(h)=nugget+sill*(h/1 km)**power_exponent (h>0).
    ``rng`` is ignored for power, never repurposed as an exponent.
    """
    h=np.asarray(h,float)
    if (h<0).any() or nugget<0 or sill<0:raise ValueError('Negative distance/variogram parameter.')
    if model=='exponential':g=sill*(-np.expm1(-h/max(float(rng),1e-12)))
    elif model=='spherical':
        r=np.clip(h/max(float(rng),1e-12),0,1);g=sill*(1.5*r-.5*r**3)
    elif model=='power':
        if not 0<power_exponent<2:raise ValueError('Power exponent must be in (0,2).')
        g=sill*h**power_exponent
    else:raise ValueError(f'Unknown variogram {model}.')
    return np.where(h>0,g+float(nugget),0.)

def _regularized_pair(a,b,model,nugget,sill,rng,power_exponent=1.):
    """Blockwise double integral J; avoids an unbounded n_A by n_B allocation."""
    _common_frame([a,b]);wa=a.normalised_weights();wb=b.normalised_weights();total=0.
    for i in range(0,len(a.x),256):
        for j in range(0,len(b.x),256):
            h=np.hypot(a.x[i:i+256,None]-b.x[None,j:j+256],a.y[i:i+256,None]-b.y[None,j:j+256])
            z=point_variogram(h,model,nugget,sill,rng,power_exponent)
            total+=np.sum(z*wa[i:i+256,None]*wb[None,j:j+256])
    return float(total)

def regularized_variogram_matrix(supports,model,nugget,sill,rng,power_exponent=1.):
    """Matrix of double integrals J_AB, including J_AA (NOT Gamma_AB)."""
    _common_frame(supports);n=len(supports);J=np.zeros((n,n))
    for i in range(n):
        for j in range(i,n):
            J[i,j]=J[j,i]=_regularized_pair(supports[i],supports[j],model,nugget,sill,rng,power_exponent)
    return J

def areal_semivariogram_matrix(supports,model,nugget,sill,rng,power_exponent=1.):
    J=regularized_variogram_matrix(supports,model,nugget,sill,rng,power_exponent)
    return J-.5*np.diag(J)[:,None]-.5*np.diag(J)[None,:]

def fit_point_variogram(supports,values,config=None):
    """Deconvolve Gamma_AB against 0.5*(z_A-z_B)^2, using training data only.

    Independent gauge measurement variances enter the empirical expectation
    as 0.5*(v_A+v_B). The optional point nugget is a different component, whose
    areal interpretation is conditional on the finite quadrature resolution.
    Default-parameter fallbacks are flagged and never reported as fitted.
    """
    cfg=config or TopKrigingConfig();z=np.asarray(values,float)
    if len(z)!=len(supports):raise ValueError('One value per support required.')
    keep=np.flatnonzero(np.isfinite(z));sup=[supports[i] for i in keep];z=z[keep];n=len(z)
    if not n:raise ValueError('No finite gauge attribute to fit.')
    _common_frame(sup)
    cx=np.array([np.average(s.x,weights=s.w) for s in sup]);cy=np.array([np.average(s.y,weights=s.w) for s in sup])
    iu,ju=np.triu_indices(n,1);sep=np.hypot(cx[iu]-cx[ju],cy[iu]-cy[ju])
    empirical=.5*(z[iu]-z[ju])**2
    variance=max(float(np.var(z)),1e-8);r0=max(float(np.median(sep)) if len(sep) else 50.,1.)
    rmax=max(cfg.max_range_factor*(float(np.max(sep)) if len(sep) else r0),2*r0)
    params=dict(model=cfg.model,nugget=.1*variance if cfg.fit_nugget else 0.,sill=variance,
                range_km=r0,power_exponent=cfg.power_exponent,status='DEFAULT_PARAMETERS',
                n_pairs=len(sep),rmse=np.nan,empirical_variance=variance,
                nugget_semantics='point_microscale_quadrature',measurement_error_separate=True)
    if len(sep)<cfg.min_pairs_for_fit:
        params['fallback_reason']='insufficient_pairs';return params
    # Cache pair distances and weights once per deconvolution. For very large
    # upstream supports, compute integrals in chunks instead of retaining O(N²).
    cache={}
    for i in range(n):
        for j in range(i,n):
            a,b=sup[i],sup[j]
            if len(a.x)*len(b.x)<=100000:
                cache[(i,j)]=(np.hypot(a.x[:,None]-b.x[None,:],a.y[:,None]-b.y[None,:]),
                              a.normalised_weights()[:,None]*b.normalised_weights()[None,:])
    free=['sill']+(['nugget'] if cfg.fit_nugget else [])
    free+=['power_exponent'] if cfg.model=='power' and cfg.fit_power_exponent else ([] if cfg.model=='power' else ['range_km'])
    bounds={'sill':(1e-12,100*variance),'nugget':(0,10*variance),'range_km':(1e-3,rmax),'power_exponent':(.05,1.95)}
    def residual(p):
        par=dict(params);par.update(zip(free,p));J=np.zeros((n,n))
        for i in range(n):
            for j in range(i,n):
                if (i,j) in cache:
                    h,w=cache[(i,j)];v=np.sum(w*point_variogram(h,par['model'],par['nugget'],par['sill'],par['range_km'],par['power_exponent']))
                else:v=_regularized_pair(sup[i],sup[j],par['model'],par['nugget'],par['sill'],par['range_km'],par['power_exponent'])
                J[i,j]=J[j,i]=v
        pred=J[iu,ju]-.5*(J[iu,iu]+J[ju,ju])
        pred+=.5*np.array([sup[i].measurement_variance+sup[j].measurement_variance for i,j in zip(iu,ju)])
        return pred-empirical
    from scipy.optimize import least_squares
    try:
        res=least_squares(residual,[params[k] for k in free],
                          bounds=([bounds[k][0] for k in free],[bounds[k][1] for k in free]),max_nfev=200)
        if res.success and np.isfinite(res.x).all():
            params.update(zip(free,map(float,res.x)));params.update(status='FITTED',rmse=float(np.sqrt(np.mean(res.fun**2))))
        else:params['fallback_reason']='optimizer_not_converged'
    except (ValueError,np.linalg.LinAlgError,FloatingPointError) as exc:params['fallback_reason']=str(exc)
    return params

def topkriging_weights(target,donors,vg,config=None):
    """Ordinary-kriging weights and complete prediction MSE.

    With independent measurement covariance V, solve
        [J-V, 1; 1',0] [lambda,mu]' = [j0,1]'.
    For ANY unbiased weights the latent-target error variance is
        2 lambda'j0 - lambda'J lambda - J00 + lambda'V lambda.
    This remains valid after nonnegative constrained optimisation; an old
    Lagrange multiplier must never be reused after clipping weights.
    """
    cfg=config or TopKrigingConfig();n=len(donors)
    if not n:return np.zeros(0),np.nan
    _common_frame([target]+list(donors))
    args=(vg['model'],vg['nugget'],vg['sill'],vg.get('range_km',1.),vg.get('power_exponent',cfg.power_exponent))
    J=regularized_variogram_matrix(donors,*args)
    j0=np.array([_regularized_pair(d,target,*args) for d in donors]);J00=_regularized_pair(target,target,*args)
    v=np.array([d.measurement_variance for d in donors]);G=J-np.diag(v)
    A=np.zeros((n+1,n+1));A[:n,:n]=G;A[:n,n]=1;A[n,:n]=1;b=np.r_[j0,1.]
    if n==1:lam=np.ones(1)
    else:
        try:lam=np.linalg.solve(A,b)[:n]
        except np.linalg.LinAlgError:
            A[:n,:n]-=np.eye(n)*cfg.condition_jitter # equivalent to small observation error
            lam=np.linalg.lstsq(A,b,rcond=None)[0][:n]
        if not np.isfinite(lam).all() or abs(lam.sum()-1)>1e-6:raise ValueError('Invalid kriging solution.')
        if not cfg.allow_negative_weights and np.any(lam<0):
            from scipy.optimize import minimize
            objective=lambda w:float(2*w@j0-w@G@w-J00)
            res=minimize(objective,np.full(n,1/n),jac=lambda w:2*(j0-G@w),method='SLSQP',
                         bounds=[(0,1)]*n,constraints={'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(n)},
                         options={'ftol':1e-12,'maxiter':500})
            if not res.success:raise ValueError(f'Nonnegative kriging failed: {res.message}')
            lam=res.x
    mse=float(2*lam@j0-lam@J@lam-J00+np.dot(lam**2,v))
    tolerance=1e-8*max(1.,abs(J00),float(np.max(np.abs(J))))
    if mse < -tolerance:raise ValueError('Negative kriging MSE: inspect supports/model conditioning.')
    return lam,max(0.,mse)

def topkriging_predict(target,donors,values,vg,config=None):
    cfg=config or TopKrigingConfig();z=np.asarray(values,float)
    if len(z)!=len(donors):raise ValueError('One attribute per donor is required.')
    keep=np.flatnonzero(np.isfinite(z));d=[donors[i] for i in keep];zz=z[keep]
    if not len(d):return dict(value=np.nan,variance=np.nan,n_donors=0,donors='',status='NO_DONOR')
    _common_frame([target]+d)
    if len(d)>cfg.max_donors:
        tx=np.average(target.x,weights=target.w);ty=np.average(target.y,weights=target.w)
        sep=[math.hypot(np.average(s.x,weights=s.w)-tx,np.average(s.y,weights=s.w)-ty) for s in d]
        order=np.argsort(sep)[:cfg.max_donors];d=[d[i] for i in order];zz=zz[order]
    lam,var=topkriging_weights(target,d,vg,cfg)
    return dict(value=float(lam@zz),variance=var,n_donors=len(d),donors=';'.join(s.name for s in d),weights=lam,
                status='OK',variogram_status=vg.get('status','USER_PARAMETERS'),support_kind=target.support_kind,
                uncertainty='conditional latent-attribute kriging MSE; no variogram-parameter uncertainty')

def topkriging_cross_validate(supports,values,config=None,compare_idw=True,idw_power=2.):
    cfg=config or TopKrigingConfig();names=[n for n in supports if n in values and np.isfinite(values[n])]
    if len(names)<3:return dict(folds=pd.DataFrame(),summary=pd.DataFrame(),verdict='INSUFFICIENT_GAUGES')
    sup=[supports[n] for n in names];z=np.array([values[n] for n in names]);rows=[]
    for i,name in enumerate(names):
        idx=[j for j in range(len(names)) if j!=i];d=[sup[j] for j in idx];vg=fit_point_variogram(d,z[idx],cfg)
        pred=topkriging_predict(sup[i],d,z[idx],vg,cfg)
        r=dict(name=name,observed=z[i],topkriging=pred['value'],topkriging_variance=pred['variance'],
               variogram_status=vg['status'],range_km=vg['range_km'])
        if compare_idw:
            sx=np.average(sup[i].x,weights=sup[i].w);sy=np.average(sup[i].y,weights=sup[i].w)
            dist=np.array([math.hypot(np.average(a.x,weights=a.w)-sx,np.average(a.y,weights=a.w)-sy) for a in d])
            w=1/np.maximum(dist,1e-9)**idw_power;r['idw']=float(w@z[idx]/w.sum())
        rows.append(r)
    folds=pd.DataFrame(rows);summary=[]
    for label in ['topkriging']+(['idw'] if compare_idw else []):
        e=folds[label]-folds.observed;folds['err_'+label]=e
        summary.append(dict(approach=label,mae=float(np.abs(e).mean()),rmse=float(np.sqrt(np.mean(e**2))),bias=float(e.mean()),n=len(e)))
    table=pd.DataFrame(summary).sort_values('rmse').reset_index(drop=True)
    return dict(folds=folds,summary=table,verdict='TOPKRIGING_BEST' if table.iloc[0].approach=='topkriging' else 'IDW_BEST')

def propagate_attribute_topkriging(gauge_values,target_supports,gauge_supports,value_col,name_col='station',config=None):
    cfg=config or TopKrigingConfig()
    if gauge_values[name_col].astype(str).duplicated().any():raise ValueError('Duplicate gauge names; aggregate explicitly first.')
    vals=gauge_values.assign(**{name_col:gauge_values[name_col].astype(str)}).set_index(name_col)[value_col].to_dict()
    names=[n for n in vals if n in gauge_supports and np.isfinite(vals[n])]
    if not names:return pd.DataFrame(columns=['target',value_col,'status'])
    donors=[gauge_supports[n] for n in names];z=np.array([vals[n] for n in names]);vg=fit_point_variogram(donors,z,cfg)
    rows=[]
    for name,s in target_supports.items():
        p=topkriging_predict(s,donors,z,vg,cfg)
        rows.append(dict(target=name,**{value_col:p['value']},kriging_variance=p['variance'],n_donors=p['n_donors'],
                         donors=p['donors'],status=p['status'],support_kind=s.support_kind,
                         **{'variogram_'+k:vg[k] for k in ('model','nugget','sill','range_km','status','power_exponent')}))
    return pd.DataFrame(rows)
