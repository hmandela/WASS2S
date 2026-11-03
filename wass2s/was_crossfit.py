"""Temporal cross-fitting from raw annual seasonal fields.

Selection and weighting use inner out-of-fold predictions. Outer probability
calibration reuses validation residuals by default; explicit inner/nested mode
cross-fits the complete selection procedure. Operational fitting is separate.
No held-out target is passed to a fitted estimator's predict method.
"""

from dataclasses import dataclass, field, asdict, fields as dataclass_fields
from typing import Any
import hashlib
import json
import numpy as np
import xarray as xr
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PowerTransformer
from .was_scientific_utils import apply_physical_support, climate_cluster_map
from .was_probability_reference import climatological_terciles
from .was_probability_calibration import normalize_probability_calibration


def _time_info(times):
    if not isinstance(times, xr.DataArray) or times.dims != ("T",):
        raise ValueError("Provide the one-dimensional T coordinate")
    idx = times.to_index()
    if idx.has_duplicates or not idx.is_monotonic_increasing:
        raise ValueError("T must be unique and sorted")
    years = tuple(int(y) for y in times.dt.year.values)
    if len(set(years)) != len(years):
        raise ValueError("Strict seasonal cross-fitting requires one target per year")
    return tuple(str(t) for t in times.values), years


@dataclass(frozen=True)
class Fold:
    train: tuple[int, ...]
    test: tuple[int, ...]


@dataclass(frozen=True)
class SplitPlan:
    """Immutable, bound to dates, with exclusion buffers in CALENDAR YEARS.

    LOO/blocked validation is retrospective. Expanding validation trains only
    on years preceding each test block. Uncovered burn-in years remain NaN.
    """

    times: tuple[str, ...]
    years: tuple[int, ...]
    folds: tuple[Fold, ...]
    mode: str
    gap_years: int
    min_train: int

    def __post_init__(self):
        if self.mode not in {"loo", "blocked", "expanding"}:
            raise ValueError("Unknown split mode")
        if not self.folds:
            raise ValueError("Insufficient history for a nonempty split plan")
        if self.gap_years < 0 or self.min_train < 2:
            raise ValueError("Invalid gap or minimum sample count")
        n = len(self.times)
        if len(self.years) != n or tuple(sorted(set(self.years))) != self.years:
            raise ValueError("Plan years must be sorted and unique")
        seen = set()
        for f in self.folds:
            tr, te = set(f.train), set(f.test)
            if (
                len(tr) != len(f.train)
                or len(te) != len(f.test)
                or len(tr) < self.min_train
                or not te
            ):
                raise ValueError("Invalid train/test partition")
            if tr & te or seen & te or not (tr | te) <= set(range(n)):
                raise ValueError("Overlapping or invalid fold positions")
            if any(
                abs(self.years[i] - self.years[j]) <= self.gap_years
                for i in tr
                for j in te
            ):
                raise ValueError("Training dates overlap the test buffer")
            if (
                self.mode == "expanding"
                and max(self.years[i] for i in tr)
                >= min(self.years[j] for j in te) - self.gap_years
            ):
                raise ValueError("Expanding plan includes future training years")
            seen |= te

    @classmethod
    def build(
        cls,
        times,
        *,
        mode="loo",
        gap_years=1,
        n_blocks=4,
        min_train=5,
        test_positions=None,
    ):
        keys, years = _time_info(times)
        if int(gap_years) != gap_years or gap_years < 0:
            raise ValueError("gap_years must be a nonnegative integer")
        if mode not in {"loo", "blocked", "expanding"}:
            raise ValueError("Unknown split mode")
        idx = np.arange(len(years))
        yy = np.asarray(years)
        if test_positions is not None:
            if mode != "loo":
                raise ValueError("test_positions is only supported for LOO")
            pos = np.asarray(test_positions)
            if (
                pos.ndim != 1
                or not np.issubdtype(pos.dtype, np.integer)
                or np.any(pos < 0)
                or np.any(pos >= len(idx))
                or len(np.unique(pos)) != len(pos)
            ):
                raise ValueError("Invalid explicit test positions")
            blocks = [np.array([i]) for i in pos]
        elif mode == "loo":
            blocks = [np.array([i]) for i in idx]
        else:
            if int(n_blocks) != n_blocks or n_blocks < 2:
                raise ValueError("n_blocks must be an integer >= 2")
            blocks = np.array_split(idx, min(int(n_blocks), len(idx)))
        folds = []
        for te in blocks:
            if not len(te):
                continue
            distance = np.min(abs(yy[:, None] - yy[te][None, :]), axis=1)
            keep = distance > gap_years
            if mode == "expanding":
                keep &= yy < yy[te].min() - gap_years
            tr = idx[keep]
            if len(tr) < min_train:
                if mode == "expanding":
                    continue
                raise ValueError(
                    f"Insufficient training years for test {yy[te].tolist()}; reduce buffer/blocks or supply more years"
                )
            folds.append(Fold(tuple(int(i) for i in tr), tuple(int(i) for i in te)))
        return cls(keys, years, tuple(folds), mode, int(gap_years), int(min_train))

    def check(self, times):
        if _time_info(times) != (self.times, self.years):
            raise ValueError("SplitPlan does not match these T coordinates")

    def to_dict(self):
        return asdict(self)

    @property
    def fingerprint(self):
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True).encode()
        ).hexdigest()


def _target(y):
    if not isinstance(y, xr.DataArray):
        raise TypeError("Target must be a DataArray")
    if "M" in y.dims:
        if y.sizes["M"] != 1:
            raise ValueError("Observation M must be singleton")
        y = y.isel(M=0, drop=True)
    if set(y.dims) != {"T", "Y", "X"}:
        raise ValueError("Target dimensions must be T,Y,X")
    y = y.transpose("T", "Y", "X").astype(float)
    _time_info(y["T"])
    if np.isinf(y.values).any():
        raise ValueError("Infinite observations are invalid")
    return y


def _fields(X, times=None):
    if isinstance(X, xr.DataArray):
        X = {"predictor": X}
    if not isinstance(X, dict) or not X:
        raise TypeError("Predictor must be a DataArray or nonempty dict of raw fields")
    out = {}
    for name, a in X.items():
        if (
            not isinstance(name, str)
            or not isinstance(a, xr.DataArray)
            or "T" not in a.dims
        ):
            raise TypeError("Each predictor needs a string name and DataArray with T")
        _time_info(a["T"])
        if times is not None and not a.get_index("T").equals(times.to_index()):
            raise ValueError(
                f"Predictor {name!r}: dates differ from target; no positional relabelling"
            )
        if np.isinf(a.values).any():
            raise ValueError(f"Infinite predictor values: {name}")
        out[name] = a
    ref = next(iter(out.values()))["T"]
    if any(not a.get_index("T").equals(ref.to_index()) for a in out.values()):
        raise ValueError("Predictor dates must agree")
    return out


def _slice(X, idx):
    return {k: a.isel(T=list(idx)) for k, a in X.items()}


class _FeatureTransform:
    """Training-only valid columns, mean imputation, trend and standardization."""

    def __init__(self, detrend=False):
        self.detrend = detrend

    def fit(self, x, years):
        self.valid = np.isfinite(x).sum(axis=0) >= 2
        x = np.asarray(x[:, self.valid], dtype=float)
        if not x.shape[1]:
            raise ValueError("No column has two finite training values")
        self.mean = np.nanmean(x, axis=0)
        self.origin = float(np.mean(years))
        t = np.asarray(years, dtype=float) - self.origin
        self.slope = np.zeros(x.shape[1])
        if self.detrend:
            for j in range(x.shape[1]):
                ok = np.isfinite(x[:, j])
                tt = t[ok]
                self.slope[j] = np.linalg.lstsq(
                    np.column_stack([np.ones(ok.sum()), tt]), x[ok, j], rcond=None
                )[0][1]
                self.mean[j] = np.mean(x[ok, j] - self.slope[j] * tt)
        z = x - (self.mean + t[:, None] * self.slope)
        self.scale = np.nanstd(z, axis=0)
        self.scale = np.where(self.scale > 1e-12, self.scale, 1.0)
        return self

    def fit_transform(self, x, years):
        return self.fit(x, years).transform(x, years)

    def transform(self, x, years):
        x = np.asarray(x[:, self.valid], dtype=float)
        z = x - (self.mean + (np.asarray(years) - self.origin)[:, None] * self.slope)
        return np.where(np.isfinite(z), z, 0.0) / self.scale


@dataclass(frozen=True)
class RegressionSpec:
    """One fixed candidate; settings compete via separate named specs.

    local: per-cell regression using matching grids and/or T/features indices.
    pcr: training-only shared PCA of flattened predictor fields, then regression.
    This PCA is not the legacy xeofs EEOF/CCA. Estimators are sklearn templates.
    """

    name: str
    fields: tuple[str, ...] = ("predictor",)
    alpha: float = 1.0
    geometry: str = "local"
    n_components: int = 3
    detrend: bool = False
    target_transform: str = "identity"
    estimator: Any = field(default=None, repr=False, compare=False)
    seed: int = 42
    min_samples: int = 5
    use_coslat: bool = False

    def create(self):
        if self.geometry not in {"local", "pcr"} or self.target_transform not in {
            "identity",
            "yeo_johnson",
        }:
            raise ValueError("Unsupported geometry or target transformation")
        if (
            not self.name
            or not self.fields
            or len(set(self.fields)) != len(self.fields)
        ):
            raise ValueError("Candidate needs a name and unique input field names")
        if (
            self.alpha < 0
            or self.min_samples < 3
            or int(self.n_components) != self.n_components
            or self.n_components < 1
        ):
            raise ValueError("Invalid regression configuration")
        return _GridRegression(self)


class _GridRegression:
    def __init__(self, spec):
        self.spec = spec

    def _matrix(self, X, iy=None, ix=None):
        pieces = []
        for name in self.spec.fields:
            if name not in X:
                raise ValueError(f"Missing raw predictor {name!r}")
            a = X[name]
            if self.spec.geometry == "local" and ("Y" in a.dims or "X" in a.dims):
                if not {"Y", "X"} <= set(a.dims):
                    raise ValueError("Local fields require both Y and X")
                if not a.get_index("Y").equals(
                    self.template.get_index("Y")
                ) or not a.get_index("X").equals(self.template.get_index("X")):
                    raise ValueError("Local predictor grid differs from target")
                a = a.isel(Y=iy, X=ix)
            dims = tuple(d for d in a.dims if d != "T")
            if name not in self.schemas:
                self.schemas[name] = (dims, {d: a.get_index(d).copy() for d in dims})
            expected, coords = self.schemas[name]
            if dims != expected or any(
                not a.get_index(d).equals(coords[d]) for d in dims
            ):
                raise ValueError(f"Predictor feature coordinates changed for {name}")
            pieces.append(
                np.asarray(a.transpose("T", *dims).values, dtype=float).reshape(
                    a.sizes["T"], -1
                )
            )
        return np.concatenate(pieces, axis=1)

    def _area_weights(self, X):
        weights = []
        for name in self.spec.fields:
            a = X[name].isel(T=0, drop=True)
            w = xr.ones_like(a, dtype=float)
            if self.spec.use_coslat and "Y" in a.dims:
                if np.any(abs(a.Y.values) > 90):
                    raise ValueError("Cosine weights require latitude degrees")
                w = w * np.sqrt(np.cos(np.deg2rad(a.Y)).clip(min=0))
            weights.append(w.transpose(*a.dims).values.ravel())
        return np.concatenate(weights)

    def fit(self, X, y):
        self.template = y.isel(T=0, drop=True).copy(deep=True)
        self.template.values[:] = np.nan
        self.reference = xr.DataArray(
            [], dims="sample", name=y.name, attrs=dict(y.attrs)
        )
        self.schemas = {}
        self.models = {}
        years = y["T"].dt.year.values.astype(float)
        self.training_years = tuple(int(t) for t in years)
        if self.spec.geometry == "pcr":
            raw = self._matrix(X)
            self.pre = _FeatureTransform(self.spec.detrend).fit(raw, years)
            z = self.pre.transform(raw, years)
            self.area = self._area_weights(X)[self.pre.valid]
            z = z * self.area
            n = min(int(self.spec.n_components), len(years) - 1, z.shape[1])
            self.pca = PCA(n_components=n, svd_solver="full").fit(z)
            global_x = self.pca.transform(z)
        for iy in range(y.sizes["Y"]):
            for ix in range(y.sizes["X"]):
                vals = y.values[:, iy, ix]
                ok = np.isfinite(vals)
                if ok.sum() < self.spec.min_samples:
                    continue
                if self.spec.geometry == "local":
                    raw = self._matrix(X, iy, ix)
                    if not np.any(np.isfinite(raw[ok]).sum(axis=0) >= 2):
                        continue
                    pre = _FeatureTransform(self.spec.detrend).fit(raw[ok], years[ok])
                    z = pre.transform(raw, years)
                else:
                    pre = None
                    z = global_x
                yy = vals[ok]
                transformer = None
                if self.spec.target_transform == "yeo_johnson" and np.std(yy) > 1e-12:
                    transformer = PowerTransformer(
                        method="yeo-johnson", standardize=False
                    )
                    yy = transformer.fit_transform(yy[:, None])[:, 0]
                ym = float(yy.mean())
                ys = float(yy.std())
                if ys < 1e-12:
                    ys = 1.0
                estimator = (
                    clone(self.spec.estimator)
                    if self.spec.estimator is not None
                    else Ridge(alpha=self.spec.alpha)
                )
                parameters = estimator.get_params(deep=True)
                fixed = {
                    k: self.spec.seed for k in parameters if k.endswith("random_state")
                }
                fixed.update({k: 1 for k in parameters if k.endswith("n_jobs")})
                estimator.set_params(**fixed)
                estimator.fit(z[ok], (yy - ym) / ys)
                self.models[iy, ix] = (pre, transformer, ym, ys, estimator)
        return self

    def predict(self, X):
        times = next(iter(X.values()))["T"]
        years = times.dt.year.values.astype(float)
        out = np.full(
            (len(times), self.template.sizes["Y"], self.template.sizes["X"]), np.nan
        )
        if self.spec.geometry == "pcr":
            z = self.pca.transform(
                self.pre.transform(self._matrix(X), years) * self.area
            )
        for (iy, ix), (pre, transformer, ym, ys, estimator) in self.models.items():
            xx = pre.transform(self._matrix(X, iy, ix), years) if pre is not None else z
            value = estimator.predict(xx) * ys + ym
            if transformer is not None:
                value = transformer.inverse_transform(value[:, None])[:, 0]
            out[:, iy, ix] = value
        result = xr.DataArray(
            out,
            dims=("T", "Y", "X"),
            coords={"T": times, "Y": self.template.Y, "X": self.template.X},
            name="prediction",
        )
        return apply_physical_support(result, self.reference)


@dataclass(frozen=True)
class CrossFitConfig:
    probability_method: str = "auto"
    inner_blocks: int = 4
    calibration_blocks: int = 4
    gap_years: int = 1
    min_train: int = 5
    min_oof: int = 3
    top_k: int | None = None
    weighting: str = "uniform"
    cluster_count: int = 1
    seed: int = 42
    climatology_start: int | None = None
    climatology_end: int | None = None
    mode: str = "blocked"  # inner selection/calibration mode
    outer_mode: str | None = None  # None: expanding when inner expands, otherwise LOO
    tercile_reference: str = "climatology"  # fixed climatology; optional strict training reference
    probability_calibration: str = "outer"

    def __post_init__(self):
        object.__setattr__(self, "probability_calibration", normalize_probability_calibration(self.probability_calibration))
        if self.tercile_reference not in {"climatology", "training"}:
            raise ValueError("tercile_reference must be climatology or training")
        if self.outer_mode not in {None, 'loo', 'blocked', 'expanding'}:
            raise ValueError('Unknown outer_mode')
        if self.mode == 'expanding' and self.outer_mode not in {None, 'expanding'}:
            raise ValueError('Expanding inner validation requires expanding outer validation')
        if self.probability_method not in {"auto", "native", "residual"}:
            raise ValueError("probability_method must be auto, native or residual")
        if (
            self.inner_blocks < 2
            or self.calibration_blocks < 2
            or self.min_train < 3
            or self.min_oof < 3
        ):
            raise ValueError("Need >=2 blocks, >=3 fit samples and >=3 OOF samples")
        if (
            self.gap_years < 0
            or self.cluster_count < 1
            or (self.top_k is not None and self.top_k < 1)
        ):
            raise ValueError("Invalid gap, clusters or top_k")
        if self.weighting not in {"uniform", "simplex"} or self.mode not in {
            "blocked",
            "expanding",
        }:
            raise ValueError("Unknown weighting or inner split mode")
        if (self.climatology_start is None) != (self.climatology_end is None):
            raise ValueError("Provide both climatology bounds or neither")
        if (
            self.climatology_start is not None
            and self.climatology_start > self.climatology_end
        ):
            raise ValueError("Reversed climatology bounds")


@dataclass
class _Bundle:
    models: dict
    weights: xr.DataArray
    clusters: xr.DataArray
    selection_losses: xr.DataArray
    training_years: tuple
    selection_plan: dict
    selection_oof: xr.DataArray | None = None

    def predict(self, X):
        times = next(iter(X.values()))["T"]
        result = (
            xr.zeros_like(self.weights.isel(candidate=0, drop=True))
            .expand_dims(T=times)
            .transpose("T", "Y", "X")
        )
        valid = xr.ones_like(result, dtype=bool)
        for name, model in self.models.items():
            w = self.weights.sel(candidate=name, drop=True)
            if not bool((w > 0).any()):
                continue
            pred = model.predict(X)
            result = result + xr.where(w > 0, pred * w, 0.0)
            valid = valid & ((w <= 0) | pred.notnull())
        return result.where(valid & (self.weights.sum("candidate") > 0)).rename(
            "prediction"
        )


    def predict_proba(self, X, thresholds):
        parts = []
        for name, model in self.models.items():
            w = self.weights.sel(candidate=name, drop=True)
            if not bool((w > 0).any()):
                continue
            p = model.predict_proba(X, thresholds)
            parts.append(xr.where(w > 0, p * w, 0.0))
        if not parts:
            raise ValueError("No selected model for native probability pooling")
        p = sum(parts).where(self.weights.sum("candidate") > 0)
        valid = p.notnull().all("probability") & (p >= 0).all("probability")
        total = p.sum("probability", skipna=False)
        return (p / total).where(valid & (total > 0)).rename("probability_forecast")


def _thresholds(y, cfg):
    """Thresholds from the explicitly supplied reference, without implicit folds."""
    q, ref = climatological_terciles(
        y, cfg.climatology_start, cfg.climatology_end, min_count=cfg.min_oof
    )
    q.attrs["tercile_reference"] = ref.attrs["tercile_reference"] = cfg.tercile_reference
    return q, ref


def _reference_metadata(y, cfg):
    """Common, serializable reference description; per-fold counts are audited."""
    return {
        "tercile_reference": cfg.tercile_reference,
        "climatology_start": int(y["T"].dt.year.min()) if cfg.climatology_start is None else int(cfg.climatology_start),
        "climatology_end": int(y["T"].dt.year.max()) if cfg.climatology_end is None else int(cfg.climatology_end),
        "tercile_reference_scope": "fixed across folds" if cfg.tercile_reference == "climatology" else "outer training fold only",
        "tercile_quantile_levels": "1/3,2/3",
        "category_definition": "PB: y <= low; PN: low < y <= high; PA: y > high",
    }


def _probabilities(pred, residuals, thresholds, reference, min_oof):
    """One residual per calibration year, in physical units, without pooling."""
    low = thresholds.isel(quantile=0, drop=True).values
    high = thresholds.isel(quantile=1, drop=True).values
    rr = residuals.values
    out = np.full((3, *pred.shape), np.nan)
    for i in range(pred.sizes["T"]):
        values = xr.DataArray(
            rr + pred.values[i],
            dims=("sample", "Y", "X"),
            coords={"Y": pred.Y, "X": pred.X},
        )
        values = apply_physical_support(values, reference).values
        valid = np.isfinite(values) & np.isfinite(low) & np.isfinite(high)
        count = valid.sum(axis=0)
        for k, event in enumerate(
            [values <= low, (values > low) & (values <= high), values > high]
        ):
            out[k, i] = np.divide(
                (event & valid).sum(axis=0),
                count,
                out=np.full_like(low, np.nan),
                where=count >= min_oof,
            )
    return xr.DataArray(
        out,
        dims=("probability", "T", "Y", "X"),
        coords={
            "probability": ["PB", "PN", "PA"],
            "T": pred["T"],
            "Y": pred.Y,
            "X": pred.X,
        },
        name="probability_forecast",
    )


@dataclass
class CrossFitResult:
    deterministic: xr.DataArray
    probabilities: xr.DataArray
    thresholds: xr.DataArray
    reference_probabilities: xr.DataArray
    weights: xr.DataArray
    calibration_count: xr.DataArray
    clusters: xr.DataArray
    selection_losses: xr.DataArray
    settings: dict
    audit: list
    split_plan: SplitPlan

    def to_dataset(self):
        ds = xr.Dataset(
            {
                "deterministic": self.deterministic,
                "probabilities": self.probabilities,
                "thresholds": self.thresholds,
                "reference_probabilities": self.reference_probabilities,
                "weights": self.weights,
                "calibration_count": self.calibration_count,
                "clusters": self.clusters,
                "selection_losses": self.selection_losses,
            }
        )
        ds.attrs.update(
            validation=self.probabilities.attrs.get("validation", "temporal cross-fitting"),
            probability_calibration=self.probabilities.attrs.get("probability_calibration", "outer"),
            split_plan=json.dumps(self.split_plan.to_dict()),
            audit=json.dumps(self.audit),
            settings=json.dumps(self.settings),
            split_fingerprint=self.split_plan.fingerprint,
        )
        ds.attrs.update(self.settings.get("probability_reference", {}))
        return ds

    def scores(self, observations):
        """Score using exactly the same reference as probability generation."""
        y = _target(observations)
        self.split_plan.check(y['T'])
        scores = score_fold_probabilities(y, self.probabilities, self.thresholds,
                                         self.reference_probabilities)
        error = (self.deterministic-y).where(np.isfinite(self.deterministic) & np.isfinite(y))
        scores['MAE'] = abs(error).mean('T')
        scores['RMSE'] = np.sqrt((error**2).mean('T'))
        scores.attrs.update(self.settings.get("probability_reference", {}))
        return scores


def score_fold_probabilities(observations, probabilities, thresholds, reference_probabilities):
    """RPS/RPSS, Brier and macro-AUC with an explicitly supplied reference.

    Thresholds may vary by T. RPSS is one minus the ratio of mean RPS values,
    not a mean of pointwise ratios. All scores share the same complete cases.
    ``ROC_AUC_macro`` is explicitly a mean one-versus-rest AUC and not GROC.
    Undefined absent-event AUCs are omitted; n_valid is returned for diagnosis.
    """
    from sklearn.metrics import roc_auc_score
    from .was_groc import ordinal_groc, PROBABILITY_SUM_ATOL
    y, p, q, ref = xr.align(observations, probabilities, thresholds,
                            reference_probabilities, join='exact')
    if list(p.probability.values) != ['PB','PN','PA'] or list(ref.probability.values) != ['PB','PN','PA']:
        raise ValueError('Expected probabilities in PB,PN,PA order')
    if q.sizes.get('quantile') != 2:
        raise ValueError('Provide exactly two ordered tercile thresholds')
    for a in (p, ref):
        complete = np.isfinite(a).all('probability')
        partial = np.isfinite(a).any('probability') & ~complete
        invalid = np.isinf(a).any() | ((a < 0) | (a > 1)).any() | partial.any()
        total = a.astype(np.float64).sum('probability', skipna=False)
        invalid = invalid | (complete & (abs(total-1) > PROBABILITY_SUM_ATOL)).any()
        if bool(invalid.compute()):
            raise ValueError('Invalid probability simplex; missing forecasts must be all NaN')
    # Use identical float64-normalized probabilities for every score, after
    # rejecting invalid values/sums above. Keep wholly missing columns missing.
    p = p.astype(np.float64)
    ref = ref.astype(np.float64)
    p = p / p.sum('probability', skipna=False)
    ref = ref / ref.sum('probability', skipna=False)
    lo, hi = q.isel(quantile=0,drop=True), q.isel(quantile=1,drop=True)
    if bool((lo > hi).any().compute()):
        raise ValueError('Reversed tercile thresholds')
    cls = xr.where(y <= lo, 0, xr.where(y <= hi, 1, 2))
    valid = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi) & np.isfinite(p).all('probability') & np.isfinite(ref).all('probability')
    onehot = xr.concat([(cls == k).astype(float) for k in range(3)],
                      dim=xr.IndexVariable('probability',['PB','PN','PA']))
    cumulative = onehot.cumsum('probability')
    rps = ((p.cumsum('probability')-cumulative)**2).sum('probability',skipna=False).where(valid)
    rps_ref = ((ref.cumsum('probability')-cumulative)**2).sum('probability',skipna=False).where(valid)
    denom = rps_ref.mean('T')
    def auc_macro(labels, probs):
        mask = np.isfinite(labels) & np.isfinite(probs).all(axis=0)
        labels, probs = labels[mask], probs[:,mask]
        values = []
        for k in range(3):
            binary = labels == k
            if len(np.unique(binary)) == 2:
                values.append(roc_auc_score(binary,probs[k]))
        return float(np.mean(values)) if values else np.nan
    macro = xr.apply_ufunc(auc_macro,cls.where(valid),p.where(valid),
        input_core_dims=[['T'],['probability','T']],output_core_dims=[[]],
        vectorize=True,dask='parallelized',output_dtypes=[float],
        dask_gufunc_kwargs={'allow_rechunk':True})
    groc = xr.apply_ufunc(ordinal_groc,cls.where(valid),p.where(valid),
        input_core_dims=[['T'],['probability','T']],output_core_dims=[[]],
        vectorize=True,dask='parallelized',output_dtypes=[float],
        dask_gufunc_kwargs={'allow_rechunk':True})
    chosen = (p.where(onehot == 1, 0)).sum('probability',skipna=False)
    with np.errstate(divide='ignore',invalid='ignore'):
        ignorance = -np.log2(chosen)
    return xr.Dataset({'RPS':rps.mean('T'),'RPS_reference':denom,
        'RPSS':(1-rps.mean('T')/denom).where(denom>0),
        'BS_PA':((p.sel(probability='PA',drop=True)-onehot.sel(probability='PA',drop=True))**2).where(valid).mean('T'),
        'Ignorance':ignorance.where(valid).mean('T'),
        'ROC_AUC_macro':macro,'GROC':groc,'n_valid':valid.sum('T')})


@dataclass
class OperationalArtifact:
    bundle: _Bundle
    residuals: xr.DataArray
    thresholds: xr.DataArray
    reference_probabilities: xr.DataArray
    target_reference: xr.DataArray
    min_oof: int
    training_times: tuple[str, ...]
    training_last_year: int
    audit: dict

    def predict(self, predictors):
        X = _fields(predictors)
        times = next(iter(X.values()))["T"]
        if int(times.dt.year.min()) <= self.training_last_year:
            raise ValueError(
                "Operational prediction requires years after the training record"
            )
        pred = self.bundle.predict(X)
        prob = self.bundle.predict_proba(X, self.thresholds) if self.audit.get("probability_method") == "native" else _probabilities(
            pred, self.residuals, self.thresholds, self.target_reference, self.min_oof
        )
        mode = self.audit.get("probability_calibration", "outer")
        native = self.audit.get("probability_method") == "native"
        calibration = self.audit.get("calibration", {})
        source = (calibration.get("source") if isinstance(calibration, dict) else None)
        provenance = {
            "probability_calibration_requested": mode,
            "probability_calibration": "native" if native else mode,
            "probability_method": self.audit.get("probability_method", "residual"),
            "residual_calibration_applied": int(not native),
            "calibration_strictly_nested": int(not native and mode == "nested"),
            "calibration_source": source or ("native training distribution" if native else "cross-fitted complete selection procedure"),
        }
        prob.attrs.update(provenance, **self.audit.get("probability_reference", {}))
        return xr.Dataset(
            {"deterministic": pred, "probabilities": prob},
            attrs={
                "purpose": "operational forecast, distinct from hindcast evaluation",
                **provenance,
                "training_last_year": self.training_last_year,
                **self.audit.get("probability_reference", {}),
            },
        )


class WAS_CrossFit:
    """Outer or nested evaluation for predeclared RegressionSpec candidates.

    Supply RAW fields and all a-priori eligible GCMs. Candidate instances are
    fresh at each fit. Full-record tuning/EOF/cluster artifacts are not accepted.
    Additional sklearn regressors can be provided as estimator templates.
    By default, a prescribed climatology defines identical categories in every
    fold.  It is separate from training-only model selection and calibration.
    Use ``tercile_reference='training'`` for changing, fold-local categories.
    """

    def __init__(self, candidates, config=None):
        self.candidates = tuple(candidates)
        self.config = config or CrossFitConfig()
        names = [s.name for s in self.candidates]
        if not names or len(set(names)) != len(names):
            raise ValueError("Candidate names must be nonempty and unique")
        for spec in self.candidates:
            from .was_crossfit_adapters import WASModelSpec
            if not isinstance(spec, (RegressionSpec, WASModelSpec)):
                raise TypeError("Use explicit RegressionSpec or WASModelSpec candidates")
            spec.create()
        self.names = names
        if self.config.probability_method == "auto":
            for spec in self.candidates:
                requested = str(getattr(spec, "init", {}).get("dist_method", "nonparam")).lower()
                if requested != "nonparam" and not getattr(spec, "native_probabilities", False):
                    raise ValueError("CrossFit auto cannot silently replace a requested parametric law; "
                        "use WAS_Cross_Validator for fold-local bestfit, or explicitly request "
                        "CrossFitConfig(probability_method='residual')")
        all_native = all(getattr(s, "native_probabilities", False) for s in self.candidates)
        if self.config.probability_method == "native" and not all_native:
            raise ValueError("Native pooling requires native probabilities for every candidate")
        self.probability_method = ("native" if all_native else "residual") if self.config.probability_method == "auto" else self.config.probability_method

    def _plan(self, times, blocks):
        c = self.config
        return SplitPlan.build(
            times,
            mode=c.mode,
            gap_years=c.gap_years,
            n_blocks=blocks,
            min_train=c.min_train,
        )

    def describe(self):
        catalog = []
        for spec in self.candidates:
            item = {
                f.name: getattr(spec, f.name)
                for f in dataclass_fields(spec)
                if f.name != "estimator"
            }
            if hasattr(spec, "estimator") and spec.estimator is not None:
                item["estimator_type"] = (
                    type(spec.estimator).__module__
                    + "."
                    + type(spec.estimator).__qualname__
                )
                item["estimator_parameters"] = spec.estimator.get_params(deep=False)
            elif isinstance(spec, RegressionSpec):
                item["estimator_type"] = "sklearn.linear_model.Ridge"
            catalog.append(item)
        return json.loads(
            json.dumps(
                {"config": asdict(self.config), "candidates": catalog,
                 "resolved_probability_method": self.probability_method}, default=repr
            )
        )

    def _selection_plan(self, times):
        plan = self._plan(times, self.config.inner_blocks)
        covered = {i for f in plan.folds for i in f.test}
        if len(covered) < self.config.min_oof:
            raise ValueError("Insufficient OOF years after temporal warm-up")
        if any(
            len(f.train) < max(s.min_samples for s in self.candidates)
            for f in plan.folds
        ):
            raise ValueError("Training fold is shorter than candidate min_samples")
        return plan

    def _calibration_plan(self, times):
        plan = self._plan(times, self.config.calibration_blocks)
        usable = []
        for f in plan.folds:
            try:
                self._selection_plan(times.isel(T=list(f.train)))
            except ValueError:
                if self.config.mode != "expanding":
                    raise
                continue  # date-only nested warm-up; never chosen from skill
            usable.append(f)
        if not usable:
            raise ValueError("Insufficient history for nested calibration")
        return SplitPlan(
            plan.times,
            plan.years,
            tuple(usable),
            plan.mode,
            plan.gap_years,
            plan.min_train,
        )

    def default_split_plan(self, times):
        """Outer validation geometry, including date-only nested warm-up."""
        c = self.config
        mode = c.outer_mode or ('expanding' if c.mode == 'expanding' else 'loo')
        plan = SplitPlan.build(times, mode=mode, gap_years=c.gap_years,
                               n_blocks=c.inner_blocks, min_train=c.min_train)
        if mode != 'expanding':
            return plan
        usable = []
        for fold in plan.folds:
            t = times.isel(T=list(fold.train))
            try:
                self._selection_plan(t)
                if self.probability_method != 'native' and c.probability_calibration == "nested":
                    self._calibration_plan(t)
            except ValueError:
                continue
            usable.append(fold)
        if not usable:
            raise ValueError('Insufficient history for nested expanding validation')
        return SplitPlan(plan.times,plan.years,tuple(usable),plan.mode,plan.gap_years,plan.min_train)

    def estimate_cost(self, observations, split_plan=None):
        """Preflight temporal feasibility and count candidate pipeline fits.

        Counts refits conservatively for all candidates, even where a zero
        weight later avoids one. Local regressors multiply by valid grid cells.
        """
        y = _target(observations)
        plan = split_plan or self.default_split_plan(y['T'])
        plan.check(y["T"])
        if self.config.mode == 'expanding' and plan.mode != 'expanding':
            raise ValueError('Expanding configuration requires an expanding outer SplitPlan')
        total = 0
        for f in plan.folds:
            times = y["T"].isel(T=list(f.train))
            selection = self._selection_plan(times)
            total += len(self.candidates) * (len(selection.folds) + 1)
            if self.probability_method == "native" or self.config.probability_calibration == "outer":
                continue
            calibration = self._calibration_plan(times)
            for cf in calibration.folds:
                inner = self._selection_plan(times.isel(T=list(cf.train)))
                total += len(self.candidates) * (len(inner.folds) + 1)
        return {
            "outer_folds": len(plan.folds),
            "candidate_pipeline_fits_upper_bound": total,
            "target_cells": y.sizes["Y"] * y.sizes["X"],
            "per_cell_regressions_upper_bound": total * y.sizes["Y"] * y.sizes["X"],
        }

    def _fit_bundle(self, X, y):
        c = self.config
        plan = self._selection_plan(y["T"])
        raw = np.full((len(self.candidates), *y.shape), np.nan)
        for f in plan.folds:
            xa, xb = _slice(X, f.train), _slice(X, f.test)
            ya = y.isel(T=list(f.train))
            for k, spec in enumerate(self.candidates):
                try:
                    estimator = spec.create().fit(xa, ya)
                    pred = estimator.predict(xb)
                except Exception as exc:
                    raise RuntimeError(
                        f"Candidate {spec.name!r} failed on training years {ya['T'].dt.year.values.tolist()}"
                    ) from exc
                raw[k, list(f.test)] = pred.values
        clusters = climate_cluster_map(
            y, c.cluster_count, random_state=c.seed, min_valid=c.min_train
        )
        weights = np.zeros((len(self.candidates), y.sizes["Y"], y.sizes["X"]))
        losses = np.full_like(weights, np.nan)
        common = np.isfinite(raw).all(axis=0) & np.isfinite(y.values)
        enough = common.sum(axis=0) >= c.min_oof
        std = y.std("T").values
        std = np.where(std > 1e-12, std, 1.0)
        for group in np.unique(clusters.values[np.isfinite(clusters.values)]):
            cells = (clusters.values == group) & enough
            use = common & cells[None, :, :]
            if not use.any():
                continue
            yy = (y.values / std)[use]
            pp = np.stack([(a / std)[use] for a in raw], axis=1)
            loss = np.mean((pp - yy[:, None]) ** 2, axis=0)
            chosen = np.argsort(loss, kind="stable")[: c.top_k or len(self.names)]
            w = np.full(len(chosen), 1 / len(chosen))
            if c.weighting == "simplex" and len(chosen) > 1:
                mat = pp[:, chosen]
                solution = minimize(
                    lambda v: np.mean((mat @ v - yy) ** 2),
                    w,
                    jac=lambda v: 2 * mat.T @ (mat @ v - yy) / len(yy),
                    method="SLSQP",
                    bounds=[(0.0, 1.0)] * len(chosen),
                    constraints={
                        "type": "eq",
                        "fun": lambda v: v.sum() - 1.0,
                        "jac": lambda v: np.ones_like(v),
                    },
                    options={"ftol": 1e-10, "maxiter": 500},
                )
                if not solution.success:
                    raise RuntimeError(f"Simplex weighting failed: {solution.message}")
                w = np.maximum(solution.x, 0)
                w /= w.sum()
            for k in range(len(self.names)):
                losses[k, cells] = loss[k]
            for k, value in zip(chosen, w):
                weights[k, cells] = value
        coords = {"candidate": self.names, "Y": y.Y, "X": y.X}
        wda = xr.DataArray(
            weights, dims=("candidate", "Y", "X"), coords=coords, name="weight"
        )
        fitted = {
            s.name: s.create().fit(X, y)
            for k, s in enumerate(self.candidates)
            if np.any(weights[k] > 0)
        }
        if not fitted:
            raise ValueError(
                "No cells have enough common OOF observations; inspect missingness or increase history"
            )
        # Reuse candidate selection OOF fits for an operational approximation.
        # Weights/selection have seen these historical targets; this is never
        # labelled as cross-fitting the complete selection procedure.
        oof_values = np.zeros(y.shape)
        valid_oof = np.ones(y.shape, dtype=bool)
        for k in range(len(self.names)):
            active = weights[k] > 0
            oof_values += np.where(active, raw[k] * weights[k], 0.0)
            valid_oof &= (~active) | np.isfinite(raw[k])
        oof_values = np.where(valid_oof & (weights.sum(axis=0) > 0), oof_values, np.nan)
        selection_oof = xr.DataArray(oof_values, dims=y.dims, coords=y.coords)
        return _Bundle(
            fitted,
            wda,
            clusters,
            xr.DataArray(losses, dims=wda.dims, coords=coords),
            tuple(int(t) for t in y["T"].dt.year.values),
            plan.to_dict(),
            selection_oof,
        )

    def _calibrate(self, X, y):
        plan = self._calibration_plan(y["T"])
        parts = []
        audit = []
        for f in plan.folds:
            xa, xb = _slice(X, f.train), _slice(X, f.test)
            ya = y.isel(T=list(f.train))
            # Cross-fit the SELECTION procedure, not just its chosen regressors.
            bundle = self._fit_bundle(xa, ya)
            parts.append(bundle.predict(xb))
            audit.append(
                {
                    "train_years": list(bundle.training_years),
                    "test_years": y["T"]
                    .dt.year.values[list(f.test)]
                    .astype(int)
                    .tolist(),
                    "selection_plan": bundle.selection_plan,
                }
            )
        oof = xr.concat(parts, dim="T").sortby("T").reindex(T=y["T"])
        return y - oof, {"plan": plan.to_dict(), "folds": audit}

    def evaluate(self, predictors, observations, split_plan=None):
        y = _target(observations)
        X = _fields(predictors, y["T"])
        c = self.config
        plan = split_plan or self.default_split_plan(y['T'])
        plan.check(y["T"])
        if self.config.mode == 'expanding' and plan.mode != 'expanding':
            raise ValueError('Expanding configuration requires an expanding outer SplitPlan')
        self.estimate_cost(y, plan)  # fail on impossible nested dates before fitting
        # Fixed climate categories are computed ONCE, before fitting any fold.
        # A prescribed event reference is distinct from fitting/tuning targets.
        shared_reference = _thresholds(y, c) if c.tercile_reference == "climatology" else None
        reference_metadata = _reference_metadata(y, c)
        cluster_parts = []
        loss_parts = []
        dets = []
        probs = []
        ths = []
        refs = []
        weights = []
        counts = []
        audits = []
        for index, f in enumerate(plan.folds):
            xa, xb = _slice(X, f.train), _slice(X, f.test)
            ya = y.isel(T=list(f.train))
            bundle = self._fit_bundle(xa, ya)
            if self.probability_method == "native":
                residuals, ca = ya.isel(T=slice(0, 0)), {"method": "native", "residual_calibration": False}
            elif c.probability_calibration == "outer":
                residuals, ca = ya.isel(T=slice(0, 0)), {"method": "outer", "source": "other outer folds"}
            else:
                residuals, ca = self._calibrate(xa, ya)
            thresholds, ref = shared_reference if shared_reference is not None else _thresholds(ya, c)
            pred = bundle.predict(xb)
            reference = xr.DataArray(
                [], dims="sample", name=ya.name, attrs=dict(ya.attrs)
            )
            prob = bundle.predict_proba(xb, thresholds) if self.probability_method == "native" else _probabilities(pred, residuals, thresholds, reference, c.min_oof)
            times = pred["T"]
            dets.append(pred)
            probs.append(prob)
            ths.append(thresholds.expand_dims(T=times))
            refs.append(ref.expand_dims(T=times))
            weights.append(bundle.weights.expand_dims(T=times))
            counts.append(residuals.notnull().sum("T").expand_dims(T=times))
            cluster_parts.append(bundle.clusters.expand_dims(T=times))
            loss_parts.append(bundle.selection_losses.expand_dims(T=times))
            audits.append(
                {
                    "fold": index,
                    "train_years": list(bundle.training_years),
                    "test_years": y["T"]
                    .dt.year.values[list(f.test)]
                    .astype(int)
                    .tolist(),
                    "selection_plan": bundle.selection_plan,
                    "calibration": ca,
                    "probability_reference": dict(thresholds.attrs),
                }
            )

        def combine(parts):
            return xr.concat(parts, dim="T").sortby("T").reindex(T=y["T"])

        if self.probability_method != "native" and c.probability_calibration == "outer":
            errors = y - combine(dets)
            probs, counts = [], []
            for index, f in enumerate(plan.folds):
                train = y.isel(T=list(f.train))
                residuals = errors.isel(T=list(f.train))
                q = ths[index].isel(T=0, drop=True)
                probs.append(_probabilities(dets[index], residuals, q, train, c.min_oof))
                counts.append(residuals.notnull().sum("T").expand_dims(T=dets[index]["T"]))
                audits[index]["calibration"].update(
                    strictly_nested=False,
                    residual_years=residuals["T"].dt.year.values.astype(int).tolist(),
                    note="other outer models may have seen this fold's verification targets")

        result = CrossFitResult(
            combine(dets).transpose("T", "Y", "X"),
            combine(probs).transpose("probability", "T", "Y", "X"),
            combine(ths).transpose("T", "quantile", "Y", "X"),
            combine(refs).transpose("probability", "T", "Y", "X"),
            combine(weights).transpose("T", "candidate", "Y", "X"),
            combine(counts).transpose("T", "Y", "X"),
            combine(cluster_parts).transpose("T", "Y", "X"),
            combine(loss_parts).transpose("T", "candidate", "Y", "X"),
            {**self.describe(), "probability_reference": reference_metadata},
            audits,
            plan,
        )
        for a in [result.deterministic, result.probabilities]:
            a.attrs.update(
                validation="temporal cross-fitting with " + ("native probabilities" if self.probability_method == "native" else c.probability_calibration + " residual calibration"),
                probability_calibration=c.probability_calibration,
                calibration_strictly_nested=int(self.probability_method != "native" and c.probability_calibration == "nested"),
                residual_calibration_applied=int(self.probability_method != "native"),
                split_fingerprint=plan.fingerprint,
            )
        for a in [result.probabilities, result.thresholds, result.reference_probabilities]:
            a.attrs.update(reference_metadata)
        return result

    def fit_operational(self, predictors, observations, *, hindcast_det=None):
        y = _target(observations)
        X = _fields(predictors, y["T"])
        thresholds, ref = _thresholds(y, self.config)
        bundle = self._fit_bundle(X, y)
        if self.probability_method == "native":
            residuals, ca = y.isel(T=slice(0, 0)), {"method": "native", "residual_calibration": False}
        elif self.config.probability_calibration == "outer":
            oof = bundle.selection_oof if hindcast_det is None else _target(hindcast_det)
            _, oof = xr.align(y, oof, join="exact")
            residuals = y - oof
            ca = {"method": "outer", "strictly_nested": False,
                  "source": "reused selection OOF with full-history weights" if hindcast_det is None else "caller-supplied outer hindcasts",
                  "note": "future target is absent; historical residuals are conditional on selected configuration"}
        else:
            if hindcast_det is not None:
                raise ValueError("Nested operational calibration must cross-fit selection; a single hindcast cannot replace it")
            residuals, ca = self._calibrate(X, y)
        reference = xr.DataArray([], dims="sample", name=y.name, attrs=dict(y.attrs))
        return OperationalArtifact(
            bundle,
            residuals,
            thresholds,
            ref,
            reference,
            self.config.min_oof,
            _time_info(y["T"])[0],
            int(y["T"].dt.year.max()),
            {
                "purpose": "operational refit",
                "probability_method": self.probability_method,
                "probability_calibration": self.config.probability_calibration,
                "selection_plan": bundle.selection_plan,
                "calibration": ca,
                "probability_reference": {
                    **_reference_metadata(y, self.config),
                    "tercile_reference_scope": "operational reference period",
                },
            },
        )
