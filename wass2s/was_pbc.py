"""Seasonal adaptation of probabilistic bias correction (PBC).

Guan et al., Nature (2026), doi:10.1038/s41586-026-11126-x, combine an
additive Debias++ correction and Persistence++ regression in cumulative
probability space. Each branch is projected onto valid CDFs before averaging.

This implementation preserves that construction, but replaces the authors'
subseasonal day-of-year windows with a caller-supplied, season-specific training
archive and optionally regularizes Persistence++ for short annual records.
Training/archive selection and observational availability belong to the calling
workflow: ``fit`` cannot establish whether a base forecast was truly out of
sample. Event thresholds are supplied once, never estimated from a fit fold.
"""

from pathlib import Path

import numpy as np
import xarray as xr


_LEVELS = np.array([1.0 / 3.0, 2.0 / 3.0])
_CATEGORIES = ["PB", "PN", "PA"]
_SCHEMA = "wass2s-seasonal-pbc-v2"
_LEGACY_SCHEMA = "wass2s-seasonal-pbc-v1"


def _feature_names(n_lags):
    return ["intercept", "climatology", *[f"lag{i}" for i in range(1, n_lags + 1)], "raw_cdf"]


def _unique_coordinates(array, dimensions):
    for dim in dimensions:
        if dim not in array.coords or array[dim].dims != (dim,):
            raise ValueError(f"PBC requires a one-dimensional {dim} coordinate")
        if not array.get_index(dim).is_unique:
            raise ValueError(f"PBC {dim} coordinates must be unique")


def _probabilities(array):
    if not isinstance(array, xr.DataArray):
        raise TypeError("PBC probabilities must be an xarray DataArray")
    if set(array.dims) != {"probability", "T", "Y", "X"}:
        raise ValueError("PBC probabilities require exactly probability,T,Y,X dimensions")
    _unique_coordinates(array, ("probability", "T", "Y", "X"))
    if list(map(str, array["probability"].values)) != _CATEGORIES:
        raise ValueError("PBC probabilities must be ordered PB,PN,PA")
    out = array.transpose("probability", "T", "Y", "X").astype(float)
    values = np.asarray(out.values)
    if np.isinf(values).any():
        raise ValueError("PBC probabilities cannot contain infinity")
    finite = np.isfinite(values)
    if ((values[finite] < -1e-8) | (values[finite] > 1 + 1e-8)).any():
        raise ValueError("PBC probability values must lie in [0,1]")
    complete = finite.all(axis=0)
    if np.any(np.abs(values.sum(axis=0)[complete] - 1) > 1e-6):
        raise ValueError("PBC complete probability vectors must sum to one")
    # A partial vector does not define a distribution. Do not fill its missing
    # component by renormalizing the remaining classes.
    values = np.where(complete[None], np.clip(values, 0, 1), np.nan)
    return out.copy(data=values)


def _thresholds(array, reference):
    if isinstance(array, (tuple, list)) and len(array) == 2:
        array = xr.concat(array, dim=xr.IndexVariable("cdf_threshold", _LEVELS))
    if not isinstance(array, xr.DataArray):
        raise TypeError("PBC thresholds must be a two-threshold DataArray or a pair of maps")
    if "quantile" in array.dims:
        array = array.rename(quantile="cdf_threshold")
    if set(array.dims) != {"cdf_threshold", "Y", "X"} or array.sizes["cdf_threshold"] != 2:
        raise ValueError("PBC thresholds require two cdf_threshold/quantile,Y,X maps")
    _unique_coordinates(array, ("cdf_threshold", "Y", "X"))
    labels = np.asarray(array["cdf_threshold"].values, dtype=float)
    if not (np.allclose(labels, _LEVELS) or np.allclose(labels, [.33, .67])):
        raise ValueError("PBC thresholds must be ordered at 1/3,2/3 (or legacy .33,.67)")
    try:
        xr.align(array, reference.isel(probability=0, T=0, drop=True), join="exact")
    except ValueError as exc:
        raise ValueError("PBC threshold grid differs from the forecast grid") from exc
    out = array.transpose("cdf_threshold", "Y", "X").assign_coords(cdf_threshold=_LEVELS)
    q = np.asarray(out.values, dtype=float)
    if np.isinf(q).any() or np.any(q[0] > q[1]):
        raise ValueError("PBC thresholds must be finite or missing, with low <= high")
    return out.astype(float)


def probabilities_to_cdf(probabilities):
    """Convert ordered PB/PN/PA probabilities to the two cumulative values."""
    p = _probabilities(probabilities)
    values = np.stack([p.values[0], p.values[0] + p.values[1]])
    return xr.DataArray(values, dims=("cdf_threshold", "T", "Y", "X"),
                        coords={"cdf_threshold": _LEVELS,
                                **{d: p[d] for d in ("T", "Y", "X")}}, name="raw_cdf")


def project_cdf(cdf):
    """Euclidean projection onto 0 <= C1 <= C2 <= 1, preserving NaNs.

    Pool crossing values BEFORE clipping. Clipping before pooling is not the
    metric projection when the two unbounded values straddle both bounds.
    """
    if not isinstance(cdf, xr.DataArray) or "cdf_threshold" not in cdf.dims or cdf.sizes["cdf_threshold"] != 2:
        raise ValueError("PBC CDF must have a two-element cdf_threshold dimension")
    other = [d for d in cdf.dims if d != "cdf_threshold"]
    source = cdf.transpose("cdf_threshold", *other)
    values = np.asarray(source.values, dtype=float).copy()
    valid = np.isfinite(values).all(axis=0)
    crossing = values[0] > values[1]
    pooled = (values[0] + values[1]) / 2.0
    values[0] = np.where(crossing, pooled, values[0])
    values[1] = np.where(crossing, pooled, values[1])
    values = np.where(valid[None], np.clip(values, 0, 1), np.nan)
    return source.copy(data=values).transpose(*cdf.dims)


def cdf_to_probabilities(cdf):
    """Convert a valid two-threshold CDF into PB/PN/PA probabilities."""
    c = project_cdf(cdf).transpose("cdf_threshold", "T", "Y", "X")
    values = np.stack([c.values[0], c.values[1] - c.values[0], 1 - c.values[1]])
    return xr.DataArray(values, dims=("probability", "T", "Y", "X"),
                        coords={"probability": _CATEGORIES,
                                **{d: c[d] for d in ("T", "Y", "X")}}, name="probabilities")


def _feature_array(array, reference, *, lag=False, label="feature"):
    expected = {"cdf_threshold", "T", "Y", "X"} | ({"lag"} if lag else set())
    if not isinstance(array, xr.DataArray) or set(array.dims) != expected:
        raise ValueError(f"PBC {label} requires dimensions {','.join(sorted(expected))}")
    if array.sizes["cdf_threshold"] != 2 or (lag and array.sizes["lag"] < 1):
        raise ValueError(f"PBC {label} requires two thresholds" + (" and at least one lag" if lag else ""))
    _unique_coordinates(array, tuple(expected))
    labels = np.asarray(array["cdf_threshold"].values, dtype=float)
    if not (np.allclose(labels, _LEVELS) or np.allclose(labels, [.33, .67])):
        raise ValueError(f"PBC {label} thresholds must be ordered 1/3,2/3")
    array = array.assign_coords(cdf_threshold=_LEVELS)
    if lag:
        lag_labels = np.asarray(array["lag"].values)
        if (not np.issubdtype(lag_labels.dtype, np.integer)
                or not np.array_equal(lag_labels, np.arange(1, len(lag_labels) + 1))):
            raise ValueError("PBC lag indicators must be ordered with integer lag=[1,...,N]")
    try:
        xr.align(array, reference, join="exact")
    except ValueError as exc:
        raise ValueError(f"PBC {label} coordinates differ from the forecast") from exc
    dims = ("cdf_threshold", "T", "lag", "Y", "X") if lag else ("cdf_threshold", "T", "Y", "X")
    values = np.asarray(array.transpose(*dims).values, dtype=float)
    finite = np.isfinite(values)
    if np.isinf(values).any() or ((values[finite] < 0) | (values[finite] > 1)).any():
        raise ValueError(f"PBC {label} must contain probabilities/indicators in [0,1] or NaN")
    if np.any(values[0] > values[1]):
        raise ValueError(f"PBC {label} must be cumulative: threshold 1 <= threshold 2")
    return values


class WAS_PBC:
    """Season-specific PBC with fixed climatological terciles.

    ``alpha`` penalizes the mean squared residual regression around the raw
    forecast, i.e. mean((O-F)-X delta)^2 + alpha*||delta[1:]||^2.
    The intercept is unpenalized. ``alpha=0`` recovers ordinary least squares
    (minimum-norm solution) for the supplied seasonal archive. The default
    ``alpha=1`` shrinks short-record slopes without selecting on test years.
    ``min_samples`` is per grid cell. Insufficient cells retain raw forecasts;
    unavailable persistence features retain the Debias++ branch.

    Lag indicators and climatology must be constructed with information that
    was available at EACH sample's issuance date. This class deliberately does
    not infer observed lags by shifting annual targets into an emission date.
    The number of persistence windows is inferred from the supplied ``lag``
    dimension, ordered newest first with integer labels 1..N. Sliding windows
    remain separate predictors, not additional independent training samples.
    """

    def __init__(self, alpha=1.0, min_samples=8, persistence=True):
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("PBC alpha must be finite and nonnegative")
        if int(min_samples) != min_samples or min_samples < 1:
            raise ValueError("PBC min_samples must be a positive integer")
        self.alpha = float(alpha)
        self.min_samples = int(min_samples)
        self.persistence = bool(persistence)
        self.state_ = None

    def fit(self, probabilities, observations, thresholds, lag_indicators=None, climatology=None):
        """Fit on an explicitly selected historical, out-of-sample archive."""
        p = _probabilities(probabilities)
        if p.sizes["T"] == 0:
            raise ValueError("PBC cannot fit an empty training archive")
        if not isinstance(observations, xr.DataArray) or set(observations.dims) != {"T", "Y", "X"}:
            raise ValueError("PBC observations require exactly T,Y,X dimensions")
        _unique_coordinates(observations, ("T", "Y", "X"))
        try:
            xr.align(observations, p, join="exact")
        except ValueError as exc:
            raise ValueError("PBC observations and training forecasts must align exactly") from exc
        q = _thresholds(thresholds, p)
        raw = probabilities_to_cdf(p)
        f = raw.values.reshape(2, p.sizes["T"], -1)
        y = observations.transpose("T", "Y", "X").values
        qv = q.values
        target = np.where(np.isfinite(y)[None] & np.isfinite(qv)[:, None],
                          y[None] <= qv[:, None], np.nan).reshape(f.shape)
        valid = np.isfinite(target) & np.isfinite(f)
        count = valid.sum(axis=1)
        enough = count >= self.min_samples
        residual = np.where(valid, target - f, 0.0)
        bias = np.divide(residual.sum(axis=1), count, out=np.zeros_like(count, dtype=float), where=count > 0)
        bias = np.where(enough, bias, 0.0)
        empirical = np.divide(np.where(valid, target, 0).sum(axis=1), count,
                              out=np.full_like(bias, np.nan), where=count > 0)
        if climatology is None:
            c = np.broadcast_to(empirical[:, None], f.shape).copy()
            climatology_mode = "training_empirical_frequency"
        else:
            c = _feature_array(climatology, raw, label="climatology").reshape(f.shape)
            climatology_mode = "supplied_asof_climatology"
        lags = None
        n_lags = 0
        if lag_indicators is not None:
            lag_values = _feature_array(lag_indicators, raw, lag=True, label="lag indicators")
            n_lags = lag_values.shape[2]
            lags = lag_values.reshape(2, p.sizes["T"], n_lags, -1)
        feature_names = _feature_names(n_lags)
        n_features = len(feature_names) - 1
        coef = np.zeros((2, f.shape[-1], n_features + 1))
        mean = np.full((2, f.shape[-1], n_features), np.nan)
        used = np.zeros((2, f.shape[-1]), dtype=bool)
        if self.persistence and lags is not None:
            features = np.concatenate([c[..., None], np.moveaxis(lags, 2, -1), f[..., None]], axis=-1)
            for k in range(2):
                for cell in np.flatnonzero(enough[k]):
                    take = valid[k, :, cell]
                    z = features[k, take, cell]
                    feature_counts = np.isfinite(z).sum(axis=0)
                    # A wholly unavailable lag cannot acquire a meaningful
                    # coefficient through an arbitrary fill value.
                    if np.any(feature_counts == 0):
                        continue
                    m = np.nansum(z, axis=0) / feature_counts
                    mean[k, cell] = m
                    z = np.where(np.isfinite(z), z, m)
                    design = np.column_stack([np.ones(z.shape[0]), z])
                    response = (target[k, take, cell] - f[k, take, cell])
                    if self.alpha == 0:
                        # Solve the authors' direct OLS, including its
                        # minimum-norm convention when regressors are
                        # collinear, then store relative to the raw prior.
                        delta = np.linalg.lstsq(design, target[k, take, cell], rcond=None)[0]
                        delta[-1] -= 1.0
                    else:
                        gram = design.T @ design / design.shape[0]
                        gram[1:, 1:] += self.alpha * np.eye(n_features)
                        delta = np.linalg.solve(gram, design.T @ response / design.shape[0])
                    if np.isfinite(delta).all():
                        coef[k, cell] = delta
                        used[k, cell] = True
        grid = (p.sizes["Y"], p.sizes["X"])
        coords = {"cdf_threshold": _LEVELS, "Y": p["Y"], "X": p["X"],
                  "regressor": feature_names, "feature": feature_names[1:]}
        self.state_ = xr.Dataset({
            "thresholds": q,
            "debias_offset": (("cdf_threshold", "Y", "X"), bias.reshape((2,) + grid)),
            "sample_count": (("cdf_threshold", "Y", "X"), count.reshape((2,) + grid)),
            "training_count": (("cdf_threshold", "Y", "X"), count.reshape((2,) + grid)),
            "persistence_used": (("cdf_threshold", "Y", "X"), used.reshape((2,) + grid).astype(np.int8)),
            "persistence_count": (("cdf_threshold", "Y", "X"), np.where(used, count, 0).reshape((2,) + grid)),
            "residual_coefficients": (("cdf_threshold", "Y", "X", "regressor"), coef.reshape((2,) + grid + (n_features + 1,))),
            "feature_mean": (("cdf_threshold", "Y", "X", "feature"), mean.reshape((2,) + grid + (n_features,))),
            "training_frequency": (("cdf_threshold", "Y", "X"), empirical.reshape((2,) + grid)),
        }, coords=coords, attrs={
            "schema": _SCHEMA, "method": "PBC adapted to seasonal WASS2S",
            "source_doi": "10.1038/s41586-026-11126-x", "alpha": self.alpha,
            "min_samples": self.min_samples, "persistence_requested": int(self.persistence),
            "persistence_inputs_supplied": int(lags is not None),
            "persistence_lag_count": int(n_lags),
            "missing_feature_policy": "training-mean imputation; entirely missing training feature disables persistence for that cell and threshold",
            "climatology_mode": climatology_mode,
            "ridge_objective": "mean((O-F-X*delta)^2)+alpha*sum(delta_slopes^2); intercept unpenalized",
            "event_definition": "PB:y<=q1; PN:q1<y<=q2; PA:y>q2",
            "training_dates": ",".join(map(str, p["T"].values)),
            "training_date_count": p.sizes["T"],
            "evaluation_scope": "caller must establish forecast archive independence and as-of availability",
        })
        self.thresholds_ = q
        self.persistence_lag_count_ = n_lags
        return self

    def predict_components(self, probabilities, lag_indicators=None, climatology=None):
        """Return raw and separately projected correction branches plus PBC."""
        if self.state_ is None:
            raise RuntimeError("PBC must be fitted before predict")
        p = _probabilities(probabilities)
        _thresholds(self.state_["thresholds"], p)
        raw = probabilities_to_cdf(p)
        f = raw.values
        offset = self.state_["debias_offset"].values[:, None]
        # Algorithms 1 and 2 first clip each corrected threshold; their final
        # projection then enforces monotonicity across thresholds.
        debias = project_cdf(raw.copy(data=np.clip(f + offset, 0, 1))).rename("debias_cdf")
        persistence_values = debias.values.copy()
        lags = None
        if lag_indicators is not None:
            lags = _feature_array(lag_indicators, raw, lag=True, label="lag indicators")
            if lags.shape[2] != self.state_.attrs["persistence_lag_count"]:
                raise ValueError("PBC query lag count differs from the fitted persistence lag count")
        if self.persistence and lags is not None:
            if climatology is None:
                if self.state_.attrs["climatology_mode"] == "supplied_asof_climatology":
                    raise ValueError("PBC requires query climatology because fit used supplied as-of climatology")
                c = np.broadcast_to(self.state_["training_frequency"].values[:, None], f.shape)
            else:
                c = _feature_array(climatology, raw, label="climatology")
            features = np.concatenate([c[..., None], np.moveaxis(lags, 2, -1), f[..., None]], axis=-1)
            means = self.state_["feature_mean"].values[:, None]
            features = np.where(np.isfinite(features), features, means)
            coef = self.state_["residual_coefficients"].values[:, None]
            correction = coef[..., 0] + np.sum(features * coef[..., 1:], axis=-1)
            available = (self.state_["persistence_used"].values[:, None].astype(bool)
                         & np.isfinite(features).all(axis=-1))
            persistence_values = np.where(available, f + correction, persistence_values)
        persistence = project_cdf(raw.copy(data=np.clip(persistence_values, 0, 1))).rename("persistence_cdf")
        combined = ((debias + persistence) / 2).rename("pbc_cdf")
        # Preserve a missing base distribution: no branch creates weather
        # forecasts where the underlying model itself supplied none.
        combined = combined.where(np.isfinite(raw).all("cdf_threshold")
                                  & np.isfinite(self.state_["thresholds"]).all("cdf_threshold"))
        result = cdf_to_probabilities(combined)
        result.attrs.update(p.attrs)
        result.attrs.update({
            "probability_postprocessing": "seasonal_pbc",
            "pbc_source_doi": "10.1038/s41586-026-11126-x",
            "pbc_methodology": "seasonal adaptation; Debias++; regularized Persistence++; per-branch isotonic projection; equal mixture",
            "pbc_alpha": self.alpha, "pbc_min_samples": self.min_samples,
            "pbc_persistence_requested": int(self.persistence),
            "pbc_query_lags_supplied": int(lag_indicators is not None),
            "pbc_persistence_lag_count": int(self.state_.attrs["persistence_lag_count"]),
            "pbc_missing_feature_policy": "training-mean imputation; unavailable persistence falls back to Debias++",
            "tercile_reference": "climatology", "tercile_quantile_levels": "1/3,2/3",
            "category_definition": "PB: y <= low; PN: low < y <= high; PA: y > high",
        })
        for key, value in self.state_["thresholds"].attrs.items():
            if key.startswith("climatology_"):
                result.attrs[key] = value
        return xr.Dataset({"raw_cdf": raw, "debias_cdf": debias,
                           "persistence_cdf": persistence, "pbc_cdf": combined,
                           "probabilities": result})

    def predict(self, probabilities, lag_indicators=None, climatology=None):
        """Apply the learned correction to a forecast on the unchanged grid."""
        return self.predict_components(probabilities, lag_indicators, climatology)["probabilities"]

    def save(self, path):
        """Persist the numerical state as NetCDF, without pickled Python code."""
        if self.state_ is None:
            raise RuntimeError("PBC must be fitted before save")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.state_.to_netcdf(path)
        return path

    @classmethod
    def load(cls, path):
        state = xr.load_dataset(path)
        source_schema = state.attrs.get("schema")
        if source_schema not in {_SCHEMA, _LEGACY_SCHEMA}:
            raise ValueError("Unrecognized PBC checkpoint schema")
        needed = {"thresholds", "debias_offset", "sample_count", "persistence_used",
                  "residual_coefficients", "feature_mean", "training_frequency"}
        if not needed.issubset(state.data_vars):
            raise ValueError("Incomplete PBC checkpoint")
        n_lags = 2 if source_schema == _LEGACY_SCHEMA else state.attrs.get("persistence_lag_count")
        if (not isinstance(n_lags, (int, np.integer)) or isinstance(n_lags, (bool, np.bool_))
                or n_lags < 0):
            raise ValueError("Invalid PBC checkpoint persistence lag count")
        expected_names = _feature_names(int(n_lags))
        if ("regressor" not in state.coords or "feature" not in state.coords
                or list(map(str, state["regressor"].values)) != expected_names
                or list(map(str, state["feature"].values)) != expected_names[1:]
                or state["residual_coefficients"].dims != ("cdf_threshold", "Y", "X", "regressor")
                or state["feature_mean"].dims != ("cdf_threshold", "Y", "X", "feature")):
            raise ValueError("PBC checkpoint feature layout disagrees with its persistence lag count")
        if source_schema == _LEGACY_SCHEMA:
            state.attrs.update(schema=_SCHEMA, source_checkpoint_schema=_LEGACY_SCHEMA,
                               persistence_lag_count=2)
        if "training_count" not in state:
            state["training_count"] = state["sample_count"]
        if "persistence_count" not in state:
            state["persistence_count"] = xr.where(state["persistence_used"], state["sample_count"], 0)
        obj = cls(alpha=state.attrs["alpha"], min_samples=state.attrs["min_samples"],
                  persistence=bool(state.attrs["persistence_requested"]))
        obj.state_ = state
        obj.thresholds_ = state["thresholds"]
        obj.persistence_lag_count_ = int(n_lags)
        return obj
