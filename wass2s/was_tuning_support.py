"""Target-independent groups and training-only preprocessing for spatial tuning.

Geographical groups replace response-derived climate groups during model selection:
inner validation observations must not determine group membership or cell weights.
A shared hyperparameter is selected using a standardized cluster-mean proxy, then
actual operational estimators are fitted separately at every cell as before.
"""
import numpy as np
import xarray as xr
from sklearn.base import BaseEstimator, RegressorMixin, clone


def geographical_cluster_map(data, n_clusters=10, random_state=42):
    """KMeans on unit-sphere grid coordinates, independent of target values.

    All grid locations participate in clustering. Cells without observations are
    masked only after grouping, so missing validation values cannot move centroids.
    """
    from sklearn.cluster import KMeans
    yy, xx = np.meshgrid(np.asarray(data.Y, float), np.asarray(data.X, float), indexing="ij")
    latitude, longitude = np.deg2rad(yy.ravel()), np.deg2rad(xx.ravel())
    coordinates = np.column_stack((np.cos(latitude)*np.cos(longitude),
                                   np.cos(latitude)*np.sin(longitude), np.sin(latitude)))
    if not np.isfinite(coordinates).all():
        raise ValueError("Spatial coordinates must be finite for geographical grouping")
    count = min(max(1, int(n_clusters)), len(coordinates), len(np.unique(coordinates, axis=0)))
    labels = np.zeros(len(coordinates), float) if count == 1 else KMeans(
        n_clusters=count, random_state=random_state, n_init=10).fit_predict(coordinates).astype(float)
    labels = labels.reshape(yy.shape)
    observed = np.isfinite(data).any("T").transpose("Y", "X").values
    labels = np.where(observed, labels, np.nan)
    result = xr.DataArray(labels, dims=("Y", "X"), coords={"Y": data.Y, "X": data.X}, name="cluster")
    result.attrs.update(grouping="target-independent geographical coordinates",
                        effective_clusters=count, tuning_target="fold-local standardized cluster mean")
    return result


# Retain the spelling used in some callers.
geometrical_cluster_map = geographical_cluster_map


class FoldLocalTargetRegressor(RegressorMixin, BaseEstimator):
    """Fit cell scaling and a scalar regional proxy entirely on the fit rows.

    ``y`` may be (sample,) or (sample, cell). Missing cells remain unavailable.
    The underlying estimator sees the mean of training-standardized cell series.
    Use :func:`fold_local_neg_mse` during CV rather than a raw-units MSE scorer.
    """
    def __init__(self, regressor):
        self.regressor = regressor

    def _matrix(self, y):
        y = np.asarray(y, float)
        if y.ndim not in (1, 2):
            raise ValueError("Tuning target must be (sample,) or (sample, cell)")
        return y[:, None] if y.ndim == 1 else y

    def fit(self, X, y):
        X, original_y = np.asarray(X, float), np.asarray(y, float)
        values = self._matrix(original_y)
        self.single_target_ = original_y.ndim == 1
        valid = np.isfinite(X).all(axis=1) & np.isfinite(values).any(axis=1)
        if valid.sum() < 2:
            raise ValueError("At least two finite training rows are required for model selection")
        X, values = X[valid], values[valid]
        counts = np.isfinite(values).sum(axis=0)
        self.center_ = np.divide(np.nansum(values, axis=0), counts,
                                out=np.full(values.shape[1], np.nan), where=counts > 0)
        squared = np.nansum((values-self.center_)**2, axis=0)
        variance = np.divide(squared, counts, out=np.full_like(squared, np.nan), where=counts > 0)
        self.scale_ = np.where(variance > 0, np.sqrt(variance), 1.)
        normalized = (values-self.center_)/self.scale_
        row_count = np.isfinite(normalized).sum(axis=1)
        target = np.divide(np.nansum(normalized, axis=1), row_count,
                           out=np.full(len(values), np.nan), where=row_count > 0)
        keep = np.isfinite(target)
        self.regressor_ = clone(self.regressor)
        params = self.regressor_.get_params(deep=True)
        neighbors = {name: min(int(value), int(keep.sum())) for name,value in params.items()
                     if name.split("__")[-1] == "n_neighbors"}
        if neighbors:
            self.regressor_.set_params(**neighbors)
        self.regressor_.fit(X[keep], target[keep])
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        signal = np.asarray(self.regressor_.predict(X), float).reshape(-1, 1)
        prediction = signal*self.scale_ + self.center_
        return prediction[:, 0] if self.single_target_ else prediction

    def score(self, X, y):
        return fold_local_neg_mse(self, X, y)


def fold_local_neg_mse(estimator, X, y):
    """Mean squared error per valid cell in TRAIN-standardized units."""
    truth = estimator._matrix(y)
    prediction = estimator._matrix(estimator.predict(X))
    errors = (truth-prediction)/estimator.scale_
    finite = np.isfinite(errors)
    if not finite.any():
        raise ValueError("No finite validation targets matching available training cells")
    return -float(np.mean(errors[finite]**2))


def fold_local_regression_score(estimator, X, y, cv=3):
    """Fresh geographical-proxy preprocessing and estimator in each inner fold."""
    from sklearn.model_selection import check_cv
    X, y = np.asarray(X, float), np.asarray(y, float)
    if isinstance(cv, (int, np.integer)):
        cv = min(int(cv), len(X))
        if cv < 2:
            raise ValueError("At least two inner folds are required")
    splitter = check_cv(cv, classifier=False)
    scores = []
    for train, valid in splitter.split(X, y):
        fitted = FoldLocalTargetRegressor(estimator).fit(X[train], y[train])
        scores.append(fold_local_neg_mse(fitted, X[valid], y[valid]))
    if not scores:
        raise ValueError("Inner splitter produced no validation folds")
    return float(np.mean(scores))
