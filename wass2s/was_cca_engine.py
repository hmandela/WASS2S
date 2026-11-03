"""Exact, deterministic CCA engine shared by every CCA branch of WASS2S.

Why this module exists
----------------------
``xeofs.cross.CCA`` (xeofs 3.x) is ``CPCCA(alpha=[0, 0])``: preprocessing
(centre, optional standardisation, sqrt-cos(lat) weights) -> PCA of each field
-> whitening -> SVD of the whitened cross-covariance -> regression of the Y
canonical scores on the X ones -> back-projection. Two properties of that
implementation made the CCA branches of WASS2S slow and not reproducible:

* The PCA step always goes through ``sklearn.utils.extmath.randomized_svd``
  (xeofs uses the exact SVD only for < 500 features AND more than 80 % of the
  rank requested) and ``BaseModelCrossSet`` never forwards ``random_state``
  to its two PCA preprocessors. Four identical calls on the evaluation data
  differed by up to 6 % (relative) at ``(3, [6, 6])``.
* Every candidate truncation ``(n_modes, n_pca_X, n_pca_Y)`` of a tuning grid
  redoes the whole pipeline, although the PCA of a field does not depend on
  the candidate: with an exact SVD the ``k``-mode truncation is the first
  ``k`` columns of the full decomposition.

With alpha = 0 the whole chain collapses to closed-form algebra. Write the
preprocessed training fields ``X = Ux Sx Vxᵀ`` and ``Y = Uy Sy Vyᵀ`` (exact
thin SVDs) and ``K = Ux[:, :nx]ᵀ Uy[:, :ny] = A diag(rho) Bᵀ``: ``rho`` are the
canonical correlations, and the xeofs prediction of a new predictor ``x`` is,
after the scaling constants of the whitening cancel,

    y = x Vx[:, :nx] Sx⁻¹ A_r diag(rho_r) B_rᵀ Sy[:ny] Vy[:, :ny]ᵀ

(then un-weighted, un-standardised and re-centred). One exact SVD per field
and per training fold serves every candidate; each candidate costs an
``nx x ny`` SVD and one back-projection. The engine reproduces xeofs to
machine precision when xeofs is forced onto the exact solver (see
``tests/test_cca_engine.py``), and is deterministic by construction.

The xeofs model object is still available for the diagnostics that need it
(components, scores, explained variances in ``plot_cca_results``) through
:class:`LazyXeofsCCA`, which fits it on first use only.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

__all__ = ["FieldBasis", "ExactCCA", "LazyXeofsCCA", "exact_cca_supported",
           "exact_pca_truncation", "bartlett_aic", "aic_truncation_table"]

_LAT_NAMES = ("lat", "latitude", "Y")


def exact_cca_supported(use_pca=True, n_pca_modes=None):
    """Whether a configuration is handled by :class:`ExactCCA`.

    Integer (or ``"all"``) PCA truncations with the PCA pre-filter on. A
    variance-fraction truncation (float) or ``use_pca=False`` keep the xeofs
    path, whose behaviour for those options is not reproduced here.
    """
    if not use_pca:
        return False
    seq = n_pca_modes if isinstance(n_pca_modes, (list, tuple, np.ndarray)) else [n_pca_modes]
    for v in seq:
        if isinstance(v, str):
            if v != "all":
                return False
        elif isinstance(v, (bool, np.bool_)) or not float(v).is_integer():
            return False
    return True


def exact_pca_truncation(v, rank):
    """Integer truncation for ``v`` (int or ``"all"``) bounded by ``rank``."""
    k = rank if isinstance(v, str) else int(v)
    return max(1, min(k, rank))


def _coslat_weights(template):
    lat = next((n for n in _LAT_NAMES if n in template.dims), None)
    if lat is None:
        raise ValueError(
            "use_coslat=True needs a latitude dimension (lat, latitude or Y); "
            f"got {template.dims}.")
    w = np.sqrt(np.clip(np.cos(np.deg2rad(template[lat].astype(float))), 0.0, 1.0))
    ones = xr.DataArray(np.ones(template.shape), dims=template.dims,
                        coords={d: template[d] for d in template.dims if d in template.coords})
    return (ones * w).transpose(*template.dims).values.ravel()


class FieldBasis:
    """Exact PCA basis of one training field (sample dimension first).

    Mirrors the xeofs preprocessing: features with a missing value are
    dropped, then ``(x - mean) / std * sqrt(cos(lat))`` with ``std`` the
    population standard deviation clipped at float32 epsilon (only when
    ``standardize``). The thin SVD ``U S Vᵀ`` of the result is exact (LAPACK)
    and truncated at its numerical rank, so every PCA truncation is a prefix.
    """

    def __init__(self, da, standardize=False, use_coslat=True, sample="T"):
        da = da.transpose(sample, ...)
        self.sample = sample
        self.template = da.isel({sample: 0}, drop=True)
        n = int(da.sizes[sample])
        values = np.asarray(da.values, dtype=np.float64).reshape(n, -1)
        self.valid = np.isfinite(values).all(axis=0)
        if not self.valid.any():
            raise ValueError("No feature is observed at every training time step.")
        A = values[:, self.valid]
        self.mean = A.mean(axis=0)
        A = A - self.mean
        if standardize:
            self.std = np.clip(A.std(axis=0), np.finfo(np.float32).eps, None)
            A /= self.std
        else:
            self.std = None
        if use_coslat:
            self.weight = _coslat_weights(self.template)[self.valid]
            A *= self.weight
        else:
            self.weight = None
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        tol = (s[0] if s.size else 0.0) * max(A.shape) * np.finfo(np.float64).eps
        rank = int(np.count_nonzero(s > tol))
        if rank < 1:
            raise ValueError("The training field has no variance.")
        self.U, self.s, self.Vt = U[:, :rank], s[:rank], Vt[:rank]
        self.rank = rank
        self.n_samples = n

    # -- forward / backward -------------------------------------------------
    def _scaled(self, da):
        da = da.transpose(self.sample, ...)
        m = int(da.sizes[self.sample])
        B = np.asarray(da.values, dtype=np.float64).reshape(m, -1)[:, self.valid] - self.mean
        if self.std is not None:
            B = B / self.std
        if self.weight is not None:
            B = B * self.weight
        return B

    def project(self, da):
        """Unnormalised PC scores (``m x rank``) of new data."""
        return self._scaled(da) @ self.Vt.T

    def reconstruct(self, scores, coords):
        """Field (sample first) from PC scores ``m x k`` (``k <= rank``)."""
        scores = np.atleast_2d(scores)
        k = scores.shape[1]
        B = scores @ self.Vt[:k]
        if self.weight is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                B = B / self.weight
        if self.std is not None:
            B = B * self.std
        B = B + self.mean
        out = np.full((B.shape[0], self.valid.size), np.nan)
        out[:, self.valid] = B
        dims = (self.sample,) + tuple(self.template.dims)
        shape = (B.shape[0],) + tuple(self.template.shape)
        crd = {d: self.template[d] for d in self.template.dims if d in self.template.coords}
        crd[self.sample] = coords
        return xr.DataArray(out.reshape(shape), dims=dims, coords=crd)


class ExactCCA:
    """``xe.cross.CCA`` (alpha = 0, PCA pre-filter) in closed form.

    Build it once per training fold; :meth:`predict` then serves any
    ``(n_modes, [n_pca_X, n_pca_Y])``. Mode counts are further capped by the
    numerical rank of each field (the callers already cap them by the sample
    size, exactly as they did for xeofs).
    """

    def __init__(self, X, Y, standardize=False, use_coslat=True, sample="T"):
        nx, ny = int(X.sizes[sample]), int(Y.sizes[sample])
        if nx != ny:
            raise ValueError(f"X and Y must share the sample dimension ({nx} != {ny}).")
        self.sample = sample
        self.bx = FieldBasis(X, standardize, use_coslat, sample)
        self.by = FieldBasis(Y, standardize, use_coslat, sample)
        self._coef = {}

    def resolve(self, n_modes, n_pca_modes):
        px, py = (n_pca_modes if isinstance(n_pca_modes, (list, tuple, np.ndarray))
                  else (n_pca_modes, n_pca_modes))
        kx = exact_pca_truncation(px, self.bx.rank)
        ky = exact_pca_truncation(py, self.by.rank)
        if int(n_modes) < 0:
            raise ValueError("n_modes must be nonnegative.")
        r = min(int(n_modes), kx, ky)
        return r, kx, ky

    def solve(self, n_modes, n_pca_modes):
        """``(coef, rho)``: PC-space regression matrix ``kx x ky`` and canonical correlations."""
        key = self.resolve(n_modes, n_pca_modes)
        if key not in self._coef:
            r, kx, ky = key
            K = self.bx.U[:, :kx].T @ self.by.U[:, :ky]
            A, rho, Bt = np.linalg.svd(K, full_matrices=False)
            coef = ((A[:, :r] / self.bx.s[:kx, None]) * rho[:r]) @ (Bt[:r] * self.by.s[:ky])
            self._coef[key] = (coef, rho[:r])
        return self._coef[key]

    def canonical_correlations(self, n_modes, n_pca_modes):
        return self.solve(n_modes, n_pca_modes)[1]

    def predict_scores(self, Zx, n_modes, n_pca_modes):
        coef, _ = self.solve(n_modes, n_pca_modes)
        return Zx[:, :coef.shape[0]] @ coef

    def predict(self, X_test, n_modes, n_pca_modes, Zx=None):
        """Predicted Y field for ``X_test`` (same dims as the training Y)."""
        X_test = X_test.transpose(self.sample, ...)
        if Zx is None:
            Zx = self.bx.project(X_test)
        return self.by.reconstruct(self.predict_scores(Zx, n_modes, n_pca_modes),
                                   X_test[self.sample].values)

    def predict_many(self, X_test, candidates):
        """Predictions aligned with ``candidates``; the test field is projected once."""
        X_test = X_test.transpose(self.sample, ...)
        Zx = self.bx.project(X_test)
        return [self.predict(X_test, c[0], c[1], Zx=Zx) for c in candidates]


class LazyXeofsCCA:
    """Stand-in for ``cca_model``: the xeofs object is fitted on first use.

    Predictions of the WASS2S classes come from :class:`ExactCCA`. The
    xeofs model is only needed for diagnostics (components, scores,
    explained-variance fractions); this proxy fits it with ``factory()`` on
    the very training fields the engine used when such an attribute is
    first requested, and delegates to it. Its PCA uses xeofs's solver (a
    seeded randomized SVD), so its patterns agree with the engine's to the
    accuracy of that solver, not to machine precision.
    """

    def __init__(self, factory, X, Y, engine=None, sample="T"):
        self.__dict__.update(_factory=factory, _X=X, _Y=Y, _fitted=None,
                             engine=engine, _sample=sample)

    def _model(self):
        if self.__dict__["_fitted"] is None:
            self.__dict__["_fitted"] = self._factory().fit(self._X, self._Y, dim=self._sample)
        return self.__dict__["_fitted"]

    def __getattr__(self, name):
        if name.startswith("__") or name in ("_factory", "_X", "_Y", "_fitted", "engine", "_sample"):
            raise AttributeError(name)
        return getattr(self._model(), name)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_fitted"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


# ---------------------------------------------------------------------------
# Truncation by an information criterion (no inner cross-validation)
# ---------------------------------------------------------------------------
def bartlett_aic(rho, n_samples, kx, ky, *, detrend_dof=0):
    """Experimental Bartlett-AIC score in one fixed pair of PCA spaces.

    Under independence, Bartlett's statistic
    ``-m * sum(log(1 - rho_i**2))`` with ``m = n - 1 - (kx + ky + 1) / 2`` is
    approximately chi-square with ``kx * ky`` degrees of freedom; a rank-r
    cross-covariance has ``d = r (kx + ky - r)`` free parameters. The criterion

        IC = m * sum_{i <= r} log(1 - rho_i**2) + 2 d

    is used as a heuristic with a small-sample likelihood correction. Scores
    across different PCA spaces are not comparable likelihoods for the same
    observed data. Selection across those spaces remains experimental.
    Centering and detrending reduce the temporal rank; saturated PCA spaces
    and numerically unit correlations must not manufacture overwhelming
    evidence by clipping their log singularity. Rank zero is a valid baseline.
    """
    rho = np.asarray(rho, dtype=float)
    r = rho.size
    if r == 0:
        return 0.0
    tolerance = 64 * np.finfo(float).eps * max(n_samples, kx, ky)
    if (kx + ky >= n_samples - 1 - detrend_dof or
            not np.isfinite(rho).all() or np.any(rho < 0) or
            np.any(rho >= 1.0 - tolerance)):
        return np.inf
    m = float(n_samples) - 1.0 - (kx + ky + 1.0) / 2.0
    if m <= 0:
        return np.inf
    return float(m * np.sum(np.log1p(-rho ** 2)) + 2.0 * r * (kx + ky - r))


def aic_truncation_table(engine, candidates, n_samples, cap=None, *, detrend_dof=0):
    """``[(candidate, resolved (r, kx, ky), IC)]`` for an :class:`ExactCCA`.

    ``cap(n_modes, n_pca_modes, n_samples) -> (n_modes, n_pca_modes)`` applies
    the caller's sample-size caps (the same as for the fits).
    """
    rows = []
    for n_modes, n_pca in candidates:
        nm, npca = cap(n_modes, n_pca, n_samples) if cap else (n_modes, n_pca)
        if n_modes == 0:
            nm = 0  # legacy sample caps force >=1; the null model must survive
        r, kx, ky = engine.resolve(nm, npca)
        rho = engine.canonical_correlations(nm, npca)
        rows.append(((n_modes, n_pca), (r, kx, ky),
                     bartlett_aic(rho, n_samples, kx, ky, detrend_dof=detrend_dof)))
    return rows
