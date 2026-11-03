"""Extreme Learning Machine by Gram matrices (replaces the ``hpelm`` package).

An ELM is a ridge regression on a random hidden layer ``H = g(X W + B)``:

    beta = (FᵀF + lambda D)⁻¹ Fᵀy

where ``F`` holds the hidden units (and, optionally, an unpenalised intercept
and the raw inputs, see ``linear_part``) and ``D`` is the identity with 0 on the
intercept. ``hpelm`` computes exactly this (``HᵀH`` accumulated by BLAS
``syrk`` over batches, then Cholesky ``posv``) with ``F = H`` and
``lambda = norm``. Doing it here brings three things hpelm cannot:

* **Nested candidates.** The random layer is drawn neuron by neuron, so the
  ``L``-neuron network is the first ``L`` columns of the ``L_max`` one and its
  Gram matrix is the leading block. ``norm`` only moves the diagonal. One pass
  over the training rows therefore serves every ``(neurons, norm)`` candidate
  of a tuning grid; each candidate costs a ``k x k`` Cholesky
  (:class:`ELMGramBank`).
* **An explicit linear part** (``linear_part=True``): ``F = [1 | X | H]``. With
  ``neurons=0`` the model is exactly a ridge regression on the inputs, so the
  tuning grid contains its own baseline and the neurons only have to explain
  what the linear part does not.
* **Seed averaging** (``n_seeds``): the prediction is the mean of ``n_seeds``
  networks with independent random layers, which removes most of the variance
  due to the random draw -- the main weakness of an ELM.

With ``linear_part=False`` and ``n_seeds=1`` (the defaults of
:class:`ELMRegressor`, used by the historical ``HPELMWrapper`` name) the
prediction is identical to hpelm's for the same ``W``, ``B`` and ``norm``
(``tests/test_elm_gram.py``). Activations follow hpelm's definitions,
including its sign convention ``sigm = 1 / (1 + exp(XW + B))`` and
``rbf_* = exp(-dist(X, W)² / B)`` with centres ``W`` and widths ``B``.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, RegressorMixin

__all__ = ["ACTIVATIONS", "effective_norm", "random_layer", "hidden_layer",
           "ELMGramBank", "ELMRegressor"]

ACTIVATIONS = ("lin", "tanh", "sigm", "rbf_l1", "rbf_l2", "rbf_linf")
_RBF_METRIC = {"rbf_l1": "cityblock", "rbf_l2": "euclidean", "rbf_linf": "chebyshev"}
# Bounded work per batch of rows (8 * rows * columns bytes).
_BATCH_BYTES = 256 * 2 ** 20


def effective_norm(norm, n_rows, scaling="per_sample"):
    """Ridge added to the diagonal of ``FᵀF``.

    ``"absolute"`` is hpelm's meaning (``norm`` as is, whose effect vanishes as
    ``FᵀF`` grows with the number of rows); ``"per_sample"`` is a ridge per row,
    ``norm * n_rows``, transferable across sample sizes.
    """
    norm = float(norm)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("ELM norm must be finite and strictly positive.")
    if str(scaling) == "per_sample":
        return norm * max(1, int(n_rows))
    if str(scaling) != "absolute":
        raise ValueError("norm_scaling must be 'per_sample' or 'absolute'.")
    return norm


def random_layer(seed, X, neurons, activation, scaling="per_sample"):
    """Random hidden layer ``(W, B)`` drawn neuron by neuron (nested in ``neurons``).

    ``W`` comes from ``RandomState(seed)`` as a ``(neurons, n_features)`` block
    (row-major, so the first ``L`` rows do not depend on ``neurons``) and ``B``
    from an independent stream, hence ``random_layer(s, X, L)`` is the first
    ``L`` neurons of ``random_layer(s, X, L_max)``. ``per_sample`` scaling
    divides ``W`` by sqrt(n_features) (unit-variance pre-activations; without
    it tanh/sigm saturate with 20-50 channels). RBF neurons take their centres
    from the rows of ``X`` and positive widths.
    """
    X = np.asarray(X, dtype=np.float64)
    n_features, neurons = X.shape[1], int(neurons)
    rng_w = np.random.RandomState(seed)
    rng_b = np.random.RandomState((int(seed) + 1_000_003) % (2 ** 32))
    if str(activation).startswith("rbf"):
        idx = rng_w.randint(0, X.shape[0], size=neurons)
        W = X[idx].T.copy()
        scale = float(np.nanstd(X)) or 1.0
        B = np.abs(rng_b.randn(neurons)) * scale + 1e-6
        return W, B
    if activation not in ACTIVATIONS:
        raise ValueError(f"Unknown ELM activation {activation!r}; use one of {ACTIVATIONS}.")
    W = rng_w.randn(neurons, n_features).T
    if str(scaling) == "per_sample":
        W = W / np.sqrt(max(1, n_features))
    B = rng_b.randn(neurons)
    return W, B


def hidden_layer(X, W, B, activation):
    """``g(X W + B)`` with hpelm's definitions."""
    if activation in _RBF_METRIC:
        return np.exp(-cdist(X, W.T, _RBF_METRIC[activation]) ** 2 / B)
    Z = X @ W + B
    if activation == "lin":
        return Z
    if activation == "tanh":
        return np.tanh(Z)
    if activation == "sigm":
        with np.errstate(over="ignore"):
            return 1.0 / (1.0 + np.exp(Z))
    raise ValueError(f"Unknown ELM activation {activation!r}; use one of {ACTIVATIONS}.")


class _Layer:
    """Design matrix ``F = [1 | X | g(XW + B)]`` of one random draw."""

    def __init__(self, W, B, activation, linear_part):
        self.W, self.B, self.activation = W, B, activation
        self.linear_part = bool(linear_part)
        self.n_linear = (1 + W.shape[0]) if self.linear_part else 0

    def columns(self, neurons):
        return self.n_linear + int(neurons)

    def design(self, X, neurons=None):
        L = self.W.shape[1] if neurons is None else int(neurons)
        parts = []
        if self.linear_part:
            parts += [np.ones((X.shape[0], 1)), X]
        if L:
            parts.append(hidden_layer(X, self.W[:, :L], self.B[:L], self.activation))
        return np.hstack(parts) if len(parts) > 1 else parts[0]

    def penalty_mask(self, k):
        d = np.ones(k)
        if self.linear_part:
            d[0] = 0.0  # intercept not penalised
        return d


def _batches(n_rows, n_cols):
    step = max(1000, int(_BATCH_BYTES // (8 * max(1, n_cols))))
    for start in range(0, n_rows, step):
        yield slice(start, min(n_rows, start + step))


def _solve(G, c, lam, d):
    A = G + np.diag(lam * d)
    try:
        return cho_solve(cho_factor(A, check_finite=False), c, check_finite=False)
    except LinAlgError:
        return np.linalg.lstsq(A, c, rcond=None)[0]


class ELMGramBank:
    """Gram matrices of one training set, for all candidates of a grid.

    ``draws`` random layers are built per activation at ``max_neurons``;
    ``FᵀF`` and ``Fᵀy`` are accumulated once, by batches of rows. Then
    :meth:`predict` returns the (seed-averaged) predictions of any list of
    ``(activation, neurons, norm)`` candidates for new rows at the cost of one
    ``k x k`` solve per candidate and draw.
    """

    def __init__(self, X, y, activations, max_neurons, seeds=(42,),
                 norm_scaling="per_sample", linear_part=True):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        if X.ndim != 2 or X.shape[0] != y.size or X.shape[0] == 0:
            raise ValueError("ELM needs a non-empty (rows, features) matrix and one target per row.")
        self.n_rows = X.shape[0]
        self.norm_scaling = norm_scaling
        self.layers = {}
        for act in dict.fromkeys(activations):
            draws = []
            for seed in seeds:
                W, B = random_layer(seed, X, max_neurons, act, norm_scaling)
                layer = _Layer(W, B, act, linear_part)
                k = layer.columns(max_neurons)
                if k == 0:
                    raise ValueError("An ELM without linear part needs at least one neuron.")
                G, c = np.zeros((k, k)), np.zeros(k)
                for sl in _batches(self.n_rows, k):
                    F = layer.design(X[sl])
                    G += F.T @ F
                    c += F.T @ y[sl]
                draws.append((layer, G, c))
            self.layers[act] = draws

    def coefficients(self, activation, neurons, norm):
        """One coefficient vector per random draw."""
        lam = effective_norm(norm, self.n_rows, self.norm_scaling)
        out = []
        for layer, G, c in self.layers[activation]:
            k = layer.columns(neurons)
            if k == 0:
                raise ValueError("An ELM without linear part needs at least one neuron.")
            out.append((layer, int(neurons), _solve(G[:k, :k], c[:k], lam, layer.penalty_mask(k))))
        return out

    def predict(self, X, candidates):
        """Seed-averaged predictions, one array per ``(activation, neurons, norm)``."""
        X = np.asarray(X, dtype=np.float64)
        results = [None] * len(candidates)
        by_act = {}
        for i, cand in enumerate(candidates):
            by_act.setdefault(cand[0], []).append(i)
        for act, idx in by_act.items():
            sums = {i: np.zeros(X.shape[0]) for i in idx}
            coefs = {i: self.coefficients(*candidates[i]) for i in idx}
            for d, (layer, _, _) in enumerate(self.layers[act]):
                for sl in _batches(X.shape[0], layer.columns(layer.W.shape[1])):
                    F = layer.design(X[sl])
                    for i in idx:
                        _, L, beta = coefs[i][d]
                        sums[i][sl] += F[:, :layer.columns(L)] @ beta
            n_draws = len(self.layers[act])
            for i in idx:
                results[i] = sums[i] / n_draws
        return results


class ELMRegressor(RegressorMixin, BaseEstimator):
    """scikit-learn ELM regressor (fit / predict), numpy only.

    ``neurons``, ``activation`` and ``norm`` have hpelm's meaning;
    ``norm_scaling`` see :func:`effective_norm`; ``linear_part`` and
    ``n_seeds`` see the module docstring. Seeds are
    ``random_state, random_state + 1, ...``.
    """

    def __init__(self, neurons=10, activation="sigm", norm=1.0, random_state=42,
                 norm_scaling="absolute", linear_part=False, n_seeds=1):
        self.neurons = neurons
        self.activation = activation
        self.norm = norm
        self.random_state = random_state
        self.norm_scaling = norm_scaling
        self.linear_part = linear_part
        self.n_seeds = n_seeds

    def _seeds(self):
        n = max(1, int(self.n_seeds)) if int(self.neurons) > 0 else 1
        base = 0 if self.random_state is None else int(self.random_state)
        return tuple(base + i for i in range(n))

    def fit(self, X, y):
        bank = ELMGramBank(X, y, [self.activation], int(self.neurons), self._seeds(),
                           self.norm_scaling, self.linear_part)
        self.coef_ = bank.coefficients(self.activation, int(self.neurons), self.norm)
        self.n_features_in_ = int(np.asarray(X).shape[1])
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        out = np.zeros(X.shape[0])
        for layer, L, beta in self.coef_:
            for sl in _batches(X.shape[0], beta.size):
                out[sl] += layer.design(X[sl], L) @ beta
        return out / len(self.coef_)

    def score(self, X, y):
        y = np.asarray(y, dtype=float).ravel()
        return -float(np.mean((self.predict(X) - y) ** 2))
