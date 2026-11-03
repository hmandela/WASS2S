"""Training-only minimum-description-length selection for PCA followed by CCA.

Reference
---------
Song, Y., Schreier, P. J., Ramirez, D., and Hasija, T. (2016),
"Canonical correlation analysis of high-dimensional data with very small
sample support", arXiv:1604.02047, equations (4), (14), and Detector 3 (18).
https://arxiv.org/abs/1604.02047

For each admissible pair of PCA ranks, minimize the MDL criterion over
signal ranks, then retain the largest detected signal rank across PCA pairs.
Scores from different PCA spaces are NOT comparable model evidences. The
paper's strict signal-rank bound, s < min(rx, ry), is retained: [1, 1] can
only select rank zero, and detection of d signals needs at least d+1 EOFs
in each field. A deterministic smallest-space tie break is our convention.

The paper assumes independent Gaussian sample pairs (and uses complex data).
For real data, twice the negative log likelihood and the real parameter
count give the same argmin criterion used below. It detects correlated
dimensions, not necessarily dimensions maximizing forecast skill. Temporal
dependence, fitted trends, and climatological transforms can make its nominal
sample-size penalty approximate; validation of predictions remains required.

Only temporal singular vectors from existing training-field decompositions
are used. No spatial decomposition or validation fitting is performed here.
"""
from __future__ import annotations

import numpy as np

__all__ = ["select_cca_mdl"]


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    try:
        integer = int(value)
        valid = np.ndim(value) == 0 and integer == value and integer >= minimum
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return integer


def _basis(basis, name):
    n = _integer(basis.n_samples, f"{name}.n_samples", 2)
    rank = _integer(basis.rank, f"{name}.rank", 1)
    raw = np.asarray(basis.U)
    if np.iscomplexobj(raw):
        raise ValueError("This MDL selector expects real-valued training fields.")
    U = np.asarray(raw, dtype=np.float64)
    if U.ndim != 2 or U.shape[0] != n or U.shape[1] < rank:
        raise ValueError(f"{name}.U does not match its sample count and numerical rank.")
    U = U[:, :rank]
    if not np.isfinite(U).all():
        raise ValueError(f"{name}.U contains nonfinite values.")
    if not np.allclose(U.T @ U, np.eye(rank), rtol=1e-10, atol=1e-10):
        raise ValueError(f"{name}.U must contain orthonormal temporal singular vectors.")
    return n, rank, U


def _pca_bounds(max_pca_modes, rank_x, rank_y):
    if isinstance(max_pca_modes, (list, tuple, np.ndarray)):
        if len(max_pca_modes) != 2:
            raise ValueError("max_pca_modes must be one integer or a pair of bounds.")
        bounds = max_pca_modes
    else:
        bounds = (max_pca_modes, max_pca_modes)
    return tuple(
        rank if isinstance(bound, str) and bound == "all"
        else min(rank, _integer(bound, "max_pca_modes", 1))
        for bound, rank in zip(bounds, (rank_x, rank_y))
    )


def select_cca_mdl(bx, by, max_modes, max_pca_modes, *, detrend_dof=0):
    """Select CCA/PCA ranks using Song et al.'s max-min MDL detector.

    Parameters
    ----------
    bx, by : objects with ``U``, ``rank``, and ``n_samples`` attributes
        Existing :class:`FieldBasis`-compatible decompositions of the same
        training samples. Their U columns are ordered by decreasing variance.
    max_modes : nonnegative integer
        Upper bound for the detected signal rank. Zero is a valid null model.
    max_pca_modes : integer, "all", or pair of these
        Search every positive integer PCA rank up to these bounds and the
        respective numerical field rank.
    detrend_dof : nonnegative integer, default 0
        Number of temporal regressors removed IN ADDITION TO the intercept
        already removed by FieldBasis centering. Use 1 for a linear trend.
        This reduces the geometric rank limit, not the sample count in MDL.

    Returns
    -------
    dict
        JSON-serializable selection and per-PCA-pair diagnostics. ``score``
        is the selected pair's MDL value (smaller is better WITHIN that pair).
        ``n_modes == 0`` means use the training mean/trend baseline, never
        force a spurious first CCA mode. No admissible pair also returns this
        baseline with ``selected_pair_admissible=False`` and a reason.

    Notes
    -----
    Centering leaves at most n-1 temporal dimensions, with another dimension
    lost for each independently fitted detrending regressor. We require
    rx+ry < n-1-detrend_dof, leaving a residual dimension beyond the joint
    PCA space. This is deliberately stricter than the paper's uncentered
    rx+ry <= n bound and avoids saturation in very short training records.
    A true shared noiseless mode also reduces the observed union rank, so
    that rank is not used as an estimate of the ambient temporal dimension.
    Numerically unit canonical correlations are rejected explicitly rather
    than clipped into artificial evidence for correlation.
    """
    n, rank_x, Ux = _basis(bx, "bx")
    ny, rank_y, Uy = _basis(by, "by")
    if n != ny:
        raise ValueError("Both training bases must have the same sample count.")
    max_modes = _integer(max_modes, "max_modes", 0)
    detrend_dof = _integer(detrend_dof, "detrend_dof", 0)
    if detrend_dof >= n - 1:
        raise ValueError("detrend_dof leaves no centered temporal residual space.")
    limit_x, limit_y = _pca_bounds(max_pca_modes, rank_x, rank_y)
    ambient_rank = n - 1 - detrend_dof
    cross = Ux[:, :limit_x].T @ Uy[:, :limit_y]
    unit_tolerance = 64 * np.finfo(np.float64).eps * max(n, rank_x, rank_y)
    table = []
    admissible = []
    rejected_counts = {}
    for rx in range(1, limit_x + 1):
        for ry in range(1, limit_y + 1):
            row = {
                "n_pca_modes": [rx, ry], "admissible": False,
                "n_modes": None, "score": None, "canonical_correlations": [],
                "mdl_scores": [], "reason": None,
            }
            if rx + ry >= ambient_rank:
                row["reason"] = "joint_pca_rank_saturates_temporal_residual_space"
            else:
                rho = np.linalg.svd(cross[:rx, :ry], compute_uv=False)
                row["canonical_correlations"] = rho.tolist()
                if np.any(rho > 1.0 + unit_tolerance):
                    row["reason"] = "canonical_correlation_exceeds_one"
                elif np.any(rho >= 1.0 - unit_tolerance):
                    row["reason"] = "numerically_unit_canonical_correlation"
                else:
                    # Only remove tiny roundoff excursions after rejecting
                    # singular cases; never floor 1-rho**2 to invent a fit.
                    rho = np.clip(rho, 0.0, 1.0)
                    largest = min(max_modes, rx - 1, ry - 1)
                    s = np.arange(largest + 1)
                    likelihood = np.r_[0.0, n * np.cumsum(np.log1p(-rho[:largest] ** 2))]
                    scores = likelihood + np.log(n) * s * (rx + ry - s)
                    # np.argmin resolves exact MDL ties toward smaller rank.
                    best = int(np.argmin(scores))
                    row.update(
                        admissible=True, n_modes=best, score=float(scores[best]),
                        canonical_correlations=rho.tolist(),
                        mdl_scores=[{"n_modes": int(k), "score": float(score)}
                                    for k, score in zip(s, scores)],
                    )
                    admissible.append(row)
            if not row["admissible"]:
                reason = row["reason"]
                rejected_counts[reason] = rejected_counts.get(reason, 0) + 1
            table.append(row)

    if admissible:
        # Maximize detected rank, NOT likelihood across different PCA spaces.
        chosen = min(admissible, key=lambda row: (
            -row["n_modes"], sum(row["n_pca_modes"]),
            max(row["n_pca_modes"]), *row["n_pca_modes"],
        ))
        modes, pca, score = chosen["n_modes"], chosen["n_pca_modes"], chosen["score"]
        fallback_reason = None
    else:
        modes, pca, score = 0, [1, 1], 0.0
        fallback_reason = "no_nonsingular_admissible_pca_pair"
    return {
        "n_modes": modes, "n_pca_modes": list(pca), "metric": "mdl",
        "score": score, "rule": "mdl", "table": table, "n_samples": n,
        "selected_pair_admissible": bool(admissible),
        "fallback_reason": fallback_reason,
        "admissible_ranks": {
            "field_rank_x": rank_x, "field_rank_y": rank_y,
            "max_pca_x": limit_x, "max_pca_y": limit_y,
            "max_modes": max_modes, "detrend_dof": detrend_dof,
            "residual_time_rank": ambient_rank,
            "max_joint_pca_rank": ambient_rank - 1,
            "n_admissible_pairs": len(admissible),
            "n_rejected_pairs": len(table) - len(admissible),
            "rejected_counts": rejected_counts,
            "signal_rank_bound": "s < min(rx, ry)",
        },
    }
