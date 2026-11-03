"""Compact U-Net and shared spatial-network calibration for the seasonal CLIs.

Corrections of 15 September 2026, retained after the network refactor of
17 September: normalisation-free networks with a linear skip and static
channels (see ``was_mme_unet`` and its subclasses), seed ensemble, training on
a coarser grid with a fold-local anomaly downscaling to the observation grid,
checkpoint per outer fold, optional process-parallel outer folds and an
optional cheaper probability calibration from outer out-of-fold residuals.
All of it applies identically to U-Net, UNet++, RISE-UNet and ResCNN.

The operational profile keeps one declared architecture. ``nested_tuned``
selects a compact candidate only from outer-training years through blocked inner
temporal folds. Each outer fit and each inner calibration fit uses a fresh
network, training-only early stopping and a fresh refit for the selected epoch
count. PyTorch is imported only when requested.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from importlib import import_module
import json
import math
import os
from pathlib import Path
import pickle
import random

import numpy as np
import xarray as xr

from .runtime import _attach, _target, cv_probability_kwargs
from .probability import probability_calibration_mode


@dataclass(frozen=True)
class _NetworkDefinition:
    """Names that vary between spatial networks; calibration stays shared."""
    name: str
    class_name: str
    module_name: str
    label: str
    slug: str
    parameter_keys: tuple
    architecture_keys: tuple

    def estimator(self):
        return getattr(import_module(self.module_name), self.class_name)


_UNET = _NetworkDefinition("UNet", "WAS_mme_UNet", "wass2s.was_mme_unet",
                           "U-Net", "unet", ("filters", "dropout", "lr", "weight_decay"),
                           ("filters",))


def _settings(args, variable):
    variable = str(variable).upper()
    if variable not in {"PRCP", "TEMP"}:
        raise ValueError("U-Net variable must be PRCP or TEMP")
    base = int(getattr(args, "unet_base_channels", 8))
    depth = int(getattr(args, "unet_depth", 3))
    epochs = int(getattr(args, "unet_epochs", 40))
    batch = int(getattr(args, "unet_batch_size", 2))
    patience = int(getattr(args, "unet_patience", 6))
    threads = int(getattr(args, "unet_threads", 1))
    dropout = float(getattr(args, "unet_dropout", .2))
    lr = float(getattr(args, "unet_learning_rate", .001))
    decay = float(getattr(args, "unet_weight_decay", .0001))
    tuning = str(getattr(args, "unet_tuning", "fixed"))
    tuning_candidates = int(getattr(args, "unet_tuning_candidates", 8))
    tuning_folds = int(getattr(args, "unet_tuning_folds", 3))
    tuning_epochs = int(getattr(args, "unet_tuning_epochs", 20))
    tuning_patience = int(getattr(args, "unet_tuning_patience", 4))
    if min(base, depth, epochs, batch, patience, threads) < 1 or depth > 5:
        raise ValueError("U-Net channels, epochs, batch, patience and threads must be positive; depth must be 1..5")
    if not 0 <= dropout < 1 or not np.isfinite(lr) or lr <= 0 or not np.isfinite(decay) or decay < 0:
        raise ValueError("U-Net requires 0<=dropout<1, positive learning rate and nonnegative weight decay")
    if tuning not in {"fixed", "nested_tuned"}:
        raise ValueError("--unet-tuning must be fixed or nested_tuned")
    if not 1 <= tuning_candidates <= 12 or not 2 <= tuning_folds <= 4 or min(tuning_epochs, tuning_patience) < 1:
        raise ValueError("U-Net tuning needs 1..12 candidates, 2..4 folds, and positive epochs/patience")
    device = str(getattr(args, "unet_device", "cpu"))
    if device not in {"cpu", "auto", "cuda"}:
        raise ValueError("U-Net device must be cpu, auto or cuda")
    method = str(getattr(args, "dist_method", "nonparam"))
    if method not in {"nonparam", "bestfit"}:
        raise ValueError("U-Net probability method must be nonparam or bestfit")
    norm = str(getattr(args, "unet_norm", "none"))
    if norm not in {"none", "group"}:
        raise ValueError("--unet-norm must be none or group")
    seeds = int(getattr(args, "unet_seeds", 3))
    if seeds < 1:
        raise ValueError("--unet-seeds must be positive")
    fast_gpu = bool(getattr(args, "unet_fast_gpu", False))
    return {
        "filters": tuple(base * 2**i for i in range(depth)), "dropout": dropout,
        "lr": lr, "weight_decay": decay, "batch_size": batch, "epochs": epochs,
        "early_stopping_patience": patience, "validation_fraction": .2,
        "norm_groups": min(base, 8), "device": device, "dist_method": method,
        "random_state": 42, "deterministic": not fast_gpu, "num_workers": 0,
        "mixed_precision": fast_gpu, "min_valid_years": 3,
        "norm": norm, "linear_skip": bool(getattr(args, "unet_linear_skip", True)),
        "add_static_channels": bool(getattr(args, "unet_static_channels", True)),
        "n_seeds": seeds, "refit_after_stopping": bool(getattr(args, "unet_refit", True)),
        "tune_hyperparameters": tuning == "nested_tuned", "temporal_buffer": int(args.cv_gap_years),
        # Deliberately compact for ~24 independent seasonal years. The final
        # architecture is refitted with --unet-epochs after selection.
        "tuning_filters": ((4, 8), (8, 16), (8, 16, 32)),
        "tuning_dropouts": (.10, .20), "tuning_lrs": (3e-4, 1e-3),
        "tuning_weight_decays": (1e-5, 1e-4), "tuning_folds": tuning_folds,
        "tuning_epochs": tuning_epochs, "tuning_patience": tuning_patience,
        "max_tuning_candidates": tuning_candidates,
        "verbose": 0,
    }, threads


@contextmanager
def _torch_runtime(device, threads, label="U-Net"):
    """Bound CPU use and restore caller RNG/determinism/thread settings."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(f"{label} requires PyTorch in the active Python environment") from exc
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"{label} CUDA was requested but PyTorch cannot access an allocated GPU")
    resolved = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    allocated = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocated:
        available = min(available, int(allocated))
    if threads > available:
        raise ValueError(f"{label} requests {threads} CPU threads but only {available} are allocated")
    previous = (torch.get_num_threads(), random.getstate(), np.random.get_state(),
                torch.get_rng_state(), torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled(),
                torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        torch.set_num_threads(threads)
        yield resolved
    finally:
        torch.set_num_threads(previous[0])
        random.setstate(previous[1])
        np.random.set_state(previous[2])
        torch.set_rng_state(previous[3])
        torch.use_deterministic_algorithms(previous[4], warn_only=previous[5])
        torch.backends.cudnn.deterministic = previous[6]
        torch.backends.cudnn.benchmark = previous[7]
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _runtime_options(args):
    """Execution choices that are not constructor parameters of WAS_mme_UNet."""
    resolution = float(getattr(args, "unet_train_resolution", 0.25) or 0.0)
    workers = int(getattr(args, "unet_fold_workers", 1))
    calibration = probability_calibration_mode(args)
    if resolution < 0 or not np.isfinite(resolution):
        raise ValueError("--unet-train-resolution must be >= 0 (0 keeps the observation grid)")
    if workers < 1:
        raise ValueError("--unet-fold-workers must be positive")
    return {"train_resolution": resolution, "fold_workers": workers,
            "probability_calibration": calibration,
            "probability_residuals": "outer_oof" if calibration == "outer" else "nested",
            "probability_inner_retune": bool(getattr(args, "probability_inner_retune", True)),
            "probability_calibration_schema": 3,
            "checkpoint": bool(getattr(args, "unet_checkpoint", True))}


def grid_factors(reference, resolution):
    """Integer block sizes (Y, X) giving roughly ``resolution`` degrees; 1 = native."""
    if not resolution or resolution <= 0:
        return 1, 1
    factors = []
    for dim in ("Y", "X"):
        values = np.asarray(reference[dim].values, dtype=float)
        if values.size < 2:
            return 1, 1
        steps = np.diff(values)
        if not np.allclose(steps, steps[0], rtol=1e-3, atol=1e-9):
            return 1, 1  # irregular grid: never coarsen implicitly
        factors.append(max(1, int(round(float(resolution) / abs(float(steps[0]))))))
    return tuple(factors)


def block_mean(field, fy, fx, min_fraction=0.5):
    """NaN-aware block average on a regular grid; not fitted, hence fold-safe."""
    if fy == 1 and fx == 1:
        return field
    dims = [d for d in field.dims if d not in ("Y", "X")] + ["Y", "X"]
    field = field.transpose(*dims)
    values = np.asarray(field.values, dtype=float)
    ny, nx = values.shape[-2:]
    py, px = (-ny) % fy, (-nx) % fx
    pad = [(0, 0)] * (values.ndim - 2) + [(0, py), (0, px)]
    padded = np.pad(values, pad, constant_values=np.nan)
    inside = np.pad(np.ones((ny, nx)), [(0, py), (0, px)])
    shape = padded.shape[:-2] + (padded.shape[-2] // fy, fy, padded.shape[-1] // fx, fx)
    blocks = padded.reshape(shape)
    valid = np.isfinite(blocks)
    count = valid.sum(axis=(-3, -1))
    total = np.where(valid, blocks, 0.0).sum(axis=(-3, -1))
    cells = inside.reshape(shape[-4:]).sum(axis=(-3, -1))
    needed = np.maximum(1, np.ceil(min_fraction * cells))
    mean = np.where(count >= needed, total / np.maximum(count, 1), np.nan)
    coords = {d: field[d] for d in dims[:-2] if d in field.coords}
    for dim, factor, size in (("Y", fy, ny), ("X", fx, nx)):
        axis = np.asarray(field[dim].values, dtype=float)
        step = axis[1] - axis[0]
        coords[dim] = axis[0] + step * (np.arange(math.ceil(size / factor)) * factor + (factor - 1) / 2.0)
    out = xr.DataArray(mean.astype(values.dtype if values.dtype.kind == "f" else float),
                       dims=dims, coords=coords, name=field.name, attrs=dict(field.attrs))
    return out


def refine_anomaly(coarse, fine):
    """Bilinear coarse-to-fine interpolation after nearest-valid filling."""
    from scipy.ndimage import distance_transform_edt
    coarse = coarse.transpose("T", "Y", "X")
    values = np.array(coarse.values, dtype=float)
    for t in range(values.shape[0]):
        missing = ~np.isfinite(values[t])
        if missing.all() or not missing.any():
            continue
        index = distance_transform_edt(missing, return_distances=False, return_indices=True)
        values[t] = values[t][tuple(index)]
    filled = coarse.copy(data=values)
    # Extend the coarse centres to the fine-grid extremes with edge values so
    # border half-cells keep linear interpolation along the other direction.
    for dim in ("Y", "X"):
        centres = np.asarray(filled[dim].values, dtype=float)
        target = np.asarray(fine[dim].values, dtype=float)
        if centres.size < 2:
            continue
        ascending = centres[1] > centres[0]
        first, last = (target.min(), target.max()) if ascending else (target.max(), target.min())
        pieces = []
        if (first < centres[0]) if ascending else (first > centres[0]):
            pieces.append(filled.isel({dim: [0]}).assign_coords({dim: [first]}))
        pieces.append(filled)
        if (last > centres[-1]) if ascending else (last < centres[-1]):
            pieces.append(filled.isel({dim: [-1]}).assign_coords({dim: [last]}))
        if len(pieces) > 1:
            filled = xr.concat(pieces, dim=dim)
    if min(filled.sizes["Y"], filled.sizes["X"]) < 2:
        out = filled.sel(Y=fine["Y"].values, X=fine["X"].values, method="nearest")
        return out.assign_coords(Y=fine["Y"].values, X=fine["X"].values)
    linear = filled.interp(Y=fine["Y"].values, X=fine["X"].values, method="linear")
    # Border half-cells lie outside the coarse centres: nearest-centre values.
    nearest = filled.sel(Y=fine["Y"].values, X=fine["X"].values, method="nearest")
    nearest = nearest.assign_coords(Y=fine["Y"].values, X=fine["X"].values)
    return linear.fillna(nearest)


class CoarseGridAdapter:
    """Train/predict on block means; map the anomaly back to the fine grid.

    The coarse network predicts physical values; they are converted to
    anomalies with the COARSE training climatology, interpolated, and rescaled
    with the FINE training climatology of the same fold.
    """

    def __init__(self, spec, fy, fx, min_samples):
        self.spec, self.fy, self.fx, self.min_samples = spec, int(fy), int(fx), int(min_samples)
        self.inner = spec.create()

    @property
    def model(self):
        return self.inner.model

    @staticmethod
    def _stats(field):
        mean = field.mean("T", skipna=True)
        sd = field.std("T", skipna=True)
        return mean, xr.where(sd > 1e-12, sd, 1.0)

    def fit(self, X, y):
        self.y = y
        self.support = y.notnull().sum("T") >= self.min_samples
        self.fine_mean, self.fine_sd = self._stats(y)
        yc = block_mean(y, self.fy, self.fx)
        self.coarse_mean, self.coarse_sd = self._stats(yc)
        self.inner.fit({"predictor": block_mean(X["predictor"], self.fy, self.fx)}, yc)
        return self

    def predict(self, X):
        from wass2s.was_scientific_utils import apply_physical_support
        predictor = X["predictor"]
        coarse = self.inner.predict({"predictor": block_mean(predictor, self.fy, self.fx)})
        anomaly = (coarse - self.coarse_mean) / self.coarse_sd
        fine = refine_anomaly(anomaly, self.y)
        pred = (self.fine_mean + self.fine_sd * fine).where(self.support)
        pred = pred.transpose("T", "Y", "X").assign_coords(T=predictor["T"].values)
        return apply_physical_support(pred, self.y)


@dataclass
class ResolutionSpec:
    """WASModelSpec wrapper; ``init`` stays inspectable for audits/tests."""
    inner: object
    fy: int
    fx: int
    min_samples: int

    @property
    def init(self):
        return self.inner.init

    def create(self):
        return CoarseGridAdapter(self.inner, self.fy, self.fx, self.min_samples)


def build_unet(args, variable):
    """Build the compact optional estimator without changing global RNG state."""
    settings, threads = _settings(args, variable)
    with _torch_runtime(settings["device"], threads) as device:
        from wass2s.was_mme_unet import WAS_mme_UNet
        return WAS_mme_UNet(**dict(settings, device=device))


def _probability_model(method):
    # Reuse the same predictive-family kernels as the other native MME methods.
    # U-Net's legacy standalone probability functions have a different API.
    from wass2s.was_mme import WAS_mme_Weighted
    return WAS_mme_Weighted(equal_weighted=True, dist_method=method)


def _prepared(args, obs, hind, future, variable):
    settings, threads = _settings(args, variable)
    return _prepare_network(obs, hind, future, variable, settings, threads, _UNET)


def _prepare_network(obs, hind, future, variable, settings, threads, network, factors=(1, 1)):
    obs = _target(obs).copy(deep=False)
    obs.attrs = dict(obs.attrs, support_policy="nonnegative" if variable == "PRCP" else "identity")
    for label, field in (("hindcast", hind), ("forecast", future)):
        if not isinstance(field, xr.DataArray) or set(field.dims) != {"M", "T", "Y", "X"}:
            raise ValueError(f"{network.label} {label} predictors require dimensions M,T,Y,X")
        for dim in ("T", "M", "Y", "X"):
            if field.get_index(dim).has_duplicates:
                raise ValueError(f"{network.label} {label} has duplicate {dim} coordinates")
        if not field.get_index("T").is_monotonic_increasing:
            raise ValueError(f"{network.label} {label} years must be sorted")
        for dim in ("Y", "X"):
            if not field.get_index(dim).equals(obs.get_index(dim)):
                raise ValueError(f"{network.label} {label} and observation {dim} grids differ")
    if not hind.get_index("T").equals(obs.get_index("T")):
        raise ValueError(f"{network.label} predictor and observation times must match exactly")
    if not hind.get_index("M").equals(future.get_index("M")):
        raise ValueError(f"{network.label} forecast predictor channels must match the historical channels in order")
    if network.slug in {"unet", "unetpp", "rise-unet"}:
        sides = (math.ceil(obs.sizes["Y"] / factors[0]), math.ceil(obs.sizes["X"] / factors[1]))
        if min(sides) < 2**len(settings["filters"]):
            raise ValueError(f"{network.label} depth {len(settings['filters'])} needs grid sides >= {2**len(settings['filters'])} "
                             f"(training grid {sides}); reduce --unet-depth or --unet-train-resolution")
    if min(obs.sizes["Y"], obs.sizes["X"]) < 1:
        raise ValueError(f"{network.label} requires a nonempty spatial grid")
    if network.slug == "rescnn" and obs.sizes["Y"] * obs.sizes["X"] < 2:
        raise ValueError("ResCNN requires at least two spatial grid cells for GroupNorm")
    if future.sizes["T"] < 1 or int(future["T"].dt.year.min()) <= int(obs["T"].dt.year.max()):
        raise ValueError(f"{network.label} operational target year must follow all training years")
    return obs.load(), hind.transpose("T", "M", "Y", "X").load(), future.transpose("T", "M", "Y", "X").load(), settings, threads


def _calibration_plan(times, config):
    from wass2s.was_crossfit import SplitPlan
    return SplitPlan.build(times, mode="blocked", gap_years=config.gap_years,
                           n_blocks=config.calibration_blocks, min_train=config.min_train)


def _fit(spec, x, y):
    fitted = spec.create().fit({"predictor": x}, y)
    return fitted


def _select_unet_spec(settings, device, x, y, *, min_samples, seed):
    return _select_network_spec(settings, device, x, y, min_samples=min_samples,
                                seed=seed, network=_UNET)


def _select_network_spec(settings, device, x, y, *, min_samples, seed, network,
                         factors=(1, 1)):
    """Select one candidate from the outer-training fold, then freeze it.

    ``WASModelSpec`` intentionally disables a native estimator's own tuning;
    selection therefore happens here before every outer fit, rather than ever
    accidentally seeing a held-out outer year.
    """
    from wass2s.was_crossfit_adapters import WASModelSpec
    selected = dict(settings)
    fy, fx = factors
    if settings["tune_hyperparameters"]:
        tuner = network.estimator()(**dict(settings, device=device, random_state=seed))
        best, _ = tuner.compute_hyperparameters(block_mean(x, fy, fx), block_mean(y, fy, fx))
        selected.update(best)
        tuning = {"enabled": True, "best_params": best,
                  "candidates": tuner.tuning_results_,
                  "performed": bool(tuner.tuning_results_),
                  "fallback": (None if tuner.tuning_results_ else
                               "too few training years; prespecified parameters retained")}
    else:
        tuning = {"enabled": False, "best_params": {key: selected[key]
                  for key in network.parameter_keys}, "candidates": []}
    selected["tune_hyperparameters"] = False
    spec = WASModelSpec(network.name, network.class_name, init=dict(selected, device=device),
                        min_samples=min_samples, seed=seed)
    return _wrap_resolution(spec, factors, min_samples), tuning


def _wrap_resolution(spec, factors, min_samples):
    fy, fx = factors
    return spec if (fy, fx) == (1, 1) else ResolutionSpec(spec, fy, fx, min_samples)


def _residuals(spec, x, y, plan, *, settings=None, device=None, network=_UNET,
               min_samples=5, seed=42, factors=(1, 1)):
    parts, audit = [], []
    for index, fold in enumerate(plan.folds):
        train, test = list(fold.train), list(fold.test)
        xa, ya = x.isel(T=train), y.isel(T=train)
        calibration_spec = spec
        selection = None
        if settings is not None and settings["tune_hyperparameters"]:
            # The calibration test year must not influence candidate selection.
            calibration_spec, selection = _select_network_spec(
                settings, device, xa, ya, min_samples=min_samples,
                seed=seed + index, network=network, factors=factors)
        fitted = _fit(calibration_spec, xa, ya)
        parts.append(fitted.predict({"predictor": x.isel(T=test)}))
        audit.append({"train_years": y["T"].dt.year.values[train].tolist(),
                      "test_years": y["T"].dt.year.values[test].tolist(),
                      "selected_epoch": int(fitted.model.stopping_diagnostics_["selected_epoch"]),
                      "selected_epochs": list(fitted.model.stopping_diagnostics_.get("selected_epochs", [])),
                      "hyperparameter_selection": selection})
        del fitted
    oof = xr.concat(parts, "T").sortby("T").reindex(T=y["T"])
    return y - oof, audit


@dataclass
class UNetOperationalArtifact:
    """A fitted, reusable network with training-only probability calibration."""
    fitted: object
    residuals: xr.DataArray
    observations: xr.DataArray
    config: object
    probability_options: dict
    dist_method: str
    threads: int
    audit: dict

    def predict(self, predictors):
        """Predict later target years; network weights/transforms are reused."""
        from wass2s.was_cross_validate import _training_probabilities
        if int(predictors["T"].dt.year.min()) <= int(self.observations["T"].dt.year.max()):
            raise ValueError("Spatial network operational target must follow the training record")
        with _torch_runtime(self.fitted.model.device, self.threads,
                            self.audit.get("network", "U-Net")):
            det = self.fitted.predict({"predictor": predictors})
        prob, q, ref, _ = _training_probabilities(
            _probability_model(self.dist_method), det, self.residuals,
            self.observations, self.config, self.probability_options)
        prob = _attach(prob, q, ref, "operational")
        # Preserve the declared procedure when reusing a serialized artifact.
        calibration = self.audit.get("probability_calibration")
        if calibration is not None:
            for result in (det, prob):
                result.attrs.update(probability_calibration=calibration,
                                    calibration_residual_source=(
                                        "outer_oof" if calibration == "outer" else "nested"))
        return det, prob

    def save(self, path):
        """Save a CPU-portable copy atomically; retain the live training device."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        portable = deepcopy(self)
        portable.fitted.model.model.to("cpu")
        portable.fitted.model.device = "cpu"
        portable.fitted.model.pin_memory = False
        portable.fitted.model.mixed_precision = False
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            with temporary.open("wb") as output:
                pickle.dump(portable, output, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path


def load_unet_artifact(path):
    """Load an artifact produced by this workflow (trusted pickle files only)."""
    with Path(path).open("rb") as source:
        artifact = pickle.load(source)
    if not isinstance(artifact, UNetOperationalArtifact):
        raise TypeError("File does not contain a CLI U-Net operational artifact")
    return artifact


def run_unet(args, obs, rawhind, rawfuture, variable, workdir=None, *, split_plan=None):
    """Return U-Net OOS hindcast and operational forecast means/probabilities."""
    return _run_spatial_network(args, obs, rawhind, rawfuture, variable, workdir,
                                split_plan=split_plan, network=_UNET,
                                settings_function=_settings,
                                artifact_type=UNetOperationalArtifact)


def _digest(*arrays):
    digest = hashlib.blake2b(digest_size=20)
    for array in arrays:
        digest.update(np.ascontiguousarray(np.asarray(array.values, dtype=np.float32)).tobytes())
        for dim in array.dims:
            digest.update(str(dim).encode())
            digest.update(np.asarray(array[dim].values).astype(str).tobytes())
    return digest.hexdigest()


def _checkpoint_load(path, fingerprint):
    if path is None or not Path(path).is_file():
        return None
    try:
        with Path(path).open("rb") as source:
            record = pickle.load(source)
    except Exception:
        return None
    return record.get("result") if record.get("fingerprint") == fingerprint else None


def _checkpoint_save(path, fingerprint, result):
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as output:
        pickle.dump({"fingerprint": fingerprint, "result": result}, output,
                    protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _outer_fold_task(index, train, test, inner, x, obs, settings, threads, device_request,
                     cfg, options, dist_method, factors, residual_mode, checkpoint,
                     fingerprint, network):
    """One outer fold; independent of every other fold, hence parallelisable."""
    cached = _checkpoint_load(checkpoint, fingerprint)
    if cached is not None:
        print(f"[sv-ml-cmme/{network.slug}] outer fold {index + 1} restored from checkpoint", flush=True)
        return cached
    from wass2s.was_crossfit_adapters import WASModelSpec
    from wass2s.was_cross_validate import _training_probabilities
    xa, ya = x.isel(T=train), obs.isel(T=train)
    with _torch_runtime(device_request, threads, network.label) as device:
        if settings["tune_hyperparameters"]:
            seed = 42 + int(obs["T"].dt.year.values[test[0]])
            spec, selection = _select_network_spec(settings, device, xa, ya,
                min_samples=cfg.min_train, seed=seed, network=network, factors=factors)
        else:
            spec = _wrap_resolution(WASModelSpec(network.name, network.class_name,
                                                 init=dict(settings, device=device),
                                                 min_samples=cfg.min_train, seed=42),
                                    factors, cfg.min_train)
            selection = {"enabled": False, "best_params": {key: settings[key]
                         for key in network.parameter_keys}, "candidates": []}
        fitted = _fit(spec, xa, ya)
        hd = fitted.predict({"predictor": x.isel(T=test)})
        diagnostics = fitted.model.stopping_diagnostics_
        record = {"train_years": ya["T"].dt.year.values.tolist(),
                  "test_years": hd["T"].dt.year.values.tolist(),
                  "selected_epoch": int(diagnostics["selected_epoch"]),
                  "selected_epochs": list(diagnostics.get("selected_epochs", [])),
                  "hyperparameter_selection": selection}
        del fitted
        result = {"hd": hd, "audit": record}
        if residual_mode == "nested":
            residuals, inner_audit = _residuals(spec, xa, ya, inner, settings=(settings if options.get("probability_inner_retune", True)
                          else dict(settings, tune_hyperparameters=False)),
                device=device, network=network, min_samples=cfg.min_train,
                seed=1042 + index * 100, factors=factors)
            hp, q, ref, _ = _training_probabilities(_probability_model(dist_method), hd, residuals, ya, cfg, options)
            record["calibration"] = inner_audit
            result.update(hp=hp, q=q, ref=ref)
        if str(device).startswith("cuda"):
            import torch
            record["max_gpu_memory_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    _checkpoint_save(checkpoint, fingerprint, result)
    print(f"[sv-ml-cmme/{network.slug}] completed outer fold {index + 1}", flush=True)
    return result


def _run_spatial_network(args, obs, rawhind, rawfuture, variable, workdir=None, *,
                         split_plan=None, network, settings_function, artifact_type):
    """Return hindcast mean/probabilities and future mean/probabilities.

    A custom SplitPlan is primarily useful for focused fold verification. The
    CLI uses the default annual LOO plan with its configured calendar-year gap.
    """
    from wass2s.was_crossfit import CrossFitConfig, SplitPlan
    from wass2s.was_crossfit_adapters import WASModelSpec
    from wass2s.was_cross_validate import _training_probabilities, _climatology_options

    variable = str(variable).upper()
    settings, threads = settings_function(args, variable)
    runtime = _runtime_options(args)
    factors = grid_factors(_target(obs), runtime["train_resolution"])
    obs, x, future, settings, threads = _prepare_network(
        obs, rawhind, rawfuture, variable, settings, threads, network, factors)
    cfg = CrossFitConfig(probability_method="residual", gap_years=int(args.cv_gap_years),
                         calibration_blocks=int(getattr(args, "probability_inner_folds", None) or args.svml_probability_folds), min_train=5,
                         tercile_reference=getattr(args, "tercile_reference", "climatology"),
                         climatology_start=int(args.clim_years[0]), climatology_end=int(args.clim_years[1]))
    outer = split_plan or SplitPlan.build(obs["T"], mode="loo", gap_years=cfg.gap_years, min_train=cfg.min_train)
    outer.check(obs["T"])
    if outer.gap_years != cfg.gap_years:
        raise ValueError(f"{network.label} custom split plan must use the configured calendar-year gap")
    residual_mode = runtime["probability_calibration"]
    if residual_mode == "outer":
        coverage = np.zeros(obs.sizes["T"], dtype=int)
        for fold in outer.folds:
            coverage[list(fold.test)] += 1
        if not np.all(coverage == 1):
            raise ValueError("outer probability calibration needs the complete outer plan")
    # Resolve every date-only split before any expensive model training.
    # Outer reuses the already available raw OOF residuals: no calibration
    # plan or extra calibration fit is needed, including for future forecasts.
    inner_plans = ([_calibration_plan(obs["T"].isel(T=list(f.train)), cfg)
                    for f in outer.folds] if residual_mode == "nested"
                   else [None] * len(outer.folds))
    operational_plan = _calibration_plan(obs["T"], cfg) if residual_mode == "nested" else None
    options = _climatology_options(cfg, obs, cfg.climatology_start, cfg.climatology_end,
                                  cv_probability_kwargs(args, variable))
    tuning_mode = "nested blocked temporal hyperparameter tuning" if settings["tune_hyperparameters"] else "fixed architecture"
    calibration_mode = ("nested temporal residual calibration" if residual_mode == "nested"
                        else "outer out-of-fold residuals excluding each target and its gap")
    strictly_nested = (residual_mode == "nested" and
                       (not settings["tune_hyperparameters"] or runtime["probability_inner_retune"]))
    if residual_mode == "nested" and not strictly_nested:
        calibration_mode += "; conditional on outer-selected hyperparameters"
    audit = {"method": f"{network.label}; {tuning_mode}; {calibration_mode}",
             "network": network.label,
             "probability_calibration": residual_mode,
             "probability_calibration_schema": 3,
             "calibration_strictly_nested": strictly_nested,
             "probability_inner_retune": runtime["probability_inner_retune"],
             "upstream_selection": deepcopy(getattr(args, "cnn_upstream_selection", {
                 "method": "caller-supplied channels",
                 "validation_scope": "upstream selection not assessed here",
             })),
             "outer_plan": outer.to_dict(), "settings": settings,
             "runtime": dict(runtime, training_grid_factors=list(factors)), "folds": []}

    checkpoint_dir = None
    base_print = None
    if workdir is not None and runtime["checkpoint"]:
        base = json.dumps({"settings": settings, "runtime": runtime, "factors": factors,
                           "cfg": repr(cfg), "options": options, "dist": args.dist_method,
                           "variable": variable, "network": network.slug,
                           "plan": outer.fingerprint, "data": _digest(obs, x)},
                          sort_keys=True, default=str)
        base_print = hashlib.blake2b(base.encode(), digest_size=20).hexdigest()
        checkpoint_dir = Path(workdir) / "models" / f"{network.slug}_folds" / base_print[:16]
    tasks = []
    for index, (fold, inner) in enumerate(zip(outer.folds, inner_plans)):
        train, test = list(fold.train), list(fold.test)
        path = None if checkpoint_dir is None else checkpoint_dir / f"fold_{index:03d}.pkl"
        tasks.append((index, train, test, inner, x, obs, settings, threads, settings["device"],
                      cfg, options, args.dist_method, factors, residual_mode, path,
                      f"{base_print}:{index}:{train}:{test}", network))
    if runtime["fold_workers"] > 1 and len(tasks) > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=min(runtime["fold_workers"], len(tasks)), backend="loky")(
            delayed(_outer_fold_task)(*task) for task in tasks)
    else:
        results = [_outer_fold_task(*task) for task in tasks]

    def combine(parts):
        return xr.concat(parts, "T").sortby("T").reindex(T=obs["T"])

    hds = [result["hd"] for result in results]
    audit["folds"] = [result["audit"] for result in results]
    calibrator = _probability_model(args.dist_method)
    if residual_mode == "nested":
        hps = [result["hp"] for result in results]
        qs = [result["q"].expand_dims(T=result["hd"]["T"]) for result in results]
        refs = [result["ref"].expand_dims(T=result["hd"]["T"]) for result in results]
    else:
        oof = combine(hds)
        hps, qs, refs = [], [], []
        for fold, hd in zip(outer.folds, hds):
            # SplitPlan validates the temporal gap, but custom plans may also
            # impose extra exclusions. Calibration must honor those same years.
            keep = list(fold.train)
            residuals = (obs - oof).isel(T=keep)
            ya = obs.isel(T=keep)
            hp, q, ref, _ = _training_probabilities(calibrator, hd, residuals, ya, cfg, options)
            hps.append(hp); qs.append(q.expand_dims(T=hd["T"])); refs.append(ref.expand_dims(T=hd["T"]))

    with _torch_runtime(settings["device"], threads, network.label) as device:
        if settings["tune_hyperparameters"]:
            operational_spec, operational_selection = _select_network_spec(
                settings, device, x, obs, min_samples=cfg.min_train, seed=4242,
                network=network, factors=factors)
        else:
            operational_spec = _wrap_resolution(
                WASModelSpec(network.name, network.class_name, init=dict(settings, device=device),
                             min_samples=cfg.min_train, seed=42), factors, cfg.min_train)
            operational_selection = {"enabled": False, "best_params": {key: settings[key]
                                     for key in network.parameter_keys}, "candidates": []}
        if residual_mode == "nested":
            residuals, operational_audit = _residuals(operational_spec, x, obs, operational_plan,
                settings=(settings if runtime["probability_inner_retune"]
                          else dict(settings, tune_hyperparameters=False)),
                device=device, network=network,
                min_samples=cfg.min_train, seed=5242, factors=factors)
        else:
            residuals, operational_audit = obs - combine(hds), "outer out-of-fold residuals"
        fitted = _fit(operational_spec, x, obs)
        audit["operational"] = {"training_years": obs["T"].dt.year.values.tolist(),
                                "selected_epoch": int(fitted.model.stopping_diagnostics_["selected_epoch"]),
                                "hyperparameter_selection": operational_selection,
                                "calibration": operational_audit}
        artifact = artifact_type(fitted, residuals, obs, cfg, options,
                                 args.dist_method, threads, audit)
        fd, fp = artifact.predict(future)
        if workdir is not None:
            model_dir = Path(workdir) / "models"
            artifact.save(model_dir / f"{network.slug}_operational.pkl")
            (model_dir / f"{network.slug}_audit.json").write_text(
                json.dumps(audit, indent=2, default=str), encoding="utf-8")
    hd = combine(hds).transpose("T", "Y", "X")
    hp = _attach(combine(hps).transpose("probability", "T", "Y", "X"),
                 combine(qs), combine(refs), "hindcast")
    for result in (hd, hp, fd, fp):
        inner_spec = getattr(operational_spec, "inner", operational_spec)
        architecture = (inner_spec.init["filters"] if network.slug == "unet" else
                        {key: inner_spec.init[key] for key in network.architecture_keys})
        result.attrs.update(model=f"SV-ML-CMME_{network.slug.upper()}", validation=audit["method"],
                            architecture=json.dumps(architecture),
                            split_fingerprint=outer.fingerprint,
                            probability_calibration=residual_mode,
                            calibration_residual_source=(
                                "outer_oof" if residual_mode == "outer" else "nested"))
    hp.attrs["calibration_strictly_nested"] = int(strictly_nested)
    hp.attrs["probability_inner_retune"] = int(runtime["probability_inner_retune"])
    hd.attrs.update(obs.attrs)
    fd.attrs.update(obs.attrs)
    return hd, hp, fd, fp
