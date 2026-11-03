"""Optional residual CNN using the same temporal verification as CLI U-Net.

Residual blocks retain the native grid. The seasonal target, model channels,
outer folds and the common nested/outer probability calibration choice are
shared with U-Net. Outer reuses raw OOF errors without calibration refits.
"""
from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np

from .unet import (
    _NetworkDefinition, _run_spatial_network, _select_network_spec,
    _torch_runtime, UNetOperationalArtifact,
)


_RESCNN = _NetworkDefinition(
    "ResCNN", "WAS_mme_ResCNN", "wass2s.was_mme_rescnn", "ResCNN", "rescnn",
    ("channels", "num_blocks", "dilations", "dropout", "lr", "weight_decay"),
    ("channels", "num_blocks", "dilations"),
)


def _settings(args, variable):
    if str(variable).upper() not in {"PRCP", "TEMP"}:
        raise ValueError("ResCNN variable must be PRCP or TEMP")
    channels = int(getattr(args, "rescnn_channels", 16))
    blocks = int(getattr(args, "rescnn_blocks", 3))
    requested_dilations = getattr(args, "rescnn_dilations", (1, 2, 4))
    dilations = tuple(int(d) for d in requested_dilations) if requested_dilations is not None else (1,) * blocks
    epochs = int(getattr(args, "rescnn_epochs", 40))
    batch = int(getattr(args, "rescnn_batch_size", 1))
    patience = int(getattr(args, "rescnn_patience", 6))
    threads = int(getattr(args, "rescnn_threads", 1))
    dropout = float(getattr(args, "rescnn_dropout", .2))
    lr = float(getattr(args, "rescnn_learning_rate", .001))
    decay = float(getattr(args, "rescnn_weight_decay", .0001))
    tuning = str(getattr(args, "rescnn_tuning", "fixed"))
    candidates = int(getattr(args, "rescnn_tuning_candidates", 8))
    folds = int(getattr(args, "rescnn_tuning_folds", 3))
    tuning_epochs = int(getattr(args, "rescnn_tuning_epochs", 20))
    tuning_patience = int(getattr(args, "rescnn_tuning_patience", 4))
    if min(channels, blocks, epochs, batch, patience, threads) < 1:
        raise ValueError("ResCNN channels, blocks, epochs, batch, patience and threads must be positive")
    if len(dilations) != blocks or any(d < 1 for d in dilations):
        raise ValueError("ResCNN requires one positive --rescnn-dilations value per block")
    if not 0 <= dropout < 1 or not np.isfinite(lr) or lr <= 0 or not np.isfinite(decay) or decay < 0:
        raise ValueError("ResCNN requires 0<=dropout<1, positive learning rate and nonnegative weight decay")
    if tuning not in {"fixed", "nested_tuned"}:
        raise ValueError("--rescnn-tuning must be fixed or nested_tuned")
    if not 1 <= candidates <= 12 or not 2 <= folds <= 4 or min(tuning_epochs, tuning_patience) < 1:
        raise ValueError("ResCNN tuning needs 1..12 candidates, 2..4 folds, and positive epochs/patience")
    device = str(getattr(args, "rescnn_device", "cpu"))
    if device not in {"cpu", "auto", "cuda"}:
        raise ValueError("ResCNN device must be cpu, auto or cuda")
    method = str(getattr(args, "dist_method", "nonparam"))
    if method not in {"nonparam", "bestfit"}:
        raise ValueError("ResCNN probability method must be nonparam or bestfit")
    return {
        "channels": channels, "num_blocks": blocks, "dilations": dilations,
        "dropout": dropout, "lr": lr, "weight_decay": decay, "batch_size": batch,
        "epochs": epochs, "early_stopping_patience": patience, "validation_fraction": .2,
        "norm_groups": min(channels, 8), "device": device, "dist_method": method,
        "random_state": 42, "deterministic": True, "num_workers": 0,
        "mixed_precision": False, "min_valid_years": 3,
        "tune_hyperparameters": tuning == "nested_tuned", "temporal_buffer": int(args.cv_gap_years),
        "tuning_channels": (8, 16), "tuning_num_blocks": (blocks,),
        "tuning_dropouts": (.10, .20), "tuning_lrs": (3e-4, 1e-3),
        "tuning_weight_decays": (1e-5, 1e-4), "tuning_folds": folds,
        "tuning_epochs": tuning_epochs, "tuning_patience": tuning_patience,
        "max_tuning_candidates": candidates, "verbose": 0,
    }, threads


def build_rescnn(args, variable):
    """Build a residual CNN without changing the caller's RNG or CPU settings."""
    settings, threads = _settings(args, variable)
    with _torch_runtime(settings["device"], threads, _RESCNN.label) as device:
        return _RESCNN.estimator()(**dict(settings, device=device))


def _select_rescnn_spec(settings, device, x, y, *, min_samples, seed):
    return _select_network_spec(settings, device, x, y, min_samples=min_samples,
                                seed=seed, network=_RESCNN)


class ResCNNOperationalArtifact(UNetOperationalArtifact):
    """Portable ResCNN state with the shared training-only probability model."""


def load_rescnn_artifact(path):
    """Load a CLI ResCNN artifact (trusted pickle files only)."""
    with Path(path).open("rb") as source:
        artifact = pickle.load(source)
    if not isinstance(artifact, ResCNNOperationalArtifact):
        raise TypeError("File does not contain a CLI ResCNN operational artifact")
    return artifact


def run_rescnn(args, obs, rawhind, rawfuture, variable, workdir=None, *, split_plan=None):
    """Return OOS hindcast and operational forecast means and probabilities."""
    return _run_spatial_network(args, obs, rawhind, rawfuture, variable, workdir,
                                split_plan=split_plan, network=_RESCNN,
                                settings_function=_settings,
                                artifact_type=ResCNNOperationalArtifact)
