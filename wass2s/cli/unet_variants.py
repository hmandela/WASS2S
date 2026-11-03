"""Compact UNet++ and RISE-UNet adaptations for WASS2S seasonal regression.

Both reuse the existing fold-local spatial-network calibration. Deep supervision
acts on training heads; early stopping and inference use the deepest head. RISE
uses residual, multi-kernel inception and squeeze-and-excitation channel blocks.
These are compact GroupNorm adaptations, not published pretrained models.
"""
from __future__ import annotations

from copy import copy
from functools import partial
from numbers import Integral

from .unet import (
    _NetworkDefinition, _settings as _unet_settings, _torch_runtime,
    _run_spatial_network, UNetOperationalArtifact, load_unet_artifact,
)

_UNETPP = _NetworkDefinition(
    "UNetPP", "WAS_mme_UNetPP", "wass2s.was_mme_unet_variants",
    "UNet++ (WASS2S compact)", "unetpp",
    ("filters", "dropout", "lr", "weight_decay", "deep_supervision"),
    ("filters", "deep_supervision"),
)
_RISE_UNET = _NetworkDefinition(
    "RISEUNet", "WAS_mme_RISEUNet", "wass2s.was_mme_unet_variants",
    "RISE-UNet (WASS2S compact)", "rise-unet",
    ("filters", "dropout", "lr", "weight_decay", "deep_supervision", "se_reduction"),
    ("filters", "deep_supervision", "se_reduction"),
)
_NETWORKS = {"unetpp": _UNETPP, "rise-unet": _RISE_UNET}


def _network(architecture):
    try:
        return _NETWORKS[architecture]
    except (KeyError, TypeError) as exc:
        raise ValueError("architecture must be 'unetpp' or 'rise-unet'") from exc


def _settings(args, variable, *, architecture):
    _network(architecture)
    # Native CLI factories have optional_cv_gap_years; run_optional already
    # supplies cv_gap_years. Normalize a copy so direct factory use also works.
    local = copy(args)
    if not hasattr(local, "cv_gap_years"):
        local.cv_gap_years = getattr(args, "optional_cv_gap_years", 1)
    settings, threads = _unet_settings(local, variable)
    deep = getattr(args, "unet_deep_supervision", True)
    if not isinstance(deep, bool):
        raise ValueError("unet_deep_supervision must be a boolean")
    settings["deep_supervision"] = deep
    if architecture == "rise-unet":
        reduction = getattr(args, "rise_se_reduction", 8)
        if isinstance(reduction, bool) or not isinstance(reduction, Integral) or reduction < 1:
            raise ValueError("rise_se_reduction must be a positive integer")
        settings["se_reduction"] = int(reduction)
    return settings, threads


def build_unet_variant(args, variable, architecture):
    """Construct one explicit variant, restoring the caller's torch state."""
    network = _network(architecture)
    settings, threads = _settings(args, variable, architecture=architecture)
    with _torch_runtime(settings["device"], threads, network.label) as device:
        return network.estimator()(**dict(settings, device=device))


class UNetPPOperationalArtifact(UNetOperationalArtifact):
    """Portable fitted UNet++ and its training-only probability calibration."""


class RISEUNetOperationalArtifact(UNetOperationalArtifact):
    """Portable fitted compact RISE-UNet and its probability calibration."""


def load_unet_variant_artifact(path):
    """Load trusted local artifacts produced by either new network."""
    artifact = load_unet_artifact(path)
    if not isinstance(artifact, (UNetPPOperationalArtifact, RISEUNetOperationalArtifact)):
        raise TypeError("File is not a UNet++ or compact RISE-UNet operational artifact")
    return artifact


def run_unet_variant(args, obs, rawhind, rawfuture, variable, workdir=None, *,
                     architecture, split_plan=None):
    """Return OOS hindcast and future deterministic/probabilistic products."""
    network = _network(architecture)
    return _run_spatial_network(
        args, obs, rawhind, rawfuture, variable, workdir,
        split_plan=split_plan, network=network,
        settings_function=partial(_settings, architecture=architecture),
        artifact_type=(UNetPPOperationalArtifact if architecture == "unetpp"
                       else RISEUNetOperationalArtifact),
    )
