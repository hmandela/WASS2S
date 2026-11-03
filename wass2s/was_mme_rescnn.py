# -*- coding: utf-8 -*-
"""Residual CNN calibration on the native WASS2S predictor/target grid.

This is a deterministic spatial post-processor, sharing U-Net's masking,
fold-local transformations, early stopping followed by a fresh refit, and
out-of-fold residual probability calibration. It changes only the architecture
and its search space. ``M`` denotes stable predictor channels (for example each
GCM's mean and spread), not persistent identities for exchangeable members.

Residual connections act on hidden feature maps. They do not prescribe an
additive correction to the raw precipitation ensemble mean. A ResCNN has no
pooling or decoder: every convolution preserves the original spatial grid.
Its skill relative to U-Net must be established on identical held-out years.

Architectural references: He et al. (2016), https://arxiv.org/abs/1512.03385;
Wu and He (2018), https://arxiv.org/abs/1803.08494. GroupNorm uses statistics
over spatial positions within a group; the convolutional receptive-field size
reported below does not bound this normalization-induced spatial dependence.

Correction of 15 September 2026 (inherited defaults): per-sample GroupNorm over
space makes a domain-uniform predictor anomaly nearly invisible, so ``norm``
defaults to 'none' and a 1x1 ``linear_skip`` from the inputs to the output is
added; both follow the shared WAS_mme_UNet constructor options.
"""
from __future__ import annotations

import itertools
from numbers import Integral

from .was_mme_unet import HAS_TORCH, WAS_mme_UNet

if HAS_TORCH:
    from .was_mme_unet import nn, _group_count, _norm_layer


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _architecture(channels, num_blocks, dilations):
    channels = _positive_integer(channels, "channels")
    num_blocks = _positive_integer(num_blocks, "num_blocks")
    if dilations is None:
        dilations = (1,) * num_blocks
    try:
        dilations = tuple(_positive_integer(d, "dilation") for d in dilations)
    except TypeError as exc:
        raise ValueError("dilations must contain one positive integer per block.") from exc
    if len(dilations) != num_blocks:
        raise ValueError("dilations must contain exactly num_blocks entries.")
    return channels, num_blocks, dilations


if HAS_TORCH:

    class ResidualBlock(nn.Module):
        """Two same-grid convolutions followed by an identity feature skip."""

        def __init__(self, channels, dilation=1, dropout=0.2, norm_groups=8,
                     norm="group"):
            super().__init__()
            bias = norm == "none"
            self.branch = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=dilation,
                          dilation=dilation, bias=bias),
                _norm_layer(norm, channels, norm_groups),
                nn.SiLU(inplace=True),
                nn.Dropout2d(dropout),
                nn.Conv2d(channels, channels, 3, padding=dilation,
                          dilation=dilation, bias=bias),
                _norm_layer(norm, channels, norm_groups),
                nn.Dropout2d(dropout),
            )
            self.activation = nn.SiLU(inplace=False)

        def forward(self, x):
            return self.activation(x + self.branch(x))


    class ResCNN(nn.Module):
        """Compact residual CNN without pooling, resizing or downsampling."""

        def __init__(self, n_channels, n_classes=1, channels=16, num_blocks=3,
                     dilations=None, dropout=0.2, norm_groups=8, norm="group",
                     linear_skip=False):
            super().__init__()
            if norm not in {"group", "none"}:
                raise ValueError("norm must be 'group' or 'none'.")
            channels, num_blocks, dilations = _architecture(channels, num_blocks, dilations)
            n_channels = _positive_integer(n_channels, "n_channels")
            n_classes = _positive_integer(n_classes, "n_classes")
            norm_groups = _positive_integer(norm_groups, "norm_groups")
            if not 0 <= float(dropout) < 1:
                raise ValueError("dropout must be in [0, 1).")
            self.channels = channels
            self.num_blocks = num_blocks
            self.dilations = dilations
            self.convolutional_receptive_field = 3 + 4 * sum(dilations)
            self.stem = nn.Sequential(
                nn.Conv2d(n_channels, channels, 3, padding=1, bias=norm == "none"),
                _norm_layer(norm, channels, norm_groups),
                nn.SiLU(inplace=True),
            )
            self.blocks = nn.Sequential(*[
                ResidualBlock(channels, d, float(dropout), norm_groups, norm)
                for d in dilations
            ])
            self.output = nn.Conv2d(channels, n_classes, kernel_size=1)
            # Linear input->output path bypassing every normalisation (see the
            # 2026-09-15 correction in was_mme_unet).
            self.linear_skip = (nn.Conv2d(n_channels, n_classes, kernel_size=1)
                                if linear_skip else None)

        def forward(self, x):
            out = self.output(self.blocks(self.stem(x)))
            if self.linear_skip is not None:
                out = out + self.linear_skip(x)
            return out


class WAS_mme_ResCNN(WAS_mme_UNet):
    """WASS2S residual CNN with the same public fitting contract as U-Net.

    Parameters
    ----------
    channels : int
        Constant feature width, default 16.
    num_blocks : int
        Number of residual blocks (two convolutions each), default 3.
    dilations : sequence[int] or None
        One dilation per block. None uses ordinary convolutions throughout.
        Explicit ``(1, 2, 4)`` enlarges convolutional context for three blocks.
    tuning_channels : sequence[int]
        Candidate widths for optional blocked temporal hyperparameter tuning.
    tuning_num_blocks : sequence[int] or None
        Candidate depths; None retains the configured depth. A uniform dilation
        can be repeated at a candidate depth; a nonuniform schedule requires
        candidates to retain its depth rather than silently changing it.
    **training_kwargs
        U-Net training options: dropout, lr, weight_decay, epochs, batch_size,
        loss, device, early_stopping_patience, temporal_buffer, random_state,
        tuning_dropouts, tuning_lrs, tuning_weight_decays, tuning_folds, etc.
        Encoder-specific filters/bilinear/tuning_filters are not accepted.
    """

    _was_unet = False
    _was_spatial_cnn = True
    _was_rescnn = True
    _network_name = "ResCNN"

    def __init__(self, channels=16, num_blocks=3, dilations=None,
                 tuning_channels=(8, 16), tuning_num_blocks=None, **training_kwargs):
        forbidden = {"filters", "bilinear", "tuning_filters"} & set(training_kwargs)
        if forbidden:
            raise TypeError(f"ResCNN has no U-Net encoder options: {sorted(forbidden)}")
        self.channels, self.num_blocks, self.dilations = _architecture(
            channels, num_blocks, dilations)
        self.tuning_channels = tuple(_positive_integer(c, "tuning_channels")
                                     for c in tuning_channels)
        self.tuning_num_blocks = ((self.num_blocks,) if tuning_num_blocks is None else
                                 tuple(_positive_integer(b, "tuning_num_blocks")
                                       for b in tuning_num_blocks))
        if not self.tuning_channels or not self.tuning_num_blocks:
            raise ValueError("ResCNN tuning architecture choices cannot be empty.")
        if len(set(self.dilations)) > 1 and any(
                b != self.num_blocks for b in self.tuning_num_blocks):
            raise ValueError("A nonuniform dilation schedule requires tuning_num_blocks "
                             "to retain num_blocks.")
        super().__init__(**training_kwargs)

    @property
    def convolutional_receptive_field(self):
        """Width in cells of the configured convolutional path (excluding GN)."""
        return 3 + 4 * sum(self.dilations)

    def _effective_params(self, best_params=None):
        params = dict(channels=self.channels, num_blocks=self.num_blocks,
                      dilations=self.dilations, dropout=self.dropout,
                      lr=self.lr, weight_decay=self.weight_decay)
        if best_params:
            if len(best_params) == 1 and 0 in best_params and isinstance(best_params[0], dict):
                best_params = best_params[0]
            params.update({key: best_params[key] for key in params if key in best_params})
        params["channels"], params["num_blocks"], params["dilations"] = _architecture(
            params["channels"], params["num_blocks"], params["dilations"])
        for key in ("dropout", "lr", "weight_decay"):
            params[key] = float(params[key])
        if not 0 <= params["dropout"] < 1 or not params["lr"] > 0 or not params["weight_decay"] >= 0:
            raise ValueError("Require dropout in [0, 1), lr > 0 and weight_decay >= 0.")
        return params

    def _validate_grid(self, height, width, params):
        # No multiple-of-two restriction applies; with norm='group' a spatial
        # group still needs more than one value when temporal batch size is one.
        if min(height, width) < 1 or (self.norm == "group" and height * width < 2):
            raise ValueError("ResCNN requires at least two spatial grid cells for GroupNorm.")

    def _build_network(self, input_channels, params):
        return ResCNN(n_channels=input_channels, channels=params["channels"],
                      num_blocks=params["num_blocks"], dilations=params["dilations"],
                      dropout=params["dropout"], norm_groups=self.norm_groups,
                      norm=self.norm, linear_skip=self.linear_skip)

    def _architecture_complexity(self, params):
        # Monotone proxy for the dominant two 3x3 convolutions per block.
        return 18 * params["channels"] ** 2 * params["num_blocks"]

    def _tuning_candidates(self):
        groups = []
        for channels, blocks in sorted(itertools.product(self.tuning_channels,
                                                         self.tuning_num_blocks),
                                       key=lambda pair: (pair[0] ** 2 * pair[1], pair)):
            dilations = (self.dilations if blocks == self.num_blocks else
                         (self.dilations[0],) * blocks)
            groups.append([
                dict(channels=channels, num_blocks=blocks, dilations=dilations,
                     dropout=dropout, lr=lr, weight_decay=decay)
                for dropout, lr, decay in sorted(itertools.product(
                    self.tuning_dropouts, self.tuning_lrs, self.tuning_weight_decays))
            ])
        # Round-robin architectures before exhausting learning-rate variants.
        limit = self.max_tuning_candidates
        selected = []
        while any(groups) and (limit <= 0 or len(selected) < limit):
            for group in groups:
                if group and (limit <= 0 or len(selected) < limit):
                    selected.append(group.pop(0))
        return selected

    def summary(self):
        params = self._effective_params(self.best_params_)
        print("WAS_mme_ResCNN — spatial nonlinear MME calibration")
        for key, value in params.items():
            print(f"{key}: {value}")
        print(f"convolutional receptive field: {3 + 4 * sum(params['dilations'])} cells")
        print(f"device: {self.device}; fitted: {self.model is not None}")


__all__ = ["WAS_mme_ResCNN"]
if HAS_TORCH:
    __all__ += ["ResCNN", "ResidualBlock"]

# Store only constructor inputs; never clone fitted weights into a new CV fold.
from .was_crossfit_adapters import capture_constructors as _capture_constructors
_capture_constructors(globals())
