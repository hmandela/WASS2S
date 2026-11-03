# -*- coding: utf-8 -*-
"""Compact UNet++ and RISE-UNet adaptations for seasonal MME regression.

Both use the fold-local preprocessing, masked losses, blocked early stopping,
full-training-set refit and residual probability calibration of WAS_mme_UNet.
They do not turn attention weights into forecast probabilities.

UNet++ uses dense nested skip connections. RISE-UNet replaces its convolutional
blocks with residual multi-kernel inception blocks and squeeze/excitation (SE),
followed by two-convolution residual/SE refinement. The deepest encoder uses
residual/SE only. Other nodes uniformly use the full RISE block, a configurable
adaptation of the author's fixed-depth block placement. These are compact
regression adaptations: GroupNorm replaces BatchNorm, widths
are configurable, and linear heads permit signed standardized rainfall/temperature anomalies.
Correction of 15 September 2026 (inherited defaults): ``norm='none'`` removes
the per-sample spatial GroupNorm that hides domain-uniform predictor anomalies,
and a 1x1 ``linear_skip`` from the inputs is added to every supervised head so
deep supervision and the deployed head see the same linear baseline.
Dropout is disabled at inference; uncertainty remains residual calibrated. Deep
supervision averages the masked training losses from the nested top-level heads.
Validation, prediction and inference ALWAYS use the deepest head, not an
average of the heads. Thus early stopping measures the deployed predictor.

Architecture references:
* UNet++: Zhou et al., https://arxiv.org/abs/1807.10165
* RISE author implementation (multi-kernel residual/SE and nested skips):
  https://github.com/kyle-lesinger/dl_dm_rzsm_subseasonal_forecast/
  blob/master/function/modelRzsmReluExtraConv.py
"""
from __future__ import annotations

import math
from numbers import Integral

from .was_mme_unet import HAS_TORCH, WAS_mme_UNet, torch, nn, F


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _filters(values):
    try:
        result = tuple(_positive_int(value, "filters") for value in values)
    except TypeError as exc:
        raise ValueError("filters must be a nonempty sequence of positive integers.") from exc
    if not result:
        raise ValueError("filters must be a nonempty sequence of positive integers.")
    return result


def _boolean(value, name):
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be True or False.")
    return value


def _group_count(channels, requested):
    # At least two channels/group whenever possible: batch=1 and a 1x1
    # bottleneck must remain valid (ordinary channels==groups would fail).
    groups = min(_positive_int(requested, "norm_groups"), max(1, channels // 2))
    while channels % groups:
        groups -= 1
    return groups


def _training_parameters(params):
    for key in ("dropout", "lr", "weight_decay"):
        value = float(params[key])
        if not math.isfinite(value):
            raise ValueError(f"{key} must be finite.")
        params[key] = value
    if not 0 <= params["dropout"] < 1 or params["lr"] <= 0 or params["weight_decay"] < 0:
        raise ValueError("Require dropout in [0, 1), lr > 0 and weight_decay >= 0.")
    return params


if HAS_TORCH:
    def _norm_layer(norm, channels, norm_groups):
        if norm == "group":
            return nn.GroupNorm(_group_count(channels, norm_groups), channels)
        if norm == "none":
            return nn.Identity()
        raise ValueError("norm must be 'group' or 'none'.")


    class _ConvNormAct(nn.Sequential):
        def __init__(self, in_channels, out_channels, kernel_size, norm_groups,
                     norm="group"):
            super().__init__(
                nn.Conv2d(in_channels, out_channels, kernel_size,
                          padding=kernel_size // 2, bias=norm == "none"),
                _norm_layer(norm, out_channels, norm_groups),
                nn.SiLU(inplace=False),
            )


    class _NestedConv(nn.Sequential):
        def __init__(self, in_channels, out_channels, dropout=0.2, norm_groups=8,
                     norm="group", **unused):
            super().__init__(
                _ConvNormAct(in_channels, out_channels, 3, norm_groups, norm),
                nn.Dropout2d(dropout),
                _ConvNormAct(out_channels, out_channels, 3, norm_groups, norm),
                nn.Dropout2d(dropout),
            )


    class SEBlock(nn.Module):
        """Global average -> channel bottleneck -> sigmoid feature gates."""
        def __init__(self, channels, reduction=8):
            super().__init__()
            channels = _positive_int(channels, "channels")
            reduction = _positive_int(reduction, "se_reduction")
            hidden = max(1, channels // reduction)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.excitation = nn.Sequential(
                nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=False),
                nn.Conv2d(hidden, channels, 1), nn.Sigmoid(),
            )

        def gates(self, x):
            return self.excitation(self.pool(x))

        def forward(self, x):
            return x * self.gates(x)


    class _ResidualSEBlock(nn.Module):
        """Two 3x3 convolutions with a projected shortcut, then SE."""
        def __init__(self, in_channels, out_channels, dropout=0.2, norm_groups=8,
                     se_reduction=8, norm="group"):
            super().__init__()
            def conv_relu(incoming, outgoing, kernel):
                return nn.Sequential(
                    nn.Conv2d(incoming, outgoing, kernel, padding=kernel // 2,
                              bias=norm == "none"),
                    _norm_layer(norm, outgoing, norm_groups),
                    nn.ReLU(inplace=False),
                )
            self.shortcut = conv_relu(in_channels, out_channels, 1)
            self.convolutions = nn.Sequential(
                conv_relu(in_channels, out_channels, 3), nn.Dropout2d(dropout),
                conv_relu(out_channels, out_channels, 3), nn.Dropout2d(dropout),
            )
            self.se = SEBlock(out_channels, se_reduction)

        def forward(self, x):
            return self.se(self.shortcut(x) + self.convolutions(x))


    class RISEBlock(nn.Module):
        """Residual inception/SE followed by a two-convolution residual/SE.

        Each 3x3/5x5/7x7 branch uses conv, GroupNorm, ReLU, spatial dropout,
        1x1 conv, GroupNorm. The fourth branch uses max-pool5 followed by the
        same 1x1 operations. Concatenation is added to a projected shortcut,
        then attended by SE. The final residual/SE stage sets output width.
        """
        def __init__(self, in_channels, out_channels, dropout=0.2, norm_groups=8,
                     se_reduction=8, norm="group"):
            super().__init__()
            in_channels = _positive_int(in_channels, "in_channels")
            out_channels = _positive_int(out_channels, "out_channels")
            _positive_int(norm_groups, "norm_groups")
            if not math.isfinite(float(dropout)) or not 0 <= dropout < 1:
                raise ValueError("dropout must be finite and in [0, 1).")
            self.branch_width = max(2, math.ceil(out_channels / 4))
            def branch(kernel):
                return nn.Sequential(
                    nn.Conv2d(in_channels, self.branch_width, kernel,
                              padding=kernel // 2, bias=norm == "none"),
                    _norm_layer(norm, self.branch_width, norm_groups),
                    nn.ReLU(inplace=False), nn.Dropout2d(dropout),
                    nn.Conv2d(self.branch_width, self.branch_width, 1, bias=norm == "none"),
                    _norm_layer(norm, self.branch_width, norm_groups),
                )
            self.branches = nn.ModuleList([branch(kernel) for kernel in (3, 5, 7)] + [
                nn.Sequential(nn.MaxPool2d(5, stride=1, padding=2), branch(1))])
            inception_width = 4 * self.branch_width
            self.shortcut = (nn.Identity() if in_channels == inception_width else
                             nn.Conv2d(in_channels, inception_width, 1, bias=False))
            self.se = SEBlock(inception_width, se_reduction)
            self.refine = _ResidualSEBlock(inception_width, out_channels, dropout,
                                           norm_groups, se_reduction, norm)

        def forward(self, x):
            merged = torch.cat([branch(x) for branch in self.branches], dim=1)
            return self.refine(self.se(merged + self.shortcut(x)))


    class NestedUNet(nn.Module):
        """Dense UNet++ with an optional RISE block at every nested node.

        ``filters`` are the widths before the bottleneck, as in WASS2S UNet.
        Widths are ``filters + (2*filters[-1],)``. For each column j>0,
        x[i,j] receives x[i,0], ..., x[i,j-1] and upsampled x[i+1,j-1].
        ``forward_heads`` returns supervised heads in increasing nested depth;
        ``forward`` always returns only the deepest (last) head.
        """
        def __init__(self, n_channels, n_classes=1, filters=(4, 8, 16),
                     bilinear=True, dropout=0.2, norm_groups=8,
                     deep_supervision=True, rise=False, se_reduction=8,
                     norm="group", linear_skip=False):
            super().__init__()
            if norm not in {"group", "none"}:
                raise ValueError("norm must be 'group' or 'none'.")
            _positive_int(n_channels, "n_channels")
            _positive_int(n_classes, "n_classes")
            self.filters = _filters(filters)
            self.bilinear = _boolean(bilinear, "bilinear")
            self.deep_supervision = _boolean(deep_supervision, "deep_supervision")
            self.rise = _boolean(rise, "rise")
            _positive_int(norm_groups, "norm_groups")
            _positive_int(se_reduction, "se_reduction")
            if not math.isfinite(float(dropout)) or not 0 <= dropout < 1:
                raise ValueError("dropout must be finite and in [0, 1).")
            self.depth = len(self.filters)
            self.widths = self.filters + (2 * self.filters[-1],)
            block = RISEBlock if self.rise else _NestedConv
            self.blocks = nn.ModuleDict()
            self.ups = nn.ModuleDict()
            self.pool = nn.MaxPool2d(2)
            for i, width in enumerate(self.widths):
                previous = n_channels if i == 0 else self.widths[i - 1]
                encoder_block = _ResidualSEBlock if self.rise and i == self.depth else block
                self.blocks[f"{i}_0"] = encoder_block(previous, width, dropout, norm_groups,
                                                       se_reduction=se_reduction, norm=norm)
            for j in range(1, self.depth + 1):
                for i in range(self.depth - j + 1):
                    key = f"{i}_{j}"
                    width = self.widths[i]
                    if self.bilinear:
                        incoming = self.widths[i + 1]
                    else:
                        self.ups[key] = nn.ConvTranspose2d(
                            self.widths[i + 1], width, kernel_size=2, stride=2)
                        incoming = width
                    self.blocks[key] = block(j * width + incoming, width,
                                              dropout, norm_groups,
                                              se_reduction=se_reduction, norm=norm)
            self.head_columns = (tuple(range(1, self.depth + 1)) if self.deep_supervision
                                 else (self.depth,))
            self.heads = nn.ModuleDict({str(j): nn.Conv2d(self.widths[0], n_classes, 1)
                                        for j in self.head_columns})
            # One shared linear input->output path added to EVERY supervised
            # head: deep supervision then trains the same linear baseline the
            # deployed deepest head uses (see the 2026-09-15 correction).
            self.linear_skip = (nn.Conv2d(n_channels, n_classes, 1)
                                if _boolean(linear_skip, "linear_skip") else None)

        def _head(self, column, node, x):
            out = self.heads[str(column)](node)
            if self.linear_skip is not None:
                out = out + self.linear_skip(x)
            return out

        def _nodes(self, x):
            if x.ndim != 4 or min(x.shape[-2:]) < 2 ** self.depth:
                raise ValueError(f"NestedUNet requires N,C,Y,X and sides >= {2 ** self.depth}.")
            nodes = {(0, 0): self.blocks["0_0"](x)}
            for i in range(1, self.depth + 1):
                nodes[i, 0] = self.blocks[f"{i}_0"](self.pool(nodes[i - 1, 0]))
            for j in range(1, self.depth + 1):
                for i in range(self.depth - j + 1):
                    key = f"{i}_{j}"
                    deep = nodes[i + 1, j - 1]
                    shape = nodes[i, 0].shape[-2:]
                    if self.bilinear:
                        deep = F.interpolate(deep, size=shape, mode="bilinear", align_corners=False)
                    else:
                        deep = self.ups[key](deep)
                        # Max-pooling odd dimensions leaves at most one missing
                        # row/column. Pad symmetrically like the existing UNet.
                        dy, dx = shape[0] - deep.shape[-2], shape[1] - deep.shape[-1]
                        deep = F.pad(deep, (dx // 2, dx - dx // 2, dy // 2, dy - dy // 2))
                    nodes[i, j] = self.blocks[key](torch.cat(
                        [nodes[i, k] for k in range(j)] + [deep], dim=1))
            return nodes

        def forward_heads(self, x):
            nodes = self._nodes(x)
            return tuple(self._head(j, nodes[0, j], x) for j in self.head_columns)

        def forward(self, x):
            nodes = self._nodes(x)
            return self._head(self.depth, nodes[0, self.depth], x)


class WAS_mme_UNetPP(WAS_mme_UNet):
    """UNet++ seasonal calibration; training options match WAS_mme_UNet.

    ``deep_supervision=True`` averages the training loss across nested heads.
    Prediction and early stopping always use the final, deepest head. Compact
    defaults target short hindcasts and the 4 GiB RTX A400; superiority over
    simpler models must be established by held-out-year validation.
    """
    _was_unet = False
    _was_unetpp = True
    _was_spatial_cnn = True
    _network_name = "UNet++"

    def __init__(self, filters=(4, 8, 16), deep_supervision=True, **training_kwargs):
        filters = _filters(filters)
        self.deep_supervision = _boolean(deep_supervision, "deep_supervision")
        if "norm_groups" in training_kwargs:
            _positive_int(training_kwargs["norm_groups"], "norm_groups")
        training_kwargs.setdefault("tuning_filters", ((4, 8), (4, 8, 16)))
        training_kwargs["tuning_filters"] = tuple(_filters(f) for f in training_kwargs["tuning_filters"])
        if not training_kwargs["tuning_filters"]:
            raise ValueError("tuning_filters cannot be empty.")
        super().__init__(filters=filters, **training_kwargs)
        _training_parameters(dict(dropout=self.dropout, lr=self.lr, weight_decay=self.weight_decay))

    def _effective_params(self, best_params=None):
        selected = best_params or {}
        if len(selected) == 1 and 0 in selected and isinstance(selected[0], dict):
            selected = selected[0]
        if "filters" in selected:
            _filters(selected["filters"])
        params = super()._effective_params(selected)
        params["deep_supervision"] = _boolean(
            selected.get("deep_supervision", self.deep_supervision), "deep_supervision")
        return _training_parameters(params)

    def _build_network(self, input_channels, params):
        return NestedUNet(input_channels, filters=params["filters"],
                          bilinear=self.bilinear, dropout=params["dropout"],
                          norm_groups=self.norm_groups,
                          deep_supervision=params.get("deep_supervision", self.deep_supervision),
                          norm=self.norm, linear_skip=self.linear_skip)

    def _training_loss_map(self, model, inputs, target, loss_fn):
        if model.deep_supervision:
            losses = [loss_fn(head.squeeze(1), target) for head in model.forward_heads(inputs)]
            return torch.stack(losses, dim=0).mean(dim=0)
        return super()._training_loss_map(model, inputs, target, loss_fn)

    def _tuning_candidates(self):
        return [self._effective_params(p) for p in super()._tuning_candidates()]

    def summary(self):
        print(f"WAS_mme_{self._network_name} — spatial nonlinear MME calibration")
        for key, value in self._effective_params(self.best_params_).items():
            print(f"{key}: {value}")
        print("prediction/early stopping head: final (deepest)")
        print(f"device: {self.device}; fitted: {self.model is not None}")


class WAS_mme_RISEUNet(WAS_mme_UNetPP):
    """Compact RISE-UNet adaptation: nested skips + residual inception + SE.

    GroupNorm, configurable widths and linear rainfall/temperature regression
    heads intentionally adapt the published soil-moisture network to WASS2S.
    ``se_reduction`` controls the SE channel bottleneck (default 8).
    """
    _was_unetpp = False
    _was_rise_unet = True
    _network_name = "RISE-UNet"

    def __init__(self, filters=(4, 8, 16), deep_supervision=True, se_reduction=8,
                 **training_kwargs):
        self.se_reduction = _positive_int(se_reduction, "se_reduction")
        super().__init__(filters=filters, deep_supervision=deep_supervision,
                         **training_kwargs)

    def _effective_params(self, best_params=None):
        selected = best_params or {}
        if len(selected) == 1 and 0 in selected and isinstance(selected[0], dict):
            selected = selected[0]
        params = super()._effective_params(selected)
        params["se_reduction"] = _positive_int(
            selected.get("se_reduction", self.se_reduction), "se_reduction")
        return params

    def _build_network(self, input_channels, params):
        return NestedUNet(input_channels, filters=params["filters"],
                          bilinear=self.bilinear, dropout=params["dropout"],
                          norm_groups=self.norm_groups,
                          deep_supervision=params.get("deep_supervision", self.deep_supervision),
                          rise=True, se_reduction=params.get("se_reduction", self.se_reduction),
                          norm=self.norm, linear_skip=self.linear_skip)


__all__ = ["WAS_mme_UNetPP", "WAS_mme_RISEUNet"]
if HAS_TORCH:
    __all__ += ["NestedUNet", "RISEBlock", "SEBlock"]

# Capture only constructor inputs; no learned weights enter another CV fold.
from .was_crossfit_adapters import capture_constructors as _capture_constructors
_capture_constructors(globals())
