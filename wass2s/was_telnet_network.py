"""TelNet neural core for seasonal probabilistic forecasting.

Reimplementation of Pinheiro & Ouarda (2025), Communications Earth &
Environment 6:222, doi:10.1038/s43247-025-02207-2, equations 1--9 and 22--24.
The authors' reference implementation is https://github.com/enzopinheiro/telnet
(GPL-3.0). This module follows its partial-convolution / variable-selection /
autoregressive latent decoder design and is distributed under GPL-3.0-or-later.

WASS2S adaptations are explicit: compact defaults, arbitrary/singleton grids,
per-sample masks, missing-target-aware loss, explicit training/validation inputs,
and no dependency on Pyro or on a global DEVICE. No feature or target scaling
is learned here: the xarray wrapper must fit those transforms on training data.
PyTorch is loaded only when a neural class, loss or training routine is used.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from typing import Mapping

import numpy as np

__all__ = ["TelNetNetwork", "PartialConv2D", "GatedResidualNetwork",
           "VariableSelectionNetwork", "empirical_crps", "crps_loss",
           "fit_network", "torch_runtime"]
_COMPONENTS = None


def _load_components():
    global _COMPONENTS
    if _COMPONENTS is not None:
        return _COMPONENTS
    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
    except ImportError as exc:
        raise ImportError("TelNet requires PyTorch; install the optional TelNet dependencies") from exc

    class PartialConv2D(nn.Module):
        """Equation 2 with a shared binary spatial mask and updated support."""
        def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                     padding=1, bias=True):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                  stride=stride, padding=padding, bias=bias)
            self.register_buffer("mask_kernel", torch.ones(1, 1, *self.conv.kernel_size))
            self.window_size = int(np.prod(self.conv.kernel_size))

        def forward(self, x, mask):
            if x.ndim != 4:
                raise ValueError("PartialConv2D input must have shape B,C,Y,X")
            if mask.ndim == 3:
                mask = mask[:, None]
            if mask.ndim != 4 or mask.shape[1] != 1:
                raise ValueError("PartialConv2D needs one spatial mask channel")
            if mask.shape[0] == 1 and x.shape[0] != 1:
                mask = mask.expand(x.shape[0], -1, -1, -1)
            if mask.shape[0] != x.shape[0] or mask.shape[-2:] != x.shape[-2:]:
                raise ValueError("Partial-convolution mask does not match the input grid")
            mask = mask.to(device=x.device, dtype=x.dtype)
            # where, rather than nan * 0, prevents masked NaNs from contaminating cells.
            values = torch.where(mask > 0, x, torch.zeros_like(x))
            output = self.conv(values)
            with torch.no_grad():
                count = F.conv2d(mask, self.mask_kernel.to(dtype=x.dtype),
                                 stride=self.conv.stride, padding=self.conv.padding)
                valid = count > 0
                ratio = self.window_size / count.clamp_min(1.)
            bias = self.conv.bias
            if bias is not None:
                bias = bias.view(1, -1, 1, 1)
                output = (output - bias) * ratio + bias
            else:
                output = output * ratio
            output = torch.where(valid, output, torch.zeros_like(output))
            return output, valid.to(dtype=x.dtype)

    class GatedResidualNetwork(nn.Module):
        """Gated residual network (paper equations 3--6)."""
        def __init__(self, input_dim, hidden_dim, output_dim=None, dropout=0.):
            super().__init__()
            output_dim = input_dim if output_dim is None else output_dim
            self.linear_in = nn.Linear(input_dim, hidden_dim)
            self.linear_hidden = nn.Linear(hidden_dim, output_dim)
            self.gate = nn.Linear(output_dim, 2 * output_dim)
            self.skip = nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim)
            self.dropout = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(output_dim)

        def forward(self, values):
            transformed = self.linear_hidden(F.elu(self.linear_in(values)))
            transformed = self.dropout(transformed)
            content, gate = self.gate(transformed).chunk(2, dim=-1)
            return self.norm(self.skip(values) + content * torch.sigmoid(gate))

    class VariableSelectionNetwork(nn.Module):
        """Instance-wise weights over K covariates plus the lagged target map."""
        def __init__(self, n_variables, hidden_dim, dropout=0.):
            super().__init__()
            self.n_variables = int(n_variables)
            self.weight_network = GatedResidualNetwork(n_variables * hidden_dim,
                hidden_dim, output_dim=n_variables, dropout=dropout)
            self.variable_networks = nn.ModuleList([
                GatedResidualNetwork(hidden_dim, hidden_dim, dropout=dropout)
                for _ in range(n_variables)])

        def forward(self, variables):
            if variables.shape[-2] != self.n_variables:
                raise ValueError("VariableSelectionNetwork received the wrong variable count")
            weights = torch.softmax(self.weight_network(variables.flatten(-2)), dim=-1)
            transformed = torch.stack([net(variables[..., i, :])
                for i, net in enumerate(self.variable_networks)], dim=-2)
            return (weights[..., None] * transformed).sum(dim=-2), weights

    class TelNetNetwork(nn.Module):
        """Partial-convolution TelNet with a learned empirical ensemble head.

        Input shapes: x_lag (B,H,Y,X), x_cov (B,H,K), calendar arrays
        (B,H)/(B,L) with month labels 0..11, and mask (B,Y,X) or (Y,X).
        Output: (B,L,N,Y,X), and weights (B,H+L,K+1). The first forecast
        weight equals the last historical variable selection, as in the
        reference decoder. Subsequent weights explain each latent forecast step.

        The original grid encoder requires dimensions divisible by four. Our
        padded stride-two encoder and global partial convolution additionally
        support small/odd WASS2S grids while preserving the three-layer design.
        """
        def __init__(self, n_covariates, height, width, history_length,
                     forecast_length, hidden_dim=32, n_members=20, dropout=.05,
                     conv_channels=None, random_state=42):
            super().__init__()
            integers = (n_covariates, height, width, history_length, forecast_length,
                        hidden_dim, n_members)
            if any(int(v) != v for v in integers) or int(n_covariates) < 0:
                raise ValueError("TelNet dimensions must be integers; n_covariates can be zero")
            if min(int(v) for v in integers[1:]) < 1 or int(hidden_dim) < 4:
                raise ValueError("TelNet dimensions must be positive and hidden_dim >= 4")
            if not np.isfinite(dropout) or not 0 <= float(dropout) < 1:
                raise ValueError("TelNet dropout must be finite in [0,1)")
            self.n_covariates = int(n_covariates)
            self.height, self.width = int(height), int(width)
            self.history_length, self.forecast_length = int(history_length), int(forecast_length)
            self.hidden_dim, self.n_members = int(hidden_dim), int(n_members)
            channels = (max(1, self.hidden_dim // 4), max(1, self.hidden_dim // 2))
            if conv_channels is not None:
                channels = tuple(int(v) for v in conv_channels)
                if len(channels) != 2 or min(channels) < 1:
                    raise ValueError("conv_channels needs two positive channel widths")
            # Constructor initialization is reproducible without consuming caller RNG.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(random_state))
                self.pconv1 = PartialConv2D(1, channels[0], kernel_size=3, stride=2, padding=1)
                self.pconv2 = PartialConv2D(channels[0], channels[1], kernel_size=3, stride=2, padding=1)
                reduced_shape = ((self.height + 3) // 4, (self.width + 3) // 4)
                self.pconv3 = PartialConv2D(channels[1], self.hidden_dim,
                    kernel_size=reduced_shape, stride=1, padding=0)
                self.covariate_encoders = nn.ModuleList([
                    nn.Linear(1, self.hidden_dim, bias=False) for _ in range(self.n_covariates)])
                self.month_embedding = nn.Embedding(12, self.hidden_dim)
                self.variable_selection = VariableSelectionNetwork(self.n_covariates + 1,
                    self.hidden_dim, dropout=0.)
                self.norm_history = nn.LayerNorm(self.hidden_dim)
                self.norm_output = nn.LayerNorm(self.hidden_dim)
                self.norm_recursive = nn.LayerNorm(self.hidden_dim)
                self.decoder = nn.LSTM(self.hidden_dim, self.hidden_dim,
                                       num_layers=1, batch_first=True)
                self.dropout = nn.Dropout(float(dropout))
                self.head = nn.Linear(self.hidden_dim, self.n_members * self.height * self.width)
                self.apply(self._initialize)

        @staticmethod
        def _initialize(layer):
            if isinstance(layer, (nn.Linear, nn.Conv2d, nn.Embedding)):
                nn.init.trunc_normal_(layer.weight, std=.02)
                if getattr(layer, "bias", None) is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.LSTM):
                for name, value in layer.named_parameters():
                    nn.init.trunc_normal_(value, std=.02) if "weight" in name else nn.init.zeros_(value)
            elif isinstance(layer, nn.LayerNorm):
                nn.init.ones_(layer.weight); nn.init.zeros_(layer.bias)

        def forward(self, x_lag, x_cov, month_history, month_future, mask):
            if x_lag.ndim != 4:
                raise ValueError("x_lag must have shape B,H,Y,X")
            b, h, y, x = x_lag.shape
            if (h, y, x) != (self.history_length, self.height, self.width):
                raise ValueError("x_lag dimensions do not match the configured network")
            if x_cov.shape != (b, h, self.n_covariates):
                raise ValueError("x_cov must have shape B,H,n_covariates")
            if month_history.shape != (b, h) or month_future.shape != (b, self.forecast_length):
                raise ValueError("Calendar arrays do not match the history/forecast lengths")
            if any(bool(((v < 0) | (v > 11)).any()) for v in (month_history, month_future)):
                raise ValueError("Calendar labels must be integers in 0..11")
            if mask.ndim == 2:
                mask = mask[None].expand(b, -1, -1)
            if mask.shape != (b, y, x):
                raise ValueError("mask must have shape B,Y,X or Y,X")
            mask = mask.to(device=x_lag.device, dtype=torch.bool)
            if not bool(torch.isfinite(x_cov).all()):
                raise ValueError("Covariates must be finite after training-only preprocessing")
            if bool((~torch.isfinite(x_lag) & mask[:, None]).any()):
                raise ValueError("Valid lagged target pixels must be finite")
            past_calendar = self.dropout(self.month_embedding(month_history.long()))
            lead_calendar = self.dropout(self.month_embedding(month_future.long()))
            rainfall_embeddings = []
            for time in range(h):
                z, support = self.pconv1(x_lag[:, time:time+1], mask)
                z, support = self.pconv2(z, support)
                z, _ = self.pconv3(z, support)
                rainfall_embeddings.append(z.flatten(1))
            lag_embedding = self.dropout(torch.stack(rainfall_embeddings, dim=1))
            cov_embedding = torch.stack([encoder(x_cov[..., i:i+1])
                for i, encoder in enumerate(self.covariate_encoders)], dim=2) if self.n_covariates else x_lag.new_empty((b,h,0,self.hidden_dim))
            cov_embedding = self.dropout(cov_embedding)
            variables = torch.cat((lag_embedding[:, :, None], cov_embedding), dim=2)
            selected, history_weights = self.variable_selection(variables + past_calendar[:, :, None])
            decoder_input = self.norm_history(selected)
            ensembles, future_weights = [], []
            current_weights = history_weights[:, -1]
            for lead in range(self.forecast_length):
                # As in the authors' code: shared LSTM, recursive latent input;
                # no observed future target and no teacher forcing at inference.
                decoded, _ = self.decoder(self.dropout(decoder_input))
                latent = self.norm_output(decoded[:, -1]) + lead_calendar[:, lead]
                ensembles.append(self.head(self.dropout(latent)).reshape(
                    b, self.n_members, self.height, self.width))
                future_weights.append(current_weights)
                if lead + 1 < self.forecast_length:
                    persisted_covariates = cov_embedding[:, -1] + past_calendar[:, -1, None]
                    recursive_variables = torch.cat((latent[:, None], persisted_covariates), dim=1)
                    selected_next, current_weights = self.variable_selection(recursive_variables)
                    decoder_input = self.norm_recursive(selected_next)[:, None]
            values = torch.stack(ensembles, dim=1)
            values = torch.where(mask[:, None, None], values, torch.zeros_like(values))
            weights = torch.cat((history_weights, torch.stack(future_weights, dim=1)), dim=1)
            return values, weights

    classes = {"PartialConv2D": PartialConv2D, "GatedResidualNetwork": GatedResidualNetwork,
               "VariableSelectionNetwork": VariableSelectionNetwork, "TelNetNetwork": TelNetNetwork}
    for name, cls in classes.items():
        cls.__module__ = __name__; cls.__qualname__ = name
    globals().update(classes)
    _COMPONENTS = {**classes, "torch": torch}
    return _COMPONENTS


def __getattr__(name):
    if name in {"PartialConv2D", "GatedResidualNetwork", "VariableSelectionNetwork", "TelNetNetwork"}:
        return _load_components()[name]
    raise AttributeError(name)


def empirical_crps(predictions, targets, *, member_dim=2, fair=False):
    """Pointwise empirical-ensemble CRPS, computed by sorting in O(N log N).

    Default is paper Eq.22 with N squared. ``fair=True`` uses N(N-1), an
    explicitly different finite-ensemble estimator; at least two members are
    required in that case. Inputs are finite tensors with the member axis added
    to the target shape. This function does not reduce the spatial/sample axes.
    """
    torch = _load_components()["torch"]
    n = predictions.shape[member_dim]
    if n < 1 or (fair and n < 2):
        raise ValueError("CRPS needs members; fair=True needs at least two")
    expected = list(predictions.shape); expected.pop(member_dim)
    if tuple(targets.shape) != tuple(expected):
        raise ValueError("CRPS target shape does not match its ensemble")
    ordered = predictions.sort(dim=member_dim).values
    rank = torch.arange(1, n+1, device=predictions.device, dtype=predictions.dtype)
    coefficient = 2*rank - n - 1
    shape = [1] * predictions.ndim; shape[member_dim] = n
    spread = (ordered * coefficient.view(shape)).sum(dim=member_dim)
    denominator = n * (n - 1) if fair else n * n
    return (predictions-targets.unsqueeze(member_dim)).abs().mean(dim=member_dim) - spread / denominator


def _loss_components(predictions, targets, mask=None, latitude_weights=None, fair=False):
    torch = _load_components()["torch"]
    valid = torch.isfinite(targets)
    if mask is not None:
        mask = torch.as_tensor(mask, dtype=torch.bool, device=targets.device)
        if mask.ndim == 2:mask = mask[None, None]
        elif mask.ndim == 3:mask = mask[:, None]
        valid = valid & mask
    if not bool(valid.any()):
        raise ValueError("No finite target cells are available for TelNet loss")
    if bool((~torch.isfinite(predictions) & valid.unsqueeze(2)).any()):
        raise FloatingPointError("Nonfinite TelNet ensemble values on valid target cells")
    safe_targets = torch.where(valid, targets, torch.zeros_like(targets))
    safe_predictions = torch.where(valid.unsqueeze(2), predictions, torch.zeros_like(predictions))
    scores = empirical_crps(safe_predictions, safe_targets, fair=fair)
    weights = valid.to(dtype=scores.dtype)
    if latitude_weights is not None:
        lat = torch.as_tensor(latitude_weights, dtype=scores.dtype, device=scores.device)
        if lat.ndim == 1:lat = lat[None, None, :, None]
        elif lat.ndim == 2:lat = lat[None, None]
        elif lat.ndim == 3:lat = lat[:, None]
        if bool((~torch.isfinite(lat) | (lat < 0)).any()):
            raise ValueError("Latitude weights must be finite and nonnegative")
        weights = weights * lat
    denominator = weights.sum()
    if not bool(denominator > 0):
        raise ValueError("TelNet loss has no positively weighted valid targets")
    return (scores * weights).sum(), denominator


def crps_loss(predictions, targets, mask=None, latitude_weights=None, *, fair=False):
    """Finite-target-aware, optionally latitude-weighted mean empirical CRPS."""
    numerator, denominator = _loss_components(predictions,targets,mask,latitude_weights,fair)
    return numerator / denominator


@contextmanager
def torch_runtime(seed=42, device="cpu", num_threads=1):
    """Temporarily bound neural runtime and restore caller RNG/thread state."""
    torch = _load_components()["torch"]
    if str(device) not in {"cpu", "cuda", "auto"}:
        raise ValueError("TelNet device must be cpu, cuda or auto")
    resolved = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else str(device)
    if resolved == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("TelNet CUDA requested but no CUDA device is available")
    if int(num_threads) != num_threads or num_threads < 1:
        raise ValueError("TelNet num_threads must be a positive integer")
    previous_threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    devices = list(range(torch.cuda.device_count())) if resolved == "cuda" else []
    try:
        torch.set_num_threads(int(num_threads))
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(seed))
            torch.use_deterministic_algorithms(True, warn_only=True)
            yield resolved
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def _tensor_inputs(inputs, device):
    torch = _load_components()["torch"]
    names = ("x_lag", "x_cov", "month_history", "month_future", "mask")
    if isinstance(inputs, Mapping):
        values = [inputs[k] for k in names]
    else:
        values = list(inputs)
        if len(values) != len(names):raise ValueError("TelNet needs five input arrays")
    tensors = [torch.as_tensor(v,device=device,dtype=torch.long if i in (2,3) else torch.bool if i==4 else torch.float32)
               for i,v in enumerate(values)]
    b = tensors[0].shape[0]
    if tensors[-1].ndim == 2:tensors[-1] = tensors[-1][None].expand(b,-1,-1)
    if b < 1 or any(v.shape[0] != b for v in tensors):
        raise ValueError("TelNet arrays need a common, nonempty sample axis")
    return tensors


def fit_network(network, train_inputs, train_targets, *, validation_inputs=None,
                validation_targets=None, epochs=20, batch_size=8,
                learning_rate=1e-3, weight_decay=1e-5, patience=5,
                min_delta=1e-5, gradient_clip=1., latitude_weights=None,
                seed=42, device="cpu", num_threads=1, fair_crps=False):
    """Fit supplied train data, with optional explicitly independent validation.

    This routine never creates a random train/validation split and never sees
    any held-out outer target. If validation is supplied, the best epoch/state
    are returned. The caller can then refit a fresh network for that epoch count
    on its entire permitted training set. Returns ``(network, history)``.
    """
    torch = _load_components()["torch"]
    if (validation_inputs is None) != (validation_targets is None):
        raise ValueError("Supply both validation inputs and targets, or neither")
    if any(int(v)!=v or v<1 for v in (epochs,batch_size,patience)):
        raise ValueError("epochs, batch_size and patience must be positive integers")
    if not np.isfinite(learning_rate) or learning_rate<=0 or not np.isfinite(weight_decay) or weight_decay<0:
        raise ValueError("Invalid TelNet learning rate or weight decay")
    if not np.isfinite(min_delta) or min_delta<0 or not np.isfinite(gradient_clip) or gradient_clip<=0:
        raise ValueError("Invalid TelNet stopping tolerance or gradient clipping")
    with torch_runtime(seed,device,num_threads) as resolved:
        network = network.to(resolved)
        training = _tensor_inputs(train_inputs,resolved)
        targets = torch.as_tensor(train_targets,device=resolved,dtype=torch.float32)
        if targets.shape[0]!=training[0].shape[0]:raise ValueError("Training targets do not match inputs")
        validation = None if validation_inputs is None else _tensor_inputs(validation_inputs,resolved)
        val_targets = None if validation_targets is None else torch.as_tensor(validation_targets,device=resolved,dtype=torch.float32)
        if validation is not None and val_targets.shape[0]!=validation[0].shape[0]:
            raise ValueError("Validation targets do not match inputs")
        optimizer = torch.optim.Adam(network.parameters(),lr=float(learning_rate),weight_decay=float(weight_decay))
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        history={"train_loss":[],"val_loss":[],"best_epoch":0,"epochs_run":0,
                 "crps_estimator":"fair" if fair_crps else "empirical_paper_eq22"}
        best_loss=float("inf");best_state=None;bad_epochs=0
        n=training[0].shape[0]
        for epoch in range(int(epochs)):
            network.train();total_num=0.;total_den=0.
            order=torch.randperm(n,generator=generator)
            for start in range(0,n,int(batch_size)):
                idx=order[start:start+int(batch_size)].to(resolved)
                batch=[v.index_select(0,idx) for v in training]
                yy=targets.index_select(0,idx)
                optimizer.zero_grad(set_to_none=True)
                ensemble,_=network(*batch)
                numerator,denominator=_loss_components(ensemble,yy,batch[-1],latitude_weights,fair_crps)
                loss=numerator/denominator
                if not bool(torch.isfinite(loss)):raise FloatingPointError("TelNet training loss is not finite")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(),float(gradient_clip),error_if_nonfinite=True)
                optimizer.step()
                total_num+=float(numerator.detach());total_den+=float(denominator.detach())
            history['train_loss'].append(total_num/total_den)
            history['epochs_run']=epoch+1
            if validation is None:
                history['best_epoch']=epoch+1
                continue
            network.eval();val_num=0.;val_den=0.
            with torch.no_grad():
                for start in range(0,validation[0].shape[0],int(batch_size)):
                    batch=[v[start:start+int(batch_size)] for v in validation]
                    yy=val_targets[start:start+int(batch_size)]
                    ensemble,_=network(*batch)
                    numerator,denominator=_loss_components(ensemble,yy,batch[-1],latitude_weights,fair_crps)
                    val_num+=float(numerator);val_den+=float(denominator)
            value=val_num/val_den;history['val_loss'].append(value)
            if value<best_loss-float(min_delta):
                best_loss=value;best_state=deepcopy(network.state_dict())
                history['best_epoch']=epoch+1;bad_epochs=0
            else:bad_epochs+=1
            if bad_epochs>=int(patience):break
        if best_state is not None:network.load_state_dict(best_state)
        network.eval()
    return network,history
