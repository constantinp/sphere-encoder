# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class SimpleEMA:
    """
    exponential moving average of models weights
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.ema_params = {}
        self.temp_stored_params = {}
        self.decay = decay

        # init EMA params
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.ema_params[name] = param.clone().detach()
            else:
                self.ema_params[name] = param

        for name, buffer in model.named_buffers():
            self.ema_params[name] = buffer

    @torch.no_grad()
    def step(self, model: nn.Module):
        """
        update EMA parameters with current model parameters
        """
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module

        for name, param in model.named_parameters():

            name = name.replace("_checkpoint_wrapped_module.", "")
            if name not in self.ema_params:
                continue

            if param.requires_grad:
                self.ema_params[name].mul_(self.decay).add_(
                    param.data, alpha=1 - self.decay
                )
            else:
                self.ema_params[name].copy_(param.data)

        for name, buffer in model.named_buffers():
            name = name.replace("_checkpoint_wrapped_module.", "")
            self.ema_params[name].copy_(buffer)

    def copy_to(self, model: nn.Module):
        """
        copy current averaged parameters into given model
        """
        for name, param in model.named_parameters():
            _dtype = param.data.dtype
            param.data.copy_(
                self.ema_params[name].data.to(device=param.device, dtype=_dtype)
            )

        for name, param in model.named_buffers():
            _dtype = param.data.dtype
            param.data.copy_(
                self.ema_params[name].data.to(device=param.device, dtype=_dtype)
            )

    def to(self, device=None, dtype=None):
        """
        move internal buffers to specified device
        """
        # .to() on the tensors handles None correctly
        for name, param in self.ema_params.items():
            self.ema_params[name] = (
                self.ema_params[name].to(device=device, dtype=dtype)
                if self.ema_params[name].is_floating_point()
                else self.ema_params[name].to(device=device)
            )

    def store(self, model: nn.Module):
        """
        store current model parameters temporarily
        """
        for name, param in model.named_parameters():
            self.temp_stored_params[name] = param.detach().cpu().clone()

        for name, buffer in model.named_buffers():
            self.temp_stored_params[name] = buffer.detach().cpu().clone()

    def restore(self, model: nn.Module):
        """
        restore parameters stored with the store method
        """
        if self.temp_stored_params is None:
            raise RuntimeError(
                "This ExponentialMovingAverage has no `store()`ed weights to `restore()`"
            )

        for name, param in model.named_parameters():
            assert (
                name in self.temp_stored_params
            ), f"{name} not found in temp_stored_params"
            param.data.copy_(self.temp_stored_params[name].data)

        for name, buffer in model.named_buffers():
            assert (
                name in self.temp_stored_params
            ), f"{name} not found in temp_stored_params"
            buffer.data.copy_(self.temp_stored_params[name].data)

        self.temp_stored_params = {}

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        for name, param in self.ema_params.items():
            if name not in state_dict and not strict:
                continue
            param.data.copy_(state_dict[name].to(param.device).data)

    def state_dict(self):
        return self.ema_params
