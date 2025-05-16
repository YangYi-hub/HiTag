from __future__ import annotations

import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

class LinearWarmupDecayLR(LambdaLR):

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int = 500000,
        warmup_steps: int = 4000,
        last_epoch: int = -1,
    ):
        assert warmup_steps < total_steps, "Warmup steps should be less than total steps."
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

        super().__init__(optimizer, self._lr_multiplier, last_epoch)

    def _lr_multiplier(self, step: int) -> float:
        if step < self.warmup_steps:
            multiplier = step / float(self.warmup_steps)
        else:
            decay_steps = self.total_steps - self.warmup_steps
            if decay_steps > 0:
                multiplier = (self.total_steps - step) / float(decay_steps)
            else:
                multiplier = 0.0

        return max(0, multiplier)


class PowerWarmupCosineDecayLR(LambdaLR):

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int = 500000,
        warmup_steps: int = 4000,
        q: float = 4.0,
        last_epoch: int = -1,
    ):
        assert warmup_steps < total_steps, (
            "Warmup steps should be less than total steps."
        )
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.q = q

        super().__init__(optimizer, self._lr_multiplier, last_epoch)

    def _lr_multiplier(self, step: int) -> float:

        if step < self.warmup_steps:
            multiplier = step / float(self.warmup_steps)
        else:
            cos_factor = (step - self.warmup_steps) / float(self.total_steps - self.warmup_steps)
            multiplier = math.cos(cos_factor * (math.pi / 2)) ** self.q

        return max(0, multiplier)

class LinearWarmupCosineDecayLR(LambdaLR):

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int,
        last_epoch: int = -1,
    ):
        assert (
            warmup_steps < total_steps
        ), "Warmup steps should be less than total steps."

        self.tsteps = total_steps#500000
        self.wsteps = warmup_steps#4000
        super().__init__(optimizer, self._lr_multiplier, last_epoch)

    def _lr_multiplier(self, step: int) -> float:
        if step < self.wsteps:
            multiplier = step / float(max(1, self.wsteps))
        else:
            cos_factor = (step - self.wsteps) / (self.tsteps - self.wsteps)
            multiplier = math.cos(cos_factor * (math.pi / 2)) ** 2
        return max(0, multiplier)


def set_weight_decay_per_param(
    model: torch.nn.Module,
    weight_decay: float,
    gain_bias_decay: float | None = None,
    exclude_params: list[str] = [],
) -> list[dict]:
    norm_classes = (
        torch.nn.modules.batchnorm._BatchNorm,
        torch.nn.LayerNorm,
        torch.nn.GroupNorm,
        torch.nn.modules.instancenorm._InstanceNorm,
        torch.nn.LocalResponseNorm,
    )

    gain_bias_decay = gain_bias_decay or weight_decay
    params = {"regular": [], "gain_bias": [], "excluded": []}
    params_weight_decay = {
        "regular": weight_decay,
        "gain_bias": gain_bias_decay,
        "excluded": 0.0,
    }
    already_added_parameters = set()

    def _add_params(module, prefix=""):
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad or p in already_added_parameters:
                continue

            already_added_parameters.add(p)

            if any([exclude_name in name for exclude_name in exclude_params]):
                params["excluded"].append(p)
            elif isinstance(module, norm_classes) or "bias" in name:

                params["gain_bias"].append(p)
            else:
                params["regular"].append(p)

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
            _add_params(child_module, prefix=child_prefix)

    _add_params(model)

    param_groups = []
    for key in params:
        if len(params[key]) > 0:
            param_groups.append(
                {"params": params[key], "weight_decay": params_weight_decay[key]}
            )
    return param_groups
