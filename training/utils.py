import torch
from torch import nn
from typing import Optional


def num_parameters(module: nn.Module, requires_grad: Optional[bool] = None) -> int:
    return sum(p.numel() for p in module.parameters() if requires_grad is None or p.requires_grad == requires_grad)


def get_default_supported_precision(training: bool, tpu: bool = False) -> str:
    """Return default precision that is supported by the hardware.

    Args:
        training: `-mixed` or `-true` version of the precision to use
        tpu: whether TPU device is used

    Returns:
        default precision that is suitable for the task and is supported by the hardware
    """
    if tpu:
        return "32-true"
    if not torch.cuda.is_available() or torch.cuda.is_bf16_supported():
        return "bf16-mixed" if training else "bf16-true"
    return "16-mixed" if training else "16-true"