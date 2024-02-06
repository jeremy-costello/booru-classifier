import torch
import torch.nn as nn
from types import MethodType
from lightning.fabric.loggers import CSVLogger
from typing import Any, Optional, Type, TypeVar


def num_parameters(module: nn.Module, requires_grad: Optional[bool] = None) -> int:
    return sum(p.numel() for p in module.parameters() if requires_grad is None or p.requires_grad == requires_grad)


T = TypeVar("T")


def step_csv_logger(*args: Any, cls: Type[T] = CSVLogger, **kwargs: Any) -> T:
    logger = cls(*args, **kwargs)

    def merge_by(dicts, key):
        from collections import defaultdict

        out = defaultdict(dict)
        for d in dicts:
            if key in d:
                out[d[key]].update(d)
        return [v for _, v in sorted(out.items())]

    def save(self) -> None:
        """Overridden to merge CSV by the step number."""
        import csv

        if not self.metrics:
            return
        metrics = merge_by(self.metrics, "step")
        keys = sorted({k for m in metrics for k in m})
        with self._fs.open(self.metrics_file_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(metrics)

    logger.experiment.save = MethodType(save, logger.experiment)

    return logger


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