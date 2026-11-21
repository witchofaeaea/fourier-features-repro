import math

import torch


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean((pred - target) ** 2).item()


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """PSNR in dB for signals in [0, 1]."""
    m = mse(pred, target)
    return float("inf") if m == 0 else -10.0 * math.log10(m)
