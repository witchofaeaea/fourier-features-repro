"""Input mappings gamma(v) from Tancik et al. (2020), Section 6.1.

All mappings take coordinates v in [0, 1)^d with shape (N, d) and return
features with shape (N, out_dim). The frequency matrix B is a fixed buffer,
it is not trained.
"""
import math

import torch
import torch.nn as nn

MAPPINGS = ("none", "basic", "positional", "gaussian")


def build_b(kind: str, in_dim: int = 2, num_freqs: int = 256,
            sigma: float = 10.0, seed: int = 0):
    """Return the frequency matrix B (shape: num_rows x in_dim) or None.

    none        -> None, gamma(v) = v
    basic       -> identity, gamma(v) = [cos(2*pi*v), sin(2*pi*v)]
    positional  -> axis-aligned, log-spaced frequencies 2^(sigma * X), X in linspace(0, 1)
                   (paper Table 2), m = num_freqs // in_dim per input dimension
    gaussian    -> B_ij ~ N(0, sigma^2), num_freqs rows
    """
    if kind == "none":
        return None
    if kind == "basic":
        return torch.eye(in_dim)
    if kind == "positional":
        m = num_freqs // in_dim
        # Paper Table 2: b_j = 2^(sigma_p * X), X in linspace(0, 1) -> frequencies 1 ... 2^sigma_p
        freqs = 2.0 ** (sigma * torch.linspace(0, 1, m))  # (m,)
        rows = [freqs[:, None] * torch.eye(in_dim)[d][None, :] for d in range(in_dim)]
        return torch.cat(rows, dim=0)  # (m * in_dim, in_dim)
    if kind == "gaussian":
        g = torch.Generator().manual_seed(seed)
        return torch.randn(num_freqs, in_dim, generator=g) * sigma
    raise ValueError(f"unknown mapping '{kind}', choose from {MAPPINGS}")


class InputMapping(nn.Module):
    """gamma(v) = [cos(2*pi*B v), sin(2*pi*B v)]  (or v itself for 'none')."""

    def __init__(self, kind: str = "gaussian", in_dim: int = 2,
                 num_freqs: int = 256, sigma: float = 10.0, seed: int = 0):
        super().__init__()
        self.kind = kind
        self.in_dim = in_dim
        B = build_b(kind, in_dim, num_freqs, sigma, seed)
        if B is None:
            self.B = None
            self.out_dim = in_dim
        else:
            self.register_buffer("B", B)
            self.out_dim = 2 * B.shape[0]

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        if self.B is None:
            return v
        proj = 2 * math.pi * v @ self.B.T  # (N, num_rows)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
