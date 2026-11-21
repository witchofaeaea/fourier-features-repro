"""Coordinate-based MLP used for the 2D image task (paper Appendix E.1):
4 linear layers, 256 channels, ReLU, sigmoid output."""
import torch
import torch.nn as nn

from .mappings import InputMapping


class FourierFeatureMLP(nn.Module):
    def __init__(self, mapping: InputMapping, hidden: int = 256,
                 num_layers: int = 4, out_dim: int = 3):
        super().__init__()
        self.mapping = mapping
        layers = []
        d = mapping.out_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers += [nn.Linear(d, out_dim), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return self.net(self.mapping(v))
