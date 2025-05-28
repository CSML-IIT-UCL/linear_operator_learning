"""Simplicial normalizarion."""

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SimNorm(nn.Module):
    """Simplicial normalization from :footcite:t:`lavoie2022simplicial`.

    Simplicial normalization splits the input into chunks of dimension :code:`dim`, applies a softmax transformation to each of the chunks separately, and concatenates them back together.

    Args:
        dim (int): Dimension of the simplicial groups.
    """

    def __init__(self, dim: int = 8):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the simplicial normalization module."""
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)

    def __repr__(self):
        """String representation of the simplicial norm module."""
        return f"SimNorm(dim={self.dim})"
