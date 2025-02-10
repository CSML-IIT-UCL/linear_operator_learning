"""Loss functions for representation learning."""

import torch
from torch import Tensor
from torch.nn import Module

from linear_operator_learning.nn import functional as F


class L2ContrastiveLoss(Module):
    """L2 Contrastive Loss. See `F.l2_contrastive_loss` for details."""

    def __init__(self, gamma: float = 1e-3) -> None:
        # TODO: Automatically determine reasonable values of gamma with the dimension, like Dani did
        # on his implementation of NCP.
        super().__init__()
        self.gamma = gamma

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:  # noqa: D102
        return F.l2_contrastive_loss(x, y) + self.gamma * F.orthm_regularization(x, y)
