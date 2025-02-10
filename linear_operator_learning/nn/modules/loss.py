"""Loss functions for representation learning."""

import torch
from torch import Tensor
from torch.nn import Module

from linear_operator_learning.nn import functional as F

__all__ = ["L2ContrastiveLoss"]


class L2ContrastiveLoss(Module):
    """L2 Contrastive Loss. See `F.l2_contrastive_loss` for details."""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:  # noqa: D102
        return F.l2_contrastive_loss(x, y)
