"""Loss functions for representation learning."""

import torch
from torch import Tensor
from torch.nn import Module

from linear_operator_learning.nn import functional as F

__all__ = ["L2ContrastiveLoss", "KLContrastiveLoss"]


class L2ContrastiveLoss(Module):
    r"""NCP/Contrastive/Mutual Information Loss based on the :math:`L^{2}` error by :footcite:t:`Kostic2024NCP`.

    .. math::

        \frac{1}{N(N-1)}\sum_{i \neq j}\langle x_{i}, y_{j} \rangle^2 - \frac{2}{N}\sum_{i=1}\langle x_{i}, y_{i} \rangle.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:  # noqa: D102
        """Forward pass of the L2 contrastive loss.

        Args:
            x (Tensor): Input features.
            y (Tensor): Output features.

        Returns:
            Tensor: Loss value.

        Shape:
            ``x``: :math:`(N, D)`, where :math:`N` is the batch size and :math:`D` is the number of features.

            ``y``: :math:`(N, D)`, where :math:`N` is the batch size and :math:`D` is the number of features.
        """
        return F.l2_contrastive_loss(x, y)


class KLContrastiveLoss(Module):
    r"""NCP/Contrastive/Mutual Information Loss based on the KL divergence.

    .. math::

        \frac{1}{N(N-1)}\sum_{i \neq j}\langle x_{i}, y_{j} \rangle - \frac{2}{N}\sum_{i=1}\log\big(\langle x_{i}, y_{i} \rangle\big).
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:  # noqa: D102
        """Forward pass of the KL contrastive loss.

        Args:
            x (Tensor): Input features.
            y (Tensor): Output features.

        Returns:
            Tensor: Loss value.

        Shape:
            ``x``: :math:`(N, D)`, where :math:`N` is the batch size and :math:`D` is the number of features.

            ``y``: :math:`(N, D)`, where :math:`N` is the batch size and :math:`D` is the number of features.
        """
        return F.kl_contrastive_loss(x, y)
