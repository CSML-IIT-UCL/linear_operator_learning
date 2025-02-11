"""Loss functions for representation learning."""

import torch
from torch import Tensor
from torch.nn import Module

from linear_operator_learning.nn import functional as F

__all__ = ["L2ContrastiveLoss"]


class VampLoss(Module):
    r"""Variational Approach for learning Markov Processes (VAMP) score by :footcite:t:`Wu2019`.

    .. math::

        -\sum_{i} \sigma_{i}(A)^{p} \qquad \text{where}~A = \big(x^{\top}x\big)^{\dagger/2}x^{\top}y\big(y^{\top}y\big)^{\dagger/2}.

    Args:
        schatten_norm (int, optional): Computes the VAMP-p score with ``p = schatten_norm``. Defaults to 2.
        center_covariances (bool, optional): Use centered covariances to compute the VAMP score. Defaults to True.
    """

    def __init__(self, schatten_norm: int = 2, center_covariances: bool = True) -> None:
        super().__init__()
        self.schatten_norm = 2
        self.center_covariances = center_covariances

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Forward pass of VampLoss.

        x (Tensor): Features for x.
        y (Tensor): Features for y.

        Raises:
            NotImplementedError: If ``schatten_norm`` is not 1 or 2.

        Shape:
            ``x``: :math:`(N, D)`, where :math:`N` is the batch size and :math:`D` is the number of features.

            ``y``: :math:`(N, D)`, where :math:`N` is the batch size and :math:`D` is the number of features.
        """


class L2ContrastiveLoss(Module):
    """L2 Contrastive Loss. See `F.l2_contrastive_loss` for details."""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:  # noqa: D102
        return F.l2_contrastive_loss(x, y)


class DPLoss(Module):
    r"""Deep Projection Loss by :footcite:t:`Kostic2023DPNets`.

    .. math::


        -\frac{\|x^{\top}y\|^{2}_{{\rm F}}}{\|x^{\top}x\|^{2}\|y^{\top}y\|^{2}}.

    Args:
        relaxed (bool, optional): Whether to use the relaxed (more numerically stable) or the full deep-projection loss. Defaults to True.
        metric_deformation (float, optional): Strength of the metric metric deformation loss: Defaults to 1.0.
        center_covariances (bool, optional): Use centered covariances to compute the Deep Projection loss. Defaults to True.
    """

    def __init__(
        self,
        relaxed: bool = True,
        metric_deformation: float = 1.0,
        center_covariances: bool = True,
    ) -> None:
        super().__init__()
        self.relaxed = relaxed
        self.metric_deformation = metric_deformation
        self.center_covariances = center_covariances

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Forward pass of DPLoss.

        Args:
        x (Tensor): Features for x.
        y (Tensor): Features for y.

        Shape:
        ``x``: :math:`(N, D)`, where :math:`N` is the batch size and :math:`D` is the number of features.

        ``y``: :math:`(N, D)`, where :math:`N` is the batch size and :math:`D` is the number of features.
        """
        return F.dp_loss(x, y, self.relaxed, self.metric_deformation, self.center_covariances)


class LogFroLoss(Module):
    r"""Logarithmic + Frobenious (metric deformation) loss by :footcite:t:`Kostic2023DPNets`.

    Defined as :math:`\text{Tr}(C^{2} - C -\ln(C))`.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, cov: Tensor) -> Tensor:
        """Forward pass of LogFroLoss.

        Args:
            cov (Tensor): A symmetric positive-definite matrix.

        Shape:
            ``cov``: :math:`(D, D)`, where :math:`D` is the number of features.
        """
        F.logfro_loss(cov)
