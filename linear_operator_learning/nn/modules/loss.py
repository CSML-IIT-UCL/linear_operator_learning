"""Loss functions for representation learning."""

from typing import Literal

import torch
from torch import Tensor
from torch.nn import Module

from linear_operator_learning.nn import functional as F
from linear_operator_learning.nn.linalg import covariance

__all__ = ["VampLoss", "L2ContrastiveLoss", "KLContrastiveLoss", "DPLoss"]

# Losses_____________________________________________________________________________________________


class _RegularizedLoss(Module):
    """Base class for regularized losses.

    Args:
        gamma (float, optional): Regularization strength.
        regularizer (literal, optional): Regularizer. Either 'orthn_fro' or 'orthn_logfro'.
    """

    def __init__(
        self, gamma: float, regularizer: Literal["orthnornal_fro", "orthnornal_logfro"]
    ) -> None:  # TODO: Automatically determine 'gamma' from dim_x and dim_y
        super().__init__()
        self.gamma = gamma

        if regularizer == "orthonormal_fro":
            self.regularizer = F.orthonormal_fro_reg
        elif regularizer == "orthonormal_logfro":
            self.regularizer = F.orthonormal_logfro_reg
        else:
            raise NotImplementedError(f"Regularizer {regularizer} not supported!")


class VampLoss(_RegularizedLoss):
    r"""Variational Approach for learning Markov Processes (VAMP) score by :footcite:t:`Wu2019`.

    .. math::

        -\sum_{i} \sigma_{i}(A)^{p} \qquad \text{where}~A = \big(x^{\top}x\big)^{\dagger/2}x^{\top}y\big(y^{\top}y\big)^{\dagger/2}.

    Args:
        schatten_norm (int, optional): Computes the VAMP-p score with ``p = schatten_norm``. Defaults to 2.
        center_covariances (bool, optional): Use centered covariances to compute the VAMP score. Defaults to True.
        gamma (float, optional): Regularization strength. Defaults to 1e-3.
        regularizer (literal, optional): Regularizer. Either 'orthn_fro' or 'orthn_logfro'. Defaults to 'orthn_fro'.
    """

    def __init__(
        self,
        schatten_norm: int = 2,
        center_covariances: bool = True,
        gamma: float = 1e-3,
        regularizer: Literal["orthn_fro", "orthn_logfro"] = "orthn_fro",
    ) -> None:
        super().__init__(gamma, regularizer)
        self.schatten_norm = schatten_norm
        self.center_covariances = center_covariances

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Forward pass of VampLoss.

        Args:
            x (Tensor): Features for x.
            y (Tensor): Features for y.

        Raises:
            NotImplementedError: If ``schatten_norm`` is not 1 or 2.

        Shape:
            ``x``: :math:`(N, D)`, where :math:`N` is the batch size and :math:`D` is the number of features.

            ``y``: :math:`(N, D)`, where :math:`N` is the batch size and :math:`D` is the number of features.
        """
        return F.vamp_loss(
            x,
            y,
            self.schatten_norm,
            self.center_covariances,
        ) + self.gamma * (self.regularizer(x) + self.regularizer(y))


class L2ContrastiveLoss(_RegularizedLoss):
    r"""NCP/Contrastive/Mutual Information Loss based on the :math:`L^{2}` error by :footcite:t:`Kostic2024NCP`.

    .. math::

        \frac{1}{N(N-1)}\sum_{i \neq j}\langle x_{i}, y_{j} \rangle^2 - \frac{2}{N}\sum_{i=1}\langle x_{i}, y_{i} \rangle.

    Args:
        gamma (float, optional): Regularization strength. Defaults to 1e-3.
        regularizer (literal, optional): Regularizer. Either 'orthn_fro' or 'orthn_logfro'. Defaults to 'orthn_fro'.
    """

    def __init__(
        self,
        gamma: float = 1e-3,
        regularizer: Literal["orthn_fro", "orthn_logfro"] = "orthn_fro",
    ) -> None:
        super().__init__(gamma, regularizer)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:  # noqa: D102
        """Forward pass of the L2 contrastive loss.

        Args:
            x (Tensor): Input features.
            y (Tensor): Output features.


        Shape:
            ``x``: :math:`(N, D)`, where :math:`N` is the batch size and :math:`D` is the number of features.

            ``y``: :math:`(N, D)`, where :math:`N` is the batch size and :math:`D` is the number of features.
        """
        return F.l2_contrastive_loss(x, y) + self.gamma * (
            self.regularizer(x) + self.regularizer(y)
        )


class DPLoss(_RegularizedLoss):
    r"""Deep Projection Loss by :footcite:t:`Kostic2023DPNets`.

    .. math::


        -\frac{\|x^{\top}y\|^{2}_{{\rm F}}}{\|x^{\top}x\|^{2}\|y^{\top}y\|^{2}}.

    Args:
        relaxed (bool, optional): Whether to use the relaxed (more numerically stable) or the full deep-projection loss. Defaults to True.
        center_covariances (bool, optional): Use centered covariances to compute the Deep Projection loss. Defaults to True.
        gamma (float, optional): Regularization strength. Defaults to 1e-3.
        regularizer (literal, optional): Regularizer. Either 'orthn_fro' or 'orthn_logfro'. Defaults to 'orthn_fro'.
    """

    def __init__(
        self,
        relaxed: bool = True,
        center_covariances: bool = True,
        gamma: float = 1e-3,
        regularizer: Literal["orthn_fro", "orthn_logfro"] = "orthn_fro",
    ) -> None:
        super().__init__(gamma, regularizer)
        self.relaxed = relaxed
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
        return F.dp_loss(
            x,
            y,
            self.relaxed,
            self.center_covariances,
        ) + self.gamma * (self.regularizer(x) + self.regularizer(y))


class KLContrastiveLoss(_RegularizedLoss):
    r"""NCP/Contrastive/Mutual Information Loss based on the KL divergence.

    .. math::

        \frac{1}{N(N-1)}\sum_{i \neq j}\langle x_{i}, y_{j} \rangle - \frac{2}{N}\sum_{i=1}\log\big(\langle x_{i}, y_{i} \rangle\big).

    Args:
        gamma (float, optional): Regularization strength. Defaults to 1e-3.
        regularizer (literal, optional): Regularizer. Either 'orthn_fro' or 'orthn_logfro'. Defaults to 'orthn_fro'.
    """

    def __init__(
        self,
        gamma: float = 1e-3,
        regularizer: Literal["orthn_fro", "orthn_logfro"] = "orthn_fro",
    ) -> None:
        super().__init__(gamma, regularizer)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:  # noqa: D102
        """Forward pass of the KL contrastive loss.

        Args:
            x (Tensor): Input features.
            y (Tensor): Output features.


        Shape:
            ``x``: :math:`(N, D)`, where :math:`N` is the batch size and :math:`D` is the number of features.

            ``y``: :math:`(N, D)`, where :math:`N` is the batch size and :math:`D` is the number of features.
        """
        return F.kl_contrastive_loss(x, y) + self.gamma * (
            self.regularizer(x) + self.regularizer(y)
        )
