"""Functional interface."""

import torch
from torch import Tensor

from linear_operator_learning.nn.linalg import covariance, sqrtmh

# Losses_____________________________________________________________________________________________


def vamp_loss(
    x: Tensor, y: Tensor, schatten_norm: int = 2, center_covariances: bool = True
) -> Tensor:
    """See :class:`linear_operator_learning.nn.VampLoss` for details."""
    cov_x, cov_y, cov_xy = (
        covariance(x, center=center_covariances),
        covariance(y, center=center_covariances),
        covariance(x, y, center=center_covariances),
    )
    if schatten_norm == 2:
        # Using least squares in place of pinv for numerical stability
        M_x = torch.linalg.lstsq(cov_x, cov_xy).solution
        M_y = torch.linalg.lstsq(cov_y, cov_xy.T).solution
        return -torch.trace(M_x @ M_y)
    elif schatten_norm == 1:
        sqrt_cov_x = sqrtmh(cov_x)
        sqrt_cov_y = sqrtmh(cov_y)
        M = torch.linalg.multi_dot(
            [
                torch.linalg.pinv(sqrt_cov_x, hermitian=True),
                cov_xy,
                torch.linalg.pinv(sqrt_cov_y, hermitian=True),
            ]
        )
        return -torch.linalg.matrix_norm(M, "nuc")
    else:
        raise NotImplementedError(f"Schatten norm {schatten_norm} not implemented")


def dp_loss(
    x: Tensor,
    y: Tensor,
    relaxed: bool = True,
    metric_deformation: float = 1.0,
    center_covariances: bool = True,
) -> Tensor:
    """See :class:`linear_operator_learning.nn.DPLoss` for details."""
    cov_x, cov_y, cov_xy = (
        covariance(x, center=center_covariances),
        covariance(y, center=center_covariances),
        covariance(x, y, center=center_covariances),
    )
    R_x = logfro_loss(cov_x)
    R_y = logfro_loss(cov_y)
    if relaxed:
        S = (torch.linalg.matrix_norm(cov_xy, ord="fro") ** 2) / (
            torch.linalg.matrix_norm(cov_x, ord=2) * torch.linalg.matrix_norm(cov_y, ord=2)
        )
    else:
        M_x = torch.linalg.lstsq(cov_x, cov_xy).solution
        M_y = torch.linalg.lstsq(cov_y, cov_xy.T).solution
        S = torch.trace(M_x @ M_y)
    return -S + 0.5 * metric_deformation * (R_x + R_y)


def l2_contrastive_loss(X: Tensor, Y: Tensor) -> Tensor:
    r"""NCP/Contrastive/Mutual Information Loss based on the :math:`L^{2}` error by :footcite:t:`Kostic2024NCP`.

    .. math::

        \frac{1}{N(N-1)}\sum_{i \neq j}\langle Y_{i}, X_{j} \rangle^2 - \frac{2}{N}\sum_{i=1}\langle Y_{i}, X_{i} \rangle.

    Args:
        X (Tensor): Input features.
        Y (Tensor): Output features.

    Shape:
        ``X``: :math:`(N, D)`, where :math:`N` is the batch size and :math:`D` is the number of features.

        ``Y``: :math:`(N, D)`, where :math:`N` is the batch size and :math:`D` is the number of features.
    """
    assert X.shape == Y.shape
    assert X.ndim == 2

    npts, dim = X.shape
    diag = 2 * torch.mean(X * Y) * dim
    square_term = torch.matmul(X, Y.T) ** 2
    off_diag = (
        torch.mean(torch.triu(square_term, diagonal=1) + torch.tril(square_term, diagonal=-1))
        * npts
        / (npts - 1)
    )
    return off_diag - diag


def kl_contrastive_loss(X: Tensor, Y: Tensor) -> Tensor:
    r"""NCP/Contrastive/Mutual Information Loss based on the KL divergence.

    .. math::

        \frac{1}{N(N-1)}\sum_{i \neq j}\langle Y_{i}, X_{j} \rangle - \frac{2}{N}\sum_{i=1}\log\big(\langle Y_{i}, X_{i} \rangle\big).

    Args:
        X (Tensor): Input features.
        Y (Tensor): Output features.

    Shape:
        ``X``: :math:`(N, D)`, where :math:`N` is the batch size and :math:`D` is the number of features.

        ``Y``: :math:`(N, D)`, where :math:`N` is the batch size and :math:`D` is the number of features.
    """
    assert X.shape == Y.shape
    assert X.ndim == 2

    npts, dim = X.shape
    log_term = torch.mean(torch.log(X * Y)) * dim
    linear_term = torch.matmul(X, Y.T)
    off_diag = (
        torch.mean(torch.triu(linear_term, diagonal=1) + torch.tril(linear_term, diagonal=-1))
        * npts
        / (npts - 1)
    )
    return off_diag - log_term


def logfro_loss(cov: Tensor) -> Tensor:
    """See :class:`linear_operator_learning.nn.LogFroLoss` for details."""
    eps = torch.finfo(cov.dtype).eps * cov.shape[0]
    vals_x = torch.linalg.eigvalsh(cov)
    vals_x = torch.where(vals_x > eps, vals_x, eps)
    loss = torch.mean(-torch.log(vals_x) + vals_x * (vals_x - 1.0))
    return loss
