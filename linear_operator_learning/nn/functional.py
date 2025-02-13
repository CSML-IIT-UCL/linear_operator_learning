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
    center_covariances: bool = True,
) -> Tensor:
    """See :class:`linear_operator_learning.nn.DPLoss` for details."""
    cov_x, cov_y, cov_xy = (
        covariance(x, center=center_covariances),
        covariance(y, center=center_covariances),
        covariance(x, y, center=center_covariances),
    )
    if relaxed:
        S = (torch.linalg.matrix_norm(cov_xy, ord="fro") ** 2) / (
            torch.linalg.matrix_norm(cov_x, ord=2) * torch.linalg.matrix_norm(cov_y, ord=2)
        )
    else:
        M_x = torch.linalg.lstsq(cov_x, cov_xy).solution
        M_y = torch.linalg.lstsq(cov_y, cov_xy.T).solution
        S = torch.trace(M_x @ M_y)
    return -S


def l2_contrastive_loss(x: Tensor, y: Tensor) -> Tensor:
    """See :class:`linear_operator_learning.nn.L2ContrastiveLoss` for details."""
    assert x.shape == y.shape
    assert x.ndim == 2

    npts, dim = x.shape
    diag = 2 * torch.mean(x * y) * dim
    square_term = torch.matmul(x, y.T) ** 2
    off_diag = (
        torch.mean(torch.triu(square_term, diagonal=1) + torch.tril(square_term, diagonal=-1))
        * npts
        / (npts - 1)
    )
    return off_diag - diag


def kl_contrastive_loss(X: Tensor, Y: Tensor) -> Tensor:
    """See :class:`linear_operator_learning.nn.KLContrastiveLoss` for details."""
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


# Regularizers______________________________________________________________________________________


def orthn_fro_reg(x: Tensor) -> Tensor:
    r"""Orthonormality regularization with Frobenious norm of covariance of x.

    .. math::

       \frac{1}{D}\lVert C_X - I\rVert_F^2 =  \frac{1}{D}\text{Tr}((C_X-I)^2).

    Args:
        x (Tensor): Input features.

    Shape:
        ``x``: :math:`(N, D)`, where :math:`N` is the batch size and :math:`D` is the number of features.
    """
    cov = covariance(x)  # shape: (D, D)
    eps = torch.finfo(cov.dtype).eps * cov.shape[0]
    vals_x = torch.linalg.eigvalsh(cov)
    vals_x = torch.where(vals_x > eps, vals_x, eps)
    reg = torch.mean((vals_x - 1.0) ** 2)
    return reg


def orthn_logfro_reg(x: Tensor) -> Tensor:
    r"""Orthonormality regularization with log-Frobenious norm of covariance of x by :footcite:t:`Kostic2023DPNets`.

    .. math::

        \frac{1}{D}\text{Tr}(C_X^{2} - C_X -\ln(C_X)).

    Args:
        x (Tensor): Input features.

    Shape:
        ``x``: :math:`(N, D)`, where :math:`N` is the batch size and :math:`D` is the number of features.
    """
    cov = covariance(x)  # shape: (D, D)
    eps = torch.finfo(cov.dtype).eps * cov.shape[0]
    vals_x = torch.linalg.eigvalsh(cov)
    vals_x = torch.where(vals_x > eps, vals_x, eps)
    reg = torch.mean(-torch.log(vals_x) + vals_x * (vals_x - 1.0))
    return reg
