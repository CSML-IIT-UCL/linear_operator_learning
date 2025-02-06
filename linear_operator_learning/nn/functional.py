"""Representation learning losses and utility functions."""

import torch

from linear_operator_learning.nn.linalg import covariance, sqrtmh


def vamp_loss(X, Y, schatten_norm: int = 2, center_covariances: bool = True):
    """Variational Approach for learning Markov Processes (VAMP) score by :footcite:t:`Wu2019`.

    Args:
        X (torch.Tensor): Covariates for the initial time steps.
        Y (torch.Tensor): Covariates for the evolved time steps.
        schatten_norm (int, optional): Computes the VAMP-p score with ``p = schatten_norm``. Defaults to 2.
        center_covariances (bool, optional): Use centered covariances to compute the VAMP score. Defaults to True.

    Raises:
        NotImplementedError: If ``schatten_norm`` is not 1 or 2.

    Shape:
        ``X``: :math:`(N, D)`, where :math:`N` is the batch size, and :math:`D` is the number of features.

        ``Y``: :math:`(N, D)`, where :math:`N` is the batch size, and :math:`D` is the number of features.

    Returns:
        torch.Tensor: VAMP score


    """
    cov_X, cov_Y, cov_XY = (
        covariance(X, center=center_covariances),
        covariance(Y, center=center_covariances),
        covariance(X, Y, center=center_covariances),
    )
    if schatten_norm == 2:
        # Using least squares in place of pinv for numerical stability
        M_X = torch.linalg.lstsq(cov_X, cov_XY).solution
        M_Y = torch.linalg.lstsq(cov_Y, cov_XY.T).solution
        return -torch.trace(M_X @ M_Y)
    elif schatten_norm == 1:
        sqrt_cov_X = sqrtmh(cov_X)
        sqrt_cov_Y = sqrtmh(cov_Y)
        M = torch.linalg.multi_dot(
            [
                torch.linalg.pinv(sqrt_cov_X, hermitian=True),
                cov_XY,
                torch.linalg.pinv(sqrt_cov_Y, hermitian=True),
            ]
        )
        return -torch.linalg.matrix_norm(M, "nuc")
    else:
        raise NotImplementedError(f"Schatten norm {schatten_norm} not implemented")


def dp_loss(
    X: torch.Tensor,
    Y: torch.Tensor,
    relaxed: bool = True,
    metric_deformation: float = 1.0,
    center_covariances: bool = True,
):
    """Deep Projection Loss by :footcite:t:`Kostic2023DPNets`.

    Args:
        X (torch.Tensor): Covariates for the initial time steps.
        Y (torch.Tensor): Covariates for the evolved time steps.
        relaxed (bool, optional): Whether to use the relaxed (more numerically stable) or the full deep-projection loss. Defaults to True.
        metric_deformation (float, optional): Strength of the metric metric deformation loss: Defaults to 1.0.
        center_covariances (bool, optional): Use centered covariances to compute the Deep Projection score. Defaults to True.

    Shape:
        ``X``: :math:`(N, D)`, where :math:`N` is the batch size, and :math:`D` is the number of features.

        ``Y``: :math:`(N, D)`, where :math:`N` is the batch size, and :math:`D` is the number of features.

    Returns:
        torch.Tensor: Deep Projection score
    """
    cov_X, cov_Y, cov_XY = (
        covariance(X, center=center_covariances),
        covariance(Y, center=center_covariances),
        covariance(X, Y, center=center_covariances),
    )
    R_X = logfro_loss(cov_X)
    R_Y = logfro_loss(cov_Y)
    if relaxed:
        S = (torch.linalg.matrix_norm(cov_XY, ord="fro") ** 2) / (
            torch.linalg.matrix_norm(cov_X, ord=2) * torch.linalg.matrix_norm(cov_Y, ord=2)
        )
    else:
        M_X = torch.linalg.lstsq(cov_X, cov_XY).solution
        M_Y = torch.linalg.lstsq(cov_Y, cov_XY.T).solution
        S = torch.trace(M_X @ M_Y)
    return -S + 0.5 * metric_deformation * (R_X + R_Y)


def L2_contrastive_loss(X: torch.Tensor, Y: torch.Tensor):
    """NCP/Contrastive/Mutual Information Loss based on the :math:`L^{2}` error.

    Args:
        X (torch.Tensor): Input features
        Y (torch.Tensor): Output features

    Shape:
        ``X``: :math:`(N, D)`, where :math:`N` is the batch size, and :math:`D` is the number of features.

        ``Y``: :math:`(N, D)`, where :math:`N` is the batch size, and :math:`D` is the number of features.

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


def KL_contrastive_score(X: torch.Tensor, Y: torch.Tensor):
    """NCP/Contrastive/Mutual Information Loss based on the KL divergence.

    Args:
        X (torch.Tensor): Input features
        Y (torch.Tensor): Output features

    Shape:
        ``X``: :math:`(N, D)`, where :math:`N` is the batch size, and :math:`D` is the number of features.

        ``Y``: :math:`(N, D)`, where :math:`N` is the batch size, and :math:`D` is the number of features.

    """
    assert X.shape == Y.shape
    assert X.ndim == 2

    npts, dim = X.shape
    log_term = torch.log(torch.mean(X * Y) * dim)
    linear_term = torch.matmul(X, Y.T)
    off_diag = (
        torch.mean(torch.triu(linear_term, diagonal=1) + torch.tril(linear_term, diagonal=-1))
        * npts
        / (npts - 1)
    )
    return off_diag - log_term


def logfro_loss(cov: torch.Tensor):
    r"""Logarithmic + Frobenious (metric deformation) loss as used in :footcite:t:`Kostic2023DPNets`, defined as :math:`\text{Tr}(C^{2} - C -\ln(C))` .

    Args:
        cov (torch.tensor): A symmetric positive-definite matrix.

    Shape:
        ``cov``: :math:`(D, D)`, where :math:`D` is the number of features.

    Returns:
        torch.tensor: Loss function
    """
    eps = torch.finfo(cov.dtype).eps * cov.shape[0]
    vals_x = torch.linalg.eigvalsh(cov)
    vals_x = torch.where(vals_x > eps, vals_x, eps)
    loss = torch.mean(-torch.log(vals_x) + vals_x * (vals_x - 1.0))
    return loss
