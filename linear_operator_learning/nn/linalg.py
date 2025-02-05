"""Linear Algebra."""

from math import sqrt

import torch


def sqrtmh(A: torch.Tensor):
    # Credits to
    """Compute the square root of a Symmetric or Hermitian positive definite matrix or batch of matrices. Credits to  `https://github.com/pytorch/pytorch/issues/25481#issuecomment-1032789228 <https://github.com/pytorch/pytorch/issues/25481#issuecomment-1032789228>`_."""
    L, Q = torch.linalg.eigh(A)
    zero = torch.zeros((), device=L.device, dtype=L.dtype)
    threshold = L.max(-1).values * L.size(-1) * torch.finfo(L.dtype).eps
    L = L.where(L > threshold.unsqueeze(-1), zero)  # zero out small components
    return (Q * L.sqrt().unsqueeze(-2)) @ Q.mH


def covariance(
    X: torch.Tensor,
    Y: torch.Tensor | None = None,
    center: bool = True,
    norm: float | None = None,
):
    """Covariance matrix.

    Args:
        X (torch.Tensor): Input covariates of shape ``(samples, features)``.
        Y (torch.Tensor | None, optional): Output covariates of shape ``(samples, features)`` Defaults to None.
        center (bool, optional): Whether to compute centered covariances. Defaults to True.
        norm (float | None, optional): Normalization factor. Defaults to None.

    Returns:
        torch.Tensor: Covariance matrix of shape ``(features, features)``. If ``Y is not None`` computes the cross-covariance between X and Y.
    """
    assert X.ndim == 2
    if norm is None:
        norm = sqrt(X.shape[0])
    else:
        assert norm > 0
        norm = sqrt(norm)
    if Y is None:
        X = X / norm
        if center:
            X = X - X.mean(dim=0, keepdim=True)
        return torch.mm(X.T, X)
    else:
        assert Y.ndim == 2
        X = X / norm
        Y = Y / norm
        if center:
            X = X - X.mean(dim=0, keepdim=True)
            Y = Y - Y.mean(dim=0, keepdim=True)
        return torch.mm(X.T, Y)
