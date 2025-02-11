"""Linear Algebra."""

from math import sqrt
from typing import NamedTuple

import torch
from torch import Tensor


def sqrtmh(A: Tensor) -> Tensor:
    """Compute the square root of a Symmetric or Hermitian positive definite matrix or batch of matrices.

    Credits to: <https://github.com/pytorch/pytorch/issues/25481#issuecomment-1032789228>.

    Args:
        A (Tensor): Symmetric or Hermitian positive definite matrix or batch of matrices.

    Shape:
        ``A``: #TODO: ADD SHAPE

        Output: #TODO: ADD SHAPE
    """
    L, Q = torch.linalg.eigh(A)
    zero = torch.zeros((), device=L.device, dtype=L.dtype)
    threshold = L.max(-1).values * L.size(-1) * torch.finfo(L.dtype).eps
    L = L.where(L > threshold.unsqueeze(-1), zero)  # zero out small components
    return (Q * L.sqrt().unsqueeze(-2)) @ Q.mH


def covariance(
    X: Tensor,
    Y: Tensor | None = None,
    center: bool = True,
    norm: float | None = None,
) -> Tensor:
    """Computes the covariance of X or cross-covariance between X and Y if Y is given.

    Args:
        X (Tensor): Input features.
        Y (Tensor | None, optional): Output features. Defaults to None.
        center (bool, optional): Whether to compute centered covariances. Defaults to True.
        norm (float | None, optional): Normalization factor. Defaults to None.

    Shape:
        ``X``: :math:`(N, D)`, where :math:`N` is the batch size and :math:`D` is the number of features.

        ``Y``: :math:`(N, D)`, where :math:`N` is the batch size and :math:`D` is the number of features.

        Output: :math:`(D, D)`, where :math:`D` is the number of features.
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


def whitening(u: Tensor, v: Tensor) -> tuple:
    """TODO: Add docs."""
    cov_u = covariance(u)
    cov_v = covariance(v)
    cov_uv = covariance(u, v)

    sqrt_cov_u_inv = torch.linalg.pinv(sqrtmh(cov_u))
    sqrt_cov_v_inv = torch.linalg.pinv(sqrtmh(cov_v))

    M = sqrt_cov_u_inv @ cov_uv @ sqrt_cov_v_inv
    e_val, sing_vec_l = torch.linalg.eigh(M @ M.T)
    e_val, sing_vec_l = filter_reduced_rank_svals(e_val, sing_vec_l)
    sing_val = torch.sqrt(e_val)
    sing_vec_r = (M.T @ sing_vec_l) / sing_val

    return sqrt_cov_u_inv, sqrt_cov_v_inv, sing_val, sing_vec_l, sing_vec_r


####################################################################################################
# TODO: THIS IS JUST COPY AND PASTE FROM OLD NCP
# Should topk and filter_reduced_rank_svals be in utils? They look like linalg to me, specially the
# filter
####################################################################################################


# Sorting and parsing
class TopKReturnType(NamedTuple):  # noqa: D101
    values: torch.Tensor
    indices: torch.Tensor


def topk(vec: torch.Tensor, k: int):  # noqa: D103
    assert vec.ndim == 1, "'vec' must be a 1D array"
    assert k > 0, "k should be greater than 0"
    sort_perm = torch.flip(torch.argsort(vec), dims=[0])  # descending order
    indices = sort_perm[:k]
    values = vec[indices]
    return TopKReturnType(values, indices)


def filter_reduced_rank_svals(values, vectors):  # noqa: D103
    eps = 2 * torch.finfo(torch.get_default_dtype()).eps
    # Filtering procedure.
    # Create a mask which is True when the real part of the eigenvalue is negative or the imaginary part is nonzero
    is_invalid = torch.logical_or(
        torch.real(values) <= eps,
        torch.imag(values) != 0
        if torch.is_complex(values)
        else torch.zeros(len(values), device=values.device),
    )
    # Check if any is invalid take the first occurrence of a True value in the mask and filter everything after that
    if torch.any(is_invalid):
        values = values[~is_invalid].real
        vectors = vectors[:, ~is_invalid]

    sort_perm = topk(values, len(values)).indices
    values = values[sort_perm]
    vectors = vectors[:, sort_perm]

    # Assert that the eigenvectors do not have any imaginary part
    assert torch.all(
        torch.imag(vectors) == 0 if torch.is_complex(values) else torch.ones(len(values))
    ), "The eigenvectors should be real. Decrease the rank or increase the regularization strength."

    # Take the real part of the eigenvectors
    vectors = torch.real(vectors)
    values = torch.real(values)
    return values, vectors
