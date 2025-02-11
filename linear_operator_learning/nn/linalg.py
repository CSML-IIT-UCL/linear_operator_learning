"""Linear Algebra."""

from math import sqrt
from typing import Literal, NamedTuple

import numpy as np
import scipy
import torch
from torch import Tensor

from linear_operator_learning.nn.structs import EigResult, FitResult

__all__ = ["eig", "evaluate_eigenfunction"]


def sqrtmh(A: Tensor) -> Tensor:
    """Compute the square root of a Symmetric or Hermitian positive definite matrix or batch of matrices.

    Credits to: <https://github.com/pytorch/pytorch/issues/25481#issuecomment-1032789228>.

    Args:
        A (Tensor): Symmetric or Hermitian positive definite matrix or batch of matrices.

    Shape:
        ``A``: :math:`(N, N)`

        Output: :math:`(N, N)`
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


def eig(
    fit_result: FitResult,
    cov_XY: Tensor,
) -> EigResult:
    """Computes the eigendecomposition of a regressor.

    Args:
        fit_result (FitResult): Fit result as defined in ``linear_operator_learning.nn.structs``.
        cov_XY (Tensor): Cross covariance matrix between the input and output data.


    Shape:
        ``cov_XY``: :math:`(D, D)`, where :math:`D` is the number of features.

        Output: ``U, V`` of shape :math:`(D, R)`, ``svals`` of shape :math:`R`
        where :math:`D` is the number of features and  :math:`R` is the rank of the regressor.
    """
    dtype_and_device = {
        "dtype": cov_XY.dtype,
        "device": cov_XY.device,
    }
    U = fit_result["U"]
    # Using the trick described in https://arxiv.org/abs/1905.11490
    M = torch.linalg.multi_dot([U.T, cov_XY, U])
    # Convertion to numpy
    M = M.numpy(force=True)
    values, lv, rv = scipy.linalg.eig(M, left=True, right=True)
    r_perm = torch.tensor(np.argsort(values), device=cov_XY.device)
    l_perm = torch.tensor(np.argsort(values.conj()), device=cov_XY.device)
    values = values[r_perm]
    # Back to torch, casting to appropriate dtype and device
    values = torch.complex(
        torch.tensor(values.real, **dtype_and_device), torch.tensor(values.imag, **dtype_and_device)
    )
    lv = torch.complex(
        torch.tensor(lv.real, **dtype_and_device), torch.tensor(lv.imag, **dtype_and_device)
    )
    rv = torch.complex(
        torch.tensor(rv.real, **dtype_and_device), torch.tensor(rv.imag, **dtype_and_device)
    )
    # Normalization in RKHS norm
    rv = U.to(rv.dtype) @ rv
    rv = rv[:, r_perm]
    rv = rv / torch.linalg.norm(rv, axis=0)
    # # Biorthogonalization
    lv = torch.linalg.multi_dot([cov_XY.T.to(lv.dtype), U.to(lv.dtype), lv])
    lv = lv[:, l_perm]
    l_norm = torch.sum(lv * rv, axis=0)
    lv = lv / l_norm
    result: EigResult = EigResult({"values": values, "left": lv, "right": rv})
    return result


def evaluate_eigenfunction(
    eig_result: EigResult,
    which: Literal["left", "right"],
    X: Tensor,
):
    """Evaluates left or right eigenfunctions of a regressor.

    Args:
        eig_result: EigResult object containing eigendecomposition results
        which: String indicating "left" or "right" eigenfunctions.
        X: Feature map of the input data


    Shape:
        ``eig_results``: ``U, V`` of shape :math:`(D, R)`, ``svals`` of shape :math:`R`
        where :math:`D` is the number of features and  :math:`R` is the rank of the regressor.

        ``X``: :math:`(N_0, D)`, where :math:`N_0` is the number of inputs to predict and :math:`D` is the number of features.

        Output: :math:`(N_0, R)`
    """
    vr_or_vl = eig_result[which]
    return X.to(vr_or_vl.dtype) @ vr_or_vl


def whitening(u: Tensor, v: Tensor) -> tuple:
    """Computes whitening matrices for ``u`` and ``v``.

    Args:
        u (Tensor): Input features.
        v (Tensor): Output features.


    Shape:
        ``u``: :math:`(N, D)`, where :math:`N` is the batch size and :math:`D` is the number of features.
        ``v``: :math:`(N, D)`, where :math:`N` is the batch size and :math:`D` is the number of features.
        ``sqrt_cov_u_inv``: :math:`(D, D)`
        ``sqrt_cov_v_inv``: :math:`(D, D)`
        ``sing_val``: :math:`(D,)`
        ``sing_vec_l``: :math:`(D, D)`
        ``sing_vec_r``: :math:`(D, D)`
    """
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
