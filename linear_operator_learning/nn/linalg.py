"""Linear Algebra."""

from math import sqrt
from typing import Literal

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


def eig(
    fit_result: FitResult,
    cov_XY: Tensor,
) -> EigResult:
    """Computes the eigendecomposition of a regressor.

    Args:
        fit_result (FitResult): Fit result as defined in ``linear_operator_learning.nn.structs``.
        cov_XY (Tensor): Cross covariance matrix between the input and output data.

    Returns:
        EigResult: as defined in ``linear_operator_learning.nn.structs``

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

    Returns:
        Tensor: Evaluated eigenfunctions

    Shape:
        ``eig_results``: ``U, V`` of shape :math:`(D, R)`, ``svals`` of shape :math:`R`
        where :math:`D` is the number of features and  :math:`R` is the rank of the regressor.

        ``X``: :math:`(N_0, D)`, where :math:`N_0` is the number of inputs to predict and :math:`D` is the number of features.

        Output: :math:`(N_0, R)`
    """
    vr_or_vl = eig_result[which]
    return X.to(vr_or_vl.dtype) @ vr_or_vl
