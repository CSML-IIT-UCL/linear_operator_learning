"""Linear algebra utilities for the `kernel` algorithms."""

from typing import Literal
from warnings import warn

import numpy as np
import scipy.linalg
from numpy import ndarray

from linear_operator_learning.kernel.structs import EigResult, FitResult
from linear_operator_learning.kernel.utils import sanitize_complex_conjugates, topk

__all__ = ["eig", "evaluate_eigenfunction"]


def eig(
    fit_result: FitResult,
    K_X: ndarray,  # Kernel matrix of the input data
    K_YX: ndarray,  # Kernel matrix between the output data and the input data
) -> EigResult:
    """Computes the eigendecomposition of the transfer operator.

    Args:
        fit_result (FitResult): Fit result as defined in ``operator_learning.structs``.
        K_X (ndarray): Kernel matrix of the input data.
        K_YX (ndarray): Kernel matrix between the output data and the input data.

    Returns:
        EigResult: as defined in ``operator_learning.structs``

    Shape:
        ``K_X``: :math:`(N, N)`, where :math:`N` is the sample size.

        ``K_YX``: :math:`(N, N)`, where :math:`N` is the sample size.

        Output: ``U, V`` of shape :math:`(N, R)`, ``svals`` of shape :math:`R`
        where :math:`N` is the sample size and  :math:`R` is the rank of the regressor.
    """
    # SUV.TZ -> V.T K_YX U (right ev = SUvr, left ev = ZVvl)
    U = fit_result["U"]
    V = fit_result["V"]
    r_dim = (K_X.shape[0]) ** (-1)

    W_YX = np.linalg.multi_dot([V.T, r_dim * K_YX, U])
    W_X = np.linalg.multi_dot([U.T, r_dim * K_X, U])

    values, vl, vr = scipy.linalg.eig(W_YX, left=True, right=True)  # Left -> V, Right -> U
    values = sanitize_complex_conjugates(values)
    r_perm = np.argsort(values)
    vr = vr[:, r_perm]
    l_perm = np.argsort(values.conj())
    vl = vl[:, l_perm]
    values = values[r_perm]

    rcond = 1000.0 * np.finfo(U.dtype).eps
    # Normalization in RKHS
    norm_r = weighted_norm(vr, W_X)
    norm_r = np.where(norm_r < rcond, np.inf, norm_r)
    vr = vr / norm_r

    # Bi-orthogonality of left eigenfunctions
    norm_l = np.diag(np.linalg.multi_dot([vl.T, W_YX, vr]))
    norm_l = np.where(np.abs(norm_l) < rcond, np.inf, norm_l)
    vl = vl / norm_l
    result: EigResult = {"values": values, "left": V @ vl, "right": U @ vr}
    return result


def evaluate_eigenfunction(
    eig_result: EigResult,
    which: Literal["left", "right"],
    K_Xin_X_or_Y: ndarray,
):
    """Evaluates left or right eigenfunctions using kernel matrices.

    Args:
        eig_result: EigResult object containing eigendecomposition results
        which: String indicating "left" or "right" eigenfunctions
        K_Xin_X_or_Y: Kernel matrix between initial conditions and input data (for right
            eigenfunctions) or output data (for left eigenfunctions)

    Returns:
        ndarray: Evaluated eigenfunctions

    Shape:
        ``eig_results``: ``U, V`` of shape :math:`(N, R)`, ``svals`` of shape :math:`R`
        where :math:`N` is the sample size and  :math:`R` is the rank of the regressor.

        ``K_Xin_X_or_Y``: :math:`(N_0, N)`, where :math:`N_0` is the number of inputs to
        predict and :math:`N` is the sample size.

        Output:
    """
    vr_or_vl = eig_result[which]
    rsqrt_dim = (K_Xin_X_or_Y.shape[1]) ** (-0.5)
    return np.linalg.multi_dot([rsqrt_dim * K_Xin_X_or_Y, vr_or_vl])


def add_diagonal_(M: ndarray, alpha: float):
    """Add alpha to the diagonal of M inplace.

    Args:
        M (ndarray): The matrix to modify inplace.
        alpha (float): The value to add to the diagonal of M.
    """
    np.fill_diagonal(M, M.diagonal() + alpha)


def stable_topk(
    vec: ndarray,
    k_max: int,
    rcond: float | None = None,
    ignore_warnings: bool = True,
):
    """Takes up to k_max indices of the top k_max values of vec. If the values are below rcond, they are discarded.

    Args:
        vec (ndarray): Vector to extract the top k indices from.
        k_max (int): Number of indices to extract.
        rcond (float, optional): Value below which the values are discarded. Defaults to None, in which case it is set according to the machine precision of vec's dtype.
        ignore_warnings (bool): If False, raise a warning when some elements are discarted for being below the requested numerical precision.

    Returns:
        tuple[ndarray, ndarray]: top values and their indices.
    """
    if rcond is None:
        rcond = 10.0 * vec.shape[0] * np.finfo(vec.dtype).eps

    top_vec, top_idxs = topk(vec, k_max)

    if all(top_vec > rcond):
        return top_vec, top_idxs
    else:
        valid = top_vec > rcond
        # In the case of multiple occurrences of the maximum vec, the indices corresponding to the first occurrence are returned.
        first_invalid = np.argmax(np.logical_not(valid))
        _first_discarded_val = np.max(np.abs(vec[first_invalid:]))

        if not ignore_warnings:
            warn(
                f"Warning: Discarted {k_max - vec.shape[0]} dimensions of the {k_max} requested due to numerical instability. Consider decreasing the k. The largest discarded value is: {_first_discarded_val:.3e}."
            )
        return top_vec[valid], top_idxs[valid]


def weighted_norm(A: ndarray, M: ndarray | None = None):
    r"""Weighted norm of the columns of A.

    Args:
        A (ndarray): 1D or 2D array. If 2D, the columns are treated as vectors.
        M (ndarray or LinearOperator, optional): Weigthing matrix. the norm of the vector :math:`a` is given by :math:`\langle a, Ma \rangle` . Defaults to None, corresponding to the Identity matrix. Warning: no checks are
        performed on M being a PSD operator.

    Returns:
        (ndarray or float): If ``A.ndim == 2`` returns 1D array of floats corresponding to the norms of
        the columns of A. Else return a float.
    """
    assert A.ndim <= 2, "'A' must be a vector or a 2D array"
    if M is None:
        norm = np.linalg.norm(A, axis=0)
    else:
        _A = np.dot(M, A)
        _A_T = np.dot(M.T, A)
        norm = np.real(np.sum(0.5 * (np.conj(A) * _A + np.conj(A) * _A_T), axis=0))
    rcond = 10.0 * A.shape[0] * np.finfo(A.dtype).eps
    norm = np.where(norm < rcond, 0.0, norm)
    return np.sqrt(norm)
