"""Generic Utilities."""

from math import sqrt

import numpy as np
from numpy import ndarray
from scipy.spatial.distance import pdist

__all__ = [
    "return_phi_dphi",
    "return_dphi_dphi",
]


def topk(vec: ndarray, k: int):
    """Get the top k values from a Numpy array.

    Args:
        vec (ndarray): A 1D numpy array
        k (int): Number of elements to keep

    Returns:
        values indices: Top k values and their indices
    """
    assert np.ndim(vec) == 1, "'vec' must be a 1D array"
    assert k > 0, "k should be greater than 0"
    sort_perm = np.flip(np.argsort(vec))  # descending order
    indices = sort_perm[:k]
    values = vec[indices]
    return values, indices


def sanitize_complex_conjugates(vec: ndarray, tol: float = 10.0):
    """On 1D complex vector, if the real parts of two elements are close, set them equal. Set to 0 the imaginary parts smaller than `tol` times the machine precision.

    Args:
        vec (ndarray): A 1D vector to sanitize.
        tol (float, optional): Tolerance for comparisons. Defaults to 10.0.

    """
    assert issubclass(vec.dtype.type, np.complexfloating), "The input element should be complex"
    assert vec.ndim == 1
    rcond = tol * np.finfo(vec.dtype).eps
    pdist_real_part = pdist(vec.real[:, None])
    # Set the same element whenever pdist is smaller than eps*tol
    condensed_idxs = np.argwhere(pdist_real_part < rcond)[:, 0]
    fuzzy_real = vec.real.copy()
    if condensed_idxs.shape[0] >= 1:
        for idx in condensed_idxs:
            i, j = _row_col_from_condensed_index(vec.real.shape[0], idx)
            avg = 0.5 * (fuzzy_real[i] + fuzzy_real[j])
            fuzzy_real[i] = avg
            fuzzy_real[j] = avg
    fuzzy_imag = vec.imag.copy()
    fuzzy_imag[np.abs(fuzzy_imag) < rcond] = 0.0
    return fuzzy_real + 1j * fuzzy_imag


def _row_col_from_condensed_index(d, index):
    # Credits to: https://stackoverflow.com/a/14839010
    b = 1 - (2 * d)
    i = (-b - sqrt(b**2 - 8 * index)) // 2
    j = index + i * (b + i + 2) // 2 + 1
    return (int(i), int(j))


def return_phi_dphi(
    kernel_X: np.ndarray,
    X: np.ndarray,
    sigma: float,
    friction: np.ndarray,
):
    r"""Returns the matrix :math:`N_{i,(k-1)n+j}=<\phi(x_i),d_k\phi(x_j)>` (matrix :math:`N` in the paper) only for
    a GAUSSIAN kernel
    where :math:`i = 1,\dots n, k=1, \dots d, j=1,\dots n`
    and :math:`N` is the number of training points and :math:`d` is the dimensionality of the system.

    Args:
        kernel_X (np.ndarray): kernel matrix of the training data
        X (np.ndarray): training data
        sigma (float): length scale of the GAUSSIAN kernel
        friction (np.ndarray): friction parameter of the physical model :math:`s(x)` in the paper
    Shape:
        ``kernel_X``: :math:`(N, N)`, where :math:`N` is the number of training data.

        ``X``: :math:`(N,d)`  where :math:`N` is the number of training data and :math:`d` the dimensionality of the system.

        ``friction``: :math:`d`  where :math:`d` is the dimensionality of the system.

    Output: :math:`<\phi(x_i),d_k\phi(x_j)>` of shape `N,Nd`, where :math:`N` is the number of training data and :math:`d` the dimension of the system.
    """  # noqa: D205
    difference = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    n = difference.shape[0]
    d = X.shape[1]
    N = np.zeros((n, n * d))
    for i in range(n):
        for j in range(n):
            for k in range(0, d):
                N[i, k * n + j] = (
                    np.sqrt(friction[k]) * difference[i, j, k] * kernel_X[i, j] / sigma**2
                )
    return N


def return_dphi_dphi(kernel_X: np.ndarray, X: np.ndarray, sigma: float, friction: np.ndarray):
    r"""Returns the matrix :math:`M_{(k-1)n + i,(l-1)n+j}=<d_k\phi(x_i),d_l\phi(x_j)>` (matrix :math:`M` in the paper) only for
    a GAUSSIAN kernel
    where :math:`i = 1,\dots n, k=1, \dots d, j=1,\dots n, l=1, \dots d`
    and :math:`N` is the number of training points and :math:`d` is the dimensionality of the system.

    Args:
        kernel_X (np.ndarray): kernel matrix of the training data
        X (np.ndarray): training data
        sigma (float): length scale of the GAUSSIAN kernel
        friction (np.ndarray): friction parameter of the physical model :math:`s(x)` in the paper
    Shape:
        ``kernel_X``: :math:`(N, N)`, where :math:`N` is the number of training data.
        ``X``: :math:`(N,d)`  where :math:`N` is the number of training data and `d` the dimensionality of the system.
        ``friction``: :math:`d`  where :math:`d` is the dimensionality of the system
    Returns: :math:`M_{(k-1)n + i,(l-1)n+j}=<d_k\phi(x_i),d_l\phi(x_j)>` of shape `N,Nd`, where :math:`N` is the number of training data and :math:`d` the dimension of the system.
    """  # noqa: D205
    difference = X[:, np.newaxis, :] - X[np.newaxis, :, :]

    d = difference.shape[2]
    n = difference.shape[0]
    M = np.zeros((n * d, n * d))
    for i in range(n):
        for j in range(n):
            for k in range(0, d):
                for m in range(0, d):
                    if m == k:
                        M[(k) * n + i, (m) * n + j] = (
                            friction[k]
                            * (1 / sigma**2 - difference[i, j, k] ** 2 / sigma**4)
                            * kernel_X[i, j]
                        )
                    else:
                        M[(k) * n + i, (m) * n + j] = (
                            np.sqrt(friction[k])
                            * np.sqrt(friction[m])
                            * (-difference[i, j, k] * difference[i, j, m] / sigma**4)
                            * kernel_X[i, j]
                        )

    return M
