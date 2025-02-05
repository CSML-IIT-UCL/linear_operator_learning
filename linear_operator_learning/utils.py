"""Generic Utilities."""

from math import sqrt

import numpy as np
from scipy.spatial.distance import pdist


def topk(vec: np.ndarray, k: int):
    """Get the top k values from a Numpy array.

    Args:
        vec (np.ndarray): A 1D numpy array
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


def sanitize_complex_conjugates(vec: np.ndarray, tol: float = 10.0):
    """On 1D complex vector, if the real parts of two elements are close, set them equal. Set to 0 the imaginary parts smaller than `tol` times the machine precision.

    Args:
        vec (np.ndarray): A 1D vector to sanitize.
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
