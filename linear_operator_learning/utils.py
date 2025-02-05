"""Generic Utilities."""

import numpy as np


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
