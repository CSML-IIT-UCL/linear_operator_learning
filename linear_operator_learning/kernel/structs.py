"""Structs used by the `kernel` algorithms."""

from typing import TypedDict

import numpy as np


class FitResult(TypedDict):
    """Return type for kernel regressors."""

    U: np.ndarray
    V: np.ndarray
    svals: np.ndarray | None


class EigResult(TypedDict):
    """Return type for eigenvalue decompositions of kernel regressors."""

    values: np.ndarray
    left: np.ndarray | None
    right: np.ndarray
