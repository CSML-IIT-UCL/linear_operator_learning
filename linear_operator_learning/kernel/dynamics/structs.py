"""Structs used by the `kernel dynamics`  algorithms."""

from typing import TypedDict

import numpy as np
from numpy import ndarray


class ModeResult(TypedDict):
    """Return type for modes decompositions of dynamics kernel regressors."""

    decay_rates: ndarray
    frequencies: ndarray
    modes: ndarray
