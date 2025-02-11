"""Structs used by the `nn` algorithms."""

from typing import TypedDict

from torch import Tensor


class FitResult(TypedDict):
    """Return type for nn regressors."""

    U: Tensor
    V: Tensor
    svals: Tensor | None


class EigResult(TypedDict):
    """Return type for eigenvalue decompositions of nn regressors."""

    values: Tensor
    left: Tensor | None
    right: Tensor
