"""Main entry point for linear operator learning."""


def __getattr__(attr):
    """This is a workaround to avoid importing all subpackages at once.

    It allows to lazily import the subpackages, e.g.:
    >>> from linear_operator_learning import kernel
    >>> from linear_operator_learning import nn
    instead of:
    >>> import linear_operator_learning
    >>> linear_operator_learning.kernel
    >>> linear_operator_learning.nn
    """
    if attr == "kernel":
        import linear_operator_learning.kernel as kernel

        return kernel
    elif attr == "nn":
        import linear_operator_learning.nn as nn

        return nn
    else:
        raise AttributeError(f"Unknown submodule {attr}")
