"""Utilities of symmetric random variables and vector spaces with known group representations."""

try:
    import symm_torch
except ImportError as e:
    raise ImportError(
        "Please install optional dependencies for symmetries: `pip install 'linear-operator-learning[symm]'`"
    ) from e
