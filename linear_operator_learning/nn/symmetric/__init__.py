"""Utilities of symmetric random variables and vector spaces with known group representations."""

import sys
import warnings


def __getattr__(attr):
    """Lazy-import submodules and warn if optional dependencies are not installed."""
    submodules = {
        "linalg": "linear_operator_learning.nn.symmetric.linalg",
        "stats": "linear_operator_learning.nn.symmetric.stats",
    }

    if attr in submodules:
        try:
            module = __import__(submodules[attr], fromlist=[""])
            setattr(sys.modules[__name__], attr, module)  # Attach to the module namespace
            return module
        except ImportError as e:
            warnings.warn(
                "Optional dependencies for symmetries are not installed. "
                "Please install them using: `pip install 'linear-operator-learning[symm]'`"
            )
            raise e

    raise AttributeError(f"Unknown submodule {attr}")
