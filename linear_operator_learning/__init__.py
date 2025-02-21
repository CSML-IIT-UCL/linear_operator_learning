"""Main entry point for linear operator learning."""

import sys


def __getattr__(attr):
    """Lazy-import submodules."""
    submodules = {
        "kernel": "linear_operator_learning.kernel",
        "nn": "linear_operator_learning.nn",
    }

    if attr in submodules:
        module = __import__(submodules[attr], fromlist=[""])
        setattr(sys.modules[__name__], attr, module)  # Attach to the module namespace
        return module

    raise AttributeError(f"Unknown submodule {attr}")
