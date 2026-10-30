"""
Shared code for:

    Regional hierarchical reorganization in antipsychotic-naive
    first-episode psychosis

Submodules
----------
``config``        path and constant definitions (edit this, or set ``KASP_DATA``)
``data``          loaders for model outputs, timeseries and clinical tables
``parcellation``  atlas, gradient, network assignment and geometry helpers
``plotting``      figure style and the parcel-wise scatter helper
``stats``         outlier rules, permutation tests and spin tests
``rendering``     cortical surface and subcortical renderings of parcel maps
``record``        the per-run log of reported statistics (``runs/<tag>/stats.csv``)

Submodules are imported lazily, so ``from kasp import config`` (and therefore
``python config.py``, which checks your paths) works before the scientific
stack is installed. The heavier modules pull in scipy, neuromaps and friends
only when you actually touch them.
"""

import importlib

__all__ = ["config", "data", "parcellation", "plotting", "stats",
           "rendering", "record"]
__version__ = "1.0.0"

_SUBMODULES = frozenset(__all__)


def __getattr__(name):
    if name in _SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _SUBMODULES)
