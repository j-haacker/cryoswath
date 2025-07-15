__all__ = [
    "__version__",
    "misc",
    "gis",
    "l1b",
    "l2",
    "l3",
    "l4",
    "test_plots",  # subpackage
]

from importlib.metadata import version as _version
from cryoswath import gis, misc, l1b, l2, l3, l4, test_plots


# copied from xarray
try:
    __version__ = _version("cryoswath")
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "9999"
