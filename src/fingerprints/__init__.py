"""Cancer Fingerprints: explain a tumour's mutations via COSMIC mutational signatures.

For research and education only. Not for diagnosis.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fingerprints")
except PackageNotFoundError:  # pragma: no cover - only when running from an uninstalled tree
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
