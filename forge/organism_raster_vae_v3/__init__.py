"""High-resolution structured continuous VAE for morphology-v2 organisms."""

from .contract import RasterVAEV3Config
from .model import StructuredRasterVAE

__all__ = ["RasterVAEV3Config", "StructuredRasterVAE"]
