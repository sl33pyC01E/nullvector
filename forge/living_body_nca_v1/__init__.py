"""Bridge between native living bodies and the causal cellular NCA."""

from .adapter import BodyRaster, LivingBodyNCARuntime, rasterize_body
from .evaluation import audit, validate

__all__ = ["BodyRaster", "LivingBodyNCARuntime", "audit", "rasterize_body", "validate"]
