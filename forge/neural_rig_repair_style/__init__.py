"""Styled animation export for the sealed all-80 neural rig repair bank.

This package is intentionally downstream of :mod:`forge.neural_rig_repair`.
It never changes repair plans, logical drivers, categorical fields, or the
sealed motion audit.  It reconstructs those audited frames exactly and then
applies the immutable per-identity presentation palette.
"""

from .authority import RepairStyleAuthority, load_repair_style_authority
from .projection import RepairedMotionClip, RepairedMotionFrame, reconstruct_clip

__all__ = [
    "RepairStyleAuthority",
    "RepairedMotionClip",
    "RepairedMotionFrame",
    "load_repair_style_authority",
    "reconstruct_clip",
]
