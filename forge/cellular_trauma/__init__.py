"""Deterministic wounds, clotting, scars, reconnection, and fragment fate."""

from .compiler import build_bank, compile_trauma, replay_bank, validate_bank
from .simulation import TraumaState

__all__ = ["TraumaState", "build_bank", "compile_trauma", "replay_bank", "validate_bank"]
