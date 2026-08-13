"""Deterministic presentation compiler for immutable categorical sprite fields.

This package is deliberately independent from torch and the live training
stack. It compiles derived RGBA/effect layers while preserving categorical,
rig, and collision authority exactly.
"""

from .compiler import compile_generation_bank, compiler_source_hash
from .procedural import (
    compile_procedural_reference_bank,
    replay_procedural_reference_bank,
)
from .replay import replay_style_bank
from .rendering import render_layers
from .source import load_generation_bank

__all__ = (
    "compile_generation_bank",
    "compiler_source_hash",
    "compile_procedural_reference_bank",
    "load_generation_bank",
    "render_layers",
    "replay_style_bank",
    "replay_procedural_reference_bank",
)
