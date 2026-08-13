"""Motion-coherent presentation for authoritative categorical sprite clips."""

from .compiler import compile_motion_style_bank, compiler_source_hash
from .replay import replay_motion_style_bank
from .rendering import render_motion_frame
from .source import load_motion_bank


__all__ = (
    "compile_motion_style_bank",
    "compiler_source_hash",
    "load_motion_bank",
    "render_motion_frame",
    "replay_motion_style_bank",
)
