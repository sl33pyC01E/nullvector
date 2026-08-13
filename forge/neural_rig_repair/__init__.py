"""CPU-only, rest-exact logical rig repair for authoritative neural fields."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "bind_repair_plan": ("binding", "bind_repair_plan"),
    "compile_motion_clip_audit": ("motion", "compile_motion_clip_audit"),
    "compile_repair_bank": ("compiler", "compile_repair_bank"),
    "compile_repair_plan": ("planner", "compile_repair_plan"),
    "compile_sample_motion_audit": ("motion", "compile_sample_motion_audit"),
    "load_repair_plan": ("planner", "load_repair_plan"),
    "load_repair_source": ("source", "load_repair_source"),
    "prepare_repair_plans": ("compiler", "prepare_repair_plans"),
    "replay_repair_bank": ("replay", "replay_repair_bank"),
    "validate_repaired_binding": ("binding", "validate_repaired_binding"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value
