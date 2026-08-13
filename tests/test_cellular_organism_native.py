from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARENA = ROOT / "game/Arena.tscn"
PROJECT = ROOT / "game/project.godot"
SCENE = ROOT / "game/CellularOrganismLab.tscn"
SCRIPT = ROOT / "game/scripts/cellular_organism_lab.gd"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_native_cellular_lab_is_additive_and_arena_remains_main() -> None:
    assert SCENE.is_file() and SCRIPT.is_file()
    project = PROJECT.read_text(encoding="utf-8")
    assert 'run/main_scene="res://Arena.tscn"' in project
    assert "CellularOrganismLab" not in project


def test_native_script_exposes_cell_damage_food_fluid_and_reproduction() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    for symbol in (
        "_break_bond",
        "_step_fluid_and_metabolism",
        "_damage_at",
        "_feed_current",
        "_can_reproduce",
        "_reproduce",
        "FLAG_EYE",
        "FLAG_MOUTH",
        "FLAG_REPRODUCTIVE",
    ):
        assert symbol in script
    assert "CELLULAR_ORGANISM_SMOKE_OK" in script
    assert "python_runtime_required" in script


def test_native_runtime_is_not_python_dependent() -> None:
    import json

    catalog = json.loads((ROOT / "game/generated/cellular_organism/v2/catalog.json").read_text(encoding="utf-8"))
    assert catalog["runtime_contract"]["python_required"] is False
    assert catalog["sample_count"] == 80
    assert catalog["totals"] == {
        "bonds": 116112,
        "eyes": 224,
        "organs": 1255,
        "phase_tethers": 1,
        "physical_cells": 34178,
    }


def test_arena_source_is_not_referenced_by_cellular_lab() -> None:
    combined = SCENE.read_text(encoding="utf-8") + SCRIPT.read_text(encoding="utf-8")
    assert "Arena.tscn" not in combined
    assert _sha(ARENA) == "c5f8b961297b43f683d40dff831cb576d89539c222f0a5a9abab3d29a1f67490"
