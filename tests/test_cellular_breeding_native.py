from __future__ import annotations

import hashlib
import json
from pathlib import Path

from forge.cellular_breeding_sync import project_runtime, validate_runtime


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/cellular_breeding_v1/cellular_breeding_manifest.json"
RUNTIME = ROOT / "game/generated/cellular_breeding/v1"
SCENE = ROOT / "game/CellularBreedingLab.tscn"
SCRIPT = ROOT / "game/scripts/cellular_organism_lab.gd"
ARENA = ROOT / "game/Arena.tscn"
PROJECT = ROOT / "game/project.godot"


def test_breeding_native_projection_is_repeatable_and_hash_closed() -> None:
    first = project_runtime(SOURCE)
    second = project_runtime(SOURCE)
    assert first == second
    assert validate_runtime(RUNTIME) == {
        "passed": True,
        "sample_count": 45,
        "cell_count": 22933,
        "bond_count": 77829,
        "organ_count": 746,
        "family_pair_count": 15,
        "bundle_id": "b956f84de997f6e805e0aa3113ac6d353264b7c046c202d52e47d5e1168fa3dc",
    }


def test_native_catalog_exposes_true_structural_lineage() -> None:
    catalog = json.loads((RUNTIME / "catalog.json").read_text(encoding="utf-8"))
    assert catalog["bank_kind"] == "two-parent-structural-cellular-breeding"
    assert catalog["runtime_contract"]["offspring_anatomy_is_forge_decoded"] is True
    assert catalog["runtime_contract"]["runtime_offspring_redecode"] is False
    assert catalog["orientation"]["contract"]["projection"] == "top_down_dorsal"
    assert catalog["runtime_contract"]["uniform_screen_gravity_disabled"] is True
    assert catalog["simulation"]["gravity"] == 0.0
    assert len(catalog["family_pair_counts"]) == 15
    for entry in catalog["species"]:
        runtime = json.loads((RUNTIME / entry["runtime"]["path"]).read_text(encoding="utf-8"))
        assert runtime["lineage"] == entry["lineage"]
        assert runtime["genome"]["structural_lineage"]["parent_ids"] == entry["lineage"]["parent_ids"]


def test_breeding_scene_uses_the_cell_physics_lab_without_changing_arena() -> None:
    scene = SCENE.read_text(encoding="utf-8")
    script = SCRIPT.read_text(encoding="utf-8")
    assert 'catalog_path = "res://generated/cellular_breeding/v1/catalog.json"' in scene
    assert "expected_species_count = 45" in scene and "lineage_mode = true" in scene
    for feature in ("_damage_at", "_step_fluid_and_metabolism", "_reproduce"):
        assert feature in script
    assert 'run/main_scene="res://Arena.tscn"' in PROJECT.read_text(encoding="utf-8")
    assert hashlib.sha256(ARENA.read_bytes()).hexdigest() == "c5f8b961297b43f683d40dff831cb576d89539c222f0a5a9abab3d29a1f67490"


def test_native_smoke_proves_all_structural_offspring() -> None:
    report = json.loads((ROOT / "outputs/cellular_breeding_topdown_godot_report.json").read_text(encoding="utf-8"))
    assert report["passed"] is True and report["species_loaded"] == 45
    assert report["cells_checked"] == 22933 and report["organs_checked"] == 746 and report["bonds_checked"] == 77829
    assert report["population_after_reproduction"] == 2
    assert report["python_runtime_required"] is False
    assert report["orientation"] == "top_down_dorsal"
    assert report["uniform_screen_gravity"] is False
    assert report["surface_fluid_model"] == "isotropic_surface_diffusion"
    assert report["surface_puddles_observed"] > 0
