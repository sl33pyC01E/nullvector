from __future__ import annotations

import hashlib
import json
from pathlib import Path

from forge.evolved_cellular_organism_sync import project_runtime, validate_runtime


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/evolved_cellular_organism_v1/evolved_cellular_organism_manifest.json"
RUNTIME = ROOT / "game/generated/evolved_cellular_organism/v1"
SCENE = ROOT / "game/EvolvedOrganismLab.tscn"
SCRIPT = ROOT / "game/scripts/cellular_organism_lab.gd"
PROJECT = ROOT / "game/project.godot"
ARENA = ROOT / "game/Arena.tscn"


def test_evolved_native_projection_is_repeatable_and_hash_closed() -> None:
    first = project_runtime(SOURCE)
    second = project_runtime(SOURCE)
    assert first == second
    validation = validate_runtime(RUNTIME)
    assert validation == {
        "passed": True,
        "sample_count": 36,
        "cell_count": 14457,
        "bond_count": 48569,
        "organ_count": 580,
        "bundle_id": "df87a2f3b460e4c648b54e26b01d369702b9b0203152e0bd5689765e7094d0d1",
    }


def test_runtime_exposes_exact_neural_lineage_and_truthful_reproduction_scope() -> None:
    catalog = json.loads((RUNTIME / "catalog.json").read_text(encoding="utf-8"))
    assert catalog["bank_kind"] == "learned-latent-evolution-cellular"
    assert catalog["generation_counts"] == {"1": 12, "2": 12, "3": 12}
    assert catalog["runtime_contract"]["neural_lineage_visible"] is True
    assert catalog["runtime_contract"]["runtime_offspring_redecode"] is False
    for entry in catalog["species"]:
        runtime = json.loads((RUNTIME / entry["runtime"]["path"]).read_text(encoding="utf-8"))
        assert runtime["lineage"] == entry["lineage"]
        assert runtime["genome"]["neural_lineage"]["lineage_sha256"] == entry["lineage"]["lineage_sha256"]


def test_evolved_scene_reuses_cell_physics_with_lineage_configuration() -> None:
    scene = SCENE.read_text(encoding="utf-8")
    script = SCRIPT.read_text(encoding="utf-8")
    assert 'catalog_path = "res://generated/evolved_cellular_organism/v1/catalog.json"' in scene
    assert "expected_species_count = 36" in scene
    assert "lineage_mode = true" in scene
    for feature in ("_damage_at", "_step_fluid_and_metabolism", "_reproduce"):
        assert feature in script


def test_native_smoke_proves_all_descendants_and_arena_is_untouched() -> None:
    report = json.loads((ROOT / "outputs/evolved_cellular_organism_godot_report.json").read_text(encoding="utf-8"))
    assert report["passed"] is True and report["species_loaded"] == 36
    assert report["cells_checked"] == 14457
    assert report["organs_checked"] == 580
    assert report["bonds_checked"] == 48569
    assert report["population_after_reproduction"] == 2
    assert report["python_runtime_required"] is False
    assert 'run/main_scene="res://Arena.tscn"' in PROJECT.read_text(encoding="utf-8")
    assert hashlib.sha256(ARENA.read_bytes()).hexdigest() == "c5f8b961297b43f683d40dff831cb576d89539c222f0a5a9abab3d29a1f67490"
