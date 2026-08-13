from __future__ import annotations

import hashlib
import json
from pathlib import Path

from forge.cellular_symmetry_sync import project_runtime, validate_runtime


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/cellular_breeding_symmetry_v1/cellular_symmetry_manifest.json"
RUNTIME = ROOT / "game/generated/cellular_symmetry/v1"
SCENE = ROOT / "game/SymmetricOrganismLab.tscn"


def test_symmetry_native_projection_is_repeatable_and_closed() -> None:
    assert project_runtime(SOURCE) == project_runtime(SOURCE)
    assert validate_runtime(RUNTIME) == {
        "passed": True,
        "sample_count": 45,
        "cell_count": 25668,
        "bond_count": 85357,
        "organ_count": 748,
        "symmetry_cells_added": 2535,
        "bundle_id": "ab80cbdf768bd1ade6dd2d8cd3f74aeb88c433dcaaac5dc84a68e1faf331ee38",
    }


def test_native_catalog_truthfully_labels_soft_additive_symmetry() -> None:
    catalog = json.loads((RUNTIME / "catalog.json").read_text(encoding="utf-8"))
    contract = catalog["runtime_contract"]
    assert contract["soft_not_hard_symmetry"] is True
    assert contract["source_cells_preserved"] is True
    assert contract["runtime_offspring_redecode"] is False
    assert catalog["symmetry_summary"]["improved_samples"] == 45


def test_native_scene_and_smoke_cover_all_refined_organisms() -> None:
    scene = SCENE.read_text(encoding="utf-8")
    assert 'catalog_path = "res://generated/cellular_symmetry/v1/catalog.json"' in scene
    assert "expected_species_count = 45" in scene and "lineage_mode = true" in scene
    report = json.loads((ROOT / "outputs/cellular_symmetry_godot_report.json").read_text(encoding="utf-8"))
    assert report["passed"] is True and report["species_loaded"] == 45
    assert report["cells_checked"] == 25668 and report["organs_checked"] == 748 and report["bonds_checked"] == 85357
    assert report["population_after_reproduction"] == 2 and report["python_runtime_required"] is False
    assert hashlib.sha256((ROOT / "game/Arena.tscn").read_bytes()).hexdigest() == "c5f8b961297b43f683d40dff831cb576d89539c222f0a5a9abab3d29a1f67490"
