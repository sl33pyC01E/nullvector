from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from forge.multifield_style_motion.hashing import canonical_json_bytes, sha256_bytes
from forge.neural_fusion_production.contract import FUSION_MODES, MUTATION_MODES
from forge.neural_fusion_production_evolution.contract import DEFAULT_FOUNDERS, DEFAULT_OUTPUT, FORMAT
from forge.neural_fusion_production_evolution.evolution import (
    _load_founders,
    validate_production_evolution,
)


MANIFEST = DEFAULT_OUTPUT / "production_evolution_manifest.json"


def _authority() -> dict:
    assert MANIFEST.is_file(), "compile the checked-in production evolution bank first"
    return validate_production_evolution(MANIFEST)


def _rehashed(payload: dict) -> bytes:
    unsigned = dict(payload)
    unsigned.pop("evolution_sha256", None)
    payload = {**unsigned, "evolution_sha256": sha256_bytes(canonical_json_bytes(unsigned))}
    return canonical_json_bytes(payload)


def _tamper_bank(tmp_path: Path, mutate) -> Path:
    root = tmp_path / "bank"
    shutil.copytree(DEFAULT_OUTPUT, root)
    path = root / "production_evolution_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    path.write_bytes(_rehashed(manifest))
    return path


def test_founders_are_exact_unique_and_span_every_family() -> None:
    manifest, founders, _ = _load_founders(DEFAULT_FOUNDERS)
    assert manifest["status"] == "ready"
    assert len(founders) == 12
    assert len({item.sample_id for item in founders}) == 12
    assert len({item.raw_fields_sha256 for item in founders}) == 12
    assert {item.family_id for item in founders} == set(range(5))


def test_production_evolution_bank_is_strict_and_complete() -> None:
    manifest = _authority()
    assert manifest["format"] == FORMAT
    assert manifest["quality_tier"] == "production-learned-latent-evolution-v1"
    assert manifest["counts"]["founders"] == 12
    assert manifest["counts"]["generations"] == 3
    assert manifest["counts"]["selected"] == 36
    assert manifest["counts"]["motion_clips"] == 108
    assert manifest["counts"]["layer_atlases"] == 252
    assert all(manifest["gates"].values())


def test_every_generation_preserves_morphology_and_operator_vocabularies() -> None:
    manifest = _authority()
    for generation in range(1, 4):
        records = [item for item in manifest["selected"] if item["generation"] == generation]
        assert len(records) == 12
        assert {item["family_id"] for item in records} == set(range(5))
        assert all(2 <= sum(item["family_id"] == family_id for item in records) <= 3 for family_id in range(5))
        assert {item["fusion_mode"] for item in records} == set(FUSION_MODES)
        assert {item["mutation_mode"] for item in records} == set(MUTATION_MODES)
        assert [item["rank"] for item in records] == list(range(12))


def test_recursive_lineage_is_closed_and_strictly_ancestral() -> None:
    manifest = _authority()
    generations = {item["specimen_id"]: item["generation"] for item in manifest["lineage_nodes"]}
    assert len(generations) == manifest["counts"]["lineage_nodes"]
    for record in manifest["selected"]:
        assert all(parent in generations for parent in record["parent_ids"])
        assert all(generations[parent] < record["generation"] for parent in record["parent_ids"])


def test_fully_rehashed_path_escape_is_rejected(tmp_path: Path) -> None:
    path = _tamper_bank(tmp_path, lambda manifest: manifest["artifacts"]["contact_sheet"].__setitem__("path", "../escape.png"))
    with pytest.raises(ValueError, match="unsafe|escapes"):
        validate_production_evolution(path)


def test_fully_rehashed_lineage_forgery_is_rejected(tmp_path: Path) -> None:
    path = _tamper_bank(tmp_path, lambda manifest: manifest["selected"][12]["parent_ids"].__setitem__(0, "nonexistent-parent"))
    with pytest.raises(ValueError, match="lineage"):
        validate_production_evolution(path)
