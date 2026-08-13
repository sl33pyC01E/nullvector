from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from forge.morphology import (
    CANVAS_SIZE,
    FAMILIES,
    LAYER_NAMES,
    SAFETY_MARGIN,
    DiskBudgetError,
    MorphologyGenome,
    canonical_manifest_json,
    compose_rgba,
    estimate_corpus_bytes,
    genome_from_seed,
    legal_field_tuples,
    plan_disk_budget,
    render_specimen,
    validate_specimen,
)
from forge.morphology.constants import JOINT_LAYER, SOCKET_LAYER
from forge.morphology.constants import (
    EMISSION_LEVEL_NAMES,
    GUIDE_CHANNEL_NAMES,
    MATERIAL_NAMES,
    PART_OWNER_NAMES,
)
from forge.morphology.disk_guard import guard_corpus_destination
from forge.morphology.preview import (
    build_contact_sheet,
    build_role_contact_sheet,
    prototype_specimens,
    role_matrix_specimens,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_genome_round_trip_and_deterministic_render() -> None:
    for family in FAMILIES:
        genome = genome_from_seed(0xDEADBEEF, family)
        restored = MorphologyGenome.from_dict(genome.to_dict())
        assert restored == genome
        assert genome.condition_vector().shape == (24,)
        assert np.isfinite(genome.condition_vector()).all()
        assert (genome.condition_vector() >= 0).all()
        assert (genome.condition_vector() <= 1).all()

        first = render_specimen(genome)
        second = render_specimen(restored)
        assert np.array_equal(first.layers, second.layers)
        assert np.array_equal(first.tokens, second.tokens)
        assert canonical_manifest_json(first) == canonical_manifest_json(second)


def test_1000_genome_fuzz_contract() -> None:
    semantic_hashes = {family: set() for family in FAMILIES}
    family_counts = {family: 0 for family in FAMILIES}
    observed_part_owners: set[int] = set()
    for index in range(1_000):
        family = FAMILIES[index % len(FAMILIES)]
        seed = (0x12340000 + index * 7_919) & 0xFFFFFFFF
        specimen = render_specimen(genome_from_seed(seed, family))
        assert validate_specimen(specimen) == [], (
            family,
            seed,
            validate_specimen(specimen),
        )
        assert specimen.layers.shape == (len(LAYER_NAMES), 48, 48)
        assert specimen.layers.dtype == np.uint8
        assert specimen.tokens.shape == (48, 48)
        assert specimen.tokens.dtype == np.uint8
        assert np.isin(specimen.layers, (0, 1)).all()
        assert int(specimen.tokens.max()) <= len(LAYER_NAMES)
        assert all(int(layer.sum()) > 0 for layer in specimen.layers)
        visible = np.logical_or.reduce(specimen.layers > 0)
        assert not visible[:SAFETY_MARGIN].any()
        assert not visible[-SAFETY_MARGIN:].any()
        assert not visible[:, :SAFETY_MARGIN].any()
        assert not visible[:, -SAFETY_MARGIN:].any()
        for name, layer_index in JOINT_LAYER.items():
            x, y = specimen.joints[name]
            assert specimen.layers[layer_index, y, x] == 1
        for name, layer_index in SOCKET_LAYER.items():
            x, y = specimen.sockets[name]
            assert specimen.layers[layer_index, y, x] == 1
        rgba = compose_rgba(specimen)
        assert rgba.shape == (CANVAS_SIZE, CANVAS_SIZE, 4)
        assert set(np.unique(rgba[..., 3]).tolist()).issubset({0, 255})
        assert not rgba[: SAFETY_MARGIN - 1, :, 3].any()
        assert not rgba[-(SAFETY_MARGIN - 1) :, :, 3].any()
        fields = specimen.training_fields()
        assert fields.guide.shape == (len(GUIDE_CHANNEL_NAMES), 48, 48)
        assert fields.guide.dtype == np.float32
        assert np.isfinite(fields.guide).all()
        assert (fields.guide >= 0).all() and (fields.guide <= 1).all()
        assert fields.part_owner.shape == (48, 48)
        assert fields.part_owner.dtype == np.uint8
        assert int(fields.part_owner.max()) < len(PART_OWNER_NAMES)
        assert int(fields.material.max()) < len(MATERIAL_NAMES)
        assert int(fields.emission_level.max()) < len(EMISSION_LEVEL_NAMES)
        assert fields.morphology_index == FAMILIES.index(family)
        assert fields.subtype_id == fields.morphology_index * 4 + specimen.genome.silhouette_variant
        assert fields.role_id == specimen.genome.role_id
        assert fields.genes.shape == (24,)
        observed_part_owners.update(np.unique(fields.part_owner).tolist())
        semantic_hashes[family].add(specimen.manifest["hashes"]["semantic_sha256"])
        family_counts[family] += 1

    assert family_counts == {family: 200 for family in FAMILIES}
    assert all(len(values) >= 190 for values in semantic_hashes.values())
    assert observed_part_owners == set(range(len(PART_OWNER_NAMES)))


def test_manifest_matches_versioned_json_schema() -> None:
    schema_path = PROJECT_ROOT / "shared" / "schema" / "morphology_manifest.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    for family in FAMILIES:
        specimen = render_specimen(genome_from_seed(44_721, family))
        assert list(validator.iter_errors(specimen.manifest)) == []


def test_disk_budget_guard_and_production_isolation() -> None:
    estimate = estimate_corpus_bytes(1_000)
    budget = plan_disk_budget(
        1_000,
        free_bytes=estimate + 10_000,
        reserve_bytes=10_000,
    )
    assert budget.estimated_bytes == estimate
    assert budget.sample_count == 1_000
    with pytest.raises(DiskBudgetError, match="guarded bytes"):
        plan_disk_budget(
            1_000,
            free_bytes=estimate,
            reserve_bytes=10_000,
        )
    with pytest.raises(DiskBudgetError, match="guarded limit"):
        plan_disk_budget(
            250_001,
            free_bytes=10**15,
            reserve_bytes=0,
        )
    with pytest.raises(DiskBudgetError, match="isolated"):
        guard_corpus_destination(
            PROJECT_ROOT / "game" / "generated" / "broad-morphology",
            1,
            reserve_bytes=0,
        )


def test_prototype_contact_sheet_contract() -> None:
    specimens = prototype_specimens(count_per_family=2)
    assert len(specimens) == len(FAMILIES) * 2
    sheet = build_contact_sheet(specimens, columns=2, scale=2)
    assert sheet.mode == "RGBA"
    assert sheet.size == (2 * CANVAS_SIZE * 2, 24 + len(FAMILIES) * CANVAS_SIZE * 2)
    role_specimens = role_matrix_specimens()
    assert len(role_specimens) == len(FAMILIES) * 8
    role_sheet = build_role_contact_sheet(role_specimens, scale=2)
    assert role_sheet.mode == "RGBA"
    assert role_sheet.size == (74 + 8 * 96, 28 + len(FAMILIES) * 96)


def test_legal_field_tuple_table_is_deterministic_and_valid() -> None:
    fields = [
        render_specimen(genome_from_seed(900 + index, FAMILIES[index % 5])).training_fields()
        for index in range(40)
    ]
    first = legal_field_tuples(fields)
    second = legal_field_tuples(tuple(reversed(fields)))
    assert np.array_equal(first, second)
    assert first.dtype == np.uint8
    assert first.shape[1] == 3
    assert [0, 0, 0] in first.tolist()
    assert int(first[:, 0].max()) < len(PART_OWNER_NAMES)
    assert int(first[:, 1].max()) < len(MATERIAL_NAMES)
    assert int(first[:, 2].max()) < len(EMISSION_LEVEL_NAMES)


def test_role_conditioning_changes_every_family_without_breaking_topology() -> None:
    for family in FAMILIES:
        base = genome_from_seed(0xA11CE55, family)
        hashes: set[str] = set()
        field_hashes: set[str] = set()
        for role_id in range(8):
            specimen = render_specimen(replace(base, role_id=role_id))
            assert validate_specimen(specimen) == []
            assert specimen.genome.role_id == role_id
            hashes.add(specimen.manifest["hashes"]["semantic_sha256"])
            field_hashes.add(specimen.training_fields().arrays_hash())
        assert len(hashes) == 8, family
        assert len(field_hashes) == 8, family
