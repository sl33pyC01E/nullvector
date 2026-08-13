from __future__ import annotations

from dataclasses import replace
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from forge.morphology import genome_from_seed, render_specimen
from forge.morphology.constants import FAMILIES, ROLE_NAMES, SAFETY_MARGIN
from forge.morphology.motion import FACING_NAMES, MOTION_NAMES
from forge.neural_rig_bridge import (
    BINDING_FORMAT,
    DRIVER_NAMES,
    REPLAY_FORMAT,
    BindingRejected,
    DerivedAnatomy,
    assert_exact_neural_motion_replay,
    assert_exact_replay,
    assert_valid_bound_frame,
    bind_neural_fields,
    bind_raw_sample_archive,
    compile_neural_motion_clip,
    derive_conditioned_anatomy,
    facing_transforms,
    motion_adapter_contract,
    render_bound_pose,
    render_facing_frame,
    replay_neural_motion_clip,
    replay_binding,
    validate_binding,
    validate_bound_frame,
    validate_neural_motion_clip,
)
from forge.neural_rig_bridge.hashing import (
    aligned_fields_hash,
    canonical_json_hash,
    evaluator_tuple_fingerprint,
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(
    family: str,
    role: int,
    *,
    salt: int = 0,
    require_all_facings: bool = False,
):
    family_id = FAMILIES.index(family)
    last_error: Exception | None = None
    for attempt in range(96):
        seed = (
            0x5A170000
            + family_id * 0x1F123
            + role * 0x10D1
            + salt * 0x10001
            + attempt * 0x9E37
        ) & 0xFFFFFFFF
        genome = replace(genome_from_seed(seed, family), role_id=role)
        specimen = render_specimen(genome)
        fields = specimen.training_fields()
        anatomy = derive_conditioned_anatomy(
            seed,
            family=family,
            subtype_id=genome.subtype_id,
            role_id=role,
        )
        try:
            binding = bind_neural_fields(
                fields.part_owner,
                fields.material,
                fields.emission_level,
                fields.guide,
                family=family,
                subtype_id=genome.subtype_id,
                role_id=role,
                anatomy=anatomy,
                genes=fields.genes,
                corpus_seed=seed,
            )
            if require_all_facings:
                for facing in FACING_NAMES:
                    render_facing_frame(binding, facing)
            return specimen, fields, anatomy, binding
        except BindingRejected as error:
            last_error = error
    raise AssertionError(
        f"no valid synthetic fixture for {family}/{role}: {last_error}"
    )


def _field_tuples(
    part: np.ndarray, material: np.ndarray, emission: np.ndarray
) -> set[tuple[int, int, int]]:
    return {
        tuple(map(int, row))
        for row in np.stack((part, material, emission), axis=-1).reshape(-1, 3)
    }


def _add_disconnected_body_tuple(fields):
    part = fields.part_owner.copy()
    material = fields.material.copy()
    emission = fields.emission_level.copy()
    physical = (part != 0) & (part != 16)
    padded = np.pad(physical, 1)
    nearby = np.logical_or.reduce(
        [padded[y : y + 48, x : x + 48] for y in range(3) for x in range(3)]
    )
    candidates = np.argwhere(~nearby)
    candidates = candidates[
        (candidates[:, 0] >= SAFETY_MARGIN)
        & (candidates[:, 0] < 48 - SAFETY_MARGIN)
        & (candidates[:, 1] >= SAFETY_MARGIN)
        & (candidates[:, 1] < 48 - SAFETY_MARGIN)
    ]
    assert len(candidates)
    y, x = map(int, candidates[0])
    source_y, source_x = map(int, np.argwhere(part == 1)[0])
    part[y, x] = 1
    material[y, x] = material[source_y, source_x]
    emission[y, x] = emission[source_y, source_x]
    return part, material, emission


def _write_raw_pair(
    tmp_path: Path,
    specimen,
    fields,
    legal_tuples: np.ndarray,
    *,
    sample_id: str = "raw_fixture_sample",
):
    raw_dir = tmp_path / "bank" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / f"{sample_id}.npz"
    np.savez_compressed(
        archive_path,
        format=np.asarray(["nullvector-multifield-raw-sample-v1"]),
        part=fields.part_owner,
        material=fields.material,
        emission=fields.emission_level,
        guide=fields.guide,
        genes=fields.genes,
        target_part=fields.part_owner,
        target_material=fields.material,
        target_emission=fields.emission_level,
        morphology=np.asarray([specimen.genome.family], dtype=np.uint8),
        subtype=np.asarray([specimen.genome.subtype_id], dtype=np.uint8),
        role=np.asarray([specimen.genome.role_id], dtype=np.uint8),
        source_index=np.asarray([0], dtype=np.int64),
        corpus_seed=np.asarray([specimen.genome.seed], dtype=np.uint32),
        sample_seed=np.asarray([123], dtype=np.uint64),
    )
    fake = "a" * 64
    raw_hash = aligned_fields_hash(
        fields.part_owner, fields.material, fields.emission_level
    )
    manifest = {
        "format": "nullvector-multifield-raw-sample-v1",
        "render_format": "nullvector-multifield-raster-v1",
        "condition": {
            "ordinal": 0,
            "sample_id": sample_id,
            "grid_mode": "fixed",
            "source_index": 0,
            "variation": 0,
            "sample_seed": 123,
            "morphology_id": specimen.genome.family,
            "morphology_name": FAMILIES[specimen.genome.family],
            "subtype_id": specimen.genome.subtype_id,
            "subtype_name": f"{FAMILIES[specimen.genome.family]}_{specimen.genome.subtype_id % 4}",
            "role_id": specimen.genome.role_id,
            "role_name": ROLE_NAMES[specimen.genome.role_id],
        },
        "raw_fields_sha256": raw_hash,
        "checkpoint_sha256": fake,
        "canonical_ema_hash": fake,
        "corpus_sha256": fake,
        "training_source_hash": fake,
        "evaluation_source_hash": fake,
        "guide_policy": {},
        "legal_tuple_fingerprint": evaluator_tuple_fingerprint(legal_tuples),
        "temperature": 0.9,
        "artifacts": {
            "fields": {
                "path": f"raw/{sample_id}.npz",
                "bytes": archive_path.stat().st_size,
                "sha256": _sha256_file(archive_path),
            },
            "rgba": {
                "path": f"raw/{sample_id}_rgba.png",
                "bytes": 1,
                "sha256": fake,
            },
            "emission": {
                "path": f"raw/{sample_id}_emission.png",
                "bytes": 1,
                "sha256": fake,
            },
        },
        "validation": {
            "format": "nullvector-multifield-generation-validation-v2",
            "sample_id": sample_id,
            "raw_fields_sha256": raw_hash,
            "errors": [],
            "topology": {},
            "margins": {"safe": True},
            "tuples": {
                "valid_fraction": 1.0,
                "invalid_pixels": 0,
                "legal_tuple_count": len(legal_tuples),
            },
            "source_similarity": {},
            "condition_adherence": {},
            "hard_gates": {
                "categorical_domains": True,
                "guide_contract": True,
                "target_contract": True,
                "condition_contract": True,
                "legal_table_contract": True,
                "nonempty": True,
                "occupancy": True,
                "visible_connected": True,
                "structural_margin": True,
                "visible_margin": True,
                "legal_tuples": True,
                "scaffold_coverage": True,
                "essential_owners": True,
            },
            "hard_valid": True,
            "condition_exact_match": True,
            "condition_in_distribution": True,
            "accepted": True,
        },
    }
    manifest_path = raw_dir / f"{sample_id}.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return archive_path, manifest_path, manifest


def test_five_families_eight_roles_and_eight_facings_fuzz_exactly() -> None:
    seen_conditions: set[tuple[str, int]] = set()
    seen_facings: set[str] = set()
    hashes: set[str] = set()
    for family in FAMILIES:
        for role in range(len(ROLE_NAMES)):
            specimen, fields, anatomy, first = _fixture(
                family, role, require_all_facings=True
            )
            second = bind_neural_fields(
                fields.part_owner,
                fields.material,
                fields.emission_level,
                fields.guide,
                family=family,
                subtype_id=specimen.genome.subtype_id,
                role_id=role,
                anatomy=anatomy,
                genes=fields.genes,
                corpus_seed=specimen.genome.seed,
            )
            assert validate_binding(first) == []
            assert first.sha256 == second.sha256
            assert first.manifest == second.manifest
            assert first.manifest["format"] == BINDING_FORMAT
            assert first.manifest["graph"]["connected"] is True
            assert {node["id"] for node in first.manifest["graph"]["nodes"]} == set(
                DRIVER_NAMES
            )
            assert all(
                np.array_equal(expected, actual)
                for expected, actual in zip(
                    (
                        fields.part_owner,
                        fields.material,
                        fields.emission_level,
                    ),
                    first.reconstruct_fields(),
                    strict=True,
                )
            )
            assert first.part_owner.flags.writeable is False
            assert first.driver_index.flags.writeable is False
            source_tuples = _field_tuples(
                first.part_owner, first.material, first.emission_level
            )
            for facing in FACING_NAMES:
                contract = motion_adapter_contract(first, facing=facing)
                assert contract == motion_adapter_contract(first, facing=facing)
                assert contract["facing"] == facing
                frame = render_facing_frame(first, facing)
                assert _field_tuples(
                    frame.part_owner, frame.material, frame.emission_level
                ) <= source_tuples
                assert frame.manifest["tuples_preserved"] is True
                seen_facings.add(facing)
            hashes.add(first.sha256)
            seen_conditions.add((family, role))
    assert len(seen_conditions) == len(FAMILIES) * len(ROLE_NAMES)
    assert seen_facings == set(FACING_NAMES)
    assert len(hashes) == len(seen_conditions)


def test_rest_adapter_is_bit_exact_and_nonidentity_copies_only_source_tuples() -> None:
    _, fields, _, binding = _fixture("plantlike", 4)
    rest = render_bound_pose(binding)
    assert np.array_equal(rest.part_owner, fields.part_owner)
    assert np.array_equal(rest.material, fields.material)
    assert np.array_equal(rest.emission_level, fields.emission_level)
    transform = np.asarray(
        ((1.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    posed = render_bound_pose(
        binding,
        {name: transform for name in DRIVER_NAMES},
        enforce_margin=False,
    )
    assert _field_tuples(
        posed.part_owner, posed.material, posed.emission_level
    ) <= _field_tuples(
        binding.part_owner, binding.material, binding.emission_level
    )


def test_replay_is_exact_and_detects_source_mutation() -> None:
    _, fields, _, binding = _fixture("animalian", 2)
    report = replay_binding(
        binding,
        fields.part_owner,
        fields.material,
        fields.emission_level,
        fields.guide,
        genes=fields.genes,
    )
    assert report["status"] == "exact"
    assert all(report["checks"].values())
    assert_exact_replay(report)

    corrupt = fields.material.copy()
    y, x = map(int, np.argwhere(fields.part_owner == 1)[0])
    corrupt[y, x] = 0
    mismatch = replay_binding(
        binding,
        fields.part_owner,
        corrupt,
        fields.emission_level,
        fields.guide,
        genes=fields.genes,
    )
    assert mismatch["status"] == "mismatch"
    assert mismatch["actual_binding_sha256"] is None


def test_anomaly_visual_island_gets_orbital_logical_attachment() -> None:
    specimen, fields, anatomy, _ = _fixture("anomaly", 3)
    part, material, emission = _add_disconnected_body_tuple(fields)
    binding = bind_neural_fields(
        part,
        material,
        emission,
        fields.guide,
        family="anomaly",
        subtype_id=specimen.genome.subtype_id,
        role_id=3,
        anatomy=anatomy,
        corpus_seed=specimen.genome.seed,
    )
    topology = binding.manifest["topology"]
    assert topology["physical_component_count"] == 2
    assert topology["logical_graph_connected"] is True
    assert topology["components"][1]["joint_type"] == "orbital"
    assert topology["components"][1]["logical_parent"] == "body"
    assert validate_binding(binding) == []


def test_non_anomaly_island_margin_tuple_and_ownership_fail_closed() -> None:
    specimen, fields, anatomy, _ = _fixture("humanoid", 2)
    part, material, emission = _add_disconnected_body_tuple(fields)
    with pytest.raises(BindingRejected, match="exactly one is required"):
        bind_neural_fields(
            part,
            material,
            emission,
            fields.guide,
            family="humanoid",
            subtype_id=specimen.genome.subtype_id,
            role_id=2,
            anatomy=anatomy,
        )

    part = fields.part_owner.copy()
    material = fields.material.copy()
    emission = fields.emission_level.copy()
    source_y, source_x = map(int, np.argwhere(part == 1)[0])
    part[1, 1] = 1
    material[1, 1] = material[source_y, source_x]
    emission[1, 1] = emission[source_y, source_x]
    with pytest.raises(BindingRejected, match="violate the 3-pixel margin"):
        bind_neural_fields(
            part,
            material,
            emission,
            fields.guide,
            family="humanoid",
            subtype_id=specimen.genome.subtype_id,
            role_id=2,
            anatomy=anatomy,
        )

    bad_material = fields.material.copy()
    bad_material[source_y, source_x] = 0
    with pytest.raises(BindingRejected, match="illegal aligned field tuples"):
        bind_neural_fields(
            fields.part_owner,
            bad_material,
            fields.emission_level,
            fields.guide,
            family="humanoid",
            subtype_id=specimen.genome.subtype_id,
            role_id=2,
            anatomy=anatomy,
        )

    no_head = fields.part_owner.copy()
    no_head_material = fields.material.copy()
    no_head_emission = fields.emission_level.copy()
    removed = no_head == 3
    no_head[removed] = 0
    no_head_material[removed] = 0
    no_head_emission[removed] = 0
    with pytest.raises(BindingRejected, match="required head owner is absent"):
        bind_neural_fields(
            no_head,
            no_head_material,
            no_head_emission,
            fields.guide,
            family="humanoid",
            subtype_id=specimen.genome.subtype_id,
            role_id=2,
            anatomy=anatomy,
        )


def test_scaffold_only_automatic_anchor_derivation_is_deterministic() -> None:
    specimen, fields, _, _ = _fixture("machine", 2)
    first = bind_neural_fields(
        fields.part_owner,
        fields.material,
        fields.emission_level,
        fields.guide,
        family="machine",
        subtype_id=specimen.genome.subtype_id,
        role_id=2,
    )
    second = bind_neural_fields(
        fields.part_owner,
        fields.material,
        fields.emission_level,
        fields.guide,
        family="machine",
        subtype_id=specimen.genome.subtype_id,
        role_id=2,
    )
    assert first.sha256 == second.sha256
    assert first.anatomy.source == "guide_owner_geometry-v1"
    assert validate_binding(first) == []


def test_raw_multifield_archive_bridge_verifies_manifest_provenance(tmp_path: Path) -> None:
    specimen, fields, _, reference = _fixture("humanoid", 2, salt=11)
    archive_path, manifest_path, manifest = _write_raw_pair(
        tmp_path, specimen, fields, reference.legal_tuples
    )
    binding = bind_raw_sample_archive(
        archive_path,
        raw_manifest_path=manifest_path,
        legal_tuples=reference.legal_tuples,
    )
    assert binding.sample_id == "raw_fixture_sample"
    assert binding.raw_fields_sha256 == manifest["raw_fields_sha256"]
    assert binding.manifest["source"]["procedural_pixel_substitution"] is False
    assert binding.anatomy.source == "conditioned_anatomy_points_v1"
    assert validate_binding(binding) == []

    manifest["raw_fields_sha256"] = "b" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BindingRejected, match="field hash disagrees"):
        bind_raw_sample_archive(
            archive_path,
            raw_manifest_path=manifest_path,
            legal_tuples=reference.legal_tuples,
        )


def test_conditioned_anatomy_contains_points_not_procedural_layers() -> None:
    specimen, _, _, _ = _fixture("plantlike", 6)
    anatomy = derive_conditioned_anatomy(
        specimen.genome.seed,
        family="plantlike",
        subtype_id=specimen.genome.subtype_id,
        role_id=6,
    )
    assert set(anatomy.joints) == set(specimen.joints)
    assert set(anatomy.sockets) == set(specimen.sockets)
    assert not hasattr(anatomy, "layers")
    assert not hasattr(anatomy, "tokens")
    assert len(anatomy.source_sha256 or "") == 64


def test_raw_archive_admission_rejects_missing_manifest_traversal_and_swaps(
    tmp_path: Path,
) -> None:
    specimen, fields, _, reference = _fixture("animalian", 3, salt=21)

    archive, manifest_path, manifest = _write_raw_pair(
        tmp_path / "missing", specimen, fields, reference.legal_tuples
    )
    with pytest.raises(BindingRejected, match="raw_manifest_path is required"):
        bind_raw_sample_archive(archive, legal_tuples=reference.legal_tuples)

    archive, manifest_path, manifest = _write_raw_pair(
        tmp_path / "traversal", specimen, fields, reference.legal_tuples
    )
    manifest["artifacts"]["fields"]["path"] = "../escape.npz"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BindingRejected, match="artifact path is unsafe"):
        bind_raw_sample_archive(
            archive,
            raw_manifest_path=manifest_path,
            legal_tuples=reference.legal_tuples,
        )

    archive, manifest_path, manifest = _write_raw_pair(
        tmp_path / "tuples", specimen, fields, reference.legal_tuples
    )
    manifest["legal_tuple_fingerprint"] = "b" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BindingRejected, match="legal tuple fingerprint"):
        bind_raw_sample_archive(
            archive,
            raw_manifest_path=manifest_path,
            legal_tuples=reference.legal_tuples,
        )

    archive, manifest_path, manifest = _write_raw_pair(
        tmp_path / "unaccepted", specimen, fields, reference.legal_tuples
    )
    manifest["validation"]["accepted"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BindingRejected, match="not accepted and hard-valid"):
        bind_raw_sample_archive(
            archive,
            raw_manifest_path=manifest_path,
            legal_tuples=reference.legal_tuples,
        )

    archive, manifest_path, manifest = _write_raw_pair(
        tmp_path / "condition", specimen, fields, reference.legal_tuples
    )
    manifest["condition"]["role_id"] = (specimen.genome.role_id + 1) % len(
        ROLE_NAMES
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BindingRejected, match="condition.role_id disagrees"):
        bind_raw_sample_archive(
            archive,
            raw_manifest_path=manifest_path,
            legal_tuples=reference.legal_tuples,
        )


def test_raw_npz_preflight_rejects_extra_duplicate_shape_and_endian_members(
    tmp_path: Path,
) -> None:
    specimen, fields, _, reference = _fixture("plantlike", 5, salt=31)

    archive, manifest_path, _ = _write_raw_pair(
        tmp_path / "extra", specimen, fields, reference.legal_tuples
    )
    with np.load(archive, allow_pickle=False) as loaded:
        arrays = {name: np.asarray(loaded[name]).copy() for name in loaded.files}
    arrays["rogue"] = np.zeros((1,), dtype=np.uint8)
    np.savez_compressed(archive, **arrays)
    with pytest.raises(BindingRejected, match="member count is not exact"):
        bind_raw_sample_archive(
            archive,
            raw_manifest_path=manifest_path,
            legal_tuples=reference.legal_tuples,
        )

    archive, manifest_path, _ = _write_raw_pair(
        tmp_path / "duplicate", specimen, fields, reference.legal_tuples
    )
    with zipfile.ZipFile(archive, mode="a") as packed:
        part_member = packed.read("part.npy")
        with pytest.warns(UserWarning, match="Duplicate name"):
            packed.writestr("part.npy", part_member)
    with pytest.raises(BindingRejected, match="member count is not exact|duplicate"):
        bind_raw_sample_archive(
            archive,
            raw_manifest_path=manifest_path,
            legal_tuples=reference.legal_tuples,
        )

    archive, manifest_path, _ = _write_raw_pair(
        tmp_path / "shape", specimen, fields, reference.legal_tuples
    )
    with np.load(archive, allow_pickle=False) as loaded:
        arrays = {name: np.asarray(loaded[name]).copy() for name in loaded.files}
    arrays["role"] = np.asarray([specimen.genome.role_id, 0], dtype=np.uint8)
    np.savez_compressed(archive, **arrays)
    with pytest.raises(BindingRejected, match="role shape"):
        bind_raw_sample_archive(
            archive,
            raw_manifest_path=manifest_path,
            legal_tuples=reference.legal_tuples,
        )

    archive, manifest_path, _ = _write_raw_pair(
        tmp_path / "endian", specimen, fields, reference.legal_tuples
    )
    with np.load(archive, allow_pickle=False) as loaded:
        arrays = {name: np.asarray(loaded[name]).copy() for name in loaded.files}
    arrays["corpus_seed"] = arrays["corpus_seed"].astype(">u4")
    np.savez_compressed(archive, **arrays)
    with pytest.raises(BindingRejected, match="corpus_seed dtype"):
        bind_raw_sample_archive(
            archive,
            raw_manifest_path=manifest_path,
            legal_tuples=reference.legal_tuples,
        )


def test_anatomy_provenance_and_bounded_legal_table_fail_closed() -> None:
    specimen, fields, anatomy, binding = _fixture("machine", 1, salt=41)
    swapped_joints = dict(anatomy.joints)
    swapped_joints["left_shoulder"], swapped_joints["right_shoulder"] = (
        swapped_joints["right_shoulder"],
        swapped_joints["left_shoulder"],
    )
    forged = DerivedAnatomy.from_mappings(
        swapped_joints,
        anatomy.sockets,
        source=anatomy.source,
        source_sha256=anatomy.source_sha256,
    )
    with pytest.raises(BindingRejected, match="declared deterministic source"):
        bind_neural_fields(
            fields.part_owner,
            fields.material,
            fields.emission_level,
            fields.guide,
            family="machine",
            subtype_id=specimen.genome.subtype_id,
            role_id=1,
            corpus_seed=specimen.genome.seed,
            anatomy=forged,
        )
    with pytest.raises(ValueError, match=r"must be an \(x, y\) pair"):
        DerivedAnatomy.from_mappings(
            {"broken": [1]},
            {},
            source="conditioned_anatomy_points_v1",
            source_sha256="a" * 64,
        )
    with pytest.raises(BindingRejected, match="unsigned 32-bit"):
        derive_conditioned_anatomy(
            -1,
            family="machine",
            subtype_id=specimen.genome.subtype_id,
            role_id=1,
        )
    oversized = np.zeros((681, 3), dtype=np.uint8)
    with pytest.raises(BindingRejected, match="680-row semantic bound"):
        bind_neural_fields(
            binding.part_owner,
            binding.material,
            binding.emission_level,
            binding.guide,
            family="machine",
            subtype_id=binding.subtype_id,
            role_id=binding.role_id,
            corpus_seed=binding.corpus_seed,
            anatomy=binding.anatomy,
            legal_tuples=oversized,
        )


def test_validation_and_motion_entrypoints_reject_forged_derived_state() -> None:
    _, _, _, binding = _fixture("humanoid", 6, salt=51)

    overlapping = binding.owner_masks.copy()
    y, x = map(int, np.argwhere(binding.part_owner == 1)[0])
    overlapping[1, y, x] = 1
    bad_masks = replace(binding, owner_masks=overlapping)
    assert any("owner_masks" in error for error in validate_binding(bad_masks))
    with pytest.raises(BindingRejected, match="owner_masks"):
        render_bound_pose(bad_masks)

    bad_joints = dict(binding.joints)
    bad_joints["root"] = replace(
        binding.joints["root"], support_point=(999, 999)
    )
    bad_anchor = replace(binding, joints=bad_joints)
    anchor_errors = validate_binding(bad_anchor)
    assert any("support is outside" in error for error in anchor_errors)
    with pytest.raises(BindingRejected):
        motion_adapter_contract(bad_anchor)

    forged_manifest = deepcopy(binding.manifest)
    forged_manifest["graph"]["nodes"][0]["pivot"] = [4, 4]
    payload = dict(forged_manifest)
    payload.pop("hashes")
    forged_manifest["hashes"] = {"binding_sha256": canonical_json_hash(payload)}
    bad_manifest = replace(binding, manifest=forged_manifest)
    assert any(
        "exact deterministic binding projection" in error
        for error in validate_binding(bad_manifest)
    )
    with pytest.raises(BindingRejected, match="deterministic binding projection"):
        motion_adapter_contract(bad_manifest)


def test_replay_assertion_recomputes_hash_and_all_checks() -> None:
    _, fields, _, binding = _fixture("animalian", 7, salt=61)
    report = replay_binding(
        binding,
        fields.part_owner,
        fields.material,
        fields.emission_level,
        fields.guide,
        genes=fields.genes,
    )
    assert_exact_replay(report)
    with pytest.raises(BindingRejected, match="keys are not exact"):
        assert_exact_replay({"format": REPLAY_FORMAT, "status": "exact"})

    tampered = deepcopy(report)
    tampered["checks"]["owner_masks_exact"] = False
    payload = dict(tampered)
    payload.pop("report_sha256")
    tampered["report_sha256"] = canonical_json_hash(payload)
    with pytest.raises(BindingRejected, match="checks failed"):
        assert_exact_replay(tampered)

    stale_hash = deepcopy(report)
    stale_hash["actual_binding_sha256"] = "b" * 64
    with pytest.raises(BindingRejected, match="hashes|report SHA-256"):
        assert_exact_replay(stale_hash)


def test_affine_domain_is_bounded_and_facing_matrices_are_independent() -> None:
    _, _, _, binding = _fixture("plantlike", 0, salt=71)
    with pytest.raises(BindingRejected, match="numeric 3x3"):
        render_bound_pose(binding, {"body": np.eye(3, dtype=bool)})
    near_singular = np.asarray(
        ((0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 1.0))
    )
    with pytest.raises(BindingRejected, match="singular"):
        render_bound_pose(binding, {"body": near_singular})
    projective = np.eye(3)
    projective[2, 0] = 0.1
    with pytest.raises(BindingRejected, match="2D affine"):
        render_bound_pose(binding, {"body": projective})

    north = facing_transforms(binding, "north")
    east = facing_transforms(binding, "east")
    assert all(np.array_equal(matrix, np.eye(3)) for matrix in north.values())
    assert not np.array_equal(east["body"], np.eye(3))
    assert all(np.array_equal(east["body"], east[name]) for name in DRIVER_NAMES)
    assert all(
        not np.shares_memory(east["body"], east[name])
        for name in DRIVER_NAMES
        if name != "body"
    )


def test_anomaly_component_count_and_dominance_loopholes_are_closed() -> None:
    specimen, fields, anatomy, _ = _fixture("anomaly", 4, salt=81)
    physical = (fields.part_owner != 0) & (fields.part_owner != 16)
    padded = np.pad(physical, 1)
    nearby = np.logical_or.reduce(
        [padded[y : y + 48, x : x + 48] for y in range(3) for x in range(3)]
    )
    safe = np.zeros((48, 48), dtype=bool)
    safe[SAFETY_MARGIN : 48 - SAFETY_MARGIN, SAFETY_MARGIN : 48 - SAFETY_MARGIN] = True
    candidates = np.argwhere((~nearby) & safe)
    selected: list[tuple[int, int]] = []
    for y, x in candidates:
        point = (int(y), int(x))
        if all(max(abs(point[0] - py), abs(point[1] - px)) > 1 for py, px in selected):
            selected.append(point)
        if len(selected) == 3:
            break
    assert len(selected) == 3
    part = fields.part_owner.copy()
    material = fields.material.copy()
    emission = fields.emission_level.copy()
    source_y, source_x = map(int, np.argwhere(part == 1)[0])
    for y, x in selected:
        part[y, x] = 1
        material[y, x] = material[source_y, source_x]
        emission[y, x] = emission[source_y, source_x]
    with pytest.raises(BindingRejected, match="maximum is 3"):
        bind_neural_fields(
            part,
            material,
            emission,
            fields.guide,
            family="anomaly",
            subtype_id=specimen.genome.subtype_id,
            role_id=4,
            anatomy=anatomy,
            corpus_seed=specimen.genome.seed,
        )


    target_count = int(np.ceil(int(physical.sum()) * 0.20))
    allowed = (~nearby) & safe
    start_y, start_x = map(int, np.argwhere(allowed)[0])
    queue = [(start_y, start_x)]
    visited = {(start_y, start_x)}
    island: list[tuple[int, int]] = []
    while queue and len(island) < target_count:
        y, x = queue.pop(0)
        island.append((y, x))
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if not (dx or dy):
                    continue
                neighbor = (y + dy, x + dx)
                if (
                    0 <= neighbor[0] < 48
                    and 0 <= neighbor[1] < 48
                    and allowed[neighbor]
                    and neighbor not in visited
                ):
                    visited.add(neighbor)
                    queue.append(neighbor)
    assert len(island) == target_count
    part = fields.part_owner.copy()
    material = fields.material.copy()
    emission = fields.emission_level.copy()
    for y, x in island:
        part[y, x] = 1
        material[y, x] = material[source_y, source_x]
        emission[y, x] = emission[source_y, source_x]
    with pytest.raises(BindingRejected, match="dominant component fraction"):
        bind_neural_fields(
            part,
            material,
            emission,
            fields.guide,
            family="anomaly",
            subtype_id=specimen.genome.subtype_id,
            role_id=4,
            anatomy=anatomy,
            corpus_seed=specimen.genome.seed,
        )


@pytest.mark.parametrize(
    ("family_index", "family"), tuple(enumerate(FAMILIES)), ids=FAMILIES
)
def test_shared_motion_program_animates_every_family_and_motion_exactly(
    family_index: int, family: str
) -> None:
    _, _, _, binding = _fixture(
        family, family_index % len(ROLE_NAMES), salt=101 + family_index
    )
    source_tuples = _field_tuples(
        binding.part_owner, binding.material, binding.emission_level
    )
    for motion_index, motion in enumerate(MOTION_NAMES):
        facing = FACING_NAMES[(family_index * 3 + motion_index) % len(FACING_NAMES)]
        clip = compile_neural_motion_clip(binding, motion, facing=facing)
        assert validate_neural_motion_clip(clip) == []
        assert clip.manifest["binding_sha256"] == binding.sha256
        assert clip.manifest["source_raw_fields_sha256"] == binding.raw_fields_sha256
        assert clip.manifest["metrics"]["procedural_pixel_substitution"] is False
        assert clip.manifest["metrics"]["all_source_tuples_preserved"] is True
        assert len(clip.frames) == clip.manifest["frame_count"]
        assert all(
            _field_tuples(
                frame.fields.part_owner,
                frame.fields.material,
                frame.fields.emission_level,
            )
            <= source_tuples
            for frame in clip.frames
        )
        report = replay_neural_motion_clip(clip)
        assert_exact_neural_motion_replay(report)


def test_neural_motion_all_facings_and_tamper_detection() -> None:
    _, _, _, binding = _fixture("animalian", 6, salt=207)
    hashes: set[str] = set()
    for facing in FACING_NAMES:
        clip = compile_neural_motion_clip(binding, "locomote", facing=facing)
        assert validate_neural_motion_clip(clip) == []
        assert_exact_neural_motion_replay(replay_neural_motion_clip(clip))
        hashes.add(clip.sha256)
    assert len(hashes) == len(FACING_NAMES)

    clip = compile_neural_motion_clip(binding, "attack", facing="southeast")
    forged_manifest = deepcopy(dict(clip.manifest))
    forged_manifest["frames"][0]["motion_frame_sha256"] = "0" * 64
    forged = replace(clip, manifest=forged_manifest)
    errors = validate_neural_motion_clip(forged)
    assert any("canonical" in error for error in errors)
    report = replay_neural_motion_clip(forged)
    assert report["passed"] is False
    with pytest.raises(ValueError, match="not exact"):
        assert_exact_neural_motion_replay(report)

    source = clip.frames[0]
    tampered_part = source.fields.part_owner.copy()
    y, x = map(int, np.argwhere(tampered_part != 0)[0])
    tampered_part[y, x] = 0
    tampered_fields = replace(source.fields, part_owner=tampered_part)
    authority_errors = validate_bound_frame(binding, tampered_fields)
    assert any(
        "hash mismatch" in error
        or "projection" in error
        or "foreground and driver" in error
        for error in authority_errors
    )
    with pytest.raises(BindingRejected):
        assert_valid_bound_frame(binding, tampered_fields)
    tampered_frame = replace(source, fields=tampered_fields)
    tampered_clip = replace(clip, frames=(tampered_frame,) + clip.frames[1:])
    clip_errors = validate_neural_motion_clip(tampered_clip)
    assert any("bound authority" in error for error in clip_errors)
