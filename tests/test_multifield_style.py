from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from forge.morphology.constants import FAMILIES, MATERIAL_NAMES, ROLE_NAMES, SUBTYPE_NAMES
from forge.multifield_style.backgrounds import load_background_crops
from forge.multifield_style.color import delta_e_oklab
from forge.multifield_style.compiler import (
    compile_generation_bank,
    compiler_source_hash,
    load_style_manifest,
)
from forge.multifield_style.hashing import aligned_fields_hash, sha256_file
from forge.multifield_style.metrics import evaluate_style
from forge.multifield_style.model import CategoricalFields, StyleCondition
from forge.multifield_style.palette import palette_for_condition
from forge.multifield_style.procedural import (
    _load_procedural_source,
    _select_family_references,
    compile_procedural_reference_bank,
    load_procedural_reference_manifest,
    replay_procedural_reference_bank,
)
from forge.multifield_style.rendering import chebyshev_ring, render_layers
from forge.multifield_style.replay import replay_style_bank
from forge.multifield_style.schema import (
    PROCEDURAL_REFERENCE_SCHEMA,
    STYLE_BANK_SCHEMA,
    validate_schema,
)
from forge.multifield_style.source import load_generation_bank


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEURAL_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "multifield_generation"
    / "milestone_epoch020_cpu5_calibrated_final"
    / "generation_manifest.json"
)
PROCEDURAL_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "morphology_prototype"
    / "morphology_prototype_manifest.json"
)
MAP_ART_ROOT = PROJECT_ROOT / "outputs" / "map_art" / "packs"


@pytest.fixture(scope="module")
def neural_bank():
    return load_generation_bank(NEURAL_MANIFEST)


@pytest.fixture(scope="module")
def backgrounds():
    return load_background_crops(MAP_ART_ROOT)


def _procedural_cases():
    manifest, _, arrays = _load_procedural_source(PROCEDURAL_MANIFEST)
    cases = []
    for ordinal, index in enumerate(_select_family_references(manifest, arrays)):
        morphology_id = int(arrays["morphologies"][index])
        subtype_id = int(arrays["subtypes"][index])
        role_id = int(arrays["roles"][index])
        fields_hash = aligned_fields_hash(
            arrays["part_owner"][index],
            arrays["material"][index],
            arrays["emission_level"][index],
        )
        fields = CategoricalFields(
            part=np.array(arrays["part_owner"][index], copy=True),
            material=np.array(arrays["material"][index], copy=True),
            emission=np.array(arrays["emission_level"][index], copy=True),
            aligned_sha256=fields_hash,
        )
        condition = StyleCondition(
            sample_id=f"procedural_ref_{ordinal}_{FAMILIES[morphology_id]}",
            ordinal=ordinal,
            sample_seed=int(arrays["seeds"][index]),
            morphology_id=morphology_id,
            morphology_name=FAMILIES[morphology_id],
            subtype_id=subtype_id,
            subtype_name=SUBTYPE_NAMES[subtype_id],
            role_id=role_id,
            role_name=ROLE_NAMES[role_id],
        )
        cases.append((condition, fields))
    return cases


def test_source_loader_binds_immutable_accepted_fields(neural_bank) -> None:
    assert neural_bank.manifest["format"] == "nullvector-multifield-generation-bank-v1"
    assert len(neural_bank.samples) == 5
    for sample in neural_bank.samples:
        assert sample.fields.aligned_sha256 == aligned_fields_hash(
            sample.fields.part,
            sample.fields.material,
            sample.fields.emission,
        )
        assert not sample.fields.part.flags.writeable
        assert not sample.fields.material.flags.writeable
        assert not sample.fields.emission.flags.writeable
        assert sample.condition.morphology_name == "humanoid"


def test_oklch_palette_is_deterministic_and_materials_are_distinct(neural_bank) -> None:
    sample = neural_bank.samples[0]
    first = palette_for_condition(sample.condition, sample.fields.aligned_sha256)
    second = palette_for_condition(sample.condition, sample.fields.aligned_sha256)
    assert first == second
    assert tuple(first["materials"]) == MATERIAL_NAMES
    mids = [first["materials"][name]["mid"] for name in MATERIAL_NAMES]
    assert len({tuple(color) for color in mids}) == 10
    assert min(
        delta_e_oklab(mids[a], mids[b])
        for a in range(10)
        for b in range(a + 1, 10)
    ) >= 0.045
    lightness = first["diagnostics"]["emission_oklab_lightness"]
    assert lightness[0] < lightness[1] < lightness[2]


def test_palette_contract_across_every_family_subtype_and_role() -> None:
    fields_hash = "42" * 32
    for family_id, family_name in enumerate(FAMILIES):
        for variant in range(4):
            subtype_id = family_id * 4 + variant
            for role_id, role_name in enumerate(ROLE_NAMES):
                condition = StyleCondition(
                    sample_id=f"matrix_{family_id}_{variant}_{role_id}",
                    ordinal=0,
                    sample_seed=family_id * 100 + variant * 10 + role_id,
                    morphology_id=family_id,
                    morphology_name=family_name,
                    subtype_id=subtype_id,
                    subtype_name=SUBTYPE_NAMES[subtype_id],
                    role_id=role_id,
                    role_name=role_name,
                )
                palette = palette_for_condition(condition, fields_hash)
                mids = [palette["materials"][name]["mid"] for name in MATERIAL_NAMES]
                assert len({tuple(color) for color in mids}) == 10
                assert palette["diagnostics"]["minimum_material_mid_delta_e_oklab"] >= 0.045
                lightness = palette["diagnostics"]["emission_oklab_lightness"]
                assert min(np.diff(lightness)) >= 0.055


def test_neural_and_five_family_render_gates(neural_bank, backgrounds) -> None:
    cases = [(sample.condition, sample.fields) for sample in neural_bank.samples]
    procedural_cases = _procedural_cases()
    cases.extend(procedural_cases)
    assert {condition.morphology_name for condition, _ in procedural_cases} == set(FAMILIES)
    for condition, fields in cases:
        before = fields.aligned_sha256
        rendered = render_layers(fields, condition)
        after = aligned_fields_hash(fields.part, fields.material, fields.emission)
        metrics = evaluate_style(
            fields,
            condition,
            rendered,
            backgrounds,
            fields_hash_after_render=after,
        )
        assert after == before
        assert metrics["passed"], {
            name: passed for name, passed in metrics["gates"].items() if not passed
        }
        assert rendered.base.shape == (48, 48, 4)
        assert set(np.unique(rendered.base[..., 3])).issubset({0, 255})
        assert np.array_equal(rendered.outline[..., 3] > 0, chebyshev_ring(rendered.masks["body"], 1))
        assert not np.any((rendered.aura[..., 3] > 0) & rendered.masks["body"])
        active_aura_alpha = rendered.aura[..., 3][rendered.aura[..., 3] > 0]
        assert active_aura_alpha.size == 0 or int(active_aura_alpha.max()) < 255


def test_compile_is_byte_deterministic_and_strictly_replays(tmp_path: Path) -> None:
    first_root, second_root = tmp_path / "bank_a", tmp_path / "bank_b"
    first = compile_generation_bank(NEURAL_MANIFEST, first_root, map_art_root=MAP_ART_ROOT)
    second = compile_generation_bank(NEURAL_MANIFEST, second_root, map_art_root=MAP_ART_ROOT)
    assert first == second
    assert (first_root / "style_manifest.json").read_bytes() == (
        second_root / "style_manifest.json"
    ).read_bytes()
    for first_entry, second_entry in zip(first["samples"], second["samples"]):
        for name in first_entry["presentation"]["artifacts"]:
            first_record = first_entry["presentation"]["artifacts"][name]
            second_record = second_entry["presentation"]["artifacts"][name]
            assert first_record == second_record
            assert (first_root / first_record["path"]).read_bytes() == (
                second_root / second_record["path"]
            ).read_bytes()
    validate_schema(load_style_manifest(first_root / "style_manifest.json"), STYLE_BANK_SCHEMA)
    replay = replay_style_bank(first_root / "style_manifest.json", map_art_root=MAP_ART_ROOT)
    assert replay["passed"], replay["errors"]
    assert all(all(sample["artifact_bytes_exact"].values()) for sample in replay["samples"])
    with pytest.raises(FileExistsError):
        compile_generation_bank(NEURAL_MANIFEST, first_root, map_art_root=MAP_ART_ROOT)


def test_replay_fails_closed_on_artifact_and_manifest_tampering(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    compile_generation_bank(NEURAL_MANIFEST, source_root, map_art_root=MAP_ART_ROOT)

    artifact_tamper = tmp_path / "artifact_tamper"
    shutil.copytree(source_root, artifact_tamper)
    artifact_path = artifact_tamper / "sprites" / "0000_f0_s00_r0_v00" / "palette.json"
    artifact_path.write_bytes(artifact_path.read_bytes() + b" ")
    replay = replay_style_bank(artifact_tamper / "style_manifest.json", map_art_root=MAP_ART_ROOT)
    assert not replay["passed"]
    assert "mismatch" in " ".join(replay["errors"])

    manifest_tamper = tmp_path / "manifest_tamper"
    shutil.copytree(source_root, manifest_tamper)
    path = manifest_tamper / "style_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["samples"][0]["presentation"]["artifacts"]["base"]["path"] = "../escape.png"
    path.write_text(json.dumps(payload), encoding="utf-8")
    replay = replay_style_bank(path, map_art_root=MAP_ART_ROOT)
    assert not replay["passed"]
    assert "schema" in " ".join(replay["errors"]).lower()


def test_source_loader_rejects_tampered_or_extra_npz_members(tmp_path: Path) -> None:
    copied_root = tmp_path / "generation"
    shutil.copytree(NEURAL_MANIFEST.parent, copied_root)
    manifest_path = copied_root / "generation_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = payload["samples"][0]["compiled_artifacts"]["fields"]
    field_path = copied_root / record["path"]
    with np.load(field_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays["unexpected"] = np.zeros((1,), dtype=np.uint8)
    np.savez_compressed(field_path, **arrays)
    record["bytes"] = field_path.stat().st_size
    record["sha256"] = sha256_file(field_path)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="member mismatch"):
        load_generation_bank(manifest_path)


def test_procedural_reference_is_explicitly_non_neural_and_replays(tmp_path: Path) -> None:
    destination = tmp_path / "procedural_reference"
    manifest = compile_procedural_reference_bank(
        PROCEDURAL_MANIFEST,
        destination,
        map_art_root=MAP_ART_ROOT,
    )
    assert manifest["neural_output"] is False
    assert manifest["source_kind"] == "authoritative-procedural-reference"
    assert manifest["authority"]["neural_output_claimed"] is False
    assert manifest["family_coverage"] == list(FAMILIES)
    assert [sample["condition"]["morphology_name"] for sample in manifest["samples"]] == list(FAMILIES)
    loaded = load_procedural_reference_manifest(
        destination / "procedural_reference_manifest.json"
    )
    validate_schema(loaded, PROCEDURAL_REFERENCE_SCHEMA)
    replay = replay_procedural_reference_bank(
        destination / "procedural_reference_manifest.json",
        map_art_root=MAP_ART_ROOT,
    )
    assert replay["passed"], replay["errors"]


def test_schema_and_source_hash_are_stable_contracts() -> None:
    source_hash = compiler_source_hash()
    assert len(source_hash) == 64
    assert all(character in "0123456789abcdef" for character in source_hash)

    assert (PROJECT_ROOT / "shared" / "schema" / STYLE_BANK_SCHEMA).is_file()
    assert (PROJECT_ROOT / "shared" / "schema" / PROCEDURAL_REFERENCE_SCHEMA).is_file()
    with pytest.raises(ValueError, match="morphology name/id mismatch"):
        StyleCondition(
            sample_id="bad",
            ordinal=0,
            sample_seed=1,
            morphology_id=0,
            morphology_name="machine",
            subtype_id=0,
            subtype_name="humanoid_0",
            role_id=0,
            role_name="striker",
        ).validate()
