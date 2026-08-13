from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np

from forge.cellular_organism.compiler import _load_arrays
from forge.cellular_organism.contract import CellFlag
from forge.cellular_organism.simulation import OrganismState
from forge.evolved_cellular_organism import replay_bank, validate_bank
from forge.evolved_cellular_organism.compiler import _load_fields, _safe_artifact
from forge.map_decorator.hashing import json_sha256
from forge.multifield_style_motion.hashing import canonical_json_bytes
from forge.neural_fusion_production.contract import FUSION_MODES, MUTATION_MODES
from forge.neural_fusion_production_evolution import validate_production_evolution


ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "outputs/evolved_cellular_organism_v1/evolved_cellular_organism_manifest.json"
EVOLUTION = ROOT / "outputs/neural_fusion_production_evolution_v1_run2/production_evolution_manifest.json"


def _manifest() -> dict:
    assert BANK.is_file(), "compile the evolved cellular organism bank first"
    return json.loads(BANK.read_text(encoding="utf-8"))


def test_all_selected_neural_descendants_are_cellularized_and_exactly_replayable() -> None:
    validation = validate_bank(BANK)
    replay = replay_bank(BANK)
    assert validation["passed"] and replay["passed"]
    assert validation["sample_count"] == replay["sample_count"] == 36
    assert replay["artifact_count"] == 37
    assert replay["exact_artifact_replay"] is True


def test_lineage_covers_generations_families_fusion_and_mutation() -> None:
    manifest = _manifest()
    species = manifest["species"]
    assert manifest["generation_counts"] == {"1": 12, "2": 12, "3": 12}
    assert set(manifest["fusion_modes"]) == set(FUSION_MODES)
    assert set(manifest["mutation_modes"]) == set(MUTATION_MODES)
    for generation in range(1, 4):
        values = [item for item in species if item["lineage"]["generation"] == generation]
        assert len(values) == 12
        assert {item["family_id"] for item in values} == set(range(5))
        assert {item["lineage"]["fusion_mode"] for item in values} == set(FUSION_MODES)
        assert {item["lineage"]["mutation_mode"] for item in values} == set(MUTATION_MODES)
    assert all(item["genome"]["neural_lineage"]["lineage_sha256"] == item["lineage"]["lineage_sha256"] for item in species)


def test_each_neural_source_pixel_is_the_same_physical_cell_and_tuple() -> None:
    manifest = _manifest()
    evolution = validate_production_evolution(EVOLUTION)
    source_by_id = {item["specimen_id"]: item for item in evolution["selected"]}
    for record in manifest["species"]:
        source = source_by_id[record["sample_id"]]
        field_path = _safe_artifact(EVOLUTION.parent, source["artifacts"]["semantic_fields"], label="test fields")
        fields = _load_fields(field_path, source["fields_sha256"])
        anatomy = _load_arrays(BANK.parent / record["arrays"]["path"])
        physical = (fields["part_owner"] > 0) & (fields["part_owner"] != 16)
        yx = np.argwhere(physical)
        assert np.array_equal(anatomy["position_xy"], yx[:, ::-1].astype(np.int16))
        assert np.array_equal(anatomy["part_owner"], fields["part_owner"][physical])
        assert np.array_equal(anatomy["material"], fields["material"][physical])
        assert np.array_equal(anatomy["emission"], fields["emission_level"][physical])


def test_every_descendant_has_distinct_essential_organs_fluid_and_connected_bonds() -> None:
    manifest = _manifest()
    for record in manifest["species"]:
        arrays = _load_arrays(BANK.parent / record["arrays"]["path"])
        kinds = {organ["kind"] for organ in record["organs"]}
        assert {"circulatory", "neural", "digestive", "reproductive", "sensory"} <= kinds
        assert int(np.count_nonzero(arrays["cell_flags"] & int(CellFlag.EYE))) >= 1
        assert float(arrays["fluid_initial"].sum()) > 0.0
        assert len(arrays["bond_ab"]) >= len(arrays["position_xy"]) - 1


def test_evolved_genome_can_feed_bleed_break_and_reproduce_without_redecoding_shape() -> None:
    record = _manifest()["species"][0]
    arrays = _load_arrays(BANK.parent / record["arrays"]["path"])
    state = OrganismState(arrays, record["genome"])
    assert state.feed(30.0) == 30.0
    state.energy += np.float32(20.0)
    assert state.can_reproduce()
    event = state.reproduce(0xE7010ED)
    assert event.parent_generation == record["lineage"]["generation"]
    assert event.child_generation == record["lineage"]["generation"] + 1
    assert event.child_genome["neural_lineage"] == record["genome"]["neural_lineage"]
    center = tuple(map(float, state.position.mean(axis=0)))
    fluid_before = float(state.fluid.sum())
    trauma = state.apply_damage(center, radius=36.0, damage=2.5, impulse=240.0)
    assert trauma["killed_cells"] > 0 and trauma["broken_bonds"] > 0
    for _ in range(12):
        state.step(1.0 / 60.0)
    assert float(state.fluid.sum()) < fluid_before


def test_stale_evolution_bank_is_not_accepted_as_authority() -> None:
    stale = ROOT / "outputs/neural_fusion_production_evolution_v1/production_evolution_manifest.json"
    assert stale.is_file()
    try:
        validate_production_evolution(stale)
    except ValueError as error:
        assert "source hash mismatch" in str(error)
    else:
        raise AssertionError("stale evolution authority unexpectedly validated")


def test_cellular_artifact_tamper_fails_closed(tmp_path: Path) -> None:
    copied = tmp_path / "bank"
    shutil.copytree(BANK.parent, copied)
    manifest_path = copied / BANK.name
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    anatomy = copied / manifest["species"][0]["arrays"]["path"]
    payload = bytearray(anatomy.read_bytes())
    payload[-1] ^= 0x01
    anatomy.write_bytes(payload)
    try:
        validate_bank(manifest_path)
    except ValueError as error:
        assert "SHA-256 differs" in str(error)
    else:
        raise AssertionError("tampered anatomy unexpectedly validated")


def test_fully_rehashed_lineage_tamper_still_fails_semantic_crosscheck(tmp_path: Path) -> None:
    copied = tmp_path / "bank"
    shutil.copytree(BANK.parent, copied)
    manifest_path = copied / BANK.name
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["species"][0]["lineage"]["mutation_strength"] += 1
    unsigned = dict(manifest)
    unsigned.pop("semantic_sha256")
    manifest["semantic_sha256"] = json_sha256(unsigned)
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    try:
        validate_bank(manifest_path)
    except ValueError as error:
        assert "genome and lineage differ" in str(error)
    else:
        raise AssertionError("rehashed lineage tamper unexpectedly validated")
