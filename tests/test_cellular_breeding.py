from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from forge.cellular_breeding import replay_bank, validate_bank
from forge.cellular_breeding.compiler import _load_fields
from forge.cellular_breeding.contract import CROSSOVER_MODES, MUTATION_MODES
from forge.cellular_organism.compiler import _load_arrays, validate_species_arrays
from forge.cellular_organism.simulation import OrganismState
from forge.map_decorator.hashing import json_sha256
from forge.multifield_style_motion.hashing import canonical_json_bytes, deterministic_npz_bytes, sha256_bytes


ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "outputs/cellular_breeding_v1/cellular_breeding_manifest.json"


def _manifest() -> dict[str, object]:
    assert BANK.is_file(), "compile the cellular breeding bank first"
    return json.loads(BANK.read_text(encoding="utf-8"))


def test_structural_breeding_bank_validates_and_exact_replays() -> None:
    validation = validate_bank(BANK)
    replay = replay_bank(BANK)
    assert validation["passed"] is True and validation["sample_count"] == 45
    assert replay["exact_replay"] is True and replay["artifact_count"] == 92
    assert replay["artifact_bytes"] == 1_256_872


def test_every_family_pair_and_operator_is_represented() -> None:
    manifest = _manifest()
    assert len(manifest["family_pair_counts"]) == 15
    assert set(manifest["family_pair_counts"].values()) == {3}
    assert set(manifest["crossover_modes"]) == set(CROSSOVER_MODES)
    assert set(manifest["mutation_modes"]) == set(MUTATION_MODES)
    assert set(manifest["family_counts"]) == {"humanoid", "animalian", "plantlike", "anomaly", "machine"}
    assert all(count >= 6 for count in manifest["family_counts"].values())


def test_offspring_inherit_both_parents_and_redecode_complete_anatomy() -> None:
    manifest = _manifest()
    for record in manifest["offspring"]:
        fields = _load_fields(BANK.parent / record["fields"]["path"])
        arrays = _load_arrays(BANK.parent / record["arrays"]["path"])
        validate_species_arrays(arrays, record["organs"], record["summary"])
        assert np.count_nonzero(fields["ancestry"] == 1) >= 8
        assert np.count_nonzero(fields["ancestry"] == 2) >= 8
        assert record["breeding"]["raster_connected"] is True
        assert record["genome"]["structural_lineage"]["parent_ids"] == record["lineage"]["parent_ids"]
        assert record["summary"]["eye_count"] >= 1
        assert float(arrays["fluid_initial"].sum()) > 0
        assert len(arrays["bond_ab"]) >= len(arrays["position_xy"]) - 1


def test_reference_offspring_can_bleed_feed_heal_and_reproduce() -> None:
    manifest = _manifest()
    for record in manifest["offspring"][::9]:
        arrays = _load_arrays(BANK.parent / record["arrays"]["path"])
        state = OrganismState(arrays, record["genome"])
        center = tuple(state.position.mean(axis=0).tolist())
        trauma = state.apply_damage(center, radius=35.0, damage=0.8, impulse=180.0)
        assert trauma["affected_cells"] > 0 and trauma["broken_bonds"] > 0
        fluid_before = float(state.fluid.sum())
        for _ in range(12):
            state.step(1.0 / 60.0)
        assert float(state.fluid.sum()) < fluid_before
        assert state.feed(20.0) == 20.0
        state.alive[:] = True
        state.health[:] = state.max_health
        state.bond_alive[:] = True
        threshold = float(record["genome"]["reproduction_energy_threshold"])
        state.energy[:] += np.float32((threshold + 5.0) / state.cell_count)
        event = state.reproduce(0xBEE000 + int(record["ordinal"]))
        assert event.child_generation == int(record["genome"]["generation"]) + 1
        assert event.child_genome["structural_lineage"] == record["genome"]["structural_lineage"]


def test_fully_rehashed_field_tamper_fails_deterministic_parent_replay(tmp_path: Path) -> None:
    copied = tmp_path / "bank"
    shutil.copytree(BANK.parent, copied)
    manifest_path = copied / BANK.name
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest["offspring"][0]
    fields_path = copied / record["fields"]["path"]
    fields = _load_fields(fields_path)
    physical = np.argwhere(fields["part_owner"] > 0)
    y, x = map(int, physical[0])
    fields["emission"][y, x] = np.uint8((int(fields["emission"][y, x]) + 1) % 4)
    payload = deterministic_npz_bytes(fields)
    fields_path.write_bytes(payload)
    record["fields"] = {"path": record["fields"]["path"], "bytes": len(payload), "sha256": sha256_bytes(payload)}
    record["offspring_fields_sha256"] = record["fields"]["sha256"]
    manifest["semantic_sha256"] = json_sha256({key: value for key, value in manifest.items() if key != "semantic_sha256"})
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(ValueError, match="deterministic field replay differs"):
        validate_bank(manifest_path)
