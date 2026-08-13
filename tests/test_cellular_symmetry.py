from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from forge.cellular_breeding.compiler import _load_fields as load_breeding_fields
from forge.cellular_symmetry import replay_bank, validate_bank
from forge.cellular_symmetry.compiler import _load_fields
from forge.cellular_organism.compiler import _load_arrays, validate_species_arrays


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/cellular_breeding_v1/cellular_breeding_manifest.json"
BANK = ROOT / "outputs/cellular_breeding_symmetry_v1/cellular_symmetry_manifest.json"


def _manifest() -> dict[str, object]:
    return json.loads(BANK.read_text(encoding="utf-8"))


def test_symmetry_bank_validates_and_exact_replays() -> None:
    validation = validate_bank(BANK)
    replay = replay_bank(BANK)
    assert validation["passed"] is True and validation["sample_count"] == 45
    assert replay["exact_replay"] is True and replay["artifact_count"] == 92
    assert replay["artifact_bytes"] == 1_390_319


def test_soft_prior_improves_every_sample_without_forcing_perfection() -> None:
    manifest = _manifest()
    summary = manifest["symmetry_summary"]
    assert summary == {
        "mean_weighted_before": 0.5370884,
        "mean_weighted_after": 0.6842649,
        "mean_improvement": 0.1471765,
        "improved_samples": 45,
        "unchanged_samples": 0,
    }
    assert manifest["policy"]["hard_symmetry_required"] is False
    assert manifest["policy"]["source_cell_deletion_allowed"] is False
    assert all(record["symmetry"]["after"]["weighted_score"] > record["symmetry"]["before"]["weighted_score"] for record in manifest["offspring"])
    assert all(record["symmetry"]["after"]["weighted_score"] < 1.0 for record in manifest["offspring"])


def test_source_cells_and_lineage_are_exactly_preserved() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8")); source_by_id = {record["sample_id"]: record for record in source["offspring"]}
    for record in _manifest()["offspring"]:
        source_record = source_by_id[record["sample_id"]]
        source_fields = load_breeding_fields(SOURCE.parent / source_record["fields"]["path"])
        refined = _load_fields(BANK.parent / record["fields"]["path"])
        inherited = source_fields["part_owner"] > 0
        assert np.array_equal(refined["part_owner"][inherited], source_fields["part_owner"][inherited])
        assert np.array_equal(refined["material"][inherited], source_fields["material"][inherited])
        assert np.array_equal(refined["emission"][inherited], source_fields["emission"][inherited])
        assert record["lineage"] == source_record["lineage"]
        assert record["parents"] == source_record["parents"]
        assert record["symmetry"]["source_cells_deleted"] == 0


def test_family_priors_are_strongest_for_chassis_and_weakest_for_anomaly() -> None:
    priors = _manifest()["policy"]["family_priors"]
    assert priors["machine"]["chassis"] > priors["humanoid"]["chassis"] > priors["animalian"]["chassis"] > priors["plantlike"]["chassis"] > priors["anomaly"]["chassis"]
    assert priors["machine"]["paired"] > priors["animalian"]["paired"] > priors["anomaly"]["paired"]
    assert priors["anomaly"]["growth_cap"] < priors["plantlike"]["growth_cap"] < priors["machine"]["growth_cap"]


def test_refined_children_have_complete_redecoded_anatomy() -> None:
    for record in _manifest()["offspring"]:
        arrays = _load_arrays(BANK.parent / record["arrays"]["path"])
        validate_species_arrays(arrays, record["organs"], record["summary"])
        assert record["summary"]["physical_cell_count"] == record["symmetry"]["original_cells"] + round(record["symmetry"]["growth_fraction"] * record["symmetry"]["original_cells"])
        assert record["summary"]["eye_count"] >= 1
        assert float(arrays["fluid_initial"].sum()) > 0
