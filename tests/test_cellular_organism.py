from __future__ import annotations

from pathlib import Path

import numpy as np

from forge.cellular_organism.compiler import _compile_arrays, validate_species_arrays
from forge.cellular_organism.contract import CELLULAR_CONTRACT_SHA256, CellFlag, contract_manifest
from forge.cellular_organism.simulation import OrganismState
from forge.cellular_organism_sync import project_runtime, validate_runtime
from forge.map_decorator.hashing import json_sha256
from forge.multifield_style.source import load_generation_bank


ROOT = Path(__file__).resolve().parents[1]
GENERATION = ROOT / "outputs/production_handoff_v2/final_best_stratified80_bank_attempt1/generation_manifest.json"


def _species(index: int = 0):
    bank = load_generation_bank(GENERATION)
    sample = bank.samples[index]
    arrays, organs, summary = _compile_arrays(sample)
    validate_species_arrays(arrays, organs, summary)
    return sample, arrays, organs, summary


def test_contract_is_self_hashing_and_pixel_cells_exclude_only_aura() -> None:
    assert CELLULAR_CONTRACT_SHA256 == json_sha256(contract_manifest())
    sample, arrays, _, summary = _species()
    physical = (sample.fields.part > 0) & (sample.fields.part != 16)
    assert summary["physical_cell_count"] == int(physical.sum())
    assert set(map(tuple, arrays["position_xy"].tolist())) == {
        (int(x), int(y)) for y, x in np.argwhere(physical)
    }


def test_every_family_has_organs_eyes_fluid_and_connected_breakable_bonds() -> None:
    bank = load_generation_bank(GENERATION)
    for index in (0, 16, 32, 48, 64):
        sample = bank.samples[index]
        arrays, organs, summary = _compile_arrays(sample)
        validate_species_arrays(arrays, organs, summary)
        kinds = {organ["kind"] for organ in organs}
        assert {"circulatory", "neural", "digestive", "reproductive", "sensory"} <= kinds
        assert np.count_nonzero(arrays["cell_flags"] & int(CellFlag.EYE)) >= 1
        assert float(arrays["fluid_initial"].sum()) > 0
        assert len(arrays["bond_ab"]) >= len(arrays["position_xy"]) - 1


def test_damage_severs_bonds_kills_cells_and_spills_fluid() -> None:
    _, arrays, _, _ = _species(16)
    state = OrganismState(arrays, {**{
        "generation": 0, "metabolic_rate": 1.0, "digestion_efficiency": 0.8,
        "tissue_regeneration_rate": 0.01, "reproduction_energy_threshold": 99999.0,
        "offspring_energy_fraction": 0.35, "mutation_rate": 0.05, "mutation_scale": 0.1,
    }})
    center = tuple(map(float, state.position.mean(axis=0)))
    before_fluid = float(state.fluid.sum())
    result = state.apply_damage(center, radius=35.0, damage=2.5, impulse=220.0)
    assert result["killed_cells"] > 0
    assert result["broken_bonds"] > 0
    for _ in range(12):
        state.step(1 / 60)
    assert float(state.fluid.sum()) < before_fluid
    assert state.status()["fluid_lost"] > 0


def test_feeding_and_reproduction_have_heritable_mutation_contract() -> None:
    sample, arrays, _, _ = _species(32)
    from forge.cellular_organism.compiler import _genome

    genome = _genome(sample.condition)
    state = OrganismState(arrays, genome)
    before = float(state.nutrient.sum())
    assert state.feed(25.0) == 25.0
    assert float(state.nutrient.sum()) > before
    state.energy += np.float32(10.0)
    assert state.can_reproduce()
    event = state.reproduce(0xBAD5EED)
    assert event.child_generation == 1
    assert event.child_seed == 0xBAD5EED
    assert event.child_genome["genome_seed"] == 0xBAD5EED
    assert state.status()["birth_count"] == 1


def test_compile_is_exactly_deterministic() -> None:
    sample, first, first_organs, first_summary = _species(48)
    second, second_organs, second_summary = _compile_arrays(sample)
    assert first_organs == second_organs
    assert first_summary == second_summary
    for name in first:
        assert np.array_equal(first[name], second[name])


def test_published_bank_and_native_projection_are_hash_closed() -> None:
    bank_manifest = ROOT / "outputs/cellular_organism_v1/cellular_organism_manifest.json"
    if not bank_manifest.exists():
        return
    first = project_runtime(bank_manifest)
    second = project_runtime(bank_manifest)
    assert first == second
    validation = validate_runtime(ROOT / "game/generated/cellular_organism/v2")
    assert validation["sample_count"] == 80
    assert validation["cell_count"] == 34178
    assert validation["bond_count"] == 116112
