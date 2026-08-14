from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from forge.cellular_organism.compiler import _load_arrays
from forge.cellular_physiology import PhysiologyState, replay_bank, validate_bank
from forge.cellular_physiology.compiler import compile_systems
from forge.cellular_physiology.contract import SYSTEM_NAMES


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/cellular_breeding_symmetry_v1/cellular_symmetry_manifest.json"
BANK = ROOT / "outputs/cellular_physiology_v3/cellular_physiology_manifest.json"


def _representative(family_id: int = 0):
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    record = next(item for item in source["offspring"] if item["family_id"] == family_id)
    arrays = _load_arrays(SOURCE.parent / record["arrays"]["path"])
    overlay, systems = compile_systems(record, arrays)
    return record, arrays, overlay, systems


def test_every_family_has_all_connected_overlapping_systems() -> None:
    for family_id in range(5):
        _, arrays, overlay, systems = _representative(family_id)
        assert [item["name"] for item in systems] == list(SYSTEM_NAMES)
        assert overlay["system_role"].shape == (8, len(arrays["position_xy"]))
        assert np.all(np.count_nonzero(overlay["system_role"] == 1, axis=1) > 0)
        assert np.count_nonzero(overlay["system_membership"] & (overlay["system_membership"] - 1)) > 0
        state = PhysiologyState(arrays, overlay)
        assert all(value > 0.99 for value in state.capacities().values())


def test_destroying_each_system_core_collapses_its_capacity() -> None:
    _, arrays, overlay, _ = _representative(0)
    for system_id, name in enumerate(SYSTEM_NAMES):
        state = PhysiologyState(arrays, overlay)
        core = np.flatnonzero(overlay["system_role"][system_id] == 1)
        state.kill_cells(core)
        assert state.capacities()[name] == 0.0


def test_heart_brain_gut_and_respiration_damage_have_distinct_cascades() -> None:
    _, arrays, overlay, _ = _representative(1)
    baseline = PhysiologyState(arrays, overlay).capacities()

    heart = PhysiologyState(arrays, overlay)
    heart.kill_cells(np.flatnonzero(overlay["system_role"][SYSTEM_NAMES.index("circulation")] == 1))
    heart_capacity = heart.capacities()
    assert heart_capacity["circulation"] == 0.0
    assert heart_capacity["respiration"] == heart_capacity["digestion"] == heart_capacity["neural"] == 0.0

    brain = PhysiologyState(arrays, overlay)
    brain.kill_cells(np.flatnonzero(overlay["system_role"][SYSTEM_NAMES.index("neural")] == 1))
    brain_capacity = brain.capacities()
    assert brain_capacity["neural"] == brain_capacity["sensory"] == brain_capacity["locomotion"] == 0.0
    assert brain_capacity["circulation"] > 0.9 and brain_capacity["digestion"] > 0.9

    gut = PhysiologyState(arrays, overlay)
    gut.kill_cells(np.flatnonzero(overlay["system_role"][SYSTEM_NAMES.index("digestion")] == 1))
    before = gut.energy
    for _ in range(30): gut.step(1 / 60, nutrient_input=0.03)
    assert gut.capacities()["digestion"] == 0.0 and gut.energy < before

    lungs = PhysiologyState(arrays, overlay); lungs.oxygen = 0.1
    lungs.kill_cells(np.flatnonzero(overlay["system_role"][SYSTEM_NAMES.index("respiration")] == 1))
    for _ in range(60): lungs.step(1 / 60)
    assert lungs.capacities()["respiration"] == 0.0 and lungs.oxygen < 0.1
    assert baseline["respiration"] > 0.99


def test_severing_motor_effectors_from_brain_reduces_locomotion_capacity() -> None:
    _, arrays, overlay, _ = _representative(4)
    state = PhysiologyState(arrays, overlay); system_id = SYSTEM_NAMES.index("locomotion")
    effectors = overlay["system_role"][system_id] == 3
    boundary = [
        index for index, (a, b) in enumerate(state.bond_ab)
        if bool(effectors[int(a)]) != bool(effectors[int(b)])
    ]
    assert boundary
    state.break_bonds(boundary)
    assert state.capacities()["locomotion"] < 0.8


def test_network_delivery_is_member_restricted_and_local() -> None:
    _, arrays, overlay, _ = _representative(0)
    system_id = SYSTEM_NAMES.index("locomotion")
    members = overlay["system_weight"][system_id] > 0
    baseline = PhysiologyState(arrays, overlay)
    assert np.all(baseline.network_delivery()["locomotion"][members] > 0.999)

    # Find a motor-network bridge. Cutting it must strand system members even
    # though the general physical body still offers an alternate attachment.
    selected = None
    for bond_index, (a_raw, b_raw) in enumerate(baseline.bond_ab):
        a, b = int(a_raw), int(b_raw)
        if not members[a] or not members[b]:
            continue
        candidate = PhysiologyState(arrays, overlay)
        candidate.break_bonds([bond_index])
        signal = candidate.network_delivery()["locomotion"]
        lost = np.flatnonzero(members & (signal <= 0))
        if len(lost) and candidate._reachable_from(np.asarray([a]))[int(lost[0])]:
            selected = candidate, lost
            break
    assert selected is not None
    candidate, lost = selected
    fields = candidate.delivery_fields()
    assert set(fields) == set(SYSTEM_NAMES)
    assert all(value.shape == (len(arrays["position_xy"]),) and value.dtype == np.float32 for value in fields.values())
    assert float(fields["locomotion"][int(lost[0])]) < 0.95
    assert candidate.capacities()["locomotion"] < 1.0


def test_partial_motor_cell_injury_produces_graded_delivery_before_death() -> None:
    _, arrays, overlay, _ = _representative(0)
    system_id = SYSTEM_NAMES.index("locomotion")
    state = PhysiologyState(arrays, overlay)
    effector = int(np.flatnonzero(overlay["system_role"][system_id] == 3)[0])
    baseline = state.capacities()["locomotion"]
    state.health[effector] = state.max_health[effector] * np.float32(0.37)
    delivery = state.network_delivery()["locomotion"]
    assert np.isclose(delivery[effector], 0.37, atol=1e-6)
    assert 0.0 < state.capacities()["locomotion"] < baseline


def test_local_fluid_loss_reduces_perfusion_without_killing_the_vessel() -> None:
    _, arrays, overlay, _ = _representative(0)
    state = PhysiologyState(arrays, overlay)
    circulation_id = SYSTEM_NAMES.index("circulation")
    conduit = int(np.flatnonzero(overlay["system_role"][circulation_id] == 2)[0])
    baseline = state.capacities()["circulation"]
    state.fluid[conduit] = state.fluid_reference[conduit] * np.float32(0.22)
    delivery = state.network_delivery()["circulation"]
    assert state.alive[conduit] and np.isclose(delivery[conduit], 0.22, atol=1e-6)
    assert 0.0 < state.capacities()["circulation"] < baseline


def test_published_physiology_bank_is_hash_closed_and_exactly_replayable() -> None:
    validation = validate_bank(BANK); replay = replay_bank(BANK)
    assert validation["passed"] is True and validation["identity_count"] == 45
    assert replay["exact_replay"] is True and replay["artifact_count"] == 47
