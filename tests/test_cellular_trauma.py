from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from forge.cellular_organism.compiler import _load_arrays as _load_anatomy
from forge.cellular_physiology.compiler import _load_overlay
from forge.cellular_trauma import TraumaState, compile_trauma, replay_bank, validate_bank


ROOT = Path(__file__).resolve().parents[1]
ANATOMY = ROOT / "outputs/cellular_breeding_symmetry_v1/cellular_symmetry_manifest.json"
PHYSIOLOGY = ROOT / "outputs/cellular_physiology_v1/cellular_physiology_manifest.json"
BANK = ROOT / "outputs/cellular_trauma_v1/cellular_trauma_manifest.json"


def _identity(family_id: int):
    source = json.loads(ANATOMY.read_text(encoding="utf-8")); physiology = json.loads(PHYSIOLOGY.read_text(encoding="utf-8"))
    record = next(item for item in source["offspring"] if item["family_id"] == family_id); p_record = next(item for item in physiology["identities"] if item["sample_id"] == record["sample_id"])
    anatomy = _load_anatomy(ANATOMY.parent / record["arrays"]["path"]); overlay = _load_overlay(PHYSIOLOGY.parent / p_record["arrays"]["path"], len(anatomy["position_xy"])); trauma, profile = compile_trauma(record, anatomy, overlay)
    return record, anatomy, overlay, trauma, profile


def _detach_nonessential_organ(state: TraumaState, anatomy: dict[str, np.ndarray], minimum: int) -> tuple[int, ...]:
    organ_id = anatomy["organ_id"]
    for organ in sorted(set(map(int, organ_id)), key=lambda value: (np.count_nonzero(organ_id == value), value)):
        cells = set(map(int, np.flatnonzero(organ_id == organ)))
        if len(cells) < minimum or len(cells) >= state.cell_count // 2: continue
        boundary = [index for index, (a, b) in enumerate(state.bond_ab) if (int(a) in cells) != (int(b) in cells)]
        if not boundary: continue
        candidate = TraumaState(anatomy, state.physiology, state.trauma, state.profile); candidate.cut_bonds(boundary)
        components = candidate.components()
        detached = [component for component in components if set(component) <= cells]
        if detached:
            state.cut_bonds(boundary); return max(detached, key=len)
    raise AssertionError("representative anatomy has no detachable nonessential organ")


def test_all_families_classify_every_cell_and_bond_with_distinct_fragment_profiles() -> None:
    fates = set()
    for family_id in range(5):
        record, anatomy, _physiology, trauma, profile = _identity(family_id)
        assert trauma["heal_class"].shape == (len(anatomy["position_xy"]),)
        assert np.all(trauma["heal_class"] > 0)
        assert trauma["bond_repair_weight"].shape == (len(anatomy["bond_ab"]),)
        assert np.all(trauma["bond_magnetic_weight"] > 0)
        assert profile["family"] == record["family"]
        fates.add(profile["detached_fate"])
    assert "biomass" in fates and {"polyp", "phase_polyp", "module_polyp"} <= fates


def test_heart_brain_and_gut_trauma_reduce_distinct_connected_capacities() -> None:
    _record, anatomy, physiology, trauma, profile = _identity(0)
    for system_id, name in ((0, "circulation"), (2, "digestion"), (3, "neural")):
        state = TraumaState(anatomy, physiology, trauma, profile); core = np.flatnonzero(physiology["system_role"][system_id] == 1); state.damage_cells(core, 100.0); capacities = state.capacities()
        assert capacities[name] == 0.0
        if name == "circulation": assert capacities["respiration"] == capacities["digestion"] == capacities["neural"] == 0.0
        if name == "neural": assert capacities["locomotion"] == capacities["sensory"] == 0.0 and capacities["circulation"] > 0.9


def test_clotting_reduces_diffuse_fluid_loss_and_reconnection_forms_scar() -> None:
    _record, anatomy, physiology, trauma, profile = _identity(1); bond = int(np.argmax(trauma["bond_magnetic_weight"]))
    clotted = TraumaState(anatomy, physiology, trauma, profile); unclotted_trauma = dict(trauma); unclotted_trauma["clotting_weight"] = np.zeros_like(trauma["clotting_weight"]); unclotted = TraumaState(anatomy, physiology, unclotted_trauma, profile)
    clotted.cut_bonds([bond]); unclotted.cut_bonds([bond])
    for _ in range(240): clotted.step(1 / 120); unclotted.step(1 / 120)
    assert clotted.fluid_lost < unclotted.fluid_lost and float(clotted.clot.max()) > 0.0
    a, b = map(int, clotted.bond_ab[bond]); assert clotted.magnetic_force(bond, 1.2) > 0.0
    assert clotted.attempt_reconnect(bond, 0.6) is True and clotted.bond_alive[bond]
    assert float(clotted.scar[[a, b]].max()) > 0.0


def test_reconnection_window_expires_and_fragment_fate_is_family_specific() -> None:
    events = {}
    for family_id in (0, 2, 3, 4):
        _record, anatomy, physiology, trauma, profile = _identity(family_id); state = TraumaState(anatomy, physiology, trauma, profile); component = _detach_nonessential_organ(state, anatomy, min(12, max(3, int(profile["polyp_min_cells"]))))
        for _ in range(int(float(profile["reconnect_window_seconds"]) * 4) + 4): state.step(0.25)
        fates = state.fragment_fates(); matching = [fate for cells, fate in fates.items() if set(cells) == set(component)]
        assert matching; events[profile["family"]] = matching[0]
        crossing = next(index for index, (a, b) in enumerate(state.bond_ab) if (int(a) in component) != (int(b) in component))
        assert state.magnetic_force(crossing, 0.1) == 0.0 and state.attempt_reconnect(crossing, 0.1) is False
    assert events["humanoid"] == "biomass"
    assert events["plantlike"] == "polyp" and events["anomaly"] == "phase_polyp" and events["machine"] == "module_polyp"


def test_published_trauma_bank_is_hash_closed_and_exactly_replayable() -> None:
    validation = validate_bank(BANK); replay = replay_bank(BANK)
    assert validation["passed"] is True and validation["identity_count"] == 45
    assert replay["exact_replay"] is True and replay["artifact_count"] == 47
