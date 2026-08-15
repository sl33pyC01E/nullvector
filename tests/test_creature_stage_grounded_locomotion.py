from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from forge.creature_stage_developmental.development import develop
from forge.creature_stage_developmental.genomes import review_genomes
from forge.creature_stage_grounded_locomotion.contract import GroundedLocomotionConfig, source_sha256
from forge.creature_stage_grounded_locomotion.physics import locomotor_modes, simulate_grounded_cycle
from forge.creature_stage_grounded_locomotion.review import _array_payload, _contact_sheet, _cycles
from forge.multifield_style_motion.hashing import deterministic_npz_bytes


@pytest.fixture(scope="module")
def authority() -> tuple:
    return _cycles(GroundedLocomotionConfig())


def test_grounded_configuration_and_source_are_strict() -> None:
    assert len(source_sha256()) == 64
    with pytest.raises(ValueError, match="gait envelope"):
        GroundedLocomotionConfig(step_stance_fraction=.2)
    with pytest.raises(ValueError, match="discrete"):
        GroundedLocomotionConfig(frame_count=12)


def test_base_families_use_distinct_physical_locomotion(authority: tuple) -> None:
    organisms, cycles = authority
    assert [cycle.primary_mode for cycle in cycles[::2]] == ["step", "step", "drag", "float", "wheel"]
    assert [int(np.argmax(organism.genome.family_mix)) for organism in organisms[::2]] == list(range(5))
    assert all(cycle.distance_px > .25 for cycle in cycles)
    assert max(cycle.maximum_contact_slip_px for cycle in cycles) < .05
    assert max(cycle.maximum_edge_strain for cycle in cycles) < .12
    assert max(cycle.loop_seam_max_abs for cycle in cycles) < .002
    assert max(cycle.vertical_axis_max_degrees for cycle in cycles) < 5.0


def test_motion_is_caused_by_contacts_except_for_floaters(authority: tuple) -> None:
    _organisms, cycles = authority
    for cycle in cycles:
        if cycle.primary_mode == "float":
            assert cycle.traction_work == 0.0
            assert not any(bool(frame.contact_active.any()) for frame in cycle.frames)
        else:
            assert cycle.traction_work > .1
            assert any(bool(frame.contact_active.any()) for frame in cycle.frames)


def test_locomotor_components_survive_cross_family_grafting(authority: tuple) -> None:
    organisms, cycles = authority
    assert {"step", "drag"}.issubset(set(locomotor_modes(organisms[1])))
    assert cycles[1].primary_mode == "step"
    assert "step" in locomotor_modes(organisms[7])
    assert float(organisms[7].genome.family_mix[3]) > .5


def test_cycle_and_review_arrays_replay_exact(authority: tuple) -> None:
    organisms, cycles = authority
    repeated = simulate_grounded_cycle(develop(review_genomes()[2]))
    assert repeated.identity_sha256 == cycles[2].identity_sha256
    arrays = _array_payload(organisms, cycles)
    assert deterministic_npz_bytes(arrays) == deterministic_npz_bytes(_array_payload(organisms, cycles))
    assert _contact_sheet(organisms, cycles) == _contact_sheet(organisms, cycles)
    assert arrays["cells_local"].shape[1] == 72
    assert arrays["trait_fields"].shape[-1] == 15
    assert arrays["contact_active"].dtype == np.uint8
