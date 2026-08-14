from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from forge.cellular_organism.compiler import _compile_arrays, _genome
from forge.cellular_organism.orientation import (
    ORIENTATION_CONTRACT_SHA256,
    SurfacePuddleField,
    orientation_manifest,
    validate_orientation,
)
from forge.cellular_organism.simulation import OrganismState
from forge.map_decorator.hashing import json_sha256
from forge.multifield_style.source import load_generation_bank


ROOT = Path(__file__).resolve().parents[1]
GENERATION = ROOT / "outputs/production_handoff_v2/final_best_stratified80_bank_attempt1/generation_manifest.json"


def test_orientation_contract_is_exact_and_self_hashed() -> None:
    manifest = orientation_manifest()
    assert manifest["contract"]["projection"] == "top_down_dorsal"
    assert manifest["contract"]["uniform_acceleration_xy"] == [0.0, 0.0]
    assert manifest["contract"]["external_fluid_model"] == "isotropic_surface_diffusion"
    assert manifest["contract_sha256"] == ORIENTATION_CONTRACT_SHA256
    assert ORIENTATION_CONTRACT_SHA256 == json_sha256(manifest["contract"])
    validate_orientation(manifest)
    corrupted = orientation_manifest(); corrupted["contract"]["uniform_acceleration_xy"] = [0.0, 1.0]
    with pytest.raises(ValueError, match="orientation"):
        validate_orientation(corrupted)


def test_stationary_surface_deposit_diffuses_radially_without_y_bias() -> None:
    field = SurfacePuddleField(65, 65, diffusion_rate=2.8, evaporation_rate=0.0)
    assert field.deposit(np.array([[32.0, 32.0]], dtype=np.float32), np.array([10.0], dtype=np.float32)) == 10.0
    initial_radius = field.rms_radius()
    for _ in range(180):
        field.step(1 / 60)
    centroid = field.centroid_xy()
    assert centroid is not None
    assert np.allclose(centroid, (32.0, 32.0), atol=1e-6)
    assert np.allclose(field.amount, np.flip(field.amount, axis=0), atol=1e-7)
    assert np.allclose(field.amount, np.flip(field.amount, axis=1), atol=1e-7)
    assert np.allclose(field.amount, field.amount.T, atol=1e-7)
    assert field.rms_radius() > initial_radius * 1.8
    assert np.isclose(field.total, 10.0, rtol=1e-5)


def test_reference_organism_rejects_screen_gravity_and_tracks_surface_fluid() -> None:
    sample = load_generation_bank(GENERATION).samples[0]
    arrays, _, _ = _compile_arrays(sample)
    state = OrganismState(arrays, _genome(sample.condition))
    with pytest.raises(ValueError, match="top-down"):
        state.step(1 / 60, gravity=True)
    center = tuple(map(float, state.position.mean(axis=0)))
    result = state.apply_damage(center, radius=34.0, damage=2.4, impulse=210.0)
    assert result["broken_bonds"] > 0
    for _ in range(30):
        state.step(1 / 60)
    status = state.status()
    assert status["surface_fluid"] > 0
    assert status["surface_spread_radius"] > 0
    assert status["orientation_contract_sha256"] == ORIENTATION_CONTRACT_SHA256


def test_native_lab_source_has_no_screen_down_acceleration() -> None:
    script = (ROOT / "game/scripts/cellular_organism_lab.gd").read_text(encoding="utf-8")
    assert "ORIENTATION_FORMAT" in script
    assert "_draw_surface_spill" in script
    assert "isotropic_surface_diffusion" in script
    assert "gravity_enabled" not in script
    assert "Vector2(0, 18.0)" not in script
    assert "velocity[cell_index].y += 28.0" not in script
