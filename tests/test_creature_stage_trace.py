from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from forge.creature_stage_trace import TraceValidationError, assert_valid_trace


def _state(step: int) -> dict:
    energy = 0.82 - step * 0.0002
    return {
        "position": [step * 0.21, step * -0.04],
        "velocity": [6.3, -1.2],
        "status": {
            "family": "humanoid",
            "generation": 0,
            "integrity": 1.0,
            "neural": 1.0,
            "circulation": 1.0,
            "respiration": 1.0,
            "digestion": 1.0,
            "senses": 1.0,
            "energy": energy,
            "hydration": 0.85,
            "dead": False,
        },
        "genes": {
            "body_scale": 1.0,
            "symmetry": 0.9,
            "appendage_bias": 0.5,
            "metabolism": 1.0,
            "fertility": 0.8,
            "bond_strength": 1.1,
        },
        "organ_alive": {"brain": 4, "heart": 3, "gut": 5},
        "organ_totals": {"brain": 4, "heart": 3, "gut": 5},
        "world_field": [0.5, 0.4, 0.3, 0.2, 0.6],
        "chunk": [0, -1],
        "biome": "verdant mire",
        "nearest_resource": {
            "type": "biomass",
            "relative": [18.0, -4.0],
            "amount": 0.8,
        },
        "inventory": {
            "biomass": 0.1,
            "mineral": 0.0,
            "fluid": 0.0,
            "spore": 0.0,
            "phase": 0.0,
        },
        "essence": 0.0,
        "construction_mass": 0.1,
        "objective_stage": 0,
        "active_creatures": 55,
        "active_projectiles": 1 if step % 19 < 4 else 0,
        "built_structures": 0,
        "known_societies": 0,
    }


def _document() -> dict:
    states = [_state(step) for step in range(241)]
    transitions = []
    for step in range(240):
        transitions.append(
            {
                "step": step,
                "dt": 1.0 / 30.0,
                "before": states[step],
                "action": {
                    "move": [1.0, 0.0],
                    "aim": [1.0, 0.0],
                    "attack": 1.0 if step % 19 < 2 else 0.0,
                    "feed": 1.0 if step % 31 < 6 else 0.0,
                    "utility": 1.0 if step in (42, 126, 210) else 0.0,
                    "sprint": step % 60 >= 45,
                },
                "after": states[step + 1],
            }
        )
    transition_bytes = json.dumps(transitions, separators=(",", ":")).encode()
    return {
        "format": "nullvector-creature-stage-causal-trace-v1",
        "world_seed": 0x4E554C4C,
        "family": "humanoid",
        "fixed_hz": 30,
        "transition_count": 240,
        "transition_sha256": hashlib.sha256(transition_bytes).hexdigest(),
        "contracts": {
            "morphology": "coordinate-conditioned-safe-scaffold-v1",
            "controller": "recurrent-18x12x10-v1",
            "world_field": "continuous-5-channel-latent-v1",
            "physiology": "cellular-organ-causal-scaffold-v1",
            "action": "move2-aim2-attack-feed-utility-sprint-v1",
        },
        "transitions": transitions,
    }


def _write(path: Path, document: dict, *, rehash: bool = False) -> Path:
    if rehash:
        compact = json.dumps(document["transitions"], separators=(",", ":")).encode()
        document["transition_sha256"] = hashlib.sha256(compact).hexdigest()
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return path


def test_valid_trace_is_strictly_replayable(tmp_path: Path) -> None:
    report = assert_valid_trace(_write(tmp_path / "trace.json", _document()))
    assert report["passed"] is True
    assert report["transition_count"] == 240
    assert report["attack_steps"] == 26
    assert report["utility_steps"] == 3
    assert report["moved_distance"] > 50.0


def test_stale_hash_is_rejected(tmp_path: Path) -> None:
    document = _document()
    document["transitions"][9]["after"]["status"]["energy"] = 0.1
    with pytest.raises(TraceValidationError, match="SHA-256"):
        assert_valid_trace(_write(tmp_path / "stale.json", document))


def test_rehashed_chain_tamper_is_rejected(tmp_path: Path) -> None:
    document = _document()
    forged_before = deepcopy(document["transitions"][10]["before"])
    forged_before["position"][0] += 7.0
    document["transitions"][10]["before"] = forged_before
    with pytest.raises(TraceValidationError, match="broken state chain"):
        assert_valid_trace(_write(tmp_path / "chain.json", document, rehash=True))


def test_rehashed_organ_impossibility_is_rejected(tmp_path: Path) -> None:
    document = _document()
    document["transitions"][20]["after"]["organ_alive"]["brain"] = 9
    with pytest.raises(TraceValidationError, match="organ alive count"):
        assert_valid_trace(_write(tmp_path / "organ.json", document, rehash=True))


def test_rehashed_bad_aim_is_rejected(tmp_path: Path) -> None:
    document = _document()
    document["transitions"][30]["action"]["aim"] = [0.0, 0.0]
    with pytest.raises(TraceValidationError, match="aim vector"):
        assert_valid_trace(_write(tmp_path / "aim.json", document, rehash=True))


def test_duplicate_key_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path / "duplicate.json", _document())
    raw = path.read_text(encoding="utf-8")
    path.write_text(raw.replace('{\n  "format"', '{\n  "format": "forged",\n  "format"', 1), encoding="utf-8")
    with pytest.raises(TraceValidationError, match="duplicate JSON key"):
        assert_valid_trace(path)
