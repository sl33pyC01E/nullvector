from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch

from ..cellular_nca.contract import canonical_json_bytes
from ..creature_stage_developmental import develop
from ..creature_stage_developmental.genomes import review_genomes
from ..living_body_substrate import LivingBody
from ..living_body_substrate.contract import ORGAN_SYSTEM
from .adapter import LivingBodyNCARuntime
from .contract import DEFAULT_AUTHORITY, FORMAT, source_sha256


AUDIT_FORMAT = "nullvector-living-body-causal-nca-promotion-audit-v1"
AUDIT_SYSTEMS = ("circulation", "respiration", "digestion", "neural")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _damage_system(body: LivingBody, system: str) -> int:
    selected = np.asarray([ORGAN_SYSTEM.get(organ) == system for organ in body.organ])
    if not selected.any():
        return 0
    body.health[selected] = np.minimum(body.health[selected], np.float32(.02))
    body.fluid[selected] = 0
    body._connectivity_alive = None
    body._connectivity_connected = None
    body._systems_cache = None
    return int(selected.sum())


def _make_cases() -> tuple[list[tuple[str, LivingBody]], dict[str, tuple[str, str | None, int]]]:
    cases: list[tuple[str, LivingBody]] = []
    metadata: dict[str, tuple[str, str | None, int]] = {}
    for genome in review_genomes():
        healthy_key = f"{genome.genome_id}/healthy"
        cases.append((healthy_key, LivingBody(develop(genome), seed=genome.seed)))
        metadata[healthy_key] = (genome.genome_id, None, 0)
        for system in AUDIT_SYSTEMS:
            body = LivingBody(develop(genome), seed=genome.seed)
            damaged_cells = _damage_system(body, system)
            if damaged_cells:
                key = f"{genome.genome_id}/{system}"
                cases.append((key, body))
                metadata[key] = (genome.genome_id, system, damaged_cells)
    return cases, metadata


@torch.inference_mode()
def _audit_once(runtime: LivingBodyNCARuntime, steps: int) -> list[dict[str, Any]]:
    runtime.states.clear()
    cases, metadata = _make_cases()
    for _ in range(steps):
        runtime.step_many(cases)
    snapshots = {key: body.snapshot() for key, body in cases}
    rows: list[dict[str, Any]] = []
    for key, body in cases:
        genome_id, ablation, damaged_cells = metadata[key]
        snapshot = snapshots[key]
        healthy = snapshots[f"{genome_id}/healthy"]
        rows.append({
            "key": key,
            "genome_id": genome_id,
            "family": int(body.family),
            "ablation": ablation,
            "damaged_cells": damaged_cells,
            "dead": snapshot.dead,
            "incapacitated": snapshot.incapacitated,
            "energy": snapshot.energy,
            "alive_cells": snapshot.alive_cells,
            "systems": snapshot.systems,
            "matched_capacity_delta": None if ablation is None else round(
                snapshot.systems[ablation] - healthy.systems[ablation], 8
            ),
        })
    return rows


def audit(
    destination: Path,
    *,
    authority: Path = DEFAULT_AUTHORITY,
    device: str = "cpu",
    steps: int = 12,
) -> dict[str, Any]:
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if steps != 12:
        raise ValueError("living-body promotion horizon drifted")
    runtime = LivingBodyNCARuntime.from_output(Path(authority), device=device, blend=.55)
    first = _audit_once(runtime, steps)
    second = _audit_once(runtime, steps)
    if first != second:
        raise ValueError("living-body causal NCA audit is not deterministic")
    healthy = [row for row in first if row["ablation"] is None]
    ablated = [row for row in first if row["ablation"] is not None]
    per_system = {
        system: [float(row["matched_capacity_delta"]) for row in ablated if row["ablation"] == system]
        for system in AUDIT_SYSTEMS
    }
    gates = {
        "all_five_families_present": {row["family"] for row in healthy} == set(range(5)),
        "all_ten_reviewed_chassis_present": len(healthy) == 10,
        "healthy_chassis_remain_alive": all(not row["dead"] for row in healthy),
        "healthy_chassis_remain_capable": all(not row["incapacitated"] for row in healthy),
        "healthy_integrity_above_0_70": all(row["systems"]["integrity"] >= .70 for row in healthy),
        "all_organ_classes_exercised": all(per_system[system] for system in AUDIT_SYSTEMS),
        "targeted_ablations_reduce_capacity": all(
            delta <= -.005 for values in per_system.values() for delta in values
        ),
        "exact_replay": True,
    }
    payload: dict[str, Any] = {
        "format": AUDIT_FORMAT,
        "status": "ready" if all(gates.values()) else "experimental",
        "bridge_format": FORMAT,
        "bridge_source_sha256": source_sha256(),
        "authority": str(Path(authority).resolve()),
        "device": str(runtime.device),
        "steps": steps,
        "case_count": len(first),
        "rows": first,
        "summary": {
            "minimum_healthy_integrity": min(row["systems"]["integrity"] for row in healthy),
            "mean_matched_capacity_delta": {
                system: round(float(np.mean(values)), 8) if values else None
                for system, values in per_system.items()
            },
        },
        "gates": gates,
        "promotion_allowed": all(gates.values()),
        "limitations": [
            "The learned update is still projected onto deterministic cell topology.",
            "This audit covers physiology, not behavior, ecology, or learned fracture.",
        ],
    }
    payload["audit_sha256"] = _sha256_bytes(canonical_json_bytes(payload))
    _atomic_write(destination, canonical_json_bytes(payload))
    return payload


def validate(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    raw = path.read_bytes()
    payload = json.loads(raw)
    if raw != canonical_json_bytes(payload):
        raise ValueError("living-body audit JSON is not canonical")
    expected = _sha256_bytes(canonical_json_bytes({key: value for key, value in payload.items() if key != "audit_sha256"}))
    if payload.get("audit_sha256") != expected:
        raise ValueError("living-body audit hash drifted")
    if payload.get("format") != AUDIT_FORMAT or payload.get("bridge_format") != FORMAT:
        raise ValueError("living-body audit format drifted")
    if payload.get("bridge_source_sha256") != source_sha256():
        raise ValueError("living-body audit source drifted")
    gates = payload.get("gates")
    if not isinstance(gates, dict) or payload.get("promotion_allowed") != all(gates.values()):
        raise ValueError("living-body audit gate/status drifted")
    if (payload.get("status") == "ready") != bool(payload["promotion_allowed"]):
        raise ValueError("living-body audit status drifted")
    return payload
