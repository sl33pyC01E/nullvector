from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Mapping

import numpy as np
from jsonschema import Draft202012Validator

from .cellular_organism.compiler import _load_arrays
from .cellular_physiology.compiler import _load_overlay, validate_bank
from .cellular_physiology.contract import SYSTEM_NAMES
from .cellular_physiology.simulation import PhysiologyState
from .config import PROJECT_ROOT
from .multifield_style.hashing import sha256_file
from .multifield_style_motion.hashing import canonical_json_bytes
from .safety import require_disk_floor


FORMAT = "nullvector-cellular-homeostasis-audit-v2"
DEFAULT_PHYSIOLOGY = PROJECT_ROOT / "outputs/cellular_physiology_v4/cellular_physiology_manifest.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/cellular_homeostasis_v2/homeostasis_report.json"
SCHEMA = PROJECT_ROOT / "shared/schema/cellular_homeostasis_report.schema.json"
SOURCE_FILES = (
    "forge/cellular_homeostasis.py",
    "shared/schema/cellular_homeostasis_report.schema.json",
)


def _round(value: float) -> float:
    return round(float(value), 8)


def _json_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-cellular-homeostasis-source-v2\0")
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        digest.update(relative.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


class HomeostasisState:
    """Reserve-aware physiology over the immutable pixel-cell anatomy.

    Structural capacity is derived from live cells and intact system-member
    bonds. Functional capacity additionally depends on locally meaningful
    reserves, so lung loss causes hypoxia over time instead of erasing the
    brain at the same instant.
    """

    def __init__(self, anatomy: Mapping[str, np.ndarray], physiology: Mapping[str, np.ndarray]) -> None:
        self.body = PhysiologyState(anatomy, physiology)
        self.oxygen = 1.0
        self.energy = 1.0
        self.nutrients = 0.35
        self.hydration = 1.0
        self.waste = 0.0
        self.shock = 0.0
        self.age = 0.0
        self.neural_failure_time = 0.0
        self.perfusion_failure_time = 0.0
        self.starvation_time = 0.0
        self.dead = False

    def kill_system_core(self, name: str) -> None:
        system_id = SYSTEM_NAMES.index(name)
        self.body.kill_cells(np.flatnonzero(self.body.roles[system_id] == 1))

    def _raw_structural(self) -> dict[str, float]:
        delivery = self.body.network_delivery()
        raw: dict[str, float] = {}
        for system_id, name in enumerate(SYSTEM_NAMES):
            weights = self.body.weights[system_id]
            members = weights > 0
            cores = self.body.roles[system_id] == 1
            core_total = float(weights[cores].sum())
            total = float(weights[members].sum())
            core_fraction = float(weights[cores & self.body.alive].sum()) / max(core_total, 1e-8)
            survival = float(weights[members & self.body.alive].sum()) / max(total, 1e-8)
            connected = float((weights * delivery[name]).sum()) / max(total, 1e-8)
            raw[name] = max(0.0, min(1.0, core_fraction ** 1.55 * (0.25 * survival + 0.75 * connected)))
        return raw

    def capacities(self) -> dict[str, float]:
        raw = self._raw_structural()
        circulation = raw["circulation"]
        respiration = raw["respiration"] * circulation ** 0.45
        digestion = raw["digestion"] * circulation ** 0.45
        neural_structure = raw["neural"] * circulation ** 0.45
        oxygen_support = max(0.0, min(1.0, self.oxygen / 0.22))
        energy_support = max(0.0, min(1.0, self.energy / 0.18))
        consciousness = neural_structure * oxygen_support ** 0.8 * (1.0 - 0.65 * self.shock)
        sensory = raw["sensory"] * consciousness ** 0.55
        locomotion = raw["locomotion"] * consciousness ** 0.65 * circulation ** 0.25 * energy_support ** 0.5
        reproduction = raw["reproduction"] * circulation ** 0.35 * digestion ** 0.5 * energy_support
        immune = raw["immune"] * circulation ** 0.55 * digestion ** 0.25 * energy_support ** 0.35
        return {
            "circulation": circulation,
            "respiration": respiration,
            "digestion": digestion,
            "neural": neural_structure,
            "sensory": sensory,
            "locomotion": locomotion,
            "reproduction": reproduction,
            "immune": immune,
            "consciousness": max(0.0, min(1.0, consciousness)),
            "oxygen_support": oxygen_support,
            "energy_support": energy_support,
        }

    def status(self) -> dict[str, Any]:
        capacities = self.capacities()
        alive_fraction = float(np.count_nonzero(self.body.alive)) / max(1, self.body.cell_count)
        return {
            **{name: _round(value) for name, value in capacities.items()},
            "oxygen": _round(self.oxygen),
            "energy": _round(self.energy),
            "nutrients": _round(self.nutrients),
            "hydration": _round(self.hydration),
            "waste": _round(self.waste),
            "shock": _round(self.shock),
            "alive_fraction": _round(alive_fraction),
            "incapacitated": bool(capacities["consciousness"] < 0.14 or capacities["locomotion"] < 0.08),
            "dead": bool(self.dead),
        }

    def step(self, dt: float, *, food: float = 0.0, water: float = 0.0) -> dict[str, Any]:
        if self.dead:
            return self.status()
        if not math.isfinite(dt) or not 0.0 < dt <= 0.1 or min(food, water) < 0 or not math.isfinite(food + water):
            raise ValueError("Homeostasis step inputs are invalid")
        self.nutrients = min(3.0, self.nutrients + food)
        self.hydration = min(1.0, self.hydration + water)
        capacity = self.capacities()
        ventilation = capacity["respiration"]
        perfusion = capacity["circulation"]
        activity = capacity["locomotion"]

        oxygen_gain = 0.62 * ventilation * perfusion ** 0.35
        oxygen_use = 0.11 + 0.09 * activity
        self.oxygen = max(0.0, min(1.0, self.oxygen + (oxygen_gain - oxygen_use) * dt))
        absorbed = min(self.nutrients, 0.30 * capacity["digestion"] * perfusion ** 0.3 * dt)
        self.nutrients -= absorbed
        self.waste = min(2.0, self.waste + absorbed * 0.32)
        energy_gain = absorbed * 2.25 * max(0.15, self.oxygen)
        energy_use = (0.025 + 0.035 * activity + 0.02 * capacity["immune"]) * dt
        self.energy = max(0.0, min(2.0, self.energy + energy_gain - energy_use))
        self.hydration = max(0.0, self.hydration - (0.0015 + 0.001 * activity) * dt)
        shock_target = max(0.0, 1.0 - perfusion) * 0.78 + max(0.0, 0.18 - self.oxygen) * 1.2
        rate = min(1.0, dt * (2.4 if shock_target > self.shock else 0.45))
        self.shock += (max(0.0, min(1.0, shock_target)) - self.shock) * rate

        current = self.capacities()
        neural_members = self.body.roles[SYSTEM_NAMES.index("neural")] > 0
        if self.oxygen < 0.12:
            injury = np.float32((0.12 - self.oxygen) * 0.22 * dt)
            self.body.health[self.body.alive & neural_members] -= injury
        if perfusion < 0.12:
            injury = np.float32((0.12 - perfusion) * 0.07 * dt)
            self.body.health[self.body.alive] -= injury
        newly_dead = self.body.alive & (self.body.health <= 0)
        if bool(newly_dead.any()):
            self.body.kill_cells(np.flatnonzero(newly_dead))

        self.neural_failure_time = self.neural_failure_time + dt if current["consciousness"] < 0.08 else max(0.0, self.neural_failure_time - dt * 0.25)
        self.perfusion_failure_time = self.perfusion_failure_time + dt if perfusion < 0.06 else max(0.0, self.perfusion_failure_time - dt * 0.5)
        self.starvation_time = self.starvation_time + dt if self.energy < 0.035 else max(0.0, self.starvation_time - dt * 0.2)
        alive_fraction = float(np.count_nonzero(self.body.alive)) / max(1, self.body.cell_count)
        self.dead = bool(
            alive_fraction < 0.34
            or self.neural_failure_time >= 8.0
            or self.perfusion_failure_time >= 7.0
            or self.starvation_time >= 14.0
        )
        self.age += dt
        return self.status()


def _load_authority(manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    validation = validate_bank(manifest_path)
    physiology = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_path = PROJECT_ROOT.joinpath(*PurePosixPath(physiology["source"]["organism_manifest"]).parts)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    return validation, physiology, {record["sample_id"]: record for record in source["offspring"]}


def _state_for(
    physiology_root: Path, identity: Mapping[str, Any], source_by_id: Mapping[str, Mapping[str, Any]]
) -> HomeostasisState:
    source_record = source_by_id[identity["sample_id"]]
    source_manifest = PROJECT_ROOT / "outputs/cellular_breeding_symmetry_v1/cellular_symmetry_manifest.json"
    anatomy = _load_arrays(source_manifest.parent.joinpath(*PurePosixPath(source_record["arrays"]["path"]).parts))
    overlay = _load_overlay(
        physiology_root.joinpath(*PurePosixPath(identity["arrays"]["path"]).parts),
        identity["physical_cell_count"],
    )
    return HomeostasisState(anatomy, overlay)


def _simulate_lesion(state: HomeostasisState, lesion: str, seconds: float, *, food_per_second: float = 0.0) -> dict[str, Any]:
    state.kill_system_core(lesion)
    initial = state.status()
    timeline = []
    dt = 0.1
    next_sample = 0.0
    while state.age < seconds and not state.dead:
        state.step(dt, food=food_per_second * dt)
        if state.age + 1e-7 >= next_sample:
            snapshot = state.status()
            timeline.append({"time": _round(state.age), **snapshot})
            next_sample += 1.0
    final = state.status()
    return {
        "lesion": lesion,
        "initial": initial,
        "timeline": timeline,
        "final": final,
        "time_to_incapacitation": next((item["time"] for item in timeline if item["incapacitated"]), None),
        "time_to_death": _round(state.age) if state.dead else None,
    }


def build_report(physiology_manifest: Path = DEFAULT_PHYSIOLOGY) -> dict[str, Any]:
    physiology_manifest = Path(physiology_manifest).resolve()
    validation, physiology, source_by_id = _load_authority(physiology_manifest)
    core_matrix = []
    for identity in physiology["identities"]:
        failures = {}
        for name in SYSTEM_NAMES:
            state = _state_for(physiology_manifest.parent, identity, source_by_id)
            baseline = state.status()
            state.kill_system_core(name)
            damaged = state.status()
            failures[name] = {
                "own_capacity": damaged[name],
                "circulation": damaged["circulation"],
                "consciousness": damaged["consciousness"],
                "locomotion": damaged["locomotion"],
                "baseline_own_capacity": baseline[name],
            }
        core_matrix.append({
            "sample_id": identity["sample_id"], "family": identity["family"],
            "family_id": identity["family_id"], "failures": failures,
        })

    scenarios = []
    for family_id in range(5):
        identity = next(item for item in physiology["identities"] if item["family_id"] == family_id)
        cases = {
            "heart": _simulate_lesion(_state_for(physiology_manifest.parent, identity, source_by_id), "circulation", 16.0),
            "respiratory": _simulate_lesion(_state_for(physiology_manifest.parent, identity, source_by_id), "respiration", 20.0),
            "digestive": _simulate_lesion(_state_for(physiology_manifest.parent, identity, source_by_id), "digestion", 40.0, food_per_second=0.08),
            "brain": _simulate_lesion(_state_for(physiology_manifest.parent, identity, source_by_id), "neural", 12.0),
        }
        scenarios.append({"sample_id": identity["sample_id"], "family": identity["family"], "family_id": family_id, "cases": cases})

    gates = {
        "all_45_identities_audited": len(core_matrix) == 45,
        "all_8_core_lesions_zero_own_capacity": all(
            item["failures"][name]["own_capacity"] == 0.0 for item in core_matrix for name in SYSTEM_NAMES
        ),
        "brain_loss_immediately_incapacitates": all(item["cases"]["brain"]["initial"]["incapacitated"] for item in scenarios),
        "brain_loss_retains_circulation": all(item["cases"]["brain"]["initial"]["circulation"] > 0.5 for item in scenarios),
        "respiratory_loss_has_nonzero_initial_consciousness": all(item["cases"]["respiratory"]["initial"]["consciousness"] > 0.5 for item in scenarios),
        "respiratory_loss_causes_delayed_incapacitation": all(
            item["cases"]["respiratory"]["time_to_incapacitation"] is not None
            and item["cases"]["respiratory"]["time_to_incapacitation"] > 0.0 for item in scenarios
        ),
        "digestive_loss_blocks_reproduction": all(item["cases"]["digestive"]["initial"]["reproduction"] == 0.0 for item in scenarios),
        "digestive_loss_cannot_convert_food_to_energy": all(
            item["cases"]["digestive"]["final"]["energy"] < item["cases"]["digestive"]["initial"]["energy"] for item in scenarios
        ),
        "heart_loss_is_fatal": all(item["cases"]["heart"]["time_to_death"] is not None for item in scenarios),
        "major_neural_loss_is_fatal": all(item["cases"]["brain"]["time_to_death"] is not None for item in scenarios),
    }
    report: dict[str, Any] = {
        "format": FORMAT,
        "status": "ready" if all(gates.values()) else "failed",
        "compiler": {"source_sha256": source_sha256(), "python_runtime_required": False},
        "authority": {
            "physiology_manifest": physiology_manifest.relative_to(PROJECT_ROOT).as_posix(),
            "physiology_manifest_sha256": sha256_file(physiology_manifest),
            "physiology_semantic_sha256": physiology["semantic_sha256"],
            "physiology_validation": validation,
        },
        "contract": {
            "structural_capacity_uses_live_cells": True,
            "structural_capacity_uses_intact_member_bonds": True,
            "functional_capacity_uses_metabolic_reserves": True,
            "respiratory_failure_is_delayed_by_oxygen_reserve": True,
            "digestion_is_required_to_convert_food": True,
            "major_neural_loss_incapacitates_before_death": True,
            "sealed_anatomy_is_not_modified": True,
        },
        "system_vocab": list(SYSTEM_NAMES),
        "identity_count": len(core_matrix),
        "core_failure_case_count": len(core_matrix) * len(SYSTEM_NAMES),
        "dynamic_scenario_count": len(scenarios) * 4,
        "core_failure_matrix": core_matrix,
        "family_scenarios": scenarios,
        "gates": gates,
    }
    report["report_sha256"] = _json_hash(report)
    Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8"))).validate(report)
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise ValueError(f"Homeostasis gates failed: {failed}")
    return report


def validate_report(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    raw = path.read_bytes()
    report = json.loads(raw)
    Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8"))).validate(report)
    if raw != canonical_json_bytes(report):
        raise ValueError("Homeostasis report is not canonical JSON")
    if report["compiler"]["source_sha256"] != source_sha256():
        raise ValueError("Homeostasis source hash differs")
    if report["report_sha256"] != _json_hash({key: value for key, value in report.items() if key != "report_sha256"}):
        raise ValueError("Homeostasis report hash differs")
    authority = PROJECT_ROOT.joinpath(*PurePosixPath(report["authority"]["physiology_manifest"]).parts)
    if sha256_file(authority) != report["authority"]["physiology_manifest_sha256"]:
        raise ValueError("Homeostasis authority hash differs")
    expected = build_report(authority)
    if report != expected:
        raise ValueError("Homeostasis exact semantic replay differs")
    return {
        "passed": True,
        "identity_count": report["identity_count"],
        "core_failure_case_count": report["core_failure_case_count"],
        "dynamic_scenario_count": report["dynamic_scenario_count"],
        "report_sha256": report["report_sha256"],
    }


def write_report(destination: Path = DEFAULT_OUTPUT, physiology_manifest: Path = DEFAULT_PHYSIOLOGY) -> dict[str, Any]:
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    report = build_report(physiology_manifest)
    payload = canonical_json_bytes(report)
    require_disk_floor(destination.parent, floor_gb=100.0, planned_bytes=len(payload) + 64 * 1024**2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return validate_report(destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit reserve-aware cellular organ failure kinetics")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--physiology", type=Path, default=DEFAULT_PHYSIOLOGY)
    build.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    validate = subparsers.add_parser("validate")
    validate.add_argument("report", type=Path)
    args = parser.parse_args(argv)
    result = write_report(args.output, args.physiology) if args.command == "build" else validate_report(args.report)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
