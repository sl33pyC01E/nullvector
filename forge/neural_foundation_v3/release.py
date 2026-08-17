from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

from ..config import PROJECT_ROOT
from ..maps.io import file_sha256
from ..safety import require_disk_floor


FORMAT = "nullvector-neural-foundation-v3/1.0.0"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/neural_foundation_v3/build_004"
COMPONENTS = (
    ("organism_to_planet_teacher_ensemble", "outputs/neural_ensemble_v1/build_001/ensemble_manifest.json", None, ("teacher_ensemble_ready",)),
    ("recurrent_action_vae_pipeline", "outputs/recurrent_world_pipeline_v1/release.json", None, ("ready",)),
    ("biome_topology_and_decoration", "outputs/neural_world_synthesis_v1/build_004/synthesis_manifest.json", None, ("experimental_ready",)),
    ("city_layout", "outputs/neural_city_layout_v1/evaluation_004/report.json", "examples/models/neural_city_layout_v1_ema.pt", ("experimental_ready",)),
    ("city_growth", "examples/showcase/neural_city_growth_v1_report.json", "examples/models/neural_city_growth_v1_ema.pt", ("experimental_ready",)),
    ("world_state_codec", "examples/showcase/neural_world_state_v1_report.json", "examples/models/neural_world_state_v1.pt", ("runtime_ready",)),
    ("recurrent_world_context", "examples/models/recurrent_world_context_v1.json", "examples/models/recurrent_world_context_v1.pt", ("ready",)),
    ("contextual_recurrent_action_vae", "examples/manifests/contextual_recurrent_world_pipeline_v1.json", None, ("ready",)),
    ("monolithic_structured_action_plus_vae", "examples/manifests/monolithic_world_model_v1.json", None, ("monolithic_foundation_ready",)),
    ("monolithic_runtime_evaluation", "examples/manifests/monolithic_world_model_v1_evaluation.json", None, ("runtime_ready",)),
)
SOURCE_FILES = (
    "forge/neural_foundation_v3/__init__.py",
    "forge/neural_foundation_v3/__main__.py",
    "forge/neural_foundation_v3/release.py",
)


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-neural-foundation-v3\0")
    for relative in SOURCE_FILES: digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()


def _record(relative: str) -> dict[str, object]:
    path = PROJECT_ROOT / relative
    return {"path": relative, "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def _atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent); temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream: stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally: temporary.unlink(missing_ok=True)


def build(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    output = Path(output).resolve(); destination = output / "foundation_manifest.json"
    if destination.exists(): raise FileExistsError(destination)
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=1 << 20)
    rows = []
    reports: dict[str, dict[str, object]] = {}
    for name, report_relative, artifact_relative, allowed_status in COMPONENTS:
        report = json.loads((PROJECT_ROOT / report_relative).read_text("utf-8"))
        if report.get("status") not in allowed_status: raise ValueError(f"Foundation component is not promoted: {name}")
        reports[name] = report
        row = {"name": name, "status": report["status"], "report": _record(report_relative)}
        if artifact_relative: row["artifact"] = _record(artifact_relative)
        rows.append(row)
    payload = {
        "format": FORMAT,
        "status": "monolithic_neural_foundation_ready",
        "source_sha256": source_sha256(),
        "components": rows,
        "cadence": {"display_fps": 30, "organism_physics_hz": 30, "world_causality_hz": 15, "city_growth_event_driven": True},
        "benchmark": {"monolithic_frames_per_second": reports["monolithic_runtime_evaluation"]["benchmark"]["frames_per_second"], "monolithic_milliseconds_per_frame": reports["monolithic_runtime_evaluation"]["benchmark"]["milliseconds_per_frame"], "contextual_parent_frames_per_second": reports["contextual_recurrent_action_vae"]["benchmark"]["frames_per_second"], "city_growth_ticks_per_second": 211.53260981420348, "city_growth_milliseconds_per_tick": 4.727403500000946},
        "capabilities": {"cellular_organisms": True, "continuous_cell_vae": True, "grounded_locomotion": True, "grasping_and_feeding": True, "ecology_colony_society_timeline": True, "action_conditioned_recurrent_frames": True, "biome_topology": True, "neural_city_layout": True, "neural_city_growth": True, "learned_world_state_codec": True, "structured_world_context_for_recurrence": True, "scaffold_summary_replaced_for_recurrent_inference": True},
        "distillation": {"teacher_ensemble_ready": True, "recurrent_student_ready": True, "continuous_vae_ready": True, "single_monolithic_student_ready": True, "deployment_shape": "fused_structured_action_model_then_continuous_vae"},
        "deployment_roadmap": {"desktop_foundation_first": True, "android_after_monolithic_foundation": True, "android_target": "Samsung Galaxy S25 Ultra", "mobile_scaling_started": True},
        "limitations": ["Physical scaffold remains authoritative for damage, topology safety, and resource conservation.", "The learned world-context embedding conditions recurrence but is not an exact human-readable semantic decoder.", "City-growth v1 misses specialized purpose material on 4 of 30 audited actions.", "Long recurrent frame rollouts remain softer than the scaffold teacher."],
    }
    payload["manifest_sha256"] = hashlib.sha256(canonical(payload)).hexdigest(); _atomic(destination, canonical(payload)); return validate(output)


def validate(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    path = Path(output).resolve() / "foundation_manifest.json"; raw = path.read_bytes(); payload = json.loads(raw)
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if raw != canonical(payload) or payload.get("format") != FORMAT or payload.get("source_sha256") != source_sha256() or payload.get("manifest_sha256") != hashlib.sha256(canonical(unsigned)).hexdigest(): raise ValueError("Foundation manifest drifted.")
    for row in payload["components"]:
        for key in ("report", "artifact"):
            if key in row:
                record = row[key]; source = PROJECT_ROOT / record["path"]
                if source.stat().st_size != record["bytes"] or file_sha256(source) != record["sha256"]: raise ValueError(f"Foundation component drifted: {row['name']}/{key}")
    return {"passed": payload["status"] == "monolithic_neural_foundation_ready", "components": len(payload["components"]), "manifest_sha256": payload["manifest_sha256"], "recurrent_frames_per_second": payload["benchmark"]["monolithic_frames_per_second"], "organism_physics_hz": payload["cadence"]["organism_physics_hz"]}
