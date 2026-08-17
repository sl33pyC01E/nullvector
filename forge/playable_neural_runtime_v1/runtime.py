from __future__ import annotations

import json
from dataclasses import dataclass

import torch

from ..composite_world_v1 import CompositeWorldRuntime
from ..config import PROJECT_ROOT
from ..creature_stage_neural_locomotion_25d.runtime import NeuralLocomotionRuntime
from ..nature_behavior_nn.contract import CHECKPOINT_FORMAT, ModelConfig
from ..nature_behavior_nn.model import NeuralNatureBehavior
from ..nature_behavior_nn.runtime import NeuralBehaviorRuntime
from ..nature_behavior_nn.training import _state_hash
from ..nature_colony_nn import NeuralColonyRuntime
from ..nature_counterfactual_nn import NeuralCounterfactualRuntime
from ..nature_society_nn import NeuralSocietyRuntime
from ..nature_timeline_nn import NeuralTimelineRuntime
from .contract import COMPOSITE, ENSEMBLE, canonical, file_sha256


def _canonical_manifest(path):
    raw = path.read_bytes()
    payload = json.loads(raw)
    if raw != canonical(payload):
        raise ValueError(f"non-canonical neural release manifest: {path.name}")
    return payload


def _component_table() -> dict[str, dict]:
    ensemble = _canonical_manifest(ENSEMBLE)
    if ensemble.get("status") != "teacher_ensemble_ready" or not ensemble.get("authority", {}).get("all_quality_gates_passed"):
        raise ValueError("teacher ensemble is not promoted")
    rows = {row["name"]: row for row in ensemble["components"]}
    required = {"locomotion_25d", "behavior", "colony", "society", "timeline", "counterfactual"}
    if not required <= rows.keys():
        raise ValueError("playable neural release is incomplete")
    for name in required:
        row = rows[name]
        if not all(row["quality_gates"].values()):
            raise ValueError(f"failed neural component in release: {name}")
        for key in ("report", "artifact"):
            record = row[key]
            path = PROJECT_ROOT / record["path"]
            if path.stat().st_size != record["bytes"] or file_sha256(path) != record["sha256"]:
                raise ValueError(f"neural release artifact drifted: {name}/{key}")
    composite = _canonical_manifest(COMPOSITE)
    quality = composite.get("quality", {})
    capabilities = composite.get("capabilities", {})
    if composite.get("status") != "composite_neural_foundation_ready" or not quality.get("runtime_loader_probe") or not all(capabilities.values()):
        raise ValueError("composite neural world is not callable")
    return rows


def _load_behavior(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    report = json.loads(path.with_suffix(".json").read_text("utf-8"))
    selected = payload.get("selected")
    state = payload.get(selected, {})
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("pinned behavior checkpoint format drifted")
    if payload.get("source_sha256") != report.get("source_sha256") or report.get("checkpoint_sha256") != file_sha256(path):
        raise ValueError("pinned behavior release provenance drifted")
    if _state_hash(state) != payload.get(f"{selected}_state_sha256"):
        raise ValueError("pinned behavior state drifted")
    model = NeuralNatureBehavior(ModelConfig(**payload["model_config"]))
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return NeuralBehaviorRuntime(model, device=device, decision_interval=3)


@dataclass(slots=True)
class PlayableNeuralRuntime:
    """The promoted neural ensemble exposed through the live game interfaces."""

    composite: CompositeWorldRuntime
    locomotion: NeuralLocomotionRuntime
    behavior: NeuralBehaviorRuntime
    colony: NeuralColonyRuntime
    society: NeuralSocietyRuntime
    timeline: NeuralTimelineRuntime
    counterfactual: NeuralCounterfactualRuntime
    component_count: int

    @classmethod
    def from_release(cls, *, device: str = "cuda") -> "PlayableNeuralRuntime":
        target = device if device != "cuda" or torch.cuda.is_available() else "cpu"
        rows = _component_table()
        artifact = lambda name: PROJECT_ROOT / rows[name]["artifact"]["path"]
        return cls(
            composite=CompositeWorldRuntime.from_release(device=target),
            locomotion=NeuralLocomotionRuntime.from_checkpoint(artifact("locomotion_25d"), device=target),
            behavior=_load_behavior(artifact("behavior"), target),
            colony=NeuralColonyRuntime.from_checkpoint(artifact("colony"), device=target),
            society=NeuralSocietyRuntime.from_checkpoint(artifact("society"), device=target),
            timeline=NeuralTimelineRuntime.from_checkpoint(artifact("timeline"), device=target),
            counterfactual=NeuralCounterfactualRuntime.from_checkpoint(artifact("counterfactual"), device=target),
            component_count=len(rows),
        )
