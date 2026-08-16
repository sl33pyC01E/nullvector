from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..creature_stage_neural_grasper_v1.feeding import FeedingState, FoodClump
from ..living_body_substrate import LivingBody
from ..living_body_substrate.contract import SYSTEMS
from .contract import CHECKPOINT_FORMAT, DynamicsConfig
from .corpus import encode_transition_features, organism_edges
from .model import LivingBodyDynamicsNet


@dataclass(frozen=True, slots=True)
class NeuralBodyTransition:
    health: np.ndarray
    fluid: np.ndarray
    scar: np.ndarray
    systems: dict[str, float]
    absorbed_mass: float
    nutrition: float
    reserve: float
    fullness_seconds: float
    energy: float
    released_energy: float
    contact_probability: float
    route_probability: float
    clump_mass: float
    safety_projected: bool


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class NeuralLivingBodyDynamicsRuntime:
    """Accepted learned cell/organ/feeding transition with physical bounds.

    The graph network is the state authority.  Projection enforces only facts
    that the pixel physics already knows: no intake without contact, no intake
    beyond available material or reserve capacity, and bounded cell fields.
    """

    def __init__(self, model: LivingBodyDynamicsNet, device: torch.device) -> None:
        self.model = model.eval().requires_grad_(False)
        self.device = device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Path,
        *,
        device: str = "cuda",
        evaluation_report: Path | None = None,
        require_promoted: bool = True,
    ) -> "NeuralLivingBodyDynamicsRuntime":
        from .training import source_sha256

        checkpoint = Path(checkpoint).resolve()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload.get("format") != CHECKPOINT_FORMAT or payload.get("source_sha256") != source_sha256():
            raise ValueError("living dynamics runtime checkpoint drifted")
        if require_promoted:
            if evaluation_report is None:
                raise ValueError("living dynamics runtime requires an acceptance report")
            report = json.loads(Path(evaluation_report).read_text(encoding="utf-8"))
            gates = report.get("gates", {})
            if report.get("source_sha256") != source_sha256() or report.get("checkpoint", {}).get("sha256") != _sha(checkpoint):
                raise ValueError("living dynamics acceptance provenance drifted")
            if not gates or not gates.get("production_promotion_allowed"):
                raise ValueError("living dynamics checkpoint is not promoted")
            if not all(value is True for value in gates.values()):
                raise ValueError("living dynamics acceptance gate failed")
        target = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        model = LivingBodyDynamicsNet(DynamicsConfig(**payload["config"]))
        model.load_state_dict(payload["ema_state"], strict=True)
        return cls(model.to(target), target)

    @torch.inference_mode()
    def predict(
        self,
        body: LivingBody,
        feeding: FeedingState,
        clump: FoodClump,
        *,
        contact: bool,
        action_kind: int,
        local_action: np.ndarray | None = None,
        activity: float = 0.0,
        delta: float = .1,
    ) -> NeuralBodyTransition:
        if local_action is None:
            local_action = np.zeros((body.organism.cell_count, 3), dtype=np.float32)
        features = encode_transition_features(
            body, action_kind=action_kind, local_action=local_action,
            feeding=feeding, clump=clump, contact=contact,
            activity=activity, delta=delta,
        )
        count = len(features)
        batch = {
            "features": torch.from_numpy(features).to(self.device),
            "edges": torch.from_numpy(organism_edges(body)).long().to(self.device),
            "graph_index": torch.zeros(count, dtype=torch.long, device=self.device),
            "family": torch.tensor([body.family], dtype=torch.long, device=self.device),
        }
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            cell, systems, feeding_value = self.model(batch)
        cell = cell.float().cpu().numpy()
        systems_value = systems[0].float().cpu().numpy()
        raw = feeding_value[0].float().cpu().numpy()
        contact_probability = float(1 / (1 + np.exp(-np.clip(raw[6], -30, 30))))
        route_probability = float(1 / (1 + np.exp(-np.clip(raw[7], -30, 30))))
        family_nutrition = float(clump.nutrition_by_family[body.family])
        requested_absorption = float(raw[0])
        available_nutrition = max(0.0, feeding.reserve_capacity - feeding.reserve)
        allowed_absorption = min(
            clump.mass,
            available_nutrition / max(clump.nutrient_density * family_nutrition, 1e-8),
        ) if family_nutrition > 0 else 0.0
        physical_gate = bool(contact) and route_probability >= .5 and not body.dead
        absorbed = min(requested_absorption, allowed_absorption) if physical_gate else 0.0
        possible_nutrition = max(0.0, absorbed * clump.nutrient_density * family_nutrition)
        nutrition = min(float(raw[1]) * 4.0, possible_nutrition, available_nutrition) if absorbed > 0 else 0.0
        reserve = float(np.clip(float(raw[2]) * feeding.reserve_capacity, 0, feeding.reserve_capacity))
        fullness = float(np.clip(float(raw[3]) * feeding.fullness_capacity_seconds, 0, feeding.fullness_capacity_seconds))
        energy = float(np.clip(float(raw[4]) * 4.0, 0, 4.0))
        released = float(np.clip(float(raw[5]) * .01, 0, feeding.reserve))
        # Mass conservation is a physical invariant, not a learned preference.
        remaining_mass = max(0.0, clump.mass - absorbed)
        if not physical_gate:
            absorbed = nutrition = 0.0
            remaining_mass = clump.mass
            if action_kind in (4, 6):
                reserve = feeding.reserve
                fullness = feeding.fullness_seconds
        return NeuralBodyTransition(
            health=np.clip(cell[:, 0], 0, 1).astype(np.float32),
            fluid=np.clip(cell[:, 1], 0, body.fluid_capacity).astype(np.float32),
            scar=np.clip(cell[:, 2], 0, 1).astype(np.float32),
            systems={name: float(np.clip(systems_value[index], 0, 1)) for index, name in enumerate(SYSTEMS)},
            absorbed_mass=float(absorbed), nutrition=float(nutrition), reserve=reserve,
            fullness_seconds=fullness, energy=energy, released_energy=released,
            contact_probability=contact_probability, route_probability=route_probability,
            clump_mass=remaining_mass, safety_projected=True,
        )

    def apply(
        self,
        body: LivingBody,
        feeding: FeedingState,
        clump: FoodClump,
        transition: NeuralBodyTransition,
    ) -> None:
        body.health = transition.health.copy()
        body.fluid = transition.fluid.copy()
        body.scar = transition.scar.copy()
        body.energy = transition.energy
        body._connectivity_alive = None
        body._connectivity_connected = None
        body._systems_health = None
        body._systems_fluid = None
        body._systems_cache = None
        feeding.reserve = transition.reserve
        feeding.fullness_seconds = transition.fullness_seconds
        feeding.consumed_mass += transition.absorbed_mass
        clump.mass = transition.clump_mass
