from __future__ import annotations

import numpy as np
import torch

from ..neural_world_state_v1.contract import CHECKPOINT_FORMAT as WORLD_CHECKPOINT_FORMAT, CONDITION_NAMES, CONTINUOUS_NAMES, WorldStateModelConfig
from ..neural_world_state_v1.model import build_model as build_world_codec
from ..recurrent_world_context_v1.contract import CHECKPOINT_FORMAT as ADAPTER_CHECKPOINT_FORMAT, ContextModelConfig
from ..recurrent_world_context_v1.model import build_model as build_adapter
from ..recurrent_world_pipeline_v1.runtime import RecurrentWorldPipeline
from .contract import CONTEXT_ADAPTER, CONTEXT_ADAPTER_SHA256, WORLD_STATE, WORLD_STATE_SHA256, file_sha256


class ContextualRecurrentWorldPipeline:
    def __init__(self, recurrent: RecurrentWorldPipeline, codec, adapter, device: torch.device) -> None:
        self.recurrent = recurrent; self.codec = codec; self.adapter = adapter; self.device = device; self.context_state = None

    @classmethod
    def load(cls, *, device: str = "cuda") -> "ContextualRecurrentWorldPipeline":
        if file_sha256(WORLD_STATE) != WORLD_STATE_SHA256 or file_sha256(CONTEXT_ADAPTER) != CONTEXT_ADAPTER_SHA256: raise ValueError("Contextual recurrent component drifted.")
        target = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu"); world_payload = torch.load(WORLD_STATE, map_location="cpu", weights_only=True); adapter_payload = torch.load(CONTEXT_ADAPTER, map_location="cpu", weights_only=True)
        if world_payload.get("format") != WORLD_CHECKPOINT_FORMAT or adapter_payload.get("format") != ADAPTER_CHECKPOINT_FORMAT or adapter_payload.get("world_state_sha256") != WORLD_STATE_SHA256: raise ValueError("Contextual recurrent provenance drifted.")
        codec = build_world_codec(WorldStateModelConfig(**world_payload["model_config"])); codec.load_state_dict(world_payload["state"], strict=True); codec.to(target).eval(); adapter = build_adapter(ContextModelConfig(**adapter_payload["model_config"])); adapter.load_state_dict(adapter_payload["state"], strict=True); adapter.to(target).eval(); return cls(RecurrentWorldPipeline.load(device=str(target)), codec, adapter, target)

    def initialize(self, previous_latent, current_latent, previous_actor, actor) -> None: self.recurrent.initialize(previous_latent, current_latent, previous_actor, actor)

    @torch.inference_mode()
    def observe_world(self, terrain, city, continuous, condition) -> np.ndarray:
        terrain = torch.as_tensor(terrain, device=self.device).long(); city = torch.as_tensor(city, device=self.device).long(); continuous = torch.as_tensor(continuous, device=self.device).float(); condition = torch.as_tensor(condition, device=self.device).float()
        if terrain.ndim == 2: terrain, city, continuous, condition = terrain[None], city[None], continuous[None], condition[None]
        if terrain.shape[-2:] != (32, 32) or city.shape != terrain.shape or continuous.shape != (len(terrain), len(CONTINUOUS_NAMES), 32, 32) or condition.shape != (len(terrain), len(CONDITION_NAMES)): raise ValueError("Contextual world observation shape drifted.")
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"): spatial, global_state, _, _ = self.codec.encode(terrain, city, continuous, condition, sample=False); context = torch.cat((global_state.float(), spatial.float().mean((2, 3))), 1); self.context_state = self.adapter(context)
        return self.context_state.float().cpu().numpy()

    @torch.inference_mode()
    def step(self, action, control, visibility, memory, *, decode: bool = True):
        if self.context_state is None: raise RuntimeError("observe_world must run before contextual recurrent stepping.")
        return self.recurrent.step(action, control, self.context_state, visibility, memory, decode=decode)

    @property
    def parameter_count(self) -> int: return self.recurrent.parameter_count + sum(parameter.numel() for module in (self.codec, self.adapter) for parameter in module.parameters())
