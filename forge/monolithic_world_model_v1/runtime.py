from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..world_frame_vae.contract import ModelConfig as DecoderConfig
from ..world_frame_vae.model import WorldFrameVAE
from ..world_latent_dit.contract import ModelConfig as RecurrentConfig
from .contract import CHECKPOINT_FORMAT, DECODER, DECODER_SHA256, DirectContextConfig, file_sha256
from .model import FusedStructuredActionModel


class MonolithicWorldRuntime:
    """Stateful runtime with one fused action model followed by one frame VAE."""

    def __init__(self, model, decoder, payload, device: torch.device) -> None:
        self.model = model; self.decoder = decoder; self.payload = payload; self.device = device
        norm = payload["normalization"]
        self.latent_mean = torch.tensor(norm["latent_mean"], device=device)[None, :, None, None]
        self.latent_std = torch.tensor(norm["latent_std"], device=device)[None, :, None, None]
        self.actor_mean = torch.tensor(norm["actor_mean"], device=device)[None]
        self.actor_std = torch.tensor(norm["actor_std"], device=device)[None]
        self.bias = float(payload["inference"]["gate_logit_bias_max"])
        self.ramp = int(payload["inference"]["gate_logit_bias_ramp_steps"])
        self.previous = self.current = self.previous_actor = self.actor = None
        self.step_index = 0

    @classmethod
    def load(cls, checkpoint: Path, *, device: str = "cuda") -> "MonolithicWorldRuntime":
        target = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        if payload.get("format") != CHECKPOINT_FORMAT or payload.get("status") != "monolithic_foundation_ready":
            raise ValueError("Monolithic world runtime checkpoint is not promoted.")
        model = FusedStructuredActionModel(DirectContextConfig(**payload["model_config"]), RecurrentConfig(**payload["recurrent_config"]))
        model.load_state_dict(payload["state"], strict=True)
        if file_sha256(DECODER) != DECODER_SHA256:
            raise ValueError("Monolithic frame VAE drifted.")
        decoder_payload = torch.load(DECODER, map_location="cpu", weights_only=True)
        decoder = WorldFrameVAE(DecoderConfig(**decoder_payload["model_config"]))
        decoder.load_state_dict(decoder_payload["state"], strict=True)
        return cls(model.to(target).eval(), decoder.to(target).eval(), payload, target)

    def initialize(self, previous_latent, current_latent, previous_actor, actor) -> None:
        self.previous = torch.as_tensor(previous_latent, device=self.device).float()
        self.current = torch.as_tensor(current_latent, device=self.device).float()
        self.previous_actor = torch.as_tensor(previous_actor, device=self.device).float()
        self.actor = torch.as_tensor(actor, device=self.device).float()
        if self.current.ndim == 3: self.previous, self.current = self.previous[None], self.current[None]
        if self.actor.ndim == 1: self.previous_actor, self.actor = self.previous_actor[None], self.actor[None]
        self.step_index = 0

    @torch.inference_mode()
    def step(self, action, control, terrain, city, continuous, condition, visibility, memory, *, decode: bool = True):
        action = torch.as_tensor(action, device=self.device).long().reshape(-1)
        control = torch.as_tensor(control, device=self.device).float().reshape(len(action), -1)
        terrain = torch.as_tensor(terrain, device=self.device).long(); city = torch.as_tensor(city, device=self.device).long()
        continuous = torch.as_tensor(continuous, device=self.device).float(); condition = torch.as_tensor(condition, device=self.device).float()
        if terrain.ndim == 2: terrain, city, continuous, condition = terrain[None], city[None], continuous[None], condition[None]
        visibility = torch.as_tensor(visibility, device=self.device).float(); memory = torch.as_tensor(memory, device=self.device).float()
        cn, pn = (self.current - self.latent_mean) / self.latent_std, (self.previous - self.latent_mean) / self.latent_std
        state = self.model.observe(terrain, city, continuous, condition)
        delta, logits = self.model.recurrent.gated_action(cn, pn, action, control, state, self.actor, visibility, memory)
        applied = self.bias * min(self.step_index / self.ramp, 1.0) if self.ramp else self.bias
        next_latent = (cn + torch.sigmoid(logits + applied) * delta) * self.latent_std + self.latent_mean
        an, pan = (self.actor - self.actor_mean) / self.actor_std, (self.previous_actor - self.actor_mean) / self.actor_std
        actor_result = self.model.recurrent.actor(an, pan, action, control, state, visibility, memory)
        next_actor = (an + .9 * (actor_result.gate >= .7) * (actor_result.state - an)) * self.actor_std + self.actor_mean
        self.previous, self.current = self.current, next_latent
        self.previous_actor, self.actor = self.actor, next_actor
        self.step_index += 1
        if not decode: return next_latent
        return self.decoder.decode(next_latent).float().clamp_(0, 1).mul_(255).byte().permute(0, 2, 3, 1).cpu().numpy()

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for module in (self.model, self.decoder) for parameter in module.parameters())
