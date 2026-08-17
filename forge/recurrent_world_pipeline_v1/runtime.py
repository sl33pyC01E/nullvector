from __future__ import annotations

import numpy as np
import torch

from ..recurrent_world_student_v5.model import PerceptionRecurrentWorldStudent
from ..world_frame_vae.contract import ModelConfig as DecoderConfig
from ..world_frame_vae.model import WorldFrameVAE
from ..world_latent_dit.contract import ModelConfig as RecurrentConfig
from .contract import DECODER, DECODER_SHA256, RECURRENT, RECURRENT_SHA256, file_sha256


class RecurrentWorldPipeline:
    def __init__(self, recurrent, decoder, recurrent_payload, device):
        self.recurrent = recurrent
        self.decoder = decoder
        self.payload = recurrent_payload
        self.device = device
        normalization = recurrent_payload["normalization"]
        self.latent_mean = torch.tensor(normalization["latent_mean"], device=device)[None, :, None, None]
        self.latent_std = torch.tensor(normalization["latent_std"], device=device)[None, :, None, None]
        self.actor_mean = torch.tensor(normalization["actor_mean"], device=device)[None]
        self.actor_std = torch.tensor(normalization["actor_std"], device=device)[None]
        inference = recurrent_payload["inference"]
        self.bias = float(inference["gate_logit_bias_max"])
        self.ramp = int(inference["gate_logit_bias_ramp_steps"])
        self.previous = self.current = self.previous_actor = self.actor = None
        self.step_index = 0

    @classmethod
    def load(cls, device="cuda"):
        if file_sha256(RECURRENT) != RECURRENT_SHA256 or file_sha256(DECODER) != DECODER_SHA256:
            raise ValueError("recurrent world pipeline component drifted")
        target = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        recurrent_payload = torch.load(RECURRENT, map_location="cpu", weights_only=True)
        recurrent = PerceptionRecurrentWorldStudent(RecurrentConfig(**recurrent_payload["model_config"]))
        recurrent.load_state_dict(recurrent_payload["state"])
        recurrent.to(target).eval()
        decoder_payload = torch.load(DECODER, map_location="cpu", weights_only=True)
        decoder = WorldFrameVAE(DecoderConfig(**decoder_payload["model_config"]))
        decoder.load_state_dict(decoder_payload["state"])
        decoder.to(target).eval()
        return cls(recurrent, decoder, recurrent_payload, target)

    def initialize(self, previous_latent, current_latent, previous_actor, actor):
        self.previous = torch.as_tensor(previous_latent, device=self.device).float()
        self.current = torch.as_tensor(current_latent, device=self.device).float()
        self.previous_actor = torch.as_tensor(previous_actor, device=self.device).float()
        self.actor = torch.as_tensor(actor, device=self.device).float()
        if self.current.ndim == 3:
            self.previous, self.current = self.previous[None], self.current[None]
        if self.actor.ndim == 1:
            self.previous_actor, self.actor = self.previous_actor[None], self.actor[None]
        self.step_index = 0

    @torch.inference_mode()
    def step(self, action, control, state, visibility, memory, *, decode=True):
        if self.current is None:
            raise RuntimeError("recurrent world pipeline is not initialized")
        action = torch.as_tensor(action, device=self.device).long().reshape(-1)
        control = torch.as_tensor(control, device=self.device).float().reshape(len(action), -1)
        state = torch.as_tensor(state, device=self.device).float()
        visibility = torch.as_tensor(visibility, device=self.device).float()
        memory = torch.as_tensor(memory, device=self.device).float()
        cn, pn = (self.current - self.latent_mean) / self.latent_std, (self.previous - self.latent_mean) / self.latent_std
        delta, logits = self.recurrent.gated_action(cn, pn, action, control, state, self.actor, visibility, memory)
        applied = self.bias * min(self.step_index / self.ramp, 1.0) if self.ramp else self.bias
        next_latent = (cn + torch.sigmoid(logits + applied) * delta) * self.latent_std + self.latent_mean
        an, pan = (self.actor - self.actor_mean) / self.actor_std, (self.previous_actor - self.actor_mean) / self.actor_std
        actor_result = self.recurrent.actor(an, pan, action, control, state, visibility, memory)
        next_actor = (an + 0.9 * (actor_result.gate >= 0.7) * (actor_result.state - an)) * self.actor_std + self.actor_mean
        self.previous, self.current = self.current, next_latent
        self.previous_actor, self.actor = self.actor, next_actor
        self.step_index += 1
        if not decode:
            return next_latent
        frame = self.decoder.decode(next_latent).float().clamp_(0, 1).mul_(255).byte().permute(0, 2, 3, 1).cpu().numpy()
        return frame

    @property
    def parameter_count(self):
        return sum(parameter.numel() for module in (self.recurrent, self.decoder) for parameter in module.parameters())
