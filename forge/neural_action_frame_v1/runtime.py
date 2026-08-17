from __future__ import annotations

import numpy as np
import torch

from ..recurrent_action_dit_v2.runtime import RecurrentActionDiTRuntime
from ..world_frame_decoder_adapt_v1 import AdaptedWorldFrameCodec
from .contract import ACTION_CHECKPOINT_SHA256, ACTION_OUTPUT, CODEC_CHECKPOINT, CODEC_CHECKPOINT_SHA256, file_sha256


class NeuralActionFrameRuntime:
    def __init__(self, action, codec, *, residual_alpha: float = 1.0):
        self.action = action
        self.codec = codec
        self.device = action.device
        self.residual_alpha = float(residual_alpha)

    @classmethod
    def from_release(cls, *, device="cuda", residual_alpha=1.0):
        if file_sha256(ACTION_OUTPUT / "runtime.pt") != ACTION_CHECKPOINT_SHA256:
            raise ValueError("recurrent action release drifted")
        if file_sha256(CODEC_CHECKPOINT) != CODEC_CHECKPOINT_SHA256:
            raise ValueError("adapted decoder release drifted")
        action = RecurrentActionDiTRuntime.from_output(ACTION_OUTPUT, device=device)
        codec = AdaptedWorldFrameCodec.from_checkpoint(CODEC_CHECKPOINT, device=device)
        return cls(action, codec, residual_alpha=residual_alpha)

    @torch.inference_mode()
    def step(self, current_frame, current_latent, previous_latent, *, action, control, state, actor_state):
        frame = torch.as_tensor(current_frame)
        if frame.ndim == 4 and frame.shape[-1] == 3:
            frame = frame.permute(0, 3, 1, 2)
        if frame.dtype == torch.uint8:
            frame = frame.float().div_(255)
        else:
            frame = frame.float()
        if frame.ndim != 4 or frame.shape[1:] != (3, 256, 256):
            raise ValueError("current frame batch must be BCHW/HWC RGB 256x256")
        predicted = self.action.step(current_latent, previous_latent, action=action, control=control, state=state, actor_state=actor_state)
        current = torch.as_tensor(current_latent, dtype=torch.float32, device=self.codec.device)
        decoded_prediction = self.codec.model.decode(predicted.to(self.codec.device)).float().cpu()
        decoded_current = self.codec.model.decode(current).float().cpu()
        next_frame = torch.clamp(frame.cpu() + self.residual_alpha * (decoded_prediction - decoded_current), 0, 1)
        return next_frame, predicted.cpu()
