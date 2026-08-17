from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from ..recurrent_world_student_v5.model import PerceptionRecurrentWorldStudent
from ..recurrent_world_student_v6.training import _normalizers
from ..world_action_natural_v10 import load
from ..world_frame_decoder_adapt_v1 import AdaptedWorldFrameCodec
from ..world_frame_vae.contract import ModelConfig
from ..world_frame_vae.model import WorldFrameVAE
from ..world_latent_dit.contract import ModelConfig as RecurrentConfig
from .contract import DEFAULT_OUTPUT, NATURAL_CORPUS, PARENT_CODEC, RECURRENT, canonical, file_sha256


@torch.inference_mode()
def _continuous_latents(sequence, payload, model, device, start, frames):
    lm, ls, am, ass = _normalizers(payload, device)
    previous = torch.from_numpy(sequence["latent"][start - 1:start]).to(device)
    current = torch.from_numpy(sequence["latent"][start:start + 1]).to(device)
    previous_actor = torch.from_numpy(sequence["actor_state"][start - 1:start]).to(device)
    actor = torch.from_numpy(sequence["actor_state"][start:start + 1]).to(device)
    result = []
    for offset in range(frames):
        index = start + offset
        action = torch.from_numpy(sequence["action"][index + 1:index + 2].astype(np.int64)).to(device)
        control = torch.from_numpy(sequence["control"][index + 1:index + 2]).to(device)
        state = torch.from_numpy(sequence["state"][index:index + 1]).to(device)
        visibility = torch.from_numpy(sequence["visibility"][index:index + 1]).to(device)
        memory = torch.from_numpy(sequence["memory"][index:index + 1]).to(device)
        cn, pn = (current - lm) / ls, (previous - lm) / ls
        delta, logits = model.gated_action(cn, pn, action, control, state, actor, visibility, memory)
        applied = 1.5 * min(offset / 2, 1.0)
        next_latent = (cn + torch.sigmoid(logits + applied) * delta) * ls + lm
        an, pan = (actor - am) / ass, (previous_actor - am) / ass
        actor_result = model.actor(an, pan, action, control, state, visibility, memory)
        next_actor = (an + 0.9 * (actor_result.gate >= 0.7) * (actor_result.state - an)) * ass + am
        result.append(next_latent.float().cpu())
        previous, current = current, next_latent
        previous_actor, actor = actor, next_actor
    return torch.cat(result)


@torch.inference_mode()
def build(output: Path = DEFAULT_OUTPUT / "showcase", *, start: int = 96, frames: int = 64):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    sequences, natural = load(NATURAL_CORPUS)
    sequence = sequences[5]
    recurrent_payload = torch.load(RECURRENT, map_location="cpu", weights_only=True)
    recurrent = PerceptionRecurrentWorldStudent(RecurrentConfig(**recurrent_payload["model_config"]))
    recurrent.load_state_dict(recurrent_payload["state"])
    recurrent.to(device).eval()
    candidates = _continuous_latents(sequence, recurrent_payload, recurrent, device, start, frames)
    parent = AdaptedWorldFrameCodec.from_checkpoint(PARENT_CODEC, device="cuda").model
    release_path = DEFAULT_OUTPUT / "runtime.pt"
    release = torch.load(release_path, map_location="cpu", weights_only=True)
    adapted = WorldFrameVAE(ModelConfig(**release["model_config"]))
    adapted.load_state_dict(release["state"])
    adapted.to(device).eval()
    parent_frames, adapted_frames = [], []
    for begin in range(0, frames, 4):
        batch = candidates[begin:begin + 4].to(device)
        parent_frames.append(parent.decode(batch).float().cpu())
        adapted_frames.append(adapted.decode(batch).float().cpu())
    parent_frames = torch.cat(parent_frames).permute(0, 2, 3, 1).numpy()
    adapted_frames = torch.cat(adapted_frames).permute(0, 2, 3, 1).numpy()
    teacher = sequence["frame"][start + 1:start + frames + 1]
    font = ImageFont.load_default()
    temp = Path(tempfile.mkdtemp(prefix="nullvector-rollout-showcase-"))
    try:
        for index in range(frames):
            canvas = Image.new("RGB", (768, 280), (4, 10, 14))
            values = (teacher[index], np.clip(parent_frames[index] * 255, 0, 255).astype(np.uint8), np.clip(adapted_frames[index] * 255, 0, 255).astype(np.uint8))
            for column, value in enumerate(values):
                canvas.paste(Image.fromarray(value), (column * 256, 24))
            draw = ImageDraw.Draw(canvas)
            for column, label in enumerate(("TEACHER", "FROZEN DECODER", "ROLLOUT-AWARE DECODER")):
                draw.text((column * 256 + 8, 7), label, fill=(65, 235, 244) if column == 2 else (190, 205, 215), font=font)
            draw.text((690, 7), f"T+{index + 1:02d}", fill=(180, 255, 90), font=font)
            canvas.save(temp / f"{index:04d}.png")
        gif = output / "rollout_decoder_comparison.gif"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "12", "-i", str(temp / "%04d.png"), "-vf", "palettegen=stats_mode=diff", str(temp / "palette.png")], check=True)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "12", "-i", str(temp / "%04d.png"), "-i", str(temp / "palette.png"), "-lavfi", "paletteuse=dither=bayer:bayer_scale=3", "-loop", "0", str(gif)], check=True)
        contact = output / "rollout_decoder_contact.png"
        sample = Image.new("RGB", (768, 4 * 280), (4, 10, 14))
        for row, index in enumerate((0, 15, 31, 63)):
            sample.paste(Image.open(temp / f"{index:04d}.png"), (0, row * 280))
        sample.save(contact)
        report = {"format": "nullvector-world-frame-rollout-decoder-v2-showcase/1.0.0", "natural_corpus_sha256": natural["manifest_sha256"], "recurrent_sha256": file_sha256(RECURRENT), "parent_codec_sha256": file_sha256(PARENT_CODEC), "adapted_codec_sha256": file_sha256(release_path), "world": 5, "start": start, "frames": frames, "fps": 12, "artifacts": {"gif": {"path": gif.name, "bytes": gif.stat().st_size, "sha256": file_sha256(gif)}, "contact": {"path": contact.name, "bytes": contact.stat().st_size, "sha256": file_sha256(contact)}}}
        report["report_sha256"] = hashlib.sha256(canonical(report)).hexdigest()
        (output / "showcase.json").write_bytes(canonical(report))
        return report
    finally:
        shutil.rmtree(temp, ignore_errors=True)


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
