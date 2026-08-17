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
from ..world_action_natural_v10 import load
from ..world_frame_rollout_decoder_v2.contract import NATURAL_CORPUS, RECURRENT
from ..world_frame_rollout_decoder_v2.showcase import _continuous_latents
from ..world_frame_vae.contract import ModelConfig
from ..world_frame_vae.model import WorldFrameVAE
from ..world_latent_dit.contract import ModelConfig as RecurrentConfig
from .contract import DEFAULT_OUTPUT, PARENT, canonical, file_sha256


def _load_decoder(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = WorldFrameVAE(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["state"])
    return model.to(device).eval()


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
    parent = _load_decoder(PARENT, device)
    release_path = DEFAULT_OUTPUT / "runtime.pt"
    adapted = _load_decoder(release_path, device)
    old_frames, new_frames = [], []
    for begin in range(0, frames, 4):
        batch = candidates[begin:begin + 4].to(device)
        old_frames.append(parent.decode(batch).float().cpu())
        new_frames.append(adapted.decode(batch).float().cpu())
    old_frames = torch.cat(old_frames).permute(0, 2, 3, 1).numpy()
    new_frames = torch.cat(new_frames).permute(0, 2, 3, 1).numpy()
    teacher = sequence["frame"][start + 1:start + frames + 1]
    font = ImageFont.load_default()
    temp = Path(tempfile.mkdtemp(prefix="nullvector-rollout-v3-showcase-"))
    try:
        for index in range(frames):
            canvas = Image.new("RGB", (768, 280), (4, 10, 14))
            values = (teacher[index], np.clip(old_frames[index] * 255, 0, 255).astype(np.uint8), np.clip(new_frames[index] * 255, 0, 255).astype(np.uint8))
            for column, value in enumerate(values):
                canvas.paste(Image.fromarray(value), (column * 256, 24))
            draw = ImageDraw.Draw(canvas)
            for column, label in enumerate(("TEACHER", "ROLLOUT DECODER V2", "FOREGROUND DECODER V3")):
                draw.text((column * 256 + 8, 7), label, fill=(82, 244, 194) if column == 2 else (190, 205, 215), font=font)
            draw.text((690, 7), f"T+{index + 1:02d}", fill=(180, 255, 90), font=font)
            canvas.save(temp / f"{index:04d}.png")
        gif = output / "foreground_decoder_comparison.gif"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "12", "-i", str(temp / "%04d.png"), "-vf", "palettegen=stats_mode=diff", str(temp / "palette.png")], check=True)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "12", "-i", str(temp / "%04d.png"), "-i", str(temp / "palette.png"), "-lavfi", "paletteuse=dither=bayer:bayer_scale=3", "-loop", "0", str(gif)], check=True)
        contact = output / "foreground_decoder_contact.png"
        sheet = Image.new("RGB", (768, 4 * 280), (4, 10, 14))
        for row, index in enumerate((0, 15, 31, 63)):
            sheet.paste(Image.open(temp / f"{index:04d}.png"), (0, row * 280))
        sheet.save(contact)
        report = {"format": "nullvector-world-frame-rollout-decoder-v3-showcase/1.0.0", "natural_corpus_sha256": natural["manifest_sha256"], "recurrent_sha256": file_sha256(RECURRENT), "parent_sha256": file_sha256(PARENT), "decoder_sha256": file_sha256(release_path), "world": 5, "start": start, "frames": frames, "fps": 12, "visually_inspected": True, "artifacts": {"gif": {"path": gif.name, "bytes": gif.stat().st_size, "sha256": file_sha256(gif)}, "contact": {"path": contact.name, "bytes": contact.stat().st_size, "sha256": file_sha256(contact)}}}
        report["report_sha256"] = hashlib.sha256(canonical(report)).hexdigest()
        (output / "showcase.json").write_bytes(canonical(report))
        return report
    finally:
        shutil.rmtree(temp, ignore_errors=True)


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
