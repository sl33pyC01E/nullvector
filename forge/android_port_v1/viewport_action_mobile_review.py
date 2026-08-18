from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..config import PROJECT_ROOT
from ..mobile_viewport_student_v1.model import splat_organisms
from ..whole_viewport_latent_v1.data import load_corpus, rows, split_episodes
from ..whole_viewport_latent_v1.decoder import load_decoder
from .viewport_action import ActionGraph, MobileGpuActionGraph, _load_action


@torch.inference_mode()
def review(release: Path, samples: int = 384):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, manifest = _load_action(release)
    original, mobile = ActionGraph(model).to(device), MobileGpuActionGraph(model).to(device)
    episodes, _ = load_corpus(PROJECT_ROOT / "outputs/action_teacher_viewport_v5/macro_corpus_v1")
    _, validation, _ = split_episodes(episodes); data = rows(validation)
    indices = np.linspace(0, len(data["frame"]) - 1, min(samples, len(data["frame"])), dtype=np.int64)
    vae, _ = load_decoder(device, Path(manifest["decoder"]["release"]))
    latent_total = rgb_total = 0.0
    for start in range(0, len(indices), 8):
        chosen = indices[start:start + 8]
        image = torch.as_tensor(data["previous_frame"][chosen], device=device).permute(0, 3, 1, 2).float() / 255
        latent = vae.encode(image)[0]
        spatial = torch.as_tensor(data["spatial"][chosen], device=device).float()
        organisms = torch.as_tensor(data["organisms"][chosen], device=device).float()
        mask = torch.as_tensor(data["organism_mask"][chosen], device=device).bool()
        common = [torch.as_tensor(data[name][chosen], device=device).float() for name in ("state", "actor_state", "actor_field", "visibility", "memory", "control")]
        action = torch.as_tensor(data["action"][chosen], device=device).long()
        reference = original(latent, spatial, organisms, mask, *common, action)
        field = splat_organisms(organisms, mask)
        candidate = mobile(latent, spatial, field, *common, F.one_hot(action, 22).float())
        latent_total += float(F.l1_loss(candidate, reference)) * len(chosen)
        rgb_total += float(F.l1_loss(vae.decode(candidate), vae.decode(reference))) * len(chosen)
    return {
        "format": "nullvector-mobile-gpu-action-approximation-review/1.0.0",
        "samples": len(indices),
        "latent_mae": latent_total / len(indices),
        "decoded_rgb_mae": rgb_total / len(indices),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=384)
    arguments = parser.parse_args()
    print(json.dumps(review(arguments.release, arguments.samples), indent=2))
