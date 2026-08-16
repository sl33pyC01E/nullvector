from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import uuid

import numpy as np
from PIL import Image
import torch

from .creature_stage_developmental.contract import FAMILIES, TISSUES
from .organism_raster_vae_v3.calibration import _canonical, _sha
from .organism_raster_vae_v3.contract import RasterVAEV3Config
from .organism_raster_vae_v5_anatomical.contract import CHECKPOINT_FORMAT
from .organism_raster_vae_v5_anatomical.dataset import AnatomicalGraphCorpus
from .organism_raster_vae_v5_anatomical.model import AnatomicalGraphRasterVAE
from .organism_raster_vae_v5_anatomical.training import _batch, source_sha256
from .safety import require_disk_floor


ROOT = Path(__file__).resolve().parents[1]
FORMAT = "nullvector-anatomical-creature-demo-bundle/1.0.0"
IDENTITIES = (5, 11, 17, 23, 29)


def _rgba(value: torch.Tensor) -> Image.Image:
    array = value.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray(np.rint(array * 255).astype(np.uint8), "RGBA")


def _local_points(organism, points: np.ndarray) -> np.ndarray:
    low = organism.cell_xy.min(0).astype(np.float32)
    high = organism.cell_xy.max(0).astype(np.float32)
    midpoint = (low + high) * .5
    return points.astype(np.float32) - midpoint[None]


@torch.inference_mode()
def sync(checkpoint: Path, destination: Path) -> Path:
    checkpoint = checkpoint.resolve()
    destination = destination.resolve()
    require_disk_floor(destination.parent, floor_gb=100, planned_bytes=1024**3)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["format"] != CHECKPOINT_FORMAT or payload["source_sha256"] != source_sha256():
        raise ValueError("demo checkpoint is not current anatomical authority")
    corpus = AnatomicalGraphCorpus()
    if payload["corpus_sha256"] != corpus.semantic_sha256:
        raise ValueError("demo checkpoint corpus drifted")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AnatomicalGraphRasterVAE(RasterVAEV3Config(**payload["config"])).to(device)
    model.load_state_dict(payload["ema_state"], strict=True)
    model.eval()

    atlas = Image.new("RGBA", (len(IDENTITIES) * 96, 16 * 96), (0, 0, 0, 0))
    specimens = []
    for family, identity in enumerate(IDENTITIES):
        organism = corpus.organisms[identity]
        rows = [identity * 16 + phase for phase in range(16)]
        for start in range(0, 16, 8):
            chosen = rows[start : start + 8]
            batch = _batch(corpus, chosen, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(
                    batch["living"], batch["family"], batch["traits"], batch["phase"],
                    batch["tokens"], batch["token_mask"], stochastic=False,
                )
            for offset, row_index in enumerate(chosen):
                _, phase = corpus.rows[row_index]
                atlas.alpha_composite(_rgba(output.rgba[offset]), (family * 96, phase * 96))

        components = []
        for index, component in enumerate(organism.genome.components):
            components.append({
                "index": index,
                "id": component.component_id,
                "kind": component.kind,
                "organ": component.organ,
                "anchor": [float(value) for value in _local_points(organism, np.asarray([component.anchor]))[0]],
                "radius": [float(value) for value in component.radius],
                "side": component.side,
            })
        component_owner = organism.component_weights.argmax(1)
        cells = []
        local_cells = _local_points(organism, organism.cell_xy)
        for index, point in enumerate(local_cells):
            component_index = int(component_owner[index])
            cells.append({
                "xy": [float(point[0]), float(point[1])],
                "tissue": TISSUES[int(organism.tissue[index])],
                "component": component_index,
                "organ": organism.genome.components[component_index].organ,
                "appendage": int(organism.appendage_index[index]),
                "side": int(organism.side[index]),
            })
        local_nodes = _local_points(organism, organism.skeleton_nodes[:, :2])
        specimens.append({
            "family": FAMILIES[family],
            "family_id": family,
            "identity": identity,
            "genome_id": organism.genome.genome_id,
            "identity_sha256": organism.identity_sha256,
            "traits": [float(value) for value in organism.genome.traits],
            "components": components,
            "cells": cells,
            "skeleton": {
                "nodes": [[float(x), float(y)] for x, y in local_nodes],
                "edges": [[int(a), int(b)] for a, b in organism.skeleton_edges],
                "edge_appendage": [int(value) for value in organism.skeleton_edge_appendage],
                "muscles": [[float(value) for value in row] for row in organism.muscles],
            },
        })

    staging = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    if destination.exists():
        raise FileExistsError(destination)
    staging.mkdir(parents=True)
    try:
        atlas_path = staging / "neural_motion_atlas.png"
        atlas.save(atlas_path, compress_level=7)
        anatomy_path = staging / "anatomy.json"
        anatomy_path.write_bytes(_canonical({"specimens": specimens}))
        manifest = {
            "format": FORMAT,
            "status": "ready",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "checkpoint": {
                "sha256": _sha(checkpoint),
                "source_sha256": payload["source_sha256"],
                "segment": payload["segment"],
                "global_step": payload["global_step"],
                "ema_state_sha256": payload["ema_state_sha256"],
            },
            "corpus_sha256": corpus.semantic_sha256,
            "layout": {"cell_size": 96, "families": 5, "phases": 16, "width": 480, "height": 1536},
            "families": list(FAMILIES),
            "artifacts": {
                "atlas": {"path": atlas_path.name, "sha256": _sha(atlas_path), "bytes": atlas_path.stat().st_size},
                "anatomy": {"path": anatomy_path.name, "sha256": _sha(anatomy_path), "bytes": anatomy_path.stat().st_size},
            },
            "capabilities": [
                "neural_raster", "sixteen_phase_motion", "cell_authority", "organ_authority",
                "joint_graph", "muscle_pairs", "damage_scaffold", "grounded_locomotion_scaffold",
            ],
        }
        manifest["manifest_sha256"] = hashlib.sha256(_canonical(manifest)).hexdigest()
        (staging / "manifest.json").write_bytes(_canonical(manifest))
        staging.replace(destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return destination / "manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(sync(args.checkpoint, args.output))


if __name__ == "__main__":
    main()
