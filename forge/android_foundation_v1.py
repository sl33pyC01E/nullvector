from __future__ import annotations

from dataclasses import asdict
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
import torch
from torch import nn

from .config import PROJECT_ROOT
from .creature_stage_developmental.contract import FAMILIES, TISSUES
from .creature_stage_developmental.development import develop
from .creature_stage_developmental.genomes import review_genomes
from .creature_stage_neural_grounded_feedback_v2.contract import (
    MAX_APPENDAGES,
    MAX_MUSCLES,
)
from .creature_stage_neural_grounded_feedback_v2.runtime import NeuralGroundedFeedbackRuntime
from .multifield_style_motion.hashing import canonical_json_bytes, sha256_file
from .organism_cell_vae_runtime_v1.runtime import ContinuousCellVAERuntime
from .safety import require_disk_floor


FORMAT = "nullvector-android-foundation-runtime/1.0.0"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/android_foundation_v1/live_grounded_v1"
CHECKPOINT = PROJECT_ROOT / "outputs/creature_stage_neural_grounded_feedback_v2/production_3000_v2/runtime.pt"
SOURCE_FILES = (
    "forge/android_foundation_v1.py",
    "forge/creature_stage_neural_grounded_feedback_v2/contract.py",
    "forge/creature_stage_neural_grounded_feedback_v2/dataset.py",
    "forge/creature_stage_neural_grounded_feedback_v2/model.py",
    "forge/creature_stage_neural_grounded_feedback_v2/runtime.py",
    "forge/creature_stage_developmental/contract.py",
    "forge/creature_stage_developmental/development.py",
    "forge/creature_stage_developmental/genomes.py",
    "forge/organism_cell_vae_runtime_v1/runtime.py",
    "forge/organism_cell_vae_v1/model.py",
)


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-android-foundation-runtime-v1\0")
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        digest.update(relative.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


class _GroundedExport(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, owner_state, global_state, owner_mask, muscle_meta, muscle_owner, muscle_mask):
        result = self.model(owner_state, global_state, owner_mask, muscle_meta, muscle_owner, muscle_mask)
        return result.muscle_activation, result.contact_logits, result.body_velocity


def _organism_record(genome, organism, styles: np.ndarray) -> dict[str, object]:
    return {
        "family": FAMILIES[int(np.argmax(genome.family_mix))],
        "family_id": int(np.argmax(genome.family_mix)),
        "genome_id": genome.genome_id,
        "seed": genome.seed,
        "traits": list(map(float, genome.traits)),
        "components": [asdict(value) for value in genome.components],
        "appendages": [asdict(value) for value in genome.appendages],
        "cells": [
            {
                "xy": [float(x), float(y)],
                "tissue": int(tissue),
                "appendage": int(owner),
                "component": int(component),
            }
            for (x, y), tissue, owner, component, style in zip(
                organism.cell_xy,
                organism.tissue,
                organism.appendage_index,
                np.argmax(organism.component_weights, axis=1),
                styles,
                strict=True,
            )
        ],
        "skeleton": {
            "nodes": organism.skeleton_nodes[:, :2].astype(float).tolist(),
            "edges": organism.skeleton_edges.astype(int).tolist(),
            "edge_appendage": organism.skeleton_edge_appendage.astype(int).tolist(),
            "muscles": np.asarray(organism.muscles, np.float32).astype(float).tolist(),
        },
    }


def build(destination: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    require_disk_floor(destination.parent, floor_gb=100, planned_bytes=128 * 1024**2)
    runtime = NeuralGroundedFeedbackRuntime.from_checkpoint(CHECKPOINT, device="cpu")
    raster = ContinuousCellVAERuntime.from_release(device="cpu")
    model = _GroundedExport(runtime.model).eval()
    genomes = review_genomes()[::2]
    organisms = tuple(develop(genome) for genome in genomes)
    if tuple(FAMILIES[int(np.argmax(item.genome.family_mix))] for item in organisms) != FAMILIES:
        raise ValueError("Android foundation family roster drifted")
    stage = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    stage.mkdir(parents=True)
    try:
        neural_styles = []
        for organism in organisms:
            features, mask = raster.organism_features(organism, organism.cell_xy, phase=0.0)
            neural_styles.append(raster.cell_styles(features, mask)[0, :organism.cell_count].numpy())
        records = []
        for genome, organism, styles in zip(genomes, organisms, neural_styles, strict=True):
            record = _organism_record(genome, organism, styles)
            for cell, style in zip(record["cells"], styles, strict=True):
                cell["neural_style"] = [round(float(value), 7) for value in style]
            records.append(record)
        anatomy = {
            "format": FORMAT,
            "source_sha256": source_sha256(),
            "families": list(FAMILIES),
            "tissues": list(TISSUES),
            "organisms": records,
        }
        anatomy_bytes = canonical_json_bytes(anatomy)
        (stage / "foundation_anatomy.json").write_bytes(anatomy_bytes)

        batch = 2
        inputs = (
            torch.zeros(batch, MAX_APPENDAGES, 23),
            torch.zeros(batch, 23),
            torch.ones(batch, MAX_APPENDAGES, dtype=torch.bool),
            torch.zeros(batch, MAX_MUSCLES, 8),
            torch.zeros(batch, MAX_MUSCLES, dtype=torch.long),
            torch.ones(batch, MAX_MUSCLES, dtype=torch.bool),
        )
        onnx_path = stage / "grounded_feedback_fp32.onnx"
        torch.onnx.export(
            model,
            inputs,
            onnx_path,
            input_names=("owner_state", "global_state", "owner_mask", "muscle_meta", "muscle_owner", "muscle_mask"),
            output_names=("muscle_activation", "contact_logits", "body_velocity"),
            dynamic_axes={
                "owner_state": {0: "batch"}, "global_state": {0: "batch"}, "owner_mask": {0: "batch"},
                "muscle_meta": {0: "batch"}, "muscle_owner": {0: "batch"}, "muscle_mask": {0: "batch"},
                "muscle_activation": {0: "batch"}, "contact_logits": {0: "batch"}, "body_velocity": {0: "batch"},
            },
            opset_version=17,
            dynamo=False,
        )
        artifacts = {
            name: {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for name, path in (("anatomy", stage / "foundation_anatomy.json"), ("grounded_controller", onnx_path))
        }
        manifest = {
            "format": FORMAT,
            "status": "ready",
            "source_sha256": source_sha256(),
            "checkpoint": {"path": CHECKPOINT.relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256_file(CHECKPOINT)},
            "scope": {
                "families": 5,
                "organisms": 5,
                "live_state_controller": True,
                "physics_authoritative": True,
                "frame_loop": False,
                "analog_ground_plane_motion": True,
                "independent_elevation": True,
                "vae_cell_style_authority": True,
            },
            "model": {"parameters": runtime.model.parameter_count, "max_appendages": MAX_APPENDAGES, "max_muscles": MAX_MUSCLES},
            "artifacts": artifacts,
        }
        manifest["semantic_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
        (stage / "foundation_manifest.json").write_bytes(canonical_json_bytes(manifest))
        os.replace(stage, destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return validate(destination)


def validate(destination: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    destination = Path(destination).resolve()
    raw = (destination / "foundation_manifest.json").read_bytes()
    manifest = json.loads(raw)
    if raw != canonical_json_bytes(manifest) or manifest.get("format") != FORMAT or manifest.get("status") != "ready":
        raise ValueError("Android foundation manifest drifted")
    semantic = manifest.pop("semantic_sha256")
    if semantic != hashlib.sha256(canonical_json_bytes(manifest)).hexdigest():
        raise ValueError("Android foundation semantic hash drifted")
    manifest["semantic_sha256"] = semantic
    if manifest.get("source_sha256") != source_sha256() or sha256_file(CHECKPOINT) != manifest["checkpoint"]["sha256"]:
        raise ValueError("Android foundation provenance drifted")
    for artifact in manifest["artifacts"].values():
        path = destination / artifact["path"]
        if not path.is_file() or path.stat().st_size != artifact["bytes"] or sha256_file(path) != artifact["sha256"]:
            raise ValueError("Android foundation artifact drifted")
    anatomy = json.loads((destination / manifest["artifacts"]["anatomy"]["path"]).read_bytes())
    if anatomy.get("families") != list(FAMILIES) or len(anatomy.get("organisms", [])) != 5:
        raise ValueError("Android foundation anatomy roster drifted")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export live grounded foundation assets for Android")
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build"); build_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    validate_parser = sub.add_parser("validate"); validate_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build(args.output) if args.command == "build" else validate(args.output)
    print(json.dumps({"status": report["status"], "scope": report["scope"], "model": report["model"], "semantic_sha256": report["semantic_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
