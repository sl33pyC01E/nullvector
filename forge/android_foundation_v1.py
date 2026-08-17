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
from .creature_stage_neural_grasper_v1.contract import MAX_APPENDAGES as GRASPER_MAX_APPENDAGES
from .creature_stage_neural_grasper_v1.runtime import NeuralGrasperRuntime
from .multifield_style_motion.hashing import canonical_json_bytes, sha256_file
from .organism_cell_vae_runtime_v1.runtime import ContinuousCellVAERuntime
from .safety import require_disk_floor


FORMAT = "nullvector-android-foundation-runtime/1.0.0"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/android_foundation_v1/live_grounded_v1"
CHECKPOINT = PROJECT_ROOT / "outputs/creature_stage_neural_grounded_feedback_v2/production_3000_v2/runtime.pt"
GRASPER_CHECKPOINT = PROJECT_ROOT / "outputs/creature_stage_neural_grasper_v1/production_v3_physical_feeder/runtime.pt"
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
    "forge/creature_stage_neural_grasper_v1/contract.py",
    "forge/creature_stage_neural_grasper_v1/model.py",
    "forge/creature_stage_neural_grasper_v1/runtime.py",
)

NCA_STATIC_CHANNELS = 85
NCA_DYNAMIC_CHANNELS = 12
NCA_BOND_CHANNELS = 8
NCA_CANVAS = 48
NCA_DIRECTIONS = ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))


def _nca_tissue_channel(tissue: int) -> int:
    # Developmental tissue vocabulary -> cellular-NCA tissue vocabulary.
    return (1, 3, 2, 2, 10, 4, 6, 6, 7, 5, 9, 14, 12, 3, 11)[tissue]


def _system_role(system: int, organ: str, kind: str, tissue: int, appendage: int) -> int:
    organ = organ.lower()
    if system == 0:  # circulation
        if organ in {"heart", "vascular", "coolant_pump", "bulb"}: return 1
        if tissue == 6: return 2
        if kind in {"soma", "circulator"}: return 3
    elif system == 1:  # respiration / gas or heat exchange
        if organ in {"lung", "coolant_pump", "photoreceptor", "singularity"}: return 1
        if tissue in {6, 7}: return 2
        if kind in {"respirator", "sensor_crown", "soma"}: return 3
    elif system == 2:  # digestion / energy conversion
        if organ in {"gut", "jaw", "transmuter", "battery", "bulb"}: return 1
        if tissue in {8, 10}: return 2
        if kind in {"mouth", "gut", "pelvis", "storage"}: return 3
    elif system == 3:  # neural control
        if organ in {"brain", "phase_brain", "processor", "meristem", "singularity"}: return 1
        if tissue == 5: return 2
        if kind in {"head", "neural_cluster", "sensor_crown"}: return 3
    elif system == 4:  # sensory
        if organ in {"eye", "optic", "photoreceptor", "singularity"}: return 1
        if tissue in {5, 9}: return 2
        if kind == "sensor_crown": return 3
    elif system == 5:  # locomotion
        if appendage >= 0 and tissue in {1, 2, 3, 11, 13}: return 3
        if tissue in {2, 3}: return 2
        if kind in {"pelvis", "soma"}: return 1
    elif system == 6:  # reproduction / growth
        if organ in {"gut", "bulb", "meristem", "battery", "phase_brain"}: return 1
        if tissue in {8, 10, 11}: return 2
        if kind in {"pelvis", "storage", "soma"}: return 3
    else:  # immune / repair
        if organ in {"heart", "bulb", "phase_brain", "processor"}: return 1
        if tissue in {0, 4, 6, 12, 13}: return 2
        if kind in {"soma", "armor", "generator"}: return 3
    return 0


def _physiology_fields(genome, organism) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    static = np.zeros((NCA_STATIC_CHANNELS, NCA_CANVAS, NCA_CANVAS), np.float32)
    state = np.zeros((NCA_DYNAMIC_CHANNELS, NCA_CANVAS, NCA_CANVAS), np.float32)
    bonds = np.zeros((NCA_BOND_CHANNELS, NCA_CANVAS, NCA_CANVAS), np.float32)
    pixels = np.rint(organism.cell_xy).astype(np.int16) + NCA_CANVAS // 2
    if np.any(pixels < 0) or np.any(pixels >= NCA_CANVAS) or len(np.unique(pixels, axis=0)) != organism.cell_count:
        raise ValueError("Android foundation NCA raster is not a one-cell-per-pixel anatomy")
    component_index = np.argmax(organism.component_weights, axis=1)
    family = int(np.argmax(genome.family_mix))
    trait = {name: float(genome.traits[index]) for index, name in enumerate((
        "size", "symmetry", "segmentation", "stiffness", "elasticity", "bone_density",
        "muscle_density", "muscle_strength", "neural_density", "vascularity", "metabolism",
        "regeneration", "grip", "sensory_range", "phase_coherence",
    ))}
    for index, ((x, y), tissue_raw, owner, component_raw) in enumerate(zip(
        pixels, organism.tissue, organism.appendage_index, component_index, strict=True,
    )):
        tissue = int(tissue_raw); component = genome.components[int(component_raw)]
        static[0, y, x] = 1
        static[_nca_tissue_channel(tissue), y, x] = 1
        flags = 0
        if component.organ in {"eye", "optic", "photoreceptor", "singularity"}: flags |= 1
        if component.kind == "mouth" or component.organ == "jaw": flags |= 2
        if component.organ in {"heart", "vascular", "coolant_pump", "bulb"}: flags |= 4
        if component.kind in {"pelvis", "storage"}: flags |= 8
        if family == 2 and tissue in {9, 11}: flags |= 16
        if component.kind in {"soma", "neural_cluster"}: flags |= 32
        if tissue == 14: flags |= 64
        if family == 3 or tissue == 12: flags |= 128
        for bit in range(8): static[15 + bit, y, x] = float((flags >> bit) & 1)
        static[23 + family, y, x] = 1
        for system in range(8):
            role = _system_role(system, component.organ, component.kind, tissue, int(owner))
            if role:
                static[28 + system * 3 + role - 1, y, x] = 1
                base = (.62, .58, .62, .60, .56, .65, .48, .56)[system]
                modifiers = (trait["vascularity"], trait["vascularity"], trait["metabolism"], trait["neural_density"], trait["sensory_range"], trait["muscle_strength"], trait["metabolism"], trait["regeneration"])
                static[52 + system, y, x] = np.clip(base + .38 * modifiers[system], .1, 1)
        heal_class = 1 + min(5, family + (1 if tissue in {1, 4, 13} else 0))
        static[60 + heal_class - 1, y, x] = 1
        static[66, y, x] = np.clip(.34 + .55 * trait["vascularity"], 0, 1)
        static[67, y, x] = np.clip(.18 + .34 * (1 - trait["regeneration"]), 0, 1)
        static[68, y, x] = np.clip(.18 + .76 * trait["regeneration"], 0, 1)
        static[69:75, y, x] = (
            .72 + .18 * trait["stiffness"], .55 + .35 * trait["vascularity"],
            .55 + .25 * trait["metabolism"], .60 + .22 * trait["metabolism"],
            .38 + .35 * trait["bone_density"], .35 + .55 * trait["stiffness"],
        )
        state[0, y, x] = 1
        state[1, y, x] = .72 if family != 4 else .62
        state[2, y, x] = .62
        state[3, y, x] = .74
        state[4, y, x] = .88
        state[8, y, x] = np.clip(static[37:40, y, x].sum() * .7 + static[40:43, y, x].sum() * .18, .05, 1)
        state[11, y, x] = 1
    lookup = {tuple(map(int, pixel)): index for index, pixel in enumerate(pixels)}
    for x, y in lookup:
        degree = 0; conductance = 0.0
        for direction_index, (dx, dy) in enumerate(NCA_DIRECTIONS):
            if (x + dx, y + dy) in lookup:
                strength = .82 if abs(dx) + abs(dy) == 1 else .64
                bonds[direction_index, y, x] = 1
                static[77 + direction_index, y, x] = strength
                degree += 1; conductance += strength
        static[75, y, x] = degree / 8
        static[76, y, x] = conductance / max(degree, 1) / .8
    return static, state, bonds, pixels


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


class _GrasperExport(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__(); self.model = model

    def forward(self, owner_meta, owner_mask, target, global_state):
        result = self.model(owner_meta, owner_mask, target, global_state)
        return (
            result.appendage_logits, result.engage_logit, result.reach, result.force,
            result.type_logits, result.brace, result.release_logit, result.throw_impulse,
        )


def _organism_record(genome, organism, styles: np.ndarray, pixels: np.ndarray) -> dict[str, object]:
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
                "nca_xy": [int(pixel[0]), int(pixel[1])],
            }
            for (x, y), tissue, owner, component, style, pixel in zip(
                organism.cell_xy,
                organism.tissue,
                organism.appendage_index,
                np.argmax(organism.component_weights, axis=1),
                styles,
                pixels,
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
    grasper_runtime = NeuralGrasperRuntime.from_checkpoint(GRASPER_CHECKPOINT, device="cpu")
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
        fields = [_physiology_fields(genome, organism) for genome, organism in zip(genomes, organisms, strict=True)]
        records = []
        for genome, organism, styles, (_, _, _, pixels) in zip(genomes, organisms, neural_styles, fields, strict=True):
            record = _organism_record(genome, organism, styles, pixels)
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
        for name, member in (
            ("foundation_cell_static.f32", 0), ("foundation_cell_state.f32", 1), ("foundation_cell_bonds.f32", 2),
        ):
            np.stack([entry[member] for entry in fields]).astype("<f4").tofile(stage / name)

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
        grasper_path = stage / "neural_grasper_fp32.onnx"
        grasper_model = _GrasperExport(grasper_runtime.model).eval()
        grasper_inputs = (
            torch.zeros(2, GRASPER_MAX_APPENDAGES, 16), torch.ones(2, GRASPER_MAX_APPENDAGES, dtype=torch.bool),
            torch.zeros(2, 18), torch.zeros(2, 10),
        )
        torch.onnx.export(
            grasper_model, grasper_inputs, grasper_path,
            input_names=("owner_meta", "owner_mask", "target", "global_state"),
            output_names=("appendage_logits", "engage_logit", "reach", "force", "type_logits", "brace", "release_logit", "throw_impulse"),
            dynamic_axes={name: {0: "batch"} for name in (
                "owner_meta", "owner_mask", "target", "global_state", "appendage_logits", "engage_logit",
                "reach", "force", "type_logits", "brace", "release_logit", "throw_impulse",
            )},
            opset_version=17, dynamo=False,
        )
        artifacts = {
            name: {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for name, path in (
                ("anatomy", stage / "foundation_anatomy.json"), ("grounded_controller", onnx_path),
                ("cell_static", stage / "foundation_cell_static.f32"),
                ("cell_state", stage / "foundation_cell_state.f32"),
                ("cell_bonds", stage / "foundation_cell_bonds.f32"),
                ("neural_grasper", grasper_path),
            )
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
                "neural_cell_physiology": True,
                "five_family_organ_fields": True,
            },
            "model": {"parameters": runtime.model.parameter_count, "grasper_parameters": grasper_runtime.model.parameter_count, "max_appendages": MAX_APPENDAGES, "max_muscles": MAX_MUSCLES},
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
