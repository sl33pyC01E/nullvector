from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import uuid

import numpy as np
import torch
from torch import Tensor

from ..creature_stage_developmental.contract import AppendageGene, DevelopmentalGenome
from ..creature_stage_developmental.development import develop
from ..creature_stage_developmental.genomes import review_genomes
from ..creature_stage_grounded_locomotion.physics import simulate_grounded_cycle
from ..creature_stage_grounded_locomotion.review import _array_payload
from ..creature_stage_neural_grounded.dataset import EXPECTED_ARRAYS, GroundedMotionTeacher, _array_digest
from ..creature_stage_neural_motion.contract import CONTROL_FEATURES
from ..multifield_style_motion.hashing import deterministic_npz_bytes
from ..safety import require_disk_floor
from .contract import BODY_SPEED_SCALE, DYNAMIC_FEATURES, MAX_CELLS, POSITION_SCALE, canonical_json_bytes, sha256_file, source_sha256


CURRICULUM_VARIANTS_PER_FAMILY = 12
LOCOMOTOR_KINDS = ("leg", "root", "wheel")
CURRICULUM_FORMAT = "nullvector-grounded-locomotor-curriculum-v1"
DEFAULT_CURRICULUM = Path(__file__).resolve().parents[2] / "outputs/creature_stage_neural_grounded_cyclic/curriculum_v1_sealed"


def _scaled_genome(base: DevelopmentalGenome, family: int, ordinal: int) -> DevelopmentalGenome:
    rng = np.random.Generator(np.random.PCG64(0x4355525249430000 ^ family * 0x9E3779B9 ^ ordinal))
    scale_x = float(rng.uniform(.84, 1.16))
    scale_y = float(rng.uniform(.86, 1.14))
    traits = tuple(float(np.clip(value + rng.uniform(-.065, .065), 0, 1)) for value in base.traits)
    components = tuple(
        replace(
            component,
            anchor=(component.anchor[0] * scale_x, component.anchor[1] * scale_y),
            radius=(
                max(.6, component.radius[0] * scale_x * float(rng.uniform(.96, 1.04))),
                max(.6, component.radius[1] * scale_y * float(rng.uniform(.96, 1.04))),
            ),
        )
        for component in base.components
    )
    phase_shift = float((ordinal * .137) % 1.0)
    appendages = tuple(
        replace(
            appendage,
            root_offset=(appendage.root_offset[0] * scale_x, appendage.root_offset[1] * scale_y),
            endpoint=(appendage.endpoint[0] * scale_x, appendage.endpoint[1] * scale_y),
            phase=(appendage.phase + phase_shift) % 1.0,
        )
        for appendage in base.appendages
    )
    parent_ids: tuple[str, ...] = ()
    family_mix = list(base.family_mix)
    # World generation keeps grafts rare.  The curriculum deliberately
    # oversamples them so every transferable locomotor receives gradients.
    if ordinal % 3:
        kind = LOCOMOTOR_KINDS[(family + ordinal) % len(LOCOMOTOR_KINDS)]
        root = next((item.component_id for item in components if item.kind == "pelvis"), components[0].component_id)
        width = max(item.radius[0] for item in components)
        bottom = max(item.anchor[1] + item.radius[1] for item in components) + float(rng.uniform(5.5, 8.5))
        prefix = f"curr_{kind}_{ordinal:02d}"
        left, right = f"{prefix}_l", f"{prefix}_r"
        segments = 2 if kind == "wheel" else 3
        graft = (
            AppendageGene(left, kind, root, (-width * .55, 1.0), (-width * .82, bottom), segments, -1, phase_shift, -1, right),
            AppendageGene(right, kind, root, (width * .55, 1.0), (width * .82, bottom), segments, 1, (phase_shift + .5) % 1.0, 1, left),
        )
        appendages += graft
        donor_family = {"leg": 1, "root": 2, "wheel": 4}[kind]
        if donor_family != family:
            family_mix[family] = .86
            family_mix[donor_family] += .14
        parent_ids = (base.genome_id, f"locomotor_{kind}")
    return replace(
        base,
        genome_id=f"curriculum_f{family}_v{ordinal:02d}",
        seed=0xC70000 + family * 0x100 + ordinal,
        family_mix=tuple(family_mix),
        traits=traits,
        components=components,
        appendages=appendages,
        generation=1,
        parent_ids=parent_ids,
    )


def _curriculum_data(variants_per_family: int) -> tuple[tuple[DevelopmentalGenome, ...], tuple[Any, ...], tuple[Any, ...]]:
    if not 2 <= variants_per_family <= 32:
        raise ValueError("grounded curriculum size drifted")
    bases = review_genomes()[::2]
    accepted: list[DevelopmentalGenome] = []
    accepted_organisms: list[Any] = []
    accepted_cycles: list[Any] = []
    for family, base in enumerate(bases):
        family_rows: list[DevelopmentalGenome] = []
        for ordinal in range(variants_per_family * 4):
            candidate = _scaled_genome(base, family, ordinal)
            organism = develop(candidate)
            if organism.cell_count > MAX_CELLS:
                continue
            cycle = simulate_grounded_cycle(organism)
            if (
                cycle.distance_px <= .20 or cycle.maximum_contact_slip_px >= .05
                or cycle.maximum_edge_strain >= .12 or cycle.loop_seam_max_abs >= .002
                or cycle.vertical_axis_max_degrees >= 5
            ):
                continue
            family_rows.append(candidate)
            accepted_organisms.append(organism)
            accepted_cycles.append(cycle)
            if len(family_rows) == variants_per_family:
                break
        if len(family_rows) != variants_per_family:
            raise RuntimeError(f"grounded curriculum family {family} could not fill its quota")
        accepted.extend(family_rows)
    return tuple(accepted), tuple(accepted_organisms), tuple(accepted_cycles)


def curriculum_genomes(variants_per_family: int = CURRICULUM_VARIANTS_PER_FAMILY) -> tuple[DevelopmentalGenome, ...]:
    return _curriculum_data(variants_per_family)[0]


def _pad_cells(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    cell_axis = {
        "cells_local": 2, "cell_mask": 1, "rest_cells": 1, "tissue": 1,
        "appendage_owner": 1, "trait_fields": 1, "component_weights": 1,
    }
    result: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        axis = cell_axis.get(name)
        if axis is None:
            result[name] = np.ascontiguousarray(value)
            continue
        if value.shape[axis] > MAX_CELLS:
            raise ValueError("grounded curriculum exceeded the cell canvas")
        shape = list(value.shape); shape[axis] = MAX_CELLS
        fill = 255 if name == "tissue" else -1 if name == "appendage_owner" else 0
        padded = np.full(shape, fill, dtype=value.dtype)
        slices = [slice(None)] * value.ndim; slices[axis] = slice(0, value.shape[axis])
        padded[tuple(slices)] = value
        result[name] = np.ascontiguousarray(padded)
    return result


def build_curriculum(output: Path = DEFAULT_CURRICULUM,
                     variants_per_family: int = CURRICULUM_VARIANTS_PER_FAMILY) -> dict[str, Any]:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=2 * 1024**3)
    genomes, organisms, cycles = _curriculum_data(variants_per_family)
    arrays = _pad_cells(_array_payload(organisms, cycles))
    archive = deterministic_npz_bytes(arrays)
    stage = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    stage.mkdir(parents=True)
    archive_path = stage / "grounded_curriculum_arrays.npz"
    archive_path.write_bytes(archive)
    records = []
    for genome, organism, cycle in zip(genomes, organisms, cycles, strict=True):
        records.append({
            "genome_id": genome.genome_id,
            "family": int(np.argmax(genome.family_mix)),
            "grafted": bool(genome.parent_ids),
            "parent_ids": list(genome.parent_ids),
            "cell_count": organism.cell_count,
            "organism_sha256": organism.identity_sha256,
            "cycle_sha256": cycle.identity_sha256,
            "primary_mode": cycle.primary_mode,
            "locomotor_modes": list(cycle.modes),
            "distance_px": round(cycle.distance_px, 9),
            "maximum_contact_slip_px": round(cycle.maximum_contact_slip_px, 9),
            "maximum_edge_strain": round(cycle.maximum_edge_strain, 9),
            "loop_seam_max_abs": round(cycle.loop_seam_max_abs, 9),
            "vertical_axis_max_degrees": round(cycle.vertical_axis_max_degrees, 9),
        })
    manifest = {
        "format": CURRICULUM_FORMAT,
        "source_sha256": source_sha256(),
        "variants_per_family": variants_per_family,
        "organism_count": len(organisms),
        "family_counts": [sum(record["family"] == family for record in records) for family in range(5)],
        "grafted_count": sum(record["grafted"] for record in records),
        "world_graft_prior": "rare; curriculum intentionally oversamples transfer cases",
        "records": records,
        "arrays": {
            "path": archive_path.name, "bytes": len(archive),
            "sha256": hashlib.sha256(archive).hexdigest(), "semantic_sha256": _array_digest(arrays),
        },
        "gates": {
            "all_five_families_balanced": all(value == variants_per_family for value in [sum(record["family"] == family for record in records) for family in range(5)]),
            "all_cells_fit_canvas": max(record["cell_count"] for record in records) <= MAX_CELLS,
            "all_cycles_valid": all(
                record["distance_px"] > .20 and record["maximum_contact_slip_px"] < .05
                and record["maximum_edge_strain"] < .12 and record["loop_seam_max_abs"] < .002
                and record["vertical_axis_max_degrees"] < 5 for record in records
            ),
            "all_locomotor_modes_covered": {mode for record in records for mode in record["locomotor_modes"]} >= {"passive", "step", "drag", "wheel"},
            "cross_family_grafts_present": sum(record["grafted"] for record in records) >= variants_per_family * 2,
        },
    }
    manifest["semantic_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    (stage / "curriculum_manifest.json").write_bytes(canonical_json_bytes(manifest))
    os.replace(stage, output)
    return validate_curriculum(output)


def validate_curriculum(root: Path = DEFAULT_CURRICULUM) -> dict[str, Any]:
    root = Path(root).resolve(); manifest_path = root / "curriculum_manifest.json"
    raw = manifest_path.read_bytes(); manifest = json.loads(raw)
    if raw != canonical_json_bytes(manifest):
        raise ValueError("grounded curriculum manifest is not canonical")
    semantic = manifest.pop("semantic_sha256")
    if semantic != hashlib.sha256(canonical_json_bytes(manifest)).hexdigest():
        raise ValueError("grounded curriculum semantic hash drifted")
    manifest["semantic_sha256"] = semantic
    if manifest["format"] != CURRICULUM_FORMAT or manifest["source_sha256"] != source_sha256() or not all(manifest["gates"].values()):
        raise ValueError("grounded curriculum contract drifted")
    archive_path = root / manifest["arrays"]["path"]
    if archive_path.stat().st_size != manifest["arrays"]["bytes"] or sha256_file(archive_path) != manifest["arrays"]["sha256"]:
        raise ValueError("grounded curriculum archive drifted")
    with np.load(archive_path, allow_pickle=False) as archive:
        if set(archive.files) != EXPECTED_ARRAYS:
            raise ValueError("grounded curriculum member census drifted")
        arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    if _array_digest(arrays) != manifest["arrays"]["semantic_sha256"]:
        raise ValueError("grounded curriculum array identity drifted")
    if arrays["cells_local"].shape != (manifest["organism_count"], 72, MAX_CELLS, 2):
        raise ValueError("grounded curriculum tensor shape drifted")
    return manifest


class CurriculumGroundedTeacher(GroundedMotionTeacher):
    """Deterministic, family-balanced physics curriculum; sentinels excluded."""

    def __init__(self, root: Path | None = DEFAULT_CURRICULUM,
                 variants_per_family: int = CURRICULUM_VARIANTS_PER_FAMILY) -> None:
        self.root = None
        if root is None:
            genomes, self.organisms, cycles = _curriculum_data(variants_per_family)
            self.arrays = _pad_cells(_array_payload(self.organisms, cycles))
            stored_semantic = None
        else:
            root = Path(root).resolve(); manifest = validate_curriculum(root)
            variants_per_family = int(manifest["variants_per_family"])
            genomes = tuple(_scaled_genome(review_genomes()[::2][family], family, ordinal) for family in range(5) for ordinal in range(variants_per_family))
            # Candidate filtering can skip ordinals, so bind the exact accepted
            # genome IDs and regenerate only those lightweight anatomies.
            accepted = {record["genome_id"]: record for record in manifest["records"]}
            if set(genome.genome_id for genome in genomes) != set(accepted):
                all_candidates = (_scaled_genome(review_genomes()[::2][family], family, ordinal) for family in range(5) for ordinal in range(variants_per_family * 4))
                genomes = tuple(genome for genome in all_candidates if genome.genome_id in accepted)
            genomes = tuple(sorted(genomes, key=lambda genome: (int(genome.genome_id[12]), genome.genome_id)))
            self.organisms = tuple(develop(genome) for genome in genomes)
            with np.load(root / manifest["arrays"]["path"], allow_pickle=False) as archive:
                self.arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
            stored_semantic = manifest["semantic_sha256"]
        self.manifest = {"cycles": [{"genome_id": item.genome.genome_id} for item in self.organisms]}
        self.static = tuple(self._static(index) for index in range(len(self.organisms)))
        if stored_semantic is None:
            digest = hashlib.sha256(b"nullvector-grounded-curriculum-v1\0")
            digest.update(str(variants_per_family).encode("ascii") + b"\0")
            for genome, organism, cycle in zip(genomes, self.organisms, cycles, strict=True):
                digest.update(genome.genome_id.encode("ascii") + b"\0")
                digest.update(organism.identity_sha256.encode("ascii") + cycle.identity_sha256.encode("ascii"))
            digest.update(_array_digest(self.arrays).encode("ascii"))
            self.semantic_sha256 = digest.hexdigest()
        else:
            expected = {record["genome_id"]: record["organism_sha256"] for record in manifest["records"]}
            if any(expected[item.genome.genome_id] != item.identity_sha256 for item in self.organisms):
                raise ValueError("grounded curriculum anatomy replay drifted")
            self.semantic_sha256 = stored_semantic
        self.family_indices = tuple(
            tuple(int(index) for index in np.flatnonzero(self.arrays["family"] == family))
            for family in range(5)
        )
        if any(len(rows) != variants_per_family for rows in self.family_indices):
            raise ValueError("grounded curriculum family balance drifted")

    def split_indices(self, split: str) -> tuple[int, ...]:
        if split in {"train", "all"}:
            return tuple(range(len(self.organisms)))
        raise ValueError("grounded curriculum has no internal evaluation split")

    def sample(self, identity: int, frame: int) -> dict[str, Any]:
        if not 0 <= identity < len(self.organisms) or not 0 <= frame < 72:
            raise ValueError("grounded curriculum coordinate drifted")
        previous = (frame - 1) % 72
        before = (frame - 2) % 72
        count = int(self.arrays["cell_mask"][identity].sum())
        rest = self.arrays["rest_cells"][identity]
        current_cells = self.arrays["cells_local"][identity, previous]
        before_cells = self.arrays["cells_local"][identity, before]
        target_cells = self.arrays["cells_local"][identity, frame]
        state = np.zeros((MAX_CELLS, 4), np.float32)
        target = np.zeros_like(state)
        state[:count, :2] = (current_cells[:count] - rest[:count]) / POSITION_SCALE
        state[:count, 2:] = (current_cells[:count] - before_cells[:count]) / POSITION_SCALE
        target[:count, :2] = (target_cells[:count] - rest[:count]) / POSITION_SCALE
        target[:count, 2:] = (target_cells[:count] - current_cells[:count]) / POSITION_SCALE
        dynamic = np.zeros((MAX_CELLS, DYNAMIC_FEATURES), np.float32)
        owners = self.arrays["appendage_owner"][identity, :count]
        modes = self.arrays["locomotor_mode"][identity]
        active = self.arrays["contact_active"][identity, frame].astype(bool)
        for cell_index, owner in enumerate(owners):
            mode = int(modes[owner]) if owner >= 0 else 0
            dynamic[cell_index, mode] = 1
            if owner >= 0 and active[owner]:
                dynamic[cell_index, 5] = 1
                dynamic[cell_index, 6:8] = np.clip(
                    (self.arrays["contact_anchor_local"][identity, frame, owner] - current_cells[cell_index]) / POSITION_SCALE,
                    -1, 1,
                )
                dynamic[cell_index, 8:10] = np.clip(self.arrays["contact_force"][identity, frame, owner], -1, 1)
        dynamic[:count, 10] = self.arrays["body_velocity_x"][identity, previous] / BODY_SPEED_SCALE
        dynamic[:count, 11] = np.clip((self.arrays["body_world_x"][identity, previous] - self.arrays["body_world_x"][identity, 0]) / 16, -1, 1)
        dynamic[:count, 12] = np.clip((self.arrays["ground_y"][identity] - current_cells[:count, 1]) / 32, -1, 1)
        dynamic[:count, 13] = float(self.arrays["grafted"][identity])
        phase = frame / 72
        dynamic[:count, 14] = np.sin(np.pi * 2 * phase); dynamic[:count, 15] = np.cos(np.pi * 2 * phase)
        controls = np.zeros(CONTROL_FEATURES, np.float32); controls[:4] = (1, 0, 1, 0)
        static, mask, adjacency = self.static[identity]
        for value in (state, target, dynamic, controls): value.setflags(write=False)
        family = int(self.arrays["family"][identity])
        return {
            "static": static, "state": state, "target": target, "dynamic": dynamic,
            "mask": mask, "adjacency": adjacency, "controls": controls,
            "family": family, "morphotype": family * 4, "motion": 2,
            "phase": phase, "identity": identity, "frame": frame,
            "body_target": float(self.arrays["body_velocity_x"][identity, frame] / BODY_SPEED_SCALE),
            "body_previous": float(self.arrays["body_velocity_x"][identity, previous] / BODY_SPEED_SCALE),
            "cell_count": count,
        }

    def batch(self, step: int, batch_size: int, device: torch.device, *, split: str = "train", frame_offset: int = 0) -> dict[str, Tensor]:
        if split != "train" or batch_size < 5 or batch_size % 5:
            raise ValueError("grounded curriculum batch contract drifted")
        rows: list[dict[str, Any]] = []
        for slot in range(batch_size):
            family = slot % 5
            token = self._mix64(0x435552524943554C ^ step * 0xD1342543DE82EF95 ^ slot * 0xA24BAED4963EE407)
            identities = self.family_indices[family]
            identity = identities[int((token // 72) % len(identities))]
            frame = int((token % 72 + frame_offset) % 72)
            rows.append(self.sample(identity, frame))
        result: dict[str, Tensor] = {}
        for name in ("static", "state", "target", "dynamic", "mask", "adjacency", "controls"):
            result[name] = torch.from_numpy(np.stack([row[name] for row in rows]).copy()).to(device)
        for name in ("family", "morphotype", "motion", "identity", "frame"):
            result[name] = torch.tensor([int(row[name]) for row in rows], dtype=torch.long, device=device)
        for name in ("phase", "body_target", "body_previous"):
            result[name] = torch.tensor([float(row[name]) for row in rows], dtype=torch.float32, device=device)
        return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Build or validate the grounded locomotor curriculum")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build"); build.add_argument("--output", type=Path, default=DEFAULT_CURRICULUM)
    check = commands.add_parser("validate"); check.add_argument("--root", type=Path, default=DEFAULT_CURRICULUM)
    args = parser.parse_args()
    print(build_curriculum(args.output) if args.command == "build" else validate_curriculum(args.root))


if __name__ == "__main__":
    main()
