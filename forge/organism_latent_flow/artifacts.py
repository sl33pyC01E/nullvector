from __future__ import annotations

from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Final

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import Tensor
import torch.nn.functional as F

from ..organism_raster_vae_v2.smoke import _load_checkpoint as load_vae_checkpoint
from ..organism_raster_vae_v2.smoke import CHECKPOINT_NAME as VAE_CHECKPOINT_NAME
from ..safety import require_disk_floor
from .contract import FORMAT, VAE_OUTPUT, canonical_json_bytes, sha256_file, source_manifest, source_sha256
from .corpus import load_latent_corpus
from .model import HierarchicalOrganismFlow, integrate_flow
from .training import CORPUS_NAME, TRAINING_NAME, load_final_checkpoint


MANIFEST_NAME: Final[str] = "organism_flow_manifest.json"
NOVEL_NAME: Final[str] = "novel_organisms.png"
ORGANS_NAME: Final[str] = "novel_organ_systems.png"
MUTATION_NAME: Final[str] = "generated_mutation_clouds.png"
GENERATION_SEED: Final[int] = 0x4F52474E4F56454C
FAMILY_NAMES: Final[tuple[str, ...]] = ("HUMANOID", "ANIMALIAN", "PLANTLIKE", "ANOMALY", "MACHINE")


def _atomic(path: Path, payload: bytes) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _png(image: Image.Image) -> bytes:
    buffer = BytesIO(); image.save(buffer, "PNG", optimize=False, compress_level=9); return buffer.getvalue()


def _rgba_image(rgba: Tensor, scale: int = 2) -> Image.Image:
    value = rgba.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    value[..., :3] *= value[..., 3:4]
    return Image.fromarray(np.rint(value * 255).astype(np.uint8), "RGBA").resize((48 * scale, 48 * scale), Image.Resampling.NEAREST)


def _system_image(physiology: Tensor, role: Tensor, alpha: Tensor, scale: int = 2) -> Image.Image:
    palette = torch.tensor(((250, 73, 86), (90, 210, 255), (255, 178, 71), (206, 91, 255), (255, 238, 110), (91, 255, 161), (255, 104, 190), (108, 238, 205)), dtype=torch.float32) / 255
    importance = torch.tensor((0.0, 4.0, 1.0, 2.2))[role.detach().cpu().long().clamp(0, 3)]
    score = physiology.detach().cpu() * (.15 + importance)
    strength, system = score.max(0); strength = (strength / 4.15).clamp(0, 1)
    rgb = palette[system].permute(2, 0, 1) * (.25 + .75 * strength[None])
    return _rgba_image(torch.cat((rgb, alpha.detach().cpu().clamp(0, 1)[None])), scale)


def _components(mask: np.ndarray) -> int:
    remaining = set(map(tuple, np.argwhere(mask)))
    count = 0
    while remaining:
        count += 1; stack = [remaining.pop()]
        while stack:
            y, x = stack.pop()
            for neighbor in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor); stack.append(neighbor)
    return count


def _conditions(tensors: dict[str, Tensor]) -> tuple[Tensor, Tensor, list[dict[str, Any]]]:
    conditions: list[Tensor] = []; families: list[int] = []; records: list[dict[str, Any]] = []
    for family in range(5):
        indices = torch.nonzero(tensors["family"] == family).flatten().tolist()
        for column in range(6):
            left = indices[column % len(indices)]; right = indices[(column * 2 + 1) % len(indices)]
            # Columns 3-5 deliberately traverse one parent pair under shared
            # noise; columns 0-2 probe stochastic diversity.
            alpha = (column + 1) / 7 if column < 3 else (column - 2) / 4
            if column >= 3:
                left, right = indices[0], indices[1]
            condition = tensors["condition"][left] * (1 - alpha) + tensors["condition"][right] * alpha
            conditions.append(condition); families.append(family)
            records.append({"family": family, "column": column, "left_index": left, "right_index": right, "alpha": alpha, "shared_noise_traversal": column >= 3})
    return torch.stack(conditions), torch.tensor(families, dtype=torch.long), records


def _noise() -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(GENERATION_SEED)
    coarse = torch.randn((30, 32, 12, 12), generator=generator)
    fine = torch.randn((30, 16, 24, 24), generator=generator)
    # The last three columns in each family use identical noise so their
    # differences are caused by continuous conditioning, not resampling.
    for family in range(5):
        base = family * 6 + 3
        coarse[base + 1:base + 3] = coarse[base:base + 1]
        fine[base + 1:base + 3] = fine[base:base + 1]
    return coarse, fine


def _generate(model: HierarchicalOrganismFlow, output_root: Path) -> dict[str, Any]:
    corpus = load_latent_corpus(output_root / CORPUS_NAME); tensors = corpus["tensors"]
    vae, _ = load_vae_checkpoint(VAE_OUTPUT / VAE_CHECKPOINT_NAME); vae.eval(); model.eval()
    condition, family, records = _conditions(tensors); coarse_noise, fine_noise = _noise()
    with torch.inference_mode():
        normalized_coarse, normalized_fine = integrate_flow(model, coarse_noise, fine_noise, condition, steps=32, guidance=1.6)
        coarse = normalized_coarse * tensors["coarse_scale"] + tensors["coarse_center"]
        fine = normalized_fine * tensors["fine_scale"] + tensors["fine_center"]
        decoded = vae.decode(coarse, fine, condition)
        reference = vae.decode(tensors["coarse_mean"], tensors["fine_mean"], tensors["condition"])
    rgba = decoded.rgba.detach().cpu(); alpha = decoded.occupancy_logits[:, 0].sigmoid().detach().cpu()
    roles = decoded.system_role_logits.argmax(2).detach().cpu(); physiology = decoded.physiology.detach().cpu()
    reference_rgba = reference.rgba.detach().cpu()
    occupancy = alpha >= .5; ratios = occupancy.float().mean((1, 2))
    components = [_components(value.numpy()) for value in occupancy]
    nearest = torch.cdist(rgba.flatten(1), reference_rgba.flatten(1), p=1).min(1).values / rgba[0].numel()
    diversity: list[float] = []
    for family_index in range(5):
        values = rgba[family_index * 6:(family_index + 1) * 6].flatten(1)
        distances = torch.pdist(values, p=1) / values.shape[1]
        diversity.append(float(distances.mean()))
    core_present = ((roles == 1) & occupancy[:, None]).flatten(1).any(1)
    metrics = {
        "generated_count": 30,
        "finite_fraction": float(torch.isfinite(rgba).flatten(1).all(1).float().mean()),
        "occupancy_ratio_min": float(ratios.min()), "occupancy_ratio_max": float(ratios.max()), "occupancy_ratio_mean": float(ratios.mean()),
        "single_component_fraction": sum(value == 1 for value in components) / len(components),
        "maximum_component_count": max(components), "mean_nearest_training_rgba_l1": float(nearest.mean()),
        "minimum_nearest_training_rgba_l1": float(nearest.min()), "mean_within_family_pairwise_rgba_l1": float(np.mean(diversity)),
        "minimum_family_pairwise_rgba_l1": min(diversity), "system_core_sample_fraction": float(core_present.float().mean()),
    }
    for index, record in enumerate(records):
        record.update({"occupancy_ratio": float(ratios[index]), "components": components[index], "nearest_training_rgba_l1": float(nearest[index]), "system_core_present": bool(core_present[index])})
    cell = 96; gap = 8; header = 44; row_height = cell + 26
    novel = Image.new("RGB", (18 + 6 * (cell + gap), header + 5 * row_height), (4, 8, 16)); draw = ImageDraw.Draw(novel)
    draw.text((10, 8), "NOVEL ORGANISM RECTIFIED FLOW // STOCHASTIC + CONDITION BREEDING", fill=(166, 250, 219)); draw.text((10, 23), "columns 1-3 stochastic / columns 4-6 shared-noise continuous parent traversal", fill=(76, 153, 132))
    organs = Image.new("RGB", novel.size, (4, 8, 16)); organ_draw = ImageDraw.Draw(organs)
    organ_draw.text((10, 8), "GENERATED INTERNAL SYSTEMS // CORE / CONDUIT / EFFECTOR", fill=(255, 176, 220)); organ_draw.text((10, 23), "eight decoded physiological networks; alpha remains generated", fill=(151, 91, 132))
    for family_index in range(5):
        y = header + family_index * row_height
        draw.text((4, y + cell + 3), FAMILY_NAMES[family_index], fill=(101, 178, 194)); organ_draw.text((4, y + cell + 3), FAMILY_NAMES[family_index], fill=(101, 178, 194))
        for column in range(6):
            index = family_index * 6 + column; x = 10 + column * (cell + gap)
            novel.paste(_rgba_image(rgba[index]).convert("RGB"), (x, y)); organs.paste(_system_image(physiology[index], roles[index], alpha[index]).convert("RGB"), (x, y))
    mutation = Image.new("RGB", (18 + 6 * (cell + gap), header + 5 * row_height), (4, 8, 16)); mutation_draw = ImageDraw.Draw(mutation)
    mutation_draw.text((10, 8), "GENERATED MUTATION CLOUDS // COARSE CHASSIS + FINE CELLULAR EDITS", fill=(255, 165, 237)); mutation_draw.text((10, 23), "each row begins with a flow sample; edits remain in the learned VAE field", fill=(155, 87, 145))
    mutation_records: list[dict[str, Any]] = []
    generator = torch.Generator().manual_seed(GENERATION_SEED ^ 0x4D555441)
    with torch.inference_mode():
        for family_index in range(5):
            index = family_index * 6; base_coarse = coarse[index:index + 1]; base_fine = fine[index:index + 1]; base_condition = condition[index:index + 1]
            coarse_delta = F.avg_pool2d(torch.randn(base_coarse.shape, generator=generator), 3, 1, 1) * tensors["coarse_scale"] * 1.6
            fine_delta = F.avg_pool2d(torch.randn(base_fine.shape, generator=generator), 3, 1, 1) * tensors["fine_scale"] * 1.35
            variants = ((base_coarse, base_fine), (base_coarse - coarse_delta, base_fine), (base_coarse + coarse_delta, base_fine), (base_coarse, base_fine - fine_delta), (base_coarse, base_fine + fine_delta), (base_coarse + coarse_delta * .8, base_fine - fine_delta * .8))
            frames: list[Tensor] = []
            y = header + family_index * row_height
            for column, (coarse_value, fine_value) in enumerate(variants):
                value = vae.decode(coarse_value, fine_value, base_condition).rgba[0].detach().cpu(); frames.append(value)
                mutation.paste(_rgba_image(value).convert("RGB"), (10 + column * (cell + gap), y))
            mutation_draw.text((4, y + cell + 3), FAMILY_NAMES[family_index], fill=(101, 178, 194))
            mutation_records.append({"family": family_index, "maximum_rgba_l1_from_base": max(float((frame - frames[0]).abs().mean()) for frame in frames[1:])})
    return {"novel": _png(novel), "organs": _png(organs), "mutation": _png(mutation), "metrics": metrics, "records": records, "mutation_records": mutation_records, "corpus_semantic_sha256": corpus["semantic"]["semantic_sha256"]}


def _artifact(path: Path) -> dict[str, Any]:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def finalize_output(output: Path) -> dict[str, Any]:
    output = Path(output).resolve(); manifest_path = output / MANIFEST_NAME
    if manifest_path.exists():
        raise FileExistsError("Finalized organism flow output is immutable.")
    model, checkpoint, contract = load_final_checkpoint(output)
    generated = _generate(model, output)
    require_disk_floor(output, floor_gb=100, planned_bytes=512 * 1024**2)
    _atomic(output / NOVEL_NAME, generated["novel"]); _atomic(output / ORGANS_NAME, generated["organs"]); _atomic(output / MUTATION_NAME, generated["mutation"])
    final_checkpoint = output / f"segment_{contract['total_steps']:07d}.pt"
    artifacts = {
        "checkpoint": _artifact(final_checkpoint), "latent_corpus": _artifact(output / CORPUS_NAME),
        "novel_organisms": _artifact(output / NOVEL_NAME), "novel_organs": _artifact(output / ORGANS_NAME), "mutation_clouds": _artifact(output / MUTATION_NAME),
    }
    hard_gates = {
        "source_and_vae_bound": generated["corpus_semantic_sha256"] == contract["corpus_semantic_sha256"],
        "training_complete": checkpoint["step"] == contract["total_steps"], "finite_generation": generated["metrics"]["finite_fraction"] == 1,
        "generation_census_30": generated["metrics"]["generated_count"] == 30,
    }
    quality = {
        "occupancy_plausible_fraction_gate": generated["metrics"]["occupancy_ratio_min"] >= .005 and generated["metrics"]["occupancy_ratio_max"] <= .75,
        "connected_fraction_gate": generated["metrics"]["single_component_fraction"] >= .5,
        "novelty_gate": generated["metrics"]["minimum_nearest_training_rgba_l1"] > 5e-4,
        "diversity_gate": generated["metrics"]["minimum_family_pairwise_rgba_l1"] > 1e-3,
        "organ_core_gate": generated["metrics"]["system_core_sample_fraction"] >= .7,
    }
    manifest: dict[str, Any] = {
        "format": FORMAT, "status": "passed" if all(hard_gates.values()) else "failed",
        "source_sha256": source_sha256(), "source_manifest": source_manifest(), "training_contract": contract,
        "corpus_semantic_sha256": generated["corpus_semantic_sha256"], "checkpoint_step": checkpoint["step"],
        "checkpoint_ema_sha256": checkpoint["ema_state_sha256"], "loss_start": checkpoint["history"][0]["loss"], "loss_end": checkpoint["history"][-1]["loss"],
        "metrics": generated["metrics"], "generation_records": generated["records"], "mutation_records": generated["mutation_records"],
        "artifacts": artifacts, "hard_gates": hard_gates, "quality_gates": quality,
        "claim_boundary": {"conditional_latent_prior_trained": True, "free_stochastic_generation": True, "unseen_generalization_proven": False, "runtime_integration_allowed": False, "production_promotion_allowed": False},
    }
    if manifest["status"] != "passed":
        raise ValueError("Organism flow hard gates failed.")
    manifest["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest(); _atomic(manifest_path, canonical_json_bytes(manifest))
    return validate_output(output)


def validate_output(output: Path) -> dict[str, Any]:
    output = Path(output).resolve(); path = output / MANIFEST_NAME
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= 8 * 1024**2:
        raise ValueError("Organism flow manifest missing or oversized.")
    encoded = path.read_bytes(); manifest = json.loads(encoded)
    if encoded != canonical_json_bytes(manifest):
        raise ValueError("Organism flow manifest is not canonical JSON.")
    stored = manifest.pop("manifest_sha256", None)
    if stored != hashlib.sha256(canonical_json_bytes(manifest)).hexdigest():
        raise ValueError("Organism flow manifest hash failed.")
    manifest["manifest_sha256"] = stored
    required = {"format", "status", "source_sha256", "source_manifest", "training_contract", "corpus_semantic_sha256", "checkpoint_step", "checkpoint_ema_sha256", "loss_start", "loss_end", "metrics", "generation_records", "mutation_records", "artifacts", "hard_gates", "quality_gates", "claim_boundary", "manifest_sha256"}
    if set(manifest) != required or manifest["format"] != FORMAT or manifest["status"] != "passed" or manifest["source_sha256"] != source_sha256() or manifest["source_manifest"] != source_manifest():
        raise ValueError("Organism flow manifest provenance drifted.")
    for record in manifest["artifacts"].values():
        if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"} or Path(record["path"]).name != record["path"]:
            raise ValueError("Organism flow artifact descriptor drifted.")
        artifact = output / record["path"]
        if not artifact.is_file() or artifact.is_symlink() or artifact.stat().st_size != record["bytes"] or sha256_file(artifact) != record["sha256"]:
            raise ValueError("Organism flow artifact identity failed.")
    model, checkpoint, contract = load_final_checkpoint(output)
    if contract != manifest["training_contract"] or checkpoint["step"] != manifest["checkpoint_step"] or checkpoint["ema_state_sha256"] != manifest["checkpoint_ema_sha256"]:
        raise ValueError("Organism flow checkpoint semantics drifted.")
    generated = _generate(model, output)
    if generated["metrics"] != manifest["metrics"] or generated["records"] != manifest["generation_records"] or generated["mutation_records"] != manifest["mutation_records"]:
        raise ValueError("Organism flow semantic replay failed.")
    for key, payload in (("novel_organisms", generated["novel"]), ("novel_organs", generated["organs"]), ("mutation_clouds", generated["mutation"])):
        if hashlib.sha256(payload).hexdigest() != manifest["artifacts"][key]["sha256"]:
            raise ValueError("Organism flow visual replay failed.")
    if not all(manifest["hard_gates"].values()) or manifest["claim_boundary"] != {"conditional_latent_prior_trained": True, "free_stochastic_generation": True, "unseen_generalization_proven": False, "runtime_integration_allowed": False, "production_promotion_allowed": False}:
        raise ValueError("Organism flow gates/claim boundary drifted.")
    return manifest
