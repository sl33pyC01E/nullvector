from __future__ import annotations

from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Final

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import Tensor

from ..organism_latent_flow.artifacts import FAMILY_NAMES, GENERATION_SEED, _conditions, _noise, _rgba_image, _system_image
from ..organism_latent_flow.contract import VAE_OUTPUT
from ..organism_latent_flow.corpus import load_latent_corpus
from ..organism_latent_flow.model import integrate_flow
from ..organism_latent_flow.training import CORPUS_NAME as FLOW_CORPUS_NAME, load_final_checkpoint as load_flow_checkpoint
from ..organism_raster_vae_v2.smoke import CHECKPOINT_NAME as VAE_CHECKPOINT_NAME, _load_checkpoint as load_vae_checkpoint
from ..safety import require_disk_floor
from .contract import FORMAT, FLOW_OUTPUT, canonical_json_bytes, sha256_file, source_manifest, source_sha256
from .model import refine_latents
from .training import TRAINING_NAME, load_final_checkpoint


MANIFEST_NAME: Final[str] = "organism_refiner_manifest.json"
COMPARISON_NAME: Final[str] = "raw_vs_refined.png"
ORGANS_NAME: Final[str] = "refined_organ_systems.png"
RECOVERY_NAME: Final[str] = "neural_corruption_recovery.png"


def _atomic(path: Path, payload: bytes) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent); temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle: handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally: temporary.unlink(missing_ok=True)


def _png(image: Image.Image) -> bytes:
    buffer = BytesIO(); image.save(buffer, "PNG", optimize=False, compress_level=9); return buffer.getvalue()


def _component_sizes(mask: np.ndarray) -> list[int]:
    if not isinstance(mask, np.ndarray) or mask.shape != (48, 48) or mask.dtype != np.bool_: raise ValueError("Organism refiner component mask drifted.")
    remaining = set(map(tuple, np.argwhere(mask))); sizes: list[int] = []
    while remaining:
        stack = [remaining.pop()]; size = 0
        while stack:
            y, x = stack.pop(); size += 1
            for neighbor in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if neighbor in remaining: remaining.remove(neighbor); stack.append(neighbor)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def _metrics(rgba: Tensor, alpha: Tensor, roles: Tensor, reference_rgba: Tensor) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if rgba.shape != (30, 4, 48, 48) or alpha.shape != (30, 48, 48) or roles.shape != (30, 8, 48, 48) or reference_rgba.shape != (45, 4, 48, 48): raise ValueError("Organism refiner metric bank drifted.")
    occupancy = alpha >= .5; components = [_component_sizes(value.numpy()) for value in occupancy]
    counts = [len(value) for value in components]; satellites = [0 if not value else (sum(value) - value[0]) / max(sum(value), 1) for value in components]
    nearest = torch.cdist(rgba.flatten(1), reference_rgba.flatten(1), p=1).min(1).values / rgba[0].numel(); ratios = occupancy.float().mean((1, 2)); cores = ((roles == 1) & occupancy[:, None]).flatten(1).any(1)
    family_diversity = []
    for family in range(5):
        values = rgba[family * 6:(family + 1) * 6].flatten(1); family_diversity.append(float((torch.pdist(values, p=1) / values.shape[1]).mean()))
    metrics = {"generated_count": 30, "finite_fraction": float(torch.isfinite(rgba).flatten(1).all(1).float().mean()), "occupancy_ratio_min": float(ratios.min()), "occupancy_ratio_max": float(ratios.max()), "occupancy_ratio_mean": float(ratios.mean()), "single_component_fraction": sum(value == 1 for value in counts) / len(counts), "mean_component_count": float(np.mean(counts)), "maximum_component_count": max(counts), "mean_satellite_cell_fraction": float(np.mean(satellites)), "maximum_satellite_cell_fraction": max(satellites), "mean_nearest_training_rgba_l1": float(nearest.mean()), "minimum_nearest_training_rgba_l1": float(nearest.min()), "mean_within_family_pairwise_rgba_l1": float(np.mean(family_diversity)), "minimum_family_pairwise_rgba_l1": min(family_diversity), "system_core_sample_fraction": float(cores.float().mean())}
    records = [{"components": counts[index], "satellite_cell_fraction": satellites[index], "occupancy_ratio": float(ratios[index]), "nearest_training_rgba_l1": float(nearest[index]), "system_core_present": bool(cores[index])} for index in range(len(rgba))]
    return metrics, records


def _topology_summary(alpha: Tensor, family: Tensor) -> dict[str, Any]:
    occupancy = alpha >= .5; rows = []
    for index, value in enumerate(occupancy):
        sizes = _component_sizes(value.numpy()); rows.append({"family": int(family[index]), "components": len(sizes), "satellite_cell_fraction": 0 if not sizes else (sum(sizes) - sizes[0]) / max(sum(sizes), 1)})
    per_family = []
    for family_index in range(5):
        selected = [row for row in rows if row["family"] == family_index]
        per_family.append({"family": family_index, "sample_count": len(selected), "mean_components": float(np.mean([row["components"] for row in selected])), "minimum_components": min(row["components"] for row in selected), "maximum_components": max(row["components"] for row in selected), "mean_satellite_cell_fraction": float(np.mean([row["satellite_cell_fraction"] for row in selected]))})
    return {"sample_count": len(rows), "mean_components": float(np.mean([row["components"] for row in rows])), "mean_satellite_cell_fraction": float(np.mean([row["satellite_cell_fraction"] for row in rows])), "per_family": per_family}


def _recovery(refiner: torch.nn.Module, vae: torch.nn.Module, tensors: dict[str, Tensor]) -> tuple[bytes, dict[str, float]]:
    clean_coarse = (tensors["coarse_mean"] - tensors["coarse_center"]) / tensors["coarse_scale"]; clean_fine = (tensors["fine_mean"] - tensors["fine_center"]) / tensors["fine_scale"]; generator = torch.Generator().manual_seed(GENERATION_SEED ^ 0x5245434F56455259); raw_coarse = torch.randn(clean_coarse.shape, generator=generator); raw_fine = torch.randn(clean_fine.shape, generator=generator); noise_coarse = .72 * raw_coarse + .28 * torch.nn.functional.avg_pool2d(raw_coarse, 3, 1, 1); noise_fine = .72 * raw_fine + .28 * torch.nn.functional.avg_pool2d(raw_fine, 3, 1, 1); corrupted_coarse = clean_coarse + .35 * noise_coarse; corrupted_fine = clean_fine + .35 * noise_fine
    with torch.inference_mode():
        refined_coarse, refined_fine = refine_latents(refiner, corrupted_coarse, corrupted_fine, tensors["condition"]); clean = vae.decode(tensors["coarse_mean"], tensors["fine_mean"], tensors["condition"]); corrupted = vae.decode(corrupted_coarse * tensors["coarse_scale"] + tensors["coarse_center"], corrupted_fine * tensors["fine_scale"] + tensors["fine_center"], tensors["condition"]); refined = vae.decode(refined_coarse * tensors["coarse_scale"] + tensors["coarse_center"], refined_fine * tensors["fine_scale"] + tensors["fine_center"], tensors["condition"])
    clean_alpha = clean.occupancy_logits[:, 0].sigmoid(); corrupted_alpha = corrupted.occupancy_logits[:, 0].sigmoid(); refined_alpha = refined.occupancy_logits[:, 0].sigmoid(); clean_counts = [len(_component_sizes(value.numpy())) for value in (clean_alpha >= .5)]; corrupted_counts = [len(_component_sizes(value.numpy())) for value in (corrupted_alpha >= .5)]; refined_counts = [len(_component_sizes(value.numpy())) for value in (refined_alpha >= .5)]
    metrics = {"corruption_sigma": .35, "coarse_mse_before": float(torch.nn.functional.mse_loss(corrupted_coarse, clean_coarse)), "coarse_mse_after": float(torch.nn.functional.mse_loss(refined_coarse, clean_coarse)), "fine_mse_before": float(torch.nn.functional.mse_loss(corrupted_fine, clean_fine)), "fine_mse_after": float(torch.nn.functional.mse_loss(refined_fine, clean_fine)), "rgba_mae_before": float((corrupted.rgba - clean.rgba).abs().mean()), "rgba_mae_after": float((refined.rgba - clean.rgba).abs().mean()), "alpha_mae_before": float((corrupted_alpha - clean_alpha).abs().mean()), "alpha_mae_after": float((refined_alpha - clean_alpha).abs().mean()), "physiology_mae_before": float((corrupted.physiology - clean.physiology).abs().mean()), "physiology_mae_after": float((refined.physiology - clean.physiology).abs().mean()), "component_count_error_before": float(np.mean([abs(left - right) for left, right in zip(corrupted_counts, clean_counts, strict=True)])), "component_count_error_after": float(np.mean([abs(left - right) for left, right in zip(refined_counts, clean_counts, strict=True)]))}
    indices = [index for family_index in range(5) for index in torch.nonzero(tensors["family"] == family_index).flatten().tolist()[:2]]; scale = 3; cell = 144; canvas = Image.new("RGB", (20 + 3 * (cell + 8), 48 + len(indices) * (cell + 18)), (4, 8, 16)); draw = ImageDraw.Draw(canvas); draw.text((10, 8), "NEURAL HOMEOSTASIS // CLEAN / CORRUPTED / REFINED", fill=(255, 213, 129)); draw.text((10, 23), "35% correlated latent injury; recovery is learned, never raster cleanup", fill=(157, 124, 77))
    for row, index in enumerate(indices):
        y = 46 + row * (cell + 18)
        for column, value in enumerate((clean.rgba[index], corrupted.rgba[index], refined.rgba[index])): canvas.paste(_rgba_image(value, scale=scale).convert("RGB"), (10 + column * (cell + 8), y))
        draw.text((10, y + cell + 2), f"family {int(tensors['family'][index])} / sample {index:02d}", fill=(101, 178, 194))
    return _png(canvas), metrics


def _generate(refiner: torch.nn.Module) -> dict[str, Any]:
    corpus = load_latent_corpus(FLOW_OUTPUT / FLOW_CORPUS_NAME); tensors = corpus["tensors"]; flow, _, flow_contract = load_flow_checkpoint(FLOW_OUTPUT); vae, _ = load_vae_checkpoint(VAE_OUTPUT / VAE_CHECKPOINT_NAME); flow.eval(); refiner.eval(); vae.eval(); condition, family, condition_records = _conditions(tensors); coarse_noise, fine_noise = _noise()
    with torch.inference_mode():
        raw_coarse, raw_fine = integrate_flow(flow, coarse_noise, fine_noise, condition, steps=flow_contract["integration"]["steps"], guidance=flow_contract["integration"]["guidance"])
        refined_coarse, refined_fine = refine_latents(refiner, raw_coarse, raw_fine, condition)
        raw_decoded = vae.decode(raw_coarse * tensors["coarse_scale"] + tensors["coarse_center"], raw_fine * tensors["fine_scale"] + tensors["fine_center"], condition)
        refined_decoded = vae.decode(refined_coarse * tensors["coarse_scale"] + tensors["coarse_center"], refined_fine * tensors["fine_scale"] + tensors["fine_center"], condition)
        reference = vae.decode(tensors["coarse_mean"], tensors["fine_mean"], tensors["condition"])
    raw_rgba = raw_decoded.rgba.detach().cpu(); refined_rgba = refined_decoded.rgba.detach().cpu(); raw_alpha = raw_decoded.occupancy_logits[:, 0].sigmoid().detach().cpu(); refined_alpha = refined_decoded.occupancy_logits[:, 0].sigmoid().detach().cpu(); raw_roles = raw_decoded.system_role_logits.argmax(2).detach().cpu(); refined_roles = refined_decoded.system_role_logits.argmax(2).detach().cpu(); reference_rgba = reference.rgba.detach().cpu()
    raw_metrics, raw_records = _metrics(raw_rgba, raw_alpha, raw_roles, reference_rgba); refined_metrics, refined_records = _metrics(refined_rgba, refined_alpha, refined_roles, reference_rgba); reference_topology = _topology_summary(reference.occupancy_logits[:, 0].sigmoid().detach().cpu(), tensors["family"]); recovery, recovery_metrics = _recovery(refiner, vae, tensors)
    for index, record in enumerate(condition_records): record.update({"raw": raw_records[index], "refined": refined_records[index]})
    cell = 96; gap = 8; header = 46; row_height = cell * 2 + 29; canvas = Image.new("RGB", (18 + 6 * (cell + gap), header + 5 * row_height), (4, 8, 16)); draw = ImageDraw.Draw(canvas); draw.text((10, 8), "NEURAL MANIFOLD REFINEMENT // RAW FLOW ABOVE / REFINED BELOW", fill=(165, 250, 215)); draw.text((10, 23), "no raster cleanup; the second image is decoded from a learned latent projection", fill=(75, 152, 130))
    organs = Image.new("RGB", (18 + 6 * (cell + gap), 44 + 5 * (cell + 26)), (4, 8, 16)); organ_draw = ImageDraw.Draw(organs); organ_draw.text((10, 8), "REFINED GENERATED ORGAN SYSTEMS // CORE / CONDUIT / EFFECTOR", fill=(255, 176, 220)); organ_draw.text((10, 23), "physiological networks decoded after neural manifold projection", fill=(151, 91, 132))
    physiology = refined_decoded.physiology.detach().cpu()
    for family_index in range(5):
        y = header + family_index * row_height; organ_y = 44 + family_index * (cell + 26)
        for column in range(6):
            index = family_index * 6 + column; x = 10 + column * (cell + gap); canvas.paste(_rgba_image(raw_rgba[index]).convert("RGB"), (x, y)); canvas.paste(_rgba_image(refined_rgba[index]).convert("RGB"), (x, y + cell)); organs.paste(_system_image(physiology[index], refined_roles[index], refined_alpha[index]).convert("RGB"), (x, organ_y))
        draw.text((4, y + cell * 2 + 3), FAMILY_NAMES[family_index], fill=(101, 178, 194)); organ_draw.text((4, organ_y + cell + 3), FAMILY_NAMES[family_index], fill=(101, 178, 194))
    latent_change = {"coarse_l1": float((refined_coarse - raw_coarse).abs().mean()), "fine_l1": float((refined_fine - raw_fine).abs().mean()), "rgba_l1": float((refined_rgba - raw_rgba).abs().mean())}
    return {"comparison": _png(canvas), "organs": _png(organs), "recovery": recovery, "raw_metrics": raw_metrics, "refined_metrics": refined_metrics, "reference_topology": reference_topology, "recovery_metrics": recovery_metrics, "records": condition_records, "latent_change": latent_change, "corpus_semantic_sha256": corpus["semantic"]["semantic_sha256"]}


def _artifact(path: Path) -> dict[str, Any]: return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def finalize_output(output: Path) -> dict[str, Any]:
    output = Path(output).resolve(); manifest_path = output / MANIFEST_NAME
    if manifest_path.exists(): raise FileExistsError("Finalized organism refiner output is immutable.")
    model, checkpoint, contract = load_final_checkpoint(output); generated = _generate(model); require_disk_floor(output, floor_gb=100, planned_bytes=512 * 1024**2); _atomic(output / COMPARISON_NAME, generated["comparison"]); _atomic(output / ORGANS_NAME, generated["organs"]); _atomic(output / RECOVERY_NAME, generated["recovery"]); final_checkpoint = output / f"segment_{contract['total_steps']:07d}.pt"
    artifacts = {"checkpoint": _artifact(final_checkpoint), "comparison": _artifact(output / COMPARISON_NAME), "organs": _artifact(output / ORGANS_NAME), "recovery": _artifact(output / RECOVERY_NAME)}
    raw, refined = generated["raw_metrics"], generated["refined_metrics"]
    hard_gates = {"source_and_prior_bound": generated["corpus_semantic_sha256"] == contract["corpus_semantic_sha256"], "training_complete": checkpoint["step"] == contract["total_steps"], "finite_generation": refined["finite_fraction"] == 1, "generation_census_30": refined["generated_count"] == 30}
    reference = generated["reference_topology"]; recovery = generated["recovery_metrics"]
    quality_gates = {"prior_component_distribution_matches_reference": abs(raw["mean_component_count"] - reference["mean_components"]) <= 3 and abs(raw["mean_satellite_cell_fraction"] - reference["mean_satellite_cell_fraction"]) <= .02, "valid_prior_topology_preserved": refined["mean_component_count"] == raw["mean_component_count"] and refined["mean_satellite_cell_fraction"] == raw["mean_satellite_cell_fraction"], "novelty_preserved": refined["minimum_nearest_training_rgba_l1"] > 5e-4, "diversity_preserved": refined["minimum_family_pairwise_rgba_l1"] > .025, "organ_cores_preserved": refined["system_core_sample_fraction"] >= .95, "latent_corruption_recovered": recovery["coarse_mse_after"] < recovery["coarse_mse_before"] and recovery["fine_mse_after"] < recovery["fine_mse_before"], "raster_corruption_recovered": recovery["rgba_mae_after"] < recovery["rgba_mae_before"] and recovery["alpha_mae_after"] < recovery["alpha_mae_before"] and recovery["physiology_mae_after"] < recovery["physiology_mae_before"], "topology_corruption_recovered": recovery["component_count_error_after"] <= recovery["component_count_error_before"]}
    manifest: dict[str, Any] = {"format": FORMAT, "status": "passed" if all(hard_gates.values()) else "failed", "source_sha256": source_sha256(), "source_manifest": source_manifest(), "training_contract": contract, "corpus_semantic_sha256": generated["corpus_semantic_sha256"], "checkpoint_step": checkpoint["step"], "checkpoint_ema_sha256": checkpoint["ema_state_sha256"], "loss_start": checkpoint["history"][0]["loss"], "loss_end": checkpoint["history"][-1]["loss"], "reference_topology": reference, "recovery_metrics": recovery, "raw_metrics": raw, "refined_metrics": refined, "generation_records": generated["records"], "latent_change": generated["latent_change"], "artifacts": artifacts, "hard_gates": hard_gates, "quality_gates": quality_gates, "claim_boundary": {"neural_manifold_refinement": True, "neural_homeostasis_assay": True, "deterministic_raster_cleanup": False, "raw_prior_preserved": True, "runtime_integration_allowed": False, "production_promotion_allowed": False}}
    if manifest["status"] != "passed": raise ValueError("Organism refiner hard gates failed.")
    manifest["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest(); _atomic(manifest_path, canonical_json_bytes(manifest)); return validate_output(output)


def validate_output(output: Path) -> dict[str, Any]:
    output = Path(output).resolve(); path = output / MANIFEST_NAME
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= 8 * 1024**2: raise ValueError("Organism refiner manifest missing or oversized.")
    encoded = path.read_bytes(); manifest = json.loads(encoded)
    if encoded != canonical_json_bytes(manifest): raise ValueError("Organism refiner manifest is not canonical JSON.")
    stored = manifest.pop("manifest_sha256", None)
    if stored != hashlib.sha256(canonical_json_bytes(manifest)).hexdigest(): raise ValueError("Organism refiner manifest hash failed.")
    manifest["manifest_sha256"] = stored; required = {"format", "status", "source_sha256", "source_manifest", "training_contract", "corpus_semantic_sha256", "checkpoint_step", "checkpoint_ema_sha256", "loss_start", "loss_end", "reference_topology", "recovery_metrics", "raw_metrics", "refined_metrics", "generation_records", "latent_change", "artifacts", "hard_gates", "quality_gates", "claim_boundary", "manifest_sha256"}
    if set(manifest) != required or manifest["format"] != FORMAT or manifest["status"] != "passed" or manifest["source_sha256"] != source_sha256() or manifest["source_manifest"] != source_manifest(): raise ValueError("Organism refiner manifest provenance drifted.")
    for record in manifest["artifacts"].values():
        if set(record) != {"path", "bytes", "sha256"} or Path(record["path"]).name != record["path"]: raise ValueError("Organism refiner artifact descriptor drifted.")
        artifact = output / record["path"]
        if not artifact.is_file() or artifact.is_symlink() or artifact.stat().st_size != record["bytes"] or sha256_file(artifact) != record["sha256"]: raise ValueError("Organism refiner artifact identity failed.")
    model, checkpoint, contract = load_final_checkpoint(output)
    if contract != manifest["training_contract"] or checkpoint["step"] != manifest["checkpoint_step"] or checkpoint["ema_state_sha256"] != manifest["checkpoint_ema_sha256"]: raise ValueError("Organism refiner checkpoint semantics drifted.")
    generated = _generate(model)
    if generated["reference_topology"] != manifest["reference_topology"] or generated["recovery_metrics"] != manifest["recovery_metrics"] or generated["raw_metrics"] != manifest["raw_metrics"] or generated["refined_metrics"] != manifest["refined_metrics"] or generated["records"] != manifest["generation_records"] or generated["latent_change"] != manifest["latent_change"]: raise ValueError("Organism refiner semantic replay failed.")
    for key, payload in (("comparison", generated["comparison"]), ("organs", generated["organs"]), ("recovery", generated["recovery"])):
        if hashlib.sha256(payload).hexdigest() != manifest["artifacts"][key]["sha256"]: raise ValueError("Organism refiner visual replay failed.")
    if not all(manifest["hard_gates"].values()) or manifest["claim_boundary"] != {"neural_manifold_refinement": True, "neural_homeostasis_assay": True, "deterministic_raster_cleanup": False, "raw_prior_preserved": True, "runtime_integration_allowed": False, "production_promotion_allowed": False}: raise ValueError("Organism refiner gates/claim boundary drifted.")
    return manifest
