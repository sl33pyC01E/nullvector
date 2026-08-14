from __future__ import annotations

from contextlib import nullcontext
from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Final

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import Tensor
from torch.utils.data._utils.collate import default_collate

from ..map_topology_neural_production.checkpoint import tensor_state_sha256
from ..safety import require_disk_floor
from .contract import CHECKPOINT_FORMAT, FORMAT, FROZEN_UPSTREAM, OrganismVAEConfig, canonical_json_bytes, organism_vae_source_sha256, sha256_file, source_manifest
from .dataset import OrganismRasterCorpus
from .model import ContinuousOrganismRasterVAE, OrganismVAEOutput, organism_vae_loss, reconstruction_metrics


MANIFEST_NAME: Final[str] = "organism_vae_manifest.json"
CHECKPOINT_NAME: Final[str] = "checkpoint.pt"
CONTACT_NAME: Final[str] = "reconstruction_contact_sheet.png"
FUSION_NAME: Final[str] = "latent_fusion_sheet.png"
MUTATION_NAME: Final[str] = "latent_mutation_sheet.png"
MAX_CHECKPOINT_BYTES: Final[int] = 256 * 1024 * 1024
SEED: Final[int] = 0x4F524756414531


def _png(image: Image.Image) -> bytes:
    buffer = BytesIO(); image.save(buffer, "PNG", optimize=False, compress_level=9); return buffer.getvalue()


def _artifact(path: Path) -> dict[str, Any]:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _atomic_bytes(path: Path, payload: bytes) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent); temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle: handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally: temporary.unlink(missing_ok=True)


def _batch(corpus: OrganismRasterCorpus, indices: list[int], device: torch.device) -> dict[str, Tensor | list[str]]:
    collated = default_collate([corpus[index] for index in indices])
    return {name: value.to(device) if isinstance(value, Tensor) else value for name, value in collated.items()}


def _model_batch(batch: dict[str, Tensor | list[str]]) -> dict[str, Tensor]:
    return {name: value for name, value in batch.items() if isinstance(value, Tensor)}


def _rgba_image(values: Tensor, scale: int = 3) -> Image.Image:
    rgba = values.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    rgba[..., :3] *= rgba[..., 3:4]
    array = np.rint(rgba * 255).astype(np.uint8)
    return Image.fromarray(array).resize((48 * scale, 48 * scale), Image.Resampling.NEAREST)


def _field_image(output: OrganismVAEOutput, index: int, scale: int = 3) -> Image.Image:
    rgba = output.rgba[index].detach().cpu().clone(); rgba[3] = output.occupancy_logits[index, 0].sigmoid().detach().cpu()
    return _rgba_image(rgba, scale)


def _physiology_image(values: Tensor, occupancy: Tensor, scale: int = 3) -> Image.Image:
    palette = torch.tensor(((250, 73, 86), (90, 210, 255), (255, 178, 71), (206, 91, 255), (255, 238, 110), (91, 255, 161), (255, 104, 190), (108, 238, 205)), dtype=torch.float32) / 255
    strength, system = values.detach().cpu().max(dim=0); rgb = palette[system].permute(2, 0, 1) * strength[None]
    alpha = occupancy.detach().cpu().clamp(0, 1); rgba = torch.cat((rgb, alpha[None]), dim=0)
    return _rgba_image(rgba, scale)


def _contact(corpus: OrganismRasterCorpus, model: ContinuousOrganismRasterVAE) -> tuple[bytes, list[str]]:
    indices = [corpus.indices_by_family[family][offset] for family in range(5) for offset in (0, 1)]
    batch = _batch(corpus, indices, torch.device("cpu")); tensors = _model_batch(batch)
    model.eval()
    with torch.inference_mode(): output = model(tensors["living_field"], tensors["family"], tensors["subtype"], tensors["role"], tensors["genes"], sample=False)
    scale = 3; cell = 48 * scale; canvas = Image.new("RGB", (24 + cell * 4 + 42, 58 + len(indices) * (cell + 18)), (4, 8, 16)); draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), "CONTINUOUS ORGANISM VAE // LEARNED RASTER + ORGAN SYSTEMS", fill=(155, 239, 255)); draw.text((12, 23), "TARGET RGBA | DECODED RGBA | TARGET PHYSIOLOGY | DECODED PHYSIOLOGY", fill=(78, 154, 181))
    for row, index in enumerate(indices):
        y = 56 + row * (cell + 18); original = _rgba_image(tensors["rgba"][row]); decoded = _field_image(output, row)
        target_systems = _physiology_image(tensors["physiology"][row], tensors["occupancy"][row]); decoded_systems = _physiology_image(output.physiology[row], output.occupancy_logits[row, 0].sigmoid())
        for column, image in enumerate((original, decoded, target_systems, decoded_systems)): canvas.paste(image.convert("RGB"), (12 + column * (cell + 8), y))
        draw.text((12, y + cell + 2), corpus.samples[index].sample_id, fill=(108, 155, 174))
    return _png(canvas), [corpus.samples[index].sample_id for index in indices]


def _fusion(corpus: OrganismRasterCorpus, model: ContinuousOrganismRasterVAE) -> tuple[bytes, list[dict[str, Any]]]:
    scale = 3; cell = 48 * scale; steps = 7; canvas = Image.new("RGB", (24 + steps * (cell + 6), 42 + 5 * (cell + 22)), (4, 8, 16)); draw = ImageDraw.Draw(canvas); draw.text((12, 8), "NEURAL FUSION // CONTINUOUS LATENT INTERPOLATION", fill=(175, 247, 143)); draw.text((12, 23), "same-family identities / decoder receives interpolated anatomy + condition", fill=(76, 143, 112)); records: list[dict[str, Any]] = []
    model.eval()
    for family in range(5):
        indices = corpus.indices_by_family[family][:2]; batch = _batch(corpus, indices, torch.device("cpu")); tensors = _model_batch(batch)
        with torch.inference_mode():
            condition = model.condition_vector(tensors["family"], tensors["subtype"], tensors["role"], tensors["genes"]); means, _ = model.encode(tensors["living_field"], condition)
            y = 40 + family * (cell + 22)
            for step in range(steps):
                alpha = step / (steps - 1); latent = means[:1] * (1 - alpha) + means[1:] * alpha; mixed_condition = condition[:1] * (1 - alpha) + condition[1:] * alpha; output = model.decode(latent, mixed_condition)
                canvas.paste(_field_image(output, 0).convert("RGB"), (12 + step * (cell + 6), y)); draw.text((12 + step * (cell + 6), y + cell + 2), f"{alpha:.2f}", fill=(99, 177, 135))
        records.append({"family": family, "left": corpus.samples[indices[0]].sample_id, "right": corpus.samples[indices[1]].sample_id})
    return _png(canvas), records


def _mutations(corpus: OrganismRasterCorpus, model: ContinuousOrganismRasterVAE) -> tuple[bytes, list[str]]:
    scale = 3; cell = 48 * scale; variants = 6; canvas = Image.new("RGB", (24 + variants * (cell + 6), 42 + 5 * (cell + 22)), (4, 8, 16)); draw = ImageDraw.Draw(canvas); draw.text((12, 8), "LATENT MUTATION // CONTINUOUS POSTERIOR PERTURBATION", fill=(255, 165, 237)); draw.text((12, 23), "base mean + bounded correlated Gaussian edits", fill=(155, 87, 145)); ids: list[str] = []; generator = torch.Generator(device="cpu").manual_seed(SEED ^ 0x4D5554415445); model.eval()
    for family in range(5):
        index = corpus.indices_by_family[family][0]; ids.append(corpus.samples[index].sample_id); batch = _batch(corpus, [index], torch.device("cpu")); tensors = _model_batch(batch)
        with torch.inference_mode():
            condition = model.condition_vector(tensors["family"], tensors["subtype"], tensors["role"], tensors["genes"]); mean, _ = model.encode(tensors["living_field"], condition); y = 40 + family * (cell + 22)
            for variant in range(variants):
                if variant == 0: latent = mean
                else:
                    noise = torch.randn(mean.shape, generator=generator); noise = torch.nn.functional.avg_pool2d(noise, 3, stride=1, padding=1); latent = mean + noise * (.18 * variant)
                output = model.decode(latent, condition); canvas.paste(_field_image(output, 0).convert("RGB"), (12 + variant * (cell + 6), y)); draw.text((12 + variant * (cell + 6), y + cell + 2), "BASE" if variant == 0 else f"M{variant}", fill=(190, 105, 179))
    return _png(canvas), ids


def _state_hash(state: dict[str, Tensor]) -> str:
    return tensor_state_sha256({name: value.detach().cpu() for name, value in state.items()})


def _load_checkpoint(path: Path) -> tuple[ContinuousOrganismRasterVAE, dict[str, Any]]:
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= MAX_CHECKPOINT_BYTES: raise ValueError("Organism VAE checkpoint is missing or oversized.")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    keys = {"format", "source_sha256", "source_manifest", "upstream", "config", "steps", "seed", "model_state", "model_state_sha256", "history"}
    if not isinstance(payload, dict) or set(payload) != keys or payload["format"] != CHECKPOINT_FORMAT or payload["source_sha256"] != organism_vae_source_sha256() or payload["source_manifest"] != source_manifest() or payload["upstream"] != FROZEN_UPSTREAM: raise ValueError("Organism VAE checkpoint provenance drifted.")
    config = OrganismVAEConfig(**payload["config"]); model = ContinuousOrganismRasterVAE(config); model.load_state_dict(payload["model_state"], strict=True)
    if _state_hash(model.state_dict()) != payload["model_state_sha256"]: raise ValueError("Organism VAE model-state hash failed.")
    if type(payload["steps"]) is not int or payload["steps"] < 1 or type(payload["seed"]) is not int or not isinstance(payload["history"], list) or len(payload["history"]) != payload["steps"]: raise ValueError("Organism VAE checkpoint training census drifted.")
    return model, payload


def run_smoke(output: Path, *, device_name: str = "cpu", steps: int = 32) -> dict[str, Any]:
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError("Organism VAE smoke publication is immutable.")
    if type(steps) is not int or not 4 <= steps <= 2_048: raise ValueError("Organism VAE smoke steps must be in [4,2048].")
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=512 * 1024 * 1024)
    if device_name == "cuda":
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError("Organism VAE CUDA smoke requires deterministic BF16 CUDA.")
        if torch.cuda.mem_get_info(0)[0] < 4 * 1024**3: raise RuntimeError("Organism VAE CUDA smoke requires 4 GiB free VRAM.")
        device = torch.device("cuda", 0); torch.cuda.reset_peak_memory_stats(device)
    elif device_name == "cpu": device = torch.device("cpu")
    else: raise ValueError("Organism VAE device must be cpu or cuda.")
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(SEED); np.random.seed(SEED & 0xffffffff)
    if device.type == "cuda": torch.cuda.manual_seed_all(SEED)
    corpus = OrganismRasterCorpus(); config = OrganismVAEConfig(width=48, latent_channels=12, residual_depth=2)
    model = ContinuousOrganismRasterVAE(config).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4); latent_generator = torch.Generator(device=device).manual_seed(SEED ^ 0x4C4154454E54); order_generator = torch.Generator(device="cpu").manual_seed(SEED ^ 0x4241544348)
    history: list[dict[str, float | int]] = []; started = time.perf_counter(); order = torch.randperm(len(corpus), generator=order_generator).tolist(); cursor = 0
    for step in range(steps):
        if cursor + 9 > len(order): order = torch.randperm(len(corpus), generator=order_generator).tolist(); cursor = 0
        indices = order[cursor:cursor + 9]; cursor += 9; batch = _model_batch(_batch(corpus, indices, device)); model.train(); optimizer.zero_grad(set_to_none=True)
        context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        with context: output_value = model(batch["living_field"], batch["family"], batch["subtype"], batch["role"], batch["genes"], generator=latent_generator, sample=True)
        loss, pieces = organism_vae_loss(output_value, batch, config, beta_scale=min(1, (step + 1) / max(8, steps // 2))); loss.float().backward(); gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(gradient)): raise FloatingPointError("Organism VAE smoke became non-finite.")
        optimizer.step(); history.append({"step": step + 1, **{name: round(float(value), 8) for name, value in pieces.items()}, "gradient_norm": round(float(gradient), 8)})
    training_seconds = time.perf_counter() - started; state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}; cpu_model = ContinuousOrganismRasterVAE(config); cpu_model.load_state_dict(state); cpu_model.eval()
    evaluation_batch = _model_batch(_batch(corpus, list(range(len(corpus))), torch.device("cpu")))
    with torch.inference_mode(): evaluation_output = cpu_model(evaluation_batch["living_field"], evaluation_batch["family"], evaluation_batch["subtype"], evaluation_batch["role"], evaluation_batch["genes"], sample=False)
    metrics = reconstruction_metrics(evaluation_output, evaluation_batch); contact, contact_ids = _contact(corpus, cpu_model); fusion, fusion_pairs = _fusion(corpus, cpu_model); mutation, mutation_ids = _mutations(corpus, cpu_model)
    staging = output.parent / f".{output.name}.tmp-{os.getpid()}-{time.time_ns()}"; staging.mkdir(parents=True)
    try:
        checkpoint_payload = {"format": CHECKPOINT_FORMAT, "source_sha256": organism_vae_source_sha256(), "source_manifest": source_manifest(), "upstream": FROZEN_UPSTREAM, "config": config.to_dict(), "steps": steps, "seed": SEED, "model_state": state, "model_state_sha256": _state_hash(state), "history": history}
        torch.save(checkpoint_payload, staging / CHECKPOINT_NAME); _load_checkpoint(staging / CHECKPOINT_NAME)
        _atomic_bytes(staging / CONTACT_NAME, contact); _atomic_bytes(staging / FUSION_NAME, fusion); _atomic_bytes(staging / MUTATION_NAME, mutation)
        artifacts = {name: _artifact(staging / filename) for name, filename in (("checkpoint", CHECKPOINT_NAME), ("reconstruction", CONTACT_NAME), ("fusion", FUSION_NAME), ("mutation", MUTATION_NAME))}
        gates = {"upstream_exact": True, "identity_census_45": len(corpus) == 45, "all_families_represented": all(corpus.indices_by_family[family] for family in range(5)), "finite_training": all(math.isfinite(float(row["loss"])) for row in history), "model_updated": history[0]["loss"] != history[-1]["loss"], "continuous_latent_nonzero_variance": metrics["latent_std_mean"] > 0, "fusion_artifact_present": len(fusion_pairs) == 5, "mutation_artifact_present": len(mutation_ids) == 5, "production_promotion_allowed": False}
        runtime = {"training_device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu", "training_precision": "bf16-autocast-float32-loss" if device.type == "cuda" else "float32", "training_seconds": training_seconds, "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0, "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0, "artifact_render_device": "cpu"}
        positive_gates = {key: value for key, value in gates.items() if key != "production_promotion_allowed"}
        manifest: dict[str, Any] = {"format": FORMAT, "status": "passed" if all(positive_gates.values()) and gates["production_promotion_allowed"] is False else "failed", "source_sha256": organism_vae_source_sha256(), "source_manifest": source_manifest(), "upstream": FROZEN_UPSTREAM, "config": config.to_dict(), "sample_count": len(corpus), "steps": steps, "seed": SEED, "metrics": metrics, "loss_start": history[0]["loss"], "loss_end": history[-1]["loss"], "history_sha256": hashlib.sha256(canonical_json_bytes(history)).hexdigest(), "model_state_sha256": checkpoint_payload["model_state_sha256"], "artifacts": artifacts, "contact_sample_ids": contact_ids, "fusion_pairs": fusion_pairs, "mutation_sample_ids": mutation_ids, "runtime": runtime, "gates": gates, "claim_boundary": {"continuous_vae_rasterizer_foundation": True, "neural_fusion_and_mutation_demonstrated": True, "generative_prior_trained": False, "production_promotion_allowed": False}}
        if manifest["status"] != "passed": raise ValueError("Organism VAE smoke safety gates failed.")
        manifest["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest(); _atomic_bytes(staging / MANIFEST_NAME, canonical_json_bytes(manifest)); require_disk_floor(output.parent, floor_gb=100, planned_bytes=0); os.replace(staging, output)
    except BaseException:
        if staging.exists(): os.replace(staging, output.parent / f"{staging.name}.failed-{time.time_ns()}")
        raise
    return validate_smoke(output)


def validate_smoke(output: Path) -> dict[str, Any]:
    output = Path(output).resolve(); manifest_path = output / MANIFEST_NAME
    if not manifest_path.is_file() or manifest_path.is_symlink() or not 0 < manifest_path.stat().st_size <= 4 * 1024 * 1024: raise ValueError("Organism VAE manifest is missing or oversized.")
    encoded = manifest_path.read_bytes(); manifest = json.loads(encoded)
    if encoded != canonical_json_bytes(manifest): raise ValueError("Organism VAE manifest is not canonical JSON.")
    stored = manifest.pop("manifest_sha256", None)
    if stored != hashlib.sha256(canonical_json_bytes(manifest)).hexdigest(): raise ValueError("Organism VAE manifest self-hash failed.")
    manifest["manifest_sha256"] = stored
    required = {"format", "status", "source_sha256", "source_manifest", "upstream", "config", "sample_count", "steps", "seed", "metrics", "loss_start", "loss_end", "history_sha256", "model_state_sha256", "artifacts", "contact_sample_ids", "fusion_pairs", "mutation_sample_ids", "runtime", "gates", "claim_boundary", "manifest_sha256"}
    if set(manifest) != required or manifest["format"] != FORMAT or manifest["status"] != "passed" or manifest["source_sha256"] != organism_vae_source_sha256() or manifest["source_manifest"] != source_manifest() or manifest["upstream"] != FROZEN_UPSTREAM: raise ValueError("Organism VAE manifest contract/provenance drifted.")
    if manifest["sample_count"] != 45 or manifest["seed"] != SEED or manifest["config"] != OrganismVAEConfig(**manifest["config"]).to_dict(): raise ValueError("Organism VAE manifest configuration drifted.")
    if set(manifest["artifacts"]) != {"checkpoint", "reconstruction", "fusion", "mutation"}: raise ValueError("Organism VAE artifact census drifted.")
    for record in manifest["artifacts"].values():
        if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"} or Path(record["path"]).name != record["path"]: raise ValueError("Organism VAE artifact descriptor drifted.")
        path = output / record["path"]
        if not path.is_file() or path.is_symlink() or path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]: raise ValueError("Organism VAE artifact identity failed.")
    model, checkpoint = _load_checkpoint(output / CHECKPOINT_NAME)
    if checkpoint["model_state_sha256"] != manifest["model_state_sha256"] or checkpoint["steps"] != manifest["steps"] or checkpoint["seed"] != manifest["seed"] or hashlib.sha256(canonical_json_bytes(checkpoint["history"])).hexdigest() != manifest["history_sha256"]: raise ValueError("Organism VAE checkpoint/manifest semantics drifted.")
    expected_gates = {"upstream_exact": True, "identity_census_45": True, "all_families_represented": True, "finite_training": True, "model_updated": True, "continuous_latent_nonzero_variance": True, "fusion_artifact_present": True, "mutation_artifact_present": True, "production_promotion_allowed": False}
    if manifest["gates"] != expected_gates or manifest["claim_boundary"] != {"continuous_vae_rasterizer_foundation": True, "neural_fusion_and_mutation_demonstrated": True, "generative_prior_trained": False, "production_promotion_allowed": False}: raise ValueError("Organism VAE claim boundary drifted.")
    corpus = OrganismRasterCorpus(); contact, ids = _contact(corpus, model); fusion, pairs = _fusion(corpus, model); mutation, mutation_ids = _mutations(corpus, model)
    for payload, name in ((contact, "reconstruction"), (fusion, "fusion"), (mutation, "mutation")):
        if hashlib.sha256(payload).hexdigest() != manifest["artifacts"][name]["sha256"]: raise ValueError("Organism VAE exact visual replay failed.")
    if ids != manifest["contact_sample_ids"] or pairs != manifest["fusion_pairs"] or mutation_ids != manifest["mutation_sample_ids"]: raise ValueError("Organism VAE visual registry replay failed.")
    evaluation_batch = _model_batch(_batch(corpus, list(range(len(corpus))), torch.device("cpu")))
    with torch.inference_mode(): evaluation_output = model(evaluation_batch["living_field"], evaluation_batch["family"], evaluation_batch["subtype"], evaluation_batch["role"], evaluation_batch["genes"], sample=False)
    if reconstruction_metrics(evaluation_output, evaluation_batch) != manifest["metrics"]: raise ValueError("Organism VAE metric replay failed.")
    return manifest
