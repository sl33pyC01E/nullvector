from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..morphology.constants import FAMILIES, ROLE_NAMES, SUBTYPE_NAMES
from ..multifield_style.model import StyleCondition
from ..neural_fusion.genetics import SAFE_MARGIN, _components, _connect, _readonly
from ..neural_fusion.hashing import array_sha256, canonical_json_bytes, sha256_bytes
from ..neural_fusion.model import FusionGenome, FusionSpecimen
from ..neural_rig_bridge.hashing import aligned_fields_hash
from ..neural_rig_repair.model import RepairSourceSample
from ..sprite_latent.codec import CodecOutput, SemanticSpriteFSQ, project_legal_tuples
from .codec import ProductionCodecAuthority, load_production_codec
from .contract import DEFAULT_PRODUCTION_MANIFEST, FUSION_MODES, MUTATION_MODES, production_fusion_source_hash


def _condition(model: SemanticSpriteFSQ, sample: RepairSourceSample) -> torch.Tensor:
    return model.condition_vector(
        torch.tensor([sample.family_id]),
        torch.tensor([sample.subtype_id]),
        torch.tensor([sample.role_id]),
        torch.from_numpy(sample.genes[None].copy()),
    )


def _fields(sample: RepairSourceSample) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(torch.from_numpy(values[None].copy()).long() for values in (sample.part_owner, sample.material, sample.emission_level))  # type: ignore[return-value]


def _hash_noise(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed & 0x7FFFFFFFFFFFFFFF)
    return torch.randn(shape, generator=generator)


def _mix_latents(a: torch.Tensor, b: torch.Tensor, *, mode: str, alpha: float, seed: int) -> torch.Tensor:
    _, channels, height, width = a.shape
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    if mode == "linear":
        return torch.lerp(a, b, alpha)
    if mode == "spatial_weave":
        phase = (seed % 997) / 997.0 * math.tau
        wave = torch.sin(xx.float() * 0.81 + yy.float() * 0.53 + phase)
        mask = torch.sigmoid((wave + (alpha - 0.5) * 2.4) * 3.0)[None, None]
        return torch.lerp(a, b, mask)
    if mode == "voronoi_mosaic":
        generator = torch.Generator().manual_seed(seed & 0x7FFFFFFF)
        sites = torch.randint(0, min(height, width), (6, 2), generator=generator)
        ownership = torch.randint(0, 2, (6,), generator=generator)
        distance = (yy[..., None] - sites[:, 0]) ** 2 + (xx[..., None] - sites[:, 1]) ** 2
        nearest = distance.argmin(dim=-1)
        hard = ownership[nearest].float()
        soft = F.avg_pool2d(hard[None, None], 3, stride=1, padding=1)
        mask = torch.clamp(0.72 * soft + 0.28 * alpha, 0.0, 1.0)
        return torch.lerp(a, b, mask)
    if mode == "radial_graft":
        cx = (seed % width) * 0.45 + width * 0.275
        cy = ((seed >> 8) % height) * 0.45 + height * 0.275
        radius = 2.2 + alpha * 5.2
        distance = torch.sqrt((xx.float() - cx) ** 2 + (yy.float() - cy) ** 2)
        mask = torch.sigmoid((radius - distance) * 2.2)[None, None]
        return torch.lerp(a, b, mask)
    if mode == "channel_crossover":
        offsets = torch.linspace(-0.3, 0.3, channels)
        if seed & 1:
            offsets = offsets.flip(0)
        weights = torch.clamp(alpha + offsets, 0.0, 1.0)[None, :, None, None]
        return torch.lerp(a, b, weights)
    if mode == "spectral_splice":
        low_a = F.avg_pool2d(a, 3, stride=1, padding=1)
        low_b = F.avg_pool2d(b, 3, stride=1, padding=1)
        low = torch.lerp(low_a, low_b, alpha)
        high = torch.lerp(a - low_a, b - low_b, 1.0 - alpha)
        return low + high
    raise ValueError(f"unsupported production latent fusion mode {mode!r}")


def _mutate_latent(mixed: torch.Tensor, donor: torch.Tensor, *, mode: str, strength: int, seed: int) -> tuple[torch.Tensor, int]:
    if mode == "none" or strength == 0:
        return mixed, 0
    _, channels, height, width = mixed.shape
    amplitude = 0.035 + 0.035 * strength
    noise = _hash_noise(tuple(mixed.shape), seed ^ 0x4D555441)
    if mode == "latent_gaussian":
        return mixed + noise * amplitude, channels * height * width
    if mode == "spatial_burst":
        yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        cx, cy = (seed % width), ((seed >> 8) % height)
        radius = 1.5 + strength * 1.3
        mask = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2).float() / (2.0 * radius * radius))[None, None]
        changed = int((mask > 0.08).sum()) * channels
        return mixed + noise * mask * amplitude * 2.0, changed
    if mode == "channel_phase":
        selected = (seed + torch.arange(channels)) % max(2, 5 - strength) == 0
        result = mixed.clone(); result[:, selected] = -result[:, selected]
        return result, int(selected.sum()) * height * width
    if mode == "donor_transplant":
        y0 = seed % max(1, height - 3); x0 = (seed >> 7) % max(1, width - 3)
        size = min(2 + strength, height - y0, width - x0)
        result = mixed.clone(); result[:, :, y0:y0 + size, x0:x0 + size] = donor[:, :, y0:y0 + size, x0:x0 + size]
        return result, channels * size * size
    if mode == "phase_wave":
        yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        phase = (seed % 360) * math.pi / 180.0
        wave = torch.sin(xx.float() * 0.71 + yy.float() * 0.37 + phase)[None, None]
        return mixed + wave * amplitude, channels * height * width
    raise ValueError(f"unsupported production latent mutation mode {mode!r}")


def _project(model: SemanticSpriteFSQ, raw: torch.Tensor, condition: torch.Tensor, legal: torch.Tensor):
    quantized = model.quantizer(raw, quantize=True)
    logits = model.decode(quantized["quantized"], condition)
    output = CodecOutput(
        part_logits=logits[0], material_logits=logits[1], emission_logits=logits[2],
        latent=quantized["quantized"], continuous_latent=quantized["continuous"],
        digits=quantized["digits"], codes=quantized["codes"], perplexity=quantized["perplexity"],
        utilization=quantized["utilization"], marginal_entropy=quantized["marginal_entropy"],
        soft_marginal_entropy=quantized["soft_marginal_entropy"], quantized=True,
    )
    return project_legal_tuples(output, legal), quantized


def production_latent_fuse(
    parent_a: RepairSourceSample,
    parent_b: RepairSourceSample,
    *,
    seed: int,
    alpha: float = 0.5,
    fusion_mode: str = "linear",
    mutation_mode: str = "none",
    mutation_strength: int = 0,
    dominant_parent: str = "auto",
    manifest_path: Path = DEFAULT_PRODUCTION_MANIFEST,
) -> FusionSpecimen:
    if parent_a.sample_id == parent_b.sample_id or fusion_mode not in FUSION_MODES or mutation_mode not in MUTATION_MODES:
        raise ValueError("production latent fusion parent/operator contract failed")
    if not 0.0 <= alpha <= 1.0 or not 0 <= mutation_strength <= 3 or dominant_parent not in {"auto", "a", "b"}:
        raise ValueError("production latent fusion parameter contract failed")
    if not np.array_equal(parent_a.legal_tuples, parent_b.legal_tuples):
        raise ValueError("production latent parents disagree on legal tuple authority")
    authority: ProductionCodecAuthority = load_production_codec(str(Path(manifest_path).resolve()))
    model = authority.model
    legal_np = np.ascontiguousarray(parent_a.legal_tuples, dtype=np.uint8)
    legal = torch.from_numpy(legal_np.copy()).long()
    dominant_key = ("a" if alpha < 0.5 else "b") if dominant_parent == "auto" else dominant_parent
    dominant = parent_a if dominant_key == "a" else parent_b
    with torch.no_grad():
        condition_a, condition_b = _condition(model, parent_a), _condition(model, parent_b)
        latent_a = model.encode(*_fields(parent_a), condition_a, quantize=False)["prequantized"]
        latent_b = model.encode(*_fields(parent_b), condition_b, quantize=False)["prequantized"]
        mixed = _mix_latents(latent_a, latent_b, mode=fusion_mode, alpha=alpha, seed=seed)
        mixed, mutation_cells = _mutate_latent(mixed, latent_b if dominant_key == "a" else latent_a, mode=mutation_mode, strength=mutation_strength, seed=seed)
        condition = torch.lerp(condition_a, condition_b, alpha)
        projected, quantized = _project(model, mixed, condition, legal)
        recon_a, qa = _project(model, latent_a, condition_a, legal)
        recon_b, qb = _project(model, latent_b, condition_b, legal)
    part = projected["part"][0].numpy().astype(np.uint8)
    material = projected["material"][0].numpy().astype(np.uint8)
    emission = projected["emission"][0].numpy().astype(np.uint8)
    safe = np.zeros((48, 48), dtype=bool); safe[SAFE_MARGIN:-SAFE_MARGIN, SAFE_MARGIN:-SAFE_MARGIN] = True
    part[~safe] = material[~safe] = emission[~safe] = 0
    child_code = part.astype(np.int32) * 40 + material.astype(np.int32) * 4 + emission
    parent_codes = []
    for projected_parent in (recon_a, recon_b):
        parent_codes.append(
            projected_parent["part"][0].numpy().astype(np.int32) * 40
            + projected_parent["material"][0].numpy().astype(np.int32) * 4
            + projected_parent["emission"][0].numpy().astype(np.int32)
        )
    distance_a = np.abs(child_code - parent_codes[0]); distance_b = np.abs(child_code - parent_codes[1])
    visible = part != 0
    provenance = np.zeros((48, 48), dtype=np.uint8)
    provenance[visible & (distance_a < distance_b)] = 1
    provenance[visible & (distance_b < distance_a)] = 2
    provenance[visible & (distance_a == distance_b)] = 3
    components_before = len(_components(visible))
    repair_pixels = _connect(part, material, emission, provenance, legal_np)
    visible = part != 0
    occupancy = float(visible.mean())
    if len(_components(visible)) != 1 or not 0.02 <= occupancy <= 0.60:
        raise ValueError("production latent fusion failed connected occupancy gates")
    tuples = np.stack((part, material, emission), axis=-1).reshape(-1, 3)
    legal_set = {tuple(map(int, row)) for row in legal_np}
    if any(tuple(map(int, row)) not in legal_set for row in tuples):
        raise ValueError("production latent fusion escaped the legal tuple vocabulary")
    contribution_a, contribution_b = int((provenance == 1).sum()), int((provenance == 2).sum())
    if min(contribution_a, contribution_b) < 8:
        raise ValueError("production latent fusion lacks bilateral parent contribution")
    genes = np.clip((1.0 - alpha) * parent_a.genes + alpha * parent_b.genes, 0.0, 1.0).astype(np.float32)
    guide = ((1.0 - alpha) * parent_a.guide + alpha * parent_b.guide).astype(np.float32)
    guide[0] = visible.astype(np.float32); guide[1] = np.isin(part, (1, 2, 3, 10)); guide[5] = part == 10
    fields_hash = aligned_fields_hash(part, material, emission)
    lineage = {
        "format": "nullvector-production-neural-latent-lineage-v1",
        "parent_a": parent_a.sample_id, "parent_b": parent_b.sample_id,
        "seed": seed, "alpha": round(float(alpha), 6), "fusion_mode": fusion_mode,
        "mutation_mode": mutation_mode, "mutation_strength": mutation_strength,
        "dominant_parent": dominant_key, "fields_sha256": fields_hash,
        "production_manifest_sha256": authority.manifest["manifest_sha256"],
        "production_checkpoint_sha256": authority.checkpoint_file_sha256,
        "production_ema_sha256": authority.ema_state_sha256,
    }
    lineage_sha = sha256_bytes(canonical_json_bytes(lineage))
    specimen_id = f"pfx_{parent_a.ordinal:02d}_{parent_b.ordinal:02d}_{lineage_sha[:12]}"
    role_id = parent_a.role_id if alpha < 0.5 else parent_b.role_id
    condition_record = StyleCondition(
        sample_id=specimen_id, ordinal=0, sample_seed=seed & 0x7FFFFFFF,
        morphology_id=dominant.family_id, morphology_name=FAMILIES[dominant.family_id],
        subtype_id=dominant.subtype_id, subtype_name=SUBTYPE_NAMES[dominant.subtype_id],
        role_id=role_id, role_name=ROLE_NAMES[role_id],
    )
    codes = quantized["codes"]
    code_a, code_b = qa["codes"], qb["codes"]
    child_unique = set(map(int, torch.unique(codes).tolist()))
    parent_unique = set(map(int, torch.unique(torch.cat((code_a.flatten(), code_b.flatten()))).tolist()))
    metrics = {
        "quality_tier": "production-learned-latent-authority-v1",
        "occupancy": round(occupancy, 9), "component_count_before_repair": components_before,
        "component_count": 1, "connective_repair_pixels": repair_pixels,
        "parent_a_attributed_pixels": contribution_a, "parent_b_attributed_pixels": contribution_b,
        "novel_decoded_pixels": int((provenance == 3).sum()), "latent_mutation_cells": mutation_cells,
        "unique_codes": len(child_unique), "novel_codes": len(child_unique - parent_unique),
        "parent_code_union": len(parent_unique),
        "latent_distance_a": round(float(torch.mean((mixed - latent_a) ** 2)), 9),
        "latent_distance_b": round(float(torch.mean((mixed - latent_b) ** 2)), 9),
        "production_manifest_sha256": authority.manifest["manifest_sha256"],
        "production_checkpoint_sha256": authority.checkpoint_file_sha256,
        "production_ema_sha256": authority.ema_state_sha256,
        "production_fusion_source_sha256": production_fusion_source_hash(),
    }
    return FusionSpecimen(
        genome=FusionGenome(
            specimen_id=specimen_id, seed=seed, parent_a_ordinal=parent_a.ordinal,
            parent_b_ordinal=parent_b.ordinal, parent_a_sample_id=parent_a.sample_id,
            parent_b_sample_id=parent_b.sample_id, dominant_parent=dominant_key,
            fusion_mode=f"production_latent_{fusion_mode}", mutation_mode=mutation_mode,
            mutation_strength=mutation_strength, mirror_donor=False,
            condition=condition_record, lineage_sha256=lineage_sha,
        ),
        part_owner=_readonly(part, np.uint8), material=_readonly(material, np.uint8),
        emission_level=_readonly(emission, np.uint8), provenance=_readonly(provenance, np.uint8),
        guide=_readonly(guide, np.float32), genes=_readonly(genes, np.float32),
        legal_tuples=_readonly(legal_np, np.uint8), fields_sha256=fields_hash,
        provenance_sha256=array_sha256("production_latent_provenance", provenance), metrics=metrics,
    )
