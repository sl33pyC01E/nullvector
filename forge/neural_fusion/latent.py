from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..morphology.constants import FAMILIES, ROLE_NAMES, SUBTYPE_NAMES
from ..multifield_style.model import StyleCondition
from ..neural_rig_bridge.hashing import aligned_fields_hash
from ..neural_rig_repair.model import RepairSourceSample
from ..sprite_latent.codec import CodecOutput, SemanticSpriteFSQ, SpriteLatentConfig, project_legal_tuples
from ..sprite_latent.smoke import validate_smoke_output
from ..sprite_latent.training import canonical_state_hash
from .genetics import SAFE_MARGIN, _components, _connect, _readonly
from .hashing import array_sha256, canonical_json_bytes, sha256_bytes
from .model import FusionGenome, FusionSpecimen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SMOKE_MANIFEST = PROJECT_ROOT / "outputs" / "sprite_latent" / "smoke_v2" / "smoke_manifest.json"
LATENT_MODES = ("linear", "spatial", "channel", "mutagenic")


def latent_source_hash() -> str:
    files = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "forge" / "sprite_latent" / "codec.py",
        PROJECT_ROOT / "forge" / "sprite_latent" / "smoke.py",
        PROJECT_ROOT / "forge" / "sprite_latent" / "training.py",
    )
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


@lru_cache(maxsize=2)
def load_latent_codec(manifest_path: str = str(DEFAULT_SMOKE_MANIFEST)) -> tuple[SemanticSpriteFSQ, dict[str, Any], str]:
    path = Path(manifest_path).resolve()
    manifest = validate_smoke_output(path)
    if manifest["scope"] != "cpu-foundation-smoke-not-production":
        raise ValueError("latent fusion v1 expects the explicitly experimental smoke codec")
    checkpoint_record = manifest["artifacts"]["checkpoint"]
    checkpoint_path = path.parent / checkpoint_record["path"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    raw_config = dict(checkpoint["config"])
    for derived in ("latent_dim", "implicit_code_count", "latent_grid_size", "quantizer", "usage_regularizer"):
        raw_config.pop(derived, None)
    raw_config["latent_levels"] = tuple(raw_config["latent_levels"])
    config = SpriteLatentConfig(**raw_config)
    model = SemanticSpriteFSQ(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    if canonical_state_hash(model) != checkpoint["state_sha256"]:
        raise ValueError("latent fusion codec state hash mismatch")
    model.eval()
    if any(parameter.device.type != "cpu" for parameter in model.parameters()) or torch.cuda.is_initialized():
        raise ValueError("latent fusion experimental codec must remain CPU-only")
    return model, manifest, str(checkpoint["state_sha256"])


def _condition(model: SemanticSpriteFSQ, sample: RepairSourceSample) -> torch.Tensor:
    return model.condition_vector(
        torch.tensor([sample.family_id], dtype=torch.long),
        torch.tensor([sample.subtype_id], dtype=torch.long),
        torch.tensor([sample.role_id], dtype=torch.long),
        torch.from_numpy(sample.genes[None].copy()),
    )


def _fields(sample: RepairSourceSample) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(sample.part_owner[None].copy()).long(),
        torch.from_numpy(sample.material[None].copy()).long(),
        torch.from_numpy(sample.emission_level[None].copy()).long(),
    )


def _project(
    model: SemanticSpriteFSQ,
    raw_latent: torch.Tensor,
    condition: torch.Tensor,
    legal: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    quantized = model.quantizer(raw_latent, quantize=True)
    logits = model.decode(quantized["quantized"], condition)
    output = CodecOutput(
        part_logits=logits[0],
        material_logits=logits[1],
        emission_logits=logits[2],
        latent=quantized["quantized"],
        continuous_latent=quantized["continuous"],
        digits=quantized["digits"],
        codes=quantized["codes"],
        perplexity=quantized["perplexity"],
        utilization=quantized["utilization"],
        marginal_entropy=quantized["marginal_entropy"],
        soft_marginal_entropy=quantized["soft_marginal_entropy"],
        quantized=True,
    )
    return project_legal_tuples(output, legal), quantized


def latent_fuse(
    parent_a: RepairSourceSample,
    parent_b: RepairSourceSample,
    *,
    seed: int,
    alpha: float = 0.5,
    mode: str = "linear",
    dominant_parent: str = "a",
    manifest_path: Path = DEFAULT_SMOKE_MANIFEST,
) -> FusionSpecimen:
    """Fuse two neural fields inside a learned FSQ latent grid.

    The current checkpoint is intentionally a CPU foundation smoke, so outputs
    are marked experimental.  Categorical decoding is clamped to the frozen
    legal tuple table and connectivity/margins are repaired deterministically.
    """

    if parent_a.sample_id == parent_b.sample_id or mode not in LATENT_MODES:
        raise ValueError("latent fusion requires distinct parents and a canonical mode")
    if dominant_parent not in {"a", "b"} or not 0.0 <= alpha <= 1.0:
        raise ValueError("latent fusion alpha/dominance contract failed")
    model, manifest, checkpoint_sha = load_latent_codec(str(Path(manifest_path).resolve()))
    legal_np = np.ascontiguousarray(parent_a.legal_tuples, dtype=np.uint8)
    legal = torch.from_numpy(legal_np.copy()).long()
    dominant = parent_a if dominant_parent == "a" else parent_b
    with torch.no_grad():
        condition_a = _condition(model, parent_a)
        condition_b = _condition(model, parent_b)
        fields_a = _fields(parent_a)
        fields_b = _fields(parent_b)
        latent_a = model.encode(*fields_a, condition_a, quantize=False)["prequantized"]
        latent_b = model.encode(*fields_b, condition_b, quantize=False)["prequantized"]
        condition = _condition(model, dominant)
        if mode == "linear":
            mixed = (1.0 - alpha) * latent_a + alpha * latent_b
        elif mode == "spatial":
            yy, xx = torch.meshgrid(torch.arange(12), torch.arange(12), indexing="ij")
            phase = (seed % 97) / 97.0
            mask = torch.sigmoid(((xx.float() / 11.0) + 0.22 * torch.sin(yy.float() + phase * 6.28) - alpha) * 8.0)[None, None]
            mixed = latent_a * (1.0 - mask) + latent_b * mask
        elif mode == "channel":
            weights = torch.tensor(
                [alpha, 1.0 - alpha, min(1.0, alpha + 0.2), max(0.0, alpha - 0.2)],
                dtype=torch.float32,
            )[None, :, None, None]
            mixed = latent_a * (1.0 - weights) + latent_b * weights
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed & 0x7FFFFFFF)
            noise = torch.randn(latent_a.shape, generator=generator) * (0.08 + 0.18 * abs(alpha - 0.5))
            mixed = (1.0 - alpha) * latent_a + alpha * latent_b + noise
        projected, quantized = _project(model, mixed, condition, legal)
        recon_a, _ = _project(model, latent_a, condition_a, legal)
        recon_b, _ = _project(model, latent_b, condition_b, legal)

    part = projected["part"][0].numpy().astype(np.uint8)
    material = projected["material"][0].numpy().astype(np.uint8)
    emission = projected["emission"][0].numpy().astype(np.uint8)
    pa, ma, ea = (recon_a[name][0].numpy() for name in ("part", "material", "emission"))
    pb, mb, eb = (recon_b[name][0].numpy() for name in ("part", "material", "emission"))
    child_code = part.astype(np.int32) * 40 + material.astype(np.int32) * 4 + emission
    code_a = pa.astype(np.int32) * 40 + ma.astype(np.int32) * 4 + ea
    code_b = pb.astype(np.int32) * 40 + mb.astype(np.int32) * 4 + eb
    provenance = np.zeros((48, 48), dtype=np.uint8)
    visible = part != 0
    provenance[visible & (child_code == code_a) & (child_code != code_b)] = 1
    provenance[visible & (child_code == code_b) & (child_code != code_a)] = 2
    shared = visible & (child_code == code_a) & (child_code == code_b)
    provenance[shared] = 1 if alpha < 0.5 else 2
    provenance[visible & (provenance == 0)] = 3

    safe = np.zeros((48, 48), dtype=bool)
    safe[SAFE_MARGIN:-SAFE_MARGIN, SAFE_MARGIN:-SAFE_MARGIN] = True
    part[~safe] = material[~safe] = emission[~safe] = provenance[~safe] = 0
    components_before = len(_components(part != 0))
    connective = _connect(part, material, emission, provenance, legal_np)
    occupancy = float((part != 0).mean())
    if not 0.02 <= occupancy <= 0.60 or len(_components(part != 0)) != 1:
        raise ValueError("latent fusion decode failed occupancy/connectivity gates")
    fields_hash = aligned_fields_hash(part, material, emission)
    lineage = {
        "format": "nullvector-neural-latent-fusion-lineage-v1",
        "parent_a": parent_a.sample_id,
        "parent_b": parent_b.sample_id,
        "seed": seed,
        "alpha": round(float(alpha), 6),
        "mode": mode,
        "dominant_parent": dominant_parent,
        "codec_state_sha256": checkpoint_sha,
        "codec_manifest_sha256": manifest["manifest_sha256"],
        "fields_sha256": fields_hash,
    }
    lineage_sha = sha256_bytes(canonical_json_bytes(lineage))
    specimen_id = f"lf_{parent_a.ordinal:02d}_{parent_b.ordinal:02d}_{lineage_sha[:12]}"
    role_id = parent_a.role_id if alpha < 0.5 else parent_b.role_id
    condition_record = StyleCondition(
        sample_id=specimen_id,
        ordinal=0,
        sample_seed=seed & 0x7FFFFFFF,
        morphology_id=dominant.family_id,
        morphology_name=FAMILIES[dominant.family_id],
        subtype_id=dominant.subtype_id,
        subtype_name=SUBTYPE_NAMES[dominant.subtype_id],
        role_id=role_id,
        role_name=ROLE_NAMES[role_id],
    )
    genes = ((1.0 - alpha) * parent_a.genes + alpha * parent_b.genes).astype(np.float32)
    guide = ((1.0 - alpha) * parent_a.guide + alpha * parent_b.guide).astype(np.float32)
    guide[0] = (part != 0).astype(np.float32)
    guide[1] = np.isin(part, (1, 2, 3, 10)).astype(np.float32)
    guide[5] = (part == 10).astype(np.float32)
    return FusionSpecimen(
        genome=FusionGenome(
            specimen_id=specimen_id,
            seed=seed,
            parent_a_ordinal=parent_a.ordinal,
            parent_b_ordinal=parent_b.ordinal,
            parent_a_sample_id=parent_a.sample_id,
            parent_b_sample_id=parent_b.sample_id,
            dominant_parent=dominant_parent,
            fusion_mode=f"latent_{mode}",
            mutation_mode="latent_noise" if mode == "mutagenic" else "none",
            mutation_strength=1 if mode == "mutagenic" else 0,
            mirror_donor=False,
            condition=condition_record,
            lineage_sha256=lineage_sha,
        ),
        part_owner=_readonly(part, np.uint8),
        material=_readonly(material, np.uint8),
        emission_level=_readonly(emission, np.uint8),
        provenance=_readonly(provenance, np.uint8),
        guide=_readonly(guide, np.float32),
        genes=_readonly(genes, np.float32),
        legal_tuples=_readonly(legal_np, np.uint8),
        fields_sha256=fields_hash,
        provenance_sha256=array_sha256("latent_fusion_provenance", provenance),
        metrics={
            "quality_tier": "experimental-smoke-codec-not-production",
            "alpha": round(float(alpha), 6),
            "latent_mode": mode,
            "occupancy": round(occupancy, 9),
            "component_count_before_repair": components_before,
            "component_count": 1,
            "connective_repair_pixels": connective,
            "parent_a_attributed_pixels": int((provenance == 1).sum()),
            "parent_b_attributed_pixels": int((provenance == 2).sum()),
            "novel_decoded_pixels": int((provenance == 3).sum()),
            "unique_codes": int(torch.unique(quantized["codes"]).numel()),
            "codec_state_sha256": checkpoint_sha,
            "codec_manifest_sha256": manifest["manifest_sha256"],
            "latent_source_sha256": latent_source_hash(),
        },
    )
