from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping

import numpy as np
import torch

from ..map_decorator.catalog import (
    EMPTY_CLASS,
    build_legal_class_masks,
    validate_decoration_fields,
)
from ..map_decorator.features import EncodedFeatures
from ..map_decorator.hashing import named_arrays_sha256
from ..maps.model import MapData, THEMES
from .contract import HEAD_CLASS_COUNTS, HEAD_NAMES, global_condition_vector
from .legality import TorchLegalMasks, apply_legal_mask, legal_masks_to_torch
from .model import CategoricalRefinementUNet


@dataclass(frozen=True, slots=True)
class SamplerConfig:
    steps: int = 8
    temperature: float = 0.85

    def __post_init__(self) -> None:
        if isinstance(self.steps, bool) or isinstance(self.temperature, bool):
            raise TypeError("Sampler numeric fields cannot be booleans.")
        if not 1 <= self.steps <= 32:
            raise ValueError("Sampler steps must be in [1, 32].")
        if not 0.05 <= self.temperature <= 4.0:
            raise ValueError("Sampler temperature must be in [0.05, 4.0].")


@dataclass(frozen=True, slots=True)
class DecorationPrediction:
    variant: np.ndarray
    decal: np.ndarray
    prop: np.ndarray
    emission: np.ndarray
    report: dict[str, object]

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in HEAD_NAMES}


def _sample_logits(
    masked_logits: torch.Tensor,
    *,
    generator: torch.Generator,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = torch.softmax(masked_logits / temperature, dim=1)
    random = torch.rand(
        masked_logits.shape,
        generator=generator,
        device=masked_logits.device,
        dtype=torch.float32,
    ).clamp_(min=torch.finfo(torch.float32).tiny, max=1.0 - torch.finfo(torch.float32).eps)
    gumbel = -torch.log(-torch.log(random))
    selected = torch.argmax(masked_logits / temperature + gumbel, dim=1)
    confidence = probabilities.gather(1, selected[:, None]).squeeze(1)
    return selected, confidence


def _object_logits_and_legal(
    decal_logits: torch.Tensor,
    prop_logits: torch.Tensor,
    decal_legal: torch.Tensor,
    prop_legal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Empty is one mutually exclusive object choice, not two independently sampled empties.
    empty = torch.logsumexp(
        torch.stack((decal_logits[:, 0], prop_logits[:, 0]), dim=1), dim=1
    ) - math.log(2.0)
    joint_logits = torch.cat(
        (empty[:, None], decal_logits[:, 1:], prop_logits[:, 1:]), dim=1
    )
    joint_legal = torch.cat(
        (decal_legal[:, :1] & prop_legal[:, :1], decal_legal[:, 1:], prop_legal[:, 1:]),
        dim=1,
    )
    if not bool(joint_legal[:, 0].all()):
        raise ValueError("The empty object class must remain legal in every cell.")
    return joint_logits, joint_legal


def _decode_joint_object(joint: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    decal_nonempty = HEAD_CLASS_COUNTS["decal"] - 1
    decal = torch.where(
        (joint >= 1) & (joint <= decal_nonempty), joint, torch.zeros_like(joint)
    )
    prop = torch.where(
        joint > decal_nonempty, joint - decal_nonempty, torch.zeros_like(joint)
    )
    return decal, prop


def _encode_joint_object(decal: torch.Tensor, prop: torch.Tensor) -> torch.Tensor:
    if bool(((decal != 0) & (prop != 0)).any()):
        raise ValueError("Cannot encode multiple object classes in one cell.")
    offset = HEAD_CLASS_COUNTS["decal"] - 1
    return torch.where(decal != 0, decal, torch.where(prop != 0, prop + offset, prop))


def _lowest_confidence_mask(confidence: torch.Tensor, fraction: float) -> torch.Tensor:
    if confidence.ndim != 3:
        raise ValueError("confidence must be [B,H,W].")
    batch, height, width = confidence.shape
    count = int(round(height * width * fraction))
    result = torch.zeros_like(confidence, dtype=torch.bool)
    if count <= 0:
        return result
    if count >= height * width:
        return torch.ones_like(result)
    for index in range(batch):
        # Stable sorting makes equal-confidence cells resolve in row-major order.
        order = torch.argsort(confidence[index].reshape(-1), stable=True)
        result[index].view(-1)[order[:count]] = True
    return result


def _numpy_fields(labels: Mapping[str, torch.Tensor]) -> dict[str, np.ndarray]:
    fields: dict[str, np.ndarray] = {}
    for name in HEAD_NAMES:
        value = labels[name]
        if value.shape[0] != 1:
            raise ValueError("Single-map refinement requires batch size one.")
        fields[name] = np.ascontiguousarray(value[0].detach().cpu().numpy(), dtype=np.uint8)
    return fields


def sample_refinement(
    model: CategoricalRefinementUNet,
    data: MapData,
    encoded: EncodedFeatures,
    *,
    generation_seed: int,
    config: SamplerConfig = SamplerConfig(),
    device: torch.device | str = "cpu",
) -> DecorationPrediction:
    """Run masked parallel refinement, enforcing legal choices before every draw."""
    if (
        isinstance(generation_seed, bool)
        or not isinstance(generation_seed, int)
        or not 0 <= generation_seed < (1 << 63)
    ):
        raise ValueError("generation_seed must be an integer in [0, 2**63).")
    if encoded.map_id != data.map_id or encoded.theme != data.theme:
        raise ValueError("Encoded features are not bound to the supplied source map.")
    if encoded.tensor.shape != (53, *data.shape):
        raise ValueError("Encoded feature tensor shape does not match the source map.")
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU and CUDA refinement devices are supported.")
    original_training = model.training
    model.eval()
    model.to(device)
    feature_tensor = torch.from_numpy(encoded.tensor.copy()).to(device=device)[None]
    conditions = torch.from_numpy(global_condition_vector(encoded)).to(device=device)[None]
    theme = torch.tensor([THEMES.index(data.theme)], dtype=torch.long, device=device)
    height, width = data.shape
    labels = {
        name: torch.zeros((1, height, width), dtype=torch.long, device=device) for name in HEAD_NAMES
    }
    masked = {
        name: torch.ones((1, height, width), dtype=torch.bool, device=device) for name in HEAD_NAMES
    }
    base_masks = build_legal_class_masks(
        data,
        protected_backbone=data.protected_backbone,
        required_clearance=data.required_clearance,
        decoration_forbidden=data.decoration_forbidden,
    )
    base_legal = legal_masks_to_torch(base_masks, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(generation_seed)
    step_hashes: list[str] = []
    prior_deterministic = torch.are_deterministic_algorithms_enabled()
    prior_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        with torch.inference_mode():
            for step in range(config.steps):
                level_value = 1.0 - step / max(config.steps - 1, 1)
                level = torch.full((1,), level_value, dtype=torch.float32, device=device)
                raw = model(feature_tensor, labels, masked, theme, conditions, level)

                variant_logits = apply_legal_mask(raw.variant, base_legal.variant, "variant")
                variant_candidate, variant_confidence = _sample_logits(
                    variant_logits, generator=generator, temperature=config.temperature
                )
                labels["variant"] = torch.where(
                    masked["variant"], variant_candidate, labels["variant"]
                )

                joint_logits, joint_legal = _object_logits_and_legal(
                    raw.decal,
                    raw.prop,
                    base_legal.decal,
                    base_legal.prop,
                )
                if joint_logits.shape != joint_legal.shape or joint_legal.dtype != torch.bool:
                    raise TypeError("Joint object legality must exactly match joint logits.")
                if not bool(torch.isfinite(joint_logits).all()):
                    raise ValueError("Joint object logits contain a non-finite value.")
                if not bool(joint_legal.any(dim=1).all()):
                    raise ValueError("Joint object legality leaves a cell without a class.")
                joint_logits = joint_logits.masked_fill(~joint_legal, -1.0e9)
                object_candidate, object_confidence = _sample_logits(
                    joint_logits, generator=generator, temperature=config.temperature
                )
                old_joint = _encode_joint_object(labels["decal"], labels["prop"])
                joint = torch.where(masked["decal"], object_candidate, old_joint)
                labels["decal"], labels["prop"] = _decode_joint_object(joint)

                current = _numpy_fields(labels)
                conditional_masks = build_legal_class_masks(
                    data,
                    protected_backbone=data.protected_backbone,
                    required_clearance=data.required_clearance,
                    decoration_forbidden=data.decoration_forbidden,
                    selected_variant=current["variant"],
                    selected_decal=current["decal"],
                    selected_prop=current["prop"],
                )
                conditional_legal = legal_masks_to_torch(conditional_masks, device=device)
                emission_logits = apply_legal_mask(
                    raw.emission, conditional_legal.emission, "emission"
                )
                emission_candidate, emission_confidence = _sample_logits(
                    emission_logits, generator=generator, temperature=config.temperature
                )
                old_emission_legal = conditional_legal.emission.gather(
                    1, labels["emission"][:, None]
                ).squeeze(1)
                emission_update = masked["emission"] | ~old_emission_legal
                labels["emission"] = torch.where(
                    emission_update, emission_candidate, labels["emission"]
                )

                observed = _numpy_fields(labels)
                step_report = validate_decoration_fields(
                    data,
                    protected_backbone=data.protected_backbone,
                    required_clearance=data.required_clearance,
                    decoration_forbidden=data.decoration_forbidden,
                    **observed,
                )
                if not step_report["passed"]:
                    raise RuntimeError(
                        f"Refinement step {step} produced an illegal prediction: {step_report}"
                    )
                if bool(
                    (observed["decal"][base_masks.hard_empty] != EMPTY_CLASS).any()
                    or (observed["prop"][base_masks.hard_empty] != EMPTY_CLASS).any()
                    or (observed["emission"][base_masks.hard_empty] != EMPTY_CLASS).any()
                ):
                    raise RuntimeError("Hard-empty cells were not zero during refinement.")
                step_hashes.append(named_arrays_sha256(observed))

                remaining = (config.steps - step - 1) / config.steps
                masked["variant"] = _lowest_confidence_mask(variant_confidence, remaining)
                object_mask = _lowest_confidence_mask(object_confidence, remaining)
                masked["decal"] = object_mask
                masked["prop"] = object_mask
                masked["emission"] = _lowest_confidence_mask(emission_confidence, remaining)
    finally:
        torch.use_deterministic_algorithms(prior_deterministic, warn_only=prior_warn_only)
        model.train(original_training)

    fields = _numpy_fields(labels)
    validation = validate_decoration_fields(
        data,
        protected_backbone=data.protected_backbone,
        required_clearance=data.required_clearance,
        decoration_forbidden=data.decoration_forbidden,
        **fields,
    )
    if not validation["passed"]:
        raise RuntimeError(f"Final refinement candidate was rejected: {validation}")
    replay_sha256 = named_arrays_sha256(fields)
    report: dict[str, object] = {
        "passed": True,
        "map_id": data.map_id,
        "theme": data.theme,
        "generation_seed": generation_seed,
        "steps": config.steps,
        "temperature": config.temperature,
        "field_sha256": replay_sha256,
        "step_field_sha256": step_hashes,
        "feature_tensor_sha256": encoded.tensor_sha256,
        "legal_validation": validation,
        "hard_empty_zero_counts": {
            name: int((fields[name][base_masks.hard_empty] == 0).sum())
            for name in ("decal", "prop", "emission")
        },
    }
    return DecorationPrediction(report=report, **fields)
