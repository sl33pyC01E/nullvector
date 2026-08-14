from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final

from ..map_decorator.catalog import MAX_DECAL_CLASSES, MAX_PROP_CLASSES, catalog_for
from ..map_decorator.hashing import json_sha256
from ..map_decorator_ml.contract import ModelConfig
from ..maps.model import THEMES


V4_CONTRACT_NAME: Final[str] = "nullvector-map-decorator-public-entropy-residual-v4"
V4_CONTRACT_VERSION: Final[str] = "4.0.0"
OBJECT_PLACEMENT_SALT: Final[int] = 0x4F424A454354
DECAL_PROPOSAL_CHANNELS: Final[int] = MAX_DECAL_CLASSES - 1
PROP_PROPOSAL_CHANNELS: Final[int] = MAX_PROP_CLASSES - 1
PROPOSAL_CHANNEL_COUNT: Final[int] = DECAL_PROPOSAL_CHANNELS + PROP_PROPOSAL_CHANNELS


@dataclass(frozen=True, slots=True)
class ProposalLocatorConfig:
    locator_channels: int = 32
    locator_blocks: int = 2
    count_hidden_channels: int = 32
    candidate_logit_prior: float = 4.0
    noncandidate_logit_prior: float = -8.0
    count_residual_scale: float = 0.25
    proposal_type_logit_prior: float = 8.0
    maximum_objects_per_head: int = 4096

    def __post_init__(self) -> None:
        integers = (
            self.locator_channels,
            self.locator_blocks,
            self.count_hidden_channels,
            self.maximum_objects_per_head,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integers):
            raise TypeError("V4 locator dimensions must be integers.")
        if not 4 <= self.locator_channels <= 256 or not 1 <= self.locator_blocks <= 8:
            raise ValueError("V4 locator tower dimensions are outside their bounded contract.")
        if not 4 <= self.count_hidden_channels <= 256:
            raise ValueError("V4 count-head width is outside its bounded contract.")
        if not 1 <= self.maximum_objects_per_head <= 4096:
            raise ValueError("V4 maximum object quota is outside its bounded contract.")
        for name, value, lower, upper in (
            ("candidate_logit_prior", self.candidate_logit_prior, 0.0, 16.0),
            ("noncandidate_logit_prior", self.noncandidate_logit_prior, -16.0, 0.0),
            ("count_residual_scale", self.count_residual_scale, 0.0, 2.0),
            ("proposal_type_logit_prior", self.proposal_type_logit_prior, 0.0, 16.0),
        ):
            if isinstance(value, bool) or not lower <= float(value) <= upper:
                raise ValueError(f"{name} is outside [{lower},{upper}].")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def proposal_channel_manifest() -> dict[str, object]:
    themes: dict[str, object] = {}
    for theme in THEMES:
        catalog = catalog_for(theme)
        themes[theme] = {
            "decal": [
                {
                    "channel": entry.class_id - 1,
                    "class_id": entry.class_id,
                    "catalog_index": entry.catalog_index,
                    "key": entry.key,
                }
                for entry in catalog.decal_classes
            ],
            "prop": [
                {
                    "channel": entry.class_id - 1,
                    "class_id": entry.class_id,
                    "catalog_index": entry.catalog_index,
                    "key": entry.key,
                }
                for entry in catalog.prop_classes
            ],
        }
    return {
        "format": "nullvector-map-decorator-public-proposal-channels/1.0.0",
        "object_placement_salt": OBJECT_PLACEMENT_SALT,
        "channel_counts": {
            "decal": DECAL_PROPOSAL_CHANNELS,
            "prop": PROP_PROPOSAL_CHANNELS,
            "total": PROPOSAL_CHANNEL_COUNT,
        },
        "themes": themes,
        "semantics": {
            "entropy": "exact public map-seed coordinate hash used by the renderer",
            "legality": "class legality and hard-empty applied before neural consumption",
            "target_fields_read": False,
            "inference_available": True,
            "neural_residual_role": "suppress proposal conflicts, refine counts, style and type",
        },
    }


def v4_contract_manifest() -> dict[str, object]:
    return {
        "contract_name": V4_CONTRACT_NAME,
        "contract_version": V4_CONTRACT_VERSION,
        "proposal_channels": proposal_channel_manifest(),
        "categorical_core": ModelConfig().to_dict(),
        "locator": ProposalLocatorConfig().to_dict(),
        "acceptance": {
            "all_target_object_cells_covered_by_public_proposals": True,
            "zero_off_proposal_object_decodes": True,
            "zero_topology_or_legality_changes": True,
            "quality_gates_unchanged_from_v2_v3": True,
        },
        "safety": {
            "cpu_foundation_only": True,
            "cuda_training_authorized": False,
            "godot_integration_authorized": False,
            "production_claim": False,
        },
    }


V4_CONTRACT_SHA256: Final[str] = json_sha256(v4_contract_manifest())
PROPOSAL_CHANNEL_MANIFEST_SHA256: Final[str] = json_sha256(proposal_channel_manifest())
