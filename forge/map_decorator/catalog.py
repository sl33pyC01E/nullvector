from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Final

import numpy as np

from ..map_art.autotile import EAST, NORTH, WEST, cardinal_match_mask
from ..map_art.styles import STYLES, style_for
from ..maps.model import THEMES, Terrain, MapData
from .contract import CATALOG_CONTRACT_NAME, CATALOG_CONTRACT_VERSION
from .features import validate_feature_inputs
from .hashing import json_sha256, named_arrays_sha256


EMPTY_CLASS: Final[int] = 0
VARIANT_CLASS_COUNT: Final[int] = 8
EMISSION_CLASS_COUNT: Final[int] = 4
_SUPPORTED_COLOR_ROLES: Final[frozenset[str]] = frozenset({"primary", "secondary"})


@dataclass(frozen=True, slots=True)
class DecorationClass:
    class_id: int
    catalog_index: int
    key: str
    kind: str
    allowed_terrain: tuple[int, ...]
    collision: bool
    occlusion: int
    color_role: str
    emission_capable: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "class_id": self.class_id,
            "catalog_index": self.catalog_index,
            "key": self.key,
            "kind": self.kind,
            "allowed_terrain": list(self.allowed_terrain),
            "collision": self.collision,
            "occlusion": self.occlusion,
            "color_role": self.color_role,
            "emission_capable": self.emission_capable,
        }


@dataclass(frozen=True, slots=True)
class ThemeCatalog:
    theme: str
    decal_classes: tuple[DecorationClass, ...]
    prop_classes: tuple[DecorationClass, ...]
    excluded_colliding_props: tuple[str, ...]

    @property
    def decal_class_count(self) -> int:
        return 1 + len(self.decal_classes)

    @property
    def prop_class_count(self) -> int:
        return 1 + len(self.prop_classes)

    def to_dict(self) -> dict[str, object]:
        return {
            "theme": self.theme,
            "empty_class": EMPTY_CLASS,
            "decal_classes": [entry.to_dict() for entry in self.decal_classes],
            "prop_classes": [entry.to_dict() for entry in self.prop_classes],
            "excluded_colliding_props": list(self.excluded_colliding_props),
        }


def _entry(class_id: int, catalog_index: int, spec: object) -> DecorationClass:
    # Attribute access keeps this module coupled to the authoritative PropSpec contract.
    color_role = str(getattr(spec, "color_role"))
    return DecorationClass(
        class_id=class_id,
        catalog_index=catalog_index,
        key=str(getattr(spec, "key")),
        kind=str(getattr(spec, "kind")),
        allowed_terrain=tuple(int(value) for value in getattr(spec, "allowed_terrain")),
        collision=bool(getattr(spec, "collision")),
        occlusion=int(getattr(spec, "occlusion")),
        color_role=color_role,
        # Existing renderer routes these two explicit catalog roles to an emissive palette.
        emission_capable=color_role in _SUPPORTED_COLOR_ROLES,
    )


@lru_cache(maxsize=len(THEMES))
def catalog_for(theme: str) -> ThemeCatalog:
    style = style_for(theme)
    decals: list[DecorationClass] = []
    props: list[DecorationClass] = []
    excluded: list[str] = []
    for catalog_index, spec in enumerate(style.props, start=1):
        if spec.kind == "decal":
            if spec.collision:
                raise RuntimeError(
                    f"Catalog decal {spec.key!r} declares collision and cannot enter the neural contract."
                )
            decals.append(_entry(len(decals) + 1, catalog_index, spec))
        elif spec.collision:
            excluded.append(spec.key)
        else:
            props.append(_entry(len(props) + 1, catalog_index, spec))
    result = ThemeCatalog(theme, tuple(decals), tuple(props), tuple(excluded))
    if any(entry.collision for entry in result.prop_classes):
        raise RuntimeError("Topology-locked prop catalogs may never expose a colliding class.")
    return result


def catalog_manifest() -> dict[str, object]:
    source_catalog = {
        theme: [
            {
                "catalog_index": index,
                "key": spec.key,
                "kind": spec.kind,
                "allowed_terrain": list(spec.allowed_terrain),
                "collision": spec.collision,
                "occlusion": spec.occlusion,
                "color_role": spec.color_role,
            }
            for index, spec in enumerate(STYLES[theme].props, start=1)
        ]
        for theme in THEMES
    }
    derived = {theme: catalog_for(theme).to_dict() for theme in THEMES}
    return {
        "contract_name": CATALOG_CONTRACT_NAME,
        "contract_version": CATALOG_CONTRACT_VERSION,
        "theme_order": list(THEMES),
        "class_semantics": {
            "variant": {"class_count": VARIANT_CLASS_COUNT, "empty_class": None},
            "decal": {"empty_class": EMPTY_CLASS, "theme_local": True},
            "prop": {
                "empty_class": EMPTY_CLASS,
                "theme_local": True,
                "colliding_entries_excluded": True,
            },
            "emission": {
                "class_count": EMISSION_CLASS_COUNT,
                "empty_class": EMPTY_CLASS,
                "levels": ["off", "low", "medium", "high"],
            },
        },
        "emission_capability_policy": {
            "terrain": (
                "exact renderer semantics: walkable exposed north/east/west edge, crystal, "
                "or odd growth variant"
            ),
            "catalog": "only explicit PropSpec color_role primary/secondary",
            "unknown_role": "force emission class 0",
            "pixels_are_never_inspected": True,
        },
        "source_catalog": source_catalog,
        "derived_catalog": derived,
    }


CATALOG_SHA256: Final[str] = json_sha256(catalog_manifest())
MAX_DECAL_CLASSES: Final[int] = 1 + max(len(catalog_for(theme).decal_classes) for theme in THEMES)
MAX_PROP_CLASSES: Final[int] = 1 + max(len(catalog_for(theme).prop_classes) for theme in THEMES)


@dataclass(frozen=True, slots=True)
class LegalClassMasks:
    variant: np.ndarray
    decal: np.ndarray
    prop: np.ndarray
    emission: np.ndarray
    hard_empty: np.ndarray
    catalog_sha256: str
    masks_sha256: str
    theme: str

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "variant": self.variant,
            "decal": self.decal,
            "prop": self.prop,
            "emission": self.emission,
            "hard_empty": self.hard_empty,
        }


def _required_point_mask(data: MapData) -> np.ndarray:
    mask = np.zeros(data.shape, dtype=bool)
    for x, y in (data.start, data.exit, *data.objectives, *data.spawns):
        mask[y, x] = True
    return mask


def _selected_object_emission_capability(
    data: MapData,
    catalog: ThemeCatalog,
    decal: np.ndarray | None,
    prop: np.ndarray | None,
) -> np.ndarray | None:
    if decal is None and prop is None:
        return None
    capability = np.zeros(data.shape, dtype=bool)
    for field, entries, name, maximum in (
        (decal, catalog.decal_classes, "selected_decal", MAX_DECAL_CLASSES),
        (prop, catalog.prop_classes, "selected_prop", MAX_PROP_CLASSES),
    ):
        if field is None:
            continue
        if not isinstance(field, np.ndarray) or field.shape != data.shape or field.dtype != np.uint8:
            raise TypeError(f"{name} must be a uint8 ndarray with shape {data.shape}.")
        if not bool((field < maximum).all()):
            raise ValueError(f"{name} contains an out-of-domain class ID.")
        for entry in entries:
            if entry.emission_capable:
                capability |= field == entry.class_id
    return capability


def build_legal_class_masks(
    data: MapData,
    *,
    protected_backbone: np.ndarray,
    required_clearance: np.ndarray,
    decoration_forbidden: np.ndarray,
    selected_variant: np.ndarray | None = None,
    selected_decal: np.ndarray | None = None,
    selected_prop: np.ndarray | None = None,
) -> LegalClassMasks:
    inputs = validate_feature_inputs(
        data, protected_backbone, required_clearance, decoration_forbidden
    )
    height, width = data.shape
    catalog = catalog_for(data.theme)
    required = _required_point_mask(data)
    hard_empty = (
        inputs.protected_backbone.astype(bool)
        | inputs.required_clearance.astype(bool)
        | inputs.decoration_forbidden.astype(bool)
        | (data.hazard != 0)
        | required
    )

    variant = np.ones((VARIANT_CLASS_COUNT, height, width), dtype=bool)
    decal = np.zeros((MAX_DECAL_CLASSES, height, width), dtype=bool)
    prop = np.zeros((MAX_PROP_CLASSES, height, width), dtype=bool)
    emission = np.zeros((EMISSION_CLASS_COUNT, height, width), dtype=bool)
    decal[EMPTY_CLASS] = True
    prop[EMPTY_CLASS] = True
    emission[EMPTY_CLASS] = True

    available = ~hard_empty
    for entry in catalog.decal_classes:
        decal[entry.class_id] = available & np.isin(data.terrain, entry.allowed_terrain)
    for entry in catalog.prop_classes:
        if entry.collision:
            raise RuntimeError(f"Colliding prop {entry.key!r} reached the legal-class mask.")
        prop[entry.class_id] = (
            available
            & data.walkability.astype(bool)
            & np.isin(data.terrain, entry.allowed_terrain)
        )

    object_capability = _selected_object_emission_capability(
        data, catalog, selected_decal, selected_prop
    )
    if object_capability is None:
        object_capability = np.zeros(data.shape, dtype=bool)
        for entry, legality in (
            *((entry, decal[entry.class_id]) for entry in catalog.decal_classes),
            *((entry, prop[entry.class_id]) for entry in catalog.prop_classes),
        ):
            if entry.emission_capable:
                object_capability |= legality
    same_terrain = cardinal_match_mask(data.terrain)
    emissive_edge_bits = NORTH | EAST | WEST
    exposed_emissive_edge = data.walkability.astype(bool) & (
        (same_terrain & emissive_edge_bits) != emissive_edge_bits
    )
    if selected_variant is None:
        growth_capability = data.terrain == int(Terrain.GROWTH)
    else:
        _validate_field(selected_variant, data.shape, VARIANT_CLASS_COUNT, "selected_variant")
        growth_capability = (
            (data.terrain == int(Terrain.GROWTH)) & ((selected_variant & 1) != 0)
        )
    terrain_capability = (
        exposed_emissive_edge
        | (data.terrain == int(Terrain.CRYSTAL))
        | growth_capability
    )
    emission_capability = available & (terrain_capability | object_capability)
    emission[1:] = emission_capability

    arrays = {
        "variant": variant,
        "decal": decal,
        "prop": prop,
        "emission": emission,
        "hard_empty": hard_empty,
    }
    for array in arrays.values():
        array.setflags(write=False)
    return LegalClassMasks(
        variant=variant,
        decal=decal,
        prop=prop,
        emission=emission,
        hard_empty=hard_empty,
        catalog_sha256=CATALOG_SHA256,
        masks_sha256=named_arrays_sha256(arrays),
        theme=data.theme,
    )


def _validate_field(field: np.ndarray, shape: tuple[int, int], classes: int, name: str) -> None:
    if not isinstance(field, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray.")
    if field.shape != shape or field.dtype != np.uint8:
        raise TypeError(f"{name} must be uint8 with shape {shape}.")
    if not bool((field < classes).all()):
        raise ValueError(f"{name} contains an out-of-domain class ID.")


def validate_decoration_fields(
    data: MapData,
    *,
    protected_backbone: np.ndarray,
    required_clearance: np.ndarray,
    decoration_forbidden: np.ndarray,
    variant: np.ndarray,
    decal: np.ndarray,
    prop: np.ndarray,
    emission: np.ndarray,
) -> dict[str, object]:
    """Reject any selection that violates a hard mask; never clamp or repair."""
    _validate_field(variant, data.shape, VARIANT_CLASS_COUNT, "variant")
    _validate_field(decal, data.shape, MAX_DECAL_CLASSES, "decal")
    _validate_field(prop, data.shape, MAX_PROP_CLASSES, "prop")
    _validate_field(emission, data.shape, EMISSION_CLASS_COUNT, "emission")
    masks = build_legal_class_masks(
        data,
        protected_backbone=protected_backbone,
        required_clearance=required_clearance,
        decoration_forbidden=decoration_forbidden,
        selected_variant=variant,
        selected_decal=decal,
        selected_prop=prop,
    )
    yy, xx = np.indices(data.shape)
    failures: list[str] = []
    for name, field, legal in (
        ("variant", variant, masks.variant),
        ("decal", decal, masks.decal),
        ("prop", prop, masks.prop),
        ("emission", emission, masks.emission),
    ):
        if not bool(legal[field, yy, xx].all()):
            failures.append(f"illegal.{name}")
    if bool(((decal != EMPTY_CLASS) & (prop != EMPTY_CLASS)).any()):
        failures.append("objects.multiple_classes_per_cell")
    catalog = catalog_for(data.theme)
    exposed_colliding = any(entry.collision for entry in catalog.prop_classes)
    if exposed_colliding:
        failures.append("props.colliding_class_exposed")
    return {
        "passed": not failures,
        "failures": failures,
        "map_id": data.map_id,
        "theme": data.theme,
        "catalog_sha256": CATALOG_SHA256,
        "legal_masks_sha256": masks.masks_sha256,
        "field_sha256": named_arrays_sha256(
            {"variant": variant, "decal": decal, "prop": prop, "emission": emission}
        ),
        "counts": {
            "decal": int((decal != EMPTY_CLASS).sum()),
            "prop": int((prop != EMPTY_CLASS).sum()),
            "emission": int((emission != EMPTY_CLASS).sum()),
            "hard_empty": int(masks.hard_empty.sum()),
        },
    }
