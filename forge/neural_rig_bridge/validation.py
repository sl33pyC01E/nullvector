from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator
import numpy as np

from ..config import PROJECT_ROOT
from ..morphology.constants import (
    CANVAS_SIZE,
    FAMILIES,
    PART_OWNER_NAMES,
    ROLE_NAMES,
    SUBTYPE_NAMES,
)
from .binding import (
    _MAX_LEGAL_TUPLES,
    _SAMPLE_ID,
    _SHA256,
    _UPSTREAM_NAME,
    _anatomy_provenance_errors,
    _anatomy_hash,
    _anchors,
    _build_driver_index,
    _field_errors,
    _manifest,
    _normalize_legal_tuples,
    _owner_layers,
    _point_errors,
    _topology_errors,
)
from .hashing import (
    aligned_fields_hash,
    array_hash,
    binder_source_hash,
    canonical_json_hash,
    owner_tuple_hash,
    tuple_fingerprint,
)
from .model import (
    BACKGROUND_DRIVER,
    BINDER_VERSION,
    BINDING_FORMAT,
    DRIVER_INDEX,
    DRIVER_NAMES,
    JOINT_DRIVER,
    SOCKET_DRIVER,
    BindingRejected,
    DerivedAnatomy,
    NeuralRigBinding,
    MIN_DRIVER_PIXELS,
    RigAnchor,
)


BINDING_SCHEMA = "neural_rig_binding.schema.json"


def validate_binding_schema(
    manifest: Mapping[str, Any], schema_path: Path | None = None
) -> list[str]:
    path = (
        Path(schema_path)
        if schema_path is not None
        else PROJECT_ROOT / "shared" / "schema" / BINDING_SCHEMA
    )
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        errors = sorted(
            Draft202012Validator(schema).iter_errors(dict(manifest)),
            key=lambda error: tuple(map(str, error.absolute_path)),
        )
        return [
            f"{'/'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
            for error in errors
        ]
    except (OSError, json.JSONDecodeError) as error:
        return [f"binding schema could not be loaded: {error}"]


def _graph_errors(manifest: Mapping[str, Any]) -> list[str]:
    graph = manifest.get("graph")
    if not isinstance(graph, Mapping):
        return ["manifest graph is missing"]
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        return ["manifest graph nodes/edges are malformed"]
    node_ids = {
        str(node.get("id")) for node in nodes if isinstance(node, Mapping)
    }
    if node_ids != set(DRIVER_NAMES):
        return ["manifest graph nodes disagree with the driver vocabulary"]
    adjacency = {name: set() for name in DRIVER_NAMES}
    for edge in edges:
        if not isinstance(edge, Mapping):
            return ["manifest graph edge is not an object"]
        parent, child = edge.get("parent"), edge.get("child")
        if parent not in adjacency or child not in adjacency:
            return ["manifest graph edge references an unknown node"]
        adjacency[str(parent)].add(str(child))
        adjacency[str(child)].add(str(parent))
    reached = {"body"}
    stack = ["body"]
    while stack:
        node = stack.pop()
        for neighbor in sorted(adjacency[node]):
            if neighbor not in reached:
                reached.add(neighbor)
                stack.append(neighbor)
    if reached != set(DRIVER_NAMES):
        return ["logical rig graph is disconnected"]
    return []


def _validate_binding_impl(
    binding: NeuralRigBinding, *, schema: bool = True
) -> list[str]:
    errors: list[str] = []
    if not isinstance(binding, NeuralRigBinding):
        return ["value is not a NeuralRigBinding"]

    identity_valid = True
    if not isinstance(binding.sample_id, str) or not _SAMPLE_ID.fullmatch(
        binding.sample_id
    ):
        errors.append("binding sample_id is not safe")
        identity_valid = False
    if binding.family not in FAMILIES:
        errors.append("binding family is unsupported")
        identity_valid = False
    elif (
        isinstance(binding.family_id, (bool, np.bool_))
        or not isinstance(binding.family_id, (int, np.integer))
        or int(binding.family_id) != FAMILIES.index(binding.family)
    ):
        errors.append("binding family_id disagrees with family")
        identity_valid = False
    if (
        isinstance(binding.subtype_id, (bool, np.bool_))
        or not isinstance(binding.subtype_id, (int, np.integer))
        or not 0 <= int(binding.subtype_id) < len(SUBTYPE_NAMES)
        or (
            binding.family in FAMILIES
            and int(binding.subtype_id) // 4 != FAMILIES.index(binding.family)
        )
    ):
        errors.append("binding subtype_id is invalid for its family")
        identity_valid = False
    if (
        isinstance(binding.role_id, (bool, np.bool_))
        or not isinstance(binding.role_id, (int, np.integer))
        or not 0 <= int(binding.role_id) < len(ROLE_NAMES)
    ):
        errors.append("binding role_id is invalid")
        identity_valid = False
    if binding.corpus_seed is not None and (
        isinstance(binding.corpus_seed, (bool, np.bool_))
        or not isinstance(binding.corpus_seed, (int, np.integer))
        or not 0 <= int(binding.corpus_seed) <= 0xFFFFFFFF
    ):
        errors.append("binding corpus_seed is invalid")
        identity_valid = False

    legal_valid = True
    if (
        not isinstance(binding.legal_tuples, np.ndarray)
        or binding.legal_tuples.ndim != 2
        or binding.legal_tuples.shape[1:] != (3,)
        or not len(binding.legal_tuples)
        or len(binding.legal_tuples) > _MAX_LEGAL_TUPLES
        or binding.legal_tuples.dtype != np.uint8
    ):
        errors.append("legal_tuples is not a bounded uint8 [N, 3] table")
        legal_valid = False
    else:
        normalized = _normalize_legal_tuples(binding.legal_tuples)
        if not np.array_equal(binding.legal_tuples, normalized):
            errors.append("legal_tuples is not uniquely sorted and canonical")
            legal_valid = False

    field_errors: list[str]
    if legal_valid:
        field_errors = _field_errors(
            binding.part_owner,
            binding.material,
            binding.emission_level,
            binding.guide,
            binding.legal_tuples,
        )
    else:
        field_errors = ["fields cannot be checked against malformed legal_tuples"]
    errors.extend(field_errors)
    fields_valid = not field_errors
    if fields_valid and binding.family in FAMILIES:
        errors.extend(_topology_errors(binding.part_owner, binding.family))

    if binding.genes is not None and (
        not isinstance(binding.genes, np.ndarray)
        or binding.genes.shape != (24,)
        or binding.genes.dtype != np.float32
        or not np.isfinite(binding.genes).all()
    ):
        errors.append("genes must be null or one finite float32 vector of length 24")

    for name, values in (
        ("part_owner", binding.part_owner),
        ("material", binding.material),
        ("emission_level", binding.emission_level),
        ("guide", binding.guide),
        ("legal_tuples", binding.legal_tuples),
        ("owner_masks", binding.owner_masks),
        ("driver_index", binding.driver_index),
        ("genes", binding.genes),
    ):
        if isinstance(values, np.ndarray) and values.flags.writeable:
            errors.append(f"{name} must be read-only")

    masks_valid = False
    if fields_valid:
        expected_masks = np.stack(
            [binding.part_owner == owner for owner in range(1, len(PART_OWNER_NAMES))]
        ).astype(np.uint8)
        if (
            not isinstance(binding.owner_masks, np.ndarray)
            or binding.owner_masks.shape != expected_masks.shape
            or binding.owner_masks.dtype != np.uint8
        ):
            errors.append(f"owner_masks must be uint8 {expected_masks.shape}")
        elif not np.array_equal(binding.owner_masks, expected_masks):
            errors.append("owner_masks do not exactly partition part_owner")
        else:
            masks_valid = True
    if masks_valid:
        reconstructed = binding.reconstruct_fields()
        if not np.array_equal(reconstructed[0], binding.part_owner):
            errors.append("owner layer reconstruction changes part_owner")
        if not np.array_equal(reconstructed[1], binding.material):
            errors.append("owner layer reconstruction changes material")
        if not np.array_equal(reconstructed[2], binding.emission_level):
            errors.append("owner layer reconstruction changes emission_level")

    driver_valid = False
    if (
        not isinstance(binding.driver_index, np.ndarray)
        or binding.driver_index.shape != (CANVAS_SIZE, CANVAS_SIZE)
        or binding.driver_index.dtype != np.uint8
    ):
        errors.append("driver_index must be uint8 48x48")
    elif fields_valid:
        foreground = binding.part_owner != 0
        if not np.all(binding.driver_index[~foreground] == BACKGROUND_DRIVER):
            errors.append("background pixels have motion drivers")
        if not np.all(binding.driver_index[foreground] < len(DRIVER_NAMES)):
            errors.append("foreground pixels have invalid motion drivers")
        for driver_id, name in enumerate(DRIVER_NAMES):
            pixel_count = int((binding.driver_index == driver_id).sum())
            if pixel_count < MIN_DRIVER_PIXELS[name]:
                errors.append(
                    f"driver {name} has {pixel_count} pixels; "
                    f"minimum is {MIN_DRIVER_PIXELS[name]}"
                )
        driver_valid = not any(
            message.startswith("background pixels")
            or message.startswith("foreground pixels")
            or message.startswith("driver ")
            for message in errors
        )

    anatomy_valid = isinstance(binding.anatomy, DerivedAnatomy)
    if not anatomy_valid:
        errors.append("anatomy must be a DerivedAnatomy")
    elif fields_valid:
        anatomy_errors = _point_errors(
            binding.anatomy, binding.part_owner, binding.guide
        )
        if not anatomy_errors and identity_valid:
            anatomy_errors.extend(
                _anatomy_provenance_errors(
                    binding.anatomy,
                    binding.part_owner,
                    binding.guide,
                    corpus_seed=(
                        int(binding.corpus_seed)
                        if binding.corpus_seed is not None
                        else None
                    ),
                    family_id=int(binding.family_id),
                    subtype_id=int(binding.subtype_id),
                    role_id=int(binding.role_id),
                )
            )
        errors.extend(anatomy_errors)
        anatomy_valid = not anatomy_errors

    for values, expected, kind in (
        (binding.joints, JOINT_DRIVER, "joint"),
        (binding.sockets, SOCKET_DRIVER, "socket"),
    ):
        if not isinstance(values, Mapping) or set(values) != set(expected):
            errors.append(f"{kind} keys disagree with the binding vocabulary")
            continue
        for name, anchor in values.items():
            if not isinstance(anchor, RigAnchor):
                errors.append(f"{kind}.{name} is not a RigAnchor")
                continue
            if anchor.name != name or anchor.kind != kind:
                errors.append(f"{kind}.{name} identity is incorrect")
            point = anchor.point
            support = anchor.support_point
            point_valid = (
                isinstance(point, tuple)
                and len(point) == 2
                and all(type(value) is int for value in point)
                and all(0 <= value < CANVAS_SIZE for value in point)
            )
            support_valid = (
                isinstance(support, tuple)
                and len(support) == 2
                and all(type(value) is int for value in support)
                and all(0 <= value < CANVAS_SIZE for value in support)
            )
            if not point_valid:
                errors.append(f"{kind}.{name} pivot is outside the canvas or malformed")
            if not support_valid:
                errors.append(
                    f"{kind}.{name} support is outside the canvas or malformed"
                )
            if not point_valid or not support_valid:
                continue
            x, y = anchor.point
            support_x, support_y = anchor.support_point
            if anchor.driver != expected[name]:
                errors.append(f"{kind}.{name} has the wrong driver")
            elif driver_valid and int(
                binding.driver_index[support_y, support_x]
            ) != DRIVER_INDEX[anchor.driver]:
                errors.append(f"{kind}.{name} support does not land on its driver")
            if max(abs(support_x - x), abs(support_y - y)) > 6:
                errors.append(f"{kind}.{name} support is too far from its pivot")
            if fields_valid and anchor.observed_owner != int(binding.part_owner[y, x]):
                errors.append(f"{kind}.{name} observed owner is stale")
            if anatomy_valid:
                anatomy_point = (
                    binding.anatomy.joints[name]
                    if kind == "joint"
                    else binding.anatomy.sockets[name]
                )
                if anchor.point != anatomy_point:
                    errors.append(f"{kind}.{name} pivot disagrees with anatomy")
                if anchor.source != binding.anatomy.source:
                    errors.append(f"{kind}.{name} source disagrees with anatomy")

    if (
        not isinstance(binding.owner_layers, tuple)
        or len(binding.owner_layers) != len(PART_OWNER_NAMES) - 1
    ):
        errors.append("owner_layers must contain every foreground owner slot")
    elif fields_valid and driver_valid:
        for owner_id, layer in enumerate(binding.owner_layers, start=1):
            mask = binding.part_owner == owner_id
            expected_drivers = tuple(
                DRIVER_NAMES[index]
                for index in sorted(
                    value
                    for value in set(map(int, np.unique(binding.driver_index[mask])))
                    if value < len(DRIVER_NAMES)
                )
            )
            if layer.owner_id != owner_id or layer.owner_name != PART_OWNER_NAMES[owner_id]:
                errors.append(f"owner layer {owner_id} identity is incorrect")
            if layer.pixel_count != int(mask.sum()):
                errors.append(f"owner layer {owner_id} pixel count is incorrect")
            if layer.drivers != expected_drivers:
                errors.append(f"owner layer {owner_id} driver set is incorrect")
            if layer.tuple_sha256 != owner_tuple_hash(
                owner_id, mask, binding.material, binding.emission_level
            ):
                errors.append(f"owner layer {owner_id} tuple hash is incorrect")

    derived_valid = fields_valid and anatomy_valid and driver_valid
    if derived_valid:
        expected_driver = _build_driver_index(binding.part_owner, binding.anatomy)
        if not np.array_equal(binding.driver_index, expected_driver):
            errors.append("driver_index is not the deterministic anatomy projection")
            derived_valid = False
        else:
            expected_joints, expected_sockets = _anchors(
                binding.part_owner, binding.anatomy, expected_driver
            )
            if dict(binding.joints) != dict(expected_joints):
                errors.append("joint bindings are not the deterministic projection")
                derived_valid = False
            if dict(binding.sockets) != dict(expected_sockets):
                errors.append("socket bindings are not the deterministic projection")
                derived_valid = False
            expected_masks, expected_layers = _owner_layers(
                binding.part_owner,
                binding.material,
                binding.emission_level,
                expected_driver,
            )
            if not np.array_equal(binding.owner_masks, expected_masks):
                errors.append("owner masks are not the deterministic projection")
                derived_valid = False
            if binding.owner_layers != expected_layers:
                errors.append("owner layers are not the deterministic projection")
                derived_valid = False

    upstream_valid = isinstance(binding.upstream_hashes, Mapping)
    if not upstream_valid:
        errors.append("upstream_hashes is not a mapping")
    else:
        if len(binding.upstream_hashes) > 32:
            errors.append("upstream_hashes contains more than 32 entries")
            upstream_valid = False
        for name, value in binding.upstream_hashes.items():
            if (
                not isinstance(name, str)
                or not _UPSTREAM_NAME.fullmatch(name)
                or not isinstance(value, str)
                or not _SHA256.fullmatch(value)
            ):
                errors.append("upstream_hashes contains an invalid name or SHA-256")
                upstream_valid = False
                break
        evaluator_fingerprint = binding.upstream_hashes.get(
            "legal_tuple_fingerprint"
        )
        if legal_valid and evaluator_fingerprint is not None:
            from .hashing import evaluator_tuple_fingerprint

            if evaluator_fingerprint != evaluator_tuple_fingerprint(
                binding.legal_tuples
            ):
                errors.append(
                    "upstream legal_tuple_fingerprint disagrees with legal_tuples"
                )
                upstream_valid = False

    manifest = binding.manifest
    if not isinstance(manifest, Mapping):
        errors.append("manifest is not a mapping")
        return errors
    if manifest.get("format") != BINDING_FORMAT:
        errors.append("manifest format is unsupported")
    if manifest.get("binder_version") != BINDER_VERSION:
        errors.append("manifest binder version is unsupported")
    if manifest.get("id") != binding.sample_id:
        errors.append("manifest id disagrees with the binding")
    condition = manifest.get("condition", {})
    if (
        condition.get("family") != binding.family
        or condition.get("family_id") != binding.family_id
        or condition.get("subtype_id") != binding.subtype_id
        or condition.get("role_id") != binding.role_id
        or condition.get("corpus_seed") != binding.corpus_seed
    ):
        errors.append("manifest condition disagrees with the binding")
    source = manifest.get("source", {})
    expected_source = {
        "raw_fields_sha256": aligned_fields_hash(
            binding.part_owner, binding.material, binding.emission_level
        ),
        "guide_sha256": array_hash("conditioning_guide", binding.guide),
        "genes_sha256": (
            array_hash("condition_genes", binding.genes)
            if binding.genes is not None
            else None
        ),
        "legal_tuples_sha256": tuple_fingerprint(binding.legal_tuples),
        "anatomy_sha256": _anatomy_hash(binding.anatomy),
        "anatomy_source": binding.anatomy.source,
        "anatomy_source_sha256": binding.anatomy.source_sha256,
        "driver_index_sha256": array_hash("driver_index", binding.driver_index),
        "binder_source_sha256": binder_source_hash(),
    }
    for name, expected in expected_source.items():
        if source.get(name) != expected:
            errors.append(f"manifest source.{name} is incorrect")
    if source.get("upstream_hashes") != dict(sorted(binding.upstream_hashes.items())):
        errors.append("manifest upstream hashes disagree with the binding")
    if source.get("pixel_authority") != "raw_neural_aligned_fields":
        errors.append("manifest does not retain neural pixel authority")
    if source.get("procedural_pixel_substitution") is not False:
        errors.append("manifest does not prohibit procedural pixel substitution")

    hashes = manifest.get("hashes", {})
    payload = dict(manifest)
    payload.pop("hashes", None)
    if hashes.get("binding_sha256") != canonical_json_hash(payload):
        errors.append("manifest binding SHA-256 is incorrect")
    errors.extend(_graph_errors(manifest))

    if derived_valid and identity_valid and upstream_valid and legal_valid:
        expected_manifest = _manifest(
            sample_id=binding.sample_id,
            family_name=binding.family,
            family_id=int(binding.family_id),
            subtype_id=int(binding.subtype_id),
            role_id=int(binding.role_id),
            corpus_seed=(
                int(binding.corpus_seed) if binding.corpus_seed is not None else None
            ),
            part=binding.part_owner,
            material=binding.material,
            emission=binding.emission_level,
            guide=binding.guide,
            genes=binding.genes,
            legal_tuples=binding.legal_tuples,
            anatomy=binding.anatomy,
            joints=binding.joints,
            sockets=binding.sockets,
            owner_layers=binding.owner_layers,
            driver_index=binding.driver_index,
            upstream_hashes=binding.upstream_hashes,
        )
        if dict(manifest) != expected_manifest:
            errors.append("manifest is not the exact deterministic binding projection")
    try:
        json.dumps(manifest, allow_nan=False)
    except (TypeError, ValueError) as error:
        errors.append(f"manifest is not strict JSON: {error}")
    if schema:
        errors.extend(f"schema: {error}" for error in validate_binding_schema(manifest))
    return errors


def validate_binding(binding: NeuralRigBinding, *, schema: bool = True) -> list[str]:
    """Return diagnostics for hostile values without leaking indexing errors."""
    try:
        return _validate_binding_impl(binding, schema=schema)
    except (
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        RecursionError,
    ) as error:
        return [f"malformed binding could not be validated safely: {type(error).__name__}: {error}"]


def assert_valid_binding(binding: NeuralRigBinding, *, schema: bool = True) -> None:
    errors = validate_binding(binding, schema=schema)
    if errors:
        raise BindingRejected(errors)
