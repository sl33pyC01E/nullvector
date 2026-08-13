from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from forge.map_quality.audit import (
    _articulation_points,
    _clearance_radius,
    _shortest_path,
    _shortest_path_optima,
    assert_exact_audit_replay,
    assert_valid_audit_report,
    audit_map,
    audit_pack,
    audit_packs,
    write_audit_report,
)
from forge.map_quality.render import (
    assert_exact_quality_showcase,
    render_quality_contact_sheet,
    render_quality_overlay,
    write_quality_showcase,
)
from forge.maps.generator import generate_map
from forge.maps.io import write_map_pack
from forge.maps.model import MapConfig, THEMES


def _generated(seed: int = 0x5155414C495459) -> object:
    return generate_map(
        seed,
        "rooms",
        MapConfig(width=48, height=44, objective_count=3, spawn_count=8),
    )


def test_shortest_path_rejects_out_of_bounds_and_blocked_endpoints() -> None:
    mask = np.ones((5, 6), dtype=bool)
    mask[2, 3] = False
    assert _shortest_path(mask, (-1, 0), (5, 4)) == ()
    assert _shortest_path(mask, (0, 0), (6, 4)) == ()
    assert _shortest_path(mask, (3, 2), (5, 4)) == ()
    assert _shortest_path(mask, (0, 0), (3, 2)) == ()


def test_clearance_uses_square_agent_chebyshev_distance() -> None:
    walkable = np.ones((7, 7), dtype=bool)
    walkable[0, :] = False
    walkable[-1, :] = False
    walkable[:, 0] = False
    walkable[:, -1] = False
    walkable[2, 2] = False
    clearance = _clearance_radius(walkable)
    assert clearance[3, 3] == 1
    assert clearance[4, 4] == 2
    assert clearance[3, 4] == 2


def test_required_target_articulation_ignores_optional_dead_end() -> None:
    # Horizontal mission corridor y=2; an optional dead-end hangs upward at x=3.
    mask = np.zeros((5, 7), dtype=bool)
    mask[2, 1:6] = True
    mask[1, 3] = True
    component = mask.copy()
    all_points, mission = _articulation_points(mask, component, (1, 2), ((5, 2),))
    assert 3 * 0 + 0 not in all_points  # exercise set semantics, not a count proxy
    assert 2 * 7 + 3 in all_points
    assert 2 * 7 + 3 in mission

    # With the required terminal before the spur, removing the spur junction no
    # longer disconnects a mission target, though it remains an articulation.
    _, mission_before_spur = _articulation_points(
        mask, component, (1, 2), ((2, 2),)
    )
    assert 2 * 7 + 3 not in mission_before_spur


def test_shortest_path_optima_are_independent_of_neighbor_order() -> None:
    mask = np.ones((5, 7), dtype=bool)
    mask[0, :] = False
    mask[-1, :] = False
    mask[:, 0] = False
    mask[:, -1] = False
    clearance = np.ones_like(mask, dtype=np.int32)
    clearance[2, 1:6] = 2
    cost = np.zeros_like(mask, dtype=np.int32)
    cost[1, 3] = 1
    result = _shortest_path_optima(mask, (1, 2), (5, 2), clearance, cost)
    assert result == {
        "length": 4,
        "widest_clearance_radius": 2,
        "minimum_cell_cost": 0,
    }


def test_target_aware_articulation_matches_removal_bfs_oracle() -> None:
    rng = np.random.Generator(np.random.PCG64(0x4152544943554C41))
    for _ in range(120):
        mask = rng.random((6, 7)) < 0.68
        points = np.argwhere(mask)
        if len(points) < 3:
            continue
        sy, sx = map(int, points[0])
        start = (sx, sy)
        from forge.map_quality.audit import _distances

        component = _distances(mask, (start,)) >= 0
        members = np.argwhere(component)
        if len(members) < 3:
            continue
        targets = tuple(
            (int(x), int(y)) for y, x in members[-min(3, len(members) - 1) :]
        )
        _, observed = _articulation_points(mask, component, start, targets)
        expected: set[int] = set()
        height, width = mask.shape
        terminal_indices = {y * width + x for x, y in (start, *targets)}
        for y_value, x_value in members:
            y = int(y_value)
            x = int(x_value)
            index = y * width + x
            if index in terminal_indices:
                continue
            removed = mask.copy()
            removed[y, x] = False
            distances = _distances(removed, (start,))
            if any(distances[ty, tx] < 0 for tx, ty in targets):
                expected.add(index)
        assert observed == expected


def test_clearance_matches_brute_force_chebyshev_distance() -> None:
    rng = np.random.Generator(np.random.PCG64(0x434C454152414E43))
    for _ in range(80):
        mask = rng.random((8, 9)) < 0.72
        mask[0, :] = False
        mask[-1, :] = False
        mask[:, 0] = False
        mask[:, -1] = False
        observed = _clearance_radius(mask)
        blocked = np.argwhere(~mask)
        for y, x in np.argwhere(mask):
            expected = min(
                max(abs(int(y) - int(by)), abs(int(x) - int(bx)))
                for by, bx in blocked
            )
            assert observed[y, x] == expected


@pytest.mark.parametrize("theme_index,theme", enumerate(THEMES))
def test_audit_is_deterministic_and_preserves_map_authority(
    theme_index: int, theme: str
) -> None:
    data = generate_map(
        0xA11D1700 + theme_index,
        theme,
        MapConfig(width=48, height=48, objective_count=3, spawn_count=8),
    )
    before = {name: array.copy() for name, array in data.arrays().items()}
    first = audit_map(data)
    second = audit_map(data)
    assert first == second
    assert first["hard_validity_preserved"]
    assert first["diagnostics"]["radius_one_safe_route_exists"]
    assert len(first["metrics"]["required_paths"]) == 4
    assert first["metrics"]["maximum_safe_detour_ratio_vs_geometric"] >= 1.0
    for name, expected in before.items():
        np.testing.assert_array_equal(data.arrays()[name], expected)


def test_pack_audit_replays_source_and_is_input_order_invariant(tmp_path: Path) -> None:
    paths = []
    for index, theme in enumerate(("arena", "garden")):
        data = generate_map(
            0x5041434B + index,
            theme,
            MapConfig(width=40, height=40, objective_count=2, spawn_count=5),
        )
        paths.append(write_map_pack(data, tmp_path / "packs", preview_scale=2))
    direct = [audit_pack(path) for path in paths]
    assert all(record["source_semantic_sha256"] for record in direct)
    assert all(record["source_manifest_sha256"] for record in direct)
    forward = audit_packs(paths)
    reverse = audit_packs(tuple(reversed(paths)))
    assert forward == reverse
    assert_valid_audit_report(forward)
    assert_exact_audit_replay(forward, paths)


def test_report_schema_hashes_and_derived_fields_fail_closed(tmp_path: Path) -> None:
    pack = write_map_pack(_generated(), tmp_path / "packs", preview_scale=2)
    report = audit_packs((pack,))
    output = tmp_path / "quality.json"
    write_audit_report(report, output)
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded == report

    for mutation in ("nested", "aggregate", "top"):
        tampered = deepcopy(report)
        if mutation == "nested":
            tampered["maps"][0]["metrics"]["zone_count"] += 1
        elif mutation == "aggregate":
            tampered["aggregate"]["elevation_levels_used"]["mean"] += 1
        else:
            tampered["unique_semantic_count"] += 1
        with pytest.raises(ValueError):
            assert_valid_audit_report(tampered)

    fully_rehashed = deepcopy(report)
    fully_rehashed["maps"][0]["metrics"]["zone_count"] += 1
    from forge.map_quality.audit import _canonical_hash

    nested = fully_rehashed["maps"][0]
    nested["report_sha256"] = _canonical_hash(
        {key: value for key, value in nested.items() if key != "report_sha256"}
    )
    fully_rehashed["report_sha256"] = _canonical_hash(
        {
            key: value
            for key, value in fully_rehashed.items()
            if key != "report_sha256"
        }
    )
    # The local checksum is deliberately not described as an authenticity
    # mechanism; exact pack replay is the authority against full rehashing.
    with pytest.raises(ValueError, match="exact source-pack replay"):
        assert_exact_audit_replay(fully_rehashed, (pack,))


def test_audit_pack_rejects_fully_rehashed_semantic_tamper(tmp_path: Path) -> None:
    pack = write_map_pack(_generated(), tmp_path / "packs", preview_scale=2)
    arrays_path = pack / "semantics.npz"
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    y, x = np.argwhere(arrays["walkability"] != 0)[0]
    arrays["elevation"][y, x] = np.int8((int(arrays["elevation"][y, x]) + 1) % 4)
    # A normal rewrite is sufficient: authoritative pack validation must reject
    # the artifact hash before the quality layer attempts to score it.
    np.savez_compressed(arrays_path, **arrays)
    with pytest.raises(ValueError, match="authoritative validation"):
        audit_pack(pack)


def test_invalid_map_never_receives_quality_status() -> None:
    data = _generated()
    broken_walkability = data.walkability.copy()
    broken_walkability[data.start[1], data.start[0]] = 0
    malformed = replace(data, walkability=broken_walkability)
    with pytest.raises(Exception):
        audit_map(malformed)


def test_audit_map_never_trusts_bogus_semantic_metadata() -> None:
    data = _generated()
    data.metadata["semantic_array_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="metadata semantic-array hash"):
        audit_map(data)


def test_quality_showcase_is_deterministic_and_source_bound(tmp_path: Path) -> None:
    packs = []
    for index, theme in enumerate(("arena", "anomaly")):
        data = generate_map(
            0x53484F5743415345 + index,
            theme,
            MapConfig(width=40, height=40, objective_count=2, spawn_count=5),
        )
        packs.append(write_map_pack(data, tmp_path / "packs", preview_scale=2))
    report = audit_packs(tuple(packs))
    first = render_quality_contact_sheet(tuple(packs), report, scale=3)
    second = render_quality_contact_sheet(tuple(reversed(packs)), report, scale=3)
    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
    overlay = render_quality_overlay(packs[0], scale=3)
    assert overlay.size == (120, 120)
    manifest_a = write_quality_showcase(tuple(packs), tmp_path / "showcase_a", scale=3)
    manifest_b = write_quality_showcase(
        tuple(reversed(packs)), tmp_path / "showcase_b", scale=3
    )
    assert manifest_a == manifest_b
    assert (
        (tmp_path / "showcase_a" / "quality_contact_sheet.png").read_bytes()
        == (tmp_path / "showcase_b" / "quality_contact_sheet.png").read_bytes()
    )
    assert_exact_quality_showcase(
        tmp_path / "showcase_a" / "showcase_manifest.json", tuple(packs)
    )
    tampered = bytearray(
        (tmp_path / "showcase_a" / "quality_contact_sheet.png").read_bytes()
    )
    tampered[-12] ^= 1
    (tmp_path / "showcase_a" / "quality_contact_sheet.png").write_bytes(tampered)
    with pytest.raises(ValueError, match="exact replay"):
        assert_exact_quality_showcase(
            tmp_path / "showcase_a" / "showcase_manifest.json", tuple(packs)
        )
