from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from forge.cellular_ecology.compiler import _build_files, _compile_map, _discover_maps, _load_npz, replay_bank, validate_bank
from forge.cellular_ecology.contract import DEFAULT_MAP_ROOT, DEFAULT_OUTPUT, FAMILIES, FIELD_NAMES
from forge.maps.io import load_map_pack


def test_authoritative_bank_validates_and_replays() -> None:
    manifest = DEFAULT_OUTPUT / "cellular_ecology_manifest.json"
    if not manifest.is_file(): pytest.skip("ecology bank not built yet")
    assert validate_bank(manifest)["passed"]
    assert replay_bank(manifest)["exact_replay"]


def test_all_six_maps_compile_with_legal_resources() -> None:
    packs = _discover_maps(DEFAULT_MAP_ROOT)
    assert [load_map_pack(pack).theme for pack in packs] == ["arena", "rooms", "caves", "archipelago", "garden", "anomaly"]
    for pack in packs:
        data = load_map_pack(pack); fields, record = _compile_map(pack)
        assert all(record["gates"].values())
        assert fields["family_suitability"].shape == (5, *data.shape)
        assert fields["resource_type"].dtype == np.uint8
        for node in record["resource_nodes"]:
            x, y = node["position"]
            assert data.decoration_forbidden[y, x] == 0
            assert data.walkability[y, x] != 0 and data.hazard[y, x] == 0


def test_family_niches_and_field_bounds_are_nontrivial() -> None:
    for pack in _discover_maps(DEFAULT_MAP_ROOT):
        fields, record = _compile_map(pack)
        for name in FIELD_NAMES:
            assert fields[name].dtype == np.float32
            assert np.isfinite(fields[name]).all()
            assert 0.0 <= float(fields[name].min()) <= float(fields[name].max()) <= 1.0
        counts = {family: 0 for family in FAMILIES}
        for node in record["resource_nodes"]: counts[node["family"]] += 1
        assert min(counts.values()) >= 3
        assert np.unique(fields["family_suitability"], axis=0).shape[0] == 5


def test_build_is_byte_deterministic() -> None:
    first, first_manifest = _build_files(DEFAULT_MAP_ROOT)
    second, second_manifest = _build_files(DEFAULT_MAP_ROOT)
    assert first == second
    assert first_manifest == second_manifest


def test_archive_loader_rejects_extra_member(tmp_path: Path) -> None:
    source, _ = _compile_map(_discover_maps(DEFAULT_MAP_ROOT)[0])
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, format=np.asarray(["nullvector-cellular-ecology-fields-v1"]), **source, junk=np.asarray([1]))
    with pytest.raises(ValueError, match="member contract"):
        _load_npz(path)


def test_manifest_tamper_fails(tmp_path: Path) -> None:
    manifest_path = DEFAULT_OUTPUT / "cellular_ecology_manifest.json"
    if not manifest_path.is_file(): pytest.skip("ecology bank not built yet")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")); manifest["map_count"] = 5
    bad = tmp_path / "manifest.json"; bad.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError): validate_bank(bad)
