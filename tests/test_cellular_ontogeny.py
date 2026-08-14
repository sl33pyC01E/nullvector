from __future__ import annotations

import numpy as np
import pytest

from forge.cellular_ontogeny.compiler import _build_files, _program, replay_bank, validate_bank
from forge.cellular_ontogeny.contract import DEFAULT_OUTPUT, DEFAULT_SOURCE
from forge.cellular_organism.compiler import _load_arrays


def _first_source():
    import json
    source = json.loads(DEFAULT_SOURCE.read_text(encoding="utf-8")); record = source["offspring"][0]
    arrays = _load_arrays(DEFAULT_SOURCE.parent / record["arrays"]["path"])
    return record, arrays


def test_program_is_connected_and_complete() -> None:
    record, arrays = _first_source(); program, stages, metrics = _program(arrays, record["organs"], int(record["lineage"]["seed"]))
    count = len(arrays["position_xy"])
    assert sorted(program["birth_order"].tolist()) == list(range(count))
    assert stages[-1]["cell_count"] == count and stages[-1]["bond_count"] == len(arrays["bond_ab"])
    assert metrics["adult_cell_count"] == count
    assert program["parent_cell"][metrics["root_cell"]] == -1
    assert np.count_nonzero(program["parent_cell"] >= 0) == count - 1


def test_all_stages_grow_monotonically() -> None:
    record, arrays = _first_source(); program, stages, _ = _program(arrays, record["organs"], int(record["lineage"]["seed"]))
    assert [stage["cell_count"] for stage in stages] == sorted(stage["cell_count"] for stage in stages)
    assert set(program["lineage_id"].tolist()) <= {1, 2, 3, 4, 5}
    assert np.all(np.diff(np.sort(program["differentiation_time"])) > 0)


def test_bank_build_is_byte_deterministic() -> None:
    first, first_manifest = _build_files(DEFAULT_SOURCE); second, second_manifest = _build_files(DEFAULT_SOURCE)
    assert first == second and first_manifest == second_manifest


def test_authoritative_bank_validates_and_replays() -> None:
    manifest = DEFAULT_OUTPUT / "cellular_ontogeny_manifest.json"
    if not manifest.is_file(): pytest.skip("ontogeny bank not built")
    assert validate_bank(manifest)["passed"]
    assert replay_bank(manifest)["exact_replay"]
