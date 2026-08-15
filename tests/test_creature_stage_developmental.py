from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from forge.creature_stage_developmental import (
    FAMILIES,
    TISSUES,
    TRAITS,
    AppendageGene,
    develop,
    pose,
    review_genomes,
)
from forge.creature_stage_developmental.contract import source_sha256
from forge.creature_stage_developmental.review import publish_review, validate_review


def _connected(node_count: int, edges: np.ndarray) -> bool:
    adjacency = [[] for _ in range(node_count)]
    for left, right in edges:
        adjacency[int(left)].append(int(right))
        adjacency[int(right)].append(int(left))
    seen = {0}
    frontier = [0]
    while frontier:
        node = frontier.pop()
        for neighbor in adjacency[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                frontier.append(neighbor)
    return len(seen) == node_count


def test_review_genomes_cover_five_priors_and_grafts() -> None:
    genomes = review_genomes()
    assert len(genomes) == 10
    assert len({genome.genome_id for genome in genomes}) == 10
    assert [int(np.argmax(genome.family_mix)) for genome in genomes] == [value for value in range(5) for _ in range(2)]
    assert all(np.isclose(sum(genome.family_mix), 1.0) for genome in genomes)
    assert all(not genomes[index].parent_ids for index in range(0, 10, 2))
    assert all(len(genomes[index].parent_ids) == 2 for index in range(1, 10, 2))
    assert FAMILIES == ("humanoid", "animalian", "plantlike", "anomaly", "machine")


def test_appendage_pairs_are_reciprocal_and_anatomically_symmetric() -> None:
    for genome in review_genomes():
        lookup = {appendage.appendage_id: appendage for appendage in genome.appendages}
        for appendage in genome.appendages:
            if appendage.paired_with is None:
                continue
            partner = lookup[appendage.paired_with]
            assert partner.paired_with == appendage.appendage_id
            assert partner.kind == appendage.kind
            assert partner.root_component == appendage.root_component
            assert partner.side == -appendage.side
            assert np.isclose(partner.root_offset[0], -appendage.root_offset[0])
            assert np.isclose(partner.root_offset[1], appendage.root_offset[1])
    genome = review_genomes()[0]
    broken = replace(genome.appendages[0], paired_with=genome.appendages[2].appendage_id)
    with pytest.raises(ValueError, match="not reciprocal"):
        replace(genome, appendages=(broken,) + genome.appendages[1:])


def test_development_produces_complete_cellular_anatomy() -> None:
    for genome in review_genomes():
        organism = develop(genome)
        assert organism.cell_xy.dtype == np.int16
        assert organism.tissue.dtype == np.uint8
        assert organism.component_weights.dtype == np.float32
        assert organism.trait_fields.dtype == np.float32
        assert organism.cell_count >= 200
        assert organism.component_weights.shape == (organism.cell_count, len(genome.components))
        assert organism.trait_fields.shape == (organism.cell_count, len(TRAITS))
        assert np.allclose(organism.component_weights.sum(axis=1), 1.0, atol=2e-6)
        assert np.all((organism.trait_fields >= 0.0) & (organism.trait_fields <= 1.0))
        present = {TISSUES[int(index)] for index in np.unique(organism.tissue)}
        assert {"bone", "muscle", "neural", "sensor"}.issubset(present)
        assert len(organism.identity_sha256) == 64


def test_skeleton_is_connected_and_every_appendage_has_antagonistic_muscles() -> None:
    for genome in review_genomes():
        organism = develop(genome)
        assert _connected(len(organism.skeleton_nodes), organism.skeleton_edges)
        assert len(organism.skeleton_edge_appendage) == len(organism.skeleton_edges)
        assert len(organism.skeleton_edge_side) == len(organism.skeleton_edges)
        assert len(organism.muscles) == len(genome.appendages) * 2
        for appendage_index, appendage in enumerate(genome.appendages):
            edges = np.flatnonzero(organism.skeleton_edge_appendage == appendage_index)
            assert len(edges) == appendage.segments + 1
            muscle = organism.muscles[organism.muscles[:, 2] == appendage_index]
            assert muscle.shape == (2, 6)
            assert sorted(muscle[:, 3].tolist()) == [-1.0, 1.0]
            assert np.allclose(muscle[:, 4], muscle[0, 4])


def test_motion_is_deterministic_loop_closed_and_nontrivial() -> None:
    for genome in review_genomes():
        organism = develop(genome)
        first = pose(organism, 0.0)
        replay = pose(organism, 0.0)
        near_loop = pose(organism, 1.0 - 1e-6)
        quarter = pose(organism, .25)
        assert np.array_equal(first.nodes, replay.nodes)
        assert np.array_equal(first.cells, replay.cells)
        assert np.max(np.abs(first.cells - near_loop.cells)) <= .005
        assert float(np.linalg.norm(quarter.cells - organism.cell_xy, axis=1).mean()) >= .20
        assert quarter.muscle_activation.shape == (len(organism.muscles),)
        assert np.any(quarter.muscle_activation > 0.0)
        assert quarter.planted_contacts.shape == (len(genome.appendages),)
        assert np.any([pose(organism, phase).planted_contacts.any() for phase in np.linspace(0.0, .875, 8)])


def test_humanoid_vertical_orientation_remains_locked() -> None:
    for genome in review_genomes()[:2]:
        organism = develop(genome)
        lookup = {component.component_id: index for index, component in enumerate(genome.components)}
        head = lookup["head"]
        pelvis = lookup["pelvis"]
        for phase in np.linspace(0.0, .875, 8):
            motion = pose(organism, float(phase))
            assert motion.nodes[head, 1] < motion.nodes[pelvis, 1] - 6.0
            assert abs(float(motion.nodes[head, 0] - motion.nodes[pelvis, 0])) < 2.0


def test_phase_rejects_nonfinite_and_out_of_range() -> None:
    organism = develop(review_genomes()[0])
    for phase in (-.01, 1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="phase drifted"):
            pose(organism, phase)


def test_source_hash_is_stable_within_process() -> None:
    assert source_sha256() == source_sha256()
    assert len(source_sha256()) == 64


def test_review_bank_exact_replay_and_tamper_rejection(tmp_path) -> None:
    output = tmp_path / "review"
    built = publish_review(output)
    checked = validate_review(output)
    assert built["passed"] is True and checked["passed"] is True
    assert built["motion_semantic_sha256"] == checked["motion_semantic_sha256"]
    assert built["motion_frame_stream_sha256"] == checked["motion_frame_stream_sha256"]
    manifest_path = output / "review_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="ascii"))
    payload["motion"]["loop_seam_max_abs"] = 1.0
    manifest_path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii")
    with pytest.raises(ValueError):
        validate_review(output)
