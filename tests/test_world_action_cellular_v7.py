from __future__ import annotations

import numpy as np
import torch

from forge.action_teacher_v1.contract import ACTIONS
from forge.action_teacher_v2.actor import ACTOR_FEATURE_NAMES
from forge.action_teacher_v2.contract import ACTOR_FEATURES, ACTOR_FIELD_SHAPE
from forge.world_action_cellular_v7 import CellularTemporalActionDiT, ModelConfig, RecoveryCheckpointStore, TrainingConfig, align_temporal_cellular, load_encoded_corpus, load_recovery_checkpoint, load_v5_latent_editor, selection_score, source_sha256, validate_encoded_corpus, write_encoded_corpus
from forge.world_action_cellular_v7.contract import CHECKPOINT_FORMAT
from forge.world_action_cellular_v7.training import ResourceGuard
from forge.world_action_sparse_v5.contract import ModelConfig as V5ModelConfig
from forge.world_action_sparse_v5.model import SparseActionDiT
from forge.world_latent_dit.contract import LATENT_CHANNELS, LATENT_SIZE


def test_temporal_alignment_keeps_action_and_following_settle():
    count = 8
    latent = np.arange(count, dtype=np.float32)[:, None, None, None] * np.ones((count, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE), np.float32)
    action = np.asarray((0, 0, 3, 0, 0, 5, 0, 0), np.uint8)
    raw = {
        "frame": np.zeros((count, 256, 256, 3), np.uint8),
        "state": np.zeros((count, 64), np.float32),
        "actor_state": np.arange(count, dtype=np.float32)[:, None] * np.ones((count, ACTOR_FEATURES), np.float32),
        "actor_field": np.zeros((count, *ACTOR_FIELD_SHAPE), np.float16),
        "control": np.zeros((count, 4), np.float32),
        "action": action,
        "tick": np.arange(count, dtype=np.int64),
    }
    episode = align_temporal_cellular(latent, raw)
    assert episode["action"].tolist() == [3, 0, 5, 0]
    assert episode["previous_action"].tolist() == [0, 3, 0, 5]
    assert episode["previous"][:, 0, 0, 0].tolist() == [0, 1, 3, 4]
    assert episode["current"][:, 0, 0, 0].tolist() == [1, 2, 4, 5]
    assert episode["target"][:, 0, 0, 0].tolist() == [2, 3, 5, 6]
    assert episode["actor_state"][:, 0].tolist() == [1, 2, 4, 5]
    assert episode["target_actor_state"][:, 0].tolist() == [2, 3, 5, 6]


def test_v5_warm_start_preserves_exact_latent_editor_and_cellular_persistence():
    torch.manual_seed(17)
    parent = SparseActionDiT(V5ModelConfig(width=64, layers=2, heads=4, patch=4))
    model = CellularTemporalActionDiT(ModelConfig(width=64, layers=2, heads=4, patch=4))
    missing = load_v5_latent_editor(model, parent)
    assert missing and all("actor" in name or "previous" in name or "velocity" in name for name in missing)
    batch = 2
    current = torch.randn(batch, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)
    previous = torch.randn_like(current)
    time = torch.zeros(batch)
    action = torch.tensor((2, 8))
    control = torch.randn(batch, 4)
    state = torch.randn(batch, 64)
    actor_state = torch.rand(batch, ACTOR_FEATURES)
    actor_field = torch.rand(batch, *ACTOR_FIELD_SHAPE)
    previous_action = torch.tensor((1, 2))
    previous_control = torch.randn(batch, 4)
    with torch.inference_mode():
        expected = parent.edit(current, time, action, control, state)[0]
        predicted, next_state, next_field, *_ = model.edit(current, previous, time, action, control, state, actor_state, actor_field, previous_action, previous_control)
    assert torch.equal(predicted, expected)
    assert torch.equal(next_state, actor_state)
    assert torch.equal(next_field, actor_field)


def test_cellular_outputs_have_trainable_gradients():
    model = CellularTemporalActionDiT(ModelConfig(width=32, layers=1, heads=4, patch=4))
    batch = 1
    current = torch.randn(batch, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)
    actor_state = torch.rand(batch, ACTOR_FEATURES)
    actor_field = torch.rand(batch, *ACTOR_FIELD_SHAPE)
    output = model.edit(current, current, torch.zeros(batch), torch.zeros(batch, dtype=torch.long), torch.zeros(batch, 4), torch.zeros(batch, 64), actor_state, actor_field, torch.zeros(batch, dtype=torch.long), torch.zeros(batch, 4))
    loss = output[1].mean() + output[2].mean()
    loss.backward()
    assert model.actor_state_out.weight.grad is not None
    assert model.actor_state_gate_out.weight.grad is not None
    assert model.actor_field_out.weight.grad is not None
    assert model.actor_field_gate_out.weight.grad is not None


def test_cellular_edit_preserves_semantic_authority_outside_topology_actions():
    model = CellularTemporalActionDiT(ModelConfig(width=32, layers=1, heads=4, patch=4))
    with torch.no_grad():
        model.actor_state_out.bias.fill_(0.25)
        model.actor_field_out.bias.fill_(0.25)
    current = torch.zeros(1, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)
    actor_state = torch.zeros(1, ACTOR_FEATURES)
    actor_field = torch.zeros(1, *ACTOR_FIELD_SHAPE)
    actor_field[:, 0, 10:12, 10:12] = 1
    actor_field[:, 1, 10:12, 10:12] = 0.5
    common = (current, current, torch.zeros(1), None, torch.zeros(1, 4), torch.zeros(1, 64), actor_state, actor_field, torch.zeros(1, dtype=torch.long), torch.zeros(1, 4))
    inspect = torch.tensor((ACTIONS.index("inspect"),))
    _, next_actor, next_field, *_ = model.edit(*common[:3], inspect, *common[4:])
    immutable = torch.tensor([name.startswith(("family_", "stage_", "family_mix_", "development_", "ecology_", "diet_")) for name in ACTOR_FEATURE_NAMES])
    assert torch.equal(next_actor[:, immutable], actor_state[:, immutable])
    assert torch.equal(next_field[:, (0, 5, 6, 7)], actor_field[:, (0, 5, 6, 7)])
    assert torch.equal(next_field[:, 1:5, :10], actor_field[:, 1:5, :10])
    assert float(next_field[:, 1, 10:12, 10:12].mean()) > 0.5
    cut = torch.tensor((ACTIONS.index("cut"),))
    _, _, cut_field, *_ = model.edit(*common[:3], cut, *common[4:])
    assert bool((cut_field[:, (0, 5, 6, 7), 10:12, 10:12] != actor_field[:, (0, 5, 6, 7), 10:12, 10:12]).any())
    assert torch.equal(cut_field[:, :, :10], actor_field[:, :, :10])
    graft = torch.tensor((ACTIONS.index("graft_organ"),))
    _, graft_actor, graft_field, *_ = model.edit(*common[:3], graft, *common[4:])
    assert bool((graft_actor[:, immutable] != actor_state[:, immutable]).any())
    assert bool((graft_field[:, (0, 5, 6, 7)] != actor_field[:, (0, 5, 6, 7)]).any())


def test_recovery_checkpoint_is_atomic_and_source_bound(tmp_path):
    store = RecoveryCheckpointStore(tmp_path)
    payload = {"format": CHECKPOINT_FORMAT, "source_sha256": source_sha256(), "corpus_sha256": "a" * 64, "step": 500, "model": {"weight": torch.ones(2)}}
    record = store.save(payload, step=500, milestone=True)
    restored = load_recovery_checkpoint(tmp_path / "latest.pt", corpus_sha256="a" * 64)
    assert restored["step"] == 500
    assert record["milestone"].endswith("step-00000500.pt")
    assert (tmp_path / "milestones" / "step-00000500.pt").read_bytes() == (tmp_path / "latest.pt").read_bytes()


def test_encoded_corpus_roundtrips_and_rejects_tamper(tmp_path):
    count = 4
    episode = {
        "previous": np.zeros((count, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE), np.float32),
        "current": np.ones((count, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE), np.float32),
        "target": np.full((count, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE), 2, np.float32),
        "previous_control": np.zeros((count, 4), np.float32),
        "control": np.ones((count, 4), np.float32),
        "previous_action": np.zeros(count, np.uint8),
        "action": np.ones(count, np.uint8),
        "state": np.zeros((count, 64), np.float32),
        "actor_state": np.zeros((count, ACTOR_FEATURES), np.float32),
        "target_actor_state": np.ones((count, ACTOR_FEATURES), np.float32),
        "actor_field": np.zeros((count, *ACTOR_FIELD_SHAPE), np.float16),
        "target_actor_field": np.ones((count, *ACTOR_FIELD_SHAPE), np.float16),
        "current_frame": np.zeros((count, 256, 256, 3), np.uint8),
        "target_frame": np.ones((count, 256, 256, 3), np.uint8),
        "current_tick": np.arange(count, dtype=np.int64),
        "target_tick": np.arange(1, count + 1, dtype=np.int64),
    }
    source = {"session_id": "synthetic-a", "manifest_sha256": "1" * 64, "arrays_sha256": "2" * 64}
    root = tmp_path / "corpus"
    manifest = write_encoded_corpus(root, (episode,), (source,), vae_checkpoint_sha256="3" * 64, vae_ema_sha256="4" * 64)
    loaded, replay = load_encoded_corpus(root)
    assert manifest["pairs"] == replay["pairs"] == count
    assert np.array_equal(loaded[0]["target_actor_field"], episode["target_actor_field"])
    artifact = root / manifest["shards"][0]["artifact"]["path"]
    data = bytearray(artifact.read_bytes()); data[-1] ^= 1; artifact.write_bytes(data)
    try:
        validate_encoded_corpus(root)
    except ValueError as error:
        assert "artifact drifted" in str(error)
    else:
        raise AssertionError("tampered encoded corpus was accepted")


def test_selection_requires_visual_and_physiological_improvement():
    good = {"latent_mae": .08, "latent_persistence_mae": .1, "changed_latent_mae": .18, "changed_latent_persistence_mae": .24, "actor_state_mae": .03, "actor_state_persistence_mae": .05, "changed_actor_field_mae": .12, "changed_actor_field_persistence_mae": .2, "correct_action_advantage": .02, "targeted_control_advantage": .01}
    bad_physiology = {**good, "actor_state_mae": .07, "changed_actor_field_mae": .27}
    inverted = {**good, "correct_action_advantage": -.03, "targeted_control_advantage": -.02}
    assert selection_score(good) < selection_score(bad_physiology)
    assert selection_score(good) < selection_score(inverted)


def test_resource_guard_rejects_unsafe_limits_and_reports_cpu_usage():
    try:
        ResourceGuard(TrainingConfig(cuda_memory_fraction=1.0), torch.device("cpu"))
    except ValueError as error:
        assert "CUDA memory fraction" in str(error)
    else:
        raise AssertionError("unsafe CUDA memory fraction was accepted")
    guard = ResourceGuard(TrainingConfig(cpu_threads=8, max_process_memory_gib=32.0), torch.device("cpu"))
    report = guard.report()
    assert report["cpu_threads"] == 8
    assert report["maximum_rss_bytes"] > 0
    assert report["maximum_cuda_allocated_bytes"] == 0
