from __future__ import annotations

import numpy as np
import torch

from forge.action_teacher_v2.contract import ACTOR_FEATURES, ACTOR_FIELD_SHAPE
from forge.world_action_cellular_v7 import CellularTemporalActionDiT, ModelConfig, RecoveryCheckpointStore, align_temporal_cellular, load_recovery_checkpoint, load_v5_latent_editor, source_sha256
from forge.world_action_cellular_v7.contract import CHECKPOINT_FORMAT
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
        predicted, next_state, next_field, _, _, _ = model.edit(current, previous, time, action, control, state, actor_state, actor_field, previous_action, previous_control)
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
    assert model.actor_field_out.weight.grad is not None


def test_recovery_checkpoint_is_atomic_and_source_bound(tmp_path):
    store = RecoveryCheckpointStore(tmp_path)
    payload = {"format": CHECKPOINT_FORMAT, "source_sha256": source_sha256(), "corpus_sha256": "a" * 64, "step": 500, "model": {"weight": torch.ones(2)}}
    record = store.save(payload, step=500, milestone=True)
    restored = load_recovery_checkpoint(tmp_path / "latest.pt", corpus_sha256="a" * 64)
    assert restored["step"] == 500
    assert record["milestone"].endswith("step-00000500.pt")
    assert (tmp_path / "milestones" / "step-00000500.pt").read_bytes() == (tmp_path / "latest.pt").read_bytes()
