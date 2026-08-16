from __future__ import annotations

import numpy as np
import torch

from forge.world_action_sparse_v5 import ModelConfig, SparseActionDiT, SparseWorldActionRuntime, spatial_control_fields
from forge.world_action_sparse_v5.training import counterfactual_control, latent_edit_mask


def test_sparse_action_model_is_exact_copy_at_zero_initialized_delta():
    model = SparseActionDiT(ModelConfig(width=64, layers=1, heads=4, patch=4)).eval()
    runtime = SparseWorldActionRuntime(model, torch.device("cpu"), {}, torch.zeros(1, 48, 1, 1), torch.ones(1, 48, 1, 1))
    current = torch.randn(2, 48, 32, 32)
    result, gate = runtime.predict_latent(current, action=np.asarray([2, 6]), control=np.asarray([[0, 0, .4, -.2], [1, 0, -.7, .5]], np.float32), state=np.zeros((2, 64), np.float32), return_gate=True)
    assert torch.equal(result, current)
    assert gate.shape == (2, 1, 32, 32)
    assert float(gate.max()) < 0.02


def test_edit_mask_tracks_and_dilates_changed_pixels():
    current = np.zeros((2, 256, 256, 3), np.uint8)
    target = current.copy()
    target[0, 120:126, 180:186] = 255
    mask = latent_edit_mask(current, target)
    assert mask.shape == (2, 1, 32, 32)
    assert mask.dtype == np.float16
    assert float(mask[0].sum()) >= 9
    assert float(mask[1].sum()) == 0
    assert mask[0, 0, 15, 22] == 1


def test_counterfactual_control_rotates_aim_and_displaces_local_actions():
    control = np.asarray([[.2, -.4, .5, .25], [0, 0, 0, 0]], np.float32)
    wrong = counterfactual_control(control)
    assert np.allclose(wrong[0], [.2 * -1, -.4 * -1, -.25, .5])
    assert np.allclose(wrong[1, 2:], [.7, -.55])


def test_sparse_spatial_fields_have_five_camera_aligned_channels():
    control = torch.tensor([[0.0, 1.0, 0.75, -0.5]], dtype=torch.float32)
    fields = spatial_control_fields(control, 32)
    assert fields.shape == (1, 5, 32, 32)
    actor = divmod(int(fields[0, 0].argmax()), 32)
    aim = divmod(int(fields[0, 1].argmax()), 32)
    assert abs(actor[0] - 16) <= 1 and abs(actor[1] - 16) <= 1
    assert aim[0] < actor[0] and aim[1] > actor[1]
    assert float(fields[0, 2, actor[0], actor[1]]) > .85
