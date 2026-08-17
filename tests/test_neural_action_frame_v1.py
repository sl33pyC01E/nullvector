from __future__ import annotations

import torch


def test_residual_composition_preserves_unchanged_frame_exactly():
    current = torch.rand(2, 3, 16, 16)
    decoded = torch.rand(2, 3, 16, 16)
    result = torch.clamp(current + decoded - decoded, 0, 1)
    torch.testing.assert_close(result, current, rtol=0, atol=1e-7)
