from __future__ import annotations

import numpy as np

from ..map_art import render_map_art


def _screen(base: np.ndarray, light: np.ndarray) -> np.ndarray:
    left = base.astype(np.float32) / 255.0
    right = light.astype(np.float32) / 255.0
    return np.clip((1.0 - (1.0 - left) * (1.0 - right)) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _soft_emission(emission: np.ndarray) -> np.ndarray:
    """Small deterministic bloom used only to author full-viewport VAE targets."""
    work = emission.astype(np.uint16)
    near = work.copy()
    far = work.copy()
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        near += np.roll(np.roll(work, dy, axis=0), dx, axis=1)
    near //= 5
    for dy, dx in ((-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1)):
        far += np.roll(np.roll(work, dy, axis=0), dx, axis=1)
    far //= 9
    return np.clip(emission.astype(np.uint16) + near // 2 + far // 4, 0, 255).astype(np.uint8)


def render_teacher_map_frames(topology) -> np.ndarray:
    """Return eight authoritative map-art frames for aesthetic teacher capture.

    These pixels are supervision only. Deployment receives the same topology as
    numeric tensors and produces the complete view through the viewport VAE.
    """
    layers = render_map_art(topology)
    frames: list[np.ndarray] = []
    for index in range(layers.hazard_color_frames.shape[0]):
        base = layers.base_color.copy()
        hazard = layers.hazard_color_frames[index]
        alpha = hazard[..., 3:4].astype(np.float32) / 255.0
        base = np.clip(hazard[..., :3] * alpha + base * (1.0 - alpha), 0, 255).astype(np.uint8)
        emission = np.maximum(layers.emissive, layers.hazard_emissive_frames[index])
        frames.append(_screen(base, _soft_emission(emission)))
    return np.stack(frames).astype(np.uint8, copy=False)
