from __future__ import annotations

import numpy as np


NORTH = 1
EAST = 2
SOUTH = 4
WEST = 8


def cardinal_match_mask(values: np.ndarray) -> np.ndarray:
    """Four-bit N/E/S/W same-value mask; out-of-bounds never matches."""
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError("Autotile inputs must be two-dimensional.")
    height, width = values.shape
    result = np.zeros((height, width), dtype=np.uint8)
    result[1:, :] |= (values[1:, :] == values[:-1, :]).astype(np.uint8) * NORTH
    result[:, :-1] |= (values[:, :-1] == values[:, 1:]).astype(np.uint8) * EAST
    result[:-1, :] |= (values[:-1, :] == values[1:, :]).astype(np.uint8) * SOUTH
    result[:, 1:] |= (values[:, 1:] == values[:, :-1]).astype(np.uint8) * WEST
    return result


def elevation_drop_mask(elevation: np.ndarray, walkability: np.ndarray) -> np.ndarray:
    """Bits mark cardinal faces where a walkable tile overlooks a lower neighbor."""
    elevation = np.asarray(elevation)
    walk = np.asarray(walkability).astype(bool)
    if elevation.ndim != 2 or elevation.shape != walk.shape:
        raise ValueError("Elevation and walkability must be aligned 2-D arrays.")
    height, width = elevation.shape
    result = np.zeros((height, width), dtype=np.uint8)
    current = elevation.astype(np.int16)
    result[1:, :] |= (
        walk[1:, :] & ((~walk[:-1, :]) | (current[1:, :] > current[:-1, :]))
    ).astype(np.uint8) * NORTH
    result[:, :-1] |= (
        walk[:, :-1] & ((~walk[:, 1:]) | (current[:, :-1] > current[:, 1:]))
    ).astype(np.uint8) * EAST
    result[:-1, :] |= (
        walk[:-1, :] & ((~walk[1:, :]) | (current[:-1, :] > current[1:, :]))
    ).astype(np.uint8) * SOUTH
    result[:, 1:] |= (
        walk[:, 1:] & ((~walk[:, :-1]) | (current[:, 1:] > current[:, :-1]))
    ).astype(np.uint8) * WEST
    return result

