from __future__ import annotations

import numpy as np

from ..neural_city_layout_v1.contract import CLASSES, GRID_SIZE
from ..neural_city_layout_v1.teacher import CLASS_INDEX, compile_city_layout
from .contract import GrowthCondition


def growth_authority_mask(condition: GrowthCondition) -> np.ndarray:
    """Cells a single construction action may physically alter.

    The selected site gets a bounded construction envelope. A two-cell-wide
    Manhattan corridor connects it to the settlement center. This prevents a
    local build decision from repainting unrelated districts while leaving the
    neural model authoritative inside the reachable construction area.
    """
    site_x = int(round(condition.site[0] * (GRID_SIZE - 1))); site_y = int(round(condition.site[1] * (GRID_SIZE - 1))); center = GRID_SIZE // 2
    yy, xx = np.mgrid[:GRID_SIZE, :GRID_SIZE]; mask = (np.abs(xx - site_x) <= 11) & (np.abs(yy - site_y) <= 11)
    x0, x1 = sorted((center, site_x)); mask[max(0, center - 1):min(GRID_SIZE, center + 2), x0:x1 + 1] = True
    y0, y1 = sorted((center, site_y)); mask[y0:y1 + 1, max(0, site_x - 1):min(GRID_SIZE, site_x + 2)] = True
    return mask


def project_neural_growth(current: np.ndarray, proposal: np.ndarray, condition: GrowthCondition) -> tuple[np.ndarray, dict[str, object]]:
    if current.shape != (GRID_SIZE, GRID_SIZE) or proposal.shape != current.shape or current.dtype != np.uint8 or proposal.dtype != np.uint8:
        raise ValueError("Neural growth projection shape/dtype drifted.")
    if int(current.max()) >= len(CLASSES) or int(proposal.max()) >= len(CLASSES):
        raise ValueError("Neural growth projection vocabulary drifted.")
    if not condition.affordable():
        return current.copy(), {"affordable": False, "authority_cells": 0, "accepted_changes": 0, "rejected_changes": int((proposal != current).sum())}
    authority = growth_authority_mask(condition)
    difference = proposal != current
    # A growth tick may occupy empty construction cells, but it cannot repaint
    # material already committed by earlier ticks. Demolition/damage is a
    # separate physical action. Keeping those authorities separate prevents a
    # small local prediction error from eroding an entire district over a long
    # free-running rollout.
    writable = authority & (current == 0)
    accepted = difference & writable
    result = current.copy()
    result[accepted] = proposal[accepted]
    rejected = difference & ~writable
    return result, {
        "affordable": True,
        "authority_cells": int(authority.sum()),
        "writable_cells": int(writable.sum()),
        "accepted_changes": int(accepted.sum()),
        "rejected_changes": int(rejected.sum()),
        "preserved_existing_cells": int((authority & (current != 0)).sum()),
    }


def _toroidal_components(field: np.ndarray) -> list[list[tuple[int, int]]]:
    occupied = field != 0
    seen = np.zeros_like(occupied)
    components: list[list[tuple[int, int]]] = []
    for raw_y, raw_x in np.argwhere(occupied):
        y, x = int(raw_y), int(raw_x)
        if seen[y, x]:
            continue
        seen[y, x] = True
        stack = [(y, x)]
        component: list[tuple[int, int]] = []
        while stack:
            cy, cx = stack.pop(); component.append((cy, cx))
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = (cy + dy) % GRID_SIZE, (cx + dx) % GRID_SIZE
                if occupied[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True; stack.append((ny, nx))
        components.append(component)
    return components


def _wrapped_delta(source: int, target: int) -> int:
    direct = target - source
    wrapped = direct - GRID_SIZE if direct > 0 else direct + GRID_SIZE
    return wrapped if abs(wrapped) < abs(direct) else direct


def compile_growth_state(raw: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """Compile a neural city tick into one accessible toroidal settlement.

    The generic city compiler removes unsupported structural pixels. This
    growth compiler additionally joins every occupied island to the center
    network with the shortest wrapped Manhattan road, changing empty cells
    only. Neural rooms and materials remain authoritative.
    """
    result, base = compile_city_layout(raw)
    added_roads = 0
    center = GRID_SIZE // 2
    components = _toroidal_components(result)
    for component in components:
        if any(y == center and x == center for y, x in component):
            continue
        y, x = min(
            component,
            key=lambda point: abs(_wrapped_delta(point[0], center)) + abs(_wrapped_delta(point[1], center)),
        )
        dx = _wrapped_delta(x, center)
        for _ in range(abs(dx)):
            x = (x + (1 if dx > 0 else -1)) % GRID_SIZE
            if result[y, x] == 0:
                result[y, x] = CLASS_INDEX["road"]; added_roads += 1
        dy = _wrapped_delta(y, center)
        for _ in range(abs(dy)):
            y = (y + (1 if dy > 0 else -1)) % GRID_SIZE
            if result[y, x] == 0:
                result[y, x] = CLASS_INDEX["road"]; added_roads += 1
    return result, {
        "edited_cells": int(base["edited_cells"]) + added_roads,
        "base_edits": int(base["edited_cells"]),
        "network_bridge_cells": added_roads,
        "occupied_cells": int((result != 0).sum()),
        "toroidal_components": len(_toroidal_components(result)),
    }
