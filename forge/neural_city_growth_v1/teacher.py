from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from ..neural_city_layout_v1.contract import CLASSES, GRID_SIZE, CityCondition
from ..neural_city_layout_v1.teacher import CLASS_INDEX, _condition, _teacher_building
from .contract import ACTIONS, PATCH_SIZE, PURPOSE_COSTS, SITE_X_INDEX, SITE_Y_INDEX, GrowthCondition


@dataclass(frozen=True, slots=True)
class GrowthExample:
    identity: str
    seed: int
    current: np.ndarray
    target: np.ndarray
    condition: GrowthCondition
    changed: np.ndarray


def _origin(city: CityCondition, index: int) -> tuple[int, int]:
    center = GRID_SIZE // 2; radius = 7 + math.sqrt(index) * (12 + city.style[1] * 7)
    angle = index * math.pi * (3 - math.sqrt(5)) + city.style[0] * math.tau + city.family * .09
    return int(round(center + math.cos(angle) * radius)), int(round(center + math.sin(angle) * radius))


def extract_local_patch(field: np.ndarray, site: tuple[float, float]) -> np.ndarray:
    if field.shape != (GRID_SIZE, GRID_SIZE): raise ValueError("Growth patch source shape drifted.")
    center_x = int(round(site[0] * (GRID_SIZE - 1))); center_y = int(round(site[1] * (GRID_SIZE - 1))); radius = PATCH_SIZE // 2
    ys = (np.arange(center_y - radius, center_y - radius + PATCH_SIZE) % GRID_SIZE).astype(int); xs = (np.arange(center_x - radius, center_x - radius + PATCH_SIZE) % GRID_SIZE).astype(int)
    return np.ascontiguousarray(field[np.ix_(ys, xs)])


def paste_local_patch(field: np.ndarray, patch: np.ndarray, site: tuple[float, float]) -> np.ndarray:
    if field.shape != (GRID_SIZE, GRID_SIZE) or patch.shape != (PATCH_SIZE, PATCH_SIZE): raise ValueError("Growth patch paste shape drifted.")
    result = field.copy(); center_x = int(round(site[0] * (GRID_SIZE - 1))); center_y = int(round(site[1] * (GRID_SIZE - 1))); radius = PATCH_SIZE // 2
    ys = (np.arange(center_y - radius, center_y - radius + PATCH_SIZE) % GRID_SIZE).astype(int); xs = (np.arange(center_x - radius, center_x - radius + PATCH_SIZE) % GRID_SIZE).astype(int); result[np.ix_(ys, xs)] = patch; return result


def local_condition_vector(condition: GrowthCondition) -> np.ndarray:
    vector = condition.vector(); vector[SITE_X_INDEX] = .5; vector[SITE_Y_INDEX] = .5; return vector


def apply_teacher_growth(current: np.ndarray, condition: GrowthCondition) -> tuple[np.ndarray, dict[str, object]]:
    if current.shape != (GRID_SIZE, GRID_SIZE) or current.dtype != np.uint8 or int(current.max()) >= len(CLASSES):
        raise ValueError("Growth current city field drifted.")
    result = current.copy()
    if not condition.affordable():
        return result, {"affordable": False, "changed_cells": 0, "spent": (0.0,) * 4}
    index = condition.stage; origin = (int(round(condition.site[0] * (GRID_SIZE - 1))), int(round(condition.site[1] * (GRID_SIZE - 1))))
    building = _teacher_building(condition.city, index, origin, condition.action)
    door = next((x, y) for x, y, material in building if material == "door")
    center = GRID_SIZE // 2; x = center; y = center
    while x != door[0]:
        if result[y % GRID_SIZE, x % GRID_SIZE] == 0: result[y % GRID_SIZE, x % GRID_SIZE] = CLASS_INDEX["road"]
        x += 1 if door[0] > x else -1
    while y != door[1]:
        if result[y % GRID_SIZE, x % GRID_SIZE] == 0: result[y % GRID_SIZE, x % GRID_SIZE] = CLASS_INDEX["road"]
        y += 1 if door[1] > y else -1
    for x, y, material in building: result[y % GRID_SIZE, x % GRID_SIZE] = CLASS_INDEX[material]
    changed = int((result != current).sum())
    return result, {"affordable": True, "changed_cells": changed, "spent": PURPOSE_COSTS[condition.action]}


def _trajectory(seed: int, stage: int) -> tuple[np.ndarray, CityCondition, tuple[str, ...]]:
    # Building target is a persistent settlement intent, not a disguised stage
    # counter. Keeping the independently sampled target intact makes every
    # growth stage valid for small, medium, and large planned settlements and
    # matches the free-running runtime contract.
    rng = np.random.default_rng(seed ^ 0x5452414A454354); city = _condition(seed)
    actions = tuple(ACTIONS[int(rng.integers(len(ACTIONS)))] for _ in range(stage + 1)); current = np.zeros((GRID_SIZE, GRID_SIZE), np.uint8)
    abundant = (1.0, 1.0, 1.0, 1.0)
    for index, action in enumerate(actions[:-1]):
        origin = _origin(city, index); site = ((origin[0] % GRID_SIZE) / (GRID_SIZE - 1), (origin[1] % GRID_SIZE) / (GRID_SIZE - 1)); current, _ = apply_teacher_growth(current, GrowthCondition(city, action, abundant, site, index))
    return current, city, actions


def build_growth_corpus(count: int, *, seed: int) -> tuple[GrowthExample, ...]:
    if not 128 <= count <= 1_000_000: raise ValueError("Growth corpus size is outside contract.")
    examples = []
    for item in range(count):
        example_seed = int((seed + item * 0x9E3779B97F4A7C15) & ((1 << 63) - 1)); stage = item % 12
        current, city, actions = _trajectory(example_seed, stage); rng = np.random.default_rng(example_seed ^ 0x5245534F55524345)
        action = actions[-1]; cost = np.asarray(PURPOSE_COSTS[action], np.float32)
        affordable = item % 5 != 0
        resources = np.clip(cost + rng.uniform(.05, .45, 4), 0, 1) if affordable else np.clip(cost * rng.uniform(0, .85, 4), 0, 1)
        if not affordable:
            resource_index = int(rng.integers(4)); resources[resource_index] = max(0, cost[resource_index] * .25)
        origin = _origin(city, stage); site = ((origin[0] % GRID_SIZE) / (GRID_SIZE - 1), (origin[1] % GRID_SIZE) / (GRID_SIZE - 1))
        condition = GrowthCondition(city, action, tuple(map(float, resources)), site, stage)
        target, _ = apply_teacher_growth(current, condition); changed = target != current
        identity = hashlib.sha256(current.tobytes() + target.tobytes() + condition.vector().tobytes()).hexdigest()
        examples.append(GrowthExample(identity, example_seed, current, target, condition, changed))
    if len({item.identity for item in examples}) != count: raise ValueError("Growth corpus identities are not unique.")
    return tuple(examples)
