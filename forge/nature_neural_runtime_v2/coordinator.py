from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
import time

import numpy as np

from ..nature_counterfactual_nn import ACTIONS as COUNTERFACTUAL_ACTIONS
from ..nature_macro_nn.state import extract_global_state, extract_patch_state


@dataclass(frozen=True, slots=True)
class EnsembleTick:
    tick: int
    macro_applied: bool
    society_applied: bool
    timeline_applied: bool
    counterfactual_applied: bool
    resource_delta_l1: float
    semantic_sha256: str


class MultiRateNeuralCoordinator:
    """Couple promoted neural specialists to one authoritative nature world.

    The physical world remains the conservation/topology authority. Learned
    specialists author intent, movement, physiology, feeding, colony roles and
    society actions inside their existing adapters. This coordinator closes the
    formerly missing macro/timeline scheduling loop and records which neural
    outputs actually changed shared state.
    """

    def __init__(self, neural, world, society, *, world_hz: float = 15.0):
        if not np.isfinite(world_hz) or world_hz <= 0:
            raise ValueError("ensemble world cadence is invalid")
        self.neural = neural
        self.world = world
        self.society = society
        self.world_hz = float(world_hz)
        self.macro_interval = max(1, round(self.world_hz / 1.0))
        self.society_interval = max(1, round(self.world_hz / .05))
        self.timeline_interval = max(1, round(self.world_hz / .02))
        self.counterfactual_interval = self.timeline_interval
        current = extract_patch_state(world, society)
        global_state = extract_global_state(world, society)
        self.previous_patch = current.copy()
        self.current_patch = current
        self.previous_global = global_state.copy()
        self.current_global = global_state
        self.timeline_forecast = neural.timeline.observe(world, society)
        self.counterfactuals = {
            item.action: item for item in neural.counterfactual.evaluate(neural.timeline.history)
        }
        self.timings_ms: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=240))
        self.calls: dict[str, int] = defaultdict(int)
        self.macro_delta_l1 = 0.0
        self.last_tick: EnsembleTick | None = None

    def _time(self, stage: str, function):
        started = time.perf_counter()
        result = function()
        self.timings_ms[stage].append((time.perf_counter() - started) * 1000.0)
        self.calls[stage] += 1
        return result

    def step_world(self, delta: float, *, publish: bool = False):
        """Advance the shared physical state through neural local policies."""
        return self._time("local_world", lambda: self.world.step(delta, publish=publish))

    def step_physiology(self, function) -> None:
        """Time the demo's bounded living-body NCA adapter."""
        self._time("cell_physiology", function)

    def _apply_macro(self) -> float:
        observed_patch = extract_patch_state(self.world, self.society)
        observed_global = extract_global_state(self.world, self.society)
        predicted, predicted_global, _gate = self.neural.macro.step(
            observed_patch, self.previous_patch, observed_global, self.previous_global
        )
        predicted = np.asarray(predicted, np.float32)
        predicted_global = np.asarray(predicted_global, np.float32)
        if predicted.shape != observed_patch.shape or predicted_global.shape != observed_global.shape:
            raise ValueError("macro neural output shape drifted")
        if not np.isfinite(predicted).all() or not np.isfinite(predicted_global).all():
            raise FloatingPointError("macro neural output became non-finite")

        # Preserve cell-scale detail and conservation projections: the model
        # authors the learned 32x32 resource delta, which is lifted onto the
        # 64x64 authority grid instead of replacing it with blocky averages.
        resource_delta = np.repeat(np.repeat(predicted[:10] - observed_patch[:10], 2, -2), 2, -1)
        before = self.world.fields.copy()
        self.world.fields[:] = np.clip(self.world.fields + resource_delta, 0.0, 1.0)
        magnitude = float(np.mean(np.abs(self.world.fields - before)))
        self.previous_patch, self.current_patch = observed_patch, predicted
        self.previous_global, self.current_global = observed_global, predicted_global
        return magnitude

    def after_world_step(self) -> EnsembleTick:
        tick = int(self.world.tick_index)
        macro = society = timeline = counterfactual = False
        delta_l1 = 0.0
        if tick % self.macro_interval == 0:
            delta_l1 = self._time("macro_patch", self._apply_macro)
            self.macro_delta_l1 += delta_l1
            macro = True
        if tick % self.society_interval == 0:
            self._time("society", lambda: self.society.step_history(1))
            society = True
        if tick % self.timeline_interval == 0:
            self.timeline_forecast = self._time(
                "timeline", lambda: self.neural.timeline.observe(self.world, self.society)
            )
            timeline = True
        if tick % self.counterfactual_interval == 0:
            values = self._time(
                "counterfactual",
                lambda: self.neural.counterfactual.evaluate(
                    self.neural.timeline.history, COUNTERFACTUAL_ACTIONS
                ),
            )
            self.counterfactuals = {item.action: item for item in values}
            counterfactual = True
        digest = hashlib.sha256()
        digest.update(self.world.snapshot().semantic_sha256.encode())
        digest.update(self.society.semantic_sha256().encode())
        digest.update(np.ascontiguousarray(self.current_patch, dtype="<f4").tobytes())
        event = EnsembleTick(
            tick, macro, society, timeline, counterfactual, delta_l1, digest.hexdigest()
        )
        self.last_tick = event
        return event

    def run_slow_cycle(self) -> EnsembleTick:
        """Run one real slow-scale cycle without waiting 50 wall-clock seconds.

        This is used by bounded evidence captures. It applies the same society,
        timeline and counterfactual functions as the cadence scheduler and does
        not fabricate or merely probe their outputs.
        """
        self._time("society", lambda: self.society.step_history(1))
        self.timeline_forecast = self._time(
            "timeline", lambda: self.neural.timeline.observe(self.world, self.society)
        )
        values = self._time(
            "counterfactual",
            lambda: self.neural.counterfactual.evaluate(
                self.neural.timeline.history, COUNTERFACTUAL_ACTIONS
            ),
        )
        self.counterfactuals = {item.action: item for item in values}
        digest = hashlib.sha256()
        digest.update(self.world.snapshot().semantic_sha256.encode())
        digest.update(self.society.semantic_sha256().encode())
        event = EnsembleTick(
            int(self.world.tick_index), False, True, True, True, 0.0, digest.hexdigest()
        )
        self.last_tick = event
        return event

    def timing_summary(self) -> dict[str, dict[str, float | int]]:
        summary = {}
        for stage in sorted(self.timings_ms):
            values = np.asarray(self.timings_ms[stage], np.float64)
            summary[stage] = {
                "calls": int(self.calls[stage]),
                "mean_ms": float(values.mean()),
                "p95_ms": float(np.quantile(values, .95)),
                "max_ms": float(values.max()),
            }
        return summary
