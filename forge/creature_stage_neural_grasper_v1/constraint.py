from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class GraspBody:
    position: np.ndarray
    velocity: np.ndarray
    mass: float


@dataclass(slots=True)
class GraspConstraint:
    attached: bool = False
    target_kind: str = "none"
    target_id: int = -1
    strain: float = 0.0


def solve_grasp(body: GraspBody, target: GraspBody, *, effector: np.ndarray, engage: bool, force: float,
                brace: float, cohesion: float, state: GraspConstraint, delta: float,
                release_impulse: np.ndarray | None = None) -> dict[str, float | bool | tuple[float, float]]:
    if delta <= 0 or body.mass <= 0 or target.mass <= 0 or not all(np.isfinite(value).all() for value in (body.position, body.velocity, target.position, target.velocity, effector)):
        raise ValueError("grasp constraint input drifted")
    if not engage:
        throw = np.zeros(2, dtype=np.float64) if release_impulse is None else np.asarray(release_impulse, dtype=np.float64)
        if throw.shape != (2,) or not np.isfinite(throw).all() or float(np.linalg.norm(throw)) > 12:
            raise ValueError("grasp release impulse drifted")
        was_attached = state.attached
        ground_impulse = np.zeros(2, dtype=np.float64)
        if was_attached and float(np.linalg.norm(throw)) > 0:
            target.velocity += throw / target.mass
            recoil = throw * (1.0 - float(np.clip(brace, 0, 1)))
            body.velocity -= recoil / body.mass
            ground_impulse = -(throw - recoil)
        state.attached = False; state.strain = 0
        return {
            "attached": False, "torn": False, "impulse": float(np.linalg.norm(throw)) if was_attached else 0.0,
            "ground_impulse": tuple(map(float, ground_impulse)), "thrown": bool(was_attached and np.linalg.norm(throw) > 0),
        }
    delta_position = target.position - effector
    distance = float(np.linalg.norm(delta_position))
    if not state.attached and distance <= 1.25:
        state.attached = True
    if not state.attached:
        return {"attached": False, "torn": False, "impulse": 0.0, "ground_impulse": (0.0, 0.0), "thrown": False}
    direction = delta_position / max(distance, 1e-6)
    relative = float(np.dot(target.velocity - body.velocity, direction))
    impulse = float(np.clip(distance * (8 + 14 * force) + relative * 1.8, -4, 4) * delta)
    body_share = target.mass / (body.mass + target.mass)
    target_share = body.mass / (body.mass + target.mass)
    body.velocity += direction * impulse * body_share * (1 - .72 * brace)
    target.velocity -= direction * impulse * target_share
    state.strain = max(0.0, distance - .45) * force
    torn = state.strain > max(.05, cohesion)
    if torn:
        state.attached = False
    return {"attached": state.attached, "torn": torn, "impulse": abs(impulse), "ground_impulse": (0.0, 0.0), "thrown": False}
