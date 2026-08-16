from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..creature_stage_developmental.development import DevelopedOrganism
@dataclass(frozen=True, slots=True)
class LimbGeometry:
    kind: str
    segments: int
    length: float
    cell_count: int


@dataclass(slots=True)
class ArticulatedBody:
    """Cell-skinned appendage chains with fixed chassis roots and bone lengths."""

    organism: DevelopedOrganism
    nodes: np.ndarray
    chain_ids: tuple[np.ndarray, ...]
    root_nodes: tuple[int, ...]
    velocities: np.ndarray
    targets: np.ndarray
    target_velocities: np.ndarray
    component_offsets: np.ndarray
    component_velocities: np.ndarray
    attached: np.ndarray
    detached_age: np.ndarray
    elapsed: float = 0.0

    @classmethod
    def from_organism(cls, organism: DevelopedOrganism) -> "ArticulatedBody":
        chains: list[np.ndarray] = []
        roots: list[int] = []
        for appendage_index in range(len(organism.genome.appendages)):
            edge_ids = np.flatnonzero(organism.skeleton_edge_appendage == appendage_index)
            if edge_ids.size < 2:
                raise ValueError("grasper appendage has no articulated chain")
            edges = organism.skeleton_edges[edge_ids]
            roots.append(int(edges[0, 0]))
            chains.append(np.asarray([int(edges[0, 1]), *[int(edge[1]) for edge in edges[1:]]], np.int16))
        endpoints = np.stack([organism.skeleton_nodes[chain[-1], :2] for chain in chains]).astype(np.float32)
        return cls(
            organism,
            organism.skeleton_nodes.copy(),
            tuple(chains),
            tuple(roots),
            np.zeros_like(organism.skeleton_nodes[:, :2], dtype=np.float32),
            endpoints.copy(),
            np.zeros_like(endpoints),
            np.zeros((len(organism.genome.components), 2), dtype=np.float32),
            np.zeros((len(organism.genome.components), 2), dtype=np.float32),
            np.ones(len(chains), dtype=np.bool_),
            np.zeros(len(chains), dtype=np.float32),
        )

    def endpoint(self, appendage: int) -> np.ndarray:
        return self.nodes[self.chain_ids[appendage][-1], :2].astype(np.float64)

    def solve(
        self,
        appendage: int,
        target: np.ndarray,
        response: float,
        *,
        delta: float = .05,
        actuation: float = 1.0,
        load: float = 0.0,
    ) -> np.ndarray:
        """Advance one inertial, length-constrained limb toward a hand target.

        This is the same physical vocabulary used by grounded locomotors:
        recurrent node velocity, damped muscle drive and iterative bone-length
        projection.  A grasp target attracts the terminal hand; it does not
        overwrite a pose or teleport intermediate joints.
        """
        if not self.attached[appendage]:
            return self.endpoint(appendage)
        if not .001 <= delta <= .25 or not 0 <= actuation <= 1 or not 0 <= load <= 1000:
            raise ValueError("articulated grasper dynamics drifted")
        gene = self.organism.genome.appendages[appendage]
        chain_ids = self.chain_ids[appendage]
        root = self.nodes[self.root_nodes[appendage], :2].astype(np.float32) + np.asarray(gene.root_offset, np.float32)
        rest = self.organism.skeleton_nodes[chain_ids, :2].astype(np.float32)
        positions = self.nodes[chain_ids, :2].astype(np.float32, copy=True)
        velocity = self.velocities[chain_ids].astype(np.float32, copy=True)
        lengths = np.linalg.norm(rest[1:] - rest[:-1], axis=1).astype(np.float32)
        desired = np.asarray(target, np.float32)
        reach = float(lengths.sum())
        reach_delta = desired - root
        distance = float(np.linalg.norm(reach_delta))
        if distance > reach and distance > 1e-6:
            desired = root + reach_delta * (reach / distance)
        gain = float(np.clip(response, 0.0, 1.0))
        # Match grounded locomotor timing: the muscle target itself has inertia
        # and a bounded speed.  A noisy controller can no longer snap the hand
        # between distant commands on consecutive frames.
        target_error = desired - self.targets[appendage]
        # A critically damped reach governor replaces frame-to-frame hand
        # interpolation.  The controller asks for a destination; muscle force
        # accelerates the reach target toward it without overshoot or a pose
        # discontinuity.
        omega = 7.5 + 2.5 * gain
        target_acceleration = target_error * (omega * omega) - self.target_velocities[appendage] * (2.0 * omega)
        acceleration_norm = float(np.linalg.norm(target_acceleration))
        if acceleration_norm > 48.0:
            target_acceleration *= 48.0 / acceleration_norm
        self.target_velocities[appendage] += target_acceleration * delta
        target_speed = float(np.linalg.norm(self.target_velocities[appendage]))
        maximum_target_speed = 8.0 + 6.0 * gain
        if target_speed > maximum_target_speed:
            self.target_velocities[appendage] *= maximum_target_speed / target_speed
        self.targets[appendage] += self.target_velocities[appendage] * delta
        filtered_target = self.targets[appendage]
        # Drive the whole articulated chain through a compliant curved muscle
        # pose.  Grounded legs use distributed actuator forces plus PBD bone
        # projection; graspers now use the same pattern instead of dragging a
        # passive rigid chain from its final pixel.  Slack becomes elbow bend,
        # while a fully extended reach naturally straightens.
        cumulative = np.concatenate((np.zeros(1, np.float32), np.cumsum(lengths)))
        fractions = cumulative / max(reach, 1e-6)
        chord = filtered_target - root
        chord_length = float(np.linalg.norm(chord))
        direction = chord / max(chord_length, 1e-6)
        normal = np.asarray((-direction[1], direction[0]), np.float32)
        rest_chord = rest[-1] - rest[0]
        rest_normal = np.asarray((-rest_chord[1], rest_chord[0]), np.float32)
        bend_sign = float(np.sign(np.mean((rest[1:-1] - rest[0]) @ rest_normal))) if len(rest) > 2 else 0.0
        if bend_sign == 0.0:
            bend_sign = -1.0 if float(gene.root_offset[0]) < 0 else 1.0
        bend = min(max(reach - chord_length, 0.0) * .62, reach * .24)
        muscle_pose = root[None] + fractions[:, None] * chord[None]
        muscle_pose += normal[None] * (bend_sign * bend * np.sin(np.pi * fractions))[:, None]
        muscle_pose[0] = root
        muscle_pose[-1] = filtered_target
        previous = positions.copy()
        substeps = 4
        step = delta / substeps
        load_scale = 1.0 / (1.0 + max(load, 0.0) * .12)
        for _ in range(substeps):
            posture_gain = (24.0 + 34.0 * gain) * actuation
            acceleration = (muscle_pose - positions) * posture_gain
            acceleration[0] = 0
            acceleration[-1] += (filtered_target - positions[-1]) * (120.0 + 180.0 * gain) * actuation * load_scale
            # Damaged but still connected neural/muscle tissue retains a very
            # small, slow residual twitch; it cannot create useful grip force.
            damage = 1.0 - actuation
            if .03 < actuation < .45:
                twitch = np.asarray((np.sin(self.elapsed * 8.1 + appendage), np.cos(self.elapsed * 6.7 + appendage)), np.float32)
                acceleration[-1] += twitch * (.42 * damage * actuation)
            velocity[1:] = velocity[1:] * np.exp(-step * (7.2 + load * .08)) + acceleration[1:] * step
            speed = np.linalg.norm(velocity[1:], axis=1, keepdims=True)
            velocity[1:] *= np.minimum(1.0, 10.0 / np.maximum(speed, 1e-6))
            positions[1:] += velocity[1:] * step
            positions[0] = root
            for _constraint in range(9):
                positions[0] = root
                for segment, length in enumerate(lengths):
                    left, right = segment, segment + 1
                    edge = positions[right] - positions[left]
                    current = max(float(np.linalg.norm(edge)), 1e-7)
                    correction = edge * ((current - float(length)) / current)
                    if left == 0:
                        positions[right] -= correction
                    else:
                        positions[left] += correction * .5
                        positions[right] -= correction * .5
                positions[0] = root
        velocity = velocity * .35 + (positions - previous) / max(delta, 1e-6) * .65
        velocity[0] = 0
        self.nodes[chain_ids, :2] = positions
        self.velocities[chain_ids] = velocity
        self.elapsed += delta
        return self.endpoint(appendage)

    def sever(self, appendage: int, *, impulse: np.ndarray | None = None) -> None:
        if not self.attached[appendage]:
            return
        self.attached[appendage] = False
        self.detached_age[appendage] = 0.0
        kick = np.zeros(2, dtype=np.float32) if impulse is None else np.asarray(impulse, dtype=np.float32)
        if kick.shape != (2,) or not np.isfinite(kick).all() or float(np.linalg.norm(kick)) > 20:
            raise ValueError("detached limb impulse drifted")
        self.velocities[self.chain_ids[appendage]] += kick

    def step_detached(self, appendage: int, *, delta: float, ground_y: float, residual: float) -> None:
        """Integrate a severed bone chain with brief residual neural twitch."""
        if self.attached[appendage]:
            return
        chain_ids = self.chain_ids[appendage]
        positions = self.nodes[chain_ids, :2].astype(np.float32, copy=True)
        velocity = self.velocities[chain_ids].astype(np.float32, copy=True)
        rest = self.organism.skeleton_nodes[chain_ids, :2].astype(np.float32)
        lengths = np.linalg.norm(rest[1:] - rest[:-1], axis=1).astype(np.float32)
        age = float(self.detached_age[appendage])
        twitch_strength = float(np.clip(residual, 0, 1)) * np.exp(-age / .48)
        substeps = 4
        step = delta / substeps
        for substep in range(substeps):
            velocity *= np.exp(-step * 2.4)
            velocity[:, 1] += 17.0 * step
            phase = (age + substep * step) * 11.0 + appendage * 1.7
            velocity[-1] += np.asarray((np.sin(phase), np.cos(phase * .83)), np.float32) * (.55 * twitch_strength * step)
            positions += velocity * step
            for _constraint in range(6):
                for segment, length in enumerate(lengths):
                    edge = positions[segment + 1] - positions[segment]
                    current = max(float(np.linalg.norm(edge)), 1e-7)
                    correction = edge * ((current - float(length)) / current) * .5
                    positions[segment] += correction
                    positions[segment + 1] -= correction
            below = positions[:, 1] > ground_y
            if below.any():
                positions[below, 1] = ground_y
                velocity[below, 1] *= -.08
                velocity[below, 0] *= .72
        self.nodes[chain_ids, :2] = positions
        self.velocities[chain_ids] = velocity
        self.detached_age[appendage] += delta
        self.elapsed += delta

    def endpoint_velocity(self, appendage: int) -> np.ndarray:
        return self.velocities[self.chain_ids[appendage][-1]].astype(np.float64)

    def pose_components(self, target_offsets: np.ndarray, *, delta: float, response: float = 1.0) -> None:
        """Critically damp chassis articulation without rotating the 2.5D body.

        Component nodes are the torso/head/pelvis skeleton.  Locomotor and
        grasper chains retain their own fixed-length constraints, while the
        chassis can kneel, bow, compress on suspension, or recover smoothly.
        """
        target = np.asarray(target_offsets, np.float32)
        if target.shape != self.component_offsets.shape or not np.isfinite(target).all():
            raise ValueError("component posture target drifted")
        if not .001 <= delta <= .25 or not 0 <= response <= 1:
            raise ValueError("component posture timing drifted")
        omega = 5.0 + 3.0 * response
        acceleration = (target - self.component_offsets) * (omega * omega) - self.component_velocities * (2.0 * omega)
        norm = np.linalg.norm(acceleration, axis=1, keepdims=True)
        acceleration *= np.minimum(1.0, 36.0 / np.maximum(norm, 1e-6))
        self.component_velocities += acceleration * delta
        speed = np.linalg.norm(self.component_velocities, axis=1, keepdims=True)
        self.component_velocities *= np.minimum(1.0, 8.0 / np.maximum(speed, 1e-6))
        self.component_offsets += self.component_velocities * delta
        count = len(self.organism.genome.components)
        self.nodes[:count, :2] = self.organism.skeleton_nodes[:count, :2] + self.component_offsets

    def cells(self) -> np.ndarray:
        points = self.organism.cell_xy.astype(np.float32, copy=True)
        # Blend component-node posture through the existing developmental
        # ownership weights. Appendage skinning below remains authoritative
        # for limb cells, so a kneel changes the chassis without melting arms.
        points += self.organism.component_weights @ self.component_offsets
        for appendage, chain_ids in enumerate(self.chain_ids):
            cell_ids = self._skinned_cell_ids(appendage)
            if cell_ids.size == 0:
                continue
            rest = self.organism.skeleton_nodes[chain_ids, :2].astype(np.float32)
            posed = self.nodes[chain_ids, :2].astype(np.float32)
            cells = points[cell_ids]
            rest_vectors = rest[1:] - rest[:-1]
            denominators = np.maximum(np.square(rest_vectors).sum(axis=1), 1e-8)
            relative = cells[:, None, :] - rest[:-1][None, :, :]
            along = np.clip((relative * rest_vectors[None]).sum(axis=2) / denominators[None], 0, 1)
            projections = rest[:-1][None] + along[:, :, None] * rest_vectors[None]
            distance = np.linalg.norm(cells[:, None] - projections, axis=2)
            segment = distance.argmin(axis=1)
            row = np.arange(cell_ids.size)
            chosen_along = along[row, segment]
            rest_direction = rest_vectors[segment]
            rest_normal = np.stack((-rest_direction[:, 1], rest_direction[:, 0]), axis=1)
            rest_normal /= np.maximum(np.linalg.norm(rest_normal, axis=1, keepdims=True), 1e-8)
            lateral = ((cells - projections[row, segment]) * rest_normal).sum(axis=1)
            posed_vector = posed[segment + 1] - posed[segment]
            posed_normal = np.stack((-posed_vector[:, 1], posed_vector[:, 0]), axis=1)
            posed_normal /= np.maximum(np.linalg.norm(posed_normal, axis=1, keepdims=True), 1e-8)
            points[cell_ids] = posed[segment] + chosen_along[:, None] * posed_vector + lateral[:, None] * posed_normal
        return points

    def _skinned_cell_ids(self, appendage: int) -> np.ndarray:
        """Return limb cells without reclassifying load-bearing torso cells.

        The same selection is used for rendering and limb-proportion gates, so
        a grasper cannot appear muscular merely because its root crosses the
        chassis.  Distal arms and legs retain the developmental tube thickness.
        """
        cell_ids = np.flatnonzero(self.organism.appendage_index == appendage)
        if cell_ids.size == 0:
            return cell_ids
        gene = self.organism.genome.appendages[appendage]
        root_component = next(component for component in self.organism.genome.components if component.component_id == gene.root_component)
        points = self.organism.cell_xy.astype(np.float32, copy=False)
        root_delta = (points[cell_ids] - np.asarray(root_component.anchor, np.float32)) / np.asarray(root_component.radius, np.float32)
        core_distance = np.max(np.abs(root_delta), axis=1) if int(np.argmax(self.organism.genome.family_mix)) == 4 else np.linalg.norm(root_delta, axis=1)
        return cell_ids[core_distance >= .92]

    def geometry(self, appendage: int) -> LimbGeometry:
        gene = self.organism.genome.appendages[appendage]
        return LimbGeometry(gene.kind, gene.segments, self.length(appendage), int(self._skinned_cell_ids(appendage).size))

    def require_peer_limbs(self, grasp_kinds: set[str], locomotor_kinds: set[str]) -> None:
        """Fail closed when ordinary graspers stop reading as locomotor peers.

        Specialized tails, tendrils, roots and hardpoints intentionally opt out.
        For arm-and-leg chassis, reach, articulated joints and visible cell mass
        remain comparable; only the terminal contact role differs.
        """
        graspers = [self.geometry(index) for index, gene in enumerate(self.organism.genome.appendages) if gene.kind in grasp_kinds]
        locomotors = [self.geometry(index) for index, gene in enumerate(self.organism.genome.appendages) if gene.kind in locomotor_kinds]
        if not graspers or not locomotors:
            return
        grasp_length = float(np.mean([limb.length for limb in graspers]))
        locomotor_length = float(np.mean([limb.length for limb in locomotors]))
        grasp_cells = float(np.mean([limb.cell_count for limb in graspers]))
        locomotor_cells = float(np.mean([limb.cell_count for limb in locomotors]))
        if {limb.segments for limb in graspers} != {limb.segments for limb in locomotors}:
            raise ValueError("grasper and locomotor joint vocabularies diverged")
        if not .80 <= grasp_length / locomotor_length <= 1.20:
            raise ValueError("grasper and locomotor reaches diverged")
        if not .70 <= grasp_cells / max(locomotor_cells, 1.0) <= 1.30:
            raise ValueError("grasper and locomotor cell masses diverged")

    def chain(self, appendage: int) -> np.ndarray:
        return self.nodes[self.chain_ids[appendage], :2].copy()

    def root(self, appendage: int) -> np.ndarray:
        gene = self.organism.genome.appendages[appendage]
        return self.nodes[self.root_nodes[appendage], :2].astype(np.float64) + np.asarray(gene.root_offset, np.float64)

    def length(self, appendage: int) -> float:
        chain = self.organism.skeleton_nodes[self.chain_ids[appendage], :2]
        return float(np.linalg.norm(chain[1:] - chain[:-1], axis=1).sum())

    def feasible(self, appendage: int, target: np.ndarray, contact_radius: float = 1.25) -> bool:
        return float(np.linalg.norm(np.asarray(target) - self.root(appendage))) <= self.length(appendage) + contact_radius

    def max_length_error(self) -> float:
        maximum = 0.0
        for chain_ids in self.chain_ids:
            rest = self.organism.skeleton_nodes[chain_ids, :2]
            posed = self.nodes[chain_ids, :2]
            error = np.abs(np.linalg.norm(rest[1:] - rest[:-1], axis=1) - np.linalg.norm(posed[1:] - posed[:-1], axis=1))
            maximum = max(maximum, float(error.max(initial=0)))
        return maximum
