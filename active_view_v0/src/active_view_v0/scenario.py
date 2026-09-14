from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from .geometry import interpolate_polyline, yaw_to_axes
from .junctions import JunctionInfo, choose_straight_pair, waypoint_after, waypoint_before


@dataclass
class MovingEvent:
    actor: Any
    start_xyz: np.ndarray
    end_xyz: np.ndarray
    start_s: float
    duration_s: float
    yaw_deg: float

    def step(self, elapsed_s: float) -> None:
        import carla

        alpha = np.clip((float(elapsed_s) - self.start_s) / self.duration_s, 0.0, 1.0)
        # Smoothstep prevents an unrealistic velocity discontinuity at trigger time.
        alpha = float(alpha * alpha * (3.0 - 2.0 * alpha))
        location = self.start_xyz + alpha * (self.end_xyz - self.start_xyz)
        self.actor.set_transform(
            carla.Transform(
                carla.Location(x=float(location[0]), y=float(location[1]), z=float(location[2])),
                carla.Rotation(yaw=float(self.yaw_deg)),
            )
        )


class ControlledOcclusionScenario:
    """Deterministic two-sided crossing events at a selected junction."""

    def __init__(self, world: Any, cfg: dict[str, Any], junction_info: JunctionInfo):
        self.world = world
        self.cfg = cfg
        self.info = junction_info
        self.actors: list[Any] = []
        self.events: list[MovingEvent] = []
        self.ego: Any | None = None
        self.route_xyz: np.ndarray | None = None
        self.route_forward_xy: np.ndarray | None = None
        self.route_right_xy: np.ndarray | None = None
        self.event_center_xyz: np.ndarray | None = None
        self._rng = random.Random(int(cfg["carla"]["seed"]))

    def setup(self) -> None:
        import carla

        self._destroy_stale_owned_actors()
        entry_wp, exit_wp = choose_straight_pair(self.info.junction)
        scenario_cfg = self.cfg["scenario"]
        before_wp = waypoint_before(entry_wp, float(scenario_cfg["ego_start_distance_m"]))
        after_wp = waypoint_after(exit_wp, float(scenario_cfg["route_exit_distance_m"]))
        route = np.asarray(
            [
                _location_array(before_wp.transform.location),
                _location_array(entry_wp.transform.location),
                _location_array(exit_wp.transform.location),
                _location_array(after_wp.transform.location),
            ],
            dtype=np.float64,
        )
        # Remove repeated endpoints returned by short/dead-end lanes.
        keep = [0]
        for index in range(1, len(route)):
            if np.linalg.norm(route[index, :2] - route[keep[-1], :2]) > 0.5:
                keep.append(index)
        route = route[keep]
        if len(route) < 2:
            raise RuntimeError("could not build a route through the selected junction")
        self.route_xyz = route
        forward, right = yaw_to_axes(entry_wp.transform.rotation.yaw)
        self.route_forward_xy = forward
        self.route_right_xy = right
        center = 0.5 * (
            _location_array(entry_wp.transform.location) + _location_array(exit_wp.transform.location)
        )
        self.event_center_xyz = center

        ego_bp = self._blueprint(["vehicle.lincoln.mkz_2020", "vehicle.tesla.model3", "vehicle.*"])
        self._set_role(ego_bp, "active_view_ego")
        start_xyz, start_yaw = interpolate_polyline(route, 0.0)
        ego_tf = carla.Transform(
            carla.Location(x=float(start_xyz[0]), y=float(start_xyz[1]), z=float(start_xyz[2] + 0.25)),
            carla.Rotation(yaw=float(start_yaw)),
        )
        self.ego = self._spawn_or_raise(ego_bp, ego_tf, "ego vehicle")
        self.ego.set_simulate_physics(False)
        self.actors.append(self.ego)

        lane_width = max(3.0, float(entry_wp.lane_width))
        self._spawn_occluder(center, forward, right, lane_width, side=1.0, longitudinal=-3.0)
        self._spawn_occluder(center, forward, right, lane_width, side=-1.0, longitudinal=4.0)
        self._spawn_crossing_events(center, forward, right, lane_width)

    def step(self, elapsed_s: float) -> None:
        import carla

        if self.ego is None or self.route_xyz is None:
            raise RuntimeError("scenario.setup() must be called before step()")
        distance = float(self.cfg["scenario"]["ego_speed_mps"]) * float(elapsed_s)
        xyz, yaw = interpolate_polyline(self.route_xyz, distance)
        self.ego.set_transform(
            carla.Transform(
                carla.Location(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2] + 0.25)),
                carla.Rotation(yaw=float(yaw)),
            )
        )
        for event in self.events:
            event.step(elapsed_s)

    def metadata(self) -> dict[str, Any]:
        if self.route_xyz is None or self.event_center_xyz is None:
            raise RuntimeError("scenario has not been set up")
        return {
            "junction": self.info.as_dict(),
            "route_xyz": self.route_xyz.tolist(),
            "event_center_xyz": self.event_center_xyz.tolist(),
            "forward_xy": self.route_forward_xy.tolist(),
            "right_xy": self.route_right_xy.tolist(),
            "controlled_actor_ids": [int(actor.id) for actor in self.actors],
        }

    def close(self) -> None:
        for actor in reversed(self.actors):
            try:
                actor.destroy()
            except Exception:
                pass
        self.actors.clear()
        self.events.clear()

    def _spawn_occluder(
        self,
        center: np.ndarray,
        forward: np.ndarray,
        right: np.ndarray,
        lane_width: float,
        side: float,
        longitudinal: float,
    ) -> None:
        import carla

        blueprint = self._blueprint(
            ["vehicle.carlamotors.carlacola", "vehicle.mercedes.sprinter", "vehicle.*"]
        )
        self._set_role(blueprint, "active_view_occluder")
        xy = center[:2] + side * 1.25 * lane_width * right + longitudinal * forward
        yaw = math.degrees(math.atan2(forward[1], forward[0]))
        transform = carla.Transform(
            carla.Location(x=float(xy[0]), y=float(xy[1]), z=float(center[2] + 0.35)),
            carla.Rotation(yaw=float(yaw)),
        )
        actor = self._spawn_or_raise(blueprint, transform, "occluding vehicle")
        actor.set_simulate_physics(False)
        self.actors.append(actor)

    def _spawn_crossing_events(
        self,
        center: np.ndarray,
        forward: np.ndarray,
        right: np.ndarray,
        lane_width: float,
    ) -> None:
        event_cfgs = [self.cfg["scenario"]["event_1"], self.cfg["scenario"]["event_2"]]
        patterns = [
            ["walker.pedestrian.0001", "walker.pedestrian.*"],
            ["vehicle.bh.crossbike", "vehicle.diamondback.century", "walker.pedestrian.*"],
        ]
        longitudinal = [-3.0, 4.0]
        for index, event_cfg in enumerate(event_cfgs):
            side = float(event_cfg["side"])
            start_xy = center[:2] + side * 2.15 * lane_width * right + longitudinal[index] * forward
            end_xy = center[:2] - side * 2.15 * lane_width * right + longitudinal[index] * forward
            start_xyz = np.asarray([start_xy[0], start_xy[1], center[2] + 0.45], dtype=np.float64)
            end_xyz = np.asarray([end_xy[0], end_xy[1], center[2] + 0.45], dtype=np.float64)
            direction = end_xy - start_xy
            yaw = math.degrees(math.atan2(direction[1], direction[0]))
            blueprint = self._blueprint(patterns[index])
            self._set_role(blueprint, f"active_view_target_{index + 1}")
            import carla

            transform = carla.Transform(
                carla.Location(x=float(start_xyz[0]), y=float(start_xyz[1]), z=float(start_xyz[2])),
                carla.Rotation(yaw=float(yaw)),
            )
            actor = self._spawn_or_raise(blueprint, transform, f"crossing target {index + 1}")
            actor.set_simulate_physics(False)
            self.actors.append(actor)
            self.events.append(
                MovingEvent(
                    actor=actor,
                    start_xyz=start_xyz,
                    end_xyz=end_xyz,
                    start_s=float(event_cfg["start_s"]),
                    duration_s=float(event_cfg["duration_s"]),
                    yaw_deg=float(yaw),
                )
            )

    def _blueprint(self, patterns: list[str]) -> Any:
        library = self.world.get_blueprint_library()
        for pattern in patterns:
            matches = list(library.filter(pattern))
            if matches:
                matches.sort(key=lambda bp: bp.id)
                return matches[self._rng.randrange(len(matches))]
        raise RuntimeError(f"no CARLA blueprint matches {patterns}")

    @staticmethod
    def _set_role(blueprint: Any, role: str) -> None:
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", role)
        if blueprint.has_attribute("is_invincible"):
            blueprint.set_attribute("is_invincible", "true")

    def _spawn_or_raise(self, blueprint: Any, transform: Any, label: str) -> Any:
        actor = self.world.try_spawn_actor(blueprint, transform)
        if actor is None:
            raise RuntimeError(
                f"failed to spawn {label} at {transform.location}; choose another junction_id "
                "or inspect the map for a collision"
            )
        return actor

    def _destroy_stale_owned_actors(self) -> None:
        for actor in self.world.get_actors():
            role = actor.attributes.get("role_name", "")
            if role.startswith("active_view_"):
                try:
                    actor.destroy()
                except Exception:
                    pass


def _location_array(location: Any) -> np.ndarray:
    return np.asarray([location.x, location.y, location.z], dtype=np.float64)

