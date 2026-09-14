from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from .corridors import ConnectedJunctions, cross_route, extend_route_straight, route_locations
from .junctions import waypoint_before


@dataclass
class VehicleRoute:
    role: str
    actor: Any
    path: list[Any]
    color: str


class RegionalIntersectionScenario:
    """Continuous, rule-following traffic across two connected junctions.

    The v1.0 scene uses two moving sensing vehicles.  A large, moving road
    vehicle runs ahead of the J2 CAV on the same legal lane, so CARLA's ray
    casting—not a hand-authored visibility mask—creates the occlusion.
    """

    COLORS = {
        "regional_ego": "#f59e0b",
        "regional_car1": "#22c55e",
        "regional_car3": "#3b82f6",
        "regional_cav2_blocker": "#ef4444",
        "regional_cav2_front_1": "#ec4899",
        "regional_cav2_front_2": "#db2777",
        "regional_cav2_rear": "#be185d",
        "regional_corridor_1": "#a855f7",
        "regional_corridor_2": "#9333ea",
        "regional_corridor_3": "#7e22ce",
        "regional_east_front": "#ec4899",
        "regional_east_rear": "#db2777",
        "regional_south_1": "#14b8a6",
        "regional_south_2": "#0d9488",
        "regional_south_3": "#0f766e",
        "regional_late_1": "#a855f7",
        "regional_late_2": "#9333ea",
        "regional_late_3": "#7e22ce",
        "regional_late_4": "#6b21a8",
        "regional_late_5": "#581c87",
    }

    ORDINARY_VEHICLE_PATTERNS = [
        "vehicle.audi.a2",
        "vehicle.tesla.model3",
        "vehicle.lincoln.mkz_2020",
        "vehicle.nissan.patrol",
        "vehicle.mercedes.coupe_2020",
        "vehicle.*",
    ]

    def __init__(
        self,
        client: Any,
        world: Any,
        traffic_manager: Any,
        cfg: dict[str, Any],
        corridor: ConnectedJunctions,
    ):
        self.client = client
        self.world = world
        self.tm = traffic_manager
        self.cfg = cfg
        self.corridor = corridor
        self.actors: list[Any] = []
        self.routes: dict[str, VehicleRoute] = {}
        self._rng = random.Random(int(cfg["carla"]["seed"]))

    def setup(self) -> None:
        self._destroy_stale_owned_actors()
        regional = self.cfg["regional"]
        corridor_yaw = _route_yaw(self.corridor.corridor_waypoints)
        ego_route_tail_m = float(regional.get("ego_route_tail_m", 180.0))
        ego_start_advance_m = float(regional.get("ego_start_advance_m", 0.0))

        # 原始走廊路线：用于布置其他车辆，位置不跟随 Ego 改变
        support_route = extend_route_straight(
            self.corridor.corridor_waypoints,
            ego_route_tail_m,
        )

        # 先向前多生成 advance 米，再裁掉开头 advance 米。
        # 这样 Ego 起点和终点一起前移，路线长度基本保持不变。
        ego_full_route = extend_route_straight(
            self.corridor.corridor_waypoints,
            ego_route_tail_m + ego_start_advance_m,
        )

        ego_route = ego_full_route
        if ego_start_advance_m > 0.0:
            ego_route = _route_suffix_after_distance(
                ego_full_route,
                ego_start_advance_m,
        )
        car1_enabled = bool(regional.get("enable_j1_cav", True))
        car1_route = None
        if car1_enabled:
            car1_route = cross_route(
                self.corridor.first.junction,
                corridor_yaw,
                float(regional["car1_start_before_junction_m"]),
                float(regional["route_exit_distance_m"]),
            )
        car3_route = cross_route(
            self.corridor.second.junction,
            corridor_yaw,
            float(regional["car3_start_before_junction_m"]),
            float(regional["route_exit_distance_m"]),
        )
        car3_route = extend_route_straight(
            car3_route,
            float(regional.get("car3_route_tail_m", 100.0)),
        )

        self._spawn_cav(
            "regional_ego",
            ego_route,
            speed_difference=float(regional.get("ego_speed_difference_pct", 35.0)),
        )
        if car1_enabled:
            assert car1_route is not None
            self._spawn_cav(
                "regional_car1",
                car1_route,
                speed_difference=float(regional.get("car1_speed_difference_pct", 15.0)),
            )
        self._spawn_cav(
            "regional_car3",
            car3_route,
            speed_difference=float(regional.get("car3_speed_difference_pct", -5.0)),
        )
        self._spawn_support_traffic(support_route, corridor_yaw)
        self._spawn_background(int(regional["background_vehicle_count"]))

    @property
    def ego(self) -> Any:
        return self.routes["regional_ego"].actor

    def metadata(self) -> dict[str, Any]:
        return {
            "corridor": self.corridor.as_dict(),
            "routes": {
                role: {
                    "actor_id": int(item.actor.id),
                    "color": item.color,
                    "xyz": [[float(p.x), float(p.y), float(p.z)] for p in item.path],
                }
                for role, item in self.routes.items()
            },
            "owned_actor_ids": [int(actor.id) for actor in self.actors],
        }

    def actor_states(self) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for actor in self.actors:
            try:
                transform = actor.get_transform()
                velocity = actor.get_velocity()
            except RuntimeError:
                continue
            output.append(
                {
                    "actor_id": int(actor.id),
                    "track_id": int(actor.id),
                    "type_id": actor.type_id,
                    "role_name": actor.attributes.get("role_name", ""),
                    "location": [
                        float(transform.location.x),
                        float(transform.location.y),
                        float(transform.location.z),
                    ],
                    "rotation": [
                        float(transform.rotation.roll),
                        float(transform.rotation.pitch),
                        float(transform.rotation.yaw),
                    ],
                    "velocity": [float(velocity.x), float(velocity.y), float(velocity.z)],
                    "bbox_extent": [
                        float(actor.bounding_box.extent.x),
                        float(actor.bounding_box.extent.y),
                        float(actor.bounding_box.extent.z),
                    ],
                }
            )
        return output

    def close(self) -> None:
        for actor in reversed(self.actors):
            try:
                actor.destroy()
            except Exception:
                pass
        self.actors.clear()
        self.routes.clear()

    def _spawn_support_traffic(self, ego_route: list[Any], corridor_yaw: float) -> None:
        """Create deterministic legal traffic at the early and late service areas.

        Every support vehicle exists from the beginning. Different start
        distances and speeds make them enter the service area at different
        times without teleporting or time-triggered spawning.
        """
        regional = self.cfg["regional"]
        for spec in regional.get("support_vehicles", []):
            role = str(spec["role"])
            route_kind = str(spec["route"])
            if route_kind == "j2_cross":
                route = cross_route(
                    self.corridor.second.junction,
                    corridor_yaw,
                    float(spec["start_offset_m"]),
                    float(regional["route_exit_distance_m"]),
                )
            elif route_kind == "j1_cross":
                route = cross_route(
                    self.corridor.first.junction,
                    corridor_yaw,
                    float(spec["start_offset_m"]),
                    float(regional["route_exit_distance_m"]),
                )
            elif route_kind == "corridor_forward":
                route = _route_suffix_after_distance(
                    ego_route,
                    float(spec["start_offset_m"]),
                )
            elif route_kind == "corridor_behind_ego":
                start = waypoint_before(ego_route[0], float(spec["start_offset_m"]))
                route = [start, *ego_route]
            elif route_kind == "corridor_oncoming":
                reference = _route_suffix_after_distance(
                    ego_route,
                    float(spec["start_offset_m"]),
                )[0]
                opposite = _opposite_driving_lane(reference, self.world.get_map())
                route = extend_route_straight(
                    [opposite],
                    float(spec.get("route_tail_m", regional["route_exit_distance_m"])),
                )
            else:
                raise ValueError(f"unknown support vehicle route: {route_kind}")
            self._spawn_cav(
                role,
                route,
                speed_difference=float(spec.get("speed_difference_pct", 20.0)),
                required=bool(spec.get("required", False)),
                blueprint_patterns=[str(item) for item in spec.get("blueprints", [])] or None,
            )

    def _spawn_cav(
        self,
        role: str,
        route: list[Any],
        speed_difference: float,
        required: bool = True,
        blueprint_patterns: list[str] | None = None,
    ) -> None:
        import carla

        patterns = {
            "regional_ego": ["vehicle.lincoln.mkz_2020", "vehicle.tesla.model3", "vehicle.*"],
            "regional_car1": ["vehicle.audi.a2", "vehicle.mini.cooper_s_2021", "vehicle.*"],
            "regional_car3": ["vehicle.mercedes.coupe_2020", "vehicle.nissan.patrol", "vehicle.*"],
            "regional_cav2_blocker": [
                "vehicle.mitsubishi.fusorosa",
                "vehicle.mercedes.sprinter",
                "vehicle.carlamotors.european_hgv",
            ],
            "regional_east_front": ["vehicle.audi.a2", "vehicle.mini.cooper_s", "vehicle.*"],
            "regional_east_rear": ["vehicle.micro.microlino", "vehicle.audi.a2", "vehicle.*"],
            "regional_south_1": ["vehicle.mini.cooper_s", "vehicle.audi.a2", "vehicle.*"],
            "regional_south_2": ["vehicle.micro.microlino", "vehicle.citroen.c3", "vehicle.*"],
            "regional_south_3": ["vehicle.audi.a2", "vehicle.mini.cooper_s", "vehicle.*"],
        }
        blueprint = self._blueprint(
            blueprint_patterns or patterns.get(role, self.ORDINARY_VEHICLE_PATTERNS)
        )
        self._set_role(blueprint, role)
        start = route[0].transform
        transform = carla.Transform(
            carla.Location(
                x=float(start.location.x),
                y=float(start.location.y),
                z=float(start.location.z + 0.25),
            ),
            start.rotation,
        )
        actor = self.world.try_spawn_actor(blueprint, transform)
        if actor is None:
            message = f"failed to spawn {role}; its start point is occupied"
            if required:
                raise RuntimeError(message + "; choose another corridor_rank or reduce background traffic")
            print(f"WARNING: {message}; continuing without this support vehicle")
            return
        self.actors.append(actor)
        path = route_locations(route, spacing_m=2.0)
        actor.set_autopilot(True, int(self.cfg["carla"]["traffic_manager_port"]))
        self.tm.auto_lane_change(actor, False)
        self.tm.vehicle_percentage_speed_difference(actor, float(speed_difference))
        self.tm.distance_to_leading_vehicle(actor, 3.0)
        try:
            self.tm.set_path(actor, path[1:])
        except (AttributeError, RuntimeError) as error:
            print(f"WARNING: TrafficManager.set_path failed for {role}: {error}")
            print("         Vehicle will continue with TrafficManager's default route.")
        self.routes[role] = VehicleRoute(
            role=role,
            actor=actor,
            path=path,
            color=self.COLORS.get(role, "#64748b"),
        )

    def _spawn_background(self, requested: int) -> None:
        center = np.asarray(self.corridor.center[:2], dtype=np.float64)
        radius = float(self.cfg["regional"]["background_spawn_radius_m"])
        spawn_points = list(self.world.get_map().get_spawn_points())
        self._rng.shuffle(spawn_points)
        spawned = 0
        for transform in spawn_points:
            if spawned >= requested:
                break
            xy = np.asarray([transform.location.x, transform.location.y], dtype=np.float64)
            if float(np.linalg.norm(xy - center)) > radius:
                continue
            if any(transform.location.distance(item.actor.get_location()) < 10.0 for item in self.routes.values()):
                continue
            blueprint = self._blueprint(self.ORDINARY_VEHICLE_PATTERNS)
            self._set_role(blueprint, "regional_background")
            actor = self.world.try_spawn_actor(blueprint, transform)
            if actor is None:
                continue
            actor.set_autopilot(True, int(self.cfg["carla"]["traffic_manager_port"]))
            self.tm.vehicle_percentage_speed_difference(actor, float(self._rng.uniform(-8.0, 18.0)))
            self.tm.distance_to_leading_vehicle(actor, float(self._rng.uniform(2.5, 5.0)))
            self.actors.append(actor)
            spawned += 1
        print(f"Background traffic: requested={requested}, spawned={spawned}")
        minimum = int(self.cfg["regional"].get("minimum_background_spawned", 0))
        if spawned < minimum:
            raise RuntimeError(
                f"only {spawned}/{requested} background vehicles spawned; "
                f"the dense-scene minimum is {minimum}. Select another corridor or reduce the minimum."
            )

    def _blueprint(self, patterns: list[str]) -> Any:
        library = self.world.get_blueprint_library()
        for pattern in patterns:
            matches = [bp for bp in library.filter(pattern) if _safe_vehicle(bp)]
            if matches:
                matches.sort(key=lambda item: item.id)
                blueprint = matches[self._rng.randrange(len(matches))]
                if blueprint.has_attribute("color"):
                    colors = list(blueprint.get_attribute("color").recommended_values)
                    if colors:
                        blueprint.set_attribute("color", self._rng.choice(colors))
                return blueprint
        raise RuntimeError(f"no safe CARLA vehicle blueprint matches {patterns}")

    @staticmethod
    def _set_role(blueprint: Any, role: str) -> None:
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", role)

    def _destroy_stale_owned_actors(self) -> None:
        for actor in self.world.get_actors():
            if actor.attributes.get("role_name", "").startswith("regional_"):
                try:
                    actor.destroy()
                except Exception:
                    pass


def _route_yaw(route: tuple[Any, ...]) -> float:
    start = route[0].transform.location
    end = route[-1].transform.location
    return float(np.degrees(np.arctan2(end.y - start.y, end.x - start.x)))


def _route_suffix_after_distance(route: list[Any], distance_m: float) -> list[Any]:
    """Return the legal remainder starting at an exact metric route position.

    This is used only to distribute ordinary traffic along the already chosen
    Ego corridor. CARLA route anchors can be tens of metres apart, so selecting
    the next anchor would collapse several requested offsets onto one spawn
    point. We instead ask the starting waypoint for the legal waypoint at the
    residual distance inside that segment.
    """
    if len(route) < 2:
        raise ValueError("route requires at least two waypoints")
    target = max(0.0, float(distance_m))
    traveled = 0.0
    for index, (start, end) in enumerate(zip(route[:-1], route[1:])):
        segment_m = float(start.transform.location.distance(end.transform.location))
        if traveled + segment_m + 1e-6 < target:
            traveled += segment_m
            continue
        residual_m = min(max(target - traveled, 0.0), segment_m)
        if residual_m <= 0.25:
            exact = start
        elif segment_m - residual_m <= 0.25:
            exact = end
        else:
            options = list(start.next(residual_m))
            exact = (
                min(
                    options,
                    key=lambda waypoint: waypoint.transform.location.distance(
                        end.transform.location
                    ),
                )
                if options
                else end
            )
        return _deduplicate_route([exact, *route[index + 1 :]])
    return list(route[-2:])


def _deduplicate_route(route: list[Any]) -> list[Any]:
    output: list[Any] = []
    for waypoint in route:
        if (
            output
            and waypoint.transform.location.distance(output[-1].transform.location) < 0.5
        ):
            continue
        output.append(waypoint)
    return output


def _opposite_driving_lane(reference: Any, carla_map: Any) -> Any:
    """Find a nearby legal driving lane whose travel direction is opposite.

    Adjacent lanes are preferred. A local map search is used as a fallback for
    divided roads where CARLA's left/right lane links stop at a median.
    """
    import carla

    reference_yaw = float(reference.transform.rotation.yaw)
    candidates: list[Any] = []
    for accessor in ("get_left_lane", "get_right_lane"):
        current = reference
        for _ in range(8):
            current = getattr(current, accessor)()
            if current is None:
                break
            if (
                current.lane_type == carla.LaneType.Driving
                and _heading_is_opposite(reference_yaw, current.transform.rotation.yaw)
            ):
                candidates.append(current)
                break
    if not candidates:
        origin = reference.transform.location
        candidates = [
            waypoint
            for waypoint in carla_map.generate_waypoints(2.0)
            if not waypoint.is_junction
            and waypoint.lane_type == carla.LaneType.Driving
            and origin.distance(waypoint.transform.location) <= 18.0
            and _heading_is_opposite(reference_yaw, waypoint.transform.rotation.yaw)
        ]
    if not candidates:
        raise RuntimeError(
            "no legal oncoming lane found near the selected corridor; "
            "choose another corridor_rank"
        )
    origin = reference.transform.location
    return min(
        candidates,
        key=lambda waypoint: origin.distance(waypoint.transform.location),
    )


def _heading_is_opposite(reference_yaw: float, candidate_yaw: float) -> bool:
    delta = abs((float(candidate_yaw) - float(reference_yaw) + 180.0) % 360.0 - 180.0)
    return delta >= 135.0


def _safe_vehicle(blueprint: Any) -> bool:
    # ``str(ActorAttribute)`` is not stable across CARLA PythonAPI builds. In
    # some 0.9.16 packages it renders the whole attribute object rather than
    # just ``car``/``truck`` and consequently rejected every vehicle.*
    if not str(blueprint.id).startswith("vehicle."):
        return False
    if blueprint.has_attribute("number_of_wheels"):
        try:
            return int(blueprint.get_attribute("number_of_wheels").as_int()) >= 4
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return True
    return True
