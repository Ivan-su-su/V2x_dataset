"""Interactive, single-tick-owner browser editor for the regional CARLA scene.

The editor never runs a collection or writes to an existing dataset directory.
Use Apply & Restart to save a separate YAML draft and respawn the scene.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import math
import queue
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np

from .bev_render import project_world_to_camera
from .config import dump_effective_config, load_config, validate_config
from .corridors import extend_route_straight, rank_connected_junctions
from .junctions import rank_junctions
from .regional_layout import build_regional_layout
from .regional_preview import _sensor_range_summary
from .regional_scenario import RegionalIntersectionScenario
from .sensors import SensorRig


ALLOWED_ROUTES = {
    "j1_cross", "j2_cross", "j2_cross_reverse", "corridor_forward",
    "corridor_behind_ego", "corridor_oncoming",
}
AIRV2X_TOWNS = ("Town01", "Town02", "Town03", "Town04", "Town06", "Town07", "Town12")


def nearest_route_distance(point_xy: Any, route_xyz: Any) -> tuple[float, float]:
    """Return arc length and lateral error at the closest point of a legal route."""
    points = np.asarray(route_xyz, dtype=np.float64)[:, :2]
    point = np.asarray(point_xy, dtype=np.float64)[:2]
    if len(points) < 2:
        raise ValueError("the selected route has fewer than two waypoints")
    start, delta = points[:-1], np.diff(points, axis=0)
    squared = np.einsum("ij,ij->i", delta, delta)
    fraction = np.clip(np.einsum("ij,ij->i", point - start, delta) /
                       np.maximum(squared, 1e-9), 0, 1)
    projected = start + fraction[:, None] * delta
    errors = np.linalg.norm(projected - point, axis=1)
    index = int(np.argmin(errors))
    distance = float(np.linalg.norm(delta[:index], axis=1).sum() +
                     fraction[index] * math.sqrt(squared[index]))
    return distance, float(errors[index])


def finite_number(value: Any, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    try:
        number = float(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name} must be a number") from error
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return number


def edited_config(cfg: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    """Apply only editor-controlled fields to a copy; preserve all other YAML settings."""
    if not isinstance(changes, dict):
        raise ValueError("edits must be an object")
    allowed = {"ego_speed_difference_pct", "car3_speed_difference_pct",
               "car3_start_before_junction_m", "background_vehicle_count",
               "j2_switch_time_s", "rsu_forward_m", "rsu_right_m",
               "uav_forward_m", "uav_right_m", "initial_candidate",
               "support_vehicles", "map_name", "junction_1_id", "junction_2_id",
               "ego_start_advance_m"}
    unexpected = set(changes) - allowed
    if unexpected:
        raise ValueError(f"unsupported edit fields: {sorted(unexpected)}")
    updated = copy.deepcopy(cfg)
    region = updated["regional"]
    if "map_name" in changes:
        name = str(changes["map_name"])
        if name not in AIRV2X_TOWNS:
            raise ValueError("select an AirV2X CARLA town from the map list")
        if name != str(cfg["carla"]["map"]):
            region["junction_1_id"] = region["junction_2_id"] = None
        updated["carla"]["map"] = name
    if "junction_1_id" in changes or "junction_2_id" in changes:
        first = changes.get("junction_1_id", region.get("junction_1_id"))
        second = changes.get("junction_2_id", region.get("junction_2_id"))
        if (first is None) != (second is None):
            raise ValueError("select both J1 and J2, or clear both")
        if first is not None:
            first, second = int(first), int(second)
            if first == second:
                raise ValueError("J1 and J2 must differ")
        region["junction_1_id"], region["junction_2_id"] = first, second
    if "ego_start_advance_m" in changes:
        region["ego_start_advance_m"] = finite_number(
            changes["ego_start_advance_m"], "ego_start_advance_m", 0, 300
        )
    for field in ("ego_speed_difference_pct", "car3_speed_difference_pct"):
        if field in changes:
            region[field] = finite_number(changes[field], field, -80.0, 100.0)
    if "car3_start_before_junction_m" in changes:
        region["car3_start_before_junction_m"] = finite_number(
            changes["car3_start_before_junction_m"], "car3_start_before_junction_m", 5.0, 500.0
        )
    if "background_vehicle_count" in changes:
        region["background_vehicle_count"] = int(finite_number(
            changes["background_vehicle_count"], "background_vehicle_count", 0, 100
        ))
    if "j2_switch_time_s" in changes:
        region.setdefault("fixed_signal_plan", {})["j2_switch_time_s"] = finite_number(
            changes["j2_switch_time_s"], "j2_switch_time_s", 0, float(region["duration_s"])
        )
    for field, group, key in (
        ("rsu_forward_m", "rsu", "offset_forward_m"),
        ("rsu_right_m", "rsu", "offset_right_m"),
    ):
        if field in changes:
            region[group][key] = finite_number(changes[field], field, -300, 300)
    for field, key in (("uav_forward_m", "uav_grid_offset_forward_m"),
                       ("uav_right_m", "uav_grid_offset_right_m")):
        if field in changes:
            region[key] = finite_number(changes[field], field, -300, 300)
    if "initial_candidate" in changes:
        value = str(changes["initial_candidate"])
        size = int(updated["grid"]["size"])
        if value not in {f"uav_r{r}_c{c}" for r in range(size) for c in range(size)}:
            raise ValueError("initial_candidate is outside the UAV grid")
        region.setdefault("uav_modes", {})["initial_candidate"] = value
    if "support_vehicles" in changes:
        incoming = changes["support_vehicles"]
        if not isinstance(incoming, list) or len(incoming) > 100:
            raise ValueError("support_vehicles must be a list of at most 100 vehicles")
        original = {str(spec["role"]): spec for spec in region["support_vehicles"]}
        new_vehicles = []
        seen = set()
        for item in incoming:
            if not isinstance(item, dict):
                raise ValueError("each vehicle must be an object")
            role = str(item.get("role", ""))
            if (not role.startswith("regional_") or not role.replace("_", "").isalnum()
                    or len(role) > 64 or role in seen):
                raise ValueError(f"invalid or duplicate vehicle role: {role}")
            seen.add(role)
            route = str(item.get("route", ""))
            if route not in ALLOWED_ROUTES:
                raise ValueError(f"unsupported route: {route}")
            old = copy.deepcopy(original.get(role, {}))
            old.update({
                "role": role,
                "route": route,
                "start_offset_m": finite_number(item.get("start_offset_m"), role + " start", 1, 500),
                "speed_difference_pct": finite_number(item.get("speed_difference_pct", 0), role + " speed", -80, 100),
                "required": bool(item.get("required", False)),
            })
            new_vehicles.append(old)
        region["support_vehicles"] = new_vehicles
    validate_config(updated)
    return updated


def pixel_to_ground(
    pixel: tuple[float, float], camera_to_world: np.ndarray,
    width: int, height: int, fov: float, ground_z: float,
) -> np.ndarray:
    """Intersect the CARLA RGB camera ray with the road-height plane."""
    u, v = pixel
    focal = width / (2 * math.tan(math.radians(fov) / 2))
    # CARLA's local camera axes are forward X, right Y, up Z.
    ray_local = np.asarray([1.0, (u - width / 2) / focal, -(v - height / 2) / focal])
    matrix = np.asarray(camera_to_world, dtype=np.float64)
    origin = matrix[:3, 3]
    direction = matrix[:3, :3] @ ray_local
    if abs(direction[2]) < 1e-8:
        raise ValueError("the selected point does not intersect the ground")
    distance = (ground_z - origin[2]) / direction[2]
    if distance <= 0:
        raise ValueError("the selected point is behind the camera")
    return origin + distance * direction


def offsets_for_pixel(
    pixel: tuple[float, float], camera_to_world: np.ndarray,
    width: int, height: int, fov: float, center: list[float],
    forward: list[float], right: list[float],
) -> dict[str, float]:
    point = pixel_to_ground(pixel, camera_to_world, width, height, fov, center[2])
    delta = point[:2] - np.asarray(center[:2])
    return {"forward_m": round(float(np.dot(delta, forward)), 2),
            "right_m": round(float(np.dot(delta, right)), 2),
            "world_xyz": [round(float(p), 2) for p in point]}


class SceneEditor:
    def __init__(self, config_path: str | Path, draft_path: str | Path | None = None):
        self.source = Path(config_path).expanduser().resolve()
        self.draft = (Path(draft_path).expanduser().resolve() if draft_path else
                      self.source.with_name(self.source.stem + ".editor.yaml"))
        if self.draft == self.source:
            raise ValueError("draft path must differ from the source config")
        self.cfg = load_config(self.source)
        if "regional" not in self.cfg:
            raise ValueError("the editor requires a regional scenario config")
        self.lock = threading.RLock()
        self.requests: queue.Queue[tuple[str, Any, queue.Queue[Any]]] = queue.Queue()
        self.stopping = threading.Event()
        self.playing = False
        self.state: dict[str, Any] = {"status": "starting", "playing": False}
        self.jpeg = b""
        self.client = self.world = self.traffic_manager = self.scenario = self.rig = None
        self.camera = self.corridor = self.layout = None
        self.junction_infos = self.pair_candidates = self.available_maps = None
        self.old_settings = None
        self.routes = {}
        self.trails: dict[str, list[list[float]]] = {}
        self.tick_index = 0
        self.thread = threading.Thread(target=self.run, name="carla-editor-tick-owner", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def command(self, name: str, payload: Any = None) -> Any:
        reply: queue.Queue[Any] = queue.Queue(maxsize=1)
        self.requests.put((name, payload, reply))
        try:
            okay, result = reply.get(timeout=120)
        except queue.Empty as error:
            raise TimeoutError("CARLA did not answer the editor command in 120 seconds") from error
        if not okay:
            raise ValueError(result)
        return result

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return copy.deepcopy(self.state)

    def image(self) -> bytes:
        with self.lock:
            return self.jpeg

    def run(self) -> None:
        try:
            self._setup()
            next_tick = time.monotonic()
            while not self.stopping.is_set():
                timeout = max(0.0, next_tick - time.monotonic()) if self.playing else 0.2
                try:
                    name, payload, reply = self.requests.get(timeout=timeout)
                except queue.Empty:
                    if self.playing:
                        self._tick()
                        next_tick = time.monotonic() + float(self.cfg["carla"]["fixed_delta_seconds"])
                    continue
                try:
                    result = self._handle(name, payload)
                    reply.put((True, result))
                except Exception as error:
                    if self.scenario is None:
                        with self.lock:
                            self.state.update(status="error", error=str(error), playing=False)
                    reply.put((False, str(error)))
                next_tick = time.monotonic()
        except Exception as error:
            with self.lock:
                self.state = {**self.state, "status": "error", "error": str(error), "playing": False}
            self._drain_requests(str(error))
        finally:
            self._cleanup(final=True)

    def _drain_requests(self, message: str) -> None:
        while not self.requests.empty():
            try:
                _, _, reply = self.requests.get_nowait()
                reply.put_nowait((False, message))
            except queue.Empty:
                break

    def _setup(self) -> None:
        import carla

        config = self.cfg
        cc = config["carla"]
        if self.client is None:
            self.client = carla.Client(str(cc["host"]), int(cc["port"]))
        self.client.set_timeout(float(cc["timeout_s"]))
        available = {entry.rsplit("/", 1)[-1] for entry in self.client.get_available_maps()}
        if str(cc["map"]) not in available:
            raise ValueError(f"CARLA map {cc['map']} is not installed on this server")
        self.world = self.client.get_world()
        if not self.world.get_map().name.endswith(str(cc["map"])):
            self.world = self.client.load_world(str(cc["map"]))
        # Keep the pre-editor settings across Apply & Restart. Otherwise the
        # second setup saves our own synchronous settings as the "original".
        if self.old_settings is None:
            self.old_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = float(cc["fixed_delta_seconds"])
        settings.no_rendering_mode = False
        self.world.apply_settings(settings)
        self.traffic_manager = self.client.get_trafficmanager(int(cc["traffic_manager_port"]))
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(int(cc["seed"]))
        region = config["regional"]
        self.available_maps = [name for name in AIRV2X_TOWNS if name in available]
        carla_map = self.world.get_map()
        self.junction_infos = rank_junctions(carla_map)
        self.pair_candidates = rank_connected_junctions(
            carla_map, float(region["min_junction_separation_m"]),
            float(region["max_junction_separation_m"]),
        )
        if not self.pair_candidates:
            raise ValueError("this town has no straight, lane-connected two-junction route")
        first_id, second_id = region.get("junction_1_id"), region.get("junction_2_id")
        if first_id is not None and second_id is not None:
            self.corridor = next((pair for pair in self.pair_candidates
                                  if pair.first.junction_id == int(first_id)
                                  and pair.second.junction_id == int(second_id)), None)
            if self.corridor is None:
                raise ValueError(f"no legal route between J1 #{first_id} and J2 #{second_id}")
        else:
            rank = int(region.get("corridor_rank", 0))
            self.corridor = self.pair_candidates[rank]
        self.layout = build_regional_layout(self.corridor, config)
        self.scenario = RegionalIntersectionScenario(
            self.client, self.world, self.traffic_manager, config, self.corridor
        )
        self.scenario.setup()
        self.routes = self.scenario.metadata()["routes"]
        self.rig = SensorRig(self.world, float(cc["fixed_delta_seconds"]))
        self.camera = self._add_map_camera()
        frame = -1
        for _ in range(max(1, int(round(float(region.get("warmup_s", 0)) / float(cc["fixed_delta_seconds"]))))):
            self.scenario.apply_fixed_signal_plan(0.0)
            frame = int(self.world.tick())
            self.scenario.apply_fixed_signal_plan(0.0)
        self.scenario.assert_fixed_signal_plan(0.0)
        self.tick_index = 0
        self.trails = {}
        self.playing = False
        self._present(frame)

    def _add_map_camera(self) -> Any:
        """Show the complete junction network so a new service route can be chosen."""
        import carla

        points = np.asarray([item.center for item in self.junction_infos], dtype=np.float64)
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        width, height, fov = 1600, 1200, 100.0
        half_horizontal = math.tan(math.radians(fov) / 2)
        half_vertical = half_horizontal * height / width
        extent = np.max(np.abs(points[:, :2] - center[:2]), axis=0) + 40
        altitude = max(float(self.cfg["regional"]["bev_camera"]["height_m"]),
                       float(extent[0] / half_vertical), float(extent[1] / half_horizontal))
        return self.rig.add_camera(
            "global_bev_rgb",
            carla.Transform(carla.Location(x=float(center[0]), y=float(center[1]),
                                           z=float(center[2] + altitude)),
                            carla.Rotation(pitch=-90.0)),
            {"image_size_x": width, "image_size_y": height, "fov": fov},
        )

    def _handle(self, name: str, payload: Any) -> Any:
        if self.scenario is None and name not in ("restart", "apply_restart"):
            raise RuntimeError("scene is not running; fix the draft and restart")
        if name == "pause":
            self.playing = False
        elif name == "play":
            if self.state.get("status") == "done":
                raise ValueError("clip ended; use Restart to play again")
            self.playing = True
        elif name == "step":
            self.playing = False
            self._tick()
        elif name == "step_second":
            self.playing = False
            count = int(round(1.0 / float(self.cfg["carla"]["fixed_delta_seconds"])))
            for _ in range(count):
                if self.state.get("status") == "done":
                    break
                self._tick()
        elif name in ("restart", "apply_restart"):
            if name == "apply_restart":
                new_cfg = edited_config(self.cfg, payload or {})
                if new_cfg["carla"]["map"] not in self.available_maps:
                    raise ValueError(f"map {new_cfg['carla']['map']} is not installed on the CARLA server")
                if new_cfg["carla"]["map"] == self.cfg["carla"]["map"]:
                    j1 = new_cfg["regional"].get("junction_1_id")
                    j2 = new_cfg["regional"].get("junction_2_id")
                    if j1 is not None and not any(
                        candidate.first.junction_id == j1 and candidate.second.junction_id == j2
                        for candidate in self.pair_candidates
                    ):
                        raise ValueError(f"J1 #{j1} → J2 #{j2} is not a connected straight route")
                # Validate, then write an atomic standalone draft. Never touch the source YAML.
                self.draft.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.draft.with_name(self.draft.name + ".tmp")
                dump_effective_config(new_cfg, temporary)
                temporary.replace(self.draft)
                self.cfg = new_cfg
            self.playing = False
            self._cleanup(final=False)
            try:
                self._setup()
            except Exception:
                self._cleanup(final=False)
                raise
        elif name == "place":
            if not isinstance(payload, dict) or payload.get("target") not in ("rsu", "uav", "ego"):
                raise ValueError("choose an editor placement tool before clicking the map")
            width = int(self.camera.attributes["image_size_x"])
            height = int(self.camera.attributes["image_size_y"])
            u = finite_number(payload.get("u"), "image u", 0, width)
            v = finite_number(payload.get("v"), "image v", 0, height)
            if payload["target"] == "ego":
                point = pixel_to_ground(
                    (u, v), np.asarray(self.camera.get_transform().get_matrix()),
                    width, height, float(self.camera.attributes["fov"]),
                    float(self.corridor.first.center[2]),
                )
                length = float(self.cfg["regional"].get("ego_route_tail_m", 180)) + 300
                route = extend_route_straight(self.corridor.corridor_waypoints, length)
                xyz = [[wp.transform.location.x, wp.transform.location.y,
                        wp.transform.location.z] for wp in route]
                distance, error = nearest_route_distance(point, xyz)
                if error > 8.0 or distance > 300.0:
                    raise ValueError("Ego start must lie near the selected legal route, within 300 m of its start")
                return {"advance_m": round(distance, 2), "lateral_error_m": round(error, 2)}
            center = (self.corridor.first.center if payload["target"] == "rsu" else
                      self.corridor.second.center)
            return offsets_for_pixel(
                (u, v), np.asarray(self.camera.get_transform().get_matrix()),
                width, height, float(self.camera.attributes["fov"]), center,
                list(self.corridor.forward_xy), list(self.corridor.right_xy),
            )
        else:
            raise ValueError(f"unknown command: {name}")
        with self.lock:
            self.state["playing"] = self.playing
        return {"playing": self.playing, "draft": str(self.draft)}

    def _tick(self) -> None:
        dt = float(self.cfg["carla"]["fixed_delta_seconds"])
        if (self.tick_index + 1) * dt >= float(self.cfg["regional"]["duration_s"]) - 1e-6:
            self.playing = False
            with self.lock:
                self.state.update(status="done", playing=False)
            return
        self.tick_index += 1
        elapsed = self.tick_index * dt
        self.scenario.apply_fixed_signal_plan(elapsed)
        frame = int(self.world.tick())
        self.scenario.apply_fixed_signal_plan(elapsed)
        self.scenario.assert_fixed_signal_plan(elapsed)
        self._present(frame)

    def _project(self, points: Any) -> list[list[float] | None]:
        width = int(self.camera.attributes["image_size_x"])
        height = int(self.camera.attributes["image_size_y"])
        uv, valid = project_world_to_camera(
            np.asarray(points, dtype=np.float64).reshape(-1, 3),
            np.asarray(self.camera.get_transform().get_inverse_matrix()), width, height,
            float(self.camera.attributes["fov"]),
        )
        return [[round(float(x), 1), round(float(y), 1)] if ok else None
                for (x, y), ok in zip(uv, valid)]

    def _present(self, frame: int) -> None:
        from PIL import Image

        measurement = self.rig.collect_frame(frame, timeout_s=20)["global_bev_rgb"]
        width, height = int(measurement.width), int(measurement.height)
        bgra = np.frombuffer(measurement.raw_data, dtype=np.uint8).reshape(height, width, 4)
        image = Image.fromarray(np.ascontiguousarray(bgra[:, :, 2::-1]), "RGB")
        data = io.BytesIO()
        image.save(data, format="JPEG", quality=80)

        actors = self.scenario.actor_states()
        projected = self._project([a["location"] for a in actors])
        for actor, uv in zip(actors, projected):
            actor["uv"] = uv
            actor["speed_mps"] = round(float(np.linalg.norm(actor["velocity"][:2])), 2)
            role = actor["role_name"]
            if role and role != "regional_background" and uv is not None:
                history = self.trails.setdefault(role, [])
                if not history or math.dist(uv, history[-1]) >= 1:
                    history.append(uv)
                    if len(history) > 400:
                        del history[0]
        for record in actors:
            record.pop("bbox_extent", None)
        route_uv = {}
        for role, route in self.routes.items():
            points = route["xyz"][::4]
            route_uv[role] = self._project(points)
        light_states = self.scenario.fixed_signal_states()
        lights = []
        for light in light_states:
            entry = dict(light)
            record = self.scenario._fixed_signal_states[int(light["traffic_light_id"])]
            entry["uv"] = self._project([[float(record["actor"].get_location().x),
                                         float(record["actor"].get_location().y),
                                         float(record["actor"].get_location().z)]])[0]
            lights.append(entry)
        layout = self.layout
        region = self.cfg["regional"]
        center_name = str(region.get("uav_modes", {}).get("initial_candidate", "uav_r2_c2"))
        grid = layout["grid"]
        selected = next((item for item in grid if item["name"] == center_name), grid[len(grid) // 2])
        ranges = _sensor_range_summary(self.cfg, self.corridor, grid)
        map_junctions = self._project([item.center for item in self.junction_infos])
        names = {item.junction_id: f"路口 {i + 1}"
                 for i, item in enumerate(self.junction_infos)}
        state = {
            "status": "running" if self.playing else "paused",
            "playing": self.playing,
            "time_s": round(self.tick_index * float(self.cfg["carla"]["fixed_delta_seconds"]), 1),
            "duration_s": float(region["duration_s"]),
            "carla_frame": frame,
            "image_width": width, "image_height": height,
            "image_version": frame,
            "actors": actors, "routes": route_uv, "trails": self.trails,
            "lights": lights, "sensor_ranges": ranges,
            "map_junctions": [
                {"id": item.junction_id, "name": names[item.junction_id], "uv": uv,
                 "world_xyz": list(item.center)}
                for item, uv in zip(self.junction_infos, map_junctions)
            ],
            "connected_pairs": [
                {"first": item.first.junction_id, "second": item.second.junction_id,
                 "name": f"{names[item.first.junction_id]} → {names[item.second.junction_id]}"}
                for item in self.pair_candidates
            ],
            "available_maps": self.available_maps,
            "agent_roles": list(region.get("agent_sensors", {"regional_ego": "ego_lidar", "regional_car3": "cav2_lidar"})),
            "junctions": self._project([self.corridor.first.center, self.corridor.second.center]),
            "junction_distance_m": float(np.linalg.norm(
                np.asarray(self.corridor.first.center[:2]) -
                np.asarray(self.corridor.second.center[:2])
            )),
            "rsu_uv": self._project([layout["rsu_xyz"]])[0],
            "uav_uv": self._project([selected["location"]])[0],
            "grid": [{"name": item["name"], "uv": uv} for item, uv in zip(
                grid, self._project([entry["location"] for entry in grid]))],
            "config": {
                "map_name": str(self.cfg["carla"]["map"]),
                "junction_1_id": self.corridor.first.junction_id,
                "junction_2_id": self.corridor.second.junction_id,
                "ego_start_advance_m": float(region.get("ego_start_advance_m", 0)),
                "ego_speed_difference_pct": float(region.get("ego_speed_difference_pct", 0)),
                "car3_speed_difference_pct": float(region.get("car3_speed_difference_pct", 0)),
                "car3_start_before_junction_m": float(region["car3_start_before_junction_m"]),
                "background_vehicle_count": int(region["background_vehicle_count"]),
                "j2_switch_time_s": float(region.get("fixed_signal_plan", {}).get("j2_switch_time_s", 20)),
                "rsu_forward_m": float(region["rsu"]["offset_forward_m"]),
                "rsu_right_m": float(region["rsu"]["offset_right_m"]),
                "uav_forward_m": float(region.get("uav_grid_offset_forward_m", 0)),
                "uav_right_m": float(region.get("uav_grid_offset_right_m", 0)),
                "initial_candidate": center_name,
                "support_vehicles": [{key: item.get(key) for key in
                    ("role", "route", "start_offset_m", "speed_difference_pct", "required")}
                    for item in region["support_vehicles"]],
            },
            "draft_path": str(self.draft),
        }
        with self.lock:
            self.jpeg = data.getvalue()
            self.state = state

    def _cleanup(self, final: bool) -> None:
        try:
            if self.rig:
                self.rig.stop()
            if self.scenario:
                self.scenario.release_fixed_signal_plan()
            if self.client and self.world:
                import carla
                ids = ([int(stream.actor.id) for stream in self.rig.streams.values()]
                       if self.rig else [])
                ids += [int(actor.id) for actor in self.scenario.actors] if self.scenario else []
                if ids:
                    self.client.apply_batch_sync(
                        [carla.command.DestroyActor(actor_id) for actor_id in dict.fromkeys(ids)],
                        False,
                    )
            if self.rig:
                self.rig.streams.clear()
            if self.scenario:
                self.scenario.actors.clear()
                self.scenario.routes.clear()
        finally:
            self.rig = self.camera = self.scenario = None
            if final and self.world is not None:
                if self.traffic_manager:
                    try:
                        self.traffic_manager.set_synchronous_mode(False)
                    except Exception:
                        pass
                if self.old_settings:
                    try:
                        self.world.apply_settings(self.old_settings)
                    except Exception:
                        pass


def make_handler(editor: SceneEditor) -> type[BaseHTTPRequestHandler]:
    html_file = Path(__file__).with_name("regional_editor.html")

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, body: bytes, mimetype: str, code: int = 200) -> None:
            self.send_response(code)
            self.send_header("Content-Type", mimetype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path == "/":
                self._reply(html_file.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/state":
                self._reply(json.dumps(editor.snapshot()).encode("utf-8"), "application/json")
            elif path == "/api/image":
                data = editor.image()
                self._reply(data, "image/jpeg" if data else "text/plain",
                            200 if data else HTTPStatus.SERVICE_UNAVAILABLE)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if urlsplit(self.path).path != "/api/control":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("invalid command size")
                request = json.loads(self.rfile.read(length).decode("utf-8"))
                result = editor.command(str(request["command"]), request.get("payload"))
                self._reply(json.dumps({"ok": True, "result": result}).encode(), "application/json")
            except (ValueError, KeyError, TimeoutError) as error:
                self._reply(json.dumps({"ok": False, "error": str(error)}).encode(),
                            "application/json", HTTPStatus.BAD_REQUEST)

        def log_message(self, format: str, *args: Any) -> None:
            if not self.path.startswith("/api/state"):
                super().log_message(format, *args)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Live regional scenario BEV editor")
    parser.add_argument("--config", required=True, help="source YAML, never modified")
    parser.add_argument("--draft", help="editable YAML output (defaults to <config>.editor.yaml)")
    parser.add_argument("--web-port", type=int, default=8765)
    args = parser.parse_args()
    editor = SceneEditor(args.config, args.draft)
    editor.start()
    server = ThreadingHTTPServer(("127.0.0.1", args.web_port), make_handler(editor))
    print(f"Live scene editor: http://127.0.0.1:{args.web_port}/", flush=True)
    print(f"Edits will be saved to: {editor.draft}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        editor.stopping.set()
        editor.thread.join(timeout=20)


if __name__ == "__main__":
    main()
