"""CARLA adapter for the existing regional traffic engine and a single UAV."""

from __future__ import annotations

import hashlib
import os
import queue
import time
from collections import Counter
from contextlib import contextmanager
from typing import Any, Iterator

from .dataset_runtime import Tick
from .dataset_scene import DatasetScene
from .geometry import transform_record
from .regional_collector import _collect_gt
from .regional_layout import build_regional_layout, resolve_regional_corridor
from .regional_scenario import RegionalIntersectionScenario
from .sensors import SensorRig
from .weather import weather_parameters


class CarlaEpisode:
    def __init__(self, carla: Any, client: Any, world: Any, scenario: Any, rig: SensorRig,
                 scene: DatasetScene, layout: dict):
        self.carla, self.world, self.scenario, self.rig = carla, world, scenario, rig
        self.client, self.scene, self.layout = client, scene, layout
        self.sensor_agent_types: dict[str, str] = {}
        self._last_uav = None
        self._actor_names: dict[int, str] = {}
        self._signals: dict[int, str] = {}
        cfg = scene.config
        roles = cfg["regional"]["agent_sensors"]
        if set(cfg["scoring"]["base_sensors"]) != set(roles.values()) | {"rsu_lidar"}:
            raise ValueError("base_sensors must match agent_sensors and rsu_lidar")
        if roles["regional_ego"] != "ego_lidar" or {"rsu_lidar", "uav_lidar"}.intersection(roles.values()):
            raise ValueError("use ego_lidar for Ego and reserve rsu_lidar and uav_lidar")
        mount = cfg["sensors"]["vehicle_lidar"]["mount"]
        extrinsic = carla.Transform(
            carla.Location(x=mount["x"], y=mount["y"], z=mount["z"]),
            carla.Rotation(pitch=mount["pitch"], yaw=mount["yaw"], roll=mount["roll"]),
        )
        for role, name in roles.items():
            route = scenario.routes.get(role)
            if route is None:
                raise RuntimeError(f"required sensing agent missing: {role}")
            rig.add_lidar(name, extrinsic, cfg["sensors"]["vehicle_lidar"], attach_to=route.actor)
            self.sensor_agent_types[name] = "vehicle"
        rig.add_lidar("rsu_lidar", self._transform(layout["rsu_xyz"]), cfg["sensors"]["rsu_lidar"])
        self.sensor_agent_types["rsu_lidar"] = "rsu"
        start = cfg["dataset"]["uav_initial_world_xyz"]
        if start is None:
            # Legacy grid metadata is only used to resolve the approved starting point.
            configured = cfg["regional"]["uav_modes"].get("initial_candidate", layout["grid"][len(layout["grid"]) // 2]["name"])
            by_name = {item["name"]: item["location"] for item in layout["grid"]}
            if configured not in by_name:
                raise ValueError(f"unknown configured UAV starting point: {configured}")
            start = by_name[configured]
        self.initial_uav_xyz = tuple(float(x) for x in start)
        self.uav = rig.add_lidar("uav_lidar", self._transform(self.initial_uav_xyz), cfg["sensors"]["uav_lidar"])
        self.sensor_agent_types["uav_lidar"] = "drone"
        self._last_uav = self.initial_uav_xyz

        self._name_actors()
        for entry in scenario.fixed_signal_metadata() + scenario.native_signal_states():
            actor = world.get_actor(entry["traffic_light_id"])
            xyz = actor.get_location()
            identity = actor.get_opendrive_id()
            self._signals[actor.id] = f"{identity}@{xyz.x:.3f},{xyz.y:.3f},{xyz.z:.3f}"

    def _name_actors(self) -> None:
        counts: Counter[str] = Counter()
        for actor in self.scenario.actors:
            role = actor.attributes.get("role_name", "") or actor.type_id
            counts[role] += 1
            self._actor_names[actor.id] = f"{role}:{counts[role]:03d}"

    def _transform(self, xyz: tuple | list) -> Any:
        return self.carla.Transform(self.carla.Location(x=xyz[0], y=xyz[1], z=xyz[2]))

    def _advance(self, elapsed_s: float) -> tuple[int, float]:
        self.scenario.apply_fixed_signal_plan(elapsed_s)
        wall_s = time.monotonic()
        frame = int(self.world.tick())
        self.scenario.apply_fixed_signal_plan(elapsed_s)
        self.scenario.assert_fixed_signal_plan(elapsed_s)
        return frame, wall_s

    def warm_up(self) -> None:
        count = round(float(self.scene.config["regional"]["warmup_s"]) / self.scene.dt)
        for _ in range(count):
            self._advance(0.0)
            self.poll()

    def tick(self, index: int, uav_xyz: tuple[float, float, float]) -> Tick:
        if self._last_uav != uav_xyz:
            response = self.client.apply_batch_sync([
                self.carla.command.ApplyTransform(self.uav.id, self._transform(uav_xyz))
            ], False)[0]
            if response.error:
                raise RuntimeError(f"CARLA could not apply UAV pose: {response.error}")
            self._last_uav = uav_xyz
        time_s = index * self.scene.dt
        frame, capture_wall_s = self._advance(time_s)
        ego = self.scenario.ego.get_transform()
        scoring = self.scene.config["scoring"]
        gt = _collect_gt(self.world, ego, roi_mode=scoring.get("roi_mode", "ego_moving"),
                         ego_roi=scoring["ego_roi"], regional_roi=self.layout["evaluation_roi"])
        ignored = set(scoring["ignore_roles"])
        gt = [item for item in gt if item["role_name"] not in ignored]
        for item in gt:
            item["logical_actor_id"] = self._actor_names.get(item["actor_id"], f"unmanaged:{item['actor_id']}")
        actors = {
            self._actor_names[state["actor_id"]]: {
                key: state[key] for key in ("type_id", "location", "rotation", "velocity")
            }
            for state in self.scenario.actor_states()
        }
        signals = {
            self._signals[signal["traffic_light_id"]]: {
                key: signal[key] for key in ("junction", "phase", "actual_state", "expected_state")
            }
            for signal in self.scenario.fixed_signal_states() + self.scenario.native_signal_states()
        }
        state = {"actors": actors, "signals": signals, "gt": gt}
        return Tick(index, frame, time_s, capture_wall_s, uav_xyz, transform_record(ego), state)

    def collect(self, frame: int) -> dict[str, Any]:
        return self.rig.collect_frame(frame, timeout_s=10.0)

    def poll(self) -> list[tuple[str, Any]]:
        out = []
        for name, stream in self.rig.streams.items():
            for _ in range(stream.queue.maxsize):
                try:
                    out.append((name, stream.queue.get_nowait()))
                except queue.Empty:
                    break
        return out


@contextmanager
def start_episode(scene: DatasetScene) -> Iterator[tuple[CarlaEpisode, dict]]:
    import carla

    cfg = scene.config
    carla_cfg = cfg["carla"]
    host = os.environ.get("ACTIVEVIEW_CARLA_HOST", carla_cfg["host"])
    port = int(os.environ.get("ACTIVEVIEW_CARLA_PORT", carla_cfg["port"]))
    tm_port = int(os.environ.get("ACTIVEVIEW_TM_PORT", carla_cfg["traffic_manager_port"]))
    # Legacy traffic code reads the TM port from its private settings object.
    effective = {**cfg, "carla": {**carla_cfg, "traffic_manager_port": tm_port}}
    client = carla.Client(host, port)
    client.set_timeout(float(carla_cfg["timeout_s"]))
    world = client.get_world()
    original = world.get_settings()
    tm = scenario = rig = None
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = scene.dt
        settings.no_rendering_mode = bool(cfg["regional"].get("lane_traffic"))
        world.apply_settings(settings)
        target_map = str(carla_cfg["map"])
        if world.get_map().name.rsplit("/", 1)[-1] == target_map:
            world = client.reload_world(False)
        else:
            world = client.load_world(target_map, False)
        carla_map = world.get_map()
        if carla_map.name.rsplit("/", 1)[-1] != target_map:
            raise RuntimeError(f"expected map {target_map}, got {carla_map.name}")
        world.set_weather(weather_parameters(cfg, carla))
        tm = client.get_trafficmanager(tm_port)
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(int(carla_cfg["seed"]))
        regional = cfg["regional"]
        corridor = resolve_regional_corridor(carla_map, regional)
        layout = build_regional_layout(corridor, cfg)
        scenario = RegionalIntersectionScenario(client, world, tm, effective, corridor)
        scenario.setup()
        rig = SensorRig(world, scene.dt)
        episode = CarlaEpisode(carla, client, world, scenario, rig, scene, layout)
        episode.warm_up()
        yield episode, {
            "server_version": client.get_server_version(),
            "client_version": client.get_client_version(),
            "map": carla_map.name,
            "opendrive_sha256": hashlib.sha256(carla_map.to_opendrive().encode()).hexdigest(),
            "corridor": corridor.as_dict(),
            **({"routes": scenario.metadata()["routes"]} if regional.get("lane_traffic") else {}),
            "carla_host": host, "carla_port": port, "traffic_manager_port": tm_port,
        }
    finally:
        if rig is not None:
            rig.stop()
        if scenario is not None:
            scenario.release_fixed_signal_plan()
        ids = [stream.actor.id for stream in rig.streams.values()] if rig else []
        if scenario is not None:
            ids.extend(actor.id for actor in scenario.actors)
        try:
            if ids:
                responses = client.apply_batch_sync([carla.command.DestroyActor(i) for i in ids], False)
                for r in responses:
                    if r.error:
                        print(f"CARLA cleanup: {r.error}", flush=True)
        finally:
            if rig is not None:
                rig.streams.clear()
            try:
                if tm is not None:
                    tm.set_synchronous_mode(False)
            finally:
                world.apply_settings(original)
