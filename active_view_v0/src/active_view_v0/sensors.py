from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class SensorStream:
    name: str
    kind: str
    actor: Any
    queue: queue.Queue[Any]


class SensorRig:
    def __init__(self, world: Any, fixed_delta_seconds: float):
        self.world = world
        self.fixed_delta_seconds = float(fixed_delta_seconds)
        self.streams: dict[str, SensorStream] = {}

    def add_lidar(self, name: str, transform: Any, attributes: dict[str, Any], attach_to: Any = None) -> Any:
        blueprint = self.world.get_blueprint_library().find("sensor.lidar.ray_cast")
        for key in (
            "channels",
            "range",
            "points_per_second",
            "rotation_frequency",
            "upper_fov",
            "lower_fov",
        ):
            if key in attributes and blueprint.has_attribute(key):
                blueprint.set_attribute(key, str(attributes[key]))
        if blueprint.has_attribute("sensor_tick"):
            blueprint.set_attribute("sensor_tick", str(self.fixed_delta_seconds))
        if blueprint.has_attribute("dropoff_general_rate"):
            blueprint.set_attribute("dropoff_general_rate", "0.0")
        if blueprint.has_attribute("noise_stddev"):
            blueprint.set_attribute("noise_stddev", "0.0")
        actor = self.world.spawn_actor(blueprint, transform, attach_to=attach_to)
        data_queue: queue.Queue[Any] = queue.Queue(maxsize=3)

        def callback(measurement: Any) -> None:
            while data_queue.full():
                try:
                    data_queue.get_nowait()
                except queue.Empty:
                    break
            data_queue.put_nowait(measurement)

        actor.listen(callback)
        self.streams[name] = SensorStream(name=name, kind="lidar", actor=actor, queue=data_queue)
        return actor

    def add_camera(
        self,
        name: str,
        transform: Any,
        attributes: dict[str, Any],
        attach_to: Any = None,
    ) -> Any:
        blueprint = self.world.get_blueprint_library().find("sensor.camera.rgb")
        for key in ("image_size_x", "image_size_y", "fov"):
            if key in attributes and blueprint.has_attribute(key):
                blueprint.set_attribute(key, str(attributes[key]))
        if blueprint.has_attribute("sensor_tick"):
            blueprint.set_attribute("sensor_tick", str(self.fixed_delta_seconds))
        actor = self.world.spawn_actor(blueprint, transform, attach_to=attach_to)
        data_queue: queue.Queue[Any] = queue.Queue(maxsize=3)

        def callback(measurement: Any) -> None:
            while data_queue.full():
                try:
                    data_queue.get_nowait()
                except queue.Empty:
                    break
            data_queue.put_nowait(measurement)

        actor.listen(callback)
        self.streams[name] = SensorStream(name=name, kind="camera", actor=actor, queue=data_queue)
        return actor

    def collect_frame(self, target_frame: int, timeout_s: float = 10.0) -> dict[str, Any]:
        return {
            name: self._measurement_for_frame(stream, int(target_frame), timeout_s)
            for name, stream in self.streams.items()
        }

    def close(self) -> None:
        for stream in self.streams.values():
            try:
                stream.actor.stop()
            except Exception:
                pass
        for stream in self.streams.values():
            try:
                stream.actor.destroy()
            except Exception:
                pass
        self.streams.clear()

    @staticmethod
    def _measurement_for_frame(stream: SensorStream, target_frame: int, timeout_s: float) -> Any:
        deadline = time.monotonic() + float(timeout_s)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"sensor {stream.name} did not produce frame {target_frame}")
            try:
                measurement = stream.queue.get(timeout=remaining)
            except queue.Empty as error:
                raise TimeoutError(f"sensor {stream.name} timed out at frame {target_frame}") from error
            frame = int(measurement.frame)
            if frame == target_frame:
                return measurement
            if frame > target_frame:
                raise RuntimeError(
                    f"sensor {stream.name} skipped requested frame {target_frame}; next available is {frame}"
                )
