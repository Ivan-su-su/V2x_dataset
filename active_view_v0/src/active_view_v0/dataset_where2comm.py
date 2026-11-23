"""Feed live, same-frame LiDAR directly to the existing detector runtime."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .dataset_runtime import Decision, Observation, Policy
from .geometry import carla_matrix
from .where2comm_eval import Where2commRuntime, _clip_points, carla_points_to_ego, resolve_checkpoint


def detector_policy(
    policy: Policy, *, opencood_root: Path, model_dir: Path, device: str,
) -> tuple[Policy, dict]:
    config = (model_dir / "config.yaml").resolve()
    weights = resolve_checkpoint(model_dir.resolve(), None)
    runtime = Where2commRuntime(opencood_root=opencood_root, config_path=config,
                                checkpoint_path=weights, device=device)

    def run(obs: Observation) -> Decision:
        ego = carla_matrix(obs.sensors["ego_lidar"].transform)
        groups: dict[str, list[np.ndarray]] = {"vehicle": [], "rsu": [], "drone": []}
        for sensor_name in sorted(obs.sensors, key=lambda x: (x != "ego_lidar", x)):
            data = obs.sensors[sensor_name]
            raw = np.frombuffer(data.raw_data, dtype="<f4").reshape(-1, 4)
            groups[obs.sensor_agent_types[sensor_name]].append(_clip_points(
                carla_points_to_ego(raw, carla_matrix(data.transform), ego), runtime.lidar_range
            ))
        detection = runtime.infer(groups)
        action = policy(obs)
        return Decision(action.dx_m, action.dy_m, {
            "frame": "source_ego_lidar_x_forward_y_left_z_up",
            **{key: value.tolist() for key, value in detection.items()},
        })

    def file_sha(path: Path) -> str:
        sha = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                sha.update(chunk)
        return sha.hexdigest()

    return run, {"config_sha256": file_sha(config), "checkpoint_sha256": file_sha(weights),
                 "checkpoint_name": weights.name, "device": device}
