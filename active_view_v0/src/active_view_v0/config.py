from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .weather import validate_weather


DEFAULTS: dict[str, Any] = {
    "carla": {
        "host": "127.0.0.1",
        "port": 2000,
        "timeout_s": 30.0,
        "map": "Town03",
        "fixed_delta_seconds": 0.1,
        "traffic_manager_port": 8000,
        "seed": 42,
        "weather": {"preset": "ClearNoon", "overrides": {}},
    },
    "output": {
        "root": "./datasets/active_view_v0",
        "run_name": "town03_pilot_001",
    },
    "scenario": {
        "junction_id": None,
        "junction_rank": 0,
        "duration_s": 30.0,
        "ego_speed_mps": 6.0,
        "ego_start_distance_m": 35.0,
        "route_exit_distance_m": 45.0,
        "roi_half_extent_m": 50.0,
        "event_1": {"start_s": 5.0, "duration_s": 6.0, "side": 1.0},
        "event_2": {"start_s": 14.0, "duration_s": 6.0, "side": -1.0},
    },
    "grid": {
        "size": 5,
        "spacing_m": 10.0,
        "height_m": 25.0,
        "keyframe_interval_s": 2.0,
    },
    "sensors": {
        "vehicle_lidar": {
            "mount": {"x": 0.0, "y": 0.0, "z": 1.8, "pitch": 0.0, "yaw": 0.0, "roll": 0.0},
            "channels": 32,
            "range": 80.0,
            "points_per_second": 200000,
            "rotation_frequency": 10.0,
            "upper_fov": 10.0,
            "lower_fov": -30.0,
        },
        "rsu_lidar": {
            "offset_forward_m": 18.0,
            "offset_right_m": 18.0,
            "height_m": 6.0,
            "channels": 64,
            "range": 100.0,
            "points_per_second": 300000,
            "rotation_frequency": 10.0,
            "upper_fov": 10.0,
            "lower_fov": -40.0,
        },
        "uav_lidar": {
            "channels": 64,
            "range": 80.0,
            "points_per_second": 200000,
            "rotation_frequency": 10.0,
            "upper_fov": -5.0,
            "lower_fov": -90.0,
        },
        "preview_cameras": {
            "enabled": True,
            "image_size_x": 960,
            "image_size_y": 540,
            "fov": 90.0,
        },
    },
    "scoring": {
        "base_sensors": ["vehicle_lidar", "rsu_lidar"],
        "ego_roi": {"min": [-140.8, -40.0], "max": [140.8, 40.0]},
        "threshold_points": {"car": 8, "pedestrian": 3, "two_wheeler": 5},
        "class_weights": {"car": 1.0, "pedestrian": 2.0, "two_wheeler": 1.5},
        "ignore_roles": ["active_view_ego", "active_view_occluder"],
        "box_margin_m": 0.15,
        "gain_weight": 0.5,
        "quality_tau_points": {"car": 40.0, "pedestrian": 12.0, "two_wheeler": 20.0},
    },
    "oracle": {
        "uav_speed_mps": 5.0,
        "movement_penalty_per_m": 0.002,
    },
}


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        user = yaml.safe_load(handle) or {}
    cfg = _deep_update(copy.deepcopy(DEFAULTS), user)
    cfg["_config_path"] = str(path)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    validate_weather(cfg["carla"].get("weather", {"preset": "ClearNoon"}))
    dt = float(cfg["carla"]["fixed_delta_seconds"])
    key_dt = float(cfg["grid"]["keyframe_interval_s"])
    size = int(cfg["grid"]["size"])
    if dt <= 0 or key_dt <= 0:
        raise ValueError("fixed_delta_seconds and keyframe_interval_s must be positive")
    ratio = key_dt / dt
    if abs(ratio - round(ratio)) > 1e-6:
        raise ValueError("keyframe_interval_s must be an integer multiple of fixed_delta_seconds")
    if size < 1 or size % 2 == 0:
        raise ValueError("grid.size must be a positive odd number")
    if float(cfg["grid"]["spacing_m"]) <= 0 or float(cfg["grid"]["height_m"]) <= 0:
        raise ValueError("grid spacing and height must be positive")
    max_step = key_dt * float(cfg["oracle"]["uav_speed_mps"])
    if float(cfg["grid"]["spacing_m"]) > max_step + 1e-6:
        raise ValueError(
            "grid.spacing_m exceeds UAV travel distance per decision; "
            "increase oracle.uav_speed_mps or decision interval"
        )
    if float(cfg["scenario"]["duration_s"]) < key_dt:
        raise ValueError("scenario.duration_s must cover at least one keyframe")
    if "regional" in cfg and float(cfg["regional"].get("duration_s", key_dt)) < key_dt:
        raise ValueError("regional.duration_s must cover at least one keyframe")
    if "regional" in cfg:
        regional = cfg["regional"]
        if float(regional.get("ego_start_advance_m", 0.0)) < 0.0:
            raise ValueError("regional.ego_start_advance_m cannot be negative")
        collection_dt = float(regional.get("collection_interval_s", key_dt))
        if collection_dt <= 0:
            raise ValueError("regional.collection_interval_s must be positive")
        collection_ratio = collection_dt / dt
        if abs(collection_ratio - round(collection_ratio)) > 1e-6:
            raise ValueError(
                "regional.collection_interval_s must be an integer multiple of fixed_delta_seconds"
            )
        if regional.get("frame_count") is not None and int(regional["frame_count"]) < 1:
            raise ValueError("regional.frame_count must be positive")
        mode_cfg = regional.get("uav_modes", {})
        decision_dt = float(mode_cfg.get("decision_interval_s", key_dt))
        decision_ratio = decision_dt / collection_dt
        if decision_dt <= 0 or abs(decision_ratio - round(decision_ratio)) > 1e-6:
            raise ValueError(
                "regional.uav_modes.decision_interval_s must be a positive integer "
                "multiple of collection_interval_s"
            )
        if abs(decision_dt - key_dt) > 1e-6:
            raise ValueError(
                "grid.keyframe_interval_s must equal the UAV decision interval for "
                "backward-compatible oracle configuration"
            )
        roles = [str(item.get("role", "")) for item in regional.get("support_vehicles", [])]
        if not roles or any(not role for role in roles):
            raise ValueError("regional.support_vehicles requires non-empty role names")
        if len(roles) != len(set(roles)):
            raise ValueError("regional.support_vehicles role names must be unique")
        if any(float(item.get("start_offset_m", 0.0)) <= 0 for item in regional["support_vehicles"]):
            raise ValueError("all support-vehicle start offsets must be positive")
        allowed_routes = {
            "j1_cross",
            "j2_cross",
            "j2_cross_reverse",
            "corridor_forward",
            "corridor_behind_ego",
            "corridor_oncoming",
        }
        unknown_routes = sorted(
            {
                str(item.get("route", ""))
                for item in regional["support_vehicles"]
            }.difference(allowed_routes)
        )
        if unknown_routes:
            raise ValueError(f"unknown regional support-vehicle routes: {unknown_routes}")
        validation = regional.get("validation", {})
        early_end = float(validation.get("early_end_s", 14.0))
        late_start = float(validation.get("late_start_s", 16.0))
        duration = float(regional.get("duration_s", key_dt))
        if not 0.0 <= early_end < late_start <= duration:
            raise ValueError("regional.validation requires 0 <= early_end < late_start <= duration")
        if int(validation.get("minimum_late_targets", 1)) < 1:
            raise ValueError("regional.validation.minimum_late_targets must be positive")
        agent_sensors = regional.get("agent_sensors")
        if agent_sensors is not None:
            if not isinstance(agent_sensors, dict) or not agent_sensors:
                raise ValueError("regional.agent_sensors must be a non-empty role-to-sensor mapping")
            if "regional_ego" not in agent_sensors:
                raise ValueError("regional.agent_sensors must include regional_ego")
            sensor_names = [str(value) for value in agent_sensors.values()]
            if len(sensor_names) != len(set(sensor_names)):
                raise ValueError("regional.agent_sensors sensor names must be unique")
    ego_roi = cfg["scoring"].get("ego_roi", {})
    roi_min = ego_roi.get("min", [])
    roi_max = ego_roi.get("max", [])
    if len(roi_min) < 2 or len(roi_max) < 2:
        raise ValueError("scoring.ego_roi min/max must contain at least x and y")
    if any(float(lo) >= float(hi) for lo, hi in zip(roi_min[:2], roi_max[:2])):
        raise ValueError("scoring.ego_roi min must be smaller than max")
    roi_mode = str(cfg["scoring"].get("roi_mode", "ego_moving"))
    if roi_mode not in {"ego_moving", "regional_fixed"}:
        raise ValueError("scoring.roi_mode must be ego_moving or regional_fixed")
    if roi_mode == "regional_fixed":
        regional_roi = cfg["scoring"].get("regional_roi", {})
        for key in ("forward_m", "right_m"):
            values = regional_roi.get(key, [])
            if len(values) != 2 or float(values[0]) >= float(values[1]):
                raise ValueError(
                    f"scoring.regional_roi.{key} must contain increasing [min, max]"
                )
    tau = cfg["scoring"].get("quality_tau_points", {})
    if any(float(value) <= 0 for value in tau.values()):
        raise ValueError("all scoring.quality_tau_points values must be positive")


def dump_effective_config(cfg: dict[str, Any], path: str | Path) -> None:
    clean = {key: value for key, value in cfg.items() if not key.startswith("_")}
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(clean, handle, sort_keys=False, allow_unicode=True)
