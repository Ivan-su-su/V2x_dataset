"""Resolve named dataset scenes without any workstation-specific paths."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import load_config, validate_config
from .regional_collector import _collection_schedule

PROJECT = Path(__file__).resolve().parents[2]
SCENE_ID = re.compile(r"Town[0-9]{2}(?:HD)?/scene_[0-9]{4,}")
RUN_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,79}")
MODES = ("offline_hover", "online_step", "online_realtime")


def digest(data: Any) -> str:
    encoded = json.dumps(data, sort_keys=True, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def data_root(value: Path | str | None = None) -> Path:
    selected = value if value is not None else os.environ.get("ACTIVEVIEW_DATA_ROOT")
    if selected is None:
        return PROJECT.parent / "datasets"
    path = Path(selected).expanduser()
    if not path.is_absolute():
        raise ValueError("data root must be an absolute path")
    return path.resolve()


def run_directory(root: Path, scene_id: str, mode: str, run_id: str) -> Path:
    if not (SCENE_ID.fullmatch(scene_id) and RUN_ID.fullmatch(run_id)):
        raise ValueError("use Town03/scene_0001 and a lowercase run ID (e.g. hover_001)")
    if mode not in (*MODES, "preview"):
        raise ValueError(f"unknown run mode: {mode}")
    return root / scene_id / mode / run_id


def _overlay(base: dict, updates: dict, prefix: str = "") -> None:
    for key, value in updates.items():
        path = f"{prefix}.{key}" if prefix else key
        if path == "regional.ego_j2_connection" and key not in base:
            base[key] = None
        if key not in base:
            raise ValueError(f"unknown scene field: {path}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{path} must be a mapping")
            if path == "carla.weather.overrides":
                base[key] = copy.deepcopy(value)  # validated by the legacy weather schema
            else:
                _overlay(base[key], value, path)
        else:
            base[key] = copy.deepcopy(value)


@dataclass(frozen=True)
class DatasetScene:
    scene_id: str
    family_id: str
    config: dict
    source_hashes: dict[str, str]

    @property
    def scene_hash(self) -> str:
        return digest(self.config)

    @property
    def dt(self) -> float:
        return float(self.config["carla"]["fixed_delta_seconds"])

    @property
    def frame_count(self) -> int:
        return _collection_schedule(self.config)[2]

    @property
    def decision_ticks(self) -> int:
        return round(float(self.config["regional"]["uav_modes"]["decision_interval_s"]) / self.dt)


def resolve_scene(scene_id: str, project: Path = PROJECT) -> DatasetScene:
    if not SCENE_ID.fullmatch(scene_id):
        raise ValueError("scene ID must look like Town03/scene_0001")
    directory = project / "scenes" / scene_id / "config"
    descriptor = directory / "scene.yaml"
    raw = yaml.safe_load(descriptor.read_text(encoding="utf-8"))
    allowed = {"schema_version", "scene_id", "family_id", "baseline_yaml", "overrides",
               "uav_initial_world_xyz"}
    if (not isinstance(raw, dict) or set(raw) - allowed
            or raw.get("schema_version") != 1 or raw.get("scene_id") != scene_id):
        raise ValueError(f"invalid scene descriptor: {descriptor}")
    family_id = raw.get("family_id")
    if not isinstance(family_id, str) or not RUN_ID.fullmatch(family_id):
        raise ValueError("family_id must use lowercase letters, digits, underscores or dashes")
    baseline_name = Path(raw["baseline_yaml"])
    project = project.resolve()
    baseline = (directory / baseline_name).resolve()
    if baseline_name.is_absolute() or not (
        baseline.is_relative_to((project / "configs").resolve())
        or baseline.is_relative_to(directory.resolve())
    ):
        raise ValueError("baseline YAML must be relative and inside configs/ or the scene config directory")
    cfg = {key: value for key, value in load_config(baseline, validate=False).items()
           if key not in {"_config_path", "output", "recipe"}}
    overrides = raw.get("overrides", {})
    if not isinstance(overrides, dict) or set(overrides) - {"regional", "scoring", "grid", "sensors", "oracle", "carla"}:
        raise ValueError("invalid scene overrides")
    if "carla" in overrides and (
        not isinstance(overrides["carla"], dict)
        or set(overrides["carla"]) - {"weather", "seed", "fixed_delta_seconds"}
    ):
        raise ValueError("CARLA scene overrides allow only weather, seed and fixed_delta_seconds")
    _overlay(cfg, overrides)
    cfg["dataset"] = {"schema_version": 1, "scene_id": scene_id, "family_id": family_id,
                      "uav_initial_world_xyz": raw.get("uav_initial_world_xyz")}
    return _resolved(cfg, {
        str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (descriptor.resolve(), baseline)
    })


def load_frozen_scene(path: str | Path) -> DatasetScene:
    source = Path(path).expanduser().resolve()
    contents = source.read_bytes()
    cfg = yaml.safe_load(contents)
    if not isinstance(cfg, dict) or "dataset" not in cfg or "output" in cfg:
        raise ValueError("pass an effective_config.yaml from activeview run, or use --scene")
    return _resolved(cfg, {source.name: hashlib.sha256(contents).hexdigest()})


def _resolved(cfg: dict, sources: dict[str, str]) -> DatasetScene:
    data = cfg["dataset"]
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("unsupported frozen scene schema")
    scene_id, family_id = data.get("scene_id", ""), data.get("family_id", "")
    if not SCENE_ID.fullmatch(scene_id) or not RUN_ID.fullmatch(family_id):
        raise ValueError("invalid scene ID or family ID")
    validate_config(cfg)
    town = scene_id.split("/")[0]
    if cfg["carla"]["map"] not in (town, f"{town}_Opt"):
        raise ValueError(f"scene is under {town} but uses {cfg['carla']['map']}")
    regional = cfg.get("regional")
    if not isinstance(regional, dict) or not regional.get("agent_sensors"):
        raise ValueError("dataset runtime requires a regional scene with explicit agent_sensors")
    if any(not re.fullmatch(r"[a-z][a-z0-9_]*_lidar", name)
           for name in regional["agent_sensors"].values()):
        raise ValueError("vehicle sensor names must be lowercase identifiers ending in _lidar")
    dt = float(cfg["carla"]["fixed_delta_seconds"])
    ticks, stride, frames = _collection_schedule(cfg)
    period = float(regional.get("uav_modes", {}).get("decision_interval_s", 0))
    max_speed = float(cfg["oracle"]["uav_speed_mps"])
    if (not math.isfinite(dt) or stride != 1 or ticks != frames or frames < 1
            or not math.isfinite(period) or period < dt
            or abs(period / dt - round(period / dt)) > 1e-6
            or not math.isfinite(max_speed) or max_speed <= 0):
        raise ValueError("runtime needs one sample per sim tick and a valid UAV decision period and speed")
    initial = data.get("uav_initial_world_xyz")
    if initial is not None and (
        not isinstance(initial, list) or len(initial) != 3
        or not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in initial)
    ):
        raise ValueError("uav_initial_world_xyz must contain three finite world coordinates in meters")
    return DatasetScene(scene_id, family_id, cfg, sources)
