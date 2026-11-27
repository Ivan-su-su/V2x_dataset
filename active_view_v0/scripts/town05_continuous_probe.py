"""Scripted reachability probe for Town05/scene_0020; not an autonomous policy."""

from functools import lru_cache
from pathlib import Path

import yaml

from active_view_v0.dataset_runtime import Decision, Observation
from active_view_v0.dataset_scene import resolve_scene


@lru_cache
def _plan():
    scene = resolve_scene("Town05/scene_0020")
    brief = Path(__file__).resolve().parents[1] / "scenes/Town05/scene_0020/config/brief.yaml"
    probe = yaml.safe_load(brief.read_text())["agents"]["uav"]["continuous_probe"]
    return scene.dt, probe["earliest_departure_s"], probe["waypoint_world_xyz_m"]


def b_waypoint(obs: Observation) -> Decision:
    """Hold A until departure, then request the world-frame B waypoint."""
    dt, departure_s, (x, y, _) = _plan()
    if obs.tick_index * dt < departure_s:
        return Decision()
    uav_x, uav_y, _ = obs.uav_world_xyz
    return Decision(dx_m=x - uav_x, dy_m=y - uav_y)
