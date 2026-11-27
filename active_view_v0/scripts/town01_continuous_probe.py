"""Hold A, then request the scripted B waypoint in Town01/scene_0012."""

from functools import lru_cache

import yaml

from active_view_v0.dataset_runtime import Decision, Observation
from active_view_v0.dataset_scene import PROJECT, resolve_scene

SCENE_ID = "Town01/scene_0012"


@lru_cache
def _plan():
    scene = resolve_scene(SCENE_ID)
    brief_path = PROJECT / "scenes" / SCENE_ID / "config" / "brief.yaml"
    brief = yaml.safe_load(brief_path.read_text(encoding="utf-8"))
    probe = brief["agents"]["uav"]["continuous_probe"]
    return scene.dt, probe["earliest_departure_s"], probe["waypoint_world_xyz_m"]


def b_waypoint(obs: Observation) -> Decision:
    dt, departure_s, (x, y, _) = _plan()
    if obs.tick_index * dt < departure_s:
        return Decision()
    uav_x, uav_y, _ = obs.uav_world_xyz
    return Decision(dx_m=x - uav_x, dy_m=y - uav_y)
