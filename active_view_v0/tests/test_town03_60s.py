from copy import deepcopy

import numpy as np

from active_view_v0.dataset_scene import resolve_scene
from active_view_v0.scene_health import lane_scene_health


def test_decoded_town03_scene_contract():
    for number in range(3, 18):
        scene = resolve_scene(f"Town03/scene_{number:04d}")
        cfg = scene.config
        regional = cfg["regional"]
        assert scene.frame_count == 600 and scene.dt == .1
        assert cfg["scoring"]["ego_roi"] == {"min": [-140.8, -40.0], "max": [140.8, 40.0]}
        assert not regional["save_overhead_rgb"]
        assert {v["group"] for v in regional["lane_traffic"]["vehicles"].values()} == {
            "ego", "car2", "with", "against", "side",
        }
        assert (regional["density_validation"]["minimum_targets_per_frame"],
                regional["density_validation"]["maximum_targets_per_frame"]) == (7, 20)
    assert resolve_scene("Town03/scene_0001").frame_count == 400
    assert resolve_scene("Town03/scene_0002").frame_count == 400


def test_lane_health_distinguishes_detector_targets_and_cleared_side_traffic():
    cfg = deepcopy(resolve_scene("Town03/scene_0007").config)
    regional = cfg["regional"]
    regional["density_validation"]["minimum_agent_progress_per_quarter_m"] = 5
    roles = {"regional_ego": "ego", "regional_car3": "car2",
             "regional_with_01": "with", "regional_against_01": "against", "regional_side_01": "side"}
    regional["lane_traffic"]["vehicles"] = {k: {"group": g, "junction_id": 356} for k, g in roles.items()}
    xyz = [[i, 0, 0] for i in range(110)]
    routes = {role: {"xyz": xyz, "junction_exit_xyz": [50, 0, 0]} for role in roles}
    ego_lidar = {"transform": {"matrix": np.eye(4).tolist()}}
    target = {"role_name": "ordinary", "class_name": "car", "actor_id": 1,
              "actor_transform": {"matrix": np.eye(4).tolist()},
              "bbox_center_local": [0, 0, -1], "bbox_extent": [2, 1, .8]}
    far = {**target, "bbox_center_local": [0, 41, -1]}
    rows = [
        {"gt": [target] * 7 + [far],
         "sensors": {"ego_lidar": ego_lidar},
         "sensing_agents": {role: {"location": [i, 0, 0], "velocity": [4, 0, 0]} for role in roles}}
        for i in range(100)
    ]
    health = lane_scene_health(rows, cfg, routes)
    assert health["passed"]
    assert health["detector_gt_min"] == health["detector_gt_max"] == 7
    assert health["side_crossings_by_junction"] == {356: 1}

    rows[8]["gt"] = [target] * 6 + [far]
    for row in rows:
        row["sensing_agents"]["regional_car3"] = {"location": [0, 0, 0], "velocity": [0, 0, 0]}
        row["sensing_agents"]["regional_side_01"] = {"location": [0, 0, 0], "velocity": [0, 0, 0]}
    health = lane_scene_health(rows, cfg, routes)
    assert not health["passed"]
    assert health["density_failure_frames"] == [8]
    assert health["side_crossings_by_junction"] == {356: 0}
    assert health["agent_motion"]["regional_car3"]["progress_per_quarter_m"] == [0] * 4
