import json

import numpy as np
import pytest

from active_view_v0.base_ablation import _miss_diagnostics, base_profiles


def test_base_profiles_change_only_ground_sensor_inputs():
    profiles = base_profiles(["ego_lidar", "car1_lidar", "car3_lidar", "rsu_lidar"])
    assert profiles["ego_only"] == ["ego_lidar"]
    assert profiles["ego_rsu"] == ["ego_lidar", "rsu_lidar"]
    assert profiles["ego_cavs"] == ["ego_lidar", "car1_lidar", "car3_lidar"]
    assert profiles["full"] == ["ego_lidar", "car1_lidar", "car3_lidar", "rsu_lidar"]


def test_base_profiles_reject_missing_canonical_sensor():
    with pytest.raises(ValueError, match="rsu_lidar"):
        base_profiles(["ego_lidar", "car1_lidar", "car3_lidar"])


def test_miss_diagnostics_uses_union_threshold_not_uav_alone(tmp_path):
    # Base and UAV can each be below threshold while their combined support
    # crosses it.  The scorer records union_visible explicitly for this case.
    details = [
        {
            "target_ids": [11, 12],
            "target_classes": ["car", "pedestrian"],
            "base_visible": [0, 0],
            "candidates": {
                "uav_r0_c0": {
                    "visible": [0, 0],
                    "union_visible": [1, 0],
                },
                "uav_r0_c1": {
                    "visible": [0, 0],
                    "union_visible": [0, 0],
                },
            },
        }
    ]
    path = tmp_path / "geometry_details.json"
    path.write_text(json.dumps(details), encoding="utf-8")
    result = _miss_diagnostics(path, np.asarray([16.0]), late_start_s=16.0)
    assert result["late_missed_target_instances"] == 2
    assert result["late_rescuable_target_instances"] == 1
    assert result["late_irrecoverable_target_instances"] == 1
    assert result["frames"][0]["rescuable_target_ids"] == [11]
    assert result["frames"][0]["irrecoverable_target_ids"] == [12]
