from pathlib import Path

import numpy as np

from active_view_v0.config import load_config
from active_view_v0.occlusion_report import _blocker_between
from active_view_v0.recovery_report import _phase_stats
from active_view_v0.solve_oracle import _score_matrix


def test_v10_config_is_reachable_at_one_hz():
    config_path = Path(__file__).parents[1] / "configs" / "active_occlusion_town03.yaml"
    cfg = load_config(config_path)
    assert cfg["grid"]["keyframe_interval_s"] == 1.0
    assert cfg["grid"]["spacing_m"] == 15.0
    assert cfg["oracle"]["uav_speed_mps"] == 15.0
    assert cfg["regional"]["agent_sensors"] == {
        "regional_ego": "ego_lidar",
        "regional_car3": "cav2_lidar",
    }
    assert cfg["scoring"]["base_sensors"] == ["ego_lidar", "cav2_lidar", "rsu_lidar"]
    assert cfg["sensors"]["vehicle_lidar"]["channels"] == 80


def test_blocker_must_be_between_and_collinear():
    valid = _blocker_between(
        np.asarray([0.0, 0.0]),
        np.asarray([10.0, 0.5]),
        np.asarray([20.0, 0.0]),
    )
    assert valid["between"]
    lateral = _blocker_between(
        np.asarray([0.0, 0.0]),
        np.asarray([10.0, 5.0]),
        np.asarray([20.0, 0.0]),
    )
    assert not lateral["between"]
    behind = _blocker_between(
        np.asarray([0.0, 0.0]),
        np.asarray([-5.0, 0.0]),
        np.asarray([20.0, 0.0]),
    )
    assert not behind["between"]


def test_phase_stats_counts_unique_rescuable_target_instances():
    details = [
        {
            "base_visible": [0, 1],
            "candidates": {
                "a": {"union_visible": [1, 1]},
                "b": {"union_visible": [1, 1]},
            },
        },
        {
            "base_visible": [0, 0],
            "candidates": {
                "a": {"union_visible": [1, 0]},
                "b": {"union_visible": [0, 1]},
            },
        },
    ]
    record = _phase_stats(details, np.asarray([True, True]))
    assert record["base_missed_instances"] == 3
    assert record["uav_rescuable_instances"] == 3
    assert record["uav_rescue_frames"] == 2


def test_recovery_objective_never_trades_one_rescue_for_quality():
    arrays = {
        "rescued_weight": np.asarray([[1.0, 0.0]]),
        "quality_gain": np.asarray([[0.0, 100.0]]),
    }
    scores, kind = _score_matrix(
        arrays,
        {"oracle": {"recovery_quality_tiebreak": 0.0001}},
        "recovery",
    )
    assert kind == "rescued_target_weight_then_quality_tiebreak"
    assert int(np.argmax(scores[0])) == 0
