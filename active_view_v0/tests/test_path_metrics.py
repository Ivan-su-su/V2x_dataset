import numpy as np

from active_view_v0.path_metrics import _compare, _evaluate_strategy


def test_path_metrics_count_only_actual_threshold_rescues():
    details = [
        {
            "target_ids": [1, 2],
            "target_classes": ["car", "car"],
            "base_visible": [1, 0],
            "candidates": {
                "uav_0": {"union_visible": [1, 0]},
                "uav_1": {"union_visible": [1, 1]},
            },
        },
        {
            "target_ids": [3, 4],
            "target_classes": ["car", "car"],
            "base_visible": [0, 0],
            "candidates": {
                "uav_0": {"union_visible": [1, 0]},
                "uav_1": {"union_visible": [0, 1]},
            },
        },
    ]
    solution = {"movement_m": 10.0, "switch_count": 1}
    record = _evaluate_strategy(
        details,
        np.asarray([0.0, 2.0]),
        np.asarray([1, 0]),
        ["uav_0", "uav_1"],
        np.asarray([[0.5, 1.0], [0.5, 0.5]]),
        np.asarray([[0.0, 0.5], [0.5, 0.5]]),
        solution,
    )
    assert record["target_instances"] == 4
    assert record["base_missed_instances"] == 3
    assert record["rescued_instances"] == 2
    assert record["union_detected_instances"] == 3
    assert np.isclose(record["target_recall"], 0.75)
    assert np.isclose(record["miss_recovery_rate"], 2.0 / 3.0)


def test_compare_reports_percentage_points_and_extra_rescues():
    fixed = {
        "rescued_instances": 2,
        "target_recall": 0.5,
        "miss_recovery_rate": 0.25,
        "mean_weighted_coverage": 0.6,
        "mean_weighted_gain": 0.1,
    }
    dynamic = {
        "rescued_instances": 4,
        "target_recall": 0.6,
        "miss_recovery_rate": 0.5,
        "mean_weighted_coverage": 0.7,
        "mean_weighted_gain": 0.2,
        "movement_m": 20.0,
        "switch_count": 2,
    }
    result = _compare(dynamic, fixed)
    assert result["additional_rescued_instances"] == 2
    assert np.isclose(result["target_recall_gain_pp"], 10.0)
    assert np.isclose(result["miss_recovery_rate_gain_pp"], 25.0)
