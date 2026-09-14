from __future__ import annotations

import math

import numpy as np

from active_view_v0.where2comm_eval import (
    boxes_hwl_to_corners,
    calculate_global_ap,
    carla_points_to_ego,
    gt_boxes_in_ego,
)


def test_carla_points_are_transformed_and_y_is_flipped() -> None:
    points = np.asarray([[1.0, 2.0, 3.0, 0.4]], dtype=np.float32)
    sensor = np.eye(4)
    sensor[:3, 3] = [10.0, 5.0, 0.0]
    ego = np.eye(4)
    ego[:3, 3] = [7.0, 1.0, 0.0]
    converted = carla_points_to_ego(points, sensor, ego)
    np.testing.assert_allclose(converted[0], [4.0, -6.0, 3.0, 0.4], atol=1e-6)


def test_gt_box_is_hwl_in_ego_coordinates() -> None:
    actor_matrix = np.eye(4)
    actor_matrix[:3, 3] = [10.0, 3.0, 0.0]
    frame = {
        "gt": [
            {
                "actor_id": 7,
                "class_name": "car",
                "role_name": "traffic",
                "actor_transform": {"matrix": actor_matrix.tolist()},
                "bbox_center_local": [1.0, 0.0, 1.0],
                "bbox_extent": [2.0, 1.0, 0.75],
            }
        ]
    }
    boxes, ids, classes = gt_boxes_in_ego(
        frame,
        np.eye(4),
        ignored_roles=[],
        accepted_classes=["car"],
        lidar_range=[-20, -20, -3, 20, 20, 3],
    )
    np.testing.assert_allclose(boxes[0], [11, -3, 1, 1.5, 2, 4, 0], atol=1e-6)
    assert ids == [7]
    assert classes == ["car"]


def test_hwl_corner_extent_and_rotation() -> None:
    boxes = np.asarray([[0, 0, 0, 2, 4, 6, math.pi / 2]], dtype=np.float32)
    corners = boxes_hwl_to_corners(boxes)[0]
    spans = corners.max(axis=0) - corners.min(axis=0)
    np.testing.assert_allclose(spans, [4, 6, 2], atol=1e-5)


def test_global_ap_sorts_across_frames_by_confidence() -> None:
    # A low-confidence TP was collected before a high-confidence FP. Global
    # sorting must put the FP first, yielding VOC AP=0.5 for one GT.
    ap, recall, precision = calculate_global_ap(
        tp=[1, 0], fp=[0, 1], scores=[0.1, 0.9], gt_total=1
    )
    assert abs(ap - 0.5) < 1e-9
    assert recall[-2] == 1.0
    assert precision[-1] == 0.0
