from dataclasses import dataclass

import numpy as np

from active_view_v0.geometry import make_regional_roi, points_in_regional_roi
from active_view_v0.regional_layout import build_regional_layout


def test_regional_roi_is_fixed_and_road_aligned():
    roi = make_regional_roi(
        [100.0, 200.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
        {"forward_m": [-10.0, 20.0], "right_m": [-5.0, 5.0]},
    )
    points = np.asarray(
        [
            [100.0, 210.0, 0.0],  # +10m forward
            [94.0, 200.0, 0.0],   # +6m right, outside
            [100.0, 221.0, 0.0],  # +21m forward, outside
        ]
    )
    assert points_in_regional_roi(points, roi).tolist() == [True, False, False]
    assert np.asarray(roi["corners_world"]).shape == (4, 3)


@dataclass
class _Junction:
    center: tuple[float, float, float]


@dataclass
class _Corridor:
    first: _Junction
    second: _Junction
    center: tuple[float, float, float]
    forward_xy: tuple[float, float]
    right_xy: tuple[float, float]


def test_layout_shifts_grid_from_j2_toward_corridor():
    corridor = _Corridor(
        first=_Junction((0.0, 0.0, 0.0)),
        second=_Junction((80.0, 0.0, 0.0)),
        center=(40.0, 0.0, 0.0),
        forward_xy=(1.0, 0.0),
        right_xy=(0.0, 1.0),
    )
    cfg = {
        "regional": {
            "rsu": {"offset_forward_m": 0.0, "offset_right_m": -18.0, "height_m": 7.0},
            "uav_grid_offset_forward_m": -15.0,
            "uav_grid_offset_right_m": 0.0,
        },
        "grid": {"size": 5, "spacing_m": 10.0, "height_m": 25.0},
        "scoring": {
            "roi_mode": "regional_fixed",
            "regional_roi": {
                "forward_m": [-110.0, 110.0],
                "right_m": [-60.0, 60.0],
            },
        },
    }
    layout = build_regional_layout(corridor, cfg)
    assert np.allclose(layout["grid_center_xyz"], [65.0, 0.0, 0.0])
    assert np.allclose(layout["grid"][12]["location"], [65.0, 0.0, 25.0])
    assert np.allclose(layout["rsu_xyz"], [0.0, -18.0, 7.0])
    assert layout["evaluation_roi"]["frame"] == "fixed_world_road_aligned"
