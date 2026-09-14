from dataclasses import dataclass

import numpy as np

from active_view_v0.bev_render import _oriented_rectangle, scene_bounds


@dataclass
class _Junction:
    center: tuple[float, float, float]


@dataclass
class _Corridor:
    first: _Junction
    second: _Junction


def test_scene_bounds_cover_both_junctions_with_margin():
    corridor = _Corridor(_Junction((0.0, 10.0, 0.0)), _Junction((60.0, -5.0, 0.0)))
    assert scene_bounds(corridor, 20.0) == (-20.0, 80.0, -25.0, 30.0)


def test_oriented_rectangle_rotates_length_axis():
    corners = _oriented_rectangle(10.0, 20.0, length=4.0, width=2.0, yaw_deg=90.0)
    assert np.allclose(corners.mean(axis=0), [10.0, 20.0])
    assert np.isclose(corners[:, 0].max() - corners[:, 0].min(), 2.0)
    assert np.isclose(corners[:, 1].max() - corners[:, 1].min(), 4.0)
