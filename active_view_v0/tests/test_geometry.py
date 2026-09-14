import numpy as np

from active_view_v0.geometry import make_grid, points_in_oriented_box, transform_points


def test_transform_and_box_membership():
    matrix = np.eye(4)
    matrix[:3, 3] = [10.0, 20.0, 0.0]
    local = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    world = transform_points(local, matrix)
    assert np.allclose(world[0], [10.0, 20.0, 0.0])
    mask = points_in_oriented_box(world, matrix, [0, 0, 0], [1, 1, 1])
    assert mask.tolist() == [True, False]


def test_make_grid_is_centered_and_row_major():
    grid = make_grid([100, 200, 0], [1, 0], [0, 1], 5, 10, 25)
    assert len(grid) == 25
    assert grid[12]["name"] == "uav_r2_c2"
    assert grid[12]["location"] == [100.0, 200.0, 25.0]
    assert grid[0]["location"] == [80.0, 180.0, 25.0]
    assert grid[-1]["location"] == [120.0, 220.0, 25.0]

