import numpy as np

from active_view_v0.bev_render import project_world_to_camera


def test_carla_camera_axis_conversion_and_projection():
    points = np.asarray(
        [
            [10.0, 0.0, 0.0],
            [10.0, 1.0, 0.0],
            [10.0, 0.0, 1.0],
            [-10.0, 0.0, 0.0],
        ]
    )
    uv, valid = project_world_to_camera(points, np.eye(4), 1000, 1000, 90.0)
    assert np.allclose(uv[0], [500.0, 500.0])
    assert uv[1, 0] > 500.0
    assert uv[2, 1] < 500.0
    assert valid.tolist() == [True, True, True, False]
