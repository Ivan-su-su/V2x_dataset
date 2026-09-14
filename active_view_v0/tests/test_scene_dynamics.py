import numpy as np

from active_view_v0.scene_dynamics import _judge_transition


def test_transition_accepts_cav_that_covers_early_and_leaves_late():
    result = _judge_transition(
        np.asarray([0.0, 2.0, 4.0, 14.0, 16.0, 18.0]),
        np.asarray([35.0, 25.0, 30.0, 75.0, 90.0, 110.0]),
        early_end_s=4.0,
        late_start_s=14.0,
        service_radius_m=60.0,
        minimum_inside_frames=2,
        minimum_outside_frames=2,
        minimum_exit_shift_m=30.0,
    )
    assert result["passed"]
    assert result["early_inside_frame_count"] == 3
    assert result["late_outside_frame_count"] == 3


def test_transition_rejects_cav_that_never_leaves():
    result = _judge_transition(
        np.asarray([0.0, 2.0, 14.0, 16.0]),
        np.asarray([20.0, 25.0, 30.0, 35.0]),
        early_end_s=2.0,
        late_start_s=14.0,
        service_radius_m=60.0,
        minimum_inside_frames=2,
        minimum_outside_frames=2,
        minimum_exit_shift_m=30.0,
    )
    assert not result["passed"]
    assert len(result["failures"]) == 2
