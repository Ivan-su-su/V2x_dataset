import numpy as np

from active_view_v0.idea_report import _judge, _scene_health


def test_judge_rejects_tied_positions():
    verdict, _ = _judge(0.1, 0.2, 0.1, 4)
    assert verdict == "UNSUPPORTED"


def test_judge_rejects_no_dynamic_gap():
    verdict, _ = _judge(0.8, 0.005, 0.0, 3)
    assert verdict == "UNSUPPORTED"


def test_judge_marks_physical_gain_as_strong():
    verdict, _ = _judge(0.8, 0.08, 0.04, 4)
    assert verdict == "STRONG_PROXY_SIGNAL"


def test_scene_health_rejects_empty_late_phase():
    scores = np.zeros((4, 25))
    scores[:2, 22] = 1.0
    scores[2:, 2] = 1.0
    result = _scene_health(
        scores,
        np.array([0.0, 2.0, 4.0, 6.0]),
        np.array([10, 10, 2, 2]),
        np.array([8, 8, 2, 2]),
        np.array([0.2, 0.2, 0.0, 0.0]),
        early_end_s=2.0,
        late_start_s=4.0,
        minimum_late_targets=6,
        minimum_rescue_frames=2,
        minimum_row_shift=2,
    )
    assert not result["passed"]
    assert result["late_target_min"] == 2


def test_scene_health_accepts_two_rescue_phases():
    scores = np.zeros((4, 25))
    scores[:2, 22] = 1.0
    scores[2:, 2] = 1.0
    result = _scene_health(
        scores,
        np.array([0.0, 2.0, 4.0, 6.0]),
        np.array([10, 10, 9, 9]),
        np.array([8, 8, 7, 7]),
        np.array([0.2, 0.2, 0.2, 0.2]),
        early_end_s=2.0,
        late_start_s=4.0,
        minimum_late_targets=6,
        minimum_rescue_frames=2,
        minimum_row_shift=2,
    )
    assert result["passed"]
