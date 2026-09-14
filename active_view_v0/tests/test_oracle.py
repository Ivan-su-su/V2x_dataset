import numpy as np

from active_view_v0.oracle import best_fixed, constrained_oracle, per_frame_oracle, switch_count


def test_oracles_distinguish_fixed_unconstrained_and_speed_constrained():
    coords = np.asarray([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    scores = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    fixed = best_fixed(scores)
    free = per_frame_oracle(scores, coords)
    constrained = constrained_oracle(
        scores, coords, delta_t_s=2.0, max_speed_mps=5.0, start_index=0
    )
    assert fixed.perception_reward == 1.0
    assert free.indices.tolist() == [0, 1, 2]
    assert constrained.indices.tolist() == [0, 1, 2]
    assert constrained.movement_m == 20.0
    assert switch_count(constrained.indices) == 2


def test_speed_limit_prevents_grid_teleportation():
    coords = np.asarray([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    scores = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 10.0]])
    constrained = constrained_oracle(
        scores, coords, delta_t_s=1.0, max_speed_mps=5.0, start_index=0
    )
    assert constrained.indices.tolist() == [0, 0]


def test_per_frame_oracle_does_not_move_only_because_of_ties():
    coords = np.asarray([[-10.0, 0.0], [0.0, 0.0], [10.0, 0.0]])
    scores = np.ones((4, 3), dtype=np.float64)
    free = per_frame_oracle(scores, coords)
    assert free.indices.tolist() == [1, 1, 1, 1]
    assert free.movement_m == 0.0
