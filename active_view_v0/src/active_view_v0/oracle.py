from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PathSolution:
    indices: np.ndarray
    objective: float
    perception_reward: float
    movement_m: float


def best_fixed(scores: np.ndarray) -> PathSolution:
    scores = _check_scores(scores)
    index = int(np.argmax(scores.sum(axis=0)))
    path = np.full(scores.shape[0], index, dtype=np.int64)
    return PathSolution(path, float(scores[:, index].sum()), float(scores[:, index].sum()), 0.0)


def per_frame_oracle(
    scores: np.ndarray,
    coordinates_xy: np.ndarray,
    tie_tolerance: float = 1e-6,
) -> PathSolution:
    scores = _check_scores(scores)
    coords = _check_coords(coordinates_xy, scores.shape[1])
    path = np.empty(scores.shape[0], dtype=np.int64)
    previous_xy = np.zeros(2, dtype=np.float64)
    for time_index in range(scores.shape[0]):
        near_best = np.flatnonzero(
            scores[time_index] >= scores[time_index].max() - float(tie_tolerance)
        )
        distance = np.linalg.norm(coords[near_best] - previous_xy[None, :], axis=1)
        # A tie is not evidence for movement: prefer the closest position, then exact score, then index.
        order = np.lexsort((near_best, -scores[time_index, near_best], distance))
        path[time_index] = int(near_best[int(order[0])])
        previous_xy = coords[path[time_index]]
    movement = _path_distance(path, coords)
    reward = float(scores[np.arange(scores.shape[0]), path].sum())
    return PathSolution(path, reward, reward, movement)


def constrained_oracle(
    scores: np.ndarray,
    coordinates_xy: np.ndarray,
    delta_t_s: float,
    max_speed_mps: float,
    movement_penalty_per_m: float = 0.0,
    start_index: int | None = None,
) -> PathSolution:
    """Dynamic-programming oracle on a discrete grid with physical transitions."""
    scores = _check_scores(scores)
    coords = _check_coords(coordinates_xy, scores.shape[1])
    max_step = float(delta_t_s) * float(max_speed_mps) + 1e-9
    if max_step < 0:
        raise ValueError("delta_t_s and max_speed_mps must be non-negative")
    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    allowed = distances <= max_step
    time_count, position_count = scores.shape
    dp = np.full((time_count, position_count), -np.inf, dtype=np.float64)
    parent = np.full((time_count, position_count), -1, dtype=np.int64)
    if start_index is None:
        dp[0, :] = scores[0, :]
    else:
        if not 0 <= int(start_index) < position_count:
            raise ValueError("start_index is outside the candidate range")
        dp[0, int(start_index)] = scores[0, int(start_index)]
    penalty = float(movement_penalty_per_m)
    for time_index in range(1, time_count):
        for current in range(position_count):
            predecessors = np.flatnonzero(allowed[:, current])
            values = dp[time_index - 1, predecessors] - penalty * distances[predecessors, current]
            best_local = int(np.argmax(values))
            previous = int(predecessors[best_local])
            if np.isfinite(values[best_local]):
                parent[time_index, current] = previous
                dp[time_index, current] = values[best_local] + scores[time_index, current]
    final = int(np.argmax(dp[-1]))
    if not np.isfinite(dp[-1, final]):
        raise RuntimeError("no feasible oracle path")
    path = np.empty(time_count, dtype=np.int64)
    path[-1] = final
    for time_index in range(time_count - 1, 0, -1):
        path[time_index - 1] = parent[time_index, path[time_index]]
    movement = _path_distance(path, coords)
    perception = float(scores[np.arange(time_count), path].sum())
    return PathSolution(path, float(dp[-1, final]), perception, movement)


def closest_to_origin(coordinates_xy: np.ndarray) -> int:
    coords = np.asarray(coordinates_xy, dtype=np.float64)
    return int(np.argmin(np.linalg.norm(coords, axis=1)))


def switch_count(indices: np.ndarray) -> int:
    indices = np.asarray(indices, dtype=np.int64)
    return int(np.count_nonzero(indices[1:] != indices[:-1]))


def _path_distance(path: np.ndarray, coords: np.ndarray) -> float:
    if len(path) < 2:
        return 0.0
    return float(np.linalg.norm(coords[path[1:]] - coords[path[:-1]], axis=1).sum())


def _check_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] < 1 or scores.shape[1] < 1:
        raise ValueError("scores must have shape [time, positions]")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain non-finite values")
    return scores


def _check_coords(coords: np.ndarray, position_count: int) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape != (position_count, 2):
        raise ValueError("coordinates_xy must have shape [positions, 2]")
    return coords
