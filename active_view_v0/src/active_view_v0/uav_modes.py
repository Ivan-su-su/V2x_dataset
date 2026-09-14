from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class DecisionWindows:
    frame_to_window: np.ndarray
    window_start_indices: np.ndarray
    window_elapsed_s: np.ndarray


def build_decision_windows(
    elapsed_s: Iterable[float], decision_interval_s: float
) -> DecisionWindows:
    """Assign dense sensor frames to causal UAV decision windows.

    A 10 Hz clip with a 1 Hz controller therefore contains ten windows.  The
    function is timestamp-based instead of assuming an exact number of frames
    per window, so it remains valid if a future collector drops a frame.
    """
    elapsed = np.asarray(list(elapsed_s), dtype=np.float64)
    if elapsed.ndim != 1 or not len(elapsed):
        raise ValueError("elapsed_s must be a non-empty vector")
    if not np.isfinite(elapsed).all() or np.any(np.diff(elapsed) < -1e-9):
        raise ValueError("elapsed_s must be finite and non-decreasing")
    interval = float(decision_interval_s)
    if interval <= 0:
        raise ValueError("decision_interval_s must be positive")
    relative = elapsed - elapsed[0]
    raw = np.floor((relative + 1e-7) / interval).astype(np.int64)
    _, frame_to_window = np.unique(raw, return_inverse=True)
    starts = np.flatnonzero(
        np.r_[True, frame_to_window[1:] != frame_to_window[:-1]]
    ).astype(np.int64)
    return DecisionWindows(
        frame_to_window=frame_to_window,
        window_start_indices=starts,
        window_elapsed_s=elapsed[starts],
    )


def aggregate_scores_by_window(
    scores: np.ndarray, windows: DecisionWindows
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(windows.frame_to_window):
        raise ValueError("scores must have shape [frames, candidates]")
    count = int(windows.frame_to_window.max()) + 1
    output = np.empty((count, values.shape[1]), dtype=np.float64)
    for index in range(count):
        output[index] = values[windows.frame_to_window == index].mean(axis=0)
    return output


def expand_window_path(path: Iterable[int], windows: DecisionWindows) -> np.ndarray:
    decisions = np.asarray(list(path), dtype=np.int64)
    expected = int(windows.frame_to_window.max()) + 1
    if decisions.shape != (expected,):
        raise ValueError(f"decision path must contain {expected} indices")
    return decisions[windows.frame_to_window]


class UAVModePlanner:
    """Generate continuous hover, Ego-tracking and patrol baselines.

    Decisions target reachable grid points at 1 Hz. Sensor locations are
    linearly interpolated at the dense collection rate, so CARLA records three
    physically moving LiDAR streams in addition to the 25 counterfactual fixed
    candidates used by the detector oracle.
    """

    MODE_NAMES = ("hover", "tracking", "patrol")

    def __init__(
        self,
        grid: list[dict[str, Any]],
        mode_cfg: dict[str, Any],
        *,
        frame_interval_s: float,
        max_speed_mps: float,
    ) -> None:
        if not grid:
            raise ValueError("UAV mode planner requires a non-empty grid")
        self.grid = list(grid)
        self.names = [str(item["name"]) for item in self.grid]
        self.xy = np.asarray([item["location"][:2] for item in self.grid], dtype=np.float64)
        self.name_to_index = {name: index for index, name in enumerate(self.names)}
        self.frame_interval_s = float(frame_interval_s)
        self.decision_interval_s = float(mode_cfg.get("decision_interval_s", 1.0))
        if self.frame_interval_s <= 0 or self.decision_interval_s <= 0:
            raise ValueError("UAV frame and decision intervals must be positive")
        ratio = self.decision_interval_s / self.frame_interval_s
        if abs(ratio - round(ratio)) > 1e-6:
            raise ValueError("UAV decision interval must be a multiple of frame interval")
        self.frames_per_decision = int(round(ratio))
        self.max_step_m = float(max_speed_mps) * self.decision_interval_s
        initial_name = str(mode_cfg.get("initial_candidate", self.names[len(self.names) // 2]))
        if initial_name not in self.name_to_index:
            raise ValueError(f"unknown initial UAV candidate: {initial_name}")
        initial = self.name_to_index[initial_name]
        self.initial_index = initial
        self.segment_start = {name: initial for name in self.MODE_NAMES}
        self.segment_target = {name: initial for name in self.MODE_NAMES}
        self.tracking_forward_m = float(mode_cfg.get("tracking_forward_offset_m", 25.0))
        self.tracking_right_m = float(mode_cfg.get("tracking_right_offset_m", 0.0))
        patrol_names = [str(value) for value in mode_cfg.get("patrol_candidates", [])]
        if not patrol_names:
            patrol_names = _default_patrol_names(self.grid, initial_name)
        unknown = sorted(set(patrol_names).difference(self.name_to_index))
        if unknown:
            raise ValueError(f"unknown patrol UAV candidates: {unknown}")
        if patrol_names[0] != initial_name:
            patrol_names.insert(0, initial_name)
        self.patrol_indices = [self.name_to_index[name] for name in patrol_names]
        self.decision_count = 0
        self.records: list[dict[str, Any]] = []

    def step(self, frame_index: int, elapsed_s: float, ego_transform: Any) -> dict[str, Any]:
        is_decision = int(frame_index) % self.frames_per_decision == 0
        if is_decision:
            if self.decision_count > 0:
                self.segment_start = dict(self.segment_target)
            desired = _tracking_target(
                ego_transform,
                self.tracking_forward_m,
                self.tracking_right_m,
            )
            self.segment_target["hover"] = self.initial_index
            self.segment_target["tracking"] = self._reachable_nearest(
                self.segment_start["tracking"], desired
            )
            patrol_target = self.patrol_indices[
                (self.decision_count + 1) % len(self.patrol_indices)
            ]
            self.segment_target["patrol"] = self._reachable_nearest(
                self.segment_start["patrol"], self.xy[patrol_target]
            )
            self.decision_count += 1
        phase = (int(frame_index) % self.frames_per_decision) / self.frames_per_decision
        modes = {
            name: self._mode_record(
                name,
                self.segment_start[name],
                self.segment_target[name],
                phase,
            )
            for name in self.MODE_NAMES
        }
        record = {
            "frame_index": int(frame_index),
            "elapsed_s": float(elapsed_s),
            "is_decision_frame": bool(is_decision),
            "decision_index": int(max(self.decision_count - 1, 0)),
            "modes": modes,
        }
        self.records.append(record)
        return record

    def summary(self) -> dict[str, Any]:
        return {
            "format_version": "active-view-uav-modes-v1.0",
            "decision_interval_s": self.decision_interval_s,
            "frames_per_decision": self.frames_per_decision,
            "max_speed_mps": self.max_step_m / self.decision_interval_s,
            "max_step_m": self.max_step_m,
            "mode_descriptions": {
                "hover": "fixed at the common initial candidate",
                "tracking": "tracks a fixed forward/right offset from Ego without GT",
                "patrol": "follows a traffic-independent predefined grid route",
            },
            "paths": {
                mode: {
                    "candidate_names_per_frame": [
                        item["modes"][mode]["target_candidate_name"] for item in self.records
                    ],
                    "candidate_names_at_decisions": [
                        item["modes"][mode]["target_candidate_name"]
                        for item in self.records
                        if item["is_decision_frame"]
                    ],
                    "location_world_per_frame": [
                        item["modes"][mode]["location_world"] for item in self.records
                    ],
                    "sensor_name": f"uav_mode_{mode}_lidar",
                }
                for mode in self.MODE_NAMES
            },
        }

    def _reachable_nearest(self, current: int, target_xy: np.ndarray) -> int:
        distance_from_current = np.linalg.norm(self.xy - self.xy[current], axis=1)
        reachable = np.flatnonzero(distance_from_current <= self.max_step_m + 1e-6)
        if not len(reachable):
            return current
        target_distance = np.linalg.norm(self.xy[reachable] - target_xy[None, :], axis=1)
        order = np.lexsort((reachable, distance_from_current[reachable], target_distance))
        return int(reachable[int(order[0])])

    def _mode_record(
        self,
        mode: str,
        start_index: int,
        target_index: int,
        phase: float,
    ) -> dict[str, Any]:
        start = np.asarray(self.grid[int(start_index)]["location"], dtype=np.float64)
        target = np.asarray(self.grid[int(target_index)]["location"], dtype=np.float64)
        location = (1.0 - float(phase)) * start + float(phase) * target
        item = self.grid[int(target_index)]
        return {
            "candidate_name": str(item["name"]),
            "target_candidate_name": str(item["name"]),
            "source_sensor": f"uav_mode_{mode}_lidar",
            "candidate_index": int(target_index),
            "segment_start_candidate": str(self.grid[int(start_index)]["name"]),
            "segment_progress": float(phase),
            "location_world": location.tolist(),
            "offset_forward_m": float(item["offset_forward_m"]),
            "offset_right_m": float(item["offset_right_m"]),
        }


def _tracking_target(ego_transform: Any, forward_m: float, right_m: float) -> np.ndarray:
    location = ego_transform.location
    yaw = np.deg2rad(float(ego_transform.rotation.yaw))
    forward = np.asarray([np.cos(yaw), np.sin(yaw)], dtype=np.float64)
    right = np.asarray([-forward[1], forward[0]], dtype=np.float64)
    return (
        np.asarray([float(location.x), float(location.y)], dtype=np.float64)
        + float(forward_m) * forward
        + float(right_m) * right
    )


def _default_patrol_names(grid: list[dict[str, Any]], initial_name: str) -> list[str]:
    by_cell = {(int(item["row"]), int(item["col"])): str(item["name"]) for item in grid}
    center = next(
        (int(item["row"]), int(item["col"]))
        for item in grid
        if str(item["name"]) == initial_name
    )
    row, col = center
    cells = [
        (row, col),
        (row, col + 1),
        (row, col + 2),
        (row - 1, col + 2),
        (row - 2, col + 2),
        (row - 2, col + 1),
        (row - 2, col),
        (row - 1, col),
        (row, col),
        (row + 1, col),
        (row + 2, col),
    ]
    return [by_cell[cell] for cell in cells if cell in by_cell]
