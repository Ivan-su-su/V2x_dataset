from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np


def carla_matrix(transform: Any) -> np.ndarray:
    """Return CARLA transform as a 4x4 matrix mapping local points to world."""
    return np.asarray(transform.get_matrix(), dtype=np.float64)


def transform_points(points_xyz: np.ndarray, local_to_world: np.ndarray) -> np.ndarray:
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points_xyz must have shape [N, 3]")
    homogeneous = np.concatenate(
        [points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)], axis=1
    )
    return (np.asarray(local_to_world, dtype=np.float64) @ homogeneous.T).T[:, :3]


def points_in_oriented_box(
    points_world: np.ndarray,
    actor_to_world: np.ndarray,
    bbox_center_local: Iterable[float],
    bbox_extent: Iterable[float],
    margin: float = 0.0,
) -> np.ndarray:
    """Boolean mask for points inside a CARLA actor bounding box."""
    points_world = np.asarray(points_world, dtype=np.float64)
    world_to_actor = np.linalg.inv(np.asarray(actor_to_world, dtype=np.float64))
    actor_points = transform_points(points_world, world_to_actor)
    center = np.asarray(list(bbox_center_local), dtype=np.float64)
    extent = np.asarray(list(bbox_extent), dtype=np.float64) + float(margin)
    return np.all(np.abs(actor_points - center[None, :]) <= extent[None, :], axis=1)


def normalize_xy(vector: Iterable[float]) -> np.ndarray:
    value = np.asarray(list(vector), dtype=np.float64)[:2]
    norm = float(np.linalg.norm(value))
    if norm < 1e-9:
        raise ValueError("zero-length XY vector")
    return value / norm


def yaw_to_axes(yaw_deg: float) -> tuple[np.ndarray, np.ndarray]:
    yaw = math.radians(float(yaw_deg))
    forward = np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    right = np.asarray([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)
    return forward, right


def make_grid(
    center_xyz: Iterable[float],
    forward_xy: Iterable[float],
    right_xy: Iterable[float],
    size: int,
    spacing_m: float,
    height_m: float,
) -> list[dict[str, Any]]:
    """Create a road-aligned odd square grid centered on a junction."""
    if size % 2 == 0:
        raise ValueError("size must be odd")
    center = np.asarray(list(center_xyz), dtype=np.float64)
    forward = normalize_xy(forward_xy)
    right = normalize_xy(right_xy)
    half = size // 2
    positions: list[dict[str, Any]] = []
    for row, forward_step in enumerate(range(-half, half + 1)):
        for col, right_step in enumerate(range(-half, half + 1)):
            offset_f = float(forward_step * spacing_m)
            offset_r = float(right_step * spacing_m)
            xy = center[:2] + offset_f * forward + offset_r * right
            positions.append(
                {
                    "name": f"uav_r{row}_c{col}",
                    "row": row,
                    "col": col,
                    "offset_forward_m": offset_f,
                    "offset_right_m": offset_r,
                    "location": [float(xy[0]), float(xy[1]), float(center[2] + height_m)],
                }
            )
    return positions


def make_regional_roi(
    center_xyz: Iterable[float],
    forward_xy: Iterable[float],
    right_xy: Iterable[float],
    bounds: dict[str, Iterable[float]],
) -> dict[str, Any]:
    """Build a fixed, road-aligned evaluation ROI in the CARLA world frame.

    The local regional x-axis follows the J1-to-J2 corridor and the y-axis is
    road-right.  Perception may still be fused in the current Ego frame; this
    object controls only which world-space targets belong to the benchmark.
    """
    center = np.asarray(list(center_xyz), dtype=np.float64)
    forward = normalize_xy(forward_xy)
    right = normalize_xy(right_xy)
    forward_bounds = np.asarray(list(bounds["forward_m"]), dtype=np.float64)
    right_bounds = np.asarray(list(bounds["right_m"]), dtype=np.float64)
    if forward_bounds.shape != (2,) or right_bounds.shape != (2,):
        raise ValueError("regional ROI forward_m/right_m must each contain [min, max]")
    if forward_bounds[0] >= forward_bounds[1] or right_bounds[0] >= right_bounds[1]:
        raise ValueError("regional ROI minimum bounds must be smaller than maximum bounds")
    regional_to_world = np.eye(4, dtype=np.float64)
    regional_to_world[:3, 0] = [forward[0], forward[1], 0.0]
    regional_to_world[:3, 1] = [right[0], right[1], 0.0]
    regional_to_world[:3, 2] = [0.0, 0.0, 1.0]
    regional_to_world[:3, 3] = center[:3]
    corners_local = np.asarray(
        [
            [forward_bounds[0], right_bounds[0], 0.0],
            [forward_bounds[1], right_bounds[0], 0.0],
            [forward_bounds[1], right_bounds[1], 0.0],
            [forward_bounds[0], right_bounds[1], 0.0],
        ],
        dtype=np.float64,
    )
    corners_world = transform_points(corners_local, regional_to_world)
    return {
        "frame": "fixed_world_road_aligned",
        "center_world": center[:3].tolist(),
        "forward_xy": forward.tolist(),
        "right_xy": right.tolist(),
        "forward_m": forward_bounds.tolist(),
        "right_m": right_bounds.tolist(),
        "regional_to_world": regional_to_world.tolist(),
        "world_to_regional": np.linalg.inv(regional_to_world).tolist(),
        "corners_world": corners_world.tolist(),
    }


def points_in_regional_roi(points_world: np.ndarray, roi: dict[str, Any]) -> np.ndarray:
    """Return a horizontal membership mask for a fixed regional ROI."""
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    local = transform_points(points, np.asarray(roi["world_to_regional"], dtype=np.float64))
    forward_bounds = np.asarray(roi["forward_m"], dtype=np.float64)
    right_bounds = np.asarray(roi["right_m"], dtype=np.float64)
    return (
        (local[:, 0] >= forward_bounds[0])
        & (local[:, 0] <= forward_bounds[1])
        & (local[:, 1] >= right_bounds[0])
        & (local[:, 1] <= right_bounds[1])
    )


def angular_difference_deg(first: float, second: float) -> float:
    return abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)


def interpolate_polyline(points: np.ndarray, distance_m: float) -> tuple[np.ndarray, float]:
    """Interpolate location and yaw along an XYZ polyline."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        raise ValueError("polyline requires at least two points")
    segments = points[1:] - points[:-1]
    lengths = np.linalg.norm(segments[:, :2], axis=1)
    remaining = max(0.0, float(distance_m))
    for index, length in enumerate(lengths):
        if length < 1e-9:
            continue
        if remaining <= length:
            alpha = remaining / length
            location = points[index] + alpha * segments[index]
            yaw = math.degrees(math.atan2(segments[index, 1], segments[index, 0]))
            return location, yaw
        remaining -= length
    segment = segments[-1]
    yaw = math.degrees(math.atan2(segment[1], segment[0]))
    return points[-1].copy(), yaw


def transform_record(transform: Any) -> dict[str, Any]:
    location = transform.location
    rotation = transform.rotation
    return {
        "location": [float(location.x), float(location.y), float(location.z)],
        "rotation": [float(rotation.roll), float(rotation.pitch), float(rotation.yaw)],
        "matrix": carla_matrix(transform).tolist(),
    }
