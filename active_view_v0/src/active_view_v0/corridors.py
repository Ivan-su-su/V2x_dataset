from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .geometry import angular_difference_deg, normalize_xy, yaw_to_axes
from .junctions import JunctionInfo, rank_junctions, waypoint_after, waypoint_before


@dataclass(frozen=True)
class ConnectedJunctions:
    first: JunctionInfo
    second: JunctionInfo
    corridor_waypoints: tuple[Any, ...]
    corridor_distance_m: float
    forward_xy: tuple[float, float]
    right_xy: tuple[float, float]
    score: float

    @property
    def center(self) -> tuple[float, float, float]:
        a = np.asarray(self.first.center, dtype=np.float64)
        b = np.asarray(self.second.center, dtype=np.float64)
        return tuple(((a + b) * 0.5).tolist())

    def as_dict(self) -> dict[str, Any]:
        return {
            "first": self.first.as_dict(),
            "second": self.second.as_dict(),
            "corridor_distance_m": float(self.corridor_distance_m),
            "forward_xy": list(self.forward_xy),
            "right_xy": list(self.right_xy),
            "rank_score": float(self.score),
            "corridor_xyz": [
                [
                    float(wp.transform.location.x),
                    float(wp.transform.location.y),
                    float(wp.transform.location.z),
                ]
                for wp in self.corridor_waypoints
            ],
        }


def rank_connected_junctions(
    carla_map: Any,
    min_separation_m: float = 30.0,
    max_separation_m: float = 120.0,
    follow_step_m: float = 2.0,
) -> list[ConnectedJunctions]:
    """Rank directed, lane-connected junction pairs.

    A candidate is accepted only when following a near-straight driving lane out
    of the first junction reaches the second junction. Euclidean proximity alone
    is not enough because neighboring CARLA junctions are not always connected.
    """
    infos = rank_junctions(carla_map)
    by_id = {info.junction_id: info for info in infos}
    best_by_direction: dict[tuple[int, int], ConnectedJunctions] = {}

    for first in infos:
        for entry, exit_wp in first.junction.get_waypoints(_driving_lane_type()):
            turn = angular_difference_deg(
                entry.transform.rotation.yaw, exit_wp.transform.rotation.yaw
            )
            if turn > 35.0:
                continue
            traced = _trace_to_next_junction(
                first.junction_id,
                exit_wp,
                max_distance_m=max_separation_m + 80.0,
                step_m=follow_step_m,
            )
            if traced is None:
                continue
            second_id, middle_path, traveled = traced
            second = by_id.get(second_id)
            if second is None:
                continue
            separation = float(
                np.linalg.norm(
                    np.asarray(first.center[:2], dtype=np.float64)
                    - np.asarray(second.center[:2], dtype=np.float64)
                )
            )
            if not min_separation_m <= separation <= max_separation_m:
                continue

            start = waypoint_before(entry, 30.0)
            end = waypoint_after(middle_path[-1], 35.0)
            route = _deduplicate_waypoints((start, entry, exit_wp, *middle_path, end))
            if len(route) < 5:
                continue
            direction = np.asarray(second.center[:2]) - np.asarray(first.center[:2])
            forward = normalize_xy(direction)
            right = np.asarray([-forward[1], forward[0]], dtype=np.float64)
            complexity = min(first.road_count, second.road_count)
            desired_separation = 65.0
            score = (
                30.0 * complexity
                + 4.0 * min(first.straight_pair_count, second.straight_pair_count)
                - abs(separation - desired_separation)
                - 0.1 * abs(traveled - separation)
            )
            candidate = ConnectedJunctions(
                first=first,
                second=second,
                corridor_waypoints=tuple(route),
                corridor_distance_m=separation,
                forward_xy=(float(forward[0]), float(forward[1])),
                right_xy=(float(right[0]), float(right[1])),
                score=float(score),
            )
            key = (first.junction_id, second.junction_id)
            previous = best_by_direction.get(key)
            if previous is None or candidate.score > previous.score:
                best_by_direction[key] = candidate

    return sorted(
        best_by_direction.values(),
        key=lambda item: (-item.score, item.first.junction_id, item.second.junction_id),
    )


def select_connected_junctions(
    carla_map: Any,
    first_id: int | None,
    second_id: int | None,
    rank: int,
    min_separation_m: float,
    max_separation_m: float,
) -> ConnectedJunctions:
    candidates = rank_connected_junctions(
        carla_map,
        min_separation_m=min_separation_m,
        max_separation_m=max_separation_m,
    )
    if not candidates:
        raise RuntimeError(
            "no lane-connected junction pair found; widen regional.min/max_junction_separation_m"
        )
    if first_id is not None or second_id is not None:
        if first_id is None or second_id is None:
            raise ValueError("junction_1_id and junction_2_id must be set together")
        for item in candidates:
            if item.first.junction_id == int(first_id) and item.second.junction_id == int(second_id):
                return item
        leading = ", ".join(
            f"{item.first.junction_id}->{item.second.junction_id}" for item in candidates[:12]
        )
        raise ValueError(f"requested pair {first_id}->{second_id} not found; candidates: {leading}")
    if not 0 <= int(rank) < len(candidates):
        raise ValueError(f"corridor_rank={rank} outside [0, {len(candidates) - 1}]")
    return candidates[int(rank)]


def cross_route(
    junction: Any,
    reference_yaw_deg: float,
    before_m: float,
    after_m: float,
    reverse: bool = False,
) -> list[Any]:
    """Choose one direction of the straight road orthogonal to the corridor.

    reverse=True selects the legal opposite-direction lane pair of the same
    transverse road. It does not reverse waypoint order because CARLA lanes are
    directed and TrafficManager requires legal lane direction.
    """
    pairs = list(junction.get_waypoints(_driving_lane_type()))
    straight_pairs = [
        pair
        for pair in pairs
        if angular_difference_deg(pair[0].transform.rotation.yaw, pair[1].transform.rotation.yaw)
        <= 35.0
    ]
    if not straight_pairs:
        straight_pairs = pairs
    if not straight_pairs:
        raise RuntimeError(f"junction {junction.id} has no driving routes")

    def key(pair: tuple[Any, Any]) -> tuple[float, float]:
        entry, exit_wp = pair
        angle = angular_difference_deg(entry.transform.rotation.yaw, reference_yaw_deg)
        orthogonal_error = abs(90.0 - min(angle, 180.0 - angle))
        turn = angular_difference_deg(entry.transform.rotation.yaw, exit_wp.transform.rotation.yaw)
        return orthogonal_error, turn

    primary = min(straight_pairs, key=key)
    if reverse:
        primary_yaw = float(primary[0].transform.rotation.yaw)
        opposite_pairs = [
            pair
            for pair in straight_pairs
            if angular_difference_deg(
                pair[0].transform.rotation.yaw,
                primary_yaw,
            )
            >= 135.0
        ]
        if not opposite_pairs:
            raise RuntimeError(
                f"junction {junction.id} has no legal reverse transverse route"
            )
        entry, exit_wp = min(
            opposite_pairs,
            key=lambda pair: (
                abs(
                    180.0
                    - angular_difference_deg(
                        pair[0].transform.rotation.yaw,
                        primary_yaw,
                    )
                ),
                key(pair),
            ),
        )
    else:
        entry, exit_wp = primary
    return _deduplicate_waypoints(
        (waypoint_before(entry, before_m), entry, exit_wp, waypoint_after(exit_wp, after_m))
    )


def extend_route_straight(
    route: list[Any] | tuple[Any, ...],
    extra_distance_m: float,
    step_m: float = 2.0,
) -> list[Any]:
    """Extend a route while choosing the smallest-heading-change successor.

    TrafficManager chooses a random turn once a supplied path ends.  The
    regional experiment needs the ego to keep following the same road for the
    full 30 seconds, so we explicitly trace a long tail beyond junction 2.
    """
    output = list(route)
    if not output:
        raise ValueError("route cannot be empty")
    current = output[-1]
    previous_yaw = float(current.transform.rotation.yaw)
    remaining = max(0.0, float(extra_distance_m))
    step = max(0.5, float(step_m))
    while remaining > 1e-6:
        distance = min(step, remaining)
        options = list(current.next(distance))
        if not options:
            break
        current = min(
            options,
            key=lambda item: angular_difference_deg(
                item.transform.rotation.yaw, previous_yaw
            ),
        )
        output.append(current)
        previous_yaw = float(current.transform.rotation.yaw)
        remaining -= distance
    return _deduplicate_waypoints(tuple(output))


def route_locations(route: list[Any] | tuple[Any, ...], spacing_m: float = 2.0) -> list[Any]:
    """Return a dense CARLA Location path suitable for TrafficManager.set_path."""
    if len(route) < 2:
        raise ValueError("route requires at least two waypoints")
    output: list[Any] = []
    for start, end in zip(route[:-1], route[1:]):
        start_location = start.transform.location
        end_location = end.transform.location
        distance = float(start_location.distance(end_location))
        count = max(1, int(math.ceil(distance / max(spacing_m, 0.5))))
        for index in range(count):
            alpha = index / count
            import carla

            output.append(
                carla.Location(
                    x=float(start_location.x + alpha * (end_location.x - start_location.x)),
                    y=float(start_location.y + alpha * (end_location.y - start_location.y)),
                    z=float(start_location.z + alpha * (end_location.z - start_location.z)),
                )
            )
    output.append(route[-1].transform.location)
    return output


def _trace_to_next_junction(
    origin_junction_id: int,
    start_waypoint: Any,
    max_distance_m: float,
    step_m: float,
) -> tuple[int, list[Any], float] | None:
    current = start_waypoint
    path: list[Any] = []
    traveled = 0.0
    previous_yaw = float(current.transform.rotation.yaw)
    left_origin = False
    while traveled < max_distance_m:
        options = list(current.next(float(step_m)))
        if not options:
            return None
        current = min(
            options,
            key=lambda item: angular_difference_deg(item.transform.rotation.yaw, previous_yaw),
        )
        traveled += float(step_m)
        previous_yaw = float(current.transform.rotation.yaw)
        path.append(current)
        if not current.is_junction:
            left_origin = True
            continue
        junction = current.get_junction()
        if left_origin and junction is not None and int(junction.id) != int(origin_junction_id):
            return int(junction.id), path, traveled
    return None


def _deduplicate_waypoints(route: tuple[Any, ...]) -> list[Any]:
    output: list[Any] = []
    for waypoint in route:
        if output and waypoint.transform.location.distance(output[-1].transform.location) < 0.5:
            continue
        output.append(waypoint)
    return output


def _driving_lane_type() -> Any:
    import carla

    return carla.LaneType.Driving
