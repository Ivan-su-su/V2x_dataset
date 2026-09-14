from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .geometry import angular_difference_deg


@dataclass(frozen=True)
class JunctionInfo:
    junction: Any
    junction_id: int
    center: tuple[float, float, float]
    extent: tuple[float, float, float]
    lane_pair_count: int
    road_count: int
    straight_pair_count: int
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "junction_id": self.junction_id,
            "center": list(self.center),
            "extent": list(self.extent),
            "lane_pair_count": self.lane_pair_count,
            "road_count": self.road_count,
            "straight_pair_count": self.straight_pair_count,
            "rank_score": self.score,
        }


def rank_junctions(carla_map: Any) -> list[JunctionInfo]:
    """Find and rank map junctions, preferring multi-road junctions with straight paths."""
    unique: dict[int, Any] = {}
    for waypoint in carla_map.generate_waypoints(2.0):
        if not waypoint.is_junction:
            continue
        junction = waypoint.get_junction()
        if junction is not None:
            unique[int(junction.id)] = junction

    infos: list[JunctionInfo] = []
    for junction_id, junction in unique.items():
        pairs = list(junction.get_waypoints(_driving_lane_type()))
        roads = {int(wp.road_id) for pair in pairs for wp in pair}
        straight = sum(
            angular_difference_deg(pair[0].transform.rotation.yaw, pair[1].transform.rotation.yaw) <= 25.0
            for pair in pairs
        )
        bbox = junction.bounding_box
        area = float(bbox.extent.x * bbox.extent.y)
        score = 20.0 * len(roads) + 4.0 * straight + len(pairs) + min(area, 400.0) / 100.0
        infos.append(
            JunctionInfo(
                junction=junction,
                junction_id=junction_id,
                center=(float(bbox.location.x), float(bbox.location.y), float(bbox.location.z)),
                extent=(float(bbox.extent.x), float(bbox.extent.y), float(bbox.extent.z)),
                lane_pair_count=len(pairs),
                road_count=len(roads),
                straight_pair_count=int(straight),
                score=float(score),
            )
        )
    return sorted(infos, key=lambda item: (-item.score, item.junction_id))


def select_junction(carla_map: Any, junction_id: int | None, rank: int = 0) -> JunctionInfo:
    infos = rank_junctions(carla_map)
    if not infos:
        raise RuntimeError("the selected CARLA map contains no junctions")
    if junction_id is not None:
        for info in infos:
            if info.junction_id == int(junction_id):
                return info
        available = ", ".join(str(item.junction_id) for item in infos[:20])
        raise ValueError(f"junction_id={junction_id} not found; leading candidates: {available}")
    if not 0 <= int(rank) < len(infos):
        raise ValueError(f"junction_rank={rank} outside [0, {len(infos) - 1}]")
    return infos[int(rank)]


def choose_straight_pair(junction: Any) -> tuple[Any, Any]:
    pairs = list(junction.get_waypoints(_driving_lane_type()))
    if not pairs:
        raise RuntimeError("selected junction has no driving-lane waypoint pairs")

    def key(pair: tuple[Any, Any]) -> tuple[float, int, int]:
        entry, exit_wp = pair
        angle = angular_difference_deg(entry.transform.rotation.yaw, exit_wp.transform.rotation.yaw)
        same_lane = int(entry.lane_id == exit_wp.lane_id)
        return angle, -same_lane, abs(int(entry.lane_id))

    return min(pairs, key=key)


def waypoint_before(waypoint: Any, distance_m: float) -> Any:
    candidates = list(waypoint.previous(float(distance_m)))
    if candidates:
        return min(
            candidates,
            key=lambda item: angular_difference_deg(
                item.transform.rotation.yaw, waypoint.transform.rotation.yaw
            ),
        )
    current = waypoint
    remaining = float(distance_m)
    while remaining > 0.0:
        options = list(current.previous(min(2.0, remaining)))
        if not options:
            break
        current = min(
            options,
            key=lambda item: angular_difference_deg(
                item.transform.rotation.yaw, waypoint.transform.rotation.yaw
            ),
        )
        remaining -= 2.0
    return current


def waypoint_after(waypoint: Any, distance_m: float) -> Any:
    candidates = list(waypoint.next(float(distance_m)))
    if candidates:
        return min(
            candidates,
            key=lambda item: angular_difference_deg(
                item.transform.rotation.yaw, waypoint.transform.rotation.yaw
            ),
        )
    return waypoint


def _driving_lane_type() -> Any:
    import carla

    return carla.LaneType.Driving

