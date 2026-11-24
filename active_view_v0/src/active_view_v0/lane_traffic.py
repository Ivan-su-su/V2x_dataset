"""Resolve explicit OpenDRIVE flows on the live map before spawning any actors."""

from __future__ import annotations

from math import ceil, isfinite


def validate_lane_traffic(regional: dict) -> None:
    plan = regional["lane_traffic"]
    if not isinstance(plan, dict) or set(plan) != {"vehicles", "junction_ids"}:
        raise ValueError("lane_traffic needs vehicles and junction_ids")
    cycle = regional.get("native_signal_cycle_s")
    if (not isinstance(cycle, dict) or set(cycle) != {"green", "yellow", "red"}
            or any(not isfinite(float(v)) or float(v) <= 0 for v in cycle.values())):
        raise ValueError("native_signal_cycle_s needs positive green, yellow and red times")
    junctions = plan["junction_ids"]
    if (not isinstance(junctions, list) or not junctions
            or any(type(j) is not int or j < 0 for j in junctions)
            or len(set(junctions)) != len(junctions)):
        raise ValueError("lane_traffic.junction_ids must contain distinct junction IDs")
    roles = plan["vehicles"]
    if not isinstance(roles, dict) or not set(regional["agent_sensors"]) <= roles.keys():
        raise ValueError("lane_traffic must route every sensing vehicle")
    for role, spec in roles.items():
        if (not isinstance(role, str) or not role.startswith("regional_") or not isinstance(spec, dict)
                or spec.get("group") not in {"ego", "car2", "with", "against", "side"}
                or not 0 < float(spec.get("speed_mps", 0)) <= 20):
            raise ValueError(f"invalid lane traffic vehicle: {role}")
        if spec["group"] == "side" and spec.get("junction_id") not in junctions:
            raise ValueError(f"{role}: side traffic requires a declared junction")
        lanes = spec.get("lanes")
        if not isinstance(lanes, list) or not lanes:
            raise ValueError(f"{role}: missing lane segments")
        for segment in lanes:
            if (not isinstance(segment, list) or len(segment) != 4
                    or type(segment[0]) is not int or segment[0] < 0
                    or type(segment[1]) is not int or segment[1] == 0
                    or any(type(s) not in (float, int) or not isfinite(s) or s < 0 for s in segment[2:])
                    or (segment[3] - segment[2]) * segment[1] > 0):
                raise ValueError(f"{role}: invalid directed lane segment {segment}")
    if (regional.get("fixed_signal_plan", {}).get("enabled") or regional.get("support_vehicles")
            or regional.get("background_vehicle_count", 0)):
        raise ValueError("lane_traffic replaces the legacy support traffic and fixed signal plan")


def resolve_lanes(carla_map, segments):
    """Sample [road ID, lane ID, start s, end s] in the declared direction."""
    import carla

    route = []
    for road, lane, start, end in segments:
        steps = max(1, ceil(abs(end - start) / 2.0))
        for i in range(steps + 1):
            if i == 1 and start == end:
                break
            station = end if i == steps else start + (end - start) * i / steps
            waypoint = carla_map.get_waypoint_xodr(road, lane, station)
            if waypoint is None or waypoint.lane_type != carla.LaneType.Driving:
                raise ValueError(f"missing driving lane {road}/{lane} at s={station:g}")
            if i == 0 and route:
                prev = route[-1]
                gap = prev.transform.location.distance(waypoint.transform.location)
                # A junction connector may end on the same physical point as the exit lane.
                if gap > 4.5 or not any(
                    (n.road_id, n.lane_id) == (road, lane)
                    and n.transform.location.distance(waypoint.transform.location) < 4
                    for distance in range(1, 7)
                    for n in prev.next(float(distance))
                ):
                    raise ValueError(f"disconnected lane {prev.road_id}/{prev.lane_id} → {road}/{lane}")
            route.append(waypoint)
    return route


def traffic_manager_path(route):
    """Target the exit road at each branch instead of interpolating inside its junction."""
    anchors = []
    for i, waypoint in enumerate(route[1:], 1):
        if waypoint.is_junction or not route[i - 1].is_junction:
            continue
        # Move slightly into the exit lane so nearest-point lookup does not
        # associate the shared connector endpoint with a different road.
        exit_index = i
        while exit_index + 1 < len(route) and not route[exit_index + 1].is_junction:
            if waypoint.transform.location.distance(route[exit_index].transform.location) >= 7:
                break
            exit_index += 1
        anchors.append(route[exit_index].transform.location)
    end = route[-1].transform.location
    if not anchors or end.distance(anchors[-1]) > 1:
        anchors.append(end)
    return anchors
