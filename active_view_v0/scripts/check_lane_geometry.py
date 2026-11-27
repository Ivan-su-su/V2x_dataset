"""Preflight registered explicit routes against an exported CARLA OpenDRIVE file."""

import argparse
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path

import carla
import numpy as np
import yaml

from active_view_v0.dataset_scene import PROJECT, resolve_scene
from active_view_v0.lane_traffic import resolve_lanes, traffic_manager_path
from active_view_v0.regional_layout import build_regional_layout, resolve_regional_corridor


def xyz(waypoint):
    p = waypoint.transform.location
    return [p.x, p.y, p.z]


def _sample_route(route, distance):
    points = np.array([xyz(wp) for wp in route])
    station = np.r_[0., np.linalg.norm(np.diff(points, axis=0), axis=1).cumsum()]
    yaw = np.unwrap(np.deg2rad([wp.transform.rotation.yaw for wp in route]))
    return (np.stack([np.interp(distance, station, axis) for axis in points.T], axis=-1),
            np.interp(distance, station, yaw))


def _nominal_car2_overlap(scene, routes, events):
    """Estimate Car2 pose overlap at target speeds; no occlusion or car following model."""
    cfg = scene.config
    regional = cfg["regional"]
    agents = regional["lane_traffic"]["vehicles"]
    frames = np.arange(scene.frame_count) * scene.dt
    distance = frames + .65 * float(regional["warmup_s"])
    ego, yaw = _sample_route(routes["regional_ego"], distance * agents["regional_ego"]["speed_mps"])
    car2, _ = _sample_route(routes["regional_car3"], distance * agents["regional_car3"]["speed_mps"])
    dx, dy, dz = (car2 - ego).T
    c, s = np.cos(yaw), np.sin(yaw)
    # CARLA +right is the opposite of OpenCOOD's Y in the Ego detector frame.
    centers = np.stack((c * dx + s * dy, s * dx - c * dy, dz), axis=-1)
    limits = np.array(cfg["scoring"]["detector_lidar_range"], dtype=float)
    limits[[2, 5]] += [-5, 5]
    in_roi = ((limits[:3] <= centers) & (centers <= limits[3:])).all(axis=-1)
    nearby = in_roi & (np.linalg.norm(car2 - ego, axis=-1) <= cfg["sensors"]["vehicle_lidar"]["range"])

    def seconds(mask):
        return round(float(np.count_nonzero(mask) * scene.dt), 2)

    result = {"ego_roi": seconds(in_roi), "ego_roi_and_lidar_range": seconds(nearby)}
    for label, event in events.items():
        start, stop = event["nominal_window_s"]
        result[f"{label.lower()}_roi_and_lidar_range"] = seconds(nearby & (start <= frames) & (frames < stop))
    return result


def _nominal_density(scene, routes):
    """Estimate ROI occupancy on planned paths, without interactions or sensing."""
    cfg = scene.config
    regional = cfg["regional"]
    speeds = {role: spec["speed_mps"] for role, spec in regional["lane_traffic"]["vehicles"].items()}
    limits = np.asarray(cfg["scoring"]["detector_lidar_range"], dtype=float)
    limits[[2, 5]] += [-5, 5]  # Same vertical margin as the detector GT filter.
    frames = np.arange(scene.frame_count) * scene.dt
    warmup = float(regional["warmup_s"])

    ignored = set(cfg["scoring"]["ignore_roles"])
    targets = [
        _sample_route(route, (frames + .675 * warmup) * speeds[role])[0]
        for role, route in routes.items() if role not in ignored
    ]
    if not targets:
        raise ValueError("nominal density requires at least one target route")
    positions = np.asarray(targets)
    output = []
    for name, speed_ratio, delay_s in (("slow", .78, 0), ("light_delay", .91, 3), ("high", .99, 0)):
        distance = (np.maximum(frames - delay_s, 0) * speed_ratio + .65 * warmup) * speeds["regional_ego"]
        ego, yaw = _sample_route(routes["regional_ego"], distance)
        x, y, z = (positions - ego).transpose(2, 0, 1)
        c, s = np.cos(yaw), np.sin(yaw)
        z += .8 - float(cfg["sensors"]["vehicle_lidar"]["mount"]["z"])
        centers = np.stack((c * x + s * y, s * x - c * y, z), axis=-1)
        counts = ((limits[:3] <= centers) & (centers <= limits[3:])).all(axis=-1).sum(axis=0)
        density = regional["density_validation"]
        lo, hi = (density[k] for k in ("minimum_targets_per_frame", "maximum_targets_per_frame"))
        in_band = np.count_nonzero((lo <= counts) & (counts <= hi))
        output.append({"case": name, "median": float(np.median(counts)), "min": int(counts.min()),
                       "max": int(counts.max()), "frames_in_band": int(in_band),
                       "fraction_in_band": round(in_band / scene.frame_count, 4)})
    return output


def check_scene(scene_id, carla_map, nominal_density=False):
    scene = resolve_scene(scene_id)
    cfg = scene.config
    regional = cfg["regional"]
    plan = regional["lane_traffic"]
    corridor = resolve_regional_corridor(carla_map, regional)
    layout = build_regional_layout(corridor, cfg)
    span_s = float(regional["duration_s"]) + float(regional["warmup_s"])
    starts = {}
    routes = {}
    for role, spec in plan["vehicles"].items():
        route = resolve_lanes(carla_map, spec["lanes"])
        routes[role] = route
        if route[0].is_junction:
            raise ValueError(f"{role}: starts inside a junction")
        distance = sum(a.transform.location.distance(b.transform.location) for a, b in zip(route, route[1:]))
        if distance < (span_s + 1.0) * spec["speed_mps"] or not traffic_manager_path(route):
            raise ValueError(f"{role}: route too short or no path anchors ({distance:.1f} m)")
        starts[role] = xyz(route[0])
        if "junction_id" in spec and spec["junction_id"] not in {
            w.get_junction().id for w in route[:-1] if w.is_junction
        }:
            raise ValueError(f"{role}: misses its designated junction")
    gap, left, right = min((math.dist(starts[a], starts[b]), a, b) for a, b in combinations(starts, 2))
    if gap < 9.0:
        raise ValueError(f"{left}, {right}: spawn spacing {gap:.2f} m")
    candidates = {node["name"]: node for node in layout["grid"]}
    if len(candidates) != 25 or scene.frame_count != 600 or not math.isclose(scene.dt, .1):
        raise ValueError("expected 25 grid points and 600 frames at 10 Hz")
    path = PROJECT / "scenes" / scene_id / "config" / "brief.yaml"
    brief = yaml.safe_load(path.read_text(encoding="utf-8"))
    uav = brief["agents"]["uav"]
    if not np.allclose(layout["grid_center_xyz"], uav["grid_center_world_xyz_m"], atol=.002):
        raise ValueError("UAV grid center differs from the scene brief")
    if not np.allclose(layout["rsu_xyz"], brief["agents"]["rsu_world_xyz_m"], atol=.002):
        raise ValueError("RSU placement differs from the scene brief")
    if not math.isclose(scene.dt * scene.decision_ticks, 1.0):
        raise ValueError("expected 1 Hz decisions")
    events = uav["event_viewpoints"]
    a = candidates[events["A"]["id"]]
    lidar_range = float(cfg["sensors"]["uav_lidar"]["range"])
    steps = None
    if uav.get("evaluation_mode", "grid") == "continuous":
        probe = uav["continuous_probe"]
        waypoint = probe["waypoint_world_xyz_m"]
        if (len(waypoint) != 3 or not all(math.isfinite(x) for x in waypoint)
                or events["B"]["id"] is not None or uav["step_count_A_to_B"] is not None):
            raise ValueError("continuous scene needs a finite B waypoint and no B grid ID or steps")
        b_ground = events["B"]["ground_area_xyz_m"]
        if min(math.dist(point["location"], b_ground) for point in candidates.values()) <= lidar_range:
            raise ValueError("continuous test B target is within the starting grid's nominal LiDAR range")
        start = candidates[uav["initial_candidate"]]["location"]
        depart_min, depart_max = (probe[f"{bound}_departure_s"] for bound in ("earliest", "latest"))
        b_start = events["B"]["nominal_window_s"][0]
        duration = math.dist(start, waypoint) / cfg["oracle"]["uav_speed_mps"]
        if not 0 <= depart_min <= depart_max <= b_start or depart_min + duration > b_start:
            raise ValueError("continuous test B waypoint is not reachable from the start in time")
        b = {"location": waypoint}
    else:
        b = candidates[events["B"]["id"]]
        steps = abs(a["row"] - b["row"]) + abs(a["col"] - b["col"])
        if steps != uav["step_count_A_to_B"] or steps < 1:
            raise ValueError(f"A/B grid path has {steps} steps")
        if cfg["grid"]["spacing_m"] > float(cfg["oracle"]["uav_speed_mps"]) * scene.dt * scene.decision_ticks + 1e-6:
            raise ValueError("speed limit cannot reach a neighboring grid point in one decision step")
    targets = [str(p.get("cohort")) for p in plan["vehicles"].values()]
    if min(targets.count("A"), targets.count("B")) < 3:
        raise ValueError("fewer than three planned cars in one cohort")
    for label, pos in (("A", a), ("B", b)):
        expected_roles = {role for role, p in plan["vehicles"].items() if p.get("cohort") == label}
        if expected_roles != set(uav["event_viewpoints"][label]["target_roles"]):
            raise ValueError(f"cohort {label} roles differ from the scene brief")
        ground = events[label].get("ground_area_xyz_m", events[label]["ground_area_xy_m"])
        distance = math.dist(pos["location"][:len(ground)], ground)
        if distance > lidar_range:
            raise ValueError(f"cohort {label} is outside the nominal LiDAR range")
    result = {"scene": scene_id, "routes": len(starts), "minimum_spawn_gap_m": round(gap, 3),
              "uav_steps": steps, "nominal_frames": scene.frame_count}
    if nominal_density:
        result["nominal_density"] = _nominal_density(scene, routes)
        if "nominal_overlap_s" in brief.get("collaborator_car2", {}):
            result["nominal_car2_overlap_s"] = _nominal_car2_overlap(scene, routes, events)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("opendrive", type=Path)
    parser.add_argument("--town", default="Town04")
    parser.add_argument("--scene", action="append", help="full ID; repeat to select scenes")
    parser.add_argument("--nominal-density", action="store_true",
                        help="estimate ROI occupancy with three Ego speed/delay settings; not live GT")
    args = parser.parse_args()
    raw = args.opendrive.read_bytes()
    carla_map = carla.Map(args.town, raw.decode("utf-8"))
    scene_ids = args.scene or [
        f"{args.town}/{p.parents[1].name}" for p in sorted(
            (PROJECT / "scenes" / args.town).glob("*/config/scene.yaml")
        )
    ]
    if not scene_ids:
        parser.error(f"no registered scenes for {args.town}")
    passed, failed = [], []
    for key in scene_ids:
        try:
            passed.append(check_scene(key, carla_map, nominal_density=args.nominal_density))
        except (KeyError, ValueError, RuntimeError) as exc:
            failed.append({"scene": key, "error": str(exc)})
    print(json.dumps({"opendrive_sha256": hashlib.sha256(raw).hexdigest(),
                      "passed": passed, "failed": failed, "live_checked": False}, indent=2))
    raise SystemExit(bool(failed))


if __name__ == "__main__":
    main()
