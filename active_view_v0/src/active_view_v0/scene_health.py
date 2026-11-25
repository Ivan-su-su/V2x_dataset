"""Validate full captures against the lane plan and detector GT convention."""

from __future__ import annotations

import json

import numpy as np

from .where2comm_eval import gt_boxes_in_ego


def lane_scene_health(rows, cfg, routes):
    plan = cfg["regional"]["lane_traffic"]["vehicles"]
    limits = cfg["regional"]["density_validation"]
    scoring = cfg["scoring"]
    failures = []
    if not rows:
        return {"passed": False, "failures": ["no frames to validate"]}
    counts = []
    moving_counts = []
    entrants = []
    departures = []
    previous_ids = None
    positions = {role: [] for role in plan}
    speeds = {role: [] for role in plan}
    for row in rows:
        sensor = row["sensors"].get("ego_lidar")
        if sensor is None:
            counts.append(None)
            moving_counts.append(None)
            entrants.append(None)
            departures.append(None)
            previous_ids = None
        else:
            _, ids, _ = gt_boxes_in_ego(
                row, np.asarray(sensor["transform"]["matrix"]),
                ignored_roles=scoring["ignore_roles"], accepted_classes=("car", "two_wheeler"),
                lidar_range=scoring["detector_lidar_range"],
            )
            counts.append(len(ids))
            present = set(ids)
            moving_counts.append(sum(float(np.linalg.norm(actor.get("velocity_world", (0., 0.))[:2])) >= .5
                                     for actor in row.get("gt", [])
                                     if actor["actor_id"] in present))
            entrants.append(None if previous_ids is None else len(present - previous_ids))
            departures.append(None if previous_ids is None else len(previous_ids - present))
            previous_ids = present
        actors = row.get("sensing_agents")
        if actors is None:
            actors = {k.rsplit(":", 1)[0]: v for k, v in row["actors"].items()}
        for role in plan:
            info = actors.get(role)
            positions[role].append(info["location"] if info else [np.nan] * 3)
            speeds[role].append(float(np.linalg.norm(info["velocity"][:2])) if info else 0.0)
    lo, hi = limits["minimum_targets_per_frame"], limits["maximum_targets_per_frame"]
    bad_frames = [i for i, n in enumerate(counts) if n is None or not lo <= n <= hi]
    observed_counts = [n for n in counts if n is not None]
    in_band = 1.0 - len(bad_frames) / len(rows)
    required_fraction = float(limits.get("target_band_min_fraction", 1.0))
    if in_band + 1e-9 < required_fraction or len(observed_counts) != len(rows):
        i = bad_frames[0]
        failures.append(f"{len(bad_frames)} frames outside {lo}–{hi} detector GTs or missing sensors; "
                        f"first index {i}, count {counts[i]}; fraction in band {in_band:.3f}")
    median = float(np.median(observed_counts)) if observed_counts else None
    median_band = limits.get("target_median_band")
    if median_band is not None and (median is None or not median_band[0] <= median <= median_band[1]):
        failures.append(f"detector GT median {median} outside {median_band}")
    # Match locations to the dense map path. 3D distance distinguishes stacked roads.
    crossing = {}
    covered = {}
    agents = {}
    dt = float(cfg["regional"]["collection_interval_s"])
    for role, spec in plan.items():
        points = np.asarray(positions[role])
        valid = np.isfinite(points).all(axis=1)
        if role not in routes or not valid[0]:
            failures.append(f"{role}: required actor or route missing at first frame")
            continue
        path = np.asarray(routes[role]["xyz"])
        distance = np.linalg.norm(points[:, None] - path[None, :], axis=2)
        station = np.r_[0., np.linalg.norm(np.diff(path, axis=0), axis=1).cumsum()]
        nearest = np.zeros(len(rows), dtype=int)
        deviation = np.full(len(rows), np.inf)
        previous = 0
        for i in np.flatnonzero(valid):
            lo_idx = np.searchsorted(station, station[previous] - 6)
            hi_idx = np.searchsorted(station, station[previous] + (14 if i > 0 else 20))
            previous = lo_idx + int(distance[i, lo_idx:hi_idx].argmin())
            nearest[i] = previous
            deviation[i] = distance[i, previous]
        along = np.where(valid, station[nearest], np.nan)
        observed = along[np.isfinite(along)]
        covered[role] = float(observed.max() - observed[0])
        if spec["group"] in {"ego", "car2"}:
            if not valid.all() or np.any(deviation > 7):
                failures.append(f"{role}: left the planned route or was absent")
            quarter_m = [float(q[-1] - q[0]) if len(q) > 1 and np.isfinite(q[[0,-1]]).all()
                         else 0.0 for q in np.array_split(along, 4)]
            stopped = np.flatnonzero(np.r_[True, np.asarray(speeds[role]) >= .5, True])
            stop_s = float((np.diff(stopped).max() - 1) * dt)
            if min(quarter_m) < limits["minimum_agent_progress_per_quarter_m"]:
                failures.append(f"{role}: insufficient progress in a 15 s quarter")
            if stop_s > limits["maximum_stopped_s"]:
                failures.append(f"{role}: stopped continuously for {stop_s:.1f} s")
            agents[role] = {"progress_per_quarter_m": quarter_m, "longest_stop_s": stop_s}
        if spec["group"] == "side":
            exit_point = np.asarray(routes[role]["junction_exit_xyz"])
            exit_idx = int(np.linalg.norm(path - exit_point, axis=1).argmin())
            passed = valid & (nearest >= exit_idx) & (deviation <= 7)
            crossing.setdefault(spec["junction_id"], []).append(bool(passed.any()))
    for jid, seen in crossing.items():
        if not any(seen):
            failures.append(f"junction {jid}: no designated side vehicle crossed")
    for group in ("with", "against", "side"):
        values = [covered.get(role, 0) for role, spec in plan.items() if spec["group"] == group]
        if sum(v >= 20 for v in values) < (len(values) + 1) // 2:
            failures.append(f"{group} traffic: fewer than half the vehicles advanced 20 m")
    return {"passed": not failures, "failures": failures, "detector_vehicle_gt_per_frame": counts,
            "detector_gt_min": min(observed_counts, default=None),
            "detector_gt_max": max(observed_counts, default=None),
            "detector_gt_median": median, "detector_gt_in_band_fraction": in_band,
            "moving_detector_gt_per_frame": moving_counts,
            "detector_gt_roi_entries_per_frame": entrants,
            "detector_gt_roi_exits_per_frame": departures,
            "density_failure_frames": bad_frames, "agent_motion": agents,
            "side_crossings_by_junction": {j: sum(v) for j, v in crossing.items()}}


def check_regional_lane_capture(path, cfg):
    status = json.loads((path / "collection_status.json").read_text(encoding="utf-8"))
    expected = int(cfg["regional"]["frame_count"])
    index = [json.loads(line) for line in (path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    if (status["status"] != "complete" or len(index) != expected
            or [i["keyframe_index"] for i in index] != list(range(expected))):
        return {"passed": False, "failures": ["incomplete regional capture"], "full_scene": False}
    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    rows = [json.loads((path / i["path"] / "frame.json").read_text(encoding="utf-8")) for i in index]
    return {"full_scene": True, **lane_scene_health(rows, cfg, meta["scenario"]["routes"])}
