"""Shared offline frame layout and online audit traces."""

from __future__ import annotations

import json
import math
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from .dataset_runtime import Observation, Tick
from .geometry import transform_record


def json_write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _append(stream, value: dict) -> None:
    stream.write(json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n")


class RunRecorder:
    def __init__(self, directory: Path, offline: bool, files: ExitStack):
        self.directory, self.offline = directory, offline
        self.trace = files.enter_context((directory / "state_trace.jsonl").open("w", encoding="utf-8"))
        self.events = files.enter_context((directory / "events.jsonl").open("w", encoding="utf-8"))
        self.manifest = (
            files.enter_context((directory / "manifest.jsonl").open("w", encoding="utf-8"))
            if offline else None
        )
        self.observed = 0
        self.count = 0
        self.times: list[tuple[int, float]] = []
        self.target_counts: list[int] = []
        self.egos: list[tuple[int, list[float]]] = []

    def event(self, data: dict) -> None:
        _append(self.events, data)

    def tick(self, tick: Tick, obs: Observation | None) -> None:
        self.count += 1
        self.times.append((tick.index, tick.capture_wall_s))
        self.target_counts.append(len(tick.state["gt"]))
        self.egos.append((tick.index, tick.ego_transform["location"]))
        sensors = {}
        frame_dir = self.directory / "frames" / f"{tick.index:06d}"
        if obs is not None:
            self.observed += 1
            for name, measurement in obs.sensors.items():
                transform = transform_record(measurement.transform)
                if name == "uav_lidar" and math.dist(transform["location"], tick.uav_world_xyz) > 0.02:
                    raise RuntimeError(f"frame {tick.frame}: UAV LiDAR pose disagrees with applied action")
                record = {
                    "modality": "lidar",
                    "agent_type": obs.sensor_agent_types[name],
                    "point_count": len(measurement.raw_data) // 16,
                    "frame_id": int(measurement.frame),
                    "sim_timestamp_s": float(measurement.timestamp),
                    "transform": transform,
                }
                if self.offline:
                    agent = name.removesuffix("_lidar")
                    relative = Path("agents") / agent / "lidar.bin"
                    path = frame_dir / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.frombuffer(measurement.raw_data, dtype="<f4").reshape(-1, 4).tofile(path)
                    record["path"] = relative.as_posix()
                sensors[name] = record
        row = {"keyframe_index": tick.index, "carla_frame": tick.frame,
               "elapsed_s": tick.sim_time_s, "capture_wall_s": tick.capture_wall_s,
               "observation_ready_wall_s": obs.ready_wall_s if obs is not None else None,
               "ego_transform": tick.ego_transform, "uav_world_xyz": tick.uav_world_xyz,
               "sensors": sensors, **tick.state}
        _append(self.trace, row)
        if self.manifest is not None:
            if obs is None:
                raise RuntimeError("offline collection requires complete frames")
            json_write(frame_dir / "frame.json", row)
            _append(self.manifest, {key: row[key] for key in ("keyframe_index", "carla_frame", "elapsed_s")}
                    | {"path": frame_dir.relative_to(self.directory).as_posix()})

    def summary(self, dt: float) -> dict:
        times = [wall for _, wall in sorted(self.times)]
        wall_span = times[-1] - times[0] if len(times) > 1 else 0.0
        locations = [loc for _, loc in sorted(self.egos)]
        return {
            "world_ticks": self.count,
            "observed_frames": self.observed,
            "missing_frames": self.count - self.observed,
            "minimum_targets": min(self.target_counts, default=0),
            "mean_targets": float(np.mean(self.target_counts)) if self.target_counts else 0.0,
            "ego_displacement_m": math.dist(locations[0], locations[-1]) if locations else 0.0,
            "effective_wall_tick_hz": (len(times) - 1) / wall_span if wall_span > 0 else None,
            "tick_intervals_over_110_percent": sum(
                b - a > 1.1 * dt for a, b in zip(times, times[1:])
            ),
        }


def _trace(path: Path) -> dict[int, dict]:
    indexed = {}
    with (path / "state_trace.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            tick = row["keyframe_index"]
            if tick in indexed:
                raise ValueError(f"duplicate tick {tick} in {path}")
            indexed[tick] = row
    return indexed


def check_run(path: Path) -> dict:
    from .config import load_config

    cfg = load_config(path / "effective_config.yaml")
    lane_traffic = cfg["regional"].get("lane_traffic")
    if lane_traffic and (path / "collection_status.json").is_file():
        from .scene_health import check_regional_lane_capture

        return check_regional_lane_capture(path, cfg)
    manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("run did not complete")
    fixed = cfg["regional"].get("fixed_signal_plan", {})
    dt = float(cfg["carla"]["fixed_delta_seconds"])
    rows = _trace(path)
    n = manifest["frame_limit"]
    failures = []
    if set(rows) != set(range(n)):
        failures.append("missing or extra world ticks")
    first_late = None
    for tick, row in sorted(rows.items()):
        if manifest["mode"] != "online_realtime" and not row["sensors"]:
            failures.append(f"missing observation at tick {tick}")
            break
        if fixed.get("enabled"):
            expected = ("j2_corridor_green" if tick * dt >= float(fixed["j2_switch_time_s"])
                        else "j2_cross_green")
            j2 = [state for state in row["signals"].values() if state["junction"] == "J2_cross_green"]
            if not j2 or any(
                state["phase"] != expected or state["actual_state"] != state["expected_state"]
                for state in j2
            ):
                failures.append(f"J2 phase or lamp states disagree with config at tick {tick}")
                break
            if expected == "j2_corridor_green" and first_late is None:
                first_late = row["elapsed_s"]
    summary = manifest["summary"]
    if lane_traffic and n != manifest["frame_count"]:
        failures.append("a full 600-frame clip is required to qualify this scene")
    if lane_traffic and rows and not failures:
        from .scene_health import lane_scene_health

        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        return {"full_scene": True, **summary,
                **lane_scene_health([rows[i] for i in range(n)], cfg, meta["simulator"]["routes"])}
    if n == manifest["frame_count"] and rows and not failures:
        density = cfg["regional"].get("density_validation", {})
        for metric, setting in (
            ("ego_displacement_m", "minimum_ego_displacement_m"),
            ("minimum_targets", "minimum_targets_per_frame"),
            ("mean_targets", "minimum_mean_targets"),
        ):
            if summary[metric] < float(density.get(setting, 0)):
                failures.append(f"{metric}: {summary[metric]:.2f} < {density[setting]}")
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        corridor = metadata["simulator"]["corridor"]
        egos = np.asarray([row["ego_transform"]["location"][:2] for row in rows.values()])
        j1 = np.asarray(corridor["first"]["center"][:2])
        j2 = np.asarray(corridor["second"]["center"][:2])
        motion = {
            "minimum_j1_cross_progress_m": (float(np.max((egos - j1) @ corridor["forward_xy"])), True),
            "maximum_ego_j2_distance_m": (float(np.min(np.linalg.norm(egos - j2, axis=1))), False),
        }
        for limit, (value, minimum) in motion.items():
            if limit in density and (value < density[limit] if minimum else value > density[limit]):
                failures.append(f"{limit}: observed {value:.2f}m, limit {density[limit]}m")
    return {"passed": not failures, "failures": failures, "first_late_phase_s": first_late,
            "full_scene": n == manifest["frame_count"], **summary}


def compare_runs(left: Path, right: Path, *, position_m: float = 0.05,
                 angle_deg: float = 0.5) -> dict:
    """Compare hover reruns by relative tick and logical identities."""
    a, b = (json.loads((path / "run_manifest.json").read_text()) for path in (left, right))
    for run in (a, b):
        if run["status"] != "complete" or run["policy"] != "hover":
            raise ValueError("compare needs two completed hover runs")
    for key in ("scene_sha256", "frame_limit", "simulator_identity"):
        if a[key] != b[key]:
            raise ValueError(f"runs differ in {key}; use the same frozen scene, frame limit and simulator")
    rows = [_trace(path) for path in (left, right)]

    max_point_count_delta = 0

    def report(failure=None):
        return {"passed": failure is None, "first_difference": failure,
                "frame_count": a["frame_limit"], "position_tolerance_m": position_m,
                "angle_tolerance_deg": angle_deg, "max_point_count_delta": max_point_count_delta}

    def pose_close(x: dict, y: dict) -> bool:
        return math.dist(x["location"], y["location"]) <= position_m and all(
            abs((u - v + 180) % 360 - 180) <= angle_deg
            for u, v in zip(x["rotation"], y["rotation"])
        )

    for i in range(a["frame_limit"]):
        x, y = (r.get(i) for r in rows)
        def fail(field):
            return report({"tick": i, "field": field})

        if x is None or y is None:
            return fail("missing tick")
        if not math.isclose(x["elapsed_s"], y["elapsed_s"], abs_tol=1e-8):
            return fail("simulation time")
        if x["signals"] != y["signals"]:
            return fail("signals")
        if set(x["actors"]) != set(y["actors"]):
            return fail("actor identities")
        for name, p in x["actors"].items():
            q = y["actors"][name]
            if p["type_id"] != q["type_id"] or not pose_close(p, q):
                return fail(f"actor {name}")
        if not pose_close(x["ego_transform"], y["ego_transform"]):
            return fail("Ego pose")
        gx, gy = ({item["logical_actor_id"]: item for item in row["gt"]} for row in (x, y))
        if set(gx) != set(gy):
            return fail("GT identities")
        for name, p in gx.items():
            q = gy[name]
            if (p["class_name"] != q["class_name"]
                    or math.dist(p["bbox_center_world"], q["bbox_center_world"]) > position_m
                    or math.dist(p["bbox_extent"], q["bbox_extent"]) > position_m):
                return fail(f"GT {name}")
        if not x["sensors"] or set(x["sensors"]) != set(y["sensors"]):
            return fail("missing or different sensors")
        for name, p in x["sensors"].items():
            q = y["sensors"][name]
            if (not pose_close(p["transform"], q["transform"])
                    or p["frame_id"] != x["carla_frame"] or q["frame_id"] != y["carla_frame"]):
                return fail(f"sensor {name}")
            max_point_count_delta = max(max_point_count_delta, abs(p["point_count"] - q["point_count"]))
    if any(len(row) != a["frame_limit"] for row in rows):
        return report({"field": "extra ticks"})
    return report()
