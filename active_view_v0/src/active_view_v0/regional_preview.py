from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .bev_render import annotate_rgb_snapshot, compose_rgb_timeline, render_timeline
from .config import dump_effective_config, load_config
from .corridors import rank_connected_junctions, select_connected_junctions
from .regional_layout import build_regional_layout
from .regional_scenario import RegionalIntersectionScenario
from .sensors import SensorRig


def preview(config_path: str | Path, overwrite: bool = False) -> Path:
    import carla

    cfg = load_config(config_path)
    regional = cfg["regional"]
    carla_cfg = cfg["carla"]
    client = carla.Client(str(carla_cfg["host"]), int(carla_cfg["port"]))
    client.set_timeout(float(carla_cfg["timeout_s"]))
    world = client.get_world()
    target_map = str(carla_cfg["map"])
    if not world.get_map().name.endswith(target_map):
        print(f"Loading {target_map}; current actors will be cleared by CARLA.")
        world = client.load_world(target_map)

    output_dir = Path(cfg["output"]["root"]).expanduser() / str(cfg["output"]["run_name"])
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite to replace this preview only")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    dump_effective_config(cfg, output_dir / "effective_config.yaml")

    corridor = select_connected_junctions(
        world.get_map(),
        regional.get("junction_1_id"),
        regional.get("junction_2_id"),
        int(regional.get("corridor_rank", 0)),
        float(regional["min_junction_separation_m"]),
        float(regional["max_junction_separation_m"]),
    )
    layout = build_regional_layout(corridor, cfg)
    rsu_xyz = layout["rsu_xyz"]
    grid = layout["grid"]
    evaluation_roi = layout["evaluation_roi"]
    sensor_ranges = _sensor_range_summary(cfg, corridor, grid)

    original_settings = world.get_settings()
    traffic_manager = client.get_trafficmanager(int(carla_cfg["traffic_manager_port"]))
    scenario: RegionalIntersectionScenario | None = None
    camera_rig: SensorRig | None = None
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = float(carla_cfg["fixed_delta_seconds"])
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(int(carla_cfg["seed"]))
        world.set_weather(carla.WeatherParameters.ClearNoon)

        scenario = RegionalIntersectionScenario(client, world, traffic_manager, cfg, corridor)
        scenario.setup()
        scenario_meta = scenario.metadata()
        camera_rig = SensorRig(world, float(carla_cfg["fixed_delta_seconds"]))
        overhead_camera = _add_overhead_camera(camera_rig, corridor, cfg)

        dt = float(carla_cfg["fixed_delta_seconds"])
        warmup_s = float(regional["warmup_s"])
        for _ in range(int(round(warmup_s / dt))):
            scenario.apply_fixed_signal_plan()
            world.tick()
            scenario.apply_fixed_signal_plan()
            scenario.assert_fixed_signal_plan()

        snapshot_times = sorted(float(value) for value in regional["snapshot_times_s"])
        total_duration = max(float(regional["duration_s"]), snapshot_times[-1] if snapshot_times else 0.0)
        snapshot_ticks = {int(round(value / dt)): value for value in snapshot_times}
        snapshots: list[dict[str, Any]] = []
        rgb_paths: list[Path] = []
        annotated_rgb_paths: list[Path] = []
        for tick_index in range(int(round(total_duration / dt)) + 1):
            scenario.apply_fixed_signal_plan()
            frame_id = int(world.tick())
            scenario.apply_fixed_signal_plan()
            scenario.assert_fixed_signal_plan()
            if tick_index not in snapshot_ticks:
                continue
            elapsed_s = snapshot_ticks[tick_index]
            measurement = camera_rig.collect_frame(frame_id, timeout_s=20.0)["global_bev_rgb"]
            rgb_path = output_dir / f"global_bev_rgb_t{elapsed_s:05.1f}.png"
            measurement.save_to_disk(str(rgb_path))
            rgb_paths.append(rgb_path)
            actors = scenario.actor_states()
            annotated_rgb_path = output_dir / f"global_bev_rgb_annotated_t{elapsed_s:05.1f}.png"
            annotated_rgb_path, visibility = annotate_rgb_snapshot(
                rgb_path,
                annotated_rgb_path,
                overhead_camera,
                actors,
                rsu_xyz,
                grid,
                [list(corridor.first.center), list(corridor.second.center)],
                elapsed_s,
                sensor_ranges=sensor_ranges,
                evaluation_roi=evaluation_roi,
            )
            annotated_rgb_paths.append(annotated_rgb_path)
            snapshots.append(
                {
                    "elapsed_s": float(elapsed_s),
                    "carla_frame": frame_id,
                    "actors": actors,
                    "rgb_path": rgb_path.name,
                    "annotated_rgb_path": annotated_rgb_path.name,
                    "tracked_actor_visible_in_rgb": visibility,
                    "fixed_signal_states": scenario.fixed_signal_states(),
                }
            )
            print(f"Snapshot t={elapsed_s:5.1f}s frame={frame_id} actors={len(snapshots[-1]['actors'])}")

        if not snapshots:
            raise RuntimeError("no snapshots were produced; regional.snapshot_times_s is empty")
        map_bev = render_timeline(
            world.get_map(),
            corridor,
            grid,
            rsu_xyz,
            scenario_meta["routes"],
            snapshots,
            output_dir / "global_bev_annotated_timeline.png",
            float(regional["bev_margin_m"]),
            evaluation_roi=evaluation_roi,
        )
        sensor_range_bev = render_timeline(
            world.get_map(),
            corridor,
            grid,
            rsu_xyz,
            scenario_meta["routes"],
            snapshots,
            output_dir / "global_bev_sensor_ranges_timeline.png",
            float(regional["bev_margin_m"]),
            sensor_ranges=sensor_ranges,
            evaluation_roi=evaluation_roi,
        )
        raw_rgb_bev = compose_rgb_timeline(
            rgb_paths,
            [float(item["elapsed_s"]) for item in snapshots],
            output_dir / "global_bev_rgb_raw_timeline.png",
        )
        rgb_bev = compose_rgb_timeline(
            annotated_rgb_paths,
            [float(item["elapsed_s"]) for item in snapshots],
            output_dir / "global_bev_rgb_annotated_timeline.png",
        )
        motion_summary = _build_motion_summary(snapshots)
        motion_report = _write_motion_report(output_dir, motion_summary)
        ego_motion = next(
            (
                item
                for item in motion_summary
                if item["role_name"] == "regional_ego"
            ),
            None,
        )
        if ego_motion is not None:
            print(
                "Ego motion: "
                f"displacement={ego_motion['displacement_m']:.1f}m, "
                f"sampled_path={ego_motion['sampled_path_m']:.1f}m, "
                f"mean_speed={ego_motion['mean_speed_mps']:.1f}m/s"
            )
        metadata = {
            "format_version": "active-view-regional-preview-v1.0",
            "map": world.get_map().name,
            "corridor": corridor.as_dict(),
            "rsu_xyz": rsu_xyz,
            "uav_grid": grid,
            "grid_center_xyz": layout["grid_center_xyz"],
            "uav_service_region": "corridor_to_junction_2_fixed_world_frame",
            "roi_mode": str(cfg["scoring"].get("roi_mode", "ego_moving")),
            "evaluation_roi": evaluation_roi,
            "sensor_ranges": sensor_ranges,
            "routes": scenario_meta["routes"],
            "fixed_signal_plan": scenario_meta["fixed_signal_plan"],
            "snapshots": snapshots,
            "motion_summary": motion_summary,
            "outputs": {
                "annotated_bev": map_bev.name,
                "sensor_range_bev": sensor_range_bev.name,
                "annotated_rgb_bev": rgb_bev.name,
                "raw_rgb_bev": raw_rgb_bev.name,
                "motion_report": motion_report.name,
            },
        }
        (output_dir / "scene_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Annotated global BEV: {map_bev}")
        print(f"CARLA RGB global BEV: {rgb_bev}")
        return output_dir
    finally:
        if scenario is not None:
            scenario.release_fixed_signal_plan()
        actor_ids: list[int] = []
        if camera_rig is not None:
            actor_ids.extend(int(stream.actor.id) for stream in camera_rig.streams.values())
        if scenario is not None:
            actor_ids.extend(int(actor.id) for actor in scenario.actors)
        actor_ids = list(dict.fromkeys(actor_ids))
        if actor_ids:
            try:
                responses = client.apply_batch_sync(
                    [carla.command.DestroyActor(actor_id) for actor_id in actor_ids], False
                )
                errors = [response.error for response in responses if response.error]
                if errors:
                    print(f"WARNING: {len(errors)} preview cleanup command(s) failed: {errors[:3]}")
            except Exception as error:
                print(f"WARNING: batched preview cleanup failed: {error}")
        if camera_rig is not None:
            camera_rig.streams.clear()
        if scenario is not None:
            scenario.actors.clear()
            scenario.routes.clear()
        try:
            traffic_manager.set_synchronous_mode(False)
        except Exception:
            pass
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass


def list_corridors(config_path: str | Path, limit: int) -> None:
    import carla

    cfg = load_config(config_path)
    carla_cfg = cfg["carla"]
    regional = cfg["regional"]
    client = carla.Client(str(carla_cfg["host"]), int(carla_cfg["port"]))
    client.set_timeout(float(carla_cfg["timeout_s"]))
    world = client.get_world()
    target_map = str(carla_cfg["map"])
    if not world.get_map().name.endswith(target_map):
        world = client.load_world(target_map)
    candidates = rank_connected_junctions(
        world.get_map(),
        float(regional["min_junction_separation_m"]),
        float(regional["max_junction_separation_m"]),
    )
    for index, item in enumerate(candidates[: int(limit)]):
        print(
            f"rank={index:02d} J1={item.first.junction_id:3d} -> J2={item.second.junction_id:3d} "
            f"distance={item.corridor_distance_m:6.1f}m roads=({item.first.road_count},{item.second.road_count}) "
            f"score={item.score:7.1f}"
        )


def _add_overhead_camera(rig: SensorRig, corridor: Any, cfg: dict[str, Any]) -> Any:
    import carla

    camera_cfg = cfg["regional"]["bev_camera"]
    center = np.asarray(corridor.center, dtype=np.float64)
    forward = np.asarray(corridor.forward_xy, dtype=np.float64)
    yaw = math.degrees(math.atan2(forward[1], forward[0]))
    return rig.add_camera(
        "global_bev_rgb",
        carla.Transform(
            carla.Location(
                x=float(center[0]),
                y=float(center[1]),
                z=float(center[2] + camera_cfg["height_m"]),
            ),
            carla.Rotation(pitch=-90.0, yaw=float(yaw)),
        ),
        camera_cfg,
    )


def _sensor_range_summary(
    cfg: dict[str, Any], corridor: Any, grid: list[dict[str, Any]]
) -> dict[str, float]:
    """Nominal horizontal ranges and the UAV's ground-intersection footprint."""
    vehicle_m = float(cfg["sensors"]["vehicle_lidar"]["range"])
    rsu_m = float(cfg["sensors"]["rsu_lidar"]["range"])
    uav_cfg = cfg["sensors"]["uav_lidar"]
    uav_slant_m = float(uav_cfg["range"])
    drone_z = float(grid[len(grid) // 2]["location"][2])
    ground_z = float(corridor.second.center[2])
    height_agl = max(0.0, drone_z - ground_z)
    range_limited = math.sqrt(max(0.0, uav_slant_m**2 - height_agl**2))
    upper_fov = float(uav_cfg["upper_fov"])
    if upper_fov < -1e-3:
        fov_limited = height_agl / math.tan(math.radians(abs(upper_fov)))
        uav_ground_m = min(range_limited, fov_limited)
    else:
        uav_ground_m = range_limited
    return {
        "vehicle_m": vehicle_m,
        "rsu_m": rsu_m,
        "uav_slant_m": uav_slant_m,
        "uav_height_agl_m": height_agl,
        "uav_ground_m": uav_ground_m,
    }


def _build_motion_summary(snapshots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_actor: dict[int, list[tuple[float, dict[str, Any]]]] = {}
    for snapshot in snapshots:
        elapsed_s = float(snapshot["elapsed_s"])
        for actor in snapshot["actors"]:
            by_actor.setdefault(int(actor["actor_id"]), []).append((elapsed_s, actor))
    output: list[dict[str, Any]] = []
    for actor_id, samples in by_actor.items():
        samples.sort(key=lambda item: item[0])
        locations = np.asarray([item[1]["location"] for item in samples], dtype=np.float64)
        speeds = np.asarray(
            [np.linalg.norm(np.asarray(item[1]["velocity"][:2], dtype=np.float64)) for item in samples],
            dtype=np.float64,
        )
        displacement = (
            float(np.linalg.norm(locations[-1, :2] - locations[0, :2]))
            if len(locations) > 1
            else 0.0
        )
        sampled_path = (
            float(np.linalg.norm(np.diff(locations[:, :2], axis=0), axis=1).sum())
            if len(locations) > 1
            else 0.0
        )
        output.append(
            {
                "actor_id": actor_id,
                "role_name": str(samples[0][1].get("role_name", "")),
                "type_id": str(samples[0][1].get("type_id", "")),
                "sample_count": len(samples),
                "first_time_s": samples[0][0],
                "last_time_s": samples[-1][0],
                "displacement_m": displacement,
                "sampled_path_m": sampled_path,
                "mean_speed_mps": float(speeds.mean()),
                "stationary_sample_fraction": float(np.mean(speeds < 0.5)),
            }
        )
    return sorted(output, key=lambda item: (item["role_name"], item["actor_id"]))


def _write_motion_report(output_dir: Path, records: list[dict[str, Any]]) -> Path:
    path = output_dir / "preview_motion_report.txt"
    lines = [
        "Active AirV2X preview motion report",
        "===================================",
        "Role / actor                                disp(m)  path(m)  mean(m/s)  stopped",
    ]
    for item in records:
        role = item["role_name"] or item["type_id"]
        label = f"{role}#{item['actor_id']}"
        lines.append(
            f"{label[:40]:40s} "
            f"{item['displacement_m']:8.1f} "
            f"{item['sampled_path_m']:8.1f} "
            f"{item['mean_speed_mps']:10.2f} "
            f"{100.0 * item['stationary_sample_fraction']:7.1f}%"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview a continuous two-junction regional active-UAV scene"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list-corridors", action="store_true")
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    if args.list_corridors:
        list_corridors(args.config, args.limit)
    else:
        preview(args.config, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
