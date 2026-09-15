from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .bev_render import annotate_rgb_snapshot
from .collector import _class_name, _save_measurements, _write_json
from .config import dump_effective_config, load_config
from .corridors import select_connected_junctions
from .geometry import carla_matrix, points_in_regional_roi, transform_points, transform_record
from .regional_layout import build_regional_layout
from .regional_preview import _add_overhead_camera, _sensor_range_summary
from .regional_scenario import RegionalIntersectionScenario
from .sensors import SensorRig
from .uav_modes import UAVModePlanner


def collect_regional(
    config_path: str | Path,
    overwrite: bool = False,
    duration_s: float | None = None,
    run_name: str | None = None,
) -> Path:
    """Collect a synchronized regional counterfactual-view pilot."""
    import carla

    cfg = load_config(config_path)
    if duration_s is not None:
        cfg["regional"]["duration_s"] = float(duration_s)
        if "frame_count" in cfg["regional"]:
            interval = float(
                cfg["regional"].get(
                    "collection_interval_s", cfg["grid"]["keyframe_interval_s"]
                )
            )
            cfg["regional"]["frame_count"] = int(round(float(duration_s) / interval))
    if run_name is not None:
        cfg["output"]["run_name"] = str(run_name)
    regional = cfg["regional"]
    carla_cfg = cfg["carla"]
    client = carla.Client(str(carla_cfg["host"]), int(carla_cfg["port"]))
    client.set_timeout(float(carla_cfg["timeout_s"]))
    world = client.get_world()
    target_map = str(carla_cfg["map"])
    if not world.get_map().name.endswith(target_map):
        print(f"Loading {target_map}; current actors will be cleared by CARLA.")
        world = client.load_world(target_map)

    run_dir = Path(cfg["output"]["root"]).expanduser() / str(cfg["output"]["run_name"])
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{run_dir} exists; pass --overwrite to replace this run only")
        shutil.rmtree(run_dir)
    (run_dir / "frames").mkdir(parents=True)
    dump_effective_config(cfg, run_dir / "effective_config.yaml")

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

    original_settings = world.get_settings()
    traffic_manager = client.get_trafficmanager(int(carla_cfg["traffic_manager_port"]))
    scenario: RegionalIntersectionScenario | None = None
    rig: SensorRig | None = None
    mode_planner: UAVModePlanner | None = None
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
        collection_interval_s = float(
            regional.get("collection_interval_s", cfg["grid"]["keyframe_interval_s"])
        )
        decision_interval_s = float(
            regional.get("uav_modes", {}).get(
                "decision_interval_s", cfg["grid"]["keyframe_interval_s"]
            )
        )
        mode_cfg = regional.get("uav_modes")
        if mode_cfg:
            mode_planner = UAVModePlanner(
                grid,
                mode_cfg,
                frame_interval_s=collection_interval_s,
                max_speed_mps=float(cfg["oracle"]["uav_speed_mps"]),
            )
        mode_sensor_names = (
            {
                mode: f"uav_mode_{mode}_lidar"
                for mode in UAVModePlanner.MODE_NAMES
            }
            if mode_planner is not None
            else {}
        )

        rig = SensorRig(world, float(carla_cfg["fixed_delta_seconds"]))
        _spawn_regional_rig(
            rig,
            scenario,
            rsu_xyz,
            grid,
            cfg,
            mode_sensor_names=mode_sensor_names,
        )
        overhead_stream = rig.streams.get("global_bev_rgb")
        overhead_camera = overhead_stream.actor if overhead_stream is not None else None
        annotation_cfg = regional.get("annotated_bev", {})
        save_annotated_bev = bool(
            annotation_cfg.get("enabled", True) and overhead_camera is not None
        )
        annotated_bev_dir = run_dir / "visualization" / "annotated_bev_frames"
        if save_annotated_bev:
            annotated_bev_dir.mkdir(parents=True, exist_ok=True)
        annotated_bev_paths: list[Path] = []
        sensor_ranges = _sensor_range_summary(cfg, corridor, grid)

        roi_mode = str(cfg["scoring"].get("roi_mode", "ego_moving"))
        ego_roi = cfg["scoring"]["ego_roi"]
        base_names = list(cfg["scoring"]["base_sensors"])
        candidate_names = [item["name"] for item in grid]
        missing = [
            name
            for name in [*base_names, *candidate_names, *mode_sensor_names.values()]
            if name not in rig.streams
        ]
        if missing:
            raise RuntimeError(f"required sensor streams were not created: {missing}")
        metadata = {
            "format_version": "active-view-regional-collection-v1.2",
            "map": world.get_map().name,
            "fixed_delta_seconds": float(carla_cfg["fixed_delta_seconds"]),
            # keyframe_interval_s remains the saved-frame interval for older
            # readers. Active controllers must use decision_interval_s.
            "keyframe_interval_s": collection_interval_s,
            "frame_interval_s": collection_interval_s,
            "decision_interval_s": decision_interval_s,
            "scenario": scenario_meta,
            "corridor": corridor.as_dict(),
            "rsu_xyz": rsu_xyz,
            "grid_center_xyz": layout["grid_center_xyz"],
            "grid": grid,
            "candidate_sensor_names": candidate_names,
            "base_sensor_names": base_names,
            "ego_roi": ego_roi,
            "roi_mode": roi_mode,
            "evaluation_roi": evaluation_roi,
            "sensor_parameters": cfg["sensors"],
            "uav_mode_names": list(mode_sensor_names),
            "uav_mode_sensor_names": mode_sensor_names,
            "visualization": {
                "annotated_overhead_frames": (
                    str(annotated_bev_dir.relative_to(run_dir))
                    if save_annotated_bev
                    else None
                ),
                "annotated_overhead_video": (
                    str(
                        (
                            run_dir
                            / "visualization"
                            / str(
                                annotation_cfg.get(
                                    "video_filename",
                                    "global_bev_rgb_annotated.mp4",
                                )
                            )
                        ).relative_to(run_dir)
                    )
                    if save_annotated_bev
                    and bool(annotation_cfg.get("make_video", True))
                    else None
                ),
            },
        }
        _write_json(run_dir / "metadata.json", metadata)

        dt = float(carla_cfg["fixed_delta_seconds"])
        warmup_ticks = int(round(float(regional["warmup_s"]) / dt))
        for _ in range(warmup_ticks):
            scenario.apply_fixed_signal_plan(0.0)
            world.tick()
            scenario.apply_fixed_signal_plan(0.0)
            scenario.assert_fixed_signal_plan(0.0)

        duration = float(regional["duration_s"])
        total_loop_ticks, key_stride, expected_frames = _collection_schedule(cfg)
        manifest_path = run_dir / "manifest.jsonl"
        keyframe_count = 0
        evaluation_target_counts: list[int] = []
        moving_target_counts: list[int] = []
        ego_locations: list[list[float]] = []
        ego_speeds: list[float] = []
        ignored_gt_roles = set(str(role) for role in cfg["scoring"].get("ignore_roles", []))
        for tick_index in range(total_loop_ticks):
            elapsed_s = tick_index * dt
            mode_record = None
            if tick_index % key_stride == 0 and mode_planner is not None:
                mode_record = mode_planner.step(
                    keyframe_count,
                    elapsed_s,
                    scenario.ego.get_transform(),
                )
                _set_uav_mode_sensor_transforms(
                    rig,
                    mode_record,
                    mode_sensor_names,
                )
            scenario.apply_fixed_signal_plan(elapsed_s)
            frame_id = int(world.tick())
            scenario.apply_fixed_signal_plan(elapsed_s)
            scenario.assert_fixed_signal_plan(elapsed_s)
            if tick_index % key_stride != 0:
                continue
            measurements = rig.collect_frame(frame_id, timeout_s=30.0)
            frame_dir = run_dir / "frames" / f"{keyframe_count:06d}"
            frame_dir.mkdir(parents=True)
            sensor_records = _save_measurements(frame_dir, measurements)
            ego_transform = scenario.ego.get_transform()
            ego_velocity = scenario.ego.get_velocity()
            ego_locations.append(
                [float(ego_transform.location.x), float(ego_transform.location.y)]
            )
            ego_speeds.append(
                float(np.hypot(float(ego_velocity.x), float(ego_velocity.y)))
            )
            if mode_planner is not None and mode_record is None:
                raise RuntimeError("UAV mode record was not prepared for a saved frame")
            gt = _collect_gt(
                world,
                ego_transform,
                roi_mode=roi_mode,
                ego_roi=ego_roi,
                regional_roi=evaluation_roi,
            )
            evaluation_gt = [item for item in gt if item.get("role_name", "") not in ignored_gt_roles]
            moving_gt = [
                item
                for item in evaluation_gt
                if float(np.linalg.norm(np.asarray(item["velocity_world"][:2], dtype=np.float64)))
                >= 0.5
            ]
            evaluation_target_counts.append(len(evaluation_gt))
            moving_target_counts.append(len(moving_gt))
            tracked_roles = set(
                str(role)
                for role in regional.get(
                    "tracked_roles",
                    list(regional.get("agent_sensors", {}))
                    or ["regional_ego", "regional_car1", "regional_car3"],
                )
            )
            actor_states = scenario.actor_states()
            sensing_agents = {
                state["role_name"]: state
                for state in actor_states
                if state["role_name"] in tracked_roles
            }
            visualization_record: dict[str, Any] = {}
            if save_annotated_bev:
                raw_rgb_record = sensor_records.get("global_bev_rgb")
                if raw_rgb_record is None:
                    raise RuntimeError(
                        "annotated BEV is enabled but global_bev_rgb was not collected"
                    )
                raw_rgb_path = frame_dir / str(raw_rgb_record["path"])
                annotated_path = annotated_bev_dir / f"{keyframe_count:06d}.png"
                annotate_rgb_snapshot(
                    raw_rgb_path,
                    annotated_path,
                    overhead_camera,
                    actor_states,
                    rsu_xyz,
                    grid,
                    [list(corridor.first.center), list(corridor.second.center)],
                    elapsed_s,
                    sensor_ranges=sensor_ranges,
                    evaluation_roi=evaluation_roi,
                )
                annotated_bev_paths.append(annotated_path)
                visualization_record["annotated_global_bev_rgb"] = str(
                    annotated_path.relative_to(run_dir)
                )
            frame_record = {
                "keyframe_index": keyframe_count,
                "carla_frame": frame_id,
                "elapsed_s": elapsed_s,
                "ego_transform": transform_record(ego_transform),
                "roi_mode": roi_mode,
                "evaluation_roi": evaluation_roi,
                "evaluation_target_count": len(evaluation_gt),
                "moving_evaluation_target_count": len(moving_gt),
                "sensing_agents": sensing_agents,
                "fixed_signal_states": scenario.fixed_signal_states(),
                "uav_decision": (
                    {
                        "is_decision_frame": mode_record["is_decision_frame"],
                        "decision_index": mode_record["decision_index"],
                        "decision_interval_s": decision_interval_s,
                    }
                    if mode_record is not None
                    else None
                ),
                "uav_modes": mode_record["modes"] if mode_record is not None else {},
                "visualization": visualization_record,
                "sensors": sensor_records,
                "gt": gt,
            }
            _write_json(frame_dir / "frame.json", frame_record)
            with manifest_path.open("a", encoding="utf-8") as manifest:
                manifest.write(
                    json.dumps(
                        {
                            "keyframe_index": keyframe_count,
                            "carla_frame": frame_id,
                            "elapsed_s": elapsed_s,
                            "path": str(frame_dir.relative_to(run_dir)),
                        }
                    )
                    + "\n"
                )
            keyframe_count += 1
            print(
                f"[{keyframe_count:03d}/{expected_frames:03d}] frame={frame_id} t={elapsed_s:5.1f}s "
                f"targets={len(evaluation_gt)} moving={len(moving_gt)} sensors={len(measurements)}"
            )
        if mode_planner is not None:
            _write_json(run_dir / "uav_modes.json", mode_planner.summary())
        video_path = None
        if save_annotated_bev and bool(annotation_cfg.get("make_video", True)):
            video_path = (
                run_dir
                / "visualization"
                / str(
                    annotation_cfg.get(
                        "video_filename",
                        "global_bev_rgb_annotated.mp4",
                    )
                )
            )
            _encode_annotated_video(
                annotated_bev_paths,
                video_path,
                fps=float(
                    annotation_cfg.get(
                        "video_fps",
                        1.0 / collection_interval_s,
                    )
                ),
                codec=str(annotation_cfg.get("video_codec", "libx264")),
                quality=int(annotation_cfg.get("video_quality", 8)),
            )
            print(f"Annotated BEV video: {video_path}")
        density_cfg = regional.get("density_validation", {})
        minimum_targets = int(density_cfg.get("minimum_targets_per_frame", 1))
        minimum_mean = float(density_cfg.get("minimum_mean_targets", 1.0))
        observed_minimum = int(min(evaluation_target_counts))
        observed_mean = float(np.mean(evaluation_target_counts))
        ego_locations_array = np.asarray(ego_locations, dtype=np.float64)
        ego_displacement_m = (
            float(np.linalg.norm(ego_locations_array[-1] - ego_locations_array[0]))
            if len(ego_locations_array) > 1
            else 0.0
        )
        ego_path_m = (
            float(np.linalg.norm(np.diff(ego_locations_array, axis=0), axis=1).sum())
            if len(ego_locations_array) > 1
            else 0.0
        )
        mean_ego_speed_mps = float(np.mean(ego_speeds))
        corridor_forward = np.asarray(corridor.forward_xy, dtype=np.float64)
        j1_center_xy = np.asarray(corridor.first.center[:2], dtype=np.float64)
        j2_center_xy = np.asarray(corridor.second.center[:2], dtype=np.float64)
        ego_j1_progress_m = (ego_locations_array - j1_center_xy) @ corridor_forward
        ego_j2_distance_m = np.linalg.norm(
            ego_locations_array - j2_center_xy,
            axis=1,
        )
        maximum_j1_progress_m = float(np.max(ego_j1_progress_m))
        minimum_j2_distance_m = float(np.min(ego_j2_distance_m))
        minimum_j1_cross_progress_m = float(
            density_cfg.get("minimum_j1_cross_progress_m", 0.0)
        )
        maximum_ego_j2_distance_m = float(
            density_cfg.get("maximum_ego_j2_distance_m", float("inf"))
        )
        minimum_ego_displacement_m = float(
            density_cfg.get("minimum_ego_displacement_m", 0.0)
        )
        minimum_mean_ego_speed_mps = float(
            density_cfg.get("minimum_mean_ego_speed_mps", 0.0)
        )
        health_failures: list[str] = []
        if observed_minimum < minimum_targets:
            health_failures.append(
                f"minimum target count {observed_minimum} < {minimum_targets}"
            )
        if observed_mean < minimum_mean:
            health_failures.append(
                f"mean target count {observed_mean:.2f} < {minimum_mean:.2f}"
            )
        if maximum_j1_progress_m < minimum_j1_cross_progress_m:
            health_failures.append(
                f"Ego never cleared J1: max progress {maximum_j1_progress_m:.2f}m < "
                f"{minimum_j1_cross_progress_m:.2f}m"
            )
        if minimum_j2_distance_m > maximum_ego_j2_distance_m:
            health_failures.append(
                f"Ego never approached J2: min distance {minimum_j2_distance_m:.2f}m > "
                f"{maximum_ego_j2_distance_m:.2f}m"
            )
        if ego_displacement_m < minimum_ego_displacement_m:
            health_failures.append(
                f"Ego displacement {ego_displacement_m:.2f}m < "
                f"{minimum_ego_displacement_m:.2f}m"
            )
        if mean_ego_speed_mps < minimum_mean_ego_speed_mps:
            health_failures.append(
                f"Ego mean speed {mean_ego_speed_mps:.2f}m/s < "
                f"{minimum_mean_ego_speed_mps:.2f}m/s"
            )
        _write_json(
            run_dir / "dataset_health.json",
            {
                "passed": not health_failures,
                "failures": health_failures,
                "minimum_targets_required": minimum_targets,
                "minimum_mean_targets_required": minimum_mean,
                "target_count_per_frame": evaluation_target_counts,
                "moving_target_count_per_frame": moving_target_counts,
                "minimum_target_count": observed_minimum,
                "mean_target_count": observed_mean,
                "mean_moving_target_count": float(np.mean(moving_target_counts)),
                "ego_displacement_m": ego_displacement_m,
                "ego_sampled_path_m": ego_path_m,
                "mean_ego_speed_mps": mean_ego_speed_mps,
                "maximum_j1_progress_m": maximum_j1_progress_m,
                "minimum_j2_distance_m": minimum_j2_distance_m,
                "minimum_j1_cross_progress_required_m": minimum_j1_cross_progress_m,
                "maximum_ego_j2_distance_required_m": maximum_ego_j2_distance_m,
                "minimum_ego_displacement_required_m": minimum_ego_displacement_m,
                "minimum_mean_ego_speed_required_mps": minimum_mean_ego_speed_mps,
            },
        )
        if health_failures:
            print("WARNING: dense-scene health gate failed: " + "; ".join(health_failures))
        _write_json(
            run_dir / "collection_status.json",
            {
                "status": "complete",
                "keyframe_count": keyframe_count,
                "expected_keyframe_count": expected_frames,
                "duration_s": duration,
                "frame_interval_s": collection_interval_s,
                "decision_interval_s": decision_interval_s,
                "annotated_bev_frame_count": len(annotated_bev_paths),
                "annotated_bev_video": (
                    str(video_path.relative_to(run_dir))
                    if video_path is not None
                    else None
                ),
            },
        )
        if keyframe_count != expected_frames:
            raise RuntimeError(
                f"collection wrote {keyframe_count} frames, expected {expected_frames}"
            )
        print(f"Regional collection complete: {run_dir} ({keyframe_count} keyframes)")
        return run_dir
    finally:
        if rig is not None:
            rig.stop()
        if scenario is not None:
            scenario.release_fixed_signal_plan()
        # CARLA 0.9.16 can terminate the whole Python interpreter when many
        # attached sensors and their parent vehicles are destroyed one by one.
        # A single server-side batch avoids calling methods on wrappers whose
        # parent may already have been destroyed.
        actor_ids: list[int] = []
        if rig is not None:
            actor_ids.extend(int(stream.actor.id) for stream in rig.streams.values())
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
                    print(f"WARNING: {len(errors)} actor cleanup command(s) failed: {errors[:3]}")
            except Exception as error:
                print(f"WARNING: batched CARLA cleanup failed: {error}")
        if rig is not None:
            rig.streams.clear()
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


def _encode_annotated_video(
    frame_paths: list[Path],
    output_path: Path,
    *,
    fps: float,
    codec: str,
    quality: int,
) -> Path:
    """Encode sequential annotated PNG frames without loading them all at once."""
    if not frame_paths:
        raise RuntimeError("cannot encode annotated BEV video: no frames were written")
    if fps <= 0.0:
        raise ValueError("annotated BEV video fps must be positive")

    import imageio.v2 as imageio

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        str(output_path),
        fps=float(fps),
        codec=str(codec),
        quality=int(quality),
        macro_block_size=2,
    ) as writer:
        for frame_path in frame_paths:
            image = imageio.imread(frame_path)
            if image.ndim == 3 and image.shape[2] == 4:
                image = image[:, :, :3]
            writer.append_data(image)
    return output_path


def _spawn_regional_rig(
    rig: SensorRig,
    scenario: RegionalIntersectionScenario,
    rsu_xyz: list[float],
    grid: list[dict[str, Any]],
    cfg: dict[str, Any],
    *,
    mode_sensor_names: dict[str, str] | None = None,
) -> None:
    import carla

    vehicle_cfg = cfg["sensors"]["vehicle_lidar"]
    mount = vehicle_cfg["mount"]
    mount_transform = carla.Transform(
        carla.Location(x=float(mount["x"]), y=float(mount["y"]), z=float(mount["z"])),
        carla.Rotation(
            pitch=float(mount["pitch"]), yaw=float(mount["yaw"]), roll=float(mount["roll"])
        ),
    )
    agent_sensor_names = cfg["regional"].get(
        "agent_sensors",
        {
            "regional_ego": "ego_lidar",
            "regional_car1": "car1_lidar",
            "regional_car3": "car3_lidar",
        },
    )
    for role, sensor_name in agent_sensor_names.items():
        route = scenario.routes.get(role)
        if route is None:
            raise RuntimeError(f"required CAV actor is missing: {role}")
        rig.add_lidar(sensor_name, mount_transform, vehicle_cfg, attach_to=route.actor)

    rsu_lidar_cfg = cfg["sensors"]["rsu_lidar"]
    rig.add_lidar(
        "rsu_lidar",
        carla.Transform(
            carla.Location(x=float(rsu_xyz[0]), y=float(rsu_xyz[1]), z=float(rsu_xyz[2])),
            carla.Rotation(),
        ),
        rsu_lidar_cfg,
    )
    uav_cfg = cfg["sensors"]["uav_lidar"]
    for item in grid:
        x, y, z = item["location"]
        rig.add_lidar(
            item["name"],
            carla.Transform(carla.Location(x=float(x), y=float(y), z=float(z)), carla.Rotation()),
            uav_cfg,
        )
    if mode_sensor_names:
        initial_name = str(
            cfg["regional"].get("uav_modes", {}).get(
                "initial_candidate", grid[len(grid) // 2]["name"]
            )
        )
        by_name = {str(item["name"]): item for item in grid}
        if initial_name not in by_name:
            raise ValueError(f"unknown initial UAV candidate: {initial_name}")
        x, y, z = by_name[initial_name]["location"]
        initial_transform = carla.Transform(
            carla.Location(x=float(x), y=float(y), z=float(z)),
            carla.Rotation(),
        )
        for sensor_name in mode_sensor_names.values():
            rig.add_lidar(sensor_name, initial_transform, uav_cfg)
    if bool(cfg["regional"].get("save_overhead_rgb", True)):
        _add_overhead_camera(rig, scenario.corridor, cfg)


def _set_uav_mode_sensor_transforms(
    rig: SensorRig,
    mode_record: dict[str, Any],
    mode_sensor_names: dict[str, str],
) -> None:
    """Move the three physical UAV sensors before the synchronous world tick."""
    import carla

    for mode, sensor_name in mode_sensor_names.items():
        location = mode_record["modes"][mode]["location_world"]
        rig.streams[sensor_name].actor.set_transform(
            carla.Transform(
                carla.Location(
                    x=float(location[0]),
                    y=float(location[1]),
                    z=float(location[2]),
                ),
                carla.Rotation(),
            )
        )


def _collect_gt(
    world: Any,
    ego_transform: Any,
    *,
    roi_mode: str,
    ego_roi: dict[str, list[float]],
    regional_roi: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    minimum = np.asarray(ego_roi["min"], dtype=np.float64)
    maximum = np.asarray(ego_roi["max"], dtype=np.float64)
    world_to_ego = np.asarray(ego_transform.get_inverse_matrix(), dtype=np.float64)
    world_to_regional = (
        None
        if regional_roi is None
        else np.asarray(regional_roi["world_to_regional"], dtype=np.float64)
    )
    actors = list(world.get_actors().filter("vehicle.*")) + list(
        world.get_actors().filter("walker.pedestrian.*")
    )
    records: list[dict[str, Any]] = []
    for actor in actors:
        transform = actor.get_transform()
        bbox = actor.bounding_box
        bbox_center_world = transform_points(
            np.asarray([[bbox.location.x, bbox.location.y, bbox.location.z]], dtype=np.float64),
            carla_matrix(transform),
        )[0]
        bbox_center_ego = transform_points(bbox_center_world[None, :], world_to_ego)[0]
        bbox_center_regional = None
        if roi_mode == "regional_fixed":
            if regional_roi is None or world_to_regional is None:
                raise ValueError("regional_fixed mode requires a resolved regional ROI")
            if not bool(points_in_regional_roi(bbox_center_world[None, :], regional_roi)[0]):
                continue
            bbox_center_regional = transform_points(
                bbox_center_world[None, :], world_to_regional
            )[0]
        elif roi_mode == "ego_moving":
            # Keep only horizontal bounds so buses are not rejected by height.
            if not np.all(
                (bbox_center_ego[:2] >= minimum[:2])
                & (bbox_center_ego[:2] <= maximum[:2])
            ):
                continue
        else:
            raise ValueError(f"unknown roi_mode: {roi_mode}")
        velocity = actor.get_velocity()
        acceleration = actor.get_acceleration()
        records.append(
            {
                "actor_id": int(actor.id),
                "track_id": int(actor.id),
                "type_id": actor.type_id,
                "class_name": _class_name(actor),
                "role_name": actor.attributes.get("role_name", ""),
                "actor_transform": transform_record(transform),
                "bbox_center_local": [float(bbox.location.x), float(bbox.location.y), float(bbox.location.z)],
                "bbox_center_world": bbox_center_world.tolist(),
                "bbox_center_ego": bbox_center_ego.tolist(),
                "bbox_center_regional": (
                    None if bbox_center_regional is None else bbox_center_regional.tolist()
                ),
                "bbox_extent": [float(bbox.extent.x), float(bbox.extent.y), float(bbox.extent.z)],
                "velocity_world": [float(velocity.x), float(velocity.y), float(velocity.z)],
                "acceleration_world": [
                    float(acceleration.x), float(acceleration.y), float(acceleration.z)
                ],
            }
        )
    return records


def _collection_schedule(cfg: dict[str, Any]) -> tuple[int, int, int]:
    """Return loop ticks, saved-frame stride, and expected saved frames."""
    regional = cfg["regional"]
    dt = float(cfg["carla"]["fixed_delta_seconds"])
    interval = float(
        regional.get("collection_interval_s", cfg["grid"]["keyframe_interval_s"])
    )
    stride_float = interval / dt
    if abs(stride_float - round(stride_float)) > 1e-6:
        raise ValueError("regional.collection_interval_s must be a multiple of fixed_delta_seconds")
    stride = int(round(stride_float))
    if "frame_count" in regional and regional["frame_count"] is not None:
        expected = int(regional["frame_count"])
        if expected < 1:
            raise ValueError("regional.frame_count must be positive")
        loop_ticks = (expected - 1) * stride + 1
        return loop_ticks, stride, expected
    duration_ticks = int(round(float(regional["duration_s"]) / dt))
    include_endpoint = bool(regional.get("include_endpoint", True))
    loop_ticks = duration_ticks + int(include_endpoint)
    expected = (max(loop_ticks - 1, 0) // stride) + int(loop_ticks > 0)
    return loop_ticks, stride, expected


def _collect_gt_in_ego_roi(
    world: Any, ego_transform: Any, roi: dict[str, list[float]]
) -> list[dict[str, Any]]:
    """Backward-compatible wrapper used by earlier tests and callers."""
    return _collect_gt(
        world,
        ego_transform,
        roi_mode="ego_moving",
        ego_roi=roi,
        regional_roi=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect the regional 25-view Active AirV2X pilot")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    collect_regional(
        args.config,
        overwrite=args.overwrite,
        duration_s=args.duration_s,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
