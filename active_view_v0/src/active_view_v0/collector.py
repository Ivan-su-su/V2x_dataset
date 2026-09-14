from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .config import dump_effective_config, load_config, validate_config
from .geometry import make_grid, transform_record
from .junctions import select_junction
from .scenario import ControlledOcclusionScenario
from .sensors import SensorRig


def collect(
    config_path: str | Path,
    overwrite: bool = False,
    duration_s: float | None = None,
    grid_size: int | None = None,
    run_name: str | None = None,
) -> Path:
    import carla

    cfg = load_config(config_path)
    if duration_s is not None:
        cfg["scenario"]["duration_s"] = float(duration_s)
    if grid_size is not None:
        cfg["grid"]["size"] = int(grid_size)
    if run_name is not None:
        cfg["output"]["run_name"] = str(run_name)
    validate_config(cfg)
    carla_cfg = cfg["carla"]
    client = carla.Client(carla_cfg["host"], int(carla_cfg["port"]))
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

    original_settings = world.get_settings()
    traffic_manager = client.get_trafficmanager(int(carla_cfg["traffic_manager_port"]))
    scenario: ControlledOcclusionScenario | None = None
    rig: SensorRig | None = None
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = float(carla_cfg["fixed_delta_seconds"])
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(int(carla_cfg["seed"]))
        world.set_weather(carla.WeatherParameters.ClearNoon)

        junction = select_junction(
            world.get_map(),
            cfg["scenario"].get("junction_id"),
            int(cfg["scenario"].get("junction_rank", 0)),
        )
        scenario = ControlledOcclusionScenario(world, cfg, junction)
        scenario.setup()
        scenario_meta = scenario.metadata()
        center = scenario_meta["event_center_xyz"]
        grid = make_grid(
            center,
            scenario_meta["forward_xy"],
            scenario_meta["right_xy"],
            int(cfg["grid"]["size"]),
            float(cfg["grid"]["spacing_m"]),
            float(cfg["grid"]["height_m"]),
        )

        rig = SensorRig(world, float(carla_cfg["fixed_delta_seconds"]))
        _spawn_rig(rig, scenario, scenario_meta, grid, cfg)
        metadata = {
            "format_version": "active-view-v0.2",
            "map": world.get_map().name,
            "fixed_delta_seconds": float(carla_cfg["fixed_delta_seconds"]),
            "keyframe_interval_s": float(cfg["grid"]["keyframe_interval_s"]),
            "scenario": scenario_meta,
            "grid": grid,
            "candidate_sensor_names": [item["name"] for item in grid],
            "base_sensor_names": list(cfg["scoring"]["base_sensors"]),
            "preview_camera_names": [
                name for name in ("overview_rgb", "ego_front_rgb", "uav_center_rgb")
                if name in rig.streams
            ],
            "sensor_parameters": cfg["sensors"],
        }
        _write_json(run_dir / "metadata.json", metadata)

        dt = float(carla_cfg["fixed_delta_seconds"])
        total_ticks = int(round(float(cfg["scenario"]["duration_s"]) / dt))
        key_stride = int(round(float(cfg["grid"]["keyframe_interval_s"]) / dt))
        manifest_path = run_dir / "manifest.jsonl"
        if metadata["preview_camera_names"]:
            _write_preview_html(run_dir, 0, metadata["preview_camera_names"])
            print(f"Live scene preview: {run_dir / 'scene_preview.html'}")

        # Sensor warmup. Old measurements are discarded when the first target frame is requested.
        for warmup_index in range(3):
            scenario.step(-3.0 * dt + warmup_index * dt)
            world.tick()

        keyframe_count = 0
        for tick_index in range(total_ticks):
            elapsed_s = tick_index * dt
            scenario.step(elapsed_s)
            frame_id = int(world.tick())
            if tick_index % key_stride != 0:
                continue
            measurements = rig.collect_frame(frame_id, timeout_s=20.0)
            frame_dir = run_dir / "frames" / f"{keyframe_count:06d}"
            frame_dir.mkdir(parents=True)
            sensor_records = _save_measurements(frame_dir, measurements)
            gt = _collect_gt(world, np.asarray(center), float(cfg["scenario"]["roi_half_extent_m"]))
            frame_record = {
                "keyframe_index": keyframe_count,
                "carla_frame": frame_id,
                "elapsed_s": elapsed_s,
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
                f"[{keyframe_count:02d}] frame={frame_id} t={elapsed_s:5.1f}s "
                f"actors={len(gt)} sensors={len(measurements)}"
            )
        print(f"Collection complete: {run_dir} ({keyframe_count} keyframes)")
        return run_dir
    finally:
        if rig is not None:
            rig.close()
        if scenario is not None:
            scenario.close()
        try:
            traffic_manager.set_synchronous_mode(False)
        except Exception:
            pass
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass


def _spawn_rig(
    rig: SensorRig,
    scenario: ControlledOcclusionScenario,
    scenario_meta: dict[str, Any],
    grid: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> None:
    import carla

    if scenario.ego is None:
        raise RuntimeError("scenario ego is unavailable")
    vehicle_cfg = cfg["sensors"]["vehicle_lidar"]
    mount = vehicle_cfg["mount"]
    rig.add_lidar(
        "vehicle_lidar",
        carla.Transform(
            carla.Location(x=float(mount["x"]), y=float(mount["y"]), z=float(mount["z"])),
            carla.Rotation(
                pitch=float(mount["pitch"]), yaw=float(mount["yaw"]), roll=float(mount["roll"])
            ),
        ),
        vehicle_cfg,
        attach_to=scenario.ego,
    )

    center = np.asarray(scenario_meta["event_center_xyz"], dtype=np.float64)
    forward = np.asarray(scenario_meta["forward_xy"], dtype=np.float64)
    right = np.asarray(scenario_meta["right_xy"], dtype=np.float64)
    rsu_cfg = cfg["sensors"]["rsu_lidar"]
    rsu_xy = (
        center[:2]
        + float(rsu_cfg["offset_forward_m"]) * forward
        + float(rsu_cfg["offset_right_m"]) * right
    )
    rig.add_lidar(
        "rsu_lidar",
        carla.Transform(
            carla.Location(
                x=float(rsu_xy[0]),
                y=float(rsu_xy[1]),
                z=float(center[2] + float(rsu_cfg["height_m"])),
            ),
            carla.Rotation(),
        ),
        rsu_cfg,
    )
    uav_cfg = cfg["sensors"]["uav_lidar"]
    for item in grid:
        x, y, z = item["location"]
        rig.add_lidar(
            item["name"],
            carla.Transform(carla.Location(x=float(x), y=float(y), z=float(z)), carla.Rotation()),
            uav_cfg,
        )
    camera_cfg = cfg["sensors"].get("preview_cameras", {})
    if not bool(camera_cfg.get("enabled", True)):
        return
    route_yaw = math.degrees(math.atan2(forward[1], forward[0]))
    overview_xy = center[:2] - 32.0 * forward
    rig.add_camera(
        "overview_rgb",
        carla.Transform(
            carla.Location(
                x=float(overview_xy[0]), y=float(overview_xy[1]), z=float(center[2] + 22.0)
            ),
            carla.Rotation(pitch=-34.0, yaw=float(route_yaw)),
        ),
        camera_cfg,
    )
    rig.add_camera(
        "ego_front_rgb",
        carla.Transform(
            carla.Location(x=1.5, z=1.8),
            carla.Rotation(pitch=-5.0),
        ),
        camera_cfg,
        attach_to=scenario.ego,
    )
    rig.add_camera(
        "uav_center_rgb",
        carla.Transform(
            carla.Location(
                x=float(center[0]),
                y=float(center[1]),
                z=float(center[2] + float(cfg["grid"]["height_m"])),
            ),
            carla.Rotation(pitch=-90.0, yaw=float(route_yaw)),
        ),
        camera_cfg,
    )


def _save_measurements(frame_dir: Path, measurements: dict[str, Any]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name, measurement in measurements.items():
        if hasattr(measurement, "width") and hasattr(measurement, "height"):
            relative = Path("rgb") / f"{name}.png"
            path = frame_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            measurement.save_to_disk(str(path))
            records[name] = {
                "modality": "rgb",
                "path": str(relative),
                "width": int(measurement.width),
                "height": int(measurement.height),
                "transform": transform_record(measurement.transform),
            }
            continue
        relative = Path("lidar") / f"{name}.bin"
        path = frame_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        array = np.frombuffer(measurement.raw_data, dtype=np.float32).reshape(-1, 4)
        array.tofile(path)
        records[name] = {
            "modality": "lidar",
            "path": str(relative),
            "point_count": int(array.shape[0]),
            "transform": transform_record(measurement.transform),
        }
    return records


def _collect_gt(world: Any, center_xyz: np.ndarray, half_extent_m: float) -> list[dict[str, Any]]:
    actors = list(world.get_actors().filter("vehicle.*")) + list(
        world.get_actors().filter("walker.pedestrian.*")
    )
    records: list[dict[str, Any]] = []
    for actor in actors:
        transform = actor.get_transform()
        location = transform.location
        if (
            abs(float(location.x) - float(center_xyz[0])) > half_extent_m
            or abs(float(location.y) - float(center_xyz[1])) > half_extent_m
        ):
            continue
        bbox = actor.bounding_box
        velocity = actor.get_velocity()
        acceleration = actor.get_acceleration()
        records.append(
            {
                "actor_id": int(actor.id),
                "type_id": actor.type_id,
                "class_name": _class_name(actor),
                "role_name": actor.attributes.get("role_name", ""),
                "actor_transform": transform_record(transform),
                "bbox_center_local": [float(bbox.location.x), float(bbox.location.y), float(bbox.location.z)],
                "bbox_extent": [float(bbox.extent.x), float(bbox.extent.y), float(bbox.extent.z)],
                "velocity_world": [float(velocity.x), float(velocity.y), float(velocity.z)],
                "acceleration_world": [
                    float(acceleration.x),
                    float(acceleration.y),
                    float(acceleration.z),
                ],
            }
        )
    return records


def _class_name(actor: Any) -> str:
    if actor.type_id.startswith("walker.pedestrian"):
        return "pedestrian"
    wheels = actor.attributes.get("number_of_wheels", "4")
    if str(wheels) == "2":
        return "two_wheeler"
    return "car"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_preview_html(run_dir: Path, frame_count: int, camera_names: list[str]) -> None:
    labels = {
        "overview_rgb": "路口斜视总览",
        "ego_front_rgb": "车辆前视",
        "uav_center_rgb": "25m 中心俯视",
    }
    panels = "\n".join(
        f'<section><h2>{labels.get(name, name)}</h2><img id="{name}" alt="{name}"></section>'
        for name in camera_names
    )
    names_json = json.dumps(camera_names)
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>ActiveView-v0 场景预览</title>
<style>
body{{margin:0;background:#0f172a;color:#e2e8f0;font-family:system-ui,sans-serif}}
header{{position:sticky;top:0;background:#111827;padding:14px 20px;z-index:2}}
main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:14px;padding:14px}}
section{{background:#1e293b;padding:12px;border-radius:10px}} h2{{font-size:18px;margin:0 0 8px}}
img{{display:block;width:100%;height:auto;background:#020617}} input[type=range]{{width:min(720px,70vw)}}
</style></head><body><header><b>ActiveView-v0 真实 CARLA RGB 场景</b><br>
<input id="slider" type="range" min="0" max="{max(0, frame_count - 1)}" value="0" step="1">
<span id="counter"></span>　<label><input id="follow" type="checkbox" checked>跟随最新帧</label>　<button id="play">播放</button></header><main>{panels}</main>
<script>
const names={names_json}; let count={frame_count}; const slider=document.getElementById('slider');
const counter=document.getElementById('counter'); let timer=null;
function pad(n){{return String(n).padStart(6,'0')}}
function show(n){{for(const name of names) document.getElementById(name).src=`frames/${{pad(n)}}/rgb/${{name}}.png`; counter.textContent=`关键帧 ${{n+1}} / ${{count}}`;}}
slider.oninput=()=>{{document.getElementById('follow').checked=false;show(Number(slider.value));}};
document.getElementById('play').onclick=()=>{{if(timer){{clearInterval(timer);timer=null;return}} timer=setInterval(()=>{{if(count<1)return;slider.value=(Number(slider.value)+1)%count;show(Number(slider.value));}},700)}};
async function refresh(){{try{{const response=await fetch(`manifest.jsonl?t=${{Date.now()}}`);if(!response.ok)return;const lines=(await response.text()).trim().split(/\n+/).filter(Boolean);count=lines.length;slider.max=Math.max(0,count-1);if(count&&document.getElementById('follow').checked){{slider.value=count-1;show(count-1)}}}}catch(e){{}}}}
refresh();setInterval(refresh,1000);
</script></body></html>"""
    (run_dir / "scene_preview.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect the ActiveView-v0 CARLA pilot")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--duration-s", type=float, default=None, help="temporary smoke-test override")
    parser.add_argument("--grid-size", type=int, default=None, help="temporary odd-grid override")
    parser.add_argument("--run-name", default=None, help="temporary output run-name override")
    args = parser.parse_args()
    collect(
        args.config,
        overwrite=args.overwrite,
        duration_s=args.duration_s,
        grid_size=args.grid_size,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
