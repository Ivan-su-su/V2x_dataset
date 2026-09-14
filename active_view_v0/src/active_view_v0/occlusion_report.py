from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config


def build_occlusion_report(run_dir: str | Path) -> tuple[Path, Path]:
    """Verify that the moving blocker causes real ray-cast visibility loss.

    The check does not trust the scripted actor roles alone.  A target counts
    only when the blocker is geometrically between CAV-2 and the target, the
    target is inside CAV-2's configured LiDAR range, CAV-2 is below the class
    point threshold, and at least one UAV candidate sees it.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    cfg = load_config(run_dir / "effective_config.yaml")
    details = json.loads((run_dir / "geometry_details.json").read_text(encoding="utf-8"))
    manifest = [
        json.loads(line)
        for line in (run_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    regional = cfg["regional"]
    blocker_role = str(regional["blocker_role"])
    target_roles = {str(role) for role in regional["occlusion_target_roles"]}
    cav_role = "regional_car3"
    cav_sensor = str(regional["agent_sensors"][cav_role])
    lidar_range = float(cfg["sensors"]["vehicle_lidar"]["range"])
    thresholds = {key: int(value) for key, value in cfg["scoring"]["threshold_points"].items()}

    frame_records: list[dict[str, Any]] = []
    physical_frames = 0
    uav_visibility_frames = 0
    system_rescue_frames = 0
    for detail, entry in zip(details, manifest):
        frame = json.loads((run_dir / entry["path"] / "frame.json").read_text(encoding="utf-8"))
        cav_state = frame.get("sensing_agents", {}).get(cav_role)
        blocker = next(
            (item for item in frame["gt"] if item.get("role_name") == blocker_role),
            None,
        )
        qualified: list[dict[str, Any]] = []
        if cav_state is not None and blocker is not None:
            cav_xy = np.asarray(cav_state["location"][:2], dtype=np.float64)
            blocker_xy = np.asarray(blocker["bbox_center_world"][:2], dtype=np.float64)
            cav_support = detail.get("base_sensor_support_points", {}).get(cav_sensor)
            if cav_support is None:
                raise KeyError(f"geometry details do not contain isolated support for {cav_sensor}")
            for target_index, role in enumerate(detail["target_roles"]):
                if role not in target_roles:
                    continue
                target_xy = np.asarray(detail["target_centers_world"][target_index][:2], dtype=np.float64)
                geometry = _blocker_between(cav_xy, blocker_xy, target_xy)
                in_range = float(np.linalg.norm(target_xy - cav_xy)) <= lidar_range
                class_name = str(detail["target_classes"][target_index])
                cav_visible = int(cav_support[target_index]) >= thresholds[class_name]
                uav_visible = any(
                    bool(candidate["visible"][target_index])
                    for candidate in detail["candidates"].values()
                )
                base_visible = bool(detail["base_visible"][target_index])
                system_rescuable = any(
                    bool(candidate["union_visible"][target_index])
                    for candidate in detail["candidates"].values()
                ) and not base_visible
                if geometry["between"] and in_range and not cav_visible:
                    qualified.append(
                        {
                            "target_id": int(detail["target_ids"][target_index]),
                            "target_role": role,
                            "distance_from_cav_m": float(np.linalg.norm(target_xy - cav_xy)),
                            "blocker_lateral_error_m": geometry["lateral_error_m"],
                            "cav2_support_points": int(cav_support[target_index]),
                            "uav_visible": uav_visible,
                            "system_rescuable": system_rescuable,
                        }
                    )
        has_physical = bool(qualified)
        has_uav_visibility = any(item["uav_visible"] for item in qualified)
        has_system_rescue = any(item["system_rescuable"] for item in qualified)
        physical_frames += int(has_physical)
        uav_visibility_frames += int(has_uav_visibility)
        system_rescue_frames += int(has_system_rescue)
        frame_records.append(
            {
                "elapsed_s": float(detail["elapsed_s"]),
                "physical_occlusion": has_physical,
                "uav_sees_occluded_target": has_uav_visibility,
                "occlusion_is_system_rescuable": has_system_rescue,
                "targets": qualified,
            }
        )

    validation = regional.get("validation", {})
    minimum_physical = int(validation.get("minimum_occlusion_frames", 3))
    minimum_rescue = int(validation.get("minimum_occlusion_rescue_frames", 2))
    failures: list[str] = []
    if physical_frames < minimum_physical:
        failures.append(f"physical occlusion appears in only {physical_frames} frames (<{minimum_physical})")
    if system_rescue_frames < minimum_rescue:
        failures.append(
            f"occlusion creates a system-level UAV rescue in only {system_rescue_frames} frames (<{minimum_rescue})"
        )
    payload = {
        "passed": not failures,
        "scope": "geometry_plus_actual_CARLA_raycast_support",
        "blocker_role": blocker_role,
        "target_roles": sorted(target_roles),
        "physical_occlusion_frame_count": physical_frames,
        "uav_visibility_frame_count": uav_visibility_frames,
        "system_rescue_frame_count": system_rescue_frames,
        "failures": failures,
        "frames": frame_records,
    }
    json_path = run_dir / "occlusion_report.json"
    text_path = run_dir / "occlusion_report.txt"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    text_path.write_text(_render_text(payload), encoding="utf-8")
    print(text_path.read_text(encoding="utf-8"))
    return json_path, text_path


def _blocker_between(
    observer_xy: np.ndarray,
    blocker_xy: np.ndarray,
    target_xy: np.ndarray,
    lateral_tolerance_m: float = 3.0,
) -> dict[str, float | bool]:
    ray = np.asarray(target_xy, dtype=np.float64) - np.asarray(observer_xy, dtype=np.float64)
    norm2 = float(np.dot(ray, ray))
    if norm2 < 1e-9:
        return {"between": False, "projection": 0.0, "lateral_error_m": float("inf")}
    relative = np.asarray(blocker_xy, dtype=np.float64) - np.asarray(observer_xy, dtype=np.float64)
    projection = float(np.dot(relative, ray) / norm2)
    closest = np.asarray(observer_xy, dtype=np.float64) + projection * ray
    lateral = float(np.linalg.norm(np.asarray(blocker_xy, dtype=np.float64) - closest))
    between = 0.05 < projection < 0.95 and lateral <= float(lateral_tolerance_m)
    return {"between": between, "projection": projection, "lateral_error_m": lateral}


def _render_text(payload: dict[str, Any]) -> str:
    lines = [
        "Active AirV2X moving-occlusion check",
        "======================================",
        f"Passed: {payload['passed']}",
        f"Physical blocker frames: {payload['physical_occlusion_frame_count']}",
        f"UAV-visible occluded-target frames: {payload['uav_visibility_frame_count']}",
        f"System-level rescue frames: {payload['system_rescue_frame_count']}",
    ]
    if payload["failures"]:
        lines.append("Failures:")
        lines.extend(f"  - {failure}" for failure in payload["failures"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate real moving-vehicle occlusion")
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    build_occlusion_report(args.run_dir)


if __name__ == "__main__":
    main()
