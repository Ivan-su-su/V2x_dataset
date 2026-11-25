from __future__ import annotations

from typing import Any

import numpy as np

from .geometry import make_grid, make_regional_roi


def resolve_regional_corridor(carla_map: Any, regional: dict[str, Any]) -> Any:
    """Resolve the legacy straight corridor or anchors for an explicit lane scene."""
    from .corridors import ConnectedJunctions, select_connected_junctions
    from .junctions import rank_junctions
    from .lane_traffic import resolve_lanes

    first_id, second_id = regional.get("junction_1_id"), regional.get("junction_2_id")
    plan = regional.get("lane_traffic") or {}
    if "layout" not in plan:
        return select_connected_junctions(
            carla_map, first_id, second_id, int(regional.get("corridor_rank", 0)),
            float(regional["min_junction_separation_m"]), float(regional["max_junction_separation_m"]),
        )
    infos = {j.junction_id: j for j in rank_junctions(carla_map)}
    missing = set(plan["junction_ids"]) - infos.keys()
    if missing:
        raise ValueError(f"explicit lane scene: map lacks junction IDs {sorted(missing)}")
    route = resolve_lanes(carla_map, plan["vehicles"]["regional_ego"]["lanes"])
    first, second = infos[first_id], infos[second_id]
    # These references label the BEV. The directed traffic plan defines connectivity.
    distance = float(np.linalg.norm(np.asarray(first.center[:2]) - second.center[:2]))
    return ConnectedJunctions(first, second, tuple(route), distance, (1.0, 0.0), (0.0, 1.0), 0.0)


def build_regional_layout(corridor: Any, cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve RSU, active-UAV grid and the fixed evaluation ROI once."""
    regional = cfg["regional"]
    first_center = np.asarray(corridor.first.center, dtype=np.float64)
    second_center = np.asarray(corridor.second.center, dtype=np.float64)
    corridor_center = np.asarray(corridor.center, dtype=np.float64)
    forward = np.asarray(corridor.forward_xy, dtype=np.float64)
    right = np.asarray(corridor.right_xy, dtype=np.float64)

    rsu_cfg = regional["rsu"]
    rsu_xy = (
        first_center[:2]
        + float(rsu_cfg["offset_forward_m"]) * forward
        + float(rsu_cfg["offset_right_m"]) * right
    )
    rsu_xyz = [
        float(rsu_xy[0]),
        float(rsu_xy[1]),
        float(first_center[2] + float(rsu_cfg["height_m"])),
    ]

    grid_center = second_center.copy()
    grid_center[:2] += (
        float(regional.get("uav_grid_offset_forward_m", 0.0)) * forward
        + float(regional.get("uav_grid_offset_right_m", 0.0)) * right
    )
    explicit = (regional.get("lane_traffic") or {}).get("layout")
    if explicit is not None:
        rsu_xyz = list(explicit["rsu_world_xyz"])
        grid_center = np.asarray(explicit["grid_center_world_xyz"], dtype=np.float64)
    grid = make_grid(
        grid_center,
        forward,
        right,
        int(cfg["grid"]["size"]),
        float(cfg["grid"]["spacing_m"]),
        float(cfg["grid"]["height_m"]),
    )

    evaluation_roi = None
    if str(cfg["scoring"].get("roi_mode", "ego_moving")) == "regional_fixed":
        roi_cfg = cfg["scoring"]["regional_roi"]
        roi_center = corridor_center.copy()
        roi_center[:2] += (
            float(roi_cfg.get("center_offset_forward_m", 0.0)) * forward
            + float(roi_cfg.get("center_offset_right_m", 0.0)) * right
        )
        evaluation_roi = make_regional_roi(roi_center, forward, right, roi_cfg)

    return {
        "rsu_xyz": rsu_xyz,
        "grid_center_xyz": grid_center.tolist(),
        "grid": grid,
        "evaluation_roi": evaluation_roi,
    }
