from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


ROLE_COLORS = {
    "regional_ego": "#f59e0b",
    "regional_car1": "#22c55e",
    "regional_car3": "#3b82f6",
    "regional_cav2_blocker": "#ef4444",
    "regional_cav2_front_1": "#ec4899",
    "regional_cav2_front_2": "#db2777",
    "regional_cav2_rear": "#be185d",
    "regional_corridor_1": "#a855f7",
    "regional_corridor_2": "#9333ea",
    "regional_corridor_3": "#7e22ce",
    "regional_east_front": "#ec4899",
    "regional_east_rear": "#db2777",
    "regional_south_1": "#14b8a6",
    "regional_south_2": "#0d9488",
    "regional_south_3": "#0f766e",
    "regional_late_1": "#a855f7",
    "regional_late_2": "#9333ea",
    "regional_late_3": "#7e22ce",
    "regional_late_4": "#6b21a8",
    "regional_late_5": "#581c87",
    "regional_background": "#94a3b8",
}
ROLE_LABELS = {
    "regional_ego": "CAV-1 / Ego",
    "regional_car1": "CAV-2 / J1",
    "regional_car3": "CAV-2 / J2 transient",
    "regional_cav2_blocker": "Moving bus / blocker",
    "regional_cav2_front_1": "Occluded target A",
    "regional_cav2_front_2": "Occluded target B",
    "regional_cav2_rear": "J2 traffic rear",
    "regional_corridor_1": "Corridor-1",
    "regional_corridor_2": "Corridor-2",
    "regional_corridor_3": "Corridor-3",
    "regional_east_front": "East-A",
    "regional_east_rear": "East-B",
    "regional_south_1": "South-A",
    "regional_south_2": "South-B",
    "regional_south_3": "South-C",
    "regional_late_1": "Late-1",
    "regional_late_2": "Late-2",
    "regional_late_3": "Late-3",
    "regional_late_4": "Late-4",
    "regional_late_5": "Late-5",
    "regional_background": "NPC",
}
SENSING_AGENT_ROLES = ("regional_ego", "regional_car1", "regional_car3")


def scene_bounds(corridor: Any, margin_m: float) -> tuple[float, float, float, float]:
    first = np.asarray(corridor.first.center[:2], dtype=np.float64)
    second = np.asarray(corridor.second.center[:2], dtype=np.float64)
    minimum = np.minimum(first, second) - float(margin_m)
    maximum = np.maximum(first, second) + float(margin_m)
    return float(minimum[0]), float(maximum[0]), float(minimum[1]), float(maximum[1])


def render_timeline(
    carla_map: Any,
    corridor: Any,
    grid: list[dict[str, Any]],
    rsu_xyz: list[float],
    routes: dict[str, dict[str, Any]],
    snapshots: list[dict[str, Any]],
    output: str | Path,
    margin_m: float,
    sensor_ranges: dict[str, float] | None = None,
    evaluation_roi: dict[str, Any] | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    output = Path(output)
    bounds = timeline_bounds(
        corridor,
        grid,
        rsu_xyz,
        routes,
        snapshots,
        margin_m,
        sensor_ranges=sensor_ranges,
        evaluation_roi=evaluation_roi,
    )
    columns = 2
    rows = int(math.ceil(len(snapshots) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(15, 7.5 * rows), constrained_layout=True)
    axes_array = np.asarray(axes, dtype=object).reshape(-1)
    for axis, snapshot in zip(axes_array, snapshots):
        render_snapshot(
            axis,
            carla_map,
            corridor,
            grid,
            rsu_xyz,
            routes,
            snapshot["actors"],
            float(snapshot["elapsed_s"]),
            margin_m,
            bounds=bounds,
            sensor_ranges=sensor_ranges,
            evaluation_roi=evaluation_roi,
        )
    for axis in axes_array[len(snapshots) :]:
        axis.set_visible(False)
    figure.suptitle(
        "Regional Active UAV Pilot — Global BEV (fixed world frame)",
        fontsize=18,
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="#f8fafc")
    plt.close(figure)
    return output


def render_snapshot(
    axis: Any,
    carla_map: Any,
    corridor: Any,
    grid: list[dict[str, Any]],
    rsu_xyz: list[float],
    routes: dict[str, dict[str, Any]],
    actors: list[dict[str, Any]],
    elapsed_s: float,
    margin_m: float,
    bounds: tuple[float, float, float, float] | None = None,
    sensor_ranges: dict[str, float] | None = None,
    evaluation_roi: dict[str, Any] | None = None,
) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon, Rectangle

    xmin, xmax, ymin, ymax = bounds or scene_bounds(corridor, margin_m)
    axis.set_facecolor("#e5e7eb")
    _draw_road_map(axis, carla_map, (xmin, xmax, ymin, ymax))

    if evaluation_roi is not None:
        roi_corners = np.asarray(evaluation_roi["corners_world"], dtype=np.float64)
        axis.add_patch(
            Polygon(
                roi_corners[:, :2],
                closed=True,
                facecolor="#facc15",
                edgecolor="#ca8a04",
                linewidth=2.2,
                linestyle="-.",
                alpha=0.08,
                zorder=2,
            )
        )
        center = roi_corners[:, :2].mean(axis=0)
        axis.text(
            center[0],
            center[1],
            "Fixed regional evaluation ROI",
            ha="center",
            va="center",
            color="#854d0e",
            fontsize=9,
            fontweight="bold",
            zorder=4,
        )

    for index, info in enumerate((corridor.first, corridor.second), start=1):
        cx, cy, _ = info.center
        ex, ey, _ = info.extent
        axis.add_patch(
            Rectangle(
                (cx - ex, cy - ey),
                2.0 * ex,
                2.0 * ey,
                edgecolor="#7c3aed" if index == 2 else "#475569",
                facecolor="none",
                linewidth=2.2,
                linestyle="--",
                zorder=4,
            )
        )
        axis.text(
            cx,
            cy + ey + 3.0,
            f"J{index}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="#312e81",
            zorder=8,
        )

    for role, route in routes.items():
        xyz = np.asarray(route["xyz"], dtype=np.float64)
        if len(xyz) < 2:
            continue
        axis.plot(
            xyz[:, 0],
            xyz[:, 1],
            color=route["color"],
            linewidth=2.0,
            alpha=0.72,
            zorder=5,
        )
        axis.annotate(
            "",
            xy=(xyz[-1, 0], xyz[-1, 1]),
            xytext=(xyz[-2, 0], xyz[-2, 1]),
            arrowprops={"arrowstyle": "->", "color": route["color"], "lw": 2.0},
            zorder=6,
        )

    grid_xy = np.asarray([item["location"][:2] for item in grid], dtype=np.float64)
    axis.scatter(
        grid_xy[:, 0],
        grid_xy[:, 1],
        s=34,
        marker="o",
        facecolors="none",
        edgecolors="#7c3aed",
        linewidths=1.1,
        alpha=0.85,
        zorder=7,
    )
    grid_size = int(round(math.sqrt(len(grid))))
    if grid_size * grid_size != len(grid):
        raise ValueError("UAV grid must contain an odd square number of candidates")
    outline = np.asarray(
        [
            grid[0]["location"][:2],
            grid[grid_size - 1]["location"][:2],
            grid[-1]["location"][:2],
            grid[-grid_size]["location"][:2],
        ]
    )
    axis.add_patch(
        Polygon(
            outline,
            closed=True,
            fill=False,
            edgecolor="#7c3aed",
            linewidth=1.8,
            linestyle=":",
            zorder=6,
        )
    )
    center_item = grid[len(grid) // 2]
    axis.scatter(
        [center_item["location"][0]],
        [center_item["location"][1]],
        s=120,
        marker="X",
        color="#7c3aed",
        edgecolors="white",
        linewidths=1.0,
        zorder=10,
    )
    axis.text(
        center_item["location"][0],
        center_item["location"][1] - 4.0,
        "UAV initial / grid center",
        ha="center",
        va="top",
        color="#5b21b6",
        fontsize=9,
        fontweight="bold",
        zorder=10,
    )

    if sensor_ranges:
        _draw_sensor_ranges_map(axis, actors, rsu_xyz, center_item["location"], sensor_ranges)

    axis.scatter(
        [rsu_xyz[0]],
        [rsu_xyz[1]],
        s=140,
        marker="P",
        color="#dc2626",
        edgecolors="white",
        linewidths=1.0,
        zorder=10,
    )
    axis.text(
        rsu_xyz[0],
        rsu_xyz[1] + 4.0,
        "RSU (360°)",
        ha="center",
        color="#991b1b",
        fontsize=10,
        fontweight="bold",
        zorder=10,
    )

    for actor in actors:
        role = actor["role_name"]
        x, y, _ = actor["location"]
        _, _, yaw = actor["rotation"]
        length = 2.0 * actor["bbox_extent"][0]
        width = 2.0 * actor["bbox_extent"][1]
        corners = _oriented_rectangle(x, y, max(length, 2.5), max(width, 1.2), yaw)
        color = ROLE_COLORS.get(role, "#94a3b8")
        axis.add_patch(
            Polygon(
                corners,
                closed=True,
                facecolor=color,
                edgecolor="#0f172a",
                linewidth=0.7,
                alpha=0.95 if role != "regional_background" else 0.7,
                zorder=9 if role != "regional_background" else 8,
            )
        )
        if role in ROLE_LABELS and role != "regional_background":
            axis.text(
                x,
                y + 2.8,
                ROLE_LABELS[role],
                ha="center",
                fontsize=9,
                color="#111827",
                fontweight="bold",
                zorder=11,
            )

    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymin, ymax)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(f"t = {elapsed_s:.1f} s", fontsize=14, fontweight="bold")
    axis.set_xlabel("CARLA world x (m)")
    axis.set_ylabel("CARLA world y (m)")
    axis.grid(color="white", linewidth=0.5, alpha=0.35)
    axis.annotate(
        "N (+y)",
        xy=(xmin + 8.0, ymax - 7.0),
        xytext=(xmin + 8.0, ymax - 22.0),
        ha="center",
        arrowprops={"arrowstyle": "-|>", "color": "#111827", "lw": 1.8},
        fontsize=9,
        fontweight="bold",
        zorder=12,
    )
    handles = [
        Line2D([0], [0], color=ROLE_COLORS["regional_ego"], lw=5, label="CAV-1 / Ego"),
        Line2D([0], [0], color=ROLE_COLORS["regional_car3"], lw=5, label="CAV-2 / J2"),
        Line2D([0], [0], color=ROLE_COLORS["regional_cav2_blocker"], lw=5,
               label="moving blocker"),
        Line2D([0], [0], marker="P", color="w", markerfacecolor="#dc2626", markersize=10, label="RSU"),
        Line2D([0], [0], marker="o", color="#7c3aed", markerfacecolor="none", linestyle="None", label="UAV candidates"),
    ]
    if sensor_ranges:
        handles.extend(
            [
                Line2D([0], [0], color="#2563eb", ls="--", lw=1.5,
                       label=f"CAV LiDAR R={sensor_ranges['vehicle_m']:.0f}m"),
                Line2D([0], [0], color="#dc2626", ls="--", lw=1.5,
                       label=f"RSU LiDAR R={sensor_ranges['rsu_m']:.0f}m"),
                Line2D([0], [0], color="#7c3aed", ls="--", lw=1.5,
                       label=f"Drone-center ground R≈{sensor_ranges['uav_ground_m']:.0f}m"),
            ]
        )
    axis.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.9)


def compose_rgb_timeline(image_paths: list[Path], times_s: list[float], output: str | Path) -> Path:
    import matplotlib.pyplot as plt

    output = Path(output)
    columns = 2
    rows = int(math.ceil(len(image_paths) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(16, 9 * rows), constrained_layout=True)
    axes_array = np.asarray(axes, dtype=object).reshape(-1)
    for axis, path, elapsed in zip(axes_array, image_paths, times_s):
        axis.imshow(plt.imread(path))
        axis.set_title(f"CARLA overhead RGB — t={elapsed:.1f}s", fontsize=14, fontweight="bold")
        axis.axis("off")
    for axis in axes_array[len(image_paths) :]:
        axis.set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, facecolor="white")
    plt.close(figure)
    return output


def timeline_bounds(
    corridor: Any,
    grid: list[dict[str, Any]],
    rsu_xyz: list[float],
    routes: dict[str, dict[str, Any]],
    snapshots: list[dict[str, Any]],
    margin_m: float,
    sensor_ranges: dict[str, float] | None = None,
    evaluation_roi: dict[str, Any] | None = None,
) -> tuple[float, float, float, float]:
    """Bounds that keep both junctions and every tracked CAV route visible."""
    points: list[list[float]] = [
        list(corridor.first.center[:2]),
        list(corridor.second.center[:2]),
        list(rsu_xyz[:2]),
    ]
    points.extend([list(item["location"][:2]) for item in grid])
    if evaluation_roi is not None:
        points.extend(
            [list(point[:2]) for point in evaluation_roi.get("corners_world", [])]
        )
    for route in routes.values():
        points.extend([list(value[:2]) for value in route["xyz"]])
    for snapshot in snapshots:
        for actor in snapshot["actors"]:
            if actor.get("role_name") in ROLE_LABELS and actor.get("role_name") != "regional_background":
                points.append(list(actor["location"][:2]))
    xy = np.asarray(points, dtype=np.float64)
    minimum = xy.min(axis=0) - float(margin_m)
    maximum = xy.max(axis=0) + float(margin_m)
    if sensor_ranges:
        range_boxes = [
            (np.asarray(rsu_xyz[:2], dtype=np.float64), float(sensor_ranges["rsu_m"])),
            (
                np.asarray(grid[len(grid) // 2]["location"][:2], dtype=np.float64),
                float(sensor_ranges["uav_ground_m"]),
            ),
        ]
        for snapshot in snapshots:
            for actor in snapshot["actors"]:
                if actor.get("role_name") in SENSING_AGENT_ROLES:
                    range_boxes.append(
                        (
                            np.asarray(actor["location"][:2], dtype=np.float64),
                            float(sensor_ranges["vehicle_m"]),
                        )
                    )
        for center, radius in range_boxes:
            minimum = np.minimum(minimum, center - radius - 5.0)
            maximum = np.maximum(maximum, center + radius + 5.0)
    return float(minimum[0]), float(maximum[0]), float(minimum[1]), float(maximum[1])


def annotate_rgb_snapshot(
    image_path: str | Path,
    output: str | Path,
    camera: Any,
    actors: list[dict[str, Any]],
    rsu_xyz: list[float],
    grid: list[dict[str, Any]],
    junction_centers: list[list[float]],
    elapsed_s: float,
    sensor_ranges: dict[str, float] | None = None,
    evaluation_roi: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, bool]]:
    """Overlay world-coordinate labels on a real CARLA RGB camera frame."""
    import matplotlib.pyplot as plt

    image_path = Path(image_path)
    output = Path(output)
    image = plt.imread(image_path)
    height, width = image.shape[:2]
    fov = float(camera.attributes["fov"])
    world_to_camera = np.asarray(camera.get_transform().get_inverse_matrix(), dtype=np.float64)

    tracked = [actor for actor in actors if actor.get("role_name") in ROLE_LABELS]
    actor_points = np.asarray([actor["location"] for actor in tracked], dtype=np.float64)
    actor_uv, actor_valid = project_world_to_camera(actor_points, world_to_camera, width, height, fov)

    grid_points = np.asarray([item["location"] for item in grid], dtype=np.float64)
    center_index = len(grid) // 2
    grid_uv, grid_valid = project_world_to_camera(grid_points, world_to_camera, width, height, fov)
    rsu_uv, rsu_valid = project_world_to_camera(
        np.asarray([rsu_xyz], dtype=np.float64), world_to_camera, width, height, fov
    )
    junction_uv, junction_valid = project_world_to_camera(
        np.asarray(junction_centers, dtype=np.float64), world_to_camera, width, height, fov
    )
    roi_uv = None
    roi_valid = None
    if evaluation_roi is not None:
        roi_uv, roi_valid = project_world_to_camera(
            np.asarray(evaluation_roi["corners_world"], dtype=np.float64),
            world_to_camera,
            width,
            height,
            fov,
        )

    dpi = 120
    figure = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axis.imshow(image)
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    axis.axis("off")

    if roi_uv is not None and roi_valid is not None and bool(np.all(roi_valid)):
        closed = np.vstack([roi_uv, roi_uv[0]])
        axis.plot(
            closed[:, 0],
            closed[:, 1],
            color="#facc15",
            linewidth=3.0,
            linestyle="-.",
            alpha=0.95,
            zorder=9,
        )

    if sensor_ranges:
        _draw_sensor_ranges_rgb(
            axis,
            tracked,
            rsu_xyz,
            grid[center_index]["location"],
            junction_centers,
            sensor_ranges,
            world_to_camera,
            width,
            height,
            fov,
        )

    if np.any(grid_valid):
        visible = grid_uv[grid_valid]
        axis.scatter(
            visible[:, 0], visible[:, 1], s=24, facecolors="none", edgecolors="#a855f7",
            linewidths=1.2, alpha=0.9, zorder=10,
        )
    if bool(grid_valid[center_index]):
        u, v = grid_uv[center_index]
        _rgb_marker(axis, u, v, "Drone init", "#7c3aed", marker="X", offset=(12, 18))

    if bool(rsu_valid[0]):
        u, v = rsu_uv[0]
        _rgb_marker(axis, u, v, "RSU", "#dc2626", marker="P", offset=(12, -18))

    for index, valid in enumerate(junction_valid):
        if valid:
            u, v = junction_uv[index]
            _rgb_marker(axis, u, v, f"J{index + 1}", "#06b6d4", marker="+", offset=(10, 16))

    visible_roles: dict[str, bool] = {
        ROLE_LABELS[actor["role_name"]]: False
        for actor in tracked
        if actor["role_name"] != "regional_background"
    }
    for actor, uv, valid in zip(tracked, actor_uv, actor_valid):
        role = actor["role_name"]
        label = ROLE_LABELS[role]
        if role == "regional_background" or not valid:
            continue
        visible_roles[label] = True
        u, v = uv
        _rgb_marker(axis, u, v, label, ROLE_COLORS[role], marker="o", offset=(12, -18))

    axis.text(
        18, 30, f"CARLA RGB  t={elapsed_s:.1f}s", color="white", fontsize=12,
        fontweight="bold", ha="left", va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#0f172a", "alpha": 0.82, "edgecolor": "none"},
        zorder=20,
    )
    missing = [name for name, visible in visible_roles.items() if not visible]
    if missing:
        axis.text(
            18, height - 22, "Outside camera: " + ", ".join(missing), color="white", fontsize=10,
            ha="left", va="bottom",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#991b1b", "alpha": 0.85, "edgecolor": "none"},
            zorder=20,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, facecolor="white")
    plt.close(figure)
    return output, visible_roles


def project_world_to_camera(
    points_xyz: np.ndarray,
    world_to_camera: np.ndarray,
    image_width: int,
    image_height: int,
    horizontal_fov_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project CARLA world XYZ into pixels using the documented UE4 axis conversion."""
    points = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=np.float64)], axis=1)
    ue_camera = (np.asarray(world_to_camera, dtype=np.float64) @ homogeneous.T).T[:, :3]
    camera_xyz = np.column_stack([ue_camera[:, 1], -ue_camera[:, 2], ue_camera[:, 0]])
    depth = camera_xyz[:, 2]
    focal = float(image_width) / (2.0 * math.tan(math.radians(float(horizontal_fov_deg)) / 2.0))
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    front = depth > 1e-4
    uv[front, 0] = focal * camera_xyz[front, 0] / depth[front] + float(image_width) / 2.0
    uv[front, 1] = focal * camera_xyz[front, 1] / depth[front] + float(image_height) / 2.0
    valid = (
        front
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < float(image_width))
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < float(image_height))
    )
    return uv, valid


def _draw_sensor_ranges_map(
    axis: Any,
    actors: list[dict[str, Any]],
    rsu_xyz: list[float],
    drone_xyz: list[float],
    ranges: dict[str, float],
) -> None:
    from matplotlib.patches import Circle

    for actor in actors:
        role = actor.get("role_name")
        if role not in SENSING_AGENT_ROLES:
            continue
        x, y, _ = actor["location"]
        color = ROLE_COLORS[role]
        axis.add_patch(
            Circle((x, y), ranges["vehicle_m"], fill=False, edgecolor=color,
                   linewidth=1.1, linestyle="--", alpha=0.45, zorder=3)
        )
    axis.add_patch(
        Circle((rsu_xyz[0], rsu_xyz[1]), ranges["rsu_m"], fill=False,
               edgecolor="#dc2626", linewidth=1.2, linestyle="--", alpha=0.5, zorder=3)
    )
    axis.add_patch(
        Circle((drone_xyz[0], drone_xyz[1]), ranges["uav_ground_m"], fill=False,
               edgecolor="#7c3aed", linewidth=1.4, linestyle="--", alpha=0.65, zorder=3)
    )


def _draw_sensor_ranges_rgb(
    axis: Any,
    actors: list[dict[str, Any]],
    rsu_xyz: list[float],
    drone_xyz: list[float],
    junction_centers: list[list[float]],
    ranges: dict[str, float],
    world_to_camera: np.ndarray,
    width: int,
    height: int,
    fov: float,
) -> None:
    for actor in actors:
        role = actor.get("role_name")
        if role not in SENSING_AGENT_ROLES:
            continue
        x, y, z = actor["location"]
        _draw_projected_circle(
            axis, [x, y, z], ranges["vehicle_m"], ROLE_COLORS[role],
            world_to_camera, width, height, fov, alpha=0.48,
        )
    _draw_projected_circle(
        axis,
        [rsu_xyz[0], rsu_xyz[1], junction_centers[0][2]],
        ranges["rsu_m"],
        "#dc2626",
        world_to_camera,
        width,
        height,
        fov,
        alpha=0.5,
    )
    _draw_projected_circle(
        axis,
        [drone_xyz[0], drone_xyz[1], junction_centers[1][2]],
        ranges["uav_ground_m"],
        "#7c3aed",
        world_to_camera,
        width,
        height,
        fov,
        alpha=0.7,
    )


def _draw_projected_circle(
    axis: Any,
    center_xyz: list[float],
    radius_m: float,
    color: str,
    world_to_camera: np.ndarray,
    width: int,
    height: int,
    fov: float,
    alpha: float,
) -> None:
    angles = np.linspace(0.0, 2.0 * math.pi, 181)
    points = np.column_stack(
        [
            center_xyz[0] + radius_m * np.cos(angles),
            center_xyz[1] + radius_m * np.sin(angles),
            np.full_like(angles, center_xyz[2]),
        ]
    )
    uv, valid = project_world_to_camera(points, world_to_camera, width, height, fov)
    plotted = uv.copy()
    plotted[~valid] = np.nan
    axis.plot(plotted[:, 0], plotted[:, 1], color=color, linewidth=2.0,
              linestyle="--", alpha=alpha, zorder=8)


def _rgb_marker(
    axis: Any,
    u: float,
    v: float,
    label: str,
    color: str,
    marker: str,
    offset: tuple[float, float],
) -> None:
    marker_style = {
        "s": 150,
        "marker": marker,
        "color": color,
        "linewidths": 2.2 if marker == "+" else 1.8,
        "zorder": 15,
    }
    if marker != "+":
        marker_style["edgecolors"] = "white"
    axis.scatter([u], [v], **marker_style)
    axis.annotate(
        label, xy=(u, v), xytext=offset, textcoords="offset points", color="white",
        fontsize=11, fontweight="bold", ha="left", va="center",
        arrowprops={"arrowstyle": "->", "color": color, "lw": 2.0},
        bbox={"boxstyle": "round,pad=0.25", "facecolor": color, "alpha": 0.9, "edgecolor": "white"},
        zorder=16,
    )


def _draw_road_map(axis: Any, carla_map: Any, bounds: tuple[float, float, float, float]) -> None:
    xmin, xmax, ymin, ymax = bounds
    for waypoint in carla_map.generate_waypoints(2.0):
        location = waypoint.transform.location
        if not (xmin - 3.0 <= location.x <= xmax + 3.0 and ymin - 3.0 <= location.y <= ymax + 3.0):
            continue
        options = [
            item
            for item in waypoint.next(2.0)
            if int(item.road_id) == int(waypoint.road_id) or item.is_junction
        ]
        if not options:
            continue
        next_wp = min(
            options,
            key=lambda item: abs(
                (float(item.transform.rotation.yaw) - float(waypoint.transform.rotation.yaw) + 180.0)
                % 360.0
                - 180.0
            ),
        )
        next_location = next_wp.transform.location
        axis.plot(
            [location.x, next_location.x],
            [location.y, next_location.y],
            color="#64748b",
            linewidth=4.5,
            solid_capstyle="round",
            zorder=1,
        )
        axis.plot(
            [location.x, next_location.x],
            [location.y, next_location.y],
            color="#e2e8f0",
            linewidth=0.45,
            alpha=0.75,
            zorder=2,
        )


def _oriented_rectangle(
    center_x: float, center_y: float, length: float, width: float, yaw_deg: float
) -> np.ndarray:
    local = np.asarray(
        [
            [length * 0.5, width * 0.5],
            [length * 0.5, -width * 0.5],
            [-length * 0.5, -width * 0.5],
            [-length * 0.5, width * 0.5],
        ],
        dtype=np.float64,
    )
    yaw = math.radians(float(yaw_deg))
    rotation = np.asarray([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    return local @ rotation.T + np.asarray([center_x, center_y])
