import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from active_view_v0.config import load_config
from active_view_v0.regional_collector import _collection_schedule
from active_view_v0.regional_preview import _build_motion_summary
from active_view_v0.regional_scenario import (
    RegionalIntersectionScenario,
    _heading_is_opposite,
    _route_suffix_after_distance,
)
from active_view_v0.uav_modes import (
    UAVModePlanner,
    aggregate_scores_by_window,
    build_decision_windows,
    expand_window_path,
)
from active_view_v0.where2comm_eval import _load_uav_mode_record


class _Location:
    x = 0.0
    y = 0.0


class _Rotation:
    yaw = 0.0


class _Transform:
    location = _Location()
    rotation = _Rotation()


class _RouteLocation:
    def __init__(self, x: float):
        self.x = float(x)
        self.y = 0.0
        self.z = 0.0

    def distance(self, other) -> float:
        return abs(self.x - other.x)


class _RouteWaypoint:
    def __init__(self, x: float):
        self.transform = SimpleNamespace(location=_RouteLocation(x))

    def next(self, distance: float):
        return [_RouteWaypoint(self.transform.location.x + float(distance))]


def _grid() -> list[dict]:
    output = []
    for row in range(5):
        for col in range(5):
            output.append(
                {
                    "name": f"uav_r{row}_c{col}",
                    "row": row,
                    "col": col,
                    "offset_forward_m": float((row - 2) * 15),
                    "offset_right_m": float((col - 2) * 15),
                    "location": [float((row - 2) * 15), float((col - 2) * 15), 40.0],
                }
            )
    return output


def test_dense_config_produces_exactly_100_frames() -> None:
    path = Path(__file__).parents[1] / "configs" / "dense_dynamic_town03_10s.yaml"
    cfg = load_config(path)
    loop_ticks, stride, expected = _collection_schedule(cfg)
    assert (loop_ticks, stride, expected) == (100, 1, 100)
    assert cfg["regional"]["collection_interval_s"] == 0.1
    assert cfg["regional"]["uav_modes"]["decision_interval_s"] == 1.0
    assert cfg["scoring"]["base_sensors"] == ["ego_lidar", "cav2_lidar", "rsu_lidar"]
    support = cfg["regional"]["support_vehicles"]
    assert len(support) == 14
    assert all("route" in item and "speed_difference_pct" in item for item in support)
    assert cfg["regional"]["background_vehicle_count"] == 14


def test_dense_20s_config_adds_deterministic_oncoming_flow() -> None:
    path = Path(__file__).parents[1] / "configs" / "dense_dynamic_town03_20s.yaml"
    cfg = load_config(path)
    assert _collection_schedule(cfg) == (200, 1, 200)
    assert cfg["regional"]["ego_speed_difference_pct"] == 0.0
    assert cfg["regional"]["ego_start_advance_m"] == 20.0
    support = cfg["regional"]["support_vehicles"]
    assert sum(item["route"] == "corridor_oncoming" for item in support) == 5
    assert all(
        item.get("required", False)
        for item in support
        if item["route"] == "corridor_oncoming"
    )
    assert cfg["regional"]["density_validation"]["minimum_ego_displacement_m"] == 40.0


def test_opposite_heading_gate() -> None:
    assert _heading_is_opposite(0.0, 180.0)
    assert _heading_is_opposite(170.0, -10.0)
    assert not _heading_is_opposite(0.0, 90.0)


def test_route_offsets_do_not_collapse_to_sparse_anchor() -> None:
    route = [_RouteWaypoint(0.0), _RouteWaypoint(100.0), _RouteWaypoint(150.0)]
    first = _route_suffix_after_distance(route, 36.0)
    second = _route_suffix_after_distance(route, 57.0)
    assert first[0].transform.location.x == 36.0
    assert second[0].transform.location.x == 57.0
    assert second[0].transform.location.distance(first[0].transform.location) == 21.0


def test_disabled_j1_cav_does_not_read_missing_car1_distance(monkeypatch) -> None:
    import active_view_v0.regional_scenario as scenario_module

    scenario = RegionalIntersectionScenario.__new__(RegionalIntersectionScenario)
    scenario.cfg = {
        "regional": {
            "enable_j1_cav": False,
            "route_exit_distance_m": 120.0,
            "car3_start_before_junction_m": 82.0,
            "background_vehicle_count": 0,
        }
    }
    scenario.corridor = SimpleNamespace(
        corridor_waypoints=(object(), object()),
        first=SimpleNamespace(junction="j1"),
        second=SimpleNamespace(junction="j2"),
    )
    spawned = []
    scenario._destroy_stale_owned_actors = lambda: None
    scenario._spawn_cav = lambda role, route, speed_difference: spawned.append(role)
    scenario._spawn_support_traffic = lambda route, yaw: None
    scenario._spawn_background = lambda count: None
    monkeypatch.setattr(scenario_module, "_route_yaw", lambda route: 0.0)
    monkeypatch.setattr(
        scenario_module, "extend_route_straight", lambda route, distance: list(route)
    )
    monkeypatch.setattr(
        scenario_module,
        "cross_route",
        lambda junction, yaw, start, end: [junction, object()],
    )

    scenario.setup()

    assert spawned == ["regional_ego", "regional_car3"]


def test_ten_hz_frames_form_ten_one_hz_decision_windows() -> None:
    elapsed = np.arange(100, dtype=np.float64) * 0.1
    windows = build_decision_windows(elapsed, 1.0)
    assert len(windows.window_start_indices) == 10
    np.testing.assert_array_equal(windows.window_start_indices, np.arange(0, 100, 10))
    scores = np.column_stack([np.arange(100), np.arange(100) + 10]).astype(np.float64)
    aggregated = aggregate_scores_by_window(scores, windows)
    assert aggregated.shape == (10, 2)
    np.testing.assert_allclose(aggregated[0], [4.5, 14.5])
    expanded = expand_window_path(np.arange(10) % 2, windows)
    np.testing.assert_array_equal(expanded[:20], [0] * 10 + [1] * 10)


def test_uav_modes_hold_between_decisions_and_obey_speed_limit() -> None:
    planner = UAVModePlanner(
        _grid(),
        {
            "decision_interval_s": 1.0,
            "initial_candidate": "uav_r2_c2",
            "tracking_forward_offset_m": 25.0,
            "tracking_right_offset_m": 0.0,
            "patrol_candidates": [
                "uav_r2_c2",
                "uav_r2_c3",
                "uav_r2_c4",
                "uav_r1_c4",
            ],
        },
        frame_interval_s=0.1,
        max_speed_mps=15.0,
    )
    records = [planner.step(index, index * 0.1, _Transform()) for index in range(100)]
    assert sum(record["is_decision_frame"] for record in records) == 10
    for mode in UAVModePlanner.MODE_NAMES:
        names = [record["modes"][mode]["candidate_name"] for record in records]
        for start in range(0, 100, 10):
            assert len(set(names[start : start + 10])) == 1
        decision_xy = np.asarray(
            [records[index]["modes"][mode]["location_world"][:2] for index in range(0, 100, 10)]
        )
        assert np.all(np.linalg.norm(np.diff(decision_xy, axis=0), axis=1) <= 15.0 + 1e-6)
        dense_xyz = np.asarray(
            [record["modes"][mode]["location_world"] for record in records]
        )
        assert np.all(
            np.linalg.norm(np.diff(dense_xyz, axis=0), axis=1) <= 1.5 + 1e-6
        )
        assert all(
            record["modes"][mode]["source_sensor"] == f"uav_mode_{mode}_lidar"
            for record in records
        )
    summary = planner.summary()
    assert len(summary["paths"]["hover"]["candidate_names_per_frame"]) == 100
    assert summary["paths"]["tracking"]["sensor_name"] == "uav_mode_tracking_lidar"


def test_where2comm_reads_continuous_mode_trajectory(tmp_path: Path) -> None:
    locations = [[float(index), 0.0, 40.0] for index in range(6)]
    payload = {
        "paths": {
            "tracking": {
                "candidate_names_per_frame": ["uav_r2_c2"] * 6,
                "location_world_per_frame": locations,
                "sensor_name": "uav_mode_tracking_lidar",
            }
        }
    }
    (tmp_path / "uav_modes.json").write_text(json.dumps(payload), encoding="utf-8")
    record = _load_uav_mode_record(
        tmp_path,
        mode="tracking",
        frame_indices=[0, 2, 5],
    )
    assert record["candidate_names"] == ["uav_r2_c2"] * 3
    np.testing.assert_allclose(record["locations"][:, 0], [0.0, 2.0, 5.0])
    assert record["movement_m"] == 5.0


def test_preview_motion_summary_reports_real_displacement() -> None:
    snapshots = [
        {
            "elapsed_s": 0.0,
            "actors": [
                {
                    "actor_id": 1,
                    "role_name": "regional_ego",
                    "type_id": "vehicle.tesla.model3",
                    "location": [0.0, 0.0, 0.0],
                    "velocity": [4.0, 0.0, 0.0],
                }
            ],
        },
        {
            "elapsed_s": 5.0,
            "actors": [
                {
                    "actor_id": 1,
                    "role_name": "regional_ego",
                    "type_id": "vehicle.tesla.model3",
                    "location": [20.0, 0.0, 0.0],
                    "velocity": [4.0, 0.0, 0.0],
                }
            ],
        },
    ]
    record = _build_motion_summary(snapshots)[0]
    assert record["displacement_m"] == 20.0
    assert record["sampled_path_m"] == 20.0
    assert record["mean_speed_mps"] == 4.0
