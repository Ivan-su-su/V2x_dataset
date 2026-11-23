import json
import math
import threading
import time
from contextlib import ExitStack
from types import SimpleNamespace

import pytest

from active_view_v0.config import dump_effective_config, load_config
from active_view_v0.dataset_record import RunRecorder, compare_runs, json_write
from active_view_v0.dataset_runtime import Decision, Tick, drive_episode
from active_view_v0.dataset_scene import PROJECT, data_root, load_frozen_scene, resolve_scene, run_directory


def test_town03_registered_scenes_preserve_baseline_and_apply_exact_override(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    original = load_config(PROJECT / "configs/dense_dynamic_town03_40s.yaml")
    base, variant = map(resolve_scene, ("Town03/scene_0001", "Town03/scene_0002"))
    assert base.config["regional"] == original["regional"]
    assert base.config["sensors"] == original["sensors"]
    assert "output" not in base.config
    assert base.scene_hash == resolve_scene(base.scene_id).scene_hash
    frozen_path = tmp_path / "effective_config.yaml"
    dump_effective_config(base.config, frozen_path)
    assert load_frozen_scene(frozen_path).scene_hash == base.scene_hash
    assert base.scene_hash != variant.scene_hash and base.family_id == variant.family_id
    assert [scene.config["regional"]["fixed_signal_plan"]["j2_switch_time_s"]
            for scene in (base, variant)] == [20, 24]
    assert base.frame_count == variant.frame_count == 400
    assert base.decision_ticks == variant.decision_ticks == 10
    monkeypatch.setenv("ACTIVEVIEW_DATA_ROOT", str(tmp_path))
    assert run_directory(data_root(), base.scene_id, "offline_hover", "smoke_001") == (
        tmp_path / "Town03/scene_0001/offline_hover/smoke_001"
    )
    with pytest.raises(ValueError):
        run_directory(data_root(), base.scene_id, "offline_hover", "../unsafe")


def transform(xyz):
    x, y, z = xyz
    return SimpleNamespace(
        location=SimpleNamespace(x=x, y=y, z=z),
        rotation=SimpleNamespace(roll=0, pitch=0, yaw=0),
        get_matrix=lambda: [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]],
    )


class FakeCarla:
    initial_uav_xyz = (0.0, 0.0, 40.0)
    sensor_agent_types = {"uav_lidar": "drone"}

    def __init__(self, *, start=50, missing=(), dt=0.02):
        self.start, self.missing, self.dt = start, missing, dt
        self.positions, self.pending = [], []
        self.later_tick = threading.Event()

    def tick(self, index, position):
        self.positions.append(position)
        if index == 2:
            self.later_tick.set()
        frame = self.start + index
        sim_time = index * self.dt
        if index not in self.missing:
            self.pending.append(("uav_lidar", SimpleNamespace(
                frame=frame, timestamp=sim_time, transform=transform(position), raw_data=bytes(16),
            )))
        ego = {"location": [float(index), 0, 0], "rotation": [0, 0, 0]}
        return Tick(index, frame, sim_time, time.monotonic(), position, ego,
                    {"actors": {"ego:001": {"type_id": "vehicle.mock", **ego}},
                     "signals": {"J2": "Red"}, "gt": []})

    def poll(self):
        items, self.pending = self.pending, []
        return items

    def collect(self, frame):
        return dict(self.poll())


def test_step_waits_before_advancing_and_uses_next_frame_pose():
    backend = FakeCarla()
    events = []

    def policy(obs):
        assert not hasattr(obs, "state")
        assert len(backend.positions) == obs.tick_index + 1
        if obs.tick_index == 0:
            return Decision(5, -2)
        return Decision()

    drive_episode(backend, mode="online_step", frames=3, dt=backend.dt, decision_ticks=1,
                  max_speed_mps=1, max_action_age_s=1, policy=policy, on_tick=lambda *_: None,
                  on_event=events.append)
    assert backend.positions == [(0, 0, 40), (5, -2, 40), (5, -2, 40)]
    assert events[0]["source_frame_id"] == 50 and events[0]["apply_tick"] == 1
    assert events[-1]["discard_reason"] == "episode_ended"


def test_realtime_keeps_ticking_during_policy_and_missing_sensors():
    backend = FakeCarla(missing=(1,))
    events, observed = [], {}

    def policy(obs):
        assert backend.later_tick.wait(timeout=1)
        return Decision(2, 0)

    drive_episode(backend, mode="online_realtime", frames=16, dt=backend.dt,
                  decision_ticks=16, max_speed_mps=5, max_action_age_s=5, policy=policy,
                  on_tick=lambda tick, obs: observed.update({tick.index: obs}), on_event=events.append)
    assert len(backend.positions) == len(observed) == 16
    assert observed[1] is None
    assert all(obs is None or obs.frame_id == obs.sensors["uav_lidar"].frame for obs in observed.values())
    command = next(e for e in events if e["event"] == "decision")
    assert command["apply_tick"] >= 3
    assert backend.positions[command["apply_tick"] - 1] == backend.initial_uav_xyz
    assert backend.positions[-1][0] > 0
    assert all(math.dist(x, y) <= 5 * backend.dt + 1e-8
               for x, y in zip(backend.positions, backend.positions[1:]))


def test_realtime_rejects_stale_result():
    backend = FakeCarla()
    events = []

    def slow(_):
        assert backend.later_tick.wait(timeout=1)
        return Decision(10, 0)

    drive_episode(backend, mode="online_realtime", frames=12, dt=backend.dt,
                  decision_ticks=12, max_speed_mps=5, max_action_age_s=0.03, policy=slow,
                  on_tick=lambda *_: None, on_event=events.append)
    assert events[0]["discard_reason"] == "stale_action"
    assert set(backend.positions) == {backend.initial_uav_xyz}


def test_compare_repeats_by_relative_tick_and_pinpoints_first_difference(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    for path, start, mode in ((left, 99, "offline_hover"), (right, 188, "online_step")):
        path.mkdir()
        with ExitStack() as files:
            recorder = RunRecorder(path, mode == "offline_hover", files)
            backend = FakeCarla(start=start)
            drive_episode(backend, mode=mode, frames=4, dt=backend.dt,
                          decision_ticks=2, max_speed_mps=5, max_action_age_s=1,
                          policy=lambda _: Decision(), on_tick=recorder.tick, on_event=recorder.event)
        json_write(path / "run_manifest.json", {"status": "complete", "policy": "hover",
                   "scene_sha256": "shared", "frame_limit": 4, "simulator_identity": {}})
    assert (left / "frames/000003/agents/uav/lidar.bin").stat().st_size == 16
    assert not (right / "frames").exists()
    assert compare_runs(left, right)["passed"]
    trace = right / "state_trace.jsonl"
    lines = [json.loads(line) for line in trace.read_text().splitlines()]
    lines[2]["signals"]["J2"] = "Green"
    trace.write_text("".join(json.dumps(line) + "\n" for line in lines))
    assert compare_runs(left, right)["first_difference"] == {"tick": 2, "field": "signals"}


def test_check_run_detects_phase_reported_at_wrong_tick(tmp_path):
    from active_view_v0.dataset_record import check_run

    scene = resolve_scene("Town03/scene_0002")
    dump_effective_config(scene.config, tmp_path / "effective_config.yaml")
    # A 25-second pilot reaches J2's switch but intentionally skips full-clip density gates.
    n = 250
    summary = {"world_ticks": n, "observed_frames": n, "missing_frames": 0}
    json_write(tmp_path / "run_manifest.json", {"status": "complete", "frame_limit": n,
               "frame_count": scene.frame_count, "mode": "offline_hover", "summary": summary})
    rows = []
    for i in range(n):
        late = i >= 240
        rows.append({"keyframe_index": i, "elapsed_s": i * scene.dt, "sensors": {"uav_lidar": {}},
                     "signals": {"J2": {"junction": "J2_cross_green",
                         "phase": "j2_corridor_green" if late else "j2_cross_green",
                         "actual_state": "Green" if late else "Red",
                         "expected_state": "Green" if late else "Red"}}})
    path = tmp_path / "state_trace.jsonl"

    def write():
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    write()
    result = check_run(tmp_path)
    assert result["passed"] and result["first_late_phase_s"] == 24.0
    rows[240]["signals"]["J2"]["actual_state"] = "Red"
    write()
    assert check_run(tmp_path)["failures"] == ["J2 phase or lamp states disagree with config at tick 240"]
