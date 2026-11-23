"""Tick ownership, causal observations and UAV command scheduling."""

from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class Tick:
    index: int
    frame: int
    sim_time_s: float
    capture_wall_s: float
    uav_world_xyz: tuple[float, float, float]
    ego_transform: dict
    state: dict  # evaluator and recorder only; never included in Observation


@dataclass(frozen=True)
class Observation:
    tick_index: int
    frame_id: int
    sim_time_s: float
    capture_wall_s: float
    ready_wall_s: float
    uav_world_xyz: tuple[float, float, float]
    ego_transform: dict
    sensors: dict[str, Any]
    sensor_agent_types: dict[str, str]


@dataclass(frozen=True)
class Decision:
    dx_m: float = 0.0
    dy_m: float = 0.0
    predictions: dict | None = None


@dataclass(frozen=True)
class Result:
    observation: Observation
    decision: Decision | None
    started_wall_s: float
    completed_wall_s: float
    error: Exception | None = None


class EpisodeBackend(Protocol):
    initial_uav_xyz: tuple[float, float, float]
    sensor_agent_types: dict[str, str]

    def tick(self, index: int, uav_xyz: tuple[float, float, float]) -> Tick: ...
    def collect(self, frame: int) -> dict[str, Any]: ...
    def poll(self) -> list[tuple[str, Any]]: ...


Policy = Callable[[Observation], Decision]
RecordTick = Callable[[Tick, Observation | None], None]
RecordEvent = Callable[[dict], None]


def _towards(xyz: tuple, goal: tuple, max_distance: float) -> tuple:
    distance = math.dist(xyz, goal)
    if distance <= max_distance:
        return goal
    return tuple(a + (b - a) * max_distance / distance for a, b in zip(xyz, goal))


def _invoke(policy: Policy, obs: Observation) -> Result:
    started = time.monotonic()
    try:
        decision = policy(obs)
        if not isinstance(decision, Decision) or not all(
            math.isfinite(x) for x in (decision.dx_m, decision.dy_m)
        ):
            raise ValueError("policy must return a Decision with finite meter offsets")
        return Result(obs, decision, started, time.monotonic())
    except Exception as exc:
        return Result(obs, None, started, time.monotonic(), exc)


class _LatestWorker:
    """At most one active call and one waiting decision observation."""

    def __init__(self, policy: Policy):
        self.policy = policy
        self.waiting: queue.Queue[Observation] = queue.Queue(maxsize=1)
        self.results: queue.Queue[Result] = queue.Queue()
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="activeview-policy")
        self.thread.start()

    def _run(self) -> None:
        while not self.stopped.is_set():
            try:
                obs = self.waiting.get(timeout=0.05)
            except queue.Empty:
                continue
            if not self.stopped.is_set():
                self.results.put(_invoke(self.policy, obs))

    def submit(self, obs: Observation) -> Observation | None:
        replaced = None
        try:
            replaced = self.waiting.get_nowait()
        except queue.Empty:
            pass
        self.waiting.put_nowait(obs)
        return replaced


def drive_episode(
    backend: EpisodeBackend, *, mode: str, frames: int, dt: float, decision_ticks: int,
    max_speed_mps: float, max_action_age_s: float, policy: Policy,
    on_tick: RecordTick, on_event: RecordEvent,
) -> None:
    """Run one episode; only this loop calls the backend tick."""
    position = target = tuple(backend.initial_uav_xyz)
    streams = backend.sensor_agent_types
    sensor_names = set(streams)
    xyz_limit = max_speed_mps * dt

    def observe(tick: Tick, measurements: dict[str, Any]) -> Observation:
        if set(measurements) != sensor_names or any(
            item.frame != tick.frame for item in measurements.values()
        ):
            raise RuntimeError(f"incomplete or mixed sensor measurements for frame {tick.frame}")
        return Observation(tick.index, tick.frame, tick.sim_time_s, tick.capture_wall_s,
                           time.monotonic(), tick.uav_world_xyz, tick.ego_transform,
                           measurements, dict(streams))

    def accept(result: Result, apply_tick: int | None, reason: str | None = None) -> None:
        nonlocal target
        obs = result.observation
        if result.error is not None:
            raise RuntimeError(f"policy failed on frame {obs.frame_id}") from result.error
        assert result.decision is not None
        dx, dy = result.decision.dx_m, result.decision.dy_m
        goal = (obs.uav_world_xyz[0] + dx, obs.uav_world_xyz[1] + dy, obs.uav_world_xyz[2])
        if not all(math.isfinite(x) for x in goal):
            raise ValueError("UAV goal is outside the numeric coordinate range")
        if apply_tick is not None:
            target = goal
        on_event({"event": "decision", "source_frame_id": obs.frame_id,
                  "source_tick": obs.tick_index, "capture_wall_s": obs.capture_wall_s,
                  "observation_ready_wall_s": obs.ready_wall_s,
                  "inference_started_wall_s": result.started_wall_s,
                  "inference_done_wall_s": result.completed_wall_s,
                  "scheduled_apply_wall_s": time.monotonic() if apply_tick is not None else None,
                  "apply_tick": apply_tick, "discard_reason": reason, "target_world_xyz": goal,
                  "predictions": result.decision.predictions})

    if mode in ("offline_hover", "online_step"):
        for index in range(frames):
            position = target
            tick = backend.tick(index, position)
            obs = observe(tick, backend.collect(tick.frame))
            on_tick(tick, obs)
            if mode == "online_step" and index % decision_ticks == 0:
                result = _invoke(policy, obs)
                accept(result, index + 1 if index + 1 < frames else None,
                       "episode_ended" if index + 1 == frames else None)
        return
    if mode != "online_realtime":
        raise ValueError(f"unsupported runtime mode: {mode}")

    worker = _LatestWorker(policy)
    # Retain up to four pending frames. Slow/missing sensors cannot block the tick owner.
    pending: dict[int, tuple[Tick, dict[str, Any]]] = {}
    submitted: dict[int, Observation] = {}
    last_submitted = last_applied = -1

    def poll_sensors(submit: bool = True) -> None:
        nonlocal last_submitted
        for name, item in backend.poll():
            if item.frame in pending:
                pending[item.frame][1][name] = item
        for frame, (tick, items) in list(pending.items()):
            if set(items) != sensor_names:
                continue
            del pending[frame]
            obs = observe(tick, items)
            on_tick(tick, obs)
            if submit and tick.index % decision_ticks == 0:
                if tick.index <= last_submitted or time.monotonic() - tick.capture_wall_s > max_action_age_s:
                    on_event({"event": "decision_skipped", "source_tick": tick.index,
                              "reason": "stale_observation"})
                    continue
                replaced = worker.submit(obs)
                submitted[obs.tick_index] = obs
                last_submitted = tick.index
                if replaced is not None:
                    submitted.pop(replaced.tick_index, None)
                    on_event({"event": "decision_skipped", "source_tick": replaced.tick_index,
                              "reason": "newer_observation"})

    def consume_results(index: int | None) -> None:
        nonlocal last_applied
        while True:
            try:
                result = worker.results.get_nowait()
            except queue.Empty:
                return
            obs = result.observation
            submitted.pop(obs.tick_index, None)
            reason = None
            if index is None:
                reason = "episode_ended"
            elif obs.tick_index <= last_applied:
                reason = "superseded"
            elif time.monotonic() - obs.capture_wall_s > max_action_age_s:
                reason = "stale_action"
            accept(result, index if reason is None else None, reason)
            if reason is None:
                last_applied = obs.tick_index

    try:
        next_start = time.monotonic()
        for index in range(frames):
            while True:
                poll_sensors()
                remaining = next_start - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(remaining, 0.002))
            consume_results(index)
            # Slower simulation work is reported, rather than altering dt or skipping world ticks.
            next_start = time.monotonic() + dt
            position = _towards(position, target, xyz_limit)
            tick = backend.tick(index, position)
            pending[tick.frame] = (tick, {})
            poll_sensors()
            if len(pending) > 4:
                expired, _ = pending.pop(next(iter(pending)))
                on_tick(expired, None)
        # Drain callbacks already in flight, without advancing the simulation.
        deadline = time.monotonic() + dt
        while pending:
            poll_sensors(submit=False)
            if not pending or time.monotonic() >= deadline:
                break
            time.sleep(0.002)
        worker.stopped.set()
        worker.thread.join(timeout=1.0)
        consume_results(None)
        for obs in submitted.values():
            on_event({"event": "decision_skipped", "source_tick": obs.tick_index,
                      "reason": "episode_ended"})
        for tick, _ in pending.values():
            on_tick(tick, None)
    finally:
        worker.stopped.set()
