"""One entry point for authoring, collecting and running dataset episodes."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import math
import os
import platform
import subprocess
import tempfile
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

from .config import dump_effective_config
from .dataset_record import RunRecorder, check_run, compare_runs, json_write
from .dataset_runtime import Decision, Observation, Policy, drive_episode
from .dataset_scene import MODES, PROJECT, DatasetScene, data_root, load_frozen_scene, resolve_scene, run_directory


@contextmanager
def _exclusive_carla(scene: DatasetScene):
    endpoint = "{}:{}".format(os.environ.get("ACTIVEVIEW_CARLA_HOST", scene.config["carla"]["host"]),
                              os.environ.get("ACTIVEVIEW_CARLA_PORT", scene.config["carla"]["port"]))
    name = hashlib.sha256(endpoint.encode()).hexdigest()[:20]
    with (Path(tempfile.gettempdir()) / f"activeview-{os.getuid()}-{name}.lock").open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another activeview command owns CARLA {endpoint}") from exc
        yield


def _git_version() -> dict:
    try:
        def git(*args):
            return subprocess.check_output(["git", "-C", str(PROJECT), *args],
                                           stderr=subprocess.DEVNULL, text=True).strip()
        return {"commit": git("rev-parse", "HEAD"), "dirty_tracked_files": bool(git("status", "--porcelain", "-uno"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty_tracked_files": None}


def _traffic_summary(scene: DatasetScene) -> dict:
    fixed_plan = scene.config["regional"].get("fixed_signal_plan", {})
    return {"scene_id": scene.scene_id, "family_id": scene.family_id, "scene_sha256": scene.scene_hash,
            "sources": scene.source_hashes, "map": scene.config["carla"]["map"],
            "frame_count": scene.frame_count, "sample_hz": 1 / scene.dt,
            "decision_hz": 1 / (scene.dt * scene.decision_ticks),
            "j2_switch_time_s": fixed_plan.get("j2_switch_time_s") if fixed_plan.get("enabled") else None}


def _policy(args) -> Policy:
    if args.policy == "hover":
        return lambda obs: Decision()
    if args.policy == "offset":
        goal: tuple[float, float] | None = None

        def offset(obs: Observation) -> Decision:
            nonlocal goal
            x, y, _ = obs.uav_world_xyz
            if goal is None:
                goal = (x + args.dx_m, y + args.dy_m)
            return Decision(goal[0] - x, goal[1] - y)

        return offset
    module, sep, name = args.policy.partition(":")
    if not sep or not module or not name:
        raise ValueError("policy must be hover, offset, or an importable module:callable")
    callback = getattr(importlib.import_module(module), name)
    if not callable(callback):
        raise ValueError("policy object is not callable")
    return callback


def _run(args, scene: DatasetScene, directory: Path) -> dict:
    from .dataset_carla import start_episode

    if args.mode == "offline_hover" and (args.policy != "hover" or args.delay_ms or args.model_dir):
        raise ValueError("offline_hover only records a fixed UAV; choose an online mode for inference")
    if bool(args.model_dir) != bool(args.opencood_root):
        raise ValueError("pass both --model-dir and --opencood-root")
    if not all(math.isfinite(x) for x in (args.dx_m, args.dy_m, args.delay_ms, args.max_action_age_s)):
        raise ValueError("offsets, delay, and max action age must be finite")
    if args.delay_ms < 0 or args.max_action_age_s <= 0:
        raise ValueError("delay must be nonnegative and max action age positive")
    if args.frames is not None and not 1 <= args.frames <= scene.frame_count:
        raise ValueError(f"--frames must be in [1, {scene.frame_count}]")
    frame_count = scene.frame_count if args.frames is None else args.frames
    policy = _policy(args)
    model = None
    if args.model_dir is not None:
        from .dataset_where2comm import detector_policy
        policy, model = detector_policy(policy, opencood_root=args.opencood_root,
                                        model_dir=args.model_dir, device=args.device)

    def maybe_delay(obs: Observation) -> Decision:
        if args.delay_ms:
            time.sleep(args.delay_ms / 1000)
        return policy(obs)

    with _exclusive_carla(scene):
        directory.mkdir(parents=True, exist_ok=False)
        dump_effective_config(scene.config, directory / "effective_config.yaml")
        status = {**_traffic_summary(scene), "status": "running", "mode": args.mode,
                  "run_id": args.run_id, "frame_limit": frame_count, "policy": args.policy,
                  "offset_world_xy_m": [args.dx_m, args.dy_m] if args.policy == "offset" else None,
                  "injected_delay_ms": args.delay_ms, "max_action_age_s": args.max_action_age_s,
                  "model": model, "git": _git_version(),
                  "client_host": {"hostname": platform.node(), "os": platform.platform(),
                                  "python": platform.python_version()}}
        manifest = directory / "run_manifest.json"
        json_write(manifest, status)
        try:
            with start_episode(scene) as (backend, simulator), ExitStack() as files:
                status["simulator_identity"] = {
                    key: simulator[key] for key in ("map", "opendrive_sha256", "server_version", "client_version")
                }
                json_write(directory / "metadata.json", {
                    "format_version": "activeview-runtime-v1", "map": simulator["map"],
                    "frame_interval_s": scene.dt, "initial_uav_world_xyz": backend.initial_uav_xyz,
                    "sensor_agent_types": backend.sensor_agent_types,
                    "base_sensor_names": list(scene.config["scoring"]["base_sensors"]),
                    "candidate_sensor_names": [], "simulator": simulator,
                })
                recorder = RunRecorder(directory, args.mode == "offline_hover", files)
                drive_episode(backend, mode=args.mode, frames=frame_count, dt=scene.dt,
                              decision_ticks=scene.decision_ticks,
                              max_speed_mps=float(scene.config["oracle"]["uav_speed_mps"]),
                              max_action_age_s=args.max_action_age_s,
                              policy=maybe_delay, on_tick=recorder.tick, on_event=recorder.event)
                summary = recorder.summary(scene.dt)
                if summary["world_ticks"] != frame_count:
                    raise RuntimeError("incomplete world tick trace")
            status.update(status="complete", summary=summary)
        except BaseException as exc:
            status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            json_write(manifest, status)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(prog="activeview", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("scenes", help="list registered scene IDs")
    for name in ("validate", "preview", "run"):
        command = commands.add_parser(name)
        source = command.add_mutually_exclusive_group(required=True)
        source.add_argument("--scene", help="registered ID, e.g. Town03/scene_0001")
        source.add_argument("--config", type=Path, help="effective_config.yaml from a prior run")
        if name != "validate":
            command.add_argument("--data-root", type=Path)
            command.add_argument("--run-id", required=True, help="unique lowercase name, e.g. hover_001")
        if name == "run":
            command.add_argument("--mode", choices=MODES, required=True)
            command.add_argument("--frames", type=int, help="limit a pilot without changing the scene YAML")
            command.add_argument("--policy", default="hover", help="hover, offset, or module:callable")
            command.add_argument("--dx-m", type=float, default=5.0)
            command.add_argument("--dy-m", type=float, default=0.0)
            command.add_argument("--delay-ms", type=float, default=0.0)
            command.add_argument("--max-action-age-s", type=float, default=1.0)
            command.add_argument("--model-dir", type=Path)
            command.add_argument("--opencood-root", type=Path)
            command.add_argument("--device", default="cuda:0")
    check = commands.add_parser("check-run", help="validate a completed run against its scene")
    check.add_argument("directory", type=Path)
    compare = commands.add_parser("compare", help="check two completed hover runs for replay drift")
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "scenes":
            for path in sorted((PROJECT / "scenes").glob("*/*/config/scene.yaml")):
                print(path.parent.parent.relative_to(PROJECT / "scenes").as_posix())
            return
        if args.command in ("compare", "check-run"):
            result = (compare_runs(args.left, args.right) if args.command == "compare"
                      else check_run(args.directory))
            print(json.dumps(result, indent=2))
            if not result["passed"]:
                raise SystemExit(1)
            return
        scene = load_frozen_scene(args.config) if args.config is not None else resolve_scene(args.scene)
        if args.command == "validate":
            print(json.dumps(_traffic_summary(scene), indent=2))
            return
        mode = args.mode if args.command == "run" else "preview"
        directory = run_directory(data_root(args.data_root), scene.scene_id, mode, args.run_id)
        if directory.exists():
            raise FileExistsError(f"{directory} already exists; choose a new --run-id")
        if args.command == "preview":
            from .regional_preview import preview
            with _exclusive_carla(scene), tempfile.TemporaryDirectory() as tmp:
                effective = {**scene.config, "output": {"root": str(directory.parent), "run_name": directory.name}}
                effective["carla"] = {**scene.config["carla"],
                                      "host": os.environ.get("ACTIVEVIEW_CARLA_HOST", scene.config["carla"]["host"]),
                                      "port": int(os.environ.get("ACTIVEVIEW_CARLA_PORT", scene.config["carla"]["port"])),
                                      "traffic_manager_port": int(os.environ.get("ACTIVEVIEW_TM_PORT", scene.config["carla"]["traffic_manager_port"]))}
                config = Path(tmp) / "preview.yaml"
                dump_effective_config(effective, config)
                preview(config)
            print(directory)
            return
        report = _run(args, scene, directory)
        print(json.dumps({"directory": str(directory), **report["summary"]}, indent=2))
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"activeview: {exc}\n")


if __name__ == "__main__":
    main()
