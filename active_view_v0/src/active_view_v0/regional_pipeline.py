from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .config import load_config
from .idea_report import build_report
from .occlusion_report import build_occlusion_report
from .path_metrics import evaluate_path_metrics
from .plot_results import plot
from .recovery_report import build_recovery_report
from .scoring import score_dataset
from .scene_dynamics import validate_coverage_transition
from .solve_oracle import solve


def run_pipeline(
    config: str | Path,
    overwrite: bool = False,
    duration_s: float | None = None,
    run_name: str | None = None,
) -> Path:
    config = Path(config).expanduser().resolve()
    cfg = load_config(config)
    if run_name is not None:
        cfg["output"]["run_name"] = str(run_name)
    run_dir = Path(cfg["output"]["root"]).expanduser() / str(cfg["output"]["run_name"])
    command = [
        sys.executable,
        "-m",
        "active_view_v0.regional_collector",
        "--config",
        str(config),
    ]
    if duration_s is not None:
        command.extend(["--duration-s", str(float(duration_s))])
    if run_name is not None:
        command.extend(["--run-name", str(run_name)])
    if overwrite:
        command.append("--overwrite")
    print("Starting isolated CARLA collection process...")
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        if _collection_is_complete(run_dir, cfg, duration_s):
            print(
                f"WARNING: collector exited with code {completed.returncode} after writing a complete run; "
                "continuing with offline evaluation."
            )
        else:
            raise RuntimeError(
                f"collector exited with code {completed.returncode} and the dataset is incomplete: {run_dir}"
            )
    validate_coverage_transition(run_dir)
    score_dataset(run_dir)
    solve(run_dir)
    evaluate_path_metrics(run_dir)
    solve(
        run_dir,
        score_key="recovery",
        output_name="recovery_oracle_results.json",
    )
    evaluate_path_metrics(
        run_dir,
        oracle_name="recovery_oracle_results.json",
        output_stem="recovery_path_binary_metrics",
    )
    if cfg.get("regional", {}).get("blocker_role"):
        build_occlusion_report(run_dir)
    else:
        print(
            "Skipping legacy occlusion report: "
            "regional.blocker_role is not configured."
        )
    build_recovery_report(run_dir)
    plot(run_dir)
    build_report(run_dir)
    print(f"Pipeline complete: {run_dir}")
    return run_dir


def _collection_is_complete(
    run_dir: Path, cfg: dict, duration_s: float | None
) -> bool:
    status_path = run_dir / "collection_status.json"
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") != "complete":
                return False
        except (OSError, ValueError, TypeError):
            return False
    manifest_path = run_dir / "manifest.jsonl"
    if not manifest_path.exists():
        return False
    try:
        entries = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]
    except (OSError, ValueError, TypeError):
        return False
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        expected = int(status.get("expected_keyframe_count", status.get("keyframe_count", -1)))
    elif cfg.get("regional", {}).get("frame_count") is not None and duration_s is None:
        expected = int(cfg["regional"]["frame_count"])
    else:
        interval = float(
            cfg.get("regional", {}).get(
                "collection_interval_s", cfg["grid"]["keyframe_interval_s"]
            )
        )
        duration = (
            float(cfg["regional"]["duration_s"])
            if duration_s is None
            else float(duration_s)
        )
        include_endpoint = bool(cfg.get("regional", {}).get("include_endpoint", True))
        expected = int(round(duration / interval)) + int(include_endpoint)
    if len(entries) != expected:
        return False
    return all((run_dir / entry["path"] / "frame.json").is_file() for entry in entries)


def evaluate_only(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir).expanduser().resolve()
    validate_coverage_transition(run_dir)
    score_dataset(run_dir)
    solve(run_dir)
    evaluate_path_metrics(run_dir)
    solve(
        run_dir,
        score_key="recovery",
        output_name="recovery_oracle_results.json",
    )
    evaluate_path_metrics(
        run_dir,
        oracle_name="recovery_oracle_results.json",
        output_stem="recovery_path_binary_metrics",
    )
    build_occlusion_report(run_dir)
    build_recovery_report(run_dir)
    plot(run_dir)
    build_report(run_dir)
    print(f"Evaluation complete: {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect and evaluate a regional active-UAV clip")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", type=Path, help="collect and evaluate using this YAML")
    mode.add_argument("--evaluate-only", type=Path, metavar="RUN_DIR")
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.evaluate_only is not None:
        evaluate_only(args.evaluate_only)
    else:
        run_pipeline(
            args.config,
            overwrite=args.overwrite,
            duration_s=args.duration_s,
            run_name=args.run_name,
        )


if __name__ == "__main__":
    main()
