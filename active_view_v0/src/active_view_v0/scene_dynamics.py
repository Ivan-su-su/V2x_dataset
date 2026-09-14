from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config


def validate_coverage_transition(run_dir: str | Path) -> tuple[Path, Path] | None:
    """Check that the transient J2 CAV really covers the area and then leaves."""
    run_dir = Path(run_dir).expanduser().resolve()
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    if "grid_center_xyz" not in metadata:
        print(
            "Coverage-transition check: SKIPPED (this run predates v0.9 grid/agent metadata)."
        )
        return None
    cfg = load_config(run_dir / "effective_config.yaml")
    manifest = [
        json.loads(line)
        for line in (run_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    grid_center = np.asarray(metadata["grid_center_xyz"][:2], dtype=np.float64)
    elapsed: list[float] = []
    distances: list[float] = []
    for item in manifest:
        frame = json.loads((run_dir / item["path"] / "frame.json").read_text(encoding="utf-8"))
        state = frame.get("sensing_agents", {}).get("regional_car3")
        if state is None:
            print(
                "Coverage-transition check: SKIPPED "
                "(this run predates v0.9 sensing-agent metadata)."
            )
            return None
        elapsed.append(float(item["elapsed_s"]))
        xy = np.asarray(state["location"][:2], dtype=np.float64)
        distances.append(float(np.linalg.norm(xy - grid_center)))
    validation = cfg["regional"].get("validation", {})
    result = _judge_transition(
        np.asarray(elapsed, dtype=np.float64),
        np.asarray(distances, dtype=np.float64),
        early_end_s=float(validation.get("early_end_s", 10.0)),
        late_start_s=float(validation.get("late_start_s", 14.0)),
        service_radius_m=float(cfg["regional"].get("transient_coverage_radius_m", 60.0)),
        minimum_inside_frames=int(validation.get("minimum_transient_inside_frames", 2)),
        minimum_outside_frames=int(validation.get("minimum_transient_outside_frames", 2)),
        minimum_exit_shift_m=float(validation.get("minimum_transient_exit_shift_m", 30.0)),
    )
    payload: dict[str, Any] = {
        "role": "regional_car3",
        "description": "CAV-2 / J2 transient sensing provider",
        "grid_center_world_xy": grid_center.tolist(),
        "elapsed_s": elapsed,
        "distance_to_uav_service_center_m": distances,
        **result,
    }
    json_path = run_dir / "coverage_transition_report.json"
    text_path = run_dir / "coverage_transition_report.txt"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    text_path.write_text(_render_text(payload), encoding="utf-8")
    print(text_path.read_text(encoding="utf-8"))
    print(f"Saved: {json_path}")
    print(f"Saved: {text_path}")
    return json_path, text_path


def _judge_transition(
    elapsed_s: np.ndarray,
    distances_m: np.ndarray,
    *,
    early_end_s: float,
    late_start_s: float,
    service_radius_m: float,
    minimum_inside_frames: int,
    minimum_outside_frames: int,
    minimum_exit_shift_m: float,
) -> dict[str, Any]:
    elapsed_s = np.asarray(elapsed_s, dtype=np.float64)
    distances_m = np.asarray(distances_m, dtype=np.float64)
    if elapsed_s.shape != distances_m.shape or elapsed_s.ndim != 1 or not len(elapsed_s):
        raise ValueError("elapsed_s and distances_m must be non-empty vectors of equal length")
    early = elapsed_s <= early_end_s + 1e-6
    late = elapsed_s >= late_start_s - 1e-6
    if not early.any() or not late.any():
        raise ValueError("coverage-transition early/late windows contain no frames")
    early_inside = int(np.count_nonzero(distances_m[early] <= service_radius_m))
    late_outside = int(np.count_nonzero(distances_m[late] > service_radius_m))
    early_median = float(np.median(distances_m[early]))
    late_median = float(np.median(distances_m[late]))
    exit_shift = late_median - early_median
    failures: list[str] = []
    if early_inside < minimum_inside_frames:
        failures.append(
            f"CAV-2 is inside the J2 service radius in only {early_inside} early frames"
        )
    if late_outside < minimum_outside_frames:
        failures.append(
            f"CAV-2 is outside the J2 service radius in only {late_outside} late frames"
        )
    if exit_shift < minimum_exit_shift_m:
        failures.append(
            f"CAV-2 median exit shift is {exit_shift:.1f}m (<{minimum_exit_shift_m:.1f}m)"
        )
    return {
        "passed": not failures,
        "failures": failures,
        "service_radius_m": service_radius_m,
        "early_window_s": [float(elapsed_s[early][0]), float(elapsed_s[early][-1])],
        "late_window_s": [float(elapsed_s[late][0]), float(elapsed_s[late][-1])],
        "early_inside_frame_count": early_inside,
        "late_outside_frame_count": late_outside,
        "early_median_distance_m": early_median,
        "late_median_distance_m": late_median,
        "median_exit_shift_m": exit_shift,
    }


def _render_text(payload: dict[str, Any]) -> str:
    lines = [
        "Sparse-coverage scene transition check",
        "======================================",
        f"Passed: {payload['passed']}",
        f"Service radius: {payload['service_radius_m']:.1f}m",
        f"CAV-2 early inside frames: {payload['early_inside_frame_count']}",
        f"CAV-2 late outside frames: {payload['late_outside_frame_count']}",
        f"Median distance early -> late: {payload['early_median_distance_m']:.1f}m -> "
        f"{payload['late_median_distance_m']:.1f}m",
        f"Median exit shift: {payload['median_exit_shift_m']:+.1f}m",
    ]
    if payload["failures"]:
        lines.append("Failures:")
        lines.extend(f"  - {failure}" for failure in payload["failures"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the transient CAV coverage event")
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    validate_coverage_transition(args.run_dir)


if __name__ == "__main__":
    main()
