from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config


def build_recovery_report(run_dir: str | Path) -> tuple[Path, Path]:
    """Judge the active-view hypothesis using recovered targets, not point density."""
    run_dir = Path(run_dir).expanduser().resolve()
    cfg = load_config(run_dir / "effective_config.yaml")
    arrays = np.load(run_dir / "geometry_scores.npz", allow_pickle=False)
    details = json.loads((run_dir / "geometry_details.json").read_text(encoding="utf-8"))
    metrics = json.loads(
        (run_dir / "recovery_path_binary_metrics.json").read_text(encoding="utf-8")
    )
    occlusion = json.loads((run_dir / "occlusion_report.json").read_text(encoding="utf-8"))
    elapsed = np.asarray(arrays["elapsed_s"], dtype=np.float64)
    target_counts = np.asarray(arrays["target_counts"], dtype=np.int64)
    validation = cfg["regional"].get("validation", {})
    early_end = float(validation.get("early_end_s", 10.0))
    late_start = float(validation.get("late_start_s", 20.0))

    phase_stats = {
        "early": _phase_stats(details, elapsed <= early_end + 1e-6),
        "late": _phase_stats(details, elapsed >= late_start - 1e-6),
    }
    comparison = metrics["dynamic_vs_best_fixed"]["center_start"]
    dynamic = metrics["strategy_metrics"]["speed_constrained_center_start"]
    minimum_targets = int(validation.get("minimum_targets_per_frame_median", 8))
    minimum_missed = int(validation.get("minimum_missed_instances_per_phase", 5))
    minimum_rescuable = int(validation.get("minimum_rescuable_instances_per_phase", 3))
    minimum_extra = int(validation.get("minimum_dynamic_extra_rescues", 2))
    minimum_recall = float(validation.get("minimum_dynamic_recall_gain_pp", 1.0))
    minimum_move = float(validation.get("minimum_dynamic_movement_m", 15.0))

    scene_failures: list[str] = []
    median_targets = float(np.median(target_counts))
    if median_targets < minimum_targets:
        scene_failures.append(
            f"median target count is {median_targets:.1f} (<{minimum_targets})"
        )
    for phase, record in phase_stats.items():
        if record["base_missed_instances"] < minimum_missed:
            scene_failures.append(
                f"{phase} base misses are {record['base_missed_instances']} (<{minimum_missed})"
            )
        if record["uav_rescuable_instances"] < minimum_rescuable:
            scene_failures.append(
                f"{phase} UAV-rescuable misses are {record['uav_rescuable_instances']} (<{minimum_rescuable})"
            )
    if not occlusion["passed"]:
        scene_failures.append("moving-occlusion validation failed")

    method_failures: list[str] = []
    if comparison["additional_rescued_instances"] < minimum_extra:
        method_failures.append(
            f"dynamic path adds {comparison['additional_rescued_instances']} rescues (<{minimum_extra})"
        )
    recall_gain = comparison["target_recall_gain_pp"]
    if recall_gain is None or recall_gain < minimum_recall:
        method_failures.append(
            f"dynamic recall gain is {recall_gain if recall_gain is not None else 'n/a'} pp (<{minimum_recall:.1f})"
        )
    if dynamic["movement_m"] < minimum_move:
        method_failures.append(
            f"dynamic path moves {dynamic['movement_m']:.1f}m (<{minimum_move:.1f}m)"
        )

    if scene_failures:
        verdict = "SCENE_NOT_INFORMATIVE"
        reason = scene_failures[0]
    elif method_failures:
        verdict = "ACTIVE_VIEW_UNSUPPORTED"
        reason = method_failures[0]
    else:
        verdict = "ACTIVE_VIEW_PROMISING"
        reason = "The center-start, speed-limited UAV rescues more targets than every fixed position."
    payload: dict[str, Any] = {
        "verdict": verdict,
        "reason": reason,
        "scope": "target_recovery_geometry_proxy_not_detector_AP",
        "median_target_count": median_targets,
        "phase_stats": phase_stats,
        "occlusion_passed": bool(occlusion["passed"]),
        "dynamic_center_start_vs_best_fixed": comparison,
        "dynamic_center_start": {
            key: dynamic[key]
            for key in ("rescued_instances", "target_recall", "miss_recovery_rate", "movement_m", "switch_count", "path_names")
        },
        "scene_failures": scene_failures,
        "method_failures": method_failures,
        "decision_thresholds": {
            "median_targets": minimum_targets,
            "missed_instances_per_phase": minimum_missed,
            "rescuable_instances_per_phase": minimum_rescuable,
            "dynamic_extra_rescues": minimum_extra,
            "dynamic_recall_gain_pp": minimum_recall,
            "dynamic_movement_m": minimum_move,
        },
    }
    json_path = run_dir / "recovery_report.json"
    text_path = run_dir / "recovery_report.txt"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    text_path.write_text(_render_text(payload), encoding="utf-8")
    print(text_path.read_text(encoding="utf-8"))
    return json_path, text_path


def _phase_stats(details: list[dict[str, Any]], mask: np.ndarray) -> dict[str, int]:
    missed = 0
    rescuable = 0
    frame_count = 0
    rescue_frames = 0
    for selected, detail in zip(mask.tolist(), details):
        if not selected:
            continue
        frame_count += 1
        base_visible = np.asarray(detail["base_visible"], dtype=bool)
        if not len(base_visible):
            continue
        any_union = np.zeros(len(base_visible), dtype=bool)
        for candidate in detail["candidates"].values():
            any_union |= np.asarray(candidate["union_visible"], dtype=bool)
        rescuable_mask = (~base_visible) & any_union
        missed += int((~base_visible).sum())
        rescuable += int(rescuable_mask.sum())
        rescue_frames += int(rescuable_mask.any())
    return {
        "frame_count": frame_count,
        "base_missed_instances": missed,
        "uav_rescuable_instances": rescuable,
        "uav_rescue_frames": rescue_frames,
    }


def _render_text(payload: dict[str, Any]) -> str:
    comparison = payload["dynamic_center_start_vs_best_fixed"]
    lines = [
        "Active AirV2X recovery-first idea check",
        "========================================",
        f"Verdict: {payload['verdict']}",
        f"Reason: {payload['reason']}",
        "",
        f"Median targets/frame: {payload['median_target_count']:.1f}",
        f"Moving occlusion passed: {payload['occlusion_passed']}",
    ]
    for phase, record in payload["phase_stats"].items():
        lines.append(
            f"{phase:5s}: missed={record['base_missed_instances']} "
            f"rescuable={record['uav_rescuable_instances']} "
            f"rescue_frames={record['uav_rescue_frames']}"
        )
    lines.extend(
        [
            "",
            "Center-start speed-limited UAV vs best fixed:",
            f"  extra rescued targets: {comparison['additional_rescued_instances']:+d}",
            f"  recall gain: {comparison['target_recall_gain_pp']:+.2f} pp",
            f"  movement: {comparison['movement_m']:.1f} m",
            "",
            "Important: pass this proxy gate before training a detector.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the recovery-first Active AirV2X report")
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    build_recovery_report(args.run_dir)


if __name__ == "__main__":
    main()
