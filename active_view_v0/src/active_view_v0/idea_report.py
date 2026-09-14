from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config


def build_report(run_dir: str | Path) -> tuple[Path, Path]:
    """Summarize whether this scene contains a useful active-view signal.

    This intentionally judges only the inexpensive LiDAR-support proxy.  It is
    a gate before detector training, not a substitute for AP/recall results.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    arrays = np.load(run_dir / "geometry_scores.npz", allow_pickle=False)
    oracle = json.loads((run_dir / "oracle_results.json").read_text(encoding="utf-8"))
    cfg = load_config(run_dir / "effective_config.yaml")
    scores = np.asarray(arrays["scores"], dtype=np.float64)
    elapsed = np.asarray(arrays["elapsed_s"], dtype=np.float64)
    target_counts = np.asarray(arrays["target_counts"], dtype=np.int64)
    base_visible_counts = np.asarray(arrays["base_visible_counts"], dtype=np.int64)
    max_binary_gain = np.asarray(arrays["gain"], dtype=np.float64).max(axis=1)
    spans = scores.max(axis=1) - scores.min(axis=1)
    solutions = oracle["solutions"]

    center = solutions["center_fixed"]
    fixed = solutions["best_fixed"]
    per_frame = solutions["per_frame_unconstrained"]
    constrained = solutions["speed_constrained_center_start"]
    constrained_best_start = solutions["speed_constrained_best_start"]
    center_reward = float(center["perception_reward"])
    fixed_reward = float(fixed["perception_reward"])
    per_frame_reward = float(per_frame["perception_reward"])
    constrained_reward = float(constrained["perception_reward"])

    # Candidate positions differ meaningfully when their spread exceeds 1%
    # of the frame's best score (with an absolute numerical floor).
    best_per_frame = scores.max(axis=1)
    informative = spans > np.maximum(1e-4, 0.01 * np.maximum(best_per_frame, 1e-6))
    informative_fraction = float(informative.mean())
    relative = {
        "best_fixed_over_center": _relative(fixed_reward, center_reward),
        "per_frame_over_best_fixed": _relative(per_frame_reward, fixed_reward),
        "speed_constrained_over_best_fixed": _relative(constrained_reward, fixed_reward),
        "speed_constrained_over_center": _relative(constrained_reward, center_reward),
    }
    validation = cfg.get("regional", {}).get("validation", {})
    scene_health = _scene_health(
        scores,
        elapsed,
        target_counts,
        base_visible_counts,
        max_binary_gain,
        early_end_s=float(validation.get("early_end_s", 14.0)),
        late_start_s=float(validation.get("late_start_s", 16.0)),
        minimum_late_targets=int(validation.get("minimum_late_targets", 6)),
        minimum_rescue_frames=int(validation.get("minimum_rescue_frames_per_phase", 2)),
        minimum_row_shift=int(validation.get("minimum_directional_row_shift", 2)),
    )
    if scene_health["passed"]:
        verdict, reason = _judge(
            informative_fraction,
            relative["per_frame_over_best_fixed"],
            relative["speed_constrained_over_best_fixed"],
            int(per_frame["switch_count"]),
        )
    else:
        verdict = "SCENE_NOT_INFORMATIVE"
        reason = "; ".join(scene_health["failures"])
    payload: dict[str, Any] = {
        "verdict": verdict,
        "reason": reason,
        "scope": "geometry_only_lidar_support_proxy_not_AP",
        "keyframe_count": int(scores.shape[0]),
        "candidate_count": int(scores.shape[1]),
        "time_s": elapsed.tolist(),
        "informative_frame_count": int(informative.sum()),
        "informative_frame_fraction": informative_fraction,
        "median_score_span": float(np.median(spans)),
        "mean_score_span": float(np.mean(spans)),
        "scene_health": scene_health,
        "target_count_per_frame": target_counts.tolist(),
        "base_missed_count_per_frame": (target_counts - base_visible_counts).tolist(),
        "max_binary_uav_gain_per_frame": max_binary_gain.tolist(),
        "relative_gaps": relative,
        "strategies": {
            name: {
                "mean_score": float(item["mean_score"]),
                "perception_reward": float(item["perception_reward"]),
                "movement_m": float(item["movement_m"]),
                "switch_count": int(item["switch_count"]),
                "path_names": item["path_names"],
            }
            for name, item in (
                ("center_fixed", center),
                ("best_fixed", fixed),
                ("per_frame_unconstrained", per_frame),
                ("speed_constrained_center_start", constrained),
                ("speed_constrained_best_start", constrained_best_start),
            )
        },
        "decision_rule": {
            "scene_not_informative": "late traffic, two-phase UAV rescue, or directional-shift checks fail",
            "unsupported": "informative frames <25%, oracle changes <2, or per-frame upper bound <= best fixed by 1%",
            "weak": "dynamic upper bound exists but physically constrained path does not beat best fixed",
            "promising": "constrained path beats best fixed and per-frame upper bound is at least 3%",
            "strong": "promising plus constrained gain >=3% and informative frames >=50%",
        },
        "next_step": _next_step(verdict),
    }
    json_path = run_dir / "idea_report.json"
    text_path = run_dir / "idea_report.txt"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    text_path.write_text(_render_text(payload), encoding="utf-8")
    print(text_path.read_text(encoding="utf-8"))
    print(f"Saved: {json_path}")
    print(f"Saved: {text_path}")
    return json_path, text_path


def _relative(value: float, reference: float) -> float | None:
    if abs(reference) < 1e-12:
        return None
    return float((value - reference) / reference)


def _judge(
    informative_fraction: float,
    per_frame_over_fixed: float | None,
    constrained_over_fixed: float | None,
    oracle_switches: int,
) -> tuple[str, str]:
    upper_gap = per_frame_over_fixed if per_frame_over_fixed is not None else -np.inf
    feasible_gap = constrained_over_fixed if constrained_over_fixed is not None else -np.inf
    if informative_fraction < 0.25:
        return "UNSUPPORTED", "Most candidate positions are effectively tied."
    if oracle_switches < 2 or upper_gap < 0.01:
        return "UNSUPPORTED", "A dynamic view has less than 1% advantage over the best fixed view."
    if feasible_gap <= 0.0 or upper_gap < 0.03:
        return "WEAK", "A dynamic upper bound exists, but the feasible path does not clearly beat best fixed."
    if feasible_gap >= 0.03 and informative_fraction >= 0.50:
        return "STRONG_PROXY_SIGNAL", "A speed-limited moving view beats best fixed by at least 3%."
    return "PROMISING_PROXY_SIGNAL", "A speed-limited moving view beats best fixed; verify with detector AP/recall."


def _scene_health(
    scores: np.ndarray,
    elapsed_s: np.ndarray,
    target_counts: np.ndarray,
    base_visible_counts: np.ndarray,
    max_binary_gain: np.ndarray,
    early_end_s: float,
    late_start_s: float,
    minimum_late_targets: int,
    minimum_rescue_frames: int,
    minimum_row_shift: int,
) -> dict[str, Any]:
    position_count = int(scores.shape[1])
    grid_size = int(round(np.sqrt(position_count)))
    if grid_size * grid_size != position_count:
        raise ValueError("scene-health check expects a square UAV candidate grid")
    best_rows = np.argmax(scores, axis=1) // grid_size
    early = elapsed_s <= float(early_end_s) + 1e-6
    late = elapsed_s >= float(late_start_s) - 1e-6
    if not early.any() or not late.any():
        raise ValueError("early/late validation windows contain no keyframes")
    early_rescue_frames = int(np.count_nonzero(max_binary_gain[early] > 1e-8))
    late_rescue_frames = int(np.count_nonzero(max_binary_gain[late] > 1e-8))
    early_row = float(np.median(best_rows[early]))
    late_row = float(np.median(best_rows[late]))
    row_shift = float(abs(late_row - early_row))
    late_target_min = int(target_counts[late].min())
    failures: list[str] = []
    if late_target_min < int(minimum_late_targets):
        failures.append(
            f"late target count drops to {late_target_min} (<{int(minimum_late_targets)})"
        )
    if early_rescue_frames < int(minimum_rescue_frames):
        failures.append(
            f"UAV rescues targets in only {early_rescue_frames} early frames"
        )
    if late_rescue_frames < int(minimum_rescue_frames):
        failures.append(
            f"UAV rescues targets in only {late_rescue_frames} late frames"
        )
    if row_shift < float(minimum_row_shift):
        failures.append(
            f"median best-row shift is {row_shift:.1f} (<{int(minimum_row_shift)})"
        )
    return {
        "passed": not failures,
        "failures": failures,
        "early_window_s": [float(elapsed_s[early][0]), float(elapsed_s[early][-1])],
        "late_window_s": [float(elapsed_s[late][0]), float(elapsed_s[late][-1])],
        "late_target_min": late_target_min,
        "early_uav_rescue_frame_count": early_rescue_frames,
        "late_uav_rescue_frame_count": late_rescue_frames,
        "early_median_best_row": early_row,
        "late_median_best_row": late_row,
        "median_best_row_shift": row_shift,
        "best_row_per_frame": best_rows.astype(int).tolist(),
        "base_missed_count_per_frame": (target_counts - base_visible_counts).astype(int).tolist(),
    }


def _next_step(verdict: str) -> str:
    if verdict == "SCENE_NOT_INFORMATIVE":
        return "Repair traffic timing/route coverage before drawing a conclusion about active movement."
    if verdict == "UNSUPPORTED":
        return "Do not train a detector yet; inspect score saturation and scene/agent coverage."
    if verdict == "WEAK":
        return "Run 3-5 seeds and one sensor-density ablation before detector training."
    return "Repeat across seeds, then replace the proxy with a frozen detector and report AP/recall."


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:+.2f}%"


def _render_text(payload: dict[str, Any]) -> str:
    gaps = payload["relative_gaps"]
    strategies = payload["strategies"]
    lines = [
        "Active AirV2X 30-second idea check",
        "===================================",
        f"Verdict: {payload['verdict']}",
        f"Reason: {payload['reason']}",
        "",
        f"Frames / candidates: {payload['keyframe_count']} / {payload['candidate_count']}",
        f"Informative frames: {payload['informative_frame_count']} "
        f"({100.0 * payload['informative_frame_fraction']:.1f}%)",
        f"Median candidate score span: {payload['median_score_span']:.4f}",
        "",
        "Scene-health gate:",
        f"  passed: {payload['scene_health']['passed']}",
        f"  late minimum targets: {payload['scene_health']['late_target_min']}",
        f"  UAV rescue frames (early / late): "
        f"{payload['scene_health']['early_uav_rescue_frame_count']} / "
        f"{payload['scene_health']['late_uav_rescue_frame_count']}",
        f"  median best-row shift: {payload['scene_health']['median_best_row_shift']:.1f}",
        f"  target counts: {payload['target_count_per_frame']}",
        f"  base missed counts: {payload['base_missed_count_per_frame']}",
        "",
        "Relative perception-reward gaps:",
        f"  best fixed vs center fixed:          {_pct(gaps['best_fixed_over_center'])}",
        f"  per-frame upper bound vs best fixed: {_pct(gaps['per_frame_over_best_fixed'])}",
        f"  speed-limited vs best fixed:         {_pct(gaps['speed_constrained_over_best_fixed'])}",
        f"  speed-limited vs center fixed:       {_pct(gaps['speed_constrained_over_center'])}",
        "",
        "Strategy summary:",
    ]
    for name, item in strategies.items():
        lines.append(
            f"  {name:36s} mean={item['mean_score']:.4f} "
            f"move={item['movement_m']:.1f}m switches={item['switch_count']}"
        )
    lines.extend(
        [
            "",
            "Important: this is a LiDAR box-support proxy, not detector AP.",
            f"Next: {payload['next_step']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge the 30-second active-view proxy signal")
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    build_report(args.run_dir)


if __name__ == "__main__":
    main()
