from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config
from .oracle import (
    PathSolution,
    best_fixed,
    closest_to_origin,
    constrained_oracle,
    per_frame_oracle,
    switch_count,
)
from .uav_modes import (
    DecisionWindows,
    aggregate_scores_by_window,
    build_decision_windows,
    expand_window_path,
)


def solve(
    run_dir: str | Path,
    *,
    evaluation_dir: str | Path | None = None,
    movement_penalty_per_m: float | None = None,
    output_name: str = "oracle_results.json",
    score_key: str = "continuous",
) -> Path:
    """Solve oracle paths for one score matrix.

    ``evaluation_dir`` may point at an ablation subdirectory containing its own
    ``geometry_scores.npz``.  The raw metadata/config are always read from
    ``run_dir``.  Existing callers retain the original top-level behaviour.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    evaluation_dir = (
        run_dir
        if evaluation_dir is None
        else Path(evaluation_dir).expanduser().resolve()
    )
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    cfg = load_config(run_dir / "effective_config.yaml")
    arrays = np.load(evaluation_dir / "geometry_scores.npz", allow_pickle=False)
    scores, score_type = _score_matrix(arrays, cfg, score_key)
    names = arrays["candidate_names"].astype(str).tolist()
    elapsed_s = np.asarray(arrays["elapsed_s"], dtype=np.float64)
    score_span = scores.max(axis=1) - scores.min(axis=1)
    near_best_count = np.sum(scores >= scores.max(axis=1, keepdims=True) - 1e-5, axis=1)
    coords = np.asarray(
        [[item["offset_forward_m"], item["offset_right_m"]] for item in metadata["grid"]],
        dtype=np.float64,
    )
    center = closest_to_origin(coords)
    center_path = np.full(scores.shape[0], center, dtype=np.int64)
    center_solution = PathSolution(
        center_path,
        float(scores[:, center].sum()),
        float(scores[:, center].sum()),
        0.0,
    )
    frame_dt = float(metadata.get("frame_interval_s", metadata["keyframe_interval_s"]))
    decision_dt = float(metadata.get("decision_interval_s", metadata["keyframe_interval_s"]))
    speed = float(cfg["oracle"]["uav_speed_mps"])
    penalty = (
        float(cfg["oracle"]["movement_penalty_per_m"])
        if movement_penalty_per_m is None
        else float(movement_penalty_per_m)
    )
    if penalty < 0:
        raise ValueError("movement_penalty_per_m must be non-negative")
    windows = build_decision_windows(elapsed_s, decision_dt)
    decision_scores = aggregate_scores_by_window(scores, windows)
    center_decision = constrained_oracle(
        decision_scores, coords, decision_dt, speed, penalty, start_index=center
    )
    best_start_decision = constrained_oracle(
        decision_scores, coords, decision_dt, speed, penalty
    )
    solutions = {
        "center_fixed": center_solution,
        "best_fixed": best_fixed(scores),
        "per_frame_unconstrained": per_frame_oracle(scores, coords),
        "speed_constrained_center_start": _expanded_decision_solution(
            center_decision, windows, scores
        ),
        "speed_constrained_best_start": _expanded_decision_solution(
            best_start_decision, windows, scores
        ),
    }
    result = {
        "score_type": score_type,
        "score_key": score_key,
        "keyframe_interval_s": frame_dt,
        "frame_interval_s": frame_dt,
        "decision_interval_s": decision_dt,
        "decision_count": int(len(decision_scores)),
        "uav_speed_mps": speed,
        "max_transition_m": decision_dt * speed,
        "movement_penalty_per_m": penalty,
        "time_s": elapsed_s.tolist(),
        "diagnostics": {
            "score_span_per_frame": score_span.tolist(),
            "near_best_position_count_per_frame": near_best_count.tolist(),
            "uninformative_frame_count": int(np.count_nonzero(score_span < 1e-5)),
        },
        "solutions": {
            key: _solution_record(value, names, coords, scores, center_solution.perception_reward)
            for key, value in solutions.items()
        },
    }
    output = evaluation_dir / output_name
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, solution in solutions.items():
        print(
            f"{name:32s} reward={solution.perception_reward:.3f} "
            f"move={solution.movement_m:5.1f}m switches={switch_count(solution.indices)}"
        )
    if np.all(score_span < 1e-5):
        print("WARNING: all candidate positions are tied on every frame; this run cannot support active movement.")
    elif np.median(near_best_count) > max(2, scores.shape[1] // 2):
        print("WARNING: most positions are near-tied; inspect scene geometry and score saturation.")
    print(f"Saved: {output}")
    return output


def _expanded_decision_solution(
    decision_solution: PathSolution,
    windows: DecisionWindows,
    frame_scores: np.ndarray,
) -> PathSolution:
    path = expand_window_path(decision_solution.indices, windows)
    reward = float(frame_scores[np.arange(len(path)), path].sum())
    return PathSolution(
        indices=path,
        objective=float(decision_solution.objective),
        perception_reward=reward,
        movement_m=float(decision_solution.movement_m),
    )


def _score_matrix(
    arrays: Any,
    cfg: dict[str, Any],
    score_key: str,
) -> tuple[np.ndarray, str]:
    if score_key == "continuous":
        return (
            np.asarray(arrays["scores"], dtype=np.float64),
            "continuous_lidar_support_proxy_not_AP",
        )
    if score_key == "recovery":
        tie_break = float(cfg["oracle"].get("recovery_quality_tiebreak", 0.0001))
        if not 0.0 <= tie_break < 0.01:
            raise ValueError("oracle.recovery_quality_tiebreak must be in [0, 0.01)")
        scores = np.asarray(arrays["rescued_weight"], dtype=np.float64).copy()
        scores += tie_break * np.asarray(arrays["quality_gain"], dtype=np.float64)
        return scores, "rescued_target_weight_then_quality_tiebreak"
    raise ValueError("score_key must be continuous or recovery")


def _solution_record(
    solution: PathSolution,
    names: list[str],
    coords: np.ndarray,
    scores: np.ndarray,
    center_reward: float,
) -> dict[str, Any]:
    path_scores = scores[np.arange(scores.shape[0]), solution.indices]
    improvement = (
        (solution.perception_reward - center_reward) / center_reward if abs(center_reward) > 1e-12 else None
    )
    return {
        "path_indices": solution.indices.tolist(),
        "path_names": [names[index] for index in solution.indices],
        "path_offsets_m": coords[solution.indices].tolist(),
        "per_frame_scores": path_scores.tolist(),
        "objective": solution.objective,
        "perception_reward": solution.perception_reward,
        "mean_score": float(path_scores.mean()),
        "movement_m": solution.movement_m,
        "switch_count": switch_count(solution.indices),
        "relative_improvement_over_center": improvement,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve fixed and speed-constrained UAV oracles")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--score-key", choices=("continuous", "recovery"), default="continuous")
    parser.add_argument("--output-name", default="oracle_results.json")
    args = parser.parse_args()
    solve(args.run_dir, score_key=args.score_key, output_name=args.output_name)


if __name__ == "__main__":
    main()
