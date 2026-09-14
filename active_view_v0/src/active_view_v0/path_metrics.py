from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def evaluate_path_metrics(
    run_dir: str | Path,
    *,
    oracle_name: str = "oracle_results.json",
    output_stem: str = "path_binary_metrics",
) -> tuple[Path, Path]:
    """Evaluate actual threshold crossings along each selected UAV path.

    These metrics complement the continuous point-support reward.  A target is
    rescued only when the selected Base+UAV union crosses its class-specific
    LiDAR support threshold while the Base alone does not.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    arrays = np.load(run_dir / "geometry_scores.npz", allow_pickle=False)
    oracle = json.loads((run_dir / oracle_name).read_text(encoding="utf-8"))
    details = json.loads((run_dir / "geometry_details.json").read_text(encoding="utf-8"))
    elapsed = np.asarray(arrays["elapsed_s"], dtype=np.float64)
    coverage = np.asarray(arrays["coverage"], dtype=np.float64)
    gain = np.asarray(arrays["gain"], dtype=np.float64)
    candidate_names = arrays["candidate_names"].astype(str).tolist()
    if len(details) != len(elapsed):
        raise ValueError("geometry_details length does not match geometry_scores")

    strategy_records: dict[str, Any] = {}
    for strategy_name, solution in oracle["solutions"].items():
        indices = np.asarray(solution["path_indices"], dtype=np.int64)
        if indices.shape != (len(elapsed),):
            raise ValueError(f"invalid path length for {strategy_name}")
        strategy_records[strategy_name] = _evaluate_strategy(
            details,
            elapsed,
            indices,
            candidate_names,
            coverage,
            gain,
            solution,
        )

    fixed = strategy_records["best_fixed"]
    center_dynamic = strategy_records["speed_constrained_center_start"]
    best_start_dynamic = strategy_records["speed_constrained_best_start"]
    payload = {
        "scope": "binary_lidar_support_proxy_not_detector_AP",
        "path_objective": oracle.get("score_type", "unknown"),
        "definition": (
            "rescued = Base is below the class point threshold and the selected "
            "Base+UAV union reaches it"
        ),
        "strategy_metrics": strategy_records,
        "dynamic_vs_best_fixed": {
            "center_start": _compare(center_dynamic, fixed),
            "best_start": _compare(best_start_dynamic, fixed),
        },
    }
    json_path = run_dir / f"{output_stem}.json"
    text_path = run_dir / f"{output_stem}.txt"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    text_path.write_text(_render_text(payload), encoding="utf-8")
    print(text_path.read_text(encoding="utf-8"))
    print(f"Saved: {json_path}")
    print(f"Saved: {text_path}")
    return json_path, text_path


def _evaluate_strategy(
    details: list[dict[str, Any]],
    elapsed: np.ndarray,
    path_indices: np.ndarray,
    candidate_names: list[str],
    coverage: np.ndarray,
    gain: np.ndarray,
    solution: dict[str, Any],
) -> dict[str, Any]:
    total_targets = 0
    base_detected = 0
    union_detected = 0
    base_missed = 0
    rescued = 0
    class_totals: dict[str, dict[str, int]] = {}
    frames: list[dict[str, Any]] = []
    for time_index, detail in enumerate(details):
        candidate_index = int(path_indices[time_index])
        candidate_name = candidate_names[candidate_index]
        candidate = detail["candidates"][candidate_name]
        base_visible = np.asarray(detail["base_visible"], dtype=bool)
        union_visible = np.asarray(candidate["union_visible"], dtype=bool)
        rescued_mask = (~base_visible) & union_visible
        missed_mask = ~base_visible
        target_count = len(base_visible)
        total_targets += target_count
        base_detected += int(base_visible.sum())
        union_detected += int(union_visible.sum())
        base_missed += int(missed_mask.sum())
        rescued += int(rescued_mask.sum())
        for index, class_name in enumerate(detail["target_classes"]):
            record = class_totals.setdefault(
                str(class_name), {"targets": 0, "base_missed": 0, "rescued": 0}
            )
            record["targets"] += 1
            record["base_missed"] += int(missed_mask[index])
            record["rescued"] += int(rescued_mask[index])
        frames.append(
            {
                "elapsed_s": float(elapsed[time_index]),
                "candidate": candidate_name,
                "target_count": target_count,
                "base_missed": int(missed_mask.sum()),
                "rescued": int(rescued_mask.sum()),
                "union_detected": int(union_visible.sum()),
                "rescued_target_ids": [
                    int(detail["target_ids"][index]) for index in np.flatnonzero(rescued_mask)
                ],
            }
        )
    for record in class_totals.values():
        record["miss_recovery_rate"] = (
            float(record["rescued"] / record["base_missed"])
            if record["base_missed"]
            else None
        )
    return {
        "target_instances": total_targets,
        "base_detected_instances": base_detected,
        "base_missed_instances": base_missed,
        "union_detected_instances": union_detected,
        "rescued_instances": rescued,
        "target_recall": float(union_detected / total_targets) if total_targets else None,
        "miss_recovery_rate": float(rescued / base_missed) if base_missed else None,
        "mean_weighted_coverage": float(
            coverage[np.arange(len(elapsed)), path_indices].mean()
        ),
        "mean_weighted_gain": float(gain[np.arange(len(elapsed)), path_indices].mean()),
        "movement_m": float(solution["movement_m"]),
        "switch_count": int(solution["switch_count"]),
        "path_names": [candidate_names[int(index)] for index in path_indices],
        "class_metrics": class_totals,
        "frames": frames,
    }


def _compare(dynamic: dict[str, Any], fixed: dict[str, Any]) -> dict[str, Any]:
    return {
        "additional_rescued_instances": int(
            dynamic["rescued_instances"] - fixed["rescued_instances"]
        ),
        "target_recall_gain_pp": _difference_pp(
            dynamic["target_recall"], fixed["target_recall"]
        ),
        "miss_recovery_rate_gain_pp": _difference_pp(
            dynamic["miss_recovery_rate"], fixed["miss_recovery_rate"]
        ),
        "weighted_coverage_gain_pp": 100.0
        * (dynamic["mean_weighted_coverage"] - fixed["mean_weighted_coverage"]),
        "weighted_gain_gain_pp": 100.0
        * (dynamic["mean_weighted_gain"] - fixed["mean_weighted_gain"]),
        "movement_m": dynamic["movement_m"],
        "switch_count": dynamic["switch_count"],
    }


def _difference_pp(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None:
        return None
    return 100.0 * (value - reference)


def _fmt(value: float | None, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:+.2f}{suffix}"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def _render_text(payload: dict[str, Any]) -> str:
    metrics = payload["strategy_metrics"]
    comparisons = payload["dynamic_vs_best_fixed"]
    lines = [
        "Active AirV2X path-level binary support check",
        "================================================",
        "A rescue requires an actual class-threshold crossing, not merely denser points.",
        "",
        "Strategy summary:",
    ]
    for name in (
        "center_fixed",
        "best_fixed",
        "per_frame_unconstrained",
        "speed_constrained_center_start",
        "speed_constrained_best_start",
    ):
        item = metrics[name]
        lines.append(
            f"  {name:36s} recall={_pct(item['target_recall'])} "
            f"rescued={item['rescued_instances']}/{item['base_missed_instances']} "
            f"MRR={_pct(item['miss_recovery_rate'])} "
            f"move={item['movement_m']:.1f}m"
        )
    lines.extend(["", "Dynamic path vs best fixed:"])
    for name in ("center_start", "best_start"):
        item = comparisons[name]
        lines.append(
            f"  {name:14s} extra_rescued={item['additional_rescued_instances']:+d} "
            f"recall={_fmt(item['target_recall_gain_pp'], 'pp')} "
            f"MRR={_fmt(item['miss_recovery_rate_gain_pp'], 'pp')}"
        )
    lines.extend(
        [
            "",
            "Important: threshold support is still a geometry proxy; detector AP/AR is the next stage.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate binary target recovery along UAV paths")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--oracle-name", default="oracle_results.json")
    parser.add_argument("--output-stem", default="path_binary_metrics")
    args = parser.parse_args()
    evaluate_path_metrics(
        args.run_dir,
        oracle_name=args.oracle_name,
        output_stem=args.output_stem,
    )


if __name__ == "__main__":
    main()
