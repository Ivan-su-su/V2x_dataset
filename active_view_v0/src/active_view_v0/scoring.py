from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config
from .geometry import points_in_oriented_box, transform_points


def score_dataset(
    run_dir: str | Path,
    *,
    base_sensors: list[str] | tuple[str, ...] | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    """Score every UAV candidate against one selected ground-agent base.

    ``base_sensors`` and ``output_dir`` are optional so the historical command
    keeps writing the usual top-level files.  Base ablations pass both options
    and therefore reuse the raw frames without overwriting the canonical v0.7
    evaluation.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    output_dir = (
        run_dir if output_dir is None else Path(output_dir).expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = _read_json(run_dir / "metadata.json")
    cfg = load_config(run_dir / "effective_config.yaml")
    scoring_cfg = cfg["scoring"]
    candidate_names = list(metadata["candidate_sensor_names"])
    base_names = (
        list(scoring_cfg["base_sensors"])
        if base_sensors is None
        else [str(name) for name in base_sensors]
    )
    if not base_names:
        raise ValueError("base_sensors must contain at least one sensor")
    manifest = [json.loads(line) for line in (run_dir / "manifest.jsonl").read_text().splitlines() if line]
    if not manifest:
        raise RuntimeError(f"no keyframes found in {run_dir}")

    scores = np.zeros((len(manifest), len(candidate_names)), dtype=np.float32)
    coverage = np.zeros_like(scores)
    gain = np.zeros_like(scores)
    quality_coverage = np.zeros_like(scores)
    quality_gain = np.zeros_like(scores)
    rescued_counts = np.zeros_like(scores, dtype=np.int32)
    rescued_weight = np.zeros_like(scores)
    union_visible_counts = np.zeros_like(scores, dtype=np.int32)
    base_visible_counts = np.zeros(len(manifest), dtype=np.int32)
    target_counts = np.zeros(len(manifest), dtype=np.int32)
    details: list[dict[str, Any]] = []
    ignore_roles = set(scoring_cfg["ignore_roles"])
    thresholds = {key: int(value) for key, value in scoring_cfg["threshold_points"].items()}
    class_weights = {key: float(value) for key, value in scoring_cfg["class_weights"].items()}
    margin = float(scoring_cfg["box_margin_m"])
    gain_weight = float(scoring_cfg["gain_weight"])
    quality_tau = {key: float(value) for key, value in scoring_cfg["quality_tau_points"].items()}

    for time_index, item in enumerate(manifest):
        frame_dir = run_dir / item["path"]
        frame = _read_json(frame_dir / "frame.json")
        targets = [
            actor
            for actor in frame["gt"]
            if actor["role_name"] not in ignore_roles and actor["class_name"] in thresholds
        ]
        target_counts[time_index] = len(targets)
        weights = np.asarray([class_weights[target["class_name"]] for target in targets], dtype=np.float64)
        denominator = float(weights.sum()) if len(weights) else 1.0
        base_support = np.zeros(len(targets), dtype=np.int32)
        base_sensor_support: dict[str, list[int]] = {}
        for sensor_name in base_names:
            points_world = _load_world_points(frame_dir, frame, sensor_name)
            sensor_support = _support_counts(points_world, targets, margin)
            base_support += sensor_support
            base_sensor_support[sensor_name] = sensor_support.tolist()
        thresholds_vector = np.asarray([thresholds[target["class_name"]] for target in targets])
        base_visible = base_support >= thresholds_vector
        tau_vector = np.asarray([quality_tau[target["class_name"]] for target in targets])
        base_quality = 1.0 - np.exp(-base_support / tau_vector) if len(targets) else np.asarray([])
        base_visible_counts[time_index] = int(base_visible.sum())

        candidate_details: dict[str, Any] = {}
        for position_index, sensor_name in enumerate(candidate_names):
            points_world = _load_world_points(frame_dir, frame, sensor_name)
            candidate_support = _support_counts(points_world, targets, margin)
            candidate_visible = candidate_support >= thresholds_vector
            candidate_quality = (
                1.0 - np.exp(-candidate_support / tau_vector) if len(targets) else np.asarray([])
            )
            union_support = base_support + candidate_support
            union_visible = union_support >= thresholds_vector
            rescued = (~base_visible) & union_visible
            coverage_value = float(weights[union_visible].sum() / denominator) if len(targets) else 0.0
            gain_value = float(weights[rescued].sum() / denominator) if len(targets) else 0.0
            union_quality = (
                1.0 - np.exp(-union_support / tau_vector) if len(targets) else np.asarray([])
            )
            rescued_quality = np.maximum(union_quality - base_quality, 0.0)
            quality_coverage_value = (
                float((weights * union_quality).sum() / denominator) if len(targets) else 0.0
            )
            quality_gain_value = (
                float((weights * rescued_quality).sum() / denominator) if len(targets) else 0.0
            )
            score = quality_coverage_value + gain_weight * quality_gain_value
            coverage[time_index, position_index] = coverage_value
            gain[time_index, position_index] = gain_value
            quality_coverage[time_index, position_index] = quality_coverage_value
            quality_gain[time_index, position_index] = quality_gain_value
            rescued_counts[time_index, position_index] = int(rescued.sum())
            rescued_weight[time_index, position_index] = float(weights[rescued].sum())
            union_visible_counts[time_index, position_index] = int(union_visible.sum())
            scores[time_index, position_index] = score
            candidate_details[sensor_name] = {
                "support_points": candidate_support.tolist(),
                "union_support_points": union_support.tolist(),
                "visible": candidate_visible.astype(int).tolist(),
                "union_visible": union_visible.astype(int).tolist(),
                "coverage": coverage_value,
                "gain": gain_value,
                "quality_coverage": quality_coverage_value,
                "quality_gain": quality_gain_value,
                "rescued_count": int(rescued.sum()),
                "rescued_weight": float(weights[rescued].sum()),
                "score": score,
            }
        details.append(
            {
                "keyframe_index": int(item["keyframe_index"]),
                "elapsed_s": float(item["elapsed_s"]),
                "target_ids": [int(target["actor_id"]) for target in targets],
                "target_classes": [target["class_name"] for target in targets],
                "target_roles": [target["role_name"] for target in targets],
                "base_support_points": base_support.tolist(),
                "base_sensor_support_points": base_sensor_support,
                "base_visible": base_visible.astype(int).tolist(),
                "base_quality": base_quality.tolist(),
                "target_centers_world": [
                    target.get("bbox_center_world", target["actor_transform"]["location"])
                    for target in targets
                ],
                "target_extents": [target["bbox_extent"] for target in targets],
                "candidates": candidate_details,
            }
        )
        best = int(np.argmax(scores[time_index]))
        print(
            f"[{time_index + 1:02d}/{len(manifest):02d}] t={item['elapsed_s']:5.1f}s "
            f"targets={len(targets)} best={candidate_names[best]} score={scores[time_index, best]:.3f}"
        )

    output_path = output_dir / "geometry_scores.npz"
    np.savez_compressed(
        output_path,
        scores=scores,
        coverage=coverage,
        gain=gain,
        quality_coverage=quality_coverage,
        quality_gain=quality_gain,
        rescued_counts=rescued_counts,
        rescued_weight=rescued_weight,
        union_visible_counts=union_visible_counts,
        candidate_names=np.asarray(candidate_names),
        frame_ids=np.asarray([item["carla_frame"] for item in manifest], dtype=np.int64),
        elapsed_s=np.asarray([item["elapsed_s"] for item in manifest], dtype=np.float64),
        target_counts=target_counts,
        base_visible_counts=base_visible_counts,
        score_span=scores.max(axis=1) - scores.min(axis=1),
        near_best_count=np.sum(scores >= scores.max(axis=1, keepdims=True) - 1e-5, axis=1),
    )
    (output_dir / "geometry_details.json").write_text(
        json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "scoring_profile.json").write_text(
        json.dumps(
            {
                "base_sensors": base_names,
                "candidate_sensor_count": len(candidate_names),
                "raw_run_dir": str(run_dir),
                "score_type": "continuous_lidar_support_proxy_not_AP",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Saved: {output_path}")
    return output_path


def _load_world_points(frame_dir: Path, frame: dict[str, Any], sensor_name: str) -> np.ndarray:
    sensor = frame["sensors"].get(sensor_name)
    if sensor is None:
        raise KeyError(f"frame {frame['keyframe_index']} has no sensor named {sensor_name}")
    points = np.fromfile(frame_dir / sensor["path"], dtype=np.float32)
    if points.size % 4 != 0:
        raise ValueError(f"invalid CARLA LiDAR binary: {frame_dir / sensor['path']}")
    points = points.reshape(-1, 4)
    return transform_points(points[:, :3], np.asarray(sensor["transform"]["matrix"], dtype=np.float64))


def _support_counts(
    points_world: np.ndarray, targets: list[dict[str, Any]], margin: float
) -> np.ndarray:
    counts = np.zeros(len(targets), dtype=np.int32)
    for index, target in enumerate(targets):
        mask = points_in_oriented_box(
            points_world,
            np.asarray(target["actor_transform"]["matrix"], dtype=np.float64),
            target["bbox_center_local"],
            target["bbox_extent"],
            margin=margin,
        )
        counts[index] = int(mask.sum())
    return counts


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute geometry-only visibility scores")
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    score_dataset(args.run_dir)


if __name__ == "__main__":
    main()
