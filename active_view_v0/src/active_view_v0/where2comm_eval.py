from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import load_config
from .oracle import constrained_oracle, per_frame_oracle
from .uav_modes import (
    aggregate_scores_by_window,
    build_decision_windows,
    expand_window_path,
)


BASE_VARIANT = "__base_only__"
PREDICTION_FORMAT = "active-view-where2comm-prediction-v1"


@dataclass(frozen=True)
class FrameInput:
    index: int
    elapsed_s: float
    frame_dir: Path
    frame: dict[str, Any]


def carla_points_to_ego(
    points: np.ndarray,
    sensor_to_world: np.ndarray,
    ego_sensor_to_world: np.ndarray,
) -> np.ndarray:
    """Transform CARLA LiDAR points into OpenCOOD's Ego-LiDAR frame.

    CARLA uses x-forward/y-right/z-up.  OpenCOOD/AirV2X uses
    x-forward/y-left/z-up, hence the final y-axis sign change.
    """
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError("CARLA LiDAR points must have shape [N, 4]")
    sensor_to_world = _matrix4(sensor_to_world, "sensor_to_world")
    ego_sensor_to_world = _matrix4(ego_sensor_to_world, "ego_sensor_to_world")
    xyz1 = np.concatenate(
        [points[:, :3].astype(np.float64), np.ones((len(points), 1), dtype=np.float64)],
        axis=1,
    )
    sensor_to_ego = np.linalg.solve(ego_sensor_to_world, sensor_to_world)
    xyz_ego_carla = (sensor_to_ego @ xyz1.T).T[:, :3]
    xyz_ego_carla[:, 1] *= -1.0
    return np.column_stack([xyz_ego_carla, points[:, 3]]).astype(np.float32)


def gt_boxes_in_ego(
    frame: dict[str, Any],
    ego_sensor_to_world: np.ndarray,
    *,
    ignored_roles: Iterable[str],
    accepted_classes: Iterable[str],
    lidar_range: Iterable[float],
) -> tuple[np.ndarray, list[int], list[str]]:
    """Return CARLA GT as OpenCOOD hwl boxes in the Ego-LiDAR frame."""
    ignored = set(ignored_roles)
    accepted = set(accepted_classes)
    limits = np.asarray(list(lidar_range), dtype=np.float64)
    if limits.shape != (6,):
        raise ValueError("lidar_range must contain six values")
    world_to_ego = np.linalg.inv(_matrix4(ego_sensor_to_world, "ego_sensor_to_world"))
    boxes: list[list[float]] = []
    actor_ids: list[int] = []
    class_names: list[str] = []
    for actor in frame.get("gt", []):
        class_name = str(actor.get("class_name", ""))
        if actor.get("role_name", "") in ignored or class_name not in accepted:
            continue
        actor_to_world = _matrix4(actor["actor_transform"]["matrix"], "actor_to_world")
        center_local = np.asarray(actor["bbox_center_local"], dtype=np.float64)
        center_world = actor_to_world @ np.r_[center_local, 1.0]
        center_ego = (world_to_ego @ center_world)[:3]
        center_ego[1] *= -1.0
        if not (
            limits[0] <= center_ego[0] <= limits[3]
            and limits[1] <= center_ego[1] <= limits[4]
            and limits[2] - 5.0 <= center_ego[2] <= limits[5] + 5.0
        ):
            continue

        forward_world = actor_to_world[:3, 0]
        forward_ego = world_to_ego[:3, :3] @ forward_world
        yaw = math.atan2(-float(forward_ego[1]), float(forward_ego[0]))
        extent = np.asarray(actor["bbox_extent"], dtype=np.float64)
        length, width, height = (2.0 * extent).tolist()
        boxes.append(
            [
                float(center_ego[0]),
                float(center_ego[1]),
                float(center_ego[2]),
                float(height),
                float(width),
                float(length),
                float(yaw),
            ]
        )
        actor_ids.append(int(actor["actor_id"]))
        class_names.append(class_name)
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 7), actor_ids, class_names


def boxes_hwl_to_corners(boxes: np.ndarray) -> np.ndarray:
    """Convert [x,y,z,h,w,l,yaw] boxes to [N,8,3] corners."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 7)
    if not len(boxes):
        return np.zeros((0, 8, 3), dtype=np.float32)
    corners = np.zeros((len(boxes), 8, 3), dtype=np.float32)
    signs = np.asarray(
        [
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, -1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
            [-1, -1, 1],
        ],
        dtype=np.float32,
    )
    for index, (x, y, z, height, width, length, yaw) in enumerate(boxes):
        local = signs * np.asarray([length / 2, width / 2, height / 2], dtype=np.float32)
        cosine, sine = math.cos(float(yaw)), math.sin(float(yaw))
        rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float32)
        local[:, :2] = local[:, :2] @ rotation.T
        local += np.asarray([x, y, z], dtype=np.float32)
        corners[index] = local
    return corners


class Where2commRuntime:
    """Thin runtime around the Airv2x_gs Where2comm model.

    This deliberately bypasses the AirV2X disk dataset loader.  That loader
    unconditionally opens RGB, depth and segmentation files even for a
    LiDAR-only checkpoint, while this benchmark already has synchronized raw
    LiDAR and exact sensor poses.
    """

    def __init__(
        self,
        *,
        opencood_root: Path,
        config_path: Path,
        checkpoint_path: Path,
        device: str,
        allow_partial_checkpoint: bool = False,
        verbose_model: bool = False,
    ) -> None:
        opencood_root = opencood_root.expanduser().resolve()
        if not (opencood_root / "opencood").is_dir():
            raise FileNotFoundError(
                f"{opencood_root} does not contain the opencood package; "
                "pass the root of Ivan-su-su/Airv2x_gs"
            )
        sys.path.insert(0, str(opencood_root))
        import torch
        from opencood.data_utils import post_processor
        from opencood.data_utils.pre_processor import build_preprocessor
        from opencood.hypes_yaml import yaml_utils
        from opencood.tools import train_utils

        self.torch = torch
        self.device = torch.device(device)
        self.verbose_model = bool(verbose_model)
        self.hypes = yaml_utils.load_yaml(str(config_path.expanduser().resolve()))
        self._validate_config()
        self.preprocessor = build_preprocessor(self.hypes["preprocess"], train=False)
        self.postprocessor = post_processor.build_postprocessor(
            self.hypes["postprocess"], dataset="airv2x", train=False
        )
        self.model = train_utils.create_model(self.hypes).to(self.device)
        self.checkpoint_report = load_checkpoint_strict(
            self.model,
            checkpoint_path,
            torch_module=torch,
            device=self.device,
            allow_partial=allow_partial_checkpoint,
        )
        coverage = 100.0 * self.checkpoint_report["coverage_by_parameter_count"]
        print(
            "Checkpoint load OK: "
            f"{self.checkpoint_report['matched_keys']}/"
            f"{self.checkpoint_report['model_keys']} keys, "
            f"{coverage:.2f}% parameters"
        )
        self.model.eval()
        self.anchor_box = torch.from_numpy(
            np.asarray(self.postprocessor.generate_anchor_box())
        ).float()
        self.lidar_range = np.asarray(
            self.hypes["preprocess"]["cav_lidar_range"], dtype=np.float32
        )
        self.max_cav = int(self.hypes["train_params"]["max_cav_num"])

    def _validate_config(self) -> None:
        method = str(self.hypes.get("model", {}).get("core_method", ""))
        if method != "airv2x_where2com":
            raise ValueError(
                f"expected model.core_method=airv2x_where2com, found {method!r}"
            )
        sensors = list(self.hypes.get("active_sensors", []))
        if sensors != ["lidar"]:
            raise ValueError(f"this adapter requires active_sensors: [lidar], found {sensors}")
        required = {"vehicle", "rsu", "drone"}
        collaborators = set(self.hypes.get("collaborators", []))
        if not required.issubset(collaborators):
            raise ValueError(
                "checkpoint config must contain vehicle, rsu and drone collaborators"
            )
        self.hypes["train"] = False
        self.hypes["model"]["args"]["train"] = False

    def infer(self, points_by_type: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
        torch = self.torch
        cav_content: dict[str, Any] = {}
        total_agents = 0
        for agent_type in ("vehicle", "rsu", "drone"):
            clouds = points_by_type.get(agent_type, [])
            total_agents += len(clouds)
            if clouds:
                features = [self.preprocessor.preprocess(cloud) for cloud in clouds]
                merged: dict[str, list[Any]] = {}
                for feature in features:
                    for key, value in feature.items():
                        merged.setdefault(key, []).extend(
                            value if isinstance(value, list) else [value]
                        )
                collated = self.preprocessor.collate_batch(merged)
                cav_content[agent_type] = {
                    "batch_merged_lidar_features_torch": _to_device(collated, self.device),
                    "batch_merged_cam_inputs": {},
                    "record_len": torch.tensor([len(clouds)], dtype=torch.int32, device=self.device),
                    "batch_idxs": [0],
                    "instance_voxel_mask": None,
                    "instance_valid_mask": None,
                }
            else:
                cav_content[agent_type] = {
                    "batch_merged_lidar_features_torch": None,
                    "batch_merged_cam_inputs": {},
                    "record_len": torch.tensor([0], dtype=torch.int32, device=self.device),
                    "batch_idxs": [],
                    "instance_voxel_mask": None,
                    "instance_valid_mask": None,
                }
        if total_agents == 0 or not points_by_type.get("vehicle"):
            raise ValueError("at least the Ego vehicle LiDAR is required")
        identity = torch.eye(4, dtype=torch.float32, device=self.device)
        cav_content["record_len"] = torch.tensor(
            [total_agents], dtype=torch.int32, device=self.device
        )
        cav_content["img_pairwise_t_matrix_collab"] = identity.reshape(1, 1, 1, 4, 4).repeat(
            1, self.max_cav, self.max_cav, 1, 1
        )
        cav_content["pairwise_t_matrix_collab"] = cav_content[
            "img_pairwise_t_matrix_collab"
        ]

        wrapper = {
            "ego": {
                **cav_content,
                "anchor_box": self.anchor_box.to(self.device),
                "transformation_matrix": identity,
            }
        }
        with torch.no_grad(), _quiet_model(self.verbose_model):
            output = self.model(cav_content)
            result = self.postprocessor.post_process_airv2x(wrapper, {"ego": output})
        pred_boxes, pred_scores, pred_labels, _ = result
        if pred_boxes is None:
            return {
                "boxes": np.zeros((0, 8, 3), dtype=np.float32),
                "scores": np.zeros((0,), dtype=np.float32),
                "labels": np.zeros((0,), dtype=np.int64),
            }
        return {
            "boxes": pred_boxes.detach().cpu().numpy().astype(np.float32),
            "scores": pred_scores.detach().cpu().numpy().astype(np.float32),
            "labels": pred_labels.detach().cpu().numpy().astype(np.int64),
        }


def load_checkpoint_strict(
    model: Any,
    checkpoint_path: Path,
    *,
    torch_module: Any,
    device: Any,
    allow_partial: bool,
) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    try:
        payload = torch_module.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch_module.load(checkpoint_path, map_location=device)
    state = payload
    if isinstance(payload, dict):
        for key in ("model_state_dict", "model_state", "state_dict", "model"):
            if key in payload and isinstance(payload[key], dict):
                state = payload[key]
                break
    if not isinstance(state, dict):
        raise TypeError("checkpoint does not contain a state dictionary")
    normalized = {}
    for key, value in state.items():
        clean = key[7:] if key.startswith("module.") else key
        if clean.startswith("cdd"):
            clean = "m" + clean[1:]
        normalized[clean] = value
    expected = model.state_dict()
    compatible = {
        key: value
        for key, value in normalized.items()
        if key in expected and tuple(value.shape) == tuple(expected[key].shape)
    }
    missing = sorted(set(expected).difference(compatible))
    unexpected = sorted(set(normalized).difference(expected))
    mismatched = sorted(
        key
        for key in set(normalized).intersection(expected)
        if tuple(normalized[key].shape) != tuple(expected[key].shape)
    )
    matched_numel = sum(int(expected[key].numel()) for key in compatible)
    total_numel = sum(int(value.numel()) for value in expected.values())
    report = {
        "checkpoint": str(checkpoint_path),
        "matched_keys": len(compatible),
        "model_keys": len(expected),
        "coverage_by_parameter_count": matched_numel / max(total_numel, 1),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatch_keys": mismatched,
        "partial_load_allowed": bool(allow_partial),
    }
    if (missing or unexpected or mismatched) and not allow_partial:
        raise RuntimeError(
            "checkpoint/config mismatch; refusing a scientifically invalid partial load:\n"
            + json.dumps(report, indent=2, ensure_ascii=False)
        )
    model.load_state_dict(compatible, strict=not allow_partial)
    return report


def run_inference(
    *,
    run_dir: Path,
    runtime: Where2commRuntime,
    output_dir: Path,
    base_sensors: list[str],
    ego_sensor: str,
    accepted_classes: list[str],
    overwrite: bool,
    limit_frames: int | None,
    limit_candidates: int | None,
) -> Path:
    metadata, cfg, frames = _load_run(run_dir)
    candidates = list(metadata["candidate_sensor_names"])
    mode_sensor_names = {
        str(mode): str(sensor_name)
        for mode, sensor_name in metadata.get("uav_mode_sensor_names", {}).items()
    }
    mode_variants = {
        mode: f"__uav_{mode}__" for mode in mode_sensor_names
    }
    variant_to_sensor = {
        mode_variants[mode]: sensor_name
        for mode, sensor_name in mode_sensor_names.items()
    }
    if limit_frames is not None:
        frames = frames[:limit_frames]
    if limit_candidates is not None:
        candidates = candidates[:limit_candidates]
    if ego_sensor not in base_sensors:
        raise ValueError(f"ego sensor {ego_sensor!r} must be present in base_sensors")
    sensor_types = _sensor_type_map(base_sensors, ego_sensor)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    ignored_roles = list(cfg["scoring"].get("ignore_roles", []))

    index_records: list[dict[str, Any]] = []
    first_frame_input_counts: dict[str, int] = {}
    first_forward_summary: dict[str, Any] | None = None
    variants = [BASE_VARIANT, *candidates, *mode_variants.values()]
    for ordinal, frame_input in enumerate(frames, start=1):
        frame = frame_input.frame
        if ego_sensor not in frame["sensors"]:
            raise KeyError(f"frame {frame_input.index} has no Ego sensor {ego_sensor}")
        ego_matrix = np.asarray(
            frame["sensors"][ego_sensor]["transform"]["matrix"], dtype=np.float64
        )
        gt_hwl, gt_ids, gt_classes = gt_boxes_in_ego(
            frame,
            ego_matrix,
            ignored_roles=ignored_roles,
            accepted_classes=accepted_classes,
            lidar_range=runtime.lidar_range,
        )
        gt_corners = boxes_hwl_to_corners(gt_hwl)
        base_points: dict[str, list[np.ndarray]] = {"vehicle": [], "rsu": [], "drone": []}
        for sensor_name in base_sensors:
            if sensor_name not in frame["sensors"]:
                raise KeyError(f"frame {frame_input.index} has no base sensor {sensor_name}")
            points = _load_sensor_points(frame_input.frame_dir, frame, sensor_name, ego_matrix)
            points = _clip_points(points, runtime.lidar_range)
            base_points[sensor_types[sensor_name]].append(points)
            if ordinal == 1:
                first_frame_input_counts[sensor_name] = int(len(points))

        if ordinal == 1:
            readable = ", ".join(
                f"{name}={count:,}" for name, count in first_frame_input_counts.items()
            )
            print(f"Dataset read OK: frame={frame_input.index}, GT={len(gt_corners)}, {readable}")

        for variant in variants:
            relative = Path("predictions") / variant / f"{frame_input.index:06d}.npz"
            target = output_dir / relative
            if target.is_file() and not overwrite:
                prediction = _read_prediction(target)
            else:
                points_by_type = {key: list(value) for key, value in base_points.items()}
                if variant != BASE_VARIANT:
                    sensor_name = variant_to_sensor.get(variant, variant)
                    if sensor_name not in frame["sensors"]:
                        raise KeyError(
                            f"frame {frame_input.index} has no UAV sensor {sensor_name} "
                            f"for variant {variant}"
                        )
                    uav_points = _load_sensor_points(
                        frame_input.frame_dir, frame, sensor_name, ego_matrix
                    )
                    clipped_uav = _clip_points(uav_points, runtime.lidar_range)
                    points_by_type["drone"] = [clipped_uav]
                    if ordinal == 1:
                        first_frame_input_counts[sensor_name] = int(len(clipped_uav))
                prediction = runtime.infer(points_by_type)
                if first_forward_summary is None:
                    first_forward_summary = {
                        "frame": frame_input.index,
                        "variant": variant,
                        "prediction_count": int(len(prediction["boxes"])),
                        "box_shape": list(prediction["boxes"].shape),
                        "score_shape": list(prediction["scores"].shape),
                        "label_shape": list(prediction["labels"].shape),
                    }
                    print(
                        "Model forward OK: "
                        f"variant={variant}, predictions={len(prediction['boxes'])}, "
                        f"boxes={tuple(prediction['boxes'].shape)}"
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    target,
                    format=np.asarray(PREDICTION_FORMAT),
                    boxes=prediction["boxes"],
                    scores=prediction["scores"],
                    labels=prediction["labels"],
                    gt_boxes=gt_corners,
                    gt_hwl=gt_hwl,
                    gt_actor_ids=np.asarray(gt_ids, dtype=np.int64),
                    gt_class_names=np.asarray(gt_classes),
                    elapsed_s=np.asarray(frame_input.elapsed_s, dtype=np.float64),
                )
            index_records.append(
                {
                    "keyframe_index": frame_input.index,
                    "elapsed_s": frame_input.elapsed_s,
                    "variant": variant,
                    "path": str(relative),
                    "prediction_count": int(len(prediction["boxes"])),
                    "gt_count": int(len(gt_corners)),
                }
            )
        print(
            f"[{ordinal:02d}/{len(frames):02d}] t={frame_input.elapsed_s:5.1f}s "
            f"GT={len(gt_corners)} variants={len(variants)}"
        )
    (output_dir / "prediction_index.json").write_text(
        json.dumps(
            {
                "format": PREDICTION_FORMAT,
                "run_dir": str(run_dir),
                "base_sensors": base_sensors,
                "ego_sensor": ego_sensor,
                "accepted_classes": accepted_classes,
                "candidate_names": candidates,
                "mode_variants": mode_variants,
                "mode_sensor_names": mode_sensor_names,
                "checkpoint_report": runtime.checkpoint_report,
                "first_frame_input_point_counts": first_frame_input_counts,
                "first_forward_summary": first_forward_summary,
                "records": index_records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return output_dir


def evaluate_cached(
    *,
    run_dir: Path,
    output_dir: Path,
    movement_penalty_per_m: float,
    utility_fp_penalty: float,
) -> Path:
    metadata, cfg, frames = _load_run(run_dir)
    index = json.loads((output_dir / "prediction_index.json").read_text(encoding="utf-8"))
    candidates = list(index["candidate_names"])
    mode_variants = {
        str(mode): str(variant)
        for mode, variant in index.get("mode_variants", {}).items()
    }
    frame_indices = sorted({int(item["keyframe_index"]) for item in index["records"]})
    frames_by_index = {item.index: item for item in frames}
    elapsed = np.asarray(
        [frames_by_index[index].elapsed_s for index in frame_indices], dtype=np.float64
    )
    cache: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    for frame_index in frame_indices:
        for variant in [BASE_VARIANT, *candidates, *mode_variants.values()]:
            cache[(frame_index, variant)] = _read_prediction(
                output_dir / "predictions" / variant / f"{frame_index:06d}.npz"
            )

    fixed_metrics = {
        candidate: evaluate_sequence(
            [cache[(frame_index, candidate)] for frame_index in frame_indices]
        )
        for candidate in candidates
    }
    best_candidate = max(
        candidates,
        key=lambda name: (
            fixed_metrics[name]["ap_50"],
            fixed_metrics[name]["recall_50"],
            fixed_metrics[name]["ap_30"],
        ),
    )
    score_matrix = np.zeros((len(frame_indices), len(candidates)), dtype=np.float64)
    for time_index, frame_index in enumerate(frame_indices):
        for candidate_index, candidate in enumerate(candidates):
            sample = cache[(frame_index, candidate)]
            matched, false_positives = frame_match_counts(sample, iou_threshold=0.5)
            score_matrix[time_index, candidate_index] = (
                matched - utility_fp_penalty * false_positives
            ) / max(len(sample["gt_boxes"]), 1)

    grid_by_name = {item["name"]: item for item in metadata["grid"]}
    coords = np.asarray(
        [
            [grid_by_name[name]["offset_forward_m"], grid_by_name[name]["offset_right_m"]]
            for name in candidates
        ],
        dtype=np.float64,
    )
    center_index = int(np.argmin(np.linalg.norm(coords, axis=1)))
    per_frame_solution = per_frame_oracle(score_matrix, coords)
    decision_interval_s = float(
        metadata.get("decision_interval_s", metadata["keyframe_interval_s"])
    )
    decision_windows = build_decision_windows(elapsed, decision_interval_s)
    decision_scores = aggregate_scores_by_window(score_matrix, decision_windows)
    speed_decision_solution = constrained_oracle(
        decision_scores,
        coords,
        decision_interval_s,
        float(cfg["oracle"]["uav_speed_mps"]),
        float(movement_penalty_per_m),
        start_index=center_index,
    )
    speed_path = expand_window_path(speed_decision_solution.indices, decision_windows)
    paths: dict[str, np.ndarray] = {
        "center_fixed": np.full(len(frame_indices), center_index, dtype=np.int64),
    }
    if not mode_variants:
        # Backward compatibility for v1.1 collections that recorded a mode
        # path but did not save continuously moving mode-specific LiDAR.
        paths.update(
            _load_uav_mode_paths(
                run_dir,
                frame_indices=frame_indices,
                candidates=candidates,
            )
        )
    paths.update({
        "detector_best_fixed": np.full(
            len(frame_indices), candidates.index(best_candidate), dtype=np.int64
        ),
        "per_frame_detector_oracle": per_frame_solution.indices,
        "speed_constrained_detector_oracle": speed_path,
    })
    strategies: dict[str, Any] = {
        "base_only": {
            **evaluate_sequence(
                [cache[(frame_index, BASE_VARIANT)] for frame_index in frame_indices]
            ),
            "path_names": [],
            "movement_m": 0.0,
        }
    }
    for name, path in paths.items():
        samples = [
            cache[(frame_index, candidates[int(path[time_index])])]
            for time_index, frame_index in enumerate(frame_indices)
        ]
        strategies[name] = {
            **evaluate_sequence(samples),
            "path_names": [candidates[int(value)] for value in path],
            "path_indices": path.tolist(),
            "movement_m": float(
                np.linalg.norm(np.diff(coords[path], axis=0), axis=1).sum()
            ),
        }
    for mode, variant in mode_variants.items():
        mode_path = _load_uav_mode_record(
            run_dir,
            mode=mode,
            frame_indices=frame_indices,
        )
        strategies[f"uav_{mode}"] = {
            **evaluate_sequence(
                [cache[(frame_index, variant)] for frame_index in frame_indices]
            ),
            "path_names": mode_path["candidate_names"],
            "location_world_per_frame": mode_path["locations"].tolist(),
            "movement_m": float(mode_path["movement_m"]),
            "prediction_variant": variant,
            "uses_continuous_sensor_stream": True,
        }
    reference = strategies["detector_best_fixed"]
    for record in strategies.values():
        record["delta_ap50_vs_best_fixed_pp"] = 100.0 * (
            record["ap_50"] - reference["ap_50"]
        )
        record["delta_recall50_vs_best_fixed_pp"] = 100.0 * (
            record["recall_50"] - reference["recall_50"]
        )

    base = strategies["base_only"]
    transfer_passed = bool(base["recall_50"] >= 0.10 and base["ap_50"] >= 0.01)
    dynamic = strategies["speed_constrained_detector_oracle"]
    payload = {
        "scope": "AirV2X-pretrained Where2comm zero-shot on CARLA; class-agnostic BEV AP",
        "coordinate_frame": "current Ego LiDAR",
        "evaluation_range": index.get("lidar_range", None),
        "frame_count": len(frame_indices),
        "candidate_count": len(candidates),
        "time_s": elapsed.tolist(),
        "transfer_sanity": {
            "passed": transfer_passed,
            "rule": "base-only Recall@0.5 >= 10% and AP@0.5 >= 1%",
            "warning": (
                None
                if transfer_passed
                else "Zero-shot detector is not reliable on CARLA. Fine-tune/domain-adapt before using UAV deltas to judge the idea."
            ),
        },
        "selection_utility": {
            "formula": "(TP@0.5 - fp_penalty * FP@0.5) / max(GT, 1)",
            "fp_penalty": float(utility_fp_penalty),
            "uses_ground_truth": True,
            "interpretation": (
                "diagnostic oracle only; dense frame utilities are averaged inside "
                "each decision window before speed-constrained planning"
            ),
        },
        "movement": {
            "uav_speed_mps": float(cfg["oracle"]["uav_speed_mps"]),
            "frame_interval_s": float(
                metadata.get("frame_interval_s", metadata["keyframe_interval_s"])
            ),
            "decision_interval_s": decision_interval_s,
            "decision_count": int(len(decision_scores)),
            "movement_penalty_per_m": float(movement_penalty_per_m),
        },
        "best_fixed_candidate": best_candidate,
        "strategies": strategies,
        "all_fixed": fixed_metrics,
        "verdict": detector_verdict(transfer_passed, dynamic, reference),
    }
    report_path = output_dir / "where2comm_report.json"
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    text_path = output_dir / "where2comm_report.txt"
    text_path.write_text(render_report(payload), encoding="utf-8")
    print(text_path.read_text(encoding="utf-8"))
    return report_path


def _load_uav_mode_paths(
    run_dir: Path,
    *,
    frame_indices: list[int],
    candidates: list[str],
) -> dict[str, np.ndarray]:
    path = run_dir / "uav_modes.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidate_to_index = {name: index for index, name in enumerate(candidates)}
    output: dict[str, np.ndarray] = {}
    for mode in ("hover", "tracking", "patrol"):
        record = payload.get("paths", {}).get(mode)
        if record is None:
            continue
        names = list(record.get("candidate_names_per_frame", []))
        if frame_indices and max(frame_indices) >= len(names):
            raise ValueError(
                f"UAV mode {mode} has {len(names)} frames but evaluation requests "
                f"frame {max(frame_indices)}"
            )
        selected = [names[index] for index in frame_indices]
        unknown = sorted(set(selected).difference(candidate_to_index))
        if unknown:
            raise ValueError(f"UAV mode {mode} references unknown candidates: {unknown}")
        output[f"uav_{mode}"] = np.asarray(
            [candidate_to_index[name] for name in selected], dtype=np.int64
        )
    return output


def _load_uav_mode_record(
    run_dir: Path,
    *,
    mode: str,
    frame_indices: list[int],
) -> dict[str, Any]:
    path = run_dir / "uav_modes.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"continuous UAV variant {mode!r} exists but {path} is missing"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    record = payload.get("paths", {}).get(mode)
    if record is None:
        raise KeyError(f"uav_modes.json has no path for mode {mode!r}")
    names = list(record.get("candidate_names_per_frame", []))
    locations = list(record.get("location_world_per_frame", []))
    if frame_indices and (
        max(frame_indices) >= len(names) or max(frame_indices) >= len(locations)
    ):
        raise ValueError(
            f"UAV mode {mode} has {len(names)} names/{len(locations)} locations, "
            f"but evaluation requests frame {max(frame_indices)}"
        )
    selected_names = [str(names[index]) for index in frame_indices]
    selected_locations = np.asarray(
        [locations[index] for index in frame_indices], dtype=np.float64
    )
    movement_m = (
        float(np.linalg.norm(np.diff(selected_locations, axis=0), axis=1).sum())
        if len(selected_locations) > 1
        else 0.0
    )
    return {
        "candidate_names": selected_names,
        "locations": selected_locations,
        "movement_m": movement_m,
    }


def evaluate_sequence(samples: list[dict[str, np.ndarray]]) -> dict[str, float | int]:
    from opencood.utils import eval_utils

    import torch

    result: dict[str, Any] = {}
    for threshold in (0.3, 0.5, 0.7):
        stats = {threshold: {"tp": [], "fp": [], "gt": 0, "score": []}}
        for sample in samples:
            boxes = torch.from_numpy(sample["boxes"]).float()
            scores = torch.from_numpy(sample["scores"]).float()
            gt = torch.from_numpy(sample["gt_boxes"]).float()
            eval_utils.caluclate_tp_fp(boxes, scores, gt, stats, threshold)
            # caluclate_tp_fp appends TP/FP in descending score order within
            # each frame but the older AirV2X eval_utils does not retain the
            # scores. Keep them here so AP can be sorted across all frames.
            stats[threshold]["score"].extend(
                np.sort(np.asarray(sample["scores"], dtype=np.float64))[::-1].tolist()
            )
        raw = stats[threshold]
        tp_total = int(sum(raw["tp"]))
        fp_total = int(sum(raw["fp"]))
        gt_total = int(raw["gt"])
        ap, _, _ = calculate_global_ap(
            raw["tp"], raw["fp"], raw["score"], gt_total
        )
        result[f"ap_{int(threshold * 100):02d}"] = float(ap)
        result[f"recall_{int(threshold * 100):02d}"] = (
            float(tp_total / gt_total) if gt_total else 0.0
        )
        result[f"tp_{int(threshold * 100):02d}"] = tp_total
        result[f"fp_{int(threshold * 100):02d}"] = fp_total
        result[f"gt_{int(threshold * 100):02d}"] = gt_total
    return result


def calculate_global_ap(
    tp: Iterable[int],
    fp: Iterable[int],
    scores: Iterable[float],
    gt_total: int,
) -> tuple[float, list[float], list[float]]:
    """VOC-style AP after globally sorting detections by confidence.

    This intentionally avoids OpenCOOD's version-dependent calculate_ap
    signature. Older AirV2X branches expose calculate_ap(result, iou), while
    newer branches add global_sort_detections.
    """
    tp_array = np.asarray(list(tp), dtype=np.float64)
    fp_array = np.asarray(list(fp), dtype=np.float64)
    score_array = np.asarray(list(scores), dtype=np.float64)
    if not (len(tp_array) == len(fp_array) == len(score_array)):
        raise ValueError(
            "TP, FP and confidence arrays must have equal length: "
            f"{len(tp_array)}, {len(fp_array)}, {len(score_array)}"
        )
    if gt_total <= 0 or not len(tp_array):
        return 0.0, [], []

    order = np.argsort(-score_array, kind="stable")
    tp_cumulative = np.cumsum(tp_array[order])
    fp_cumulative = np.cumsum(fp_array[order])
    recall = tp_cumulative / float(gt_total)
    precision = tp_cumulative / np.maximum(tp_cumulative + fp_cumulative, 1e-12)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])
    changed = np.where(mrec[1:] != mrec[:-1])[0] + 1
    ap = float(np.sum((mrec[changed] - mrec[changed - 1]) * mpre[changed]))
    return ap, mrec.tolist(), mpre.tolist()


def frame_match_counts(
    sample: dict[str, np.ndarray], *, iou_threshold: float
) -> tuple[int, int]:
    from opencood.utils import eval_utils

    import torch

    stats = {iou_threshold: {"tp": [], "fp": [], "gt": 0, "score": []}}
    eval_utils.caluclate_tp_fp(
        torch.from_numpy(sample["boxes"]).float(),
        torch.from_numpy(sample["scores"]).float(),
        torch.from_numpy(sample["gt_boxes"]).float(),
        stats,
        iou_threshold,
    )
    return int(sum(stats[iou_threshold]["tp"])), int(sum(stats[iou_threshold]["fp"]))


def detector_verdict(
    transfer_passed: bool, dynamic: dict[str, Any], fixed: dict[str, Any]
) -> dict[str, str]:
    if not transfer_passed:
        return {
            "code": "DOMAIN_TRANSFER_FAILED",
            "reason": "The AirV2X checkpoint does not detect CARLA reliably enough to judge active UAV movement.",
        }
    ap_gain = 100.0 * (dynamic["ap_50"] - fixed["ap_50"])
    recall_gain = 100.0 * (dynamic["recall_50"] - fixed["recall_50"])
    if ap_gain >= 1.0 and recall_gain >= 1.0:
        return {
            "code": "ACTIVE_VIEW_SIGNAL",
            "reason": "The speed-constrained detector oracle improves both AP@0.5 and Recall@0.5 by at least 1 point over the best fixed UAV.",
        }
    return {
        "code": "NO_CLEAR_ACTIVE_VIEW_SIGNAL",
        "reason": "The speed-constrained detector oracle does not improve both AP@0.5 and Recall@0.5 by 1 point over the best fixed UAV.",
    }


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "Active AirV2X — Where2comm checkpoint evaluation",
        "=================================================",
        f"Verdict: {payload['verdict']['code']}",
        f"Reason: {payload['verdict']['reason']}",
        f"Transfer sanity: {'PASS' if payload['transfer_sanity']['passed'] else 'FAIL'}",
        f"Frames / candidates: {payload['frame_count']} / {payload['candidate_count']}",
        f"Best fixed UAV: {payload['best_fixed_candidate']}",
        "",
        "Class-agnostic BEV metrics:",
    ]
    for name, record in payload["strategies"].items():
        lines.append(
            f"  {name:36s} AP30={100*record['ap_30']:6.2f} "
            f"AP50={100*record['ap_50']:6.2f} R50={100*record['recall_50']:6.2f} "
            f"move={record['movement_m']:6.1f}m"
        )
    lines.extend(
        [
            "",
            "The detector-oracle uses GT only to test whether useful movement exists; it is not a deployable policy.",
            "If transfer sanity fails, fine-tune on CARLA before interpreting UAV-policy differences.",
            "",
        ]
    )
    return "\n".join(lines)


def resolve_checkpoint(model_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    candidates = sorted(model_dir.glob("net_epoch_bestval_at*.pth"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise RuntimeError("multiple best checkpoints found; pass --checkpoint explicitly")
    candidates = sorted(model_dir.glob("net_epoch*.pth"), key=_checkpoint_epoch)
    if not candidates:
        raise FileNotFoundError(
            f"no net_epoch*.pth under {model_dir}; checkpoint files are gitignored in Airv2x_gs"
        )
    return candidates[-1]


def _checkpoint_epoch(path: Path) -> int:
    digits = "".join(character for character in path.stem if character.isdigit())
    return int(digits or -1)


def _load_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], list[FrameInput]]:
    run_dir = run_dir.expanduser().resolve()
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    cfg = load_config(run_dir / "effective_config.yaml")
    manifest = [
        json.loads(line)
        for line in (run_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    frames = []
    for item in manifest:
        frame_dir = run_dir / item["path"]
        frames.append(
            FrameInput(
                index=int(item["keyframe_index"]),
                elapsed_s=float(item["elapsed_s"]),
                frame_dir=frame_dir,
                frame=json.loads((frame_dir / "frame.json").read_text(encoding="utf-8")),
            )
        )
    if not frames:
        raise RuntimeError(f"no frames found in {run_dir}")
    return metadata, cfg, frames


def _sensor_type_map(base_sensors: list[str], ego_sensor: str) -> dict[str, str]:
    result = {}
    for name in base_sensors:
        lowered = name.lower()
        if name == ego_sensor or "cav" in lowered or "car" in lowered or "vehicle" in lowered:
            result[name] = "vehicle"
        elif "rsu" in lowered:
            result[name] = "rsu"
        else:
            raise ValueError(
                f"cannot infer agent type for base sensor {name!r}; name it with ego/cav/car/vehicle or rsu"
            )
    return result


def _load_sensor_points(
    frame_dir: Path,
    frame: dict[str, Any],
    sensor_name: str,
    ego_matrix: np.ndarray,
) -> np.ndarray:
    record = frame["sensors"].get(sensor_name)
    if record is None:
        raise KeyError(f"frame {frame.get('keyframe_index')} has no sensor {sensor_name}")
    if record.get("modality") != "lidar":
        raise ValueError(f"sensor {sensor_name} is not LiDAR")
    raw = np.fromfile(frame_dir / record["path"], dtype=np.float32)
    if raw.size % 4:
        raise ValueError(f"invalid CARLA LiDAR binary: {frame_dir / record['path']}")
    points = raw.reshape(-1, 4)
    return carla_points_to_ego(
        points,
        np.asarray(record["transform"]["matrix"], dtype=np.float64),
        ego_matrix,
    )


def _clip_points(points: np.ndarray, lidar_range: np.ndarray) -> np.ndarray:
    limits = np.asarray(lidar_range, dtype=np.float32)
    mask = np.all(
        (points[:, :3] >= limits[:3][None, :])
        & (points[:, :3] <= limits[3:][None, :]),
        axis=1,
    )
    points = points[mask]
    # Match OpenCOOD's standard ego-point removal after projection.
    keep = ~((np.abs(points[:, 0]) < 1.95) & (np.abs(points[:, 1]) < 1.05))
    return points[keep]


def _read_prediction(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        return {key: np.asarray(arrays[key]) for key in arrays.files}


def _matrix4(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must have shape [4,4]")
    return matrix


def _to_device(value: Any, device: Any) -> Any:
    if hasattr(value, "to") and callable(value.to):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    return value


@contextlib.contextmanager
def _quiet_model(verbose: bool):
    """Suppress per-frame debug prints and the model's hard-coded PNG write."""
    if verbose:
        yield
        return
    old_imwrite = None
    try:
        import cv2

        old_imwrite = cv2.imwrite
        cv2.imwrite = lambda *args, **kwargs: True
    except ImportError:
        cv2 = None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            yield
    finally:
        if old_imwrite is not None and cv2 is not None:
            cv2.imwrite = old_imwrite


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an AirV2X Where2comm checkpoint on Active AirV2X CARLA frames"
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--opencood-root", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--base-sensors", nargs="+", default=None)
    parser.add_argument("--ego-sensor", default="ego_lidar")
    parser.add_argument("--classes", nargs="+", default=["car", "two_wheeler"])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-partial-checkpoint", action="store_true")
    parser.add_argument("--verbose-model", action="store_true")
    parser.add_argument("--limit-frames", type=int, default=None)
    parser.add_argument("--limit-candidates", type=int, default=None)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="run exactly one frame and one UAV candidate, then report read/load/forward/eval status",
    )
    parser.add_argument("--movement-penalty-per-m", type=float, default=0.0)
    parser.add_argument("--utility-fp-penalty", type=float, default=0.05)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        if args.evaluate_only:
            parser.error("--smoke-test cannot be combined with --evaluate-only")
        args.limit_frames = 1
        args.limit_candidates = 1

    run_dir = args.run_dir.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_dir / "where2comm_eval"
    )
    opencood_root = args.opencood_root.expanduser().resolve()
    if not (opencood_root / "opencood").is_dir():
        raise FileNotFoundError(
            f"{opencood_root} does not contain the opencood package"
        )
    if str(opencood_root) not in sys.path:
        sys.path.insert(0, str(opencood_root))
    _, cfg, _ = _load_run(run_dir)
    base_sensors = (
        list(args.base_sensors)
        if args.base_sensors is not None
        else list(cfg["scoring"]["base_sensors"])
    )
    if not args.evaluate_only:
        config_path = (
            args.config.expanduser().resolve()
            if args.config is not None
            else model_dir / "config.yaml"
        )
        if not config_path.is_file():
            raise FileNotFoundError(
                f"model config not found: {config_path}; pass --config with the exact training config"
            )
        checkpoint = resolve_checkpoint(model_dir, args.checkpoint)
        runtime = Where2commRuntime(
            opencood_root=opencood_root,
            config_path=config_path,
            checkpoint_path=checkpoint,
            device=args.device,
            allow_partial_checkpoint=args.allow_partial_checkpoint,
            verbose_model=args.verbose_model,
        )
        run_inference(
            run_dir=run_dir,
            runtime=runtime,
            output_dir=output_dir,
            base_sensors=base_sensors,
            ego_sensor=args.ego_sensor,
            accepted_classes=list(args.classes),
            overwrite=args.overwrite,
            limit_frames=args.limit_frames,
            limit_candidates=args.limit_candidates,
        )
        index_path = output_dir / "prediction_index.json"
        index_payload = json.loads(index_path.read_text(encoding="utf-8"))
        index_payload["lidar_range"] = runtime.lidar_range.tolist()
        index_path.write_text(
            json.dumps(index_payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    report_path = evaluate_cached(
        run_dir=run_dir,
        output_dir=output_dir,
        movement_penalty_per_m=args.movement_penalty_per_m,
        utility_fp_penalty=args.utility_fp_penalty,
    )
    if args.smoke_test:
        print("SMOKE TEST PASSED")
        print(f"Result: {report_path}")


if __name__ == "__main__":
    main()
