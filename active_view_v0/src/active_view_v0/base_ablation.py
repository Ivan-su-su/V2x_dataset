from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config
from .scoring import score_dataset
from .solve_oracle import solve


PROFILE_ORDER = ("ego_only", "ego_rsu", "ego_cavs", "full")


def base_profiles(configured_base: list[str]) -> dict[str, list[str]]:
    """Return the controlled ground-agent subsets used by the benchmark."""
    configured = list(dict.fromkeys(str(name) for name in configured_base))
    required = {"ego_lidar", "car1_lidar", "car3_lidar", "rsu_lidar"}
    missing = sorted(required.difference(configured))
    if missing:
        raise ValueError(
            "the canonical four-profile ablation requires these configured sensors: "
            + ", ".join(missing)
        )
    return {
        "ego_only": ["ego_lidar"],
        "ego_rsu": ["ego_lidar", "rsu_lidar"],
        "ego_cavs": ["ego_lidar", "car1_lidar", "car3_lidar"],
        "full": configured,
    }


def run_base_ablation(run_dir: str | Path, *, overwrite: bool = False) -> Path:
    """Run all base-agent subsets on an already collected raw episode."""
    run_dir = Path(run_dir).expanduser().resolve()
    _validate_raw_run(run_dir)
    cfg = load_config(run_dir / "effective_config.yaml")
    profiles = base_profiles(list(cfg["scoring"]["base_sensors"]))
    root = run_dir / "base_ablations"
    if root.exists() and not overwrite:
        raise FileExistsError(
            f"ablation output already exists: {root}; pass --overwrite to recompute it"
        )
    root.mkdir(parents=True, exist_ok=True)

    configured_penalty = float(cfg["oracle"]["movement_penalty_per_m"])
    profile_payloads: dict[str, dict[str, Any]] = {}
    for profile_name in PROFILE_ORDER:
        sensors = profiles[profile_name]
        profile_dir = root / profile_name
        profile_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Base ablation: {profile_name} ({', '.join(sensors)}) ===")
        score_dataset(run_dir, base_sensors=sensors, output_dir=profile_dir)
        penalized_path = solve(
            run_dir,
            evaluation_dir=profile_dir,
            movement_penalty_per_m=configured_penalty,
            output_name="oracle_penalized.json",
        )
        speed_only_path = solve(
            run_dir,
            evaluation_dir=profile_dir,
            movement_penalty_per_m=0.0,
            output_name="oracle_speed_only.json",
        )
        profile_payloads[profile_name] = _summarize_profile(
            profile_name,
            sensors,
            profile_dir,
            penalized_path,
            speed_only_path,
            cfg,
        )

    report = _build_report(run_dir, configured_penalty, profile_payloads)
    json_path = root / "base_ablation_report.json"
    text_path = root / "base_ablation_report.txt"
    csv_path = root / "base_ablation_summary.csv"
    diagnostics_path = root / "late_miss_diagnostics.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    text_path.write_text(_render_text(report), encoding="utf-8")
    _write_csv(csv_path, report)
    diagnostics_path.write_text(
        json.dumps(
            {
                name: payload["miss_diagnostics"]
                for name, payload in profile_payloads.items()
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _plot_summary(root, report)
    print("\n" + text_path.read_text(encoding="utf-8"))
    print(f"Saved base ablation outputs: {root}")
    return root


def _validate_raw_run(run_dir: Path) -> None:
    required = ("metadata.json", "effective_config.yaml", "manifest.jsonl")
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete raw run {run_dir}; missing: {', '.join(missing)}")


def _summarize_profile(
    profile_name: str,
    sensors: list[str],
    profile_dir: Path,
    penalized_path: Path,
    speed_only_path: Path,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    arrays = np.load(profile_dir / "geometry_scores.npz", allow_pickle=False)
    scores = np.asarray(arrays["scores"], dtype=np.float64)
    elapsed = np.asarray(arrays["elapsed_s"], dtype=np.float64)
    targets = np.asarray(arrays["target_counts"], dtype=np.int64)
    base_visible = np.asarray(arrays["base_visible_counts"], dtype=np.int64)
    gain = np.asarray(arrays["gain"], dtype=np.float64)
    penalized = json.loads(penalized_path.read_text(encoding="utf-8"))
    speed_only = json.loads(speed_only_path.read_text(encoding="utf-8"))
    fixed = penalized["solutions"]["best_fixed"]
    per_frame = penalized["solutions"]["per_frame_unconstrained"]
    penalized_center = penalized["solutions"]["speed_constrained_center_start"]
    penalized_best_start = penalized["solutions"]["speed_constrained_best_start"]
    speed_center = speed_only["solutions"]["speed_constrained_center_start"]
    speed_best_start = speed_only["solutions"]["speed_constrained_best_start"]
    late_start = float(
        cfg.get("regional", {}).get("validation", {}).get("late_start_s", 16.0)
    )
    miss_diagnostics = _miss_diagnostics(
        profile_dir / "geometry_details.json", elapsed, late_start
    )
    return {
        "profile": profile_name,
        "base_sensors": sensors,
        "target_count_total": int(targets.sum()),
        "base_visible_count_total": int(base_visible.sum()),
        "base_missed_count_total": int((targets - base_visible).sum()),
        "uav_rescue_frame_count": int(np.count_nonzero(gain.max(axis=1) > 1e-8)),
        "mean_score_span": float(np.mean(scores.max(axis=1) - scores.min(axis=1))),
        "strategies": {
            "best_fixed": _strategy(fixed),
            "per_frame_unconstrained": _strategy(per_frame),
            "speed_only_center_start": _strategy(speed_center),
            "speed_only_best_start": _strategy(speed_best_start),
            "penalized_center_start": _strategy(penalized_center),
            "penalized_best_start": _strategy(penalized_best_start),
        },
        "relative_gaps": {
            "per_frame_over_best_fixed": _relative(per_frame, fixed),
            "speed_only_center_over_best_fixed": _relative(speed_center, fixed),
            "speed_only_best_start_over_best_fixed": _relative(speed_best_start, fixed),
            "penalized_center_over_best_fixed": _relative(penalized_center, fixed),
            "penalized_best_start_over_best_fixed": _relative(penalized_best_start, fixed),
        },
        "miss_diagnostics": miss_diagnostics,
    }


def _strategy(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "mean_score": float(item["mean_score"]),
        "perception_reward": float(item["perception_reward"]),
        "movement_m": float(item["movement_m"]),
        "switch_count": int(item["switch_count"]),
        "path_names": item["path_names"],
    }


def _relative(value: dict[str, Any], reference: dict[str, Any]) -> float | None:
    numerator = float(value["perception_reward"])
    denominator = float(reference["perception_reward"])
    if abs(denominator) < 1e-12:
        return None
    return (numerator - denominator) / denominator


def _miss_diagnostics(details_path: Path, elapsed: np.ndarray, late_start_s: float) -> dict[str, Any]:
    details = json.loads(details_path.read_text(encoding="utf-8"))
    frames: list[dict[str, Any]] = []
    late_missed = 0
    late_rescuable = 0
    late_irrecoverable = 0
    late_class_counts: dict[str, int] = {}
    for time_index, detail in enumerate(details):
        base_visible = np.asarray(detail["base_visible"], dtype=bool)
        missed_indices = np.flatnonzero(~base_visible)
        union_visibility = [
            np.asarray(candidate["union_visible"], dtype=bool)
            for candidate in detail["candidates"].values()
        ]
        rescued_any = (
            np.logical_or.reduce(union_visibility) & ~base_visible
            if union_visibility
            else np.zeros_like(base_visible)
        )
        rescuable_indices = np.flatnonzero(rescued_any)
        irrecoverable_indices = np.flatnonzero((~base_visible) & (~rescued_any))
        frame = {
            "elapsed_s": float(elapsed[time_index]),
            "missed_target_count": int(len(missed_indices)),
            "rescuable_by_any_uav_count": int(len(rescuable_indices)),
            "irrecoverable_by_grid_count": int(len(irrecoverable_indices)),
            "missed_target_ids": [int(detail["target_ids"][index]) for index in missed_indices],
            "rescuable_target_ids": [
                int(detail["target_ids"][index]) for index in rescuable_indices
            ],
            "irrecoverable_target_ids": [
                int(detail["target_ids"][index]) for index in irrecoverable_indices
            ],
        }
        frames.append(frame)
        if elapsed[time_index] >= late_start_s - 1e-6:
            late_missed += len(missed_indices)
            late_rescuable += len(rescuable_indices)
            late_irrecoverable += len(irrecoverable_indices)
            for index in missed_indices:
                class_name = str(detail["target_classes"][index])
                late_class_counts[class_name] = late_class_counts.get(class_name, 0) + 1
    return {
        "late_start_s": late_start_s,
        "late_missed_target_instances": int(late_missed),
        "late_rescuable_target_instances": int(late_rescuable),
        "late_irrecoverable_target_instances": int(late_irrecoverable),
        "late_rescuable_fraction": (
            float(late_rescuable / late_missed) if late_missed else None
        ),
        "late_missed_class_counts": late_class_counts,
        "frames": frames,
    }


def _build_report(
    run_dir: Path,
    configured_penalty: float,
    profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    full_gap = profiles["full"]["relative_gaps"]["speed_only_best_start_over_best_fixed"]
    sparse_gaps = [
        profiles[name]["relative_gaps"]["speed_only_best_start_over_best_fixed"]
        for name in ("ego_only", "ego_rsu", "ego_cavs")
    ]
    best_sparse = max(value for value in sparse_gaps if value is not None)
    if full_gap is not None and full_gap >= 0.03:
        verdict = "FULL_COOP_ACTIVE_SIGNAL"
        reason = "The speed-limited UAV beats best fixed by at least 3% even with the full ground base."
    elif best_sparse >= 0.03 and (full_gap is None or full_gap < 0.03):
        verdict = "SPARSE_INFRASTRUCTURE_SIGNAL"
        reason = (
            "Active movement is useful only after removing some ground agents; "
            "the defensible task is robustness to sparse infrastructure or agent dropout."
        )
    else:
        verdict = "ACTIVE_SIGNAL_WEAK_ACROSS_BASES"
        reason = (
            "No ground-base profile gives a 3% speed-limited gain over its own best fixed view."
        )
    return {
        "verdict": verdict,
        "reason": reason,
        "scope": "same_frames_same_GT_same_ROI_geometry_only_lidar_support_proxy_not_AP",
        "raw_run_dir": str(run_dir),
        "configured_movement_penalty_per_m": configured_penalty,
        "profile_order": list(PROFILE_ORDER),
        "profiles": profiles,
        "interpretation_rule": {
            "full_coop_active_signal": "full-base speed-only best-start gain >= 3%",
            "sparse_infrastructure_signal": (
                "some reduced-base speed-only best-start gain >= 3%, but full-base gain < 3%"
            ),
            "weak": "all speed-only best-start gains < 3%",
        },
        "fairness_note": (
            "Car1/Car3 platform actors remain excluded from GT in every profile; only their sensor "
            "inputs change. Speed-only isolates reachability; penalized additionally includes the "
            "configured movement cost."
        ),
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:+.2f}%"


def _render_text(report: dict[str, Any]) -> str:
    lines = [
        "Active AirV2X base-agent ablation",
        "=================================",
        f"Verdict: {report['verdict']}",
        f"Reason: {report['reason']}",
        "",
        "All profiles reuse the same frames, GT, Ego ROI and 25 UAV candidates.",
        "",
        "Profile results:",
    ]
    for name in report["profile_order"]:
        profile = report["profiles"][name]
        gaps = profile["relative_gaps"]
        diag = profile["miss_diagnostics"]
        lines.extend(
            [
                f"  {name}: {', '.join(profile['base_sensors'])}",
                f"    base missed target-instances: {profile['base_missed_count_total']}",
                f"    UAV rescue frames: {profile['uav_rescue_frame_count']}",
                f"    oracle upper bound vs best fixed: "
                f"{_pct(gaps['per_frame_over_best_fixed'])}",
                f"    speed-only center/best start vs best fixed: "
                f"{_pct(gaps['speed_only_center_over_best_fixed'])} / "
                f"{_pct(gaps['speed_only_best_start_over_best_fixed'])}",
                f"    penalized center/best start vs best fixed: "
                f"{_pct(gaps['penalized_center_over_best_fixed'])} / "
                f"{_pct(gaps['penalized_best_start_over_best_fixed'])}",
                f"    late missed/rescuable/irrecoverable: "
                f"{diag['late_missed_target_instances']} / "
                f"{diag['late_rescuable_target_instances']} / "
                f"{diag['late_irrecoverable_target_instances']}",
            ]
        )
    lines.extend(
        [
            "",
            "Interpret speed-only before penalized results: if speed-only is weak, tuning the movement penalty cannot rescue the idea.",
            "Important: this is still a LiDAR box-support proxy, not detector AP.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_csv(path: Path, report: dict[str, Any]) -> None:
    fields = [
        "profile",
        "base_sensors",
        "base_missed_count_total",
        "uav_rescue_frame_count",
        "best_fixed_mean_score",
        "per_frame_gain_pct",
        "speed_only_center_gain_pct",
        "speed_only_best_start_gain_pct",
        "penalized_center_gain_pct",
        "penalized_best_start_gain_pct",
        "late_missed",
        "late_rescuable",
        "late_irrecoverable",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in report["profile_order"]:
            profile = report["profiles"][name]
            gaps = profile["relative_gaps"]
            diag = profile["miss_diagnostics"]
            writer.writerow(
                {
                    "profile": name,
                    "base_sensors": "+".join(profile["base_sensors"]),
                    "base_missed_count_total": profile["base_missed_count_total"],
                    "uav_rescue_frame_count": profile["uav_rescue_frame_count"],
                    "best_fixed_mean_score": profile["strategies"]["best_fixed"]["mean_score"],
                    "per_frame_gain_pct": _as_pct(gaps["per_frame_over_best_fixed"]),
                    "speed_only_center_gain_pct": _as_pct(
                        gaps["speed_only_center_over_best_fixed"]
                    ),
                    "speed_only_best_start_gain_pct": _as_pct(
                        gaps["speed_only_best_start_over_best_fixed"]
                    ),
                    "penalized_center_gain_pct": _as_pct(
                        gaps["penalized_center_over_best_fixed"]
                    ),
                    "penalized_best_start_gain_pct": _as_pct(
                        gaps["penalized_best_start_over_best_fixed"]
                    ),
                    "late_missed": diag["late_missed_target_instances"],
                    "late_rescuable": diag["late_rescuable_target_instances"],
                    "late_irrecoverable": diag["late_irrecoverable_target_instances"],
                }
            )


def _as_pct(value: float | None) -> float | None:
    return None if value is None else 100.0 * value


def _plot_summary(root: Path, report: dict[str, Any]) -> Path:
    import matplotlib.pyplot as plt

    labels = list(report["profile_order"])
    metric_keys = (
        "per_frame_over_best_fixed",
        "speed_only_best_start_over_best_fixed",
        "penalized_best_start_over_best_fixed",
    )
    metric_labels = ("unconstrained upper bound", "speed-only", "speed + movement cost")
    values = np.asarray(
        [
            [
                100.0 * (report["profiles"][name]["relative_gaps"][key] or 0.0)
                for key in metric_keys
            ]
            for name in labels
        ],
        dtype=np.float64,
    )
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.24
    figure, axis = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    for index, metric_label in enumerate(metric_labels):
        axis.bar(x + (index - 1) * width, values[:, index], width, label=metric_label)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.axhline(3.0, color="tab:red", linewidth=1.0, linestyle="--", label="3% signal gate")
    axis.set_xticks(x, labels)
    axis.set_ylabel("perception reward gain over each profile's best fixed view (%)")
    axis.set_title("Base-agent ablation on identical raw frames")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=2)
    output = root / "base_ablation_summary.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Ego/RSU/CAV/full base-agent ablations on one collected episode"
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_base_ablation(args.run_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
