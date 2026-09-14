from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def plot(run_dir: str | Path) -> Path:
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir).expanduser().resolve()
    arrays = np.load(run_dir / "geometry_scores.npz", allow_pickle=False)
    scores = arrays["scores"]
    elapsed = arrays["elapsed_s"]
    names = arrays["candidate_names"].astype(str)
    result = json.loads((run_dir / "oracle_results.json").read_text(encoding="utf-8"))
    per_frame = np.asarray(result["solutions"]["per_frame_unconstrained"]["path_indices"])
    constrained = np.asarray(result["solutions"]["speed_constrained_center_start"]["path_indices"])
    fixed = np.asarray(result["solutions"]["best_fixed"]["path_indices"])
    center = np.asarray(result["solutions"]["center_fixed"]["path_indices"])

    figure, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    image = axes[0].imshow(scores, aspect="auto", cmap="viridis", origin="lower")
    axes[0].plot(per_frame, np.arange(len(elapsed)), "w.-", label="per-frame oracle")
    axes[0].plot(constrained, np.arange(len(elapsed)), "r.-", label="speed-constrained")
    axes[0].set_xlabel("candidate index (row-major 5x5 grid)")
    axes[0].set_ylabel("keyframe")
    axes[0].legend(loc="upper right")
    figure.colorbar(image, ax=axes[0], label="geometry score")

    axes[1].plot(elapsed, scores[np.arange(len(elapsed)), per_frame], label="per-frame oracle")
    axes[1].plot(elapsed, scores[np.arange(len(elapsed)), constrained], label="speed-constrained")
    axes[1].plot(elapsed, scores[np.arange(len(elapsed)), fixed], label=f"best fixed: {names[fixed[0]]}")
    axes[1].plot(
        elapsed,
        scores[np.arange(len(elapsed)), center],
        "--",
        label=f"center fixed: {names[center[0]]}",
    )
    axes[1].set_xlabel("simulation time (s)")
    axes[1].set_ylabel("geometry score")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    output = run_dir / "active_view_validation.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"Saved: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ActiveView-v0 geometry scores and oracle paths")
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    plot(args.run_dir)


if __name__ == "__main__":
    main()
