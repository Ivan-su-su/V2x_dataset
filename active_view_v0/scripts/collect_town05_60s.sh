#!/usr/bin/env bash
set -euo pipefail

# Start CARLA and activate the collection environment first.
# bash scripts/collect_town05_60s.sh          # Collect scenes 0001–0015
# bash scripts/collect_town05_60s.sh 004      # Collect only scene 0004
# BATCH=20261125_220000 bash scripts/collect_town05_60s.sh  # Resume a batch

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  echo "Usage: bash $0 [scene number 1–15 ...]"
  echo "Optional environment variables: PROJECT, ACTIVEVIEW_DATA_ROOT, DATA_ROOT, BATCH"
  exit 0
fi

if (( $# == 0 )); then
  set -- {1..15}
fi
for selected in "$@"; do
  if [[ ! "$selected" =~ ^[0-9]{1,4}$ ]] || (( 10#$selected < 1 || 10#$selected > 15 )); then
    echo "无效场景编号：$selected（应为 0001–0015）" >&2
    exit 2
  fi
done

PROJECT="${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
DATA_ROOT="${DATA_ROOT:-${ACTIVEVIEW_DATA_ROOT:-/mnt/disk_4/suyi/datasets/active_view_v1}/Town05/grid_60s}"
BATCH="${BATCH:-$(date +%Y%m%d_%H%M%S)}"
if [[ ! "$BATCH" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "无效批次名：$BATCH" >&2
  exit 2
fi

mkdir -p "$DATA_ROOT"
BATCH_DIR="$(cd "$DATA_ROOT" && pwd -P)/$BATCH"
LOG_DIR="$BATCH_DIR/logs"
mkdir -p "$LOG_DIR" "$BATCH_DIR/configs"
cd "$PROJECT"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

collection_complete() {
  python - "$1/collection_status.json" <<'PY'
import json
import sys
from pathlib import Path

try:
    status = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except (OSError, ValueError):
    sys.exit(1)
sys.exit(0 if isinstance(status, dict) and status.get("status") == "complete"
         and status.get("keyframe_count") == 600 else 1)
PY
}

failed=()
echo "采集批次：$BATCH_DIR"
for selected in "$@"; do
  printf -v scene 'scene_%04d' "$((10#$selected))"
  scene_dir="$BATCH_DIR/$scene"
  config="$BATCH_DIR/configs/$scene.yaml"
  log="$LOG_DIR/$scene.log"

  if [[ -e "$scene_dir" ]]; then
    if collection_complete "$scene_dir"; then
      echo "SKIP $scene：已有完整的 600 帧"
    else
      echo "SKIP $scene：已有不完整的输出；请用新的 BATCH 补采" >&2
      failed+=("$scene")
    fi
    continue
  fi

  echo "START $scene  日志：$log"
  if (
    python - "$scene" "$BATCH_DIR" "$config" <<'PY' &&
import math
import os
import sys

from active_view_v0.config import dump_effective_config
from active_view_v0.dataset_scene import resolve_scene

scene, root, config = sys.argv[1:]
spec = resolve_scene(f"Town05/{scene}")
cfg = spec.config
if (spec.frame_count != 600 or not math.isclose(spec.dt, 0.1)
        or not math.isclose(cfg["regional"]["collection_interval_s"], 0.1)
        or cfg["grid"]["size"] != 5):
    raise SystemExit(f"{spec.scene_id}: 需要 600 帧、10 Hz、5×5 UAV 网格")

for name, key, cast in (
    ("ACTIVEVIEW_CARLA_HOST", "host", str),
    ("ACTIVEVIEW_CARLA_PORT", "port", int),
    ("ACTIVEVIEW_TM_PORT", "traffic_manager_port", int),
):
    if name in os.environ:
        cfg["carla"][key] = cast(os.environ[name])
cfg["output"] = {"root": root, "run_name": scene}
dump_effective_config(cfg, config)
PY
    python -u -m active_view_v0.regional_collector --config "$config"
  ) 2>&1 | tee -a "$log" && collection_complete "$scene_dir"; then
    echo "DONE $scene  数据：$scene_dir"
  else
    echo "FAILED $scene  查看：$log" >&2
    failed+=("$scene")
  fi
done

if (( ${#failed[@]} > 0 )); then
  echo "未完成采集：${failed[*]}。日志：$LOG_DIR" >&2
  exit 1
fi
echo "所选场景采集完成。采集批次：$BATCH_DIR"
