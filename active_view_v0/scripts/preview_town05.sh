#!/usr/bin/env bash
set -euo pipefail

# Start CARLA and activate the simulation environment first.
# bash scripts/preview_town05.sh               # All scenes
# bash scripts/preview_town05.sh 002 008 011   # Selected scenes
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  echo "Usage: bash $0 [scene number 1–27 ...] (default: 1–15)"
  echo "Optional environment variables: ACTIVEVIEW_DATA_ROOT, RUN_ID"
  exit 0
fi
if (( $# == 0 )); then
  set -- {1..15}
fi
for selected in "$@"; do
  if [[ ! "$selected" =~ ^[0-9]{1,4}$ ]] || (( 10#$selected < 1 || 10#$selected > 27 )); then
    echo "无效场景编号：$selected（应为 0001–0027）" >&2
    exit 2
  fi
done

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
DATA_ROOT="${ACTIVEVIEW_DATA_ROOT:-/mnt/disk_4/suyi/datasets/active_view_v1}"
RUN_ID="${RUN_ID:-preview_$(date +%Y%m%d_%H%M%S)}"
failed=()
for selected in "$@"; do
  printf -v scene 'scene_%04d' "$((10#$selected))"
  echo "=== Town05/$scene ==="
  if python -u -m active_view_v0.dataset_cli preview \
      --scene "Town05/$scene" --data-root "$DATA_ROOT" --run-id "$RUN_ID"; then
    echo "总览：$DATA_ROOT/Town05/$scene/preview/$RUN_ID/global_bev_rgb_annotated_timeline.png"
  else
    failed+=("$scene")
  fi
done
if (( ${#failed[@]} > 0 )); then
  echo "未完成预览：${failed[*]}" >&2
  exit 1
fi
