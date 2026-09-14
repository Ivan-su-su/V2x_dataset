#!/usr/bin/env bash
set -euo pipefail

ACTIVE_ROOT="${ACTIVE_ROOT:-/mnt/disk_4/suyi/active_airv2x}"
CONDA_EXE_PATH="${CONDA_EXE_PATH:-/home/suyi/miniconda3/bin/conda}"
DETECT_ENV="${DETECT_ENV:-${ACTIVE_ROOT}/conda_envs/activeair-det}"
AIRV2X_REPO="${AIRV2X_REPO:-${ACTIVE_ROOT}/code/Airv2x_gs}"
ACTIVE_VIEW_REPO="${ACTIVE_VIEW_REPO:-${ACTIVE_ROOT}/code/active_view_v0}"
MODEL_DIR="${MODEL_DIR:-${ACTIVE_ROOT}/checkpoints/airv2x_intermediate_where2comm/release}"

if [[ ! -x "${CONDA_EXE_PATH}" ]]; then
    echo "conda executable not found: ${CONDA_EXE_PATH}" >&2
    exit 1
fi
if [[ ! -f "${ACTIVE_VIEW_REPO}/scripts/requirements-where2comm-smoke.txt" ]]; then
    echo "active_view_v0 smoke requirements not found under ${ACTIVE_VIEW_REPO}" >&2
    exit 1
fi

eval "$("${CONDA_EXE_PATH}" shell.bash hook)"

if [[ ! -x "${DETECT_ENV}/bin/python" ]]; then
    conda create \
        -p "${DETECT_ENV}" \
        --override-channels \
        -c conda-forge \
        python=3.10 \
        pip=23.3 \
        setuptools=68 \
        wheel \
        cmake=3.27 \
        -y
fi

conda activate "${DETECT_ENV}"
python -m pip install \
    torch==1.13.1+cu117 \
    torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install spconv-cu117==2.3.6
python -m pip install \
    -r "${ACTIVE_VIEW_REPO}/scripts/requirements-where2comm-smoke.txt"
python -m pip install \
    "git+https://github.com/klintan/pypcd.git"

if [[ ! -d "${AIRV2X_REPO}/.git" ]]; then
    git clone \
        --branch 0911version \
        --single-branch \
        https://github.com/Ivan-su-su/Airv2x_gs.git \
        "${AIRV2X_REPO}"
else
    CURRENT_BRANCH="$(git -C "${AIRV2X_REPO}" branch --show-current)"
    if [[ "${CURRENT_BRANCH}" != "0911version" ]]; then
        echo "Airv2x_gs is on ${CURRENT_BRANCH}; expected 0911version." >&2
        echo "Do not silently switch a working tree with user changes." >&2
        exit 1
    fi
fi

# Avoid `pip install -r Airv2x_gs/requirements.txt`: that file pins packages
# from the original Python-3.7 environment.  PYTHONPATH is sufficient here.
cd "${AIRV2X_REPO}"
python opencood/utils/setup.py build_ext --inplace

cd "${ACTIVE_VIEW_REPO}"
python -m pip install -e . --no-deps

mkdir -p "${MODEL_DIR}"
if [[ ! -s "${MODEL_DIR}/config.yaml" ]]; then
    curl -L --retry 20 --retry-all-errors \
        -o "${MODEL_DIR}/config.yaml" \
        https://huggingface.co/xiangbog/AirV2X-Perception-Checkpoints/resolve/main/airv2x_intermediate_where2comm/release/config.yaml
fi
if [[ ! -s "${MODEL_DIR}/net_epoch16.pth" ]]; then
    curl -L --retry 20 --retry-all-errors \
        -o "${MODEL_DIR}/net_epoch16.pth" \
        https://huggingface.co/xiangbog/AirV2X-Perception-Checkpoints/resolve/main/airv2x_intermediate_where2comm/release/net_epoch16.pth
fi
echo "4f0ce7c234489d6a1540944e6f656e0668e2917eea8d29b1da03885c638ceb2b  ${MODEL_DIR}/net_epoch16.pth" \
    | sha256sum -c -

PYTHONPATH="${AIRV2X_REPO}:${ACTIVE_VIEW_REPO}/src" python - <<'PY'
import torch
import torchvision
import spconv.pytorch as spconv
from cumm import tensorview
import open3d
from pypcd import pypcd

from opencood.models.airv2x_where2com import Airv2xWhere2com
from opencood.utils.box_overlaps import bbox_overlaps
from active_view_v0.where2comm_eval import carla_points_to_ego

assert torch.cuda.is_available(), "PyTorch cannot see CUDA"
print("Environment check OK")
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("spconv:", spconv.__version__ if hasattr(spconv, "__version__") else "imported")
print("GPU count:", torch.cuda.device_count())
print("GPU 0:", torch.cuda.get_device_name(0))
PY

echo
echo "WHERE2COMM ENVIRONMENT READY"
echo "Activate with: conda activate ${DETECT_ENV}"
echo "Checkpoint dir: ${MODEL_DIR}"
