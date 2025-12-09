#!/bin/bash
#SBATCH --job-name=build-flash-attn
#SBATCH --account=torch_pr_40_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G

# Batch job that compiles and installs flash-attention from source inside the
# requested conda environment. The build runs with --no-build-isolation so that
# it can re-use the already provisioned CUDA/Torch stack.

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-dcagent}"
FLASH_ATTN_REPO="${FLASH_ATTN_REPO:-https://github.com/Dao-AILab/flash-attention.git}"
FLASH_ATTN_REF="${FLASH_ATTN_REF:-main}"
FLASH_ATTN_BUILD_DIR="${FLASH_ATTN_BUILD_DIR:-${SCRATCH:-${HOME}}/flash-attention}"
PIP_EXTRA_ARGS="${PIP_EXTRA_ARGS:-}"

_activate_conda() {
    local conda_root=""
    if [[ -n "${SCRATCH:-}" && -d "${SCRATCH}/miniconda3" ]]; then
        conda_root="${SCRATCH}/miniconda3"
    elif [[ -d "${HOME}/miniconda3" ]]; then
        conda_root="${HOME}/miniconda3"
    fi

    if [[ -z "$conda_root" ]]; then
        echo "ERROR: Could not find miniconda3 under \$SCRATCH or \$HOME." >&2
        exit 1
    fi

    # shellcheck disable=SC1090
    source "${conda_root}/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
}

_sync_repo() {
    local repo_path="$1"
    if [[ ! -d "$repo_path/.git" ]]; then
        echo "Cloning $FLASH_ATTN_REPO to $repo_path"
        git clone "$FLASH_ATTN_REPO" "$repo_path"
    fi

    echo "Checking out ${FLASH_ATTN_REF}"
    git -C "$repo_path" fetch --all --tags
    git -C "$repo_path" checkout "$FLASH_ATTN_REF"
    git -C "$repo_path" submodule update --init --recursive
}

echo "=== Activating conda environment (${CONDA_ENV}) ==="
_activate_conda

echo
echo "=== Runtime environment ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "FLASH_ATTN_REPO=${FLASH_ATTN_REPO}"
echo "FLASH_ATTN_REF=${FLASH_ATTN_REF}"
echo "FLASH_ATTN_BUILD_DIR=${FLASH_ATTN_BUILD_DIR}"
echo
nvidia-smi || true

echo
echo "=== CUDA paths ==="
echo "CUDA_HOME=${CUDA_HOME:-<unset>}"
echo "CUDA_PATH=${CUDA_PATH:-<unset>}"
echo "CUDATOOLKIT_ROOT_DIR=${CUDATOOLKIT_ROOT_DIR:-<unset>}"
echo "PATH entries with CUDA:"
tr ':' '\n' <<< "${PATH}" | grep -i 'cuda' || true
if command -v nvcc >/dev/null 2>&1; then
    echo "nvcc=$(command -v nvcc)"
    nvcc --version || true
else
    echo "nvcc=<not found in PATH>"
fi

mkdir -p "$FLASH_ATTN_BUILD_DIR"
_sync_repo "$FLASH_ATTN_BUILD_DIR"

# Default to Hopper (SM90) if the user did not provide an explicit arch list.
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    export TORCH_CUDA_ARCH_LIST="90"
fi

echo
echo "=== Building flash-attention (TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}) ==="
pushd "$FLASH_ATTN_BUILD_DIR" >/dev/null
python -m pip install --no-build-isolation -v ${PIP_EXTRA_ARGS} .
popd >/dev/null

echo
echo "=== Installed flash-attention version ==="
python - <<'PY'
import importlib
import pkg_resources
name = "flash-attn"
spec = importlib.util.find_spec("flash_attn")
print("Module path:", getattr(spec, "origin", spec))
dist = pkg_resources.get_distribution(name)
print("Installed distribution:", dist)
PY

echo
echo "flash-attention build completed successfully."
