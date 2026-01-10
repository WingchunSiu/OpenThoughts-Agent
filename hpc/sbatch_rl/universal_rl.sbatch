#!/bin/bash
#SBATCH --time={time_limit}
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus_per_node}
#SBATCH --output={experiments_dir}/logs/%x_%j.out
#SBATCH --job-name={job_name}
{sbatch_extra_directives}

# ==============================================================================
# Universal RL Training SBATCH Template
# ==============================================================================
# This template handles SkyRL-based reinforcement learning training jobs.
# It sets up Ray cluster, environment variables, and runs the SkyRL entrypoint.
#
# Usage: The launcher writes a JSON config file and substitutes {config_path}
# ==============================================================================

set -euo pipefail
ulimit -c 0  # Disable core dumps to avoid filling disk space

# Handle bash completion scripts that use BASH_COMPLETION_DEBUG
if [ -z "${BASH_COMPLETION_DEBUG+x}" ]; then
  export BASH_COMPLETION_DEBUG=""
fi

# Guard conda deactivate scripts from set -u complaints
export CONDA_BACKUP_CXX="${CONDA_BACKUP_CXX:-}"
export CONDA_BACKUP_CC="${CONDA_BACKUP_CC:-}"
export CONDA_BACKUP_FC="${CONDA_BACKUP_FC:-}"

# --- Module loading (cluster-specific, substituted by launcher) ---
set +u
{module_commands}
set -u

# --- Environment setup ---
if [ -n "${DCFT_PRIVATE:-}" ]; then
  WORKDIR="$DCFT_PRIVATE"
elif [ -n "${DCFT:-}" ]; then
  WORKDIR="$DCFT"
else
  WORKDIR="$PWD"
fi
cd "$WORKDIR"

if [ -z "${DCFT:-}" ]; then
  export DCFT="$WORKDIR"
fi

# --- Conda activation (cluster-specific, substituted by launcher) ---
set +u
{conda_activate}
set -u

# --- RL Environment Activation ---
# The RL environment is separate from datagen due to dependency conflicts.
# It's created by: ./hpc/setup_rl_env.sh
RL_ENV_DIR="${RL_ENV_DIR:-$WORKDIR/envs/rl}"
if [[ -d "$RL_ENV_DIR" ]]; then
  echo "Activating RL environment: $RL_ENV_DIR"
  source "$RL_ENV_DIR/bin/activate"
elif [[ -n "${DCFT_RL_ENV:-}" ]] && [[ -d "$DCFT_RL_ENV" ]]; then
  echo "Activating RL environment from DCFT_RL_ENV: $DCFT_RL_ENV"
  source "$DCFT_RL_ENV/bin/activate"
else
  echo "Warning: RL environment not found at $RL_ENV_DIR"
  echo "Run ./hpc/setup_rl_env.sh to create it, or set DCFT_RL_ENV"
fi

# --- Source environment files ---
if [ -n "${DCFT:-}" ] && [ -f "$DCFT/hpc/dotenv/{cluster_env_file}" ]; then
  source "$DCFT/hpc/dotenv/{cluster_env_file}"
fi
if [ -n "${DC_AGENT_SECRET_ENV:-}" ] && [ -f "$DC_AGENT_SECRET_ENV" ]; then
  set -a
  source "$DC_AGENT_SECRET_ENV"
  set +a
fi
if [ -n "${DCFT_ACTIVATE_ENV:-}" ]; then
  set +u
  eval "$DCFT_ACTIVATE_ENV"
  set -u
fi

# --- CUDA path detection (Perlmutter and similar) ---
{cuda_setup}

# --- NCCL/Networking settings (cluster-specific) ---
{nccl_exports}

# --- Standard environment variables ---
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"

# --- RL-specific environment variables ---
{rl_env_exports}

# --- vLLM/Ray settings ---
export VLLM_USE_V1=1
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook

# --- Hydra debugging ---
export HYDRA_FULL_ERROR=1

# --- Ray logging ---
# Use $SCRATCH for Ray temp/logs because Ray creates very long socket paths
# that can exceed the 107-byte Unix socket path limit (AF_UNIX).
export RAY_TMPDIR="${SCRATCH:-/tmp}/ray_tmp"
export RAY_LOG_DIR="${SCRATCH:-/tmp}/ray_logs"

# --- HuggingFace/WandB paths ---
export HF_HOME="${HF_HOME:-${HF_HUB_CACHE:-/tmp/hf_home}}"
export WANDB_DIR="${DCFT_WANDB_DIR:-{experiments_dir}/wandb}"

# --- Create experiment directories ---
# Note: {experiments_dir} is an absolute path substituted by the launcher
mkdir -p "{experiments_dir}"
mkdir -p "{experiments_dir}/logs"
mkdir -p "{experiments_dir}/exports"

# Ray directories in $SCRATCH (shorter paths for Unix socket limit)
mkdir -p "${SCRATCH:-/tmp}/ray_tmp"
mkdir -p "${SCRATCH:-/tmp}/ray_logs"

# --- SSH Tunneling (JSC clusters only) ---
{ssh_tunnel_setup}

# --- Clean up stale Ray instances on all nodes ---
# This prevents collisions with Ray sessions from previous/crashed jobs.
# Run on all allocated nodes using srun.
echo "Cleaning up stale Ray instances on all nodes..."
for node in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
  srun --export=ALL --nodes=1 --ntasks=1 --overlap -w "$node" \
    ray stop --force 2>/dev/null || true
done
sleep 2  # Brief pause to let processes terminate

# --- Run the RL training job via Python runner ---
# Ray cluster management is handled by hpc/ray_utils.py using proper srun calls
echo "=== Universal RL Training Runner ==="
echo "Config: {config_path}"
echo "Working directory: $WORKDIR"
echo "========================================"

python -m hpc.rl_launch_utils --config "{config_path}"
