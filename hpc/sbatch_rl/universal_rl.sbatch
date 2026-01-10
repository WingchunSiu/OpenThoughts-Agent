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

# --- Ray logging (store all logs in experiments_dir) ---
export RAY_TMPDIR="$DCFT/{experiments_dir}/ray_tmp"
export RAY_LOG_DIR="$DCFT/{experiments_dir}/ray_logs"

# --- Distributed training setup ---
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT={master_port}
export NUM_NODES=$SLURM_JOB_NUM_NODES
export NUM_GPUS_PER_NODE={gpus_per_node}
export NUM_GPUS=$((NUM_GPUS_PER_NODE*SLURM_NNODES))

# --- HuggingFace/WandB paths ---
export HF_HOME="${HF_HOME:-${HF_HUB_CACHE:-/tmp/hf_home}}"
export WANDB_DIR="${DCFT_WANDB_DIR:-$DCFT/{experiments_dir}/wandb}"

# --- Create experiment directories ---
mkdir -p "$DCFT/{experiments_dir}"
mkdir -p "$DCFT/{experiments_dir}/logs"
mkdir -p "$DCFT/{experiments_dir}/exports"
mkdir -p "$DCFT/{experiments_dir}/ray_tmp"
mkdir -p "$DCFT/{experiments_dir}/ray_logs"

# --- SSH Tunneling (JSC clusters only) ---
{ssh_tunnel_setup}

# --- Ray cluster setup ---
RAY_PORT={ray_port}

echo "=== Ray Cluster Setup ==="
echo "MASTER_ADDR: $MASTER_ADDR"
echo "SLURM_NODEID: $SLURM_NODEID"
echo "NUM_NODES: $NUM_NODES"
echo "NUM_GPUS_PER_NODE: $NUM_GPUS_PER_NODE"

if [ "$SLURM_NODEID" = "0" ]; then
  echo "Starting Ray head node on $HOSTNAME..."
  ray start --head \
    --port=$RAY_PORT \
    --num-cpus={cpus_per_node} \
    --num-gpus={gpus_per_node} \
    --block &
  RAY_PID=$!
  sleep 15
  echo "Ray head started (PID: $RAY_PID)"
else
  echo "Starting Ray worker on $HOSTNAME, connecting to $MASTER_ADDR:$RAY_PORT..."
  sleep 10  # Wait for head to start
  ray start \
    --address="$MASTER_ADDR:$RAY_PORT" \
    --num-cpus={cpus_per_node} \
    --num-gpus={gpus_per_node} \
    --block &
  RAY_PID=$!
  sleep 5
  echo "Ray worker started (PID: $RAY_PID)"
fi

# --- Run SkyRL on head node only ---
if [ "$SLURM_NODEID" = "0" ]; then
  echo "=== Universal RL Training Runner ==="
  echo "Config: {config_path}"
  echo "Working directory: $WORKDIR"
  echo "Nodes: $NUM_NODES, GPUs/node: $NUM_GPUS_PER_NODE"
  echo "======================================"

  # Change to SkyRL directory if available
  if [ -n "${SKYRL_HOME:-}" ] && [ -d "$SKYRL_HOME/skyrl-train" ]; then
    cd "$SKYRL_HOME/skyrl-train"
    echo "Changed to SKYRL_HOME: $(pwd)"
  fi

  # Run SkyRL
  {skyrl_command}
  EXIT_CODE=$?

  echo "SkyRL exited with code: $EXIT_CODE"

  # Cleanup Ray
  ray stop
  exit $EXIT_CODE
else
  # Worker nodes just keep Ray running
  echo "Worker node $SLURM_NODEID waiting for head node..."
  wait $RAY_PID
fi
