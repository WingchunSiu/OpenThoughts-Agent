#!/bin/bash
export TRACE_SCRIPT=data/swesmith_with_plain_docker/generate_abstract.py
export TRACE_STAGE=traces
export TRACE_TASKS_PATH=/Users/benjaminfeuer/Documents/stackexchange-overflow-sandboxes
export TRACE_TARGET_REPO=DCAgent2/gemini-3-pro-preview-stackexchange-overflow-32ep-512k-v3-traces
export TRACE_MODEL=gemini/gemini-3-pro-preview
export TRACE_ENGINE=google_gemini
export TRACE_OUTPUT_DIR=./experiments/datagen-gemini-3-pro-preview-stackexchange-overflow-32ep-512k-v3/outputs/traces
export TRACE_USE_GPU=0
export TRACE_EPISODES=last
export TRACE_EXPORT_FILTER=none
export TRACE_DATASET_TYPE=SFT
export TRACE_EXPORT_SUBAGENTS=1
export TRACE_JOBS_DIR=/Users/benjaminfeuer/Documents/OpenThoughts-Agent/experiments/datagen-gemini-3-pro-preview-stackexchange-overflow-32ep-512k-v3/trace_jobs
export TRACE_REQUIRE_ENDPOINT=0
export TRACE_WAIT_FOR_ENDPOINT=0
export TRACE_HEALTH_MAX_ATTEMPTS=20
export TRACE_HEALTH_RETRY_DELAY=30
export TRACE_DISABLE_VERIFICATION=0
export TRACE_EVAL_ONLY=0
export TRACE_NUM_GPUS=0
export TRACE_BACKEND=none
export TRACE_TOTAL_TASKS=10000
export TRACE_CHUNK_COUNT=1
export TRACE_JOB_INDEX=0
export TRACE_TARGET_REPO_BASE=DCAgent2/gemini-3-pro-preview-stackexchange-overflow-32ep-512k-v3-traces
export TRACE_HARBOR_CONFIG=experiments/datagen-gemini-3-pro-preview-stackexchange-overflow-32ep-512k-v3/configs/trace/datagen_engine_gemini-3-pro-preview_trace.yaml
export PROJECT_ROOT=/Users/benjaminfeuer/Documents/OpenThoughts-Agent
export PYTHON_EXECUTABLE=/Users/benjaminfeuer/miniconda3/envs/llama-factory/bin/python
export TRACE_ENGINE_CONFIG_PATH=experiments/datagen-gemini-3-pro-preview-stackexchange-overflow-32ep-512k-v3/configs/datagen_config.resolved.yaml
export TRACE_RAY_PORT=6379
export TRACE_API_PORT=8000
export TRACE_TENSOR_PARALLEL_SIZE=1
export TRACE_PIPELINE_PARALLEL_SIZE=1
export TRACE_DATA_PARALLEL_SIZE=1
export TRACE_NUM_NODES=1
export TRACE_GPUS_PER_NODE=0
export SANDBOX_CPU=1
export SANDBOX_MEMORY_GB=1
export SANDBOX_DISK_GB=3
export DAYTONA_API_KEY=dtn_2cf3cc9807d66a3402be8829c1a85f90ae8681f9ca3ead8711cc25d053c5600f
export TRACE_CPUS_PER_NODE=16
set -euo pipefail

echo "=== Local Trace Generation (dip) ==="

WORKDIR="${DCFT:-${PROJECT_ROOT:-$PWD}}"
cd "$WORKDIR"
export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"

SCRIPT="${TRACE_SCRIPT:?TRACE_SCRIPT must be set}"
TASKS_PATH="${TRACE_TASKS_PATH:?TRACE_TASKS_PATH must be set}"
TARGET_REPO="${TRACE_TARGET_REPO:?TRACE_TARGET_REPO must be set}"
OUTPUT_DIR="${TRACE_OUTPUT_DIR:-}"
HARDCFG="${TRACE_HARBOR_CONFIG:-}"

source "$WORKDIR/hpc/sbatch_data/trace_helpers.sh"
load_trace_chunk_env
ensure_tasks_ready_or_exit "$TASKS_PATH" "${TRACE_TASKS_READY_TIMEOUT:-1800}"

if [ -n "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

if [ "${TRACE_REQUIRE_ENDPOINT:-0}" = "1" ]; then
    echo "ERROR: TRACE_REQUIRE_ENDPOINT=1 is not supported for local execution." >&2
    exit 1
fi

CMD=("${PYTHON_EXECUTABLE:-python}" "$SCRIPT" --stage "${TRACE_STAGE:-traces}" --target-repo "$TARGET_REPO" --tasks-input "$TASKS_PATH")

if [ -n "$OUTPUT_DIR" ]; then
    CMD+=(--output-dir "$OUTPUT_DIR")
fi
if [ -n "$HARDCFG" ]; then
    CMD+=(--trace-harbor-config "$HARDCFG")
fi
if [ -n "${TRACE_JOBS_DIR:-}" ]; then
    CMD+=(--trace-jobs-dir "$TRACE_JOBS_DIR")
fi
if [ -n "${TRACE_MODEL:-}" ]; then
    CMD+=(--trace-model "$TRACE_MODEL")
fi
if [ -n "${TRACE_EPISODES:-}" ]; then
    CMD+=(--trace-episodes "$TRACE_EPISODES")
fi
if [ -n "${TRACE_EXPORT_FILTER:-}" ]; then
    CMD+=(--trace-export-filter "$TRACE_EXPORT_FILTER")
fi
if [ -n "${TRACE_DATASET_TYPE:-}" ]; then
    CMD+=(--trace-dataset-type "$TRACE_DATASET_TYPE")
fi
if [ "${TRACE_EXPORT_SUBAGENTS:-1}" = "1" ]; then
    CMD+=(--trace-export-subagents)
else
    CMD+=(--trace-skip-subagents)
fi
if [ -n "${TRACE_ENGINE:-}" ]; then
    CMD+=(--engine "$TRACE_ENGINE")
fi
if [ -n "${TRACE_AGENT_TIMEOUT_SEC:-}" ]; then
    CMD+=(--trace-agent-timeout-sec "$TRACE_AGENT_TIMEOUT_SEC")
fi
if [ -n "${TRACE_VERIFIER_TIMEOUT_SEC:-}" ]; then
    CMD+=(--trace-verifier-timeout-sec "$TRACE_VERIFIER_TIMEOUT_SEC")
fi
if [ "$TRACE_DISABLE_VERIFICATION" = "1" ]; then
    CMD+=(--disable-verification)
fi
if [ -n "${TRACE_ENGINE_CONFIG_PATH:-}" ]; then
    CMD+=(--engine-config "$TRACE_ENGINE_CONFIG_PATH")
fi
if [ "$TRACE_EVAL_ONLY" = "1" ]; then
    CMD+=(--trace-eval-only)
fi
if [ -n "${TRACE_TASK_TYPE:-}" ]; then
    CMD+=(--task-type "$TRACE_TASK_TYPE")
fi
# Local execution cannot support SLURM chunk arrays.
if [ "${TRACE_CHUNK_MODE:-0}" = "1" ]; then
    echo "ERROR: TRACE chunking (--chunk_size) is not supported for local execution." >&2
    exit 1
fi

echo "Running trace locally: ${CMD[*]}"
"${CMD[@]}"
