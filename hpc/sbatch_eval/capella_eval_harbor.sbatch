#!/bin/bash
#SBATCH --account {account}
#SBATCH --partition {partition}
#SBATCH --nodes {num_nodes}
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task={cpus_per_node}
#SBATCH --gpus-per-node={gpus_per_node}
#SBATCH --time={time_limit}
#SBATCH --exclusive
#SBATCH --output={experiments_dir}/logs/%x_%j.out
#SBATCH --job-name={job_name}
#SBATCH --mail-type=END,TIME_LIMIT,FAIL
#SBATCH --mail-user=bf996@nyu.edu

set -euo pipefail

JOB_SUBMIT_ORDER=${JOB_SUBMIT_ORDER:-}
if [ -n "$JOB_SUBMIT_ORDER" ]; then
  if [[ "$JOB_SUBMIT_ORDER" =~ ^[0-9]+$ ]]; then
    SLEEP_DURATION=$((30 * JOB_SUBMIT_ORDER))
    echo "Submit order delay: sleeping ${SLEEP_DURATION}s (JOB_SUBMIT_ORDER=${JOB_SUBMIT_ORDER})"
    sleep "$SLEEP_DURATION"
  else
    echo "Warning: JOB_SUBMIT_ORDER='${JOB_SUBMIT_ORDER}' is not an integer; skipping launch delay"
  fi
fi

if [ -n "${DCFT:-}" ] && [ -f "$DCFT/hpc/dotenv/zih_capella.env" ]; then
  # shellcheck disable=SC1090
  source "$DCFT/hpc/dotenv/zih_capella.env"
elif [ -n "${DC_AGENT:-}" ] && [ -f "$DC_AGENT/hpc/dotenv/zih_capella.env" ]; then
  # shellcheck disable=SC1090
  source "$DC_AGENT/hpc/dotenv/zih_capella.env"
fi

module load CUDA/12.8.0

if [ -n "${DCFT_ACTIVATE_ENV:-}" ]; then
  set +u
  eval "$DCFT_ACTIVATE_ENV"
  set -u
fi

SECRET_FILE="${DC_AGENT_SECRET_ENV:-${KEYS:-}}"
if [[ -n "${SECRET_FILE}" ]]; then
    if [[ -f "${SECRET_FILE}" ]]; then
        echo "Sourcing secrets from ${SECRET_FILE}"
        set -a
        # shellcheck disable=SC1090
        source "${SECRET_FILE}"
        set +a
    else
        echo "Warning: secrets file ${SECRET_FILE} not found; eval uploads may fail." >&2
    fi
else
    echo "Warning: DC_AGENT_SECRET_ENV is not set; eval uploads may fail." >&2
fi

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=23
export HF_HOME=${HF_HOME:-$SCRATCH/.hf_home}
export OUTLINES_CACHE_DIR="${OUTLINES_CACHE_DIR:-/tmp/.outlines}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SCRATCH/triton_cache}"

if [ -n "$DCFT_PRIVATE" ]; then
  WORKDIR="$DCFT_PRIVATE"
elif [ -n "$DCFT" ]; then
  WORKDIR="$DCFT"
elif [ -n "$DC_AGENT" ]; then
  WORKDIR="$DC_AGENT"
else
  WORKDIR="$PWD"
fi
cd "$WORKDIR"

if [ -z "${DCFT:-}" ]; then
  export DCFT="$WORKDIR"
fi

EXPERIMENTS_DIR="$WORKDIR/{experiments_dir}"
mkdir -p "$EXPERIMENTS_DIR/logs" "$EXPERIMENTS_DIR/jobs"

export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"
export PATH="$WORKDIR/.local/bin:${PATH}"

# Preserve GPU ordinals for Ray
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}"
export RAY_NOSET_CUDA_VISIBLE_DEVICES="${RAY_NOSET_CUDA_VISIBLE_DEVICES:-1}"

RAY_TMPDIR_BASE="${SCRATCH:-/tmp}/ray_sessions"
RAY_TMPDIR="${RAY_TMPDIR_BASE}/ray_${SLURM_JOB_ID:-$$}"
mkdir -p "$RAY_TMPDIR"
echo "Ray session directory: $RAY_TMPDIR"

SRUN_EXPORT_ENV="ALL,PATH=$PATH,LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-},PYTHONPATH=$PYTHONPATH,HF_HOME=$HF_HOME,RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=$RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES,RAY_NOSET_CUDA_VISIBLE_DEVICES=$RAY_NOSET_CUDA_VISIBLE_DEVICES,RAY_TMPDIR=$RAY_TMPDIR"
RAY_ENV_VARS="PATH=$PATH LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-} PYTHONPATH=$PYTHONPATH HF_HOME=$HF_HOME RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=$RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES RAY_NOSET_CUDA_VISIBLE_DEVICES=$RAY_NOSET_CUDA_VISIBLE_DEVICES RAY_TMPDIR=$RAY_TMPDIR"

EVAL_DATASET_PATH={eval_dataset_path}
EVAL_DATASET_SLUG={harbor_dataset}
EVAL_HARBOR_CONFIG={eval_harbor_config}
EVAL_AGENT={eval_agent_name}
EVAL_MODEL={eval_model}
EVAL_ENV={eval_env}
EVAL_N_CONCURRENT={eval_n_concurrent}
EVAL_N_ATTEMPTS={eval_n_attempts}
EVAL_EXPECTED_TRIALS={eval_expected_trials}
EVAL_BENCHMARK_REPO={eval_benchmark_repo}
EVAL_UPLOAD_USERNAME={eval_upload_username}
EVAL_UPLOAD_MODE={eval_upload_mode}
EVAL_HF_REPO_PREFIX={eval_hf_repo_prefix}
AGENT_KWARGS_PATH={agent_kwargs_path}
TRACE_ENDPOINT_JSON={trace_endpoint_json}
TRACE_WAIT_FOR_ENDPOINT={trace_wait_for_endpoint}
TRACE_HEALTH_MAX_ATTEMPTS={trace_health_max_attempts}
TRACE_HEALTH_RETRY_DELAY={trace_health_retry_delay}
TRACE_REQUIRE_ENDPOINT={trace_require_endpoint}
TRACE_BACKEND={trace_backend}
TRACE_ENGINE={trace_engine}
TRACE_RAY_PORT={trace_ray_port}
TRACE_API_PORT={trace_api_port}
TRACE_TENSOR_PARALLEL_SIZE={trace_tensor_parallel_size}
TRACE_PIPELINE_PARALLEL_SIZE={trace_pipeline_parallel_size}
TRACE_DATA_PARALLEL_SIZE={trace_data_parallel_size}
TRACE_GPUS_PER_NODE={trace_gpus_per_node}
NEEDS_VLLM_SERVER={needs_vllm_server}

export EVAL_DATASET_PATH EVAL_DATASET_SLUG EVAL_HARBOR_CONFIG EVAL_AGENT EVAL_MODEL
export EVAL_ENV EVAL_N_CONCURRENT EVAL_N_ATTEMPTS EVAL_EXPECTED_TRIALS
export EVAL_BENCHMARK_REPO EVAL_UPLOAD_USERNAME EVAL_UPLOAD_MODE EVAL_HF_REPO_PREFIX
export AGENT_KWARGS_PATH TRACE_ENDPOINT_JSON TRACE_WAIT_FOR_ENDPOINT TRACE_HEALTH_MAX_ATTEMPTS TRACE_HEALTH_RETRY_DELAY
export TRACE_REQUIRE_ENDPOINT TRACE_BACKEND TRACE_ENGINE TRACE_RAY_PORT TRACE_API_PORT TRACE_TENSOR_PARALLEL_SIZE TRACE_PIPELINE_PARALLEL_SIZE TRACE_DATA_PARALLEL_SIZE TRACE_GPUS_PER_NODE NEEDS_VLLM_SERVER

echo "Working directory: $WORKDIR"
echo "Experiments directory: $EXPERIMENTS_DIR"

wait_for_external_endpoint() {
  if [ "$TRACE_WAIT_FOR_ENDPOINT" != "1" ]; then
    echo "Trace endpoint wait skipped (flag=$TRACE_WAIT_FOR_ENDPOINT)"
    return 0
  fi
  if [ -z "$TRACE_ENDPOINT_JSON" ]; then
    echo "ERROR: TRACE_ENDPOINT_JSON is empty but wait was requested."
    return 1
  fi
  if [ -f "$TRACE_ENDPOINT_JSON" ]; then
    echo "Removing stale endpoint JSON: $TRACE_ENDPOINT_JSON"
    rm -f "$TRACE_ENDPOINT_JSON"
  fi
  echo "Waiting for eval endpoint JSON..."
  local wait_elapsed=0
  local wait_max=600
  while [ ! -f "$TRACE_ENDPOINT_JSON" ] && [ $wait_elapsed -lt $wait_max ]; do
    sleep 5
    wait_elapsed=$((wait_elapsed + 5))
    if [ $((wait_elapsed % 30)) -eq 0 ]; then
      echo "Still waiting for $TRACE_ENDPOINT_JSON... (${wait_elapsed}s elapsed)"
    fi
  done
  if [ ! -f "$TRACE_ENDPOINT_JSON" ]; then
    echo "ERROR: Trace endpoint JSON not found after ${wait_max}s: $TRACE_ENDPOINT_JSON"
    return 1
  fi
  echo "✓ Trace endpoint JSON present"
  python scripts/vllm/wait_for_endpoint.py \
    --endpoint-json "$TRACE_ENDPOINT_JSON" \
    --max-attempts "${TRACE_HEALTH_MAX_ATTEMPTS:-20}" \
    --retry-delay "${TRACE_HEALTH_RETRY_DELAY:-30}" \
    --health-path "v1/models"
}

ray_pids=()
nodes_array=()
VLLM_PID=""

cleanup_inline_vllm() {
  if [ -n "${VLLM_PID:-}" ]; then
    kill "$VLLM_PID" >/dev/null 2>&1 || true
    wait "$VLLM_PID" >/dev/null 2>&1 || true
  fi
  if [ "${#nodes_array[@]}" -gt 0 ]; then
    for node in "${nodes_array[@]}"; do
      srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$node" ray stop --force >/dev/null 2>&1 || true
    done
  fi
  if [ "${#ray_pids[@]}" -gt 0 ]; then
    for pid in "${ray_pids[@]}"; do
      wait "$pid" >/dev/null 2>&1 || true
    done
  fi
  rm -rf "$RAY_TMPDIR" >/dev/null 2>&1 || true
}

launch_inline_vllm() {
  if [ "$TRACE_BACKEND" != "ray" ]; then
    echo "ERROR: Inline eval hosting currently requires trace_backend=ray."
    exit 1
  fi
  if [ -z "$TRACE_ENDPOINT_JSON" ]; then
    echo "ERROR: TRACE_ENDPOINT_JSON must be set for inline eval hosting."
    exit 1
  fi
  if [ -f "$TRACE_ENDPOINT_JSON" ]; then
    echo "Removing stale endpoint JSON: $TRACE_ENDPOINT_JSON"
    rm -f "$TRACE_ENDPOINT_JSON"
  fi
  local gpus_per_node="${TRACE_GPUS_PER_NODE:-0}"
  if [ -z "$gpus_per_node" ] || [ "$gpus_per_node" -le 0 ]; then
    echo "ERROR: TRACE_GPUS_PER_NODE must be > 0 when launching inline eval hosting."
    exit 1
  fi
  local nodes
  nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
  if [ -z "$nodes" ]; then
    echo "ERROR: Unable to resolve SLURM hostnames for inline eval hosting."
    exit 1
  fi
  read -r -a nodes_array <<<"$nodes"
  local head_node="${nodes_array[0]}"
  local head_node_ip
  head_node_ip=$(srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$head_node" hostname --ip-address)
  head_node_ip=${head_node_ip%% *}

  local ray_port="${TRACE_RAY_PORT:-6379}"
  local api_port="${TRACE_API_PORT:-8000}"
  local tp_size="${TRACE_TENSOR_PARALLEL_SIZE:-1}"
  local pp_size="${TRACE_PIPELINE_PARALLEL_SIZE:-1}"
  local dp_size="${TRACE_DATA_PARALLEL_SIZE:-1}"
  local num_nodes="$SLURM_JOB_NUM_NODES"
  export TRACE_GPUS_PER_NODE="$gpus_per_node"
  local total_gpus
  total_gpus=$(python - <<'PY'
import os
print(int(os.environ.get("SLURM_JOB_NUM_NODES", "1")) * int(os.environ.get("TRACE_GPUS_PER_NODE", "0")))
PY
)

  local ip_head="${head_node_ip}:${ray_port}"
  export RAY_ADDRESS="$ip_head"

  start_ray_node() {
    local node_name=$1
    local start_cmd=$2
    srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$node_name" bash -c "env $RAY_ENV_VARS $start_cmd" &
    ray_pids+=($!)
  }

  local head_cmd="ray start --head --node-ip-address=${head_node_ip} --port=${ray_port} --num-gpus=${gpus_per_node} --num-cpus={cpus_per_node} --block"
  echo "Starting Ray head on ${head_node} (${head_node_ip})"
  start_ray_node "$head_node" "$head_cmd"

  for ((i = 1; i < SLURM_JOB_NUM_NODES; i++)); do
    node_i=${nodes_array[$i]}
    worker_cmd="ray start --address ${ip_head} --num-gpus=${gpus_per_node} --num-cpus={cpus_per_node} --block"
    echo "Starting Ray worker on ${node_i}"
    start_ray_node "$node_i" "$worker_cmd"
    sleep 3
  done

  trap cleanup_inline_vllm EXIT

  echo "Waiting for Ray cluster resources..."
  python scripts/ray/wait_for_cluster.py \
    --address "$ip_head" \
    --expected-gpus "$total_gpus" \
    --expected-nodes "$num_nodes" \
    --timeout 600 \
    --poll-interval 10

  echo "Launching vLLM Ray controller on ${head_node_ip}:${api_port}"
  local controller_log="$EXPERIMENTS_DIR/logs/{job_name}_vllm.log"
  local controller_env="env PATH=$PATH LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-} HF_HOME=$HF_HOME PYTHONPATH=$PYTHONPATH RAY_TMPDIR=$RAY_TMPDIR"

  srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$head_node" \
    $controller_env \
    python scripts/vllm/start_vllm_ray_controller.py \
      --ray-address "$ip_head" \
      --host "$head_node_ip" \
      --port "$api_port" \
      --endpoint-json "$TRACE_ENDPOINT_JSON" \
      --tensor-parallel-size "$tp_size" \
      --pipeline-parallel-size "$pp_size" \
      --data-parallel-size "$dp_size" \
    >> "$controller_log" 2>&1 &
  VLLM_PID=$!

  echo "vLLM controller PID: $VLLM_PID (log: $controller_log)"

  local wait_elapsed=0
  local wait_timeout="${TRACE_ENDPOINT_WAIT_TIMEOUT:-180}"
  local wait_interval="${TRACE_ENDPOINT_WAIT_INTERVAL:-5}"

  while true; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      wait "$VLLM_PID" || true
      echo "ERROR: vLLM controller exited before writing endpoint JSON; see $controller_log"
      exit 1
    fi
    if [ -f "$TRACE_ENDPOINT_JSON" ]; then
      echo "✓ Trace endpoint JSON detected after ${wait_elapsed}s: $TRACE_ENDPOINT_JSON"
      break
    fi
    if [ "$wait_elapsed" -ge "$wait_timeout" ]; then
      echo "ERROR: Trace endpoint JSON not found within ${wait_timeout}s (controller PID $VLLM_PID still running)"
      exit 1
    fi
    sleep "$wait_interval"
    wait_elapsed=$((wait_elapsed + wait_interval))
    if [ $((wait_elapsed % 30)) -eq 0 ]; then
      echo "Waiting for trace endpoint JSON... ${wait_elapsed}s elapsed"
    fi
  done

  python scripts/vllm/wait_for_endpoint.py \
    --endpoint-json "$TRACE_ENDPOINT_JSON" \
    --max-attempts "${TRACE_HEALTH_MAX_ATTEMPTS:-20}" \
    --retry-delay "${TRACE_HEALTH_RETRY_DELAY:-60}" \
    --health-path "v1/models"
}

AGENT_KWARG_FLAGS=()
if [ -n "$AGENT_KWARGS_PATH" ] && [ -f "$AGENT_KWARGS_PATH" ]; then
  echo "Loading agent kwargs from $AGENT_KWARGS_PATH"
  while IFS= read -r line; do
    AGENT_KWARG_FLAGS+=("--agent-kwarg" "$line")
  done < <(python - <<'PY'
import json
import os
path = os.environ["AGENT_KWARGS_PATH"]
data = json.loads(open(path).read())
for key, value in data.items():
    print("{}={}".format(key, value))
PY
)
fi

if [ "$NEEDS_VLLM_SERVER" = "1" ]; then
  launch_inline_vllm
else
  if ! wait_for_external_endpoint; then
    exit 1
  fi
fi

if [ -n "$TRACE_ENDPOINT_JSON" ] && [ -f "$TRACE_ENDPOINT_JSON" ]; then
  has_api_base=0
  has_metrics_endpoint=0
  for kv in "${AGENT_KWARG_FLAGS[@]}"; do
    case "$kv" in
      api_base=*) has_api_base=1 ;;
      metrics_endpoint=*) has_metrics_endpoint=1 ;;
    esac
  done
  if [ $has_api_base -eq 0 ] || [ $has_metrics_endpoint -eq 0 ]; then
    endpoint_api_base=""
    endpoint_metrics_endpoint=""
    while IFS='=' read -r key value; do
      case "$key" in
        api_base) endpoint_api_base="$value" ;;
        metrics_endpoint) endpoint_metrics_endpoint="$value" ;;
      esac
    done < <(python - <<'PY'
import json
import os
path = os.environ.get("TRACE_ENDPOINT_JSON")
if not path:
    raise SystemExit(0)
try:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
except FileNotFoundError:
    raise SystemExit(0)
base_url = (data.get("endpoint_url") or "").rstrip("/")
api_path = (data.get("api_path") or "").strip()
if base_url and api_path:
    base_url = f"{base_url.rstrip('/')}/{api_path.lstrip('/')}"
api_base = f"{base_url.rstrip('/')}/v1" if base_url else ""
def derive_metrics(base: str) -> str:
    cleaned = base.rstrip("/")
    if cleaned.endswith("/v1"):
        cleaned = cleaned[:-3].rstrip("/")
    return f"{cleaned}/metrics" if cleaned else ""
metrics = derive_metrics(api_base) if api_base else ""
if api_base:
    print("api_base={}".format(api_base))
if metrics:
    print("metrics_endpoint={}".format(metrics))
PY
)
    if [ $has_api_base -eq 0 ] && [ -n "$endpoint_api_base" ]; then
      echo "Injecting api_base from endpoint JSON: $endpoint_api_base"
      AGENT_KWARG_FLAGS+=("--agent-kwarg" "api_base=$endpoint_api_base")
    fi
    if [ $has_metrics_endpoint -eq 0 ] && [ -n "$endpoint_metrics_endpoint" ]; then
      echo "Injecting metrics_endpoint from endpoint JSON: $endpoint_metrics_endpoint"
      AGENT_KWARG_FLAGS+=("--agent-kwarg" "metrics_endpoint=$endpoint_metrics_endpoint")
    fi
  fi
fi

DATASET_ARGS=()
if [ -n "$EVAL_DATASET_SLUG" ]; then
  DATASET_ARGS=(--dataset "$EVAL_DATASET_SLUG")
elif [ -n "$EVAL_DATASET_PATH" ]; then
  DATASET_ARGS=(-p "$EVAL_DATASET_PATH")
else
  echo "ERROR: Provide either --harbor-dataset or --eval-dataset-path/--trace-input-path"
  exit 2
fi

TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
SAFE_MODEL=$(echo "$EVAL_MODEL" | tr '/:' '_')
SAFE_DATASET=$(echo {dataset_label} | tr '/:' '_')
RUN_TAG="${SAFE_DATASET}_${SAFE_MODEL}_${TIMESTAMP}"
RUN_DIR="$EXPERIMENTS_DIR/jobs/$RUN_TAG"
mkdir -p "$RUN_DIR"
export RUN_TAG RUN_DIR

echo "Run tag: $RUN_TAG"
echo "Run dir: $RUN_DIR"

HARBOR_VERSION=$(python - <<'PY'
try:
    import harbor
    print(harbor.__version__)
except Exception:
    print("unknown")
PY
)
export HARBOR_VERSION=${HARBOR_VERSION:-unknown}

echo "Creating Supabase job row..."
DB_JOB_ID=$(python - <<'PY'
import json
import os
import sys
from database.unified_db.utils import create_job_entry_started

model = os.environ["EVAL_MODEL"]
benchmark = os.environ["EVAL_BENCHMARK_REPO"]
job_name = os.environ["RUN_TAG"]
username = os.environ["EVAL_UPLOAD_USERNAME"]
slurm_job_id = os.environ.get("SLURM_JOB_ID", "unknown")
agent = os.environ["EVAL_AGENT"]
harbor_version = os.environ.get("HARBOR_VERSION", "unknown")
n_trials = int(os.environ.get("EVAL_EXPECTED_TRIALS", "0") or "0")
n_attempts = int(os.environ.get("EVAL_N_ATTEMPTS", "1") or "1")

config = {"agent": agent, "env": os.environ.get("EVAL_ENV", "daytona")}
result = create_job_entry_started(
    model_hf_name=model,
    benchmark_hf_name=benchmark,
    job_name=job_name,
    username=username,
    slurm_job_id=slurm_job_id,
    agent_name=agent,
    config=config,
    n_trials=n_trials,
    n_rep_eval=n_attempts,
    harbor_package_version=harbor_version,
)
if not result.get("success"):
    print(result.get("error", "unknown error"), file=sys.stderr)
    sys.exit(1)
print(result["job"]["id"])
PY
) || {
  echo "Failed to create Supabase job entry"
  exit 1
}
export DB_JOB_ID

cat <<EOF > "$RUN_DIR/meta.env"
MODEL=$EVAL_MODEL
BENCHMARK_REPO=$EVAL_BENCHMARK_REPO
RUN_TAG=$RUN_TAG
SLURM_JOB_ID=$SLURM_JOB_ID
DB_JOB_ID=$DB_JOB_ID
EOF

HARBOR_CMD=(harbor jobs start "${DATASET_ARGS[@]}" --config "$EVAL_HARBOR_CONFIG" --job-name "$RUN_TAG" --agent "$EVAL_AGENT" --model "$EVAL_MODEL" --env "$EVAL_ENV" --n-concurrent "$EVAL_N_CONCURRENT" --k-attempts "$EVAL_N_ATTEMPTS")
if [ "${#AGENT_KWARG_FLAGS[@]}" -gt 0 ]; then
  HARBOR_CMD+=("${AGENT_KWARG_FLAGS[@]}")
fi

set +e
"${HARBOR_CMD[@]}"
SB_EXIT=$?
set -e

if [ ${SB_EXIT:-0} -ne 0 ]; then
  echo "Harbor exited with status ${SB_EXIT}"
  exit ${SB_EXIT}
fi

RESULT_FILE="$RUN_DIR/result.json"
DAYTONA_THRESHOLD=3

if [ -f "$RESULT_FILE" ]; then
  echo "Checking Daytona errors..."
  DAYTONA_COUNT=$(python - <<'PY' "$RESULT_FILE"
import json
import sys
path = sys.argv[1]
data = json.load(open(path))
total = 0
if "stats" in data and "evals" in data["stats"]:
    for eval_stats in data["stats"]["evals"].values():
        if "exception_stats" in eval_stats and "DaytonaError" in eval_stats["exception_stats"]:
            errors = eval_stats["exception_stats"]["DaytonaError"]
            if isinstance(errors, list):
                total += len(errors)
print(total)
PY
  )
  echo "Daytona errors: ${DAYTONA_COUNT}"
  if [ "${DAYTONA_COUNT:-0}" -gt "$DAYTONA_THRESHOLD" ]; then
    echo "Too many Daytona errors (${DAYTONA_COUNT}); skipping upload."
    exit 0
  fi
else
  echo "Warning: result.json not found at $RESULT_FILE"
fi

UPLOAD_LOG="$EXPERIMENTS_DIR/logs/upload_${SLURM_JOB_ID}.log"
mkdir -p "$(dirname "$UPLOAD_LOG")"

export RUN_DIR
export RUN_TAG
export DB_JOB_ID
export UPLOAD_USERNAME=$EVAL_UPLOAD_USERNAME
export UPLOAD_MODE=$EVAL_UPLOAD_MODE

set +e
python - <<'PY' 2>&1 | tee -a "$UPLOAD_LOG"
import os
import re
import hashlib
from unified_db.utils import upload_eval_results

run_dir = os.environ["RUN_DIR"]
run_tag = os.environ["RUN_TAG"]
username = os.environ.get("UPLOAD_USERNAME", "unknown")
error_mode = os.environ.get("UPLOAD_MODE", "skip_on_error")
prefix = os.environ.get("EVAL_HF_REPO_PREFIX", "DCAgent2")
hf_token = os.environ.get("HF_TOKEN")

if not hf_token:
    raise SystemExit("HF_TOKEN must be set for eval uploads.")

safe_tag = re.sub(r"[^A-Za-z0-9._-]+", "-", run_tag).strip("-")
if len(safe_tag) > 96:
    digest = hashlib.sha1(run_tag.encode()).hexdigest()[:8]
    safe_tag = "{}{}".format(safe_tag[:88], digest)

hf_repo_id = "{}/{}".format(prefix, safe_tag)
print("[eval-upload] Uploading {} -> {}".format(run_dir, hf_repo_id))
upload_eval_results(
    job_dir=run_dir,
    username=username,
    error_mode=error_mode,
    hf_repo_id=hf_repo_id,
    hf_token=hf_token,
    register_benchmark=True,
)
print("[eval-upload] Upload complete.")
PY
UPLOAD_EXIT=${PIPESTATUS[0]}
set -e
if [ $UPLOAD_EXIT -ne 0 ]; then
  echo "Upload failed with exit code: $UPLOAD_EXIT"
  exit $UPLOAD_EXIT
fi

echo "Eval workflow finished."
