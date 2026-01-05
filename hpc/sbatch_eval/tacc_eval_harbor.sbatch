#!/bin/bash
#SBATCH -p {partition}
#SBATCH --time={time_limit}
#SBATCH --nodes {num_nodes}
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task={cpus_per_node}
#SBATCH --account {account}
#SBATCH --output={experiments_dir}/logs/%x_%j.out
#SBATCH --job-name={job_name}

set -euo pipefail

module load gcc/14.2.0
module load cuda/12.8
module load tacc-apptainer

if [ -n "${DCFT:-}" ] && [ -f "$DCFT/hpc/dotenv/tacc.env" ]; then
  source "$DCFT/hpc/dotenv/tacc.env"
fi
if [ -n "${DC_AGENT_SECRET_ENV:-}" ] && [ -f "$DC_AGENT_SECRET_ENV" ]; then
  source "$DC_AGENT_SECRET_ENV"
fi
if [ -n "${DCFT_ACTIVATE_ENV:-}" ]; then
  eval "$DCFT_ACTIVATE_ENV"
fi

WORKDIR="${DCFT_PRIVATE:-${DCFT:-$PWD}}"
cd "$WORKDIR"

EXPERIMENTS_DIR="$WORKDIR/{experiments_dir}"
mkdir -p "$EXPERIMENTS_DIR/logs" "$EXPERIMENTS_DIR/jobs"

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
VLLM_JOB_ID={vllm_job_id}

export EVAL_DATASET_PATH EVAL_DATASET_SLUG EVAL_HARBOR_CONFIG EVAL_AGENT EVAL_MODEL
export EVAL_ENV EVAL_N_CONCURRENT EVAL_N_ATTEMPTS EVAL_EXPECTED_TRIALS
export EVAL_BENCHMARK_REPO EVAL_UPLOAD_USERNAME EVAL_UPLOAD_MODE EVAL_HF_REPO_PREFIX
export AGENT_KWARGS_PATH TRACE_ENDPOINT_JSON TRACE_WAIT_FOR_ENDPOINT TRACE_HEALTH_MAX_ATTEMPTS TRACE_HEALTH_RETRY_DELAY VLLM_JOB_ID

echo "Working directory: $WORKDIR"
echo "Experiments directory: $EXPERIMENTS_DIR"

cleanup_vllm() {
  if [ -n "$VLLM_JOB_ID" ]; then
    echo "Tearing down VLLM job $VLLM_JOB_ID"
    scancel "$VLLM_JOB_ID" >/dev/null 2>&1 || echo "Warning: failed to cancel VLLM job $VLLM_JOB_ID"
  fi
}
trap cleanup_vllm EXIT

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
    print(f"{key}={value}")
PY
)
fi

if [ "$TRACE_WAIT_FOR_ENDPOINT" = "1" ] && [ -n "$TRACE_ENDPOINT_JSON" ] && [ -f "$TRACE_ENDPOINT_JSON" ]; then
  echo "Removing stale endpoint JSON: $TRACE_ENDPOINT_JSON"
  rm -f "$TRACE_ENDPOINT_JSON"
fi

if [ "$TRACE_WAIT_FOR_ENDPOINT" = "1" ]; then
  if [ -z "$TRACE_ENDPOINT_JSON" ]; then
    echo "ERROR: TRACE_WAIT_FOR_ENDPOINT is set, but TRACE_ENDPOINT_JSON is empty."
    exit 1
  fi
  echo "Waiting for eval endpoint JSON..."
  TRACE_ENDPOINT_WAIT=0
  TRACE_ENDPOINT_WAIT_MAX=600
  while [ ! -f "$TRACE_ENDPOINT_JSON" ] && [ $TRACE_ENDPOINT_WAIT -lt $TRACE_ENDPOINT_WAIT_MAX ]; do
    sleep 5
    TRACE_ENDPOINT_WAIT=$((TRACE_ENDPOINT_WAIT + 5))
    if [ $((TRACE_ENDPOINT_WAIT % 30)) -eq 0 ]; then
      echo "Still waiting for $TRACE_ENDPOINT_JSON... (${TRACE_ENDPOINT_WAIT}s elapsed)"
    fi
  done
  if [ ! -f "$TRACE_ENDPOINT_JSON" ]; then
    echo "ERROR: Trace endpoint JSON not found after ${TRACE_ENDPOINT_WAIT_MAX}s: $TRACE_ENDPOINT_JSON"
    exit 1
  fi
  echo "✓ Trace endpoint JSON present"
  python scripts/vllm/wait_for_endpoint.py \
    --endpoint-json "$TRACE_ENDPOINT_JSON" \
    --max-attempts "${TRACE_HEALTH_MAX_ATTEMPTS:-20}" \
    --retry-delay "${TRACE_HEALTH_RETRY_DELAY:-30}" \
    --health-path "v1/models"
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
    print(f"api_base={api_base}")
if metrics:
    print(f"metrics_endpoint={metrics}")
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
    safe_tag = f"{safe_tag[:88]}{digest}"

hf_repo_id = f"{prefix}/{safe_tag}"
print(f"[eval-upload] Uploading {run_dir} -> {hf_repo_id}")
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
