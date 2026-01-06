#!/usr/bin/env python3
"""
Local trace generation runner.

Starts a single-node Ray cluster + vLLM controller and then launches a Harbor job
to generate traces from tasks. Designed for non-SLURM Linux hosts where we have
exclusive access to the box.

Usage:
    python run_tracegen.py \
        --harbor-config harbor_configs/default.yaml \
        --tasks-input-path /path/to/tasks \
        --datagen-config datagen_configs/my_config.yaml \
        --upload-hf-repo my-org/my-traces
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import pty
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENTS_DIR = REPO_ROOT / "trace_runs"
DEFAULT_ENDPOINT = "vllm_endpoint.json"

# Import shared utilities for local runners
from hpc.local_runner_utils import (
    ManagedProcess,
    maybe_int,
    start_ray,
    start_vllm_controller,
    wait_for_endpoint,
    terminate_processes,
    _build_vllm_cli_args,
)
from hpc.cli_utils import run_harbor_cli


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local trace generation with Ray/vLLM server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--harbor-config",
        required=True,
        help="Path to Harbor job config YAML.",
    )
    parser.add_argument(
        "--tasks-input-path",
        required=True,
        help="Path to tasks directory (input for trace generation).",
    )
    parser.add_argument(
        "--datagen-config",
        required=True,
        help="Path to datagen YAML with vLLM settings.",
    )
    parser.add_argument(
        "--model",
        help="Model identifier (overrides datagen config).",
    )
    parser.add_argument(
        "--agent",
        default="terminus-2",
        help="Harbor agent name to run (default: terminus-2).",
    )
    parser.add_argument(
        "--trace-env",
        default="daytona",
        help="Harbor environment name (default: daytona).",
    )
    parser.add_argument(
        "--n-concurrent",
        type=int,
        default=64,
        help="Concurrent trace trials (default: 64).",
    )
    parser.add_argument(
        "--n-attempts",
        type=int,
        default=3,
        help="Retries per task (default: 3).",
    )
    parser.add_argument(
        "--experiments-dir",
        default=str(DEFAULT_EXPERIMENTS_DIR),
        help="Directory for logs + endpoint JSON.",
    )
    parser.add_argument(
        "--job-name",
        help="Optional Harbor job name override.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host/IP for Ray + vLLM (default: 127.0.0.1).",
    )
    parser.add_argument("--ray-port", type=int, default=6379, help="Ray head port.")
    parser.add_argument("--api-port", type=int, default=8000, help="vLLM OpenAI server port.")
    parser.add_argument("--gpus", type=int, help="GPUs to expose to Ray.")
    parser.add_argument("--cpus", type=int, help="CPUs to expose to Ray.")
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--pipeline-parallel-size", type=int)
    parser.add_argument("--data-parallel-size", type=int)
    parser.add_argument("--health-max-attempts", type=int, default=20)
    parser.add_argument("--health-retry-delay", type=int, default=30)
    parser.add_argument("--harbor-binary", default="harbor", help="Harbor CLI executable.")
    parser.add_argument(
        "--agent-kwarg",
        action="append",
        default=[],
        help="Additional --agent-kwarg entries (key=value).",
    )
    parser.add_argument(
        "--controller-log",
        help="Optional path for vLLM controller stdout/stderr.",
    )
    parser.add_argument(
        "--ray-log",
        help="Optional path for Ray stdout/stderr.",
    )
    parser.add_argument(
        "--endpoint-json",
        help="Optional endpoint JSON path.",
    )
    parser.add_argument(
        "--harbor-log",
        help="Optional path for Harbor CLI stdout/stderr.",
    )
    parser.add_argument(
        "--harbor-extra-arg",
        action="append",
        default=[],
        help="Additional passthrough args for `harbor jobs start`.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing Harbor.",
    )
    # HuggingFace upload options
    parser.add_argument(
        "--upload-hf-repo",
        help="Hugging Face repo id to upload traces to (e.g., my-org/my-traces).",
    )
    parser.add_argument(
        "--upload-hf-token",
        help="Hugging Face token for upload (defaults to $HF_TOKEN).",
    )
    parser.add_argument(
        "--upload-hf-private",
        action="store_true",
        help="Create/overwrite the Hugging Face repo as private.",
    )
    return parser.parse_args()


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _default_job_name(tasks_label: str, model_label: str) -> str:
    sanitized_tasks = Path(tasks_label).name.replace("/", "-").replace(" ", "_")
    sanitized_model = model_label.replace("/", "-").replace(" ", "_")
    return f"tracegen-{sanitized_tasks}-{sanitized_model}-{_timestamp()}"


def _generate_served_model_id() -> str:
    return str(int(time.time() * 1_000_000))


def _hosted_vllm_alias(served_id: str) -> str:
    return f"hosted_vllm/{served_id}"


def _apply_datagen_defaults(args: argparse.Namespace) -> None:
    """Load datagen config and extract vLLM settings."""
    args._vllm_cli_args: List[str] = []
    args._vllm_env_vars: dict[str, str] = {}

    cfg_path = Path(args.datagen_config).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Datagen config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as handle:
        datagen_cfg = yaml.safe_load(handle) or {}
    args.datagen_config = str(cfg_path)

    engine_cfg = datagen_cfg.get("engine") or {}
    backend_cfg = datagen_cfg.get("backend") or {}
    vllm_cfg = datagen_cfg.get("vllm_server") or {}

    if args.model is None:
        args.model = vllm_cfg.get("model_path") or engine_cfg.get("model")

    tp_default = maybe_int(vllm_cfg.get("tensor_parallel_size")) or maybe_int(
        backend_cfg.get("tensor_parallel_size")
    )
    pp_default = maybe_int(vllm_cfg.get("pipeline_parallel_size")) or maybe_int(
        backend_cfg.get("pipeline_parallel_size")
    )
    dp_default = maybe_int(vllm_cfg.get("data_parallel_size")) or maybe_int(
        backend_cfg.get("data_parallel_size")
    )

    if args.tensor_parallel_size is None and tp_default:
        args.tensor_parallel_size = tp_default
    if args.pipeline_parallel_size is None and pp_default:
        args.pipeline_parallel_size = pp_default
    if args.data_parallel_size is None and dp_default:
        args.data_parallel_size = dp_default

    if args.ray_port is None:
        args.ray_port = maybe_int(backend_cfg.get("ray_port")) or 6379
    if args.api_port is None:
        args.api_port = maybe_int(backend_cfg.get("api_port")) or 8000

    # Build CLI args and env vars from vllm_server config
    merged_cfg = {**engine_cfg, **vllm_cfg}
    cli_args, env_vars = _build_vllm_cli_args(merged_cfg)
    args._vllm_cli_args = cli_args
    args._vllm_env_vars = env_vars


def _run_endpoint_health_check(
    endpoint_json: Path,
    attempts: int,
    delay: int,
) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "vllm" / "wait_for_endpoint.py"),
        "--endpoint-json",
        str(endpoint_json),
        "--max-attempts",
        str(attempts),
        "--retry-delay",
        str(delay),
        "--health-path",
        "v1/models",
    ]
    subprocess.run(cmd, check=True)


def _load_endpoint_metadata(endpoint_json: Path) -> dict:
    data = json.loads(endpoint_json.read_text())
    base_url = (data.get("endpoint_url") or "").rstrip("/")
    api_base = f"{base_url}/v1" if base_url else ""
    metrics = base_url.rstrip("/")
    if metrics.endswith("/v1"):
        metrics = metrics[:-3].rstrip("/")
    metrics = f"{metrics}/metrics" if metrics else ""
    data["api_base"] = api_base
    data["metrics_endpoint"] = metrics
    return data


def _parse_agent_kwarg_strings(entries: List[str]) -> tuple[dict[str, Any], List[str]]:
    overrides: dict[str, Any] = {}
    passthrough: List[str] = []
    for entry in entries:
        if "=" not in entry:
            passthrough.append(entry)
            continue
        key, raw_value = entry.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not key:
            passthrough.append(entry)
            continue
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        overrides[key] = value
    return overrides, passthrough


def _build_harbor_command(
    args: argparse.Namespace,
    endpoint_meta: dict,
) -> List[str]:
    harbor_model = getattr(args, "_harbor_model_name", args.model)
    tasks_label = args.tasks_input_path
    job_model_label = args.model or harbor_model or "model"
    job_name = args.job_name or _default_job_name(tasks_label, job_model_label)
    args._harbor_job_name = job_name

    # Build agent kwargs
    agent_kwargs: dict[str, Any] = {}
    if endpoint_meta.get("metrics_endpoint"):
        agent_kwargs["metrics_endpoint"] = endpoint_meta["metrics_endpoint"]
    if endpoint_meta.get("api_base"):
        agent_kwargs["api_base"] = endpoint_meta["api_base"]

    override_kwargs, passthrough = _parse_agent_kwarg_strings(list(args.agent_kwarg or []))
    agent_kwargs.update(override_kwargs)

    cmd = [
        args.harbor_binary,
        "jobs",
        "start",
        "--config",
        args.harbor_config,
        "--job-name",
        job_name,
        "--agent",
        args.agent,
        "--model",
        harbor_model,
        "--env",
        args.trace_env,
        "--n-concurrent",
        str(args.n_concurrent),
        "--n-attempts",
        str(args.n_attempts),
        "-p",
        args.tasks_input_path,
    ]

    # Add agent kwargs
    for key, value in agent_kwargs.items():
        if isinstance(value, (dict, list)):
            cmd.extend(["--agent-kwarg", f"{key}={json.dumps(value)}"])
        else:
            cmd.extend(["--agent-kwarg", f"{key}={value}"])

    for passthrough_kw in passthrough:
        cmd.extend(["--agent-kwarg", passthrough_kw])

    # Standard export flags for traces
    extra_args = list(args.harbor_extra_arg or [])

    def _flag_present(flag: str) -> bool:
        return any(arg == flag or arg.startswith(f"{flag}=") for arg in extra_args)

    if not (_flag_present("--export-traces") or _flag_present("--no-export-traces")):
        extra_args.append("--export-traces")
    if not (_flag_present("--export-verifier-metadata") or _flag_present("--no-export-verifier-metadata")):
        extra_args.append("--export-verifier-metadata")
    if not _flag_present("--export-episodes"):
        extra_args.extend(["--export-episodes", "last"])

    for extra in extra_args:
        cmd.append(extra)

    return cmd


def _upload_traces_to_hf(args: argparse.Namespace) -> None:
    """Upload generated traces to HuggingFace Hub."""
    hf_repo = args.upload_hf_repo
    if not hf_repo:
        print("[upload] No --upload-hf-repo specified, skipping HuggingFace upload.")
        return

    if args.dry_run:
        print("[upload] Skipping HuggingFace upload because --dry-run was set.")
        return

    hf_token = args.upload_hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("[upload] No HF token provided; skipping HuggingFace upload.")
        return

    job_name = getattr(args, "_harbor_job_name", None)
    jobs_dir_path = getattr(args, "_jobs_dir_path", None)
    if not job_name or jobs_dir_path is None:
        print("[upload] Unable to determine job directory; upload skipped.")
        return

    run_dir = Path(jobs_dir_path) / job_name
    traces_dir = run_dir / "traces"
    if not traces_dir.exists():
        print(f"[upload] Traces directory {traces_dir} does not exist; upload skipped.")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[upload] huggingface_hub not installed; skipping HuggingFace upload.")
        return

    print(f"[upload] Uploading traces from {traces_dir} to {hf_repo}")

    api = HfApi(token=hf_token)

    # Create repo if it doesn't exist
    try:
        api.create_repo(
            repo_id=hf_repo,
            repo_type="dataset",
            private=args.upload_hf_private,
            exist_ok=True,
        )
    except Exception as e:
        print(f"[upload] Warning: Could not create repo: {e}")

    # Upload the traces directory
    try:
        api.upload_folder(
            folder_path=str(traces_dir),
            repo_id=hf_repo,
            repo_type="dataset",
            path_in_repo="traces",
            commit_message=f"Upload traces from {job_name}",
        )
        print(f"[upload] Successfully uploaded traces to https://huggingface.co/datasets/{hf_repo}")
    except Exception as e:
        print(f"[upload] Failed to upload traces: {e}")


def _resolve_jobs_dir_path(jobs_dir_value: Optional[str]) -> Path:
    raw_value = jobs_dir_value or "jobs"
    path = Path(raw_value)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def main() -> None:
    args = _parse_args()
    _apply_datagen_defaults(args)

    # Set defaults
    if args.tensor_parallel_size is None:
        args.tensor_parallel_size = 1
    if args.pipeline_parallel_size is None:
        args.pipeline_parallel_size = 1
    if args.data_parallel_size is None:
        args.data_parallel_size = 1
    if args.model is None:
        raise ValueError("Provide --model or supply a datagen config with vllm_server.model_path.")

    served_model_id = _generate_served_model_id()
    args._served_model_id = served_model_id
    args._harbor_model_name = _hosted_vllm_alias(served_model_id)

    if args.gpus is None:
        args.gpus = max(
            1,
            args.tensor_parallel_size * args.pipeline_parallel_size * args.data_parallel_size,
        )
    if args.cpus is None:
        args.cpus = os.cpu_count() or 16

    args.harbor_config = str(Path(args.harbor_config).expanduser().resolve())
    args.tasks_input_path = str(Path(args.tasks_input_path).expanduser().resolve())

    # Load Harbor config to get jobs_dir
    harbor_config_data = {}
    try:
        with open(args.harbor_config, "r", encoding="utf-8") as harbor_handle:
            harbor_config_data = yaml.safe_load(harbor_handle) or {}
    except FileNotFoundError:
        harbor_config_data = {}

    jobs_dir_value = harbor_config_data.get("jobs_dir") if isinstance(harbor_config_data, dict) else None
    args._jobs_dir_path = _resolve_jobs_dir_path(jobs_dir_value)

    experiments_dir = Path(args.experiments_dir).expanduser().resolve()
    experiments_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = experiments_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    endpoint_json = Path(args.endpoint_json or (experiments_dir / DEFAULT_ENDPOINT))
    if endpoint_json.exists():
        endpoint_json.unlink()

    os.chdir(REPO_ROOT)

    ray_log = Path(args.ray_log) if args.ray_log else logs_dir / "ray.log"
    controller_log = Path(args.controller_log) if args.controller_log else logs_dir / "vllm_controller.log"
    harbor_log = Path(args.harbor_log).expanduser().resolve() if args.harbor_log else None

    processes: List[ManagedProcess] = []

    def _handle_signal(signum, _frame):
        print(f"\nSignal {signum} received; shutting down...", file=sys.stderr)
        terminate_processes(processes)
        subprocess.run(["ray", "stop", "--force"], check=False)
        sys.exit(1)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    print("=== Local Trace Generation ===")
    print(f"  Model: {args.model}")
    print(f"  Tasks: {args.tasks_input_path}")
    print(f"  TP/PP/DP: {args.tensor_parallel_size}/{args.pipeline_parallel_size}/{args.data_parallel_size}")
    print(f"  GPUs: {args.gpus}")
    print("==============================")

    controller_script = REPO_ROOT / "scripts" / "vllm" / "start_vllm_ray_controller.py"

    ray_proc = start_ray(
        host=args.host,
        ray_port=args.ray_port,
        num_gpus=args.gpus,
        num_cpus=args.cpus,
        log_path=ray_log,
    )
    processes.append(ray_proc)

    vllm_proc = start_vllm_controller(
        model=args.model,
        host=args.host,
        ray_port=args.ray_port,
        api_port=args.api_port,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        data_parallel_size=args.data_parallel_size,
        endpoint_path=endpoint_json,
        controller_script=controller_script,
        log_path=controller_log,
        served_model_name=getattr(args, "_served_model_id", None),
        extra_cli_args=getattr(args, "_vllm_cli_args", []),
        extra_env_vars=getattr(args, "_vllm_env_vars", {}),
    )
    processes.append(vllm_proc)

    try:
        wait_for_endpoint(endpoint_json, vllm_proc)
        _run_endpoint_health_check(endpoint_json, args.health_max_attempts, args.health_retry_delay)
        endpoint_meta = _load_endpoint_metadata(endpoint_json)
        harbor_cmd = _build_harbor_command(args, endpoint_meta)
        print("Harbor command:", " ".join(harbor_cmd))

        if not args.dry_run:
            run_harbor_cli(harbor_cmd, harbor_log)
            _upload_traces_to_hf(args)
        else:
            print("[dry-run] Would run Harbor and upload traces.")

    finally:
        terminate_processes(processes[::-1])
        subprocess.run(["ray", "stop", "--force"], check=False)


if __name__ == "__main__":
    main()
