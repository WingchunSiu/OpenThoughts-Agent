#!/usr/bin/env python3
"""
Local eval runner.

Starts a single-node Ray cluster + vLLM controller and then launches a Harbor eval
job that targets the freshly booted endpoint. Designed for non-SLURM Linux hosts
where we have exclusive access to the box.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENTS_DIR = REPO_ROOT / "eval_runs"
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


def _resolve_jobs_dir_path(jobs_dir_value: Optional[str]) -> Path:
    raw_value = jobs_dir_value or "jobs"
    path = Path(raw_value)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _ensure_database_module_path() -> None:
    db_path = REPO_ROOT / "database"
    if db_path.exists():
        db_path_str = str(db_path)
        if db_path_str not in sys.path:
            sys.path.insert(0, db_path_str)


def _sanitize_hf_repo_id(repo_id: str, max_length: int = 96) -> str:
    """
    Sanitize a Hugging Face repo_id to comply with naming rules.
    Based on the sbatch uploader helper used in HPC eval jobs.
    """

    def collapse(value: str) -> str:
        prev = None
        while value != prev:
            prev = value
            value = value.replace("--", "-").replace("..", ".")
        return value

    org, name = repo_id.split("/", 1) if "/" in repo_id else (None, repo_id)
    name = re.sub(r"[^A-Za-z0-9._-]", "-", name)
    name = collapse(name).strip("-.")
    if not name:
        name = "repo"
    limit = max_length - (len(org) + 1 if org else 0)
    if len(name) > limit > 8:
        digest = hashlib.sha1(name.encode()).hexdigest()[:8]
        keep = max(1, limit - len(digest))
        base = name[:keep].rstrip("-.") or "r"
        name = collapse(f"{base}{digest}").strip("-.")
    if name[0] in "-.":
        name = f"r{name[1:]}"
    if name[-1] in "-.":
        name = f"{name[:-1]}0"
    return f"{org}/{name}" if org else name


def _derive_default_hf_repo_id(args: argparse.Namespace, job_name: str) -> str:
    benchmark_repo = args.eval_benchmark_repo or ""
    if "/" in benchmark_repo:
        org = benchmark_repo.split("/", 1)[0]
    else:
        org = benchmark_repo or "openthoughts-agent"
    return f"{org}/{job_name}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Harbor evals against a local Ray/vLLM server.")
    parser.add_argument("--harbor-config", required=True, help="Path to Harbor eval YAML.")
    parser.add_argument(
        "--dataset",
        help="Harbor dataset slug (e.g., terminal-bench@2.0). Mutually exclusive with --dataset-path.",
    )
    parser.add_argument(
        "--dataset-path",
        help="Path to a Harbor task directory. Mutually exclusive with --dataset.",
    )
    parser.add_argument("--model", help="Trace model identifier (used for Harbor + vLLM).")
    parser.add_argument("--agent", help="Harbor agent name to run (default terminus-2).")
    parser.add_argument("--eval-env", default="daytona", help="Harbor environment name.")
    parser.add_argument("--n-concurrent", type=int, default=16, help="Concurrent eval trials.")
    parser.add_argument("--n-attempts", type=int, default=3, help="Retries (Harbor --n-attempts flag).")
    parser.add_argument("--eval-benchmark-repo", required=True, help="Supabase benchmark repo id.")
    parser.add_argument(
        "--experiments-dir",
        default=str(DEFAULT_EXPERIMENTS_DIR),
        help="Directory for logs + endpoint JSON.",
    )
    parser.add_argument("--job-name", help="Optional Harbor job name override.")
    parser.add_argument("--host", default="127.0.0.1", help="Host/IP for Ray + vLLM.")
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
        help="Optional endpoint JSON path. Defaults to <experiments_dir>/vllm_endpoint.json",
    )
    parser.add_argument(
        "--harbor-extra-arg",
        action="append",
        default=[],
        help="Additional passthrough args for `harbor jobs start`.",
    )
    parser.add_argument(
        "--harbor-log",
        help="Optional path for Harbor CLI stdout/stderr.",
    )
    parser.add_argument(
        "--datagen-config",
        help="Optional datagen YAML whose vLLM settings will seed defaults for this script.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing Harbor.",
    )
    parser.add_argument(
        "--upload-to-database",
        action="store_true",
        help="After Harbor finishes, upload traces + metrics to Supabase (same flow as HPC eval jobs).",
    )
    parser.add_argument(
        "--upload-username",
        help="Username to attribute in Supabase (defaults to $UPLOAD_USERNAME or current user).",
    )
    parser.add_argument(
        "--upload-error-mode",
        choices=["skip_on_error", "rollback_on_error"],
        default="skip_on_error",
        help="Supabase upload error handling (default: skip_on_error).",
    )
    parser.add_argument(
        "--upload-hf-repo",
        help="Hugging Face repo id used when uploading traces (defaults to <org>/<job_name>).",
    )
    parser.add_argument(
        "--upload-hf-token",
        help="Hugging Face token for trace upload (defaults to $HF_TOKEN).",
    )
    parser.add_argument(
        "--upload-hf-private",
        action="store_true",
        help="Create/overwrite the Hugging Face repo as private.",
    )
    parser.add_argument(
        "--upload-hf-episodes",
        choices=["last", "all"],
        default="last",
        help="Which episodes to upload to Hugging Face when pushing traces.",
    )
    parser.add_argument(
        "--upload-forced-update",
        action="store_true",
        help="Allow overwriting existing Supabase records for the same job.",
    )
    return parser.parse_args()


def _ensure_mutually_exclusive(dataset: Optional[str], dataset_path: Optional[str]) -> None:
    if dataset and dataset_path:
        raise ValueError("Specify either --dataset or --dataset-path (not both).")
    if not dataset and not dataset_path:
        raise ValueError("Must provide --dataset or --dataset-path.")


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _default_job_name(dataset_label: str, model_label: str) -> str:
    sanitized_dataset = dataset_label.replace("/", "-").replace(" ", "_")
    sanitized_model = model_label.replace("/", "-").replace(" ", "_")
    return f"eval-{sanitized_dataset}-{sanitized_model}-{_timestamp()}"


def _generate_served_model_id() -> str:
    return str(int(time.time() * 1_000_000))


def _hosted_vllm_alias(served_id: str) -> str:
    return f"hosted_vllm/{served_id}"


def _deep_copy(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _apply_nested_key(target: dict, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = target
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


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


def _serialize_agent_kwargs(kwargs: dict) -> List[str]:
    serialized: List[str] = []
    for key, value in kwargs.items():
        if isinstance(value, (dict, list)):
            serialized.append(f"{key}={json.dumps(value)}")
        else:
            serialized.append(f"{key}={value}")
    return serialized


def _apply_datagen_defaults(args: argparse.Namespace) -> None:
    args._vllm_cli_args: List[str] = []
    args._vllm_env_vars: dict[str, str] = {}
    if not args.datagen_config:
        return

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
        args.ray_port = maybe_int(backend_cfg.get("ray_port")) or args.ray_port
    if args.api_port is None:
        args.api_port = maybe_int(backend_cfg.get("api_port")) or args.api_port

    # Build CLI args and env vars from vllm_server config (pass-through to vLLM)
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


def _maybe_upload_results(args: argparse.Namespace) -> None:
    if not getattr(args, "upload_to_database", False):
        return
    if args.dry_run:
        print("[upload] Skipping Supabase upload because --dry-run was set.")
        return
    job_name = getattr(args, "_harbor_job_name", None)
    jobs_dir_path = getattr(args, "_jobs_dir_path", None)
    if not job_name or jobs_dir_path is None:
        print("[upload] Unable to determine job directory; upload skipped.")
        return
    run_dir = Path(jobs_dir_path) / job_name
    if not run_dir.exists():
        print(f"[upload] Expected Harbor job directory {run_dir} does not exist; upload skipped.")
        return

    _ensure_database_module_path()
    try:
        from unified_db.utils import upload_eval_results
    except ImportError as exc:
        raise RuntimeError(
            "Supabase upload helpers are unavailable. Install the database extras "
            "(e.g., `pip install .[cloud]`) or ensure dcagents-leaderboard is on PYTHONPATH."
        ) from exc

    username = args.upload_username or os.environ.get("UPLOAD_USERNAME") or getpass.getuser()
    hf_repo_id = args.upload_hf_repo or _derive_default_hf_repo_id(args, job_name)
    hf_token = args.upload_hf_token or os.environ.get("HF_TOKEN")
    if hf_repo_id:
        hf_repo_id = _sanitize_hf_repo_id(hf_repo_id)
    if hf_repo_id and not hf_token:
        print("[upload] HF repo requested but no token provided; skipping HF upload step.")
        hf_repo_id = None
    print(f"[upload] Uploading Harbor job '{job_name}' from {run_dir}")
    result = upload_eval_results(
        run_dir,
        username=username,
        error_mode=args.upload_error_mode,
        agent_name=args.agent,
        model_name=args.model,
        benchmark_name=args.dataset or args.dataset_path,
        register_benchmark=True,
        hf_repo_id=hf_repo_id,
        hf_private=args.upload_hf_private,
        hf_token=hf_token,
        hf_episodes=args.upload_hf_episodes,
        forced_update=args.upload_forced_update,
    )
    uploaded = result.get("n_trials_uploaded")
    job_id = result.get("job_id")
    hf_dataset = result.get("hf_dataset_url")
    print(
        f"[upload] Upload complete (job_id={job_id}, trials={uploaded}, hf_dataset={hf_dataset or 'n/a'})."
    )


def _build_harbor_command(
    args: argparse.Namespace,
    dataset_label: str,
    endpoint_meta: dict,
) -> List[str]:
    harbor_model = getattr(args, "_harbor_model_name", args.model)
    job_model_label = args.model or harbor_model or "model"
    job_name = args.job_name or _default_job_name(dataset_label, job_model_label)
    args._harbor_job_name = job_name
    base_agent_kwargs = _deep_copy(getattr(args, "_base_agent_kwargs", {}) or {})
    if endpoint_meta.get("metrics_endpoint"):
        base_agent_kwargs["metrics_endpoint"] = endpoint_meta["metrics_endpoint"]
    if endpoint_meta.get("api_base"):
        base_agent_kwargs["api_base"] = endpoint_meta["api_base"]
    override_kwargs, passthrough = _parse_agent_kwarg_strings(list(args.agent_kwarg or []))
    for dotted_key, override_value in override_kwargs.items():
        _apply_nested_key(base_agent_kwargs, dotted_key, override_value)
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
        args.eval_env,
        "--n-concurrent",
        str(args.n_concurrent),
        "--n-attempts",
        str(args.n_attempts),
    ]
    if args.dataset:
        cmd.extend(["--dataset", args.dataset])
    else:
        cmd.extend(["-p", args.dataset_path])
    serialized_kwargs = _serialize_agent_kwargs(base_agent_kwargs)
    for kw in serialized_kwargs:
        cmd.extend(["--agent-kwarg", kw])
    for passthrough_kw in passthrough:
        cmd.extend(["--agent-kwarg", passthrough_kw])
    extra_args = list(args.harbor_extra_arg or [])

    def _flag_present(flag: str) -> bool:
        return any(arg == flag or arg.startswith(f"{flag}=") for arg in extra_args)

    if not (_flag_present("--export-traces") or _flag_present("--no-export-traces")):
        extra_args.append("--export-traces")
    if not (
        _flag_present("--export-verifier-metadata") or _flag_present("--no-export-verifier-metadata")
    ):
        extra_args.append("--export-verifier-metadata")
    if not (_flag_present("--export-episodes")):
        extra_args.extend(["--export-episodes", "last"])

    for extra in extra_args:
        cmd.append(extra)

    return cmd


def _run_harbor_cli(cmd: List[str], log_path: Path | None) -> None:
    # Use shared PTY-based runner
    from hpc.cli_utils import run_harbor_cli
    run_harbor_cli(cmd, log_path)


def main() -> None:
    args = _parse_args()
    _apply_datagen_defaults(args)

    if args.agent is None:
        args.agent = "terminus-2"
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
    if args.ray_port is None:
        args.ray_port = 6379
    if args.api_port is None:
        args.api_port = 8000
    if args.gpus is None:
        args.gpus = max(
            1,
            args.tensor_parallel_size * args.pipeline_parallel_size * args.data_parallel_size,
        )
    if args.cpus is None:
        args.cpus = os.cpu_count() or 16
    args.harbor_config = str(Path(args.harbor_config).expanduser().resolve())
    harbor_config_data = {}
    try:
        with open(args.harbor_config, "r", encoding="utf-8") as harbor_handle:
            harbor_config_data = yaml.safe_load(harbor_handle) or {}
    except FileNotFoundError:
        harbor_config_data = {}
    if isinstance(harbor_config_data, dict):
        jobs_dir_value = harbor_config_data.get("jobs_dir")
    else:
        jobs_dir_value = None
    args._jobs_dir_path = _resolve_jobs_dir_path(jobs_dir_value)
    agents_def = harbor_config_data.get("agents") if isinstance(harbor_config_data, dict) else None
    first_agent = agents_def[0] if agents_def else {}
    base_agent_kwargs = first_agent.get("kwargs") if isinstance(first_agent, dict) else {}
    args._base_agent_kwargs = _deep_copy(base_agent_kwargs or {})
    if args.dataset_path:
        args.dataset_path = str(Path(args.dataset_path).expanduser().resolve())

    _ensure_mutually_exclusive(args.dataset, args.dataset_path)

    experiments_dir = Path(args.experiments_dir).expanduser().resolve()
    experiments_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = experiments_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    endpoint_json = Path(args.endpoint_json or (experiments_dir / DEFAULT_ENDPOINT))
    if endpoint_json.exists():
        endpoint_json.unlink()

    os.chdir(REPO_ROOT)

    dataset_label = args.dataset or args.dataset_path or "dataset"
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
        harbor_cmd = _build_harbor_command(args, dataset_label, endpoint_meta)
        print("Harbor command:", " ".join(harbor_cmd))
        if not args.dry_run:
            _run_harbor_cli(harbor_cmd, harbor_log)
            _maybe_upload_results(args)
        elif args.upload_to_database:
            print("[upload] --upload-to-database ignored because --dry-run was requested.")
    finally:
        terminate_processes(processes[::-1])
        subprocess.run(["ray", "stop", "--force"], check=False)


if __name__ == "__main__":
    main()
