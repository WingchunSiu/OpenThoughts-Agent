"""Shared utilities for local Ray/vLLM runners.

This module consolidates common code used by:
- eval/local/run_eval.py
- data/local/run_tracegen.py

It provides managed subprocess handling for Ray clusters and vLLM servers,
datagen config parsing, Docker runtime setup, and Harbor command building.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from hpc.vllm_utils import _build_vllm_cli_args
from hpc.launch_utils import generate_served_model_id, hosted_vllm_alias
from hpc.arg_groups import (
    add_harbor_args,
    add_model_compute_args,
    add_ray_vllm_args,
    add_log_path_args,
)

# Re-export model-specific utilities for backward compatibility
from hpc.model_utils import (
    is_gpt_oss_model,
    setup_gpt_oss_tiktoken,
    get_model_specific_env_vars,
    GPT_OSS_TIKTOKEN_FILES,
)

# Re-export harbor utilities for backward compatibility
from hpc.harbor_utils import (
    load_harbor_config,
    get_harbor_env_from_config,
    extract_agent_kwargs_from_config,
    apply_nested_key,
    parse_agent_kwarg_strings,
    serialize_agent_kwargs,
    default_job_name,
    build_harbor_command,
)


@dataclass
class ManagedProcess:
    """A subprocess with graceful shutdown support."""

    name: str
    proc: subprocess.Popen
    _log_handle: Optional[object] = field(default=None, repr=False)

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the process gracefully, falling back to kill if needed."""
        if self.proc.poll() is not None:
            return
        try:
            self.proc.terminate()
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        finally:
            if self._log_handle:
                try:
                    self._log_handle.close()
                except Exception:
                    pass


def maybe_int(value: object) -> Optional[int]:
    """Parse a value as int, returning None if not possible."""
    if value in (None, "", "None"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _open_log_file(log_path: Optional[Path]) -> tuple:
    """Open a log file with line buffering for real-time tail access.

    Returns:
        Tuple of (stdout_dest, stderr_dest, log_file_handle)
    """
    if log_path:
        log_file = open(log_path, "w", encoding="utf-8", buffering=1)
        return log_file, log_file, log_file
    return None, None, None


def start_ray(
    host: str,
    ray_port: int,
    num_gpus: int,
    num_cpus: int,
    log_path: Optional[Path] = None,
) -> ManagedProcess:
    """Start a single-node Ray cluster head.

    Args:
        host: IP address to bind to
        ray_port: Port for Ray head node
        num_gpus: Number of GPUs to expose
        num_cpus: Number of CPUs to expose
        log_path: Optional path for Ray logs (line-buffered)

    Returns:
        ManagedProcess wrapping the Ray head process
    """
    cmd = [
        "ray",
        "start",
        "--head",
        f"--node-ip-address={host}",
        f"--port={ray_port}",
        f"--num-gpus={num_gpus}",
        f"--num-cpus={num_cpus}",
        "--dashboard-host=0.0.0.0",
        "--block",
    ]

    env = os.environ.copy()
    stdout, stderr, log_file = _open_log_file(log_path)

    popen = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, env=env)
    process = ManagedProcess(name="ray", proc=popen, _log_handle=log_file)
    return process


def start_vllm_controller(
    model: str,
    host: str,
    ray_port: int,
    api_port: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    data_parallel_size: int,
    endpoint_path: Path,
    controller_script: Path,
    log_path: Optional[Path] = None,
    served_model_name: Optional[str] = None,
    extra_cli_args: Optional[List[str]] = None,
    extra_env_vars: Optional[dict] = None,
) -> ManagedProcess:
    """Start a vLLM controller process.

    Args:
        model: Model path/name for vLLM
        host: IP address to bind to
        ray_port: Ray head port to connect to
        api_port: Port for vLLM OpenAI-compatible API
        tensor_parallel_size: Number of GPUs for tensor parallelism
        pipeline_parallel_size: Number of pipeline stages
        data_parallel_size: Number of data parallel replicas
        endpoint_path: Path to write endpoint JSON
        controller_script: Path to start_vllm_ray_controller.py
        log_path: Optional path for vLLM logs (line-buffered)
        served_model_name: Optional custom model name for the API
        extra_cli_args: Additional CLI args to pass to vLLM
        extra_env_vars: Additional environment variables

    Returns:
        ManagedProcess wrapping the vLLM controller process
    """
    env = os.environ.copy()
    env["VLLM_MODEL_PATH"] = model
    env["PYTHONUNBUFFERED"] = "1"  # Ensure real-time log output

    if extra_env_vars:
        env.update(extra_env_vars)

    cmd = [
        sys.executable,
        str(controller_script),
        "--ray-address",
        f"{host}:{ray_port}",
        "--host",
        host,
        "--port",
        str(api_port),
        "--model",
        model,
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--pipeline-parallel-size",
        str(pipeline_parallel_size),
        "--data-parallel-size",
        str(data_parallel_size),
        "--endpoint-json",
        str(endpoint_path),
    ]

    if served_model_name:
        cmd.extend(["--served-model-name", served_model_name])

    if extra_cli_args:
        cmd.extend(extra_cli_args)

    stdout, stderr, log_file = _open_log_file(log_path)

    popen = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, env=env)
    process = ManagedProcess(name="vllm_controller", proc=popen, _log_handle=log_file)
    return process


def wait_for_endpoint(
    endpoint_path: Path,
    controller: ManagedProcess,
    timeout: int = 300,
) -> None:
    """Wait for the vLLM endpoint JSON file to be created.

    Args:
        endpoint_path: Path to the endpoint JSON file
        controller: The vLLM controller process to monitor
        timeout: Maximum seconds to wait

    Raises:
        RuntimeError: If the controller exits before creating the endpoint
        TimeoutError: If the endpoint is not created within timeout
    """
    start = time.time()
    while time.time() - start < timeout:
        if controller.proc.poll() is not None:
            raise RuntimeError(
                "vLLM controller exited before writing the endpoint JSON. Check logs."
            )
        if endpoint_path.exists():
            return
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for endpoint JSON at {endpoint_path}")


def terminate_processes(processes: List[ManagedProcess]) -> None:
    """Terminate a list of managed processes in order."""
    for proc in processes:
        try:
            proc.stop()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Shared utilities for local runners (eval + tracegen)
# ---------------------------------------------------------------------------

# REPO_ROOT is needed for health check script path - computed lazily
_REPO_ROOT: Optional[Path] = None


def _get_repo_root() -> Path:
    """Get the repository root directory."""
    global _REPO_ROOT
    if _REPO_ROOT is None:
        # This file is at hpc/local_runner_utils.py, so repo root is parent
        _REPO_ROOT = Path(__file__).resolve().parent.parent
    return _REPO_ROOT


def resolve_jobs_dir_path(jobs_dir_value: Optional[str], repo_root: Optional[Path] = None) -> Path:
    """Resolve jobs_dir from harbor config to an absolute path.

    Args:
        jobs_dir_value: The jobs_dir value from harbor config (or None)
        repo_root: Repository root path (defaults to auto-detected)

    Returns:
        Absolute path to jobs directory
    """
    if repo_root is None:
        repo_root = _get_repo_root()
    raw_value = jobs_dir_value or "jobs"
    path = Path(raw_value)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path


def run_endpoint_health_check(
    endpoint_json: Path,
    max_attempts: int,
    retry_delay: int,
    repo_root: Optional[Path] = None,
) -> None:
    """Run the vLLM endpoint health check script.

    Args:
        endpoint_json: Path to the endpoint JSON file
        max_attempts: Maximum number of health check attempts
        retry_delay: Delay in seconds between attempts
        repo_root: Repository root path (defaults to auto-detected)

    Raises:
        subprocess.CalledProcessError: If health check fails
    """
    if repo_root is None:
        repo_root = _get_repo_root()

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "vllm" / "wait_for_endpoint.py"),
        "--endpoint-json",
        str(endpoint_json),
        "--max-attempts",
        str(max_attempts),
        "--retry-delay",
        str(retry_delay),
        "--health-path",
        "v1/models",
    ]
    subprocess.run(cmd, check=True)


def load_endpoint_metadata(endpoint_json: Path) -> Dict[str, Any]:
    """Load and parse vLLM endpoint metadata from JSON file.

    Computes api_base and metrics_endpoint URLs from the endpoint_url.

    Args:
        endpoint_json: Path to the endpoint JSON file

    Returns:
        Dict with endpoint data plus computed api_base and metrics_endpoint
    """
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


def apply_datagen_defaults(args: argparse.Namespace) -> None:
    """Load datagen config and apply defaults to args.

    Extracts vLLM settings from datagen config YAML and sets:
    - args.model (if not set)
    - args.tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    - args.ray_port, api_port
    - args._vllm_cli_args, args._vllm_env_vars
    - args._engine_type (openai, anthropic, vllm_local, etc.)
    - args._needs_local_vllm (whether to start Ray/vLLM server)

    Args:
        args: Parsed argparse namespace with datagen_config attribute
    """
    args._vllm_cli_args: List[str] = []
    args._vllm_env_vars: Dict[str, str] = {}
    args._engine_type: str = "vllm_local"  # Default to local vLLM
    args._needs_local_vllm: bool = True  # Default to needing local server

    datagen_config = getattr(args, "datagen_config", None)
    if not datagen_config:
        return

    cfg_path = Path(datagen_config).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Datagen config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as handle:
        datagen_cfg = yaml.safe_load(handle) or {}
    args.datagen_config = str(cfg_path)

    engine_cfg = datagen_cfg.get("engine") or {}
    backend_cfg = datagen_cfg.get("backend") or {}
    vllm_cfg = datagen_cfg.get("vllm_server") or {}

    # Determine engine type and whether we need local vLLM
    engine_type = engine_cfg.get("type", "vllm_local").lower()
    args._engine_type = engine_type

    # API-based engines don't need local Ray/vLLM
    api_engines = {"openai", "anthropic", "azure", "together", "fireworks", "groq"}
    args._needs_local_vllm = engine_type not in api_engines

    # Model path
    if getattr(args, "model", None) is None:
        args.model = vllm_cfg.get("model_path") or engine_cfg.get("model")

    # Parallelism settings
    tp_default = maybe_int(vllm_cfg.get("tensor_parallel_size")) or maybe_int(
        backend_cfg.get("tensor_parallel_size")
    )
    pp_default = maybe_int(vllm_cfg.get("pipeline_parallel_size")) or maybe_int(
        backend_cfg.get("pipeline_parallel_size")
    )
    dp_default = maybe_int(vllm_cfg.get("data_parallel_size")) or maybe_int(
        backend_cfg.get("data_parallel_size")
    )

    if getattr(args, "tensor_parallel_size", None) is None and tp_default:
        args.tensor_parallel_size = tp_default
    if getattr(args, "pipeline_parallel_size", None) is None and pp_default:
        args.pipeline_parallel_size = pp_default
    if getattr(args, "data_parallel_size", None) is None and dp_default:
        args.data_parallel_size = dp_default

    # Port settings
    if getattr(args, "ray_port", None) is None:
        args.ray_port = maybe_int(backend_cfg.get("ray_port")) or 6379
    if getattr(args, "api_port", None) is None:
        args.api_port = maybe_int(backend_cfg.get("api_port")) or 8000

    # Build CLI args and env vars from vllm_server config
    merged_cfg = {**engine_cfg, **vllm_cfg}
    cli_args, env_vars = _build_vllm_cli_args(merged_cfg)
    args._vllm_cli_args = cli_args
    args._vllm_env_vars = env_vars

    # Setup tiktoken encodings for GPT-OSS models
    if is_gpt_oss_model(args.model):
        _, tiktoken_env = setup_gpt_oss_tiktoken()
        args._vllm_env_vars.update(tiktoken_env)


def setup_docker_runtime_if_needed(env_type: str) -> None:
    """Configure Docker/Podman runtime if using docker backend.

    Detects available Docker/Podman runtime, sets DOCKER_HOST environment
    variable, and verifies connectivity.

    Args:
        env_type: Harbor environment type (daytona, docker, modal)

    Raises:
        SystemExit: If docker backend requested but no runtime found
    """
    if env_type.lower() != "docker":
        return

    # Import docker runtime utilities (lazy import to avoid circular deps)
    from hpc.docker_runtime import (
        detect_docker_runtime,
        setup_docker_environment,
        DockerRuntimeType,
        check_docker_connectivity,
    )

    print("[docker] Detecting Docker/Podman runtime...")
    runtime = detect_docker_runtime()

    if runtime.runtime_type == DockerRuntimeType.UNAVAILABLE:
        print("[docker] ERROR: Docker backend requested but no Docker/Podman runtime found.")
        print("[docker] Please ensure Docker or Podman is installed and running,")
        print("[docker] or set DOCKER_HOST to point to a remote Docker daemon.")
        sys.exit(1)

    # Set up environment variables
    env = setup_docker_environment(runtime)
    os.environ.update(env)

    print(f"[docker] Runtime type: {runtime.runtime_type.value}")
    print(f"[docker] DOCKER_HOST: {runtime.docker_host}")

    # Verify connectivity
    if not check_docker_connectivity(timeout=10):
        print("[docker] WARNING: Docker daemon not responding. Continuing anyway...")
    else:
        print("[docker] Docker daemon is accessible.")


# ---------------------------------------------------------------------------
# LocalHarborRunner base class
# ---------------------------------------------------------------------------
# Note: generate_served_model_id and hosted_vllm_alias are imported from
# hpc.launch_utils to avoid duplication.


class LocalHarborRunner:
    """Base class for local Harbor runners (tracegen, eval).

    This class encapsulates the common workflow for running Harbor jobs locally:
    1. Parse and validate arguments
    2. Set up defaults for parallelism, ports, etc.
    3. Start Ray cluster and vLLM controller
    4. Wait for endpoint to be ready
    5. Build and run Harbor command
    6. Clean up processes

    Subclasses should override:
    - JOB_PREFIX: Job name prefix (e.g., "tracegen", "eval")
    - DEFAULT_EXPERIMENTS_SUBDIR: Subdirectory for experiments (e.g., "trace_runs")
    - DEFAULT_N_CONCURRENT: Default concurrent trials
    - DATAGEN_CONFIG_REQUIRED: Whether datagen_config is required
    - get_env_type(): Return the environment type from args
    - get_dataset_label(): Return dataset label for job naming
    - get_dataset_for_harbor(): Return (dataset_slug, dataset_path) tuple
    - validate_args(): Additional argument validation
    - post_harbor_hook(): Called after Harbor completes (for uploads)
    - print_banner(): Print startup banner
    """

    # Subclass configuration - override these in subclasses
    JOB_PREFIX: str = "job"
    DEFAULT_EXPERIMENTS_SUBDIR: str = "runs"
    DEFAULT_N_CONCURRENT: int = 16
    DATAGEN_CONFIG_REQUIRED: bool = False
    DEFAULT_ENDPOINT_FILENAME: str = "vllm_endpoint.json"

    def __init__(self, args: argparse.Namespace, repo_root: Path):
        """Initialize the runner.

        Args:
            args: Parsed command-line arguments
            repo_root: Path to repository root
        """
        self.args = args
        self.repo_root = repo_root
        self.processes: List[ManagedProcess] = []
        self._endpoint_json: Optional[Path] = None
        self._endpoint_meta: Optional[Dict[str, Any]] = None
        self._harbor_job_name: Optional[str] = None

    @classmethod
    def add_common_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add common arguments shared by all local runners.

        Uses shared argument groups from hpc.arg_groups for consistency
        with cloud launchers.

        Args:
            parser: ArgumentParser to add arguments to
        """
        # Harbor core configuration (--harbor-config, --agent, --job-name, --agent-kwarg, --harbor-extra-arg)
        add_harbor_args(parser, config_required=True)

        # Model and compute (--model, --n-concurrent, --n-attempts, --gpus, --dry-run)
        # Note: n_attempts default is 1; subclasses can override via their own defaults
        add_model_compute_args(
            parser,
            model_required=False,
            default_n_concurrent=cls.DEFAULT_N_CONCURRENT,
            default_n_attempts=1,
            n_attempts_help="Times to run each task for repeated trials (default: 1). Not retries on failure.",
        )

        # Ray/vLLM configuration (--host, --ray-port, --api-port, parallelism, health checks)
        add_ray_vllm_args(parser)

        # Log paths (--harbor-binary, --controller-log, --ray-log, --harbor-log)
        add_log_path_args(parser)

        # Local-runner-specific arguments (not in shared arg_groups)
        parser.add_argument("--cpus", type=int, help="CPUs to expose to Ray.")
        parser.add_argument(
            "--endpoint-json",
            help="Optional endpoint JSON path.",
        )

    def get_env_type(self) -> str:
        """Get the environment type from --harbor-env.

        Subclasses must override this method.
        """
        raise NotImplementedError("Subclasses must implement get_env_type()")

    def get_dataset_label(self) -> str:
        """Get the dataset label for job naming.

        Subclasses must override this method.
        """
        raise NotImplementedError("Subclasses must implement get_dataset_label()")

    def get_dataset_for_harbor(self) -> Tuple[Optional[str], Optional[str]]:
        """Return (dataset_slug, dataset_path) for harbor command.

        Subclasses must override this method.

        Returns:
            Tuple of (dataset_slug, dataset_path) - one should be None
        """
        raise NotImplementedError("Subclasses must implement get_dataset_for_harbor()")

    def get_experiments_dir(self) -> Path:
        """Get the experiments directory path.

        Can be overridden by subclasses for custom logic.
        """
        if hasattr(self.args, "experiments_dir") and self.args.experiments_dir:
            return Path(self.args.experiments_dir).expanduser().resolve()
        return self.repo_root / self.DEFAULT_EXPERIMENTS_SUBDIR

    def validate_args(self) -> None:
        """Validate arguments - subclasses can override for additional checks."""
        pass

    def post_harbor_hook(self) -> None:
        """Called after Harbor completes - for uploads, etc.

        Subclasses should override this method to implement upload logic.
        """
        pass

    def print_banner(self) -> None:
        """Print startup banner - subclasses should override."""
        args = self.args
        needs_local_vllm = getattr(args, "_needs_local_vllm", True)
        engine_type = getattr(args, "_engine_type", "vllm_local")

        print(f"=== Local {self.JOB_PREFIX.title()} Runner ===")
        print(f"  Model: {args.model}")
        if needs_local_vllm:
            print(f"  TP/PP/DP: {args.tensor_parallel_size}/{args.pipeline_parallel_size}/{args.data_parallel_size}")
            print(f"  GPUs: {args.gpus}")
        else:
            print(f"  Engine: {engine_type} (API)")
        print("=" * 35)

    def setup(self) -> None:
        """Set up the runner - apply defaults, configure environment."""
        args = self.args

        # Apply datagen config defaults
        apply_datagen_defaults(args)

        # Set up Docker runtime if using docker backend
        setup_docker_runtime_if_needed(self.get_env_type())

        # Set parallelism defaults (only relevant for local vLLM)
        if args.tensor_parallel_size is None:
            args.tensor_parallel_size = 1
        if args.pipeline_parallel_size is None:
            args.pipeline_parallel_size = 1
        if args.data_parallel_size is None:
            args.data_parallel_size = 1

        # Validate model - required for local vLLM, optional for API engines
        needs_local_vllm = getattr(args, "_needs_local_vllm", True)
        if args.model is None and needs_local_vllm:
            raise ValueError("Provide --model or supply a datagen config with vllm_server.model_path.")

        # Generate served model ID (only for local vLLM)
        if needs_local_vllm:
            served_model_id = generate_served_model_id()
            args._served_model_id = served_model_id
            args._harbor_model_name = hosted_vllm_alias(served_model_id)
        else:
            # For API engines, use the model from datagen config directly
            args._served_model_id = None
            args._harbor_model_name = args.model

        # Set GPU/CPU defaults
        if args.gpus is None:
            args.gpus = max(
                1,
                args.tensor_parallel_size * args.pipeline_parallel_size * args.data_parallel_size,
            )
        if args.cpus is None:
            args.cpus = os.cpu_count() or 16

        # Set port defaults
        if args.ray_port is None:
            args.ray_port = 6379
        if args.api_port is None:
            args.api_port = 8000

        # Resolve paths
        args.harbor_config = str(Path(args.harbor_config).expanduser().resolve())

        # Load Harbor config
        harbor_config_data = load_harbor_config(args.harbor_config)
        jobs_dir_value = harbor_config_data.get("jobs_dir") if isinstance(harbor_config_data, dict) else None
        args._jobs_dir_path = resolve_jobs_dir_path(jobs_dir_value, self.repo_root)
        args._harbor_config_data = harbor_config_data

        # Subclass-specific validation
        self.validate_args()

    def _setup_directories(self) -> Tuple[Path, Path]:
        """Set up experiments and logs directories.

        Returns:
            Tuple of (experiments_dir, logs_dir)
        """
        experiments_dir = self.get_experiments_dir()
        experiments_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = experiments_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        return experiments_dir, logs_dir

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        import signal as sig

        def _handle_signal(signum, _frame):
            print(f"\nSignal {signum} received; shutting down...", file=sys.stderr)
            self.cleanup()
            sys.exit(1)

        sig.signal(sig.SIGINT, _handle_signal)
        sig.signal(sig.SIGTERM, _handle_signal)

    def cleanup(self) -> None:
        """Clean up processes."""
        terminate_processes(self.processes[::-1])
        # Only stop Ray if we started it (local vLLM engines)
        needs_local_vllm = getattr(self.args, "_needs_local_vllm", True)
        if needs_local_vllm:
            subprocess.run(["ray", "stop", "--force"], check=False)

    def run(self) -> None:
        """Main entry point - start services and run Harbor."""
        args = self.args
        needs_local_vllm = getattr(args, "_needs_local_vllm", True)

        # Set up directories
        experiments_dir, logs_dir = self._setup_directories()

        # Set up endpoint JSON path (only used for local vLLM)
        self._endpoint_json = Path(args.endpoint_json or (experiments_dir / self.DEFAULT_ENDPOINT_FILENAME))
        if self._endpoint_json.exists():
            self._endpoint_json.unlink()

        # Change to repo root
        os.chdir(self.repo_root)

        # Set up log paths
        ray_log = Path(args.ray_log) if args.ray_log else logs_dir / "ray.log"
        controller_log = Path(args.controller_log) if args.controller_log else logs_dir / "vllm_controller.log"
        harbor_log = Path(args.harbor_log).expanduser().resolve() if args.harbor_log else None

        # Set up signal handlers
        self._setup_signal_handlers()

        # Print banner
        self.print_banner()

        # Start Ray and vLLM only if needed (local vLLM engine)
        if needs_local_vllm:
            controller_script = self.repo_root / "scripts" / "vllm" / "start_vllm_ray_controller.py"

            ray_proc = start_ray(
                host=args.host,
                ray_port=args.ray_port,
                num_gpus=args.gpus,
                num_cpus=args.cpus,
                log_path=ray_log,
            )
            self.processes.append(ray_proc)

            # Start vLLM controller
            vllm_proc = start_vllm_controller(
                model=args.model,
                host=args.host,
                ray_port=args.ray_port,
                api_port=args.api_port,
                tensor_parallel_size=args.tensor_parallel_size,
                pipeline_parallel_size=args.pipeline_parallel_size,
                data_parallel_size=args.data_parallel_size,
                endpoint_path=self._endpoint_json,
                controller_script=controller_script,
                log_path=controller_log,
                served_model_name=getattr(args, "_served_model_id", None),
                extra_cli_args=getattr(args, "_vllm_cli_args", []),
                extra_env_vars=getattr(args, "_vllm_env_vars", {}),
            )
            self.processes.append(vllm_proc)
        else:
            engine_type = getattr(args, "_engine_type", "unknown")
            print(f"[engine] Using {engine_type} API engine - skipping local Ray/vLLM startup")

        try:
            # Wait for endpoint and run health check (only for local vLLM)
            if needs_local_vllm:
                wait_for_endpoint(self._endpoint_json, vllm_proc)
                run_endpoint_health_check(
                    self._endpoint_json,
                    args.health_max_attempts,
                    args.health_retry_delay,
                    self.repo_root,
                )
                self._endpoint_meta = load_endpoint_metadata(self._endpoint_json)
            else:
                # For API engines, no local endpoint metadata
                self._endpoint_meta = None

            # Compute job name
            harbor_model = getattr(args, "_harbor_model_name", args.model)
            job_model_label = args.model or harbor_model or "model"
            dataset_label = self.get_dataset_label()
            job_name = args.job_name or default_job_name(self.JOB_PREFIX, dataset_label, job_model_label)
            self._harbor_job_name = job_name
            args._harbor_job_name = job_name

            # Get dataset info
            dataset_slug, dataset_path = self.get_dataset_for_harbor()

            # Build Harbor command
            harbor_cmd = build_harbor_command(
                harbor_binary=args.harbor_binary,
                harbor_config_path=args.harbor_config,
                harbor_config_data=getattr(args, "_harbor_config_data", {}),
                job_name=job_name,
                agent_name=args.agent,
                model_name=harbor_model,
                env_type=self.get_env_type(),
                n_concurrent=args.n_concurrent,
                n_attempts=args.n_attempts,
                endpoint_meta=self._endpoint_meta,
                agent_kwarg_overrides=list(args.agent_kwarg or []),
                harbor_extra_args=list(args.harbor_extra_arg or []),
                dataset_slug=dataset_slug,
                dataset_path=dataset_path,
            )
            print("Harbor command:", " ".join(harbor_cmd))

            if not args.dry_run:
                # Import here to avoid circular imports
                from hpc.cli_utils import run_harbor_cli
                run_harbor_cli(harbor_cmd, harbor_log)
                self.post_harbor_hook()
            else:
                print(f"[dry-run] Would run Harbor {self.JOB_PREFIX} job.")

        finally:
            self.cleanup()


__all__ = [
    # Process management
    "ManagedProcess",
    "start_ray",
    "start_vllm_controller",
    "wait_for_endpoint",
    "terminate_processes",
    # Config and setup utilities
    "maybe_int",
    "apply_datagen_defaults",
    "setup_docker_runtime_if_needed",
    "load_harbor_config",
    "get_harbor_env_from_config",
    "resolve_jobs_dir_path",
    # Endpoint utilities
    "run_endpoint_health_check",
    "load_endpoint_metadata",
    # Harbor command building
    "extract_agent_kwargs_from_config",
    "default_job_name",
    "apply_nested_key",
    "parse_agent_kwarg_strings",
    "serialize_agent_kwargs",
    "build_harbor_command",
    # Model ID utilities (re-exported from launch_utils)
    "generate_served_model_id",
    "hosted_vllm_alias",
    # Base runner class
    "LocalHarborRunner",
    # Re-exports
    "_build_vllm_cli_args",
]
