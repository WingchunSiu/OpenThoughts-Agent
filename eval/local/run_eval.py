#!/usr/bin/env python3
"""
Local eval runner.

Starts a single-node Ray cluster + vLLM controller and then launches a Harbor eval
job that targets the freshly booted endpoint. Designed for non-SLURM Linux hosts
where we have exclusive access to the box.
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
from typing import Iterable, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENTS_DIR = REPO_ROOT / "eval_runs"
DEFAULT_ENDPOINT = "vllm_endpoint.json"

VALUE_ENV_FIELDS = {
    "max_model_len": "VLLM_MAX_MODEL_LEN",
    "max_num_seqs": "VLLM_MAX_NUM_SEQS",
    "gpu_memory_utilization": "VLLM_GPU_MEMORY_UTILIZATION",
    "swap_space": "VLLM_SWAP_SPACE",
    "max_seq_len_to_capture": "VLLM_MAX_SEQ_LEN_TO_CAPTURE",
    "custom_model_name": "VLLM_CUSTOM_MODEL_NAME",
    "tool_call_parser": "VLLM_TOOL_CALL_PARSER",
    "reasoning_parser": "VLLM_REASONING_PARSER",
    "hf_overrides": "VLLM_HF_OVERRIDES",
    "cpu_offload_gb": "VLLM_CPU_OFFLOAD_GB",
    "kv_offloading_size": "VLLM_KV_OFFLOADING_SIZE",
    "kv_offloading_backend": "VLLM_KV_OFFLOADING_BACKEND",
    "max_output_tokens": "VLLM_MAX_OUTPUT_TOKENS",
    "logging_level": "VLLM_LOGGING_LEVEL",
}

BOOL_ENV_FIELDS = {
    "use_deep_gemm": "VLLM_USE_DEEP_GEMM",
    "enable_expert_parallel": "VLLM_ENABLE_EXPERT_PARALLEL",
    "enable_auto_tool_choice": "VLLM_ENABLE_AUTO_TOOL_CHOICE",
    "trust_remote_code": "VLLM_TRUST_REMOTE_CODE",
    "disable_log_requests": "VLLM_DISABLE_LOG_REQUESTS",
}


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
    parser.add_argument(
        "--expected-trials",
        type=int,
        help="Optional expected trial count (for bookkeeping only).",
    )
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
    return parser.parse_args()


def _ensure_mutually_exclusive(dataset: Optional[str], dataset_path: Optional[str]) -> None:
    if dataset and dataset_path:
        raise ValueError("Specify either --dataset or --dataset-path (not both).")
    if not dataset and not dataset_path:
        raise ValueError("Must provide --dataset or --dataset-path.")


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _default_job_name(dataset_label: str, model: str) -> str:
    sanitized_dataset = dataset_label.replace("/", "-").replace(" ", "_")
    sanitized_model = model.replace("/", "-")
    return f"eval-{sanitized_dataset}-{sanitized_model}-{_timestamp()}"


class ManagedProcess:
    def __init__(self, name: str, popen: subprocess.Popen):
        self.name = name
        self.proc = popen

    def stop(self, timeout: float = 10.0) -> None:
        if self.proc.poll() is not None:
            return
        try:
            self.proc.terminate()
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def _maybe_int(value: object) -> Optional[int]:
    if value in (None, "", "None"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _prepare_vllm_env_from_datagen(cfg: dict) -> dict[str, str]:
    env: dict[str, str] = {}
    for field, env_key in VALUE_ENV_FIELDS.items():
        value = cfg.get(field)
        if value in (None, "", "None"):
            continue
        env[env_key] = str(value)
    for field, env_key in BOOL_ENV_FIELDS.items():
        flag = cfg.get(field)
        if isinstance(flag, bool) and flag:
            env[env_key] = "1"
    return env


def _apply_datagen_defaults(args: argparse.Namespace) -> None:
    args._vllm_env_overrides = {}
    args._vllm_extra_args: List[str] = []
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

    tp_default = _maybe_int(vllm_cfg.get("tensor_parallel_size")) or _maybe_int(
        backend_cfg.get("tensor_parallel_size")
    )
    pp_default = _maybe_int(vllm_cfg.get("pipeline_parallel_size")) or _maybe_int(
        backend_cfg.get("pipeline_parallel_size")
    )
    dp_default = _maybe_int(vllm_cfg.get("data_parallel_size")) or _maybe_int(
        backend_cfg.get("data_parallel_size")
    )

    if args.tensor_parallel_size is None and tp_default:
        args.tensor_parallel_size = tp_default
    if args.pipeline_parallel_size is None and pp_default:
        args.pipeline_parallel_size = pp_default
    if args.data_parallel_size is None and dp_default:
        args.data_parallel_size = dp_default

    if args.ray_port is None:
        args.ray_port = _maybe_int(backend_cfg.get("ray_port")) or args.ray_port
    if args.api_port is None:
        args.api_port = _maybe_int(backend_cfg.get("api_port")) or args.api_port

    env_overrides = _prepare_vllm_env_from_datagen({**engine_cfg, **vllm_cfg})
    args._vllm_env_overrides = env_overrides

    extra_args = vllm_cfg.get("extra_args")
    if isinstance(extra_args, list) and extra_args:
        args._vllm_extra_args = [str(item) for item in extra_args]
def _start_ray(args: argparse.Namespace, log_path: Path | None) -> ManagedProcess:
    cmd = [
        "ray",
        "start",
        "--head",
        f"--node-ip-address={args.host}",
        f"--port={args.ray_port}",
        f"--num-gpus={args.gpus}",
        f"--num-cpus={args.cpus}",
        "--dashboard-host=0.0.0.0",
        "--block",
    ]
    env = os.environ.copy()
    stdout = stderr = None
    if log_path:
        log_file = open(log_path, "w", encoding="utf-8")
        stdout = log_file
        stderr = log_file
    else:
        log_file = None
    popen = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, env=env)
    process = ManagedProcess("ray", popen)
    process._log_handle = log_file  # type: ignore[attr-defined]
    return process


def _start_vllm_controller(
    args: argparse.Namespace,
    endpoint_path: Path,
    log_path: Path | None,
) -> ManagedProcess:
    env = os.environ.copy()
    env.update(
        {
            "VLLM_MODEL_PATH": args.model,
            "VLLM_TENSOR_PARALLEL_SIZE": str(args.tensor_parallel_size),
            "VLLM_PIPELINE_PARALLEL_SIZE": str(args.pipeline_parallel_size),
            "VLLM_DATA_PARALLEL_SIZE": str(args.data_parallel_size),
            "VLLM_ENDPOINT_JSON_PATH": str(endpoint_path),
        }
    )
    env.update(getattr(args, "_vllm_env_overrides", {}))
    extra_cli = getattr(args, "_vllm_extra_args", None)
    if extra_cli:
        env["VLLM_SERVER_EXTRA_ARGS_JSON"] = json.dumps(extra_cli)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "vllm" / "start_vllm_ray_controller.py"),
        "--ray-address",
        f"{args.host}:{args.ray_port}",
        "--host",
        args.host,
        "--port",
        str(args.api_port),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--pipeline-parallel-size",
        str(args.pipeline_parallel_size),
        "--data-parallel-size",
        str(args.data_parallel_size),
        "--endpoint-json",
        str(endpoint_path),
    ]
    stdout = stderr = None
    if log_path:
        log_file = open(log_path, "w", encoding="utf-8")
        stdout = log_file
        stderr = log_file
    else:
        log_file = None
    popen = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, env=env)
    process = ManagedProcess("vllm_controller", popen)
    process._log_handle = log_file  # type: ignore[attr-defined]
    return process


def _wait_for_endpoint(endpoint_path: Path, controller: ManagedProcess, timeout: int = 300) -> None:
    start = time.time()
    while time.time() - start < timeout:
        if controller.proc.poll() is not None:
            raise RuntimeError("vLLM controller exited before writing the endpoint JSON. Check logs.")
        if endpoint_path.exists():
            return
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for endpoint JSON at {endpoint_path}")


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


def _build_harbor_command(
    args: argparse.Namespace,
    dataset_label: str,
    endpoint_meta: dict,
) -> List[str]:
    job_name = args.job_name or _default_job_name(dataset_label, args.model)
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
        args.model,
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
    if args.expected_trials:
        cmd.extend(["--expected-trials", str(args.expected_trials)])

    agent_kwargs: List[str] = list(args.agent_kwarg or [])
    keys = {kw.split("=", 1)[0] for kw in agent_kwargs if "=" in kw}
    if endpoint_meta.get("api_base") and "api_base" not in keys:
        agent_kwargs.append(f"api_base={endpoint_meta['api_base']}")
    if endpoint_meta.get("metrics_endpoint") and "metrics_endpoint" not in keys:
        agent_kwargs.append(f"metrics_endpoint={endpoint_meta['metrics_endpoint']}")
    for kw in agent_kwargs:
        cmd.extend(["--agent-kwarg", kw])

    for extra in args.harbor_extra_arg or []:
        cmd.append(extra)

    return cmd


def _run_harbor_cli(cmd: List[str], log_path: Path | None) -> None:
    if log_path:
        with open(log_path, "w", encoding="utf-8") as harbor_log_file:
            print(f"Streaming Harbor output to {log_path}")
            subprocess.run(
                cmd,
                check=True,
                stdout=harbor_log_file,
                stderr=subprocess.STDOUT,
            )
        return

    master_fd, slave_fd = pty.openpty()
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            text=False,
        )
        os.close(slave_fd)
        while True:
            try:
                data = os.read(master_fd, 4096)
            except OSError as exc:
                if exc.errno != errno.EIO:
                    raise
                break
            if not data:
                break
            os.write(sys.stdout.fileno(), data)
    finally:
        os.close(master_fd)
    ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)


def _terminate(processes: Iterable[ManagedProcess]) -> None:
    for proc in processes:
        try:
            proc.stop()
        finally:
            log_handle = getattr(proc, "_log_handle", None)
            if log_handle:
                log_handle.close()


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
        _terminate(processes)
        subprocess.run(["ray", "stop", "--force"], check=False)
        sys.exit(1)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    ray_proc = _start_ray(args, ray_log)
    processes.append(ray_proc)
    vllm_proc = _start_vllm_controller(args, endpoint_json, controller_log)
    processes.append(vllm_proc)

    try:
        _wait_for_endpoint(endpoint_json, vllm_proc)
        _run_endpoint_health_check(endpoint_json, args.health_max_attempts, args.health_retry_delay)
        endpoint_meta = _load_endpoint_metadata(endpoint_json)
        harbor_cmd = _build_harbor_command(args, dataset_label, endpoint_meta)
        print("Harbor command:", " ".join(harbor_cmd))
        if not args.dry_run:
            _run_harbor_cli(harbor_cmd, harbor_log)
    finally:
        _terminate(processes[::-1])
        subprocess.run(["ray", "stop", "--force"], check=False)


if __name__ == "__main__":
    main()
