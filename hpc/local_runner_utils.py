"""Shared utilities for local Ray/vLLM runners.

This module consolidates common code used by:
- eval/local/run_eval.py
- data/local/run_tracegen.py

It provides managed subprocess handling for Ray clusters and vLLM servers.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from hpc.vllm_utils import _build_vllm_cli_args


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


__all__ = [
    "ManagedProcess",
    "maybe_int",
    "start_ray",
    "start_vllm_controller",
    "wait_for_endpoint",
    "terminate_processes",
    "_build_vllm_cli_args",  # Re-export from vllm_utils
]
