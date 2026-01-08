"""Harbor CLI utilities for HPC launchers.

This module provides utilities for interacting with the Harbor CLI:
- Config loading and parsing
- Agent kwargs extraction and serialization
- Command building for harbor jobs start

These utilities are shared across all execution paths:
- Local runners (data/local/run_tracegen.py, eval/local/run_eval.py)
- Cloud launchers (data/cloud/launch_tracegen_cloud.py, eval/cloud/launch_eval_cloud.py)
- HPC SLURM launchers (hpc/launch.py --job_type datagen/eval)
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml


# ---------------------------------------------------------------------------
# Harbor config loading
# ---------------------------------------------------------------------------


def load_harbor_config(harbor_config_path: str) -> Dict[str, Any]:
    """Load and parse Harbor config YAML.

    Args:
        harbor_config_path: Path to harbor config file

    Returns:
        Parsed harbor config dict (empty dict if file not found)
    """
    try:
        with open(harbor_config_path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except FileNotFoundError:
        return {}


def get_harbor_env_from_config(
    harbor_config: Union[str, Dict[str, Any], None],
    default: str = "daytona",
) -> str:
    """Extract Harbor environment type from config.

    Reads the `environment.type` field from Harbor YAML config to determine
    the sandbox backend (daytona, docker, modal, apptainer).

    Args:
        harbor_config: Either a path to Harbor config YAML, a parsed config dict,
                      or None.
        default: Default environment type if not found in config (default: "daytona").

    Returns:
        Environment type string: "daytona", "docker", "modal", "apptainer", etc.

    Examples:
        >>> get_harbor_env_from_config("hpc/harbor_yaml/trace_docker_32concurrency_ctx131k.yaml")
        'docker'
        >>> get_harbor_env_from_config({"environment": {"type": "daytona"}})
        'daytona'
        >>> get_harbor_env_from_config(None)
        'daytona'
    """
    if harbor_config is None:
        return default

    # Load config if path provided
    if isinstance(harbor_config, str):
        config_dict = load_harbor_config(harbor_config)
    else:
        config_dict = harbor_config

    # Extract environment.type
    env_config = config_dict.get("environment") or {}
    env_type = env_config.get("type")

    if env_type and isinstance(env_type, str):
        return env_type.lower()

    return default


# ---------------------------------------------------------------------------
# Agent kwargs utilities
# ---------------------------------------------------------------------------


def extract_agent_kwargs_from_config(harbor_config: dict, agent_name: str) -> dict:
    """Extract kwargs for the specified agent from harbor config.

    The Harbor YAML is the ground truth for agent configuration. This function
    finds the agent by name and returns a copy of its kwargs dict.

    Args:
        harbor_config: Parsed harbor config dict (from YAML)
        agent_name: Name of the agent to find (e.g., "terminus-2")

    Returns:
        Copy of the agent's kwargs dict, or empty dict if not found
    """
    agents = harbor_config.get("agents", [])
    for agent in agents:
        if agent.get("name") == agent_name:
            return copy.deepcopy(agent.get("kwargs", {}))
    # Fallback: return first agent's kwargs if no match (backwards compat)
    if agents and isinstance(agents[0], dict):
        return copy.deepcopy(agents[0].get("kwargs", {}))
    return {}


def apply_nested_key(target: dict, dotted_key: str, value: Any) -> None:
    """Apply a value to a nested dict using dotted key notation.

    Args:
        target: Dict to modify in-place
        dotted_key: Key like "model_info.max_tokens" for nested access
        value: Value to set
    """
    parts = dotted_key.split(".")
    cursor = target
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def parse_agent_kwarg_strings(entries: List[str]) -> Tuple[Dict[str, Any], List[str]]:
    """Parse --agent-kwarg CLI entries into overrides and passthrough.

    Args:
        entries: List of "key=value" strings (or passthrough entries without =)

    Returns:
        Tuple of (overrides dict, passthrough list)
    """
    overrides: Dict[str, Any] = {}
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


def serialize_agent_kwargs(kwargs: dict) -> List[str]:
    """Serialize agent kwargs dict to CLI argument strings.

    Args:
        kwargs: Dict of agent kwargs

    Returns:
        List of "key=value" strings suitable for --agent-kwarg
    """
    serialized: List[str] = []
    for key, value in kwargs.items():
        if isinstance(value, (dict, list)):
            serialized.append(f"{key}={json.dumps(value)}")
        else:
            serialized.append(f"{key}={value}")
    return serialized


# ---------------------------------------------------------------------------
# Job naming utilities
# ---------------------------------------------------------------------------


def _timestamp() -> str:
    """Generate a timestamp string for job names."""
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def default_job_name(prefix: str, dataset_label: str, model_label: str) -> str:
    """Generate a default job name.

    Args:
        prefix: Job type prefix (e.g., "eval", "tracegen")
        dataset_label: Dataset/tasks identifier
        model_label: Model identifier

    Returns:
        Formatted job name like "eval-dataset-model-20240101_120000"
    """
    sanitized_dataset = Path(dataset_label).name.replace("/", "-").replace(" ", "_")
    sanitized_model = model_label.replace("/", "-").replace(" ", "_")
    return f"{prefix}-{sanitized_dataset}-{sanitized_model}-{_timestamp()}"


# ---------------------------------------------------------------------------
# Harbor command building
# ---------------------------------------------------------------------------


def build_harbor_command(
    harbor_binary: str,
    harbor_config_path: str,
    harbor_config_data: dict,
    job_name: str,
    agent_name: str,
    model_name: str,
    env_type: str,
    n_concurrent: int,
    n_attempts: int,
    endpoint_meta: Optional[dict],
    agent_kwarg_overrides: List[str],
    harbor_extra_args: List[str],
    dataset_slug: Optional[str] = None,
    dataset_path: Optional[str] = None,
    jobs_dir: Optional[str] = None,
) -> List[str]:
    """Build the harbor jobs start command.

    The Harbor YAML is the ground truth for agent configuration. This function:
    1. Extracts ALL kwargs from the Harbor YAML for the specified agent
    2. Overrides with endpoint-specific values (api_base, metrics_endpoint) if using local vLLM
    3. Applies CLI --agent-kwarg overrides with highest precedence

    Args:
        harbor_binary: Path to harbor CLI
        harbor_config_path: Path to harbor config YAML
        harbor_config_data: Parsed harbor config dict
        job_name: Name for this harbor job
        agent_name: Agent to run (e.g., "terminus-2")
        model_name: Model identifier for --model flag
        env_type: Environment type for --env flag (daytona, docker, modal, apptainer)
        n_concurrent: Number of concurrent trials
        n_attempts: Number of attempts per task
        endpoint_meta: Dict with api_base and metrics_endpoint from vLLM (None for API engines)
        agent_kwarg_overrides: Raw --agent-kwarg strings from CLI
        harbor_extra_args: Additional args to pass through to harbor
        dataset_slug: Harbor dataset slug (mutually exclusive with dataset_path)
        dataset_path: Path to tasks directory (mutually exclusive with dataset_slug)
        jobs_dir: Override for --jobs-dir (where Harbor writes job outputs)

    Returns:
        Complete harbor command as list of strings
    """
    # Build agent kwargs - start with ALL kwargs from Harbor YAML as ground truth
    agent_kwargs = extract_agent_kwargs_from_config(harbor_config_data, agent_name)

    # Override with endpoint-specific values (only for local vLLM, not API engines)
    if endpoint_meta:
        if endpoint_meta.get("metrics_endpoint"):
            agent_kwargs["metrics_endpoint"] = endpoint_meta["metrics_endpoint"]
        if endpoint_meta.get("api_base"):
            agent_kwargs["api_base"] = endpoint_meta["api_base"]

    # CLI --agent-kwarg flags take highest precedence (supports dotted keys)
    override_kwargs, passthrough = parse_agent_kwarg_strings(agent_kwarg_overrides)
    for dotted_key, override_value in override_kwargs.items():
        apply_nested_key(agent_kwargs, dotted_key, override_value)

    # Build base command
    cmd = [
        harbor_binary,
        "jobs",
        "start",
        "--config",
        harbor_config_path,
        "--job-name",
        job_name,
        "--agent",
        agent_name,
        "--model",
        model_name,
        "--env",
        env_type,
        "--n-concurrent",
        str(n_concurrent),
        "--n-attempts",
        str(n_attempts),
    ]

    # Add dataset (slug or path)
    if dataset_slug:
        cmd.extend(["--dataset", dataset_slug])
    elif dataset_path:
        cmd.extend(["-p", dataset_path])

    # Add jobs_dir if specified
    if jobs_dir:
        cmd.extend(["--jobs-dir", jobs_dir])

    # Add serialized agent kwargs
    for kw in serialize_agent_kwargs(agent_kwargs):
        cmd.extend(["--agent-kwarg", kw])
    for passthrough_kw in passthrough:
        cmd.extend(["--agent-kwarg", passthrough_kw])

    # Process extra args with sensible defaults
    extra_args = list(harbor_extra_args or [])

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
